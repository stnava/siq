import os
import sys
import queue
import threading
import math

# Ensure repo root and tests dir are on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
tests_dir = os.path.dirname(os.path.abspath(__file__))
if tests_dir not in sys.path:
    sys.path.insert(0, tests_dir)

# Configure Keras to use PyTorch backend for GPU MPS/CUDA acceleration
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import keras
keras.config.enable_unsafe_deserialization()
from keras import ops
import ants
import antspynet
import siq
from siq.get_data import compute_gmsd, compute_hfen


def prefetch_generator(gen, maxsize=4):
    """Wrap a generator with a background prefetch thread.

    While the GPU is executing a training step, the CPU thread is already
    generating the NEXT batch. This eliminates the ~1.7 s/step stall where
    the GPU sits idle waiting for new data.

    Parameters
    ----------
    gen : generator
        The source generator (e.g. siq.blind_sr_generator).
    maxsize : int
        Maximum number of batches to prefetch into the queue (default 4).
        Set to 0 to disable prefetching (returns gen unchanged).

    Yields
    ------
    tuple
        (x_batch, y_batch) from the underlying generator.
    """
    if maxsize <= 0:
        yield from gen
        return

    q = queue.Queue(maxsize=maxsize)
    _sentinel = object()

    def _worker():
        try:
            for item in gen:
                q.put(item)
        except Exception as e:
            print(f"[PrefetchGenerator] Worker exception: {e}")
        finally:
            q.put(_sentinel)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    while True:
        item = q.get()
        if item is _sentinel:
            break
        yield item


class OneCycleLR(keras.optimizers.schedules.LearningRateSchedule):
    """OneCycle learning-rate schedule for fast convergence.

    Linearly warms the LR from base_lr to max_lr over the first
    ``pct_start`` fraction of steps, then applies cosine annealing
    from max_lr down to final_lr for the remaining steps.

    Parameters
    ----------
    max_lr : float
        Peak learning rate (e.g. 2e-4).
    total_steps : int
        Total number of training steps this schedule covers.
    pct_start : float
        Fraction of steps used for the warmup phase (default 0.3).
    div_factor : float
        base_lr = max_lr / div_factor (default 25.0).
    final_div : float
        final_lr = max_lr / final_div (default 1e4).
    """

    def __init__(self, max_lr, total_steps, pct_start=0.3,
                 div_factor=25.0, final_div=1e4):
        super().__init__()
        self.max_lr = float(max_lr)
        self.total_steps = int(total_steps)
        self.pct_start = float(pct_start)
        self.base_lr = self.max_lr / div_factor
        self.final_lr = self.max_lr / final_div

    def __call__(self, step):
        step = keras.ops.cast(step, "float32")
        warmup_steps = float(self.total_steps) * self.pct_start

        # Linear warmup phase
        warmup_lr = self.base_lr + (self.max_lr - self.base_lr) * (
            step / max(1.0, warmup_steps)
        )
        # Cosine annealing phase
        progress = (step - warmup_steps) / max(
            1.0, float(self.total_steps) - warmup_steps
        )
        cosine_lr = self.final_lr + 0.5 * (self.max_lr - self.final_lr) * (
            1.0 + keras.ops.cos(math.pi * keras.ops.clip(progress, 0.0, 1.0))
        )
        return keras.ops.where(step < warmup_steps, warmup_lr, cosine_lr)

    def get_config(self):
        return {
            "max_lr": self.max_lr,
            "total_steps": self.total_steps,
            "pct_start": self.pct_start,
            "div_factor": self.max_lr / self.base_lr,
            "final_div": self.max_lr / self.final_lr,
        }

def rank_normalize_batch(x_tensor):
    """Apply per-sample rank normalization to match antspyt1w resnet_grader preprocessing.

    The ResNet grader was trained with ``ants.rank_intensity`` applied to each
    volume before inference. That transform maps every voxel's intensity to its
    fractional rank within the volume (output in [0, 1], uniform distribution).
    Without this, we feed linear [0,1] data to a network that learned rank-uniform
    features, causing a significant distribution mismatch.

    This function replicates that transform on a batched Keras/torch tensor:
      1. Convert to numpy (CPU, float32).
      2. For each sample: flatten → double argsort → divide by (N-1) → reshape.
      3. Convert back to a Keras tensor on the original device.

    Parameters
    ----------
    x_tensor : Keras tensor, shape (B, D, H, W, C)
        Input batch in [0, 1] linear intensity range.

    Returns
    -------
    Keras tensor, same shape and device as input, values in [0, 1] (rank-uniform).
    """
    x_np = ops.convert_to_numpy(x_tensor).astype("float32")
    out = np.empty_like(x_np)
    B = x_np.shape[0]
    for b in range(B):
        vol = x_np[b, ..., 0].ravel()
        # double argsort = rank transform
        ranks = np.argsort(np.argsort(vol)).astype("float32")
        n = len(ranks)
        ranks /= max(n - 1, 1)
        out[b, ..., 0] = ranks.reshape(x_np.shape[1:-1])
    return ops.convert_to_tensor(out, dtype="float32")


def set_core_trainable(model, trainable=True):
    count = 0
    for layer in model.layers:
        if isinstance(layer, (keras.layers.Conv3D, keras.layers.Conv2D)):
            # Ignore channel attention 1x1 convs and global scaling layers
            if layer.kernel_size != (1, 1) and layer.kernel_size != (1, 1, 1) and "_ca_" not in layer.name:
                layer.trainable = trainable
                count += 1
    print(f"Set layer.trainable={trainable} for {count} core Conv layers.")

def apply_icnr_initialization(model, factor=2):
    """
    Applies ICNR (Initialization Checkerboard Free) initialization to all Conv2D and Conv3D
    layers that are immediately followed by PixelShuffle.
    """
    print("Applying ICNR weight initialization to PixelShuffle-preceding convolutional layers...")
    for layer in model.layers:
        if isinstance(layer, (keras.layers.Conv2D, keras.layers.Conv3D)):
            weights = layer.get_weights()
            if not weights:
                continue
            w = weights[0]
            is_3d = len(w.shape) == 5
            if isinstance(factor, (list, tuple)):
                num_subpixels = math.prod(factor)
            else:
                num_subpixels = factor**3 if is_3d else factor**2
            
            # Check if last dimension is divisible by num_subpixels and layer name is an upsampler preceding PixelShuffle
            is_upsampler = ("preshuffle_conv" in layer.name or 
                            "_up_conv1" in layer.name or 
                            "_up_conv2" in layer.name or 
                            "_down_conv2" in layer.name)
            if w.shape[-1] % num_subpixels == 0 and is_upsampler:
                out_channels = w.shape[-1] // num_subpixels
                print(f"  ICNR initializing layer: {layer.name} with shape {w.shape} (factor={factor})")
                
                base_shape = list(w.shape[:-1]) + [out_channels]
                initializer = keras.initializers.GlorotUniform()
                base_w = keras.ops.convert_to_numpy(initializer(base_shape, dtype=layer.dtype))
                
                # Tile the base weights along the last dimension
                new_w = np.tile(base_w, (1,) * (len(base_shape) - 1) + (num_subpixels,))
                
                if len(weights) > 1:
                    b = weights[1]
                    base_b = np.zeros([out_channels], dtype=b.dtype)
                    new_b = np.tile(base_b, num_subpixels)
                    layer.set_weights([new_w, new_b])
                else:
                    layer.set_weights([new_w])

def lowess_smooth(x, y, x_query, span=100):
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    if len(x) < 5:
        return float(y[-1]) if len(y) > 0 else 1.0
        
    distances = np.abs(x - x_query)
    max_d = np.max(distances)
    if max_d == 0:
        return float(y[-1])
        
    u = distances / (max_d * 1.001)
    weights = (1.0 - u**3)**3
    weights[u >= 1] = 0.0
    weights = np.maximum(weights, 1e-4)
    
    dx = x - x_query
    W = np.diag(weights)
    X = np.vstack([np.ones_like(dx), dx]).T
    
    try:
        XTW = X.T @ W
        beta = np.linalg.solve(XTW @ X, XTW @ y)
        return float(beta[0])
    except np.linalg.LinAlgError:
        return float(np.sum(weights * y) / np.sum(weights))

class LossHistoryTracker:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.iterations = []
        self.raw_mae = []
        self.raw_percep = []
        self.raw_tv = []
        
    def add(self, iteration, mae, percep, tv):
        self.iterations.append(iteration)
        self.raw_mae.append(mae)
        self.raw_percep.append(percep)
        self.raw_tv.append(tv)
        if len(self.iterations) > self.window_size:
            self.iterations.pop(0)
            self.raw_mae.pop(0)
            self.raw_percep.pop(0)
            self.raw_tv.pop(0)

def get_smoothed_losses_and_weights(tracker, target_pcts, current_iteration, arg4=1.0, arg5=None, beta_damp=0.98, target_total_loss=1.0):
    # Support both new signature (tracker, target_pcts, current_iteration, current_weights, target_total_loss, beta_damp)
    # and legacy signature (tracker, target_pcts, current_iteration, original_weight_sum, current_weights, beta_damp)
    if isinstance(arg4, dict):
        current_weights = arg4
        if isinstance(arg5, (int, float)):
            target_total_loss = float(arg5)
    else:
        target_total_loss = float(arg4)
        current_weights = arg5 if arg5 is not None else {'mae': 1.0, 'percep': 1.0, 'tv': 1.0}

    if len(tracker.iterations) < 5:
        return current_weights, {
            'mae': tracker.raw_mae[-1] if tracker.raw_mae else 1.0,
            'percep': tracker.raw_percep[-1] if tracker.raw_percep else 1.0,
            'tv': tracker.raw_tv[-1] if tracker.raw_tv else 1.0
        }
        
    smooth_mae = max(1e-8, lowess_smooth(tracker.iterations, tracker.raw_mae, current_iteration))
    smooth_percep = max(1e-8, lowess_smooth(tracker.iterations, tracker.raw_percep, current_iteration))
    smooth_tv = max(1e-8, lowess_smooth(tracker.iterations, tracker.raw_tv, current_iteration))
    
    smoothed = {'mae': smooth_mae, 'percep': smooth_percep, 'tv': smooth_tv}
    
    t_mae = target_pcts.get('mae', 30.0)
    t_percep = target_pcts.get('percep', 65.0)
    t_tv = target_pcts.get('tv', 5.0)
    
    total_t = t_mae + t_percep + t_tv
    p_mae = t_mae / total_t
    p_percep = t_percep / total_t
    p_tv = t_tv / total_t
    
    w_mae_tgt = (p_mae * target_total_loss) / smooth_mae
    w_percep_tgt = (p_percep * target_total_loss) / smooth_percep
    w_tv_tgt = (p_tv * target_total_loss) / smooth_tv
    
    # Exponentially damped step toward LOWESS target (beta_damp controls inertia).
    # No soft decay over iterations — the hard freeze in step_dynamic_balancer
    # is the sole mechanism for locking weights at convergence.
    effective_step = 1.0 - beta_damp
    
    new_mae   = (1.0 - effective_step) * current_weights['mae']   + effective_step * w_mae_tgt
    new_percep = (1.0 - effective_step) * current_weights['percep'] + effective_step * w_percep_tgt
    new_tv    = (1.0 - effective_step) * current_weights['tv']    + effective_step * w_tv_tgt
    
    return {'mae': new_mae, 'percep': new_percep, 'tv': new_tv}, smoothed

def auto_weight_loss_multi(mdl, feature_extractor, x, y, feature=2.0, tv=0.1, verbose=True):
    y = ops.convert_to_tensor(y)
    y_pred = mdl(x)
    squared_difference = ops.square(y - y_pred)
    myax = list(range(1, len(y.shape)))
    msqTerm = ops.mean(squared_difference, axis=myax)
    
    f_true = feature_extractor(y)
    f_pred = feature_extractor(y_pred)
    if isinstance(f_true, list):
        feat_term = 0.0
        for ft, fp in zip(f_true, f_pred):
            feat_term += ops.mean(ops.square(ft - fp))
        mean_feat = float(feat_term)
    else:
        mean_feat = float(ops.mean(ops.square(f_true - f_pred)))
        
    msqw = 10.0
    mean_msq = float(ops.mean(msqTerm))
    featw = feature * msqw * mean_msq / (mean_feat + 1e-8)
    
    dim = len(y.shape) - 2
    if dim == 2:
        diff_h = ops.mean(ops.abs(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]))
        diff_w = ops.mean(ops.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]))
        tv_val = float(diff_h + diff_w)
    else:
        diff_d = ops.mean(ops.abs(y_pred[:, 1:, :, :, :] - y_pred[:, :-1, :, :, :]))
        diff_h = ops.mean(ops.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :]))
        diff_w = ops.mean(ops.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :]))
        tv_val = float(diff_d + diff_h + diff_w)
        
    tvw = tv * msqw * mean_msq / (tv_val + 1e-8)
    
    if verbose:
        print("MSQ: " + str(float(msqw * mean_msq)))
        print("Feat: " + str(float(featw * mean_feat)))
        print("Tv: " + str(float(tv_val * tvw)))
        
    return [float(msqw), float(featw), float(tvw)]

def main():
    import argparse
    parser = argparse.ArgumentParser(description="SIQ Super-Resolution Refinement Pipeline")
    parser.add_argument("model", choices=["espcn", "ldbpn", "ref-dbpn", "wdsr", "rcan", "carn", "espcn-rc", "wdsr-rc", "srfbn", "san", "asdbpn"], default="espcn", nargs="?", help="Model type to refine (default: espcn)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for training (default: 1)")
    parser.add_argument("--dim", type=int, choices=[2, 3], default=3, help="Dimensionality (2 or 3) (default: 3)")
    parser.add_argument("--stage1-iter", type=int, default=100, help="Max iterations for Stage 1 (default: 100)")
    parser.add_argument("--stage2-iter", type=int, default=2000, help="Max iterations for Stage 2 (default: 2000)")
    parser.add_argument("--stage3-iter", type=int, default=5000, help="Max iterations for Stage 3 (default: 5000)")
    parser.add_argument("--target-percep", type=float, default=65.0, help="Target perceptual loss percentage contribution (default: 65.0)")
    parser.add_argument("--target-mae", type=float, default=30.0, help="Target MAE loss percentage contribution (default: 30.0)")
    parser.add_argument("--target-tv", type=float, default=5.0, help="Target TV loss percentage contribution (default: 5.0)")
    parser.add_argument("--target-total-loss", type=float, default=1.0, help="Target total loss scale for auto-balancing (default: 1.0)")
    parser.add_argument("--anneal-iter", type=int, default=100, help="Number of iterations in Stage 1 to linearly anneal perceptual and TV loss weights (default: 100, 0 to disable)")
    parser.add_argument("--dampening", type=float, default=0.98, help="Dampening factor beta for weight transition (default: 0.98)")
    parser.add_argument("--smooth-window", type=int, default=100, help="LOWESS smoothing window size (default: 100)")
    parser.add_argument("--update-freq", type=int, default=10, help="Weight update frequency in iterations (default: 10)")
    parser.add_argument("--use-layer2", action="store_true", default=False, help="Enable Layer 2 procedural shape simulations (default: False)")
    parser.add_argument("--projection-kernel-size", type=int, default=None, help="Projection kernel size for AS-DBPN (default: 6 for asdbpn, None for others)")
    parser.add_argument("--checkpoint-freq", type=int, default=50, help="Frequency (in iterations) to save convergence checkpoints and update visual report (default: 50)")
    parser.add_argument("--eval-freq", type=int, default=20, help="Frequency (in iterations) to evaluate metrics during warmup (default: 20)")
    parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory to save convergence checkpoints")
    parser.add_argument("--report-dir", type=str, default=None, help="Directory to save visual report assets")
    parser.add_argument("--warmup-max-iter", type=int, default=500, help="Maximum iterations for MSE warmup (default: 500)")
    parser.add_argument("--lr-patch-size", type=int, default=None, help="Low-resolution patch size (default: 32 for 3D, 48 for 2D)")
    parser.add_argument("--stage1-lr", type=float, default=None, help="Learning rate for Stage 1 (default: 1e-4 for 3D, 5e-5 for 2D)")
    parser.add_argument("--stage2-lr", type=float, default=None, help="Learning rate for Stage 2 (default: 5e-5 for 3D, 2e-5 for 2D)")
    parser.add_argument("--stage3-lr", type=float, default=None, help="Learning rate for Stage 3 (default: 2e-5 for 3D, 1e-5 for 2D)")
    parser.add_argument("--skip-warmup", action="store_true", default=False, help="Skip MSE warmup phase (useful when resuming/fine-tuning from an existing trained model)")
    parser.add_argument("--load-model", type=str, default=None, help="Explicit path to pretrained/checkpoint model to load weights from")
    parser.add_argument("--factor", nargs="+", type=int, default=[2], help="Super-resolution scaling factor (e.g. 2 or 1 1 2) (default: 2)")
    parser.add_argument("--transfer-from", type=str, default=None, help="Explicit path to model to transfer compatible weights from (supports factor mismatch)")
    parser.add_argument("--reset-history", action="store_true", default=False, help="Reset convergence history for new run")
    parser.add_argument(
        "--balancer-anneal-iters", type=int, default=0,
        help="Deprecated no-op. Freeze/clamp/anneal mechanisms have been removed. "
             "Balancer now runs as pure continuous LOWESS weight tracking. Default: 0.")
    parser.add_argument(
        "--stage-patience", type=int, default=0,
        help="Early convergence detection per stage (Stages 1 & 2). "
             "Tracks a rolling window of the last N checkpoints across ALL metrics "
             "(PSNR, SSIM, GMSD, HFEN, corr). Computes the linear slope of each metric "
             "normalised by its window mean (unitless relative slope per checkpoint). "
             "A stage is declared converged when ALL metric slopes are below a small "
             "threshold (neither improving nor deteriorating). "
             "0 = disabled (default). Recommended: 5.")
    # ---------------------------------------------------------------
    # Speed optimisation flags (see faster_training_plan.md)
    # ---------------------------------------------------------------
    parser.add_argument(
        "--perceptual-backend", choices=["vgg", "resnet"], default="vgg",
        help="Perceptual feature extractor backend for 3D. "
             "'vgg'  = pseudo-3D VGG19 layer 6 (canonical per Avants et al. medRxiv). "
             "'resnet' = native 3D ResNet grader layer 6 (60x faster, "
             "validated as equal-or-better in Avants et al. 2023). (default: vgg)")
    parser.add_argument(
        "--balancer-freq", type=int, default=1,
        help="How many steps between full loss-component diagnostic forward passes "
             "used by the dynamic balancer. 1 = every step (original behaviour). "
             "25 = ~60%% step-time reduction by amortising the extra model+VGG "
             "forward passes. (default: 1)")
    parser.add_argument(
        "--prefetch-size", type=int, default=0,
        help="Number of batches to prefetch in a background thread while the GPU "
             "trains. Hides the ~1.7 s/step CPU data-generation stall. "
             "0 = disabled (original behaviour). 4 = recommended. (default: 0)")
    parser.add_argument(
        "--use-onecycle", action="store_true", default=False,
        help="Use a OneCycleLR schedule instead of flat learning-rate blocks. "
             "Warms up over 30%% of steps then cosine-anneals to near-zero; "
             "typically reaches equivalent quality in ~40%% fewer iterations.")
    parser.add_argument(
        "--onecycle-max-lr", type=float, default=2e-4,
        help="Peak learning rate for the OneCycleLR schedule (default: 2e-4).")
    parser.add_argument(
        "--init-l1-weight", type=float, default=None,
        help="Pre-calibrated MAE (L1) loss weight. When provided along with "
             "--init-feat-weight and --init-tv-weight, skips the auto-calibration "
             "forward pass and starts training immediately at the target 65/30/5 ratio. "
             "Compute via: w = (target_pct * target_total_loss) / median_raw_loss.")
    parser.add_argument(
        "--init-feat-weight", type=float, default=None,
        help="Pre-calibrated perceptual (feature) loss weight (see --init-l1-weight).")
    parser.add_argument(
        "--init-tv-weight", type=float, default=None,
        help="Pre-calibrated total-variation loss weight (see --init-l1-weight).")
    parser.add_argument(
        "--clip-norm", type=float, default=None,
        help="Global gradient clipping norm applied to every optimizer (Stage 1/2/3). "
             "When the ResNet perceptual backend is used, the feature extractor can "
             "gradually drive the model into a degenerate high-feature-loss / low-PSNR "
             "state during long Stage 3 runs — a slow-burn collapse observed consistently "
             "at Iter ~1500. Gradient clipping (e.g. --clip-norm 1.0) bounds each weight "
             "update's L2 norm, preventing any single large perceptual gradient from "
             "destabilising the model. Opt-in only; default=None (no clipping). "
             "Recommended value for ResNet backend Stage 3: 1.0.")
    parser.add_argument(
        "--edge-weight", type=float, default=0.0,
        help="Weight for gradient-magnitude edge loss term in hybrid_loss. "
             "Computes MSE between gradient magnitude maps of y_true and y_pred "
             "via finite differences, directly penalizing gradient inconsistency "
             "and improving GMSD metric. 0.0=disabled (default). Try 1.0-5.0.")
    parser.add_argument(
        "--gms-weight", type=float, default=0.0,
        help="Weight for differentiable GMS (Gradient Magnitude Similarity) loss. "
             "Minimizes GMSD metric directly by: (a) penalising GMS map variance "
             "(std(GMS) = GMSD metric) and (b) encouraging GMS values toward 1.0 "
             "everywhere (perfect gradient match). Directly optimises the GMSD "
             "evaluation metric. 0.0=disabled (default). Try 5.0-20.0.")
    parser.add_argument(
        "--checkerboard-weight", "--cbi-weight", dest="checkerboard_weight", type=float, default=0.0,
        help="Weight for differentiable 3D/2D alternating parity checkerboard loss (CBI). "
             "Directly penalises Nyquist-frequency (+1, -1) transposed-convolution ringing "
             "artifacts during training. 0.0=disabled (default). Try 1.0-5.0.")
    parser.add_argument(
        "--val-image", type=str, default=None,
        help="Path to real MRI volume for validation monitoring (e.g. FPA participant T1w). "
             "If None, defaults to the BLAST FPA participant if present on disk, else OASIS.")
    parser.add_argument(
        "--val-shift", nargs="+", type=int, default=None,
        help="Voxel shift from image center for validation crop (e.g. 40 0 40).")
    parser.add_argument(
        "--val-box-size", type=int, default=None,
        help="Half-width in LR voxels of validation patch crop (e.g. 32 -> 64x64 LR / 128x128 HR). "
             "If None: in 2D, defaults to the FULL brain slice (no cropping), showing the entire brain in reports; "
             "in 3D, defaults to 32 (64x64x64 LR / 128x128x128 HR).")
    parser.add_argument(
        "--start-stage", type=int, choices=[1, 2, 3], default=None,
        help="Explicitly begin training from the specified stage (e.g. 3 to skip Stage 1 & 2).")
    parser.add_argument(
        "--from-scratch", action="store_true", default=False,
        help="Build a brand new model from scratch and ignore existing refined or baseline weights.")
    args = parser.parse_args()
    
    try:
        keras.config.enable_unsafe_deserialization()
    except Exception:
        pass
        
    model_type = args.model
    batch_size = args.batch_size
    dim = args.dim

    if len(args.factor) == 1:
        factor_tuple = tuple([args.factor[0]] * dim)
    elif len(args.factor) == dim:
        factor_tuple = tuple(args.factor)
    else:
        raise ValueError(f"--factor must specify 1 or {dim} integer values, got {args.factor}")
    
    is_default_factor = (factor_tuple == tuple([2] * dim))
    factor_str = "x".join(str(f) for f in factor_tuple)

    lr_patch_size = args.lr_patch_size if args.lr_patch_size is not None else (32 if dim == 3 else 48)
    if isinstance(lr_patch_size, (list, tuple)):
        lr_patch_shape = tuple(lr_patch_size)
    else:
        lr_patch_shape = tuple([lr_patch_size] * dim)
    hr_patch_shape = tuple(p * f for p, f in zip(lr_patch_shape, factor_tuple))
    hr_patch_size = hr_patch_shape[0]

    stage1_lr = args.stage1_lr if args.stage1_lr is not None else (1e-4 if dim == 3 else 5e-5)
    stage2_lr = args.stage2_lr if args.stage2_lr is not None else (5e-5 if dim == 3 else 2e-5)
    stage3_lr = args.stage3_lr if args.stage3_lr is not None else (2e-5 if dim == 3 else 1e-5)
    
    stage1_max = args.stage1_iter
    stage2_max = args.stage2_iter
    stage3_max = args.stage3_iter
    target_pcts = {
        'mae': args.target_mae,
        'percep': args.target_percep,
        'tv': args.target_tv
    }
    custom_objects = None
            
    print(f"Initializing {model_type.upper()} {dim}D Refinement Pipeline (factor={factor_tuple})...")
    print(f"  Configuration: batch_size={batch_size}, lr_patch={lr_patch_shape} -> hr_patch={hr_patch_shape}, stage1_lr={stage1_lr}, stage2_lr={stage2_lr}, stage3_lr={stage3_lr}")
    workspace_dir = "."
    scratch_dir = os.path.join(workspace_dir, "scratch")
    os.makedirs(scratch_dir, exist_ok=True)
    
    import pandas as pd
    last_iteration = 0
    csv_log_path = os.path.join(workspace_dir, f"loss_contributions_{model_type}_{dim}d.csv" if is_default_factor else f"loss_contributions_{model_type}_{dim}d_{factor_str}.csv")
    if args.reset_history and os.path.exists(csv_log_path):
        try:
            os.remove(csv_log_path)
            print(f"Reset history requested: removed {csv_log_path}")
        except Exception as e:
            pass
    elif os.path.exists(csv_log_path):
        try:
            df = pd.read_csv(csv_log_path)
            if len(df) > 0:
                last_iteration = int(df['iteration'].iloc[-1])
                print(f"Detected last logged training iteration: {last_iteration}")
        except Exception as e:
            print(f"Could not read last iteration from CSV: {e}")

    # 1. Load Real MRI validation patches for monitoring (e.g. FPA Participant T1w)
    default_fpa = "/Users/stnava/data/blast_cohorts/BIDS/FPA/sub-BLAST022/ses-01/anat/sub-BLAST022_ses-01_run-001_T1w.nii.gz"
    if args.val_image and os.path.exists(args.val_image):
        val_img_path = args.val_image
    elif os.path.exists(default_fpa):
        val_img_path = default_fpa
    else:
        val_img_path = antspynet.get_antsxnet_data("oasis")

    if args.val_shift:
        val_shift = list(args.val_shift)
        if len(val_shift) < dim:
            val_shift = val_shift + [0] * (dim - len(val_shift))
    elif "sub-BLAST" in val_img_path or "FPA" in val_img_path:
        val_shift = [40, 0, 40] if dim == 3 else [40, 40, 25]  # Z=+25 drops slice by 15 voxels from +40
    else:
        val_shift = [0] * dim

    print(f"Loading Real MRI validation volume from: {val_img_path} (shift={val_shift})...")
    img = ants.image_read(val_img_path)
    img = ants.iMath(ants.iMath(img, 'TruncateIntensity', 0.001, 0.999), 'Normalize')

    # For 2D training with a 3D NIfTI, extract an axial slice
    if dim == 2 and img.dimension == 3:
        z_shift = int(round(val_shift[2])) if len(val_shift) > 2 else 25
        mid_z = img.shape[2] // 2 + z_shift
        mid_z = max(0, min(img.shape[2] - 1, mid_z))
        img = ants.slice_image(img, axis=2, idx=mid_z)
        print(f"  [2D mode] Extracted axial slice {mid_z} (z_shift={z_shift}) from 3D volume for 2D validation.")

    if dim == 2:
        # Crop tightly to head to eliminate excess black background
        mask = ants.get_mask(img, low_thresh=0.03, cleanup=2)
        mask_dil = ants.iMath(mask, 'MD', 2)
        img = ants.crop_image(img, mask_dil)
        h, w = img.shape
        img = ants.crop_indices(img, [0, 0], [h - (h % 2), w - (w % 2)])
        print(f"  [2D mode] Tightly cropped to head: shape={img.shape}")

    print(f"Simulating Validation Low Resolution (factor={factor_tuple})...")
    val_target_spacing = [img.spacing[i] * factor_tuple[i] for i in range(dim)]
    low_res = ants.resample_image(img, val_target_spacing, use_voxels=False, interp_type=0)
    
    if dim == 2:
        if args.val_image is None and not os.path.exists(default_fpa):
            print("Loading r16 validation image for 2D super-resolution monitoring...")
            img = ants.image_read(ants.get_data("r16"))
            img = ants.crop_image(img)
            val_target_spacing = [img.spacing[i] * factor_tuple[i] for i in range(dim)]
            low_res = ants.resample_image(img, val_target_spacing, use_voxels=False, interp_type=0)
        
    if dim == 2 and args.val_box_size is None:
        # 2D default: use the full brain slice so the visual report shows the complete brain
        lr_patch = low_res
        hr_patch = img
        print(f"  [2D mode] Using tightly cropped head slice ({img.shape}) for validation report.")
    else:
        lr_box = args.val_box_size if args.val_box_size is not None else (32 if dim == 3 else 48)
        mid_lr = [low_res.shape[i] // 2 + int(round(val_shift[i] / factor_tuple[i])) for i in range(dim)]
        mid_hr = [img.shape[i] // 2 + val_shift[i] for i in range(dim)]
        lr_patch_low = [max(0, mid_lr[i] - lr_box) for i in range(dim)]
        lr_patch_high = [min(low_res.shape[i], mid_lr[i] + lr_box) for i in range(dim)]
        lr_patch = ants.crop_indices(low_res, lr_patch_low, lr_patch_high)
        
        hr_patch_low = [max(0, mid_hr[i] - lr_box * factor_tuple[i]) for i in range(dim)]
        hr_patch_high = [min(img.shape[i], mid_hr[i] + lr_box * factor_tuple[i]) for i in range(dim)]
        hr_patch = ants.crop_indices(img, hr_patch_low, hr_patch_high)
    gt_np = hr_patch.numpy()

    # Initialize Visual Convergence Reporter
    from tests.visual_convergence_report import VisualConvergenceReporter
    ckpt_dir = args.checkpoint_dir if args.checkpoint_dir else (f"checkpoints/{model_type}_{dim}d" if is_default_factor else f"checkpoints/{model_type}_{dim}d_{factor_str}")
    rep_dir = args.report_dir if args.report_dir else (f"reports/{model_type}_{dim}d" if is_default_factor else f"reports/{model_type}_{dim}d_{factor_str}")
    html_name = f"{model_type}_{dim}d_report.html" if is_default_factor else f"{model_type}_{dim}d_{factor_str}_report.html"
    reporter = VisualConvergenceReporter(workspace_dir=workspace_dir, checkpoint_dir=ckpt_dir, report_dir=rep_dir, html_filename=html_name, reset_history=args.reset_history)
    reporter.setup_validation_patches(lr_patch, hr_patch, factor=factor_tuple)
    if not args.reset_history and len(reporter.history) > 0:
        last_iteration = max(last_iteration, int(reporter.history[-1].get("iteration", 0)))
        print(f"Detected last logged convergence iteration from history: {last_iteration}")
    
    # 2. Cache disabled by default (generating raw volumes on-the-fly)
    print("Cache disabled by default. Training volumes will be generated raw on the fly.")
    hr_base_cache = None

    # 3. Define Simulation Classes mixture
    simulation_classes = {
        "brain_procedural": 1.0 / 9.0,
        "layered": 1.0 / 9.0,
        "sinewave": 1.0 / 9.0,
        "organic_blobs": 1.0 / 9.0,
        "vessel_tubes": 1.0 / 9.0,
        "cellular_voronoi": 1.0 / 9.0,
        "geometric_phantoms": 1.0 / 9.0,
        "grid_patterns": 1.0 / 9.0,
        "fractal_noise": 1.0 / 9.0
    }
    
    # 4. Instantiate Model
    skip_stages_1_2 = True if (args.start_stage and args.start_stage >= 3) else False
    if model_type == "espcn":
        if dim == 2:
            output_model_path = os.path.join(workspace_dir, "espcn_2d_attention_refined.keras")
            best_model_path = os.path.join(workspace_dir, "espcn_2d_attention_clean_best_mdl.keras")
            custom_objects = {"PixelShuffle2D": siq.PixelShuffle2D, "LearnableScale": siq.LearnableScale}
        else:
            output_model_path = os.path.join(workspace_dir, "espcn_3d_attention_refined.keras")
            best_model_path = os.path.join(workspace_dir, "espcn_3d_attention_clean_best_mdl.keras")
            custom_objects = {"PixelShuffle3D": siq.PixelShuffle3D, "LearnableScale": siq.LearnableScale}
        
        if os.path.exists(output_model_path):
            print(f"Resuming training: loading existing refined CA-ESPCN model from {output_model_path}...")
            model = keras.models.load_model(output_model_path, custom_objects=custom_objects, compile=False)
            if last_iteration >= stage2_max:
                skip_stages_1_2 = True
        elif os.path.exists(best_model_path):
            print(f"Starting fresh: loading baseline CA-ESPCN model from {best_model_path}...")
            model = keras.models.load_model(best_model_path, custom_objects=custom_objects, compile=False)
        else:
            if dim == 2:
                print("Baseline model not found. Building a new Attention-Enhanced ESPCN 2D model...")
                model = siq.create_espcn_2d_attention(
                    input_shape=(None, None, 1),
                    factor=2,
                    n_filters=128,
                    n_res_blocks=8,
                    use_global_skip=True
                )
            else:
                print("Baseline model not found. Building a new Attention-Enhanced ESPCN 3D model...")
                model = siq.create_espcn_3d_attention(
                    input_shape=(None, None, None, 1),
                    factor=2,
                    n_filters=128,
                    n_res_blocks=8,
                    use_global_skip=True
                )
    elif model_type == "ldbpn":
        if dim == 2:
            output_model_path = os.path.join(workspace_dir, "ldbpn_2d_refined.keras")
            best_model_path = os.path.join(workspace_dir, "ldbpn_2d_best_mdl.keras")
            custom_objects = {"PixelShuffle2D": siq.PixelShuffle2D}
        else:
            output_model_path = os.path.join(workspace_dir, "ldbpn_3d_refined.keras")
            best_model_path = os.path.join(workspace_dir, "ldbpn_3d_best_mdl.keras")
            custom_objects = {"PixelShuffle3D": siq.PixelShuffle3D}
        
        if os.path.exists(output_model_path):
            print(f"Resuming training: loading existing refined L-DBPN model from {output_model_path}...")
            model = keras.models.load_model(output_model_path, custom_objects=custom_objects, compile=False)
            if last_iteration >= stage2_max:
                skip_stages_1_2 = True
        elif os.path.exists(best_model_path):
            print(f"Starting fresh: loading baseline L-DBPN model from {best_model_path}...")
            model = keras.models.load_model(best_model_path, custom_objects=custom_objects, compile=False)
        else:
            if dim == 2:
                print("Baseline model not found. Building a new Lightweight DBPN 2D model...")
                model = siq.create_ldbpn_2d(
                    input_shape=(None, None, 1),
                    factor=2,
                    n_filters=64,
                    n_stages=3
                )
            else:
                print("Baseline model not found. Building a new Lightweight DBPN 3D model...")
                model = siq.create_ldbpn_3d(
                    input_shape=(None, None, None, 1),
                    factor=2,
                    n_filters=64,
                    n_stages=3
                )
    elif model_type == "wdsr":
        if dim == 2:
            output_model_path = os.path.join(workspace_dir, "wdsr_2d_refined.keras")
            best_model_path = os.path.join(workspace_dir, "wdsr_2d_best_mdl.keras")
            custom_objects = {"PixelShuffle2D": siq.PixelShuffle2D, "LearnableScale": siq.LearnableScale}
        else:
            output_model_path = os.path.join(workspace_dir, "wdsr_3d_refined.keras")
            best_model_path = os.path.join(workspace_dir, "wdsr_3d_best_mdl.keras")
            custom_objects = {"PixelShuffle3D": siq.PixelShuffle3D, "LearnableScale": siq.LearnableScale}
        
        if os.path.exists(output_model_path):
            print(f"Resuming training: loading existing refined WDSR model from {output_model_path}...")
            model = keras.models.load_model(output_model_path, custom_objects=custom_objects, compile=False)
            if last_iteration >= stage2_max:
                skip_stages_1_2 = True
        elif os.path.exists(best_model_path):
            print(f"Starting fresh: loading baseline WDSR model from {best_model_path}...")
            model = keras.models.load_model(best_model_path, custom_objects=custom_objects, compile=False)
        else:
            if dim == 2:
                print("Baseline model not found. Building a new WDSR 2D model...")
                model = siq.create_wdsr_2d(
                    input_shape=(None, None, 1),
                    factor=2,
                    n_filters=64,
                    n_res_blocks=8,
                    expansion_ratio=4,
                    use_global_skip=True
                )
            else:
                print("Baseline model not found. Building a new WDSR 3D model...")
                model = siq.create_wdsr_3d(
                    input_shape=(None, None, None, 1),
                    factor=2,
                    n_filters=64,
                    n_res_blocks=8,
                    expansion_ratio=4,
                    use_global_skip=True
                )
    elif model_type == "rcan":
        if dim == 2:
            output_model_path = os.path.join(workspace_dir, "rcan_2d_refined.keras")
            best_model_path = os.path.join(workspace_dir, "rcan_2d_best_mdl.keras")
            custom_objects = {"PixelShuffle2D": siq.PixelShuffle2D, "LearnableScale": siq.LearnableScale}
        else:
            output_model_path = os.path.join(workspace_dir, "rcan_3d_refined.keras")
            best_model_path = os.path.join(workspace_dir, "rcan_3d_best_mdl.keras")
            custom_objects = {"PixelShuffle3D": siq.PixelShuffle3D, "LearnableScale": siq.LearnableScale}
        
        if os.path.exists(output_model_path):
            print(f"Resuming training: loading existing refined RCAN model from {output_model_path}...")
            model = keras.models.load_model(output_model_path, custom_objects=custom_objects, compile=False)
            if last_iteration >= stage2_max:
                skip_stages_1_2 = True
        elif os.path.exists(best_model_path):
            print(f"Starting fresh: loading baseline RCAN model from {best_model_path}...")
            model = keras.models.load_model(best_model_path, custom_objects=custom_objects, compile=False)
        else:
            if dim == 2:
                print("Baseline model not found. Building a new RCAN 2D model...")
                model = siq.create_rcan_2d(
                    input_shape=(None, None, 1),
                    factor=2,
                    n_filters=64,
                    n_groups=3,
                    n_blocks=4,
                    use_global_skip=True
                )
            else:
                print("Baseline model not found. Building a new RCAN 3D model...")
                model = siq.create_rcan_3d(
                    input_shape=(None, None, None, 1),
                    factor=2,
                    n_filters=64,
                    n_groups=3,
                    n_blocks=4,
                    use_global_skip=True
                )
    elif model_type == "carn":
        if dim == 2:
            output_model_path = os.path.join(workspace_dir, "carn_2d_refined.keras")
            best_model_path = os.path.join(workspace_dir, "carn_2d_best_mdl.keras")
            custom_objects = {"PixelShuffle2D": siq.PixelShuffle2D, "LearnableScale": siq.LearnableScale}
        else:
            output_model_path = os.path.join(workspace_dir, "carn_3d_refined.keras")
            best_model_path = os.path.join(workspace_dir, "carn_3d_best_mdl.keras")
            custom_objects = {"PixelShuffle3D": siq.PixelShuffle3D, "LearnableScale": siq.LearnableScale}
        
        if os.path.exists(output_model_path):
            print(f"Resuming training: loading existing refined CARN model from {output_model_path}...")
            model = keras.models.load_model(output_model_path, custom_objects=custom_objects, compile=False)
            if last_iteration >= stage2_max:
                skip_stages_1_2 = True
        elif os.path.exists(best_model_path):
            print(f"Starting fresh: loading baseline CARN model from {best_model_path}...")
            model = keras.models.load_model(best_model_path, custom_objects=custom_objects, compile=False)
        else:
            if dim == 2:
                print("Baseline model not found. Building a new CARN 2D model...")
                model = siq.create_carn_2d(
                    input_shape=(None, None, 1),
                    factor=2,
                    n_filters=64,
                    n_blocks=3,
                    use_global_skip=True
                )
            else:
                print("Baseline model not found. Building a new CARN 3D model...")
                model = siq.create_carn_3d(
                    input_shape=(None, None, None, 1),
                    factor=2,
                    n_filters=64,
                    n_blocks=3,
                    use_global_skip=True
                )
    elif model_type == "espcn-rc":
        if dim == 2:
            output_model_path = os.path.join(workspace_dir, "espcn_2d_resize_conv_refined.keras")
            best_model_path = os.path.join(workspace_dir, "espcn_2d_resize_conv_best_mdl.keras")
            custom_objects = {"LearnableScale": siq.LearnableScale}
        else:
            raise ValueError("espcn-rc is only implemented in 2D for step-artifact mitigation pilots.")
        
        if os.path.exists(output_model_path):
            print(f"Resuming training: loading existing refined ESPCN Resize Conv model from {output_model_path}...")
            model = keras.models.load_model(output_model_path, custom_objects=custom_objects, compile=False)
            if last_iteration >= stage2_max:
                skip_stages_1_2 = True
        elif os.path.exists(best_model_path):
            print(f"Starting fresh: loading baseline ESPCN Resize Conv model from {best_model_path}...")
            model = keras.models.load_model(best_model_path, custom_objects=custom_objects, compile=False)
        else:
            print("Baseline model not found. Building a new ESPCN Resize Conv 2D model...")
            model = siq.create_espcn_2d_resize_conv(
                input_shape=(None, None, 1),
                factor=2,
                n_filters=64,
                n_res_blocks=8,
                use_global_skip=True
            )
            
    elif model_type == "wdsr-rc":
        if dim == 2:
            output_model_path = os.path.join(workspace_dir, "wdsr_2d_resize_conv_refined.keras")
            best_model_path = os.path.join(workspace_dir, "wdsr_2d_resize_conv_best_mdl.keras")
            custom_objects = {"LearnableScale": siq.LearnableScale}
        else:
            raise ValueError("wdsr-rc is only implemented in 2D for step-artifact mitigation pilots.")
        
        if os.path.exists(output_model_path):
            print(f"Resuming training: loading existing refined WDSR Resize Conv model from {output_model_path}...")
            model = keras.models.load_model(output_model_path, custom_objects=custom_objects, compile=False)
            if last_iteration >= stage2_max:
                skip_stages_1_2 = True
        elif os.path.exists(best_model_path):
            print(f"Starting fresh: loading baseline WDSR Resize Conv model from {best_model_path}...")
            model = keras.models.load_model(best_model_path, custom_objects=custom_objects, compile=False)
        else:
            print("Baseline model not found. Building a new WDSR Resize Conv 2D model...")
            model = siq.create_wdsr_2d_resize_conv(
                input_shape=(None, None, 1),
                factor=2,
                n_filters=64,
                n_res_blocks=8,
                use_global_skip=True
            )
    elif model_type == "srfbn":
        if dim == 2:
            output_model_path = os.path.join(workspace_dir, "srfbn_2d_refined.keras")
            best_model_path = os.path.join(workspace_dir, "srfbn_2d_best_mdl.keras")
            custom_objects = {"LearnableScale": siq.LearnableScale, "LearnableSharpening": siq.LearnableSharpening}
        else:
            output_model_path = os.path.join(workspace_dir, "srfbn_3d_refined.keras")
            best_model_path = os.path.join(workspace_dir, "srfbn_3d_best_mdl.keras")
            custom_objects = {"LearnableScale": siq.LearnableScale, "LearnableSharpening3D": siq.LearnableSharpening3D}
        
        if os.path.exists(output_model_path):
            print(f"Resuming training: loading existing refined SRFBN model from {output_model_path}...")
            model = keras.models.load_model(output_model_path, custom_objects=custom_objects, compile=False)
            if last_iteration >= stage2_max:
                skip_stages_1_2 = True
        elif os.path.exists(best_model_path):
            print(f"Starting fresh: loading baseline SRFBN model from {best_model_path}...")
            model = keras.models.load_model(best_model_path, custom_objects=custom_objects, compile=False)
        else:
            if dim == 2:
                print("Baseline model not found. Building a new SRFBN 2D model...")
                model = siq.create_srfbn_2d(
                    input_shape=(None, None, 1),
                    factor=2,
                    n_filters=64,
                    n_steps=8,
                    use_global_skip=True
                )
            else:
                print("Baseline model not found. Building a new SRFBN 3D model...")
                model = siq.create_srfbn_3d(
                    input_shape=(None, None, None, 1),
                    factor=2,
                    n_filters=64,
                    n_steps=8,
                    use_global_skip=True
                )
    elif model_type == "asdbpn":
        if dim == 2:
            output_model_path = os.path.join(workspace_dir, "asdbpn_2d_refined.keras" if is_default_factor else f"asdbpn_2d_{factor_str}_refined.keras")
            best_model_path = os.path.join(workspace_dir, "asdbpn_2d_best_mdl.keras" if is_default_factor else f"asdbpn_2d_{factor_str}_best_mdl.keras")
            custom_objects = {"LearnableScale": siq.LearnableScale, "LearnableSharpening": siq.LearnableSharpening}
        else:
            output_model_path = os.path.join(workspace_dir, "asdbpn_3d_refined.keras" if is_default_factor else f"asdbpn_3d_{factor_str}_refined.keras")
            best_model_path = os.path.join(workspace_dir, "asdbpn_3d_best_mdl.keras" if is_default_factor else f"asdbpn_3d_{factor_str}_best_mdl.keras")
            custom_objects = {
                "LearnableScale": siq.LearnableScale,
                "LearnableSharpening3D": siq.LearnableSharpening3D,
                "TrilinearUpSampling3D": siq.TrilinearUpSampling3D
            }
        
        proj_k = args.projection_kernel_size if args.projection_kernel_size is not None else 6
        source_transfer_path = args.transfer_from if args.transfer_from else (args.load_model if (not is_default_factor and args.load_model) else None)
        
        ckpt_best_cqs = os.path.join(ckpt_dir, "asdbpn_3d_best_cqs.keras")
        ckpt_best_psnr = os.path.join(ckpt_dir, "asdbpn_3d_best_psnr.keras")
        ckpt_best_path = ckpt_best_cqs if os.path.exists(ckpt_best_cqs) else ckpt_best_psnr
        if not args.from_scratch and args.load_model and os.path.exists(args.load_model):
            print(f"Loading AS-DBPN model from explicit path: {args.load_model}...")
            model = keras.models.load_model(args.load_model, custom_objects=custom_objects, compile=False, safe_mode=False)
            if (args.start_stage and args.start_stage >= 3) or last_iteration >= stage2_max:
                skip_stages_1_2 = True
        elif not args.from_scratch and os.path.exists(output_model_path) and not args.reset_history:
            print(f"Resuming training: loading existing refined AS-DBPN model from {output_model_path}...")
            model = keras.models.load_model(output_model_path, custom_objects=custom_objects, compile=False, safe_mode=False)
            if (args.start_stage and args.start_stage >= 3) or last_iteration >= stage2_max:
                skip_stages_1_2 = True
        elif not args.from_scratch and os.path.exists(ckpt_best_path) and not args.reset_history:
            print(f"Resuming training: loading best champion checkpoint from {ckpt_best_path}...")
            model = keras.models.load_model(ckpt_best_path, custom_objects=custom_objects, compile=False, safe_mode=False)
            if (args.start_stage and args.start_stage >= 3) or last_iteration >= stage2_max:
                skip_stages_1_2 = True
        elif not args.from_scratch and source_transfer_path and os.path.exists(source_transfer_path):
            print(f"[Transfer Learning] Initializing AS-DBPN {dim}D with factor={factor_tuple} (projection_kernel_size={proj_k})...")
            if dim == 2:
                model = siq.create_asdbpn_2d(
                    input_shape=(None, None, 1),
                    factor=factor_tuple,
                    n_filters=128,
                    n_steps=4,
                    use_global_skip=True,
                    projection_kernel_size=proj_k
                )
            else:
                model = siq.create_asdbpn_3d(
                    input_shape=(None, None, None, 1),
                    factor=factor_tuple,
                    n_filters=64,
                    n_steps=4,
                    use_global_skip=True,
                    projection_kernel_size=proj_k
                )
            print(f"[Transfer Learning] Transferring compatible weights from: {source_transfer_path}...")
            src_m, _ = siq.load_siq_model(source_transfer_path)
            siq.transfer_siq_weights(src_m, model, verbose=True)
        elif not args.from_scratch and os.path.exists(best_model_path) and not args.reset_history:
            print(f"Starting fresh: loading baseline AS-DBPN model from {best_model_path}...")
            model = keras.models.load_model(best_model_path, custom_objects=custom_objects, compile=False, safe_mode=False)
        else:
            if dim == 2:
                print(f"Building a fresh AS-DBPN 2D model (factor={factor_tuple})...")
                model = siq.create_asdbpn_2d(
                    input_shape=(None, None, 1),
                    factor=factor_tuple,
                    n_filters=128,
                    n_steps=4,
                    use_global_skip=True,
                    projection_kernel_size=proj_k
                )
            else:
                print(f"Baseline model not found. Building a new AS-DBPN 3D model (factor={factor_tuple}, n_filters=64, n_steps=4, projection_kernel_size={proj_k})...")
                model = siq.create_asdbpn_3d(
                    input_shape=(None, None, None, 1),
                    factor=factor_tuple,
                    n_filters=64,
                    n_steps=4,
                    use_global_skip=True,
                    projection_kernel_size=proj_k
                )
    elif model_type == "san":
        if dim == 2:
            output_model_path = os.path.join(workspace_dir, "san_2d_refined.keras")
            best_model_path = os.path.join(workspace_dir, "san_2d_best_mdl.keras")
            custom_objects = {"PixelShuffle2D": siq.PixelShuffle2D, "LearnableScale": siq.LearnableScale}
        else:
            output_model_path = os.path.join(workspace_dir, "san_3d_refined.keras")
            best_model_path = os.path.join(workspace_dir, "san_3d_best_mdl.keras")
            custom_objects = {"PixelShuffle3D": siq.PixelShuffle3D, "LearnableScale": siq.LearnableScale}
        
        if os.path.exists(output_model_path):
            print(f"Resuming training: loading existing refined SAN model from {output_model_path}...")
            model = keras.models.load_model(output_model_path, custom_objects=custom_objects, compile=False)
            if last_iteration >= stage2_max:
                skip_stages_1_2 = True
        elif os.path.exists(best_model_path):
            print(f"Starting fresh: loading baseline SAN model from {best_model_path}...")
            model = keras.models.load_model(best_model_path, custom_objects=custom_objects, compile=False)
        else:
            if dim == 2:
                print("Baseline model not found. Building a new SAN 2D model...")
                model = siq.create_san_2d(
                    input_shape=(None, None, 1),
                    factor=2,
                    n_filters=64,
                    n_groups=3,
                    n_blocks=4,
                    use_global_skip=True
                )
            else:
                print("Baseline model not found. Building a new SAN 3D model...")
                model = siq.create_san_3d(
                    input_shape=(None, None, None, 1),
                    factor=2,
                    n_filters=64,
                    n_groups=3,
                    n_blocks=4,
                    use_global_skip=True
                )
    else: # ref-dbpn
        if dim == 2:
            output_model_path = os.path.join(workspace_dir, "ref_dbpn_2d_refined.keras")
            best_model_path = os.path.join(workspace_dir, "ref_dbpn_2d_best_mdl.keras")
        else:
            output_model_path = os.path.join(workspace_dir, "ref_dbpn_3d_refined.keras")
            best_model_path = os.path.join(workspace_dir, "exp_baseline_best.keras")
        
        if os.path.exists(output_model_path):
            print(f"Resuming training: loading existing refined Reference DBPN model from {output_model_path}...")
            model = keras.models.load_model(output_model_path, compile=False)
            if last_iteration >= stage2_max:
                skip_stages_1_2 = True
        elif os.path.exists(best_model_path):
            print(f"Starting fresh: loading baseline Reference DBPN model from {best_model_path}...")
            model = keras.models.load_model(best_model_path, compile=False)
        else:
            if dim == 2:
                print("Baseline model not found. Building a new Reference DBPN 2D model...")
                model = siq.default_dbpn(
                    strider=[2, 2],
                    dimensionality=2,
                    option="large"
                )
            else:
                print("Baseline model not found. Building a new Reference DBPN 3D model...")
                model = siq.default_dbpn(
                    strider=[2, 2, 2],
                    dimensionality=3,
                    option="large"
                )
        
    # Apply ICNR initialization when starting fresh (not resuming or transferring weights)
    if (args.from_scratch or not os.path.exists(output_model_path)) and not source_transfer_path:
        apply_icnr_initialization(model, factor=factor_tuple)
        
    # 5. Load feature extractor for perceptual loss (VGG Layers [3, 6, 9])
    if dim == 2:
        print("Loading 2D VGG feature extractor (Layers [3, 6, 9])...")
        def build_vgg_2d(inshape=[hr_patch_size, hr_patch_size], layers=[3, 6, 9]):
            inputs = keras.layers.Input(shape=(inshape[0], inshape[1], 1))
            x = keras.layers.Concatenate(axis=-1)([inputs, inputs, inputs])
            vgg19 = keras.applications.VGG19(include_top=False, weights="imagenet", input_shape=(inshape[0], inshape[1], 3))
            conv_layers = [l for l in vgg19.layers if isinstance(l, keras.layers.Conv2D)]
            outputs = [conv_layers[min(lyr, len(conv_layers)-1)].output for lyr in layers]
            feature_model = keras.Model(inputs=vgg19.inputs, outputs=outputs)
            feature_model.trainable = False
            return keras.Model(inputs=inputs, outputs=feature_model(x))
        feature_extractor = build_vgg_2d(inshape=[hr_patch_shape[0], hr_patch_shape[1]], layers=[3, 6, 9])
    else:
        if args.perceptual_backend == "resnet":
            print("Loading native 3D ResNet grader feature extractor (Layer 6) — 60x faster than VGG, "
                  "validated as equal-or-better (Avants et al. 2023 medRxiv)...")
            feature_extractor = siq.get_grader_feature_network(layer=6)
            print(f"  ResNet grader output shape: {feature_extractor.output.shape}")
        else:
            fe_inshape = list(hr_patch_shape)
            print(f"Loading pseudo-3D VGG feature extractor (Layer 6, inshape={fe_inshape}) — canonical layer from Avants et al. medRxiv paper...")
            feature_extractor = siq.pseudo_3d_vgg_features_unbiased(inshape=fe_inshape, layer=6)

    feature_extractor.trainable = False
    
    # 6. Hybrid loss function variables (using auto_weight_loss mimicking successful training)
    msq_weight_var = keras.Variable(0.0, dtype="float32")
    feat_weight_var = keras.Variable(0.0, dtype="float32")
    tv_weight_var = keras.Variable(0.0, dtype="float32")
    l1_weight_var = keras.Variable(0.0, dtype="float32")
    edge_weight_var = keras.Variable(args.edge_weight, dtype="float32")  # gradient-magnitude edge loss
    # GMS and CBI are ZERO during Stages 1 & 2 (perceptual-only curriculum).
    # They are activated only at Stage 3 start (see stage 3 setup below).
    # This prevents GMS from overwhelming the VGG perceptual signal in Stage 2
    # (GMS was 54% vs VGG 0.8% with naive simultaneous activation).
    gms_weight_var = keras.Variable(0.0, dtype="float32")   # activated at Stage 3
    cbi_weight_var = keras.Variable(0.0, dtype="float32")   # activated at Stage 3
    
    wts_csv = os.path.join(workspace_dir, f"{model_type}_{dim}d_refined_training_weights.csv" if is_default_factor else f"{model_type}_{dim}d_{factor_str}_refined_training_weights.csv")
    wts_loaded = False
    if os.path.exists(wts_csv) and not args.reset_history:
        print(f"Loading preset weights from {wts_csv}...")
        try:
            wtsdf = pd.read_csv(wts_csv)
            if 'l1' in wtsdf.columns and float(wtsdf['l1'].iloc[0]) > 1e-4:
                w_mae_init = float(wtsdf['l1'].iloc[0])
                w_percep_init = float(wtsdf['feat'].iloc[0])
                w_tv_init = float(wtsdf['tv'].iloc[0])
                wts_loaded = True
        except Exception as e:
            print(f"Could not read weights from CSV: {e}")
            wts_loaded = False
            
    # ── Provenance & Transfer Learning of Loss Weights ────────────────────────
    transferred_wts = None
    provenance_model_candidates = [
        source_transfer_path,
        args.load_model if args.load_model and os.path.exists(args.load_model) else None,
        output_model_path if os.path.exists(output_model_path) else None,
        best_model_path if os.path.exists(best_model_path) else None
    ]
    for cand in provenance_model_candidates:
        if cand:
            transferred_wts = siq.extract_siq_loss_weights(cand, verbose=True)
            if transferred_wts is not None:
                break

    if (args.init_l1_weight is not None
            and args.init_feat_weight is not None
            and args.init_tv_weight is not None):
        w_mae_init   = args.init_l1_weight
        w_percep_init = args.init_feat_weight
        w_tv_init    = args.init_tv_weight
        wts_loaded   = True
        print(f"[Pre-calibrated weights] L1={w_mae_init:.6f}  "
              f"Feat={w_percep_init:.6f}  TV={w_tv_init:.6f}  "
              f"(bypassing auto-calibration)")
    elif transferred_wts is not None:
        w_mae_init = transferred_wts.get("l1", 1.0)
        w_percep_init = transferred_wts.get("feat", 0.0)
        w_tv_init = transferred_wts.get("tv", 0.0)
        if transferred_wts.get("gms", 0.0) > 0 and args.gms_weight == 0.0:
            # Store as Stage 3 value — do NOT assign to gms_weight_var yet (Stage 2 = 0)
            args.gms_weight = transferred_wts["gms"]
        if transferred_wts.get("cbi", 0.0) > 0 and args.checkerboard_weight == 0.0:
            # Store as Stage 3 value — do NOT assign to cbi_weight_var yet (Stage 2 = 0)
            args.checkerboard_weight = transferred_wts["cbi"]
        if transferred_wts.get("edge", 0.0) > 0 and args.edge_weight == 0.0:
            args.edge_weight = transferred_wts["edge"]
            edge_weight_var.assign(args.edge_weight)
        wts_loaded = True
        print(f"[Transfer Learning] Transferred loss weights from source: "
              f"L1={w_mae_init:.6f}, Feat={w_percep_init:.6e}, TV={w_tv_init:.6f}, "
              f"GMS(Stage3)={args.gms_weight:.4f}, "
              f"CBI(Stage3)={args.checkerboard_weight:.4f}")

    if not wts_loaded:

        print("Computing automatic systematic loss weights using a sample clean training batch...")
        # Temporary generator to obtain clean patches for calibration
        temp_gen = siq.blind_sr_generator(
            hr_base_cache=None,
            batch_size=batch_size,
            lr_patch_size=lr_patch_size,
            factor=factor_tuple,
            blur_sigma_range=(0.0, 0.0),
            noise_std_range=(0.0, 0.0),
            simulation_classes=simulation_classes,
            zoom_range=(1.0, 1.0),
            use_cache=False,
            dimensionality=dim,
            use_layer2=args.use_layer2
        )
        x_init, y_init = next(temp_gen)
        x_init_t = ops.convert_to_tensor(x_init, dtype="float32")
        y_init_t = ops.convert_to_tensor(y_init, dtype="float32")
        y_pred_init = ops.stop_gradient(model(x_init_t, training=False))
        
        init_mae = float(ops.mean(ops.abs(y_init_t - y_pred_init)))
        
        # Apply same preprocessing as during training
        y_init_rn = rank_normalize_batch(y_init_t) if (args.perceptual_backend == "resnet" and dim == 3) else y_init_t
        y_pred_rn = rank_normalize_batch(y_pred_init) if (args.perceptual_backend == "resnet" and dim == 3) else y_pred_init
        f_true_init = feature_extractor(y_init_rn)
        f_pred_init = feature_extractor(y_pred_rn)
        if not isinstance(f_true_init, list):
            f_true_init = [f_true_init]
            f_pred_init = [f_pred_init]
        init_percep = sum(float(ops.mean(ops.square(ft - fp))) for ft, fp in zip(f_true_init, f_pred_init))
        
        if dim == 2:
            diff_h = ops.mean(ops.abs(y_pred_init[:, 1:, :, :] - y_pred_init[:, :-1, :, :]))
            diff_w = ops.mean(ops.abs(y_pred_init[:, :, 1:, :] - y_pred_init[:, :, :-1, :]))
            init_tv = float(diff_h + diff_w)
        else:
            diff_d = ops.mean(ops.abs(y_pred_init[:, 1:, :, :, :] - y_pred_init[:, :-1, :, :, :]))
            diff_h = ops.mean(ops.abs(y_pred_init[:, :, 1:, :, :] - y_pred_init[:, :, :-1, :, :]))
            diff_w = ops.mean(ops.abs(y_pred_init[:, :, :, 1:, :] - y_pred_init[:, :, :, :-1, :]))
            init_tv = float(diff_d + diff_h + diff_w)
            
        # Also measure GMS raw loss for calibration info (not used in weight init — fixed at args.gms_weight)
        if args.gms_weight > 0:
            if dim == 2:
                _gd2 = ops.pad(ops.abs(y_init_t[:,1:,:,:] - y_init_t[:,:-1,:,:]), [[0,0],[0,1],[0,0],[0,0]])
                _gh2 = ops.pad(ops.abs(y_init_t[:,:,1:,:] - y_init_t[:,:,:-1,:]), [[0,0],[0,0],[0,1],[0,0]])
                _gd2p = ops.pad(ops.abs(y_pred_init[:,1:,:,:] - y_pred_init[:,:-1,:,:]), [[0,0],[0,1],[0,0],[0,0]])
                _gh2p = ops.pad(ops.abs(y_pred_init[:,:,1:,:] - y_pred_init[:,:,:-1,:]), [[0,0],[0,0],[0,1],[0,0]])
                m2t = ops.sqrt(ops.square(_gd2)+ops.square(_gh2)+1e-8)
                m2p = ops.sqrt(ops.square(_gd2p)+ops.square(_gh2p)+1e-8)
            else:
                _gd2 = ops.pad(ops.abs(y_init_t[:,1:,:,:,:] - y_init_t[:,:-1,:,:,:]), [[0,0],[0,1],[0,0],[0,0],[0,0]])
                _gh2 = ops.pad(ops.abs(y_init_t[:,:,1:,:,:] - y_init_t[:,:,:-1,:,:]), [[0,0],[0,0],[0,1],[0,0],[0,0]])
                _gw2 = ops.pad(ops.abs(y_init_t[:,:,:,1:,:] - y_init_t[:,:,:,:-1,:]), [[0,0],[0,0],[0,0],[0,1],[0,0]])
                _gd2p = ops.pad(ops.abs(y_pred_init[:,1:,:,:,:] - y_pred_init[:,:-1,:,:,:]), [[0,0],[0,1],[0,0],[0,0],[0,0]])
                _gh2p = ops.pad(ops.abs(y_pred_init[:,:,1:,:,:] - y_pred_init[:,:,:-1,:,:]), [[0,0],[0,0],[0,1],[0,0],[0,0]])
                _gw2p = ops.pad(ops.abs(y_pred_init[:,:,:,1:,:] - y_pred_init[:,:,:,:-1,:]), [[0,0],[0,0],[0,0],[0,1],[0,0]])
                m2t = ops.sqrt(ops.square(_gd2)+ops.square(_gh2)+ops.square(_gw2)+1e-8)
                m2p = ops.sqrt(ops.square(_gd2p)+ops.square(_gh2p)+ops.square(_gw2p)+1e-8)
            _gms_c = 0.0026
            _gms_map = (2.0*m2t*m2p + _gms_c) / (ops.square(m2t)+ops.square(m2p)+_gms_c)
            _gms_mean = ops.mean(_gms_map, keepdims=True)
            init_gms = float(ops.mean(ops.square(_gms_map - _gms_mean)) + ops.mean(ops.square(1.0 - _gms_map)))
            print(f"Initial raw GMS loss: {init_gms:.6f} (fixed weight={args.gms_weight:.4f}, contribution={init_gms*args.gms_weight:.4f})")
        else:
            init_gms = 0.0

        print(f"Initial raw loss components: MAE={init_mae:.6f}, Perceptual={init_percep:.6f}, TV={init_tv:.6f}")
        
        t_mae = target_pcts.get('mae', 30.0)
        t_percep = target_pcts.get('percep', 65.0)
        t_tv = target_pcts.get('tv', 5.0)
        total_t = t_mae + t_percep + t_tv
        p_mae = t_mae / total_t
        p_percep = t_percep / total_t
        p_tv = t_tv / total_t
        
        target_total_loss = args.target_total_loss
        if args.anneal_iter > 0:
            # Start with 100% MAE and 0% perceptual / TV, smoothly ramping up over anneal_iter iterations
            w_mae_init = (1.0 * target_total_loss) / max(1e-8, init_mae)
            w_percep_init = 0.0
            w_tv_init = 0.0
        else:
            w_mae_init = (p_mae * target_total_loss) / max(1e-8, init_mae)
            w_percep_init = (p_percep * target_total_loss) / max(1e-8, init_percep)
            w_tv_init = (p_tv * target_total_loss) / max(1e-8, init_tv)
        
        pd.DataFrame([[0.0, w_percep_init, w_tv_init, w_mae_init]], columns=["msq", "feat", "tv", "l1"]).to_csv(wts_csv, index=False)
        print(f"Saved initial systematic weights to {wts_csv}")
    
    msq_weight_var.assign(0.0)
    l1_weight_var.assign(w_mae_init)
    feat_weight_var.assign(w_percep_init)
    tv_weight_var.assign(w_tv_init)

    print(f"Systematic dynamic weight starting values: MSE={float(ops.convert_to_numpy(msq_weight_var)):.4f}, MAE (L1)={float(ops.convert_to_numpy(l1_weight_var)):.6f}, Feat={float(ops.convert_to_numpy(feat_weight_var)):.8e}, TV={float(ops.convert_to_numpy(tv_weight_var)):.6f}")

    # ---------------------------------------------------------------
    # Preprocessing wrapper for the perceptual feature extractor.
    # VGG has Rescaling(255, -127.5) baked in; ResNet was trained on
    # rank-intensity-normalised [0,1] inputs (ants.rank_intensity).
    # fe_preprocess applies that transform before every feature call
    # so the ResNet sees the correct input distribution.
    # ---------------------------------------------------------------
    if args.perceptual_backend == "resnet" and dim == 3:
        print("[ResNet] Rank-normalization preprocessing enabled "
              "(matching antspyt1w resnet_grader training distribution).")
        def _call_fe(tensor):
            return feature_extractor(rank_normalize_batch(tensor), training=False)
    else:
        def _call_fe(tensor):
            return feature_extractor(tensor)

    def _compute_cbi_block(target, dim, factor_tuple):
        up_axes = [i for i, f in enumerate(factor_tuple) if f > 1]
        if len(up_axes) == 0:
            up_axes = list(range(dim))

        if len(up_axes) == 1:
            ax = up_axes[0]
            if dim == 2:
                if ax == 0:
                    return (target[:, 1:, :, :] - target[:, :-1, :, :]) / 2.0
                else:
                    return (target[:, :, 1:, :] - target[:, :, :-1, :]) / 2.0
            else: # dim == 3
                if ax == 0:
                    return (target[:, 1:, :, :, :] - target[:, :-1, :, :, :]) / 2.0
                elif ax == 1:
                    return (target[:, :, 1:, :, :] - target[:, :, :-1, :, :]) / 2.0
                else:
                    return (target[:, :, :, 1:, :] - target[:, :, :, :-1, :]) / 2.0
        elif len(up_axes) == 2:
            if dim == 2:
                c00 = target[:, :-1, :-1, :]
                c10 = target[:, 1:,  :-1, :]
                c01 = target[:, :-1, 1:,  :]
                c11 = target[:, 1:,  1:,  :]
                return (c00 - c10 - c01 + c11) / 4.0
            else: # dim == 3
                ax0, ax1 = up_axes[0], up_axes[1]
                if ax0 == 0 and ax1 == 1:
                    c00 = target[:, :-1, :-1, :, :]
                    c10 = target[:, 1:,  :-1, :, :]
                    c01 = target[:, :-1, 1:,  :, :]
                    c11 = target[:, 1:,  1:,  :, :]
                elif ax0 == 0 and ax1 == 2:
                    c00 = target[:, :-1, :, :-1, :]
                    c10 = target[:, 1:,  :, :-1, :]
                    c01 = target[:, :-1, :, 1:,  :]
                    c11 = target[:, 1:,  :, 1:,  :]
                else: # ax0 == 1 and ax1 == 2
                    c00 = target[:, :, :-1, :-1, :]
                    c10 = target[:, :, 1:,  :-1, :]
                    c01 = target[:, :, :-1, 1:,  :]
                    c11 = target[:, :, 1:,  1:,  :]
                return (c00 - c10 - c01 + c11) / 4.0
        else: # 3 axes (isotropic 3D)
            c000 = target[:, :-1, :-1, :-1, :]
            c100 = target[:, 1:,  :-1, :-1, :]
            c010 = target[:, :-1, 1:,  :-1, :]
            c110 = target[:, 1:,  1:,  :-1, :]
            c001 = target[:, :-1, :-1, 1:,  :]
            c101 = target[:, 1:,  :-1, 1:,  :]
            c011 = target[:, :-1, 1:,  1:,  :]
            c111 = target[:, 1:,  1:,  1:,  :]
            return (c000 - c100 - c010 + c110 - c001 + c101 + c011 - c111) / 8.0

    def hybrid_loss(y_true, y_pred):
        # L2 Loss (MSE)
        squared_diff = ops.square(y_true - y_pred)
        l2_term = ops.mean(squared_diff, axis=list(range(1, len(y_true.shape))))
        
        # L1 Loss (MAE) for sharper edges
        abs_diff = ops.abs(y_true - y_pred)
        l1_term = ops.mean(abs_diff, axis=list(range(1, len(y_true.shape))))
        
        # Perceptual Loss (multi-layer) — fe_preprocess applied internally by _call_fe
        f_true_list = _call_fe(y_true)
        f_pred_list = _call_fe(y_pred)
        if not isinstance(f_true_list, list):
            f_true_list = [f_true_list]
            f_pred_list = [f_pred_list]
        feat_term = 0.0
        for f_t, f_p in zip(f_true_list, f_pred_list):
            feat_term += ops.mean(ops.square(f_t - f_p), axis=list(range(1, len(f_t.shape))))
        
        # Total Variation Loss
        if dim == 2:
            diff_h = ops.mean(ops.abs(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]), axis=list(range(1, len(y_pred.shape)-1)))
            diff_w = ops.mean(ops.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]), axis=list(range(1, len(y_pred.shape)-1)))
            tv_term = diff_h + diff_w
        else:
            diff_d = ops.mean(ops.abs(y_pred[:, 1:, :, :, :] - y_pred[:, :-1, :, :, :]), axis=list(range(1, len(y_pred.shape)-1)))
            diff_h = ops.mean(ops.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :]), axis=list(range(1, len(y_pred.shape)-1)))
            diff_w = ops.mean(ops.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :]), axis=list(range(1, len(y_pred.shape)-1)))
            tv_term = diff_d + diff_h + diff_w
        
        # Gradient Magnitude Edge Loss — penalises gradient magnitude mismatch
        # (directly reduces GMSD by enforcing consistent sharpness everywhere)
        if dim == 2:
            gy_true = ops.mean(ops.abs(y_true[:, 1:, :, :] - y_true[:, :-1, :, :]), axis=list(range(1, len(y_true.shape))))
            gx_true = ops.mean(ops.abs(y_true[:, :, 1:, :] - y_true[:, :, :-1, :]), axis=list(range(1, len(y_true.shape))))
            gy_pred = ops.mean(ops.abs(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]), axis=list(range(1, len(y_pred.shape))))
            gx_pred = ops.mean(ops.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]), axis=list(range(1, len(y_pred.shape))))
            edge_term = ops.square(gy_true - gy_pred) + ops.square(gx_true - gx_pred)
        else:
            gd_true = ops.mean(ops.abs(y_true[:, 1:, :, :, :] - y_true[:, :-1, :, :, :]), axis=list(range(1, len(y_true.shape))))
            gh_true = ops.mean(ops.abs(y_true[:, :, 1:, :, :] - y_true[:, :, :-1, :, :]), axis=list(range(1, len(y_true.shape))))
            gw_true = ops.mean(ops.abs(y_true[:, :, :, 1:, :] - y_true[:, :, :, :-1, :]), axis=list(range(1, len(y_true.shape))))
            gd_pred = ops.mean(ops.abs(y_pred[:, 1:, :, :, :] - y_pred[:, :-1, :, :, :]), axis=list(range(1, len(y_pred.shape))))
            gh_pred = ops.mean(ops.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :]), axis=list(range(1, len(y_pred.shape))))
            gw_pred = ops.mean(ops.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :]), axis=list(range(1, len(y_pred.shape))))
            edge_term = (ops.square(gd_true - gd_pred) + ops.square(gh_true - gh_pred) +
                         ops.square(gw_true - gw_pred))

        # Differentiable GMS loss — directly minimises GMSD evaluation metric
        # GMS(x,y) = (2*|∇x|*|∇y| + c) / (|∇x|² + |∇y|² + c) ∈ [0,1]
        # We minimise: var(GMS) + MSE(GMS, 1.0) — reduces std and pushes toward perfect match
        if args.gms_weight > 1e-8:
            gms_c = 0.0026
            if dim == 2:
                _gd_t = ops.pad(ops.abs(y_true[:, 1:, :, :] - y_true[:, :-1, :, :]), [[0,0],[0,1],[0,0],[0,0]])
                _gh_t = ops.pad(ops.abs(y_true[:, :, 1:, :] - y_true[:, :, :-1, :]), [[0,0],[0,0],[0,1],[0,0]])
                _gd_p = ops.pad(ops.abs(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]), [[0,0],[0,1],[0,0],[0,0]])
                _gh_p = ops.pad(ops.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]), [[0,0],[0,0],[0,1],[0,0]])
                m_t = ops.sqrt(ops.square(_gd_t) + ops.square(_gh_t) + 1e-8)
                m_p = ops.sqrt(ops.square(_gd_p) + ops.square(_gh_p) + 1e-8)
            else:
                _gd_t = ops.pad(ops.abs(y_true[:, 1:, :, :, :] - y_true[:, :-1, :, :, :]), [[0,0],[0,1],[0,0],[0,0],[0,0]])
                _gh_t = ops.pad(ops.abs(y_true[:, :, 1:, :, :] - y_true[:, :, :-1, :, :]), [[0,0],[0,0],[0,1],[0,0],[0,0]])
                _gw_t = ops.pad(ops.abs(y_true[:, :, :, 1:, :] - y_true[:, :, :, :-1, :]), [[0,0],[0,0],[0,0],[0,1],[0,0]])
                _gd_p = ops.pad(ops.abs(y_pred[:, 1:, :, :, :] - y_pred[:, :-1, :, :, :]), [[0,0],[0,1],[0,0],[0,0],[0,0]])
                _gh_p = ops.pad(ops.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :]), [[0,0],[0,0],[0,1],[0,0],[0,0]])
                _gw_p = ops.pad(ops.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :]), [[0,0],[0,0],[0,0],[0,1],[0,0]])
                m_t = ops.sqrt(ops.square(_gd_t) + ops.square(_gh_t) + ops.square(_gw_t) + 1e-8)
                m_p = ops.sqrt(ops.square(_gd_p) + ops.square(_gh_p) + ops.square(_gw_p) + 1e-8)
            gms_map = (2.0 * m_t * m_p + gms_c) / (ops.square(m_t) + ops.square(m_p) + gms_c)
            gms_mean = ops.mean(gms_map, keepdims=True)
            gms_term = ops.mean(ops.square(gms_map - gms_mean)) + ops.mean(ops.square(1.0 - gms_map))
        else:
            gms_term = ops.zeros_like(l1_term)
        # Differentiable Alternating Parity Filter (Checkerboard Penalty)
        # Factor-aware: 1D (-1)^i/2, 2D (-1)^(i+j)/4, 3D (-1)^(i+j+k)/8 along upsampled axes
        # Penalises the Nyquist (+1, -1) deconvolution ringing artifact directly during backprop
        if args.checkerboard_weight > 1e-8:
            _cb_target = y_pred - y_true
            cbi_block = _compute_cbi_block(_cb_target, dim, factor_tuple)
            cbi_term = ops.mean(ops.abs(cbi_block), axis=list(range(1, len(y_pred.shape))))
        else:
            cbi_term = ops.zeros_like(l1_term)

        return (l2_term * msq_weight_var + 
                l1_term * l1_weight_var + 
                feat_term * feat_weight_var + 
                tv_term * tv_weight_var +
                edge_term * edge_weight_var +
                gms_term * gms_weight_var +
                cbi_term * cbi_weight_var)

    def get_current_loss_weights():
        return {
            "msq": float(ops.convert_to_numpy(msq_weight_var)),
            "l1": float(ops.convert_to_numpy(l1_weight_var)),
            "feat": float(ops.convert_to_numpy(feat_weight_var)),
            "tv": float(ops.convert_to_numpy(tv_weight_var)),
            "edge": float(ops.convert_to_numpy(edge_weight_var)),
            "gms": float(ops.convert_to_numpy(gms_weight_var)),
            "cbi": float(ops.convert_to_numpy(cbi_weight_var)),
        }

    def check_stage_convergence(history_deque, stage_name, patience):
        """
        Multi-metric slope convergence check.

        Fits a linear slope to the last `patience` checkpoint entries for each
        of PSNR, SSIM, GMSD, HFEN, corr.  Each slope is normalised by the
        window mean of that metric (unitless relative slope per checkpoint).

        Convergence = ALL |normalised_slope| < 0.005  (0.5 % per checkpoint).
        This is direction-agnostic: it fires when the model has genuinely
        stabilised, even if PSNR is drifting slightly while GMSD improves.

        Returns (converged: bool, slope_summary: str).
        """
        if len(history_deque) < patience:
            return False, "insufficient history"
        window = list(history_deque)[-patience:]
        xs = np.arange(len(window), dtype=np.float32)
        metrics = {
            "PSNR":  [e.get("val_psnr",  0.0) for e in window],
            "SSIM":  [e.get("val_ssim",  0.0) for e in window],
            "GMSD":  [e.get("val_gmsd",  1.0) for e in window],
            "HFEN":  [e.get("val_hfen",  1.0) for e in window],
            "Corr":  [e.get("val_corr",  0.0) for e in window],
        }
        slopes = {}
        for name, vals in metrics.items():
            ys = np.array(vals, dtype=np.float32)
            mean_y = np.mean(np.abs(ys))
            if mean_y < 1e-6:
                slopes[name] = 0.0
                continue
            # linear regression slope via least-squares
            slope = float(np.polyfit(xs, ys, 1)[0])
            slopes[name] = slope / mean_y   # normalised: change-per-ckpt / mean
        summary = "  ".join(f"{k}:{v:+.4f}" for k, v in slopes.items())
        threshold = 0.005   # 0.5% per checkpoint — all must be below this
        converged = all(abs(v) < threshold for v in slopes.values())
        return converged, summary

    def print_loss_components(stage_name, iteration, max_iter, x_batch, y_batch, loss):
        # Convert y_batch to a Keras tensor to prevent PyTorch/numpy subtraction errors
        y_true_tensor = ops.convert_to_tensor(y_batch, dtype="float32")
        y_pred_batch = ops.stop_gradient(model(x_batch, training=False))
        
        # Compute terms using ops
        l2_val = float(ops.mean(ops.square(y_true_tensor - y_pred_batch)))
        l1_val = float(ops.mean(ops.abs(y_true_tensor - y_pred_batch)))
        
        f_true_batch = _call_fe(y_true_tensor)
        f_pred_batch = _call_fe(y_pred_batch)
        if not isinstance(f_true_batch, list):
            f_true_batch = [f_true_batch]
            f_pred_batch = [f_pred_batch]
        feat_val = 0.0
        for f_t, f_p in zip(f_true_batch, f_pred_batch):
            feat_val += float(ops.mean(ops.square(f_t - f_p)))
        
        if dim == 2:
            diff_h = ops.mean(ops.abs(y_pred_batch[:, 1:, :, :] - y_pred_batch[:, :-1, :, :]))
            diff_w = ops.mean(ops.abs(y_pred_batch[:, :, 1:, :] - y_pred_batch[:, :, :-1, :]))
            tv_val = float(diff_h + diff_w)
            # Edge loss: gradient magnitude MSE
            gy_t = ops.mean(ops.abs(y_true_tensor[:, 1:, :, :] - y_true_tensor[:, :-1, :, :]))
            gx_t = ops.mean(ops.abs(y_true_tensor[:, :, 1:, :] - y_true_tensor[:, :, :-1, :]))
            gy_p = ops.mean(ops.abs(y_pred_batch[:, 1:, :, :] - y_pred_batch[:, :-1, :, :]))
            gx_p = ops.mean(ops.abs(y_pred_batch[:, :, 1:, :] - y_pred_batch[:, :, :-1, :]))
            edge_val = float(ops.square(gy_t - gy_p) + ops.square(gx_t - gx_p))
        else:
            diff_d = ops.mean(ops.abs(y_pred_batch[:, 1:, :, :, :] - y_pred_batch[:, :-1, :, :, :]))
            diff_h = ops.mean(ops.abs(y_pred_batch[:, :, 1:, :, :] - y_pred_batch[:, :, :-1, :, :]))
            diff_w = ops.mean(ops.abs(y_pred_batch[:, :, :, 1:, :] - y_pred_batch[:, :, :, :-1, :]))
            tv_val = float(diff_d + diff_h + diff_w)
            # Edge loss: gradient magnitude MSE
            gd_t = ops.mean(ops.abs(y_true_tensor[:, 1:, :, :, :] - y_true_tensor[:, :-1, :, :, :]))
            gh_t = ops.mean(ops.abs(y_true_tensor[:, :, 1:, :, :] - y_true_tensor[:, :, :-1, :, :]))
            gw_t = ops.mean(ops.abs(y_true_tensor[:, :, :, 1:, :] - y_true_tensor[:, :, :, :-1, :]))
            gd_p = ops.mean(ops.abs(y_pred_batch[:, 1:, :, :, :] - y_pred_batch[:, :-1, :, :, :]))
            gh_p = ops.mean(ops.abs(y_pred_batch[:, :, 1:, :, :] - y_pred_batch[:, :, :-1, :, :]))
            gw_p = ops.mean(ops.abs(y_pred_batch[:, :, :, 1:, :] - y_pred_batch[:, :, :, :-1, :]))
            edge_val = float(ops.square(gd_t - gd_p) + ops.square(gh_t - gh_p) + ops.square(gw_t - gw_p))
        
        # Weighted terms (using ops.convert_to_numpy to avoid PyTorch warnings)
        w_l2 = l2_val * float(ops.convert_to_numpy(msq_weight_var))
        w_l1 = l1_val * float(ops.convert_to_numpy(l1_weight_var))
        w_feat = feat_val * float(ops.convert_to_numpy(feat_weight_var))
        w_tv = tv_val * float(ops.convert_to_numpy(tv_weight_var))
        w_edge = edge_val * float(ops.convert_to_numpy(edge_weight_var))

        # GMS term — compute inline (mirrors hybrid_loss GMS block)
        gms_weight_val = float(ops.convert_to_numpy(gms_weight_var))
        if gms_weight_val > 1e-8:
            gms_c = 0.0026
            if dim == 2:
                _gd_t = ops.pad(ops.abs(y_true_tensor[:, 1:, :, :] - y_true_tensor[:, :-1, :, :]), [[0,0],[0,1],[0,0],[0,0]])
                _gh_t = ops.pad(ops.abs(y_true_tensor[:, :, 1:, :] - y_true_tensor[:, :, :-1, :]), [[0,0],[0,0],[0,1],[0,0]])
                _gd_p = ops.pad(ops.abs(y_pred_batch[:, 1:, :, :] - y_pred_batch[:, :-1, :, :]), [[0,0],[0,1],[0,0],[0,0]])
                _gh_p = ops.pad(ops.abs(y_pred_batch[:, :, 1:, :] - y_pred_batch[:, :, :-1, :]), [[0,0],[0,0],[0,1],[0,0]])
                m_t = ops.sqrt(ops.square(_gd_t) + ops.square(_gh_t) + 1e-8)
                m_p = ops.sqrt(ops.square(_gd_p) + ops.square(_gh_p) + 1e-8)
            else:
                _gd_t = ops.pad(ops.abs(y_true_tensor[:, 1:, :, :, :] - y_true_tensor[:, :-1, :, :, :]), [[0,0],[0,1],[0,0],[0,0],[0,0]])
                _gh_t = ops.pad(ops.abs(y_true_tensor[:, :, 1:, :, :] - y_true_tensor[:, :, :-1, :, :]), [[0,0],[0,0],[0,1],[0,0],[0,0]])
                _gw_t = ops.pad(ops.abs(y_true_tensor[:, :, :, 1:, :] - y_true_tensor[:, :, :, :-1, :]), [[0,0],[0,0],[0,0],[0,1],[0,0]])
                _gd_p = ops.pad(ops.abs(y_pred_batch[:, 1:, :, :, :] - y_pred_batch[:, :-1, :, :, :]), [[0,0],[0,1],[0,0],[0,0],[0,0]])
                _gh_p = ops.pad(ops.abs(y_pred_batch[:, :, 1:, :, :] - y_pred_batch[:, :, :-1, :, :]), [[0,0],[0,0],[0,1],[0,0],[0,0]])
                _gw_p = ops.pad(ops.abs(y_pred_batch[:, :, :, 1:, :] - y_pred_batch[:, :, :, :-1, :]), [[0,0],[0,0],[0,0],[0,1],[0,0]])
                m_t = ops.sqrt(ops.square(_gd_t) + ops.square(_gh_t) + ops.square(_gw_t) + 1e-8)
                m_p = ops.sqrt(ops.square(_gd_p) + ops.square(_gh_p) + ops.square(_gw_p) + 1e-8)
            gms_map = (2.0 * m_t * m_p + gms_c) / (ops.square(m_t) + ops.square(m_p) + gms_c)
            gms_mean = ops.mean(gms_map, keepdims=True)
            gms_raw = float(ops.mean(ops.square(gms_map - gms_mean)) + ops.mean(ops.square(1.0 - gms_map)))
        else:
            gms_raw = 0.0
        w_gms = gms_raw * gms_weight_val

        # CBI (Checkerboard) loss raw calculation
        cbi_weight_val = float(ops.convert_to_numpy(cbi_weight_var))
        if cbi_weight_val > 1e-8:
            _cb_target = y_pred_batch - y_true_tensor
            _cbi_block = _compute_cbi_block(_cb_target, dim, factor_tuple)
            cbi_raw = float(ops.mean(ops.abs(_cbi_block)))
        else:
            cbi_raw = 0.0
        w_cbi = cbi_raw * cbi_weight_val

        total_calculated = w_l2 + w_l1 + w_feat + w_tv + w_edge + w_gms + w_cbi
        # Avoid division by zero
        denom = total_calculated if total_calculated > 1e-8 else 1.0

        pct_l2 = w_l2 / denom * 100
        pct_l1 = w_l1 / denom * 100
        pct_feat = w_feat / denom * 100
        pct_tv = w_tv / denom * 100
        pct_edge = w_edge / denom * 100
        pct_gms = w_gms / denom * 100
        pct_cbi = w_cbi / denom * 100

        print(f"{stage_name} Iter {iteration:03d}/{max_iter} - Loss: {loss:.6f}")
        print(f"  [Loss Components] Raw: L2={l2_val:.6f}, L1={l1_val:.6f}, Feat={feat_val:.6f}, TV={tv_val:.6f}, GMS={gms_raw:.6f}, CBI={cbi_raw:.6f}, Edge={edge_val:.6f}")
        print(f"  [Loss Contributions] MSE={w_l2:.4f} ({pct_l2:.1f}%), L1={w_l1:.4f} ({pct_l1:.1f}%), Feat={w_feat:.4f} ({pct_feat:.1f}%), TV={w_tv:.4f} ({pct_tv:.1f}%), GMS={w_gms:.4f} ({pct_gms:.1f}%), CBI={w_cbi:.4f} ({pct_cbi:.1f}%), Edge={w_edge:.4f} ({pct_edge:.1f}%)")
        print(f"  [Loss Weights] L1={w_l1/l1_val if l1_val > 1e-8 else 0.0:.6f}, Feat={w_feat/feat_val if feat_val > 1e-8 else 0.0:.6f}, TV={w_tv/tv_val if tv_val > 1e-8 else 0.0:.6f}, GMS={gms_weight_val:.4f}, CBI={cbi_weight_val:.4f}, Edge={args.edge_weight:.4f}")
        
        # Log to CSV
        csv_log_path = os.path.join(workspace_dir, f"loss_contributions_{model_type}_{dim}d.csv" if is_default_factor else f"loss_contributions_{model_type}_{dim}d_{factor_str}.csv")
        try:
            with open(csv_log_path, "a") as f:
                f.write(f"{stage_name},{iteration},{loss:.6f},{l2_val:.6f},{l1_val:.6f},{feat_val:.6f},{tv_val:.6f},"
                        f"{w_l2:.6f},{w_l1:.6f},{w_feat:.6f},{w_tv:.6f},{pct_l2:.2f},{pct_l1:.2f},{pct_feat:.2f},{pct_tv:.2f}\n")
        except Exception as e:
            print(f"  [Warning] Failed to write loss contributions to CSV: {e}")

        # Update preset weights file
        try:
            pd.DataFrame([get_current_loss_weights()]).to_csv(wts_csv, index=False)
        except Exception as e:
            pass

    best_val_loss = float("inf")
    
    # Initialize loss contributions CSV file and history tracker
    tracker = LossHistoryTracker(window_size=args.smooth_window)
    csv_log_path = os.path.join(workspace_dir, f"loss_contributions_{model_type}_{dim}d.csv" if is_default_factor else f"loss_contributions_{model_type}_{dim}d_{factor_str}.csv")
    if last_iteration == 0:
        with open(csv_log_path, "w") as f:
            f.write("stage,iteration,loss,l2_raw,l1_raw,feat_raw,tv_raw,w_l2,w_l1,w_feat,w_tv,pct_l2,pct_l1,pct_feat,pct_tv\n")
    else:
        if not os.path.exists(csv_log_path):
            with open(csv_log_path, "w") as f:
                f.write("stage,iteration,loss,l2_raw,l1_raw,feat_raw,tv_raw,w_l2,w_l1,w_feat,w_tv,pct_l2,pct_l1,pct_feat,pct_tv\n")
        else:
            # Populate tracker history from CSV if resuming
            try:
                df_csv = pd.read_csv(csv_log_path)
                sub_df = df_csv.tail(args.smooth_window)
                for idx, row in sub_df.iterrows():
                    tracker.add(int(row['iteration']), float(row['l1_raw']), float(row['feat_raw']), float(row['tv_raw']))
                print(f"Pre-populated tracker with {len(tracker.iterations)} iterations of history from {csv_log_path}")
            except Exception as e:
                print(f"Failed to pre-populate tracker from CSV: {e}")

    def step_dynamic_balancer(iteration, x_batch, y_batch):
        """Update dynamic loss weights, with lazy raw-loss computation.

        When ``--balancer-freq N`` is set (N > 1), the expensive extra model
        forward pass and feature-extractor forward passes are only run every
        N steps. The LOWESS-smoothed weight update still runs every
        ``--update-freq`` steps, using the last known raw component values.
        This amortises ~3.6 s of per-step overhead by a factor of N.
        """
        # ── Expensive diagnostic path (model + feature forward) ──────────
        if iteration % args.balancer_freq == 0:
            y_true_tensor = ops.convert_to_tensor(y_batch, dtype="float32")
            y_pred_batch = ops.stop_gradient(model(x_batch, training=False))
            raw_l1 = float(ops.mean(ops.abs(y_true_tensor - y_pred_batch)))

            f_true_batch = _call_fe(y_true_tensor)
            f_pred_batch = _call_fe(y_pred_batch)
            if isinstance(f_true_batch, list):
                raw_feat = sum(float(ops.mean(ops.square(ft - fp))) for ft, fp in zip(f_true_batch, f_pred_batch))
            else:
                raw_feat = float(ops.mean(ops.square(f_true_batch - f_pred_batch)))

            if dim == 2:
                diff_h = ops.mean(ops.abs(y_pred_batch[:, 1:, :, :] - y_pred_batch[:, :-1, :, :]))
                diff_w = ops.mean(ops.abs(y_pred_batch[:, :, 1:, :] - y_pred_batch[:, :, :-1, :]))
                raw_tv = float(diff_h + diff_w)
            else:
                diff_d = ops.mean(ops.abs(y_pred_batch[:, 1:, :, :, :] - y_pred_batch[:, :-1, :, :, :]))
                diff_h = ops.mean(ops.abs(y_pred_batch[:, :, 1:, :, :] - y_pred_batch[:, :, :-1, :, :]))
                diff_w = ops.mean(ops.abs(y_pred_batch[:, :, :, 1:, :] - y_pred_batch[:, :, :, :-1, :]))
                raw_tv = float(diff_d + diff_h + diff_w)

            tracker.add(iteration, raw_l1, raw_feat, raw_tv)

        # ── Weight-update path (cheap, every update_freq steps) ──────────
        # Purely updates L1/Feat/TV weights via LOWESS-smoothed target tracking.
        # No hidden freezes, clamps, or anneals — the caller controls everything
        # via explicit --init-*-weight and --dampening flags.
        if iteration % args.update_freq == 0:
            current_w = {
                'mae': float(ops.convert_to_numpy(l1_weight_var)),
                'percep': float(ops.convert_to_numpy(feat_weight_var)),
                'tv': float(ops.convert_to_numpy(tv_weight_var))
            }
            new_w, smoothed_losses = get_smoothed_losses_and_weights(
                tracker, target_pcts, iteration, current_w,
                target_total_loss=args.target_total_loss,
                beta_damp=args.dampening
            )
            l1_weight_var.assign(new_w['mae'])
            feat_weight_var.assign(new_w['percep'])
            tv_weight_var.assign(new_w['tv'])


    if last_iteration == 0 and not args.skip_warmup:
        import time
        print("\n=======================================================")
        print("Dedicated MSE-Only Warmup Phase (Targeting Bilinear Parity)")
        print("=======================================================")
        
        # Calculate target Bilinear PSNR on validation patch
        lr_patch_temp = ants.image_clone(lr_patch)
        hr_patch_temp = ants.image_clone(hr_patch)
        lr_patch_temp.set_spacing([hr_patch_temp.spacing[i] * factor_tuple[i] for i in range(dim)])
        hr_patch_temp.set_spacing(hr_patch_temp.spacing)

        target_psnr = float(reporter.bilinear_metrics.get("psnr", 25.0))
        print(f"[Warmup Gate] Bilinear baseline target PSNR: {target_psnr:.4f} dB")
        
        # Compile model with pure MSE loss for warmup
        warmup_lr = 1e-4 if model_type in ["ref-dbpn", "ldbpn", "asdbpn"] else 5e-5
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=warmup_lr), loss="mse")
        
        # Instantiate generator for warmup (using clean mixed geometries)
        train_gen_warmup = siq.blind_sr_generator(
            hr_base_cache=None,
            batch_size=batch_size,
            lr_patch_size=lr_patch_size,
            factor=factor_tuple,
            blur_sigma_range=(0.0, 0.0),
            noise_std_range=(0.0, 0.0),
            simulation_classes=simulation_classes,
            zoom_range=(0.8, 1.2),
            use_cache=False,
            dimensionality=dim,
            use_layer2=args.use_layer2
        )
        
        warmup_max_iter = args.warmup_max_iter
        start_time_warmup = time.time()
        warmup_completed = False
        
        # Record initial pre-training baseline at iteration 0 (if not already recorded)
        if not any(r.get("iteration") == 0 for r in reporter.history):
            try:
                reporter.record_checkpoint(model, 0, "Initial (Pre-Warmup)", 1.0, is_convergence_step=True, loss_weights=get_current_loss_weights())
            except Exception as e:
                print(f"[Warning] Failed to record initial state checkpoint: {e}")
        
        best_warmup_psnr = -1.0
        iters_since_best = 0
        for warmup_iter in range(1, warmup_max_iter + 1):
            x_batch, y_batch = next(train_gen_warmup)
            mse_loss = model.train_on_batch(x_batch, y_batch)
            
            is_eval = (warmup_iter % args.eval_freq == 0 or warmup_iter == 1)
            is_ckpt = (warmup_iter % args.checkpoint_freq == 0)
            
            if is_eval or is_ckpt:
                entry = reporter.record_checkpoint(model, warmup_iter, "Warmup", mse_loss, is_convergence_step=is_ckpt, loss_weights=get_current_loss_weights())
                val_psnr = entry.get("val_psnr", 0.0)
                val_ssim = entry.get("val_ssim", 0.0)
                print(f"[Warmup] Iter {warmup_iter:04d}/{warmup_max_iter} - MSE Loss: {mse_loss:.6f} - Val PSNR: {val_psnr:.2f} dB (Target: {target_psnr:.2f} dB) - Val SSIM: {val_ssim:.4f}")
                
                # Check stopping condition: surpassed Bilinear parity
                if val_psnr >= target_psnr:
                    elapsed_warmup = time.time() - start_time_warmup
                    print(f"\n[Warmup Gate] Parity with Bilinear reached! Val PSNR ({val_psnr:.2f} dB) >= Target ({target_psnr:.2f} dB)")
                    print(f"[Warmup Gate] Reached target in {warmup_iter} iterations ({elapsed_warmup:.1f}s). Saving checkpoint and exiting warmup phase.")
                    model.save(output_model_path)
                    warmup_completed = True
                    break
                    
                # Track plateau (patience of 100 iterations after reaching at least 250 iterations)
                if val_psnr > best_warmup_psnr + 0.02:
                    best_warmup_psnr = val_psnr
                    iters_since_best = 0
                else:
                    iters_since_best += args.eval_freq
                    if warmup_iter >= 250 and iters_since_best >= 100:
                        print(f"\n[Warmup Gate] PSNR plateaued near {val_psnr:.2f} dB (peak: {best_warmup_psnr:.2f} dB, no >0.02 dB gain in {iters_since_best} iters). Exiting warmup to begin refinement.")
                        model.save(output_model_path)
                        warmup_completed = True
                        break
                    
        if not warmup_completed:
            print(f"\n[Warmup Gate] Finished max warmup iterations ({warmup_max_iter}) without reaching target PSNR. Proceeding to normal stages.")
            model.save(output_model_path)
    elif args.skip_warmup:
        base_iter = stage2_max if (skip_stages_1_2 and stage2_max > 0) else 0
        if not any(r.get("iteration") == base_iter for r in reporter.history):
            try:
                base_stage = "Stage 2 Completion" if (skip_stages_1_2 and stage2_max > 0) else "Initial (Warm-Started)"
                reporter.record_checkpoint(model, base_iter, base_stage, 0.01, is_convergence_step=True, loss_weights=get_current_loss_weights())
                print(f"[Convergence Checkpoint] Recorded warm-started baseline at Step {base_iter} ({base_stage}).")
            except Exception as e:
                print(f"[Warning] Failed to record warm-started baseline: {e}")

    skip_stage_1 = (args.start_stage and args.start_stage >= 2) or last_iteration >= stage1_max
    if not skip_stages_1_2 and not skip_stage_1:
        # ==============================================================
        # Stage 1: Warmup & Adaptation on Clean Mixed Classes (Iter 1-100)
        print("\n=======================================================")
        print(f"Stage 1: Adaptation Phase (Clean Mixed Geometries) (Iter 1-100)")
        print("=======================================================")
        
        if model_type in ["espcn", "espcn-rc"]:
            set_core_trainable(model, trainable=False)
        
        train_gen_clean = prefetch_generator(
            siq.blind_sr_generator(
                hr_base_cache=None,
                batch_size=batch_size,
                lr_patch_size=lr_patch_size,
                factor=factor_tuple,
                blur_sigma_range=(0.0, 0.0),
                noise_std_range=(0.0, 0.0),
                simulation_classes=simulation_classes,
                zoom_range=(0.75, 1.3),
                use_cache=False,
                dimensionality=dim,
                use_layer2=args.use_layer2
            ),
            maxsize=args.prefetch_size,
        )

        if args.use_onecycle:
            stage1_schedule = OneCycleLR(max_lr=args.onecycle_max_lr, total_steps=stage1_max)
            print(f"[OneCycleLR] Stage 1: max_lr={args.onecycle_max_lr:.2e}, total_steps={stage1_max}")
            _opt_kw = {"clipnorm": args.clip_norm} if args.clip_norm is not None else {}
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=stage1_schedule, **_opt_kw), loss=hybrid_loss)
        else:
            _opt_kw = {"clipnorm": args.clip_norm} if args.clip_norm is not None else {}
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=stage1_lr, **_opt_kw), loss=hybrid_loss)
        if args.clip_norm is not None:
            print(f"[Gradient Clipping] Stage 1 optimizer: clipnorm={args.clip_norm}")
        
        for iteration in range(max(1, last_iteration + 1), stage1_max + 1):
            x_batch, y_batch = next(train_gen_clean)
            loss = model.train_on_batch(x_batch, y_batch)
            step_dynamic_balancer(iteration, x_batch, y_batch)
            
            # Print iteration log immediately
            print(f"Stage 1 Iter {iteration:03d}/{stage1_max} - Loss: {loss:.6f}")
            
            # Track and log heavy loss components and monitor metrics every checkpoint_freq iterations
            if iteration % args.checkpoint_freq == 0 or iteration == 1 or iteration == max(1, last_iteration + 1) or iteration == stage1_max:
                print_loss_components("Stage 1", iteration, stage1_max, x_batch, y_batch, loss)
                # Save actual training batch inputs sent to model.train_on_batch
                try:
                    os.makedirs(os.path.join(scratch_dir, "training_samples"), exist_ok=True)
                    x_img_actual = ants.from_numpy(np.squeeze(x_batch[0]))
                    y_img_actual = ants.from_numpy(np.squeeze(y_batch[0]))
                    actual_lr_png = os.path.join(scratch_dir, "training_samples", f"stage1_iter_{iteration}_lr_input.png")
                    actual_hr_png = os.path.join(scratch_dir, "training_samples", f"stage1_iter_{iteration}_hr_target.png")
                    if dim == 2:
                        ants.plot(x_img_actual, filename=actual_lr_png, title=f"Actual LR Input Iter {iteration}")
                        ants.plot(y_img_actual, filename=actual_hr_png, title=f"Actual HR Target Iter {iteration}")
                    else:
                        ants.plot(x_img_actual, filename=actual_lr_png, title=f"Actual LR Input Iter {iteration}", axis=2)
                        ants.plot(y_img_actual, filename=actual_hr_png, title=f"Actual HR Target Iter {iteration}", axis=2)
                    print(f"  --> Saved actual training batch images to {actual_lr_png} and {actual_hr_png}")
                except Exception as e:
                    print(f"  [Warning] Failed to save actual training batch images: {e}")

                is_ckpt = (iteration % args.checkpoint_freq == 0 or iteration == stage1_max)
                entry = reporter.record_checkpoint(model, iteration, "Stage 1 Adaptation", loss, is_convergence_step=is_ckpt, loss_weights=get_current_loss_weights())
                
                if loss < best_val_loss:
                    best_val_loss = loss
                    model.save(output_model_path)
                
                # Early convergence detection for Stage 1 — multi-metric slope check.
                # Guard: skip slope history update on the SAME iteration as a balancer
                # update, because the weight shift produces a transient metric change
                # that corrupts the slope signal.
                _balancer_fired_s1 = (iteration % args.balancer_freq == 0)
                if args.stage_patience > 0 and is_ckpt and entry is not None:
                    if not hasattr(reporter, "_s1_ckpt_history"):
                        reporter._s1_ckpt_history = []
                    if not _balancer_fired_s1:
                        reporter._s1_ckpt_history.append(entry)
                        converged, slope_summary = check_stage_convergence(
                            reporter._s1_ckpt_history, "Stage 1", args.stage_patience)
                        print(f"  [Stage 1 Convergence] slopes: {slope_summary}")
                        if converged:
                            print(f"\n[Stage 1 Early Stop] All metric slopes flat over "
                                  f"{args.stage_patience} checkpoints ({args.stage_patience * args.checkpoint_freq} iters). "
                                  f"Advancing to Stage 2.")
                            break
                    else:
                        print(f"  [Stage 1 Convergence] skipped (balancer fired this step)")

        # ==============================================================
        # Stage 2: Joint Fine-Tuning with Rician Noise (Iter 101-2000)
        # ==============================================================
        print("\n=======================================================")
        print(f"Stage 2: Robustness Fine-Tuning Phase (Iter 101-2000)")
        print("=======================================================")
        
        if os.path.exists(output_model_path):
            print(f"Loading best Stage 1 checkpoint from {output_model_path}...")
            model = keras.models.load_model(output_model_path, custom_objects=custom_objects, compile=False)

        if model_type in ["espcn", "espcn-rc"]:
            set_core_trainable(model, trainable=True)

        train_gen_robust = prefetch_generator(
            siq.blind_sr_generator(
                hr_base_cache=None,
                batch_size=batch_size,
                lr_patch_size=lr_patch_size,
                factor=factor_tuple,
                blur_sigma_range=(0.0, 0.0),
                noise_std_range=(0.0, 0.02),
                use_rician_noise=True,
                simulation_classes=simulation_classes,
                zoom_range=(0.75, 1.3),
                use_cache=False,
                dimensionality=dim,
                use_layer2=args.use_layer2
            ),
            maxsize=args.prefetch_size,
        )

        stage2_steps = stage2_max - stage1_max
        if args.use_onecycle:
            stage2_schedule = OneCycleLR(max_lr=args.onecycle_max_lr, total_steps=stage2_steps)
            print(f"[OneCycleLR] Stage 2: max_lr={args.onecycle_max_lr:.2e}, total_steps={stage2_steps}")
            _opt_kw = {"clipnorm": args.clip_norm} if args.clip_norm is not None else {}
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=stage2_schedule, **_opt_kw), loss=hybrid_loss)
        else:
            _opt_kw = {"clipnorm": args.clip_norm} if args.clip_norm is not None else {}
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=stage2_lr, **_opt_kw), loss=hybrid_loss)
        if args.clip_norm is not None:
            print(f"[Gradient Clipping] Stage 2 optimizer: clipnorm={args.clip_norm}")

        start_iter = max(stage1_max + 1, last_iteration + 1)
        for iteration in range(start_iter, stage2_max + 1):
            x_batch, y_batch = next(train_gen_robust)
            loss = model.train_on_batch(x_batch, y_batch)
            step_dynamic_balancer(iteration, x_batch, y_batch)
            
            # Print iteration log immediately
            print(f"Stage 2 Iter {iteration:03d}/{stage2_max} - Loss: {loss:.6f}")
            
            # Track and log heavy loss components and monitor metrics every checkpoint_freq iterations
            if iteration % args.checkpoint_freq == 0 or iteration == stage1_max + 1 or iteration == start_iter or iteration == stage2_max:
                print_loss_components("Stage 2", iteration, stage2_max, x_batch, y_batch, loss)
                # Save actual training batch inputs sent to model.train_on_batch
                try:
                    os.makedirs(os.path.join(scratch_dir, "training_samples"), exist_ok=True)
                    x_img_actual = ants.from_numpy(np.squeeze(x_batch[0]))
                    y_img_actual = ants.from_numpy(np.squeeze(y_batch[0]))
                    actual_lr_png = os.path.join(scratch_dir, "training_samples", f"stage2_iter_{iteration}_lr_input.png")
                    actual_hr_png = os.path.join(scratch_dir, "training_samples", f"stage2_iter_{iteration}_hr_target.png")
                    if dim == 2:
                        ants.plot(x_img_actual, filename=actual_lr_png, title=f"Actual LR Input Iter {iteration}")
                        ants.plot(y_img_actual, filename=actual_hr_png, title=f"Actual HR Target Iter {iteration}")
                    else:
                        ants.plot(x_img_actual, filename=actual_lr_png, title=f"Actual LR Input Iter {iteration}", axis=2)
                        ants.plot(y_img_actual, filename=actual_hr_png, title=f"Actual HR Target Iter {iteration}", axis=2)
                    print(f"  --> Saved actual training batch images to {actual_lr_png} and {actual_hr_png}")
                except Exception as e:
                    print(f"  [Warning] Failed to save actual training batch images: {e}")

                is_ckpt = (iteration % args.checkpoint_freq == 0 or iteration == stage2_max)
                entry = reporter.record_checkpoint(model, iteration, "Stage 2 Robustness", loss, is_convergence_step=is_ckpt, loss_weights=get_current_loss_weights())
                
                if loss < best_val_loss:
                    best_val_loss = loss
                    model.save(output_model_path)
                
                # Early convergence detection for Stage 2 — multi-metric slope check.
                # Guard: skip slope history update on the SAME iteration as a balancer
                # update, because the weight shift produces a transient metric change
                # that corrupts the slope signal.
                _balancer_fired_s2 = (iteration % args.balancer_freq == 0)
                if args.stage_patience > 0 and is_ckpt and entry is not None:
                    if not hasattr(reporter, "_s2_ckpt_history"):
                        reporter._s2_ckpt_history = []
                    if not _balancer_fired_s2:
                        reporter._s2_ckpt_history.append(entry)
                        converged, slope_summary = check_stage_convergence(
                            reporter._s2_ckpt_history, "Stage 2", args.stage_patience)
                        print(f"  [Stage 2 Convergence] slopes: {slope_summary}")
                        if converged:
                            print(f"\n[Stage 2 Early Stop] All metric slopes flat over "
                                  f"{args.stage_patience} checkpoints ({args.stage_patience * args.checkpoint_freq} iters). "
                                  f"Advancing to Stage 3.")
                            break
                    else:
                        print(f"  [Stage 2 Convergence] skipped (balancer fired this step)")
    else:
        print("\nSkipping Stage 1 & Stage 2 (already refined). Proceeding directly to Stage 3 (Dedicated Refinement)...")

    # ==============================================================
    # Stage 3: Dedicated Refinement Strategy (Iter 2001-5000)
    # ==============================================================
    print("\n=======================================================")
    print(f"Stage 3: Dedicated Refinement Phase (High-Fidelity Brain Focus) (Iter 2001-5000)")
    print("=======================================================")
    
    if skip_stages_1_2:
        if model is None:
            model = keras.models.load_model(output_model_path, custom_objects=custom_objects, compile=False)
        if model_type in ["espcn", "espcn-rc"]:
            set_core_trainable(model, trainable=True)
    else:
        if os.path.exists(output_model_path):
            print(f"Loading best Stage 2 checkpoint from {output_model_path}...")
            model = keras.models.load_model(output_model_path, custom_objects=custom_objects, compile=False)
            if model_type in ["espcn", "espcn-rc"]:
                set_core_trainable(model, trainable=True)

    refinement_classes = {
        "brain_procedural": 1.0 / 9.0,
        "layered": 1.0 / 9.0,
        "sinewave": 1.0 / 9.0,
        "organic_blobs": 1.0 / 9.0,
        "vessel_tubes": 1.0 / 9.0,
        "cellular_voronoi": 1.0 / 9.0,
        "geometric_phantoms": 1.0 / 9.0,
        "grid_patterns": 1.0 / 9.0,
        "fractal_noise": 1.0 / 9.0
    }

    train_gen_refine = prefetch_generator(
        siq.blind_sr_generator(
            hr_base_cache=None,
            batch_size=batch_size,
            lr_patch_size=lr_patch_size,
            factor=factor_tuple,
            blur_sigma_range=(0.0, 0.0),
            noise_std_range=(0.0, 0.01),
            use_rician_noise=True,
            simulation_classes=refinement_classes,
            zoom_range=(0.75, 1.3),
            use_cache=False,
            dimensionality=dim,
            use_layer2=args.use_layer2
        ),
        maxsize=args.prefetch_size,
    )

    stage3_steps = stage3_max - stage2_max
    if args.use_onecycle:
        stage3_schedule = OneCycleLR(max_lr=args.onecycle_max_lr * 0.5, total_steps=stage3_steps)
        print(f"[OneCycleLR] Stage 3: max_lr={args.onecycle_max_lr * 0.5:.2e}, total_steps={stage3_steps}")
        _opt_kw = {"clipnorm": args.clip_norm} if args.clip_norm is not None else {}
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=stage3_schedule, **_opt_kw), loss=hybrid_loss)
    else:
        _opt_kw = {"clipnorm": args.clip_norm} if args.clip_norm is not None else {}
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=stage3_lr, **_opt_kw), loss=hybrid_loss)
    if args.clip_norm is not None:
        print(f"[Gradient Clipping] Stage 3 optimizer: clipnorm={args.clip_norm} "
              f"(ResNet collapse prevention — Iter ~1500 without clipping)")
    best_val_loss = float("inf")

    # === Curriculum transition: activate CBI for checkerboard suppression ===
    # Stage 2: GMS=0, CBI=0 — VGG Feat (65%) guides spatial + perceptual learning.
    # Stage 3: CBI activated only. GMS stays at 0 permanently.
    # Rationale: GMS (gradient magnitude similarity) is an evaluation METRIC, not a loss.
    # Used as a loss, it blurs edges: blurred SR edges spanning a sub-pixel displacement
    # score higher on GMS than sharp edges at the correct position. VGG Feat already
    # enforces perceptual sharpness without blurring. GMS is reported in metrics but
    # NEVER used as a training signal. gms_weight_var stays 0.0 throughout all stages.

    cbi_weight_var.assign(args.checkerboard_weight)
    print(f"[Stage 3 Curriculum] CBI activated: {args.checkerboard_weight:.4f} | GMS=0 (excluded: blurs edges)")
    print(f"[Stage 3 Curriculum] Loss regime: L1 + VGG-Feat + TV + CBI (GMS is evaluation-only)")


    if skip_stages_1_2 and last_iteration < stage2_max:
        start_iter = stage2_max + 1
    else:
        start_iter = max(stage2_max + 1, last_iteration + 1)
    for iteration in range(start_iter, stage3_max + 1):
        x_batch, y_batch = next(train_gen_refine)
        loss = model.train_on_batch(x_batch, y_batch)
        step_dynamic_balancer(iteration, x_batch, y_batch)
        
        # Print iteration log immediately
        print(f"Stage 3 Iter {iteration:03d}/{stage3_max} - Loss: {loss:.6f}")
        
        # Track and log heavy loss components and monitor metrics every checkpoint_freq iterations
        if iteration % args.checkpoint_freq == 0 or iteration == stage2_max + 1 or iteration == start_iter or iteration == stage3_max:
            print_loss_components("Stage 3", iteration, stage3_max, x_batch, y_batch, loss)
            # Save actual training batch inputs sent to model.train_on_batch
            try:
                os.makedirs(os.path.join(scratch_dir, "training_samples"), exist_ok=True)
                x_img_actual = ants.from_numpy(np.squeeze(x_batch[0]))
                y_img_actual = ants.from_numpy(np.squeeze(y_batch[0]))
                actual_lr_png = os.path.join(scratch_dir, "training_samples", f"stage3_iter_{iteration}_lr_input.png")
                actual_hr_png = os.path.join(scratch_dir, "training_samples", f"stage3_iter_{iteration}_hr_target.png")
                if dim == 2:
                    ants.plot(x_img_actual, filename=actual_lr_png, title=f"Actual LR Input Iter {iteration}")
                    ants.plot(y_img_actual, filename=actual_hr_png, title=f"Actual HR Target Iter {iteration}")
                else:
                    ants.plot(x_img_actual, filename=actual_lr_png, title=f"Actual LR Input Iter {iteration}", axis=2)
                    ants.plot(y_img_actual, filename=actual_hr_png, title=f"Actual HR Target Iter {iteration}", axis=2)
                print(f"  --> Saved actual training batch images to {actual_lr_png} and {actual_hr_png}")
            except Exception as e:
                print(f"  [Warning] Failed to save actual training batch images: {e}")

            is_ckpt = (iteration % args.checkpoint_freq == 0 or iteration == stage3_max)
            entry = reporter.record_checkpoint(model, iteration, "Stage 3 Refinement", loss, is_convergence_step=is_ckpt, loss_weights=get_current_loss_weights())
            
            if loss < best_val_loss:
                best_val_loss = loss
                model.save(output_model_path)

    print(f"{model_type.upper()} {dim}D Refinement Pipeline Complete.")

if __name__ == "__main__":
    main()

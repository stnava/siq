import os
import sys

# Configure Keras to use PyTorch backend for GPU MPS/CUDA acceleration
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import keras
from keras import ops
import ants
import antspynet
import siq

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

def get_smoothed_losses_and_weights(tracker, target_pcts, current_iteration, original_weight_sum, current_weights, beta_damp=0.98):
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
    t_mae /= total_t
    t_percep /= total_t
    t_tv /= total_t
    
    w_mae_raw = t_mae / smooth_mae
    w_percep_raw = t_percep / smooth_percep
    w_tv_raw = t_tv / smooth_tv
    
    sum_raw = w_mae_raw + w_percep_raw + w_tv_raw
    if sum_raw > 1e-8:
        w_mae_tgt = original_weight_sum * (w_mae_raw / sum_raw)
        w_percep_tgt = original_weight_sum * (w_percep_raw / sum_raw)
        w_tv_tgt = original_weight_sum * (w_tv_raw / sum_raw)
    else:
        w_mae_tgt = current_weights['mae']
        w_percep_tgt = current_weights['percep']
        w_tv_tgt = current_weights['tv']
        
    new_mae = beta_damp * current_weights['mae'] + (1.0 - beta_damp) * w_mae_tgt
    new_percep = beta_damp * current_weights['percep'] + (1.0 - beta_damp) * w_percep_tgt
    new_tv = beta_damp * current_weights['tv'] + (1.0 - beta_damp) * w_tv_tgt
    
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
    parser.add_argument("model", choices=["espcn", "ldbpn", "ref-dbpn", "wdsr", "rcan", "carn", "espcn-rc", "wdsr-rc"], default="espcn", nargs="?", help="Model type to refine (default: espcn)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for training (default: 1)")
    parser.add_argument("--dim", type=int, choices=[2, 3], default=3, help="Dimensionality (2 or 3) (default: 3)")
    parser.add_argument("--stage1-iter", type=int, default=100, help="Max iterations for Stage 1 (default: 100)")
    parser.add_argument("--stage2-iter", type=int, default=2000, help="Max iterations for Stage 2 (default: 2000)")
    parser.add_argument("--stage3-iter", type=int, default=5000, help="Max iterations for Stage 3 (default: 5000)")
    parser.add_argument("--target-percep", type=float, default=65.0, help="Target perceptual loss percentage contribution (default: 65.0)")
    parser.add_argument("--target-mae", type=float, default=30.0, help="Target MAE loss percentage contribution (default: 30.0)")
    parser.add_argument("--target-tv", type=float, default=5.0, help="Target TV loss percentage contribution (default: 5.0)")
    parser.add_argument("--dampening", type=float, default=0.98, help="Dampening factor beta for weight transition (default: 0.98)")
    parser.add_argument("--smooth-window", type=int, default=100, help="LOWESS smoothing window size (default: 100)")
    parser.add_argument("--update-freq", type=int, default=10, help="Weight update frequency in iterations (default: 10)")
    args = parser.parse_args()
    
    model_type = args.model
    batch_size = args.batch_size
    dim = args.dim
    stage1_max = args.stage1_iter
    stage2_max = args.stage2_iter
    stage3_max = args.stage3_iter
    target_pcts = {
        'mae': args.target_mae,
        'percep': args.target_percep,
        'tv': args.target_tv
    }
    custom_objects = None
            
    print(f"Initializing {model_type.upper()} {dim}D Refinement Pipeline...")
    scratch_dir = "/Users/stnava/.gemini/antigravity-cli/brain/bf9e3239-711d-4a46-8dd4-8a3f33959db5/scratch"
    workspace_dir = "."
    
    import pandas as pd
    last_iteration = 0
    csv_log_path = os.path.join(workspace_dir, f"loss_contributions_{model_type}_{dim}d.csv")
    if os.path.exists(csv_log_path):
        try:
            df = pd.read_csv(csv_log_path)
            if len(df) > 0:
                last_iteration = int(df['iteration'].iloc[-1])
                print(f"Detected last logged training iteration: {last_iteration}")
        except Exception as e:
            print(f"Could not read last iteration from CSV: {e}")

    # 1. Load Real MRI (OASIS) validation patches for monitoring
    print("Loading Real MRI (OASIS) validation volume...")
    img_path = antspynet.get_antsxnet_data("oasis")
    img = ants.image_read(img_path)
    
    print("Simulating Validation Low Resolution...")
    low_res = ants.resample_image(img, [s*2 for s in img.spacing], use_voxels=False, interp_type=0)
    
    if dim == 2:
        print("Loading r16 validation image for 2D super-resolution monitoring...")
        img = ants.image_read(ants.get_data("r16"))
        img = ants.crop_image(img)
        low_res = ants.resample_image(img, [s*2 for s in img.spacing], use_voxels=False, interp_type=0)
        
    mid_lr = [s//2 for s in low_res.shape]
    lr_patch = ants.crop_indices(low_res, [m - 24 for m in mid_lr], [m + 24 for m in mid_lr])
    
    mid_hr = [s//2 for s in img.shape]
    hr_patch = ants.crop_indices(img, [m - 48 for m in mid_hr], [m + 48 for m in mid_hr])
    gt_np = hr_patch.numpy()
    
    # 2. Cache disabled by default (generating raw volumes on-the-fly)
    print("Cache disabled by default. Training volumes will be generated raw on the fly.")
    hr_base_cache = None

    # 3. Define Simulation Classes mixture
    simulation_classes = {
        "brain_procedural": 0.35,
        "layered": 0.25,
        "sinewave": 0.25,
        "organic_blobs": 0.15
    }
    
    # 4. Instantiate Model
    skip_stages_1_2 = False
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
            if last_iteration >= 2000:
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
            if last_iteration >= 2000:
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
            if last_iteration >= 2000:
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
            if last_iteration >= 2000:
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
            if last_iteration >= 2000:
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
            if last_iteration >= 2000:
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
            if last_iteration >= 2000:
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
            if last_iteration >= 2000:
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
        
    # Apply ICNR initialization when starting fresh (not resuming a refined run)
    if not os.path.exists(output_model_path):
        apply_icnr_initialization(model, factor=2)
        
    # 5. Load feature extractor for perceptual loss (VGG Layers [3, 6, 9])
    if dim == 2:
        print("Loading 2D VGG feature extractor (Layers [3, 6, 9])...")
        def build_vgg_2d(inshape=[96, 96], layers=[3, 6, 9]):
            inputs = keras.layers.Input(shape=(inshape[0], inshape[1], 1))
            x = keras.layers.Concatenate(axis=-1)([inputs, inputs, inputs])
            vgg19 = keras.applications.VGG19(include_top=False, weights="imagenet", input_shape=(inshape[0], inshape[1], 3))
            conv_layers = [l for l in vgg19.layers if isinstance(l, keras.layers.Conv2D)]
            outputs = [conv_layers[min(lyr, len(conv_layers)-1)].output for lyr in layers]
            feature_model = keras.Model(inputs=vgg19.inputs, outputs=outputs)
            feature_model.trainable = False
            return keras.Model(inputs=inputs, outputs=feature_model(x))
        feature_extractor = build_vgg_2d(inshape=[96, 96], layers=[3, 6, 9])
    else:
        print("Loading pseudo-3D VGG feature extractors (Layers [3, 6, 9])...")
        fe_3 = siq.pseudo_3d_vgg_features_unbiased(inshape=[96, 96, 96], layer=3)
        fe_6 = siq.pseudo_3d_vgg_features_unbiased(inshape=[96, 96, 96], layer=6)
        fe_9 = siq.pseudo_3d_vgg_features_unbiased(inshape=[96, 96, 96], layer=9)
        
        inputs = keras.layers.Input(shape=(96, 96, 96, 1))
        o3 = fe_3(inputs)
        o6 = fe_6(inputs)
        o9 = fe_9(inputs)
        feature_extractor = keras.Model(inputs=inputs, outputs=[o3, o6, o9])
        
    feature_extractor.trainable = False
    
    # 6. Hybrid loss function variables (using auto_weight_loss mimicking successful training)
    msq_weight_var = keras.Variable(0.0, dtype="float32")
    feat_weight_var = keras.Variable(0.0, dtype="float32")
    tv_weight_var = keras.Variable(0.0, dtype="float32")
    l1_weight_var = keras.Variable(0.0, dtype="float32")
    
    wts_csv = os.path.join(workspace_dir, f"{model_type}_{dim}d_refined_training_weights.csv")
    wts_loaded = False
    if os.path.exists(wts_csv):
        print(f"Loading preset weights from {wts_csv}...")
        try:
            wtsdf = pd.read_csv(wts_csv)
            wts = [float(wtsdf['msq'].iloc[0]), float(wtsdf['feat'].iloc[0]), float(wtsdf['tv'].iloc[0])]
            wts_loaded = True
        except Exception as e:
            print(f"Could not read weights from CSV: {e}")
            
    if not wts_loaded:
        print("Computing automatic loss weights using a sample clean training batch...")
        # Temporary generator to obtain clean patches for calibration
        temp_gen = siq.blind_sr_generator(
            hr_base_cache=None,
            batch_size=1,
            lr_patch_size=48,
            factor=2,
            blur_sigma_range=(0.0, 0.0),
            noise_std_range=(0.0, 0.0),
            simulation_classes=simulation_classes,
            zoom_range=(1.0, 1.0),
            use_cache=False,
            dimensionality=dim
        )
        x_init, y_init = next(temp_gen)
        x_init_t = ops.convert_to_tensor(x_init, dtype="float32")
        y_init_t = ops.convert_to_tensor(y_init, dtype="float32")
        
        wts = auto_weight_loss_multi(
            model,
            feature_extractor,
            x_init_t,
            y_init_t,
            feature=2.0,
            tv=0.1,
            verbose=True
        )
        print(f"Automatic weights computed: MSE={wts[0]}, Feat={wts[1]}, TV={wts[2]}")
        pd.DataFrame([[wts[0], wts[1], wts[2]]], columns=["msq", "feat", "tv"]).to_csv(wts_csv, index=False)
        print(f"Saved weights to {wts_csv}")
        
    # Generate a sample batch to compute initial raw loss values for calibration / weight scaling
    temp_gen = siq.blind_sr_generator(
        hr_base_cache=None,
        batch_size=1,
        lr_patch_size=48,
        factor=2,
        blur_sigma_range=(0.0, 0.0),
        noise_std_range=(0.0, 0.0),
        simulation_classes=simulation_classes,
        zoom_range=(1.0, 1.0),
        use_cache=False,
        dimensionality=dim
    )
    x_init, y_init = next(temp_gen)
    x_init_t = ops.convert_to_tensor(x_init, dtype="float32")
    y_init_t = ops.convert_to_tensor(y_init, dtype="float32")
    y_pred_init = ops.stop_gradient(model(x_init_t, training=False))
    
    init_mae = float(ops.mean(ops.abs(y_init_t - y_pred_init)))
    
    f_true_init = feature_extractor(y_init_t)
    f_pred_init = feature_extractor(y_pred_init)
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
        
    print(f"Initial raw loss components: MAE={init_mae:.6f}, Perceptual={init_percep:.6f}, TV={init_tv:.6f}")
    
    original_weight_sum = (0.5 * wts[1]) + wts[1] + (0.1 * wts[2])
    
    # Starting weights scaled to target percentages
    w_mae_init_raw = (target_pcts['mae'] / 100.0) / max(1e-8, init_mae)
    w_percep_init_raw = (target_pcts['percep'] / 100.0) / max(1e-8, init_percep)
    w_tv_init_raw = (target_pcts['tv'] / 100.0) / max(1e-8, init_tv)
    
    init_sum = w_mae_init_raw + w_percep_init_raw + w_tv_init_raw
    w_mae_init = original_weight_sum * (w_mae_init_raw / init_sum)
    w_percep_init = original_weight_sum * (w_percep_init_raw / init_sum)
    w_tv_init = original_weight_sum * (w_tv_init_raw / init_sum)
    
    msq_weight_var.assign(0.0)
    l1_weight_var.assign(w_mae_init)
    feat_weight_var.assign(w_percep_init)
    tv_weight_var.assign(w_tv_init)
    
    print(f"Custom dynamic weight starting values: MSE={msq_weight_var.value}, MAE (L1)={l1_weight_var.value}, Feat={feat_weight_var.value}, TV={tv_weight_var.value}")

    def hybrid_loss(y_true, y_pred):
        # L2 Loss (MSE)
        squared_diff = ops.square(y_true - y_pred)
        l2_term = ops.mean(squared_diff, axis=list(range(1, len(y_true.shape))))
        
        # L1 Loss (MAE) for sharper edges
        abs_diff = ops.abs(y_true - y_pred)
        l1_term = ops.mean(abs_diff, axis=list(range(1, len(y_true.shape))))
        
        # Perceptual Loss (multi-layer)
        f_true_list = feature_extractor(y_true)
        f_pred_list = feature_extractor(y_pred)
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
        
        return (l2_term * msq_weight_var + 
                l1_term * l1_weight_var + 
                feat_term * feat_weight_var + 
                tv_term * tv_weight_var)

    def print_loss_components(stage_name, iteration, max_iter, x_batch, y_batch, loss):
        # Convert y_batch to a Keras tensor to prevent PyTorch/numpy subtraction errors
        y_true_tensor = ops.convert_to_tensor(y_batch, dtype="float32")
        y_pred_batch = ops.stop_gradient(model(x_batch, training=False))
        
        # Compute terms using ops
        l2_val = float(ops.mean(ops.square(y_true_tensor - y_pred_batch)))
        l1_val = float(ops.mean(ops.abs(y_true_tensor - y_pred_batch)))
        
        f_true_batch = feature_extractor(y_true_tensor)
        f_pred_batch = feature_extractor(y_pred_batch)
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
        else:
            diff_d = ops.mean(ops.abs(y_pred_batch[:, 1:, :, :, :] - y_pred_batch[:, :-1, :, :, :]))
            diff_h = ops.mean(ops.abs(y_pred_batch[:, :, 1:, :, :] - y_pred_batch[:, :, :-1, :, :]))
            diff_w = ops.mean(ops.abs(y_pred_batch[:, :, :, 1:, :] - y_pred_batch[:, :, :, :-1, :]))
            tv_val = float(diff_d + diff_h + diff_w)
        
        # Weighted terms (using ops.convert_to_numpy to avoid PyTorch warnings)
        w_l2 = l2_val * float(ops.convert_to_numpy(msq_weight_var))
        w_l1 = l1_val * float(ops.convert_to_numpy(l1_weight_var))
        w_feat = feat_val * float(ops.convert_to_numpy(feat_weight_var))
        w_tv = tv_val * float(ops.convert_to_numpy(tv_weight_var))
        
        total_calculated = w_l2 + w_l1 + w_feat + w_tv
        # Avoid division by zero
        denom = total_calculated if total_calculated > 1e-8 else 1.0
        
        pct_l2 = w_l2 / denom * 100
        pct_l1 = w_l1 / denom * 100
        pct_feat = w_feat / denom * 100
        pct_tv = w_tv / denom * 100
        
        print(f"{stage_name} Iter {iteration:03d}/{max_iter} - Loss: {loss:.6f}")
        print(f"  [Loss Components] Raw: L2 (MSE)={l2_val:.6f}, L1={l1_val:.6f}, Feat={feat_val:.6f}, TV={tv_val:.6f}")
        print(f"  [Loss Contributions] Weighted: MSE={w_l2:.4f} ({pct_l2:.1f}%), L1={w_l1:.4f} ({pct_l1:.1f}%), Feat={w_feat:.4f} ({pct_feat:.1f}%), TV={w_tv:.4f} ({pct_tv:.1f}%)")
        print(f"  [Loss Weights] L1={w_l1/l1_val if l1_val > 1e-8 else 0.0:.6f}, Feat={w_feat/feat_val if feat_val > 1e-8 else 0.0:.6f}, TV={w_tv/tv_val if tv_val > 1e-8 else 0.0:.6f}")
        
        # Log to CSV
        csv_log_path = os.path.join(workspace_dir, f"loss_contributions_{model_type}_{dim}d.csv")
        try:
            with open(csv_log_path, "a") as f:
                f.write(f"{stage_name},{iteration},{loss:.6f},{l2_val:.6f},{l1_val:.6f},{feat_val:.6f},{tv_val:.6f},"
                        f"{w_l2:.6f},{w_l1:.6f},{w_feat:.6f},{w_tv:.6f},{pct_l2:.2f},{pct_l1:.2f},{pct_feat:.2f},{pct_tv:.2f}\n")
        except Exception as e:
            print(f"  [Warning] Failed to write loss contributions to CSV: {e}")

        # Update preset weights file
        try:
            pd.DataFrame([[
                0.0,
                float(ops.convert_to_numpy(feat_weight_var)),
                float(ops.convert_to_numpy(tv_weight_var)),
                float(ops.convert_to_numpy(l1_weight_var))
            ]], columns=["msq", "feat", "tv", "l1"]).to_csv(wts_csv, index=False)
        except Exception as e:
            pass

    best_val_loss = float("inf")
    
    # Initialize loss contributions CSV file and history tracker
    tracker = LossHistoryTracker(window_size=args.smooth_window)
    csv_log_path = os.path.join(workspace_dir, f"loss_contributions_{model_type}_{dim}d.csv")
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
        # Calculate raw loss values for this batch to add to tracker
        y_true_tensor = ops.convert_to_tensor(y_batch, dtype="float32")
        y_pred_batch = ops.stop_gradient(model(x_batch, training=False))
        raw_l1 = float(ops.mean(ops.abs(y_true_tensor - y_pred_batch)))
        
        f_true_batch = feature_extractor(y_true_tensor)
        f_pred_batch = feature_extractor(y_pred_batch)
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
        
        # Update weights every update_freq iterations
        if iteration % args.update_freq == 0:
            current_w = {
                'mae': float(ops.convert_to_numpy(l1_weight_var)),
                'percep': float(ops.convert_to_numpy(feat_weight_var)),
                'tv': float(ops.convert_to_numpy(tv_weight_var))
            }
            new_w, smoothed_losses = get_smoothed_losses_and_weights(
                tracker, target_pcts, iteration, original_weight_sum, current_w, beta_damp=args.dampening
            )
            l1_weight_var.assign(new_w['mae'])
            feat_weight_var.assign(new_w['percep'])
            tv_weight_var.assign(new_w['tv'])

    if not skip_stages_1_2:
        # ==============================================================
        # Stage 1: Warmup & Adaptation on Clean Mixed Classes (Iter 1-100)
        print("\n=======================================================")
        print(f"Stage 1: Adaptation Phase (Clean Mixed Geometries) (Iter 1-100)")
        print("=======================================================")
        
        if model_type in ["espcn", "espcn-rc"]:
            set_core_trainable(model, trainable=False)
        
        train_gen_clean = siq.blind_sr_generator(
            hr_base_cache=None,
            batch_size=batch_size,
            lr_patch_size=48,
            factor=2,
            blur_sigma_range=(0.0, 0.0),
            noise_std_range=(0.0, 0.0),
            simulation_classes=simulation_classes,
            zoom_range=(0.75, 1.3),
            use_cache=False,
            dimensionality=dim
        )
        
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-5), loss=hybrid_loss)
        
        for iteration in range(max(1, last_iteration + 1), stage1_max + 1):
            x_batch, y_batch = next(train_gen_clean)
            loss = model.train_on_batch(x_batch, y_batch)
            step_dynamic_balancer(iteration, x_batch, y_batch)
            
            # Print iteration log immediately
            print(f"Stage 1 Iter {iteration:03d}/{stage1_max} - Loss: {loss:.6f}")
            
            # Track and log heavy loss components and monitor metrics every 50 iterations
            if iteration % 50 == 0 or iteration == 1 or iteration == max(1, last_iteration + 1):
                print_loss_components("Stage 1", iteration, stage1_max, x_batch, y_batch, loss)
                # Save actual training batch inputs sent to model.train_on_batch
                try:
                    os.makedirs(os.path.join(scratch_dir, "training_samples"), exist_ok=True)
                    x_img_actual = ants.from_numpy(np.squeeze(x_batch))
                    y_img_actual = ants.from_numpy(np.squeeze(y_batch))
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

                sr_img = siq.inference(lr_patch, model, method="antspynet", verbose=False)
                ants.copy_image_info(hr_patch, sr_img)
                sr_np = sr_img.numpy()
                corr = float(np.corrcoef(sr_np.flatten(), gt_np.flatten())[0, 1])
                psnr = float(antspynet.psnr(hr_patch, sr_img))
                ssim = float(antspynet.ssim(hr_patch, sr_img))
                
                print(f"  [OASIS Monitor] Corr: {corr:.4f}, PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
                
                if loss < best_val_loss:
                    best_val_loss = loss
                    model.save(output_model_path)
                    print(f"  --> Saved checkpoint to {output_model_path}")

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
        
        train_gen_robust = siq.blind_sr_generator(
            hr_base_cache=None,
            batch_size=batch_size,
            lr_patch_size=48,
            factor=2,
            blur_sigma_range=(0.0, 0.0),
            noise_std_range=(0.0, 0.02),
            use_rician_noise=True,
            simulation_classes=simulation_classes,
            zoom_range=(0.75, 1.3),
            use_cache=False,
            dimensionality=dim
        )
        
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-5), loss=hybrid_loss)
        
        start_iter = max(stage1_max + 1, last_iteration + 1)
        for iteration in range(start_iter, stage2_max + 1):
            x_batch, y_batch = next(train_gen_robust)
            loss = model.train_on_batch(x_batch, y_batch)
            step_dynamic_balancer(iteration, x_batch, y_batch)
            
            # Print iteration log immediately
            print(f"Stage 2 Iter {iteration:03d}/{stage2_max} - Loss: {loss:.6f}")
            
            # Track and log heavy loss components and monitor metrics every 50 iterations
            if iteration % 50 == 0 or iteration == stage1_max + 1 or iteration == start_iter:
                print_loss_components("Stage 2", iteration, stage2_max, x_batch, y_batch, loss)
                # Save actual training batch inputs sent to model.train_on_batch
                try:
                    os.makedirs(os.path.join(scratch_dir, "training_samples"), exist_ok=True)
                    x_img_actual = ants.from_numpy(np.squeeze(x_batch))
                    y_img_actual = ants.from_numpy(np.squeeze(y_batch))
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

                sr_img = siq.inference(lr_patch, model, method="antspynet", verbose=False)
                ants.copy_image_info(hr_patch, sr_img)
                sr_np = sr_img.numpy()
                corr = float(np.corrcoef(sr_np.flatten(), gt_np.flatten())[0, 1])
                psnr = float(antspynet.psnr(hr_patch, sr_img))
                ssim = float(antspynet.ssim(hr_patch, sr_img))
                
                print(f"  [OASIS Monitor] Corr: {corr:.4f}, PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
                
                if loss < best_val_loss:
                    best_val_loss = loss
                    model.save(output_model_path)
                    print(f"  --> Saved checkpoint to {output_model_path}")
    else:
        print("\nSkipping Stage 1 & Stage 2 (already refined). Proceeding directly to Stage 3 (Dedicated Refinement)...")

    # ==============================================================
    # Stage 3: Dedicated Refinement Strategy (Iter 2001-5000)
    # ==============================================================
    print("\n=======================================================")
    print(f"Stage 3: Dedicated Refinement Phase (High-Fidelity Brain Focus) (Iter 2001-5000)")
    print("=======================================================")
    
    if skip_stages_1_2:
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
        "brain_procedural": 0.60,
        "layered": 0.20,
        "sinewave": 0.10,
        "organic_blobs": 0.10
    }

    train_gen_refine = siq.blind_sr_generator(
        hr_base_cache=None,
        batch_size=batch_size,
        lr_patch_size=48,
        factor=2,
        blur_sigma_range=(0.0, 0.0),
        noise_std_range=(0.0, 0.01),
        use_rician_noise=True,
        simulation_classes=refinement_classes,
        zoom_range=(0.75, 1.3),
        use_cache=False,
        dimensionality=dim
    )
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-6), loss=hybrid_loss)
    best_val_loss = float("inf")

    start_iter = max(stage2_max + 1, last_iteration + 1)
    for iteration in range(start_iter, stage3_max + 1):
        x_batch, y_batch = next(train_gen_refine)
        loss = model.train_on_batch(x_batch, y_batch)
        step_dynamic_balancer(iteration, x_batch, y_batch)
        
        # Print iteration log immediately
        print(f"Stage 3 Iter {iteration:03d}/{stage3_max} - Loss: {loss:.6f}")
        
        # Track and log heavy loss components and monitor metrics every 50 iterations
        if iteration % 50 == 0 or iteration == stage2_max + 1 or iteration == start_iter:
            print_loss_components("Stage 3", iteration, stage3_max, x_batch, y_batch, loss)
            # Save actual training batch inputs sent to model.train_on_batch
            try:
                os.makedirs(os.path.join(scratch_dir, "training_samples"), exist_ok=True)
                x_img_actual = ants.from_numpy(np.squeeze(x_batch))
                y_img_actual = ants.from_numpy(np.squeeze(y_batch))
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

            sr_img = siq.inference(lr_patch, model, method="antspynet", verbose=False)
            ants.copy_image_info(hr_patch, sr_img)
            sr_np = sr_img.numpy()
            corr = float(np.corrcoef(sr_np.flatten(), gt_np.flatten())[0, 1])
            psnr = float(antspynet.psnr(hr_patch, sr_img))
            ssim = float(antspynet.ssim(hr_patch, sr_img))
            
            print(f"  [OASIS Monitor] Corr: {corr:.4f}, PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
            
            if loss < best_val_loss:
                best_val_loss = loss
                model.save(output_model_path)
                print(f"  --> Saved checkpoint to {output_model_path}")

    print(f"{model_type.upper()} {dim}D Refinement Pipeline Complete.")

if __name__ == "__main__":
    main()

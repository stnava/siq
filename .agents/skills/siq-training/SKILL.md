---
name: siq-training
description: >-
  Cheatsheet for launching, resuming, and diagnosing siq super-resolution
  training runs. Covers ResNet perceptual backend (rank-normalization
  requirement), pre-calibrated loss weights, lazy balancer, prefetch, and
  standard command-line invocations for asdbpn 3D refinement.
---

# siq Training Cheatsheet

## Standard Resume Command (ResNet backend, pre-calibrated)

```bash
PYTHONUNBUFFERED=1 python tests/train_model_refinement.py asdbpn \
  --dim 3 --batch-size 4 --lr-patch-size 32 \
  --projection-kernel-size 6 \
  --skip-warmup \
  --load-model checkpoints/asdbpn_3d/asdbpn_3d_best_psnr.keras \
  --stage1-iter 50 --stage2-iter 750 --stage3-iter 3000 \
  --stage2-lr 2.5e-5 --stage3-lr 1e-5 \
  --clip-norm 1.0 \
  --checkpoint-freq 25 \
  --balancer-freq 25 --update-freq 25 \
  --prefetch-size 4 \
  --perceptual-backend resnet \
  --anneal-iter 0 \
  --dampening 0.97 \
  --init-l1-weight   3.788569 \
  --init-feat-weight 565.098821 \
  --init-tv-weight   0.462579 \
  2>&1 | tee logs/run_$(date +%Y%m%d_%H%M).log
```

## Re-Calibrating Weights (run when model improves significantly)

Measure median raw losses over 10 Rician batches from the current best
checkpoint with rank-normalization applied, then compute:

```python
# target_total_loss = 1.0; targets: MAE=30%, Feat=65%, TV=5%
w_l1   = 0.30 / median_l1
w_feat = 0.65 / median_feat
w_tv   = 0.05 / median_tv
```

Pass the results as `--init-l1-weight W --init-feat-weight W --init-tv-weight W`
to bypass auto-calibration and start training in the correct 30/65/5 ratio
from iteration 1.

**Last measured medians** (asdbpn_3d_best_psnr.keras, 10 Rician batches,
rank-normalized, 2026-09-28):

| Component | Median raw | Weight | Target % |
|-----------|-----------|--------|----------|
| L1 (MAE)  | 0.079186  | 3.789  | 30%      |
| Feat (ResNet) | 0.001150 | 565.1 | 65%   |
| TV        | 0.108090  | 0.463  | 5%       |

## Critical: ResNet Preprocessing (rank-normalization)

`siq.get_grader_feature_network()` has **NO internal rescaling**. It was trained
via `antspyt1w.resnet_grader()` which calls `ants.rank_intensity` (→ uniform
[0,1] distribution) before inference. Feeding raw linear [0,1] patches causes
~8× feature magnitude mismatch and PSNR degradation during training.

Always apply `rank_normalize_batch()` before every ResNet feature extractor call:

```python
def rank_normalize_batch(x_tensor):
    """Per-sample rank normalization matching ants.rank_intensity."""
    x_np = ops.convert_to_numpy(x_tensor).astype("float32")
    out = np.empty_like(x_np)
    for b in range(x_np.shape[0]):
        vol = x_np[b, ..., 0].ravel()
        ranks = np.argsort(np.argsort(vol)).astype("float32")
        ranks /= max(len(ranks) - 1, 1)
        out[b, ..., 0] = ranks.reshape(x_np.shape[1:-1])
    return ops.convert_to_tensor(out, dtype="float32")
```

In the training script, route all feature extractor calls through `_call_fe()`:

```python
if args.perceptual_backend == "resnet" and dim == 3:
    def _call_fe(tensor):
        return feature_extractor(rank_normalize_batch(tensor), training=False)
else:
    def _call_fe(tensor):
        return feature_extractor(tensor)
```

VGG (`pseudo_3d_vgg_features_unbiased`) does **NOT** need rank-normalization —
it bakes in `Rescaling(255.0, -127.5)` internally.

## BatchNorm Moving Stats (resnet_grader.h5)

All BN layers have `moving_mean=0, moving_var=1` — stored in the h5 file,
not a load failure. BN layers are effectively identity transforms. Outputs are
identical in `training=False` and `training=True`. This is expected and correct.

## Speedup Summary (Apple Silicon MPS, batch=4, patch 32³→64³)

| Optimization              | Time saved      | Flag                          |
|---------------------------|-----------------|-------------------------------|
| ResNet vs VGG perceptual  | ~4.5 s/step     | `--perceptual-backend resnet` |
| Lazy balancer             | ~3.6 s amortised| `--balancer-freq 10`          |
| Prefetch generator        | ~1.67 s overlap | `--prefetch-size 4`           |
| Pre-calibrated weights    | ~4 s at startup | `--init-*-weight`             |
| **Total**                 | **~12.9 → ~2 s/step (~6×)** |                    |

## Loss Balance Targets

| Component           | Target | Notes                     |
|---------------------|--------|---------------------------|
| MAE (L1)            | 30%    | Sharp edges               |
| Perceptual (ResNet) | 65%    | Structural fidelity       |
| Total Variation     | 5%     | Smoothness regularizer    |
| MSE (L2)            | 0%     | Disabled in Stage 2+      |

## Balancer Dampening Sensitivity

The `--dampening` flag controls how aggressively the LOWESS weight balancer
moves toward the 30/65/5 target ratio each update step. Too low = oscillation;
too high = too slow to correct pre-calibrated weights that drift.

| dampening | balancer-freq | Behavior |
|-----------|---------------|----------|
| 0.92 | 10 | ❌ Over-corrects — loss balance oscillates ±20pp per update (Run C) |
| **0.97** | **25** | ✅ **Stable** — converges to target in ~100 steps smoothly (Run D) |
| 0.98 (default) | 25 | ⚠️ Slow — takes ~200+ steps to correct from fresh pre-calibrated weights |

At `dampening=d`, each update moves weights by approximately `(1-d) × decay_factor`:
- `d=0.92` → ~8% per step → overshoots, oscillates
- `d=0.97` → ~3% per step → smooth convergence

Always pair `--dampening 0.97` with `--balancer-freq 25 --update-freq 25`.
Smaller freq values (e.g. 10) compound oscillation by applying corrections
before the smoother has enough data points.

## Gradient Clipping & Collapse Prevention (ResNet Perceptual Loss)

When training with the ResNet perceptual backend in Stage 3 refinement, unconstrained
feature gradients can accumulate and cause sudden model collapse (PSNR dropping from ~27.4 dB
to ~20 dB around Iteration 1500–1525).

### Why It Happens
- The ResNet feature extractor has a fixed feature scale that yields raw feature MSE ~0.001.
- With feature weight ~500–565, even small differences between predictions and rank-normalized targets can produce large gradient vectors.
- Over 1,000+ steps of refinement, isolated gradient spikes can push convolutional filter weights out of the linear response regime, causing sudden feature divergence.

### The Proven Solution
1. **Explicit Gradient Clipping**: `--clip-norm 1.0` (bounds the total L2 norm of the gradient vector to 1.0 per optimizer step across all 6 compile sites).
2. **Refined Stage 3 Learning Rate**: `--stage3-lr 1e-5` (prevents overshoot while allowing subtle high-frequency sharpening).

In Run F, this configuration ran through all 3,000 steps with **zero collapse**, setting an all-time record:
- **PSNR: 27.488 dB** (+0.39 dB vs Bilinear)
- **SSIM: 0.9376**
- **HFEN: 0.4457**
- **Corr: 0.9359**

## Checkerboard Artifact Mitigation (3D Transposed-Convolution)

Transposed-convolution layers (`Conv3DTranspose(6, 6, 6, strides=2)`) in AS-DBPN naturally introduce Nyquist-frequency grid resonance when expanding resolution. Without mitigation, raw model output exhibits checkerboard artifacts ($\text{CBI} \approx 0.073$, $3.7\times$ higher than ground truth MRI).

### 1. In-Training Alternating Parity Loss (`--cbi-weight 2.0`)
- **Residual Error Target Required**: Naive alternating parity filtering on $\hat{y}$ alone penalizes natural anatomical edges (~`0.025-0.28`). Applying it to the **residual error $(\hat{y} - y_{\text{true}})$** zeroes out true anatomy (`0.000`) and isolates 100% pure transposed-convolution artifact error:
  ```python
  # In hybrid_loss:
  _cb_target = y_pred - y_true
  cbi_block = (c000 - c100 - c010 + c110 - c001 + c101 + c011 - c111) / 8.0
  cbi_term = ops.mean(ops.abs(cbi_block))
  ```
- **Performance**: Setting `--cbi-weight 2.0` in `train_model_refinement.py` permanently unlearns the grid artifact within 25 iterations: raw validation CBI drops from `0.0729` to `0.0189` (matching ground truth `0.0198`), while raw PSNR jumps by **+0.72 dB** (surpassing `27.60 dB`) and SSIM reaches `0.9395`.

### 2. Post-Processing Auto-Notch Filter (`anti_checkerboard='auto'`)
For legacy checkpoints or unregularized models, `siq.inference()` applies an analytical data-driven notch filter:
```python
sigma = siq.estimate_anti_checkerboard_sigma(vol)  # sigma = sqrt(ln(excess)) / pi
```
If Nyquist spectral excess $\le 1.2\times$, $\sigma = 0.0$ (no blur applied to clean inputs).

## Model I/O & Provenance Standards

Never guess model input/output configurations or normalization methods. All models must be loaded and saved with companion `_config.json` provenance files:
```python
# Save:
siq.save_siq_model("model.keras", model, config)  # writes model.keras and model_config.json

# Load:
model, config = siq.load_siq_model("model.keras")  # loads weights and companion _config.json

# Inference:
sr = siq.inference(img, model, config=config)     # uses provenance config for full-volume prediction
```

## Key File Locations

| File | Purpose |
|------|---------|
| `checkpoints/asdbpn_3d/asdbpn_3d_best_psnr.keras` | Best PSNR checkpoint |
| `asdbpn_3d_refined_training_weights.csv` | Auto-saved weight state |
| `loss_contributions_asdbpn_3d.csv` | Per-step loss component log |
| `asdbpn_3d_report.html` | Convergence dashboard |
| `tests/watch_report.py` | Live dashboard watcher |
| `~/.antspyt1w/resnet_grader.h5` | ResNet grader weights |

## Release Workflow: Bump Version → Tag → Commit → Push

### Step 1 — Decide the bump level
| Change type | Bump |
|-------------|------|
| Training script improvements, loss terms, diagnostics | **patch** (0.10.6 → 0.10.7) |
| New public siq API function or changed function signature | **minor** (0.10.x → 0.11.0) |
| Breaking change to siq API | **major** (0.x.y → 1.0.0) |

### Step 2 — Confirm current state
```bash
grep "^version" pyproject.toml          # current version
git tag --sort=-creatordate | head -3   # latest tag
git log --oneline <last-tag>..HEAD      # commits since tag
```

### Step 3 — One-liner bump, tag, and push (replace OLD and NEW)
```bash
OLD=0.10.6; NEW=0.10.7
sed -i '' "s/version = \"$OLD\"/version = \"$NEW\"/" pyproject.toml
git add pyproject.toml && git commit -m "chore: bump version to $NEW"
git tag v$NEW && git push origin main && git push origin v$NEW
```

> [!NOTE]
> Version string lives in `pyproject.toml` under `[project]`.
> Always push both the branch (`main`) and the tag separately.
> Patch bumps cover training script changes; minor bumps for new public API.

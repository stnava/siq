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
  --checkpoint-freq 25 \
  --balancer-freq 10 --update-freq 10 \
  --prefetch-size 4 \
  --perceptual-backend resnet \
  --anneal-iter 0 \
  --dampening 0.92 \
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

## Key File Locations

| File | Purpose |
|------|---------|
| `checkpoints/asdbpn_3d/asdbpn_3d_best_psnr.keras` | Best PSNR checkpoint |
| `asdbpn_3d_refined_training_weights.csv` | Auto-saved weight state |
| `loss_contributions_asdbpn_3d.csv` | Per-step loss component log |
| `asdbpn_3d_report.html` | Convergence dashboard |
| `tests/watch_report.py` | Live dashboard watcher |
| `~/.antspyt1w/resnet_grader.h5` | ResNet grader weights |

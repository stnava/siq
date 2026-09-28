---
trigger: always_on
---

# siq Perceptual Loss Rules

## ResNet Grader: Rank Normalization Required

The ResNet grader (`siq.get_grader_feature_network`) expects **rank-intensity
normalized** inputs matching `ants.rank_intensity` (uniform [0,1] distribution).
The model has NO internal rescaling layer. Feeding raw linear [0,1] patches
causes ~8× feature magnitude mismatch and measurable PSNR degradation during
training.

Always apply `rank_normalize_batch()` (double-argsort divided by N-1) before
passing tensors to the ResNet feature extractor. See the `siq-training` skill
for the implementation.

## VGG Extractor: No External Rescaling

`pseudo_3d_vgg_features_unbiased` bakes in `Rescaling(255.0, -127.5)` — do
**NOT** apply additional rescaling to its inputs.

## Always Use a Dispatch Wrapper

Never call `feature_extractor(tensor)` directly in training code. Always route
through a `_call_fe(tensor)` closure that dispatches on `args.perceptual_backend`
so rank-normalization is applied consistently across `hybrid_loss`,
`print_loss_components`, `step_dynamic_balancer`, and weight calibration.

## Backend Switching Requires Weight Re-Calibration

When switching perceptual backends (vgg ↔ resnet), always re-calibrate loss
weights using the 10-batch median method (see `siq-training` skill). Raw feature
magnitudes differ by 10–100× between backends; the auto-calibrator's single-batch
estimate will be wrong at steady-state training time.

## BatchNorm Moving Stats (resnet_grader.h5)

All BN layers in `resnet_grader.h5` have `moving_mean=0, moving_var=1`. This is
correct — stored in the h5, not a load failure. Outputs are identical in
`training=False` and `training=True`. Do not attempt to "fix" this.

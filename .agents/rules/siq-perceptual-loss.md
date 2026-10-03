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

## Lean into Super-Resolution (Never Smother SR with Linear Blending)

Super-resolution models must be evaluated and saved with pure model output (`linear_blend = 1.0` or disabled):
- Never use linear blending (`linear_blend < 1.0`) as a crutch to artificially boost PSNR/SSIM at the cost of high-frequency sharpness.
- Acknowledge and document the fundamental **Perception–Distortion Tradeoff** (Blau & Michaeli, 2018): sharp, detailed anatomical boundary reconstruction will typically have a slightly lower PSNR (~23–25 dB) than blurred linear averaging (~26–27 dB), but restores true anatomical sharpness.

## Independent Multi-Dimensional Perceptual Metrics

When evaluating or refining models toward the perceptual / SR world, always evaluate independent perceptual metrics alongside PSNR and SSIM:
- `siq.compute_acutance_ratio(gt, pred)`: Measures 1st-order gradient magnitude sharpness ($\frac{\|\nabla y_{\text{pred}}\|}{\|\nabla y_{\text{true}}\|}$). Bilinear discards ~21% (`~0.789`); champion SR achieves `~0.95–1.03`.
- `siq.compute_laplacian_energy_ratio(gt, pred)`: Measures 2nd-order high-frequency structural energy ($\frac{\mathcal{E}_{\Delta}(y_{\text{pred}})}{\mathcal{E}_{\Delta}(y_{\text{true}})}$). Bilinear discards ~56% (`~0.441`); champion SR recovers `~0.55–0.62` (+25% to +40% more detail).
- `siq.compute_spectral_energy_ratio(gt, pred, factor=factor)`: Measures high-frequency Fourier spectrum recovery in the upsampled frequency band.

## Perceptual Composite Score (PCS) Standard

For perceptual refinement stages, champion model selection must use PCS rather than pure PSNR or standard CQS:
$$\text{PCS} = \text{val\_ssim} + 0.5 \cdot \text{val\_acutance} + 0.5 \cdot \text{val\_laplacian} - \text{val\_gmsd} - \text{val\_cbi}$$
PCS rewards structural preservation, anatomical edge sharpness, and high-frequency Laplacian detail while penalizing gradient dispersion and checkerboard artifacts.

## Pointwise Directional Gradient & Acutance Loss (Architecture Specific)

- **Feed-Forward Architectures (AS-DBPN)**: Always compute **pointwise directional gradient $L_1$ error** plus **gradient magnitude acutance matching** (`--edge-weight 2.0` in Stage 3):
  $$\mathcal{L}_{\text{edge}} = \frac{1}{N} \sum |\nabla y_{\text{true}} - \nabla y_{\text{pred}}| + \frac{1}{N} \sum |\|\nabla y_{\text{pred}}\| - \|\nabla y_{\text{true}}\||$$
- **Iterative Back-Projection Architectures (DBPN & L-DBPN)**: Always set `--edge-weight 0.0`. Iterative projection units natively minimize reconstruction residual errors; adding an external finite-difference edge loss steals ~30% of the gradient budget and blurs boundary reconstructions.

## Keep CQS & PCS on Hand for Performance Tracking

Never judge models by PSNR alone. Always keep **CQS** (Composite Quality Score: $\text{SSIM} - \text{GMSD} - \text{CBI}$) and **PCS** (Perceptual Composite Score: $\text{SSIM} + 0.5\cdot\text{Acutance} + 0.5\cdot\text{Laplacian} - \text{GMSD} - \text{CBI}$) on hand using native `siq.compute_cqs()` and `siq.compute_pcs()` to track model quality.

## 4-Stage Curriculum Staging Invariant (Never Truncate Stage 1 or 2)

When training super-resolution models from scratch or without pre-calibrated weights, training must strictly follow the 4-stage progression:

| Stage | Purpose | Data & Augmentations | Loss Regime | Minimum Iteration Scope |
|:---|:---|:---|:---|:---:|
| **Stage 0 (Warmup)** | Reach Bilinear Parity | Clean patches, no noise, higher LR (`1e-4`). | 100% Pure MSE | Until validation PSNR $\ge$ Bilinear baseline (or max 1,000 steps) |
| **Stage 1** | Gentle Perceptual Entry & Weight Tuning | Clean data, zero noise. | $L_1$-dominant entry ($L_1 \approx 70\%$, Feat $\approx 25\%$, TV $\approx 5\%$). LOWESS balancer smoothly settles weights. | **$\ge$ 500 iterations** (`--stage1-iter 500`) |
| **Stage 2** | Bulk Robustness Training | Add Rician noise (`noise_std_range=(0.0, 0.02)`), zoom scaling `(0.75, 1.3)`. | Established weights ($L_1 \approx 30\%$, Feat $\approx 65\%$, TV $\approx 5\%$). | **$\ge$ 1,000 iterations** (`--stage2-iter 1500`) |
| **Stage 3** | Dedicated Perceptual Refinement | Dedicated brain/tissue classes, fine LR (`1e-5`). | Pointwise directional gradient $L_1$ + gradient magnitude acutance (`--edge-weight 2.0`), factor-aware CBI, pure SR (`linear_blend = 1.0`), PCS champion selection. | **$\ge$ 3,500 iterations** (`--stage3-iter 5000`) |

### Strict Constraints:
1. **Never Compress Stage 1 Below 500 Steps**: Abbreviating Stage 1 (e.g. to 50 steps) denies the LOWESS balancer the ~100–200 steps required to reach steady-state targets smoothly at `dampening=0.97`, shocking the convolutional filters with erratic feature updates.
2. **Never Collapse Stages 1 and 2**: Stage 1 isolates weight calibration on clean geometry; Stage 2 introduces noise robustness once weights are stationary. Conflating them causes optimizer oscillation.



## Every SR Claim Needs a Reference Table (Benchmark Invariant)

Never report a super-resolution model as "good" from its own numbers or from PSNR/PCS alone:
- Run `siq.sr_benchmark(...)` (see the `siq-benchmark` skill). The model must beat the best classical interpolator / bilinear+unsharp on **CQS** on the `heldout` case set (real slices, disjoint subjects); to claim *learned* gain it must also beat the in-sample linear least-squares ceiling.
- Report PCSc (bounded sharpness) next to PCS: PCS rewards over-sharpening beyond the ground truth.
- Public pretrained models (super-image EDSR/MSRN/...) are evaluated through `siq.public_model_method` with an explicit phase calibration (`half-pixel` or `auto`); never compare an uncalibrated public model.
- Never conclude from procedural-simulation validation alone: procedural-trained models lost to bilinear on real held-out slices. Real-slice held-out evaluation is mandatory.

## Pre-Normalized Benchmark & Direct Inference Invariant

When evaluating pre-normalized $[0, 1]$ benchmark slices or patches:
- Never pass pre-normalized data through preprocessing that re-applies `TruncateIntensity` and `Normalize`, or re-normalizes output tensors upon linear blending.
- Doing so skews slice gain by $\sim 19\%$ (collapsing PSNR to $\sim 18\text{ dB}$ on models that achieve $>30\text{ dB}$ raw).
- For benchmark validation cases with known $[0, 1]$ bounds, evaluate via direct forward prediction while preserving direction-cosine reflection compensation and architecture-aware phase shifts.


---
trigger: always_on
---

# siq Checkerboard Artifact Mitigation & Provenance Rules

## Checkerboard Loss on Prediction Error (Not Output Alone)

The alternating-parity checkerboard filter ($K_{i,j,k} = \frac{1}{8}(-1)^{i+j+k}$) must **ALWAYS** be evaluated on the residual prediction error $(\hat{y} - y_{\text{true}})$, never on $\hat{y}$ alone.

Natural anatomical edges produce high-frequency alternating-parity energy (~0.025–0.28) even in clean ground truth MRI. Evaluating on $(\hat{y} - y_{\text{true}})$ zeroes out true anatomy (`0.000`) and isolates 100% pure transposed-convolution artifact error.

## Always Enable CBI Regularization During 3D Transposed-Conv Training

When refining models with transposed-convolution layers (e.g. AS-DBPN 3D), always pass `--cbi-weight 2.0` (or set `cbi_weight_var > 0`). Without this loss term, raw model output exhibits severe checkerboard grid resonance ($\text{CBI} \approx 0.073$, $3.7\times$ ground truth).

## Data-Driven Anti-Checkerboard Filter (No Fixed Sigma)

Never require users to specify a manual filtering sigma. In `siq.inference()`, `anti_checkerboard='auto'` must dynamically solve for the sub-voxel notch bandwidth:
$$\sigma = \frac{\sqrt{\ln(\text{excess})}}{\pi}$$
where `excess` is the measured 1D Nyquist spectral energy ratio relative to low frequencies. If excess $\le 1.2\times$, $\sigma = 0.0$ (no blur applied to clean inputs).

## Provenance I/O Invariant

Always use `siq.load_siq_model(path)` (returning `model, config`) and `siq.save_siq_model(path, model, config)`. Companion `_config.json` files must always travel with `.keras` files so inference callers never guess normalization quantile bounds or patching schemes.

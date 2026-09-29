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

Loss weights (`loss_weights` dict covering `l1`, `feat`, `tv`, `msq`, `edge`, `gms`, `cbi`) must always be saved to and loaded from companion `_config.json`. Any downstream transfer learning, checkpointing, or model refinement must load these calibrated weights via `siq.extract_siq_loss_weights(source)` rather than recalibrating from scratch.

## Visual Display Normalization (Histogram Equalization vs. Rank Normalization)

When generating visual reports, montages, or diagnostic slice plots:
- Use `ants.histogram_equalize_image(img, number_of_histogram_bins=256)` for display contrast.
- Never use `ants.rank_intensity` for full-brain visual displays; it skews midtones across background/CSF voxels and blows out gray/white matter parenchyma.
- Display contrast normalization must strictly be applied to the display arrays only — never alter the underlying tensors used for loss calculation or quantitative metrics (PSNR, SSIM, GMSD, CBI).

## Amplitude-Based Sub-Voxel Notch Calibration

`siq.estimate_anti_checkerboard_sigma(vol)` must solve for $\sigma$ using the 1D **amplitude spectrum** ($|F|$), never power ($|F|^2$). Power ratio squares the excess, artificially inflating $\sigma$ by $\sqrt{2}\times$ and causing over-blurring. Cap maximum bandwidth at $\sigma \le 0.40$ to guarantee $>96\%$ anatomical edge gradient preservation.

## Anisotropic Super-Resolution Reporting & Path Invariant

When training models with non-uniform downsampling factors (e.g. `--factor 1 1 2`):
- Report images and checkpoints must reside in factor-specific directories:
  - Reports: `reports/{model}_{dim}d_{factor_str}` (e.g. `reports/asdbpn_3d_1x1x2/`)
  - Checkpoints: `checkpoints/{model}_{dim}d_{factor_str}` (e.g. `checkpoints/asdbpn_3d_1x1x2/`)
  - HTML Dashboards: `{model}_{dim}d_{factor_str}_report.html` (e.g. `asdbpn_3d_1x1x2_report.html`)
- HTML dashboard generators (`render_convergence_dashboard.py` and `visual_convergence_report.py`) must **NEVER** hardcode default paths (`reports/asdbpn_3d/`). All viewport and table `src=` / `href=` links must be dynamically derived relative to `report_dir` and `checkpoint_dir`.
- Resumption precedence: When resuming a run without `--reset-history`, training scripts must always load existing in-progress checkpoints or refined models before falling back to `--transfer-from`.


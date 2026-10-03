---
name: siq-qc
description: Standardized Super-Resolution Spatial Alignment & Quality Control (QC) guide. Explains sub-pixel phase cross-correlation, directional gradient edge error correlation, status thresholds, CLI usage, and dashboard integration.
---

# Super-Resolution Spatial Alignment & Quality Control (QC)

This skill documents the automated, low-overhead (<35ms) Quality Control (QC) protocol for verifying that super-resolution models are free of sub-pixel phase shifts, directional gradient biases, and grid artifacts.

---

## 1. Motivation: Spatial Alignment Biases

In deep learning super-resolution for medical imaging, two subtle bugs can introduce systematic sub-pixel misalignments:
1. **Transposed-Convolution / Skip-Connection Phase Shift**:
   `Conv3DTranspose` (stride 2) or `UpSampling3D` with bilinear interpolation can place child voxels $+0.5$ voxels off the continuous ITK physical grid center.
2. **Discrete Reflection Coordinate Shift**:
   Flipping negative direction-cosine axes before inference and flipping back reverses child voxel order within parent voxels, introducing a $+1.0$ voxel shift ($f_d - 1$).

### Diagnostic Symptoms of Unaligned Models:
- High directional edge error correlation ($r_d > 0.25$) where prediction error aligns with boundary gradients.
- Sub-pixel FFT phase cross-correlation lag ($\approx 0.35 - 0.50$ voxels).
- Depressed validation PSNR (up to $-2.5$ dB lower than bilinear interpolation).

---

## 2. Core Diagnostics

### A. Sub-Pixel FFT Phase Cross-Correlation
Measures the translational registration vector $(\Delta_y, \Delta_x, \Delta_z)$ between ground truth and model output using Fourier cross-correlation with un-normalized spectral amplitudes (`normalization=None`):
$$\Delta = \arg\max_{\mathbf{s}} \mathcal{F}^{-1}\{Y_{\text{true}}(\mathbf{u}) Y_{\text{pred}}^*(\mathbf{u})\}$$
- **Target**: Zero lag ($\|\Delta\|_\infty < 0.05$ voxels).
- **Function**: `siq.compute_phase_shift(y_true, y_pred, upsample_factor=100)`

### B. Directional Edge Error Correlation
Computes the Pearson correlation magnitude $|r_d|$ between the signed prediction error $(y_{\text{pred}} - y_{\text{true}})$ and the true directional Sobel spatial gradient $\nabla_d y_{\text{true}}$ along each spatial axis:
$$r_d = \left|\mathrm{corr}\left(y_{\text{pred}} - y_{\text{true}}, \nabla_d y_{\text{true}}\right)\right|$$
By Taylor expansion, spatial misregistration causes error directly proportional to directional gradients ($y(\mathbf{x} - \boldsymbol{\delta}) - y(\mathbf{x}) \approx -\boldsymbol{\delta} \cdot \nabla y(\mathbf{x})$).
- **Target**: $|r_d| < 0.02$ across all axes.
- **Function**: `siq.compute_edge_error_correlation(y_true, y_pred)`

### C. Structural & Artifact Metrics
Evaluated using native `siq` metric routines:
- PSNR & SSIM vs Bilinear Baseline
- GMSD (Gradient Magnitude Similarity Deviation)
- CBI (Checkerboard Index on prediction error)
- CQS (Composite Quality Score: $\text{SSIM} - \text{GMSD} - \text{CBI}$)
- PCS (Perceptual Composite Score)

---

## 3. QC Status Gates & Thresholds

| Status | Phase Shift $\|\Delta\|_\infty$ | Edge Correlation $\max_d |r_d|$ | PSNR Delta vs Bilinear | Action Required |
|:---:|:---:|:---:|:---:|:---|
| **✅ PASS** | $< 0.08$ voxels | $< 0.020$ | $\ge -0.2$ dB | None (model is zero-lag aligned and sharp). |
| **⚠️ WARN** | $0.08 - 0.15$ voxels | $0.020 - 0.050$ | $-1.0$ to $-0.2$ dB | Monitor; verify coordinate flipping logic. |
| **❌ FAIL** | $\ge 0.15$ voxels | $\ge 0.050$ | $< -1.0$ dB | Critical misregistration; check deconvolution shift. |

---

## 4. Python API Usage

```python
import siq

# 1. Run full automated QC check (<35ms overhead)
qc_results = siq.compute_alignment_qc(
    y_true=hr_image,
    y_pred=sr_image,
    bilinear=bilinear_image,
    phase_thresh=0.08,
    edge_corr_thresh=0.02,
    verbose=True,
)

print(qc_results["status"])        # 'PASS', 'WARN', or 'FAIL'
print(qc_results["summary_table"])  # Formatted ASCII diagnostics table
```

---

## 5. Standalone CLI Tool: `scripts/sr_alignment_qc.py`

Run standalone QC directly on NIfTI volumes or Keras model checkpoints:

```bash
# Evaluate a trained model against a validation scan:
python scripts/sr_alignment_qc.py \
    --model dbpn_small_3d_champion.keras \
    --val-image path/to/validation_t1.nii.gz \
    --output-json reports/model_qc.json \
    --output-html reports/model_qc.html

# Evaluate pre-generated GT and SR volumes:
python scripts/sr_alignment_qc.py \
    --gt path/to/gt.nii.gz \
    --sr path/to/sr.nii.gz \
    --bilinear path/to/bilinear.nii.gz \
    --output-html reports/sr_qc.html
```

---

## 6. Training Integration (`VisualConvergenceReporter`)

The QC engine is automatically executed at every validation checkpoint:
1. `VisualConvergenceReporter` runs `compute_alignment_qc` on the validation slice/volume.
2. Metrics are appended to `convergence_history.csv`:
   - `val_phase_shift_y`, `val_phase_shift_x`, `val_phase_shift_z`
   - `val_edge_corr_y`, `val_edge_corr_x`, `val_edge_corr_z`
   - `qc_status`
3. The interactive HTML dashboard renders a dedicated **SR Alignment & QC Status** card displaying live status badges and phase/edge correlation vectors.

---

## 7. Pre-flight Pair Audit & Champion Gating

- `siq.audit_pair_alignment(gen, factor)` runs at the start of `train_blind_sr_kitchen_sink` (~1-2 s) and aborts when any training pair has a non-zero integer decimation offset. This catches the "reflection applied after pair formation" bug **before** training. Bypass: `SIQ_SKIP_ALIGNMENT_AUDIT=1`.
- `VisualConvergenceReporter` blocks champion promotion when `qc_status == "FAIL"`.
- Diagnostic recipe: for each flip state, `siq.decimation_offset(lr, hr, factor)`. Aligned pairs give `(0, 0, 0)`; a reflected-after-degradation pair gives `f_d - 1` on the reflected axes.
- Note: `compute_phase_shift` uses `normalization=None` (the default phase normalization biases sub-voxel estimates by ~20%).

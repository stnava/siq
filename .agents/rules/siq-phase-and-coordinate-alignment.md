---
trigger: always_on
---

# siq Coordinate & Phase Alignment Invariants

## 1. Sub-Voxel Receptive Field Phase Shift (`align_phase=True`)

Transposed convolutions (`ConvTranspose` / `Conv2DTranspose` / `Conv3DTranspose`) and nearest/bilinear upsamplers expand an input voxel $k$ into $f$ child voxels along each upsampled axis.

In continuous index coordinates:
- The receptive field center of mass of those $f$ output voxels is located at $f \cdot k + \frac{f - 1}{2}$.
- In contrast, ANTs/ITK physical spatial grids place the continuous center of voxel $k$ at $f \cdot k + 0.0$.
- For factor $f = 2$, this introduces an exact $+0.5$-voxel phase shift along each upsampled dimension.

### The Error Gradient Ridge Artifact
Without phase compensation, residual error approximates $(y_{\text{pred}} - y_{\text{true}}) \approx 0.5 \nabla y_{\text{true}}$. Every sharp anatomical boundary exhibits bright halo ridges in residual difference maps, severely correlating with true spatial gradients ($|r_y| \approx 0.32$) and dragging PSNR down by 8–10 dB.

### Mandatory Compensation
In `siq.inference()`, `siq.overlapping_patch_inference()`, and any super-resolution inference pipeline:
- `align_phase=True` must ALWAYS be active by default.
- The output array must be shifted by $-\frac{f_d - 1}{2}$ voxels along each upsampled axis $d$ using cubic spline shift (`scipy.ndimage.shift(mode='nearest')`).

## 2. Direction-Cosine Sign Invariant (ANTs ↔ Deep Learning Grid)

Real NIfTI MRI scans (e.g. LPS or RAS orientations) often have negative diagonal elements in their direction cosine matrix (`direction[i,i] < 0`).
- Deep learning tensor libraries (NumPy, PyTorch, Keras) operate strictly in standard array index order where indices increase along canonical positive directions.
- Direct evaluation without direction sign alignment presents mirrored anatomy to models trained on synthetic identity-direction data.

### Mandatory Axis Flipping
Before executing `model.predict()`:
- Identify axes where `direction[i,i] < 0`.
- Flip those spatial axes in NumPy array format prior to inference.
- Flip the output array back along the same axes before re-attaching ANTs image metadata.

## 3. Training Generator Spacing & Crop Alignment Invariants

In `siq.blind_sr_generator` and `blind_sr_generator_simple`:
- **Spacing Invariant**: Downsampling to low-resolution must ALWAYS use `use_voxels=False` with exact factor spacing:
  `lr_target_spacing = tuple(float(s * f) for s, f in zip(image.spacing, factor_tuple))`
  Using `use_voxels=True` gives spacing $\approx \frac{N-1}{N/f-1} \neq f$, introducing cumulative fractional pitch errors across patches.
- **Crop Alignment Invariant**: High-resolution patch start indices must be strictly locked to low-resolution patch start indices:
  `hr_starts = [lr_starts[i] * factor_tuple[i] for i in range(dim)]`
  Never allow independent integer division `(large_shape - patch_shape) // 2` to introduce a 1-voxel rounding mismatch.

## 4. Native Metrics & Scale Alignment Invariant

- Never use `antspynet.psnr` or `antspynet.ssim`. They apply internal joint min-max normalization (`max(img1, img2)`), corrupting metrics whenever models produce $[0, 1]$ normalized outputs against raw integer GT.
- Always use native `siq.compute_psnr()`, `siq.compute_ssim()`, `siq.compute_gmsd()`, `siq.compute_hfen()`, and `siq.compute_checkerboard_index()`.
- Ground truth and super-resolved images must strictly be verified to span $[0, 1]$ before passing into metric functions.

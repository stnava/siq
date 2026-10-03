---
trigger: always_on
---

# siq Coordinate & Phase Alignment Invariants

## 1. Sub-Voxel Receptive Field Phase Shift & Architecture Scope

Transposed convolutions (`ConvTranspose` / `Conv2DTranspose` / `Conv3DTranspose`) expand an input voxel $k$ into $f$ child voxels along each upsampled axis:
- The continuous receptive field center of mass of those $f$ output voxels is located at $f \cdot k + \frac{f - 1}{2}$.
- In contrast, ANTs/ITK physical spatial grids place the continuous center of voxel $k$ at $f \cdot k + 0.0$.
- For factor $f = 2$, this introduces an exact $+0.5$-voxel phase shift along each upsampled dimension.

### Architecture Scope Invariant for Phase Shift:
- **Transposed-Convolution Models (AS-DBPN)**: Always apply $-\frac{f_d - 1}{2}$ phase shift along upsampled axes ($f_d > 1$) using cubic spline shift (`scipy.ndimage.shift(mode='nearest')`).
- **Sub-Pixel & Nearest-Neighbor Back-Projection Models (DBPN, L-DBPN, ESPCN, WDSR)**: Do NOT apply transposed-convolution phase shift. These architectures natively place child voxels on the discrete integer grid without deconvolution phase error; applying a $-0.5$ shift corrupts true alignment.

## 2. Direction-Cosine Sign Invariant & Discrete Reflection Compensation

Real NIfTI MRI scans (e.g. LPS or RAS orientations) often have negative diagonal elements in their direction cosine matrix (`direction[i,i] < 0`).
- Deep learning tensor libraries (NumPy, PyTorch, Keras) operate strictly in standard array index order where indices increase along canonical positive directions.
- Direct evaluation without direction sign alignment presents mirrored anatomy to models trained on synthetic identity-direction data.

### Mandatory Axis Flipping & Discrete Reflection Compensation
Before executing `model.predict()`:
1. Identify axes where `direction[i,i] < 0` (`flip_axes`).
2. Flip those spatial axes in NumPy array format prior to inference.
3. Flip the model output back along the same axes.
4. **Discrete Reflection Compensation**:
   Flipping an input of size $N$ ($k' = N - 1 - k$), upsampling by $f$, and flipping the output of size $fN$ back reverses the child voxel ordering within each parent voxel, introducing a $+ (f_d - 1)$ voxel shift along each flipped axis (+1.0 voxel for $f=2$).
   - For models without transposed convolutions (`DBPN`, `L-DBPN`, `ESPCN`, `WDSR`), shift by $-(f_d - 1)$ along each flipped axis $d$ to restore exact zero-lag physical coordinate alignment.
   - For transposed-convolution models (`AS-DBPN`), the internal transposed convolution phase shift and discrete reflection shift combine such that shifting by $-\frac{f_d - 1}{2}$ along all upsampled axes achieves zero-lag alignment.

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

## 5. Pair-Formation Invariant: Augment HR First, Degrade Second (Never Reflect a Formed Pair)

The training convention is decimation-aligned: LR voxel `k` sits at HR coordinate `f·k` (`use_voxels=False`, exact factor spacing, shared origin). This is **not reflection symmetric**:
- `np.flip` (or any `np.rot90`, which contains a reflection) applied to an already-formed (LR, HR) pair moves the alignment to HR offset `f_d - 1` along each reflected axis (verified: best HR decimation offset becomes 1 instead of 0 for `f = 2`).
- The network then trains on a mixture of aligned and shifted targets and learns a partial systematic shift (observed: ~0.2 voxel, PSNR -1 dB vs bilinear within 50 steps).

Rules:
- Geometric augmentation (flips, rot90, transposes) must be applied to the **HR volume before** blur/resample/crop, i.e. via `siq.augment_geometry_hr`. LR is always derived from the augmented HR.
- Never call `np.flip`/`np.rot90`/`fliplr`/`flipud` on training pairs in `siq/blind_sr.py` (enforced by `tests/test_alignment_contract.py`).
- Every training script runs `siq.audit_pair_alignment` once at start (`siq.decimation_offset` must be all zeros); bypass only with `SIQ_SKIP_ALIGNMENT_AUDIT=1`.
- A checkpoint whose alignment QC status is `FAIL` must never be promoted to champion.
- Inference-side flip compensation uses `siq.reflection_shift(factor, flipped_axes)` = `(f_d - 1)` per flipped axis.

## 6. Mandatory Alignment QC

Use `siq.compute_alignment_qc` (FFT phase lag, directional edge-error correlation with the **signed** error, PSNR/SSIM vs bilinear). Thresholds: PASS phase < 0.08 vox and edge-corr < 0.02; FAIL phase >= 0.15, edge-corr >= 0.05, or PSNR < bilinear - 1 dB. It runs at every validation checkpoint in `VisualConvergenceReporter`; the standalone CLI is `scripts/sr_alignment_qc.py` (see the `siq-qc` skill).

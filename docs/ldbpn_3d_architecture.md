# 3D Lightweight Deep Back-Projection Network (L-DBPN 3D)
## Architecture, Empirical Evidence, and Recommended Full 3D Training Pipeline

---

## 1. Executive Summary & Recommended Defaults

The **3D Lightweight Deep Back-Projection Network** ([`siq.create_ldbpn_3d`](file:///Users/stnava/code/siq/siq/espcn.py#L373)) is the recommended architecture for full 3D volumetric medical super-resolution in `siq`.

It synthesizes the mathematical rigor of iterative residual back-projection error correction (Haris et al., CVPR 2018) with sub-pixel convolution ([`PixelShuffle3D`](file:///Users/stnava/code/siq/siq/espcn.py#L40)) and compact $3\times 3\times 3$ convolutional filters.

```python
import siq

# Default production instantiation:
model = siq.create_ldbpn_3d(
    input_shape=(None, None, None, 1),
    factor=(1, 1, 2),  # or isotropic factor=2
    n_filters=32,      # f=32
    n_stages=3         # s=3
)
```

### Parameter Specification

| Hyperparameter | Recommended Value | Empirical Rationale |
|:---|:---:|:---|
| **Stages (`n_stages` / `s`)** | **`3`** | Optimal depth for iterative error correction: provides 3 progressive stages of residual feedback while keeping computational graph depth and latency low. |
| **Base Filters (`n_filters` / `f`)** | **`32`** | Sweet spot for 3D volumetric representation: yields 2.06M total parameters (7.85 MB FP32), fitting comfortably in GPU VRAM alongside large 3D training batches. |
| **Upsampling Primitive** | **`PixelShuffle3D`** | Eliminates transposed-convolution checkerboard grid artifacts ($\text{CBI} \approx 0$) by mapping channels directly into sub-voxels. |
| **Kernel Size** | **$3\times 3\times 3$** | Requires only 27 weights per filter—an **$8\times$ reduction in MACs** compared to legacy $6\times 6\times 6$ kernels (216 weights). |
| **Edge Loss Weight** | **`--edge-weight 0.0`** | Back-projection units naturally synthesize sharp structural boundaries without auxiliary edge gradient loss terms. |
| **Anti-Checkerboard Weight** | **`--cbi-weight 0.0`** | `PixelShuffle3D` is immune to deconvolution ringing; regularizer is unneeded and avoids unnecessary smoothing. |
| **Linear Blend Fraction** | **`--linear-blend 1.0`** | Pure super-resolution model output; avoids compromising high-frequency anatomical detail with linear interpolation blending. |
| **Model Selection Metric** | **`--selection-metric pcs`** | Perceptual Composite Score ($\text{PCS} = \text{SSIM} + 0.5 \text{Acutance} + 0.5 \text{Laplacian} - \text{GMSD} - \text{CBI}$) selects true anatomical sharpness rather than blurred PSNR averages. |

---

## 2. Empirical Benchmark Matrix

Benchmarked on identical hardware (Apple Silicon M-series unified memory) using standard $32 \times 32 \times 32$ single-channel 3D MRI patches:

| Model Architecture | Back-Projection Mechanism | Upsampling Type | Parameters | Model Size (FP32) | Forward Pass ($32^3$) | Checkerboard Susceptibility |
|:---|:---|:---|---:|---:|---:|:---:|
| **`ldbpn_3d (s=3, f=32)` (Recommended)** | **True Residual Back-Projection** | **`PixelShuffle3D`** | **2,057,281** | **7.85 MB** | **273.2 ms** | **Zero (Immune)** |
| `ldbpn_3d (s=2, f=32)` (Ultra-fast) | True Residual Back-Projection | `PixelShuffle3D` | 1,277,089 | 4.87 MB | 161.4 ms | Zero (Immune) |
| `ldbpn_3d (s=3, f=48)` (High-capacity) | True Residual Back-Projection | `PixelShuffle3D` | 4,625,761 | 17.65 MB | ~580 ms | Zero (Immune) |
| `default_dbpn('tiny')` | True Residual Back-Projection | Nearest + $3^3$ Conv | 427,233 | 1.63 MB | 346.9 ms | Low (Step artifacts) |
| `default_dbpn('small')` | True Residual Back-Projection | Nearest + $6^3$ Conv | 3,294,561 | 12.57 MB | 852.1 ms | Low (Step artifacts) |
| `default_dbpn('large')` | True Residual Back-Projection | Nearest + $6^3$ Conv | 22,286,721 | 85.02 MB | ~2,100 ms | Low (Step artifacts) |
| `asdbpn_3d (s=4, f=64)` | Recurrent Loop (No error term) | $6^3$ `Conv3DTranspose` | 1,836,706 | 7.01 MB | 960.1 ms | **Severe ($\text{CBI} \approx 0.073$)** |

---

## 3. Findings & Rationale from Recent Experiments

### A. True Back-Projection vs. Recurrent Autoencoders (DBPN vs. AS-DBPN)
In AS-DBPN (`create_asdbpn_3d`), features pass through a recurrent loop of shared `Conv3DTranspose` and strided `Conv3D`. However, AS-DBPN **does not compute residual difference blocks**. It functions as a recurrent autoencoder.

In contrast, L-DBPN 3D computes explicit spatial error corrections at both resolutions in each stage:

```
[Up-Projection Unit]
L_in --------> Conv3D + PixelShuffle3D ------> H_0 ------------------------(+)--> H_out
                     |                          |                           ^
                     |                          v                           |
                     |                   Strided Conv3D                     |
                     |                          |                           |
                     v                          v                           |
                    (-) <-------------------- L_0                           |
                     |                                                      |
                     +---> e_lr ---> Conv3D + PixelShuffle3D ---> e_hr -----+

[Down-Projection Unit]
H_in --------> Strided Conv3D ---------------> L_0 ------------------------(+)--> L_out
                     |                          |                           ^
                     |                          v                           |
                     |              Conv3D + PixelShuffle3D                 |
                     |                          |                           |
                     v                          v                           |
                    (-) <-------------------- H_0                           |
                     |                                                      |
                     +---> e_hr ------------> Strided Conv3D ---> e_lr -----+
```

1. **Residual Self-Correction**: $e_{\text{lr}} = L_{\text{in}} - \text{Down}(H_0)$ explicitly calculates what information was lost when the high-resolution estimate was downsampled back to low-resolution space.
2. **Dense Multi-Scale Concatenation**: Intermediate representations from all stages $[H_0, H_1, H_2]$ are concatenated before the reconstruction block:
   $$H_{\text{final}} = \text{Concat}([H_0, H_1, H_2])$$
   This ensures the final convolution receives coarse spatial context and fine high-frequency edge corrections simultaneously.

### B. Auxiliary Edge Loss (`--edge-weight 0.0`) is Redundant
Experiments comparing standard DBPN with and without auxiliary edge loss demonstrated:
- With `--edge-weight 0.0`, DBPN naturally achieved **`0.6300` Laplacian detail** and **`0.8743` acutance ratio** in early stages, outperforming models trained with explicit edge loss penalties.
- Computing edge loss on gradient magnitude differences or GMS variance tends to reward blurred, dispersed edges under sub-voxel spatial uncertainty.
- Because back-projection units directly penalize residual high-frequency downsampling error, auxiliary gradient loss terms are superfluous. **Keep `--edge-weight 0.0`**.

### C. Immunity to Transposed-Convolution Checkerboard Grid Resonance
- Transposed convolutions with stride $> 1$ (`Conv3DTranspose`) produce periodic filter overlaps, creating Nyquist-frequency grid resonance ($\text{CBI} \approx 0.073$, $3.7\times$ ground truth).
- `PixelShuffle3D` reorganizes $C \times (f_x f_y f_z)$ feature channels into spatial sub-voxels via array reshaping without spatial overlap.
- Furthermore, `apply_icnr_initialization` initializes preceding convolutions with tiled weights, guaranteeing uniform nearest-neighbor interpolation at iteration 0. As a result, **`--cbi-weight 0.0`** is optimal, preventing unnecessary anti-aliasing blur.

### D. Computational Scaling: $3\times 3\times 3$ vs. $6\times 6\times 6$ Kernels
In 3D convolutions, volume scales cubically:
- $6\times 6\times 6$ kernel = $216$ weights per channel pair.
- $3\times 3\times 3$ kernel = $27$ weights per channel pair (**$8\times$ fewer MACs**).

L-DBPN achieves **273 ms** forward pass latency on $32^3$ volumes, compared to **960 ms** for AS-DBPN and **852 ms** for legacy DBPN-Small.

---

## 4. Python API: Real Code Examples

### A. Model Instantiation & Inspection
```python
import keras
import numpy as np
import siq

# 1. Instantiate recommended 3D model for anisotropic (1x1x2) acquisition
model = siq.create_ldbpn_3d(
    input_shape=(None, None, None, 1),
    factor=(1, 1, 2),
    n_filters=32,
    n_stages=3
)

print(f"Total Parameters: {model.count_params():,}")
# Output: Total Parameters: 2,057,281

# 2. Verify forward pass on arbitrary patch dimensions
dummy_lr = np.random.randn(1, 16, 16, 16, 1).astype(np.float32)
sr_output = model(dummy_lr, training=False)
print("Input shape:", dummy_lr.shape)
print("Output shape:", sr_output.shape)
# Output: [1, 16, 16, 32, 1]  (doubled along z-axis)
```

### B. Understanding Sub-Pixel Convolution (`PixelShuffle3D`)
```python
import keras
from siq.espcn import PixelShuffle3D

# Factor (1, 1, 2) expands channels by 1 * 1 * 2 = 2
# An input with 64 channels becomes a spatial volume with 2x depth and 32 channels:
shuffle = PixelShuffle3D(factor=(1, 1, 2))
features = keras.ops.zeros((1, 16, 16, 16, 64))
upsampled = shuffle(features)
print("Upsampled shape:", upsampled.shape)
# Output: [1, 16, 16, 32, 32]
```

### C. Extracting Multi-Stage Intermediate Projections
Because L-DBPN preserves all stage representations in its computational graph, intermediate reconstructions can be extracted for analysis:
```python
# Extract intermediate high-resolution estimates [H_0, H_1, H_2]
stage_outputs = [
    model.get_layer("stage_0_up_add").output,
    model.get_layer("stage_1_up_add").output,
    model.get_layer("stage_2_up_add").output,
    model.output
]
multi_stage_model = keras.Model(inputs=model.inputs, outputs=stage_outputs)

h0, h1, h2, final_sr = multi_stage_model(dummy_lr)
print("Stage 0 shape:", h0.shape)
print("Stage 1 shape:", h1.shape)
print("Stage 2 shape:", h2.shape)
print("Final SR shape:", final_sr.shape)
```

---

## 5. Recommended Full 3D Training Pipeline (The Canonical Command)

Based on the 4-stage curriculum staging invariant, prefetch optimizations, and LOWESS loss auto-balancing, the following command represents the state-of-the-art recipe for training a production 3D L-DBPN model from scratch.

### A. Anisotropic Clinical Protocol (e.g. $1\times 1\times 2$ mm Thick Slices)

```bash
PYTHONUNBUFFERED=1 python scripts/train_model_refinement.py ldbpn \
  --dim 3 \
  --factor 1 1 2 \
  --batch-size 4 \
  --from-scratch \
  --reset-history \
  --stage1-iter 500 \
  --stage2-iter 1500 \
  --stage3-iter 5000 \
  --target-percep 65.0 \
  --target-mae 30.0 \
  --target-tv 5.0 \
  --edge-weight 0.0 \
  --cbi-weight 0.0 \
  --linear-blend 1.0 \
  --selection-metric pcs \
  --checkpoint-freq 25 \
  --prefetch-size 4 \
  --balancer-freq 25 \
  --smooth-window 100 \
  --dampening 0.97
```

### B. Isotropic Protocol (e.g. $2\times 2\times 2$ mm Downsampled Acquisition)

```bash
PYTHONUNBUFFERED=1 python scripts/train_model_refinement.py ldbpn \
  --dim 3 \
  --factor 2 2 2 \
  --batch-size 4 \
  --from-scratch \
  --reset-history \
  --stage1-iter 500 \
  --stage2-iter 1500 \
  --stage3-iter 5000 \
  --target-percep 65.0 \
  --target-mae 30.0 \
  --target-tv 5.0 \
  --edge-weight 0.0 \
  --cbi-weight 0.0 \
  --linear-blend 1.0 \
  --selection-metric pcs \
  --checkpoint-freq 25 \
  --prefetch-size 4 \
  --balancer-freq 25 \
  --smooth-window 100 \
  --dampening 0.97
```

---

## 6. Detailed Flag Rationale

| CLI Flag | Value | Rationale |
|:---|:---:|:---|
| `ldbpn` | Positional | Selects `siq.create_ldbpn_3d` with recommended defaults ($s=3, f=32$). |
| `--dim 3` | `3` | Full 3D volumetric tensor pipeline with 3D patch extraction and validation. |
| `--factor 1 1 2` | `(1, 1, 2)` | Matches common clinical MRI anisotropic slice thickness acquisitions. |
| `--from-scratch` | Flag | Applies ICNR initialization to `PixelShuffle3D` preceding convolutions and starts optimization fresh. |
| `--stage1-iter 500` | `500` | **Mandatory Curriculum Invariant**: Never abbreviate Stage 1 below 500 iterations. Allows the LOWESS dynamic balancer to smoothly settle $L_1$/Feat/TV weights on clean data without shock. |
| `--stage2-iter 1500` | `1500` | Injects Rician noise (`noise_std_range=(0.0, 0.02)`) and stochastic zoom (`0.75, 1.3`) once loss weights are stationary. |
| `--stage3-iter 5000` | `5000` | Dedicated high-fidelity anatomical refinement with fine learning rate ($2\times 10^{-5}$) and PCS checkpoint selection. |
| `--target-percep 65.0` | `65.0%` | Anchors feature/perceptual loss contribution to 65% of the total gradient budget. |
| `--target-mae 30.0` | `30.0%` | Anchors spatial $L_1$ fidelity to 30% of total gradient budget. |
| `--target-tv 5.0` | `5.0%` | Anchors total variation smoothness regularization to 5%. |
| `--edge-weight 0.0` | `0.0` | Disables unproven gradient edge loss; DBPN back-projection handles gradient steepness natively. |
| `--cbi-weight 0.0` | `0.0` | Disables checkerboard regularization; `PixelShuffle3D` eliminates deconvolution gridding. |
| `--linear-blend 1.0` | `1.0` | Disables linear baseline blending, producing pure super-resolved anatomical sharpness. |
| `--selection-metric pcs` | `pcs` | Selects champion model using Perceptual Composite Score ($\text{SSIM} + 0.5 \text{Acutance} + 0.5 \text{Laplacian} - \text{GMSD} - \text{CBI}$). |
| `--prefetch-size 4` | `4` | Spawns background CPU worker threads to synthesize procedural 3D MRI patches asynchronously, hiding I/O latency. |
| `--balancer-freq 25` | `25` | Amortizes diagnostic forward passes across 25 steps, reducing training wall-clock time by ~60%. |

---

## 7. Inference & Coordinate Alignment

When evaluating trained L-DBPN 3D checkpoints on real NIfTI volumes:

```python
import ants
import siq

# Load model and its provenance companion config
model, config = siq.load_siq_model("ldbpn_3d_best_mdl.keras")

# Load real MRI volume
img_lr = ants.image_read("patient_t1_lowres.nii.gz")

# Full volumetric inference with automatic coordinate alignment and phase shift correction
img_sr = siq.inference(
    img_lr,
    model=model,
    config=config,
    align_phase=True,    # Compensates sub-voxel -0.5 voxel receptive field phase offset
    verbose=True
)

ants.image_write(img_sr, "patient_t1_superres.nii.gz")
```

### Invariants Observed:
1. **Receptive Field Phase Shift (`align_phase=True`)**:
   `scipy.ndimage.shift` compensates for the $-\frac{f_d - 1}{2}$ continuous coordinate offset between continuous deep learning convolution centers and physical ITK voxel grids.
2. **Direction Cosine Alignment**:
   Inference checks `image.direction` for negative diagonal elements, flips NumPy axes before the neural network forward pass, and flips them back before attaching ANTs image headers.
3. **Artifact Prevention**:
   No sub-voxel deconvolution notch filtering is required because `PixelShuffle3D` produces clean, Nyquist-bounded Fourier spectra.

# 2D 2x super-resolution reference results (siq v0.10.20)

Generated with `siq.sr_benchmark` (see `.agents/skills/siq-benchmark/SKILL.md`):

```
python scripts/sr_reference_benchmark.py --public edsr-base --public msrn-bam \
  --model real_mse_matched=<results/2d_matched_real_mse/..._latest.keras> \
  --model fast_curriculum_procedural=<results/2d_curriculum/fast_s0/..._latest.keras> \
  --out docs/benchmarks --name sr_reference_2d_2x
```

**Case sets.** `r16c`: cropped ANTs r16 (1 slice, nearest decimation, out-of-domain smooth image).
`heldout`: 24 real BLAST/EAS axial T1 slices from 8 subjects disjoint from the 40-subject training cache,
nearest decimation (siq training contract). `heldout_aa`: same slices with Gaussian(1) anti-aliasing before
decimation (the degradation public networks were trained for). Public models use the analytic half-pixel
phase correction (`phase="half-pixel"`; `"auto"` agrees to 0.03 voxel).

**Models.**
* `real_mse_matched` - DBPN-small, plain MSE, 1200 steps, trained on the real-slice cache with the nearest-decimation degradation
  (`train_2d_curriculum.py --cache ... --degradation matched --mse-only --iters 1200 0 0 0`).
* `fast_curriculum_procedural` - DBPN-small, 4-stage perceptual curriculum (`--fast`, ~10 min) on procedural simulated images.
* `PUBLIC eugenesiow/*` - super-image pretrained EDSR-base / MSRN-bam (DIV2K natural images).

**Reading the table.** CQS decides (PSNR is informational; it rewards blur). `PCSc` bounds the sharpness reward that
raw `PCS` leaves unbounded. A model has *learned gain* only if it beats the in-sample linear oracle ("beats ceiling").

**Findings at this commit.**
1. Only `real_mse_matched` on `heldout` (in-domain) beats best-classical and the linear ceiling (CQS +0.0084 / +0.0048); the margin is small.
2. Procedurally-trained models lose to bilinear on real held-out slices (CQS -0.056 vs bilinear on `heldout`): never validate on procedural data alone.
3. Public EDSR/MSRN are a valid reference only with anti-aliased input + phase correction (`heldout_aa`: +0.10 CQS over bilinear, still 0.0085 below the linear oracle). On nearest-decimated input they amplify aliasing and lose.
4. Under nearest decimation the linear headroom is tiny (oracle +0.011 CQS over bilinear); under anti-aliasing it is large (+0.11).

### r16c

| method | psnr | ssim | gmsd | cbi | cqs | pcs | pcsc | acutance | laplacian |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| nearest | 23.39 | 0.8961 | 0.1709 | 0.0402 | 0.6850 | 2.0381 | 0.3318 | 1.0110 | 1.6953 |
| bilinear | 28.76 | 0.9671 | 0.1255 | 0.0255 | 0.8161 | 1.5339 | 0.5339 | 0.8271 | 0.6086 |
| bspline | 29.63 | 0.9733 | 0.1126 | 0.0245 | 0.8362 | 1.6808 | 0.6808 | 0.9707 | 0.7185 |
| windowed-sinc | 29.66 | 0.9722 | 0.1154 | 0.0245 | 0.8323 | 1.7039 | 0.7039 | 0.9918 | 0.7515 |
| linear-LS REAL (K=9) | 27.88 | 0.9593 | 0.1388 | 0.0255 | 0.7950 | 1.5655 | 0.5655 | 0.7501 | 0.7909 |
| linear-LS oracle in-sample (K=9) | 30.49 | 0.9731 | 0.1140 | 0.0243 | 0.8348 | 1.6189 | 0.6189 | 0.8926 | 0.6754 |
| MODEL real_mse_matched | 30.67 | 0.9695 | 0.1159 | 0.0243 | 0.8293 | 1.6402 | 0.6402 | 0.9120 | 0.7096 |
| MODEL fast_curriculum_procedural | 28.39 | 0.9618 | 0.1334 | 0.0276 | 0.8009 | 1.7036 | 0.7036 | 0.9323 | 0.8732 |
| PUBLIC eugenesiow/edsr-base | 27.32 | 0.9528 | 0.1334 | 0.0394 | 0.7801 | 2.0824 | 0.4777 | 1.3010 | 1.3037 |
| PUBLIC eugenesiow/msrn-bam | 26.98 | 0.9513 | 0.1348 | 0.0424 | 0.7741 | 2.1191 | 0.4291 | 1.3264 | 1.3635 |
| bil+unsharp best-CQS (s=2.0,a=0.25) | 29.00 | 0.9701 | 0.1191 | 0.0256 | 0.8254 | 1.6925 | 0.6925 | 0.9991 | 0.7351 |
| bil+unsharp best-PCSc (s=1.0,a=1.0) | 27.78 | 0.9602 | 0.1274 | 0.0288 | 0.8040 | 1.8867 | 0.7214 | 1.1497 | 1.0157 |

**Verdicts (CQS; PSNR is informational):**

| method | Δ vs bilinear | Δ vs best classical | Δ vs linear ceiling | beats classical | beats ceiling |
|:--|:-:|:-:|:-:|:-:|:-:|
| MODEL real_mse_matched | +0.0133 | -0.0069 | -0.0055 | no | no |
| MODEL fast_curriculum_procedural | -0.0152 | -0.0353 | -0.0340 | no | no |
| PUBLIC eugenesiow/edsr-base | -0.0360 | -0.0561 | -0.0547 | no | no |
| PUBLIC eugenesiow/msrn-bam | -0.0420 | -0.0621 | -0.0607 | no | no |

### heldout

| method | psnr | ssim | gmsd | cbi | cqs | pcs | pcsc | acutance | laplacian |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| nearest | 23.13 | 0.8274 | 0.1739 | 0.0969 | 0.5566 | 1.6132 | 0.4895 | 1.0026 | 1.1105 |
| bilinear | 26.98 | 0.9105 | 0.1449 | 0.0700 | 0.6956 | 1.2588 | 0.2588 | 0.6985 | 0.4278 |
| bspline | 27.03 | 0.9121 | 0.1375 | 0.0712 | 0.7034 | 1.4226 | 0.4226 | 0.8982 | 0.5402 |
| windowed-sinc | 26.90 | 0.9100 | 0.1389 | 0.0718 | 0.6992 | 1.4478 | 0.4478 | 0.9330 | 0.5642 |
| linear-LS REAL (K=9) | 26.46 | 0.8980 | 0.1472 | 0.0709 | 0.6799 | 1.2716 | 0.2716 | 0.6594 | 0.5240 |
| linear-LS oracle in-sample (K=9) | 27.44 | 0.9150 | 0.1386 | 0.0694 | 0.7070 | 1.3335 | 0.3335 | 0.7701 | 0.4830 |
| MODEL real_mse_matched | 26.76 | 0.9164 | 0.1374 | 0.0672 | 0.7118 | 1.4109 | 0.3845 | 0.8757 | 0.5227 |
| MODEL fast_curriculum_procedural | 25.43 | 0.8652 | 0.1537 | 0.0716 | 0.6399 | 1.4535 | 0.3868 | 0.9818 | 0.6454 |
| PUBLIC eugenesiow/edsr-base | 23.60 | 0.8670 | 0.1674 | 0.1085 | 0.5911 | 2.1062 | 0.0760 | 1.7839 | 1.2463 |
| PUBLIC eugenesiow/msrn-bam | 23.61 | 0.8676 | 0.1667 | 0.1093 | 0.5915 | 2.1100 | 0.0731 | 1.7848 | 1.2521 |
| bil+unsharp best-CQS (s=2.0,a=0.25) | 26.87 | 0.9137 | 0.1405 | 0.0706 | 0.7026 | 1.4317 | 0.4317 | 0.9283 | 0.5300 |
| bil+unsharp best-PCSc (s=1.0,a=0.5) | 26.68 | 0.9111 | 0.1425 | 0.0716 | 0.6970 | 1.4631 | 0.4631 | 0.9413 | 0.5910 |

**Verdicts (CQS; PSNR is informational):**

| method | Δ vs bilinear | Δ vs best classical | Δ vs linear ceiling | beats classical | beats ceiling |
|:--|:-:|:-:|:-:|:-:|:-:|
| MODEL real_mse_matched | +0.0161 | +0.0084 | +0.0048 | yes | yes |
| MODEL fast_curriculum_procedural | -0.0557 | -0.0635 | -0.0671 | no | no |
| PUBLIC eugenesiow/edsr-base | -0.1045 | -0.1123 | -0.1159 | no | no |
| PUBLIC eugenesiow/msrn-bam | -0.1041 | -0.1119 | -0.1154 | no | no |

### heldout_aa

| method | psnr | ssim | gmsd | cbi | cqs | pcs | pcsc | acutance | laplacian |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| nearest | 24.08 | 0.8298 | 0.1917 | 0.0797 | 0.5583 | 1.0844 | 0.0844 | 0.4304 | 0.6217 |
| bilinear | 25.45 | 0.8623 | 0.1883 | 0.0757 | 0.5983 | 0.8806 | -0.1194 | 0.3607 | 0.2041 |
| bspline | 26.24 | 0.8859 | 0.1709 | 0.0745 | 0.6405 | 0.9701 | -0.0299 | 0.4249 | 0.2343 |
| windowed-sinc | 26.31 | 0.8882 | 0.1685 | 0.0743 | 0.6454 | 0.9848 | -0.0152 | 0.4341 | 0.2448 |
| linear-LS REAL (K=9) | 25.27 | 0.8567 | 0.1885 | 0.0757 | 0.5925 | 0.9012 | -0.0988 | 0.3447 | 0.2727 |
| linear-LS oracle in-sample (K=9) | 28.09 | 0.9250 | 0.1351 | 0.0786 | 0.7113 | 1.4423 | 0.3878 | 0.8308 | 0.6314 |
| MODEL real_mse_matched | 22.58 | 0.8733 | 0.1754 | 0.0737 | 0.6243 | 1.0885 | 0.0867 | 0.6323 | 0.2960 |
| MODEL fast_curriculum_procedural | 22.15 | 0.8344 | 0.1778 | 0.0740 | 0.5827 | 1.1217 | 0.1159 | 0.6935 | 0.3846 |
| PUBLIC eugenesiow/edsr-base | 27.89 | 0.9152 | 0.1434 | 0.0691 | 0.7027 | 1.1105 | 0.1105 | 0.5029 | 0.3127 |
| PUBLIC eugenesiow/msrn-bam | 27.89 | 0.9152 | 0.1433 | 0.0691 | 0.7028 | 1.1104 | 0.1104 | 0.5033 | 0.3119 |
| bil+unsharp best-CQS (s=2.0,a=1.0) | 26.87 | 0.9117 | 0.1549 | 0.0737 | 0.6830 | 1.2496 | 0.2496 | 0.7517 | 0.3813 |
| bil+unsharp best-PCSc (s=2.0,a=1.0) | 26.87 | 0.9117 | 0.1549 | 0.0737 | 0.6830 | 1.2496 | 0.2496 | 0.7517 | 0.3813 |

**Verdicts (CQS; PSNR is informational):**

| method | Δ vs bilinear | Δ vs best classical | Δ vs linear ceiling | beats classical | beats ceiling |
|:--|:-:|:-:|:-:|:-:|:-:|
| MODEL real_mse_matched | +0.0260 | -0.0588 | -0.0870 | no | no |
| MODEL fast_curriculum_procedural | -0.0156 | -0.1004 | -0.1286 | no | no |
| PUBLIC eugenesiow/edsr-base | +0.1045 | +0.0197 | -0.0085 | yes | no |
| PUBLIC eugenesiow/msrn-bam | +0.1045 | +0.0198 | -0.0085 | yes | no |


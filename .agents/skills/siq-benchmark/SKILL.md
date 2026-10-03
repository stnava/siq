---
name: siq-benchmark
description: Standard SR benchmarking — reference baselines (classical, unsharp, linear least-squares ceiling), our models, and public pretrained models (super-image EDSR/MSRN/...) on common case sets with phase calibration and CQS verdicts. Use whenever a super-resolution model is evaluated, compared, or claimed to be good.
---

# siq SR benchmarking (`siq.benchmark`)

**Rule of thumb:** no SR claim without a reference table. A model must beat the best classical
interpolator / unsharp mask on CQS, and — to claim *learned* gain — the in-sample linear
least-squares ceiling. PSNR is informational only.

## One call
```python
import siq
results, md = siq.sr_benchmark(
    models={"mine": "path/to/model.keras"},          # siq models (+ _config.json)
    public=["edsr-base", "msrn-bam"],                # zoo names or HF ids (super-image)
    phase="half-pixel",                              # or "auto" (empirical Lucas-Kanade) or None / float
)
print(md)    # tables per case set + verdicts; also results/sr_reference/reference.{json,md}
```
CLI: `python scripts/sr_reference_benchmark.py --model mine=path.keras --public edsr-base [--phase auto] [--cases heldout_aa]`
List the public zoo: `--list-public` / `siq.list_public_models()`.

## Case sets (`siq.build_cases`)
| case | what | use for |
|:--|:--|:--|
| `r16c` | cropped ANTs r16, nearest decimation | smooth out-of-domain check |
| `heldout` | real axial T1 slices, subjects disjoint from the training cache, nearest decimation | **siq's training contract** |
| `heldout_aa` | same slices, Gaussian(1) anti-alias then decimation | what public networks were trained for |

Build caches once: `scripts/build_real_slice_cache.py` (training) and
`... --skip-subjects 40 --n-subjects 8 --stride 8 --out results/heldout_slice_cache.npy` (held-out).

## Adding a method
* siq model: `siq.siq_model_method(name, path)`.
* Any callable `fn(lr_ants, gt_ants) -> np.ndarray` (ONNX, Torch, OpenCV...): `siq.callable_method(name, fn, group="public", phase="half-pixel"|"auto"|float, factor=..., calibration_cases=...)`.
* New super-image architecture: add to `PUBLIC_ZOO` in `siq/benchmark.py` (repo id, class, verified flag).

## Phase / alignment (the dominant confound)
Natural-image SR networks assume half-pixel-centre alignment; siq is decimation-aligned. Without a
−(f−1)/2 voxel shift EDSR loses ~2–4 dB. `phase="half-pixel"` is the analytic fix; `"auto"` estimates
it with `compute_shift_lk` on calibration cases. Always report which was used (`_meta.phase_shift`).

## Installing public models
`pip install "siq[public-sr]"` (declared extra), or isolated from the main env:
`pip install --target /tmp/sipkgs super-image "huggingface_hub<0.30"`; adapters add
`$SIQ_PUBLIC_SR_PATH` (default `/tmp/sipkgs`) to `sys.path`. Weights download from HuggingFace on first use.

## Metrics
PSNR (info), SSIM, GMSD, CBI, CQS, PCS (repo definition) and **PCSc** =
SSIM − 0.5|1−acutance| − 0.5|1−laplacian| − GMSD − CBI. PCS rewards unbounded over-sharpening
(unsharp amount 2 → PCS 2.6 at acutance 2.4); prefer PCSc when comparing sharpness.

## Reference results (2026-10, 2× 2D)
See `results/sr_reference/reference.md`. Key facts: on nearest-decimated real slices the
linear ceiling is only ≈ +0.5 dB over bilinear; on anti-aliased slices EDSR (phase-corrected) gains
≈ +2.4 dB over bilinear, still below the per-image linear oracle (28.09 dB).

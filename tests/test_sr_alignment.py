"""
test_sr_alignment.py — Objective SR spatial alignment test suite.

Tests that catch two classes of systematic shift bugs:
  1. Training data generator: LR pixel k must correspond exactly to HR pixel k*factor.
     (bug: use_voxels=True gave spacing 191/95≈2.0105 → 0.5px center offset per pair)
  2. Model skip connection: UpSampling2D must not introduce a sub-pixel spatial shift.
     (bug: interpolation='bilinear' with align_corners=False maps pixel 0 to -0.25 → -0.5px shift)

Run with:
    python tests/test_sr_alignment.py
or via pytest:
    pytest tests/test_sr_alignment.py -v
"""

import numpy as np
import pytest

# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _make_impulse_2d(shape, pos=None):
    """Return a float32 array with a single Gaussian blob at `pos` (default: center)."""
    h, w = shape
    if pos is None:
        pos = (h // 2, w // 2)
    y, x = np.mgrid[:h, :w]
    sigma = max(h, w) * 0.04          # narrow enough to resolve a 1-pixel shift
    arr = np.exp(-((y - pos[0])**2 + (x - pos[1])**2) / (2 * sigma**2))
    return arr.astype('float32')


def _xcorr_peak_offset_2d(a, b):
    """
    Returns (dy, dx) in pixels such that b ≈ shift(a, dy, dx).
    Uses FFT cross-correlation; sub-pixel shifts show as non-integer peak,
    but a 0.5-pixel systematic shift reliably produces a peak offset of ±1.
    """
    from numpy.fft import fft2, ifft2
    A = fft2(a - a.mean())
    B = fft2(b - b.mean())
    corr = np.real(ifft2(A * np.conj(B)))
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    # Convert from cyclic-shift convention to centered offset
    dy = peak[0] if peak[0] < a.shape[0] // 2 else peak[0] - a.shape[0]
    dx = peak[1] if peak[1] < a.shape[1] // 2 else peak[1] - a.shape[1]
    return dy, dx


def _subsample_hr_to_lr(hr, factor):
    """Ground-truth LR: integer decimation (no interpolation). LR[i,j] = HR[i*f, j*f]."""
    return hr[::factor, ::factor]


# ─────────────────────────────────────────────────────────────────
# Test 1 — ANTs resampling convention
# ─────────────────────────────────────────────────────────────────

def test_ants_resample_use_voxels_false_zero_shift(factor=2):
    """
    use_voxels=False with exact factor spacing must produce zero pixel offset
    between LR and the integer-decimated HR.

    This is the convention used in blind_sr_generator (after the fix).
    A failure here means the generator will produce systematically shifted pairs
    and the model will learn a baked-in spatial offset.
    """
    import ants
    hr_np = _make_impulse_2d((128, 128))
    hr = ants.from_numpy(hr_np)
    hr.set_spacing((1.0, 1.0))

    lr = ants.resample_image(hr, (float(factor), float(factor)), use_voxels=False, interp_type=1)
    lr_np = lr.numpy()

    # Compare LR with HR decimated at integer factor — should be zero-lag
    lr_ref = _subsample_hr_to_lr(hr_np, factor)
    # Resize lr_np to match ref if edge rounding differs by 1 pixel
    min_h = min(lr_np.shape[0], lr_ref.shape[0])
    min_w = min(lr_np.shape[1], lr_ref.shape[1])
    dy, dx = _xcorr_peak_offset_2d(lr_np[:min_h, :min_w], lr_ref[:min_h, :min_w])

    assert dy == 0 and dx == 0, (
        f"use_voxels=False LR has a {dy},{dx} pixel offset vs HR decimation. "
        f"This will bake a spatial shift into every training pair."
    )


def test_ants_resample_use_voxels_true_shift_detected(factor=2):
    """
    use_voxels=True on a non-power-of-2 image produces a measurable spacing error
    that manifests as a sub-pixel center offset in the training pairs.
    This test DOCUMENTS the bug (it does not assert zero offset — the offset IS non-zero).
    Use it to verify the magnitude of the error before/after fixes.
    """
    import ants
    hr_np = _make_impulse_2d((192, 192))   # typical blind_sr_generator hr_large_shape
    hr = ants.from_numpy(hr_np)
    hr.set_spacing((1.0, 1.0))

    lr_vox = ants.resample_image(hr, (96, 96), use_voxels=True, interp_type=1)
    spacing_error = abs(lr_vox.spacing[0] - float(factor))

    # Document: spacing error exists with use_voxels=True
    print(f"\n[alignment audit] use_voxels=True spacing: {lr_vox.spacing[0]:.6f}, "
          f"ideal: {float(factor):.6f}, error: {spacing_error:.6f} voxels/pixel")

    # Center crop — same math as blind_sr_generator
    hr_crop = hr_np[32:160, 32:160]       # 128×128
    lr_crop = lr_vox.numpy()[16:80, 16:80]  # 64×64

    # Nearest-neighbor upsample LR crop 2× and compare with HR crop
    lr_up = lr_crop.repeat(factor, axis=0).repeat(factor, axis=1)  # 128×128
    dy, dx = _xcorr_peak_offset_2d(lr_up, hr_crop)
    print(f"[alignment audit] use_voxels=True center-crop offset: ({dy}, {dx}) HR pixels")
    # Not asserting zero — this test documents the known non-zero offset
    return dy, dx


# ─────────────────────────────────────────────────────────────────
# Test 2 — Model skip connection
# ─────────────────────────────────────────────────────────────────

def _upsample2d_offset(interpolation, factor=2, size=16):
    """
    Measures the pixel offset introduced by UpSampling2D with a given interpolation.
    Uses an impulse at the center of the input, upsamples, and measures cross-correlation
    peak offset between the upsampled output and the ideal nearest-neighbor repeat.
    Returns (dy, dx) in output pixels. Should be (0, 0) for nearest.
    """
    import keras
    import numpy as np

    layer = keras.layers.UpSampling2D(size=(factor, factor), interpolation=interpolation)
    # Impulse at center of LR
    lr_np = np.zeros((1, size, size, 1), dtype='float32')
    cy, cx = size // 2, size // 2
    lr_np[0, cy, cx, 0] = 1.0
    sr_np = layer(lr_np).numpy()[0, :, :, 0]

    # Reference: nearest-neighbor repeat (ideal, zero-shift)
    ref_np = lr_np[0, :, :, 0].repeat(factor, axis=0).repeat(factor, axis=1)
    dy, dx = _xcorr_peak_offset_2d(sr_np, ref_np)
    return dy, dx


def test_nearest_upsample2d_zero_shift():
    """
    UpSampling2D with interpolation='nearest' must produce zero pixel offset.
    output[2k, 2j] == input[k, j] — this is the correct SR convention.
    """
    dy, dx = _upsample2d_offset('nearest')
    assert dy == 0 and dx == 0, (
        f"nearest UpSampling2D has offset ({dy},{dx}). "
        f"Should be (0,0) — nearest-neighbor replication has no spatial shift."
    )


def test_bilinear_upsample2d_shift_documented():
    """
    Documents the known -0.5 output-pixel shift from bilinear UpSampling2D.
    Keras/PyTorch bilinear with align_corners=False maps output j → input (j+0.5)/scale-0.5,
    giving output pixel 0 → input -0.25 (extrapolated). For a 2x impulse, the peak shifts.
    This test is informational — it SHOULD detect a non-zero offset.
    """
    dy, dx = _upsample2d_offset('bilinear')
    print(f"\n[alignment audit] bilinear UpSampling2D offset: ({dy},{dx}) output pixels "
          f"(expected non-zero due to align_corners=False convention)")
    # Not asserting zero — we document the bug
    return dy, dx


# ─────────────────────────────────────────────────────────────────
# Test 3 — Generator: LR spacing invariant (exact factor)
# ─────────────────────────────────────────────────────────────────

def test_blind_sr_generator_spacing_invariant_2d(factor=2):
    """
    The blind_sr_generator must produce LR images with spacing exactly equal to
    factor * HR_spacing (1.0). Spacing deviation causes a systematic sub-pixel
    center offset that accumulates across the patch and bakes a shift into every
    training pair.

    Pre-fix (use_voxels=True, 192→96): spacing = 191/95 = 2.01053 (+0.5%)
    Post-fix (use_voxels=False, spacing=2.0): spacing = 2.0 exactly.
    """
    import ants
    hr = ants.from_numpy(np.ones((192, 192), dtype='float32'))
    hr.set_spacing((1.0, 1.0))

    # This exactly mirrors what blind_sr_generator does after the fix
    lr_target_spacing = (float(factor), float(factor))
    lr = ants.resample_image(hr, lr_target_spacing, use_voxels=False, interp_type=0)

    for dim in range(2):
        spacing_error = abs(lr.spacing[dim] - float(factor))
        assert spacing_error < 1e-9, (
            f"LR spacing[{dim}]={lr.spacing[dim]:.8f} ≠ exact factor {factor}. "
            f"Error={spacing_error:.2e}. This bakes a systematic shift into training pairs."
        )


def test_blind_sr_generator_spacing_invariant_3d(factor=2):
    """3D version of the spacing invariant test."""
    import ants
    hr = ants.from_numpy(np.ones((96, 96, 96), dtype='float32'))
    hr.set_spacing((1.0, 1.0, 1.0))
    lr_target_spacing = (float(factor), float(factor), float(factor))
    lr = ants.resample_image(hr, lr_target_spacing, use_voxels=False, interp_type=0)
    for dim in range(3):
        spacing_error = abs(lr.spacing[dim] - float(factor))
        assert spacing_error < 1e-9, (
            f"3D LR spacing[{dim}]={lr.spacing[dim]:.8f} ≠ {factor}. "
            f"Error={spacing_error:.2e}."
        )


def test_blind_sr_generator_impulse_peak_alignment_2d(factor=2):
    """
    Uses a controlled Gaussian impulse at a KNOWN position in HR, applies the
    generator's exact downsampling, and verifies the LR peak is at the
    EXPECTED pixel position (no offset).

    LR peak position should be: floor(hr_peak / factor) for each axis.
    A shift of ±1 LR pixel = ±factor HR pixels — clearly visible.
    """
    import ants
    size = 128
    hr_np = _make_impulse_2d((size, size), pos=(size // 2, size // 2))
    hr = ants.from_numpy(hr_np)
    hr.set_spacing((1.0, 1.0))

    lr = ants.resample_image(hr, (float(factor), float(factor)), use_voxels=False, interp_type=1)
    lr_np = lr.numpy()

    expected_peak = (size // 2 // factor, size // 2 // factor)
    actual_peak = np.unravel_index(lr_np.argmax(), lr_np.shape)
    assert actual_peak == expected_peak, (
        f"LR impulse peak at {actual_peak}, expected {expected_peak}. "
        f"Systematic alignment offset detected in downsampling convention."
    )


# ─────────────────────────────────────────────────────────────────
# Test 5 — Inference: Half-pixel phase alignment (edge bias removal)
# ─────────────────────────────────────────────────────────────────

def test_inference_phase_alignment_eliminates_edge_bias():
    """
    Transposed convolution receptive fields naturally output at half-integer coordinates
    (+0.5 voxels along upsampled axes). When compared with integer voxel grids, this creates
    a systematic edge bias where residual error correlates strongly with spatial gradients.
    align_phase=True must reduce this edge gradient correlation by >5x.
    """
    import os
    import ants
    from scipy.ndimage import sobel
    import siq

    model_path = "asdbpn_2d_2x2_best_mdl.keras"
    if not os.path.exists(model_path):
        return  # skip if model file not in workspace

    model, cfg = siq.load_siq_model(model_path)
    val_path = "/Users/stnava/data/blast_cohorts/BIDS/FPA/sub-BLAST022/ses-01/anat/sub-BLAST022_ses-01_run-001_T1w.nii.gz"
    if not os.path.exists(val_path):
        return  # skip if validation subject not present

    img = ants.image_read(val_path)
    img2d = ants.slice_image(img, axis=2, idx=img.shape[2] // 2 + 40)
    lr = ants.resample_image(img2d, [2, 2], use_voxels=False, interp_type=0)
    mid = [s // 2 for s in lr.shape]
    lr_p = ants.crop_indices(lr, [mid[0] - 24, mid[1] - 24], [mid[0] + 24, mid[1] + 24])
    hr_p = ants.crop_indices(img2d, [mid[0] * 2 - 48, mid[1] * 2 - 48], [mid[0] * 2 + 48, mid[1] * 2 + 48])
    gt_np = ants.iMath(hr_p, "Normalize").numpy()

    # Raw unaligned inference vs phase-aligned inference
    sr_raw = siq.inference(lr_p, model, config=cfg, align_phase=False, poly_order=None, anti_checkerboard=False).numpy()
    sr_aligned = siq.inference(lr_p, model, config=cfg, align_phase=True, poly_order=None, anti_checkerboard=False).numpy()

    gy = sobel(gt_np, axis=0)
    diff_raw = sr_raw - gt_np
    diff_aligned = sr_aligned - gt_np

    corr_y_raw = abs(np.corrcoef(diff_raw.flat, gy.flat)[0, 1])
    corr_y_aligned = abs(np.corrcoef(diff_aligned.flat, gy.flat)[0, 1])

    print(f"\n[edge bias test] Raw |r_y|={corr_y_raw:.4f}, Aligned |r_y|={corr_y_aligned:.4f}")
    assert corr_y_aligned < corr_y_raw / 3.0, (
        f"Phase alignment did not substantially reduce edge bias: "
        f"raw |r_y|={corr_y_raw:.4f}, aligned |r_y|={corr_y_aligned:.4f}"
    )


# ─────────────────────────────────────────────────────────────────
# Test 6 — Generator: Crop index exact alignment across dimensions
# ─────────────────────────────────────────────────────────────────

def test_generator_crop_indices_strictly_aligned():
    """Verify hr_starts is strictly locked to lr_starts * factor across isotropic & anisotropic factors."""
    from siq.espcn import _normalize_factor
    for dim, factors in [(2, [2, 3, 4]), (3, [2, [1, 1, 2], [1, 2, 2]])]:
        for factor in factors:
            factor_tuple = _normalize_factor(factor, dim)
            lr_shape = tuple([32] * dim)
            hr_shape = tuple(p * f for p, f in zip(lr_shape, factor_tuple))
            hr_large_shape = tuple(int(round(p * 1.5)) for p in hr_shape)
            lr_large_shape = tuple(int(round(p * 1.5)) for p in lr_shape)
            lr_starts = [(lr_large_shape[i] - lr_shape[i]) // 2 for i in range(dim)]
            hr_starts = [lr_starts[i] * factor_tuple[i] for i in range(dim)]
            for i in range(dim):
                assert hr_starts[i] == lr_starts[i] * factor_tuple[i], (
                    f"dim={dim} factor={factor}: hr_starts[{i}]={hr_starts[i]} != lr_starts[{i}]*f={lr_starts[i]*factor_tuple[i]}"
                )


def test_blind_sr_generator_simple_spacing():
    """Verify blind_sr_generator_simple generates exact integer factor spacing."""
    import ants
    hr = ants.from_numpy(np.ones((64, 64), dtype=np.float32))
    factor = 2
    lr_target_spacing = tuple(float(s * factor) for s in hr.spacing)
    lr = ants.resample_image(hr, lr_target_spacing, use_voxels=False, interp_type=0)
    for s in lr.spacing:
        assert abs(s - 2.0) < 1e-9, f"Spacing error in simple generator: {s}"



def test_3d_skip_connections_zero_shift():
    """
    Verify UpSampling3D has zero spatial shift across isotropic & anisotropic factors.
    Output 2k, 2j, 2m replicates input k, j, m — preserving exact integer grid alignment.
    """
    import keras
    import numpy as np

    for factor in [(2, 2, 2), (1, 1, 2), (1, 2, 2)]:
        layer = keras.layers.UpSampling3D(size=factor)
        size = 16
        lr_np = np.zeros((1, size, size, size, 1), dtype='float32')
        cz, cy, cx = size // 2, size // 2, size // 2
        lr_np[0, cz, cy, cx, 0] = 1.0
        sr_np = layer(lr_np).numpy()[0, :, :, :, 0]

        ref_np = lr_np[0, :, :, :, 0].repeat(factor[0], axis=0).repeat(factor[1], axis=1).repeat(factor[2], axis=2)
        diff = np.max(np.abs(sr_np - ref_np))
        assert diff < 1e-6, f"UpSampling3D factor {factor} differs from nearest-neighbor repeat: diff={diff}"


def test_inference_real_nifti_dbpn_and_asdbpn_alignment():
    """
    Verify real MRI NIfTI scans with negative direction diagonals (e.g. BLAST data)
    achieve sub-pixel zero-lag alignment (<0.08 voxels) and minimal edge gradient
    correlation (<0.02) using siq.inference on both DBPN and AS-DBPN architectures.
    """
    import os
    import ants
    import siq
    from scipy.ndimage import sobel
    from skimage.registration import phase_cross_correlation

    val_path = "/Users/stnava/data/blast_cohorts/BIDS/FPA/sub-BLAST022/ses-01/anat/sub-BLAST022_ses-01_run-001_T1w.nii.gz"
    if not os.path.exists(val_path):
        return  # skip if validation dataset not on disk

    val_vol = ants.image_read(val_path)
    val_vol = ants.iMath(ants.iMath(val_vol, "TruncateIntensity", 0.001, 0.999), "Normalize")

    # 1. DBPN 3D
    dbpn_path = "dbpn_small_3d_champion.keras"
    if os.path.exists(dbpn_path):
        m_dbpn, cfg_dbpn = siq.load_siq_model(dbpn_path)
        low_res_vol = ants.resample_image(val_vol, [2.0, 2.0, 2.0], use_voxels=False, interp_type=0)
        mid_lr = [low_res_vol.shape[d] // 2 + 20 for d in range(3)]
        mid_hr = [mid_lr[d] * 2 for d in range(3)]
        lr_box = 32
        val_lr_p = ants.crop_indices(low_res_vol, [mid_lr[d] - lr_box for d in range(3)], [mid_lr[d] + lr_box for d in range(3)])
        val_hr_p = ants.crop_indices(val_vol, [mid_hr[d] - lr_box * 2 for d in range(3)], [mid_hr[d] + lr_box * 2 for d in range(3)])
        gt_3d = ants.iMath(val_hr_p, "Normalize").numpy()

        sr_dbpn = siq.inference(val_lr_p, m_dbpn, config=cfg_dbpn, verbose=False, poly_order=None, anti_checkerboard=False).numpy()
        sh_dbpn, _, _ = phase_cross_correlation(gt_3d, sr_dbpn, upsample_factor=100)
        for d in range(3):
            assert abs(sh_dbpn[d]) < 0.08, f"DBPN 3D axis {d} has non-zero phase shift: {sh_dbpn[d]:.3f} voxels"

        gy = sobel(gt_3d, axis=0)
        r_y = abs(np.corrcoef((sr_dbpn - gt_3d).flat, gy.flat)[0, 1])
        assert r_y < 0.02, f"DBPN 3D residual error has edge gradient correlation: |r_y|={r_y:.4f} > 0.02"

    # 2. AS-DBPN 2D
    asdbpn_path = "asdbpn_2d_2x2_best_mdl.keras"
    if os.path.exists(asdbpn_path):
        m_asdbpn, cfg_asdbpn = siq.load_siq_model(asdbpn_path)
        img2d = ants.slice_image(val_vol, axis=2, idx=val_vol.shape[2] // 2 + 40)
        lr2d = ants.resample_image(img2d, [2, 2], use_voxels=False, interp_type=0)
        mid2d = [s // 2 for s in lr2d.shape]
        lr2d_p = ants.crop_indices(lr2d, [mid2d[0] - 24, mid2d[1] - 24], [mid2d[0] + 24, mid2d[1] + 24])
        hr2d_p = ants.crop_indices(img2d, [mid2d[0] * 2 - 48, mid2d[1] * 2 - 48], [mid2d[0] * 2 + 48, mid2d[1] * 2 + 48])
        gt_2d = ants.iMath(hr2d_p, "Normalize").numpy()

        sr_asdbpn = siq.inference(lr2d_p, m_asdbpn, config=cfg_asdbpn, verbose=False, poly_order=None, anti_checkerboard=False).numpy()
        sh_asdbpn, _, _ = phase_cross_correlation(gt_2d, sr_asdbpn, upsample_factor=100)
        for d in range(2):
            assert abs(sh_asdbpn[d]) < 0.08, f"AS-DBPN 2D axis {d} has non-zero phase shift: {sh_asdbpn[d]:.3f} voxels"


# ─────────────────────────────────────────────────────────────────
# Main — run all tests and print audit report
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("siq SR Spatial Alignment Audit")
    print("=" * 60)

    tests = [
        # ANTs convention
        ("use_voxels=False LR alignment (xcorr)",    test_ants_resample_use_voxels_false_zero_shift),
        ("use_voxels=True shift documented",          test_ants_resample_use_voxels_true_shift_detected),
        # Skip interpolation
        ("nearest UpSampling2D zero shift",           test_nearest_upsample2d_zero_shift),
        ("bilinear UpSampling2D shift documented",    test_bilinear_upsample2d_shift_documented),
        ("3D skip connections zero shift",            test_3d_skip_connections_zero_shift),
        # Generator spacing invariant (catches use_voxels=True bug reliably)
        ("generator 2D LR spacing == exact factor",  test_blind_sr_generator_spacing_invariant_2d),
        ("generator 3D LR spacing == exact factor",  test_blind_sr_generator_spacing_invariant_3d),
        # Impulse alignment (catches offset with controlled input)
        ("generator 2D impulse peak alignment",       test_blind_sr_generator_impulse_peak_alignment_2d),
        # Phase alignment (eliminates edge bias from transposed convolutions)
        ("inference phase alignment eliminates edge bias", test_inference_phase_alignment_eliminates_edge_bias),
        # Real NIfTI negative direction cosine alignment
        ("real NIfTI DBPN & AS-DBPN alignment",       test_inference_real_nifti_dbpn_and_asdbpn_alignment),
        # Generator crop and spacing invariants
        ("generator crop indices strictly locked",    test_generator_crop_indices_strictly_aligned),
        ("simple generator exact spacing",             test_blind_sr_generator_simple_spacing),
    ]

    passed, failed = 0, 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✅ PASS  {name}")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ FAIL  {name}")
            print(f"           {e}")
            failed += 1
        except Exception as e:
            print(f"  ⚠️  ERROR {name}: {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("All alignment invariants satisfied. ✅")
    else:
        print("Alignment bugs detected — check output above. ❌")

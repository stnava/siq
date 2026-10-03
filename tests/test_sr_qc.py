"""
tests/test_sr_qc.py — Test suite for Super-Resolution Spatial Alignment & Quality Control (QC).
"""

import os
import json
import pytest
import numpy as np
from scipy.ndimage import shift as nd_shift, gaussian_filter

import siq


def _create_synthetic_test_volume(shape=(48, 48, 48)):
    """Creates a synthetic brain-like volume with sharp edge boundaries."""
    vol = np.zeros(shape, dtype=np.float32)
    cz, cy, cx = [s // 2 for s in shape]
    # Concentric spheres with distinct contrast
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    r1 = np.sqrt((z - cz)**2 + (y - cy)**2 + (x - cx)**2)
    vol[r1 < shape[0] * 0.4] = 0.5
    vol[r1 < shape[0] * 0.25] = 0.8
    vol[r1 < shape[0] * 0.1] = 0.2
    # Add light smoothing to create realistic anatomical gradient transitions
    vol = gaussian_filter(vol, sigma=1.0)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-8)
    return vol


def test_compute_phase_shift_zero_lag():
    """Verify compute_phase_shift accurately measures zero translation on identical inputs."""
    gt = _create_synthetic_test_volume()
    shift = siq.compute_phase_shift(gt, gt)
    for s in shift:
        assert abs(s) < 1e-3, f"Identical volumes produced non-zero shift: {shift}"


def test_compute_phase_shift_synthetic_translation():
    """Verify compute_phase_shift measures applied translation with sub-pixel precision."""
    gt = _create_synthetic_test_volume()
    known_shift = (0.5, -0.4, 0.2)
    sr_shifted = nd_shift(gt, known_shift, order=3, mode="nearest")

    measured_shift = siq.compute_phase_shift(gt, sr_shifted)
    for i, (m, k) in enumerate(zip(measured_shift, known_shift)):
        # phase cross correlation: registration vector needed to align sr back to gt
        assert abs(m - (-k)) < 0.05, f"Dim {i}: measured {m:.3f} differs from expected {-k:.3f}"


def test_compute_edge_error_correlation():
    """Verify directional edge error correlation discriminates between aligned and misaligned outputs."""
    gt = _create_synthetic_test_volume()
    
    # 1. Aligned output with slight noise/blur
    sr_aligned = gaussian_filter(gt, sigma=0.5)
    r_aligned = siq.compute_edge_error_correlation(gt, sr_aligned)
    for r in r_aligned:
        assert r < 0.02, f"Aligned output produced high edge correlation: {r_aligned}"

    # 2. Sub-pixel misaligned output (+0.5 voxel shift creates halo error ridges)
    sr_misaligned = nd_shift(gt, (0.5, 0.5, 0.0), order=3, mode="nearest")
    r_misaligned = siq.compute_edge_error_correlation(gt, sr_misaligned)
    assert r_misaligned[0] > 0.15, f"Y axis misaligned output failed to trigger edge error correlation: {r_misaligned[0]}"
    assert r_misaligned[1] > 0.15, f"X axis misaligned output failed to trigger edge error correlation: {r_misaligned[1]}"


def test_compute_alignment_qc_pass_and_fail():
    """Verify compute_alignment_qc properly assigns PASS, WARN, and FAIL statuses."""
    gt = _create_synthetic_test_volume()
    bilinear = gaussian_filter(gt, sigma=1.2)

    # 1. Pass case: sharp aligned model output
    sr_good = gaussian_filter(gt, sigma=0.2)
    qc_pass = siq.compute_alignment_qc(gt, sr_good, bilinear=bilinear, verbose=False)
    assert qc_pass["status"] in ["PASS", "WARN"]
    assert qc_pass["qc_pass"] is True
    assert "SR SPATIAL ALIGNMENT & BIAS QC REPORT" in qc_pass["summary_table"]

    # 2. Fail case: severe sub-pixel phase shift
    sr_shifted = nd_shift(gt, (1.0, 1.0, 0.5), order=3, mode="nearest")
    qc_fail = siq.compute_alignment_qc(gt, sr_shifted, bilinear=bilinear, verbose=False)
    assert qc_fail["status"] == "FAIL"
    assert qc_fail["qc_pass"] is False
    assert qc_fail["max_phase_shift"] >= 0.15


def test_sr_alignment_qc_cli(tmp_path):
    """Verify sr_alignment_qc.py CLI generates terminal output, JSON, and HTML reports."""
    import ants
    gt = _create_synthetic_test_volume((32, 32, 32))
    sr = gaussian_filter(gt, sigma=0.3)
    
    gt_file = os.path.join(tmp_path, "gt.nii.gz")
    sr_file = os.path.join(tmp_path, "sr.nii.gz")
    json_file = os.path.join(tmp_path, "qc.json")
    html_file = os.path.join(tmp_path, "qc.html")

    ants.from_numpy(gt).to_file(gt_file)
    ants.from_numpy(sr).to_file(sr_file)

    cmd = (
        f"python scripts/sr_alignment_qc.py --gt {gt_file} --sr {sr_file} "
        f"--output-json {json_file} --output-html {html_file}"
    )
    ret = os.system(cmd)
    assert ret == 0, f"sr_alignment_qc.py CLI exited with code {ret}"

    assert os.path.exists(json_file), "JSON QC report was not generated"
    assert os.path.exists(html_file), "HTML QC report was not generated"

    with open(json_file) as f:
        data = json.load(f)
    assert "status" in data
    assert "max_phase_shift" in data
    assert "max_edge_correlation" in data


def test_shift_lk_recovers_injected_shift_sign_and_magnitude():
    """LK shift estimate recovers known sub-voxel shifts (registration convention)."""
    gt = _create_synthetic_test_volume((40, 40, 40))
    for k in [(0.1, 0.0, 0.0), (0.0, -0.2, 0.1), (0.3, 0.3, -0.3)]:
        pred = nd_shift(gt, k, order=3, mode="nearest")
        est = siq.compute_shift_lk(gt, pred)
        for e, kk in zip(est, k):
            assert abs(e - (-kk)) < 0.04, f"injected {k} -> LK {est}"


def test_shift_lk_is_gain_offset_robust():
    """A pure contrast/brightness change must not be reported as a shift."""
    gt = _create_synthetic_test_volume((40, 40, 40))
    pred = 1.3 * gt + 0.1
    est = siq.compute_shift_lk(gt, pred)
    assert max(abs(e) for e in est) < 0.02, est


def test_alignment_qc_self_calibrating_vs_bilinear():
    """Estimator bias common to the (aligned) bilinear baseline is cancelled."""
    gt = _create_synthetic_test_volume((48, 48, 48))
    bilinear = gaussian_filter(gt, sigma=1.0)
    sharp_aligned = gaussian_filter(gt, sigma=0.3)
    qc = siq.compute_alignment_qc(gt, sharp_aligned, bilinear=bilinear)
    assert qc["status"] in ("PASS", "WARN")
    assert qc["max_phase_shift"] < 0.08
    shifted = nd_shift(gt, (0.3, 0.3, 0.0), order=3, mode="nearest")
    qc2 = siq.compute_alignment_qc(gt, shifted, bilinear=bilinear)
    assert qc2["status"] == "FAIL" and qc2["max_phase_shift"] >= 0.15
    assert "shift_rel" in qc2 and "lk_shift" in qc2

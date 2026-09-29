import pytest
import numpy as np
import siq


def test_checkerboard_index_isotropic():
    # 3D isotropic alternating pattern
    shape = (16, 16, 16)
    vol = np.zeros(shape, dtype=np.float32)
    for x in range(16):
        for y in range(16):
            for z in range(16):
                vol[x, y, z] = 0.05 * ((-1) ** (x + y + z))

    gt = np.random.RandomState(42).randn(*shape).astype(np.float32)
    cbi = siq.compute_checkerboard_index(vol + gt, gt)
    assert cbi > 0.03


def test_checkerboard_index_anisotropic_1d():
    # 1D alternating pattern along Z in 3D volume
    shape = (16, 16, 16)
    vol = np.zeros(shape, dtype=np.float32)
    for z in range(16):
        vol[:, :, z] = 0.05 * ((-1) ** z)

    gt = np.random.RandomState(42).randn(*shape).astype(np.float32)

    # 3D filter is blind to 1D alternating signal along a single axis
    cbi_iso = siq.compute_checkerboard_index(vol + gt, gt, factor=None)
    assert cbi_iso < 1e-4

    # 1D factor-aware filter detects it directly
    cbi_1d = siq.compute_checkerboard_index(vol + gt, gt, factor=(1, 1, 2))
    assert cbi_1d > 0.03


def test_checkerboard_index_clean_gt():
    shape = (16, 16, 16)
    gt = np.random.RandomState(42).randn(*shape).astype(np.float32)
    # When prediction == gt, residual checkerboard error is 0.0
    cbi = siq.compute_checkerboard_index(gt, gt, factor=(1, 1, 2))
    assert cbi < 1e-5

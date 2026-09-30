import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
import siq


def test_tenengrad_and_acutance_ratio_2d():
    # Crisp checkerboard / synthetic grid
    y_true = np.zeros((64, 64), dtype=np.float32)
    y_true[16:48, 16:48] = 1.0

    # Blurred version
    y_pred = gaussian_filter(y_true, sigma=1.5)

    t_true = siq.compute_tenengrad(y_true)
    t_pred = siq.compute_tenengrad(y_pred)
    assert t_true > 0.0
    assert t_pred < t_true

    ratio = siq.compute_acutance_ratio(y_true, y_pred)
    assert 0.0 < ratio < 1.0
    assert np.isclose(siq.compute_acutance_ratio(y_true, y_true), 1.0)


def test_tenengrad_and_acutance_ratio_3d():
    y_true = np.zeros((32, 32, 32), dtype=np.float32)
    y_true[8:24, 8:24, 8:24] = 1.0
    y_pred = gaussian_filter(y_true, sigma=1.5)

    ratio = siq.compute_acutance_ratio(y_true, y_pred)
    assert 0.0 < ratio < 1.0
    assert np.isclose(siq.compute_acutance_ratio(y_true, y_true), 1.0)


def test_laplacian_energy_ratio():
    y_true = np.random.rand(64, 64).astype(np.float32)
    y_pred = gaussian_filter(y_true, sigma=1.0)

    ratio = siq.compute_laplacian_energy_ratio(y_true, y_pred)
    assert 0.0 < ratio < 1.0
    assert np.isclose(siq.compute_laplacian_energy_ratio(y_true, y_true), 1.0)


def test_spectral_energy_ratio_factor_aware():
    y_true = np.random.rand(48, 48, 48).astype(np.float32)
    y_pred = gaussian_filter(y_true, sigma=1.5)

    # 3D isotropic factor (2, 2, 2)
    ratio_iso = siq.compute_spectral_energy_ratio(y_true, y_pred, factor=(2, 2, 2))
    assert 0.0 < ratio_iso < 1.0
    assert np.isclose(siq.compute_spectral_energy_ratio(y_true, y_true, factor=(2, 2, 2)), 1.0)

    # Anisotropic factor (1, 1, 2)
    ratio_aniso = siq.compute_spectral_energy_ratio(y_true, y_pred, factor=(1, 1, 2))
    assert 0.0 < ratio_aniso < 1.0


def test_ms_ssim_2d_and_3d():
    # 2D
    y_true_2d = np.random.rand(64, 64).astype(np.float32)
    y_pred_2d = gaussian_filter(y_true_2d, sigma=0.5)

    ms_2d = siq.compute_ms_ssim(y_true_2d, y_pred_2d)
    assert 0.0 < ms_2d < 1.0
    assert np.isclose(siq.compute_ms_ssim(y_true_2d, y_true_2d), 1.0, atol=1e-4)

    # 3D
    y_true_3d = np.random.rand(32, 32, 32).astype(np.float32)
    y_pred_3d = gaussian_filter(y_true_3d, sigma=0.5)

    ms_3d = siq.compute_ms_ssim(y_true_3d, y_pred_3d)
    assert 0.0 < ms_3d < 1.0
    assert np.isclose(siq.compute_ms_ssim(y_true_3d, y_true_3d), 1.0, atol=1e-4)


def test_lpips_2d_and_3d():
    y_true_2d = np.random.rand(64, 64).astype(np.float32)
    y_pred_2d = gaussian_filter(y_true_2d, sigma=1.0)

    # Identical should be 0.0
    dist_self_2d = siq.compute_lpips(y_true_2d, y_true_2d)
    assert np.isclose(dist_self_2d, 0.0, atol=1e-6)

    # Blurred should have positive perceptual distance
    dist_pred_2d = siq.compute_lpips(y_true_2d, y_pred_2d)
    assert dist_pred_2d > 0.0

    # 3D Tri-Planar
    y_true_3d = np.random.rand(32, 32, 32).astype(np.float32)
    y_pred_3d = gaussian_filter(y_true_3d, sigma=1.0)

    dist_self_3d = siq.compute_lpips(y_true_3d, y_true_3d, num_slices=8)
    assert np.isclose(dist_self_3d, 0.0, atol=1e-6)

    dist_pred_3d = siq.compute_lpips(y_true_3d, y_pred_3d, num_slices=8)
    assert dist_pred_3d > 0.0


def test_compute_perceptual_metrics_comprehensive():
    y_true = np.random.rand(48, 48).astype(np.float32)
    y_pred = gaussian_filter(y_true, sigma=0.8)

    metrics = siq.compute_perceptual_metrics(y_true, y_pred, factor=(2, 2))
    expected_keys = [
        "psnr", "ssim", "ms_ssim", "acutance_ratio",
        "laplacian_ratio", "spectral_ratio", "lpips", "gmsd", "cbi"
    ]
    for k in expected_keys:
        assert k in metrics
        assert metrics[k] is not None
        assert isinstance(metrics[k], float)

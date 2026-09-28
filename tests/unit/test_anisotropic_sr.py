import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["KERAS_BACKEND"] = "torch"

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest
import numpy as np
import torch
if hasattr(torch, "set_default_device"):
    torch.set_default_device("cpu")

import siq.espcn as espcn


def test_normalize_factor():
    """Verify factor normalization for scalars, tuples, and error handling."""
    assert espcn._normalize_factor(2, 2) == (2, 2)
    assert espcn._normalize_factor((1, 2), 2) == (1, 2)
    assert espcn._normalize_factor([1, 2], 2) == (1, 2)
    assert espcn._normalize_factor(2, 3) == (2, 2, 2)
    assert espcn._normalize_factor((1, 1, 2), 3) == (1, 1, 2)
    assert espcn._normalize_factor([2, 2, 4], 3) == (2, 2, 4)

    with pytest.raises(ValueError):
        espcn._normalize_factor((1, 2), 3)
    with pytest.raises(ValueError):
        espcn._normalize_factor((1, 1, 2), 2)
    with pytest.raises(ValueError):
        espcn._normalize_factor("invalid", 2)


@pytest.mark.parametrize("factor", [(1, 2), (2, 1), (2, 4)])
def test_anisotropic_2d_models(factor):
    """Test that all 2D models accept anisotropic factors and produce correct output shapes."""
    in_h, in_w = 8, 8
    inputs = np.zeros((1, in_h, in_w, 1), dtype=np.float32)
    expected_shape = (1, in_h * factor[0], in_w * factor[1], 1)

    models = [
        espcn.create_espcn_2d_attention(factor=factor, n_filters=16, n_res_blocks=1),
        espcn.create_ldbpn_2d(factor=factor, n_filters=16, n_stages=1),
        espcn.create_wdsr_2d(factor=factor, n_filters=16, n_res_blocks=1),
        espcn.create_rcan_2d(factor=factor, n_filters=16, n_groups=1, n_blocks=1),
        espcn.create_carn_2d(factor=factor, n_filters=16, n_blocks=1),
        espcn.create_espcn_2d_resize_conv(factor=factor, n_filters=16, n_res_blocks=1),
        espcn.create_wdsr_2d_resize_conv(factor=factor, n_filters=16, n_res_blocks=1),
        espcn.create_srfbn_2d(factor=factor, n_filters=16, n_steps=2),
        espcn.create_san_2d(factor=factor, n_filters=16, n_groups=1, n_blocks=1),
        espcn.create_asdbpn_2d(factor=factor, n_filters=16, n_steps=2),
    ]

    for model in models:
        out = model(inputs)
        assert tuple(out.shape) == expected_shape, f"{model.name}: expected {expected_shape}, got {out.shape}"


@pytest.mark.parametrize("factor", [(1, 1, 2), (2, 2, 1), (2, 2, 4)])
def test_anisotropic_3d_models(factor):
    """Test that all 3D models accept anisotropic factors and produce correct output shapes."""
    in_d, in_h, in_w = 4, 4, 4
    inputs = np.zeros((1, in_d, in_h, in_w, 1), dtype=np.float32)
    expected_shape = (1, in_d * factor[0], in_h * factor[1], in_w * factor[2], 1)

    models = [
        espcn.create_espcn_3d(factor=factor, n_filters=16),
        espcn.create_espcn_3d_residual(factor=factor, n_filters=16, n_res_blocks=1),
        espcn.create_espcn_3d_attention(factor=factor, n_filters=16, n_res_blocks=1),
        espcn.create_ldbpn_3d(factor=factor, n_filters=16, n_stages=1),
        espcn.create_wdsr_3d(factor=factor, n_filters=16, n_res_blocks=1),
        espcn.create_rcan_3d(factor=factor, n_filters=16, n_groups=1, n_blocks=1),
        espcn.create_carn_3d(factor=factor, n_filters=16, n_blocks=1),
        espcn.create_srfbn_3d(factor=factor, n_filters=16, n_steps=2),
        espcn.create_san_3d(factor=factor, n_filters=16, n_groups=1, n_blocks=1),
        espcn.create_asdbpn_3d(factor=factor, n_filters=16, n_steps=2),
    ]

    for model in models:
        out = model(inputs)
        assert tuple(out.shape) == expected_shape, f"{model.name}: expected {expected_shape}, got {out.shape}"

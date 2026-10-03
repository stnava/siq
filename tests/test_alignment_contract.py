"""
tests/test_alignment_contract.py — LR/HR alignment contract regression tests.

Guards against the recurring bug class where a reflection (np.flip / np.rot90)
applied to an already-formed (LR, HR) pair shifts alignment by (f-1) voxels.
"""

import ast
import os
import numpy as np
import pytest
import ants
from scipy.ndimage import gaussian_filter

import siq
from siq.alignment import (augment_geometry_hr, decimation_offset,
                           audit_pair_alignment, reflection_shift)


def _hr(shape=(48, 48, 48), seed=0):
    rng = np.random.default_rng(seed)
    v = gaussian_filter(rng.standard_normal(shape), 1.5).astype("float32")
    return (v - v.min()) / (v.max() - v.min())


def _lr(hr, factor):
    return ants.resample_image(
        ants.from_numpy(np.ascontiguousarray(hr)), tuple(float(f) for f in factor),
        use_voxels=False, interp_type=0).numpy()


@pytest.mark.parametrize("factor", [(2, 2, 2), (1, 1, 2), (1, 2, 2)])
def test_aligned_pair_has_zero_offset(factor):
    hr = _hr()
    assert decimation_offset(_lr(hr, factor), hr, factor) == (0, 0, 0)


@pytest.mark.parametrize("factor", [(2, 2, 2), (1, 1, 2), (1, 2, 2)])
def test_reflection_after_pairing_is_detected(factor):
    """Negative control: flipping an existing pair moves offset to f-1 per axis."""
    hr = _hr()
    lr = _lr(hr, factor)
    for ax in range(3):
        lr_f, hr_f = np.flip(lr, axis=ax), np.flip(hr, axis=ax)
        off = decimation_offset(lr_f, hr_f, factor)
        assert off[ax] == factor[ax] - 1, (factor, ax, off)
        assert reflection_shift(factor, [ax])[ax] == factor[ax] - 1


@pytest.mark.parametrize("factor", [(2, 2, 2), (1, 1, 2), (1, 2, 2)])
def test_hr_first_augmentation_preserves_alignment(factor):
    rng = np.random.RandomState(1)
    for seed in range(12):
        hr = augment_geometry_hr(_hr(seed=seed), factor, rng=rng)
        lr = _lr(hr, factor)
        assert decimation_offset(lr, hr, factor) == (0, 0, 0)


def test_blind_sr_generator_pairs_are_aligned():
    """The real training generator must emit aligned pairs (red before the fix)."""
    np.random.seed(0)
    gen = siq.blind_sr_generator(
        batch_size=4, lr_patch_size=16, factor=2, use_cache=False,
        noise_std_range=(0.0, 0.0), blur_sigma_range=(0.0, 0.0), interp_types=(0,))
    assert audit_pair_alignment(gen, (2, 2, 2), n_batches=3, verbose=False)


def test_audit_aborts_on_misaligned_generator():
    def bad_gen():
        while True:
            hr = _hr((32, 32, 32))
            lr = _lr(hr, (2, 2, 2))
            yield (np.flip(lr, 0)[None, ..., None].copy(), np.flip(hr, 0)[None, ..., None].copy())
    with pytest.raises(RuntimeError):
        audit_pair_alignment(bad_gen(), (2, 2, 2), n_batches=1, verbose=False)


def test_no_post_pair_reflections_in_generator_modules():
    """Lint: reflection/rot90 calls are only allowed in siq/alignment.py,
    inference flip-compensation code, and display-only scripts."""
    root = os.path.dirname(os.path.dirname(__file__))
    forbidden = {"flip", "rot90", "fliplr", "flipud"}
    guarded = [os.path.join(root, "siq", "blind_sr.py")]
    for path in guarded:
        tree = ast.parse(open(path).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr in forbidden:
                pytest.fail(f"{os.path.basename(path)}:{node.lineno} uses {node.func.attr}; "
                            f"route geometric augmentation through siq.alignment")


def test_no_interpolating_global_skips_in_architectures():
    """Lint: every `global_skip` must be exact nearest-neighbor replication.
    A bilinear/trilinear skip (align_corners=False) shifts the output by a
    sub-voxel phase offset (the earlier 2D fix bae6ae2 never reached 3D)."""
    root = os.path.dirname(os.path.dirname(__file__))
    src = open(os.path.join(root, "siq", "espcn.py")).read().splitlines()
    for i, line in enumerate(src, 1):
        if "global_skip" in line and "=" in line and "name=" in line:
            low = line.lower()
            assert "trilinear" not in low and "bilinear" not in low, \
                f"espcn.py:{i} global_skip must use nearest-neighbor replication: {line.strip()}"


def test_refinement_and_kitchen_sink_run_preflight_audit():
    """Every training entry point that forms LR/HR pairs must run the audit."""
    root = os.path.dirname(os.path.dirname(__file__))
    for rel in ("siq/blind_sr.py", "siq/curriculum.py", "scripts/train_model_refinement.py"):
        assert "audit_pair_alignment" in open(os.path.join(root, rel)).read(), \
            f"{rel} does not run siq.audit_pair_alignment before training"

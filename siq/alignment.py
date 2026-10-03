"""
siq.alignment — single source of truth for the LR/HR alignment convention.

Convention
----------
LR voxel ``k`` is located at HR coordinate ``f * k`` (decimation-aligned,
``ants.resample_image(..., use_voxels=False)`` with exact factor spacing and a
shared origin).  This convention is **not reflection symmetric**: reflecting an
already-formed (LR, HR) pair moves the alignment to HR offset ``f - 1`` along
each reflected axis (verified empirically: best HR decimation offset becomes 1
instead of 0 for ``f = 2``).  Reflections, including those hidden inside
``np.rot90``, must therefore be applied to the HR volume **before** the LR
image is derived.  Pure axis permutations (transposes) are safe at any stage.
"""

import numpy as np


def reflection_shift(factor, flipped_axes):
    """Voxel shift ``(f_d - 1)`` introduced along each reflected axis ``d``.

    Returns a tuple with one entry per axis of ``factor`` (0 for axes that
    were not reflected or have ``f_d == 1``).
    """
    factor = tuple(int(f) for f in factor)
    flipped = set(int(a) for a in flipped_axes)
    return tuple(float(f - 1) if d in flipped else 0.0 for d, f in enumerate(factor))


def augment_geometry_hr(hr, factor, rng=None):
    """Random geometric augmentation applied to an HR array BEFORE degradation.

    Applies random per-axis flips, random 90-degree rotations and random axis
    permutations.  Rotations/permutations only mix axes that have identical
    upsampling factors AND identical extents so that LR/HR shape relations are
    preserved for anisotropic factors.

    Returns the augmented array (a contiguous copy).
    """
    rng = np.random if rng is None else rng
    factor = tuple(int(f) for f in factor)
    dim = hr.ndim
    out = hr

    # Reflections are safe here because the LR image is derived afterwards.
    for axis in range(dim):
        if rng.choice([True, False]):
            out = np.flip(out, axis=axis)

    def _mixable(a, b):
        return factor[a] == factor[b] and out.shape[a] == out.shape[b]

    pairs = [(a, b) for a in range(dim) for b in range(a + 1, dim) if _mixable(a, b)]
    if pairs:
        k = int(rng.randint(0, 4)) if hasattr(rng, "randint") else int(rng.integers(0, 4))
        if k > 0:
            idx = int(rng.randint(0, len(pairs))) if hasattr(rng, "randint") else int(rng.integers(0, len(pairs)))
            out = np.rot90(out, k=k, axes=pairs[idx])
        if rng.choice([True, False]):
            a, b = pairs[int(rng.randint(0, len(pairs))) if hasattr(rng, "randint") else int(rng.integers(0, len(pairs)))]
            perm = list(range(dim))
            perm[a], perm[b] = perm[b], perm[a]
            out = np.transpose(out, perm)
    return np.ascontiguousarray(out)


def decimation_offset(lr, hr, factor, max_offset=None, sigmas=(0.0, 0.5, 0.75, 1.0, 1.5, 2.0),
                      trim=3, return_scores=False):
    """Best integer HR decimation offset per axis for an (LR, HR) numpy pair.

    A correctly aligned pair returns all zeros.  A value of ``f_d - 1`` on an
    axis is the signature of a reflected-after-degradation pair.

    Robust to the degradations the training generator applies: the score is
    ``1 - Pearson r`` (invariant to the affine min-max rescaling of LR and HR),
    the HR candidate is compared at a grid of blur levels (the LR blur is
    unknown; the correct offset at the matched blur gives near-zero residual
    while a wrong offset cannot), and borders are trimmed to avoid edge effects.
    """
    import itertools
    from scipy.ndimage import gaussian_filter
    lr = np.asarray(lr, dtype=np.float64)
    hr = np.asarray(hr, dtype=np.float64)
    factor = tuple(int(f) for f in factor)
    ranges = [range(1) if f <= 1 else range(f if max_offset is None else min(f, max_offset + 1))
              for f in factor]
    offsets = list(itertools.product(*ranges))
    best_score = {off: np.inf for off in offsets}
    for sg in sigmas:
        hs = gaussian_filter(hr, sg, mode="nearest") if sg > 0 else hr
        for off in offsets:
            sl = tuple(slice(o, None, f) for o, f in zip(off, factor))
            h = hs[sl]
            n = tuple(min(a, b) for a, b in zip(lr.shape, h.shape))
            s = tuple(slice(trim, k - trim) if k > 2 * trim + 2 else slice(0, k) for k in n)
            a, b = lr[s].ravel(), h[s].ravel()
            if a.std() < 1e-8 or b.std() < 1e-8:
                continue
            score = 1.0 - float(np.corrcoef(a, b)[0, 1])
            if score < best_score[off]:
                best_score[off] = score
    best = min(offsets, key=lambda o: best_score[o])
    best = tuple(int(o) for o in best)
    return (best, best_score) if return_scores else best




def audit_pair_alignment(generator, factor, n_batches=2, verbose=True):
    """Pre-flight check: draw pairs from a real generator and verify alignment.

    ``generator`` must yield ``(x, y)`` batches with channel-last arrays.
    Raises ``RuntimeError`` if any pair has a non-zero integer decimation
    offset (the generator is injecting a reflection-style shift).  Intended to
    run once, at the start of every training script (~1-2 s).
    """
    factor = tuple(int(f) for f in factor)
    bad = []
    total = 0
    for _ in range(n_batches):
        batch = next(generator)
        x, y = batch[0], batch[1]
        if isinstance(y, (tuple, list)):
            y = y[0]
        for i in range(len(x)):
            lr = np.asarray(x[i, ..., 0])
            hr = np.asarray(y[i, ..., 0])
            off = decimation_offset(lr, hr, factor)
            total += 1
            if any(o != 0 for o in off):
                bad.append(off)
    if verbose:
        print(f"[siq.alignment] pair audit: {total - len(bad)}/{total} pairs aligned", flush=True)
    if bad:
        raise RuntimeError(
            f"LR/HR alignment audit FAILED: {len(bad)}/{total} training pairs have a non-zero "
            f"decimation offset (e.g. {bad[0]}). Reflections/rot90 were likely applied after "
            f"LR/HR pair formation. See siq/alignment.py."
        )
    return True

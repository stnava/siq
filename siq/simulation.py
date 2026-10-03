import math
import numpy as np
try:
    import ants
except ImportError:
    ants = None


def simulate_image(shaper=[32, 32, 32], n_levels=10, multiply=False):  # pragma: no cover
    """Generate an image of given shape and number of levels."""
    img = ants.from_numpy(np.random.normal(0, 1.0, size=shaper)) * 0
    for k in range(n_levels):
        temp = ants.from_numpy(np.random.normal(0, 1.0, size=shaper))
        temp = ants.smooth_image(temp, n_levels)
        temp = ants.threshold_image(temp, "Otsu", 1)
        if multiply:
            temp = temp * k
        img = img + temp
    return img


def _sample_param(p, default_val=None, is_int=False):  # pragma: no cover
    """
    Internal helper to sample a parameter from various formats:
    - Scalar: returned as-is.
    - Tuple/List: sampled uniformly between [0] and [1].
    - Callable: called to get a value.
    - Dict: uses 'type' to determine distribution (uniform, gaussian, poisson).
    """
    if p is None:
        p = default_val
    if isinstance(p, (int, float)):
        return p
    if isinstance(p, (list, tuple)):
        if is_int:
            return np.random.randint(p[0], p[1])
        return np.random.uniform(p[0], p[1])
    if callable(p):
        return p()
    if isinstance(p, dict):
        dist_type = p.get("type", "uniform")
        if dist_type == "uniform":
            low, high = p.get("low", 0), p.get("high", 1)
            return np.random.randint(low, high) if is_int else np.random.uniform(low, high)
        if dist_type in ["gaussian", "normal"]:
            val = np.random.normal(p.get("mean", 0), p.get("std", 1))
            return int(round(val)) if is_int else val
        if dist_type == "poisson":
            val = np.random.poisson(p.get("lam", 1))
            if "scale" in p:
                val = float(val) * float(p["scale"])
            return int(round(val)) if is_int else val
    return p


def simulate_image_multi_scale(
    large_shape=(48, 48, 48),
    scale_range=(0.7, 1.4),
    n_levels_range=(3, 9),
    sigma_range={'type': 'poisson', 'lam': 0.1},
    multiply=None,
    interp_types=(0, 2),
    min_sim_shape=24
):  # pragma: no cover
    """
    Generates a simulated tissue-like image at a random scale and orientation.
    Returns an ants.image of shape large_shape.
    """
    scale_factor = _sample_param(scale_range, (0.7, 1.4))
    sim_shape = [int(round(s * scale_factor)) for s in large_shape]
    sim_shape = [max(min_sim_shape, s) for s in sim_shape]

    n_levels = _sample_param(n_levels_range, (3, 9), is_int=True)
    if multiply is None:
        multiply = np.random.choice([True, False])

    img_np = np.zeros(sim_shape, dtype="float32")

    for k in range(n_levels):
        temp_np = np.random.normal(0, 1.0, size=sim_shape).astype("float32")
        temp = ants.from_numpy(temp_np)

        sigma = _sample_param(sigma_range, {'type': 'poisson', 'lam': 0.1})
        if sigma > 0:
            temp = ants.smooth_image(temp, sigma)

        temp = ants.threshold_image(temp, "Otsu", 1)
        temp_np = temp.numpy()

        if multiply:
            temp_np = temp_np * (k + 1)

        img_np += temp_np

    img = ants.from_numpy(img_np)
    interp = np.random.choice(interp_types)
    img_large = ants.resample_image(img, large_shape, use_voxels=True, interp_type=interp)

    return img_large


def add_rician_noise(array, noise_std):  # pragma: no cover
    """Applies Rician noise to a numpy array or ANTs image."""
    is_ants = hasattr(array, "numpy")
    arr = array.numpy() if is_ants else array
    n1 = np.random.normal(0, noise_std, arr.shape).astype(arr.dtype)
    n2 = np.random.normal(0, noise_std, arr.shape).astype(arr.dtype)
    noisy = np.sqrt((arr + n1)**2 + n2**2)
    noisy = np.clip(noisy, 0.0, 1.0)
    if is_ants:
        res = ants.image_clone(array)
        res[:] = noisy
        return res
    return noisy


def simulate_brain_procedural(shape, zoom_range=(0.7, 1.4), use_layer2=False):  # pragma: no cover
    """
    Procedurally generates a 2D or 3D patch resembling brain anatomy (CSF, GM, WM, Ventricles)
    with stochastic folding and coordinate zoom.
    """
    ndim = len(shape)
    coords = [np.linspace(-1, 1, s) for s in shape]
    grid = np.meshgrid(*coords, indexing="ij")

    if use_layer2:
        s_axis = np.random.uniform(zoom_range[0], zoom_range[1], size=ndim)
        grid = [g * s_axis[i] for i, g in enumerate(grid)]
        if ndim == 2:
            shear_val = np.random.uniform(-0.12, 0.12)
            grid = [grid[0] + shear_val * grid[1], grid[1]]
            warp_amp = np.random.uniform(0.04, 0.10)
            warp_freq = np.random.uniform(3.5, 6.5)
            grid = [grid[0] + warp_amp * np.sin(warp_freq * grid[1]),
                    grid[1] + warp_amp * np.cos(warp_freq * grid[0])]
        else:
            sh_xy = np.random.uniform(-0.12, 0.12)
            sh_xz = np.random.uniform(-0.12, 0.12)
            sh_yz = np.random.uniform(-0.12, 0.12)
            grid = [grid[0] + sh_xy * grid[1] + sh_xz * grid[2],
                    grid[1] + sh_yz * grid[2],
                    grid[2]]
            warp_amp = np.random.uniform(0.04, 0.10)
            warp_freq = np.random.uniform(3.5, 6.5)
            grid = [grid[0] + warp_amp * np.sin(warp_freq * grid[1]) * np.cos(warp_freq * grid[2]),
                    grid[1] + warp_amp * np.cos(warp_freq * grid[0]) * np.sin(warp_freq * grid[2]),
                    grid[2] + warp_amp * np.sin(warp_freq * grid[0]) * np.cos(warp_freq * grid[1])]
    else:
        s = np.random.uniform(zoom_range[0], zoom_range[1])
        grid = [g * s for g in grid]

    from scipy.ndimage import gaussian_filter
    img_np = np.zeros(shape, dtype="float32")

    # Base ellipse/sphere envelope
    r_sq = sum(g**2 for g in grid)
    mask_brain = r_sq < 0.85

    # Tissue intensities: CSF ~0.15, GM ~0.55, WM ~0.85
    noise_gm = gaussian_filter(np.random.normal(0, 1.0, size=shape).astype("float32"), sigma=1.5)
    noise_wm = gaussian_filter(np.random.normal(0, 1.0, size=shape).astype("float32"), sigma=2.0)

    img_np[mask_brain] = 0.55 + 0.12 * noise_gm[mask_brain]
    wm_mask = mask_brain & (noise_wm > 0.1) & (r_sq < 0.6)
    img_np[wm_mask] = 0.85 + 0.08 * noise_gm[wm_mask]

    # Ventricle core
    ventricle_mask = (r_sq < 0.12) & (noise_wm < -0.3)
    img_np[ventricle_mask] = 0.12 + 0.04 * noise_gm[ventricle_mask]

    # CSF sulci / boundary
    csf_mask = mask_brain & (noise_gm < -0.6)
    img_np[csf_mask] = 0.18 + 0.04 * noise_gm[csf_mask]

    img_np = np.clip(img_np, 0.0, 1.0)
    return ants.from_numpy(img_np)


def simulate_sinewave(shape, zoom_range=(0.7, 1.4), use_layer2=False):  # pragma: no cover
    """Procedurally generates N-dimensional multi-frequency sinusoidal wave coordinates."""
    ndim = len(shape)
    coords = [np.linspace(-1, 1, s) for s in shape]
    grid = np.meshgrid(*coords, indexing="ij")

    if use_layer2:
        s_axis = np.random.uniform(zoom_range[0], zoom_range[1], size=ndim)
        grid = [g * s_axis[i] for i, g in enumerate(grid)]
        if ndim == 2:
            shear_val = np.random.uniform(-0.12, 0.12)
            grid = [grid[0] + shear_val * grid[1], grid[1]]
            warp_amp = np.random.uniform(0.04, 0.10)
            warp_freq = np.random.uniform(3.5, 6.5)
            grid = [grid[0] + warp_amp * np.sin(warp_freq * grid[1]),
                    grid[1] + warp_amp * np.cos(warp_freq * grid[0])]
        else:
            sh_xy = np.random.uniform(-0.12, 0.12)
            sh_xz = np.random.uniform(-0.12, 0.12)
            sh_yz = np.random.uniform(-0.12, 0.12)
            grid = [grid[0] + sh_xy * grid[1] + sh_xz * grid[2],
                    grid[1] + sh_yz * grid[2],
                    grid[2]]
            warp_amp = np.random.uniform(0.04, 0.10)
            warp_freq = np.random.uniform(3.5, 6.5)
            grid = [grid[0] + warp_amp * np.sin(warp_freq * grid[1]) * np.cos(warp_freq * grid[2]),
                    grid[1] + warp_amp * np.cos(warp_freq * grid[0]) * np.sin(warp_freq * grid[2]),
                    grid[2] + warp_amp * np.sin(warp_freq * grid[0]) * np.cos(warp_freq * grid[1])]
    else:
        s = np.random.uniform(zoom_range[0], zoom_range[1])
        grid = [g * s for g in grid]

    img_np = np.zeros(shape, dtype="float32")
    num_waves = np.random.randint(2, 5)
    for _ in range(num_waves):
        freqs = [np.random.uniform(2.0, 8.0) for _ in range(ndim)]
        phase = np.random.uniform(0, 2 * np.pi)
        amp = np.random.uniform(0.2, 0.5)

        wave_term = sum(f * g for f, g in zip(freqs, grid)) + phase
        img_np += amp * np.sin(wave_term)

    img_min, img_max = img_np.min(), img_np.max()
    if img_max > img_min:
        img_np = (img_np - img_min) / (img_max - img_min)

    return ants.from_numpy(img_np)


def simulate_layered(shape, zoom_range=(0.7, 1.4), use_layer2=False):  # pragma: no cover
    """Procedurally generates N-dimensional rotated planar strip layers."""
    ndim = len(shape)
    coords = [np.linspace(-1, 1, s) for s in shape]
    grid = np.meshgrid(*coords, indexing="ij")

    if use_layer2:
        s_axis = np.random.uniform(zoom_range[0], zoom_range[1], size=ndim)
        grid = [g * s_axis[i] for i, g in enumerate(grid)]
        if ndim == 2:
            shear_val = np.random.uniform(-0.12, 0.12)
            grid = [grid[0] + shear_val * grid[1], grid[1]]
            warp_amp = np.random.uniform(0.04, 0.10)
            warp_freq = np.random.uniform(3.5, 6.5)
            grid = [grid[0] + warp_amp * np.sin(warp_freq * grid[1]),
                    grid[1] + warp_amp * np.cos(warp_freq * grid[0])]
        else:
            sh_xy = np.random.uniform(-0.12, 0.12)
            sh_xz = np.random.uniform(-0.12, 0.12)
            sh_yz = np.random.uniform(-0.12, 0.12)
            grid = [grid[0] + sh_xy * grid[1] + sh_xz * grid[2],
                    grid[1] + sh_yz * grid[2],
                    grid[2]]
            warp_amp = np.random.uniform(0.04, 0.10)
            warp_freq = np.random.uniform(3.5, 6.5)
            grid = [grid[0] + warp_amp * np.sin(warp_freq * grid[1]) * np.cos(warp_freq * grid[2]),
                    grid[1] + warp_amp * np.cos(warp_freq * grid[0]) * np.sin(warp_freq * grid[2]),
                    grid[2] + warp_amp * np.sin(warp_freq * grid[0]) * np.cos(warp_freq * grid[1])]
    else:
        s = np.random.uniform(zoom_range[0], zoom_range[1])
        grid = [g * s for g in grid]

    normal = np.random.normal(size=ndim)
    normal /= np.linalg.norm(normal)

    projection = sum(normal[i] * grid[i] for i in range(ndim))

    proj_min, proj_max = projection.min(), projection.max()
    num_layers = np.random.randint(4, 9)
    thresholds = np.sort(np.random.uniform(proj_min, proj_max, num_layers - 1))

    img_np = np.zeros(shape, dtype="float32")
    last_t = proj_min
    for i in range(num_layers):
        if i < num_layers - 1:
            t = thresholds[i]
            mask = (projection >= last_t) & (projection < t)
        else:
            mask = (projection >= last_t)

        intensity = np.random.uniform(0.1, 0.95)
        img_np[mask] = intensity
        last_t = t

    texture = np.random.normal(0, 0.015, size=shape).astype("float32")
    img_np += texture
    img_np = np.clip(img_np, 0.0, 1.0)

    return ants.from_numpy(img_np)


def simulate_vessel_tubes(shape, zoom_range=(0.7, 1.4), use_layer2=False):  # pragma: no cover
    """
    Procedurally generates tubular/vessel-like tree structures using Bezier curves
    and Euclidean distance fields.
    """
    ndim = len(shape)
    coords = [np.linspace(-1, 1, s) for s in shape]
    grid = np.meshgrid(*coords, indexing="ij")

    if use_layer2:
        s_axis = np.random.uniform(zoom_range[0], zoom_range[1], size=ndim)
        grid = [g * s_axis[i] for i, g in enumerate(grid)]
        if ndim == 2:
            warp_amp = np.random.uniform(0.04, 0.08)
            warp_freq = np.random.uniform(3.0, 5.0)
            grid = [grid[0] + warp_amp * np.sin(warp_freq * grid[1]),
                    grid[1] + warp_amp * np.cos(warp_freq * grid[0])]
        else:
            warp_amp = np.random.uniform(0.04, 0.08)
            warp_freq = np.random.uniform(3.0, 5.0)
            grid = [grid[0] + warp_amp * np.sin(warp_freq * grid[1]) * np.cos(warp_freq * grid[2]),
                    grid[1] + warp_amp * np.cos(warp_freq * grid[0]) * np.sin(warp_freq * grid[2]),
                    grid[2] + warp_amp * np.sin(warp_freq * grid[0]) * np.cos(warp_freq * grid[1])]
    else:
        s = np.random.uniform(zoom_range[0], zoom_range[1])
        grid = [g * s for g in grid]

    flat_grid = np.stack([g.ravel() for g in grid], axis=-1)

    num_pts = np.random.randint(4, 7)
    ctrl_pts = np.random.uniform(-0.8, 0.8, size=(num_pts, ndim))

    M = 200
    t = np.linspace(0, 1, M, dtype="float32")
    curve_pts = np.zeros((M, ndim), dtype="float32")
    for i in range(num_pts):
        coeff = float(math.comb(num_pts - 1, i)) * ((1 - t) ** (num_pts - 1 - i)) * (t ** i)
        for d in range(ndim):
            curve_pts[:, d] += coeff * ctrl_pts[i, d]

    base_radius = np.random.uniform(0.08, 0.16)
    from scipy.spatial import cKDTree
    tree = cKDTree(curve_pts)
    dists, _ = tree.query(flat_grid, distance_upper_bound=base_radius, workers=-1)
    mask = (dists < base_radius).reshape(shape)
    intensity = np.zeros(shape, dtype="float32")
    val = np.random.uniform(0.6, 0.9)
    intensity[mask] = val

    texture = np.random.normal(0, 0.015, size=shape).astype("float32")
    intensity[mask] += texture[mask]

    intensity = np.clip(intensity, 0.0, 1.0)
    return ants.from_numpy(intensity)


def simulate_cellular_voronoi(shape, zoom_range=(0.7, 1.4), use_layer2=False):  # pragma: no cover
    """Procedurally generates honeycomb or cellular Voronoi tessellation meshes."""
    ndim = len(shape)
    coords = [np.linspace(-1, 1, s) for s in shape]
    grid = np.meshgrid(*coords, indexing="ij")

    if use_layer2:
        s_axis = np.random.uniform(zoom_range[0], zoom_range[1], size=ndim)
        grid = [g * s_axis[i] for i, g in enumerate(grid)]
        if ndim == 2:
            warp_amp = np.random.uniform(0.03, 0.06)
            warp_freq = np.random.uniform(4.0, 6.0)
            grid = [grid[0] + warp_amp * np.sin(warp_freq * grid[1]),
                    grid[1] + warp_amp * np.cos(warp_freq * grid[0])]
        else:
            warp_amp = np.random.uniform(0.03, 0.06)
            warp_freq = np.random.uniform(4.0, 6.0)
            grid = [grid[0] + warp_amp * np.sin(warp_freq * grid[1]) * np.cos(warp_freq * grid[2]),
                    grid[1] + warp_amp * np.cos(warp_freq * grid[0]) * np.sin(warp_freq * grid[2]),
                    grid[2] + warp_amp * np.sin(warp_freq * grid[0]) * np.cos(warp_freq * grid[1])]
    else:
        s = np.random.uniform(zoom_range[0], zoom_range[1])
        grid = [g * s for g in grid]

    flat_grid = np.stack([g.ravel() for g in grid], axis=-1)

    num_seeds = np.random.randint(15, 30)
    seeds = np.random.uniform(-1.0, 1.0, size=(num_seeds, ndim))

    grid_sq = np.sum(flat_grid**2, axis=-1, keepdims=True)
    seeds_sq = np.sum(seeds**2, axis=-1, keepdims=True).T
    cross = 2 * (flat_grid @ seeds.T)
    dists_sq = np.maximum(grid_sq - cross + seeds_sq, 0.0)
    dists = np.sqrt(dists_sq)

    sorted_dists = np.partition(dists, 1, axis=-1)
    d1 = sorted_dists[:, 0]
    d2 = sorted_dists[:, 1]

    diff = (d2 - d1).reshape(shape)
    thickness = np.random.uniform(0.02, 0.06)

    intensity = np.zeros(shape, dtype="float32")
    mask = diff < thickness

    val = np.random.uniform(0.7, 0.95)
    intensity[mask] = val

    nearest_idx = np.argmin(dists, axis=-1).reshape(shape)
    cell_greys = np.random.uniform(0.1, 0.4, size=num_seeds)
    bg_intensity = cell_greys[nearest_idx]

    intensity[~mask] = bg_intensity[~mask]

    texture = np.random.normal(0, 0.015, size=shape).astype("float32")
    intensity += texture
    intensity = np.clip(intensity, 0.0, 1.0)

    return ants.from_numpy(intensity)


def simulate_geometric_phantoms(shape, zoom_range=(0.7, 1.4), use_layer2=False):  # pragma: no cover
    """Procedurally compiles geometric phantom circles, ellipsoids, and boxes."""
    ndim = len(shape)
    coords = [np.linspace(-1, 1, s) for s in shape]
    grid = np.meshgrid(*coords, indexing="ij")

    if use_layer2:
        s_axis = np.random.uniform(zoom_range[0], zoom_range[1], size=ndim)
        grid = [g * s_axis[i] for i, g in enumerate(grid)]
        if ndim == 2:
            warp_amp = np.random.uniform(0.04, 0.08)
            warp_freq = np.random.uniform(3.0, 5.0)
            grid = [grid[0] + warp_amp * np.sin(warp_freq * grid[1]),
                    grid[1] + warp_amp * np.cos(warp_freq * grid[0])]
        else:
            warp_amp = np.random.uniform(0.04, 0.08)
            warp_freq = np.random.uniform(3.0, 5.0)
            grid = [grid[0] + warp_amp * np.sin(warp_freq * grid[1]) * np.cos(warp_freq * grid[2]),
                    grid[1] + warp_amp * np.cos(warp_freq * grid[0]) * np.sin(warp_freq * grid[2]),
                    grid[2] + warp_amp * np.sin(warp_freq * grid[0]) * np.cos(warp_freq * grid[1])]
    else:
        s = np.random.uniform(zoom_range[0], zoom_range[1])
        grid = [g * s for g in grid]

    intensity = np.zeros(shape, dtype="float32")

    if ndim == 2:
        X, Y = grid[0], grid[1]
        mask_bg = (X**2 / 0.8**2 + Y**2 / 0.6**2) < 1.0
        intensity[mask_bg] = np.random.uniform(0.15, 0.3)
    else:
        X, Y, Z = grid[0], grid[1], grid[2]
        mask_bg = (X**2 / 0.8**2 + Y**2 / 0.6**2 + Z**2 / 0.6**2) < 1.0
        intensity[mask_bg] = np.random.uniform(0.15, 0.3)

    num_shapes = np.random.randint(4, 8)
    for _ in range(num_shapes):
        cx = np.random.uniform(-0.5, 0.5)
        cy = np.random.uniform(-0.5, 0.5)
        val = np.random.uniform(0.4, 0.9)

        rx = np.random.uniform(0.1, 0.25)
        ry = np.random.uniform(0.1, 0.25)

        shape_type = np.random.choice(["ellipse", "box"])
        if ndim == 2:
            if shape_type == "ellipse":
                mask = ((X - cx)**2 / rx**2 + (Y - cy)**2 / ry**2) < 1.0
            else:
                mask = (np.abs(X - cx) < rx) & (np.abs(Y - cy) < ry)
        else:
            cz = np.random.uniform(-0.5, 0.5)
            rz = np.random.uniform(0.1, 0.25)
            if shape_type == "ellipse":
                mask = ((X - cx)**2 / rx**2 + (Y - cy)**2 / ry**2 + (Z - cz)**2 / rz**2) < 1.0
            else:
                mask = (np.abs(X - cx) < rx) & (np.abs(Y - cy) < ry) & (np.abs(Z - cz) < rz)

        intensity[mask] = val

    texture = np.random.normal(0, 0.015, size=shape).astype("float32")
    intensity += texture
    intensity = np.clip(intensity, 0.0, 1.0)

    return ants.from_numpy(intensity)


def simulate_grid_patterns(shape, zoom_range=(0.7, 1.4), use_layer2=False):  # pragma: no cover
    """Procedurally generates orthotropic grid lines or checkerboard patterns."""
    ndim = len(shape)
    coords = [np.linspace(-1, 1, s) for s in shape]
    grid = np.meshgrid(*coords, indexing="ij")

    if use_layer2:
        s_axis = np.random.uniform(zoom_range[0], zoom_range[1], size=ndim)
        grid = [g * s_axis[i] for i, g in enumerate(grid)]
        if ndim == 2:
            warp_amp = np.random.uniform(0.04, 0.08)
            warp_freq = np.random.uniform(3.0, 5.0)
            grid = [grid[0] + warp_amp * np.sin(warp_freq * grid[1]),
                    grid[1] + warp_amp * np.cos(warp_freq * grid[0])]
        else:
            warp_amp = np.random.uniform(0.04, 0.08)
            warp_freq = np.random.uniform(3.0, 5.0)
            grid = [grid[0] + warp_amp * np.sin(warp_freq * grid[1]) * np.cos(warp_freq * grid[2]),
                    grid[1] + warp_amp * np.cos(warp_freq * grid[0]) * np.sin(warp_freq * grid[2]),
                    grid[2] + warp_amp * np.sin(warp_freq * grid[0]) * np.cos(warp_freq * grid[1])]
    else:
        s = np.random.uniform(zoom_range[0], zoom_range[1])
        grid = [g * s for g in grid]

    pattern_type = np.random.choice(["checker", "lines"])
    freq = np.random.uniform(6.0, 12.0)
    img_np = np.zeros(shape, dtype="float32")

    if pattern_type == "checker":
        term = 1.0
        for g in grid:
            term *= np.sign(np.sin(freq * np.pi * g))
        img_np = 0.5 + 0.3 * term
    else:
        width = np.random.uniform(0.03, 0.08)
        term = np.zeros(shape, dtype="float32")
        for g in grid:
            line_mask = np.abs(np.sin(freq * np.pi * g)) < width
            term[line_mask] = 1.0
        img_np = 0.2 + 0.6 * term

    texture = np.random.normal(0, 0.015, size=shape).astype("float32")
    img_np += texture
    img_np = np.clip(img_np, 0.0, 1.0)

    return ants.from_numpy(img_np)


def simulate_fractal_noise(shape, zoom_range=(0.7, 1.4), use_layer2=False):  # pragma: no cover
    """Procedurally compiles fractional Brownian motion (fBm) fractal noise."""
    ndim = len(shape)
    coords = [np.linspace(-1, 1, s) for s in shape]
    grid = np.meshgrid(*coords, indexing="ij")

    if use_layer2:
        s_axis = np.random.uniform(zoom_range[0], zoom_range[1], size=ndim)
        grid = [g * s_axis[i] for i, g in enumerate(grid)]
        if ndim == 2:
            warp_amp = np.random.uniform(0.04, 0.08)
            warp_freq = np.random.uniform(3.0, 5.0)
            grid = [grid[0] + warp_amp * np.sin(warp_freq * grid[1]),
                    grid[1] + warp_amp * np.cos(warp_freq * grid[0])]
        else:
            warp_amp = np.random.uniform(0.04, 0.08)
            warp_freq = np.random.uniform(3.0, 5.0)
            grid = [grid[0] + warp_amp * np.sin(warp_freq * grid[1]) * np.cos(warp_freq * grid[2]),
                    grid[1] + warp_amp * np.cos(warp_freq * grid[0]) * np.sin(warp_freq * grid[2]),
                    grid[2] + warp_amp * np.sin(warp_freq * grid[0]) * np.cos(warp_freq * grid[1])]
    else:
        s = np.random.uniform(zoom_range[0], zoom_range[1])
        grid = [g * s for g in grid]

    img_np = np.zeros(shape, dtype="float32")
    num_octaves = 4
    amplitude = 0.5
    frequency = np.random.uniform(2.0, 4.0)

    for _ in range(num_octaves):
        phase = np.random.uniform(0, 2 * np.pi)
        direction = np.random.normal(size=ndim)
        direction /= np.linalg.norm(direction)

        proj = sum(direction[i] * grid[i] for i in range(ndim))
        img_np += amplitude * np.sin(frequency * np.pi * proj + phase)

        amplitude *= 0.5
        frequency *= 2.0

    img_min, img_max = img_np.min(), img_np.max()
    if img_max > img_min:
        img_np = 0.1 + 0.8 * (img_np - img_min) / (img_max - img_min)

    texture = np.random.normal(0, 0.015, size=shape).astype("float32")
    img_np += texture
    img_np = np.clip(img_np, 0.0, 1.0)

    return ants.from_numpy(img_np)

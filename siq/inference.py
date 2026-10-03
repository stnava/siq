import time
import warnings
import numpy as np
from scipy.ndimage import shift as nd_shift, gaussian_filter
import keras
from keras import ops, Model

try:
    import ants
except ImportError:
    ants = None

try:
    import antspynet
except ImportError:
    antspynet = None

try:
    import antspyt1w
except ImportError:
    antspyt1w = None


def estimate_anti_checkerboard_sigma(vol, max_sigma=0.40):
    """
    Empirically calculates the optimal sub-voxel smoothing sigma to attenuate
    transposed-convolution checkerboard artifacts based on the measured Nyquist
    frequency spectral excess.
    """
    if hasattr(vol, 'numpy'):
        vol = vol.numpy()
    vol = np.asarray(vol, dtype=np.float32)
    sigmas = []
    ndim = vol.ndim
    for ax in range(ndim):
        fft_1d = np.fft.rfft(vol, axis=ax)
        axes_to_mean = tuple(i for i in range(ndim) if i != ax)
        mean_amp = np.mean(np.abs(fft_1d), axis=axes_to_mean)
        if len(mean_amp) < 4:
            continue
        a_nyquist = mean_amp[-1]
        idx_start = max(1, int(len(mean_amp) * 0.70))
        idx_end = max(idx_start + 1, int(len(mean_amp) * 0.95))
        a_baseline = np.mean(mean_amp[idx_start:idx_end]) + 1e-12
        excess_ratio = a_nyquist / a_baseline
        if excess_ratio > 1.2:
            sigma_ax = np.sqrt(np.log(excess_ratio)) / np.pi
            sigmas.append(float(sigma_ax))
        else:
            sigmas.append(0.0)

    optimal_sigma = float(np.mean(sigmas)) if sigmas else 0.0
    return min(max_sigma, optimal_sigma)


def gaussian_weight_map_numpy(shape, sigma=0.4):
    coords = [np.linspace(-1, 1, s) for s in shape]
    grids = np.meshgrid(*coords, indexing='ij')
    dist_sq = sum(g**2 for g in grids)
    weight = np.exp(-dist_sq / (2 * sigma**2))
    return weight


def _model_has_transposed_conv(model):
    """Returns True if the model (or any sub-layer) contains transposed convolution layers."""
    for l in getattr(model, "layers", []):
        cn = l.__class__.__name__
        if "Conv" in cn and "Transpose" in cn:
            return True
        if hasattr(l, "layers") and _model_has_transposed_conv(l):
            return True
    return False


def overlapping_patch_inference(
    image,
    model,
    target_range=(-127.5, 127.5),
    patch_size=(64, 64, 64),
    overlap=16,
    batch_size=1,
    verbose=False,
    volume_normalized=False,
    align_phase=True,
):
    """Overlapping patch inference with Gaussian-weighted blending."""
    if target_range[0] > target_range[1]:
        target_range = target_range[::-1]

    shape_length = len(model.inputs[0].shape)
    if shape_length == 5 and image.dimension != 3:
        raise ValueError("Expecting 3D input for this model.")
    elif shape_length == 4 and image.dimension != 2:
        raise ValueError("Expecting 2D input for this model.")

    if len(patch_size) != image.dimension:
        patch_size = tuple([patch_size[0]] * image.dimension)

    stride = tuple([p - overlap for p in patch_size])

    dim = image.dimension
    diag = [float(image.direction[i, i]) for i in range(dim)]
    flip_axes = tuple(i for i, d in enumerate(diag) if d < 0)

    image_array = image.numpy()
    if flip_axes:
        image_array = np.flip(image_array, axis=flip_axes).copy()
    if image.components == 1:
        image_array = np.expand_dims(image_array, axis=-1)

    if verbose:
        print(f"Image array shape: {image_array.shape}")

    D, H, W, C = image_array.shape

    pad_d = (stride[0] - (D - patch_size[0]) % stride[0]) % stride[0]
    pad_h = (stride[1] - (H - patch_size[1]) % stride[1]) % stride[1]
    pad_w = (stride[2] - (W - patch_size[2]) % stride[2]) % stride[2]

    overlap_d = patch_size[0] - stride[0]
    overlap_h = patch_size[1] - stride[1]
    overlap_w = patch_size[2] - stride[2]

    total_pad = (
        (overlap_d // 2, overlap_d // 2 + pad_d),
        (overlap_h // 2, overlap_h // 2 + pad_h),
        (overlap_w // 2, overlap_w // 2 + pad_w),
        (0, 0)
    )

    padded = np.pad(image_array, total_pad, mode='edge')

    patches = []
    coords = []

    nD = (padded.shape[0] - patch_size[0]) // stride[0] + 1
    nH = (padded.shape[1] - patch_size[1]) // stride[1] + 1
    nW = (padded.shape[2] - patch_size[2]) // stride[2] + 1

    for d in range(nD):
        for h in range(nH):
            for w in range(nW):
                z = d * stride[0]
                y = h * stride[1]
                x = w * stride[2]
                patch = padded[z:z + patch_size[0], y:y + patch_size[1], x:x + patch_size[2], :]
                patches.append(patch)
                coords.append((z, y, x))

    image_patches = np.stack(patches)
    padded_shape = padded.shape

    if volume_normalized:
        pass
    else:
        img_min = image_patches.min()
        img_max = image_patches.max()
        if img_max > img_min:
            image_patches = (image_patches - img_min) / (img_max - img_min) * (target_range[1] - target_range[0]) + target_range[0]
        else:
            image_patches = image_patches - img_min + target_range[0]

    if verbose:
        print(f"Prediction (patch-wise overlapping): {len(image_patches)} patches")
        start_time = time.time()

    try:
        from tqdm import tqdm
        use_tqdm = verbose
    except ImportError:
        use_tqdm = False

    predictions = []
    num_batches = int(np.ceil(len(image_patches) / batch_size))

    if use_tqdm:
        batch_iter = tqdm(range(num_batches), desc="Inferring patches")
    else:
        batch_iter = range(num_batches)

    for b in batch_iter:
        batch_data = image_patches[b * batch_size : (b + 1) * batch_size]
        pred = model.predict(batch_data, verbose=0)
        predictions.append(pred)

    prediction = np.concatenate(predictions, axis=0)

    if verbose:
        elapsed_time = time.time() - start_time
        if not use_tqdm:
            print(f"  (elapsed time: {elapsed_time:.3f}s)")
        print("Reconstruct intensities and blend")

    if volume_normalized:
        prediction = prediction.clip(0.0, 1.0)
    else:
        intensity_range = image.range()
        pred_min = prediction.min()
        pred_max = prediction.max()
        if pred_max > pred_min:
            prediction = (prediction - pred_min) / (pred_max - pred_min) * (intensity_range[1] - intensity_range[0]) + intensity_range[0]
        else:
            prediction = prediction - pred_min + intensity_range[0]

    expansion_factor = np.asarray(prediction.shape[1:-1]) / np.asarray(image_patches.shape[1:-1])
    expansion_factor = tuple(expansion_factor.astype(int))

    out_patch_size = tuple([p * e for p, e in zip(patch_size, expansion_factor)])

    weight = gaussian_weight_map_numpy(out_patch_size)
    weight = np.expand_dims(weight, axis=-1)

    out_shape = (
        padded_shape[0] * expansion_factor[0],
        padded_shape[1] * expansion_factor[1],
        padded_shape[2] * expansion_factor[2],
        prediction.shape[-1]
    )
    canvas = np.zeros(out_shape, dtype=np.float32)
    counts = np.zeros(out_shape, dtype=np.float32)

    for i, (z, y, x) in enumerate(coords):
        oz = z * expansion_factor[0]
        oy = y * expansion_factor[1]
        ox = x * expansion_factor[2]

        canvas[oz:oz + out_patch_size[0], oy:oy + out_patch_size[1], ox:ox + out_patch_size[2], :] += prediction[i] * weight
        counts[oz:oz + out_patch_size[0], oy:oy + out_patch_size[1], ox:ox + out_patch_size[2], :] += weight

    canvas /= (counts + 1e-8)

    crop_D = image_array.shape[0] * expansion_factor[0]
    crop_H = image_array.shape[1] * expansion_factor[1]
    crop_W = image_array.shape[2] * expansion_factor[2]

    z_start = total_pad[0][0] * expansion_factor[0]
    y_start = total_pad[1][0] * expansion_factor[1]
    x_start = total_pad[2][0] * expansion_factor[2]

    final_vol = canvas[z_start:z_start + crop_D, y_start:y_start + crop_H, x_start:x_start + crop_W, :]

    if flip_axes:
        final_vol = np.flip(final_vol, axis=flip_axes).copy()

    has_tc = _model_has_transposed_conv(model)
    shift_vec = []
    for d, f in enumerate(expansion_factor):
        if f <= 1:
            shift_vec.append(0.0)
        elif has_tc:
            shift_vec.append(-float(f - 1) / 2.0 if align_phase else 0.0)
        else:
            if d in flip_axes:
                shift_vec.append(-float(f - 1))
            else:
                shift_vec.append(0.0)

    if any(abs(s) > 1e-4 for s in shift_vec):
        s_tuple = tuple(shift_vec)
        if final_vol.ndim == 4:
            for c in range(final_vol.shape[-1]):
                final_vol[..., c] = nd_shift(final_vol[..., c], s_tuple, order=3, mode='nearest')
        else:
            final_vol = nd_shift(final_vol, s_tuple, order=3, mode='nearest')

    if image.components == 1:
        final_vol = np.squeeze(final_vol, axis=-1)
        final_image = ants.from_numpy(final_vol)
    else:
        channels = []
        for i in range(image.components):
            channels.append(ants.from_numpy(final_vol[..., i]))
        final_image = ants.merge_channels(channels)

    new_spacing = tuple(np.asarray(image.spacing) / np.asarray(expansion_factor))
    ants.set_spacing(final_image, new_spacing)
    ants.set_direction(final_image, image.direction)
    ants.set_origin(final_image, image.origin)

    return final_image


def inference(
    image,
    mdl,
    config=None,
    truncation=None,
    segmentation=None,
    target_range=[1, 0],
    poly_order='hist',
    dilation_amount=0,
    method='antspynet',
    patch_size=(64, 64, 64),
    patch_overlap=16,
    batch_size=1,
    verbose=False,
    anti_checkerboard='auto',
    anti_checkerboard_sigma=None,
    align_phase=True,
    linear_blend=None,
    pre_normalized=None,
):
    """
    Perform super-resolution inference on an input image, optionally guided by segmentation.

    Parameters
    ----------
    image : ants.ANTsImage
        Input image to be super-resolved.
    mdl : keras.Model
        Trained super-resolution model.
    config : dict, optional
        Provenance config dictionary loaded via ``siq.load_siq_model()``.
    truncation : tuple or list of float, optional
        Percentile values (e.g. [0.001, 0.999]) for intensity truncation before model input.
    segmentation : ants.ANTsImage, optional
        A labeled segmentation mask.
    target_range : list of float
        Intensity range used for scaling the input.
    poly_order : int, str or None
        Determines how to match intensity between super-resolved image and original.
    pre_normalized : bool, optional
        If True, the input is already normalized to [0, 1]; truncation and min-max
        re-normalization are skipped to preserve exact radiometric fidelity.
        If None, automatically detected if values lie within [0.0, 1.0].
    """
    # ── Provenance / config resolution ──────────────────────────────────────
    if config is None:
        warnings.warn(
            "siq.inference() called without a provenance config. "
            "This is deprecated and may produce incorrect results due to "
            "normalization mismatch. Use siq.load_siq_model() to obtain "
            "both the model and its config, then pass config here.",
            DeprecationWarning, stacklevel=2
        )
        _trunc_q = list(truncation) if truncation is not None else [0.001, 0.999]
        _patch_overlap = patch_overlap
        _model_patch = list(mdl.input_shape[1:-1])
        _output_clip = True
        _anti_cb = anti_checkerboard
        _anti_cb_sigma = anti_checkerboard_sigma
        _align_phase = align_phase
    else:
        _norm = config.get('normalization', {})
        _trunc_q = _norm.get('truncate_quantiles', [0.001, 0.999])
        _infer = config.get('inference', {})
        _patch_overlap = _infer.get('patch_overlap', patch_overlap)
        _model_patch = config.get('input_patch_shape', list(mdl.input_shape[1:]))[:-1]
        _output_clip = True
        _anti_cb = _infer.get('anti_checkerboard', anti_checkerboard)
        _anti_cb_sigma = _infer.get('anti_checkerboard_sigma', anti_checkerboard_sigma)
        _align_phase = _infer.get('align_phase', align_phase)

    def apply_intensity_match(sr_image, reference_image, order, verbose=False):
        if order is None:
            return sr_image
        if verbose:
            print("Applying intensity match with", order)
        if order == 'hist':
            return ants.histogram_match_image(sr_image, reference_image)
        else:
            return ants.regression_match_image(sr_image, reference_image, poly_order=order)

    # ── Normalization preflight ─────────────────────────────────────────────
    # If pre_normalized is None, auto-detect if the input image is already
    # normalized to [0, 1]. In particular, if min >= -1e-4 and max <= 1.0 + 1e-4
    # and max > 0.05, it is pre-normalized (e.g. benchmark slices, cached arrays).
    if pre_normalized is None:
        try:
            imin = float(image.min())
            imax = float(image.max())
            is_prenorm = (-1e-4 <= imin) and (imax <= 1.0 + 1e-4) and (imax > 0.05)
        except Exception:
            is_prenorm = False
    else:
        is_prenorm = bool(pre_normalized)

    pimg = ants.image_clone(image)
    if is_prenorm:
        pimg_norm = pimg
        if poly_order == 'hist':
            poly_order = None
    else:
        pimg = ants.iMath(pimg, 'TruncateIntensity', _trunc_q[0], _trunc_q[1])
        pimg_norm = ants.iMath(pimg, 'Normalize')

    input_shape = mdl.inputs[0].shape
    num_channels = int(input_shape[-1])

    if segmentation is not None:
        from .get_data import region_wise_super_resolution_blended
        if num_channels == 1:
            if verbose:
                print("Using region-wise super resolution due to single-channel model with segmentation.")
            sr = region_wise_super_resolution_blended(
                pimg, segmentation, mdl,
                dilation_amount=dilation_amount,
                verbose=verbose
            )
            ref = ants.resample_image_to_target(pimg, sr)
            return apply_intensity_match(sr, ref, poly_order, verbose)
        else:
            mynp = segmentation.numpy()
            mynp = list(np.unique(mynp)[1:len(mynp)].astype(int))
            upFactor = []
            if len(input_shape) == 5:
                testarr = np.zeros([1, 8, 8, 8, 2])
                testarrout = mdl(testarr)
                for k in range(3):
                    upFactor.append(int(testarrout.shape[k + 1] / testarr.shape[k + 1]))
            elif len(input_shape) == 4:
                testarr = np.zeros([1, 8, 8, 2])
                testarrout = mdl(testarr)
                for k in range(2):
                    upFactor.append(int(testarrout.shape[k + 1] / testarr.shape[k + 1]))

            concat_output = mdl.output
            dimensionality = len(concat_output.shape) - 2
            split_outputs = ops.split(concat_output, 2, axis=dimensionality + 1)
            inference_model = Model(inputs=mdl.input, outputs=split_outputs)

            temp = antspyt1w.super_resolution_segmentation_per_label(
                pimg,
                segmentation,
                upFactor,
                inference_model,
                segmentation_numbers=mynp,
                target_range=target_range,
                dilation_amount=dilation_amount,
                poly_order=poly_order,
                max_lab_plus_one=True
            )
            imgsr = temp['super_resolution']
            ref = ants.resample_image_to_target(pimg, imgsr)
            temp['super_resolution'] = apply_intensity_match(imgsr, ref, poly_order, verbose)
            return temp

    # ── Step 2: Dispatch based on method ────────────────────────────────────
    input_shape_list = list(pimg_norm.shape)
    _upfactor = config.get('upsample_factor', 2) if config is not None else 2

    if method == 'patchwise':
        effective_patch = []
        for i, p in enumerate(_model_patch):
            if p is None:
                effective_patch.append(patch_size[i] if i < len(patch_size) else 64)
            else:
                effective_patch.append(int(p))
        effective_patch = tuple(effective_patch)

        if verbose:
            print(f"[siq] overlapping_patch_inference: Gaussian-blended patches {effective_patch} (overlap={_patch_overlap})")
        imgsr = overlapping_patch_inference(
            pimg_norm, mdl,
            patch_size=effective_patch,
            overlap=_patch_overlap,
            batch_size=batch_size,
            verbose=verbose,
            volume_normalized=True,
            align_phase=_align_phase,
        )
    else:
        # Default: direct full-volume inference in a single forward pass
        if verbose:
            if isinstance(_upfactor, (list, tuple)):
                print(f"[siq] Direct full-volume inference: shape {input_shape_list} -> {[f * s for f, s in zip(_upfactor, input_shape_list)]}")
            else:
                print(f"[siq] Direct full-volume inference: shape {input_shape_list} -> {[_upfactor * s for s in input_shape_list]}")

        dim = pimg_norm.dimension
        diag = [float(pimg_norm.direction[i, i]) for i in range(dim)]
        flip_axes = tuple(i for i, d in enumerate(diag) if d < 0)

        arr_np = pimg_norm.numpy().astype('float32')
        if flip_axes:
            arr_np = np.flip(arr_np, axis=flip_axes).copy()

        arr = arr_np[np.newaxis, ..., np.newaxis]
        out = mdl.predict(arr, verbose=0)[0, ..., 0]

        if flip_axes:
            out = np.flip(out, axis=flip_axes).copy()

        has_tc = _model_has_transposed_conv(mdl)
        if isinstance(_upfactor, (list, tuple)):
            f_list = list(_upfactor)
        else:
            f_list = [_upfactor] * dim

        shift_vec = []
        for d, f in enumerate(f_list):
            if f <= 1:
                shift_vec.append(0.0)
            elif has_tc:
                shift_vec.append(-float(f - 1) / 2.0 if _align_phase else 0.0)
            else:
                if d in flip_axes:
                    shift_vec.append(-float(f - 1))
                else:
                    shift_vec.append(0.0)

        if any(abs(s) > 1e-4 for s in shift_vec):
            out = nd_shift(out, tuple(shift_vec), order=3, mode='nearest')

        if _output_clip:
            out = np.clip(out, 0.0, 1.0)
        if isinstance(_upfactor, (list, tuple)):
            new_spacing = tuple(float(s) / f for s, f in zip(pimg_norm.spacing, _upfactor))
        else:
            new_spacing = tuple(float(s) / _upfactor for s in pimg_norm.spacing)
        imgsr = ants.from_numpy(out)
        ants.set_spacing(imgsr, new_spacing)
        ants.set_direction(imgsr, pimg_norm.direction)
        ants.set_origin(imgsr, pimg_norm.origin)

    # ── Step 3: Anti-Checkerboard Sub-Voxel Notch Filter ────────────────────
    if _anti_cb and _anti_cb not in [False, 0, 'none', 'false', 'False']:
        if _anti_cb_sigma is None or _anti_cb_sigma in ['auto', 'Auto']:
            effective_sigma = estimate_anti_checkerboard_sigma(imgsr.numpy())
            if verbose:
                print(f"[siq] Empirically calculated anti-checkerboard sigma={effective_sigma:.3f} (data-driven Nyquist attenuation)")
        else:
            effective_sigma = float(_anti_cb_sigma)

        if effective_sigma > 0.05:
            if verbose and (_anti_cb_sigma is not None and _anti_cb_sigma not in ['auto', 'Auto']):
                print(f"[siq] Applying anti-checkerboard sub-voxel notch filter (sigma={effective_sigma:.3f})")
            imgsr_np = gaussian_filter(imgsr.numpy().astype(np.float32), sigma=effective_sigma)
            if _output_clip:
                imgsr_np = np.clip(imgsr_np, 0.0, 1.0)
            imgsr_cleaned = ants.from_numpy(imgsr_np)
            ants.copy_image_info(imgsr, imgsr_cleaned)
            imgsr = imgsr_cleaned
        elif verbose:
            print("[siq] No checkerboard artifact detected (excess <= 1.2x); zero anti-checkerboard filtering applied.")

    ref = ants.resample_image_to_target(pimg, imgsr)
    _linear_blend = linear_blend
    if _linear_blend is None and config is not None:
        _linear_blend = config.get('inference', {}).get('linear_blend', None)

    # Only blend when genuinely requested strictly between 0 and 1
    # (1.0 or None means pure model output)
    if _linear_blend is not None and 0.0 < _linear_blend < 1.0:
        ref_arr = ref.numpy().astype(np.float32)
        sr_arr = imgsr.numpy().astype(np.float32)
        fused_np = (1.0 - _linear_blend) * ref_arr + _linear_blend * sr_arr
        if _output_clip:
            fused_np = np.clip(fused_np, 0.0, 1.0)
        imgsr_fused = ants.from_numpy(fused_np)
        ants.copy_image_info(imgsr, imgsr_fused)
        imgsr = imgsr_fused

    return apply_intensity_match(imgsr, ref, poly_order, verbose)

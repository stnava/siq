import ants
import numpy as np
import keras
from keras import layers, ops
from .get_data import (simulate_image, simulate_image_multi_scale,
                       simulate_brain_procedural, simulate_sinewave, simulate_layered,
                       simulate_vessel_tubes, simulate_cellular_voronoi,
                       simulate_geometric_phantoms, simulate_grid_patterns,
                       simulate_fractal_noise,
                       add_rician_noise, get_grader_feature_network, _sample_param,
                       pseudo_3d_vgg_features_unbiased, vgg_features_2d, save_siq_model, default_siq_config)
from .espcn import create_espcn_3d, create_espcn_3d_residual, _normalize_factor
from .alignment import augment_geometry_hr, audit_pair_alignment
import os
import time
import random

def blind_sr_generator_simple(batch_size=4, patch_size=(32, 32, 32), factor=2, feature_model=None):
    """
    Simple generator for Blind Super-Resolution.
    Uses basic simulate_image and simple degradation (resampling).
    """
    while True:
        x_batch = []
        y_batch = []
        f_batch = []
        for _ in range(batch_size):
            levels = np.random.randint(5, 25)
            hr_img = simulate_image(shaper=patch_size, n_levels=levels)
            hr_np = hr_img.numpy().astype("float32")
            hr_np = (hr_np - hr_np.min()) / (hr_np.max() - hr_np.min() + 1e-8)
            
            lr_target_spacing = tuple(float(s * factor) for s in hr_img.spacing)
            lr_img = ants.resample_image(hr_img, lr_target_spacing, use_voxels=False, interp_type=0)
            lr_np = lr_img.numpy().astype("float32")
            lr_np = (lr_np - lr_np.min()) / (lr_np.max() - lr_np.min() + 1e-8)
            lr_np = np.clip(lr_np + np.random.normal(0, 0.01, lr_np.shape), 0, 1)
            
            x_batch.append(np.expand_dims(lr_np, -1))
            y_batch.append(np.expand_dims(hr_np, -1))
            
            if feature_model:
                f_hr = feature_model.predict(np.expand_dims(y_batch[-1], 0), verbose=0)
                f_batch.append(f_hr[0])
                
        x_out = np.array(x_batch, dtype="float32")
        y_out = np.array(y_batch, dtype="float32")
        
        if feature_model:
            f_out = np.array(f_batch, dtype="float32")
            yield x_out, (y_out, f_out)
        else:
            yield x_out, y_out

DEFAULT_SIMULATION_CLASSES = {
    "brain_procedural": 1.0 / 9.0,
    "layered": 1.0 / 9.0,
    "sinewave": 1.0 / 9.0,
    "organic_blobs": 1.0 / 9.0,
    "vessel_tubes": 1.0 / 9.0,
    "cellular_voronoi": 1.0 / 9.0,
    "geometric_phantoms": 1.0 / 9.0,
    "grid_patterns": 1.0 / 9.0,
    "fractal_noise": 1.0 / 9.0,
}

def blind_sr_generator(
    hr_base_cache=None,
    batch_size=4,
    lr_patch_size=16,
    factor=2,
    gamma_range=(0.6, 1.7),
    blur_sigma_range={"type": "poisson", "lam": 0.2, "scale": 0.5},
    noise_std_range=(0.0, 0.03),
    interp_types=(0, 1, 2),
    sim_params=None,
    simulation_classes=None,
    use_rician_noise=False,
    zoom_range=(0.7, 1.4),
    cache_size=1024,
    use_cache=True,
    dimensionality=3,
    use_layer2=False
):
    """
    Advanced generator for Blind Super-Resolution.
    Features: multi-scale simulation, stochastic blur, downsampling, noise, and spatial augmentations.
    
    Args:
        hr_base_cache: List of ants.images or a single 4D/3D numpy array.
                       If None, generates a fallback cache dynamically.
        batch_size: Number of samples per batch.
        lr_patch_size: Size of the low-resolution patch. High-res will be lr_patch_size * factor.
        factor: Super-resolution factor (integer).
        gamma_range: Range for random gamma contrast perturbation.
        blur_sigma_range: Range for random Gaussian blur sigma (default: Poisson lam=0.2, scale=0.5, biased toward minimal blur).
        noise_std_range: Range for random additive Gaussian noise standard deviation.
        interp_types: Tuple of available interpolation types for resampling.
        sim_params: Optional dict of parameters for simulate_image_multi_scale.
        simulation_classes: Optional dict of {class_name: frequency} summing to 1.0. Defaults to equal mixture of all 9 procedural simulation classes.
        use_rician_noise: Whether to use Rician noise instead of additive Gaussian noise.
        zoom_range: Range of scales for stochastic scaling of coordinate grids.
        cache_size: Size of fallback cache if hr_base_cache is None.
        use_cache: If False, generates training volumes raw on the fly (no cache).
        dimensionality: Dimensionality of generator (2 or 3).
    """
    from .espcn import _normalize_factor
    factor_tuple = _normalize_factor(factor, dimensionality)
    if isinstance(lr_patch_size, (list, tuple)):
        lr_patch_shape = tuple(int(s) for s in lr_patch_size)
    else:
        lr_patch_shape = tuple([int(lr_patch_size)] * dimensionality)
    hr_patch_shape = tuple(p * f for p, f in zip(lr_patch_shape, factor_tuple))
    hr_large_shape = tuple(int(round(p * 1.5)) for p in hr_patch_shape)
    lr_large_shape = tuple(int(round(p * 1.5)) for p in lr_patch_shape)
    
    if sim_params is None:
        sim_params = {}
        
    if simulation_classes is None:
        simulation_classes = DEFAULT_SIMULATION_CLASSES.copy()
    else:
        total_freq = sum(simulation_classes.values())
        if not np.isclose(total_freq, 1.0):
            raise ValueError(f"Frequencies in simulation_classes must sum to 1.0, got {total_freq}")
            
    classes = list(simulation_classes.keys())
    probs = list(simulation_classes.values())
    
    # Pre-generate fallback cache if none provided and cache is enabled
    if use_cache and hr_base_cache is None:
        hr_base_cache = []
        for _ in range(cache_size):
            sim_class = np.random.choice(classes, p=probs)
            if sim_class == "organic_blobs":
                s_params = sim_params.copy()
                if "scale_range" not in s_params:
                    s_params["scale_range"] = zoom_range
                vol = simulate_image_multi_scale(hr_large_shape, **s_params)
            elif sim_class == "brain_procedural":
                vol = simulate_brain_procedural(hr_large_shape, zoom_range=zoom_range, use_layer2=use_layer2)
            elif sim_class == "sinewave":
                vol = simulate_sinewave(hr_large_shape, zoom_range=zoom_range, use_layer2=use_layer2)
            elif sim_class == "layered":
                vol = simulate_layered(hr_large_shape, zoom_range=zoom_range, use_layer2=use_layer2)
            elif sim_class == "vessel_tubes":
                vol = simulate_vessel_tubes(hr_large_shape, zoom_range=zoom_range, use_layer2=use_layer2)
            elif sim_class == "cellular_voronoi":
                vol = simulate_cellular_voronoi(hr_large_shape, zoom_range=zoom_range, use_layer2=use_layer2)
            elif sim_class == "geometric_phantoms":
                vol = simulate_geometric_phantoms(hr_large_shape, zoom_range=zoom_range, use_layer2=use_layer2)
            elif sim_class == "grid_patterns":
                vol = simulate_grid_patterns(hr_large_shape, zoom_range=zoom_range, use_layer2=use_layer2)
            elif sim_class == "fractal_noise":
                vol = simulate_fractal_noise(hr_large_shape, zoom_range=zoom_range, use_layer2=use_layer2)
            else:
                raise ValueError(f"Unknown simulation class: {sim_class}")
            hr_base_cache.append(vol)
        
    is_numpy_cache = hr_base_cache is not None and hasattr(hr_base_cache, "shape") and len(hr_base_cache.shape) == (dimensionality + 1)
    
    while True:
        x_batch = []
        y_batch = []
        for _ in range(batch_size):
            if not use_cache or hr_base_cache is None:
                # 1. Generate directly on the fly
                sim_class = np.random.choice(classes, p=probs)
                if sim_class == "organic_blobs":
                    s_params = sim_params.copy()
                    if "scale_range" not in s_params:
                        s_params["scale_range"] = zoom_range
                    vol = simulate_image_multi_scale(hr_large_shape, **s_params)
                elif sim_class == "brain_procedural":
                    vol = simulate_brain_procedural(hr_large_shape, zoom_range=zoom_range, use_layer2=use_layer2)
                elif sim_class == "sinewave":
                    vol = simulate_sinewave(hr_large_shape, zoom_range=zoom_range, use_layer2=use_layer2)
                elif sim_class == "layered":
                    vol = simulate_layered(hr_large_shape, zoom_range=zoom_range, use_layer2=use_layer2)
                elif sim_class == "vessel_tubes":
                    vol = simulate_vessel_tubes(hr_large_shape, zoom_range=zoom_range, use_layer2=use_layer2)
                elif sim_class == "cellular_voronoi":
                    vol = simulate_cellular_voronoi(hr_large_shape, zoom_range=zoom_range, use_layer2=use_layer2)
                elif sim_class == "geometric_phantoms":
                    vol = simulate_geometric_phantoms(hr_large_shape, zoom_range=zoom_range, use_layer2=use_layer2)
                elif sim_class == "grid_patterns":
                    vol = simulate_grid_patterns(hr_large_shape, zoom_range=zoom_range, use_layer2=use_layer2)
                elif sim_class == "fractal_noise":
                    vol = simulate_fractal_noise(hr_large_shape, zoom_range=zoom_range, use_layer2=use_layer2)
                else:
                    raise ValueError(f"Unknown simulation class: {sim_class}")
                hr_large = vol
                hr_large_np = hr_large.numpy().astype("float32")
            else:
                # 1. Sample from cache
                if is_numpy_cache:
                    idx = np.random.randint(0, hr_base_cache.shape[0])
                    vol = hr_base_cache[idx]
                    h_size = hr_large_shape[0]
                    if vol.shape[0] > h_size:
                        if dimensionality == 2:
                            x_s = np.random.randint(0, vol.shape[0] - h_size + 1)
                            y_s = np.random.randint(0, vol.shape[1] - h_size + 1)
                            hr_large_np = vol[x_s:x_s+h_size, y_s:y_s+h_size].astype("float32")
                        else:
                            x_s = np.random.randint(0, vol.shape[0] - h_size + 1)
                            y_s = np.random.randint(0, vol.shape[1] - h_size + 1)
                            z_s = np.random.randint(0, vol.shape[2] - h_size + 1)
                            hr_large_np = vol[x_s:x_s+h_size, y_s:y_s+h_size, z_s:z_s+h_size].astype("float32")
                    else:
                        hr_large_np = vol.astype("float32")
                    hr_large = ants.from_numpy(hr_large_np)
                else:
                    hr_large = random.choice(hr_base_cache)
                    hr_large_np = hr_large.numpy().astype("float32")
            
            # Geometric augmentation on the HR volume BEFORE degradation so that the
            # LR image is derived on the (reflected) HR grid — alignment preserved.
            hr_large_np = augment_geometry_hr(hr_large_np, factor_tuple)

            # Normalize and Gamma
            hr_min, hr_max = hr_large_np.min(), hr_large_np.max()
            if hr_max > hr_min:
                hr_large_np = (hr_large_np - hr_min) / (hr_max - hr_min + 1e-8)
            
            gamma = _sample_param(gamma_range, (0.6, 1.7))
            hr_large_np = np.clip(hr_large_np ** gamma, 0, 1)
            
            if use_layer2:
                # 1. Modality/Contrast Negation (50% probability)
                if np.random.choice([True, False]):
                    hr_large_np = 1.0 - hr_large_np
                # 2. Stochastic Bias Field Inhomogeneity
                if np.random.choice([True, False]):
                    grid_bias = [np.linspace(-1, 1, s) for s in hr_large_shape]
                    mesh_bias = np.meshgrid(*grid_bias, indexing="ij")
                    cx = np.random.uniform(-0.5, 0.5)
                    cy = np.random.uniform(-0.5, 0.5)
                    strength = np.random.uniform(0.08, 0.22)
                    if dimensionality == 3:
                           cz = np.random.uniform(-0.5, 0.5)
                           dist_sq = (mesh_bias[0] - cx)**2 + (mesh_bias[1] - cy)**2 + (mesh_bias[2] - cz)**2
                    else:
                           dist_sq = (mesh_bias[0] - cx)**2 + (mesh_bias[1] - cy)**2
                    bias_field = 1.0 - strength * dist_sq
                    hr_large_np = np.clip(hr_large_np * bias_field, 0.0, 1.0)
            
            hr_large = ants.from_numpy(hr_large_np)
            
            # 2. Stochastic Degradation (Blur + Resample)
            sigma = _sample_param(blur_sigma_range, {"type": "poisson", "lam": 0.2, "scale": 0.5})
            lr_large = ants.smooth_image(hr_large, sigma) if sigma > 0.1 else ants.image_clone(hr_large)
            
            interp = np.random.choice(interp_types)
            # Use exact factor spacing (use_voxels=False) so that LR pixel k aligns
            # exactly with HR pixel k*factor. use_voxels=True gives spacing 191/95≈2.0105
            # instead of 2.0, causing a systematic 0.5-pixel center offset between every
            # training (LR, HR) pair — the model learns this shift and applies it at
            # inference time, producing SR output shifted relative to the GT.
            lr_target_spacing = tuple(float(f) for f in factor_tuple)  # HR has spacing 1.0
            lr_large = ants.resample_image(lr_large, lr_target_spacing, use_voxels=False, interp_type=interp)
            
            lr_large_np = lr_large.numpy()
            lr_min, lr_max = lr_large_np.min(), lr_large_np.max()
            if lr_max > lr_min:
                lr_large_np = (lr_large_np - lr_min) / (lr_max - lr_min + 1e-8)
                
            hr_large_np = hr_large.numpy()
            
            # 3. Central Cropping
            lr_starts = [(lr_large_shape[i] - lr_patch_shape[i]) // 2 for i in range(dimensionality)]
            lr_ends = [lr_starts[i] + lr_patch_shape[i] for i in range(dimensionality)]
            hr_starts = [lr_starts[i] * factor_tuple[i] for i in range(dimensionality)]
            hr_ends = [hr_starts[i] + hr_patch_shape[i] for i in range(dimensionality)]
            
            if dimensionality == 2:
                hr_crop = hr_large_np[hr_starts[0]:hr_ends[0], hr_starts[1]:hr_ends[1]]
                lr_crop = lr_large_np[lr_starts[0]:lr_ends[0], lr_starts[1]:lr_ends[1]]
            else:
                hr_crop = hr_large_np[hr_starts[0]:hr_ends[0], hr_starts[1]:hr_ends[1], hr_starts[2]:hr_ends[2]]
                lr_crop = lr_large_np[lr_starts[0]:lr_ends[0], lr_starts[1]:lr_ends[1], lr_starts[2]:lr_ends[2]]
                
            # 4. Geometric augmentation (flips / rot90 / transposes) is applied to the HR
            # volume BEFORE degradation (see augment_geometry_hr above). Reflecting an
            # already-formed (LR, HR) pair would shift alignment by (f-1) voxels.
                
            # 5. Noise
            noise_std = _sample_param(noise_std_range, (0.0, 0.03))
            if noise_std > 0.0:
                if use_rician_noise:
                    lr_crop = add_rician_noise(lr_crop, noise_std)
                else:
                    lr_crop = np.clip(lr_crop + np.random.normal(0, noise_std, lr_crop.shape), 0, 1)
                
            x_batch.append(np.expand_dims(lr_crop, -1))
            y_batch.append(np.expand_dims(hr_crop, -1))
            
        yield np.array(x_batch, dtype="float32"), np.array(y_batch, dtype="float32")

def train_blind_espcn_perceptual(factor=2, epochs=20, steps_per_epoch=50, feature_weight=2.0):
    """
    Legacy training function for Perceptual Blind SR.
    """
    patch_size_hr = (64, 64, 64)
    
    base_model = create_espcn_3d(input_shape=(32, 32, 32, 1), factor=factor)
    
    try:
        feature_extractor = get_grader_feature_network(layer=6)
        feature_extractor.trainable = False
        has_feature = True
        print("Loaded ResNet grader for perceptual loss.")
    except Exception as e:
        print(f"Could not load perceptual model: {e}. Falling back to MSE only.")
        has_feature = False

    if has_feature:
        inputs = base_model.input
        hr_output = base_model.output
        feat_output = feature_extractor(hr_output)
        
        train_model = keras.Model(inputs=inputs, outputs=[hr_output, feat_output])
        train_model.compile(
            optimizer=keras.optimizers.Adam(5e-5),
            loss=["mse", "mse"],
            loss_weights=[1.0, feature_weight]
        )
        gen = blind_sr_generator_simple(batch_size=4, patch_size=patch_size_hr, factor=factor, feature_model=feature_extractor)
    else:
        train_model = base_model
        train_model.compile(optimizer=keras.optimizers.Adam(5e-5), loss="mse")
        gen = blind_sr_generator_simple(batch_size=4, patch_size=patch_size_hr, factor=factor)

    print("Starting Perceptual Blind SR training...")
    train_model.fit(gen, epochs=epochs, steps_per_epoch=steps_per_epoch)
    
    base_model.save("espcn_3d_perceptual.keras")
    return base_model

def prepare_2d_validation(val_image=None, factor=(2, 2)):
    """Build a (LR, HR) 2D validation pair from a path, an ANTsImage, or "r16".

    Default (``None`` / ``"r16"``) is the head-cropped ANTs ``r16`` slice. The HR
    image is cropped to a multiple of ``factor`` and the LR image is produced with
    the standard alignment convention (``use_voxels=False``, exact factor spacing,
    shared origin), identical to the 3D validation harness.
    """
    factor = tuple(int(f) for f in factor)
    if val_image is None or (isinstance(val_image, str) and val_image.lower() == "r16"):
        img = ants.image_read(ants.get_data("r16"))
    elif isinstance(val_image, str):
        img = ants.image_read(val_image)
    else:
        img = val_image
    if img.dimension != 2:
        raise ValueError(f"2D validation image required, got dimension {img.dimension}")
    img = ants.crop_image(img, ants.get_mask(img))
    img = ants.iMath(img, "Normalize")
    shp = [(s // f) * f for s, f in zip(img.shape, factor)]
    img = ants.crop_indices(img, [0, 0], shp)
    img = ants.iMath(ants.iMath(img, "TruncateIntensity", 0.001, 0.999), "Normalize")
    lr_spacing = [float(sp * f) for sp, f in zip(img.spacing, factor)]
    lr = ants.resample_image(img, lr_spacing, use_voxels=False, interp_type=0)
    return lr, img


def train_blind_sr_kitchen_sink(
    output_prefix="espcn_3d_blind",
    factor=2,
    iterations=1000,
    hr_base_cache=None,
    learning_rate=1e-4,
    use_residual=True,
    msq_weight=10.0,
    l1_weight=None,
    feat_weight=2.0,
    tv_weight=0.1,
    feature_type="grader",
    feature_layer=6,
    model=None,
    pretrain_iterations=0,
    batch_size=4,
    lr_patch_size=16,
    eval_freq=50,
    checkpoint_freq=100,
    val_image=None,
    enable_html_report=True,
    dimensionality=3,
    **generator_kwargs
):
    """
    Advanced 'Kitchen-Sink' training loop for Blind Super-Resolution (2D or 3D).
    Uses custom loss (L1/MSE + Perceptual + TV) with support for VGG or ResNet Grader.
    
    Args:
        output_prefix: Prefix for saved model files.
        factor: Super-resolution scaling factor (integer or tuple).
        iterations: Number of perceptual training iterations.
        hr_base_cache: Optional cache of HR images.
        learning_rate: Adam optimizer learning rate.
        use_residual: If True and model is None, builds residual ESPCN.
        msq_weight: Weight for MSE loss.
        l1_weight: Weight for L1 (MAE) loss (enforces sharp boundaries).
        feat_weight: Weight for perceptual loss.
        tv_weight: Weight for Total Variation loss.
        feature_type: 'vgg' (pseudo-3D VGG19 in 3D, VGG19 in 2D) or 'grader' (3D ResNet grader).
        feature_layer: Internal layer index for feature extraction (default: 6).
        model: Optional pre-instantiated model (e.g. siq.default_dbpn(..., option='small')).
        pretrain_iterations: Number of MSE-only warmup iterations before activating perceptual loss.
        batch_size: Number of patches per batch.
        lr_patch_size: Size of low-resolution patch.
        dimensionality: 2 or 3. In 2D the identical pipeline (generator, audit, VGG-L6
            perceptual loss, QC reporter) is used on 2D patches for fast iteration.
        val_image: Path to validation image, an ANTsImage, or the string "r16"
            (cropped ANTs r16 2D slice; the default in 2D).
        **generator_kwargs: Passed to blind_sr_generator.
    """
    dim = int(dimensionality)
    if dim not in (2, 3):
        raise ValueError(f"dimensionality must be 2 or 3, got {dimensionality}")
    factor_tuple = _normalize_factor(factor, dim)
    if isinstance(lr_patch_size, (list, tuple)):
        hr_patch_shape = tuple(int(p * f) for p, f in zip(lr_patch_size, factor_tuple))
    else:
        hr_patch_shape = tuple(int(lr_patch_size * f) for f in factor_tuple)
    generator_kwargs["dimensionality"] = dim
    red_axes = list(range(1, dim + 2))  # all non-batch axes (spatial + channel)

    # 1. Instantiate or use provided Model
    if model is None:
        if dim == 2:
            from .get_data import default_dbpn
            model = default_dbpn(strider=list(factor_tuple), dimensionality=2,
                                 nChannelsIn=1, nChannelsOut=1,
                                 sigmoid_second_channel=False, option="small")
        elif use_residual:
            model = create_espcn_3d_residual(input_shape=(None, None, None, 1), factor=factor, n_filters=128, n_res_blocks=8)
        else:
            model = create_espcn_3d(input_shape=(None, None, None, 1), factor=factor)

        
    # 2. Setup Perceptual Model
    try:
        if feature_type == "vgg" and dim == 2:
            feature_extractor = vgg_features_2d(inshape=list(hr_patch_shape), layer=feature_layer)
            feature_extractor.trainable = False
            print(f"Loaded 2D VGG19 Layer {feature_layer} feature extractor for perceptual loss.")
        elif feature_type == "vgg":
            feature_extractor = pseudo_3d_vgg_features_unbiased(inshape=list(hr_patch_shape), layer=feature_layer)
            feature_extractor.trainable = False
            print(f"Loaded pseudo-3D VGG19 Layer {feature_layer} feature extractor for perceptual loss.")
        else:
            feature_extractor = get_grader_feature_network(layer=feature_layer)
            feature_extractor.trainable = False
            print(f"Loaded 3D ResNet grader Layer {feature_layer} feature extractor for perceptual loss.")
    except Exception as e:
        print(f"Warning: Could not load perceptual model ({e}). Using MSE only.")
        feature_extractor = None

    # 3. Custom Loss with dynamic weights
    msq_weight_var = keras.Variable(float(msq_weight), dtype="float32")
    l1_weight_var = keras.Variable(float(l1_weight) if l1_weight is not None else 0.0, dtype="float32")
    feat_weight_var = keras.Variable(float(feat_weight), dtype="float32") if feature_extractor else keras.Variable(0.0)
    tv_weight_var = keras.Variable(float(tv_weight), dtype="float32")

    def _call_fe(tensor):
        if feature_type == "grader":
            # Rank normalization required for ResNet grader
            s = ops.shape(tensor)
            flat = ops.reshape(tensor, (s[0], -1))
            ranks = ops.cast(ops.argsort(ops.argsort(flat, axis=-1), axis=-1), "float32")
            denom = ops.cast(ops.shape(flat)[-1] - 1, "float32")
            return feature_extractor(ops.reshape(ranks / ops.maximum(denom, 1.0), s), training=False)
        else:
            # VGG has internal Rescaling(255, -127.5) baked in
            return feature_extractor(tensor, training=False)
    
    def custom_loss(y_true, y_pred):
        abs_diff = ops.abs(y_true - y_pred)
        l1_term = ops.mean(abs_diff, axis=red_axes)
        
        squared_diff = ops.square(y_true - y_pred)
        msq_term = ops.mean(squared_diff, axis=red_axes)
        
        loss = l1_term * l1_weight_var + msq_term * msq_weight_var
        
        if feature_extractor:
            f_true = _call_fe(y_true)
            f_pred = _call_fe(y_pred)
            if not isinstance(f_true, list):
                f_true = [f_true]
                f_pred = [f_pred]
            feat_term = sum(ops.mean(ops.square(ft - fp), axis=list(range(1, len(ft.shape))))
                            for ft, fp in zip(f_true, f_pred))
            loss = loss + feat_term * feat_weight_var
            
        # TV Term
        tv_sum = 0.0
        for _ax in range(1, dim + 1):
            hi = [slice(None)] * (dim + 2); lo = [slice(None)] * (dim + 2)
            hi[_ax] = slice(1, None); lo[_ax] = slice(None, -1)
            tv_sum = tv_sum + ops.mean(ops.abs(y_pred[tuple(hi)] - y_pred[tuple(lo)]), axis=red_axes)
        loss = loss + tv_sum * tv_weight_var
        
        return loss

    # 4. Compile and Train
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=custom_loss)
    
    # Initialize Generator
    gen = blind_sr_generator(
        hr_base_cache=hr_base_cache,
        batch_size=batch_size,
        lr_patch_size=lr_patch_size,
        factor=factor,
        **generator_kwargs
    )

    # Pre-flight alignment audit: abort before wasting a training run on
    # misaligned (LR, HR) pairs (set SIQ_SKIP_ALIGNMENT_AUDIT=1 to bypass).
    if os.environ.get("SIQ_SKIP_ALIGNMENT_AUDIT", "0") != "1":
        audit_pair_alignment(gen, factor_tuple, n_batches=2)
    
    current_loss_weights = {
        "msq": float(msq_weight),
        "l1": float(l1_weight) if l1_weight is not None else 0.0,
        "feat": float(feat_weight),
        "tv": float(tv_weight),
        "feature_type": feature_type,
        "feature_layer": feature_layer
    }

    # 5. Initialize Validation Convergence Reporter (if enabled)
    reporter = None
    if enable_html_report:
        try:
            from tests.visual_convergence_report import VisualConvergenceReporter
        except Exception:
            try:
                from scripts.visual_convergence_report import VisualConvergenceReporter
            except Exception:
                try:
                    from visual_convergence_report import VisualConvergenceReporter
                except Exception as _err:
                    print(f"[blind_sr] Note: VisualConvergenceReporter could not be imported: {_err}")
                    VisualConvergenceReporter = None

        if VisualConvergenceReporter is not None:
            default_fpa = "/Users/stnava/data/blast_cohorts/BIDS/FPA/sub-BLAST022/ses-01/anat/sub-BLAST022_ses-01_run-001_T1w.nii.gz"
            val_path = val_image if (isinstance(val_image, str) and os.path.exists(val_image)) else (default_fpa if os.path.exists(default_fpa) else None)
            if val_path is None and dim == 3:
                try:
                    import antspynet
                    val_path = antspynet.get_antsxnet_data("oasis")
                except Exception:
                    val_path = None
                    
            if dim == 3 and val_path and os.path.exists(val_path):
                try:
                    print(f"Loading Real MRI validation volume for live monitoring: {val_path}...", flush=True)
                    val_vol = ants.image_read(val_path)
                    val_vol = ants.iMath(ants.iMath(val_vol, "TruncateIntensity", 0.001, 0.999), "Normalize")
                    val_target_spacing = [val_vol.spacing[d] * factor_tuple[d] for d in range(3)]
                    low_res_vol = ants.resample_image(val_vol, val_target_spacing, use_voxels=False, interp_type=0)
                    
                    lr_box = 32
                    val_shift = [40, 0, 40] if ("sub-BLAST" in val_path or "FPA" in val_path) else [0, 0, 0]
                    mid_lr = [low_res_vol.shape[d] // 2 + int(round(val_shift[d] / factor_tuple[d])) for d in range(3)]
                    mid_hr = [mid_lr[d] * factor_tuple[d] for d in range(3)]
                    
                    val_lr_patch = ants.crop_indices(low_res_vol, [max(0, mid_lr[d] - lr_box) for d in range(3)], [min(low_res_vol.shape[d], mid_lr[d] + lr_box) for d in range(3)])
                    val_hr_patch = ants.crop_indices(val_vol, [max(0, mid_hr[d] - lr_box * factor_tuple[d]) for d in range(3)], [min(val_vol.shape[d], mid_hr[d] + lr_box * factor_tuple[d]) for d in range(3)])
                    
                    reporter = VisualConvergenceReporter(
                        workspace_dir=".",
                        checkpoint_dir=f"checkpoints/{output_prefix}",
                        report_dir=f"reports/{output_prefix}",
                        html_filename=f"{output_prefix}_report.html",
                        selection_metric="pcs",
                        reset_history=True,
                    )
                    reporter.setup_validation_patches(val_lr_patch, val_hr_patch, factor=factor_tuple)
                    print(f"Validation Convergence Reporter initialized. Live dashboard: {output_prefix}_report.html", flush=True)
                except Exception as e:
                    print(f"Warning: Failed to setup validation patches for HTML report: {e}", flush=True)
                    reporter = None

    if enable_html_report and dim == 2 and reporter is None:
        try:
            try:
                from tests.visual_convergence_report import VisualConvergenceReporter as _VCR
            except Exception:
                from scripts.visual_convergence_report import VisualConvergenceReporter as _VCR
            val_lr_patch, val_hr_patch = prepare_2d_validation(val_image, factor_tuple)
            reporter = _VCR(
                workspace_dir=".",
                checkpoint_dir=f"checkpoints/{output_prefix}",
                report_dir=f"reports/{output_prefix}",
                html_filename=f"{output_prefix}_report.html",
                selection_metric="pcs",
                reset_history=True,
            )
            reporter.setup_validation_patches(val_lr_patch, val_hr_patch, factor=factor_tuple)
            print(f"2D validation (HR {val_hr_patch.shape} -> LR {val_lr_patch.shape}) ready. "
                  f"Live dashboard: {output_prefix}_report.html", flush=True)
        except Exception as e:
            print(f"Warning: Failed to setup 2D validation for HTML report: {e}", flush=True)
            reporter = None

    if reporter is not None:
        try:
            x_init, y_init = next(gen)
            reporter.record_checkpoint(model, 0, "Initial", 1.0, is_convergence_step=True, loss_weights=current_loss_weights, train_batch=(x_init, y_init))
        except Exception as e:
            print(f"Warning: Failed to record initial state checkpoint: {e}", flush=True)

    # Warmup Phase (if requested)
    if pretrain_iterations > 0:
        print(f"Starting warmup pretraining for {pretrain_iterations} iterations (MSE only)...", flush=True)
        active_feat_wt = float(feat_weight_var.value)
        feat_weight_var.assign(0.0)
        for i in range(1, pretrain_iterations + 1):
            x, y = next(gen)
            loss = model.train_on_batch(x, y)
            if i % 50 == 0 or i == 1:
                print(f"Warmup Iteration {i}/{pretrain_iterations} - loss: {float(loss):.6f}", flush=True)
            if reporter is not None and (i % eval_freq == 0 or i == pretrain_iterations):
                is_ckpt = (i % checkpoint_freq == 0 or i == pretrain_iterations)
                reporter.record_checkpoint(model, i, "Warmup", float(loss), is_convergence_step=is_ckpt, loss_weights=current_loss_weights, train_batch=(x, y))
        feat_weight_var.assign(active_feat_wt)

    # Main Perceptual Phase
    print(f"Starting Blind SR perceptual training for {iterations} iterations...", flush=True)
    for i in range(1, iterations + 1):
        x, y = next(gen)
        loss = model.train_on_batch(x, y)
        
        if i % 50 == 0 or i == 1:
            print(f"Iteration {i}/{iterations} - loss: {float(loss):.6f}", flush=True)
            
        step_num = pretrain_iterations + i
        if reporter is not None and (i % eval_freq == 0 or i == 1 or i == iterations):
            is_ckpt = (i % checkpoint_freq == 0 or i == iterations)
            entry = reporter.record_checkpoint(model, step_num, "Perceptual", float(loss), is_convergence_step=is_ckpt, loss_weights=current_loss_weights, train_batch=(x, y))
            if entry and entry.get("is_best", 0) == 1:
                config = default_siq_config(model)
                config["model_type"] = "blind_sr"
                config["upsample_factor"] = factor_tuple if len(factor_tuple) > 1 else factor_tuple[0]
                config["loss_weights"] = current_loss_weights
                save_siq_model(f"{output_prefix}_best.keras", model, config)
        elif i % checkpoint_freq == 0:
            config = default_siq_config(model)
            config["model_type"] = "blind_sr"
            config["upsample_factor"] = factor_tuple if len(factor_tuple) > 1 else factor_tuple[0]
            config["loss_weights"] = current_loss_weights
            save_siq_model(f"{output_prefix}_best.keras", model, config)
            
    best_path = f"{output_prefix}_best.keras"
    if not os.path.exists(best_path):
        config = default_siq_config(model)
        config["model_type"] = "blind_sr"
        config["upsample_factor"] = factor_tuple if len(factor_tuple) > 1 else factor_tuple[0]
        config["loss_weights"] = current_loss_weights
        save_siq_model(best_path, model, config)
        
    latest_path = f"{output_prefix}_latest.keras"
    config = default_siq_config(model)
    config["model_type"] = "blind_sr"
    config["upsample_factor"] = factor_tuple if len(factor_tuple) > 1 else factor_tuple[0]
    config["loss_weights"] = current_loss_weights
    save_siq_model(latest_path, model, config)
    print(f"\nModel and config saved to {best_path} and {latest_path}", flush=True)
    if reporter is not None:
        print(f"Live HTML Convergence Dashboard available at: {output_prefix}_report.html", flush=True)
    return model

if __name__ == "__main__":
    train_blind_espcn_perceptual(epochs=1, steps_per_epoch=10)

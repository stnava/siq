import ants
import numpy as np
import keras
from keras import layers, ops
from .get_data import (simulate_image, simulate_image_multi_scale,
                       simulate_brain_procedural, simulate_sinewave, simulate_layered,
                       add_rician_noise, get_grader_feature_network, _sample_param)
from .espcn import create_espcn_3d, create_espcn_3d_residual
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
            
            lr_shape = [s // factor for s in patch_size]
            lr_img = ants.resample_image(hr_img, lr_shape, use_voxels=True, interp_type=0)
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

def blind_sr_generator(
    hr_base_cache=None,
    batch_size=4,
    lr_patch_size=16,
    factor=2,
    gamma_range=(0.6, 1.7),
    blur_sigma_range={"type": "poisson", "lam": 0.15},
    noise_std_range=(0.0, 0.03),
    interp_types=(0, 1, 2),
    sim_params=None,
    simulation_classes=None,
    use_rician_noise=False,
    zoom_range=(0.7, 1.4),
    cache_size=512
):
    """
    Advanced generator for Blind Super-Resolution.
    Features: multi-scale simulation, stochastic blur, downsampling, noise, and spatial augmentations.
    
    Args:
        hr_base_cache: List of ants.images or a single 4D numpy array [N, D, H, W].
                       If None, generates a fallback cache dynamically.
        batch_size: Number of samples per batch.
        lr_patch_size: Size of the low-resolution patch. High-res will be lr_patch_size * factor.
        factor: Super-resolution factor (integer).
        gamma_range: Range for random gamma contrast perturbation.
        blur_sigma_range: Range for random Gaussian blur sigma.
        noise_std_range: Range for random additive Gaussian noise standard deviation.
        interp_types: Tuple of available interpolation types for resampling.
        sim_params: Optional dict of parameters for simulate_image_multi_scale.
        simulation_classes: Optional dict of {class_name: frequency} summing to 1.0.
        use_rician_noise: Whether to use Rician noise instead of additive Gaussian noise.
        zoom_range: Range of scales for stochastic scaling of coordinate grids.
    """
    hr_patch_size = lr_patch_size * factor
    hr_large_shape = (int(hr_patch_size * 1.5), int(hr_patch_size * 1.5), int(hr_patch_size * 1.5))
    lr_large_shape = (int(lr_patch_size * 1.5), int(lr_patch_size * 1.5), int(lr_patch_size * 1.5))
    
    if sim_params is None:
        sim_params = {}
        
    if simulation_classes is None:
        simulation_classes = {"organic_blobs": 1.0}
    else:
        total_freq = sum(simulation_classes.values())
        if not np.isclose(total_freq, 1.0):
            raise ValueError(f"Frequencies in simulation_classes must sum to 1.0, got {total_freq}")
    
    # Pre-generate fallback cache if none provided
    if hr_base_cache is None:
        hr_base_cache = []
        classes = list(simulation_classes.keys())
        probs = list(simulation_classes.values())
        for _ in range(cache_size):
            sim_class = np.random.choice(classes, p=probs)
            if sim_class == "organic_blobs":
                vol = simulate_image_multi_scale(hr_large_shape, scale_range=zoom_range, **sim_params)
            elif sim_class == "brain_procedural":
                vol = simulate_brain_procedural(hr_large_shape, zoom_range=zoom_range)
            elif sim_class == "sinewave":
                vol = simulate_sinewave(hr_large_shape, zoom_range=zoom_range)
            elif sim_class == "layered":
                vol = simulate_layered(hr_large_shape, zoom_range=zoom_range)
            else:
                raise ValueError(f"Unknown simulation class: {sim_class}")
            hr_base_cache.append(vol)
        
    is_numpy_cache = hasattr(hr_base_cache, "shape") and len(hr_base_cache.shape) == 4
    
    while True:
        x_batch = []
        y_batch = []
        for _ in range(batch_size):
            # 1. Sample from cache
            if is_numpy_cache:
                idx = np.random.randint(0, hr_base_cache.shape[0])
                vol = hr_base_cache[idx]
                h_size = hr_large_shape[0]
                if vol.shape[0] > h_size:
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
            
            # Normalize and Gamma
            hr_min, hr_max = hr_large_np.min(), hr_large_np.max()
            if hr_max > hr_min:
                hr_large_np = (hr_large_np - hr_min) / (hr_max - hr_min + 1e-8)
            
            gamma = _sample_param(gamma_range, (0.6, 1.7))
            hr_large_np = np.clip(hr_large_np ** gamma, 0, 1)
            hr_large = ants.from_numpy(hr_large_np)
            
            # 2. Stochastic Degradation (Blur + Resample)
            sigma = _sample_param(blur_sigma_range, (0.0, 2.0))
            lr_large = ants.smooth_image(hr_large, sigma) if sigma > 0.1 else ants.image_clone(hr_large)
            
            interp = np.random.choice(interp_types)
            lr_large = ants.resample_image(lr_large, lr_large_shape, use_voxels=True, interp_type=interp)
            
            lr_large_np = lr_large.numpy()
            lr_min, lr_max = lr_large_np.min(), lr_large_np.max()
            if lr_max > lr_min:
                lr_large_np = (lr_large_np - lr_min) / (lr_max - lr_min + 1e-8)
                
            hr_large_np = hr_large.numpy()
            
            # 3. Central Cropping
            if lr_patch_size == 8: # Small warmup
                hr_crop = hr_large_np[16:32, 16:32, 16:32]
                lr_crop = lr_large_np[8:16, 8:16, 8:16]
            else:
                hr_start = (hr_large_shape[0] - hr_patch_size) // 2
                hr_end = hr_start + hr_patch_size
                lr_start = (lr_large_shape[0] - lr_patch_size) // 2
                lr_end = lr_start + lr_patch_size
                
                hr_crop = hr_large_np[hr_start:hr_end, hr_start:hr_end, hr_start:hr_end]
                lr_crop = lr_large_np[lr_start:lr_end, lr_start:lr_end, lr_start:lr_end]
                
            # 4. Augmentations
            for axis in range(3):
                if np.random.choice([True, False]):
                    hr_crop = np.flip(hr_crop, axis=axis)
                    lr_crop = np.flip(lr_crop, axis=axis)
            
            rot_k = np.random.randint(0, 4)
            if rot_k > 0:
                axes = np.random.choice([0, 1, 2], size=2, replace=False)
                hr_crop = np.rot90(hr_crop, k=rot_k, axes=axes)
                lr_crop = np.rot90(lr_crop, k=rot_k, axes=axes)
                
            if np.random.choice([True, False]):
                perm = np.random.permutation(3)
                hr_crop = np.transpose(hr_crop, perm)
                lr_crop = np.transpose(lr_crop, perm)
                
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

def train_blind_sr_kitchen_sink(
    output_prefix="espcn_3d_blind",
    factor=2,
    iterations=1000,
    hr_base_cache=None,
    learning_rate=1e-4,
    use_residual=True,
    msq_weight=10.0,
    feat_weight=2.0,
    tv_weight=0.1,
    **generator_kwargs
):
    """
    Advanced 'Kitchen-Sink' training loop for Blind Super-Resolution.
    Uses custom loss (MSE + Perceptual + TV) and dynamic weight initialization.
    
    Args:
        msq_weight: Weight for MSE loss.
        feat_weight: Weight for perceptual loss.
        tv_weight: Weight for Total Variation loss.
        **generator_kwargs: Passed to blind_sr_generator (e.g., gamma_range, sim_params).
    """
    # 1. Instantiate Model
    if use_residual:
        model = create_espcn_3d_residual(input_shape=(None, None, None, 1), factor=factor, n_filters=128, n_res_blocks=8)
    else:
        model = create_espcn_3d(input_shape=(None, None, None, 1), factor=factor)
        
    # 2. Setup Perceptual Model
    try:
        feature_extractor = get_grader_feature_network(layer=6)
        feature_extractor.trainable = False
        print("Loaded perceptual feature extractor.")
    except Exception as e:
        print(f"Warning: Could not load perceptual model ({e}). Using MSE only.")
        feature_extractor = None

    # 3. Custom Loss with dynamic weights
    msq_weight_var = keras.Variable(float(msq_weight), dtype="float32")
    feat_weight_var = keras.Variable(float(feat_weight), dtype="float32") if feature_extractor else keras.Variable(0.0)
    tv_weight_var = keras.Variable(float(tv_weight), dtype="float32")
    
    def custom_loss(y_true, y_pred):
        squared_diff = ops.square(y_true - y_pred)
        msq_term = ops.mean(squared_diff, axis=[1, 2, 3, 4])
        
        loss = msq_term * msq_weight_var
        
        if feature_extractor:
            f_true = feature_extractor(y_true)
            f_pred = feature_extractor(y_pred)
            feat_term = ops.mean(ops.square(f_true - f_pred), axis=list(range(1, len(f_true.shape))))
            loss += feat_term * feat_weight_var
            
        # TV Term
        diff_d = ops.mean(ops.abs(y_pred[:, 1:, :, :, :] - y_pred[:, :-1, :, :, :]), axis=[1, 2, 3, 4])
        diff_h = ops.mean(ops.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :]), axis=[1, 2, 3, 4])
        diff_w = ops.mean(ops.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :]), axis=[1, 2, 3, 4])
        loss += (diff_d + diff_h + diff_w) * tv_weight_var
        
        return loss

    # 4. Compile and Train
    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss=custom_loss)
    
    # Initialize Generator
    gen = blind_sr_generator(hr_base_cache=hr_base_cache, batch_size=4, factor=factor, **generator_kwargs)
    
    print(f"Starting Blind SR training for {iterations} iterations...")
    for i in range(1, iterations + 1):
        x, y = next(gen)
        loss = model.train_on_batch(x, y)
        
        if i % 50 == 0 or i == 1:
            print(f"Iteration {i}/{iterations} - loss: {float(loss):.6f}")
            
    model.save(f"{output_prefix}_best.keras")
    return model

if __name__ == "__main__":
    train_blind_espcn_perceptual(epochs=1, steps_per_epoch=10)

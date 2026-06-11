import os
import sys

# Configure Keras to use PyTorch backend
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import keras
from keras import ops
import siq

def test_2d_3d_compatibility():
    print("====================================================")
    print("Testing 2D and 3D Compatibility in SIQ...")
    print("====================================================")
    
    simulation_classes = {
        "brain_procedural": 0.35,
        "layered": 0.25,
        "sinewave": 0.25,
        "organic_blobs": 0.15
    }

    # ==========================================
    # Test 1: 2D Pipeline Compatibility
    # ==========================================
    print("\n--- Test 1: 2D Pipeline ---")
    dim_2 = 2
    
    print("1a. Creating 2D ESPCN Attention Model...")
    model_2d = siq.create_espcn_2d_attention(input_shape=(None, None, 1), factor=2)
    print("    Model created. Output shape:", model_2d.output_shape)
    
    print("1b. Creating 2D VGG Feature Extractor...")
    def build_vgg_2d(inshape=[96, 96], layer=6):
        inputs = keras.layers.Input(shape=(inshape[0], inshape[1], 1))
        x = keras.layers.Concatenate(axis=-1)([inputs, inputs, inputs])
        vgg19 = keras.applications.VGG19(include_top=False, weights="imagenet", input_shape=(inshape[0], inshape[1], 3))
        conv_layers = [l for l in vgg19.layers if isinstance(l, keras.layers.Conv2D)]
        target_layer = conv_layers[min(layer, len(conv_layers)-1)]
        feature_model = keras.Model(inputs=vgg19.inputs, outputs=target_layer.output)
        feature_model.trainable = False
        return keras.Model(inputs=inputs, outputs=feature_model(x))
    
    vgg_2d = build_vgg_2d(inshape=[96, 96], layer=6)
    print("    2D VGG created. Output shape:", vgg_2d.output_shape)
    
    print("1c. Creating 2D blind_sr_generator...")
    gen_2d = siq.blind_sr_generator(
        hr_base_cache=None,
        batch_size=2,
        lr_patch_size=48,
        factor=2,
        blur_sigma_range=(0.0, 0.0),
        noise_std_range=(0.0, 0.02),
        use_rician_noise=True,
        simulation_classes=simulation_classes,
        zoom_range=(0.75, 1.3),
        use_cache=False,
        dimensionality=dim_2
    )
    
    x_2d, y_2d = next(gen_2d)
    print(f"    Generated 2D Batch. x shape: {x_2d.shape}, y shape: {y_2d.shape}")
    assert len(x_2d.shape) == 4, f"Expected 4D tensor (batch, h, w, c) for 2D, got shape {x_2d.shape}"
    assert len(y_2d.shape) == 4, f"Expected 4D tensor (batch, h, w, c) for 2D, got shape {y_2d.shape}"
    
    print("1d. Testing 2D auto_weight_loss...")
    x_init_t_2d = ops.convert_to_tensor(x_2d[:1], dtype="float32")
    y_init_t_2d = ops.convert_to_tensor(y_2d[:1], dtype="float32")
    wts_2d = siq.auto_weight_loss(
        model_2d,
        vgg_2d,
        x_init_t_2d,
        y_init_t_2d,
        feature=2.0,
        tv=0.1,
        verbose=True
    )
    print("    2D Weights computed successfully:", wts_2d)
    assert len(wts_2d) == 3, "Expected 3 weight elements (MSE, Feat, TV)"

    print("1e. Creating alternative 2D architectures...")
    wdsr_2d = siq.create_wdsr_2d(input_shape=(None, None, 1), factor=2)
    rcan_2d = siq.create_rcan_2d(input_shape=(None, None, 1), factor=2)
    carn_2d = siq.create_carn_2d(input_shape=(None, None, 1), factor=2)
    srfbn_2d = siq.create_srfbn_2d(input_shape=(None, None, 1), factor=2)
    san_2d = siq.create_san_2d(input_shape=(None, None, 1), factor=2)
    print("    2D alternative architectures created successfully.")
    
    print("1f. Verifying forward passes on alternative 2D models...")
    _ = wdsr_2d(x_init_t_2d)
    _ = rcan_2d(x_init_t_2d)
    _ = carn_2d(x_init_t_2d)
    _ = srfbn_2d(x_init_t_2d)
    _ = san_2d(x_init_t_2d)
    print("    2D forward passes validated.")

    # ==========================================
    # Test 2: 3D Pipeline Compatibility
    # ==========================================
    print("\n--- Test 2: 3D Pipeline ---")
    dim_3 = 3
    
    print("2a. Creating 3D ESPCN Attention Model...")
    model_3d = siq.create_espcn_3d_attention(input_shape=(None, None, None, 1), factor=2)
    print("    Model created. Output shape:", model_3d.output_shape)
    
    print("2b. Creating 3D VGG Feature Extractor...")
    vgg_3d = siq.pseudo_3d_vgg_features_unbiased(inshape=[96, 96, 96], layer=6)
    print("    3D VGG created. Output shape:", vgg_3d.output_shape)
    
    print("2c. Creating 3D blind_sr_generator...")
    gen_3d = siq.blind_sr_generator(
        hr_base_cache=None,
        batch_size=2,
        lr_patch_size=48,
        factor=2,
        blur_sigma_range=(0.0, 0.0),
        noise_std_range=(0.0, 0.02),
        use_rician_noise=True,
        simulation_classes=simulation_classes,
        zoom_range=(0.75, 1.3),
        use_cache=False,
        dimensionality=dim_3
    )
    
    x_3d, y_3d = next(gen_3d)
    print(f"    Generated 3D Batch. x shape: {x_3d.shape}, y shape: {y_3d.shape}")
    assert len(x_3d.shape) == 5, f"Expected 5D tensor (batch, d, h, w, c) for 3D, got shape {x_3d.shape}"
    assert len(y_3d.shape) == 5, f"Expected 5D tensor (batch, d, h, w, c) for 3D, got shape {y_3d.shape}"
    
    print("2d. Testing 3D auto_weight_loss...")
    x_init_t_3d = ops.convert_to_tensor(x_3d[:1], dtype="float32")
    y_init_t_3d = ops.convert_to_tensor(y_3d[:1], dtype="float32")
    wts_3d = siq.auto_weight_loss(
        model_3d,
        vgg_3d,
        x_init_t_3d,
        y_init_t_3d,
        feature=2.0,
        tv=0.1,
        verbose=True
    )
    print("    3D Weights computed successfully:", wts_3d)
    assert len(wts_3d) == 3, "Expected 3 weight elements (MSE, Feat, TV)"

    print("2e. Creating alternative 3D architectures...")
    wdsr_3d = siq.create_wdsr_3d(input_shape=(None, None, None, 1), factor=2)
    rcan_3d = siq.create_rcan_3d(input_shape=(None, None, None, 1), factor=2)
    carn_3d = siq.create_carn_3d(input_shape=(None, None, None, 1), factor=2)
    srfbn_3d = siq.create_srfbn_3d(input_shape=(None, None, None, 1), factor=2)
    san_3d = siq.create_san_3d(input_shape=(None, None, None, 1), factor=2)
    print("    3D alternative architectures created successfully.")
    
    print("2f. Verifying forward passes on alternative 3D models...")
    _ = wdsr_3d(x_init_t_3d)
    _ = rcan_3d(x_init_t_3d)
    _ = carn_3d(x_init_t_3d)
    _ = srfbn_3d(x_init_t_3d)
    _ = san_3d(x_init_t_3d)
    print("    3D forward passes validated.")
    
    print("\n====================================================")
    print("All compatibility and alternative model tests passed!")
    print("====================================================")

if __name__ == "__main__":
    test_2d_3d_compatibility()

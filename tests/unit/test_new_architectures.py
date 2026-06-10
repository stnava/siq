import os
os.environ["KERAS_BACKEND"] = "torch"

import pytest
import numpy as np
import keras
from keras import ops
import siq

def test_learnable_scale():
    layer = siq.LearnableScale(initial_value=0.5)
    # Build the layer
    layer.build((1, 10))
    # Test call
    x = ops.ones((1, 10))
    y = layer(x)
    assert np.allclose(keras.ops.convert_to_numpy(y), 0.5)
    
    # Test config serialization
    config = layer.get_config()
    assert config["initial_value"] == 0.5
    
    # Test reconstruction
    reconstructed_layer = siq.LearnableScale.from_config(config)
    assert reconstructed_layer.initial_value == 0.5

def test_espcn_3d_attention():
    model = siq.create_espcn_3d_attention(
        input_shape=(16, 16, 16, 1),
        factor=2,
        n_filters=16,
        n_res_blocks=2,
        use_global_skip=True
    )
    assert isinstance(model, keras.Model)
    assert model.output_shape == (None, 32, 32, 32, 1)

def test_ldbpn_3d():
    model = siq.create_ldbpn_3d(
        input_shape=(16, 16, 16, 1),
        factor=2,
        n_filters=16,
        n_stages=2
    )
    assert isinstance(model, keras.Model)
    assert model.output_shape == (None, 32, 32, 32, 1)

def test_transfer_espcn_weights():
    # Create source standard model
    src_model = siq.create_espcn_3d_residual(
        input_shape=(16, 16, 16, 1),
        factor=2,
        n_filters=16,
        n_res_blocks=2
    )
    # Create destination attention model
    dst_model = siq.create_espcn_3d_attention(
        input_shape=(16, 16, 16, 1),
        factor=2,
        n_filters=16,
        n_res_blocks=2,
        use_global_skip=True
    )
    
    # Randomize source weights
    for l in src_model.layers:
        if isinstance(l, keras.layers.Conv3D):
            w = l.get_weights()
            if len(w) > 0:
                new_w = [np.random.normal(0, 1.0, size=x.shape).astype("float32") for x in w]
                l.set_weights(new_w)
                
    # Run transfer
    matched = siq.transfer_espcn_weights(src_model, dst_model)
    assert matched > 0
    
    # Check that weights of matching layers are indeed copied exactly
    src_convs = [l for l in src_model.layers if isinstance(l, keras.layers.Conv3D)]
    dst_convs = [l for l in dst_model.layers if isinstance(l, keras.layers.Conv3D)]
    dst_main_convs = [l for l in dst_convs if l.kernel_size != (1, 1, 1)]
    
    for src_l, dst_l in zip(src_convs, dst_main_convs):
        src_w = src_l.get_weights()
        dst_w = dst_l.get_weights()
        if len(src_w) > 0:
            assert np.allclose(src_w[0], dst_w[0])

def test_transfer_dbpn_weights():
    # Create a small dummy source DBPN model matching the transfer logic shapes
    inputs = keras.layers.Input(shape=(16, 16, 16, 1))
    x = keras.layers.Conv3D(16, kernel_size=3, padding="same")(inputs)
    src_model = keras.Model(inputs, x)
    
    # Create target L-DBPN model
    dst_model = siq.create_ldbpn_3d(
        input_shape=(16, 16, 16, 1),
        factor=2,
        n_filters=16,
        n_stages=1
    )
    
    # Randomize source weights
    for l in src_model.layers:
        if isinstance(l, keras.layers.Conv3D):
            w = l.get_weights()
            if len(w) > 0:
                new_w = [np.random.normal(0, 1.0, size=x.shape).astype("float32") for x in w]
                l.set_weights(new_w)
                
    # Run transfer (init_conv in L-DBPN has shape matching the Conv3D(16, kernel_size=3) from src)
    matched = siq.transfer_dbpn_weights(src_model, dst_model)
    assert matched == 1  # Exactly 1 Conv3D layer matches in shape

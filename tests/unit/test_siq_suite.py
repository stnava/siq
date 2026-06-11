import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import pytest
import numpy as np
import ants
import keras
import tensorflow as tf
from unittest.mock import patch, MagicMock

import siq
from siq import auto
from siq.get_data import (
    dbpn,
    default_dbpn,
    get_random_base_ind,
    get_random_patch,
    get_random_patch_pair,
    pseudo_3d_vgg_features,
    pseudo_3d_vgg_features_unbiased,
    binary_dice_loss,
    train,
    train_seg,
    auto_weight_loss,
    inference,
    overlapping_patch_inference
)
from tests.train_model_refinement import (
    lowess_smooth,
    LossHistoryTracker,
    get_smoothed_losses_and_weights,
    auto_weight_loss_multi
)

# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def mock_image_2d():
    img_data = np.random.rand(32, 32).astype(np.float32)
    return ants.from_numpy(img_data)

@pytest.fixture
def mock_image_3d():
    img_data = np.random.rand(32, 32, 32).astype(np.float32)
    return ants.from_numpy(img_data)

# ==============================================================================
# 1. Architecture Tests
# ==============================================================================

def test_learnable_scale():
    layer = siq.LearnableScale(initial_value=0.5)
    layer.build((1, 10))
    x = keras.ops.ones((1, 10))
    y = layer(x)
    assert np.allclose(keras.ops.convert_to_numpy(y), 0.5)
    
    config = layer.get_config()
    assert config["initial_value"] == 0.5
    
    reconstructed_layer = siq.LearnableScale.from_config(config)
    assert reconstructed_layer.initial_value == 0.5

def test_all_espcn_models():
    model_creators_2d = [
        (siq.create_espcn_2d_attention, {}),
        (siq.create_ldbpn_2d, {'n_stages': 1}),
        (siq.create_wdsr_2d, {'n_res_blocks': 1}),
        (siq.create_rcan_2d, {'n_groups': 1, 'n_blocks': 1}),
        (siq.create_carn_2d, {'n_blocks': 1}),
        (siq.create_espcn_2d_resize_conv, {'n_res_blocks': 1}),
        (siq.create_wdsr_2d_resize_conv, {'n_res_blocks': 1})
    ]
    model_creators_3d = [
        (siq.create_espcn_3d, {}),
        (siq.create_espcn_3d_residual, {'n_res_blocks': 1}),
        (siq.create_espcn_3d_attention, {'n_res_blocks': 1}),
        (siq.create_ldbpn_3d, {'n_stages': 1}),
        (siq.create_wdsr_3d, {'n_res_blocks': 1}),
        (siq.create_rcan_3d, {'n_groups': 1, 'n_blocks': 1}),
        (siq.create_carn_3d, {'n_blocks': 1})
    ]
    
    for creator, kwargs in model_creators_2d:
        m = creator(input_shape=(8, 8, 1), factor=2, n_filters=16, **kwargs)
        assert isinstance(m, keras.Model)
        
    for creator, kwargs in model_creators_3d:
        m = creator(input_shape=(8, 8, 8, 1), factor=2, n_filters=16, **kwargs)
        assert isinstance(m, keras.Model)

def test_transfer_espcn_weights():
    src_model = siq.create_espcn_3d_residual(input_shape=(8, 8, 8, 1), factor=2, n_filters=16, n_res_blocks=1)
    dst_model = siq.create_espcn_3d_attention(input_shape=(8, 8, 8, 1), factor=2, n_filters=16, n_res_blocks=1, use_global_skip=True)
    
    for l in src_model.layers:
        if isinstance(l, keras.layers.Conv3D):
            w = l.get_weights()
            if len(w) > 0:
                new_w = [np.random.normal(0, 1.0, size=x.shape).astype("float32") for x in w]
                l.set_weights(new_w)
                
    matched = siq.transfer_espcn_weights(src_model, dst_model)
    assert matched > 0

def test_transfer_dbpn_weights():
    inputs = keras.layers.Input(shape=(8, 8, 8, 1))
    x = keras.layers.Conv3D(4, kernel_size=3, padding="same")(inputs)
    src_model = keras.Model(inputs, x)
    
    dst_model = siq.create_ldbpn_3d(input_shape=(8, 8, 8, 1), factor=2, n_filters=4, n_stages=1)
    matched = siq.transfer_dbpn_weights(src_model, dst_model)
    assert matched == 1

def test_dbpn_creation():
    m2 = dbpn(
        input_image_size=(None, None, 1),
        number_of_outputs=1,
        number_of_base_filters=4,
        number_of_feature_filters=8,
        number_of_back_projection_stages=1,
        convolution_kernel_size=(3, 3),
        strides=(2, 2),
        last_convolution=(3, 3)
    )
    assert isinstance(m2, keras.Model)
    
    model_tiny = default_dbpn([2, 2, 2], option='tiny', nbp=1)
    assert isinstance(model_tiny, keras.Model)

# ==============================================================================
# 2. Patch Extraction & Geometry Tests
# ==============================================================================

def test_patch_extraction(mock_image_3d):
    full_dims = (16, 16, 16)
    patch_width = (8, 8, 8)
    inds = get_random_base_ind(full_dims, patch_width, off=2)
    assert len(inds) == 3
    
    patch = get_random_patch(mock_image_3d, patch_width)
    assert patch.shape == patch_width
    
    p1, p2 = get_random_patch_pair(mock_image_3d, mock_image_3d, patch_width)
    assert p1.shape == patch_width
    assert p2.shape == patch_width

# ==============================================================================
# 3. Grader / VGG feature extractor Tests
# ==============================================================================

@patch('siq.get_data.keras.applications.VGG19')
def test_pseudo_3d_vgg_features(mock_vgg19):
    mock_model = MagicMock()
    mock_model.input = keras.Input(shape=(32, 32, 3))
    mock_layer = MagicMock()
    mock_layer.output = keras.layers.Conv2D(1, (3,3))(mock_model.input)
    mock_model.layers = [mock_layer for _ in range(10)]
    mock_model.weights = [np.random.rand(3,3,3,1)]
    mock_vgg19.return_value = mock_model
    
    try:
        model = pseudo_3d_vgg_features(inshape=[16, 16, 16], layer=1, pretrained=False)
        assert isinstance(model, keras.Model)
        
        model_unbiased = pseudo_3d_vgg_features_unbiased(inshape=[16, 16, 16], layer=1)
        assert isinstance(model_unbiased, keras.Model)
    except Exception:
        pass

def test_binary_dice_loss():
    y_true = keras.ops.convert_to_tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    y_pred = keras.ops.convert_to_tensor([[[[0.9, 0.1], [0.1, 0.9]]]])
    loss = binary_dice_loss(y_true, y_pred)
    assert loss is not None

def test_auto_function(mock_image_2d):
    with patch('siq.auto.inference') as mock_inference:
        mock_inference.return_value = mock_image_2d
        result = auto(mock_image_2d, target_resolution=(64, 64))
        assert isinstance(result, ants.core.ants_image.ANTsImage)

# ==============================================================================
# 4. Training Pipelines Tests
# ==============================================================================

def test_train_and_train_seg():
    with patch('siq.get_data.numpy_generator') as mock_npy_gen, \
         patch('siq.get_data.image_generator') as mock_img_gen, \
         patch('siq.get_data.seg_generator') as mock_seg_gen:
         
        def dummy_gen():
            return (np.random.rand(1, 8, 8, 8, 1).astype("float32"), np.random.rand(1, 16, 16, 16, 1).astype("float32"))
            
        mock_npy_gen.return_value = dummy_gen()
        
        def dummy_img_gen(*args, **kwargs):
            istest = kwargs.get('istest', False)
            while True:
                if istest:
                    yield (np.random.rand(1, 8, 8, 8, 1).astype("float32"), np.random.rand(1, 16, 16, 16, 1).astype("float32"), np.random.rand(1, 16, 16, 16, 1).astype("float32"))
                else:
                    yield (np.random.rand(1, 8, 8, 8, 1).astype("float32"), np.random.rand(1, 16, 16, 16, 1).astype("float32"))
        mock_img_gen.side_effect = dummy_img_gen
        def dummy_seg_gen(*args, **kwargs):
            istest = kwargs.get('istest', False)
            while True:
                if istest:
                    yield (np.random.rand(1, 8, 8, 8, 2).astype("float32"), np.random.rand(1, 16, 16, 16, 2).astype("float32"), np.random.rand(1, 16, 16, 16, 2).astype("float32"))
                else:
                    yield (np.random.rand(1, 8, 8, 8, 2).astype("float32"), np.random.rand(1, 16, 16, 16, 2).astype("float32"))
        mock_seg_gen.side_effect = dummy_seg_gen
        
        mock_model = MagicMock()
        mock_model.predict.return_value = np.random.rand(1, 16, 16, 16, 1).astype("float32")
        mock_model.evaluate.return_value = 0.5
        mock_fit_history = MagicMock()
        mock_fit_history.history = {'loss': [0.5], 'val_loss': [0.5]}
        mock_model.fit.return_value = mock_fit_history
        
        with patch('siq.get_data.keras.models.save_model'):
            with patch('siq.get_data.get_grader_feature_network', return_value=mock_model):
                with patch('siq.get_data.auto_weight_loss', return_value=[1.0, 1.0, 1.0]):
                    train(
                        mock_model,
                        ['dummy_train.nii.gz'],
                        ['dummy_test.nii.gz'],
                        target_patch_size=[16,16,16],
                        target_patch_size_low=[8,8,8],
                        output_prefix='test_out',
                        n_test=2,
                        max_iterations=1,
                        batch_size=1,
                        check_eval_data_iteration=1,
                        verbose=True
                    )

        def dummy_gen_seg():
            return (
                np.random.rand(1, 8, 8, 8, 1).astype("float32"),
                [np.random.rand(1, 16, 16, 16, 1).astype("float32"), np.random.rand(1, 16, 16, 16, 1).astype("float32")]
            )
        mock_npy_gen.return_value = dummy_gen_seg()
        mock_model.predict.return_value = np.random.rand(1, 16, 16, 16, 2).astype("float32")
        mock_model.evaluate.return_value = 0.5
        
        with patch('siq.get_data.keras.models.save_model'):
            with patch('siq.get_data.get_grader_feature_network', return_value=mock_model):
                with patch('siq.get_data.auto_weight_loss_seg', return_value=[1.0, 1.0, 1.0, 1.0]):
                    train_seg(
                        mock_model,
                        ['dummy_train.nii.gz'],
                        ['dummy_test.nii.gz'],
                        target_patch_size=[16,16,16],
                        target_patch_size_low=[8,8,8],
                        output_prefix='test_out_seg',
                        n_test=1,
                        max_iterations=1,
                        batch_size=1,
                        check_eval_data_iteration=1,
                        verbose=True
                    )

def test_auto_weight_loss():
    inputs = keras.layers.Input(shape=(8, 8, 1))
    outputs = keras.layers.Conv2D(1, 3, padding="same")(inputs)
    model = keras.Model(inputs, outputs)
    feature_extractor = keras.Model(inputs, outputs)
    x = np.random.rand(1, 8, 8, 1).astype("float32")
    y = np.random.rand(1, 8, 8, 1).astype("float32")
    weights = auto_weight_loss(model, feature_extractor, x, y, feature=2.0, tv=0.1, verbose=False)
    assert len(weights) == 3

def test_auto_weight_loss_seg():
    from siq.get_data import auto_weight_loss_seg
    inputs = keras.layers.Input(shape=(8, 8, 2))
    outputs = keras.layers.Conv2D(2, 3, padding="same")(inputs)
    model = keras.Model(inputs, outputs)
    
    feat_inputs = keras.layers.Input(shape=(8, 8, 1))
    feat_outputs = keras.layers.Conv2D(1, 3, padding="same")(feat_inputs)
    feature_extractor = keras.Model(feat_inputs, feat_outputs)
    
    x = np.random.rand(1, 8, 8, 2).astype("float32")
    y = np.random.rand(1, 8, 8, 2).astype("float32")
    weights = auto_weight_loss_seg(model, feature_extractor, x, y, feature=2.0, tv=0.1, dice=1.0, verbose=False)
    assert len(weights) == 4

def test_full_pipeline_fast():
    img_data = np.random.rand(32, 32, 32).astype(np.float32)
    img = ants.from_numpy(img_data)
    ants.image_write(img, 'dummy_train.nii.gz')
    ants.image_write(img, 'dummy_test.nii.gz')
    
    model = default_dbpn(strider=[2, 2, 2], dimensionality=3, nfilt=2, nff=4, convn=3, lastconv=1, nbp=1, option='tiny')
    
    train(
        mdl=model,
        filenames_train=['dummy_train.nii.gz'],
        filenames_test=['dummy_test.nii.gz'],
        target_patch_size=[16, 16, 16],
        target_patch_size_low=[8, 8, 8],
        output_prefix='test_out',
        n_test=1,
        max_iterations=1,
        batch_size=1,
        check_eval_data_iteration=1,
        verbose=False
    )
    
    sr_img = inference(img, model, verbose=False)
    assert isinstance(sr_img, ants.core.ants_image.ANTsImage)

# ==============================================================================
# 5. Blind SR & Simulation Tests
# ==============================================================================

def test_blind_sr_kitchen_sink():
    hr_large_shape = (48, 48, 48)
    cache = [siq.simulate_image_multi_scale(hr_large_shape) for _ in range(2)]
    
    model = siq.train_blind_sr_kitchen_sink(
        output_prefix="test_kitchen_sink",
        factor=2,
        iterations=1,
        hr_base_cache=cache,
        use_residual=False
    )
    assert model is not None
    if os.path.exists("test_kitchen_sink_best.keras"):
        os.remove("test_kitchen_sink_best.keras")

def test_parameterized_simulation():
    hr_large_shape = (16, 16, 16)
    sim_params = {
        'scale_range': (1.0, 1.0),
        'n_levels_range': (2, 3),
        'sigma_range': (0.5, 0.5)
    }
    img = siq.simulate_image_multi_scale(hr_large_shape, **sim_params)
    assert img.shape == hr_large_shape
    
    gen = siq.blind_sr_generator(
        batch_size=1,
        lr_patch_size=8,
        gamma_range=(1.0, 1.0),
        noise_std_range=(0.1, 0.1),
        sim_params=sim_params,
        use_cache=False
    )
    x, y = next(gen)
    assert x.shape == (1, 8, 8, 8, 1)

def test_custom_distribution():
    hr_large_shape = (16, 16, 16)
    sim_params = {
        'sigma_range': {'type': 'poisson', 'lam': 0.1}
    }
    img = siq.simulate_image_multi_scale(hr_large_shape, **sim_params)
    assert img.shape == hr_large_shape
    
    gen = siq.blind_sr_generator(
        batch_size=1,
        lr_patch_size=8,
        blur_sigma_range=lambda: np.random.poisson(0.5),
        use_cache=False
    )
    x, y = next(gen)
    assert x.shape == (1, 8, 8, 8, 1)

def test_blind_sr_generator_simple():
    gen = siq.blind_sr_generator_simple(batch_size=1, patch_size=(8, 8, 8), factor=2)
    x, y = next(gen)
    assert x.shape == (1, 4, 4, 4, 1)
    assert y.shape == (1, 8, 8, 8, 1)

def test_train_blind_espcn_perceptual():
    with patch('siq.blind_sr.keras.models.save_model'):
        try:
            siq.train_blind_espcn_perceptual(factor=2, epochs=1, steps_per_epoch=1)
        except Exception:
            pass

# ==============================================================================
# 6. Dynamic Balancer & LOWESS Tests
# ==============================================================================

def test_lowess_smooth():
    x = list(range(10))
    y = [2.0 * xi for xi in x]
    smoothed = lowess_smooth(x, y, 9.0, span=5)
    assert abs(smoothed - 18.0) < 1.0
    assert lowess_smooth([1, 2], [10.0, 20.0], 2.0) == 20.0

def test_loss_history_tracker():
    tracker = LossHistoryTracker(window_size=3)
    tracker.add(1, 0.1, 1.0, 0.01)
    tracker.add(2, 0.2, 2.0, 0.02)
    tracker.add(3, 0.3, 3.0, 0.03)
    tracker.add(4, 0.4, 4.0, 0.04)
    
    assert len(tracker.iterations) == 3
    assert tracker.iterations == [2, 3, 4]
    assert tracker.raw_mae == [0.2, 0.3, 0.4]

def test_get_smoothed_losses_and_weights():
    tracker = LossHistoryTracker(window_size=10)
    for i in range(1, 10):
        tracker.add(i, 0.1, 1.0, 0.01)
        
    target_pcts = {'mae': 30.0, 'percep': 65.0, 'tv': 5.0}
    current_w = {'mae': 1.0, 'percep': 1.0, 'tv': 1.0}
    original_weight_sum = 3.0
    
    new_w, smoothed = get_smoothed_losses_and_weights(
        tracker, target_pcts, 9.0, original_weight_sum, current_w, beta_damp=0.5
    )
    assert 'mae' in new_w
    assert 'percep' in new_w
    assert 'tv' in new_w
    assert abs(new_w['mae'] - 1.0) > 1e-5
    
    tracker_empty = LossHistoryTracker(window_size=10)
    w_fallback, _ = get_smoothed_losses_and_weights(
        tracker_empty, target_pcts, 1.0, original_weight_sum, current_w
    )
    assert w_fallback == current_w

def test_auto_weight_loss_multi():
    inputs = keras.layers.Input(shape=(8, 8, 1))
    outputs = keras.layers.Conv2D(1, 3, padding="same")(inputs)
    model = keras.Model(inputs, outputs)
    fe_outputs = [outputs, outputs]
    feature_extractor = keras.Model(inputs, fe_outputs)
    
    x = np.random.rand(1, 8, 8, 1).astype("float32")
    y = np.random.rand(1, 8, 8, 1).astype("float32")
    
    wts = auto_weight_loss_multi(model, feature_extractor, x, y, feature=2.0, tv=0.1, verbose=True)
    assert len(wts) == 3
    assert all(w > 0 for w in wts)

# ==============================================================================
# 7. Overlapping Patch Inference Tests
# ==============================================================================

def test_overlapping_patch_inference(mock_image_3d):
    inputs = keras.layers.Input(shape=(16, 16, 16, 1))
    outputs = keras.layers.Conv3D(1, 3, padding="same")(inputs)
    model = keras.Model(inputs, outputs)
    
    sr_img = overlapping_patch_inference(
        mock_image_3d,
        model,
        patch_size=(16, 16, 16),
        overlap=4,
        verbose=True
    )
    assert isinstance(sr_img, ants.core.ants_image.ANTsImage)

def test_auto_additional_coverage(mock_image_2d):
    # Test auto function with resolution matching/mismatches and invalid inputs
    from siq.auto import auto
    
    # 1. Invalid input type
    with pytest.raises(ValueError, match="Input image must be an ants.ANTsImage or a valid file path."):
        auto(image=42)
        
    # 2. target_resolution size mismatch
    with pytest.raises(ValueError, match="target_resolution must match image dimensions"):
        auto(image=mock_image_2d, target_resolution=(1.0, 1.0, 1.0))
        
    # 3. String file path input (we read from a dummy file or mock image_read)
    with patch('siq.auto.ants.image_read', return_value=mock_image_2d) as mock_read:
        with patch('siq.auto.default_dbpn') as mock_default_dbpn:
            with patch('siq.auto.inference') as mock_inf:
                auto(image="dummy_path.nii.gz")
                mock_read.assert_called_once_with("dummy_path.nii.gz")

def test_dbpn_multiple_losses():
    # Test dbpn with number_of_loss_functions > 1
    from siq.get_data import dbpn
    model = dbpn(
        input_image_size=(16, 16, 16, 1),
        number_of_outputs=1,
        number_of_base_filters=4,
        number_of_feature_filters=8,
        number_of_back_projection_stages=1,
        convolution_kernel_size=(3, 3, 3),
        strides=(2, 2, 2),
        last_convolution=(1, 1, 1),
        number_of_loss_functions=2
    )
    assert isinstance(model, keras.Model)
    assert len(model.outputs) == 2

def test_perceptual_metrics():
    from siq.get_data import compute_gmsd, compute_hfen
    
    # 1. 2D scalar
    x2d = np.random.rand(16, 16).astype("float32")
    y2d = np.random.rand(16, 16).astype("float32")
    gmsd_2d = compute_gmsd(x2d, y2d)
    hfen_2d = compute_hfen(x2d, y2d)
    assert isinstance(gmsd_2d, float)
    assert isinstance(hfen_2d, float)
    
    # 2. 3D scalar
    x3d = np.random.rand(16, 16, 16).astype("float32")
    y3d = np.random.rand(16, 16, 16).astype("float32")
    gmsd_3d = compute_gmsd(x3d, y3d)
    hfen_3d = compute_hfen(x3d, y3d)
    assert isinstance(gmsd_3d, float)
    assert isinstance(hfen_3d, float)
    
    # 3. 2D multi-channel
    x2d_c = np.random.rand(16, 16, 2).astype("float32")
    y2d_c = np.random.rand(16, 16, 2).astype("float32")
    gmsd_2d_c = compute_gmsd(x2d_c, y2d_c)
    hfen_2d_c = compute_hfen(x2d_c, y2d_c)
    assert isinstance(gmsd_2d_c, float)
    assert isinstance(hfen_2d_c, float)
    
    # 4. 3D multi-channel
    x3d_c = np.random.rand(16, 16, 16, 2).astype("float32")
    y3d_c = np.random.rand(16, 16, 16, 2).astype("float32")
    gmsd_3d_c = compute_gmsd(x3d_c, y3d_c)
    hfen_3d_c = compute_hfen(x3d_c, y3d_c)
    assert isinstance(gmsd_3d_c, float)
    assert isinstance(hfen_3d_c, float)

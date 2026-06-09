import pytest
import numpy as np
import ants
import os
import tensorflow as tf
from siq.get_data import default_dbpn, train, inference, train_seg, pseudo_3d_vgg_features, pseudo_3d_vgg_features_unbiased

def test_full_pipeline_fast():
    # 1. Create dummy data
    img_data = np.random.rand(64, 64, 64).astype(np.float32)
    img = ants.from_numpy(img_data)
    ants.image_write(img, 'dummy_train.nii.gz')
    ants.image_write(img, 'dummy_test.nii.gz')

    # 2. Create tiny model
    model = default_dbpn(
        strider=[2, 2, 2],
        dimensionality=3,
        nfilt=4,
        nff=8,
        convn=3,
        lastconv=1,
        nbp=1,
        option='tiny'
    )
    
    # 3. Train on dummy data for 1 iteration
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
    
    # 4. Inference
    sr_img = inference(img, model, verbose=False)
    assert isinstance(sr_img, ants.core.ants_image.ANTsImage)
    
def test_full_seg_pipeline():
    img_data = np.random.rand(64, 64, 64).astype(np.float32)
    img = ants.from_numpy(img_data)
    
    seg_data = (np.random.rand(64, 64, 64) > 0.5).astype(np.float32)
    seg = ants.from_numpy(seg_data)
    
    merged = ants.merge_channels([img, seg])
    ants.image_write(merged, 'dummy_train_seg.nii.gz')

    model = default_dbpn(
        strider=[2, 2, 2],
        dimensionality=3,
        nfilt=4,
        nff=8,
        convn=3,
        lastconv=1,
        nbp=1,
        option='tiny'
    )

    train_seg(
        mdl=model,
        filenames_train=['dummy_train_seg.nii.gz'],
        filenames_test=['dummy_train_seg.nii.gz'],
        target_patch_size=[16, 16, 16],
        target_patch_size_low=[8, 8, 8],
        output_prefix='test_out_seg',
        n_test=1,
        max_iterations=1,
        batch_size=1,
        check_eval_data_iteration=1,
        verbose=False
    )

def test_vgg_features():
    feat1 = pseudo_3d_vgg_features([32,32,32,1], layer=1, pretrained=False)
    assert isinstance(feat1, tf.keras.Model)
    
    feat2 = pseudo_3d_vgg_features_unbiased([32,32,32,1], layer=1)
    assert isinstance(feat2, tf.keras.Model)

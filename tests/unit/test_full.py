import pytest
import numpy as np
import ants
import os
import tensorflow as tf
from siq.get_data import default_dbpn, train, inference

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
    
    # Clean up
    if os.path.exists('dummy_train.nii.gz'): os.remove('dummy_train.nii.gz')
    if os.path.exists('dummy_test.nii.gz'): os.remove('dummy_test.nii.gz')

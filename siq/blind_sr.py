import ants
import numpy as np
import keras
from keras import layers
from .get_data import simulate_image, get_grader_feature_network
from .espcn import create_espcn_3d
import os
import time

def blind_sr_generator(batch_size=4, patch_size=(32, 32, 32), factor=2, feature_model=None):
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
                # Precompute features for ground truth
                f_hr = feature_model.predict(np.expand_dims(y_batch[-1], 0), verbose=0)
                f_batch.append(f_hr[0])
                
        x_out = np.array(x_batch, dtype="float32")
        y_out = np.array(y_batch, dtype="float32")
        
        if feature_model:
            f_out = np.array(f_batch, dtype="float32")
            yield x_out, [y_out, f_out]
        else:
            yield x_out, y_out

def train_blind_espcn_perceptual(factor=2, epochs=20, steps_per_epoch=50, feature_weight=2.0):
    patch_size_hr = (64, 64, 64)
    patch_size_lr = (32, 32, 32)
    
    # 1. Create ESPCN model
    base_model = create_espcn_3d(input_shape=(32, 32, 32, 1), factor=factor)
    
    # 2. Get Perceptual Model (Grader)
    try:
        feature_extractor = get_grader_feature_network(layer=6)
        feature_extractor.trainable = False
        has_feature = True
        print("Loaded ResNet grader for perceptual loss.")
    except Exception as e:
        print(f"Could not load perceptual model: {e}. Falling back to MSE only.")
        has_feature = False

    if has_feature:
        # 3. Create Multi-output model
        inputs = base_model.input
        hr_output = base_model.output
        feat_output = feature_extractor(hr_output)
        
        train_model = keras.Model(inputs=inputs, outputs=[hr_output, feat_output])
        train_model.compile(
            optimizer=keras.optimizers.Adam(5e-5),
            loss=["mse", "mse"],
            loss_weights=[1.0, feature_weight]
        )
        gen = blind_sr_generator(batch_size=4, patch_size=patch_size_hr, factor=factor, feature_model=feature_extractor)
    else:
        train_model = base_model
        train_model.compile(optimizer=keras.optimizers.Adam(5e-5), loss="mse")
        gen = blind_sr_generator(batch_size=4, patch_size=patch_size_hr, factor=factor)

    print("Starting Perceptual Blind SR training...")
    train_model.fit(gen, epochs=epochs, steps_per_epoch=steps_per_epoch)
    
    base_model.save("espcn_3d_perceptual.keras")
    return base_model

if __name__ == "__main__":
    train_blind_espcn_perceptual(epochs=5, steps_per_epoch=20)

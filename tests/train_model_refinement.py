import os
import sys

# Configure Keras to use PyTorch backend for GPU MPS/CUDA acceleration
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import keras
from keras import ops
import ants
import antspynet
import siq

def set_core_trainable(model, trainable=True):
    count = 0
    for layer in model.layers:
        if isinstance(layer, keras.layers.Conv3D):
            if layer.kernel_size != (1, 1, 1) and "_ca_" not in layer.name:
                layer.trainable = trainable
                count += 1
    print(f"Set layer.trainable={trainable} for {count} core Conv3D layers.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="SIQ Super-Resolution Refinement Pipeline")
    parser.add_argument("model", choices=["espcn", "ldbpn", "ref-dbpn"], default="espcn", nargs="?", help="Model type to refine (default: espcn)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for training (default: 1)")
    args = parser.parse_args()
    
    model_type = args.model
    batch_size = args.batch_size
            
    print(f"Initializing {model_type.upper()} Refinement Pipeline...")
    scratch_dir = "/Users/stnava/.gemini/antigravity-cli/brain/bf9e3239-711d-4a46-8dd4-8a3f33959db5/scratch"
    workspace_dir = "."
    
    # 1. Load Real MRI (OASIS) validation patches for monitoring
    print("Loading Real MRI (OASIS) validation volume...")
    img_path = antspynet.get_antsxnet_data("oasis")
    img = ants.image_read(img_path)
    
    print("Simulating Validation Low Resolution...")
    low_res = ants.resample_image(img, [s*2 for s in img.spacing], use_voxels=False, interp_type=0)
    
    mid_lr = [s//2 for s in low_res.shape]
    lr_patch = ants.crop_indices(low_res, [m - 24 for m in mid_lr], [m + 24 for m in mid_lr])
    
    mid_hr = [s//2 for s in img.shape]
    hr_patch = ants.crop_indices(img, [m - 48 for m in mid_hr], [m + 48 for m in mid_hr])
    gt_np = hr_patch.numpy()
    
    # 2. Try loading pre-generated cache
    cache_path = "hr_base_cache_512x128.npy"
    if os.path.exists(cache_path):
        print(f"Loading cached volumes from local {os.path.abspath(cache_path)}...")
        hr_base_cache = np.load(cache_path)
    else:
        fallback_path = os.path.join(scratch_dir, "hr_base_cache_512x128.npy")
        if os.path.exists(fallback_path):
            print(f"Loading cached volumes from fallback {os.path.abspath(fallback_path)}...")
            hr_base_cache = np.load(fallback_path)
        else:
            print("Pre-generated cache not found. Fallback: generator will dynamically create a 512-volume cache on the fly.")
            hr_base_cache = None

    # 3. Define Simulation Classes mixture
    simulation_classes = {
        "brain_procedural": 0.35,
        "layered": 0.25,
        "sinewave": 0.25,
        "organic_blobs": 0.15
    }
    
    # 4. Instantiate Model
    skip_stages_1_2 = False
    if model_type == "espcn":
        output_model_path = os.path.join(workspace_dir, "espcn_3d_attention_refined.keras")
        best_model_path = os.path.join(workspace_dir, "espcn_3d_attention_clean_best_mdl.keras")
        custom_objects = {"PixelShuffle3D": siq.PixelShuffle3D, "LearnableScale": siq.LearnableScale}
        
        if os.path.exists(output_model_path):
            print(f"Resuming training: loading existing refined CA-ESPCN model from {output_model_path}...")
            model = keras.models.load_model(output_model_path, custom_objects=custom_objects, compile=False)
            skip_stages_1_2 = True
        elif os.path.exists(best_model_path):
            print(f"Starting fresh: loading baseline CA-ESPCN model from {best_model_path}...")
            model = keras.models.load_model(best_model_path, custom_objects=custom_objects, compile=False)
        else:
            print("Baseline model not found. Building a new Attention-Enhanced ESPCN 3D model...")
            model = siq.create_espcn_3d_attention(
                input_shape=(None, None, None, 1),
                factor=2,
                n_filters=128,
                n_res_blocks=8,
                use_global_skip=True
            )
    elif model_type == "ldbpn":
        output_model_path = os.path.join(workspace_dir, "ldbpn_3d_refined.keras")
        best_model_path = os.path.join(workspace_dir, "ldbpn_3d_best_mdl.keras")
        
        if os.path.exists(output_model_path):
            print(f"Resuming training: loading existing refined L-DBPN model from {output_model_path}...")
            model = keras.models.load_model(output_model_path, compile=False)
            skip_stages_1_2 = True
        elif os.path.exists(best_model_path):
            print(f"Starting fresh: loading baseline L-DBPN model from {best_model_path}...")
            model = keras.models.load_model(best_model_path, compile=False)
        else:
            print("Baseline model not found. Building a new Lightweight DBPN 3D model...")
            model = siq.create_ldbpn_3d(
                input_shape=(None, None, None, 1),
                factor=2,
                n_filters=64,
                n_stages=3
            )
    else: # ref-dbpn
        output_model_path = os.path.join(workspace_dir, "ref_dbpn_3d_refined.keras")
        best_model_path = os.path.join(workspace_dir, "exp_baseline_best.keras")
        
        if os.path.exists(output_model_path):
            print(f"Resuming training: loading existing refined Reference DBPN model from {output_model_path}...")
            model = keras.models.load_model(output_model_path, compile=False)
            skip_stages_1_2 = True
        elif os.path.exists(best_model_path):
            print(f"Starting fresh: loading baseline Reference DBPN model from {best_model_path}...")
            model = keras.models.load_model(best_model_path, compile=False)
        else:
            print("Baseline model not found. Building a new Reference DBPN 3D model...")
            model = siq.default_dbpn(
                strider=[2, 2, 2],
                dimensionality=3,
                option="large"
            )
        
    # 5. Load feature extractor for perceptual loss
    print("Loading pseudo-3D VGG feature extractor...")
    feature_extractor = siq.pseudo_3d_vgg_features_unbiased(inshape=[96, 96, 96], layer=4)
    feature_extractor.trainable = False
    
    # 6. Hybrid loss function variables
    msq_weight_var = keras.Variable(1.0, dtype="float32")
    feat_weight_var = keras.Variable(0.1, dtype="float32")
    tv_weight_var = keras.Variable(0.005, dtype="float32")
    l1_weight_var = keras.Variable(0.5, dtype="float32")

    def hybrid_loss(y_true, y_pred):
        # L2 Loss (MSE)
        squared_diff = ops.square(y_true - y_pred)
        l2_term = ops.mean(squared_diff, axis=[1, 2, 3, 4])
        
        # L1 Loss (MAE) for sharper edges
        abs_diff = ops.abs(y_true - y_pred)
        l1_term = ops.mean(abs_diff, axis=[1, 2, 3, 4])
        
        # Perceptual Loss
        f_true = feature_extractor(y_true)
        f_pred = feature_extractor(y_pred)
        feat_term = ops.mean(ops.square(f_true - f_pred), axis=[1, 2, 3, 4])
        
        # Total Variation Loss
        diff_d = ops.mean(ops.abs(y_pred[:, 1:, :, :, :] - y_pred[:, :-1, :, :, :]), axis=[1, 2, 3, 4])
        diff_h = ops.mean(ops.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :]), axis=[1, 2, 3, 4])
        diff_w = ops.mean(ops.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :]), axis=[1, 2, 3, 4])
        tv_term = diff_d + diff_h + diff_w
        
        return (l2_term * msq_weight_var + 
                l1_term * l1_weight_var + 
                feat_term * feat_weight_var + 
                tv_term * tv_weight_var)

    best_val_loss = float("inf")

    if not skip_stages_1_2:
        # ==============================================================
        # Stage 1: Warmup & Adaptation on Clean Mixed Classes (Iter 1-100)
        # ==============================================================
        print("\n=======================================================")
        print("Stage 1: Adaptation Phase (Clean Mixed Geometries)")
        print("=======================================================")
        
        if model_type == "espcn":
            set_core_trainable(model, trainable=False)
        
        train_gen_clean = siq.blind_sr_generator(
            hr_base_cache=hr_base_cache,
            batch_size=batch_size,
            lr_patch_size=48,
            factor=2,
            blur_sigma_range=(0.0, 0.0),
            noise_std_range=(0.0, 0.0),
            simulation_classes=simulation_classes,
            zoom_range=(0.75, 1.3)
        )
        
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=2e-5), loss=hybrid_loss)
        
        for iteration in range(1, 101):
            x_batch, y_batch = next(train_gen_clean)
            loss = model.train_on_batch(x_batch, y_batch)
            
            if iteration % 25 == 0 or iteration == 1:
                sr_img = siq.inference(lr_patch, model, method="antspynet", verbose=False)
                ants.copy_image_info(hr_patch, sr_img)
                sr_np = sr_img.numpy()
                corr = float(np.corrcoef(sr_np.flatten(), gt_np.flatten())[0, 1])
                psnr = float(antspynet.psnr(hr_patch, sr_img))
                ssim = float(antspynet.ssim(hr_patch, sr_img))
                
                print(f"Stage 1 Iter {iteration:03d}/100 - Loss: {loss:.6f}")
                print(f"  [OASIS Monitor] Corr: {corr:.4f}, PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
                
                if loss < best_val_loss:
                    best_val_loss = loss
                    model.save(output_model_path)
                    print(f"  --> Saved checkpoint to {output_model_path}")

        # ==============================================================
        # Stage 2: Joint Fine-Tuning with Rician Noise & Blur (Iter 101-250)
        # ==============================================================
        print("\n=======================================================")
        print("Stage 2: Robustness Fine-Tuning Phase (Low LR + Noise)")
        print("=======================================================")
        
        if os.path.exists(output_model_path):
            print(f"Loading best Stage 1 checkpoint from {output_model_path}...")
            if model_type == "espcn":
                custom_objs_load = {"PixelShuffle3D": siq.PixelShuffle3D, "LearnableScale": siq.LearnableScale}
                model = keras.models.load_model(output_model_path, custom_objects=custom_objs_load, compile=False)
            else:
                model = keras.models.load_model(output_model_path, compile=False)
        
        if model_type == "espcn":
            set_core_trainable(model, trainable=True)
        
        train_gen_robust = siq.blind_sr_generator(
            hr_base_cache=hr_base_cache,
            batch_size=batch_size,
            lr_patch_size=48,
            factor=2,
            blur_sigma_range=(0.0, 0.0),
            noise_std_range=(0.0, 0.02),
            use_rician_noise=True,
            simulation_classes=simulation_classes,
            zoom_range=(0.75, 1.3)
        )
        
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-7), loss=hybrid_loss)
        
        for iteration in range(101, 251):
            x_batch, y_batch = next(train_gen_robust)
            loss = model.train_on_batch(x_batch, y_batch)
            
            if iteration % 25 == 0 or iteration == 101:
                sr_img = siq.inference(lr_patch, model, method="antspynet", verbose=False)
                ants.copy_image_info(hr_patch, sr_img)
                sr_np = sr_img.numpy()
                corr = float(np.corrcoef(sr_np.flatten(), gt_np.flatten())[0, 1])
                psnr = float(antspynet.psnr(hr_patch, sr_img))
                ssim = float(antspynet.ssim(hr_patch, sr_img))
                
                print(f"Stage 2 Iter {iteration:03d}/250 - Loss: {loss:.6f}")
                print(f"  [OASIS Monitor] Corr: {corr:.4f}, PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
                
                if loss < best_val_loss:
                    best_val_loss = loss
                    model.save(output_model_path)
                    print(f"  --> Saved checkpoint to {output_model_path}")
    else:
        print("\nSkipping Stage 1 & Stage 2 (already refined). Proceeding directly to Stage 3 (Dedicated Refinement)...")

    # ==============================================================
    # Stage 3: Dedicated Refinement Strategy (Iter 251-450)
    # ==============================================================
    print("\n=======================================================")
    print("Stage 3: Dedicated Refinement Phase (High-Fidelity Brain Focus)")
    print("=======================================================")
    
    if skip_stages_1_2:
        # Load existing refined model with compile=False and make it trainable
        if model_type == "espcn":
            custom_objs_load = {"PixelShuffle3D": siq.PixelShuffle3D, "LearnableScale": siq.LearnableScale}
            model = keras.models.load_model(output_model_path, custom_objects=custom_objs_load, compile=False)
            set_core_trainable(model, trainable=True)
        else:
            model = keras.models.load_model(output_model_path, compile=False)
    else:
        # Load best Stage 2 checkpoint if it exists
        if os.path.exists(output_model_path):
            print(f"Loading best Stage 2 checkpoint from {output_model_path}...")
            if model_type == "espcn":
                custom_objs_load = {"PixelShuffle3D": siq.PixelShuffle3D, "LearnableScale": siq.LearnableScale}
                model = keras.models.load_model(output_model_path, custom_objects=custom_objs_load, compile=False)
                set_core_trainable(model, trainable=True)
            else:
                model = keras.models.load_model(output_model_path, compile=False)

    refinement_classes = {
        "brain_procedural": 0.60,
        "layered": 0.20,
        "sinewave": 0.10,
        "organic_blobs": 0.10
    }

    train_gen_refine = siq.blind_sr_generator(
        hr_base_cache=hr_base_cache,
        batch_size=batch_size,
        lr_patch_size=48,
        factor=2,
        blur_sigma_range=(0.0, 0.0),
        noise_std_range=(0.0, 0.01),
        use_rician_noise=True,
        simulation_classes=refinement_classes,
        zoom_range=(0.75, 1.3)
    )
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-7), loss=hybrid_loss)
    best_val_loss = float("inf")

    for iteration in range(251, 451):
        x_batch, y_batch = next(train_gen_refine)
        loss = model.train_on_batch(x_batch, y_batch)
        
        if iteration % 25 == 0 or iteration == 251:
            sr_img = siq.inference(lr_patch, model, method="antspynet", verbose=False)
            ants.copy_image_info(hr_patch, sr_img)
            sr_np = sr_img.numpy()
            corr = float(np.corrcoef(sr_np.flatten(), gt_np.flatten())[0, 1])
            psnr = float(antspynet.psnr(hr_patch, sr_img))
            ssim = float(antspynet.ssim(hr_patch, sr_img))
            
            print(f"Stage 3 Iter {iteration:03d}/450 - Loss: {loss:.6f}")
            print(f"  [OASIS Monitor] Corr: {corr:.4f}, PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
            
            if loss < best_val_loss:
                best_val_loss = loss
                model.save(output_model_path)
                print(f"  --> Saved checkpoint to {output_model_path}")

    print(f"{model_type.upper()} Refinement Pipeline Complete.")

if __name__ == "__main__":
    main()

import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
import ants
import antspynet
import siq

def main():
    backend = os.environ.get("KERAS_BACKEND", "tensorflow")
    print(f"Using backend: {backend}")

    print("Loading Real MRI (OASIS) image...")
    img_path = antspynet.get_antsxnet_data('oasis')
    img = ants.image_read(img_path)
    
    # Save original full (cached/static, or generated if not present)
    if not os.path.exists('docs/images/original_full.png'):
        print("Generating original full slice image...")
        ants.plot(img, filename='docs/images/original_full.png', title="Original High-Res (Full)", axis=2)

    print("Simulating Low Resolution (2x downsample)...")
    low_res = ants.resample_image(img, [s*2 for s in img.spacing], use_voxels=False, interp_type=0)
    
    if not os.path.exists('docs/images/lowres_full.png'):
        print("Generating lowres full slice image...")
        blurry = ants.resample_image_to_target(low_res, img, interp_type='linear')
        ants.plot(blurry, filename='docs/images/lowres_full.png', title="Low Resolution (Full)", axis=2)

    # Define zoom cropping indices
    # Low-res patch: 32x32x32 from center
    mid_lr = [s//2 for s in low_res.shape]
    lower_lr = [m - 16 for m in mid_lr]
    upper_lr = [m + 16 for m in mid_lr]
    lr_patch = ants.crop_indices(low_res, lower_lr, upper_lr)
    
    # High-res patch: 64x64x64 from center
    mid_hr = [s//2 for s in img.shape]
    lower_hr = [m - 32 for m in mid_hr]
    upper_hr = [m + 32 for m in mid_hr]
    hr_patch = ants.crop_indices(img, lower_hr, upper_hr)

    # Save original high-res patch plot
    ants.plot(hr_patch, filename='docs/images/original_zoom.png', title="Original Detail", axis=2)
    
    # Save blurry low-res patch plot (resampled to match high-res resolution)
    lr_patch_upsampled = ants.resample_image_to_target(lr_patch, hr_patch, interp_type='linear')
    ants.plot(lr_patch_upsampled, filename='docs/images/lowres_zoom.png', title="Low-Res Detail", axis=2)

    # 1. ESPCN Inference on Patch
    espcn_path = "espcn_3d_blind_kitchen_sink_best_mdl.keras"
    if os.path.exists(espcn_path):
        print("Running ESPCN inference on patch...")
        custom_objects = {"PixelShuffle3D": siq.PixelShuffle3D}
        model_espcn = keras.models.load_model(espcn_path, custom_objects=custom_objects, compile=False)
        
        sr_espcn_patch = siq.inference(lr_patch, model_espcn, method='antspynet', verbose=True)
        ants.plot(sr_espcn_patch, filename='docs/images/superres_zoom_espcn.png', title="ESPCN Detail", axis=2)
    else:
        print("ESPCN model not found.")

    # 1.5. Attention ESPCN Inference on Patch
    attention_path = "espcn_3d_attention_best_mdl.keras"
    if os.path.exists(attention_path):
        print("Running Attention ESPCN inference on patch...")
        custom_objects = {"PixelShuffle3D": siq.PixelShuffle3D, "LearnableScale": siq.LearnableScale}
        model_attn = keras.models.load_model(attention_path, custom_objects=custom_objects, compile=False)
        
        sr_attn_patch = siq.inference(lr_patch, model_attn, method='antspynet', verbose=True)
        ants.plot(sr_attn_patch, filename='docs/images/superres_zoom_attention.png', title="Attention ESPCN Detail", axis=2)
    else:
        print("Attention ESPCN model not found.")

    # 2. DBPN Inference on Patch
    dbpn_path = os.path.expanduser("~/.antspymm/siq_smallshort_train_2x2x2_1chan_featvggL6_best.keras")
    if os.path.exists(dbpn_path):
        print("Running DBPN inference on patch...")
        model_dbpn = keras.models.load_model(dbpn_path, compile=False)
        sr_dbpn_patch = siq.inference(lr_patch, model_dbpn, method='antspynet', verbose=True)
        ants.plot(sr_dbpn_patch, filename='docs/images/superres_zoom.png', title="DBPN Detail", axis=2)
    else:
        print("DBPN model not found.")

    print("Done!")

if __name__ == "__main__":
    main()

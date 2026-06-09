import os
if os.environ.get("KERAS_BACKEND") == "torch":
    import torch
    torch.backends.mps.is_available = lambda: False
    torch.backends.mps.is_built = lambda: False
    torch.set_default_device("cpu")
import ants
import antspynet
import siq
import keras

def main():
    print("Loading Real MRI (OASIS) image...")
    # Load a highly detailed, standard real T1 MRI brain from OASIS dataset
    img_path = antspynet.get_antsxnet_data('oasis')
    img = ants.image_read(img_path)
    
    print("Plotting Original Full...")
    ants.plot(img, filename='docs/images/original_full.png', title="Original High-Res (Full)", axis=2)

    print("Simulating Low Resolution (2x downsample)...")
    # simulate low resolution by downsampling
    low_res = ants.resample_image(img, [s*2 for s in img.spacing], use_voxels=False, interp_type=0)
    
    # upsample back to original grid to be perfectly comparable
    blurry = ants.resample_image_to_target(low_res, img, interp_type='linear')
    print("Plotting Low Res (Blurry) Full...")
    ants.plot(blurry, filename='docs/images/lowres_full.png', title="Low Resolution (Full)", axis=2)

    print("Loading pre-trained SIQ model...")
    model_path = os.path.expanduser('~/.antspymm/siq_default_sisr_2x2x2_1chan_featvggL6_best_mdl.h5')
    backend = keras.backend.backend()
    if os.path.exists(model_path):
        model = keras.models.load_model(model_path, compile=False)
        print("Running SIQ inference...")
        # siq.inference takes care of everything
        sr_img = siq.inference(low_res, model, verbose=False)
        
        print("Plotting Super-Resolved Full...")
        ants.plot(sr_img, filename=f'docs/images/superres_full_{backend}.png', title=f"SIQ Super-Resolved (Full - {backend})", axis=2)
    else:
        print(f"Model {model_path} not found. Using a tiny DBPN to demonstrate the pipeline.")
        sr_img = siq.auto(low_res, target_resolution=img.shape)
        ants.plot(sr_img, filename=f'docs/images/superres_full_{backend}.png', title=f"SIQ Auto-Resolved (Full - {backend})", axis=2)

    # Now create zoomed-in patches
    print("Extracting and saving zoomed patches...")
    mid = [s//2 for s in img.shape]
    lower = [m - 30 for m in mid]
    upper = [m + 30 for m in mid]
    
    orig_crop = ants.crop_indices(img, lower, upper)
    blurry_crop = ants.crop_indices(blurry, lower, upper)
    sr_crop = ants.crop_indices(sr_img, lower, upper)

    ants.plot(orig_crop, filename='docs/images/original_zoom.png', title="Original Detail", axis=2)
    ants.plot(blurry_crop, filename='docs/images/lowres_zoom.png', title="Low Res Detail", axis=2)
    ants.plot(sr_crop, filename=f'docs/images/superres_zoom_{backend}.png', title=f"SIQ Detail ({backend})", axis=2)
    print("Done!")

if __name__ == "__main__":
    main()

import os
import ants
import siq
import tensorflow as tf

def main():
    print("Loading MNI image...")
    # Load a standard 3D brain image
    mni_path = ants.get_ants_data('mni')
    img = ants.image_read(mni_path)
    
    # We will just take the head or a cropped region to make the SR effect visible
    # Actually let's just use the whole thing
    print("Plotting Original...")
    ants.plot(img, filename='docs/images/original.png', title="Original High-Res", axis=2)

    print("Simulating Low Resolution (2x downsample)...")
    # simulate_image(img, target_resolution, noise_sd=0, add_noise=False)
    # returns [resam, orig, up_nearest]
    # actually let's just manually resample to ensure it works smoothly
    low_res = ants.resample_image(img, [s/2 for s in img.spacing], use_voxels=False, interp_type=1) 
    # interp_type=1 is nearest neighbor, or we can use 0 for linear. 
    # standard downsample would be linear or gaussian. Let's use standard (linear)
    low_res = ants.resample_image(img, [s*2 for s in img.spacing], use_voxels=False, interp_type=0)
    
    # To compare on the same grid, upsample it back with nearest neighbor or b-spline
    blurry = ants.resample_image_to_target(low_res, img, interp_type='linear')
    print("Plotting Low Res (Blurry)...")
    ants.plot(blurry, filename='docs/images/lowres.png', title="Low Resolution (Interpolated)", axis=2)

    print("Loading pre-trained SIQ model...")
    model_path = os.path.expanduser('~/.antspymm/siq_default_sisr_2x2x2_1chan_featvggL6_best_mdl.h5')
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path, compile=False)
        print("Running SIQ inference...")
        # run inference on the blurry or low_res image? 
        # Usually models take the low_res image directly or the interpolated one.
        # Let's use the new auto mode! But auto mode creates a new model if we don't pass one.
        # Let's use siq.get_data.inference directly.
        
        sr_img = siq.inference(low_res, model, verbose=False)
        
        print("Plotting Super-Resolved...")
        ants.plot(sr_img, filename='docs/images/superres.png', title="SIQ Super-Resolved", axis=2)
    else:
        print(f"Model {model_path} not found. Using a tiny DBPN to demonstrate the pipeline.")
        sr_img = siq.auto(low_res, target_resolution=img.shape)
        ants.plot(sr_img, filename='docs/images/superres.png', title="SIQ Auto-Resolved", axis=2)

if __name__ == "__main__":
    main()

"""
End-to-end example for multi-task (image + segmentation) super-resolution.

This script demonstrates the full workflow for a dual-channel model:
1.  Setting up environment variables for optimal performance.
2.  Fetching a standard public dataset and splitting it correctly.
3.  Instantiating a 2-channel DBPN model suitable for multi-task learning.
4.  Running a brief demonstration training using `siq.train_seg`.
5.  Loading the trained model artifact.
6.  Preparing a realistic low-resolution test case with both an image and a mask.
7.  Performing inference to get both a super-resolved image and segmentation.
8.  Saving all relevant images for visual comparison and verification.
"""
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
from pathlib import Path
import glob as glob


def main():
    # --- 1. Configuration and Setup ---
    print("--- Step 1: Configuring Environment & Parameters ---")
    mynt = "8"
    os.environ["TF_NUM_INTEROP_THREADS"] = mynt
    os.environ["TF_NUM_INTRAOP_THREADS"] = mynt
    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = mynt

    UPSAMPLE_FACTOR = 2
    OUTPUT_DIR = Path("./siq_multi_task_example_output/")
    MODEL_PREFIX = OUTPUT_DIR / "siq_multi_task_demo_model"
    LOW_RES_PATCH_SIZE = [16, 16, 16]
    HIGH_RES_PATCH_SIZE = [dim * UPSAMPLE_FACTOR for dim in LOW_RES_PATCH_SIZE]

    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"All outputs will be saved in: {OUTPUT_DIR.resolve()}")

    # --- 2. Data Acquisition and Splitting ---
    print("\n--- Step 2: Fetching and Splitting Data ---")
    all_files = glob.glob(os.path.expanduser("~/.antspyt1w/2*T1w*gz"))
    if len(all_files) == 0:
        print("No input data found in ~/.antspyt1w/2*T1w*gz. Skipping demonstration.")
        return

    train_files = all_files[:-1]
    test_files = all_files[-1:]
    print(f"Using {len(train_files)} file(s) for training.")
    print(f"Using {len(test_files)} file(s) for testing.")

    # --- 3. Model Initialization for Multi-Task Learning ---
    print("\n--- Step 3: Initializing the Multi-Task Super-Resolution Model ---")
    strides = [UPSAMPLE_FACTOR] * 3
    model = siq.default_dbpn(
        strides,
        number_of_outputs=2,
        number_of_channels=2
    )
    print("Multi-task model created successfully. Summary:")
    model.summary()

    # --- 4. Model Training ---
    print("\n--- Step 4: Starting a Short Demonstration Training ---")
    training_history = siq.train_seg(
        mdl=model,
        filenames_train=train_files,
        filenames_test=test_files,
        output_prefix=str(MODEL_PREFIX),
        target_patch_size=HIGH_RES_PATCH_SIZE,
        target_patch_size_low=LOW_RES_PATCH_SIZE,
        n_test=2,
        learning_rate=5e-05,
        max_iterations=5,
        verbose=True
    )
    print("Demonstration training complete.")

    # --- 5. Inference on a Test Case ---
    print("\n--- Step 5: Running Inference on a Test Case ---")
    best_model_path = f"{MODEL_PREFIX}_best_mdl.keras"
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Trained model not found at {best_model_path}.")

    print(f"Loading trained multi-task model from: {best_model_path}")
    trained_model = keras.models.load_model(best_model_path, compile=False)

    print("Preparing low-resolution input image and a simulated segmentation mask...")
    test_image_high_res = ants.crop_image(ants.image_read(test_files[0]))
    test_image_high_res = ants.resample_image(test_image_high_res, [2, 2, 2])

    segmentation_high_res = ants.threshold_image(test_image_high_res, "Otsu", 1)

    low_res_spacing = [s * UPSAMPLE_FACTOR for s in test_image_high_res.spacing]
    test_image_low_res = ants.resample_image(test_image_high_res, low_res_spacing, use_voxels=False, interp_type=0)
    segmentation_low_res = ants.resample_image(segmentation_high_res, low_res_spacing, use_voxels=False, interp_type=1)

    print("Applying multi-task super-resolution model...")
    inference_result = siq.inference(
        image=test_image_low_res,
        mdl=trained_model,
        segmentation=segmentation_low_res,
        verbose=True
    )

    super_resolved_image = inference_result['super_resolution']
    super_resolved_seg = inference_result.get('segmentation', None)

    # --- 6. Save Outputs for Verification ---
    print("\n--- Step 6: Saving Images for Comparison ---")
    path_lr_image = OUTPUT_DIR / "test_input_low_res_image.nii.gz"
    path_lr_seg = OUTPUT_DIR / "test_input_low_res_seg.nii.gz"
    path_sr_image = OUTPUT_DIR / "test_output_super_res_image.nii.gz"
    path_gt_image = OUTPUT_DIR / "test_ground_truth_high_res_image.nii.gz"
    path_gt_seg = OUTPUT_DIR / "test_ground_truth_high_res_seg.nii.gz"

    ants.image_write(test_image_low_res, str(path_lr_image))
    ants.image_write(segmentation_low_res, str(path_lr_seg))
    ants.image_write(test_image_high_res, str(path_gt_image))
    ants.image_write(segmentation_high_res, str(path_gt_seg))

    ants.image_write(super_resolved_image, str(path_sr_image))
    if super_resolved_seg:
        path_sr_seg = OUTPUT_DIR / "test_output_super_res_seg.nii.gz"
        ants.image_write(super_resolved_seg, str(path_sr_seg))
        print(f"  - Output Segmentation: {path_sr_seg.name}")

    print("\n--- Example Finished ---")
    print(f"Check the directory '{OUTPUT_DIR.resolve()}' for the following files:")
    print(f"  - Input Image: {path_lr_image.name}")
    print(f"  - Input Segmentation: {path_lr_seg.name}")
    print(f"  - Output Image: {path_sr_image.name}")
    print(f"  - Ground Truth Image: {path_gt_image.name}")
    print(f"  - Ground Truth Segmentation: {path_gt_seg.name}")


if __name__ == "__main__":
    main()

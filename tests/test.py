"""
A comprehensive, end-to-end example for using the `siq` package.

This script demonstrates the full workflow:
1.  Fetching a standard public dataset.
2.  Properly splitting data for training and testing.
3.  Instantiating a default super-resolution model.
4.  Running a brief training loop to generate a model file.
5.  Loading the trained model artifact.
6.  Performing inference on a low-resolution test image.
7.  Saving the results for visual comparison.
"""
import os
import ants
import antspynet
import siq
import keras
from pathlib import Path
import glob as glob


def main():
    # --- 1. Configuration and Setup ---
    print("--- Step 1: Configuring Environment ---")
    UPSAMPLE_FACTOR = 2
    OUTPUT_DIR = Path("./siq_example_output/")
    MODEL_PREFIX = OUTPUT_DIR / "siq_demo_model"
    LOW_RES_PATCH_SIZE = [16, 16, 16]
    HIGH_RES_PATCH_SIZE = [dim * UPSAMPLE_FACTOR for dim in LOW_RES_PATCH_SIZE]

    # Create the output directory.
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

    # --- 3. Model Initialization ---
    print("\n--- Step 3: Initializing the Super-Resolution Model ---")
    strides = [UPSAMPLE_FACTOR] * 3
    model = siq.default_dbpn(strides)
    print("Model created successfully. Summary:")
    model.summary()

    # --- 4. Model Training ---
    print("\n--- Step 4: Starting a Short Demonstration Training ---")
    training_history = siq.train(
        mdl=model,
        filenames_train=train_files,
        filenames_test=test_files,
        output_prefix=str(MODEL_PREFIX),
        target_patch_size=HIGH_RES_PATCH_SIZE,
        target_patch_size_low=LOW_RES_PATCH_SIZE,
        n_test=2,
        learning_rate=5e-05,
        max_iterations=5,
        verbose=True,
    )
    print("Demonstration training complete.")

    # --- 5. Inference on a Test Image ---
    print("\n--- Step 5: Running Inference on a Test Image ---")
    best_model_path = f"{MODEL_PREFIX}_best_mdl.keras"
    if not os.path.exists(best_model_path):
        raise FileNotFoundError(f"Trained model not found at {best_model_path}.")

    print(f"Loading trained model from: {best_model_path}")
    trained_model = keras.models.load_model(best_model_path, compile=False)

    print("Preparing low-resolution input image...")
    test_image_high_res = ants.crop_image(ants.image_read(test_files[0]))
    test_image_high_res = ants.resample_image(test_image_high_res, [2, 2, 2])
    low_res_spacing = [s * UPSAMPLE_FACTOR for s in test_image_high_res.spacing]
    test_image_low_res = ants.resample_image(test_image_high_res, low_res_spacing, use_voxels=False, interp_type=0)

    print("Applying super-resolution model...")
    super_resolved_image = siq.inference(test_image_low_res, trained_model, verbose=True)

    # --- 6. Save Outputs for Verification ---
    print("\n--- Step 6: Saving Images for Comparison ---")
    low_res_path = OUTPUT_DIR / "test_image_low_res.nii.gz"
    super_res_path = OUTPUT_DIR / "test_image_super_resolved.nii.gz"
    ground_truth_path = OUTPUT_DIR / "test_image_ground_truth.nii.gz"

    ants.image_write(test_image_low_res, str(low_res_path))
    ants.image_write(super_resolved_image, str(super_res_path))
    ants.image_write(test_image_high_res, str(ground_truth_path))

    print("\n--- Example Finished ---")
    print(f"Check the directory '{OUTPUT_DIR.resolve()}' for the following files:")
    print(f"  - Input: {low_res_path.name}")
    print(f"  - Output: {super_res_path.name}")
    print(f"  - Ground Truth: {ground_truth_path.name}")


if __name__ == "__main__":
    main()

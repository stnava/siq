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
import ants
import antspynet
import siq
import tensorflow as tf
from pathlib import Path
import glob as glob

# --- 1. Configuration and Setup ---
# These settings can improve performance on multi-core systems.
print("--- Step 1: Configuring Environment & Parameters ---")
mynt="8"
os.environ["TF_NUM_INTEROP_THREADS"] = mynt
os.environ["TF_NUM_INTRAOP_THREADS"] = mynt
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = mynt

# Use clear variables for important parameters.
UPSAMPLE_FACTOR = 2
OUTPUT_DIR = Path("./siq_multi_task_example_output/")
MODEL_PREFIX = OUTPUT_DIR / "siq_multi_task_demo_model"
LOW_RES_PATCH_SIZE = [16, 16, 16]
HIGH_RES_PATCH_SIZE = [dim * UPSAMPLE_FACTOR for dim in LOW_RES_PATCH_SIZE]

# Create the output directory.
OUTPUT_DIR.mkdir(exist_ok=True)
print(f"All outputs will be saved in: {OUTPUT_DIR.resolve()}")


# --- 2. Data Acquisition and Splitting ---
# We robustly download a standard dataset to ensure the example is reproducible.
print("\n--- Step 2: Fetching and Splitting Data ---")
all_files=glob.glob( os.path.expanduser( "~/.antspyt1w/2*T1w*gz" ) )

# CRITICAL: Always use separate files for training and testing to avoid data leakage.
train_files = all_files[:-1]
test_files = all_files[-1:]  # Use the last file for testing.
print(f"Using {len(train_files)} file(s) for training.")
print(f"Using {len(test_files)} file(s) for testing.")


# --- 3. Model Initialization for Multi-Task Learning ---
# We create a model with 2 input and 2 output channels.
# Channel 0: Intensity Image
# Channel 1: Segmentation Mask
print("\n--- Step 3: Initializing the Multi-Task Super-Resolution Model ---")
strides = [UPSAMPLE_FACTOR] * 3
model = siq.default_dbpn(
    strides,
    nChannelsIn=2,
    nChannelsOut=2,
    sigmoid_second_channel=True  # Apply sigmoid to the segmentation output channel.
)
print("2-channel model created successfully. Summary:")
model.summary()


# --- 4. Model Training with `train_seg` ---
# Run a very short training loop to demonstrate the function and save a model file.
# The loss function will combine image similarity (MSE, Perceptual) and
# segmentation similarity (Dice).
print("\n--- Step 4: Starting a Short Demonstration Training for Multi-Task Model ---")
# NOTE: For a real model, increase `max_iterations` significantly (e.g., to 10000+).
mdlfn = f"{MODEL_PREFIX}_best_mdl.keras"
if os.path.exists(mdlfn):
    print(f"Model already exists at {mdlfn}, loading it.")
    model = tf.keras.models.load_model(mdlfn, compile=False)
else:
    print("Training a new model from scratch.")
    training_history = siq.train_seg(
        mdl=model,
        filenames_train=train_files,
        filenames_test=test_files,
        output_prefix=str(MODEL_PREFIX),
        target_patch_size=HIGH_RES_PATCH_SIZE,
        target_patch_size_low=LOW_RES_PATCH_SIZE,
        n_test=2,
        learning_rate=5e-05,
        max_iterations=5,  # Keep low for a quick demo.
        verbose=True
    )
    print("Demonstration training complete.")


# --- 5. Inference on a Test Image and Segmentation ---
print("\n--- Step 5: Running Inference on a Test Case ---")

# First, load the BEST model that was saved during the training loop.
best_model_path = f"{MODEL_PREFIX}_best_mdl.keras"
if not os.path.exists(best_model_path):
    raise FileNotFoundError(f"Trained model not found at {best_model_path}. Training may have failed.")

print(f"Loading trained multi-task model from: {best_model_path}")
trained_model = tf.keras.models.load_model(best_model_path, compile=False)

# Second, prepare a low-resolution test case (both image and segmentation).
print("Preparing low-resolution input image and segmentation...")
test_image_high_res = ants.image_read(test_files[0]).iMath("Normalize")

# Create a ground-truth segmentation from the high-res image.
# We'll use a simple threshold here for demonstration purposes.
segmentation_high_res = ants.threshold_image(test_image_high_res, "Otsu", 2).threshold_image(2, 2)

# Simulate the low-resolution inputs by downsampling both high-res sources.
low_res_spacing = [s * UPSAMPLE_FACTOR for s in test_image_high_res.spacing]
test_image_low_res = ants.resample_image(test_image_high_res, low_res_spacing, use_voxels=False, interp_type=0)
segmentation_low_res = ants.resample_image(segmentation_high_res, low_res_spacing, use_voxels=False, interp_type=1) # Use Nearest Neighbor for masks

# Now, run inference using both the low-res image and its corresponding segmentation.
print("Applying multi-task super-resolution model...")
inference_result = siq.inference(
    test_image_low_res,
    trained_model,
    segmentation=segmentation_low_res,
    verbose=True
)


# --- 6. Save All Outputs for Verification ---
# To verify the result, we save the full set of images.
print("\n--- Step 6: Saving All Images for Comparison ---")

# The output from inference with segmentation is a dictionary.
# Let's assume the keys are 'super_resolution' and 'super_resolution_segmentation'
# or handle the case where it might be a single image (if logic changes).
if isinstance(inference_result, dict):
    super_resolved_image = inference_result['super_resolution']
    super_resolved_seg = inference_result.get('super_resolution_segmentation', None) # Safely get seg
else: # Handle case of single-image output for robustness
    super_resolved_image = inference_result
    super_resolved_seg = None


# Define clear output paths
path_lr_image = OUTPUT_DIR / "test_input_low_res_image.nii.gz"
path_lr_seg = OUTPUT_DIR / "test_input_low_res_seg.nii.gz"
path_sr_image = OUTPUT_DIR / "test_output_super_res_image.nii.gz"
path_gt_image = OUTPUT_DIR / "test_ground_truth_high_res_image.nii.gz"
path_gt_seg = OUTPUT_DIR / "test_ground_truth_high_res_seg.nii.gz"

# Save the inputs and ground truths
ants.image_write(test_image_low_res, str(path_lr_image))
ants.image_write(segmentation_low_res, str(path_lr_seg))
ants.image_write(test_image_high_res, str(path_gt_image))
ants.image_write(segmentation_high_res, str(path_gt_seg))

# Save the model's outputs
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

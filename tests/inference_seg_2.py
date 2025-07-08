import ants
import keras
from siq import inference

# Load the super-resolution model
model_path = "siq_smallshort_train_2x2x2_1chan_featgraderL6_best.keras"
model = keras.models.load_model(model_path)

# Load the image
image = ants.image_read("t1_rand.nii.gz")
segmentation = ants.image_read("seg.nii.gz")

# Run inference WITH segmentation (automatically uses region-wise super-resolution if appropriate)
sr_result_all_labels = inference(image, model, 
    segmentation=segmentation, target_range=[0, 1],
    truncation=[0.00002, 0.9999], 
    dilation_amount=8,
    poly_order='hist', verbose=True)

# ants.image_write( sr_result_all_labels, "sr_with_seg.nii.gz")



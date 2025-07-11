# SIQ: Super-Resolution Image Quantification

[![PyPI version](https://badge.fury.io/py/siq.svg)](https://badge.fury.io/py/siq)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](httpss://github.com/your-repo/siq)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SIQ** is a powerful and flexible Python library for deep learning-based image super-resolution, with a focus on medical imaging applications. It provides a comprehensive toolkit for every stage of the super-resolution workflow, from data generation and model training to robust inference and evaluation.

The library is built on `TensorFlow/Keras` and `ANTSpy` and is designed to handle complex, real-world challenges such as **anisotropic super-resolution** (where different upsampling factors are needed for each axis) and **multi-task learning** (e.g., simultaneously upsampling an image and its segmentation mask).

---

## Key Features

SIQ is more than just an inference tool; it's a complete framework that facilitates:

*   **Flexible Data Generation:** Automatically create paired low-resolution and high-resolution patches for training with `siq.image_generator`. Includes support for multi-channel data (e.g., image + segmentation) via `siq.seg_generator`.
*   **Advanced Model Architectures:** Easily instantiate powerful Deep Back-Projection Networks (DBPN) for 2D and 3D with `siq.default_dbpn`, customized for any upsampling factor and multi-task outputs.
*   **Perceptual Loss Training:** Go beyond simple Mean Squared Error. SIQ includes tools for using pre-trained feature extractors (`siq.get_grader_feature_network`, `siq.pseudo_3d_vgg_features_unbiased`) to optimize for perceptual quality.
*   **Intelligent Loss Weighting:** Automatically balance complex, multi-component loss functions (e.g., MSE + Perceptual + Dice) with a single command (`siq.auto_weight_loss`, `siq.auto_weight_loss_seg`) to ensure stable training.
*   **End-to-End Training Pipelines:** Train models from start to finish with the high-level `siq.train` and `siq.train_seg` functions, which handle data generation, validation, and model saving.
*   **Robust Inference:** Apply your trained models to new images with `siq.inference`, including specialized logic for region-wise and blended super-resolution when guided by a segmentation mask.
*   **Comprehensive Evaluation:** Systematically benchmark and compare model performance with `siq.compare_models`, which calculates PSNR, SSIM, and Dice metrics against baseline methods.

---

## Installation

You can install the official release directly from PyPI:

```bash
pip install siq
```

To install the latest development version from this repository:

```bash
git clone https://github.com/stnava/siq.git
cd siq
pip install .
```

---

## Quick Start: A 5-Minute Example

The examples demonstrate the core workflow: training a model on publicly available data and using it for inference.

```bash
tests/test.py
tests/test_seg.py
```

## Pre-trained Models and Compatibility

We provide a collection of pre-trained models to get you started without requiring you to train from scratch.

*   **[Download Pre-trained Models from Figshare](https://figshare.com/articles/software/SIQ_reference_super_resolution_models/27079987)**

### Important Note on Keras/TensorFlow Versions

The deep learning ecosystem evolves quickly. Models saved with older versions of TensorFlow/Keras (as `.h5` files) may have trouble loading in newer versions (TF >= 2.16) due to the transition to the `.keras` format.

If you encounter issues loading a legacy `.h5` model, we provide a robust conversion script. This utility will convert your old `.h5` files into the modern `.keras` format.

**Usage:**

```python
import siq

# Define the directory containing your old .h5 models
source_dir = "~/.antspymm/" # Or wherever you downloaded the models
output_dir = "./converted_keras_models"

# Convert the models
siq.convert_h5_to_keras_format(
    search_directory=source_dir,
    output_directory=output_dir,
    exclude_patterns=["*weights.h5"] # Skips files that are just weights
)
```

After running this, you can load the converted models from the `converted_keras_models` directory using `siq.read_srmodel` or `tf.keras.models.load_model`.

---

## For Developers

### Setting Up the Environment

This package is tested with Python 3.11 and TensorFlow 2.17. For optimal CPU performance, especially on Linux, you may want to set these environment variables:

```bash
export TF_ENABLE_ONEDNN_OPTS=1
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=8
export TF_NUM_INTRAOP_THREADS=8
export TF_NUM_INTEROP_THREADS=8
```

### Publishing a New Release

To publish a new version of `siq` to PyPI:

```bash
# Ensure build and twine are installed
python3 -m pip install build twine

# Clean previous builds
rm -rf build/ siq.egg-info/ dist/

# Build the package
python3 -m build .

# Upload to PyPI
python3 -m twine upload --repository siq dist/*
```

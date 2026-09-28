# SIQ: Super-Resolution Image Quantification

[![PyPI version](https://badge.fury.io/py/siq.svg)](https://badge.fury.io/py/siq)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](httpss://github.com/your-repo/siq)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SIQ** is a powerful and flexible Python library for deep learning-based image super-resolution, with a focus on medical imaging applications. It provides a comprehensive toolkit for every stage of the super-resolution workflow, from data generation and model training to robust inference and evaluation.

The library is built on `TensorFlow/Keras` and `ANTSpy` and is designed to handle complex, real-world challenges such as **anisotropic super-resolution** (where different upsampling factors are needed for each axis) and **multi-task learning** (e.g., simultaneously upsampling an image and its segmentation mask).

---

Function-specific documentation is [here](https://stnava.github.io/siq/siq/get_data.html).

## New in Keras 3 Integration

- **Dual Backend Support:** Seamlessly switch between PyTorch and TensorFlow.
- **3D ESPCN Architecture:** High-performance super-resolution using Pixel Shuffling, optimized for Apple Silicon (MPS).
- **Patch-wise Inference:** Memory-efficient inference with Gaussian blending to eliminate stitching artifacts.
- **Blind Perceptual Training:** Synthetic data simulation for robust, general-purpose MRI enhancement.
- **Low-Latency AS-DBPN:** A highly optimized dual-recurrent back-projection model designed for fast, high-fidelity blind super-resolution.

## Key Innovations of the AS-DBPN Network

The **Attention-Guided Shared Deep Back-Projection Network (AS-DBPN)** is a state-of-the-art super-resolution architecture that achieves top-tier quality metrics while maintaining a highly optimized runtime profile:

1. **Lightweight Recurrent Back-Projection**: Uses a shared-weight recurrent loop with $T=4$ steps to iteratively project features between low- and high-resolution spaces, providing a parameter-efficient alternative to deep feedforward networks.
2. **Simplified Hybrid Attention**: Replaces computationally heavy inner-loop attention blocks with a single **Socrat/Channel Attention (SOCA/RCAN)** block at the final reconstruction layer. This eliminates redundant operations, resulting in a **4.4x speedup / 77% latency reduction** (MPS latency reduced from **261.13 ms to 59.05 ms**).
3. **Shared Layer Normalization**: Introduces shared `LayerNormalization` inside the recurrent loop to stabilize scaling dynamics and prevent gradient collapse during iterative feedback steps.
4. **Generalization Under Blind Conditions**: Trained via a curriculum of mixed geometries, Rician noise, and out-of-focus blur, the model is resilient to blind, ill-posed degradations, achieving **23.70 dB PSNR** on the `r16` brain validation patch (+0.84 dB over SAN) and ranking **#3 overall** on the Mixed Simulation Class benchmark.

## Evaluations & Visual Reports

GitHub does not render standalone `.html` files inline, so the interactive reports below are linked through [htmlpreview.github.io](https://htmlpreview.github.io), which renders them straight from this repo:

*   **[9-Class Simulation Benchmark: Overall Model Comparison](https://htmlpreview.github.io/?https://github.com/stnava/siq/blob/main/summary_results.html)** — interactive PSNR/SSIM/GMSD/HFEN/Correlation comparison across all 12 models (11 architectures + bilinear baseline) and 9 synthetic simulation classes (brain, vessels, fractal noise, sinewave, etc.).
*   **[`r16` Brain MRI Qualitative Comparison](https://htmlpreview.github.io/?https://github.com/stnava/siq/blob/main/r16_comparison.html)** — side-by-side visual comparison of ground truth, bilinear, SAN, and AS-DBPN reconstructions on the classic `r16` test image.
*   **[Simulated Example Gallery](https://htmlpreview.github.io/?https://github.com/stnava/siq/blob/main/docs/simulated_examples.html)** — low-res/high-res pairs across all 9 simulation classes used for blind training and evaluation.

Static summaries of the same benchmark are also available as Markdown tables: **[Class Performance Summary](class_performance_summary.md)** and **[Rank-of-Ranks & SRFBN Analysis](rank_of_ranks_and_srfbn_analysis.md)**.

### What the current results show

*   **Rank of Ranks (9-class benchmark, full retrain):** **REF-DBPN** ranks #1 overall (score 2.98), followed by **SAN** (#2, 3.67) and **LDBPN** (#3, 4.22). All 11 trained architectures beat the **Bilinear** baseline (10.58) by a wide margin.
*   **AS-DBPN (the new low-latency model)** trades a small amount of rank-score for a **4.4x speedup**: MPS latency drops from 261.13 ms to **59.05 ms**, while still beating SAN by **+0.84 dB PSNR** on the `r16` validation patch and landing **#3 overall** on the mixed-class benchmark — the best quality-per-millisecond tradeoff in the collection.
*   **Heavier back-projection models (REF-DBPN, LDBPN) consistently top the accuracy leaderboard** but cost far more latency (REF-DBPN: 648 ms vs. ~15–23 ms for the lightweight models), which motivated the AS-DBPN design.
*   **SRFBN bilinear-bypass fix**: SRFBN previously collapsed to an MSE-minimizing bilinear shortcut. Zero-initializing its `LearnableScale` skip parameter forced the recurrent conv branch to train from a genuine 15.53 dB start; the retrained model now shows fully active convolutional weights and **8–13% HFEN improvement** (sharper high-frequency detail) on structured classes like `sinewave`, `layered`, and `fractal_noise`.

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

**Note:** `tests/train_model_refinement.py` and `generate_summary_images.py` force `KERAS_BACKEND=torch` at import time (for MPS/CUDA acceleration via PyTorch), so refining/training a model requires `torch` to be installed in addition to the `requirements.txt` dependencies:

```bash
pip install torch
```

### Model Refinement & Fine-Tuning

To refine and fine-tune pre-trained models using our advanced mixed-modality simulation engine (supporting brain structures, sinewaves, layered strips, Rician noise, and coordinate zoom):

```bash
# Refine the Channel Attention ESPCN model (default: batch size 1)
python tests/train_model_refinement.py espcn --batch-size 1

# Refine the Lightweight DBPN model
python tests/train_model_refinement.py ldbpn --batch-size 2

# Refine the Reference DBPN model
python tests/train_model_refinement.py ref-dbpn --batch-size 1
```

Every model accepts `--dim {2,3}` (**default: 3**), so 3D training needs no extra flags:

```bash
# Refine the 3D AS-DBPN model (low-latency blind super-resolution)
python tests/train_model_refinement.py asdbpn --dim 3 --batch-size 1

# Equivalent to the above, since --dim defaults to 3
python tests/train_model_refinement.py asdbpn --batch-size 1
```

`espcn-rc` and `wdsr-rc` are 2D-only (resize-conv checkerboard-mitigation pilots) and will raise an error if run with `--dim 3`; every other model type (`espcn`, `ldbpn`, `ref-dbpn`, `wdsr`, `rcan`, `carn`, `srfbn`, `san`, `asdbpn`) supports both dimensionalities.

The script executes a curriculum-based training sequence across 3 stages:
1. **Stage 1: Adaptation Phase (Clean Mixed Geometries)**: 100 warmup iterations on clean mixed geometries.
2. **Stage 2: Robustness Fine-Tuning Phase (Low LR + Noise)**: 150 joint fine-tuning iterations introducing Rician noise.
3. **Stage 3: Dedicated Refinement Phase (High-Fidelity Brain Focus)**: 200 iterations focusing on high-fidelity brain anatomy structures with a very low learning rate ($1 \times 10^{-7}$).

**Intelligent Restart & Skipping:**
* If a refined model checkpoint already exists (e.g., `espcn_3d_attention_refined.keras`, `ldbpn_3d_refined.keras`, or `ref_dbpn_3d_refined.keras`), the script automatically loads it, **skips Stage 1 and Stage 2**, and proceeds directly to **Stage 3 (Dedicated Refinement)**.
* To train from scratch starting from the baseline model, delete or rename the existing refined `.keras` file in the workspace.

### Generating Evaluation Reports

`generate_summary_images.py` builds the interactive HTML comparison report (see [Evaluations & Visual Reports](#evaluations--visual-reports)) from whichever refined checkpoints are present in the repo root. It also accepts `--dim`:

```bash
# 2D report (default): reads *_2d_refined.keras, writes summary_results.html
python generate_summary_images.py

# 3D report: reads *_3d_refined.keras, writes summary_results_3d.html
python generate_summary_images.py --dim 3
```

Any model without a matching checkpoint for that dimensionality shows up as "TBD" in the report rather than causing an error, so it's safe to run before every model has been trained in 3D.

### Publishing a New Release

To publish a new version of `siq` to PyPI:

```bash
# Ensure build and twine are installed
python -m pip install build twine

# Clean previous builds
rm -rf build/ siq.egg-info/ dist/

# Build the package
python -m build .

# Upload to PyPI
python -m twine upload --repository siq dist/*
```

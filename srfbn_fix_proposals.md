# Proposals to Fix SRFBN Bilinear Bypass Collapse (Smoothness Failure)

This document analyzes the root cause of the **SRFBN smoothness failure** (where it acts as a simple bilinear bypass) and proposes four concrete solutions to resolve it.

---

## 1. Root Cause Analysis Recap

During our weight audit of `srfbn_2d_refined.keras`, we discovered that:
1. The global skip connection uses bilinear upsampling with a learnable scale initialized to `1.0`.
2. The recurrent layers share the same weights across 4 steps, introducing severe gradient vanishing issues during backpropagation.
3. **Warmup Gate Bug**: During the dedicated MSE Warmup Gate, training exits as soon as the model's validation PSNR reaches the bilinear baseline target. Since SRFBN starts with a scale of `1.0` on the bilinear skip connection, it achieves bilinear parity on **iteration 1**, instantly exiting the warmup phase before the recurrent feedback branch has learned anything.
4. Consequently, the optimizer takes the path of least resistance: it drives the recurrent convolutional weights to near-zero (`fb_up` mean: `-0.0013`, `fb_down` mean: `-0.0006`) and relies entirely on the bilinear shortcut (scale: `1.0004`).

---

## 2. Proposed Solutions

We have identified four viable solutions, ranked from easiest/most effective to more complex.

### Proposal A: Initialize Bypass Scale to `0.0` (Recommended)
* **Concept**: Initialize the global skip connection scale (`scaled_global_skip`) to `0.0` instead of `1.0` inside `create_srfbn_2d`.
* **Why it works**: By starting with `scale = 0.0`, the model cannot rely on the bilinear bypass to meet the Warmup Gate criteria. The validation PSNR will start very low (~10-12 dB) and the model will be forced to train the recurrent convolutional layers to reconstruct the image and reach the bilinear target (~22 dB).
* **Code Change**:
  ```diff
  # In siq/espcn.py (line 897)
  -        scaled_skip = LearnableScale(initial_value=1.0, name="scaled_global_skip")(skip)
  +        scaled_skip = LearnableScale(initial_value=0.0, name="scaled_global_skip")(skip)
  ```

---

### Proposal B: Freeze Bypass during Warmup & Stage 1
* **Concept**: Keep the global skip scale initialized to `1.0` (or `0.0`) but freeze the bypass layer (`trainable=False` or multiply by constant `0.0`) during the Warmup Gate and Stage 1 adaptation.
* **Why it works**: Forces the convolutional/recurrent branch to learn image upsampling and edge features under pure MSE and early hybrid loss. The bypass is only unfrozen in Stage 2 (LOESS tuning) to add global structural alignment.
* **Code Change**:
  ```python
  # In tests/train_model_refinement.py (during Warmup Gate)
  for layer in model.layers:
      if "scaled_global_skip" in layer.name:
          layer.trainable = False
  ```

---

### Proposal C: Gradient Clipping & Recurrent Layer Normalization
* **Concept**: Stabilize the gradient flow through the recurrent loops.
  1. Add gradient clipping (`clipnorm=1.0` or `clipvalue=1.0`) to the Adam optimizer configuration.
  2. Insert a `LayerNormalization` or `GroupNormalization` layer after each recurrent feedback step to stabilize hidden state magnitude.
* **Why it works**: Minimizes the vanishing/exploding gradient variance, making the recurrent convolutional path easier to optimize and reducing the likelihood of the optimizer discarding it in favor of the skip connection.
* **Code Change**:
  ```diff
  # In tests/train_model_refinement.py
  - model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-5), loss=hybrid_loss)
  + model.compile(optimizer=keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0), loss=hybrid_loss)
  ```

---

### Proposal D: Replace Bilinear Skip with PixelShuffle/Convolutional Skip
* **Concept**: Replace the raw bilinear global skip connection with a learnable convolutional shortcut (e.g., a $1 \times 1$ conv followed by PixelShuffle, or a separate Conv2DTranspose).
* **Why it works**: Because the shortcut contains learnable weights, it does not act as a free identity bypass from day one. The optimizer must train the shortcut weights, removing the easy cheat path.
* **Code Change**:
  ```diff
  # In siq/espcn.py (line 896)
  -        skip = layers.UpSampling2D(size=(factor, factor), interpolation="bilinear", name="global_skip")(inputs)
  -        scaled_skip = LearnableScale(initial_value=1.0, name="scaled_global_skip")(skip)
  +        # Learnable sub-pixel mapping shortcut
  +        skip = layers.Conv2D(factor**2, kernel_size=3, padding="same", name="skip_conv")(inputs)
  +        scaled_skip = PixelShuffle2D(factor=factor, name="skip_shuffle")(skip)
  ```

---

## 3. Comparison of Solutions

| Solution | Implementation Complexity | Expected Sharpness | Risk of Instability | Recommendation |
| :--- | :---: | :---: | :---: | :---: |
| **A: Init Scale to 0.0** | **Low** (1 line) | **High** | Low | **(Recommended)** |
| **B: Freeze Bypass** | **Medium** (10 lines) | **High** | Low | Good Alternative |
| **C: Gradient Clipping** | **Low** (1 line) | **Medium** | Low | Complementary |
| **D: PixelShuffle Skip** | **Medium** | **High** | Low | Solid, but changes architecture |

---

## 4. Discussion & Plan

If we proceed with **Proposal A** (the cleanest and most direct fix):
1. We will update `siq/espcn.py` to initialize the global skip connection scale to `0.0` for SRFBN.
2. We will clean existing SRFBN checkpoints.
3. We will run the retraining script for SRFBN. The Warmup Gate will now run for several hundred iterations (as it has to learn how to upsample from scratch to hit ~22 dB), and the recurrent branch weights will become active and produce sharp details.
4. We will regenerate the html summary and class performance reports to see if the sharpness and HFEN metric improve.

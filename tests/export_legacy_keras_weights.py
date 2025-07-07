"""
Migration Step 1: Export Legacy Model Weights.

This script should be run in an environment with legacy Keras (tf.keras)
installed and accessible, typically by setting the environment variable:
`export TF_USE_LEGACY_KERAS=1`

It performs the following actions:
1. Scans a directory for all legacy model files (e.g., 'siq*mdl.h5').
2. For each model, it loads the full model structure.
3. It saves ONLY the model's weights to a new file with a '.weights.h5' suffix.
   This weights file is portable across Keras versions.
"""
import os
from pathlib import Path

# --- Configuration ---
# Ensure we are using legacy Keras for this script
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow as tf
print(f"Using Keras version (legacy)")

# Define the location of your legacy models
MODEL_DIR = Path(os.path.expanduser("~/.antspymm/"))
MODEL_GLOB_PATTERN = "siq*mdl.h5"

def export_weights_from_legacy_models():
    """Finds legacy models and saves their weights."""
    print(f"Searching for models in: {MODEL_DIR}")
    legacy_model_paths = list(MODEL_DIR.glob(MODEL_GLOB_PATTERN))

    if not legacy_model_paths:
        print("No legacy models found matching the pattern. Exiting.")
        return

    print(f"Found {len(legacy_model_paths)} models to process.")
    
    for model_path in legacy_model_paths:
        try:
            # Generate the new filename for the weights
            weights_path = model_path.with_suffix('').with_suffix('.weights.h5')
            
            print(f"\nProcessing: {model_path.name}")
            print(f" -> Loading legacy model...")
            
            # Load the full legacy model
            model = tf.keras.models.load_model(model_path, compile=False)
            
            # Save only the weights - this is the key step
            model.save_weights(weights_path)
            
            print(f" ==> Successfully saved weights to: {weights_path.name}")

        except Exception as e:
            print(f" !! ERROR processing {model_path.name}: {e}")
            print(" !! Skipping this model.")

if __name__ == "__main__":
    export_weights_from_legacy_models()
    print("\nWeight export process complete.")

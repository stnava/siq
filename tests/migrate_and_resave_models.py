"""
Migration Step 2: Reconstruct, Load Weights, and Re-save in Keras 3.

This script should be run in an environment with modern Keras 3.
`export TF_USE_LEGACY_KERAS=0`

This version is updated to handle both 1-channel and 2-channel models
and uses the correct function signature for model reconstruction.

It performs the following actions:
1. Scans a directory for the '.weights.h5' files created by Script 1.
2. For each weights file, it parses the filename to determine the original
   model's architecture parameters (shape, channels, etc.).
3. It programmatically reconstructs the model architecture from scratch using
   the correct function signature.
4. It loads the saved weights into this new model instance.
5. It saves the complete, functional model in the new Keras 3 native
   format ('.keras'), which is ready for future use.
"""
import os
import re
from pathlib import Path

# --- Configuration ---
# Ensure we are using modern Keras 3 for this script
os.environ['TF_USE_LEGACY_KERAS'] = '0'
import keras
import siq  # Assuming `siq` module with `default_dbpn` is available
print(f"Using Keras version (modern): {keras.__version__}")

# Define the location of your models and weights
MODEL_DIR = Path(os.path.expanduser("~/.antspymm/"))
WEIGHTS_GLOB_PATTERN = "siq*weights.h5"

# --- Helper Functions for Modularity ---

def parse_model_filename(filename: str) -> dict | None:
    """
    Parses a model filename to extract its architectural parameters.
    """
    params = {}
    try:
        OPTION_PATTERN = r'siq_(\w+?)short'
        SHAPE_PATTERN = r'(\d+)x(\d+)x(\d+)'
        NCHAN_PATTERN = r'_(\d+)chan'

        option_match = re.search(OPTION_PATTERN, filename)
        params['option'] = option_match.group(1) if option_match else None

        shape_match = re.search(SHAPE_PATTERN, filename)
        # This parsed 'shape' will be passed to the 'strider' argument
        params['shape'] = [int(g) for g in shape_match.groups()] if shape_match else None

        nchan_match = re.search(NCHAN_PATTERN, filename)
        params['nchan'] = int(nchan_match.group(1)) if nchan_match else None

        if params['shape'] is None or params['nchan'] is None:
            raise ValueError("Could not parse essential shape or channel info.")
            
        return params
    except (AttributeError, ValueError) as e:
        print(f"  -> Warning: Could not parse parameters from filename '{filename}'. Reason: {e}")
        return None

def reconstruct_siq_model(params: dict) -> keras.Model:
    """
    Reconstructs an empty SIQ model instance from parsed parameters using
    the correct function signature.
    """
    n_channels = params.get('nchan', 1)
    
    # *** THIS IS THE CORRECTED FUNCTION CALL ***
    # The parsed 'shape' is now correctly passed to the 'strider' argument.
    return siq.default_dbpn(
        strider=params['shape'],
        option=params['option'],
        nChannelsIn=n_channels,
        nChannelsOut=n_channels,
        sigmoid_second_channel=(n_channels == 2)
    )

# --- Main Execution Logic ---

def migrate_and_resave_models():
    """Finds weights files, reconstructs models, and saves them in Keras 3 format."""
    print(f"Searching for weights files in: {MODEL_DIR}")
    weights_paths = list(MODEL_DIR.glob(WEIGHTS_GLOB_PATTERN))

    if not weights_paths:
        print("No '.weights.h5' files found. Please run Script 1 first. Exiting.")
        return

    print(f"Found {len(weights_paths)} weights files to migrate.")
    
    for weights_path in weights_paths:
        print(f"\nMigrating: {weights_path.name}")
        
        # 1. Parse parameters from filename
        params = parse_model_filename(weights_path.name)
        if not params:
            print(" !! Skipping this file due to parsing error.")
            continue
            
        print(f"  -> Parsed params: option={params['option']}, strider_shape={params['shape']}, nchan={params['nchan']}")

        try:
            # 2. Reconstruct the model architecture using the corrected function call
            print("  -> Reconstructing empty model architecture...")
            model = reconstruct_siq_model(params)
            
            # 3. Load the legacy weights.
            print(f"  -> Loading weights from {weights_path.name}...")
            model.load_weights(weights_path)
            
            # 4. (Optional but recommended) Compile the model.
            model.compile()

            # 5. Save the full, functional model in the modern .keras format
            new_filename = f"{weights_path.stem.replace('_weights', '')}.keras"
            keras3_model_path = weights_path.with_name(new_filename)
            model.save(keras3_model_path)
            
            print(f" ==> Successfully migrated and saved new model to: {keras3_model_path.name}")
            
        except Exception as e:
            print(f" !! ERROR during migration for {weights_path.name}: {e}")
            print(" !! This often indicates an architecture mismatch between the legacy model "
                  "and the newly reconstructed one. Verify that `siq.default_dbpn` with the "
                  "current parameters creates the exact same layers and layer names.")
            print(" !! Skipping this model.")

if __name__ == "__main__":
    migrate_and_resave_models()
    print("\nModel migration process complete.")
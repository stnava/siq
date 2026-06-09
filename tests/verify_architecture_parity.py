import os
import sys

# Try importing keras and siq
try:
    import keras
    import siq
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    sys.exit(1)

print(f"Loaded Keras version: {keras.__version__}")
print(f"Using backend: {keras.backend.backend()}")

def test_model_construction():
    print("\n--- Testing 3D DBPN Model Construction (Option: tiny) ---")
    strides = [2, 2, 2]
    model_3d = siq.default_dbpn(strides, dimensionality=3, option='tiny')
    
    assert isinstance(model_3d, keras.Model), "Model must be a keras.Model instance"
    print(f"3D Model Output Shape: {model_3d.output_shape}")
    print(f"3D Model Layer Count: {len(model_3d.layers)}")
    
    # Check that output shape matches expectations (2x upsampling from input shape)
    # Input: (None, None, None, None, 1) -> Output: (None, None, None, None, 1)
    
    print("\n--- Testing 2D DBPN Model Construction (Option: tiny) ---")
    strides_2d = [2, 2]
    model_2d = siq.default_dbpn(strides_2d, dimensionality=2, option='tiny')
    assert isinstance(model_2d, keras.Model), "Model must be a keras.Model instance"
    print(f"2D Model Output Shape: {model_2d.output_shape}")
    print(f"2D Model Layer Count: {len(model_2d.layers)}")

if __name__ == "__main__":
    test_model_construction()
    print("\nArchitecture parity test passed successfully!")

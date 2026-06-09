import keras
from keras import layers, ops

def pixel_shuffle_3d(inputs, factor=2):
    """
    Implementation of 3D Pixel Shuffle for Keras/Keras3.
    Args:
        inputs: (batch, d, h, w, c)
        factor: scaling factor (integer)
    """
    input_shape = ops.shape(inputs)
    batch_size = input_shape[0]
    d, h, w = input_shape[1], input_shape[2], input_shape[3]
    channels = input_shape[4]
    
    new_channels = channels // (factor ** 3)
    
    # Reshape: (batch, d, h, w, f, f, f, new_c)
    x = ops.reshape(inputs, (batch_size, d, h, w, factor, factor, factor, new_channels))
    
    # Transpose to (batch, d, f, h, f, w, f, new_c)
    x = ops.transpose(x, (0, 1, 4, 2, 5, 3, 6, 7))
    
    # Reshape to (batch, d*f, h*f, w*f, new_c)
    new_d, new_h, new_w = d * factor, h * factor, w * factor
    return ops.reshape(x, (batch_size, new_d, new_h, new_w, new_channels))

@keras.saving.register_keras_serializable(package="siq")
class PixelShuffle3D(layers.Layer):
    def __init__(self, factor=2, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, inputs):
        return pixel_shuffle_3d(inputs, self.factor)

    def compute_output_shape(self, input_shape):
        factor = self.factor
        d = input_shape[1] * factor if input_shape[1] is not None else None
        h = input_shape[2] * factor if input_shape[2] is not None else None
        w = input_shape[3] * factor if input_shape[3] is not None else None
        c = input_shape[4] // (factor ** 3) if input_shape[4] is not None else None
        return (input_shape[0], d, h, w, c)

def create_espcn_3d(input_shape=(None, None, None, 1), factor=2, n_filters=64):
    """
    Creates a 3D ESPCN model.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Feature extraction
    x = layers.Conv3D(n_filters, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.Conv3D(n_filters // 2, kernel_size=3, padding="same", activation="relu")(x)
    
    # Last layer before shuffle must have factor**3 channels
    # To output 1 channel, we need factor**3 channels here
    out_channels = 1
    x = layers.Conv3D(out_channels * (factor ** 3), kernel_size=3, padding="same")(x)
    
    # Pixel Shuffle
    outputs = PixelShuffle3D(factor=factor)(x)
    
    return keras.Model(inputs, outputs, name="espcn_3d")

def create_espcn_3d_residual(input_shape=(None, None, None, 1), factor=2, n_filters=64, n_res_blocks=4):
    """
    Creates a 3D Residual ESPCN model.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Initial projection
    x = layers.Conv3D(n_filters, kernel_size=3, padding="same", activation="relu")(inputs)
    
    # Residual blocks
    for i in range(n_res_blocks):
        res = layers.Conv3D(n_filters, kernel_size=3, padding="same", activation="relu")(x)
        res = layers.Conv3D(n_filters, kernel_size=3, padding="same")(res)
        x = layers.add([x, res])
        x = layers.Activation("relu")(x)
        
    # Shrinking
    x = layers.Conv3D(n_filters // 2, kernel_size=3, padding="same", activation="relu")(x)
    
    # Last conv before shuffle: output 32 * factor**3 channels
    out_channels = 32
    x = layers.Conv3D(out_channels * (factor ** 3), kernel_size=3, padding="same")(x)
    
    # Pixel Shuffle transition to high-resolution space
    x = PixelShuffle3D(factor=factor)(x)
    
    # Non-linear processing in high-resolution space
    x = layers.Conv3D(32, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.Conv3D(16, kernel_size=3, padding="same", activation="relu")(x)
    outputs = layers.Conv3D(1, kernel_size=3, padding="same")(x)
    
    return keras.Model(inputs, outputs, name="espcn_3d_residual")



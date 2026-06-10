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

@keras.saving.register_keras_serializable(package="siq")
class LearnableScale(layers.Layer):
    """
    Keras layer that multiplies input by a learnable scalar,
    initialized to a constant value. Used for neutral skip connections.
    """
    def __init__(self, initial_value=0.0, **kwargs):
        super().__init__(**kwargs)
        self.initial_value = initial_value

    def build(self, input_shape):
        self.scale = self.add_weight(
            shape=(),
            initializer=keras.initializers.Constant(self.initial_value),
            trainable=True,
            name="scale"
        )

    def call(self, inputs):
        return inputs * self.scale
        
    def get_config(self):
        config = super().get_config()
        config.update({"initial_value": self.initial_value})
        return config

def channel_attention_block(input_tensor, reduction_ratio=16, name_prefix=""):
    channels = input_tensor.shape[-1]
    squeeze = layers.GlobalAveragePooling3D(name=f"{name_prefix}_ca_squeeze")(input_tensor)
    squeeze = layers.Reshape((1, 1, 1, channels), name=f"{name_prefix}_ca_reshape")(squeeze)
    
    excitation = layers.Conv3D(channels // reduction_ratio, kernel_size=1, activation='relu', name=f"{name_prefix}_ca_conv1")(squeeze)
    excitation = layers.Conv3D(channels, kernel_size=1, activation='sigmoid',
                               kernel_initializer='zeros', bias_initializer='ones', name=f"{name_prefix}_ca_conv2")(excitation)
    
    return layers.Multiply(name=f"{name_prefix}_ca_scale")([input_tensor, excitation])

def create_espcn_3d_attention(input_shape=(None, None, None, 1), factor=2, n_filters=64, n_res_blocks=8, use_global_skip=True):
    """
    Creates a 3D Residual ESPCN model with Channel Attention and optional Global Skip.
    Initial attention weights output 1.0, and the global skip weight outputs 0.0,
    enabling perfect identity initialization from standard ESPCN weights.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Initial projection
    x = layers.Conv3D(n_filters, kernel_size=3, padding="same", activation="relu", name="init_conv")(inputs)
    
    # Residual blocks with Channel Attention
    for i in range(n_res_blocks):
        res = layers.Conv3D(n_filters, kernel_size=3, padding="same", activation="relu", name=f"res_{i}_conv1")(x)
        res = layers.Conv3D(n_filters, kernel_size=3, padding="same", name=f"res_{i}_conv2")(res)
        res = channel_attention_block(res, reduction_ratio=16, name_prefix=f"res_{i}")
        x = layers.add([x, res], name=f"res_{i}_add")
        x = layers.Activation("relu", name=f"res_{i}_relu")(x)
        
    # Shrinking
    x = layers.Conv3D(n_filters // 2, kernel_size=3, padding="same", activation="relu", name="shrink_conv")(x)
    
    # Last conv before shuffle
    out_channels = 32
    x = layers.Conv3D(out_channels * (factor ** 3), kernel_size=3, padding="same", name="preshuffle_conv")(x)
    
    # Pixel Shuffle
    outputs = PixelShuffle3D(factor=factor, name="pixel_shuffle")(x)
    
    # Non-linear processing in high-res space
    x = layers.Conv3D(32, kernel_size=3, padding="same", activation="relu", name="hr_conv1")(outputs)
    x = layers.Conv3D(16, kernel_size=3, padding="same", activation="relu", name="hr_conv2")(x)
    outputs = layers.Conv3D(1, kernel_size=3, padding="same", name="hr_conv3")(x)
    
    # Optional Global Skip connection
    if use_global_skip:
        skip = layers.UpSampling3D(size=(factor, factor, factor), name="global_skip")(inputs)
        # Learnable scale initialized to 0.0
        scaled_skip = LearnableScale(initial_value=0.0, name="scaled_global_skip")(skip)
        outputs = layers.add([outputs, scaled_skip], name="add_global_skip")
        
    return keras.Model(inputs, outputs, name="espcn_3d_attention")

def up_projection_unit(lr_input, n_filters, factor=2, name_prefix=""):
    x = layers.Conv3D(n_filters * (factor ** 3), kernel_size=3, padding="same", activation="relu", name=f"{name_prefix}_up_conv1")(lr_input)
    h_temp = PixelShuffle3D(factor=factor, name=f"{name_prefix}_up_shuffle1")(x)
    
    l_temp = layers.Conv3D(n_filters, kernel_size=3, strides=factor, padding="same", activation="relu", name=f"{name_prefix}_up_down1")(h_temp)
    e_lr = layers.subtract([lr_input, l_temp], name=f"{name_prefix}_up_sub")
    
    x_err = layers.Conv3D(n_filters * (factor ** 3), kernel_size=3, padding="same", activation="relu", name=f"{name_prefix}_up_conv2")(e_lr)
    e_hr = PixelShuffle3D(factor=factor, name=f"{name_prefix}_up_shuffle2")(x_err)
    
    return layers.add([h_temp, e_hr], name=f"{name_prefix}_up_add")

def down_projection_unit(hr_input, n_filters, factor=2, name_prefix=""):
    l_temp = layers.Conv3D(n_filters, kernel_size=3, strides=factor, padding="same", activation="relu", name=f"{name_prefix}_down_conv1")(hr_input)
    
    x = layers.Conv3D(n_filters * (factor ** 3), kernel_size=3, padding="same", activation="relu", name=f"{name_prefix}_down_conv2")(l_temp)
    h_temp = PixelShuffle3D(factor=factor, name=f"{name_prefix}_down_shuffle1")(x)
    
    e_hr = layers.subtract([hr_input, h_temp], name=f"{name_prefix}_down_sub")
    e_lr = layers.Conv3D(n_filters, kernel_size=3, strides=factor, padding="same", activation="relu", name=f"{name_prefix}_down_conv3")(e_hr)
    
    return layers.add([l_temp, e_lr], name=f"{name_prefix}_down_add")

def create_ldbpn_3d(input_shape=(None, None, None, 1), factor=2, n_filters=64, n_stages=3):
    """
    Creates a Lightweight 3D Deep Back-Projection Network (L-DBPN) using PixelShuffle3D.
    """
    inputs = layers.Input(shape=input_shape)
    l0 = layers.Conv3D(n_filters, kernel_size=3, padding="same", activation="relu", name="init_conv")(inputs)
    
    l_list = [l0]
    h_list = []
    
    for i in range(n_stages):
        if len(l_list) > 1:
            l_cat = layers.concatenate(l_list, name=f"l_cat_{i}")
            l_proj = layers.Conv3D(n_filters, kernel_size=1, padding="same", activation="relu", name=f"l_proj_{i}")(l_cat)
        else:
            l_proj = l_list[0]
            
        h = up_projection_unit(l_proj, n_filters, factor, name_prefix=f"stage_{i}")
        h_list.append(h)
        
        if len(h_list) > 1:
            h_cat = layers.concatenate(h_list, name=f"h_cat_{i}")
            h_proj = layers.Conv3D(n_filters, kernel_size=1, padding="same", activation="relu", name=f"h_proj_{i}")(h_cat)
        else:
            h_proj = h_list[0]
            
        l = down_projection_unit(h_proj, n_filters, factor, name_prefix=f"stage_{i}")
        l_list.append(l)
        
    h_final_cat = layers.concatenate(h_list, name="h_final_cat")
    x = layers.Conv3D(n_filters, kernel_size=3, padding="same", activation="relu", name="recon_conv1")(h_final_cat)
    outputs = layers.Conv3D(1, kernel_size=3, padding="same", name="recon_conv2")(x)
    
    return keras.Model(inputs, outputs, name="ldbpn_3d")

def transfer_espcn_weights(src_model, dst_model):
    """
    Transfers weights from standard Residual ESPCN model (src_model)
    to a Channel Attention / Skip-enabled ESPCN model (dst_model) by shape.
    """
    src_convs = [l for l in src_model.layers if isinstance(l, layers.Conv3D)]
    dst_convs = [l for l in dst_model.layers if isinstance(l, layers.Conv3D)]
    
    # Skip attention 1x1x1 convs
    dst_main_convs = [l for l in dst_convs if l.kernel_size != (1, 1, 1)]
    
    matched = 0
    for src_l, dst_l in zip(src_convs, dst_main_convs):
        src_w = src_l.get_weights()
        dst_w = dst_l.get_weights()
        if len(src_w) > 0 and len(dst_w) > 0:
            if src_w[0].shape == dst_w[0].shape:
                dst_l.set_weights(src_w)
                matched += 1
    return matched

def transfer_dbpn_weights(src_model, dst_model):
    """
    Transfers weights from legacy DBPN model (src_model)
    to Lightweight DBPN model (dst_model) by shape.
    """
    src_convs = [l for l in src_model.layers if isinstance(l, layers.Conv3D)]
    dst_convs = [l for l in dst_model.layers if isinstance(l, layers.Conv3D)]
    
    matched = 0
    for src_l in src_convs:
        src_w = src_l.get_weights()
        if len(src_w) == 0:
            continue
        for dst_l in dst_convs:
            dst_w = dst_l.get_weights()
            if len(dst_w) > 0 and not hasattr(dst_l, "_weight_transferred"):
                if src_w[0].shape == dst_w[0].shape:
                    dst_l.set_weights(src_w)
                    dst_l._weight_transferred = True
                    matched += 1
                    break
                    
    for dst_l in dst_convs:
        if hasattr(dst_l, "_weight_transferred"):
            delattr(dst_l, "_weight_transferred")
            
    return matched

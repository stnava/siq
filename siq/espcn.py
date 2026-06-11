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
    def __init__(self, initial_value=1.0, **kwargs):
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
    
    excitation = layers.Conv3D(max(1, channels // reduction_ratio), kernel_size=1, activation='relu', name=f"{name_prefix}_ca_conv1")(squeeze)
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
        # Learnable scale initialized to 1.0
        scaled_skip = LearnableScale(initial_value=1.0, name="scaled_global_skip")(skip)
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

# ==============================================================================
# 2D Super-Resolution Models
# ==============================================================================

def pixel_shuffle_2d(inputs, factor=2):
    """
    Implementation of 2D Pixel Shuffle for Keras/Keras3.
    Args:
        inputs: (batch, h, w, c)
        factor: scaling factor (integer)
    """
    input_shape = ops.shape(inputs)
    batch_size = input_shape[0]
    h, w = input_shape[1], input_shape[2]
    channels = input_shape[3]
    
    new_channels = channels // (factor ** 2)
    
    # Reshape: (batch, h, w, f, f, new_c)
    x = ops.reshape(inputs, (batch_size, h, w, factor, factor, new_channels))
    
    # Transpose to (batch, h, f, w, f, new_c)
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    
    # Reshape to (batch, h*f, w*f, new_c)
    new_h, new_w = h * factor, w * factor
    return ops.reshape(x, (batch_size, new_h, new_w, new_channels))

@keras.saving.register_keras_serializable(package="siq")
class PixelShuffle2D(keras.layers.Layer):
    def __init__(self, factor=2, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, inputs):
        return pixel_shuffle_2d(inputs, self.factor)

    def compute_output_shape(self, input_shape):
        factor = self.factor
        h = input_shape[1] * factor if input_shape[1] is not None else None
        w = input_shape[2] * factor if input_shape[2] is not None else None
        c = input_shape[3] // (factor ** 2) if input_shape[3] is not None else None
        return (input_shape[0], h, w, c)

def create_espcn_2d_attention(input_shape=(None, None, 1), factor=2, n_filters=64, n_res_blocks=8, use_global_skip=True):
    """
    Creates a 2D Residual ESPCN model with Channel Attention and optional Global Skip.
    Identical design to the 3D version but adapted for 2D.
    """
    inputs = keras.layers.Input(shape=input_shape)
    
    # Initial projection
    x = keras.layers.Conv2D(n_filters, kernel_size=3, padding="same", activation="relu", name="init_conv")(inputs)
    
    # Residual blocks with Channel Attention
    for i in range(n_res_blocks):
        res = keras.layers.Conv2D(n_filters, kernel_size=3, padding="same", activation="relu", name=f"res_{i}_conv1")(x)
        res = keras.layers.Conv2D(n_filters, kernel_size=3, padding="same", name=f"res_{i}_conv2")(res)
        
        # 2D Channel Attention
        channels = res.shape[-1]
        squeeze = keras.layers.GlobalAveragePooling2D(name=f"res_{i}_ca_squeeze")(res)
        squeeze = keras.layers.Reshape((1, 1, channels), name=f"res_{i}_ca_reshape")(squeeze)
        excitation = keras.layers.Conv2D(max(1, channels // 16), kernel_size=1, activation='relu', name=f"res_{i}_ca_conv1")(squeeze)
        excitation = keras.layers.Conv2D(channels, kernel_size=1, activation='sigmoid',
                                         kernel_initializer='zeros', bias_initializer='ones', name=f"res_{i}_ca_conv2")(excitation)
        res = keras.layers.Multiply(name=f"res_{i}_ca_scale")([res, excitation])
        
        x = keras.layers.add([x, res], name=f"res_{i}_add")
        x = keras.layers.Activation("relu", name=f"res_{i}_relu")(x)
        
    # Shrinking
    x = keras.layers.Conv2D(n_filters // 2, kernel_size=3, padding="same", activation="relu", name="shrink_conv")(x)
    
    # Last conv before shuffle
    out_channels = 32
    x = keras.layers.Conv2D(out_channels * (factor ** 2), kernel_size=3, padding="same", name="preshuffle_conv")(x)
    
    # Pixel Shuffle
    outputs = PixelShuffle2D(factor=factor, name="pixel_shuffle")(x)
    
    # Non-linear processing in high-res space
    x = keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu", name="hr_conv1")(outputs)
    x = keras.layers.Conv2D(16, kernel_size=3, padding="same", activation="relu", name="hr_conv2")(x)
    outputs = keras.layers.Conv2D(1, kernel_size=3, padding="same", name="hr_conv3")(x)
    
    # Optional Global Skip connection
    if use_global_skip:
        skip = keras.layers.UpSampling2D(size=(factor, factor), name="global_skip")(inputs)
        scaled_skip = LearnableScale(initial_value=1.0, name="scaled_global_skip")(skip)
        outputs = keras.layers.add([outputs, scaled_skip], name="add_global_skip")
        
    return keras.Model(inputs, outputs, name="espcn_2d_attention")

def up_projection_unit_2d(lr_input, n_filters, factor=2, name_prefix=""):
    x = keras.layers.Conv2D(n_filters * (factor ** 2), kernel_size=3, padding="same", activation="relu", name=f"{name_prefix}_up_conv1")(lr_input)
    h_temp = PixelShuffle2D(factor=factor, name=f"{name_prefix}_up_shuffle1")(x)
    
    l_temp = keras.layers.Conv2D(n_filters, kernel_size=3, strides=factor, padding="same", activation="relu", name=f"{name_prefix}_up_down1")(h_temp)
    e_lr = keras.layers.subtract([lr_input, l_temp], name=f"{name_prefix}_up_sub")
    
    x_err = keras.layers.Conv2D(n_filters * (factor ** 2), kernel_size=3, padding="same", activation="relu", name=f"{name_prefix}_up_conv2")(e_lr)
    e_hr = PixelShuffle2D(factor=factor, name=f"{name_prefix}_up_shuffle2")(x_err)
    
    return keras.layers.add([h_temp, e_hr], name=f"{name_prefix}_up_add")

def down_projection_unit_2d(hr_input, n_filters, factor=2, name_prefix=""):
    l_temp = keras.layers.Conv2D(n_filters, kernel_size=3, strides=factor, padding="same", activation="relu", name=f"{name_prefix}_down_conv1")(hr_input)
    
    x = keras.layers.Conv2D(n_filters * (factor ** 2), kernel_size=3, padding="same", activation="relu", name=f"{name_prefix}_down_conv2")(l_temp)
    h_temp = PixelShuffle2D(factor=factor, name=f"{name_prefix}_down_shuffle1")(x)
    
    e_hr = keras.layers.subtract([hr_input, h_temp], name=f"{name_prefix}_down_sub")
    e_lr = keras.layers.Conv2D(n_filters, kernel_size=3, strides=factor, padding="same", activation="relu", name=f"{name_prefix}_down_conv3")(e_hr)
    
    return keras.layers.add([l_temp, e_lr], name=f"{name_prefix}_down_add")

def create_ldbpn_2d(input_shape=(None, None, 1), factor=2, n_filters=64, n_stages=3):
    """
    Creates a Lightweight 2D Deep Back-Projection Network (L-DBPN) using PixelShuffle2D.
    """
    inputs = keras.layers.Input(shape=input_shape)
    l0 = keras.layers.Conv2D(n_filters, kernel_size=3, padding="same", activation="relu", name="init_conv")(inputs)
    
    l_list = [l0]
    h_list = []
    
    for i in range(n_stages):
        if len(l_list) > 1:
            l_cat = keras.layers.concatenate(l_list, name=f"l_cat_{i}")
            l_proj = keras.layers.Conv2D(n_filters, kernel_size=1, padding="same", activation="relu", name=f"l_proj_{i}")(l_cat)
        else:
            l_proj = l_list[0]
            
        h = up_projection_unit_2d(l_proj, n_filters, factor, name_prefix=f"stage_{i}")
        h_list.append(h)
        
        if len(h_list) > 1:
            h_cat = keras.layers.concatenate(h_list, name=f"h_cat_{i}")
            h_proj = keras.layers.Conv2D(n_filters, kernel_size=1, padding="same", activation="relu", name=f"h_proj_{i}")(h_cat)
        else:
            h_proj = h_list[0]
            
        l = down_projection_unit_2d(h_proj, n_filters, factor, name_prefix=f"stage_{i}")
        l_list.append(l)
        
    h_final_cat = keras.layers.concatenate(h_list, name="h_final_cat")
    x = keras.layers.Conv2D(n_filters, kernel_size=3, padding="same", activation="relu", name="recon_conv1")(h_final_cat)
    outputs = keras.layers.Conv2D(1, kernel_size=3, padding="same", name="recon_conv2")(x)
    
    return keras.Model(inputs, outputs, name="ldbpn_2d")

# ==============================================================================
# WDSR (Wide Activation Super-Resolution) Models
# ==============================================================================

def wdsr_block_2d(x, n_filters, expansion_ratio=4, name_prefix=""):
    expanded_filters = n_filters * expansion_ratio
    res = keras.layers.Conv2D(expanded_filters, kernel_size=3, padding="same", name=f"{name_prefix}_conv1")(x)
    res = keras.layers.Activation("relu", name=f"{name_prefix}_relu")(res)
    res = keras.layers.Conv2D(n_filters, kernel_size=3, padding="same", name=f"{name_prefix}_conv2")(res)
    return keras.layers.add([x, res], name=f"{name_prefix}_add")

def wdsr_block_3d(x, n_filters, expansion_ratio=4, name_prefix=""):
    expanded_filters = n_filters * expansion_ratio
    res = keras.layers.Conv3D(expanded_filters, kernel_size=3, padding="same", name=f"{name_prefix}_conv1")(x)
    res = keras.layers.Activation("relu", name=f"{name_prefix}_relu")(res)
    res = keras.layers.Conv3D(n_filters, kernel_size=3, padding="same", name=f"{name_prefix}_conv2")(res)
    return keras.layers.add([x, res], name=f"{name_prefix}_add")

def create_wdsr_2d(input_shape=(None, None, 1), factor=2, n_filters=64, n_res_blocks=8, expansion_ratio=4, use_global_skip=True):
    """
    Creates a 2D Wide Activation Super-Resolution (WDSR) model.
    """
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv2D(n_filters, kernel_size=3, padding="same", name="init_conv")(inputs)
    for i in range(n_res_blocks):
        x = wdsr_block_2d(x, n_filters, expansion_ratio, name_prefix=f"wdsr_{i}")
    x = keras.layers.Conv2D(n_filters // 2, kernel_size=3, padding="same", activation="relu", name="shrink_conv")(x)
    
    # preshuffle
    out_channels = 32
    x = keras.layers.Conv2D(out_channels * (factor ** 2), kernel_size=3, padding="same", name="preshuffle_conv")(x)
    outputs = PixelShuffle2D(factor=factor, name="pixel_shuffle")(x)
    
    x = keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu", name="hr_conv1")(outputs)
    x = keras.layers.Conv2D(16, kernel_size=3, padding="same", activation="relu", name="hr_conv2")(x)
    outputs = keras.layers.Conv2D(1, kernel_size=3, padding="same", name="hr_conv3")(x)
    
    if use_global_skip:
        skip = keras.layers.UpSampling2D(size=(factor, factor), name="global_skip")(inputs)
        scaled_skip = LearnableScale(initial_value=1.0, name="scaled_global_skip")(skip)
        outputs = keras.layers.add([outputs, scaled_skip], name="add_global_skip")
        
    return keras.Model(inputs, outputs, name="wdsr_2d")

def create_wdsr_3d(input_shape=(None, None, None, 1), factor=2, n_filters=64, n_res_blocks=8, expansion_ratio=4, use_global_skip=True):
    """
    Creates a 3D Wide Activation Super-Resolution (WDSR) model.
    """
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv3D(n_filters, kernel_size=3, padding="same", name="init_conv")(inputs)
    for i in range(n_res_blocks):
        x = wdsr_block_3d(x, n_filters, expansion_ratio, name_prefix=f"wdsr_{i}")
    x = keras.layers.Conv3D(n_filters // 2, kernel_size=3, padding="same", activation="relu", name="shrink_conv")(x)
    
    # preshuffle
    out_channels = 32
    x = keras.layers.Conv3D(out_channels * (factor ** 3), kernel_size=3, padding="same", name="preshuffle_conv")(x)
    outputs = PixelShuffle3D(factor=factor, name="pixel_shuffle")(x)
    
    x = keras.layers.Conv3D(32, kernel_size=3, padding="same", activation="relu", name="hr_conv1")(outputs)
    x = keras.layers.Conv3D(16, kernel_size=3, padding="same", activation="relu", name="hr_conv2")(x)
    outputs = keras.layers.Conv3D(1, kernel_size=3, padding="same", name="hr_conv3")(x)
    
    if use_global_skip:
        skip = keras.layers.UpSampling3D(size=(factor, factor, factor), name="global_skip")(inputs)
        scaled_skip = LearnableScale(initial_value=1.0, name="scaled_global_skip")(skip)
        outputs = keras.layers.add([outputs, scaled_skip], name="add_global_skip")
        
    return keras.Model(inputs, outputs, name="wdsr_3d")

# ==============================================================================
# RCAN (Residual Channel Attention Network) Models
# ==============================================================================

def rcab_2d(x, n_filters, reduction=16, name_prefix=""):
    res = keras.layers.Conv2D(n_filters, kernel_size=3, padding="same", activation="relu", name=f"{name_prefix}_conv1")(x)
    res = keras.layers.Conv2D(n_filters, kernel_size=3, padding="same", name=f"{name_prefix}_conv2")(res)
    
    # 2D Channel Attention
    channels = n_filters
    squeeze = keras.layers.GlobalAveragePooling2D(name=f"{name_prefix}_ca_squeeze")(res)
    squeeze = keras.layers.Reshape((1, 1, channels), name=f"{name_prefix}_ca_reshape")(squeeze)
    excitation = keras.layers.Conv2D(max(1, channels // reduction), kernel_size=1, activation='relu', name=f"{name_prefix}_ca_conv1")(squeeze)
    excitation = keras.layers.Conv2D(channels, kernel_size=1, activation='sigmoid', name=f"{name_prefix}_ca_conv2")(excitation)
    res = keras.layers.Multiply(name=f"{name_prefix}_ca_scale")([res, excitation])
    
    return keras.layers.add([x, res], name=f"{name_prefix}_add")

def rcab_3d(x, n_filters, reduction=16, name_prefix=""):
    res = keras.layers.Conv3D(n_filters, kernel_size=3, padding="same", activation="relu", name=f"{name_prefix}_conv1")(x)
    res = keras.layers.Conv3D(n_filters, kernel_size=3, padding="same", name=f"{name_prefix}_conv2")(res)
    
    # 3D Channel Attention
    channels = n_filters
    squeeze = keras.layers.GlobalAveragePooling3D(name=f"{name_prefix}_ca_squeeze")(res)
    squeeze = keras.layers.Reshape((1, 1, 1, channels), name=f"{name_prefix}_ca_reshape")(squeeze)
    excitation = keras.layers.Conv3D(max(1, channels // reduction), kernel_size=1, activation='relu', name=f"{name_prefix}_ca_conv1")(squeeze)
    excitation = keras.layers.Conv3D(channels, kernel_size=1, activation='sigmoid', name=f"{name_prefix}_ca_conv2")(excitation)
    res = keras.layers.Multiply(name=f"{name_prefix}_ca_scale")([res, excitation])
    
    return keras.layers.add([x, res], name=f"{name_prefix}_add")

def residual_group_2d(x, n_filters, n_blocks=4, name_prefix=""):
    res = x
    for i in range(n_blocks):
        res = rcab_2d(res, n_filters, name_prefix=f"{name_prefix}_rcab_{i}")
    res = keras.layers.Conv2D(n_filters, kernel_size=3, padding="same", name=f"{name_prefix}_conv")(res)
    return keras.layers.add([x, res], name=f"{name_prefix}_add")

def residual_group_3d(x, n_filters, n_blocks=4, name_prefix=""):
    res = x
    for i in range(n_blocks):
        res = rcab_3d(res, n_filters, name_prefix=f"{name_prefix}_rcab_{i}")
    res = keras.layers.Conv3D(n_filters, kernel_size=3, padding="same", name=f"{name_prefix}_conv")(res)
    return keras.layers.add([x, res], name=f"{name_prefix}_add")

def create_rcan_2d(input_shape=(None, None, 1), factor=2, n_filters=64, n_groups=3, n_blocks=4, use_global_skip=True):
    """
    Creates a 2D Residual Channel Attention Network (RCAN) model.
    """
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv2D(n_filters, kernel_size=3, padding="same", name="init_conv")(inputs)
    for i in range(n_groups):
        x = residual_group_2d(x, n_filters, n_blocks=n_blocks, name_prefix=f"rg_{i}")
    x = keras.layers.Conv2D(n_filters // 2, kernel_size=3, padding="same", activation="relu", name="shrink_conv")(x)
    
    # preshuffle
    out_channels = 32
    x = keras.layers.Conv2D(out_channels * (factor ** 2), kernel_size=3, padding="same", name="preshuffle_conv")(x)
    outputs = PixelShuffle2D(factor=factor, name="pixel_shuffle")(x)
    
    x = keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu", name="hr_conv1")(outputs)
    x = keras.layers.Conv2D(16, kernel_size=3, padding="same", activation="relu", name="hr_conv2")(x)
    outputs = keras.layers.Conv2D(1, kernel_size=3, padding="same", name="hr_conv3")(x)
    
    if use_global_skip:
        skip = keras.layers.UpSampling2D(size=(factor, factor), name="global_skip")(inputs)
        scaled_skip = LearnableScale(initial_value=1.0, name="scaled_global_skip")(skip)
        outputs = keras.layers.add([outputs, scaled_skip], name="add_global_skip")
        
    return keras.Model(inputs, outputs, name="rcan_2d")

def create_rcan_3d(input_shape=(None, None, None, 1), factor=2, n_filters=64, n_groups=3, n_blocks=4, use_global_skip=True):
    """
    Creates a 3D Residual Channel Attention Network (RCAN) model.
    """
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv3D(n_filters, kernel_size=3, padding="same", name="init_conv")(inputs)
    for i in range(n_groups):
        x = residual_group_3d(x, n_filters, n_blocks=n_blocks, name_prefix=f"rg_{i}")
    x = keras.layers.Conv3D(n_filters // 2, kernel_size=3, padding="same", activation="relu", name="shrink_conv")(x)
    
    # preshuffle
    out_channels = 32
    x = keras.layers.Conv3D(out_channels * (factor ** 3), kernel_size=3, padding="same", name="preshuffle_conv")(x)
    outputs = PixelShuffle3D(factor=factor, name="pixel_shuffle")(x)
    
    x = keras.layers.Conv3D(32, kernel_size=3, padding="same", activation="relu", name="hr_conv1")(outputs)
    x = keras.layers.Conv3D(16, kernel_size=3, padding="same", activation="relu", name="hr_conv2")(x)
    outputs = keras.layers.Conv3D(1, kernel_size=3, padding="same", name="hr_conv3")(x)
    
    if use_global_skip:
        skip = keras.layers.UpSampling3D(size=(factor, factor, factor), name="global_skip")(inputs)
        scaled_skip = LearnableScale(initial_value=1.0, name="scaled_global_skip")(skip)
        outputs = keras.layers.add([outputs, scaled_skip], name="add_global_skip")
        
    return keras.Model(inputs, outputs, name="rcan_3d")

# ==============================================================================
# CARN (Cascading Residual Network) Models
# ==============================================================================

def carn_block_2d(x, n_filters, name_prefix=""):
    b1 = keras.layers.Conv2D(n_filters, kernel_size=3, padding="same", activation="relu", name=f"{name_prefix}_c1")(x)
    b2 = keras.layers.Conv2D(n_filters, kernel_size=3, padding="same", activation="relu", name=f"{name_prefix}_c2")(b1)
    
    # cascade b1 and b2
    concat = keras.layers.concatenate([b1, b2], name=f"{name_prefix}_concat")
    proj = keras.layers.Conv2D(n_filters, kernel_size=1, padding="same", name=f"{name_prefix}_proj")(concat)
    
    return keras.layers.add([x, proj], name=f"{name_prefix}_add")

def carn_block_3d(x, n_filters, name_prefix=""):
    b1 = keras.layers.Conv3D(n_filters, kernel_size=3, padding="same", activation="relu", name=f"{name_prefix}_c1")(x)
    b2 = keras.layers.Conv3D(n_filters, kernel_size=3, padding="same", activation="relu", name=f"{name_prefix}_c2")(b1)
    
    # cascade b1 and b2
    concat = keras.layers.concatenate([b1, b2], name=f"{name_prefix}_concat")
    proj = keras.layers.Conv3D(n_filters, kernel_size=1, padding="same", name=f"{name_prefix}_proj")(concat)
    
    return keras.layers.add([x, proj], name=f"{name_prefix}_add")

def create_carn_2d(input_shape=(None, None, 1), factor=2, n_filters=64, n_blocks=3, use_global_skip=True):
    """
    Creates a 2D Cascading Residual Network (CARN) model.
    """
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv2D(n_filters, kernel_size=3, padding="same", name="init_conv")(inputs)
    block_outputs = []
    current = x
    for i in range(n_blocks):
        current = carn_block_2d(current, n_filters, name_prefix=f"carn_{i}")
        block_outputs.append(current)
    
    # cascade global
    concat = keras.layers.concatenate(block_outputs, name="global_concat")
    x = keras.layers.Conv2D(n_filters, kernel_size=1, padding="same", name="global_proj")(concat)
    x = keras.layers.Conv2D(n_filters // 2, kernel_size=3, padding="same", activation="relu", name="shrink_conv")(x)
    
    # preshuffle
    out_channels = 32
    x = keras.layers.Conv2D(out_channels * (factor ** 2), kernel_size=3, padding="same", name="preshuffle_conv")(x)
    outputs = PixelShuffle2D(factor=factor, name="pixel_shuffle")(x)
    
    x = keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu", name="hr_conv1")(outputs)
    x = keras.layers.Conv2D(16, kernel_size=3, padding="same", activation="relu", name="hr_conv2")(x)
    outputs = keras.layers.Conv2D(1, kernel_size=3, padding="same", name="hr_conv3")(x)
    
    if use_global_skip:
        skip = keras.layers.UpSampling2D(size=(factor, factor), name="global_skip")(inputs)
        scaled_skip = LearnableScale(initial_value=1.0, name="scaled_global_skip")(skip)
        outputs = keras.layers.add([outputs, scaled_skip], name="add_global_skip")
        
    return keras.Model(inputs, outputs, name="carn_2d")

def create_carn_3d(input_shape=(None, None, None, 1), factor=2, n_filters=64, n_blocks=3, use_global_skip=True):
    """
    Creates a 3D Cascading Residual Network (CARN) model.
    """
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv3D(n_filters, kernel_size=3, padding="same", name="init_conv")(inputs)
    block_outputs = []
    current = x
    for i in range(n_blocks):
        current = carn_block_3d(current, n_filters, name_prefix=f"carn_{i}")
        block_outputs.append(current)
    
    # cascade global
    concat = keras.layers.concatenate(block_outputs, name="global_concat")
    x = keras.layers.Conv3D(n_filters, kernel_size=1, padding="same", name="global_proj")(concat)
    x = keras.layers.Conv3D(n_filters // 2, kernel_size=3, padding="same", activation="relu", name="shrink_conv")(x)
    
    # preshuffle
    out_channels = 32
    x = keras.layers.Conv3D(out_channels * (factor ** 3), kernel_size=3, padding="same", name="preshuffle_conv")(x)
    outputs = PixelShuffle3D(factor=factor, name="pixel_shuffle")(x)
    
    x = keras.layers.Conv3D(32, kernel_size=3, padding="same", activation="relu", name="hr_conv1")(outputs)
    x = keras.layers.Conv3D(16, kernel_size=3, padding="same", activation="relu", name="hr_conv2")(x)
    outputs = keras.layers.Conv3D(1, kernel_size=3, padding="same", name="hr_conv3")(x)
    
    if use_global_skip:
        skip = keras.layers.UpSampling3D(size=(factor, factor, factor), name="global_skip")(inputs)
        scaled_skip = LearnableScale(initial_value=1.0, name="scaled_global_skip")(skip)
        outputs = keras.layers.add([outputs, scaled_skip], name="add_global_skip")
        
    return keras.Model(inputs, outputs, name="carn_3d")


def create_espcn_2d_resize_conv(input_shape=(None, None, 1), factor=2, n_filters=64, n_res_blocks=8, use_global_skip=True):
    """
    Creates a 2D Residual ESPCN model using Bilinear Resize + Conv instead of PixelShuffle
    to mitigate checkerboard / step artifacts.
    """
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv2D(n_filters, kernel_size=3, padding="same", activation="relu", name="init_conv")(inputs)
    
    for i in range(n_res_blocks):
        res = keras.layers.Conv2D(n_filters, kernel_size=3, padding="same", activation="relu", name=f"res_{i}_conv1")(x)
        res = keras.layers.Conv2D(n_filters, kernel_size=3, padding="same", name=f"res_{i}_conv2")(res)
        
        # Channel Attention
        channels = res.shape[-1]
        squeeze = keras.layers.GlobalAveragePooling2D(name=f"res_{i}_ca_squeeze")(res)
        squeeze = keras.layers.Reshape((1, 1, channels), name=f"res_{i}_ca_reshape")(squeeze)
        excitation = keras.layers.Conv2D(max(1, channels // 16), kernel_size=1, activation='relu', name=f"res_{i}_ca_conv1")(squeeze)
        excitation = keras.layers.Conv2D(channels, kernel_size=1, activation='sigmoid',
                                         kernel_initializer='zeros', bias_initializer='ones', name=f"res_{i}_ca_conv2")(excitation)
        res = keras.layers.Multiply(name=f"res_{i}_ca_scale")([res, excitation])
        x = keras.layers.add([x, res], name=f"res_{i}_add")
        
    x = keras.layers.Conv2D(n_filters // 2, kernel_size=3, padding="same", activation="relu", name="shrink_conv")(x)
    
    # Bilinear Resize + Conv instead of PixelShuffle
    out_channels = 32
    x = keras.layers.UpSampling2D(size=(factor, factor), interpolation="bilinear", name="resize_upsample")(x)
    outputs = keras.layers.Conv2D(out_channels, kernel_size=3, padding="same", name="resize_conv")(x)
    
    x = keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu", name="hr_conv1")(outputs)
    x = keras.layers.Conv2D(16, kernel_size=3, padding="same", activation="relu", name="hr_conv2")(x)
    outputs = keras.layers.Conv2D(1, kernel_size=3, padding="same", name="hr_conv3")(x)
    
    if use_global_skip:
        skip = keras.layers.UpSampling2D(size=(factor, factor), name="global_skip")(inputs)
        scaled_skip = LearnableScale(initial_value=1.0, name="scaled_global_skip")(skip)
        outputs = keras.layers.add([outputs, scaled_skip], name="add_global_skip")
        
    return keras.Model(inputs, outputs, name="espcn_2d_resize_conv")


def create_wdsr_2d_resize_conv(input_shape=(None, None, 1), factor=2, n_filters=64, n_res_blocks=8, expansion_ratio=4, use_global_skip=True):
    """
    Creates a 2D WDSR model using Bilinear Resize + Conv instead of PixelShuffle
    to mitigate checkerboard / step artifacts.
    """
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv2D(n_filters, kernel_size=3, padding="same", name="init_conv")(inputs)
    for i in range(n_res_blocks):
        x = wdsr_block_2d(x, n_filters, expansion_ratio, name_prefix=f"wdsr_{i}")
    x = keras.layers.Conv2D(n_filters // 2, kernel_size=3, padding="same", activation="relu", name="shrink_conv")(x)
    
    # Bilinear Resize + Conv instead of PixelShuffle
    out_channels = 32
    x = keras.layers.UpSampling2D(size=(factor, factor), interpolation="bilinear", name="resize_upsample")(x)
    outputs = keras.layers.Conv2D(out_channels, kernel_size=3, padding="same", name="resize_conv")(x)
    
    x = keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu", name="hr_conv1")(outputs)
    x = keras.layers.Conv2D(16, kernel_size=3, padding="same", activation="relu", name="hr_conv2")(x)
    outputs = keras.layers.Conv2D(1, kernel_size=3, padding="same", name="hr_conv3")(x)
    
    if use_global_skip:
        skip = keras.layers.UpSampling2D(size=(factor, factor), name="global_skip")(inputs)
        scaled_skip = LearnableScale(initial_value=1.0, name="scaled_global_skip")(skip)
        outputs = keras.layers.add([outputs, scaled_skip], name="add_global_skip")
        
    return keras.Model(inputs, outputs, name="wdsr_2d_resize_conv")


# ==============================================================================
# SRFBN (Super-Resolution Feedback Network) Models
# ==============================================================================

def create_srfbn_2d(input_shape=(None, None, 1), factor=2, n_filters=64, n_steps=4, use_global_skip=True):
    """
    Creates a 2D Super-Resolution Feedback Network (SRFBN) model.
    It uses a recurrent feedback block across n_steps to iteratively refine low-resolution representations.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Feature extraction block
    F_in = layers.Conv2D(n_filters, kernel_size=3, padding="same", activation="relu", name="init_conv")(inputs)
    
    # Instantiate recurrent layers to share weights across steps
    if n_steps > 1:
        project_layer = layers.Conv2D(n_filters, kernel_size=1, padding="same", activation="relu", name="fb_project")
    up_layer = layers.Conv2DTranspose(n_filters, kernel_size=factor, strides=factor, padding="same", activation="relu", name="fb_up")
    down_layer = layers.Conv2D(n_filters, kernel_size=factor, strides=factor, padding="same", activation="relu", name="fb_down")
    
    # Recurrent feedback loop
    L_t = F_in
    H_t = None
    
    for t in range(n_steps):
        # Feedback block (FB)
        if t == 0:
            x = F_in
        else:
            x = layers.Concatenate(axis=-1, name=f"fb_{t}_concat")([F_in, L_t])
            x = project_layer(x)
        
        # Up-projection (Deconvolution)
        H_t = up_layer(x)
        
        # Down-projection (Stride Conv)
        L_t = down_layer(H_t)
        
    # Reconstruction block using final HR representation H_t
    outputs = layers.Conv2D(n_filters // 2, kernel_size=3, padding="same", activation="relu", name="recon_conv1")(H_t)
    outputs = layers.Conv2D(1, kernel_size=3, padding="same", name="recon_conv2")(outputs)
    
    if use_global_skip:
        skip = layers.UpSampling2D(size=(factor, factor), interpolation="bilinear", name="global_skip")(inputs)
        scaled_skip = LearnableScale(initial_value=1.0, name="scaled_global_skip")(skip)
        outputs = layers.add([outputs, scaled_skip], name="add_global_skip")
        
    return keras.Model(inputs, outputs, name="srfbn_2d")


def create_srfbn_3d(input_shape=(None, None, None, 1), factor=2, n_filters=64, n_steps=4, use_global_skip=True):
    """
    Creates a 3D Super-Resolution Feedback Network (SRFBN) model.
    """
    inputs = layers.Input(shape=input_shape)
    
    # Feature extraction block
    F_in = layers.Conv3D(n_filters, kernel_size=3, padding="same", activation="relu", name="init_conv")(inputs)
    
    # Instantiate recurrent layers to share weights across steps
    if n_steps > 1:
        project_layer = layers.Conv3D(n_filters, kernel_size=1, padding="same", activation="relu", name="fb_project")
    up_layer = layers.Conv3DTranspose(n_filters, kernel_size=factor, strides=factor, padding="same", activation="relu", name="fb_up")
    down_layer = layers.Conv3D(n_filters, kernel_size=factor, strides=factor, padding="same", activation="relu", name="fb_down")
    
    # Recurrent feedback loop
    L_t = F_in
    H_t = None
    
    for t in range(n_steps):
        # Feedback block (FB)
        if t == 0:
            x = F_in
        else:
            x = layers.Concatenate(axis=-1, name=f"fb_{t}_concat")([F_in, L_t])
            x = project_layer(x)
        
        # Up-projection (Deconvolution)
        H_t = up_layer(x)
        
        # Down-projection (Stride Conv)
        L_t = down_layer(H_t)
        
    # Reconstruction block using final HR representation H_t
    outputs = layers.Conv3D(n_filters // 2, kernel_size=3, padding="same", activation="relu", name="recon_conv1")(H_t)
    outputs = layers.Conv3D(1, kernel_size=3, padding="same", name="recon_conv2")(outputs)
    
    if use_global_skip:
        skip = layers.UpSampling3D(size=(factor, factor, factor), name="global_skip")(inputs)
        scaled_skip = LearnableScale(initial_value=1.0, name="scaled_global_skip")(skip)
        outputs = layers.add([outputs, scaled_skip], name="add_global_skip")
        
    return keras.Model(inputs, outputs, name="srfbn_3d")


# ==============================================================================
# SAN (Second-order Attention Network) Models
# ==============================================================================

def soca_block_2d(input_tensor, reduction_ratio=16, name_prefix=""):
    """
    Second-order Channel Attention (SOCA) block for 2D.
    It uses channel-wise variance to model second-order statistics.
    """
    channels = input_tensor.shape[-1]
    mean = layers.GlobalAveragePooling2D(keepdims=True, name=f"{name_prefix}_soca_mean")(input_tensor)
    sq_diff = layers.Lambda(lambda inputs: (inputs[0] - inputs[1])**2, name=f"{name_prefix}_soca_sq_diff")([input_tensor, mean])
    variance = layers.GlobalAveragePooling2D(keepdims=True, name=f"{name_prefix}_soca_var")(sq_diff)
    
    excitation = layers.Conv2D(max(1, channels // reduction_ratio), kernel_size=1, activation='relu', name=f"{name_prefix}_soca_conv1")(variance)
    excitation = layers.Conv2D(channels, kernel_size=1, activation='sigmoid', name=f"{name_prefix}_soca_conv2")(excitation)
    
    return layers.Multiply(name=f"{name_prefix}_soca_scale")([input_tensor, excitation])


def soca_block_3d(input_tensor, reduction_ratio=16, name_prefix=""):
    """
    Second-order Channel Attention (SOCA) block for 3D.
    It uses channel-wise variance to model second-order statistics.
    """
    channels = input_tensor.shape[-1]
    mean = layers.GlobalAveragePooling3D(keepdims=True, name=f"{name_prefix}_soca_mean")(input_tensor)
    sq_diff = layers.Lambda(lambda inputs: (inputs[0] - inputs[1])**2, name=f"{name_prefix}_soca_sq_diff")([input_tensor, mean])
    variance = layers.GlobalAveragePooling3D(keepdims=True, name=f"{name_prefix}_soca_var")(sq_diff)
    
    excitation = layers.Conv3D(max(1, channels // reduction_ratio), kernel_size=1, activation='relu', name=f"{name_prefix}_soca_conv1")(variance)
    excitation = layers.Conv3D(channels, kernel_size=1, activation='sigmoid', name=f"{name_prefix}_soca_conv2")(excitation)
    
    return layers.Multiply(name=f"{name_prefix}_soca_scale")([input_tensor, excitation])


def lsrab_2d(x, n_filters, reduction=16, name_prefix=""):
    """
    Local Second-order Residual Attention Block (LSRAB) for 2D.
    """
    res = layers.Conv2D(n_filters, kernel_size=3, padding="same", activation="relu", name=f"{name_prefix}_conv1")(x)
    res = layers.Conv2D(n_filters, kernel_size=3, padding="same", name=f"{name_prefix}_conv2")(res)
    res = soca_block_2d(res, reduction_ratio=reduction, name_prefix=name_prefix)
    return layers.add([x, res], name=f"{name_prefix}_add")


def lsrab_3d(x, n_filters, reduction=16, name_prefix=""):
    """
    Local Second-order Residual Attention Block (LSRAB) for 3D.
    """
    res = layers.Conv3D(n_filters, kernel_size=3, padding="same", activation="relu", name=f"{name_prefix}_conv1")(x)
    res = layers.Conv3D(n_filters, kernel_size=3, padding="same", name=f"{name_prefix}_conv2")(res)
    res = soca_block_3d(res, reduction_ratio=reduction, name_prefix=name_prefix)
    return layers.add([x, res], name=f"{name_prefix}_add")


def residual_group_soca_2d(x, n_filters, n_blocks=4, name_prefix=""):
    """
    Residual Group of LSRAB blocks in 2D.
    """
    res = x
    for i in range(n_blocks):
        res = lsrab_2d(res, n_filters, name_prefix=f"{name_prefix}_lsrab_{i}")
    res = layers.Conv2D(n_filters, kernel_size=3, padding="same", name=f"{name_prefix}_conv")(res)
    return layers.add([x, res], name=f"{name_prefix}_add")


def residual_group_soca_3d(x, n_filters, n_blocks=4, name_prefix=""):
    """
    Residual Group of LSRAB blocks in 3D.
    """
    res = x
    for i in range(n_blocks):
        res = lsrab_3d(res, n_filters, name_prefix=f"{name_prefix}_lsrab_{i}")
    res = layers.Conv3D(n_filters, kernel_size=3, padding="same", name=f"{name_prefix}_conv")(res)
    return layers.add([x, res], name=f"{name_prefix}_add")


def create_san_2d(input_shape=(None, None, 1), factor=2, n_filters=64, n_groups=3, n_blocks=4, use_global_skip=True):
    """
    Creates a 2D Second-order Attention Network (SAN) model.
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(n_filters, kernel_size=3, padding="same", name="init_conv")(inputs)
    
    # Stack residual groups
    res = x
    for i in range(n_groups):
        res = residual_group_soca_2d(res, n_filters, n_blocks=n_blocks, name_prefix=f"group_{i}")
    res = layers.Conv2D(n_filters, kernel_size=3, padding="same", name="group_conv")(res)
    x = layers.add([x, res], name="group_add")
    
    # Upsampling via PixelShuffle
    x = layers.Conv2D(n_filters * (factor ** 2), kernel_size=3, padding="same", name="pre_shuffle_conv")(x)
    x = PixelShuffle2D(factor=factor, name="pixel_shuffle")(x)
    outputs = layers.Conv2D(1, kernel_size=3, padding="same", name="final_conv")(x)
    
    if use_global_skip:
        skip = layers.UpSampling2D(size=(factor, factor), interpolation="bilinear", name="global_skip")(inputs)
        scaled_skip = LearnableScale(initial_value=1.0, name="scaled_global_skip")(skip)
        outputs = layers.add([outputs, scaled_skip], name="add_global_skip")
        
    return keras.Model(inputs, outputs, name="san_2d")


def create_san_3d(input_shape=(None, None, None, 1), factor=2, n_filters=64, n_groups=3, n_blocks=4, use_global_skip=True):
    """
    Creates a 3D Second-order Attention Network (SAN) model.
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv3D(n_filters, kernel_size=3, padding="same", name="init_conv")(inputs)
    
    # Stack residual groups
    res = x
    for i in range(n_groups):
        res = residual_group_soca_3d(res, n_filters, n_blocks=n_blocks, name_prefix=f"group_{i}")
    res = layers.Conv3D(n_filters, kernel_size=3, padding="same", name="group_conv")(res)
    x = layers.add([x, res], name="group_add")
    
    # Upsampling via PixelShuffle
    x = layers.Conv3D(n_filters * (factor ** 3), kernel_size=3, padding="same", name="pre_shuffle_conv")(x)
    x = PixelShuffle3D(factor=factor, name="pixel_shuffle")(x)
    outputs = layers.Conv3D(1, kernel_size=3, padding="same", name="final_conv")(x)
    
    if use_global_skip:
        skip = layers.UpSampling3D(size=(factor, factor, factor), name="global_skip")(inputs)
        scaled_skip = LearnableScale(initial_value=1.0, name="scaled_global_skip")(skip)
        outputs = layers.add([outputs, scaled_skip], name="add_global_skip")
        
    return keras.Model(inputs, outputs, name="san_3d")

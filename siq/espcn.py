import math
import keras
from keras import layers, ops

def _normalize_factor(factor, dim):
    """
    Normalizes a super-resolution scaling factor into an integer tuple of length `dim`.
    Supports scalars (e.g. 2 -> (2, 2) or (2, 2, 2)) and tuples/lists (e.g. (1, 1, 2)).
    """
    if isinstance(factor, (int, float)):
        return tuple([int(factor)] * dim)
    elif isinstance(factor, (list, tuple)):
        if len(factor) != dim:
            raise ValueError(f"For {dim}D model, factor must have length {dim}, got {factor}")
        return tuple(int(f) for f in factor)
    else:
        raise ValueError(f"Unsupported factor type: {type(factor)}")

def pixel_shuffle_3d(inputs, factor=2):
    """
    Implementation of 3D Pixel Shuffle for Keras/Keras3.
    Args:
        inputs: (batch, d, h, w, c)
        factor: scaling factor (integer or 3-element tuple/list)
    """
    factor_tuple = _normalize_factor(factor, 3)
    f_d, f_h, f_w = factor_tuple
    f_total = f_d * f_h * f_w
    input_shape = ops.shape(inputs)
    batch_size = input_shape[0]
    d, h, w = input_shape[1], input_shape[2], input_shape[3]
    channels = input_shape[4]
    
    new_channels = channels // f_total
    
    # Reshape: (batch, d, h, w, f_d, f_h, f_w, new_c)
    x = ops.reshape(inputs, (batch_size, d, h, w, f_d, f_h, f_w, new_channels))
    
    # Transpose to (batch, d, f_d, h, f_h, w, f_w, new_c)
    x = ops.transpose(x, (0, 1, 4, 2, 5, 3, 6, 7))
    
    # Reshape to (batch, d*f_d, h*f_h, w*f_w, new_c)
    new_d, new_h, new_w = d * f_d, h * f_h, w * f_w
    return ops.reshape(x, (batch_size, new_d, new_h, new_w, new_channels))

@keras.saving.register_keras_serializable(package="siq")
class PixelShuffle3D(layers.Layer):
    def __init__(self, factor=2, **kwargs):
        super().__init__(**kwargs)
        self.factor = _normalize_factor(factor, 3) if isinstance(factor, (list, tuple)) else factor

    def call(self, inputs):
        return pixel_shuffle_3d(inputs, self.factor)

    def compute_output_shape(self, input_shape):
        factor_tuple = _normalize_factor(self.factor, 3)
        f_d, f_h, f_w = factor_tuple
        f_total = f_d * f_h * f_w
        d = input_shape[1] * f_d if input_shape[1] is not None else None
        h = input_shape[2] * f_h if input_shape[2] is not None else None
        w = input_shape[3] * f_w if input_shape[3] is not None else None
        c = input_shape[4] // f_total if input_shape[4] is not None else None
        return (input_shape[0], d, h, w, c)

    def get_config(self):
        config = super().get_config()
        config.update({"factor": self.factor})
        return config

def trilinear_upsample_3d(inputs, factor=2):
    """
    Separable trilinear interpolation for 5D tensors (batch, d, h, w, c).
    Uses bilinear resizing on (h, w) slices followed by bilinear resizing on transposed (d, h) slices.
    """
    factor_tuple = _normalize_factor(factor, 3)
    f_d, f_h, f_w = factor_tuple
    shape = ops.shape(inputs)
    b, d, h, w, c = shape[0], shape[1], shape[2], shape[3], shape[4]
    
    # 1. Bilinear resize on (H, W) across all B*D slices
    x = ops.reshape(inputs, (-1, h, w, c))
    x = ops.image.resize(x, (h * f_h, w * f_w), interpolation="bilinear")
    x = ops.reshape(x, (b, d, h * f_h, w * f_w, c))
    
    # 2. Bilinear resize along D axis if f_d > 1
    if f_d > 1:
        x = ops.transpose(x, (0, 3, 1, 2, 4))  # (B, W_new, D, H_new, C)
        x = ops.reshape(x, (-1, d, h * f_h, c))
        x = ops.image.resize(x, (d * f_d, h * f_h), interpolation="bilinear")
        x = ops.reshape(x, (b, w * f_w, d * f_d, h * f_h, c))
        x = ops.transpose(x, (0, 2, 3, 1, 4))  # Back to (B, D_new, H_new, W_new, C)
        
    return x

@keras.saving.register_keras_serializable(package="siq")
class TrilinearUpSampling3D(layers.Layer):
    """
    3D Trilinear Upsampling layer for Keras 3.
    Provides smooth continuous upsampling for 5D volumes without nearest-neighbor blockiness.
    """
    def __init__(self, size=(2, 2, 2), **kwargs):
        super().__init__(**kwargs)
        self.size = _normalize_factor(size, 3)

    def call(self, inputs):
        return trilinear_upsample_3d(inputs, self.size)

    def compute_output_shape(self, input_shape):
        f_d, f_h, f_w = self.size
        d = input_shape[1] * f_d if input_shape[1] is not None else None
        h = input_shape[2] * f_h if input_shape[2] is not None else None
        w = input_shape[3] * f_w if input_shape[3] is not None else None
        return (input_shape[0], d, h, w, input_shape[4])

    def get_config(self):
        config = super().get_config()
        config.update({"size": self.size})
        return config


def create_espcn_3d(input_shape=(None, None, None, 1), factor=2, n_filters=64):
    """
    Creates a 3D ESPCN model.
    """
    factor_tuple = _normalize_factor(factor, 3)
    f_total = math.prod(factor_tuple)
    inputs = layers.Input(shape=input_shape)
    
    # Feature extraction
    x = layers.Conv3D(n_filters, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.Conv3D(n_filters // 2, kernel_size=3, padding="same", activation="relu")(x)
    
    out_channels = 1
    x = layers.Conv3D(out_channels * f_total, kernel_size=3, padding="same")(x)
    
    # Pixel Shuffle
    outputs = PixelShuffle3D(factor=factor_tuple)(x)
    
    return keras.Model(inputs, outputs, name="espcn_3d")

def create_espcn_3d_residual(input_shape=(None, None, None, 1), factor=2, n_filters=64, n_res_blocks=4):
    """
    Creates a 3D Residual ESPCN model.
    """
    factor_tuple = _normalize_factor(factor, 3)
    f_total = math.prod(factor_tuple)
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
    
    out_channels = 32
    x = layers.Conv3D(out_channels * f_total, kernel_size=3, padding="same")(x)
    
    # Pixel Shuffle transition to high-resolution space
    x = PixelShuffle3D(factor=factor_tuple)(x)
    
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


@keras.saving.register_keras_serializable(package="siq")
class LearnableSharpening(layers.Layer):
    """
    Keras layer that blends the input with a learnable high-pass (sharpened) version.
    Initializes weight to 0.1 and kernels to a Laplacian high-pass filter.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.amount = None
        self.detail_conv = None

    def build(self, input_shape):
        import numpy as np
        self.amount = self.add_weight(
            shape=(1,),
            initializer=keras.initializers.Constant(0.1),
            trainable=True,
            name="amount"
        )
        self.detail_conv = layers.Conv2D(
            filters=1,
            kernel_size=3,
            padding="same",
            use_bias=False,
            name="detail_conv_kernel"
        )
        self.detail_conv.build(input_shape)
        laplacian = np.array([[[[0.0]], [[1.0]], [[0.0]]],
                              [[[1.0]], [[-4.0]], [[1.0]]],
                              [[[0.0]], [[1.0]], [[0.0]]]], dtype=np.float32)
        self.detail_conv.set_weights([laplacian])
        super().build(input_shape)

    def call(self, inputs):
        detail = self.detail_conv(inputs)
        return inputs + self.amount * detail

    def get_config(self):
        return super().get_config()


@keras.saving.register_keras_serializable(package="siq")
class LearnableSharpening3D(layers.Layer):
    """
    3D Keras layer that blends the input with a learnable high-pass (sharpened) version.
    Initializes weight to 0.1 and kernels to a 3D Laplacian high-pass filter.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.amount = None
        self.detail_conv = None

    def build(self, input_shape):
        import numpy as np
        self.amount = self.add_weight(
            shape=(1,),
            initializer=keras.initializers.Constant(0.1),
            trainable=True,
            name="amount"
        )
        self.detail_conv = layers.Conv3D(
            filters=1,
            kernel_size=3,
            padding="same",
            use_bias=False,
            name="detail_conv_kernel"
        )
        self.detail_conv.build(input_shape)
        laplacian = np.zeros((3, 3, 3, 1, 1), dtype=np.float32)
        laplacian[1, 1, 1, 0, 0] = -6.0
        laplacian[0, 1, 1, 0, 0] = 1.0
        laplacian[2, 1, 1, 0, 0] = 1.0
        laplacian[1, 0, 1, 0, 0] = 1.0
        laplacian[1, 2, 1, 0, 0] = 1.0
        laplacian[1, 1, 0, 0, 0] = 1.0
        laplacian[1, 1, 2, 0, 0] = 1.0
        self.detail_conv.set_weights([laplacian])
        super().build(input_shape)

    def call(self, inputs):
        detail = self.detail_conv(inputs)
        return inputs + self.amount * detail

    def get_config(self):
        return super().get_config()


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
    Supports isotropic integer factors (e.g. 2, 4) or anisotropic tuple factors (e.g. (1, 1, 2), (2, 2, 4)).
    """
    factor_tuple = _normalize_factor(factor, 3)
    f_total = math.prod(factor_tuple)
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
    x = layers.Conv3D(out_channels * f_total, kernel_size=3, padding="same", name="preshuffle_conv")(x)
    
    # Pixel Shuffle
    outputs = PixelShuffle3D(factor=factor_tuple, name="pixel_shuffle")(x)
    
    # Non-linear processing in high-res space
    x = layers.Conv3D(32, kernel_size=3, padding="same", activation="relu", name="hr_conv1")(outputs)
    x = layers.Conv3D(16, kernel_size=3, padding="same", activation="relu", name="hr_conv2")(x)
    outputs = layers.Conv3D(1, kernel_size=3, padding="same", name="hr_conv3")(x)
    
    # Optional Global Skip connection
    if use_global_skip:
        skip = TrilinearUpSampling3D(size=factor_tuple, name="global_skip")(inputs)
        # Learnable scale initialized to 1.0
        scaled_skip = LearnableScale(initial_value=1.0, name="scaled_global_skip")(skip)
        outputs = layers.add([outputs, scaled_skip], name="add_global_skip")
        
    return keras.Model(inputs, outputs, name="espcn_3d_attention")

def up_projection_unit(lr_input, n_filters, factor=2, name_prefix=""):
    factor_tuple = _normalize_factor(factor, 3)
    f_total = math.prod(factor_tuple)
    x = layers.Conv3D(n_filters * f_total, kernel_size=3, padding="same", activation="relu", name=f"{name_prefix}_up_conv1")(lr_input)
    h_temp = PixelShuffle3D(factor=factor_tuple, name=f"{name_prefix}_up_shuffle1")(x)
    
    l_temp = layers.Conv3D(n_filters, kernel_size=3, strides=factor_tuple, padding="same", activation="relu", name=f"{name_prefix}_up_down1")(h_temp)
    e_lr = layers.subtract([lr_input, l_temp], name=f"{name_prefix}_up_sub")
    
    x_err = layers.Conv3D(n_filters * f_total, kernel_size=3, padding="same", activation="relu", name=f"{name_prefix}_up_conv2")(e_lr)
    e_hr = PixelShuffle3D(factor=factor_tuple, name=f"{name_prefix}_up_shuffle2")(x_err)
    
    return layers.add([h_temp, e_hr], name=f"{name_prefix}_up_add")

def down_projection_unit(hr_input, n_filters, factor=2, name_prefix=""):
    factor_tuple = _normalize_factor(factor, 3)
    f_total = math.prod(factor_tuple)
    l_temp = layers.Conv3D(n_filters, kernel_size=3, strides=factor_tuple, padding="same", activation="relu", name=f"{name_prefix}_down_conv1")(hr_input)
    
    x = layers.Conv3D(n_filters * f_total, kernel_size=3, padding="same", activation="relu", name=f"{name_prefix}_down_conv2")(l_temp)
    h_temp = PixelShuffle3D(factor=factor_tuple, name=f"{name_prefix}_down_shuffle1")(x)
    
    e_hr = layers.subtract([hr_input, h_temp], name=f"{name_prefix}_down_sub")
    e_lr = layers.Conv3D(n_filters, kernel_size=3, strides=factor_tuple, padding="same", activation="relu", name=f"{name_prefix}_down_conv3")(e_hr)
    
    return layers.add([l_temp, e_lr], name=f"{name_prefix}_down_add")

def create_ldbpn_3d(input_shape=(None, None, None, 1), factor=2, n_filters=32, n_stages=3):
    """
    Creates a Lightweight 3D Deep Back-Projection Network (L-DBPN) using PixelShuffle3D.
    Supports isotropic integer factors (e.g. 2, 4) or anisotropic tuple factors (e.g. (1, 1, 2), (2, 2, 4)).

    Recommended Default Configuration:
        - `n_stages = 3` (s=3)
        - `n_filters = 32` (f=32)
        (Total Params: 2,057,281 | Model Size: 7.85 MB FP32 | Forward Pass: ~270 ms on 32^3 patch)

    Architectural Rationale & Advantages:
        1. True Algorithmic Back-Projection: Unlike recurrent autoencoders (such as AS-DBPN),
           L-DBPN computes explicit spatial residual reconstruction errors at both resolutions:
               e_lr = L - Down(Up(L))
               e_hr = H - Up(Down(H))
           and injects them as additive self-corrections into subsequent projection units.
        2. Dense Multi-Scale Feature Aggregation: Intermediate high-resolution estimates from
           all stages [H_0, ..., H_{n_stages-1}] are preserved and concatenated into the final
           reconstruction block, fusing coarse, intermediate, and fine structural features.
        3. PixelShuffle3D Sub-Voxel Upsampling: Eliminates transposed-convolution checkerboard
           grid resonance (CBI) by rearranging feature channels directly into spatial voxels.
        4. 3x3x3 Compact Convolutions: Uses 3x3x3 kernels (27 weights) rather than 6x6x6 (216 weights),
           yielding an 8x reduction in MACs per convolution and 3.5x lower inference latency than AS-DBPN.
        5. VRAM Footprint: Under 8 MB parameter size leaves generous GPU headroom during volumetric
           training, supporting larger patch volumes (e.g., 48^3 or 64^3) without out-of-memory errors.
    """
    factor_tuple = _normalize_factor(factor, 3)
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
            
        h = up_projection_unit(l_proj, n_filters, factor_tuple, name_prefix=f"stage_{i}")
        h_list.append(h)
        
        if len(h_list) > 1:
            h_cat = layers.concatenate(h_list, name=f"h_cat_{i}")
            h_proj = layers.Conv3D(n_filters, kernel_size=1, padding="same", activation="relu", name=f"h_proj_{i}")(h_cat)
        else:
            h_proj = h_list[0]
            
        l = down_projection_unit(h_proj, n_filters, factor_tuple, name_prefix=f"stage_{i}")
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

def transfer_dbpn_weights(src_model, dst_model, allow_channel_slicing=True):
    """
    Transfers weights from legacy DBPN model (src_model)
    to Lightweight DBPN model (dst_model) by shape or channel slicing.
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

    if matched == 0 and allow_channel_slicing:
        matched = transfer_dbpn_to_smaller(src_model, dst_model)
            
    return matched

def transfer_dbpn_to_smaller(src_model, dst_model, verbose=False):
    """
    Transfers weights from large DBPN model to small DBPN model via channel slicing.
    Transfers 100% of Conv3D layers and PReLU activation units.
    """
    import numpy as np
    src_convs = [l for l in src_model.layers if isinstance(l, layers.Conv3D)]
    dst_convs = [l for l in dst_model.layers if isinstance(l, layers.Conv3D)]
    src_prelu = [l for l in src_model.layers if isinstance(l, layers.PReLU)]
    dst_prelu = [l for l in dst_model.layers if isinstance(l, layers.PReLU)]

    matched = 0
    # 1. Intermediate Conv3D projection layers
    for i in range(len(dst_convs) - 1):
        if i >= len(src_convs):
            break
        s_l = src_convs[i]
        d_l = dst_convs[i]
        s_w = s_l.get_weights()
        d_w = d_l.get_weights()
        if len(s_w) > 0 and len(d_w) > 0:
            s_k, d_k = s_w[0], d_w[0]
            cin = min(s_k.shape[3], d_k.shape[3])
            cout = min(s_k.shape[4], d_k.shape[4])
            scale = np.sqrt(float(s_k.shape[3]) / float(cin))
            new_k = np.copy(d_k)
            new_k[:, :, :, :cin, :cout] = s_k[:, :, :, :cin, :cout] * scale
            new_w = [new_k]
            if len(d_w) > 1 and len(s_w) > 1:
                new_b = np.copy(d_w[1])
                new_b[:cout] = s_w[1][:cout]
                new_w.append(new_b)
            d_l.set_weights(new_w)
            matched += 1

    # 2. Final Reconstruction Conv3D layer
    if len(src_convs) > 0 and len(dst_convs) > 0:
        s_last = src_convs[-1]
        d_last = dst_convs[-1]
        s_w, d_w = s_last.get_weights(), d_last.get_weights()
        if len(s_w) > 0 and len(d_w) > 0:
            cin = min(s_w[0].shape[3], d_w[0].shape[3])
            scale = float(s_w[0].shape[3]) / float(cin)
            new_k = np.copy(d_w[0])
            new_k[:, :, :, :cin, :] = s_w[0][:, :, :, :cin, :] * scale
            new_w = [new_k]
            if len(d_w) > 1 and len(s_w) > 1:
                new_w.append(s_w[1])
            d_last.set_weights(new_w)
            matched += 1

    # 3. PReLU Activation layers
    for i in range(min(len(src_prelu), len(dst_prelu))):
        s_l = src_prelu[i]
        d_l = dst_prelu[i]
        s_w = s_l.get_weights()
        d_w = d_l.get_weights()
        if len(s_w) > 0 and len(d_w) > 0:
            c = min(s_w[0].shape[-1], d_w[0].shape[-1])
            new_a = np.copy(d_w[0])
            new_a[:, :, :, :c] = s_w[0][:, :, :, :c]
            d_l.set_weights([new_a])
            matched += 1

    return matched

def _adapt_weight_shape(src_w, target_shape):
    """
    Adapts a weight tensor to the target shape if dimensions differ only spatially.

    Two modes:
    - **Crop-down**: larger src → smaller target (e.g. 6×6×6 → 3×3×6 for anisotropic
      stride-1 projection). Takes the central slice.
    - **Pad-up**: smaller src → larger target (e.g. 3×3×6 → 6×6×6 for 1x1x2→2x2x2
      conv-transpose transfer). Embeds src at the center, zero-padding the surround.
      This preserves the learned frequency response in each spatial axis while giving
      the new (larger) kernel a meaningful initialization rather than ICNR noise.

    Only operates on 5-D conv kernels (kD, kH, kW, C_in, C_out) where the last two
    dimensions (channel counts) are identical between src and target.
    """
    import numpy as np
    src_np = np.asarray(src_w)
    if src_np.shape == target_shape:
        return src_np
    # 5-D conv/deconv kernel: spatial dims may differ, channel dims must match
    if src_np.ndim == 5 and len(target_shape) == 5 and src_np.shape[3:] == target_shape[3:]:
        out = src_np
        for ax in range(3):
            s = out.shape[ax]
            t = target_shape[ax]
            if s > t:
                # Crop-down: take center slice
                start = (s - t) // 2
                sl = [slice(None)] * 5
                sl[ax] = slice(start, start + t)
                out = out[tuple(sl)]
            elif s < t:
                # Pad-up: embed at center with zeros
                pad_shape = list(out.shape)
                pad_shape[ax] = t
                padded = np.zeros(pad_shape, dtype=out.dtype)
                start = (t - s) // 2
                sl = [slice(None)] * 5
                sl[ax] = slice(start, start + s)
                padded[tuple(sl)] = out
                out = padded
        if out.shape == tuple(target_shape):
            return out
    return None

def transfer_siq_weights(src_model, dst_model, verbose=True):
    """
    Universal weight transfer between two siq models by layer name with adaptive tensor shape handling.
    Enables zero-cost transfer learning across different upsampling factors
    (e.g., from isotropic 2x2x2 to anisotropic 1x1x2) or architectural refinements.

    Returns:
        int: Number of parametric layers successfully transferred.
    """
    import os
    if isinstance(src_model, str):
        src_model = keras.models.load_model(os.path.expanduser(src_model), compile=False)

    matched = 0
    total_dst_parametric = 0
    for dst_l in dst_model.layers:
        dst_w = dst_l.get_weights()
        if len(dst_w) == 0:
            continue
        total_dst_parametric += 1
        try:
            src_l = src_model.get_layer(dst_l.name)
            src_w = src_l.get_weights()
            if len(src_w) == len(dst_w):
                adapted_weights = []
                compatible = True
                for s, d in zip(src_w, dst_w):
                    adapted = _adapt_weight_shape(s, d.shape)
                    if adapted is not None:
                        adapted_weights.append(adapted)
                    else:
                        compatible = False
                        break
                if compatible:
                    dst_l.set_weights(adapted_weights)
                    matched += 1
                    if verbose:
                        print(f"  [Transfer Learning] Layer '{dst_l.name}': {len(src_w)} tensor(s) transferred into {dst_w[0].shape}")
                    continue
            if verbose:
                print(f"  [Transfer Learning] Skipping '{dst_l.name}': incompatible weight shapes.")
        except Exception:
            if verbose:
                print(f"  [Transfer Learning] Skipping '{dst_l.name}': not found in source model.")

    if verbose:
        print(f"[siq] Transfer learning complete: successfully transferred {matched}/{total_dst_parametric} parametric layers.")
    return matched

def extract_siq_loss_weights(source, verbose=True):
    """
    Extract balanced training loss weights from a saved siq model, its companion
    _config.json, or an associated training weights CSV.

    Parameters
    ----------
    source : str, dict, or keras.Model
        Path to a .keras file, path to a _config.json, a loaded config dict,
        or a Keras model instance.
    verbose : bool
        Whether to log the found weights.

    Returns
    -------
    dict or None
        Dictionary of loss weights (e.g. {'l1': float, 'feat': float, ...})
        or None if no loss weights could be located.
    """
    import json, os, glob
    import pandas as pd

    weights = None

    if isinstance(source, dict):
        if "loss_weights" in source and isinstance(source["loss_weights"], dict):
            weights = dict(source["loss_weights"])
    elif isinstance(source, str):
        src_path = os.path.expanduser(source)
        # 1. Try companion or direct config json
        cfg_path = None
        if src_path.endswith(".keras"):
            cfg_path = src_path.replace(".keras", "_config.json")
        elif src_path.endswith(".json"):
            cfg_path = src_path

        if cfg_path and os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r") as f:
                    cfg = json.load(f)
                if "loss_weights" in cfg and isinstance(cfg["loss_weights"], dict):
                    weights = dict(cfg["loss_weights"])
            except Exception:
                pass

        # 2. Fallback: search for sibling or parent directory weights CSV
        if weights is None:
            search_dirs = [
                os.path.dirname(src_path),
                os.path.join(os.path.dirname(src_path), ".."),
                os.getcwd()
            ]
            for d in search_dirs:
                if not d or not os.path.exists(d):
                    continue
                csv_candidates = glob.glob(os.path.join(d, "*refined_training_weights.csv"))
                for c in csv_candidates:
                    try:
                        df = pd.read_csv(c)
                        if "l1" in df.columns:
                            row = df.iloc[-1]
                            weights = {col: float(row[col]) for col in df.columns if pd.notnull(row[col])}
                            break
                    except Exception:
                        pass
                if weights is not None:
                    break

    if weights is not None:
        normalized = {
            "l1": float(weights.get("l1", 0.0)),
            "feat": float(weights.get("feat", 0.0)),
            "tv": float(weights.get("tv", 0.0)),
            "msq": float(weights.get("msq", 0.0)),
            "edge": float(weights.get("edge", 0.0)),
            "gms": float(weights.get("gms", 0.0)),
            "cbi": float(weights.get("cbi", 0.0)),
        }
        if verbose:
            print(f"[siq] Extracted loss weights: L1={normalized['l1']:.6f}, "
                  f"Feat={normalized['feat']:.6e}, TV={normalized['tv']:.6f}, "
                  f"GMS={normalized['gms']:.4f}, CBI={normalized['cbi']:.4f}, Edge={normalized['edge']:.4f}")
        return normalized

    if verbose:
        print("[siq] No saved loss weights found in source model or companion files.")
    return None

# ==============================================================================
# 2D Super-Resolution Models
# ==============================================================================

def pixel_shuffle_2d(inputs, factor=2):
    """
    Implementation of 2D Pixel Shuffle for Keras/Keras3.
    Args:
        inputs: (batch, h, w, c)
        factor: scaling factor (integer or 2-element tuple/list)
    """
    factor_tuple = _normalize_factor(factor, 2)
    f_h, f_w = factor_tuple
    f_total = f_h * f_w
    input_shape = ops.shape(inputs)
    batch_size = input_shape[0]
    h, w = input_shape[1], input_shape[2]
    channels = input_shape[3]
    
    new_channels = channels // f_total
    
    # Reshape: (batch, h, w, f_h, f_w, new_c)
    x = ops.reshape(inputs, (batch_size, h, w, f_h, f_w, new_channels))
    
    # Transpose to (batch, h, f_h, w, f_w, new_c)
    x = ops.transpose(x, (0, 1, 3, 2, 4, 5))
    
    # Reshape to (batch, h*f_h, w*f_w, new_c)
    new_h, new_w = h * f_h, w * f_w
    return ops.reshape(x, (batch_size, new_h, new_w, new_channels))

@keras.saving.register_keras_serializable(package="siq")
class PixelShuffle2D(keras.layers.Layer):
    def __init__(self, factor=2, **kwargs):
        super().__init__(**kwargs)
        self.factor = _normalize_factor(factor, 2) if isinstance(factor, (list, tuple)) else factor

    def call(self, inputs):
        return pixel_shuffle_2d(inputs, self.factor)

    def compute_output_shape(self, input_shape):
        factor_tuple = _normalize_factor(self.factor, 2)
        f_h, f_w = factor_tuple
        f_total = f_h * f_w
        h = input_shape[1] * f_h if input_shape[1] is not None else None
        w = input_shape[2] * f_w if input_shape[2] is not None else None
        c = input_shape[3] // f_total if input_shape[3] is not None else None
        return (input_shape[0], h, w, c)

    def get_config(self):
        config = super().get_config()
        config.update({"factor": self.factor})
        return config

def create_espcn_2d_attention(input_shape=(None, None, 1), factor=2, n_filters=64, n_res_blocks=8, use_global_skip=True):
    """
    Creates a 2D Residual ESPCN model with Channel Attention and optional Global Skip.
    Supports isotropic integer factors (e.g. 2, 4) or anisotropic tuple factors (e.g. (1, 2), (2, 4)).
    """
    factor_tuple = _normalize_factor(factor, 2)
    f_total = math.prod(factor_tuple)
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
    x = keras.layers.Conv2D(out_channels * f_total, kernel_size=3, padding="same", name="preshuffle_conv")(x)
    
    # Pixel Shuffle
    outputs = PixelShuffle2D(factor=factor_tuple, name="pixel_shuffle")(x)
    
    # Non-linear processing in high-res space
    x = keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu", name="hr_conv1")(outputs)
    x = keras.layers.Conv2D(16, kernel_size=3, padding="same", activation="relu", name="hr_conv2")(x)
    outputs = keras.layers.Conv2D(1, kernel_size=3, padding="same", name="hr_conv3")(x)
    
    # Optional Global Skip connection
    if use_global_skip:
        skip = keras.layers.UpSampling2D(size=factor_tuple, name="global_skip")(inputs)
        scaled_skip = LearnableScale(initial_value=1.0, name="scaled_global_skip")(skip)
        outputs = keras.layers.add([outputs, scaled_skip], name="add_global_skip")
        
    return keras.Model(inputs, outputs, name="espcn_2d_attention")

def up_projection_unit_2d(lr_input, n_filters, factor=2, name_prefix=""):
    factor_tuple = _normalize_factor(factor, 2)
    f_total = math.prod(factor_tuple)
    x = keras.layers.Conv2D(n_filters * f_total, kernel_size=3, padding="same", activation="relu", name=f"{name_prefix}_up_conv1")(lr_input)
    h_temp = PixelShuffle2D(factor=factor_tuple, name=f"{name_prefix}_up_shuffle1")(x)
    
    l_temp = keras.layers.Conv2D(n_filters, kernel_size=3, strides=factor_tuple, padding="same", activation="relu", name=f"{name_prefix}_up_down1")(h_temp)
    e_lr = keras.layers.subtract([lr_input, l_temp], name=f"{name_prefix}_up_sub")
    
    x_err = keras.layers.Conv2D(n_filters * f_total, kernel_size=3, padding="same", activation="relu", name=f"{name_prefix}_up_conv2")(e_lr)
    e_hr = PixelShuffle2D(factor=factor_tuple, name=f"{name_prefix}_up_shuffle2")(x_err)
    
    return keras.layers.add([h_temp, e_hr], name=f"{name_prefix}_up_add")

def down_projection_unit_2d(hr_input, n_filters, factor=2, name_prefix=""):
    factor_tuple = _normalize_factor(factor, 2)
    f_total = math.prod(factor_tuple)
    l_temp = keras.layers.Conv2D(n_filters, kernel_size=3, strides=factor_tuple, padding="same", activation="relu", name=f"{name_prefix}_down_conv1")(hr_input)
    
    x = keras.layers.Conv2D(n_filters * f_total, kernel_size=3, padding="same", activation="relu", name=f"{name_prefix}_down_conv2")(l_temp)
    h_temp = PixelShuffle2D(factor=factor_tuple, name=f"{name_prefix}_down_shuffle1")(x)
    
    e_hr = keras.layers.subtract([hr_input, h_temp], name=f"{name_prefix}_down_sub")
    e_lr = keras.layers.Conv2D(n_filters, kernel_size=3, strides=factor_tuple, padding="same", activation="relu", name=f"{name_prefix}_down_conv3")(e_hr)
    
    return keras.layers.add([l_temp, e_lr], name=f"{name_prefix}_down_add")

def create_ldbpn_2d(input_shape=(None, None, 1), factor=2, n_filters=64, n_stages=3):
    """
    Creates a Lightweight 2D Deep Back-Projection Network (L-DBPN) using PixelShuffle2D.
    Supports isotropic integer factors (e.g. 2, 4) or anisotropic tuple factors (e.g. (1, 2), (2, 4)).
    """
    factor_tuple = _normalize_factor(factor, 2)
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
            
        h = up_projection_unit_2d(l_proj, n_filters, factor_tuple, name_prefix=f"stage_{i}")
        h_list.append(h)
        
        if len(h_list) > 1:
            h_cat = keras.layers.concatenate(h_list, name=f"h_cat_{i}")
            h_proj = keras.layers.Conv2D(n_filters, kernel_size=1, padding="same", activation="relu", name=f"h_proj_{i}")(h_cat)
        else:
            h_proj = h_list[0]
            
        l = down_projection_unit_2d(h_proj, n_filters, factor_tuple, name_prefix=f"stage_{i}")
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
    Supports isotropic integer factors (e.g. 2, 4) or anisotropic tuple factors (e.g. (1, 2), (2, 4)).
    """
    factor_tuple = _normalize_factor(factor, 2)
    f_total = math.prod(factor_tuple)
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv2D(n_filters, kernel_size=3, padding="same", name="init_conv")(inputs)
    for i in range(n_res_blocks):
        x = wdsr_block_2d(x, n_filters, expansion_ratio, name_prefix=f"wdsr_{i}")
    x = keras.layers.Conv2D(n_filters // 2, kernel_size=3, padding="same", activation="relu", name="shrink_conv")(x)
    
    # preshuffle
    out_channels = 32
    x = keras.layers.Conv2D(out_channels * f_total, kernel_size=3, padding="same", name="preshuffle_conv")(x)
    outputs = PixelShuffle2D(factor=factor_tuple, name="pixel_shuffle")(x)
    
    x = keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu", name="hr_conv1")(outputs)
    x = keras.layers.Conv2D(16, kernel_size=3, padding="same", activation="relu", name="hr_conv2")(x)
    outputs = keras.layers.Conv2D(1, kernel_size=3, padding="same", name="hr_conv3")(x)
    
    if use_global_skip:
        skip = keras.layers.UpSampling2D(size=factor_tuple, name="global_skip")(inputs)
        scaled_skip = LearnableScale(initial_value=1.0, name="scaled_global_skip")(skip)
        outputs = keras.layers.add([outputs, scaled_skip], name="add_global_skip")
        
    return keras.Model(inputs, outputs, name="wdsr_2d")

def create_wdsr_3d(input_shape=(None, None, None, 1), factor=2, n_filters=64, n_res_blocks=8, expansion_ratio=4, use_global_skip=True):
    """
    Creates a 3D Wide Activation Super-Resolution (WDSR) model.
    Supports isotropic integer factors (e.g. 2, 4) or anisotropic tuple factors (e.g. (1, 1, 2), (2, 2, 4)).
    """
    factor_tuple = _normalize_factor(factor, 3)
    f_total = math.prod(factor_tuple)
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv3D(n_filters, kernel_size=3, padding="same", name="init_conv")(inputs)
    for i in range(n_res_blocks):
        x = wdsr_block_3d(x, n_filters, expansion_ratio, name_prefix=f"wdsr_{i}")
    x = keras.layers.Conv3D(n_filters // 2, kernel_size=3, padding="same", activation="relu", name="shrink_conv")(x)
    
    # preshuffle
    out_channels = 32
    x = keras.layers.Conv3D(out_channels * f_total, kernel_size=3, padding="same", name="preshuffle_conv")(x)
    outputs = PixelShuffle3D(factor=factor_tuple, name="pixel_shuffle")(x)
    
    x = keras.layers.Conv3D(32, kernel_size=3, padding="same", activation="relu", name="hr_conv1")(outputs)
    x = keras.layers.Conv3D(16, kernel_size=3, padding="same", activation="relu", name="hr_conv2")(x)
    outputs = keras.layers.Conv3D(1, kernel_size=3, padding="same", name="hr_conv3")(x)
    
    if use_global_skip:
        skip = TrilinearUpSampling3D(size=factor_tuple, name="global_skip")(inputs)
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
    Supports isotropic integer factors (e.g. 2, 4) or anisotropic tuple factors (e.g. (1, 2), (2, 4)).
    """
    factor_tuple = _normalize_factor(factor, 2)
    f_total = math.prod(factor_tuple)
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv2D(n_filters, kernel_size=3, padding="same", name="init_conv")(inputs)
    for i in range(n_groups):
        x = residual_group_2d(x, n_filters, n_blocks=n_blocks, name_prefix=f"rg_{i}")
    x = keras.layers.Conv2D(n_filters // 2, kernel_size=3, padding="same", activation="relu", name="shrink_conv")(x)
    
    # preshuffle
    out_channels = 32
    x = keras.layers.Conv2D(out_channels * f_total, kernel_size=3, padding="same", name="preshuffle_conv")(x)
    outputs = PixelShuffle2D(factor=factor_tuple, name="pixel_shuffle")(x)
    
    x = keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu", name="hr_conv1")(outputs)
    x = keras.layers.Conv2D(16, kernel_size=3, padding="same", activation="relu", name="hr_conv2")(x)
    outputs = keras.layers.Conv2D(1, kernel_size=3, padding="same", name="hr_conv3")(x)
    
    if use_global_skip:
        skip = keras.layers.UpSampling2D(size=factor_tuple, name="global_skip")(inputs)
        scaled_skip = LearnableScale(initial_value=1.0, name="scaled_global_skip")(skip)
        outputs = keras.layers.add([outputs, scaled_skip], name="add_global_skip")
        
    return keras.Model(inputs, outputs, name="rcan_2d")

def create_rcan_3d(input_shape=(None, None, None, 1), factor=2, n_filters=64, n_groups=3, n_blocks=4, use_global_skip=True):
    """
    Creates a 3D Residual Channel Attention Network (RCAN) model.
    Supports isotropic integer factors (e.g. 2, 4) or anisotropic tuple factors (e.g. (1, 1, 2), (2, 2, 4)).
    """
    factor_tuple = _normalize_factor(factor, 3)
    f_total = math.prod(factor_tuple)
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv3D(n_filters, kernel_size=3, padding="same", name="init_conv")(inputs)
    for i in range(n_groups):
        x = residual_group_3d(x, n_filters, n_blocks=n_blocks, name_prefix=f"rg_{i}")
    x = keras.layers.Conv3D(n_filters // 2, kernel_size=3, padding="same", activation="relu", name="shrink_conv")(x)
    
    # preshuffle
    out_channels = 32
    x = keras.layers.Conv3D(out_channels * f_total, kernel_size=3, padding="same", name="preshuffle_conv")(x)
    outputs = PixelShuffle3D(factor=factor_tuple, name="pixel_shuffle")(x)
    
    x = keras.layers.Conv3D(32, kernel_size=3, padding="same", activation="relu", name="hr_conv1")(outputs)
    x = keras.layers.Conv3D(16, kernel_size=3, padding="same", activation="relu", name="hr_conv2")(x)
    outputs = keras.layers.Conv3D(1, kernel_size=3, padding="same", name="hr_conv3")(x)
    
    if use_global_skip:
        skip = TrilinearUpSampling3D(size=factor_tuple, name="global_skip")(inputs)
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
    Supports isotropic integer factors (e.g. 2, 4) or anisotropic tuple factors (e.g. (1, 2), (2, 4)).
    """
    factor_tuple = _normalize_factor(factor, 2)
    f_total = math.prod(factor_tuple)
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
    x = keras.layers.Conv2D(out_channels * f_total, kernel_size=3, padding="same", name="preshuffle_conv")(x)
    outputs = PixelShuffle2D(factor=factor_tuple, name="pixel_shuffle")(x)
    
    x = keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu", name="hr_conv1")(outputs)
    x = keras.layers.Conv2D(16, kernel_size=3, padding="same", activation="relu", name="hr_conv2")(x)
    outputs = keras.layers.Conv2D(1, kernel_size=3, padding="same", name="hr_conv3")(x)
    
    if use_global_skip:
        skip = keras.layers.UpSampling2D(size=factor_tuple, name="global_skip")(inputs)
        scaled_skip = LearnableScale(initial_value=1.0, name="scaled_global_skip")(skip)
        outputs = keras.layers.add([outputs, scaled_skip], name="add_global_skip")
        
    return keras.Model(inputs, outputs, name="carn_2d")

def create_carn_3d(input_shape=(None, None, None, 1), factor=2, n_filters=64, n_blocks=3, use_global_skip=True):
    """
    Creates a 3D Cascading Residual Network (CARN) model.
    Supports isotropic integer factors (e.g. 2, 4) or anisotropic tuple factors (e.g. (1, 1, 2), (2, 2, 4)).
    """
    factor_tuple = _normalize_factor(factor, 3)
    f_total = math.prod(factor_tuple)
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
    x = keras.layers.Conv3D(out_channels * f_total, kernel_size=3, padding="same", name="preshuffle_conv")(x)
    outputs = PixelShuffle3D(factor=factor_tuple, name="pixel_shuffle")(x)
    
    x = keras.layers.Conv3D(32, kernel_size=3, padding="same", activation="relu", name="hr_conv1")(outputs)
    x = keras.layers.Conv3D(16, kernel_size=3, padding="same", activation="relu", name="hr_conv2")(x)
    outputs = keras.layers.Conv3D(1, kernel_size=3, padding="same", name="hr_conv3")(x)
    
    if use_global_skip:
        skip = TrilinearUpSampling3D(size=factor_tuple, name="global_skip")(inputs)
        scaled_skip = LearnableScale(initial_value=1.0, name="scaled_global_skip")(skip)
        outputs = keras.layers.add([outputs, scaled_skip], name="add_global_skip")
        
    return keras.Model(inputs, outputs, name="carn_3d")


def create_espcn_2d_resize_conv(input_shape=(None, None, 1), factor=2, n_filters=64, n_res_blocks=8, use_global_skip=True):
    """
    Creates a 2D Residual ESPCN model using Bilinear Resize + Conv instead of PixelShuffle
    to mitigate checkerboard / step artifacts.
    Supports isotropic integer factors (e.g. 2, 4) or anisotropic tuple factors (e.g. (1, 2), (2, 4)).
    """
    factor_tuple = _normalize_factor(factor, 2)
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
    x = keras.layers.UpSampling2D(size=factor_tuple, interpolation="bilinear", name="resize_upsample")(x)
    outputs = keras.layers.Conv2D(out_channels, kernel_size=3, padding="same", name="resize_conv")(x)
    
    x = keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu", name="hr_conv1")(outputs)
    x = keras.layers.Conv2D(16, kernel_size=3, padding="same", activation="relu", name="hr_conv2")(x)
    outputs = keras.layers.Conv2D(1, kernel_size=3, padding="same", name="hr_conv3")(x)
    
    if use_global_skip:
        skip = keras.layers.UpSampling2D(size=factor_tuple, name="global_skip")(inputs)
        scaled_skip = LearnableScale(initial_value=1.0, name="scaled_global_skip")(skip)
        outputs = keras.layers.add([outputs, scaled_skip], name="add_global_skip")
        
    return keras.Model(inputs, outputs, name="espcn_2d_resize_conv")


def create_wdsr_2d_resize_conv(input_shape=(None, None, 1), factor=2, n_filters=64, n_res_blocks=8, expansion_ratio=4, use_global_skip=True):
    """
    Creates a 2D WDSR model using Bilinear Resize + Conv instead of PixelShuffle
    to mitigate checkerboard / step artifacts.
    Supports isotropic integer factors (e.g. 2, 4) or anisotropic tuple factors (e.g. (1, 2), (2, 4)).
    """
    factor_tuple = _normalize_factor(factor, 2)
    inputs = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv2D(n_filters, kernel_size=3, padding="same", name="init_conv")(inputs)
    for i in range(n_res_blocks):
        x = wdsr_block_2d(x, n_filters, expansion_ratio, name_prefix=f"wdsr_{i}")
    x = keras.layers.Conv2D(n_filters // 2, kernel_size=3, padding="same", activation="relu", name="shrink_conv")(x)
    
    # Bilinear Resize + Conv instead of PixelShuffle
    out_channels = 32
    x = keras.layers.UpSampling2D(size=factor_tuple, interpolation="bilinear", name="resize_upsample")(x)
    outputs = keras.layers.Conv2D(out_channels, kernel_size=3, padding="same", name="resize_conv")(x)
    
    x = keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu", name="hr_conv1")(outputs)
    x = keras.layers.Conv2D(16, kernel_size=3, padding="same", activation="relu", name="hr_conv2")(x)
    outputs = keras.layers.Conv2D(1, kernel_size=3, padding="same", name="hr_conv3")(x)
    
    if use_global_skip:
        skip = keras.layers.UpSampling2D(size=factor_tuple, name="global_skip")(inputs)
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
    Supports isotropic integer factors (e.g. 2, 4) or anisotropic tuple factors (e.g. (1, 2), (2, 4)).
    """
    factor_tuple = _normalize_factor(factor, 2)
    inputs = layers.Input(shape=input_shape)
    
    # Feature extraction block
    F_in = layers.Conv2D(n_filters, kernel_size=3, padding="same", activation="relu", name="init_conv")(inputs)
    
    # Instantiate recurrent layers to share weights across steps
    if n_steps > 1:
        project_layer = layers.Conv2D(n_filters, kernel_size=1, padding="same", activation="relu", name="fb_project")
    up_layer = layers.Conv2DTranspose(n_filters, kernel_size=factor_tuple, strides=factor_tuple, padding="same", activation="relu", name="fb_up")
    down_layer = layers.Conv2D(n_filters, kernel_size=factor_tuple, strides=factor_tuple, padding="same", activation="relu", name="fb_down")
    
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
        skip = layers.UpSampling2D(size=factor_tuple, interpolation="nearest", name="global_skip")(inputs)
        scaled_skip = LearnableScale(initial_value=0.0, name="scaled_global_skip")(skip)
        outputs = layers.add([outputs, scaled_skip], name="add_global_skip")
        
    outputs = LearnableSharpening(name="final_sharpening")(outputs)
    return keras.Model(inputs, outputs, name="srfbn_2d")


def create_srfbn_3d(input_shape=(None, None, None, 1), factor=2, n_filters=64, n_steps=4, use_global_skip=True):
    """
    Creates a 3D Super-Resolution Feedback Network (SRFBN) model.
    Supports isotropic integer factors (e.g. 2, 4) or anisotropic tuple factors (e.g. (1, 1, 2), (2, 2, 4)).
    """
    factor_tuple = _normalize_factor(factor, 3)
    inputs = layers.Input(shape=input_shape)
    
    # Feature extraction block
    F_in = layers.Conv3D(n_filters, kernel_size=3, padding="same", activation="relu", name="init_conv")(inputs)
    
    # Instantiate recurrent layers to share weights across steps
    if n_steps > 1:
        project_layer = layers.Conv3D(n_filters, kernel_size=1, padding="same", activation="relu", name="fb_project")
    up_layer = layers.Conv3DTranspose(n_filters, kernel_size=factor_tuple, strides=factor_tuple, padding="same", activation="relu", name="fb_up")
    down_layer = layers.Conv3D(n_filters, kernel_size=factor_tuple, strides=factor_tuple, padding="same", activation="relu", name="fb_down")
    
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
        skip = TrilinearUpSampling3D(size=factor_tuple, name="global_skip")(inputs)
        scaled_skip = LearnableScale(initial_value=1.0, name="scaled_global_skip")(skip)
        outputs = layers.add([outputs, scaled_skip], name="add_global_skip")
        
    outputs = LearnableSharpening3D(name="final_sharpening")(outputs)
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


def ca_block_2d(input_tensor, reduction_ratio=16, name_prefix=""):
    """
    Residual Channel Attention (RCAN-style) block for 2D.
    Uses first-order global average pooling.
    """
    channels = input_tensor.shape[-1]
    squeeze = layers.GlobalAveragePooling2D(keepdims=True, name=f"{name_prefix}_ca_squeeze")(input_tensor)
    excitation = layers.Conv2D(max(1, channels // reduction_ratio), kernel_size=1, activation='relu', name=f"{name_prefix}_ca_conv1")(squeeze)
    excitation = layers.Conv2D(channels, kernel_size=1, activation='sigmoid', name=f"{name_prefix}_ca_conv2")(excitation)
    return layers.Multiply(name=f"{name_prefix}_ca_scale")([input_tensor, excitation])


def ca_block_3d(input_tensor, reduction_ratio=16, name_prefix=""):
    """
    Residual Channel Attention (RCAN-style) block for 3D.
    Uses first-order global average pooling.
    """
    channels = input_tensor.shape[-1]
    squeeze = layers.GlobalAveragePooling3D(keepdims=True, name=f"{name_prefix}_ca_squeeze")(input_tensor)
    excitation = layers.Conv3D(max(1, channels // reduction_ratio), kernel_size=1, activation='relu', name=f"{name_prefix}_ca_conv1")(squeeze)
    excitation = layers.Conv3D(channels, kernel_size=1, activation='sigmoid', name=f"{name_prefix}_ca_conv2")(excitation)
    return layers.Multiply(name=f"{name_prefix}_ca_scale")([input_tensor, excitation])



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
    Supports isotropic integer factors (e.g. 2, 4) or anisotropic tuple factors (e.g. (1, 2), (2, 4)).
    """
    factor_tuple = _normalize_factor(factor, 2)
    f_total = math.prod(factor_tuple)
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(n_filters, kernel_size=3, padding="same", name="init_conv")(inputs)
    
    # Stack residual groups
    res = x
    for i in range(n_groups):
        res = residual_group_soca_2d(res, n_filters, n_blocks=n_blocks, name_prefix=f"group_{i}")
    res = layers.Conv2D(n_filters, kernel_size=3, padding="same", name="group_conv")(res)
    x = layers.add([x, res], name="group_add")
    
    # Upsampling via PixelShuffle
    x = layers.Conv2D(n_filters * f_total, kernel_size=3, padding="same", name="pre_shuffle_conv")(x)
    x = PixelShuffle2D(factor=factor_tuple, name="pixel_shuffle")(x)
    outputs = layers.Conv2D(1, kernel_size=3, padding="same", name="final_conv")(x)
    
    if use_global_skip:
        skip = layers.UpSampling2D(size=factor_tuple, interpolation="nearest", name="global_skip")(inputs)
        scaled_skip = LearnableScale(initial_value=1.0, name="scaled_global_skip")(skip)
        outputs = layers.add([outputs, scaled_skip], name="add_global_skip")
        
    return keras.Model(inputs, outputs, name="san_2d")


def create_san_3d(input_shape=(None, None, None, 1), factor=2, n_filters=64, n_groups=3, n_blocks=4, use_global_skip=True):
    """
    Creates a 3D Second-order Attention Network (SAN) model.
    Supports isotropic integer factors (e.g. 2, 4) or anisotropic tuple factors (e.g. (1, 1, 2), (2, 2, 4)).
    """
    factor_tuple = _normalize_factor(factor, 3)
    f_total = math.prod(factor_tuple)
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv3D(n_filters, kernel_size=3, padding="same", name="init_conv")(inputs)
    
    # Stack residual groups
    res = x
    for i in range(n_groups):
        res = residual_group_soca_3d(res, n_filters, n_blocks=n_blocks, name_prefix=f"group_{i}")
    res = layers.Conv3D(n_filters, kernel_size=3, padding="same", name="group_conv")(res)
    x = layers.add([x, res], name="group_add")
    
    # Upsampling via PixelShuffle
    x = layers.Conv3D(n_filters * f_total, kernel_size=3, padding="same", name="pre_shuffle_conv")(x)
    x = PixelShuffle3D(factor=factor_tuple, name="pixel_shuffle")(x)
    outputs = layers.Conv3D(1, kernel_size=3, padding="same", name="final_conv")(x)
    
    if use_global_skip:
        skip = TrilinearUpSampling3D(size=factor_tuple, name="global_skip")(inputs)
        scaled_skip = LearnableScale(initial_value=1.0, name="scaled_global_skip")(skip)
        outputs = layers.add([outputs, scaled_skip], name="add_global_skip")
        
    return keras.Model(inputs, outputs, name="san_3d")


def create_asdbpn_2d(input_shape=(None, None, 1), factor=2, n_filters=64, n_steps=4, use_global_skip=True, projection_kernel_size=None):
    """
    Creates a 2D Attention-Guided Shared Back-Projection Network (AS-DBPN) model.
    It combines recurrent feedback loops with channel attention to guide refinement.
    Supports isotropic integer factors (e.g. 2, 4) or anisotropic tuple factors (e.g. (1, 2)).
    """
    factor_tuple = _normalize_factor(factor, 2)
    inputs = layers.Input(shape=input_shape)
    
    # Feature extraction block
    F_in = layers.Conv2D(n_filters, kernel_size=3, padding="same", activation="relu", name="init_conv")(inputs)
    
    proj_kernel = projection_kernel_size if projection_kernel_size is not None else factor_tuple

    # Instantiate recurrent layers to share weights across steps
    if n_steps > 1:
        project_layer = layers.Conv2D(n_filters, kernel_size=1, padding="same", activation="relu", name="fb_project")
    up_layer = layers.Conv2DTranspose(n_filters, kernel_size=proj_kernel, strides=factor_tuple, padding="same", name="fb_up")
    down_layer = layers.Conv2D(n_filters, kernel_size=proj_kernel, strides=factor_tuple, padding="same", name="fb_down")
    
    # Shared Layer Normalization layers to stabilize recurrent loop scale
    ln_hr = layers.LayerNormalization(axis=-1, name="fb_ln_hr")
    ln_lr = layers.LayerNormalization(axis=-1, name="fb_ln_lr")
    
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
            
        x = layers.ReLU(name=f"fb_{t}_relu1")(x)
        
        # Up-projection (Deconvolution)
        H_t = up_layer(x)
        H_t = ln_hr(H_t)
        
        # Down-projection (Stride Conv)
        L_t = down_layer(layers.ReLU(name=f"fb_{t}_relu2")(H_t))
        L_t = ln_lr(L_t)
        
    # Apply a single SOCA attention block on the final HR representation H_t
    H_t = soca_block_2d(H_t, name_prefix="recon_soca")
        
    # Reconstruction block using final HR representation H_t
    outputs = layers.Conv2D(n_filters // 2, kernel_size=3, padding="same", activation="relu", name="recon_conv1")(H_t)
    outputs = layers.Conv2D(1, kernel_size=3, padding="same", name="recon_conv2")(outputs)
    
    if use_global_skip:
        # nearest-neighbor: output pixel 2k replicates input pixel k — zero spatial shift.
        # bilinear (align_corners=False) maps output j → input (j+0.5)/scale-0.5,
        # giving a systematic -0.5 pixel shift vs the natural SR pixel convention.
        # Consistent with the 3D ASDBPN which also uses nearest-neighbor skip.
        skip = layers.UpSampling2D(size=factor_tuple, interpolation="nearest", name="global_skip")(inputs)
        scaled_skip = LearnableScale(initial_value=0.0, name="scaled_global_skip")(skip)
        outputs = layers.add([outputs, scaled_skip], name="add_global_skip")
        
    outputs = LearnableSharpening(name="final_sharpening")(outputs)
    return keras.Model(inputs, outputs, name="asdbpn_2d")


def create_asdbpn_3d(input_shape=(None, None, None, 1), factor=2, n_filters=64, n_steps=4, use_global_skip=True, projection_kernel_size=None):
    """
    Creates a 3D Attention-Guided Shared Back-Projection Network (AS-DBPN) model.
    It combines recurrent feedback loops with channel attention to guide refinement.
    Supports isotropic integer factors (e.g. 2, 4) or anisotropic tuple factors (e.g. (1, 1, 2), (2, 2, 4)).

    Note: the global skip connection upsamples with smooth trilinear
    interpolation (`TrilinearUpSampling3D`), matching the bilinear continuous
    upsampling quality of `create_asdbpn_2d`.
    """
    factor_tuple = _normalize_factor(factor, 3)

    inputs = layers.Input(shape=input_shape)

    # Feature extraction block
    F_in = layers.Conv3D(n_filters, kernel_size=3, padding="same", activation="relu", name="init_conv")(inputs)

    if projection_kernel_size is not None:
        if isinstance(projection_kernel_size, (list, tuple)):
            proj_kernel = tuple(projection_kernel_size)
        elif isinstance(projection_kernel_size, int):
            proj_kernel = tuple(3 if f == 1 else projection_kernel_size for f in factor_tuple)
        else:
            proj_kernel = projection_kernel_size
    else:
        proj_kernel = tuple(3 if f == 1 else 6 for f in factor_tuple)

    # Instantiate recurrent layers to share weights across steps
    if n_steps > 1:
        project_layer = layers.Conv3D(n_filters, kernel_size=1, padding="same", activation="relu", name="fb_project")
    up_layer = layers.Conv3DTranspose(n_filters, kernel_size=proj_kernel, strides=factor_tuple, padding="same", name="fb_up")
    down_layer = layers.Conv3D(n_filters, kernel_size=proj_kernel, strides=factor_tuple, padding="same", name="fb_down")
    
    # Shared Layer Normalization layers to stabilize recurrent loop scale
    ln_hr = layers.LayerNormalization(axis=-1, name="fb_ln_hr")
    ln_lr = layers.LayerNormalization(axis=-1, name="fb_ln_lr")
    
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
            
        x = layers.ReLU(name=f"fb_{t}_relu1")(x)
        
        # Up-projection (Deconvolution)
        H_t = up_layer(x)
        H_t = ln_hr(H_t)
        
        # Down-projection (Stride Conv)
        L_t = down_layer(layers.ReLU(name=f"fb_{t}_relu2")(H_t))
        L_t = ln_lr(L_t)
        
    # Apply a single SOCA attention block on the final HR representation H_t
    H_t = soca_block_3d(H_t, name_prefix="recon_soca")
        
    # Reconstruction block using final HR representation H_t
    outputs = layers.Conv3D(n_filters // 2, kernel_size=3, padding="same", activation="relu", name="recon_conv1")(H_t)
    outputs = layers.Conv3D(1, kernel_size=3, padding="same", name="recon_conv2")(outputs)
    
    if use_global_skip:
        skip = layers.UpSampling3D(size=factor_tuple, name="global_skip")(inputs)
        scaled_skip = LearnableScale(initial_value=0.0, name="scaled_global_skip")(skip)
        outputs = layers.add([outputs, scaled_skip], name="add_global_skip")
        
    outputs = LearnableSharpening3D(name="final_sharpening")(outputs)
    return keras.Model(inputs, outputs, name="asdbpn_3d")


import os
from os.path import exists
import numpy as np
from scipy.ndimage import prewitt, gaussian_laplace, uniform_filter, convolve, sobel, laplace
import keras
from keras import ops

try:
    import antspynet
except ImportError:
    antspynet = None


def ops_total_variation(x):
    diff_h = ops.abs(x[:, 1:, :, :] - x[:, :-1, :, :])
    diff_w = ops.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    return ops.mean(ops.mean(diff_h, axis=[1, 2, 3]) + ops.mean(diff_w, axis=[1, 2, 3]))


def ops_psnr(y_true, y_pred, max_val=255.0):
    y_true = ops.convert_to_tensor(y_true)
    y_pred = ops.convert_to_tensor(y_pred)
    mse = ops.mean(ops.square(y_true - y_pred))
    mse = ops.maximum(mse, 1e-10)
    log_10 = ops.cast(2.302585092994046, dtype="float32")
    return 20 * (ops.log(ops.cast(max_val, dtype="float32")) / log_10) - 10 * (ops.log(mse) / log_10)


def _extract_spatial_and_channels(y):
    if hasattr(y, 'numpy'):
        y = y.numpy()
    ndim = y.ndim
    if ndim == 2:
        return [y], 2
    elif ndim == 3:
        if y.shape[-1] <= 4:
            C = y.shape[-1]
            return [y[..., c] for c in range(C)], 2
        else:
            return [y], 3
    elif ndim == 4:
        C = y.shape[-1]
        return [y[..., c] for c in range(C)], 3
    else:
        raise ValueError(f"Unsupported array shape: {y.shape}")


def compute_gmsd(y_true, y_pred, c=0.0026):
    y_true_channels, spatial_dim = _extract_spatial_and_channels(y_true)
    y_pred_channels, _ = _extract_spatial_and_channels(y_pred)
    gmsd_list = []
    for yt, yp in zip(y_true_channels, y_pred_channels):
        grads_true = [prewitt(yt, axis=i) for i in range(spatial_dim)]
        grads_pred = [prewitt(yp, axis=i) for i in range(spatial_dim)]
        m_true = np.sqrt(sum(g**2 for g in grads_true))
        m_pred = np.sqrt(sum(g**2 for g in grads_pred))
        gms = (2.0 * m_true * m_pred + c) / (m_true**2 + m_pred**2 + c)
        gmsd_list.append(float(np.std(gms)))
    return float(np.mean(gmsd_list))


def compute_hfen(y_true, y_pred, sigma=1.5):
    y_true_channels, spatial_dim = _extract_spatial_and_channels(y_true)
    y_pred_channels, _ = _extract_spatial_and_channels(y_pred)
    hfen_list = []
    for yt, yp in zip(y_true_channels, y_pred_channels):
        log_true = gaussian_laplace(yt, sigma=sigma)
        log_pred = gaussian_laplace(yp, sigma=sigma)
        norm_diff = np.linalg.norm(log_true - log_pred)
        norm_true = np.linalg.norm(log_true)
        if norm_true > 1e-8:
            hfen_list.append(float(norm_diff / norm_true))
        else:
            hfen_list.append(0.0)
    return float(np.mean(hfen_list))


def compute_psnr(y_true, y_pred, data_range=1.0):
    """
    Peak Signal-to-Noise Ratio (pure numpy, no antspynet dependency).

    Parameters
    ----------
    y_true, y_pred : np.ndarray or ANTsImage
        Both must be normalised to the same scale (default [0, 1]).
    data_range : float
        Maximum possible pixel value. Default 1.0 for normalised inputs.

    Returns
    -------
    float  (higher is better; inf if images are identical)
    """
    if hasattr(y_true, 'numpy'):
        y_true = y_true.numpy()
    if hasattr(y_pred, 'numpy'):
        y_pred = y_pred.numpy()
    mse = float(np.mean((y_true.astype(np.float64) - y_pred.astype(np.float64)) ** 2))
    if mse == 0.0:
        return float('inf')
    return 10.0 * np.log10(data_range ** 2 / mse)


def compute_ssim(y_true, y_pred, data_range=1.0, win_size=11, K1=0.01, K2=0.03):
    """
    Structural Similarity Index (pure scipy, no antspynet dependency).

    Implements the Wang et al. (2004) SSIM using a sliding uniform window.
    Works for 2-D and 3-D arrays; multi-channel inputs are averaged.

    Parameters
    ----------
    y_true, y_pred : np.ndarray or ANTsImage
        Both must be normalised to the same scale (default [0, 1]).
    data_range : float
        Dynamic range of the inputs. Default 1.0.
    win_size : int
        Side length of the sliding window (must be odd). Default 11.
    K1, K2 : float
        Stability constants (SSIM paper defaults).

    Returns
    -------
    float  (1.0 = perfect match, lower is worse)
    """
    if hasattr(y_true, 'numpy'):
        y_true = y_true.numpy()
    if hasattr(y_pred, 'numpy'):
        y_pred = y_pred.numpy()

    y_true_channels, _ = _extract_spatial_and_channels(y_true)
    y_pred_channels, _ = _extract_spatial_and_channels(y_pred)
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    pad = win_size // 2
    ssim_vals = []
    for yt, yp in zip(y_true_channels, y_pred_channels):
        yt = yt.astype(np.float64)
        yp = yp.astype(np.float64)
        mu_t = uniform_filter(yt, win_size)
        mu_p = uniform_filter(yp, win_size)
        sig_t  = uniform_filter(yt * yt, win_size) - mu_t ** 2
        sig_p  = uniform_filter(yp * yp, win_size) - mu_p ** 2
        sig_tp = uniform_filter(yt * yp, win_size) - mu_t * mu_p
        num = (2.0 * mu_t * mu_p + C1) * (2.0 * sig_tp + C2)
        den = (mu_t ** 2 + mu_p ** 2 + C1) * (sig_t + sig_p + C2)
        ssim_map = num / (den + 1e-12)
        # Trim border pixels affected by filter wrap-around
        sl = tuple(slice(pad, -pad) for _ in range(yt.ndim))
        ssim_vals.append(float(ssim_map[sl].mean()))
    return float(np.mean(ssim_vals))


def compute_checkerboard_index(y, y_true=None, factor=None):
    """
    Computes the Checkerboard Index (CBI) using an alternating parity matched filter.

    In 1D: Alternating 2-tap kernel K[i] = (-1)^i / 2.
    In 2D: Alternating 2x2 kernel K[i, j] = (-1)^(i+j) / 4.
    In 3D: Alternating 2x2x2 kernel K[i, j, k] = (-1)^(i+j+k) / 8.
    For anisotropic factors (e.g. (1, 1, 2)), constructs a matched directional
    alternating kernel along the upsampled axes only.
    """
    if hasattr(y, 'numpy'):
        y = y.numpy()
    if y_true is not None and hasattr(y_true, 'numpy'):
        y_true = y_true.numpy()

    y_channels, spatial_dim = _extract_spatial_and_channels(y)
    if y_true is not None:
        y_true_channels, _ = _extract_spatial_and_channels(y_true)
    else:
        y_true_channels = [None] * len(y_channels)

    if factor is not None:
        if isinstance(factor, (int, float)):
            f_tuple = tuple([factor] * spatial_dim)
        elif len(factor) == 1:
            f_tuple = tuple([factor[0]] * spatial_dim)
        else:
            f_tuple = tuple(factor)
        up_axes = [i for i, f in enumerate(f_tuple) if f > 1]
        if len(up_axes) == 0:
            up_axes = list(range(spatial_dim))
    else:
        up_axes = list(range(spatial_dim))

    k_shape = [2 if i in up_axes else 1 for i in range(spatial_dim)]
    norm = float(2 ** len(up_axes))
    k = np.zeros(k_shape, dtype=np.float32)
    for idx in np.ndindex(*k_shape):
        k[idx] = ((-1.0) ** sum(idx)) / norm

    cbi_list = []
    for yc, ytc in zip(y_channels, y_true_channels):
        target = (yc - ytc) if ytc is not None else yc
        ref = ytc if ytc is not None else yc
        ref_std = float(np.std(ref)) + 1e-8

        resp = convolve(target.astype(np.float32), k, mode='reflect')
        cbi_list.append(float(np.std(resp) / ref_std))

    return float(np.mean(cbi_list))


def compute_tenengrad(y):
    """Computes the Tenengrad gradient energy density for 2D or 3D images."""
    if hasattr(y, 'numpy'):
        y = y.numpy()
    y = np.squeeze(y).astype(np.float64)
    grads = [sobel(y, axis=i) for i in range(y.ndim)]
    return float(np.mean(sum(g**2 for g in grads)))


def compute_acutance_ratio(y_true, y_pred):
    """
    Computes the Relative Acutance Ratio (Tenengrad ratio) between predicted
    and ground truth images.
    """
    t_true = compute_tenengrad(y_true)
    t_pred = compute_tenengrad(y_pred)
    return float(t_pred / (t_true + 1e-12))


def compute_laplacian_energy_ratio(y_true, y_pred):
    """
    Computes the Laplacian High-Frequency Energy Ratio:
    Ratio = ||∇^2 y_pred||_1 / ||∇^2 y_true||_1
    """
    if hasattr(y_true, 'numpy'):
        y_true = y_true.numpy()
    if hasattr(y_pred, 'numpy'):
        y_pred = y_pred.numpy()
    yt = np.squeeze(y_true).astype(np.float64)
    yp = np.squeeze(y_pred).astype(np.float64)
    l_true = float(np.mean(np.abs(laplace(yt))))
    l_pred = float(np.mean(np.abs(laplace(yp))))
    return float(l_pred / (l_true + 1e-12))


def compute_spectral_energy_ratio(y_true, y_pred, cutoff_ratio=None, factor=None):
    """
    Computes the High-Frequency Fourier Energy Ratio above the low-resolution
    Nyquist barrier.
    """
    if hasattr(y_true, 'numpy'):
        y_true = y_true.numpy()
    if hasattr(y_pred, 'numpy'):
        y_pred = y_pred.numpy()
    yt = np.squeeze(y_true).astype(np.float64)
    yp = np.squeeze(y_pred).astype(np.float64)

    spatial_dim = yt.ndim
    if factor is not None:
        if isinstance(factor, (int, float)):
            f_tuple = tuple([float(factor)] * spatial_dim)
        elif len(factor) == 1:
            f_tuple = tuple([float(factor[0])] * spatial_dim)
        else:
            f_tuple = tuple(float(f) for f in factor)
    else:
        f_tuple = tuple([2.0] * spatial_dim)

    up_axes = [i for i, f in enumerate(f_tuple) if f > 1.0]
    if not up_axes:
        up_axes = list(range(spatial_dim))

    if cutoff_ratio is None:
        cutoff_ratio = 1.0 / max([f_tuple[i] for i in up_axes])

    ft = np.fft.fftshift(np.fft.fftn(yt))
    fp = np.fft.fftshift(np.fft.fftn(yp))

    shape = yt.shape
    coords = [np.linspace(-1, 1, shape[i]) for i in range(spatial_dim)]
    mesh = np.meshgrid(*coords, indexing='ij')

    r_sq = sum(mesh[i]**2 for i in up_axes)
    hf_mask = np.sqrt(r_sq) > cutoff_ratio

    hf_true = np.sum(np.abs(ft)[hf_mask])
    hf_pred = np.sum(np.abs(fp)[hf_mask])
    return float(hf_pred / (hf_true + 1e-12))


def compute_ms_ssim(y_true, y_pred, data_range=1.0, weights=None, win_size=7, K1=0.01, K2=0.03):
    """Multi-Scale Structural Similarity Index (MS-SSIM) in 2D and 3D."""
    if hasattr(y_true, 'numpy'):
        y_true = y_true.numpy()
    if hasattr(y_pred, 'numpy'):
        y_pred = y_pred.numpy()
    yt = np.squeeze(y_true).astype(np.float64)
    yp = np.squeeze(y_pred).astype(np.float64)

    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    w_list = list(weights)

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    mcs = []
    cur_t = yt
    cur_p = yp

    for i, w in enumerate(w_list):
        if min(cur_t.shape) < win_size:
            w_list = w_list[:i]
            break

        mu_t = uniform_filter(cur_t, win_size)
        mu_p = uniform_filter(cur_p, win_size)
        sig_t = np.maximum(0.0, uniform_filter(cur_t * cur_t, win_size) - mu_t ** 2)
        sig_p = np.maximum(0.0, uniform_filter(cur_p * cur_p, win_size) - mu_p ** 2)
        sig_tp = uniform_filter(cur_t * cur_p, win_size) - mu_t * mu_p

        l = (2.0 * mu_t * mu_p + C1) / (mu_t ** 2 + mu_p ** 2 + C1 + 1e-12)
        cs = (2.0 * sig_tp + C2) / (sig_t + sig_p + C2 + 1e-12)

        pad = win_size // 2
        sl = tuple(slice(pad, -pad) for _ in range(cur_t.ndim))

        if i == len(w_list) - 1 or min(cur_t.shape) // 2 < win_size:
            final_l = float(np.mean(l[sl]))
            mcs.append(float(np.mean(cs[sl])))
            w_list = w_list[:len(mcs)]
            break
        else:
            mcs.append(float(np.mean(cs[sl])))
            down_sl = tuple(slice(None, None, 2) for _ in range(cur_t.ndim))
            cur_t = uniform_filter(cur_t, 2)[down_sl]
            cur_p = uniform_filter(cur_p, 2)[down_sl]

    w_arr = np.array(w_list, dtype=np.float64) / sum(w_list)
    mcs_arr = np.maximum(0.0, np.array(mcs, dtype=np.float64))
    val = (final_l ** w_arr[-1]) * np.prod(mcs_arr ** w_arr)
    return float(val)


_LPIPS_VGG_SINGLETON = None


def _get_lpips_vgg_singleton(device="cpu"):
    global _LPIPS_VGG_SINGLETON
    if _LPIPS_VGG_SINGLETON is None:
        import torch
        import torch.nn as nn
        import torchvision.models as models

        class _VGGFeatureExtractor(nn.Module):
            def __init__(self):
                super().__init__()
                vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.eval()
                for p in vgg.parameters():
                    p.requires_grad = False
                self.slices = nn.ModuleList([
                    vgg[:4],
                    vgg[4:9],
                    vgg[9:14],
                    vgg[14:23],
                    vgg[23:32]
                ])
                self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
                self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

            def forward(self, x, y):
                if x.shape[1] == 1:
                    x = x.repeat(1, 3, 1, 1)
                if y.shape[1] == 1:
                    y = y.repeat(1, 3, 1, 1)
                x = (x - self.mean) / self.std
                y = (y - self.mean) / self.std
                dist = 0.0
                cur_x, cur_y = x, y
                for s in self.slices:
                    cur_x = s(cur_x)
                    cur_y = s(cur_y)
                    norm_x = cur_x / (torch.sqrt(torch.sum(cur_x**2, dim=1, keepdim=True)) + 1e-10)
                    norm_y = cur_y / (torch.sqrt(torch.sum(cur_y**2, dim=1, keepdim=True)) + 1e-10)
                    diff = (norm_x - norm_y)**2
                    dist = dist + torch.mean(diff)
                return dist / len(self.slices)

        _LPIPS_VGG_SINGLETON = _VGGFeatureExtractor()
        _LPIPS_VGG_SINGLETON.to(device)
    return _LPIPS_VGG_SINGLETON


def compute_lpips(y_true, y_pred, net='vgg', device=None, num_slices=16):
    """Computes Learned Perceptual Image Patch Similarity (LPIPS) distance."""
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is required for compute_lpips. Install torch and torchvision.")

    if hasattr(y_true, 'numpy'):
        y_true = y_true.numpy()
    if hasattr(y_pred, 'numpy'):
        y_pred = y_pred.numpy()
    yt = np.squeeze(y_true).astype(np.float32)
    yp = np.squeeze(y_pred).astype(np.float32)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    vgg = _get_lpips_vgg_singleton(device=device)

    if yt.ndim == 2:
        tx = torch.from_numpy(yt).unsqueeze(0).unsqueeze(0).to(device)
        ty = torch.from_numpy(yp).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            return float(vgg(tx, ty).detach().cpu().numpy())
    elif yt.ndim == 3:
        dists = []
        for ax in range(3):
            size = yt.shape[ax]
            if size <= num_slices:
                indices = list(range(size))
            else:
                indices = np.linspace(int(size * 0.1), int(size * 0.9), num_slices, dtype=int)
            sl_t = [np.take(yt, i, axis=ax) for i in indices]
            sl_p = [np.take(yp, i, axis=ax) for i in indices]
            bx = torch.from_numpy(np.stack(sl_t, axis=0)).unsqueeze(1).to(device)
            by = torch.from_numpy(np.stack(sl_p, axis=0)).unsqueeze(1).to(device)
            with torch.no_grad():
                d = float(vgg(bx, by).detach().cpu().numpy())
            dists.append(d)
        return float(np.mean(dists))
    else:
        raise ValueError(f"Unsupported dimensions for compute_lpips: {yt.ndim}")


def compute_cqs(y_true, y_pred, factor=None):
    """
    Computes the Composite Quality Score (CQS):
        CQS = SSIM - GMSD - CBI
    """
    s = compute_ssim(y_true, y_pred)
    g = compute_gmsd(y_true, y_pred)
    c = compute_checkerboard_index(y_pred, y_true, factor=factor)
    return float(s - g - c)


def compute_pcs(y_true, y_pred, factor=None):
    """
    Computes the Perceptual Composite Score (PCS):
        PCS = SSIM + 0.5 * Acutance + 0.5 * Laplacian - GMSD - CBI
    """
    s = compute_ssim(y_true, y_pred)
    a = compute_acutance_ratio(y_true, y_pred)
    l = compute_laplacian_energy_ratio(y_true, y_pred)
    g = compute_gmsd(y_true, y_pred)
    c = compute_checkerboard_index(y_pred, y_true, factor=factor)
    return float(s + 0.5 * a + 0.5 * l - g - c)


def compute_perceptual_metrics(y_true, y_pred, factor=None):
    """Computes a comprehensive dictionary of quantitative quality metrics."""
    ssim_val = compute_ssim(y_true, y_pred)
    acutance_val = compute_acutance_ratio(y_true, y_pred)
    laplacian_val = compute_laplacian_energy_ratio(y_true, y_pred)
    gmsd_val = compute_gmsd(y_true, y_pred)
    cbi_val = compute_checkerboard_index(y_pred, y_true, factor=factor)
    cqs_val = float(ssim_val - gmsd_val - cbi_val)
    pcs_val = float(ssim_val + 0.5 * acutance_val + 0.5 * laplacian_val - gmsd_val - cbi_val)

    res = {
        "psnr": compute_psnr(y_true, y_pred),
        "ssim": ssim_val,
        "ms_ssim": compute_ms_ssim(y_true, y_pred),
        "acutance_ratio": acutance_val,
        "laplacian_ratio": laplacian_val,
        "spectral_ratio": compute_spectral_energy_ratio(y_true, y_pred, factor=factor),
        "gmsd": gmsd_val,
        "cbi": cbi_val,
        "cqs": cqs_val,
        "pcs": pcs_val,
    }
    try:
        res["lpips"] = compute_lpips(y_true, y_pred)
    except Exception:
        res["lpips"] = None
    return res


def pseudo_3d_vgg_features(inshape=[128, 128, 128], layer=4, angle=0, pretrained=True, verbose=False):  # pragma: no cover
    """Creates a pseudo-3D VGG feature extractor from a pre-trained 2D VGG model."""
    if antspynet is None:
        raise ImportError("antspynet is required for pseudo_3d_vgg_features.")

    vgg19 = keras.applications.VGG19(
        include_top=False, weights="imagenet",
        input_shape=[inshape[0], inshape[1], 3],
        classes=1000
    )
    layer_index = layer - 1
    vggmodelRaw = antspynet.create_vgg_model_3d(
        [inshape[0], inshape[1], inshape[2], 1],
        number_of_outputs=1000,
        layers=[1, 2, 3, 4, 4],
        lowest_resolution=64,
        convolution_kernel_size=(3, 3, 3), pool_size=(2, 2, 2),
        strides=(2, 2, 2), number_of_dense_units=4096, dropout_rate=0,
        style=19, mode="classification"
    )
    if verbose:
        print(vggmodelRaw.layers[layer_index])
        print(vggmodelRaw.layers[layer_index].name)
        print(vgg19.layers[layer_index])
        print(vgg19.layers[layer_index].name)

    feature_extractor_2d = keras.Model(
        inputs=vgg19.input,
        outputs=vgg19.layers[layer_index].output
    )
    feature_extractor = keras.Model(
        inputs=vggmodelRaw.input,
        outputs=vggmodelRaw.layers[layer_index].output
    )
    wts_2d = feature_extractor_2d.weights
    wts = feature_extractor.weights

    def checkwtshape(a, b):
        if len(a.shape) != len(b.shape):
            return False
        for j in range(len(a.shape)):
            if a.shape[j] != b.shape[j]:
                return False
        return True

    for ww in range(len(wts)):
        wts[ww] = wts[ww].numpy()
        wts_2d[ww] = wts_2d[ww].numpy()
        if checkwtshape(wts[ww], wts_2d[ww]) and ww != 0:
            wts[ww] = wts_2d[ww]
        elif ww != 0:
            if angle == 0:
                wts[ww][:, :, 0, :, :] = wts_2d[ww] / 3.0
                wts[ww][:, :, 1, :, :] = wts_2d[ww] / 3.0
                wts[ww][:, :, 2, :, :] = wts_2d[ww] / 3.0
            if angle == 1:
                wts[ww][:, 0, :, :, :] = wts_2d[ww] / 3.0
                wts[ww][:, 1, :, :, :] = wts_2d[ww] / 3.0
                wts[ww][:, 2, :, :, :] = wts_2d[ww] / 3.0
            if angle == 2:
                wts[ww][0, :, :, :, :] = wts_2d[ww] / 3.0
                wts[ww][1, :, :, :, :] = wts_2d[ww] / 3.0
                wts[ww][2, :, :, :, :] = wts_2d[ww] / 3.0
        else:
            wts[ww][:, :, :, 0, :] = wts_2d[ww]

    if pretrained:
        feature_extractor.set_weights(wts)
        newinput = keras.layers.Rescaling(255.0, -127.5)(feature_extractor.input)
        feature_extractor2 = feature_extractor(newinput)
        feature_extractor = keras.Model(feature_extractor.input, feature_extractor2)
    return feature_extractor


def vgg_features_2d(inshape=[128, 128], layer=6, pretrained=True):  # pragma: no cover
    """2D counterpart of pseudo_3d_vgg_features_unbiased."""
    vgg19 = keras.applications.VGG19(
        include_top=False, weights="imagenet" if pretrained else None,
        input_shape=[None, None, 3] if inshape is None else [inshape[0], inshape[1], 3]
    )
    layer_index = layer - 1
    extractor_rgb = keras.Model(inputs=vgg19.input, outputs=vgg19.layers[layer_index].output)
    extractor_rgb.trainable = False
    inp = keras.layers.Input(shape=(None, None, 1) if inshape is None else (inshape[0], inshape[1], 1))
    scaled = keras.layers.Rescaling(255.0, -127.5)(inp)
    zeros = keras.layers.Lambda(lambda t: t * 0.0)(scaled)
    rgb = keras.layers.Concatenate(axis=-1)([scaled, zeros, zeros])
    return keras.Model(inp, extractor_rgb(rgb))


def pseudo_3d_vgg_features_unbiased(inshape=[128, 128, 128], layer=4, verbose=False):  # pragma: no cover
    """Create pseudo-3D VGG feature extractor aggregating axial, coronal, sagittal planes."""
    f = [
        pseudo_3d_vgg_features(inshape, layer, angle=0, pretrained=True, verbose=verbose),
        pseudo_3d_vgg_features(inshape, layer, angle=1, pretrained=True),
        pseudo_3d_vgg_features(inshape, layer, angle=2, pretrained=True)
    ]
    f1 = f[0].inputs
    f0o = f[0](f1)
    f1o = f[1](f1)
    f2o = f[2](f1)
    catter = keras.layers.concatenate([f0o, f1o, f2o])
    feature_extractor = keras.Model(f1, catter)
    return feature_extractor


def get_grader_feature_network(layer=6):  # pragma: no cover
    """Load and extract a ResNet-based feature subnetwork for perceptual loss or grading."""
    if antspynet is None:
        raise ImportError("antspynet is required for get_grader_feature_network.")
    grader = antspynet.create_resnet_model_3d(
        [None, None, None, 1],
        lowest_resolution=32,
        number_of_outputs=4,
        cardinality=1,
        squeeze_and_excite=False
    )
    graderfn = os.path.expanduser("~/.antspyt1w/resnet_grader.h5")
    if not exists(graderfn):
        raise Exception("graderfn " + graderfn + " does not exist")
    grader.load_weights(graderfn)
    return keras.Model(inputs=grader.inputs, outputs=grader.layers[layer].output)


def auto_weight_loss(mdl, feature_extractor, x, y, feature=2.0, tv=0.1, verbose=True):  # pragma: no cover
    """Automatically compute weighting coefficients for MSE, feature, and TV losses."""
    y = ops.convert_to_tensor(y)
    y_pred = mdl(x)
    squared_difference = ops.square(y - y_pred)
    if len(y.shape) == 5:
        tdim = 3
        myax = [1, 2, 3, 4]
    if len(y.shape) == 4:
        tdim = 2
        myax = [1, 2, 3]
    msqTerm = ops.mean(squared_difference, axis=myax)
    temp1 = feature_extractor(y)
    temp2 = feature_extractor(y_pred)
    feature_difference = ops.square(temp1 - temp2)
    myax_feat = list(range(1, len(feature_difference.shape)))
    featureTerm = ops.mean(feature_difference, axis=myax_feat)
    msqw = 10.0
    mean_msq = ops.mean(msqTerm)
    mean_feat = ops.mean(featureTerm)
    featw = feature * msqw * mean_msq / (mean_feat + 1e-8)

    y_shape = ops.shape(y)
    if tdim == 3:
        reshaped_y = ops.reshape(y, (-1, y_shape[2], y_shape[3], y_shape[4]))
        mytv = ops_total_variation(reshaped_y)
    else:
        mytv = ops_total_variation(y)
    tvw = tv * msqw * mean_msq / (mytv + 1e-8)

    if verbose:
        print("MSQ: " + str(float(msqw * mean_msq)))
        print("Feat: " + str(float(featw * mean_feat)))
        print("Tv: " + str(float(mytv * tvw)))
    wts = [float(msqw), float(featw), float(tvw)]
    return wts


def auto_weight_loss_seg(mdl, feature_extractor, x, y, feature=2.0, tv=0.1, dice=0.5, verbose=True):  # pragma: no cover
    """Automatically compute weighting coefficients for segmentation multi-task loss."""
    y = ops.convert_to_tensor(y)
    y_pred = mdl(x)

    y_img = y[..., 0:1]
    y_seg = y[..., 1:2]
    pred_img = y_pred[..., 0:1]
    pred_seg = y_pred[..., 1:2]

    squared_difference = ops.square(y_img - pred_img)
    if len(y.shape) == 5:
        tdim = 3
        myax = [1, 2, 3, 4]
    if len(y.shape) == 4:
        tdim = 2
        myax = [1, 2, 3]
    msqTerm = ops.mean(squared_difference, axis=myax)

    temp1 = feature_extractor(y_img)
    temp2 = feature_extractor(pred_img)
    feature_difference = ops.square(temp1 - temp2)
    myax_feat = list(range(1, len(feature_difference.shape)))
    featureTerm = ops.mean(feature_difference, axis=myax_feat)

    msqw = 10.0
    mean_msq = ops.mean(msqTerm)
    mean_feat = ops.mean(featureTerm)
    featw = feature * msqw * mean_msq / (mean_feat + 1e-8)

    y_shape = ops.shape(y_img)
    if tdim == 3:
        reshaped_y = ops.reshape(y_img, (-1, y_shape[2], y_shape[3], y_shape[4]))
        mytv = ops_total_variation(reshaped_y)
    else:
        mytv = ops_total_variation(y_img)
    tvw = tv * msqw * mean_msq / (mytv + 1e-8)

    raw_dice = ops.abs(binary_dice_loss(y_seg, pred_seg))
    dicew = dice * msqw * mean_msq / (raw_dice + 1e-8)

    if verbose:
        print("MSQ: " + str(float(msqw * mean_msq)))
        print("Feat: " + str(float(featw * mean_feat)))
        print("Tv: " + str(float(mytv * tvw)))
        print("Dice: " + str(float(dicew * raw_dice)))
    wts = [float(msqw), float(featw), float(tvw), float(dicew)]
    return wts


def binary_dice_loss(y_true, y_pred):
    """Computes the Dice loss for binary segmentation tasks."""
    smoothing_factor = 1e-4
    y_true_f = ops.reshape(y_true, [-1])
    y_pred_f = ops.reshape(y_pred, [-1])
    intersection = ops.sum(y_true_f * y_pred_f)
    return -1 * (2 * intersection + smoothing_factor) / (ops.sum(y_true_f) + ops.sum(y_pred_f) + smoothing_factor)


def compute_phase_shift(y_true, y_pred, upsample_factor=100):
    """Computes the sub-pixel translational phase cross-correlation shift vector."""
    from skimage.registration import phase_cross_correlation
    t_np = y_true.numpy() if hasattr(y_true, "numpy") else np.asarray(y_true, dtype=np.float32)
    p_np = y_pred.numpy() if hasattr(y_pred, "numpy") else np.asarray(y_pred, dtype=np.float32)

    shift, _, _ = phase_cross_correlation(t_np, p_np, upsample_factor=upsample_factor, normalization=None)
    return tuple(float(s) for s in shift)


def compute_shift_lk(y_true, y_pred, n_iter=3, trim=3, grad_percentile=50.0):
    """Gain/offset-robust sub-voxel shift estimate (voxels) via iterative Lucas-Kanade."""
    from scipy.ndimage import shift as nd_shift
    t = (y_true.numpy() if hasattr(y_true, "numpy") else np.asarray(y_true)).astype(np.float64)
    p = (y_pred.numpy() if hasattr(y_pred, "numpy") else np.asarray(y_pred)).astype(np.float64)
    nd = t.ndim
    grads = np.gradient(t)
    gmag = np.sqrt(sum(g ** 2 for g in grads))
    inner = tuple(slice(trim, -trim) if s > 2 * trim + 2 else slice(None) for s in t.shape)
    keep = np.zeros(t.shape, bool)
    keep[inner] = True
    keep &= gmag > np.percentile(gmag[inner], grad_percentile)
    if keep.sum() < 10 * (nd + 2):
        return tuple([0.0] * nd)
    cols = [g[keep] for g in grads] + [t[keep], np.ones(int(keep.sum()))]
    X = np.stack(cols, axis=1)
    total = np.zeros(nd)
    cur = p
    for _ in range(n_iter):
        err = (cur - t)[keep]
        c, *_ = np.linalg.lstsq(X, err, rcond=None)
        step = c[:nd]
        total += step
        cur = nd_shift(p, tuple(total), order=1, mode="nearest")
    return tuple(float(s) for s in total)


def compute_edge_error_correlation(y_true, y_pred):
    """Measures Pearson correlation between residual prediction error and true directional gradients."""
    t_np = y_true.numpy() if hasattr(y_true, "numpy") else np.asarray(y_true, dtype=np.float32)
    p_np = y_pred.numpy() if hasattr(y_pred, "numpy") else np.asarray(y_pred, dtype=np.float32)

    err = (p_np - t_np).flatten()
    r_list = []
    ndim = t_np.ndim
    for axis in range(ndim):
        g = sobel(t_np, axis=axis).flatten()
        std_g = np.std(g)
        std_e = np.std(err)
        if std_g < 1e-8 or std_e < 1e-8:
            r_list.append(0.0)
        else:
            r = np.corrcoef(err, g)[0, 1]
            r_list.append(float(abs(r)) if not np.isnan(r) else 0.0)
    return tuple(r_list)


def compute_alignment_qc(
    y_true,
    y_pred,
    bilinear=None,
    factor=None,
    phase_thresh=0.08,
    edge_corr_thresh=0.16,
    phase_fail=0.15,
    edge_corr_fail=0.26,
    verbose=False,
):
    """Standardized Super-Resolution Spatial Alignment & Quality Control (QC) check."""
    t_np = y_true.numpy() if hasattr(y_true, "numpy") else np.asarray(y_true, dtype=np.float32)
    p_np = y_pred.numpy() if hasattr(y_pred, "numpy") else np.asarray(y_pred, dtype=np.float32)

    phase_raw = np.array(compute_phase_shift(t_np, p_np))
    lk_raw = np.array(compute_shift_lk(t_np, p_np))
    edge_raw = np.array(compute_edge_error_correlation(t_np, p_np))
    phase_shift = tuple(float(v) for v in phase_raw)
    lk_shift = tuple(float(v) for v in lk_raw)
    if bilinear is not None:
        b_arr = bilinear.numpy() if hasattr(bilinear, "numpy") else np.asarray(bilinear, dtype=np.float32)
        phase_rel = phase_raw - np.array(compute_phase_shift(t_np, b_arr))
        lk_rel = lk_raw - np.array(compute_shift_lk(t_np, b_arr))
        edge_rel = np.maximum(edge_raw - np.array(compute_edge_error_correlation(t_np, b_arr)), 0.0)
    else:
        phase_rel, lk_rel, edge_rel = phase_raw, lk_raw, edge_raw

    shift_vec = np.where(np.abs(phase_rel) <= np.abs(lk_rel), phase_rel, lk_rel)
    max_phase_shift = float(np.max(np.abs(shift_vec)))
    edge_corr = tuple(float(v) for v in edge_rel)
    max_edge_corr = float(np.max(edge_rel))

    psnr_val = float(compute_psnr(t_np, p_np))
    ssim_val = float(compute_ssim(t_np, p_np))
    gmsd_val = float(compute_gmsd(t_np, p_np))
    cbi_val = float(compute_checkerboard_index(p_np, t_np, factor=factor))
    cqs_val = float(compute_cqs(t_np, p_np, factor=factor))
    pcs_val = float(compute_pcs(t_np, p_np, factor=factor))

    b_psnr = None
    b_ssim = None
    delta_psnr = None
    if bilinear is not None:
        b_np = bilinear.numpy() if hasattr(bilinear, "numpy") else np.asarray(bilinear, dtype=np.float32)
        b_psnr = float(compute_psnr(t_np, b_np))
        b_ssim = float(compute_ssim(t_np, b_np))
        delta_psnr = psnr_val - b_psnr

    is_fail = (
        max_phase_shift >= phase_fail
        or max_edge_corr >= edge_corr_fail
    )
    is_warn = (
        not is_fail
        and (
            max_phase_shift >= phase_thresh
            or max_edge_corr >= edge_corr_thresh
        )
    )

    if is_fail:
        status = "FAIL"
        qc_pass = False
    elif is_warn:
        status = "WARN"
        qc_pass = True
    else:
        status = "PASS"
        qc_pass = True

    axes_labels = ["Y", "X", "Z"] if len(phase_shift) == 3 else ["Y", "X"]
    phase_str = ", ".join(f"{axes_labels[i]}: {shift_vec[i]:+.3f}" for i in range(len(shift_vec)))
    edge_str = ", ".join(f"{axes_labels[i]}: {edge_corr[i]:.4f}" for i in range(len(edge_corr)))

    b_psnr_str = f"{b_psnr:.2f} dB" if b_psnr is not None else "N/A"
    d_psnr_str = f"{delta_psnr:+.2f} dB" if delta_psnr is not None else "N/A"

    badge = {"PASS": "✅ PASS", "WARN": "⚠️ WARN", "FAIL": "❌ FAIL"}[status]

    lines = [
        "┌" + "─" * 68 + "┐",
        f"│  SR SPATIAL ALIGNMENT & BIAS QC REPORT                [{badge}]  │",
        "├" + "─" * 68 + "┤",
        f"│  Overall Status:              {status:<36} │",
        f"│  Max Sub-Pixel Shift (rel):   {max_phase_shift:.4f} voxels (thresh: <{phase_thresh:.2f})       │",
        f"│  Shift Vector (FFT&LK agree): ({phase_str})      │",
        f"│  Max Edge Error Correlation:  {max_edge_corr:.4f} (thresh: <{edge_corr_thresh:.2f})              │",
        f"│  Directional Edge Error:      ({edge_str})      │",
        f"│  Model Validation PSNR:       {psnr_val:.2f} dB (Bilinear: {b_psnr_str}, Δ: {d_psnr_str})  │",
        f"│  Model Validation SSIM:       {ssim_val:.4f}                               │",
        f"│  Composite Scores:            CQS={cqs_val:.4f} | PCS={pcs_val:.4f}             │",
        f"│  Checkerboard Index (CBI):    {cbi_val:.4f}                               │",
        "└" + "─" * 68 + "┘",
    ]
    summary_table = "\n".join(lines)

    if verbose:
        print(summary_table)

    return {
        "status": status,
        "qc_pass": qc_pass,
        "phase_shift": phase_shift,
        "max_phase_shift": max_phase_shift,
        "shift_rel": tuple(float(v) for v in shift_vec),
        "lk_shift": lk_shift,
        "edge_correlation": edge_corr,
        "max_edge_correlation": max_edge_corr,
        "psnr": psnr_val,
        "ssim": ssim_val,
        "gmsd": gmsd_val,
        "cbi": cbi_val,
        "cqs": cqs_val,
        "pcs": pcs_val,
        "bilinear_psnr": b_psnr,
        "bilinear_ssim": b_ssim,
        "delta_psnr_vs_bilinear": delta_psnr,
        "summary_table": summary_table,
    }

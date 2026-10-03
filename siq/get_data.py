from pathlib import Path
from pathlib import PurePath
import os
import pandas as pd
import math
import os.path
from os import path
from os.path import exists
import pickle
import sys
import numpy as np
import random
import functools
from operator import mul
from scipy.sparse.linalg import svds
from scipy.stats import pearsonr
from scipy.ndimage import prewitt, gaussian_laplace
import re

import ants
import antspynet
import antspyt1w
import keras
from keras import Model
from keras import ops
from keras.layers import (Input, Add, Subtract,
                          PReLU, Concatenate,
                          UpSampling2D, UpSampling3D,
                          Conv2D, Conv2DTranspose,
                          Conv3D, Conv3DTranspose)



# define the DBPN network - this uses a model definition that is general to<br>
# both 2D and 3D. recommended parameters for different upsampling rates can<br>
# be found in the papers by Haris et al.  We make one significant change to<br>
# the original architecture by allowing standard interpolation for upsampling<br>
# instead of convolutional upsampling.  this is controlled by the interpolation<br>
# option.



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
    from scipy.ndimage import uniform_filter
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

    Parameters
    ----------
    y : np.ndarray or ANTsImage
        Super-resolved image/volume to evaluate.
    y_true : np.ndarray or ANTsImage, optional
        Ground truth reference. If provided, measures the residual checkerboard
        error: std((y - y_true) * K) / std(y_true). If None, measures no-reference
        high-frequency checkerboard energy: std(y * K) / std(y).
    factor : int, tuple, or list, optional
        Upsampling factor per axis. If provided, the alternating filter is applied
        only across dimensions where factor > 1.

    Returns
    -------
    float
        Checkerboard Index (lower is better, 0 is no checkerboard pattern).
    """
    from scipy.ndimage import convolve
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
    """
    Computes the Tenengrad gradient energy density for 2D or 3D images.

    Tenengrad(y) = (1/N) * sum_{x} ||\nabla y(x)||^2
    using Sobel operators along each spatial dimension.

    Parameters
    ----------
    y : np.ndarray or ANTsImage
        Input 2D or 3D image/volume.

    Returns
    -------
    float
        Average squared gradient magnitude (higher means sharper edges).
    """
    from scipy.ndimage import sobel
    if hasattr(y, 'numpy'):
        y = y.numpy()
    y = np.squeeze(y).astype(np.float64)
    grads = [sobel(y, axis=i) for i in range(y.ndim)]
    return float(np.mean(sum(g**2 for g in grads)))


def compute_acutance_ratio(y_true, y_pred):
    """
    Computes the Relative Acutance Ratio (Tenengrad ratio) between predicted
    and ground truth images.

    Acutance Ratio = Tenengrad(y_pred) / Tenengrad(y_true)

    Parameters
    ----------
    y_true, y_pred : np.ndarray or ANTsImage
        Ground truth and reconstructed images/volumes.

    Returns
    -------
    float
        Ratio of edge sharpness:
        - 1.0 indicates perfect preservation of ground truth edge acutance.
        - < 1.0 indicates blurring / low-pass smoothing (e.g. bilinear is ~0.79).
        - > 1.0 indicates edge sharpening or high-frequency overshoot.
    """
    t_true = compute_tenengrad(y_true)
    t_pred = compute_tenengrad(y_pred)
    return float(t_pred / (t_true + 1e-12))


def compute_laplacian_energy_ratio(y_true, y_pred):
    """
    Computes the Laplacian High-Frequency Energy Ratio:

    Ratio = ||\nabla^2 y_pred||_1 / ||\nabla^2 y_true||_1

    Parameters
    ----------
    y_true, y_pred : np.ndarray or ANTsImage
        Ground truth and reconstructed images/volumes.

    Returns
    -------
    float
        Ratio of Laplacian high-frequency edge energy.
        Bilinear typically loses >50% of detail (ratio ~0.44), whereas
        effective super-resolution recovers fine structure (ratio ~0.80 - 1.05).
    """
    from scipy.ndimage import laplace
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

    Ratio = sum_{r > cutoff} |F_pred(k)| / sum_{r > cutoff} |F_true(k)|
    where r in [0, 1] is the normalized radial frequency in the Fourier domain
    across upsampled axes. For 2x super-resolution, the LR Nyquist limit corresponds
    to cutoff_ratio = 0.5.

    Parameters
    ----------
    y_true, y_pred : np.ndarray or ANTsImage
        Ground truth and reconstructed 2D or 3D images.
    cutoff_ratio : float, optional
        Normalized radial frequency cutoff in [0, 1]. If None, automatically derived
        from factor (1 / max_upsample_factor).
    factor : int, tuple, or list, optional
        Upsampling factor per axis. If anisotropic (e.g. (1, 1, 2)), the frequency
        cutoff is evaluated across only the upsampled axes.

    Returns
    -------
    float
        Ratio of high-frequency Fourier amplitude spectrum.
        Values near 1.0 indicate realistic spectral content extrapolation.
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
    """
    Multi-Scale Structural Similarity Index (MS-SSIM) in 2D and 3D.

    Implements Wang et al. (2003) across dyadic downsampling scales:
    MS-SSIM = [l_M(x, y)]^alpha_M * prod_{j=1}^M [cs_j(x, y)]^beta_j

    Parameters
    ----------
    y_true, y_pred : np.ndarray or ANTsImage
        Ground truth and reconstructed images/volumes.
    data_range : float
        Dynamic range of the inputs. Default 1.0.
    weights : list of float, optional
        Relative weights per scale. Defaults to Wang et al. 5-scale weights:
        [0.0448, 0.2856, 0.3001, 0.2363, 0.1333].
    win_size : int
        Local window size for uniform filtering. Default 7.
    K1, K2 : float
        Stability constants. Defaults 0.01, 0.03.

    Returns
    -------
    float
        MS-SSIM score in [0, 1] (1.0 = identical).
    """
    from scipy.ndimage import uniform_filter
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
    """
    Computes Learned Perceptual Image Patch Similarity (LPIPS) distance.

    For 2D inputs: computes perceptual distance across normalized deep features.
    For 3D inputs: computes tri-planar perceptual distance by sampling orthogonal
    slices across Axial, Coronal, and Sagittal anatomical planes:
        LPIPS_3D = (1/3) * (LPIPS_axial + LPIPS_coronal + LPIPS_sagittal)

    Parameters
    ----------
    y_true, y_pred : np.ndarray or ANTsImage
        Ground truth and reconstructed images/volumes in [0, 1].
    net : str, optional
        Feature backbone ('vgg' by default).
    device : str or torch.device, optional
        PyTorch device ('cuda', 'mps', or 'cpu'). Defaults to best available.
    num_slices : int, optional
        Number of slices per anatomical plane to evaluate for 3D volumes. Default 16.

    Returns
    -------
    float
        Perceptual distance (lower is better; 0.0 = perceptual identity).
    """
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

    Simultaneously rewards structural fidelity (SSIM in [0, 1]), gradient edge
    sharpness (low GMSD), and artifact cleanliness (low CBI).

    Parameters
    ----------
    y_true : np.ndarray or ANTsImage
        Ground truth reference image/volume.
    y_pred : np.ndarray or ANTsImage
        Predicted / reconstructed super-resolution image/volume.
    factor : int, tuple, or list, optional
        Super-resolution upsampling factor (e.g. 2 or (1, 1, 2)).

    Returns
    -------
    float
        Composite Quality Score.
    """
    s = compute_ssim(y_true, y_pred)
    g = compute_gmsd(y_true, y_pred)
    c = compute_checkerboard_index(y_pred, y_true, factor=factor)
    return float(s - g - c)


def compute_pcs(y_true, y_pred, factor=None):
    """
    Computes the Perceptual Composite Score (PCS):
        PCS = SSIM + 0.5 * Acutance + 0.5 * Laplacian - GMSD - CBI

    Integrates structural preservation, 1st-order gradient edge sharpness (acutance),
    and 2nd-order high-frequency Laplacian detail while penalizing gradient dispersion
    and checkerboard ringing artifacts.

    Parameters
    ----------
    y_true : np.ndarray or ANTsImage
        Ground truth reference image/volume.
    y_pred : np.ndarray or ANTsImage
        Predicted / reconstructed super-resolution image/volume.
    factor : int, tuple, or list, optional
        Super-resolution upsampling factor (e.g. 2 or (1, 1, 2)).

    Returns
    -------
    float
        Perceptual Composite Score.
    """
    s = compute_ssim(y_true, y_pred)
    a = compute_acutance_ratio(y_true, y_pred)
    l = compute_laplacian_energy_ratio(y_true, y_pred)
    g = compute_gmsd(y_true, y_pred)
    c = compute_checkerboard_index(y_pred, y_true, factor=factor)
    return float(s + 0.5 * a + 0.5 * l - g - c)


def compute_perceptual_metrics(y_true, y_pred, factor=None):
    """
    Computes a comprehensive dictionary of quantitative quality and independent
    perceptual metrics for 2D or 3D images.

    Returns
    -------
    dict
        Dictionary containing:
        - psnr: Peak Signal-to-Noise Ratio (dB)
        - ssim: Structural Similarity Index
        - ms_ssim: Multi-Scale Structural Similarity
        - acutance_ratio: Tenengrad edge acutance relative to ground truth
        - laplacian_ratio: Laplacian high-frequency edge energy ratio
        - spectral_ratio: Fourier high-frequency energy ratio above LR Nyquist
        - lpips: Deep perceptual feature distance (lower is better)
        - gmsd: Gradient Magnitude Similarity Deviation
        - cbi: Checkerboard Index
        - cqs: Composite Quality Score (SSIM - GMSD - CBI)
        - pcs: Perceptual Composite Score (SSIM + 0.5*Acutance + 0.5*Laplacian - GMSD - CBI)
    """
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


def estimate_anti_checkerboard_sigma(vol, max_sigma=0.40):
    """
    Empirically calculates the optimal sub-voxel smoothing sigma to attenuate
    transposed-convolution checkerboard artifacts based on the measured Nyquist
    frequency spectral excess.

    Uses the 1D amplitude spectrum (|F|) rather than power (|F|^2) to avoid
    over-filtering and preserve fine anatomical edges (>95% edge gradient retention).

    Parameters
    ----------
    vol : np.ndarray or ANTsImage
        Input image/volume.
    max_sigma : float, optional
        Maximum allowed smoothing sigma in voxels. Default is 0.40.

    Returns
    -------
    float
        Calculated sigma in voxels (0.0 if no artifact is detected).
    """
    if hasattr(vol, 'numpy'):
        vol = vol.numpy()
    vol = np.asarray(vol, dtype=np.float32)
    sigmas = []
    ndim = vol.ndim
    for ax in range(ndim):
        fft_1d = np.fft.rfft(vol, axis=ax)
        axes_to_mean = tuple(i for i in range(ndim) if i != ax)
        mean_amp = np.mean(np.abs(fft_1d), axis=axes_to_mean)
        if len(mean_amp) < 4:
            continue
        a_nyquist = mean_amp[-1]
        idx_start = max(1, int(len(mean_amp) * 0.70))
        idx_end = max(idx_start + 1, int(len(mean_amp) * 0.95))
        a_baseline = np.mean(mean_amp[idx_start:idx_end]) + 1e-12
        excess_ratio = a_nyquist / a_baseline
        if excess_ratio > 1.2:
            sigma_ax = np.sqrt(np.log(excess_ratio)) / np.pi
            sigmas.append(float(sigma_ax))
        else:
            sigmas.append(0.0)

    optimal_sigma = float(np.mean(sigmas)) if sigmas else 0.0
    return min(max_sigma, optimal_sigma)


def dbpn(input_image_size,
                                                 number_of_outputs=1,
                                                 number_of_base_filters=64,
                                                 number_of_feature_filters=256,
                                                 number_of_back_projection_stages=7,
                                                 convolution_kernel_size=(12, 12),
                                                 strides=(8, 8),
                                                 last_convolution=(3, 3),
                                                 number_of_loss_functions=1,
                                                 interpolation = 'nearest'
                                                ):
    """
    Creates a Deep Back-Projection Network (DBPN) for single image super-resolution.

    This function constructs a Keras model based on the DBPN architecture, which
    can be configured for either 2D or 3D inputs. The network uses iterative
    up- and down-projection blocks to refine the high-resolution image estimate. A
    key modification from the original paper is the option to use standard
    interpolation for upsampling instead of deconvolution layers.

    Reference:
     - Haris, M., Shakhnarovich, G., & Ukita, N. (2018). Deep Back-Projection
       Networks For Super-Resolution. In CVPR.

    Parameters
    ----------
    input_image_size : tuple or list
        The shape of the input image, including the channel.
        e.g., `(None, None, 1)` for 2D or `(None, None, None, 1)` for 3D.

    number_of_outputs : int, optional
        The number of channels in the output image. Default is 1.

    number_of_base_filters : int, optional
        The number of filters in the up/down projection blocks. Default is 64.

    number_of_feature_filters : int, optional
        The number of filters in the initial feature extraction layer. Default is 256.

    number_of_back_projection_stages : int, optional
        The number of iterative back-projection stages (T in the paper). Default is 7.

    convolution_kernel_size : tuple or list, optional
        The kernel size for the main projection convolutions. Should match the
        dimensionality of the input. Default is (12, 12).

    strides : tuple or list, optional
        The strides for the up/down sampling operations, defining the
        super-resolution factor. Default is (8, 8).

    last_convolution : tuple or list, optional
        The kernel size of the final reconstruction convolution. Default is (3, 3).

    number_of_loss_functions : int, optional
        If greater than 1, the model will have multiple identical output branches.
        Typically set to 1. Default is 1.

    interpolation : str, optional
        The interpolation method to use for upsampling layers if not using
        transposed convolutions. 'nearest' or 'bilinear'. Default is 'nearest'.

    Returns
    -------
    keras.Model
        A Keras model implementing the DBPN architecture for the specified
        parameters.
    """
    idim = len( input_image_size ) - 1
    if idim == 2:
        myconv = Conv2D
        myconv_transpose = Conv2DTranspose
        myupsampling = UpSampling2D
        shax = ( 1, 2 )
        firstConv = (3,3)
        firstStrides=(1,1)
        smashConv=(1,1)
    if idim == 3:
        myconv = Conv3D
        myconv_transpose = Conv3DTranspose
        myupsampling = UpSampling3D
        shax = ( 1, 2, 3 )
        firstConv = (3,3,3)
        firstStrides=(1,1,1)
        smashConv=(1,1,1)
    def up_block_2d(L, number_of_filters=64, kernel_size=(12, 12), strides=(8, 8),
                    include_dense_convolution_layer=True):
        if include_dense_convolution_layer == True:
            L = myconv(filters = number_of_filters,
                       use_bias=True,
                       kernel_size=smashConv,
                       strides=firstStrides,
                       padding='same')(L)
            L = PReLU(alpha_initializer='zero',
                      shared_axes=shax)(L)
        # Scale up
        if idim == 2:
            H0 = myupsampling( size = strides, interpolation=interpolation )(L)
        if idim == 3:
            H0 = myupsampling( size = strides )(L)
        H0 = myconv(filters=number_of_filters,
                    kernel_size=firstConv,
                    strides=firstStrides,
                    use_bias=True,
                    padding='same')(H0)
        H0 = PReLU(alpha_initializer='zero',
                   shared_axes=shax)(H0)
        # Scale down
        L0 = myconv(filters=number_of_filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    kernel_initializer='glorot_uniform',
                    padding='same')(H0)
        L0 = PReLU(alpha_initializer='zero',
                   shared_axes=shax)(L0)
        # Residual
        E = Subtract()([L0, L])
        # Scale residual up
        if idim == 2:
            H1 = myupsampling( size = strides, interpolation=interpolation  )(E)
        if idim == 3:
            H1 = myupsampling( size = strides )(E)
        H1 = myconv(filters=number_of_filters,
                    kernel_size=firstConv,
                    strides=firstStrides,
                    use_bias=True,
                    padding='same')(H1)
        H1 = PReLU(alpha_initializer='zero',
                   shared_axes=shax)(H1)
        # Output feature map
        up_block = Add()([H0, H1])
        return up_block
    def down_block_2d(H, number_of_filters=64, kernel_size=(12, 12), strides=(8, 8),
                    include_dense_convolution_layer=True):
        if include_dense_convolution_layer == True:
            H = myconv(filters = number_of_filters,
                       use_bias=True,
                       kernel_size=smashConv,
                       strides=firstStrides,
                       padding='same')(H)
            H = PReLU(alpha_initializer='zero',
                      shared_axes=shax)(H)
        # Scale down
        L0 = myconv(filters=number_of_filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    kernel_initializer='glorot_uniform',
                    padding='same')(H)
        L0 = PReLU(alpha_initializer='zero',
                   shared_axes=shax)(L0)
        # Scale up
        if idim == 2:
            H0 = myupsampling( size = strides, interpolation=interpolation )(L0)
        if idim == 3:
            H0 = myupsampling( size = strides )(L0)
        H0 = myconv(filters=number_of_filters,
                    kernel_size=firstConv,
                    strides=firstStrides,
                    use_bias=True,
                    padding='same')(H0)
        H0 = PReLU(alpha_initializer='zero',
                   shared_axes=shax)(H0)
        # Residual
        E = Subtract()([H0, H])
        # Scale residual down
        L1 = myconv(filters=number_of_filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    kernel_initializer='glorot_uniform',
                    padding='same')(E)
        L1 = PReLU(alpha_initializer='zero',
                   shared_axes=shax)(L1)
        # Output feature map
        down_block = Add()([L0, L1])
        return down_block
    inputs = Input(shape=input_image_size)
    # Initial feature extraction
    model = myconv(filters=number_of_feature_filters,
                   kernel_size=firstConv,
                   strides=firstStrides,
                   padding='same',
                   kernel_initializer='glorot_uniform')(inputs)
    model = PReLU(alpha_initializer='zero',
                  shared_axes=shax)(model)
    # Feature smashing
    model = myconv(filters=number_of_base_filters,
                   kernel_size=smashConv,
                   strides=firstStrides,
                   padding='same',
                   kernel_initializer='glorot_uniform')(model)
    model = PReLU(alpha_initializer='zero',
                  shared_axes=shax)(model)
    # Back projection
    up_projection_blocks = []
    down_projection_blocks = []
    model = up_block_2d(model, number_of_filters=number_of_base_filters,
      kernel_size=convolution_kernel_size, strides=strides)
    up_projection_blocks.append(model)
    for i in range(number_of_back_projection_stages):
        if i == 0:
            model = down_block_2d(model, number_of_filters=number_of_base_filters,
              kernel_size=convolution_kernel_size, strides=strides)
            down_projection_blocks.append(model)
            model = up_block_2d(model, number_of_filters=number_of_base_filters,
              kernel_size=convolution_kernel_size, strides=strides)
            up_projection_blocks.append(model)
            model = Concatenate()(up_projection_blocks)
        else:
            model = down_block_2d(model, number_of_filters=number_of_base_filters,
              kernel_size=convolution_kernel_size, strides=strides,
              include_dense_convolution_layer=True)
            down_projection_blocks.append(model)
            model = Concatenate()(down_projection_blocks)
            model = up_block_2d(model, number_of_filters=number_of_base_filters,
              kernel_size=convolution_kernel_size, strides=strides,
              include_dense_convolution_layer=True)
            up_projection_blocks.append(model)
            model = Concatenate()(up_projection_blocks)
    outputs = myconv(filters=number_of_outputs,
                     kernel_size=last_convolution,
                     strides=firstStrides,
                     padding = 'same',
                     kernel_initializer = "glorot_uniform")(model)
    if number_of_loss_functions == 1:
        deep_back_projection_network_model = Model(inputs=inputs, outputs=outputs)
    else:
        outputList=[]
        for k in range(number_of_loss_functions):
            outputList.append(outputs)
        deep_back_projection_network_model = Model(inputs=inputs, outputs=outputList)
    return deep_back_projection_network_model


# generate a random corner index for a patch

def get_random_base_ind( # pragma: no cover
 full_dims, patchWidth, off=8 ):
    """
    Generates a random top-left corner index for a patch.

    This utility function computes a valid starting index (e.g., [x, y, z])
    for extracting a patch from a larger volume, ensuring the patch fits entirely
    within the volume's boundaries, accounting for an offset.

    Parameters
    ----------
    full_dims : tuple or list
        The dimensions of the full volume (e.g., img.shape).

    patchWidth : tuple or list
        The dimensions of the patch to be extracted.

    off : int, optional
        An offset from the edge of the volume to avoid sampling near borders.
        Default is 8.

    Returns
    -------
    list
        A list of integers representing the starting coordinates for the patch.
    """
    baseInd = [None,None,None]
    for k in range(3):
        baseInd[k]=random.sample( range( off, full_dims[k]-1-patchWidth[k] ), 1 )[0]
    return baseInd


# extract a random patch
def get_random_patch( img, patchWidth ):
    """
    Extracts a random patch from an image with non-zero variance.

    This function repeatedly samples a random patch of a specified width from
    the input image until it finds one where the standard deviation of pixel
    intensities is greater than zero. This is useful for avoiding blank or
    uniform patches during training data generation.

    Parameters
    ----------
    img : ants.ANTsImage
        The source image from which to extract a patch.

    patchWidth : tuple or list
        The desired dimensions of the output patch.

    Returns
    -------
    ants.ANTsImage
        A randomly extracted patch from the input image.
    """
    mystd = 0
    while mystd == 0:
        inds = get_random_base_ind( full_dims = img.shape, patchWidth=patchWidth, off=8 )
        hinds = [None,None,None]
        for k in range(len(inds)):
            hinds[k] = inds[k] + patchWidth[k]
        myimg = ants.crop_indices( img, inds, hinds )
        mystd = myimg.std()
    return myimg

def get_random_patch_pair( # pragma: no cover
 img, img2, patchWidth ):
    """
    Extracts a corresponding random patch from a pair of images.

    This function finds a single random location and extracts a patch of the
    same size and position from two different input images. It ensures that
    both extracted patches have non-zero variance. This is useful for creating
    paired training data (e.g., low-res and high-res images).

    Parameters
    ----------
    img : ants.ANTsImage
        The first source image.

    img2 : ants.ANTsImage
        The second source image, spatially aligned with the first.

    patchWidth : tuple or list
        The desired dimensions of the output patches.

    Returns
    -------
    tuple of ants.ANTsImage
        A tuple containing two corresponding patches: (patch_from_img, patch_from_img2).
    """
    mystd = mystd2 = 0
    ct = 0
    while mystd == 0 or mystd2 == 0:
        inds = get_random_base_ind( full_dims = img.shape, patchWidth=patchWidth, off=8  )
        hinds = [None,None,None]
        for k in range(len(inds)):
            hinds[k] = inds[k] + patchWidth[k]
        myimg = ants.crop_indices( img, inds, hinds )
        myimg2 = ants.crop_indices( img2, inds, hinds )
        mystd = myimg.std()
        mystd2 = myimg2.std()
        ct = ct + 1
        if ( ct > 20 ):
            return myimg, myimg2
    return myimg, myimg2

def pseudo_3d_vgg_features( inshape = [128,128,128], layer = 4, angle=0, pretrained=True, verbose=False ): # pragma: no cover
    """
    Creates a pseudo-3D VGG feature extractor from a pre-trained 2D VGG model.

    This function constructs a 3D VGG-style network and initializes its weights
    by "stretching" the weights from a pre-trained 2D VGG19 model (trained on
    ImageNet) along a specified axis. This is a technique to transfer 2D
    perceptual knowledge to a 3D domain for tasks like perceptual loss.

    Parameters
    ----------
    inshape : list of int, optional
        The input shape of the 3D volume, e.g., `[128, 128, 128]`. Default is `[128,128,128]`.

    layer : int, optional
        The block number of the VGG network from which to extract features. For
        VGG19, this corresponds to block `layer` (e.g., layer=4 means 'block4_conv...').
        Default is 4.

    angle : int, optional
        The axis along which to project the 2D weights:
        - 0: Axial plane (stretches along Z)
        - 1: Coronal plane (stretches along Y)
        - 2: Sagittal plane (stretches along X)
        Default is 0.

    pretrained : bool, optional
        If True, loads the stretched ImageNet weights. If False, the model is
        randomly initialized. Default is True.

    verbose : bool, optional
        If True, prints information about the layers being used. Default is False.

    Returns
    -------
    tf.keras.Model
        A Keras model that takes a 3D volume as input and outputs the pseudo-3D
        feature map from the specified layer and angle.
    """
    def getLayerScaleFactorForTransferLearning( k, w3d, w2d ):
        myfact = np.round( np.prod( w3d[k].shape ) / np.prod(  w2d[k].shape) )
        return myfact
    vgg19 = keras.applications.VGG19(
            include_top = False, weights = "imagenet",
            input_shape = [inshape[0],inshape[1],3],
            classes = 1000 )
    def findLayerIndex( layerName, mdl ):
          for k in range( len( mdl.layers ) ):
            if layerName == mdl.layers[k].name :
                return k - 1
          return None
    layer_index = layer-1 # findLayerIndex( 'block2_conv2', vgg19 )
    vggmodelRaw = antspynet.create_vgg_model_3d(
            [inshape[0],inshape[1],inshape[2],1],
            number_of_outputs = 1000,
            layers = [1, 2, 3, 4, 4],
            lowest_resolution = 64,
            convolution_kernel_size= (3, 3, 3), pool_size = (2, 2, 2),
            strides = (2, 2, 2), number_of_dense_units= 4096, dropout_rate = 0,
            style = 19, mode = "classification")
    if verbose:
        print( vggmodelRaw.layers[layer_index] )
        print( vggmodelRaw.layers[layer_index].name )
        print( vgg19.layers[layer_index] )
        print( vgg19.layers[layer_index].name )
    feature_extractor_2d = keras.Model(
            inputs = vgg19.input,
            outputs = vgg19.layers[layer_index].output)
    feature_extractor = keras.Model(
            inputs = vggmodelRaw.input,
            outputs = vggmodelRaw.layers[layer_index].output)
    wts_2d = feature_extractor_2d.weights
    wts = feature_extractor.weights
    def checkwtshape( a, b ):
        if len(a.shape) != len(b.shape):
                return False
        for j in range(len(a.shape)):
            if a.shape[j] != b.shape[j]:
                return False
        return True
    for ww in range(len(wts)):
        wts[ww]=wts[ww].numpy()
        wts_2d[ww]=wts_2d[ww].numpy()
        if checkwtshape( wts[ww], wts_2d[ww] ) and ww != 0:
            wts[ww]=wts_2d[ww]
        elif ww != 0:
            # FIXME - should allow doing this across different angles
            if angle == 0:
                wts[ww][:,:,0,:,:]=wts_2d[ww]/3.0
                wts[ww][:,:,1,:,:]=wts_2d[ww]/3.0
                wts[ww][:,:,2,:,:]=wts_2d[ww]/3.0
            if angle == 1:
                wts[ww][:,0,:,:,:]=wts_2d[ww]/3.0
                wts[ww][:,1,:,:,:]=wts_2d[ww]/3.0
                wts[ww][:,2,:,:,:]=wts_2d[ww]/3.0
            if angle == 2:
                wts[ww][0,:,:,:,:]=wts_2d[ww]/3.0
                wts[ww][1,:,:,:,:]=wts_2d[ww]/3.0
                wts[ww][2,:,:,:,:]=wts_2d[ww]/3.0
        else:
            wts[ww][:,:,:,0,:]=wts_2d[ww]
    if pretrained:
        feature_extractor.set_weights( wts )
        newinput = keras.layers.Rescaling(  255.0, -127.5  )( feature_extractor.input )
        feature_extractor2 = feature_extractor( newinput )
        feature_extractor = keras.Model( feature_extractor.input, feature_extractor2 )
    return feature_extractor

def vgg_features_2d( inshape = [128,128], layer = 6, pretrained=True ): # pragma: no cover
    """
    2D counterpart of `pseudo_3d_vgg_features_unbiased` ("same style" in 2D).

    Uses the same pre-trained ImageNet VGG19 feature layer index (``layer-1``),
    the same input convention (single channel, ``Rescaling(255, -127.5)`` baked
    in, so inputs are expected in [0, 1]) and the same single-channel first
    layer weights as the pseudo-3D extractor (channel 0 of the RGB kernel).

    Parameters
    ----------
    inshape : list of int
        Spatial input shape ``[H, W]``.
    layer : int
        VGG19 layer index offset (default 6, canonical layer from Avants et al.).
    pretrained : bool
        Load ImageNet weights.

    Returns
    -------
    keras.Model
        Model mapping a ``(H, W, 1)`` image in [0, 1] to the VGG19 feature map.
    """
    vgg19 = keras.applications.VGG19(
        include_top=False, weights="imagenet" if pretrained else None,
        input_shape=[None, None, 3] if inshape is None else [inshape[0], inshape[1], 3])
    layer_index = layer - 1
    extractor_rgb = keras.Model(inputs=vgg19.input, outputs=vgg19.layers[layer_index].output)
    extractor_rgb.trainable = False
    inp = keras.layers.Input(shape=(None, None, 1) if inshape is None else (inshape[0], inshape[1], 1))
    scaled = keras.layers.Rescaling(255.0, -127.5)(inp)
    # channel 0 only (zeros elsewhere) == single-channel kernel used in 3D
    zeros = keras.layers.Lambda(lambda t: t * 0.0)(scaled)
    rgb = keras.layers.Concatenate(axis=-1)([scaled, zeros, zeros])
    return keras.Model(inp, extractor_rgb(rgb))

def pseudo_3d_vgg_features_unbiased( inshape = [128,128,128], layer = 4, verbose=False ): # pragma: no cover

    """
    Create a pseudo-3D VGG-style feature extractor by aggregating axial, coronal,
    and sagittal VGG feature representations.

    This model extracts features along each principal axis using pre-trained 2D
    VGG-style networks and concatenates them to form an unbiased pseudo-3D feature space.

    Parameters
    ----------
    inshape : list of int, optional
        The input shape of the 3D volume, default is [128, 128, 128].

    layer : int, optional
        The VGG feature layer to extract. Higher values correspond to deeper
        layers in the pseudo-3D VGG backbone.

    verbose : bool, optional
        If True, prints debug messages during model construction.

    Returns
    -------
    tf.keras.Model
        A TensorFlow Keras model that takes a 3D input volume and outputs the
        concatenated pseudo-3D feature representation from the specified layer.

    Notes
    -----
    This is useful for perceptual loss or feature comparison in super-resolution
    and image synthesis tasks. The same input is processed in three anatomical
    planes (axial, coronal, sagittal), and features are concatenated.

    See Also
    --------
    pseudo_3d_vgg_features : Generates VGG features from a single anatomical plane.
    """
    f = [
        pseudo_3d_vgg_features( inshape, layer, angle=0, pretrained=True, verbose=verbose ),
        pseudo_3d_vgg_features( inshape, layer, angle=1, pretrained=True ),
        pseudo_3d_vgg_features( inshape, layer, angle=2, pretrained=True ) ]
    f1=f[0].inputs
    f0o=f[0]( f1 )
    f1o=f[1]( f1 )
    f2o=f[2]( f1 )
    catter = keras.layers.concatenate( [f0o, f1o, f2o ])
    feature_extractor = keras.Model( f1, catter )
    return feature_extractor

def get_grader_feature_network( # pragma: no cover
 layer=6 ):
    """
    Load and extract a ResNet-based feature subnetwork for perceptual loss or quality grading.

    This function loads a pre-trained 3D ResNet model ("grader") used for
    perceptual feature extraction and returns a subnetwork that outputs activations
    from a specified internal layer.

    Parameters
    ----------
    layer : int, optional
        The index of the internal ResNet layer whose output should be used as
        the feature representation. Default is layer 6.

    Returns
    -------
    tf.keras.Model
        A Keras model that outputs features from the specified layer of the
        pre-trained 3D ResNet grader model.

    Raises
    ------
    Exception
        If the pre-trained weights file (`resnet_grader.h5`) is not found.

    Notes
    -----
    The pre-trained weights should be located in: `~/.antspyt1w/resnet_grader.keras`

    This model is typically used to compute perceptual loss by comparing
    intermediate activations between target and prediction volumes.

    See Also
    --------
    antspynet.create_resnet_model_3d : Constructs the base ResNet model.
    """
    grader = antspynet.create_resnet_model_3d(
        [None,None,None,1],
        lowest_resolution = 32,
        number_of_outputs = 4,
        cardinality = 1,
        squeeze_and_excite = False )
    # the folder and data below as available from antspyt1w get_data
    graderfn = os.path.expanduser( "~/.antspyt1w/resnet_grader.h5" )
    if not exists( graderfn ):
        raise Exception("graderfn " + graderfn + " does not exist")
    grader.load_weights( graderfn)
    #    feature_extractor_23 = tf.keras.Model( inputs=grader.inputs, outputs=grader.layers[23].output )
    #   feature_extractor_44 = tf.keras.Model( inputs=grader.inputs, outputs=grader.layers[44].output )
    return keras.Model( inputs=grader.inputs, outputs=grader.layers[layer].output )



def default_dbpn( # pragma: no cover

    strider, # length should equal dimensionality
    dimensionality = 3,
    nfilt=64,
    nff = 256,
    convn = 6,
    lastconv = 3,
    nbp=7,
    nChannelsIn=1,
    nChannelsOut=1,
    option = None,
    intensity_model=None,
    segmentation_model=None,
    sigmoid_second_channel=False,
    clone_intensity_to_segmentation=False,
    pro_seg = 0,
    freeze = False,
    verbose=False
 ):
    """
    Constructs a DBPN model based on input parameters, and can optionally
    use external models for intensity or segmentation processing.

    Args:
        strider (list): List of strides, length must match `dimensionality`.

        dimensionality (int): Number of dimensions (2 or 3). Default is 3.

        nfilt (int): Number of base filters. Default is 64.

        nff (int): Number of feature filters. Default is 256.

        convn (int): Convolution kernel size. Default is 6.

        lastconv (int): Size of the last convolution. Default is 3.
        
        nbp (int): Number of back projection stages. Default is 7.

        nChannelsIn (int): Number of input channels. Default is 1.

        nChannelsOut (int): Number of output channels. Default is 1.

        option (str): Model size option ('tiny', 'small', 'medium', 'large'). Default is None.
        intensity_model (tf.keras.Model): Optional external intensity model.

        segmentation_model (tf.keras.Model): Optional external segmentation model.

        sigmoid_second_channel (bool): If True, applies sigmoid to second channel in output.
        clone_intensity_to_segmentation (bool): If True, clones intensity model weights to segmentation.

        pro_seg (int): If greater than 0, adds a segmentation arm.

        freeze (bool): If True, freezes the layers of the intensity/segmentation models.

        verbose (bool): If True, prints detailed logs.

    Returns:

        Model: A Keras model based on the specified configuration.

    Raises:

        Exception: If `len(strider)` is not equal to `dimensionality`.
    """
    if option == 'tiny':
        nfilt=32
        nff = 64
        convn = 3
        lastconv = 1
        nbp=2
    elif option == 'small':
        nfilt=32
        nff = 64
        convn = 6
        lastconv = 3
        nbp=4
    elif option == 'medium':
        nfilt=64
        nff = 128
        convn = 6
        lastconv = 3
        nbp=4
    else:
        option='large'
    if verbose:
        print("Building mode of size: " + option)
        if intensity_model is not None:
            print("user-passed intensity model will be frozen - only segmentation will train")
        if segmentation_model is not None:
            print("user-passed segmentation model will be frozen - only intensity will train")

    if len(strider) != dimensionality:
        raise Exception("len(strider) != dimensionality")
    # **model instantiation**: these are close to defaults for the 2x network.<br>
    # empirical evidence suggests that making covolutions and strides evenly<br>
    # divisible by each other reduces artifacts.  2*3=6.
    # ofn='./models/dsr3d_'+str(strider)+'up_' + str(nfilt) + '_' + str( nff ) + '_' + str(convn)+ '_' + str(lastconv)+ '_' + str(os.environ['CUDA_VISIBLE_DEVICES'])+'_v0.0.keras'
    if dimensionality == 2 :
        mdl = dbpn( (None,None,nChannelsIn),
            number_of_outputs=nChannelsOut,
            number_of_base_filters=nfilt,
            number_of_feature_filters=nff,
            number_of_back_projection_stages=nbp,
            convolution_kernel_size=(convn, convn),
            strides=(strider[0], strider[1]),
            last_convolution=(lastconv, lastconv),
            number_of_loss_functions=1,
            interpolation='nearest')
    if dimensionality == 3 :
        mdl = dbpn( (None,None,None,nChannelsIn),
            number_of_outputs=nChannelsOut,
            number_of_base_filters=nfilt,
            number_of_feature_filters=nff,
            number_of_back_projection_stages=nbp,
            convolution_kernel_size=(convn, convn, convn),
            strides=(strider[0], strider[1], strider[2]),
            last_convolution=(lastconv, lastconv, lastconv), number_of_loss_functions=1, interpolation='nearest')
    if sigmoid_second_channel and pro_seg != 0 :
        if dimensionality == 2 :
            input_image_size = (None,None,2)
            if intensity_model is None:
                intensity_model = dbpn( (None,None,1),
                    number_of_outputs=1,
                    number_of_base_filters=nfilt,
                    number_of_feature_filters=nff,
                    number_of_back_projection_stages=nbp,
                    convolution_kernel_size=(convn, convn),
                    strides=(strider[0], strider[1]),
                    last_convolution=(lastconv, lastconv),
                    number_of_loss_functions=1,
                    interpolation='nearest')
            else:
                if freeze:
                    for layer in intensity_model.layers:
                        layer.trainable = False
            if segmentation_model is None:
                segmentation_model = dbpn( (None,None,1),
                        number_of_outputs=1,
                        number_of_base_filters=nfilt,
                        number_of_feature_filters=nff,
                        number_of_back_projection_stages=nbp,
                        convolution_kernel_size=(convn, convn),
                        strides=(strider[0], strider[1]),
                        last_convolution=(lastconv, lastconv),
                        number_of_loss_functions=1, interpolation='linear')
            else:
                if freeze:
                    for layer in segmentation_model.layers:
                        layer.trainable = False
        if dimensionality == 3 :
            input_image_size = (None,None,None,2)
            if intensity_model is None:
                intensity_model = dbpn( (None,None,None,1),
                    number_of_outputs=1,
                    number_of_base_filters=nfilt,
                    number_of_feature_filters=nff,
                    number_of_back_projection_stages=nbp,
                    convolution_kernel_size=(convn, convn, convn),
                    strides=(strider[0], strider[1], strider[2]),
                    last_convolution=(lastconv, lastconv, lastconv),
                    number_of_loss_functions=1, interpolation='nearest')
            else:
                if freeze:
                    for layer in intensity_model.layers:
                        layer.trainable = False
            if segmentation_model is None:
                segmentation_model = dbpn( (None,None,None,1),
                        number_of_outputs=1,
                        number_of_base_filters=nfilt,
                        number_of_feature_filters=nff,
                        number_of_back_projection_stages=nbp,
                        convolution_kernel_size=(convn, convn, convn),
                        strides=(strider[0], strider[1], strider[2]),
                        last_convolution=(lastconv, lastconv, lastconv),
                        number_of_loss_functions=1, interpolation='linear')
            else:
                if freeze:
                    for layer in segmentation_model.layers:
                        layer.trainable = False
        if verbose:
            print( "len intensity_model layers : " + str( len( intensity_model.layers )))
            print( "len intensity_model weights : " + str( len( intensity_model.weights )))
            print( "len segmentation_model layers : " + str( len( segmentation_model.layers )))
            print( "len segmentation_model weights : " + str( len( segmentation_model.weights )))
        if clone_intensity_to_segmentation:
            for k in range(len( segmentation_model.weights )):
                if k < len( intensity_model.weights ):
                    if intensity_model.weights[k].shape == segmentation_model.weights[k].shape:
                        segmentation_model.weights[k] = intensity_model.weights[k]
        inputs = keras.Input(shape=input_image_size)
        insplit = ops.split( inputs, 2, axis=dimensionality+1)
        outputs = [
            intensity_model( insplit[0] ),
            ops.sigmoid( segmentation_model( insplit[1] ) ) ]
        mdlout = ops.concatenate( outputs, axis=dimensionality+1 )
        return Model(inputs=inputs, outputs=mdlout )
    if pro_seg > 0 and intensity_model is not None:
        if verbose and freeze:
            print("Add a segmentation arm to the end. freeze intensity. intensity_model(seg) => conv => sigmoid")
        if verbose and not freeze:
            print("Add a segmentation arm to the end. freeze intensity. intensity_model(seg) => conv => sigmoid")
        if freeze:
            for layer in intensity_model.layers:
                layer.trainable = False
        if dimensionality == 2 :
            input_image_size = (None,None,2)
        elif dimensionality == 3 :
            input_image_size = (None, None,None,2)
        if dimensionality == 2:
            myconv = Conv2D
            firstConv = (convn,convn)
            firstStrides=(1,1)
            smashConv=(pro_seg,pro_seg)
        if dimensionality == 3:
            myconv = Conv3D
            firstConv = (convn,convn,convn)
            firstStrides=(1,1,1)
            smashConv=(pro_seg,pro_seg,pro_seg)
        inputs = keras.Input(shape=input_image_size)
        insplit = ops.split( inputs, 2, axis=dimensionality+1)
        # define segmentation arm
        seggit = intensity_model( insplit[1] )
        L0 = myconv(filters=nff,
                    kernel_size=firstConv,
                    strides=firstStrides,
                    kernel_initializer='glorot_uniform',
                    padding='same')(seggit)
        L1 = myconv(filters=nff,
                    kernel_size=firstConv,
                    strides=firstStrides,
                    kernel_initializer='glorot_uniform',
                    padding='same')(L0)
        L2 = myconv(filters=1,
                    kernel_size=smashConv,
                    strides=firstStrides,
                    kernel_initializer='glorot_uniform',
                    padding='same')(L1)
        outputs = [
            intensity_model( insplit[0] ),
            ops.sigmoid( L2 ) ]
        mdlout = ops.concatenate( outputs, axis=dimensionality+1 )
        return Model(inputs=inputs, outputs=mdlout )
    return mdl


def image_patch_training_data_from_filenames(
    filenames,
    target_patch_size,
    target_patch_size_low,
    nPatches = 128,
    istest   = False,
    patch_scaler=True,
    to_tensorflow = False,
    verbose = False
    ):
    """
    Generates a batch of paired high- and low-resolution image patches for training.

    This function creates training data by taking a list of high-resolution source
    images, extracting random patches, and then downsampling them to create
    low-resolution counterparts. This provides the (input, ground_truth) pairs
    needed to train a super-resolution model.

    Parameters
    ----------
    filenames : list of str
        A list of file paths to the high-resolution source images.

    target_patch_size : tuple or list of int
        The dimensions of the high-resolution (ground truth) patch to extract,
        e.g., `(128, 128, 128)`.

    target_patch_size_low : tuple or list of int
        The dimensions of the low-resolution (input) patch to generate. The ratio
        between `target_patch_size` and `target_patch_size_low` determines the
        super-resolution factor.

    nPatches : int, optional
        The number of patch pairs to generate in this batch. Default is 128.

    istest : bool, optional
        If True, the function also generates a third output array containing patches
        that have been naively upsampled using linear interpolation. This is useful
        for calculating baseline evaluation metrics (e.g., PSNR) against which the
        model's performance can be compared. Default is False.

    patch_scaler : bool, optional
        If True, scales the intensity of each high-resolution patch to the [0, 1]
        range before creating the downsampled version. This can help with
        training stability. Default is True.

    to_tensorflow : bool, optional
        If True, casts the output NumPy arrays to TensorFlow tensors. Default is False.

    verbose : bool, optional
        If True, prints progress messages during patch generation. Default is False.

    Returns
    -------
    tuple
        A tuple of NumPy arrays or TensorFlow tensors.
        - If `istest` is False: `(patchesResam, patchesOrig)`
            - `patchesResam`: The batch of low-resolution input patches (X_train).
            - `patchesOrig`: The batch of high-resolution ground truth patches (y_train).
        - If `istest` is True: `(patchesResam, patchesOrig, patchesUp)`
            - `patchesUp`: The batch of baseline, linearly-upsampled patches.
    """
    if verbose:
        print("begin image_patch_training_data_from_filenames")
    tardim = len( target_patch_size )
    strider = []
    for j in range( tardim ):
        strider.append( np.round( target_patch_size[j]/target_patch_size_low[j]) )
    if tardim == 3:
        shaperhi = (nPatches,target_patch_size[0],target_patch_size[1],target_patch_size[2],1)
        shaperlo = (nPatches,target_patch_size_low[0],target_patch_size_low[1],target_patch_size_low[2],1)
    if tardim == 2: # pragma: no cover
        shaperhi = (nPatches,target_patch_size[0],target_patch_size[1],1)
        shaperlo = (nPatches,target_patch_size_low[0],target_patch_size_low[1],1)
    patchesOrig = np.zeros(shape=shaperhi)
    patchesResam = np.zeros(shape=shaperlo)
    patchesUp = None
    if istest:
        patchesUp = np.zeros(shape=patchesOrig.shape)
    for myn in range(nPatches):
            if verbose:
                print(myn)
            imgfn = random.sample( filenames, 1 )[0]
            if verbose:
                print(imgfn)
            img = ants.image_read( imgfn ).iMath("Normalize")
            if img.components > 1:
                img = ants.split_channels(img)[0]
            img = ants.crop_image( img, ants.threshold_image( img, 0.05, 1 ) )
            ants.set_origin( img, ants.get_center_of_mass(img) )
            img = ants.iMath(img,"Normalize")
            spc = ants.get_spacing( img )
            newspc = []
            for jj in range(len(spc)):
                newspc.append(spc[jj]*strider[jj])
            interp_type = random.choice( [0,1] )
            if True:
                imgp = get_random_patch( img, target_patch_size )
                imgpmin = imgp.min()
                if patch_scaler:
                    imgp = imgp - imgpmin
                    imgpmax = imgp.max()
                    if imgpmax > 0 :
                        imgp = imgp / imgpmax
                rimgp = ants.resample_image( imgp, newspc, use_voxels = False, interp_type=interp_type  )
                if istest:
                    rimgbi = ants.resample_image( rimgp, spc, use_voxels = False, interp_type=0  )
                if tardim == 3:
                    patchesOrig[myn,:,:,:,0] = imgp.numpy()
                    patchesResam[myn,:,:,:,0] = rimgp.numpy()
                    if istest:
                        patchesUp[myn,:,:,:,0] = rimgbi.numpy()
                if tardim == 2: # pragma: no cover
                    patchesOrig[myn,:,:,0] = imgp.numpy()
                    patchesResam[myn,:,:,0] = rimgp.numpy()
                    if istest:
                        patchesUp[myn,:,:,0] = rimgbi.numpy()
    if to_tensorflow:
        patchesOrig = ops.convert_to_tensor( patchesOrig, dtype="float32")
        patchesResam = ops.convert_to_tensor( patchesResam, dtype="float32")
    if istest:
        if to_tensorflow:
            patchesUp = ops.convert_to_tensor( patchesUp, dtype="float32")
    return patchesResam, patchesOrig, patchesUp


def seg_patch_training_data_from_filenames( # pragma: no cover

    filenames,
    target_patch_size,
    target_patch_size_low,
    nPatches = 128,
    istest   = False,
    patch_scaler=True,
    to_tensorflow = False,
    verbose = False
    ):
    """
    Generates a batch of paired training data containing both images and segmentations.

    This function extends `image_patch_training_data_from_filenames` by adding a
    second channel to the data. For each extracted image patch, it also generates
    a corresponding segmentation mask using Otsu's thresholding. This is useful for
    training multi-task models that perform super-resolution on both an image and
    its associated segmentation simultaneously.

    Parameters
    ----------
    filenames : list of str
        A list of file paths to the high-resolution source images.

    target_patch_size : tuple or list of int
        The dimensions of the high-resolution patch, e.g., `(128, 128, 128)`.

    target_patch_size_low : tuple or list of int
        The dimensions of the low-resolution input patch.

    nPatches : int, optional
        The number of patch pairs to generate. Default is 128.

    istest : bool, optional
        If True, also generates a third output array containing baseline upsampled
        intensity images (channel 0 only). Default is False.

    patch_scaler : bool, optional
        If True, scales the intensity of each image patch to the [0, 1] range.
        Default is True.

    to_tensorflow : bool, optional
        If True, casts the output NumPy arrays to TensorFlow tensors. Default is False.

    verbose : bool, optional
        If True, prints progress messages. Default is False.

    Returns
    -------
    tuple
        A tuple of multi-channel NumPy arrays or TensorFlow tensors. The structure
        is the same as `image_patch_training_data_from_filenames`, but each
        array has a channel dimension of 2:
        - Channel 0: The intensity image.
        - Channel 1: The binary segmentation mask.
    """
    if verbose:
        print("begin seg_patch_training_data_from_filenames")
    tardim = len( target_patch_size )
    strider = []
    nchan = 2
    for j in range( tardim ):
        strider.append( np.round( target_patch_size[j]/target_patch_size_low[j]) )
    if tardim == 3:
        shaperhi = (nPatches,target_patch_size[0],target_patch_size[1],target_patch_size[2],nchan)
        shaperlo = (nPatches,target_patch_size_low[0],target_patch_size_low[1],target_patch_size_low[2],nchan)
    if tardim == 2:
        shaperhi = (nPatches,target_patch_size[0],target_patch_size[1],nchan)
        shaperlo = (nPatches,target_patch_size_low[0],target_patch_size_low[1],nchan)
    patchesOrig = np.zeros(shape=shaperhi)
    patchesResam = np.zeros(shape=shaperlo)
    patchesUp = None
    if istest:
        patchesUp = np.zeros(shape=patchesOrig.shape)
    for myn in range(nPatches):
            if verbose:
                print(myn)
            imgfn = random.sample( filenames, 1 )[0]
            if verbose:
                print(imgfn)
            img = ants.image_read( imgfn ).iMath("Normalize")
            if img.components > 1:
                img = ants.split_channels(img)[0]
            img = ants.crop_image( img, ants.threshold_image( img, 0.05, 1 ) )
            ants.set_origin( img, ants.get_center_of_mass(img) )
            img = ants.iMath(img,"Normalize")
            spc = ants.get_spacing( img )
            newspc = []
            for jj in range(len(spc)):
                newspc.append(spc[jj]*strider[jj])
            interp_type = random.choice( [0,1] )
            seg_class = random.choice( [1,2] )
            if True:
                imgp = get_random_patch( img, target_patch_size )
                imgpmin = imgp.min()
                if patch_scaler:
                    imgp = imgp - imgpmin
                    imgpmax = imgp.max()
                    if imgpmax > 0 :
                        imgp = imgp / imgpmax
                segp = ants.threshold_image( imgp, "Otsu", 2 ).threshold_image( seg_class, seg_class )
                rimgp = ants.resample_image( imgp, newspc, use_voxels = False, interp_type=interp_type  )
                rsegp = ants.resample_image( segp, newspc, use_voxels = False, interp_type=interp_type  )
                if istest:
                    rimgbi = ants.resample_image( rimgp, spc, use_voxels = False, interp_type=0  )
                if tardim == 3:
                    patchesOrig[myn,:,:,:,0] = imgp.numpy()
                    patchesResam[myn,:,:,:,0] = rimgp.numpy()
                    patchesOrig[myn,:,:,:,1] = segp.numpy()
                    patchesResam[myn,:,:,:,1] = rsegp.numpy()
                    if istest:
                        patchesUp[myn,:,:,:,0] = rimgbi.numpy()
                if tardim == 2:
                    patchesOrig[myn,:,:,0] = imgp.numpy()
                    patchesResam[myn,:,:,0] = rimgp.numpy()
                    patchesOrig[myn,:,:,1] = segp.numpy()
                    patchesResam[myn,:,:,1] = rsegp.numpy()
                    if istest:
                        patchesUp[myn,:,:,0] = rimgbi.numpy()
    if to_tensorflow:
        patchesOrig = ops.convert_to_tensor( patchesOrig, dtype="float32")
        patchesResam = ops.convert_to_tensor( patchesResam, dtype="float32")
    if istest:
        if to_tensorflow:
            patchesUp = ops.convert_to_tensor( patchesUp, dtype="float32")
    return patchesResam, patchesOrig, patchesUp

def read( filename ): # pragma: no cover
    """
    Reads an image or a NumPy array from a file.

    This function acts as a wrapper to intelligently load data. It checks the
    file extension to decide whether to use `ants.image_read` for standard
    medical image formats (e.g., .nii.gz, .mha) or `numpy.load` for `.npy` files.

    Parameters
    ----------
    filename : str
        The full path to the file to be read.

    Returns
    -------
    ants.ANTsImage or np.ndarray
        The loaded data object, either as an ANTsImage or a NumPy array.
    """
    import re
    isnpy = len( re.sub( ".npy", "", filename ) ) != len( filename )
    if not isnpy:
        myoutput = ants.image_read( filename )
    else:
        myoutput = np.load( filename )
    return myoutput


def auto_weight_loss( # pragma: no cover
 mdl, feature_extractor, x, y, feature=2.0, tv=0.1, verbose=True ):
    """
    Automatically compute weighting coefficients for a combined loss function
    based on intensity (MSE), perceptual similarity (feature), and total variation (TV).

    Parameters
    ----------
    mdl : tf.keras.Model
        A trained or untrained model to evaluate predictions on input `x`.

    feature_extractor : tf.keras.Model
        A model that extracts intermediate features from the input. Commonly a VGG or ResNet
        trained on a perceptual task.

    x : tf.Tensor
        Input batch to the model.

    y : tf.Tensor
        Ground truth target for `x`, typically a batch of 2D or 3D volumes.

    feature : float, optional
        Weighting factor for the feature (perceptual) term in the loss. Default is 2.0.

    tv : float, optional
        Weighting factor for the total variation term in the loss. Default is 0.1.

    verbose : bool, optional
        If True, prints each component of the loss and its scaled value.

    Returns
    -------
    list of float
        A list of computed weights in the order:
        `[msq_weight, feature_weight, tv_weight]`

    Notes
    -----
    The total loss (to be used during training) can then be constructed as:

        `L = msq_weight * MSE + feature_weight * perceptual_loss + tv_weight * TV`

    This function is typically used to balance loss terms before training.
    """    
    y = ops.convert_to_tensor(y)
    y_pred = mdl( x )
    squared_difference = ops.square( y - y_pred)
    if len( y.shape ) == 5:
            tdim = 3
            myax = [1,2,3,4]
    if len( y.shape ) == 4:
            tdim = 2
            myax = [1,2,3]
    msqTerm = ops.mean(squared_difference, axis=myax)
    temp1 = feature_extractor(y)
    temp2 = feature_extractor(y_pred)
    feature_difference = ops.square(temp1-temp2)
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
    
    if verbose :
        print( "MSQ: " + str( float(msqw * mean_msq) ) )
        print( "Feat: " + str( float(featw * mean_feat) ) )
        print( "Tv: " + str(  float(mytv * tvw) ) )
    wts = [float(msqw), float(featw), float(tvw)]
    return wts

def auto_weight_loss_seg( # pragma: no cover
 mdl, feature_extractor, x, y, feature=2.0, tv=0.1, dice=0.5, verbose=True ):
    """
    Automatically compute weighting coefficients for a combined loss function
    that includes MSE, perceptual similarity, total variation, and segmentation Dice loss.

    Parameters
    ----------
    mdl : tf.keras.Model
        A segmentation + super-resolution model that outputs both image and label predictions.

    feature_extractor : tf.keras.Model
        Feature extractor model used to compute perceptual similarity loss.

    x : tf.Tensor
        Input tensor to the model.

    y : tf.Tensor
        Target tensor with two channels: [intensity_image, segmentation_label].

    feature : float, optional
        Relative weight of the perceptual feature loss term. Default is 2.0.

    tv : float, optional
        Relative weight of the total variation (TV) term. Default is 0.1.

    dice : float, optional
        Relative weight of the Dice loss term (for segmentation agreement). Default is 0.5.

    verbose : bool, optional
        If True, prints the scaled values of each component loss.

    Returns
    -------
    list of float
        A list of loss term weights in the order:
        `[msq_weight, feature_weight, tv_weight, dice_weight]`

    Notes
    -----
    - The input and output tensors must be shaped such that the last axis is 2:
      channel 0 is intensity, channel 1 is segmentation.
    - This is useful for dual-task networks that predict both high-res images
      and associated segmentation masks.

    See Also
    --------
    binary_dice_loss : Computes Dice loss between predicted and ground-truth masks.
    """    
    y = ops.convert_to_tensor(y)
    y_pred = mdl( x )
    if len( y.shape ) == 5:
            tdim = 3
            myax = [1,2,3,4]
    if len( y.shape ) == 4:
            tdim = 2
            myax = [1,2,3]
    y_intensity = ops.split( y, 2, axis=tdim+1 )[0]
    y_seg = ops.split( y, 2, axis=tdim+1 )[1]
    y_intensity_p = ops.split( y_pred, 2, axis=tdim+1 )[0]
    y_seg_p = ops.split( y_pred, 2, axis=tdim+1 )[1]
    squared_difference = ops.square( y_intensity - y_intensity_p )
    msqTerm = ops.mean(squared_difference, axis=myax)
    temp1 = feature_extractor(y_intensity)
    temp2 = feature_extractor(y_intensity_p)
    feature_difference = ops.square(temp1-temp2)
    myax_feat = list(range(1, len(feature_difference.shape)))
    featureTerm = ops.mean(feature_difference, axis=myax_feat)
    msqw = 10.0
    mean_msq = ops.mean(msqTerm)
    mean_feat = ops.mean(featureTerm)
    featw = feature * msqw * mean_msq / (mean_feat + 1e-8)
    
    y_shape = ops.shape(y_intensity)
    if tdim == 3:
        reshaped_y = ops.reshape(y_intensity, (-1, y_shape[2], y_shape[3], y_shape[4]))
        mytv = ops_total_variation(reshaped_y)
    else:
        mytv = ops_total_variation(y_intensity)
    tvw = tv * msqw * mean_msq / (mytv + 1e-8)
    
    mydice = binary_dice_loss( y_seg, y_seg_p )
    mydice = ops.mean( mydice )
    dicewt = dice * msqw * mean_msq / (mydice + 1e-8)
    
    if verbose :
        print( "MSQ: " + str( float(msqw * mean_msq) ) )
        print( "Feat: " + str( float(featw * mean_feat) ) )
        print( "Tv: " + str(  float(mytv * tvw) ) )
        print( "Dice: " + str( float(mydice * dicewt) ) )
    wts = [float(msqw), float(featw), float(tvw), float(dicewt)]
    return wts


def numpy_generator( filenames ): # pragma: no cover
    """
    A placeholder or stub for a data generator.

    This generator yields a tuple of `None` values once and then stops. It is
    likely intended as a template or for debugging purposes where a generator
    object is required but no actual data needs to be processed.

    Parameters
    ----------
    filenames : any
        An argument that is not used by the function.

    Yields
    ------
    tuple
        A single tuple `(None, None, None)`.
    """
    patchesResam=patchesOrig=patchesUp=None
    yield (patchesResam, patchesOrig,patchesUp)

def image_generator(
    filenames,
    nPatches,
    target_patch_size,
    target_patch_size_low,
    patch_scaler=True,
    istest=False,
    verbose = False ):
    """
    Creates an infinite generator of paired image patches for model training.

    This function continuously generates batches of low-resolution (input) and
    high-resolution (ground truth) image patches. It is designed to be fed
    directly into a Keras `model.fit()` call.

    Parameters
    ----------
    filenames : list of str
        List of file paths to the high-resolution source images.

    nPatches : int
        The number of patch pairs to generate and yield in each batch.

    target_patch_size : tuple or list of int
        The dimensions of the high-resolution (ground truth) patches.

    target_patch_size_low : tuple or list of int
        The dimensions of the low-resolution (input) patches.

    patch_scaler : bool, optional
        If True, scales patch intensities to [0, 1]. Default is True.

    istest : bool, optional
        If True, the generator will also yield a third item: a baseline
        linearly upsampled version of the low-resolution patch for comparison.
        Default is False.

    verbose : bool, optional
        If True, passes verbosity to the underlying patch generation function.
        Default is False.

    Yields
    -------
    tuple
        A tuple of TensorFlow tensors ready for training or evaluation.
        - If `istest` is False: `(low_res_batch, high_res_batch)`
        - If `istest` is True: `(low_res_batch, high_res_batch, baseline_upsampled_batch)`

    See Also
    --------
    image_patch_training_data_from_filenames : The function that performs the
                                               underlying patch extraction.
    """
    while True:
        patchesResam, patchesOrig, patchesUp = image_patch_training_data_from_filenames(
            filenames,
            target_patch_size = target_patch_size,
            target_patch_size_low = target_patch_size_low,
            nPatches = nPatches,
            istest   = istest,
            patch_scaler=patch_scaler,
            to_tensorflow = True,
            verbose = verbose )
        if istest:
            yield (patchesResam, patchesOrig,patchesUp)
        yield (patchesResam, patchesOrig)


def seg_generator( # pragma: no cover

    filenames,
    nPatches,
    target_patch_size,
    target_patch_size_low,
    patch_scaler=True,
    istest=False,
    verbose = False ):
    """
    Creates an infinite generator of paired image and segmentation patches.

    This function continuously generates batches of multi-channel patches, where
    one channel is the intensity image and the other is a segmentation mask.
    It is designed for training multi-task super-resolution models.

    Parameters
    ----------

    filenames : list of str
        List of file paths to the high-resolution source images.

    nPatches : int
        The number of patch pairs to generate and yield in each batch.

    target_patch_size : tuple or list of int
        The dimensions of the high-resolution patches.

    target_patch_size_low : tuple or list of int
        The dimensions of the low-resolution patches.

    patch_scaler : bool, optional
        If True, scales the intensity channel of patches to [0, 1]. Default is True.

    istest : bool, optional
        If True, yields an additional baseline upsampled patch for comparison.
        Default is False.

    verbose : bool, optional
        If True, passes verbosity to the underlying patch generation function.
        Default is False.

    Yields
    -------
    tuple
        A tuple of multi-channel TensorFlow tensors. Each tensor has two channels:
        Channel 0 contains the intensity image, and Channel 1 contains the
        segmentation mask.

    See Also
    --------
    seg_patch_training_data_from_filenames : The function that performs the
                                             underlying patch extraction.
    image_generator : A similar generator for intensity-only data.
    """
    while True:
        patchesResam, patchesOrig, patchesUp = seg_patch_training_data_from_filenames(
            filenames,
            target_patch_size = target_patch_size,
            target_patch_size_low = target_patch_size_low,
            nPatches = nPatches,
            istest   = istest,
            patch_scaler=patch_scaler,
            to_tensorflow = True,
            verbose = verbose )
        if istest:
            yield (patchesResam, patchesOrig,patchesUp)
        yield (patchesResam, patchesOrig)


def train(
    mdl,
    filenames_train,
    filenames_test,
    target_patch_size,
    target_patch_size_low,
    output_prefix,
    n_test = 8,
    learning_rate=5e-5,
    feature_layer = 6,
    feature = 2,
    tv = 0.1,
    max_iterations = 1000,
    batch_size = 1,
    save_all_best = False,
    feature_type = 'grader',
    check_eval_data_iteration = 20,
    verbose = False  ):
    """
    Orchestrates the training process for a super-resolution model.

    This function handles the entire training loop, including setting up data
    generators, defining a composite loss function, automatically balancing loss
    weights, iteratively training the model, periodically evaluating performance,
    and saving the best-performing model weights.

    Parameters
    ----------
    mdl : tf.keras.Model
        The Keras model to be trained.

    filenames_train : list of str
        List of file paths for the training dataset.

    filenames_test : list of str
        List of file paths for the validation/testing dataset.

    target_patch_size : tuple or list
        The dimensions of the high-resolution target patches.

    target_patch_size_low : tuple or list
        The dimensions of the low-resolution input patches.

    output_prefix : str
        A prefix for all output files (e.g., model weights, training logs).

    n_test : int, optional
        The number of validation patches to use for evaluation. Default is 8.

    learning_rate : float, optional
        The learning rate for the Adam optimizer. Default is 5e-5.

    feature_layer : int, optional
        The layer index from the feature extractor to use for perceptual loss.
        Default is 6.

    feature : float, optional
        The relative weight of the perceptual (feature) loss term. Default is 2.0.

    tv : float, optional
        The relative weight of the Total Variation (TV) regularization term.
        Default is 0.1.

    max_iterations : int, optional
        The total number of training iterations to run. Default is 1000.

    batch_size : int, optional
        The batch size for training. Note: this implementation is optimized for
        batch_size=1 and may need adjustment for larger batches. Default is 1.

    save_all_best : bool, optional
        If True, saves a new model file every time validation loss improves.
        If False, overwrites the single best model file. Default is False.

    feature_type : str, optional
        The type of feature extractor for perceptual loss. Options: 'grader',
        'vgg', 'vggrandom'. Default is 'grader'.

    check_eval_data_iteration : int, optional
        The frequency (in iterations) at which to run validation and save logs.
        Default is 20.

    verbose : bool, optional
        If True, prints detailed progress information. Default is False.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the training history, with columns for training
        loss, validation loss, PSNR, and baseline PSNR over iterations.
    """
    colnames = ['train_loss','test_loss','best','eval_psnr','eval_psnr_lin']
    training_path = np.zeros( [ max_iterations, len(colnames) ] )
    training_weights = np.zeros( [1,3] )
    if verbose:
        print("begin get feature extractor " + feature_type)
    if feature_type == 'grader':
        feature_extractor = get_grader_feature_network( feature_layer )
    elif feature_type == 'vggrandom':
        feature_extractor = pseudo_3d_vgg_features( target_patch_size, feature_layer, pretrained=False )
    elif feature_type == 'vgg':
        feature_extractor = pseudo_3d_vgg_features_unbiased( target_patch_size, feature_layer )
    else:
        raise Exception("feature type does not exist")
    if verbose:
        print("begin train generator")
    mydatgen = image_generator(
        filenames_train,
        nPatches=1,
        target_patch_size=target_patch_size,
        target_patch_size_low=target_patch_size_low,
        istest=False , verbose=False)
    if verbose:
        print("begin test generator")
    mydatgenTest = image_generator( filenames_test, nPatches=1,
        target_patch_size=target_patch_size,
        target_patch_size_low=target_patch_size_low,
        istest=True, verbose=True)
    patchesResamTeTf, patchesOrigTeTf, patchesUpTeTf = next( mydatgenTest )
    if len( patchesOrigTeTf.shape ) == 5:
            tdim = 3
            myax = [1,2,3,4]
    if len( patchesOrigTeTf.shape ) == 4:
            tdim = 2
            myax = [1,2,3]
    if verbose:
        print("begin train generator #2 at dim: " + str( tdim))
    mydatgenTest = image_generator( filenames_test, nPatches=1,
        target_patch_size=target_patch_size,
        target_patch_size_low=target_patch_size_low,
        istest=True, verbose=True)
    patchesResamTeTfB, patchesOrigTeTfB, patchesUpTeTfB = next( mydatgenTest )
    for k in range( n_test - 1 ):
        mydatgenTest = image_generator( filenames_test, nPatches=1,
            target_patch_size=target_patch_size,
            target_patch_size_low=target_patch_size_low,
            istest=True, verbose=True)
        temp0, temp1, temp2 = next( mydatgenTest )
        patchesResamTeTfB = ops.concatenate( [patchesResamTeTfB,temp0],axis=0)
        patchesOrigTeTfB = ops.concatenate( [patchesOrigTeTfB,temp1],axis=0)
        patchesUpTeTfB = ops.concatenate( [patchesUpTeTfB,temp2],axis=0)
    if verbose:
        print("begin auto_weight_loss")
    wts_csv = output_prefix + "_training_weights.csv"
    if exists( wts_csv ):
        wtsdf = pd.read_csv( wts_csv )
        wts = [wtsdf['msq'][0], wtsdf['feat'][0], wtsdf['tv'][0]]
        if verbose:
            print( "preset weights:" )
    else:
        wts = auto_weight_loss( mdl, feature_extractor, patchesResamTeTf, patchesOrigTeTf,
            feature=feature, tv=tv )
        for k in range(len(wts)):
            training_weights[0,k]=wts[k]
        pd.DataFrame(training_weights, columns = ["msq","feat","tv"] ).to_csv( wts_csv )
        if verbose:
            print( "automatic weights:" )
    if verbose:
        print( wts )
    def my_loss_6( y_true, y_pred, msqwt = wts[0], fw = wts[1], tvwt = wts[2], mybs = batch_size ):
        """Composite loss: MSE + Perceptual Loss + Total Variation."""
        squared_difference = ops.square(y_true - y_pred)
        if len( y_true.shape ) == 5:
            tdim = 3
            myax = [1,2,3,4]
        if len( y_true.shape ) == 4:
            tdim = 2
            myax = [1,2,3]
        msqTerm = ops.mean(squared_difference, axis=myax)
        temp1 = feature_extractor(y_true)
        temp2 = feature_extractor(y_pred)
        feature_difference = ops.square(temp1-temp2)
        featureTerm = ops.mean(feature_difference, axis=myax)
        loss = msqTerm * msqwt + featureTerm * fw
        if tdim == 3:
            reshaped_pred = ops.reshape(y_pred, (-1, y_pred.shape[2], y_pred.shape[3], y_pred.shape[4]))
            mytv = ops_total_variation(reshaped_pred) * tvwt
        if tdim == 2:
            mytv = ops_total_variation(y_pred) * tvwt
        return( loss + mytv )
    if verbose:
        print("begin model compilation")
    opt = keras.optimizers.Adam( learning_rate=learning_rate )
    mdl.compile(optimizer=opt, loss=my_loss_6)
    # set up some parameters for tracking performance
    bestValLoss=1e12
    bestSSIM=0.0
    bestQC0 = -1000
    bestQC1 = -1000
    if verbose:
        print( "begin training", flush=True  )
    for myrs in range( max_iterations ):
        tracker = mdl.fit( mydatgen,  epochs=2, steps_per_epoch=4, verbose=1,
            validation_data=(patchesResamTeTf,patchesOrigTeTf) )
        training_path[myrs,0]=tracker.history['loss'][0]
        training_path[myrs,1]=tracker.history['val_loss'][0]
        training_path[myrs,2]=0
        print( "ntrain: " + str(myrs) + " loss " + str( tracker.history['loss'][0] ) + ' val-loss ' + str(tracker.history['val_loss'][0]), flush=True  )
        if myrs % check_eval_data_iteration == 0:
            myofn = output_prefix + "_best_mdl.keras"
            if save_all_best:
                myofn = output_prefix + "_" + str(myrs)+ "_mdl.keras"
            tester = mdl.evaluate( patchesResamTeTfB, patchesOrigTeTfB )
            if ( tester < bestValLoss ):
                print("MyIT " + str( myrs ) + " IS BEST!! " + str( tester ) + myofn, flush=True )
                bestValLoss = tester
                mdl.save( myofn )
                training_path[myrs,2]=1
            pp = mdl.predict( patchesResamTeTfB, batch_size = 1 )
            pp = ops.convert_to_tensor(pp)
            myssimSR = ops_psnr( pp * 220, patchesOrigTeTfB * 220, max_val=255 )
            myssimSR = float(ops.mean( myssimSR ))
            myssimBI = ops_psnr( patchesUpTeTfB * 220, patchesOrigTeTfB * 220, max_val=255 )
            myssimBI = float(ops.mean( myssimBI ))
            print( myofn + " : " + "PSNR Lin: " + str( myssimBI ) + " SR: " + str( myssimSR ), flush=True  )
            training_path[myrs,3]=myssimSR # psnr
            training_path[myrs,4]=myssimBI # psnrlin
            pd.DataFrame(training_path, columns = colnames ).to_csv( output_prefix + "_training.csv" )
    training_path = pd.DataFrame(training_path, columns = colnames )
    return training_path


def binary_dice_loss(y_true, y_pred):
    """
    Computes the Dice loss for binary segmentation tasks.

    The Dice coefficient is a common metric for comparing the overlap of two samples.
    This loss function computes `1 - DiceCoefficient`, making it suitable for
    minimization during training. A smoothing factor is added to avoid division
    by zero when both the prediction and the ground truth are empty.

    Parameters
    ----------
    y_true : tf.Tensor
        The ground truth binary segmentation mask. Values should be 0 or 1.

    y_pred : tf.Tensor
        The predicted binary segmentation mask, typically with values in [0, 1]
        from a sigmoid activation.

    Returns
    -------
    tf.Tensor
        A scalar tensor representing the Dice loss. The value ranges from -1 (perfect
        match) to 0 (no overlap), though it's typically used as `1 - dice_coeff`
        or just `-dice_coeff` (as here).
    """
    smoothing_factor = 1e-4
    y_true_f = ops.reshape(y_true, [-1])
    y_pred_f = ops.reshape(y_pred, [-1])
    intersection = ops.sum(y_true_f * y_pred_f)
    # This is -1 * Dice Similarity Coefficient
    return -1 * (2 * intersection + smoothing_factor)/(ops.sum(y_true_f) +
            ops.sum(y_pred_f) + smoothing_factor)

def train_seg( # pragma: no cover

    mdl,
    filenames_train,
    filenames_test,
    target_patch_size,
    target_patch_size_low,
    output_prefix,
    n_test = 8,
    learning_rate=5e-5,
    feature_layer = 6,
    feature = 2,
    tv = 0.1,
    dice = 0.5,
    max_iterations = 1000,
    batch_size = 1,
    save_all_best = False,
    feature_type = 'grader',
    check_eval_data_iteration = 20,
    verbose = False  ):
    """
    Orchestrates training for a multi-task image and segmentation SR model.

    This function extends the `train` function to handle models that predict
    both a super-resolved image and a super-resolved segmentation mask. It uses
    a four-component composite loss: MSE (for image), a perceptual loss (for
    image), Total Variation (for image), and Dice loss (for segmentation).

    Parameters
    ----------
    mdl : tf.keras.Model
        The 2-channel Keras model to be trained.

    filenames_train : list of str
        List of file paths for the training dataset.

    filenames_test : list of str
        List of file paths for the validation/testing dataset.

    target_patch_size : tuple or list
        The dimensions of the high-resolution target patches.

    target_patch_size_low : tuple or list
        The dimensions of the low-resolution input patches.

    output_prefix : str
        A prefix for all output files.

    n_test : int, optional
        Number of validation patches for evaluation. Default is 8.

    learning_rate : float, optional
        Learning rate for the Adam optimizer. Default is 5e-5.

    feature_layer : int, optional
        Layer from the feature extractor for perceptual loss. Default is 6.

    feature : float, optional
        Relative weight of the perceptual loss term. Default is 2.0.

    tv : float, optional
        Relative weight of the Total Variation regularization term. Default is 0.1.

    dice : float, optional
        Relative weight of the Dice loss term for the segmentation mask.
        Default is 0.5.

    max_iterations : int, optional
        Total number of training iterations. Default is 1000.

    batch_size : int, optional
        The batch size for training. Default is 1.

    save_all_best : bool, optional
        If True, saves all models that improve validation loss. Default is False.

    feature_type : str, optional
        Type of feature extractor for perceptual loss. Default is 'grader'.

    check_eval_data_iteration : int, optional
        Frequency (in iterations) for running validation. Default is 20.

    verbose : bool, optional
        If True, prints detailed progress information. Default is False.

    Returns
    -------

    pd.DataFrame
        A DataFrame containing the training history, including columns for losses
        and evaluation metrics like PSNR and Dice score.

    See Also
    --------
    train : The training function for single-task (intensity-only) models.
    """
    colnames = ['train_loss','test_loss','best','eval_psnr','eval_psnr_lin','eval_msq','eval_dice']
    training_path = np.zeros( [ max_iterations, len(colnames) ] )
    training_weights = np.zeros( [1,4] )
    if verbose:
        print("begin get feature extractor")
    if feature_type == 'grader':
        feature_extractor = get_grader_feature_network( feature_layer )
    elif feature_type == 'vggrandom':
        feature_extractor = pseudo_3d_vgg_features( target_patch_size, feature_layer, pretrained=False )
    else:
        feature_extractor = pseudo_3d_vgg_features_unbiased( target_patch_size, feature_layer  )
    if verbose:
        print("begin train generator")
    mydatgen = seg_generator(
        filenames_train,
        nPatches=1,
        target_patch_size=target_patch_size,
        target_patch_size_low=target_patch_size_low,
        istest=False , verbose=False)
    if verbose:
        print("begin test generator")
    mydatgenTest = seg_generator( filenames_test, nPatches=1,
        target_patch_size=target_patch_size,
        target_patch_size_low=target_patch_size_low,
        istest=True, verbose=True)
    patchesResamTeTf, patchesOrigTeTf, patchesUpTeTf = next( mydatgenTest )
    if len( patchesOrigTeTf.shape ) == 5:
            tdim = 3
            myax = [1,2,3,4]
    if len( patchesOrigTeTf.shape ) == 4:
            tdim = 2
            myax = [1,2,3]
    if verbose:
        print("begin train generator #2 at dim: " + str( tdim))
    mydatgenTest = seg_generator( filenames_test, nPatches=1,
        target_patch_size=target_patch_size,
        target_patch_size_low=target_patch_size_low,
        istest=True, verbose=True)
    patchesResamTeTfB, patchesOrigTeTfB, patchesUpTeTfB = next( mydatgenTest )
    for k in range( n_test - 1 ):
        mydatgenTest = seg_generator( filenames_test, nPatches=1,
            target_patch_size=target_patch_size,
            target_patch_size_low=target_patch_size_low,
            istest=True, verbose=True)
        temp0, temp1, temp2 = next( mydatgenTest )
        patchesResamTeTfB = ops.concatenate( [patchesResamTeTfB,temp0],axis=0)
        patchesOrigTeTfB = ops.concatenate( [patchesOrigTeTfB,temp1],axis=0)
        patchesUpTeTfB = ops.concatenate( [patchesUpTeTfB,temp2],axis=0)
    if verbose:
        print("begin auto_weight_loss_seg")
    wts_csv = output_prefix + "_training_weights.csv"
    if exists( wts_csv ):
        wtsdf = pd.read_csv( wts_csv )
        wts = [wtsdf['msq'][0], wtsdf['feat'][0], wtsdf['tv'][0], wtsdf['dice'][0]]
        if verbose:
            print( "preset weights:" )
    else:
        wts = auto_weight_loss_seg( mdl, feature_extractor, patchesResamTeTf, patchesOrigTeTf,
            feature=feature, tv=tv, dice=dice )
        for k in range(len(wts)):
            training_weights[0,k]=wts[k]
        pd.DataFrame(training_weights, columns = ["msq","feat","tv","dice"] ).to_csv( wts_csv )
        if verbose:
            print( "automatic weights:" )
    if verbose:
        print( wts )
    def my_loss_6( y_true, y_pred, msqwt = wts[0], fw = wts[1], tvwt = wts[2], dicewt=wts[3], mybs = batch_size ):
        """Composite loss: MSE + Perceptual + TV + Dice."""
        if len( y_true.shape ) == 5:
            tdim = 3
            myax = [1,2,3,4]
        if len( y_true.shape ) == 4:
            tdim = 2
            myax = [1,2,3]
        y_intensity = ops.split( y_true, 2, axis=tdim+1 )[0]
        y_seg = ops.split( y_true, 2, axis=tdim+1 )[1]
        y_intensity_p = ops.split( y_pred, 2, axis=tdim+1 )[0]
        y_seg_p = ops.split( y_pred, 2, axis=tdim+1 )[1]
        squared_difference = ops.square(y_intensity - y_intensity_p)
        msqTerm = ops.mean(squared_difference, axis=myax)
        temp1 = feature_extractor(y_intensity)
        temp2 = feature_extractor(y_intensity_p)
        feature_difference = ops.square(temp1-temp2)
        featureTerm = ops.mean(feature_difference, axis=myax)
        loss = msqTerm * msqwt + featureTerm * fw
        y_pred_intensity = ops.split( y_pred, 2, axis=tdim+1 )[0]
        if tdim == 3:
            reshaped_pred = ops.reshape(y_pred_intensity, (-1, y_pred_intensity.shape[2], y_pred_intensity.shape[3], y_pred_intensity.shape[4]))
            mytv = ops_total_variation(reshaped_pred) * tvwt
        if tdim == 2:
            mytv = ops_total_variation(y_pred_intensity) * tvwt
        dicer = ops.mean( dicewt * binary_dice_loss( y_seg, y_seg_p ) )
        return( loss + mytv + dicer )
    if verbose:
        print("begin model compilation")
    opt = keras.optimizers.Adam( learning_rate=learning_rate )
    mdl.compile(optimizer=opt, loss=my_loss_6)
    # set up some parameters for tracking performance
    bestValLoss=1e12
    bestSSIM=0.0
    bestQC0 = -1000
    bestQC1 = -1000
    if verbose:
        print( "begin training", flush=True  )
    for myrs in range( max_iterations ):
        tracker = mdl.fit( mydatgen,  epochs=2, steps_per_epoch=4, verbose=1,
            validation_data=(patchesResamTeTf,patchesOrigTeTf) )
        training_path[myrs,0]=tracker.history['loss'][0]
        training_path[myrs,1]=tracker.history['val_loss'][0]
        training_path[myrs,2]=0
        print( "ntrain: " + str(myrs) + " loss " + str( tracker.history['loss'][0] ) + ' val-loss ' + str(tracker.history['val_loss'][0]), flush=True  )
        if myrs % check_eval_data_iteration == 0:
            myofn = output_prefix + "_best_mdl.keras"
            if save_all_best:
                myofn = output_prefix + "_" + str(myrs)+ "_mdl.keras"
            tester = mdl.evaluate( patchesResamTeTfB, patchesOrigTeTfB )
            if ( tester < bestValLoss ):
                print("MyIT " + str( myrs ) + " IS BEST!! " + str( tester ) + myofn, flush=True )
                bestValLoss = tester
                mdl.save( myofn )
                training_path[myrs,2]=1
            pp = mdl.predict( patchesResamTeTfB, batch_size = 1 )
            pp = ops.convert_to_tensor(pp)
            pp = ops.split( pp, 2, axis=tdim+1 )
            y_orig = ops.split( patchesOrigTeTfB, 2, axis=tdim+1 )
            y_up = ops.split( patchesUpTeTfB, 2, axis=tdim+1 )
            myssimSR = ops_psnr( pp[0] * 220, y_orig[0]* 220, max_val=255 )
            myssimSR = float(ops.mean( myssimSR ))
            myssimBI = ops_psnr( y_up[0] * 220, y_orig[0]* 220, max_val=255 )
            myssimBI = float(ops.mean( myssimBI ))
            squared_difference = ops.square(y_orig[0] - pp[0])
            msqTerm = float(ops.mean(squared_difference))
            dicer = binary_dice_loss( y_orig[1], pp[1] )
            dicer = float(ops.mean( dicer ))
            print( myofn + " : " + "PSNR Lin: " + str( myssimBI ) + " SR: " + str( myssimSR ) + " MSQ: " + str(msqTerm) + " DICE: " + str(dicer), flush=True  )
            training_path[myrs,3]=myssimSR # psnr
            training_path[myrs,4]=myssimBI # psnrlin
            training_path[myrs,5]=msqTerm # msq
            training_path[myrs,6]=dicer # dice
            pd.DataFrame(training_path, columns = colnames ).to_csv( output_prefix + "_training.csv" )
    training_path = pd.DataFrame(training_path, columns = colnames )
    return training_path


def read_srmodel( srfilename, custom_objects=None ): # pragma: no cover
    """
    Load a super-resolution model (h5, .keras, or SavedModel format),
    and determine its upsampling factor.

    Parameters
    ----------
    srfilename : str
        Path to the model file (.h5, .keras, or a SavedModel folder).
    custom_objects : dict, optional
        Dictionary of custom objects used in the model (e.g. {'TFOpLambda': tf.keras.layers.Lambda(...)})

    Returns
    -------
    model : tf.keras.Model
        The loaded model.
    upsampling_factor : list of int
        List describing the upsampling factor:
        - For 3D input: [x_up, y_up, z_up, channels]
        - For 2D input: [x_up, y_up, channels]

    Example
    -------
    >>> mdl, up = read_srmodel("mymodel.keras")
    >>> mdl, up = read_srmodel("my_weights.h5", custom_objects={"TFOpLambda": tf.keras.layers.Lambda(tf.identity)})
    """

    # Expand path and detect format
    srfilename = os.path.expanduser(srfilename)
    ext = os.path.splitext(srfilename)[1].lower()

    if os.path.isdir(srfilename):
        # SavedModel directory
        model = keras.models.load_model(srfilename, custom_objects=custom_objects, compile=False)
    elif ext in ['.h5', '.keras']:
        model = keras.models.load_model(srfilename, custom_objects=custom_objects, compile=False)
    else:
        raise ValueError(f"Unsupported model format: {ext}")

    # Determine channel index
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    chanindex = 3 if len(input_shape) == 4 else 4
    nchan = int(input_shape[chanindex])

    # Run dummy input to compute upsampling factor
    try:
        if len(input_shape) == 5:  # 3D
            dummy_input = np.zeros([1, 8, 8, 8, nchan])
        else:  # 2D
            dummy_input = np.zeros([1, 8, 8, nchan])

        # Handle named inputs if necessary
        try:
            output = model(dummy_input)
        except Exception:
            output = model({model.input_names[0]: dummy_input})

        outshp = output.shape
        if len(input_shape) == 5:
            return model, [int(outshp[1]/8), int(outshp[2]/8), int(outshp[3]/8), nchan]
        else:
            return model, [int(outshp[1]/8), int(outshp[2]/8), nchan]

    except Exception as e:
        raise RuntimeError(f"Could not infer upsampling factor. Error: {e}")


def save_siq_model(  # pragma: no cover
    model_path,
    model,
    config,
    loss_weights=None,
    archive=False,
    tag="",
    notes="",
    verbose=True,
):
    """Save a siq model and its mandatory provenance config together.

    Always use this instead of ``model.save()`` directly so that the
    companion ``_config.json`` is always present alongside the weights.

    Parameters
    ----------
    model_path : str
        Destination path, must end in ``.keras``.
    model : keras.Model
        Trained model to save.
    config : dict
        Provenance dict as returned by :func:`load_siq_model` or built by
        the training reporter. Must contain at minimum:
        ``normalization``, ``input_patch_shape``, ``inference``.
    loss_weights : dict, optional
        Dictionary of loss weights at save time to store in provenance config.
    archive : bool
        If ``True``, also copy the model into the immutable ``model_archive/``
        directory and register it in ``model_registry.json``. Use this for
        champion checkpoints (new CQS best). Default ``False``.
    tag : str
        Human-readable label for the registry entry (e.g. ``"v0.10.14_cqs"``).
        Only used when ``archive=True``.
    notes : str
        Free-form notes stored in the registry entry.
        Only used when ``archive=True``.
    verbose : bool
        Print save path.
    """
    import json
    model.save(model_path)
    if loss_weights is not None:
        config["loss_weights"] = {k: float(v) for k, v in loss_weights.items()}
    config_path = model_path.replace('.keras', '_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    if verbose:
        print(f"Saved model:  {model_path}")
        print(f"Saved config: {config_path}")
    if archive:
        from .model_registry import archive_model as _archive
        _archive(model_path, config, tag=tag, notes=notes, verbose=verbose)


def load_siq_model(model_path, custom_objects=None, verbose=True):  # pragma: no cover
    """Load a siq model AND its mandatory provenance config.

    This is the canonical read entry point. It raises ``FileNotFoundError``
    if the companion ``_config.json`` is missing — no silent fallbacks,
    no guessing about normalization.

    Parameters
    ----------
    model_path : str
        Path to the ``.keras`` file.
    custom_objects : dict, optional
        Custom Keras objects required to deserialize the model.
    verbose : bool
        Print loaded shapes and normalization method.

    Returns
    -------
    model : keras.Model
        The loaded model, ready for inference.
    config : dict
        Provenance dict. Key entries:

        - ``normalization``   — how to prepare inputs (always volume-level [0,1])
        - ``input_patch_shape``  — spatial + channel shape the model expects
        - ``output_patch_shape`` — spatial + channel shape the model produces
        - ``upsample_factor``    — integer upsampling factor per spatial axis
        - ``inference``          — dispatch rules (single-pass vs overlapping)
        - ``training``           — training hyperparameters at save time
        - ``val_metrics``        — validation metrics at save time
        - ``loss_weights``       — balanced training loss weights at save time

    Raises
    ------
    FileNotFoundError
        If no ``_config.json`` companion exists alongside ``model_path``.

    Example
    -------
    >>> model, config = siq.load_siq_model('checkpoints/asdbpn_3d_step_1000.keras')
    >>> sr = siq.inference(image, model, config)
    """
    import json, keras
    model_path = os.path.expanduser(model_path)
    config_path = model_path.replace('.keras', '_config.json')

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"No provenance config found at:\n  {config_path}\n"
            f"Cannot safely run inference without knowing normalization method.\n"
            f"Use siq.save_siq_model(path, model, config) to save models with provenance,\n"
            f"or manually create a _config.json using the siq.default_siq_config() template."
        )

    with open(config_path) as f:
        config = json.load(f)

    keras.config.enable_unsafe_deserialization()
    model = keras.models.load_model(
        model_path, custom_objects=custom_objects, compile=False, safe_mode=False
    )

    if verbose:
        print(f"[siq] Loaded model: {model_path}")
        print(f"  Input patch:    {config['input_patch_shape']}")
        print(f"  Output patch:   {config['output_patch_shape']}")
        print(f"  Upsample:       {config['upsample_factor']}×")
        print(f"  Normalization:  {config['normalization']['method']} → {config['normalization']['output_range']}")
        if 'val_metrics' in config:
            m = config['val_metrics']
            print(f"  Val metrics:    PSNR={m.get('val_psnr',float('nan')):.3f} "
                  f"SSIM={m.get('val_ssim',float('nan')):.4f} "
                  f"GMSD={m.get('val_gmsd',float('nan')):.4f} "
                  f"HFEN={m.get('val_hfen',float('nan')):.4f}")
        if 'loss_weights' in config:
            lw = config['loss_weights']
            print(f"  Loss weights:   L1={lw.get('l1', 0.0):.4f} "
                  f"Feat={lw.get('feat', 0.0):.2e} "
                  f"TV={lw.get('tv', 0.0):.4f} "
                  f"GMS={lw.get('gms', 0.0):.2f} "
                  f"CBI={lw.get('cbi', 0.0):.2f}")
    return model, config


def default_siq_config(model=None):  # pragma: no cover
    """Return a default provenance config dict, optionally populated from a model.

    Use this as a template when manually constructing configs for legacy
    checkpoints that pre-date automated provenance saving.

    Parameters
    ----------
    model : keras.Model, optional
        If provided, ``input_patch_shape`` and ``output_patch_shape`` are
        populated from the model's input/output shapes.

    Returns
    -------
    dict
        Config dict with all required keys filled with sensible defaults.
    """
    cfg = {
        "model_type": "asdbpn_3d",
        "siq_version": "unknown",
        "saved_at": "unknown",
        "input_patch_shape": [64, 64, 64, 1],
        "output_patch_shape": [128, 128, 128, 1],
        "upsample_factor": 2,
        "normalization": {
            "method": "volume",
            "truncate_quantiles": [0.001, 0.999],
            "output_range": [0.0, 1.0],
            "note": (
                "Apply ants.iMath(vol,'TruncateIntensity',0.001,0.999) then "
                "ants.iMath(vol,'Normalize') to the WHOLE volume. "
                "Do NOT normalize per-patch."
            )
        },
        "inference": {
            "preferred_method": "direct_single_patch_if_fits",
            "patch_overlap": 16,
            "output_clip": [0.0, 1.0],
            "antspynet_wrapper": False,
            "anti_checkerboard": "auto",
            "anti_checkerboard_sigma": "auto",
        },
    }
    if model is not None:
        in_s = list(model.input_shape[1:])
        out_s = list(model.output_shape[1:])
        # Dynamic-size models: record a dimension-correct placeholder patch (64^ndim -> 128^ndim),
        # never a 3D placeholder for a 2D model.
        ndim = len(in_s) - 1
        if any(s is None for s in in_s[:-1]):
            cfg["input_patch_shape"] = [64] * ndim + [in_s[-1] if in_s[-1] is not None else 1]
        else:
            cfg["input_patch_shape"] = in_s
        if any(s is None for s in out_s[:-1]):
            cfg["output_patch_shape"] = [128] * ndim + [out_s[-1] if out_s[-1] is not None else 1]
        else:
            cfg["output_patch_shape"] = out_s
        try:
            dim_factors = []
            for o, i in zip(out_s[:-1], in_s[:-1]):
                if o is not None and i is not None:
                    dim_factors.append(int(round(o / i)))
            if dim_factors and len(dim_factors) == len(out_s) - 1:
                cfg["upsample_factor"] = dim_factors[0] if len(set(dim_factors)) == 1 else dim_factors
            else:
                ndim = len(in_s) - 1
                dummy_shape = [1] + [8] * ndim + [in_s[-1] if in_s[-1] is not None else 1]
                dummy = keras.ops.zeros(dummy_shape)
                out_dummy = model(dummy)
                dim_factors = [int(round(out_dummy.shape[d+1] / dummy.shape[d+1])) for d in range(ndim)]
                cfg["upsample_factor"] = dim_factors[0] if len(set(dim_factors)) == 1 else dim_factors
        except Exception:
            cfg["upsample_factor"] = 2
    return cfg


def simulate_image( # pragma: no cover
 shaper=[32,32,32], n_levels=10, multiply=False ):
    """
    generate an image of given shape and number of levels

    Arguments
    ---------
    shaper : [x,y,z] or [x,y]

    n_levels : int

    multiply : boolean

    Returns
    -------

    ants.image

    """
    img = ants.from_numpy( np.random.normal( 0, 1.0, size=shaper ) ) * 0
    for k in range(n_levels):
        temp = ants.from_numpy( np.random.normal( 0, 1.0, size=shaper ) )
        temp = ants.smooth_image( temp, n_levels )
        temp = ants.threshold_image( temp, "Otsu", 1 )
        if multiply:
            temp = temp * k
        img = img + temp
    return img


def _sample_param(p, default_val=None, is_int=False): # pragma: no cover
    """
    Internal helper to sample a parameter from various formats:
    - Scalar: returned as-is.
    - Tuple/List: sampled uniformly between [0] and [1].
    - Callable: called to get a value.
    - Dict: uses 'type' to determine distribution (uniform, gaussian, poisson).
    """
    if p is None:
        p = default_val
    if isinstance(p, (int, float)):
        return p
    if isinstance(p, (list, tuple)):
        if is_int:
            return np.random.randint(p[0], p[1])
        return np.random.uniform(p[0], p[1])
    if callable(p):
        return p()
    if isinstance(p, dict):
        dist_type = p.get("type", "uniform")
        if dist_type == "uniform":
            low, high = p.get("low", 0), p.get("high", 1)
            return np.random.randint(low, high) if is_int else np.random.uniform(low, high)
        if dist_type in ["gaussian", "normal"]:
            val = np.random.normal(p.get("mean", 0), p.get("std", 1))
            return int(round(val)) if is_int else val
        if dist_type == "poisson":
            val = np.random.poisson(p.get("lam", 1))
            if "scale" in p:
                val = float(val) * float(p["scale"])
            return int(round(val)) if is_int else val
    return p

def simulate_image_multi_scale(
    large_shape=(48, 48, 48),
    scale_range=(0.7, 1.4),
    n_levels_range=(3, 9),
    sigma_range={'type': 'poisson', 'lam': 0.1},
    multiply=None,
    interp_types=(0, 2),
    min_sim_shape=24
): # pragma: no cover
    """
    Generates a simulated tissue-like image at a random scale and orientation.
    Returns an ants.image of shape large_shape.

    Parameters
    ----------
    large_shape : tuple
        Target shape for the output image.
    scale_range : tuple, dict, or callable
        Control for the random scale factor applied to large_shape.
    n_levels_range : tuple, dict, or callable
        Control for the number of noise levels/layers.
    sigma_range : tuple, dict, or callable
        Control for the Gaussian smoothing sigma per level.
        Defaults to Poisson distribution with lambda=0.1.
    multiply : bool or None
        If True, layers are multiplied by their index. If None, randomized.
    interp_types : tuple
        Available interpolation types for resampling.
    min_sim_shape : int
        Minimum size for any dimension during simulation.
    """
    # 1. Randomize the simulation size to vary structure scale/zoom
    scale_factor = _sample_param(scale_range, (0.7, 1.4))
    sim_shape = [int(round(s * scale_factor)) for s in large_shape]
    sim_shape = [max(min_sim_shape, s) for s in sim_shape]
    
    n_levels = _sample_param(n_levels_range, (3, 9), is_int=True)
    if multiply is None:
        multiply = np.random.choice([True, False])
    
    img_np = np.zeros(sim_shape, dtype="float32")
    
    # 2. Generate multi-scale layers
    for k in range(n_levels):
        temp_np = np.random.normal(0, 1.0, size=sim_shape).astype("float32")
        temp = ants.from_numpy(temp_np)
        
        # Randomize sigma per level
        sigma = _sample_param(sigma_range, {'type': 'poisson', 'lam': 0.1})
        if sigma > 0:
            temp = ants.smooth_image(temp, sigma)
        
        # Otsu thresholding
        temp = ants.threshold_image(temp, "Otsu", 1)
        temp_np = temp.numpy()
        
        if multiply:
            temp_np = temp_np * (k + 1)
            
        img_np += temp_np
        
    # 3. Convert to ANTs image and resample to large_shape
    img = ants.from_numpy(img_np)
    interp = np.random.choice(interp_types) 
    img_large = ants.resample_image(img, large_shape, use_voxels=True, interp_type=interp)
    
    return img_large


def add_rician_noise(array, noise_std): # pragma: no cover
    """
    Applies Rician noise to a numpy array or ANTs image.
    """
    is_ants = hasattr(array, "numpy")
    arr = array.numpy() if is_ants else array
    n1 = np.random.normal(0, noise_std, arr.shape).astype(arr.dtype)
    n2 = np.random.normal(0, noise_std, arr.shape).astype(arr.dtype)
    noisy = np.sqrt((arr + n1)**2 + n2**2)
    noisy = np.clip(noisy, 0.0, 1.0)
    if is_ants:
        res = ants.image_clone(array)
        res[:] = noisy
        return res
    return noisy


def simulate_brain_procedural(shape, zoom_range=(0.7, 1.4), use_layer2=False): # pragma: no cover
    """
    Procedurally generates a 2D or 3D patch resembling brain anatomy (CSF, GM, WM, Ventricles)
    with stochastic folding and coordinate zoom.
    """
    ndim = len(shape)
    coords = [np.linspace(-1, 1, s) for s in shape]
    grid = np.meshgrid(*coords, indexing="ij")
    
    if use_layer2:
        s_axis = np.random.uniform(zoom_range[0], zoom_range[1], size=ndim)
        grid = [g * s_axis[i] for i, g in enumerate(grid)]
        if ndim == 2:
            shear_val = np.random.uniform(-0.12, 0.12)
            grid = [grid[0] + shear_val * grid[1], grid[1]]
            warp_amp = np.random.uniform(0.04, 0.10)
            warp_freq = np.random.uniform(3.5, 6.5)
            grid = [grid[0] + warp_amp * np.sin(warp_freq * grid[1]),
                    grid[1] + warp_amp * np.cos(warp_freq * grid[0])]
        else:
            sh_xy = np.random.uniform(-0.12, 0.12)
            sh_xz = np.random.uniform(-0.12, 0.12)
            sh_yz = np.random.uniform(-0.12, 0.12)
            grid = [grid[0] + sh_xy * grid[1] + sh_xz * grid[2],
                    grid[1] + sh_yz * grid[2],
                    grid[2]]
            warp_amp = np.random.uniform(0.04, 0.10)
            warp_freq = np.random.uniform(3.5, 6.5)
            grid = [grid[0] + warp_amp * np.sin(warp_freq * grid[1]) * np.cos(warp_freq * grid[2]),
                    grid[1] + warp_amp * np.cos(warp_freq * grid[0]) * np.sin(warp_freq * grid[2]),
                    grid[2] + warp_amp * np.sin(warp_freq * grid[0]) * np.cos(warp_freq * grid[1])]
    else:
        s = np.random.uniform(zoom_range[0], zoom_range[1])
        grid = [g * s for g in grid]
    
    if ndim == 3:
        X, Y, Z = grid[0], grid[1], grid[2]
        R = np.sqrt(X**2 + Y**2 + Z**2)
        
        freq = np.random.uniform(5.0, 9.0)
        amplitude = np.random.uniform(0.1, 0.2)
        folding = amplitude * np.sin(freq * X) * np.cos(freq * Y) * np.sin(freq * Z)
        modulated_R = R + folding
        
        seg = np.zeros(shape, dtype="float32")
        
        brain_mask = modulated_R < np.random.uniform(0.75, 0.85)
        wm_mask = modulated_R < np.random.uniform(0.5, 0.55)
        
        v_offset = 0.15 * (s_axis[0] if use_layer2 else s)
        v_radius = 0.12 * (s_axis[0] if use_layer2 else s)
        v1 = ((X - v_offset)**2 + (Y)**2 + (Z)**2) < v_radius**2
        v2 = ((X + v_offset)**2 + (Y)**2 + (Z)**2) < v_radius**2
        ventricles = v1 | v2
        
        seg[brain_mask] = 2
        seg[wm_mask] = 3
        seg[ventricles & wm_mask] = 1
        
        intensity_map = np.zeros(shape, dtype="float32")
        csf_val = np.random.uniform(0.1, 0.25)
        gm_val = np.random.uniform(0.5, 0.65)
        wm_val = np.random.uniform(0.8, 0.98)
        
        intensity_map[seg == 1] = csf_val
        intensity_map[seg == 2] = gm_val
        intensity_map[seg == 3] = wm_val
        
        texture = np.random.normal(0, 0.015, size=shape).astype("float32")
        intensity_map[seg > 0] += texture[seg > 0]
        if use_layer2:
            ripple = 0.03 * np.sin(20.0 * X) * np.cos(20.0 * Y) * np.sin(20.0 * Z)
            intensity_map[seg == 3] += ripple[seg == 3]
        intensity_map = np.clip(intensity_map, 0.0, 1.0)
        
        return ants.from_numpy(intensity_map)
    else:
        # 2D case
        X, Y = grid[0], grid[1]
        R = np.sqrt(X**2 + Y**2)
        
        freq = np.random.uniform(5.0, 9.0)
        amplitude = np.random.uniform(0.1, 0.2)
        folding = amplitude * np.sin(freq * X) * np.cos(freq * Y)
        modulated_R = R + folding
        
        seg = np.zeros(shape, dtype="float32")
        
        brain_mask = modulated_R < np.random.uniform(0.75, 0.85)
        wm_mask = modulated_R < np.random.uniform(0.5, 0.55)
        
        v_offset = 0.15 * (s_axis[0] if use_layer2 else s)
        v_radius = 0.12 * (s_axis[0] if use_layer2 else s)
        v1 = ((X - v_offset)**2 + (Y)**2) < v_radius**2
        v2 = ((X + v_offset)**2 + (Y)**2) < v_radius**2
        ventricles = v1 | v2
        
        seg[brain_mask] = 2
        seg[wm_mask] = 3
        seg[ventricles & wm_mask] = 1
        
        intensity_map = np.zeros(shape, dtype="float32")
        csf_val = np.random.uniform(0.1, 0.25)
        gm_val = np.random.uniform(0.5, 0.65)
        wm_val = np.random.uniform(0.8, 0.98)
        
        intensity_map[seg == 1] = csf_val
        intensity_map[seg == 2] = gm_val
        intensity_map[seg == 3] = wm_val
        
        texture = np.random.normal(0, 0.015, size=shape).astype("float32")
        intensity_map[seg > 0] += texture[seg > 0]
        if use_layer2:
            ripple = 0.03 * np.sin(20.0 * X) * np.cos(20.0 * Y)
            intensity_map[seg == 3] += ripple[seg == 3]
        intensity_map = np.clip(intensity_map, 0.0, 1.0)
        
        return ants.from_numpy(intensity_map)


def simulate_sinewave(shape, zoom_range=(0.7, 1.4), use_layer2=False): # pragma: no cover
    """
    Procedurally generates N-dimensional multi-frequency sinusoidal wave coordinates.
    """
    ndim = len(shape)
    coords = [np.linspace(-1, 1, s) for s in shape]
    grid = np.meshgrid(*coords, indexing="ij")
    
    if use_layer2:
        s_axis = np.random.uniform(zoom_range[0], zoom_range[1], size=ndim)
        grid = [g * s_axis[i] for i, g in enumerate(grid)]
        if ndim == 2:
            shear_val = np.random.uniform(-0.12, 0.12)
            grid = [grid[0] + shear_val * grid[1], grid[1]]
            warp_amp = np.random.uniform(0.04, 0.10)
            warp_freq = np.random.uniform(3.5, 6.5)
            grid = [grid[0] + warp_amp * np.sin(warp_freq * grid[1]),
                    grid[1] + warp_amp * np.cos(warp_freq * grid[0])]
        else:
            sh_xy = np.random.uniform(-0.12, 0.12)
            sh_xz = np.random.uniform(-0.12, 0.12)
            sh_yz = np.random.uniform(-0.12, 0.12)
            grid = [grid[0] + sh_xy * grid[1] + sh_xz * grid[2],
                    grid[1] + sh_yz * grid[2],
                    grid[2]]
            warp_amp = np.random.uniform(0.04, 0.10)
            warp_freq = np.random.uniform(3.5, 6.5)
            grid = [grid[0] + warp_amp * np.sin(warp_freq * grid[1]) * np.cos(warp_freq * grid[2]),
                    grid[1] + warp_amp * np.cos(warp_freq * grid[0]) * np.sin(warp_freq * grid[2]),
                    grid[2] + warp_amp * np.sin(warp_freq * grid[0]) * np.cos(warp_freq * grid[1])]
    else:
        s = np.random.uniform(zoom_range[0], zoom_range[1])
        grid = [g * s for g in grid]
    
    img_np = np.zeros(shape, dtype="float32")
    num_waves = np.random.randint(2, 5)
    for _ in range(num_waves):
        freqs = [np.random.uniform(2.0, 8.0) for _ in range(ndim)]
        phase = np.random.uniform(0, 2 * np.pi)
        amp = np.random.uniform(0.2, 0.5)
        
        wave_term = sum(f * g for f, g in zip(freqs, grid)) + phase
        img_np += amp * np.sin(wave_term)
        
    img_min, img_max = img_np.min(), img_np.max()
    if img_max > img_min:
        img_np = (img_np - img_min) / (img_max - img_min)
        
    return ants.from_numpy(img_np)


def simulate_layered(shape, zoom_range=(0.7, 1.4), use_layer2=False): # pragma: no cover
    """
    Procedurally generates N-dimensional rotated planar strip layers.
    """
    ndim = len(shape)
    coords = [np.linspace(-1, 1, s) for s in shape]
    grid = np.meshgrid(*coords, indexing="ij")
    
    if use_layer2:
        s_axis = np.random.uniform(zoom_range[0], zoom_range[1], size=ndim)
        grid = [g * s_axis[i] for i, g in enumerate(grid)]
        if ndim == 2:
            shear_val = np.random.uniform(-0.12, 0.12)
            grid = [grid[0] + shear_val * grid[1], grid[1]]
            warp_amp = np.random.uniform(0.04, 0.10)
            warp_freq = np.random.uniform(3.5, 6.5)
            grid = [grid[0] + warp_amp * np.sin(warp_freq * grid[1]),
                    grid[1] + warp_amp * np.cos(warp_freq * grid[0])]
        else:
            sh_xy = np.random.uniform(-0.12, 0.12)
            sh_xz = np.random.uniform(-0.12, 0.12)
            sh_yz = np.random.uniform(-0.12, 0.12)
            grid = [grid[0] + sh_xy * grid[1] + sh_xz * grid[2],
                    grid[1] + sh_yz * grid[2],
                    grid[2]]
            warp_amp = np.random.uniform(0.04, 0.10)
            warp_freq = np.random.uniform(3.5, 6.5)
            grid = [grid[0] + warp_amp * np.sin(warp_freq * grid[1]) * np.cos(warp_freq * grid[2]),
                    grid[1] + warp_amp * np.cos(warp_freq * grid[0]) * np.sin(warp_freq * grid[2]),
                    grid[2] + warp_amp * np.sin(warp_freq * grid[0]) * np.cos(warp_freq * grid[1])]
    else:
        s = np.random.uniform(zoom_range[0], zoom_range[1])
        grid = [g * s for g in grid]
    
    normal = np.random.normal(size=ndim)
    normal /= np.linalg.norm(normal)
    
    projection = sum(normal[i] * grid[i] for i in range(ndim))
    
    proj_min, proj_max = projection.min(), projection.max()
    num_layers = np.random.randint(4, 9)
    thresholds = np.sort(np.random.uniform(proj_min, proj_max, num_layers - 1))
    
    img_np = np.zeros(shape, dtype="float32")
    last_t = proj_min
    for i in range(num_layers):
        if i < num_layers - 1:
            t = thresholds[i]
            mask = (projection >= last_t) & (projection < t)
        else:
            mask = (projection >= last_t)
        
        intensity = np.random.uniform(0.1, 0.95)
        img_np[mask] = intensity
        last_t = t
        
    texture = np.random.normal(0, 0.015, size=shape).astype("float32")
    img_np += texture
    img_np = np.clip(img_np, 0.0, 1.0)
    
    return ants.from_numpy(img_np)


def optimize_upsampling_shape( # pragma: no cover
 spacing, modality='T1', roundit=False, verbose=False ):
    """
    Compute the optimal upsampling shape string (e.g., '2x2x2') based on image voxel spacing
    and imaging modality. This output is used to select an appropriate pretrained 
    super-resolution model filename.

    Parameters
    ----------
    spacing : sequence of float
        Voxel spacing (physical size per voxel in mm) from the input image.
        Typically obtained from `ants.get_spacing(image)`.

    modality : str, optional
        Imaging modality. Affects resolution thresholds:
        - 'T1' : anatomical MRI (default minimum spacing: 0.35 mm)
        - 'DTI' : diffusion MRI (default minimum spacing: 1.0 mm)
        - 'NM' : nuclear medicine (e.g., PET/SPECT, minimum spacing: 0.25 mm)

    roundit : bool, optional
        If True, uses rounded integer ratios for the upsampling shape.
        Otherwise, uses floor division with constraints.

    verbose : bool, optional
        If True, prints detailed internal values and logic.

    Returns
    -------
    str
        Optimal upsampling shape string in the form 'AxBxC',
        e.g., '2x2x2', '4x4x2'.

    Notes
    -----
    - The function prevents upsampling ratios that would result in '1x1x1'
      by defaulting to '2x2x2'.
    - It also avoids uncommon ratios like '5' by rounding to the nearest valid option.
    - The returned string is commonly used to populate a model filename template:
      
      Example:
          >>> bestup = optimize_upsampling_shape(ants.get_spacing(t1_img), modality='T1')
          >>> model = re.sub('bestup', bestup, 'siq_smallshort_train_bestup_1chan.keras')
    """
    minspc = min( list( spacing ) )
    maxspc = max( list( spacing ) )
    ratio = maxspc/minspc
    if ratio == 1.0:
        ratio = 0.5
    roundratio = np.round( ratio )
    tarshaperaw = []
    tarshape = []
    tarshaperound = []
    for k in range( len( spacing ) ):
        locrat = spacing[k]/minspc
        newspc = spacing[k] * roundratio
        tarshaperaw.append( locrat )
        if modality == "NM":
            if verbose:
                print("Using minspacing: 0.25")
            if newspc < 0.25 :
                locrat = spacing[k]/0.25
        elif modality == "DTI":
            if verbose:
                print("Using minspacing: 1.0")
            if newspc < 1.0 :
                locrat = spacing[k]/1.0
        else: # assume T1
            if verbose:
                print("Using minspacing: 0.35")
            if newspc < 0.35 :
                locrat = spacing[k]/0.35
        myint = int( locrat )
        if ( myint == 0 ):
            myint = 1
        if myint == 5:
            myint = 4
        if ( myint > 6 ):
            myint = 6
        tarshape.append( str( myint ) )
        tarshaperound.append( str( int(np.round( locrat )) ) )
    if verbose:
        print("before emendation:")
        print( tarshaperaw )
        print( tarshaperound )
        print( tarshape )
    allone = True
    if roundit:
        tarshape = tarshaperound
    for k in range( len( tarshape ) ):
        if tarshape[k] != "1":
            allone=False
    if allone:
        tarshape = ["2","2","2"] # default
    return "x".join(tarshape)

def compare_models( # pragma: no cover
 model_filenames, img, n_classes=3,
    poly_order='hist',
    identifier=None, noise_sd=0.1,verbose=False ):
    """
    Evaluates and compares the performance of multiple super-resolution models on a given image.

    This function provides a standardized way to benchmark SR models. For each model,
    it performs the following steps:
    1. Loads the model and determines its upsampling factor.
    2. Downsamples the high-resolution input image (`img`) to create a low-resolution
       input, simulating a real-world scenario.
    3. Adds Gaussian noise to the low-resolution input to test for robustness.
    4. Runs inference using the model to generate a super-resolved output.
    5. Generates a baseline output by upsampling the low-res input with linear interpolation.
    6. Calculates PSNR and SSIM metrics comparing both the model's output and the
       baseline against the original high-resolution image.
    7. If a dual-channel (image + segmentation) model is detected, it also calculates
       Dice scores for segmentation performance.
    8. Aggregates all results into a pandas DataFrame for easy comparison.

    Parameters
    ----------
    model_filenames : list of str
        A list of file paths to the Keras models (.h5, .keras) to be compared.

    img : ants.ANTsImage
        The high-resolution ground truth image. This image will be downsampled to
        create the input for the models.

    n_classes : int, optional
        The number of classes for Otsu's thresholding when auto-generating a
        segmentation for evaluating dual-channel models. Default is 3.

    poly_order : str or int, optional
        Method for intensity matching between the SR output and the reference.
        Options: 'hist' for histogram matching (default), an integer for
        polynomial regression, or None to disable.

    identifier : str, optional
        A custom identifier for the output DataFrame. If None, it is inferred
        from the model filename. Default is None.

    noise_sd : float, optional
        Standard deviation of the additive Gaussian noise applied to the
        downsampled image before inference. Default is 0.1.
        
    verbose : bool, optional
        If True, prints detailed progress and intermediate values. Default is False.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row corresponds to a model. Columns contain evaluation
        metrics (PSNR.SR, SSIM.SR, DICE.SR), baseline metrics (PSNR.LIN, SSIM.LIN,
        DICE.NN), and metadata.

    Notes
    -----
    When evaluating a 2-channel (segmentation) model, the primary metric for the
    segmentation task is the Dice score (`DICE.SR`). The intensity metrics (PSNR, SSIM)
    are still computed on the first channel.
    """
    padding=4
    mydf = pd.DataFrame()
    for k in range( len( model_filenames ) ):
        srmdl, upshape = read_srmodel( model_filenames[k] )
        if verbose:
            print( model_filenames[k] )
            print( upshape )
        tarshape = []
        inspc = ants.get_spacing(img)
        for j in range(len(img.shape)):
            tarshape.append( float(upshape[j]) * inspc[j] )
        # uses linear interp
        dimg=ants.resample_image( img, tarshape, use_voxels=False, interp_type=0 )
        dimg = ants.add_noise_to_image( dimg,'additivegaussian', [0,noise_sd] )
        import math
        dicesr=math.nan
        dicenn=math.nan
        if upshape[3] == 2:
            seghigh = ants.threshold_image( img,"Otsu",n_classes)
            seglow = ants.resample_image( seghigh, tarshape, use_voxels=False, interp_type=1 )
            dimgup=inference( dimg, srmdl, segmentation = seglow, poly_order=poly_order, verbose=verbose )
            dimgupseg = dimgup['super_resolution_segmentation']
            dimgup = dimgup['super_resolution']
            segblock = ants.resample_image_to_target( seghigh, dimgupseg, interp_type='nearestNeighbor'  )
            segimgnn = ants.resample_image_to_target( seglow, dimgupseg, interp_type='nearestNeighbor' )
            segblock[ dimgupseg == 0 ] = 0
            segimgnn[ dimgupseg == 0 ] = 0
            dicenn = ants.label_overlap_measures(segblock, segimgnn)['MeanOverlap'][0]
            dicesr = ants.label_overlap_measures(segblock, dimgupseg)['MeanOverlap'][0]
        else:
            dimgup=inference( dimg, srmdl, poly_order=poly_order, verbose=verbose )
        dimglin = ants.resample_image_to_target( dimg, dimgup, interp_type='linear' )
        imgblock = ants.resample_image_to_target( img, dimgup, interp_type='linear'  )
        dimgup[ imgblock == 0.0 ]=0.0
        dimglin[ imgblock == 0.0 ]=0.0
        padder = []
        dimwarning=False
        for jj in range(img.dimension):
            padder.append( padding )
            if img.shape[jj] != imgblock.shape[jj]:
                dimwarning=True
        if dimwarning:
            print("NOTE: dimensions of downsampled to upsampled image do not match!!!")
            print("we force them to match but this suggests results may not be reliable.")
        temp = os.path.basename( model_filenames[k] )
        temp = re.sub( "siq_default_sisr_", "", temp )
        temp = re.sub( "_best_mdl.keras", "", temp )
        temp = re.sub( "_best_mdl.h5", "", temp )
        if verbose and dimwarning:
            print( "original img shape" )
            print( img.shape )
            print( "resampled img shape" )
            print( imgblock.shape )
        a=[]
        imgshape = []
        for aa in range(len(upshape)):
            a.append( str(upshape[aa]) )
            if aa < len(imgblock.shape):
                imgshape.append( str( imgblock.shape[aa] ) )
        if identifier is None:
            identifier=temp
        mydict = {
            "identifier":identifier,
            "imgshape":"x".join(imgshape),
            "mdl": temp,
            "mdlshape":"x".join(a),
            "PSNR.LIN": antspynet.psnr( imgblock, dimglin ),
            "PSNR.SR": antspynet.psnr( imgblock, dimgup ),
            "SSIM.LIN": antspynet.ssim( imgblock, dimglin ),
            "SSIM.SR": antspynet.ssim( imgblock, dimgup ),
            "DICE.NN": dicenn,
            "DICE.SR": dicesr,
            "dimwarning": dimwarning }
        if verbose:
            print( mydict )
        temp = pd.DataFrame.from_records( [mydict], index=[0] )
        mydf = pd.concat( [mydf,temp], axis=0 )
        # end loop
    return mydf




def region_wise_super_resolution( # pragma: no cover
image, mask, super_res_model, dilation_amount=4, verbose=False):
    """
    Apply super-resolution model to each labeled region in the mask independently.

    Arguments
    ---------
    image : ANTsImage
        Input image.

    mask : ANTsImage
        Integer-labeled segmentation mask with non-zero regions to upsample.

    super_res_model : tf.keras.Model
        Trained super-resolution model.

    dilation_amount : int
        Number of morphological dilations applied to each label region before cropping.

    verbose : bool
        If True, print detailed status.

    Returns
    -------
    ANTsImage : Full-size super-resolved image with per-label inference and stitching.
    """
    import ants
    import numpy as np

    upFactor = []
    input_shape = super_res_model.inputs[0].shape
    test_shape = [1, 8, 8, 1] if len(input_shape) == 4 else [1, 8, 8, 8, 1]
    test_input = np.zeros(test_shape, dtype=np.float32)
    test_output = super_res_model(test_input)

    for k in range(len(test_shape) - 2):  # ignore batch + channel
        upFactor.append(int(test_output.shape[k + 1] / test_input.shape[k + 1]))

    original_size = mask.shape  # e.g., (x, y, z)
    new_size = tuple(int(s * f) for s, f in zip(original_size, upFactor))
    # Use exact SR spacing (use_voxels=False) so canvas aligns pixel-perfectly
    # with the model SR output. use_voxels=True gives spacing ≈ s*(N-1)/(N*f-1)
    # instead of s/f, causing sub-pixel blending misalignment.
    new_spacing = tuple(float(sp) / f for sp, f in zip(mask.spacing, upFactor))
    upsampled_mask = ants.resample_image(mask, new_spacing, use_voxels=False, interp_type=1)
    upsampled_image = ants.resample_image(image, new_spacing, use_voxels=False, interp_type=0)

    unique_labels = list(np.unique(upsampled_mask.numpy()))
    if 0 in unique_labels:
        unique_labels.remove(0)

    outimg = ants.image_clone(upsampled_image)

    for lab in unique_labels:
        if verbose:
            print(f"Processing label: {lab}")
        regionmask = ants.threshold_image(mask, lab, lab).iMath("MD", dilation_amount)
        cropped = ants.crop_image(image, regionmask)
        if cropped.shape[0] == 0:
            continue
        subimgsr = inference(
            cropped, super_res_model, verbose=verbose, align_phase=True
        )
        stitched = ants.decrop_image(subimgsr, outimg)
        outimg[upsampled_mask == lab] = stitched[upsampled_mask == lab]

    return outimg


def region_wise_super_resolution_blended( # pragma: no cover
image, mask, super_res_model, dilation_amount=4, verbose=False):
    """
    Apply super-resolution model to labeled regions with smooth blending to minimize stitching artifacts.

    This version uses a weighted-averaging scheme based on distance transforms
    to create seamless transitions between super-resolved regions and the background.

    Arguments
    ---------
    image : ANTsImage
        Input low-resolution image.

    mask : ANTsImage
        Integer-labeled segmentation mask.

    super_res_model : tf.keras.Model
        Trained super-resolution model.

    dilation_amount : int
        Number of morphological dilations applied to each label region before cropping.
        This provides context to the SR model.

    verbose : bool
        If True, print detailed status.

    Returns
    -------
    ANTsImage : Full-size, super-resolved image with seamless blending.
    """
    import ants
    import numpy as np
    epsilon32 = np.finfo(np.float32).eps
    normalize_weight_maps = True  # Default behavior to normalize weight maps
    # --- Step 1: Determine upsampling factor and prepare initial images ---
    upFactor = []
    input_shape = super_res_model.inputs[0].shape
    test_shape = [1, 8, 8, 1] if len(input_shape) == 4 else [1, 8, 8, 8, 1]
    test_input = np.zeros(test_shape, dtype=np.float32)
    test_output = super_res_model(test_input)
    for k in range(len(test_shape) - 2):
        upFactor.append(int(test_output.shape[k + 1] / test_input.shape[k + 1]))

    original_size = image.shape
    new_size = tuple(int(s * f) for s, f in zip(original_size, upFactor))
    # Use exact SR spacing so canvas aligns with model output (same fix as region_wise_super_resolution)
    new_spacing = tuple(float(sp) / f for sp, f in zip(image.spacing, upFactor))

    # The initial upsampled image will serve as our background
    background_sr_image = ants.resample_image(image, new_spacing, use_voxels=False, interp_type=0)

    # --- Step 2: Initialize accumulator and weight sum canvases ---
    # These must be float type for accumulation
    accumulator = ants.image_clone(background_sr_image).astype('float32') * 0.0
    weight_sum = ants.image_clone(accumulator)

    unique_labels = [l for l in np.unique(mask.numpy()) if l != 0]

    for lab in unique_labels:
        if verbose:
            print(f"Blending label: {lab}")

        # --- Step 3: Super-resolve a dilated patch (provides context to the model) ---
        region_mask_dilated = ants.threshold_image(mask, lab, lab).iMath("MD", dilation_amount)
        cropped_lowres = ants.crop_image(image, region_mask_dilated)
        if cropped_lowres.shape[0] == 0:
            continue
            
        # Apply the model to the cropped low-res patch
        sr_patch = inference(
            cropped_lowres, super_res_model, verbose=verbose, align_phase=True
        )
        
        # Place the super-resolved patch back onto a full-sized canvas
        sr_patch_full_size = ants.decrop_image(sr_patch, accumulator)

        # --- Step 4: Create a smooth weight map for this region ---
        # We use the *non-dilated* mask for the weight map to ensure a sharp focus on the target region.
        region_mask_original = ants.threshold_image(mask, lab, lab)
        
        # Resample the original region mask to the high-res grid
        weight_map = ants.resample_image(region_mask_original, new_spacing, use_voxels=False, interp_type=0)
        weight_map = ants.smooth_image(weight_map, sigma=2.0,
                                        sigma_in_physical_coordinates=False)
        if normalize_weight_maps:
            weight_map = ants.iMath(weight_map, "Normalize")
        # --- Step 5: Accumulate the weighted values and the weights themselves ---
        accumulator += sr_patch_full_size * weight_map
        weight_sum += weight_map

    # --- Step 6: Final Combination ---
    # Normalize the accumulator by the total weight at each pixel
    weight_sum_np = weight_sum.numpy()
    accumulator_np = accumulator.numpy()
    
    # Create a mask of pixels where blending occurred
    blended_mask = weight_sum_np > 0.0 # Use a small epsilon for float safety

    # Start with the original upsampled image as the base
    final_image_np = background_sr_image.numpy()
    
    # Perform the weighted average only where weights are non-zero
    final_image_np[blended_mask] = accumulator_np[blended_mask] / weight_sum_np[blended_mask]
    
    # Re-insert any non-blended background regions that were processed
    # This handles cases where regions overlap; the weighted average takes care of it.
    
    return ants.from_numpy(final_image_np, origin=background_sr_image.origin, 
                           spacing=background_sr_image.spacing, direction=background_sr_image.direction)
     


def gaussian_weight_map_numpy(shape, sigma=0.4):
    import numpy as np
    coords = [np.linspace(-1, 1, s) for s in shape]
    grids = np.meshgrid(*coords, indexing='ij')
    dist_sq = sum(g**2 for g in grids)
    weight = np.exp(-dist_sq / (2 * sigma**2))
    return weight


def _model_has_transposed_conv(model):
    """Returns True if the model (or any sub-layer) contains transposed convolution layers."""
    for l in getattr(model, "layers", []):
        cn = l.__class__.__name__
        if "Conv" in cn and "Transpose" in cn:
            return True
        if hasattr(l, "layers") and _model_has_transposed_conv(l):
            return True
    return False


def overlapping_patch_inference(
    image,
    model,
    target_range=(-127.5, 127.5),
    patch_size=(64, 64, 64),
    overlap=16,
    batch_size=1,
    verbose=False,
    volume_normalized=False,
    align_phase=True,
):
    """Overlapping patch inference with Gaussian-weighted blending.

    Parameters
    ----------
    volume_normalized : bool
        If True, the image is already in [0,1] from volume-level normalization
        (siq standard: TruncateIntensity + Normalize on the whole volume).
        Per-patch/per-batch renormalization is skipped; output is clipped [0,1].
        If False (legacy), the volume is rescaled to target_range before
        patching and rescaled back to input intensity range afterward.
    """
    import ants
    import numpy as np
    import time
    
    if target_range[0] > target_range[1]:
        target_range = target_range[::-1]

    # Validate image dim
    shape_length = len(model.inputs[0].shape)
    if shape_length == 5 and image.dimension != 3:
        raise ValueError("Expecting 3D input for this model.")
    elif shape_length == 4 and image.dimension != 2:
        raise ValueError("Expecting 2D input for this model.")

    if len(patch_size) != image.dimension:
        patch_size = tuple([patch_size[0]] * image.dimension)
    
    stride = tuple([p - overlap for p in patch_size])

    dim = image.dimension
    diag = [float(image.direction[i, i]) for i in range(dim)]
    flip_axes = tuple(i for i, d in enumerate(diag) if d < 0)

    image_array = image.numpy()
    if flip_axes:
        image_array = np.flip(image_array, axis=flip_axes).copy()
    if image.components == 1:
        image_array = np.expand_dims(image_array, axis=-1)

    if verbose:
        print(f"Image array shape: {image_array.shape}")

    # 1. Extract patches
    D, H, W, C = image_array.shape
    
    pad_d = (stride[0] - (D - patch_size[0]) % stride[0]) % stride[0]
    pad_h = (stride[1] - (H - patch_size[1]) % stride[1]) % stride[1]
    pad_w = (stride[2] - (W - patch_size[2]) % stride[2]) % stride[2]
    
    overlap_d = patch_size[0] - stride[0]
    overlap_h = patch_size[1] - stride[1]
    overlap_w = patch_size[2] - stride[2]
    
    total_pad = (
        (overlap_d//2, overlap_d//2 + pad_d),
        (overlap_h//2, overlap_h//2 + pad_h),
        (overlap_w//2, overlap_w//2 + pad_w),
        (0, 0)
    )
    
    padded = np.pad(image_array, total_pad, mode='edge')
    
    patches = []
    coords = []
    
    nD = (padded.shape[0] - patch_size[0]) // stride[0] + 1
    nH = (padded.shape[1] - patch_size[1]) // stride[1] + 1
    nW = (padded.shape[2] - patch_size[2]) // stride[2] + 1
    
    for d in range(nD):
        for h in range(nH):
            for w in range(nW):
                z = d * stride[0]
                y = h * stride[1]
                x = w * stride[2]
                patch = padded[z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2], :]
                patches.append(patch)
                coords.append((z, y, x))
                
    image_patches = np.stack(patches)
    padded_shape = padded.shape

    # Scale intensities
    if volume_normalized:
        # Image was already normalized to [0,1] at volume level before calling
        # this function — consistent with siq training normalization.
        # Do NOT renormalize per-patch or per-batch; pass through as-is.
        pass
    else:
        # Legacy path: linearly rescale volume to target_range before inference.
        img_min = image_patches.min()
        img_max = image_patches.max()
        if img_max > img_min:
            image_patches = (image_patches - img_min) / (img_max - img_min) * (target_range[1] - target_range[0]) + target_range[0]
        else:
            image_patches = image_patches - img_min + target_range[0]

    # 2. Batch Inference
    if verbose:
        print(f"Prediction (patch-wise overlapping): {len(image_patches)} patches")
        start_time = time.time()
        
    try:
        from tqdm import tqdm
        use_tqdm = verbose
    except ImportError:
        use_tqdm = False

    predictions = []
    num_batches = int(np.ceil(len(image_patches) / batch_size))
    
    if use_tqdm:
        batch_iter = tqdm(range(num_batches), desc="Inferring patches")
    else:
        batch_iter = range(num_batches)

    for b in batch_iter:
        batch_data = image_patches[b*batch_size : (b+1)*batch_size]
        pred = model.predict(batch_data, verbose=0)
        predictions.append(pred)

    prediction = np.concatenate(predictions, axis=0)

    if verbose:
        elapsed_time = time.time() - start_time
        if not use_tqdm:
            print(f"  (elapsed time: {elapsed_time:.3f}s)")
        print("Reconstruct intensities and blend")

    # Reconstruct intensities
    if volume_normalized:
        # Clip output to [0,1] — same range as normalized input.
        # Do NOT rescale back to original intensity; model output is calibrated
        # to [0,1] since that is what it was trained to produce.
        prediction = prediction.clip(0.0, 1.0)
    else:
        # Legacy path: rescale output back to original intensity range.
        intensity_range = image.range()
        pred_min = prediction.min()
        pred_max = prediction.max()
        if pred_max > pred_min:
            prediction = (prediction - pred_min) / (pred_max - pred_min) * (intensity_range[1] - intensity_range[0]) + intensity_range[0]
        else:
            prediction = prediction - pred_min + intensity_range[0]

    # Calculate expansion factor
    expansion_factor = np.asarray(prediction.shape[1:-1]) / np.asarray(image_patches.shape[1:-1])
    expansion_factor = tuple(expansion_factor.astype(int))

    scale_factor = expansion_factor[0]
    out_patch_size = tuple([p * e for p, e in zip(patch_size, expansion_factor)])
    
    # 3. Blending
    weight = gaussian_weight_map_numpy(out_patch_size)
    weight = np.expand_dims(weight, axis=-1)

    out_shape = (padded_shape[0]*expansion_factor[0], padded_shape[1]*expansion_factor[1], padded_shape[2]*expansion_factor[2], prediction.shape[-1])
    canvas = np.zeros(out_shape, dtype=np.float32)
    counts = np.zeros(out_shape, dtype=np.float32)

    for i, (z, y, x) in enumerate(coords):
        oz = z * expansion_factor[0]
        oy = y * expansion_factor[1]
        ox = x * expansion_factor[2]
        
        canvas[oz:oz+out_patch_size[0], oy:oy+out_patch_size[1], ox:ox+out_patch_size[2], :] += prediction[i] * weight
        counts[oz:oz+out_patch_size[0], oy:oy+out_patch_size[1], ox:ox+out_patch_size[2], :] += weight

    canvas /= (counts + 1e-8)

    # 4. Crop to original dimensions * scale
    crop_D = image_array.shape[0] * expansion_factor[0]
    crop_H = image_array.shape[1] * expansion_factor[1]
    crop_W = image_array.shape[2] * expansion_factor[2]

    z_start = total_pad[0][0] * expansion_factor[0]
    y_start = total_pad[1][0] * expansion_factor[1]
    x_start = total_pad[2][0] * expansion_factor[2]

    final_vol = canvas[z_start:z_start+crop_D, y_start:y_start+crop_H, x_start:x_start+crop_W, :]

    if flip_axes:
        final_vol = np.flip(final_vol, axis=flip_axes).copy()

    has_tc = _model_has_transposed_conv(model)
    shift_vec = []
    for d, f in enumerate(expansion_factor):
        if f <= 1:
            shift_vec.append(0.0)
        elif has_tc:
            shift_vec.append(-float(f - 1) / 2.0 if align_phase else 0.0)
        else:
            if d in flip_axes:
                shift_vec.append(-float(f - 1))
            else:
                shift_vec.append(0.0)

    if any(abs(s) > 1e-4 for s in shift_vec):
        from scipy.ndimage import shift as _nd_shift
        s_tuple = tuple(shift_vec)
        if final_vol.ndim == 4:
            for c in range(final_vol.shape[-1]):
                final_vol[..., c] = _nd_shift(final_vol[..., c], s_tuple, order=3, mode='nearest')
        else:
            final_vol = _nd_shift(final_vol, s_tuple, order=3, mode='nearest')

    if image.components == 1:
        final_vol = np.squeeze(final_vol, axis=-1)
        final_image = ants.from_numpy(final_vol)
    else:
        channels = []
        for i in range(image.components):
            channels.append(ants.from_numpy(final_vol[..., i]))
        final_image = ants.merge_channels(channels)

    # Re-apply spacing and direction
    new_spacing = tuple(np.asarray(image.spacing) / np.asarray(expansion_factor))
    ants.set_spacing(final_image, new_spacing)
    ants.set_direction(final_image, image.direction)
    ants.set_origin(final_image, image.origin)

    return final_image

def inference( # pragma: no cover
    image,
    mdl,
    config=None,
    truncation=None,
    segmentation=None,
    target_range=[1, 0],
    poly_order='hist',
    dilation_amount=0,
    method='antspynet',
    patch_size=(64, 64, 64),
    patch_overlap=16,
    batch_size=1,
    verbose=False,
    anti_checkerboard='auto',
    anti_checkerboard_sigma=None,
    align_phase=True,
    linear_blend=None):

    """
    Perform super-resolution inference on an input image, optionally guided by segmentation.

    This function uses a trained deep learning model to enhance the resolution of a medical image.
    It optionally applies label-wise inference if a segmentation mask is provided.

    Parameters
    ----------
    image : ants.ANTsImage
        Input image to be super-resolved.

    mdl : keras.Model
        Trained super-resolution model, typically from ANTsPyNet.

    config : dict, optional
        Provenance config dictionary loaded via ``siq.load_siq_model()``.
        Contains normalization parameters, patch sizes, and inference flags.

    anti_checkerboard : bool or str, optional
        If True or 'auto' (default), automatically measures the volume's high-frequency
        Nyquist spectral excess and applies the data-driven empirical smoothing sigma
        needed to eliminate the transposed-convolution checkerboard artifact.
        If False, no anti-checkerboard filtering is performed. Default is 'auto'.

    anti_checkerboard_sigma : float or str, optional
        Explicit standard deviation in voxels for the anti-checkerboard filter.
        If None or 'auto' (default), automatically calculated from the volume's
        empirical Nyquist spectral power via :func:`estimate_anti_checkerboard_sigma`.


    truncation : tuple or list of float, optional
        Percentile values (e.g., [0.01, 0.99]) for intensity truncation before model input.
        If None, no truncation is applied.

    segmentation : ants.ANTsImage, optional
        A labeled segmentation mask. If provided, super-resolution is performed per label
        using `region_wise_super_resolution` or `super_resolution_segmentation_per_label`.

    target_range : list of float
        Intensity range used for scaling the input before applying the model.
        Default is [1, 0] (internal default for `apply_super_resolution_model_to_image`).

    poly_order : int, str or None
        Determines how to match intensity between the super-resolved image and the original.
        Options:
          - 'hist' : use histogram matching
          - int >= 1 : perform polynomial regression of this order
          - None : no intensity adjustment

    dilation_amount : int
        Number of dilation steps applied to each segmentation label during
        region-based super-resolution (if segmentation is provided).

    method : str
        Method for inference: 'antspynet' (default, uses whole image) or 'patchwise' (uses overlapping patches to save memory).

    patch_size : tuple
        Size of patches for 'patchwise' method.

    patch_overlap : int
        Overlap between patches for 'patchwise' method.

    batch_size : int
        Batch size for 'patchwise' method.

    verbose : bool
        If True, print progress and status messages.

    Returns
    -------
    ANTsImage or dict
        - If `segmentation` is None, returns a single ANTsImage (super-resolved image).
        - If `segmentation` is provided, returns a dictionary with:
            - 'super_resolution': ANTsImage
            - other entries may include label-wise results or metadata.

    Examples
    --------
    >>> import ants
    >>> import antspynet
    >>> from siq import inference
    >>> img = ants.image_read("lowres.nii.gz")
    >>> model, cfg = siq.load_siq_model("mymodel.keras")
    >>> srimg = siq.inference(img, model, cfg, verbose=True)

    >>> seg = ants.image_read("mask.nii.gz")
    >>> sr_result = inference(img, model, cfg, segmentation=seg)
    >>> srimg = sr_result['super_resolution']
    """
    import ants
    import numpy as np
    import antspynet
    import antspyt1w
    from siq import region_wise_super_resolution
    import warnings

    # ── Provenance / config resolution ──────────────────────────────────────
    if config is None:
        warnings.warn(
            "siq.inference() called without a provenance config. "
            "This is deprecated and may produce incorrect results due to "
            "normalization mismatch. Use siq.load_siq_model() to obtain "
            "both the model and its config, then pass config here.",
            DeprecationWarning, stacklevel=2
        )
        # Fall back to reasonable defaults so old code doesn't break immediately
        _trunc_q = list(truncation) if truncation is not None else [0.001, 0.999]
        _patch_overlap = patch_overlap
        _model_patch = list(mdl.input_shape[1:-1])
        _output_clip = True
        _anti_cb = anti_checkerboard
        _anti_cb_sigma = anti_checkerboard_sigma
        _align_phase = align_phase
    else:
        _norm = config.get('normalization', {})
        _trunc_q = _norm.get('truncate_quantiles', [0.001, 0.999])
        _infer = config.get('inference', {})
        _patch_overlap = _infer.get('patch_overlap', patch_overlap)
        _model_patch = config.get('input_patch_shape', list(mdl.input_shape[1:]))[:-1]  # drop channel
        _output_clip = True  # always clip to [0,1] with siq models
        _anti_cb = _infer.get('anti_checkerboard', anti_checkerboard)
        _anti_cb_sigma = _infer.get('anti_checkerboard_sigma', anti_checkerboard_sigma)
        _align_phase = _infer.get('align_phase', align_phase)

    def apply_intensity_match(sr_image, reference_image, order, verbose=False):
        if order is None:
            return sr_image
        if verbose:
            print("Applying intensity match with", order)
        if order == 'hist':
            return ants.histogram_match_image(sr_image, reference_image)
        else:
            return ants.regression_match_image(sr_image, reference_image, poly_order=order)

    pimg = ants.image_clone(image)
    # Apply truncation from provenance config
    pimg = ants.iMath(pimg, 'TruncateIntensity', _trunc_q[0], _trunc_q[1])

    input_shape = mdl.inputs[0].shape
    num_channels = int(input_shape[-1])

    if segmentation is not None:
        if num_channels == 1:
            if verbose:
                print("Using region-wise super resolution due to single-channel model with segmentation.")
            sr = region_wise_super_resolution_blended(
                pimg, segmentation, mdl,
                dilation_amount=dilation_amount,
                verbose=verbose
            )
            ref = ants.resample_image_to_target(pimg, sr)
            return apply_intensity_match(sr, ref, poly_order, verbose)
        else:
            mynp = segmentation.numpy()
            mynp = list(np.unique(mynp)[1:len(mynp)].astype(int))
            upFactor = []
            if len(input_shape) == 5:
                testarr = np.zeros([1, 8, 8, 8, 2])
                testarrout = mdl(testarr)
                for k in range(3):
                    upFactor.append(int(testarrout.shape[k + 1] / testarr.shape[k + 1]))
            elif len(input_shape) == 4:
                testarr = np.zeros([1, 8, 8, 2])
                testarrout = mdl(testarr)
                for k in range(2):
                    upFactor.append(int(testarrout.shape[k + 1] / testarr.shape[k + 1]))
            
            # antspyt1w.super_resolution_segmentation_per_label expects the model's predict() 
            # method to return a list of predictions for each output when multi-channel. 
            # In Keras 3, we compile with a single concatenated output, but we wrap the model 
            # to split the concatenated output along the channel axis during inference.
            concat_output = mdl.output
            dimensionality = len(concat_output.shape) - 2
            split_outputs = ops.split(concat_output, 2, axis=dimensionality + 1)
            inference_model = Model(inputs=mdl.input, outputs=split_outputs)

            temp = antspyt1w.super_resolution_segmentation_per_label(
                pimg,
                segmentation,
                upFactor,
                inference_model,
                segmentation_numbers=mynp,
                target_range=target_range,
                dilation_amount=dilation_amount,
                poly_order=poly_order,
                max_lab_plus_one=True
            )
            imgsr = temp['super_resolution']
            ref = ants.resample_image_to_target(pimg, imgsr)
            temp['super_resolution'] = apply_intensity_match(imgsr, ref, poly_order, verbose)
            return temp

    # Default path: no segmentation
    # ── Step 1: Volume-level normalization (matches siq training pipeline) ──
    # Always normalize the whole volume to [0,1] BEFORE any patching.
    # This is consistent with training (TruncateIntensity + Normalize).
    # antspynet.apply_super_resolution_model_to_image() does per-patch
    # min-max renormalization which is INCOMPATIBLE with siq-trained models
    # and must NOT be used here.
    pimg_norm = ants.iMath(pimg, 'Normalize')   # → [0,1] volume-level

    # ── Step 2: Dispatch based on method ────────────────────────────────────
    # Fully convolutional networks (DBPN, etc.) process 3D volumes in a single pass
    # without patch stitching artifacts. Only use overlapping patches when
    # explicitly requested via method='patchwise'.
    input_shape_list = list(pimg_norm.shape)
    _upfactor = config.get('upsample_factor', 2) if config is not None else 2

    if method == 'patchwise':
        # Patchwise mode requested: Gaussian-blended overlapping patches.
        # Ensure patch size contains valid integers (not None from dynamic shapes).
        effective_patch = []
        for i, p in enumerate(_model_patch):
            if p is None:
                effective_patch.append(patch_size[i] if i < len(patch_size) else 64)
            else:
                effective_patch.append(int(p))
        effective_patch = tuple(effective_patch)

        if verbose:
            print(f"[siq] overlapping_patch_inference: Gaussian-blended patches {effective_patch} (overlap={_patch_overlap})")
        imgsr = overlapping_patch_inference(
            pimg_norm, mdl,
            patch_size=effective_patch,
            overlap=_patch_overlap,
            batch_size=batch_size,
            verbose=verbose,
            volume_normalized=True,   # skip per-patch renorm — volume already normalized
            align_phase=_align_phase,
        )
    else:
        # Default: direct full-volume inference in a single forward pass (no patch stitching)
        if verbose:
            if isinstance(_upfactor, (list, tuple)):
                print(f"[siq] Direct full-volume inference: shape {input_shape_list} -> {[f * s for f, s in zip(_upfactor, input_shape_list)]}")
            else:
                print(f"[siq] Direct full-volume inference: shape {input_shape_list} -> {[_upfactor * s for s in input_shape_list]}")
        # ── Axis-flip normalization (ANTs ↔ Keras/PyTorch convention) ────────────
        # ANTs stores data in physical (x, y[, z]) order; the direction matrix
        # diagonal encodes the sign of each physical axis relative to voxel index.
        #   direction[i,i] > 0  →  voxel dim i increases in the +physical direction
        #   direction[i,i] < 0  →  voxel dim i is FLIPPED (increases in − direction)
        #
        # Synthetic training data (from ants.from_numpy) always has direction=identity,
        # so all diagonals are +1.  Real MRI (e.g. LPS/RAS NIfTI) can have negative
        # diagonals (e.g. BLAST data direction ≈ [[-1,0],[0,-1]]).
        # Without correction the model receives a mirrored image relative to training.
        #
        # Fix: flip negative-diagonal axes → model sees identity-convention data →
        # flip the output back → SR is in the original ANTs coordinate frame.
        dim = pimg_norm.dimension
        diag = [float(pimg_norm.direction[i, i]) for i in range(dim)]
        flip_axes = tuple(i for i, d in enumerate(diag) if d < 0)

        arr_np = pimg_norm.numpy().astype('float32')
        if flip_axes:
            arr_np = np.flip(arr_np, axis=flip_axes).copy()

        arr = arr_np[np.newaxis, ..., np.newaxis]
        out = mdl.predict(arr, verbose=0)[0, ..., 0]

        if flip_axes:
            out = np.flip(out, axis=flip_axes).copy()

        # ── Architecture-aware phase alignment & reflection shift compensation ──
        # Transposed convolutions (AS-DBPN) have their receptive field center of mass
        # shifted by +(factor - 1)/2 voxels along upsampled axes, requiring a -(factor - 1)/2 shift.
        # Sub-pixel shuffle and nearest-neighbor back-projection (DBPN, L-DBPN, ESPCN)
        # do not produce deconvolution phase error. However, when an axis is flipped due to
        # negative direction cosines, discrete grid reflection reverses child voxels,
        # introducing a +(factor - 1) voxel offset that must be compensated with -(factor - 1).
        has_tc = _model_has_transposed_conv(mdl)
        if isinstance(_upfactor, (list, tuple)):
            f_list = list(_upfactor)
        else:
            f_list = [_upfactor] * dim

        shift_vec = []
        for d, f in enumerate(f_list):
            if f <= 1:
                shift_vec.append(0.0)
            elif has_tc:
                shift_vec.append(-float(f - 1) / 2.0 if _align_phase else 0.0)
            else:
                if d in flip_axes:
                    shift_vec.append(-float(f - 1))
                else:
                    shift_vec.append(0.0)

        if any(abs(s) > 1e-4 for s in shift_vec):
            from scipy.ndimage import shift as _nd_shift
            out = _nd_shift(out, tuple(shift_vec), order=3, mode='nearest')

        if _output_clip:
            out = np.clip(out, 0.0, 1.0)
        if isinstance(_upfactor, (list, tuple)):
            new_spacing = tuple(float(s) / f for s, f in zip(pimg_norm.spacing, _upfactor))
        else:
            new_spacing = tuple(float(s) / _upfactor for s in pimg_norm.spacing)
        imgsr = ants.from_numpy(out)
        ants.set_spacing(imgsr, new_spacing)
        ants.set_direction(imgsr, pimg_norm.direction)
        ants.set_origin(imgsr, pimg_norm.origin)


    # ── Step 3: Anti-Checkerboard Sub-Voxel Notch Filter ────────────────────
    if _anti_cb and _anti_cb not in [False, 0, 'none', 'false', 'False']:
        if _anti_cb_sigma is None or _anti_cb_sigma in ['auto', 'Auto']:
            # Empirically determine optimal sigma from the volume's Nyquist spectral excess
            effective_sigma = estimate_anti_checkerboard_sigma(imgsr.numpy())
            if verbose:
                print(f"[siq] Empirically calculated anti-checkerboard sigma={effective_sigma:.3f} (data-driven Nyquist attenuation)")
        else:
            effective_sigma = float(_anti_cb_sigma)

        if effective_sigma > 0.05:
            if verbose and (_anti_cb_sigma is not None and _anti_cb_sigma not in ['auto', 'Auto']):
                print(f"[siq] Applying anti-checkerboard sub-voxel notch filter (sigma={effective_sigma:.3f})")
            from scipy.ndimage import gaussian_filter
            imgsr_np = gaussian_filter(imgsr.numpy().astype(np.float32), sigma=effective_sigma)
            if _output_clip:
                imgsr_np = np.clip(imgsr_np, 0.0, 1.0)
            imgsr_cleaned = ants.from_numpy(imgsr_np)
            ants.copy_image_info(imgsr, imgsr_cleaned)
            imgsr = imgsr_cleaned
        elif verbose:
            print("[siq] No checkerboard artifact detected (excess <= 1.2x); zero anti-checkerboard filtering applied.")

    ref = ants.resample_image_to_target(pimg, imgsr)
    _linear_blend = linear_blend
    if _linear_blend is None and config is not None:
        _linear_blend = config.get('inference', {}).get('linear_blend', None)
        
    if _linear_blend is not None and _linear_blend > 0:
        ref_norm = ants.iMath(ref, "Normalize")
        sr_norm = ants.iMath(imgsr, "Normalize")
        fused_np = (1.0 - _linear_blend) * ref_norm.numpy() + _linear_blend * sr_norm.numpy()
        if _output_clip:
            fused_np = np.clip(fused_np, 0.0, 1.0)
        imgsr_fused = ants.from_numpy(fused_np.astype(np.float32))
        ants.copy_image_info(imgsr, imgsr_fused)
        imgsr = imgsr_fused

    return apply_intensity_match(imgsr, ref, poly_order, verbose)



def simulate_vessel_tubes(shape, zoom_range=(0.7, 1.4), use_layer2=False): # pragma: no cover
    """
    Procedurally generates tubular/vessel-like tree structures using Bezier curves
    and Euclidean distance fields.
    """
    import math
    ndim = len(shape)
    coords = [np.linspace(-1, 1, s) for s in shape]
    grid = np.meshgrid(*coords, indexing="ij")
    
    if use_layer2:
        s_axis = np.random.uniform(zoom_range[0], zoom_range[1], size=ndim)
        grid = [g * s_axis[i] for i, g in enumerate(grid)]
        if ndim == 2:
            warp_amp = np.random.uniform(0.04, 0.08)
            warp_freq = np.random.uniform(3.0, 5.0)
            grid = [grid[0] + warp_amp * np.sin(warp_freq * grid[1]),
                    grid[1] + warp_amp * np.cos(warp_freq * grid[0])]
        else:
            warp_amp = np.random.uniform(0.04, 0.08)
            warp_freq = np.random.uniform(3.0, 5.0)
            grid = [grid[0] + warp_amp * np.sin(warp_freq * grid[1]) * np.cos(warp_freq * grid[2]),
                    grid[1] + warp_amp * np.cos(warp_freq * grid[0]) * np.sin(warp_freq * grid[2]),
                    grid[2] + warp_amp * np.sin(warp_freq * grid[0]) * np.cos(warp_freq * grid[1])]
    else:
        s = np.random.uniform(zoom_range[0], zoom_range[1])
        grid = [g * s for g in grid]
        
    flat_grid = np.stack([g.ravel() for g in grid], axis=-1)
    
    num_pts = np.random.randint(4, 7)
    ctrl_pts = np.random.uniform(-0.8, 0.8, size=(num_pts, ndim))
    
    M = 200
    t = np.linspace(0, 1, M, dtype="float32")
    curve_pts = np.zeros((M, ndim), dtype="float32")
    for i in range(num_pts):
        coeff = float(math.comb(num_pts - 1, i)) * ((1 - t) ** (num_pts - 1 - i)) * (t ** i)
        for d in range(ndim):
            curve_pts[:, d] += coeff * ctrl_pts[i, d]
            
    base_radius = np.random.uniform(0.08, 0.16)
    from scipy.spatial import cKDTree
    tree = cKDTree(curve_pts)
    dists, _ = tree.query(flat_grid, distance_upper_bound=base_radius, workers=-1)
    mask = (dists < base_radius).reshape(shape)
    intensity = np.zeros(shape, dtype="float32")
    val = np.random.uniform(0.6, 0.9)
    intensity[mask] = val
    
    texture = np.random.normal(0, 0.015, size=shape).astype("float32")
    intensity[mask] += texture[mask]
    
    intensity = np.clip(intensity, 0.0, 1.0)
    return ants.from_numpy(intensity)


def simulate_cellular_voronoi(shape, zoom_range=(0.7, 1.4), use_layer2=False): # pragma: no cover
    """
    Procedurally generates honeycomb or cellular Voronoi tessellation meshes.
    """
    ndim = len(shape)
    coords = [np.linspace(-1, 1, s) for s in shape]
    grid = np.meshgrid(*coords, indexing="ij")
    
    if use_layer2:
        s_axis = np.random.uniform(zoom_range[0], zoom_range[1], size=ndim)
        grid = [g * s_axis[i] for i, g in enumerate(grid)]
        if ndim == 2:
            warp_amp = np.random.uniform(0.03, 0.06)
            warp_freq = np.random.uniform(4.0, 6.0)
            grid = [grid[0] + warp_amp * np.sin(warp_freq * grid[1]),
                    grid[1] + warp_amp * np.cos(warp_freq * grid[0])]
        else:
            warp_amp = np.random.uniform(0.03, 0.06)
            warp_freq = np.random.uniform(4.0, 6.0)
            grid = [grid[0] + warp_amp * np.sin(warp_freq * grid[1]) * np.cos(warp_freq * grid[2]),
                    grid[1] + warp_amp * np.cos(warp_freq * grid[0]) * np.sin(warp_freq * grid[2]),
                    grid[2] + warp_amp * np.sin(warp_freq * grid[0]) * np.cos(warp_freq * grid[1])]
    else:
        s = np.random.uniform(zoom_range[0], zoom_range[1])
        grid = [g * s for g in grid]
        
    flat_grid = np.stack([g.ravel() for g in grid], axis=-1)
    
    num_seeds = np.random.randint(15, 30)
    seeds = np.random.uniform(-1.0, 1.0, size=(num_seeds, ndim))
    
    grid_sq = np.sum(flat_grid**2, axis=-1, keepdims=True)
    seeds_sq = np.sum(seeds**2, axis=-1, keepdims=True).T
    cross = 2 * (flat_grid @ seeds.T)
    dists_sq = np.maximum(grid_sq - cross + seeds_sq, 0.0)
    dists = np.sqrt(dists_sq)
    
    sorted_dists = np.partition(dists, 1, axis=-1)
    d1 = sorted_dists[:, 0]
    d2 = sorted_dists[:, 1]
    
    diff = (d2 - d1).reshape(shape)
    thickness = np.random.uniform(0.02, 0.06)
    
    intensity = np.zeros(shape, dtype="float32")
    mask = diff < thickness
    
    val = np.random.uniform(0.7, 0.95)
    intensity[mask] = val
    
    nearest_idx = np.argmin(dists, axis=-1).reshape(shape)
    cell_greys = np.random.uniform(0.1, 0.4, size=num_seeds)
    bg_intensity = cell_greys[nearest_idx]
    
    intensity[~mask] = bg_intensity[~mask]
    
    texture = np.random.normal(0, 0.015, size=shape).astype("float32")
    intensity += texture
    intensity = np.clip(intensity, 0.0, 1.0)
    
    return ants.from_numpy(intensity)


def simulate_geometric_phantoms(shape, zoom_range=(0.7, 1.4), use_layer2=False): # pragma: no cover
    """
    Procedurally compiles geometric phantom circles, ellipsoids, and boxes.
    """
    ndim = len(shape)
    coords = [np.linspace(-1, 1, s) for s in shape]
    grid = np.meshgrid(*coords, indexing="ij")
    
    if use_layer2:
        s_axis = np.random.uniform(zoom_range[0], zoom_range[1], size=ndim)
        grid = [g * s_axis[i] for i, g in enumerate(grid)]
        if ndim == 2:
            warp_amp = np.random.uniform(0.04, 0.08)
            warp_freq = np.random.uniform(3.0, 5.0)
            grid = [grid[0] + warp_amp * np.sin(warp_freq * grid[1]),
                    grid[1] + warp_amp * np.cos(warp_freq * grid[0])]
        else:
            warp_amp = np.random.uniform(0.04, 0.08)
            warp_freq = np.random.uniform(3.0, 5.0)
            grid = [grid[0] + warp_amp * np.sin(warp_freq * grid[1]) * np.cos(warp_freq * grid[2]),
                    grid[1] + warp_amp * np.cos(warp_freq * grid[0]) * np.sin(warp_freq * grid[2]),
                    grid[2] + warp_amp * np.sin(warp_freq * grid[0]) * np.cos(warp_freq * grid[1])]
    else:
        s = np.random.uniform(zoom_range[0], zoom_range[1])
        grid = [g * s for g in grid]
        
    intensity = np.zeros(shape, dtype="float32")
    
    if ndim == 2:
        X, Y = grid[0], grid[1]
        mask_bg = (X**2 / 0.8**2 + Y**2 / 0.6**2) < 1.0
        intensity[mask_bg] = np.random.uniform(0.15, 0.3)
    else:
        X, Y, Z = grid[0], grid[1], grid[2]
        mask_bg = (X**2 / 0.8**2 + Y**2 / 0.6**2 + Z**2 / 0.6**2) < 1.0
        intensity[mask_bg] = np.random.uniform(0.15, 0.3)
        
    num_shapes = np.random.randint(4, 8)
    for _ in range(num_shapes):
        cx = np.random.uniform(-0.5, 0.5)
        cy = np.random.uniform(-0.5, 0.5)
        val = np.random.uniform(0.4, 0.9)
        
        rx = np.random.uniform(0.1, 0.25)
        ry = np.random.uniform(0.1, 0.25)
        
        shape_type = np.random.choice(["ellipse", "box"])
        if ndim == 2:
            if shape_type == "ellipse":
                mask = ((X - cx)**2 / rx**2 + (Y - cy)**2 / ry**2) < 1.0
            else:
                mask = (np.abs(X - cx) < rx) & (np.abs(Y - cy) < ry)
        else:
            cz = np.random.uniform(-0.5, 0.5)
            rz = np.random.uniform(0.1, 0.25)
            if shape_type == "ellipse":
                mask = ((X - cx)**2 / rx**2 + (Y - cy)**2 / ry**2 + (Z - cz)**2 / rz**2) < 1.0
            else:
                mask = (np.abs(X - cx) < rx) & (np.abs(Y - cy) < ry) & (np.abs(Z - cz) < rz)
                
        intensity[mask] = val
        
    texture = np.random.normal(0, 0.015, size=shape).astype("float32")
    intensity += texture
    intensity = np.clip(intensity, 0.0, 1.0)
    
    return ants.from_numpy(intensity)


def simulate_grid_patterns(shape, zoom_range=(0.7, 1.4), use_layer2=False): # pragma: no cover
    """
    Procedurally generates orthotropic grid lines or checkerboard patterns.
    """
    ndim = len(shape)
    coords = [np.linspace(-1, 1, s) for s in shape]
    grid = np.meshgrid(*coords, indexing="ij")
    
    if use_layer2:
        s_axis = np.random.uniform(zoom_range[0], zoom_range[1], size=ndim)
        grid = [g * s_axis[i] for i, g in enumerate(grid)]
        if ndim == 2:
            warp_amp = np.random.uniform(0.04, 0.08)
            warp_freq = np.random.uniform(3.0, 5.0)
            grid = [grid[0] + warp_amp * np.sin(warp_freq * grid[1]),
                    grid[1] + warp_amp * np.cos(warp_freq * grid[0])]
        else:
            warp_amp = np.random.uniform(0.04, 0.08)
            warp_freq = np.random.uniform(3.0, 5.0)
            grid = [grid[0] + warp_amp * np.sin(warp_freq * grid[1]) * np.cos(warp_freq * grid[2]),
                    grid[1] + warp_amp * np.cos(warp_freq * grid[0]) * np.sin(warp_freq * grid[2]),
                    grid[2] + warp_amp * np.sin(warp_freq * grid[0]) * np.cos(warp_freq * grid[1])]
    else:
        s = np.random.uniform(zoom_range[0], zoom_range[1])
        grid = [g * s for g in grid]
        
    pattern_type = np.random.choice(["checker", "lines"])
    freq = np.random.uniform(6.0, 12.0)
    img_np = np.zeros(shape, dtype="float32")
    
    if pattern_type == "checker":
        term = 1.0
        for g in grid:
            term *= np.sign(np.sin(freq * np.pi * g))
        img_np = 0.5 + 0.3 * term
    else:
        width = np.random.uniform(0.03, 0.08)
        term = np.zeros(shape, dtype="float32")
        for g in grid:
            line_mask = np.abs(np.sin(freq * np.pi * g)) < width
            term[line_mask] = 1.0
        img_np = 0.2 + 0.6 * term
        
    texture = np.random.normal(0, 0.015, size=shape).astype("float32")
    img_np += texture
    img_np = np.clip(img_np, 0.0, 1.0)
    
    return ants.from_numpy(img_np)


def simulate_fractal_noise(shape, zoom_range=(0.7, 1.4), use_layer2=False): # pragma: no cover
    """
    Procedurally compiles fractional Brownian motion (fBm) fractal noise.
    """
    ndim = len(shape)
    coords = [np.linspace(-1, 1, s) for s in shape]
    grid = np.meshgrid(*coords, indexing="ij")
    
    if use_layer2:
        s_axis = np.random.uniform(zoom_range[0], zoom_range[1], size=ndim)
        grid = [g * s_axis[i] for i, g in enumerate(grid)]
        if ndim == 2:
            warp_amp = np.random.uniform(0.04, 0.08)
            warp_freq = np.random.uniform(3.0, 5.0)
            grid = [grid[0] + warp_amp * np.sin(warp_freq * grid[1]),
                    grid[1] + warp_amp * np.cos(warp_freq * grid[0])]
        else:
            warp_amp = np.random.uniform(0.04, 0.08)
            warp_freq = np.random.uniform(3.0, 5.0)
            grid = [grid[0] + warp_amp * np.sin(warp_freq * grid[1]) * np.cos(warp_freq * grid[2]),
                    grid[1] + warp_amp * np.cos(warp_freq * grid[0]) * np.sin(warp_freq * grid[2]),
                    grid[2] + warp_amp * np.sin(warp_freq * grid[0]) * np.cos(warp_freq * grid[1])]
    else:
        s = np.random.uniform(zoom_range[0], zoom_range[1])
        grid = [g * s for g in grid]
        
    img_np = np.zeros(shape, dtype="float32")
    num_octaves = 4
    amplitude = 0.5
    frequency = np.random.uniform(2.0, 4.0)
    
    for _ in range(num_octaves):
        phase = np.random.uniform(0, 2 * np.pi)
        direction = np.random.normal(size=ndim)
        direction /= np.linalg.norm(direction)
        
        proj = sum(direction[i] * grid[i] for i in range(ndim))
        img_np += amplitude * np.sin(frequency * np.pi * proj + phase)
        
        amplitude *= 0.5
        frequency *= 2.0
        
    img_min, img_max = img_np.min(), img_np.max()
    if img_max > img_min:
        img_np = 0.1 + 0.8 * (img_np - img_min) / (img_max - img_min)
        
    texture = np.random.normal(0, 0.015, size=shape).astype("float32")
    img_np += texture
    img_np = np.clip(img_np, 0.0, 1.0)
    
    return ants.from_numpy(img_np)


def compute_phase_shift(y_true, y_pred, upsample_factor=100):
    """
    Computes the sub-pixel translational phase cross-correlation shift vector
    between ground truth and super-resolved prediction.

    Parameters
    ----------
    y_true : np.ndarray or ANTsImage
        Ground truth reference image array (2D or 3D).
    y_pred : np.ndarray or ANTsImage
        Super-resolution predicted image array (2D or 3D).
    upsample_factor : int, optional
        Sub-pixel registration precision factor. Default is 100 (0.01 voxel resolution).

    Returns
    -------
    tuple of float
        Shift vector (dy, dx[, dz]) in voxels such that y_pred ≈ shift(y_true, delta).
        Zero lag corresponds to (0.0, 0.0, 0.0).
    """
    from skimage.registration import phase_cross_correlation
    t_np = y_true.numpy() if hasattr(y_true, "numpy") else np.asarray(y_true, dtype=np.float32)
    p_np = y_pred.numpy() if hasattr(y_pred, "numpy") else np.asarray(y_pred, dtype=np.float32)

    shift, _, _ = phase_cross_correlation(t_np, p_np, upsample_factor=upsample_factor, normalization=None)
    return tuple(float(s) for s in shift)


def compute_shift_lk(y_true, y_pred, n_iter=3, trim=3, grad_percentile=50.0):
    """
    Gain/offset-robust sub-voxel shift estimate (voxels) via iterative Lucas-Kanade.

    Solves ``y_pred - y_true ≈ -delta · ∇y_true + a·y_true + b`` jointly for the
    shift vector ``delta`` and nuisance gain ``a`` / offset ``b`` by least squares
    over voxels with significant gradient. Unlike the Pearson edge correlation it is
    expressed in voxels, is insensitive to global contrast changes (a sharper or
    brighter model is not mistaken for a shifted one) and is jointly solved across
    axes (no cross-axis leakage). Returns the same registration convention as
    :func:`compute_phase_shift` (shift to apply to ``y_pred`` to align it with
    ``y_true``). Zero lag is ``(0.0, ..., 0.0)``.
    """
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
    """
    Measures the Pearson correlation between residual prediction error |y_pred - y_true|
    and true directional spatial gradients along each axis.

    Systematic sub-voxel phase shifts or uncompensated reflection shifts manifest as
    prominent halo boundary ridges in residual maps, causing high correlation (|r_d| > 0.20)
    with spatial gradients. Aligned, zero-lag outputs produce |r_d| < 0.02.

    Parameters
    ----------
    y_true : np.ndarray or ANTsImage
        Ground truth reference.
    y_pred : np.ndarray or ANTsImage
        Super-resolution prediction.

    Returns
    -------
    tuple of float
        Directional edge error correlation (|r_0|, |r_1|[, |r_2|]) along each spatial axis.
    """
    from scipy.ndimage import sobel
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
    """
    Standardized, high-throughput (<35ms overhead) Super-Resolution Spatial Alignment & Quality Control (QC) check.

    Evaluates:
      1. Sub-pixel phase cross-correlation lag vector (dy, dx[, dz])
      2. Directional edge error correlation (r_0, r_1[, r_2]) against true Sobel gradients
      3. Structural fidelity (PSNR, SSIM, GMSD, HFEN, CBI, CQS, PCS) vs bilinear baseline

    Calibrated, self-controlled shift estimate:
      Two independent estimators are used - FFT phase cross-correlation and a
      gain/offset-robust Lucas-Kanade fit (``compute_shift_lk``). When a bilinear
      reference (aligned by construction) is supplied, both are evaluated on it too and
      subtracted, which cancels estimator bias (patch-size/border effects). The gated
      per-axis shift is the smaller magnitude of the two relative estimates, so a
      FAIL requires both estimators to agree. Edge correlation is reported on the
      same relative basis. Calibration (injected known shifts on a real model output):
      edge correlation ~ 1.5-2 x shift (voxels) for small shifts, so 0.16 / 0.26
      correspond to ~0.08 / 0.15 voxel.

    Status Thresholds (voxels / unitless):
      - PASS: max |shift| < phase_thresh (0.08) and max edge corr < edge_corr_thresh (0.16)
      - WARN: max |shift| in [0.08, 0.15) or max edge corr in [0.16, 0.26)
      - FAIL: max |shift| >= phase_fail (0.15) or edge corr >= edge_corr_fail (0.26)
      PSNR vs bilinear is reported for information only and never fails or warns
      (PSNR rewards blur; see the perception-distortion tradeoff).

    Returns
    -------
    dict
        Comprehensive QC dictionary containing status, metrics, and formatted summary table.
    """
    t_np = y_true.numpy() if hasattr(y_true, "numpy") else np.asarray(y_true, dtype=np.float32)
    p_np = y_pred.numpy() if hasattr(y_pred, "numpy") else np.asarray(y_pred, dtype=np.float32)

    # 1. Sub-pixel shift estimators: FFT phase cross-correlation + Lucas-Kanade
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
    # Gate on agreement of both estimators (smaller magnitude per axis)
    shift_vec = np.where(np.abs(phase_rel) <= np.abs(lk_rel), phase_rel, lk_rel)
    max_phase_shift = float(np.max(np.abs(shift_vec)))
    edge_corr = tuple(float(v) for v in edge_rel)
    max_edge_corr = float(np.max(edge_rel))

    # 3. Standard metrics
    psnr_val = float(compute_psnr(t_np, p_np))
    ssim_val = float(compute_ssim(t_np, p_np))
    gmsd_val = float(compute_gmsd(t_np, p_np))
    cbi_val = float(compute_checkerboard_index(p_np, t_np, factor=factor))
    cqs_val = float(compute_cqs(t_np, p_np, factor=factor))
    pcs_val = float(compute_pcs(t_np, p_np, factor=factor))

    # 4. Bilinear comparison (if provided)
    b_psnr = None
    b_ssim = None
    delta_psnr = None
    if bilinear is not None:
        b_np = bilinear.numpy() if hasattr(bilinear, "numpy") else np.asarray(bilinear, dtype=np.float32)
        b_psnr = float(compute_psnr(t_np, b_np))
        b_ssim = float(compute_ssim(t_np, b_np))
        delta_psnr = psnr_val - b_psnr

    # 5. Determine QC status
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

    # Build ASCII formatted summary table
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




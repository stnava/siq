import os
import sys
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ants
import antspynet
import keras
from siq.get_data import compute_gmsd, compute_hfen


def save_orthogonal_slice_montage(img_arr, out_path, title=None, vmin=None, vmax=None, cmap="gray"):
    """
    Renders mid-axial (Z), mid-coronal (Y), and mid-sagittal (X) slices
    side-by-side into a single high-resolution PNG image.
    Applies histogram equalization for crisp tissue contrast per project rules.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if isinstance(img_arr, ants.ANTsImage):
        raw_arr = img_arr.numpy()
    elif isinstance(img_arr, np.ndarray):
        raw_arr = np.squeeze(img_arr)
    else:
        raw_arr = np.asarray(img_arr)
    # Mask out background air (< 0.01) before histogram equalization
    # to prevent tiny sub-voxel deconvolution ripples from being amplified into stark grid noise
    clamped_arr = np.where(raw_arr < 0.01, 0.0, raw_arr).astype(np.float32)
    try:
        temp_img = ants.from_numpy(clamped_arr)
        disp_img = ants.histogram_equalize_image(temp_img, number_of_histogram_bins=256)
        img_arr = disp_img.numpy()
    except Exception:
        img_arr = raw_arr

    img_arr = np.squeeze(img_arr)

    if img_arr.ndim == 2:
        # 2D image — show the single slice in all three panels
        h, w = img_arr.shape
        slice_z = slice_y = slice_x = np.rot90(img_arr)
        plane_labels = ["Image (XY)", "Image (XY)", "Image (XY)"]
    else:
        d, h, w = img_arr.shape
        slice_z = np.rot90(img_arr[:, :, w // 2])
        slice_y = np.rot90(img_arr[:, h // 2, :])
        slice_x = np.rot90(img_arr[d // 2, :, :])
        plane_labels = ["Axial (Z-plane)", "Coronal (Y-plane)", "Sagittal (X-plane)"]

    if vmin is None:
        vmin = np.percentile(img_arr, 1)
    if vmax is None:
        vmax = np.percentile(img_arr, 99)
        if vmax <= vmin:
            vmax = vmin + 1.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), facecolor="#0b0f19")

    axes[0].imshow(slice_z, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title(plane_labels[0], color="#e2e8f0", fontsize=11, fontweight="bold", pad=8)
    axes[0].axis("off")

    axes[1].imshow(slice_y, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title(plane_labels[1], color="#e2e8f0", fontsize=11, fontweight="bold", pad=8)
    axes[1].axis("off")

    axes[2].imshow(slice_x, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[2].set_title(plane_labels[2], color="#e2e8f0", fontsize=11, fontweight="bold", pad=8)
    axes[2].axis("off")

    if title:
        fig.suptitle(title, color="#f8fafc", fontsize=13, fontweight="bold", y=0.98)

    plt.subplots_adjust(wspace=0.04, hspace=0, left=0.01, right=0.99, bottom=0.02, top=0.90 if title else 0.96)
    plt.savefig(out_path, dpi=130, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_difference_montage(sr_arr, gt_arr, out_path, title=None, vmax=None):
    """
    Renders the absolute difference |SR - GT| across the three orthogonal planes
    using a thermal colormap (magma) with a colorbar to highlight residual errors.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if isinstance(sr_arr, ants.ANTsImage):
        sr_arr = sr_arr.numpy()
    if isinstance(gt_arr, ants.ANTsImage):
        gt_arr = gt_arr.numpy()
        
    diff_arr = np.abs(np.squeeze(sr_arr) - np.squeeze(gt_arr))

    if diff_arr.ndim == 2:
        slice_z = slice_y = slice_x = np.rot90(diff_arr)
        plane_labels = ["Error |SR - GT|", "Error |SR - GT|", "Error |SR - GT|"]
    else:
        d, h, w = diff_arr.shape
        slice_z = np.rot90(diff_arr[:, :, w // 2])
        slice_y = np.rot90(diff_arr[:, h // 2, :])
        slice_x = np.rot90(diff_arr[d // 2, :, :])
        plane_labels = ["Axial Error |SR - GT|", "Coronal Error |SR - GT|", "Sagittal Error |SR - GT|"]

    if vmax is None:
        vmax = max(1.0, float(np.percentile(diff_arr, 99.5)))

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), facecolor="#0b0f19")

    im0 = axes[0].imshow(slice_z, cmap="magma", vmin=0, vmax=vmax)
    axes[0].set_title(plane_labels[0], color="#f87171", fontsize=11, fontweight="bold", pad=8)
    axes[0].axis("off")

    im1 = axes[1].imshow(slice_y, cmap="magma", vmin=0, vmax=vmax)
    axes[1].set_title(plane_labels[1], color="#f87171", fontsize=11, fontweight="bold", pad=8)
    axes[1].axis("off")

    im2 = axes[2].imshow(slice_x, cmap="magma", vmin=0, vmax=vmax)
    axes[2].set_title(plane_labels[2], color="#f87171", fontsize=11, fontweight="bold", pad=8)
    axes[2].axis("off")

    cbar_ax = fig.add_axes([0.92, 0.12, 0.015, 0.72])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.ax.yaxis.set_tick_params(color="#94a3b8")
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#94a3b8', fontsize=9)
    cbar.set_label("Voxel Absolute Error", color="#94a3b8", fontsize=10, labelpad=8)

    if title:
        fig.suptitle(title, color="#f8fafc", fontsize=13, fontweight="bold", y=0.98)

    plt.subplots_adjust(wspace=0.04, hspace=0, left=0.01, right=0.90, bottom=0.02, top=0.90 if title else 0.96)
    plt.savefig(out_path, dpi=130, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_4way_comparison_montage(orig_arr, down_arr, bilin_arr, sr_arr, out_path, title="Identical-Scale 4-Way Comparison: Original vs Downsampled vs Linear vs SR"):
    """
    Renders an identical-scale 4-row comparative montage:
    Row 1: Original Image (Ground Truth HR)
    Row 2: Downsampled Image (Low-Resolution Input)
    Row 3: Linear Upsampled Image (Bilinear)
    Row 4: SR Upsampled Image (3D AS-DBPN Reconstructed)
    Across Axial, Coronal, and Sagittal orthogonal cross-sections.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    def _to_eq(a):
        if isinstance(a, ants.ANTsImage):
            arr = a.numpy()
        elif isinstance(a, np.ndarray):
            arr = np.squeeze(a)
        else:
            return a
        # Mask out background air (< 0.01) before histogram equalization
        clean_arr = np.where(arr < 0.01, 0.0, arr).astype(np.float32)
        try:
            temp = ants.from_numpy(clean_arr)
            return ants.histogram_equalize_image(temp, number_of_histogram_bins=256).numpy()
        except Exception:
            return clean_arr

    orig_arr = _to_eq(orig_arr)
    down_arr = _to_eq(down_arr)
    bilin_arr = _to_eq(bilin_arr)
    sr_arr = _to_eq(sr_arr)

    vmin = float(np.percentile(orig_arr, 1))
    vmax = float(np.percentile(orig_arr, 99))
    if vmax <= vmin:
        vmax = vmin + 1.0

    d_shape = orig_arr.shape
    is_2d = (orig_arr.ndim == 2)

    if is_2d:
        h2, w2 = d_shape
        def _slices(a):
            s = np.rot90(a)
            return s, s, s
        col_titles = ["Image (XY)", "Image (XY)", "Image (XY)"]
        row_label_sr = "4. SR Upsampled Image (2D AS-DBPN)"
    else:
        d, h2, w2 = d_shape
        def _slices(a):
            return np.rot90(a[:, :, w2 // 2]), np.rot90(a[:, h2 // 2, :]), np.rot90(a[d // 2, :, :])
        col_titles = ["Axial (Z-plane)", "Coronal (Y-plane)", "Sagittal (X-plane)"]
        row_label_sr = "4. SR Upsampled Image (3D AS-DBPN)"

    fig, axes = plt.subplots(4, 3, figsize=(12, 15.5), facecolor="#0b0f19")
    rows_data = [
        ("1. Original Image (Ground Truth HR)", orig_arr, "#38bdf8"),
        ("2. Downsampled Image (LR Input - Same FOV)", down_arr, "#f59e0b"),
        ("3. Linear Upsampled Image (Bilinear)", bilin_arr, "#a855f7"),
        (row_label_sr, sr_arr, "#10b981")
    ]

    for row_idx, (row_label, arr, col) in enumerate(rows_data):
        sz, sy, sx = _slices(arr)
        axes[row_idx, 0].imshow(sz, cmap="gray", vmin=vmin, vmax=vmax)
        axes[row_idx, 0].set_ylabel(row_label, color=col, fontsize=11, fontweight="bold", labelpad=10)
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])
        for spine in axes[row_idx, 0].spines.values():
            spine.set_color("#334155")
            spine.set_linewidth(1.5)

        axes[row_idx, 1].imshow(sy, cmap="gray", vmin=vmin, vmax=vmax)
        axes[row_idx, 1].axis("off")

        axes[row_idx, 2].imshow(sx, cmap="gray", vmin=vmin, vmax=vmax)
        axes[row_idx, 2].axis("off")

        if row_idx == 0:
            axes[0, 0].set_title(col_titles[0], color="#e2e8f0", fontsize=12, fontweight="bold", pad=8)
            axes[0, 1].set_title(col_titles[1], color="#e2e8f0", fontsize=12, fontweight="bold", pad=8)
            axes[0, 2].set_title(col_titles[2], color="#e2e8f0", fontsize=12, fontweight="bold", pad=8)

    if title:
        fig.suptitle(title, color="#f8fafc", fontsize=14, fontweight="bold", y=0.99)
    plt.subplots_adjust(wspace=0.04, hspace=0.06, left=0.18, right=0.99, bottom=0.01, top=0.96)
    plt.savefig(out_path, dpi=130, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def generate_svg_chart(x_vals, y_vals, title, y_label, baseline_val=None, baseline_label=None, color="#3b82f6", height=220, width=540, stage_markers=None, x_label="Step"):
    """
    Generates a standalone, dependency-free SVG chart for training metric trajectories.
    Supports stage demarcation lines and non-overlapping cumulative progressions.
    """
    if not x_vals or not y_vals or len(x_vals) == 0:
        return f'<div style="color: #64748b; padding: 20px;">No data recorded yet for {title}.</div>'
    
    pad_left = 65
    pad_right = 30
    pad_top = 35
    pad_bottom = 40
    
    w = width
    h = height
    plot_w = w - pad_left - pad_right
    plot_h = h - pad_top - pad_bottom
    
    min_x = min(x_vals)
    max_x = max(x_vals)
    if min_x == max_x:
        max_x = min_x + 1
        
    all_y = list(y_vals)
    if baseline_val is not None:
        all_y.append(baseline_val)
    min_y = min(all_y)
    max_y = max(all_y)
    
    # Add 8% padding to Y range
    range_y = max(1e-6, max_y - min_y)
    min_y -= range_y * 0.08
    max_y += range_y * 0.08
    range_y = max_y - min_y
    
    def map_x(x):
        return pad_left + ((x - min_x) / (max_x - min_x)) * plot_w
        
    def map_y(y):
        return pad_top + plot_h - ((y - min_y) / range_y) * plot_h
        
    svg = f'<svg viewBox="0 0 {w} {h}" class="chart-svg" style="width: 100%; height: auto; display: block;">\n'
    # Background rect
    svg += f'  <rect width="{w}" height="{h}" fill="#0f172a" rx="8" />\n'
    
    # Title
    svg += f'  <text x="{pad_left}" y="22" fill="#e2e8f0" font-family="Outfit, sans-serif" font-size="13" font-weight="600">{title}</text>\n'
    
    # Grid lines (4 horizontal)
    for i in range(5):
        val = min_y + (i / 4.0) * range_y
        y_pos = map_y(val)
        svg += f'  <line x1="{pad_left}" y1="{y_pos:.1f}" x2="{w - pad_right}" y2="{y_pos:.1f}" stroke="rgba(255,255,255,0.06)" stroke-width="1" />\n'
        svg += f'  <text x="{pad_left - 8}" y="{y_pos + 4:.1f}" fill="#64748b" font-family="monospace" font-size="10" text-anchor="end">{val:.2f}</text>\n'
        
    # Baseline line if present
    if baseline_val is not None:
        b_y = map_y(baseline_val)
        svg += f'  <line x1="{pad_left}" y1="{b_y:.1f}" x2="{w - pad_right}" y2="{b_y:.1f}" stroke="#f59e0b" stroke-width="1.5" stroke-dasharray="4,3" opacity="0.85" />\n'
        svg += f'  <text x="{w - pad_right}" y="{b_y - 5:.1f}" fill="#f59e0b" font-family="sans-serif" font-size="10" font-weight="600" text-anchor="end">{baseline_label or "Baseline"}: {baseline_val:.2f}</text>\n'
        
    # Stage boundary vertical dividers
    if stage_markers:
        for bx, b_name in stage_markers:
            if min_x < bx < max_x:
                mx = map_x(bx)
                svg += f'  <line x1="{mx:.1f}" y1="{pad_top}" x2="{mx:.1f}" y2="{pad_top + plot_h}" stroke="#64748b" stroke-width="1.2" stroke-dasharray="3,3" opacity="0.65" />\n'
                svg += f'  <text x="{mx + 4:.1f}" y="{pad_top + 14}" fill="#94a3b8" font-family="Outfit, sans-serif" font-size="9" font-weight="600">{b_name}</text>\n'

    # Points and line
    pts = [f"{map_x(x):.1f},{map_y(y):.1f}" for x, y in zip(x_vals, y_vals)]
    svg += f'  <polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" />\n'
    
    # Draw points
    for x, y in zip(x_vals, y_vals):
        cx = map_x(x)
        cy = map_y(y)
        svg += f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="3.5" fill="#0f172a" stroke="{color}" stroke-width="2" />\n'
        
    # Latest value pill
    latest_x = map_x(x_vals[-1])
    latest_y = map_y(y_vals[-1])
    svg += f'  <circle cx="{latest_x:.1f}" cy="{latest_y:.1f}" r="5.5" fill="{color}" />\n'
    svg += f'  <text x="{pad_left + plot_w}" y="22" fill="{color}" font-family="monospace" font-size="12" font-weight="bold" text-anchor="end">Current: {y_vals[-1]:.2f}</text>\n'
    
    # X axis labels
    svg += f'  <text x="{pad_left}" y="{h - 12}" fill="#64748b" font-family="monospace" font-size="10">{x_label} {min_x}</text>\n'
    svg += f'  <text x="{pad_left + plot_w}" y="{h - 12}" fill="#64748b" font-family="monospace" font-size="10" text-anchor="end">{x_label} {max_x}</text>\n'
    
    svg += '</svg>\n'
    return svg


class VisualConvergenceReporter:
    """
    Manages regular convergence checkpointing, metric tracking, orthogonal slice
    montage generation, and live HTML report updating during 3D AS-DBPN training.
    """
    def __init__(self, workspace_dir=".", checkpoint_dir="checkpoints/asdbpn_3d", report_dir="reports/asdbpn_3d", html_filename="asdbpn_3d_report.html", reset_history=False, selection_metric="cqs"):
        self.workspace_dir = os.path.abspath(workspace_dir)
        self.checkpoint_dir = os.path.join(self.workspace_dir, checkpoint_dir)
        self.report_dir = os.path.join(self.workspace_dir, report_dir)
        self.html_path = os.path.join(self.workspace_dir, html_filename)
        self.selection_metric = selection_metric
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.report_dir, exist_ok=True)
        
        self.csv_path = os.path.join(self.checkpoint_dir, "convergence_history.csv")
        self.history = []
        self.best_score = -float('inf')
        self.best_cqs = -float('inf')
        self.best_psnr = -1.0
        self.best_iter = 0
        self.best_stage = None
        self.start_time = time.time()
        
        if reset_history and os.path.exists(self.csv_path):
            try:
                os.remove(self.csv_path)
                print(f"[Convergence Reporter] Reset history requested: cleared {self.csv_path}")
            except Exception as e:
                print(f"[Convergence Reporter] Warning: Could not remove old history: {e}")
        
        # Load existing history if available
        if os.path.exists(self.csv_path):
            try:
                df = pd.read_csv(self.csv_path)
                self.history = df.to_dict("records")
                if len(self.history) > 0:
                    for r in self.history:
                        psnr = float(r.get("val_psnr", 0.0))
                        ssim = float(r.get("val_ssim", 0.0))
                        gmsd = float(r.get("val_gmsd", 0.0))
                        cbi = float(r.get("val_cbi", 0.0))
                        cqs = float(r.get("val_cqs", ssim - gmsd - cbi))
                        if psnr > self.best_psnr:
                            self.best_psnr = psnr
                        if int(r.get("is_best", 0)) == 1 or cqs > self.best_cqs:
                            self.best_cqs = cqs
                            self.best_score = cqs if self.selection_metric == "cqs" else psnr
                            self.best_iter = int(r.get("iteration", 0))
                            self.best_stage = str(r.get("stage", ""))
                print(f"[Convergence Reporter] Loaded {len(self.history)} existing history entries. Best CQS: {self.best_cqs:.4f} (PSNR: {self.best_psnr:.2f} dB) at iter {self.best_iter}")
            except Exception as e:
                print(f"[Convergence Reporter] Warning: Could not read existing convergence history: {e}")
                
        # Cache for validation patches
        self.lr_patch = None
        self.hr_patch = None
        self.gt_np = None
        self.bilinear_metrics = {}
        self.ldbpn_metrics = {}

    def setup_validation_patches(self, lr_patch, hr_patch, factor=None):
        """
        Initializes validation patches and computes static baselines (Ground Truth, Bilinear, LDBPN).
        """
        self.lr_patch = lr_patch
        self.hr_patch = hr_patch
        self.gt_np = hr_patch.numpy()
        self.factor = factor
        
        # 1. Ground Truth (Original Image) Orthogonal Montage
        gt_path = os.path.join(self.report_dir, "val3d_ground_truth.png")
        orig_path = os.path.join(self.report_dir, "val3d_original.png")
        hr_shape_str = "x".join(str(s) for s in self.hr_patch.shape)
        save_orthogonal_slice_montage(self.hr_patch, gt_path, title=f"1. Original Image (Ground Truth High Resolution - {hr_shape_str})")
        save_orthogonal_slice_montage(self.hr_patch, orig_path, title=f"1. Original Image (Ground Truth High Resolution - {hr_shape_str})")
        
        # 2. Downsampled Image (LR Input displayed at identical scale via nearest-neighbor)
        lr_temp = ants.image_clone(lr_patch)
        hr_temp = ants.image_clone(hr_patch)
        if factor is not None and isinstance(factor, (list, tuple)):
            lr_temp.set_spacing([hr_temp.spacing[i] * factor[i] for i in range(len(factor))])
        else:
            f = factor if factor is not None else 2.0
            lr_temp.set_spacing([s * f for s in hr_temp.spacing])
        hr_temp.set_spacing(hr_temp.spacing)
        self.downsampled_img = ants.resample_image_to_target(lr_temp, hr_temp, interp_type=1)
        down_path = os.path.join(self.report_dir, "val3d_downsampled.png")
        f_str = f"{factor}" if factor is not None else "2x"
        save_orthogonal_slice_montage(self.downsampled_img, down_path, title=f"2. Downsampled Image (LR Input - {f_str} Downsampled, Identical Scale)")

        # 3. Bilinear Baseline (Linear Upsampled Image)
        bilinear_sr = ants.resample_image_to_target(lr_temp, hr_temp, interp_type=0)
        self.bilinear_sr = bilinear_sr
        bilinear_np = bilinear_sr.numpy()
        
        lin_psnr = float(antspynet.psnr(hr_temp, bilinear_sr))
        lin_ssim = float(antspynet.ssim(hr_temp, bilinear_sr))
        lin_gmsd = float(compute_gmsd(self.gt_np, bilinear_np))
        lin_hfen = float(compute_hfen(self.gt_np, bilinear_np))
        lin_corr = float(np.corrcoef(bilinear_np.flatten(), self.gt_np.flatten())[0, 1])
        
        self.bilinear_metrics = {
            "psnr": lin_psnr,
            "ssim": lin_ssim,
            "gmsd": lin_gmsd,
            "hfen": lin_hfen,
            "corr": lin_corr
        }
        
        bilinear_path = os.path.join(self.report_dir, "val3d_bilinear.png")
        lin_path = os.path.join(self.report_dir, "val3d_linear_upsampled.png")
        save_orthogonal_slice_montage(bilinear_sr, bilinear_path, title=f"3. Linear Upsampled Image (Bilinear Baseline - {lin_psnr:.2f} dB, SSIM: {lin_ssim:.4f})")
        save_orthogonal_slice_montage(bilinear_sr, lin_path, title=f"3. Linear Upsampled Image (Bilinear Baseline - {lin_psnr:.2f} dB, SSIM: {lin_ssim:.4f})")
        
        diff_bilinear_path = os.path.join(self.report_dir, "diff3d_bilinear.png")
        save_difference_montage(bilinear_np, self.gt_np, diff_bilinear_path, title="Bilinear Residual Error |Bilinear - Ground Truth|")
        
        # 3. LDBPN 3D Baseline (if available in workspace)
        ldbpn_path_keras = os.path.join(self.workspace_dir, "ldbpn_3d_refined.keras")
        if os.path.exists(ldbpn_path_keras):
            try:
                import siq
                ldbpn_model = keras.models.load_model(ldbpn_path_keras, custom_objects={"PixelShuffle3D": siq.PixelShuffle3D}, compile=False)
                ldbpn_sr = siq.inference(self.lr_patch, ldbpn_model, method="antspynet", verbose=False)
                ants.copy_image_info(self.hr_patch, ldbpn_sr)
                ldbpn_np = ldbpn_sr.numpy()
                
                ld_psnr = float(antspynet.psnr(self.hr_patch, ldbpn_sr))
                ld_ssim = float(antspynet.ssim(self.hr_patch, ldbpn_sr))
                ld_gmsd = float(compute_gmsd(self.gt_np, ldbpn_np))
                ld_hfen = float(compute_hfen(self.gt_np, ldbpn_np))
                ld_corr = float(np.corrcoef(ldbpn_np.flatten(), self.gt_np.flatten())[0, 1])
                
                self.ldbpn_metrics = {
                    "psnr": ld_psnr,
                    "ssim": ld_ssim,
                    "gmsd": ld_gmsd,
                    "hfen": ld_hfen,
                    "corr": ld_corr
                }
                
                ldbpn_img_path = os.path.join(self.report_dir, "val3d_ldbpn.png")
                save_orthogonal_slice_montage(ldbpn_sr, ldbpn_img_path, title=f"L-DBPN 3D Baseline (PSNR: {ld_psnr:.2f} dB, SSIM: {ld_ssim:.4f})")
                
                diff_ldbpn_path = os.path.join(self.report_dir, "diff3d_ldbpn.png")
                save_difference_montage(ldbpn_np, self.gt_np, diff_ldbpn_path, title="L-DBPN 3D Residual Error |L-DBPN - Ground Truth|")
                print(f"[Convergence Reporter] Evaluated L-DBPN 3D baseline: PSNR: {ld_psnr:.2f} dB, SSIM: {ld_ssim:.4f}")
            except Exception as e:
                print(f"[Convergence Reporter] Could not evaluate LDBPN 3D baseline: {e}")
                
        print(f"[Convergence Reporter] Validation baselines initialized: Bilinear PSNR={lin_psnr:.2f} dB, SSIM={lin_ssim:.4f}")

    def record_checkpoint(self, model, iteration, stage_name, train_loss, is_convergence_step=True, loss_weights=None):
        """
        Evaluates the model on the validation patch, writes checkpoint files,
        renders visualization slices, logs metrics to CSV, and generates the updated HTML report.
        """
        import siq
        t0 = time.time()
        
        # 1. Run inference with provenance config
        cfg = getattr(self, "model_config", None)
        if cfg is None:
            cfg = siq.default_siq_config(model)
        # Use raw model output for validation: no intensity matching (poly_order=None),
        # no anti-checkerboard Gaussian blur (anti_checkerboard=False).
        # Both post-processing steps smear edges and cause apparent sub-voxel shift
        # in the comparison montage, and contaminate PSNR/SSIM/CBI training metrics.
        sr_img = siq.inference(self.lr_patch, model, config=cfg, verbose=False,
                               poly_order=None, anti_checkerboard=False)
        ants.copy_image_info(self.hr_patch, sr_img)
        sr_np = sr_img.numpy()
        
        val_psnr = float(antspynet.psnr(self.hr_patch, sr_img))
        val_ssim = float(antspynet.ssim(self.hr_patch, sr_img))
        val_gmsd = float(compute_gmsd(self.gt_np, sr_np))
        val_hfen = float(compute_hfen(self.gt_np, sr_np))
        val_corr = float(np.corrcoef(sr_np.flatten(), self.gt_np.flatten())[0, 1])
        val_cbi = float(siq.compute_checkerboard_index(sr_np, self.gt_np, factor=getattr(self, "factor", None)))
        val_cqs = float(val_ssim - val_gmsd - val_cbi)
        
        # Determine champion model.
        # Stages are prioritized hierarchically so downstream perceptual refinement (Stage 3)
        # is never blocked by an artificially high PSNR from an earlier L1/MSE stage (Stage 2).
        stage_rank_map = {
            "Warmup Gate": 0, "Initial Warmup": 0, "Warmup": 0,
            "Stage 1 Adaptation": 1, "Stage 1": 1,
            "Stage 2 Robustness": 2, "Stage 2": 2,
            "Stage 3 Refinement": 3, "Stage 3 Joint Fine-Tuning": 3, "Stage 3": 3,
        }
        cur_rank = stage_rank_map.get(stage_name, 0)
        best_rank = stage_rank_map.get(self.best_stage, -1) if self.best_stage is not None else -1

        is_new_best = False
        if cur_rank > best_rank:
            is_new_best = True
        elif cur_rank == best_rank:
            if self.selection_metric == "cqs":
                is_new_best = val_cqs > self.best_score
            elif self.selection_metric == "psnr":
                is_new_best = val_psnr > self.best_psnr
            else:
                is_new_best = val_cqs > self.best_score
        else:
            is_new_best = False

        if is_new_best:
            self.best_score = val_cqs if self.selection_metric == "cqs" else val_psnr
            self.best_cqs = val_cqs
            self.best_iter = iteration
            self.best_stage = stage_name

        is_new_peak_psnr = val_psnr > self.best_psnr
        if is_new_peak_psnr:
            self.best_psnr = val_psnr
            
        elapsed_eval = time.time() - t0
        
        # 2. Checkpoint filenames
        ckpt_filename = f"asdbpn_3d_step_{iteration:04d}.keras"
        ckpt_path = os.path.join(self.checkpoint_dir, ckpt_filename)

        # Build provenance config — saved alongside every .keras file so
        # any reader knows exactly how to run inference correctly.
        import json, datetime
        try:
            import siq as _siq
            _siq_ver = getattr(_siq, '__version__', 'unknown')
        except Exception:
            _siq_ver = 'unknown'
        in_s = list(model.input_shape[1:])
        out_s = list(model.output_shape[1:])
        _input_patch  = [64, 64, 64, in_s[-1] if in_s[-1] is not None else 1] if any(s is None for s in in_s[:-1]) else in_s
        _output_patch = [128, 128, 128, out_s[-1] if out_s[-1] is not None else 1] if any(s is None for s in out_s[:-1]) else out_s
        try:
            if getattr(self, "factor", None) is not None:
                _upfactor = list(self.factor) if isinstance(self.factor, (list, tuple)) else self.factor
            elif model.output_shape[1] is not None and model.input_shape[1] is not None:
                _upfactor = int(round(model.output_shape[1] / model.input_shape[1]))
            else:
                _upfactor = 2
        except Exception:
            _upfactor = 2
        provenance_config = {
            "model_type": "asdbpn_3d",
            "siq_version": _siq_ver,
            "saved_at": datetime.datetime.utcnow().isoformat() + "Z",
            "stage": stage_name,
            "iteration": iteration,
            "input_patch_shape": _input_patch,
            "output_patch_shape": _output_patch,
            "upsample_factor": _upfactor,
            "normalization": {
                "method": "volume",
                "truncate_quantiles": [0.001, 0.999],
                "output_range": [0.0, 1.0],
                "note": (
                    "Apply ants.iMath(vol,'TruncateIntensity',0.001,0.999) then "
                    "ants.iMath(vol,'Normalize') to the WHOLE volume before any patching. "
                    "Do NOT normalize per-patch — that is inconsistent with training."
                )
            },
            "inference": {
                "preferred_method": "direct_single_patch_if_fits",
                "patch_overlap": 16,
                "output_clip": [0.0, 1.0],
                "antspynet_wrapper": False,
                "note": (
                    "Use siq.inference(img, model) which dispatches: "
                    "single model.predict() if input==patch_size, else "
                    "Gaussian-blended overlapping patches with volume_normalized=True. "
                    "Never use antspynet.apply_super_resolution_model_to_image() — "
                    "it applies per-patch min-max renormalization incompatible with siq training."
                )
            },
            "val_metrics": {
                "val_psnr": float(val_psnr),
                "val_ssim": float(val_ssim),
                "val_gmsd": float(val_gmsd),
                "val_cbi": float(val_cbi),
                "val_cqs": float(val_cqs),
                "val_hfen": float(val_hfen),
            }
        }
        if loss_weights is not None:
            provenance_config["loss_weights"] = {k: float(v) for k, v in loss_weights.items()}
            try:
                import pandas as pd
                wts_row = {
                    "msq": float(loss_weights.get("msq", 0.0)),
                    "feat": float(loss_weights.get("feat", 0.0)),
                    "tv": float(loss_weights.get("tv", 0.0)),
                    "l1": float(loss_weights.get("l1", 0.0)),
                    "edge": float(loss_weights.get("edge", 0.0)),
                    "gms": float(loss_weights.get("gms", 0.0)),
                    "cbi": float(loss_weights.get("cbi", 0.0)),
                }
                wts_df = pd.DataFrame([wts_row])
                f_s = f"{_upfactor[0]}x{_upfactor[1]}x{_upfactor[2]}" if isinstance(_upfactor, (list, tuple)) else f"{_upfactor}x{_upfactor}x{_upfactor}"
                wts_fn = "asdbpn_3d_refined_training_weights.csv" if f_s == "2x2x2" else f"asdbpn_3d_{f_s}_refined_training_weights.csv"
                wts_df.to_csv(os.path.join(self.workspace_dir, wts_fn), index=False)
                wts_df.to_csv(os.path.join(self.checkpoint_dir, wts_fn), index=False)
            except Exception:
                pass

        def _save_config(keras_path, cfg):
            cfg_path = keras_path.replace('.keras', '_config.json')
            with open(cfg_path, 'w') as _f:
                json.dump(cfg, _f, indent=2)

        if is_convergence_step:
            model.save(ckpt_path)
            _save_config(ckpt_path, provenance_config)
            
        # Always maintain latest refined model at repo root for generate_summary_images.py
        if isinstance(_upfactor, (list, tuple)):
            f_s = "x".join(str(f) for f in _upfactor)
        else:
            f_s = f"{_upfactor}x{_upfactor}x{_upfactor}"
        dim_str = "2d" if (isinstance(_upfactor, (list, tuple)) and len(_upfactor) == 2) else "3d"
        refined_fn = f"asdbpn_{dim_str}_{f_s}_refined.keras"
        best_fn = f"asdbpn_{dim_str}_{f_s}_best_mdl.keras"
        # Legacy alias for default 3D 2x2x2
        if f_s == "2x2x2":
            refined_fn = "asdbpn_3d_refined.keras"
            best_fn = "asdbpn_3d_best_mdl.keras"
        refined_root_path = os.path.join(self.workspace_dir, refined_fn)
        model.save(refined_root_path)
        _save_config(refined_root_path, provenance_config)
        
        if is_new_best:
            best_cqs_ckpt = os.path.join(self.checkpoint_dir, "asdbpn_3d_best_cqs.keras")
            model.save(best_cqs_ckpt)
            _save_config(best_cqs_ckpt, provenance_config)
            best_root_path = os.path.join(self.workspace_dir, best_fn)
            model.save(best_root_path)
            _save_config(best_root_path, provenance_config)
            # Immutable archive: copy to model_archive/ + register in model_registry.json
            try:
                import siq as _siq
                factor_str = "x".join(str(f) for f in provenance_config.get("upsample_factor", []))
                _siq.archive_model(
                    best_cqs_ckpt,
                    provenance_config,
                    tag=f"v{provenance_config.get('siq_version', '?')}_{factor_str}_cqs_champion",
                    notes=f"Auto-archived at iter {iteration} by VisualConvergenceReporter",
                    verbose=True,
                )
            except Exception as _ae:
                print(f"[archive] WARNING: could not archive champion: {_ae}")

        if is_new_peak_psnr:
            best_psnr_ckpt = os.path.join(self.checkpoint_dir, "asdbpn_3d_best_psnr.keras")
            model.save(best_psnr_ckpt)
            _save_config(best_psnr_ckpt, provenance_config)
            
        # 3. Render Visual Images
        step_img_name = f"step_{iteration:04d}_ortho.png"
        step_img_path = os.path.join(self.report_dir, step_img_name)
        save_orthogonal_slice_montage(
            sr_img, step_img_path,
            title=f"3D AS-DBPN Step {iteration} [{stage_name}] (PSNR: {val_psnr:.2f} dB, SSIM: {val_ssim:.4f})"
        )
        
        diff_img_name = f"step_{iteration:04d}_diff.png"
        diff_img_path = os.path.join(self.report_dir, diff_img_name)
        save_difference_montage(
            sr_np, self.gt_np, diff_img_path,
            title=f"Residual Error |AS-DBPN (Iter {iteration}) - GT|"
        )
        
        # Maintain current and best pointers in reports
        current_img_path = os.path.join(self.report_dir, "val3d_asdbpn_current.png")
        save_orthogonal_slice_montage(sr_img, current_img_path, title=f"Latest AS-DBPN Output (Iter {iteration})")
        
        current_diff_path = os.path.join(self.report_dir, "diff3d_asdbpn_current.png")
        save_difference_montage(sr_np, self.gt_np, current_diff_path, title=f"Latest Residual Error (Iter {iteration})")
        
        # Maintain SR Upsampled image pointer (identical scale)
        sr_upsampled_path = os.path.join(self.report_dir, "val3d_sr_upsampled.png")
        save_orthogonal_slice_montage(sr_img, sr_upsampled_path, title=f"4. SR Upsampled Image (3D AS-DBPN Step {iteration} - PSNR: {val_psnr:.2f} dB, SSIM: {val_ssim:.4f})")
        
        # Maintain 4-Way Unified Comparative Montage
        if hasattr(self, "downsampled_img") and hasattr(self, "bilinear_sr") and self.downsampled_img is not None and self.bilinear_sr is not None:
            comp_path = os.path.join(self.report_dir, "val3d_4way_comparison.png")
            save_4way_comparison_montage(self.hr_patch, self.downsampled_img, self.bilinear_sr, sr_img, comp_path, title=f"Identical-Scale 4-Way Comparison (Step {iteration} - PSNR: {val_psnr:.2f} dB, SSIM: {val_ssim:.4f})")
            
        if is_new_best:
            best_img_path = os.path.join(self.report_dir, "val3d_asdbpn_best.png")
            save_orthogonal_slice_montage(sr_img, best_img_path, title=f"Best AS-DBPN Output (Iter {iteration}, CQS: {val_cqs:.4f}, PSNR: {val_psnr:.2f} dB)")
            best_diff_path = os.path.join(self.report_dir, "diff3d_asdbpn_best.png")
            save_difference_montage(sr_np, self.gt_np, best_diff_path, title=f"Best Residual Error (Iter {iteration}, CQS: {val_cqs:.4f})")
            
        # 4. Append to CSV
        timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = {
            "iteration": iteration,
            "stage": stage_name,
            "train_loss": float(train_loss) if train_loss is not None else 0.0,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim,
            "val_gmsd": val_gmsd,
            "val_cbi": val_cbi,
            "val_cqs": val_cqs,
            "val_hfen": val_hfen,
            "val_corr": val_corr,
            "is_best": 1 if is_new_best else 0,
            "checkpoint_file": ckpt_filename if is_convergence_step else "asdbpn_3d_refined.keras",
            "ortho_image": step_img_name,
            "diff_image": diff_img_name,
            "timestamp": timestamp_str
        }
        self.history.append(entry)
        
        # Write to CSV
        df = pd.DataFrame(self.history)
        df.to_csv(self.csv_path, index=False)
        
        # 5. Build HTML report
        self.render_html_report(latest_entry=entry, is_new_best=is_new_best)
        
        psnr_delta = val_psnr - self.bilinear_metrics.get("psnr", 27.10)
        sign = "+" if psnr_delta >= 0 else ""
        print(f"\n[Convergence Checkpoint] Iter {iteration:04d} ({stage_name}) - Loss: {train_loss:.6f}")
        print(f"  --> PSNR: {val_psnr:.2f} dB ({sign}{psnr_delta:.2f} dB vs Bilinear) | SSIM: {val_ssim:.4f} | GMSD: {val_gmsd:.4f} | CBI: {val_cbi:.4f} | CQS: {val_cqs:.4f} | HFEN: {val_hfen:.4f} | Corr: {val_corr:.4f}")
        if is_new_best:
            print(f"  ★ NEW PEAK VALIDATION CQS! ({val_cqs:.4f}) [PSNR: {val_psnr:.2f} dB, SSIM: {val_ssim:.4f}, CBI: {val_cbi:.4f}] -> Saved champion model checkpoints")
        elif is_new_peak_psnr:
            print(f"  ★ New peak PSNR ({val_psnr:.2f} dB) -> Saved asdbpn_3d_best_psnr.keras")
        print(f"  --> Convergence report refreshed: {self.html_path} (eval took {elapsed_eval:.2f}s)\n")
        
        return entry

    def render_html_report(self, latest_entry=None, is_new_best=False):
        """
        Compiles the visual HTML dashboard with interactive viewports, SVG charts, and convergence history.
        """
        try:
            from tests.render_convergence_dashboard import render_html_dashboard_from_csv
            if render_html_dashboard_from_csv(
                csv_path=self.csv_path,
                html_path=self.html_path,
                bilinear_metrics=self.bilinear_metrics,
                ldbpn_metrics=self.ldbpn_metrics,
                report_dir=self.report_dir,
                checkpoint_dir=self.checkpoint_dir,
                factor=getattr(self, "factor", None)
            ):
                return True
        except Exception:
            pass

        if latest_entry is None and len(self.history) > 0:
            latest_entry = self.history[-1]
            
        cur_iter = latest_entry["iteration"] if latest_entry else 0
        cur_stage = latest_entry["stage"] if latest_entry else "Initializing"
        cur_psnr = latest_entry["val_psnr"] if latest_entry else 0.0
        cur_ssim = latest_entry["val_ssim"] if latest_entry else 0.0
        cur_gmsd = latest_entry["val_gmsd"] if latest_entry else 0.0
        cur_hfen = latest_entry["val_hfen"] if latest_entry else 0.0
        cur_corr = latest_entry["val_corr"] if latest_entry else 0.0
        cur_loss = latest_entry["train_loss"] if latest_entry else 0.0
        
        lin_psnr = self.bilinear_metrics.get("psnr", 27.10)
        lin_ssim = self.bilinear_metrics.get("ssim", 0.9285)
        lin_hfen = self.bilinear_metrics.get("hfen", 0.4500)
        
        psnr_gain = cur_psnr - lin_psnr
        gain_sign = "+" if psnr_gain >= 0 else ""
        gain_color = "#10b981" if psnr_gain >= 0 else "#f59e0b"
        
        # Compute monotonic cumulative global steps and extract stage transition markers
        warmup_max_iter = 0
        for r in self.history:
            st = str(r.get("stage", ""))
            if "Warmup" in st or "Initial" in st:
                warmup_max_iter = max(warmup_max_iter, int(r.get("iteration", 0)))
        if warmup_max_iter == 0:
            warmup_max_iter = 150

        global_steps = []
        stage_markers = []
        prev_stage = None
        for r in self.history:
            st = str(r.get("stage", "Unknown"))
            it = int(r.get("iteration", 0))
            if "Warmup" in st or "Initial" in st:
                g_step = it
            else:
                g_step = warmup_max_iter + it
            if global_steps and g_step <= global_steps[-1]:
                g_step = global_steps[-1] + 1
            global_steps.append(g_step)
            if prev_stage is not None and st != prev_stage:
                short_stage = st.replace("Phase", "").replace("Joint Fine-Tuning", "Stage 3").strip()
                stage_markers.append((g_step, short_stage))
            prev_stage = st
            
        psnrs = [float(r["val_psnr"]) for r in self.history]
        ssims = [float(r["val_ssim"]) for r in self.history]
        losses = [float(r["train_loss"]) for r in self.history]
        
        svg_psnr = generate_svg_chart(global_steps, psnrs, "Validation PSNR Trajectory (dB)", "PSNR (dB)", baseline_val=lin_psnr, baseline_label="Bilinear", color="#10b981", stage_markers=stage_markers, x_label="Global Step")
        svg_ssim = generate_svg_chart(global_steps, ssims, "Validation SSIM Progression", "SSIM", baseline_val=lin_ssim, baseline_label="Bilinear", color="#8b5cf6", stage_markers=stage_markers, x_label="Global Step")
        svg_loss = generate_svg_chart(global_steps, losses, "Hybrid Training Loss", "Loss", color="#38bdf8", stage_markers=stage_markers, x_label="Global Step")
        
        # Build checkpoint rows (newest first)
        checkpoint_rows = ""
        for idx, r in enumerate(reversed(self.history)):
            orig_idx = len(self.history) - 1 - idx
            g_step = global_steps[orig_idx]
            best_tag = ' <span style="background: rgba(16,185,129,0.2); color: #10b981; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem;">★ Best</span>' if r.get("is_best", 0) else ""
            p_val = float(r["val_psnr"])
            p_diff = p_val - lin_psnr
            p_sign = "+" if p_diff >= 0 else ""
            p_color = "#10b981" if p_diff >= 0 else "#e2e8f0"
            
            html_dir = os.path.dirname(os.path.abspath(self.html_path))
            rep_rel = os.path.relpath(self.report_dir, html_dir)
            ckpt_rel_dir = os.path.relpath(self.checkpoint_dir, html_dir)
            ortho_rel = f"{rep_rel}/{r.get('ortho_image', '')}"
            diff_rel = f"{rep_rel}/{r.get('diff_image', '')}"
            ckpt_rel = f"{ckpt_rel_dir}/{r.get('checkpoint_file', '')}"
            
            checkpoint_rows += f"""
            <tr>
                <td><strong>Step {g_step}</strong> <span style="font-size: 0.78rem; color: #64748b;">(Iter {r['iteration']})</span>{best_tag}</td>
                <td><span class="stage-badge">{r['stage']}</span></td>
                <td style="color: {p_color}; font-weight: bold;">{p_val:.2f} dB <span style="font-size: 0.8em; opacity: 0.8;">({p_sign}{p_diff:.2f})</span></td>
                <td>{float(r['val_ssim']):.4f}</td>
                <td>{float(r['val_hfen']):.4f}</td>
                <td>{float(r['val_gmsd']):.4f}</td>
                <td>{float(r.get('val_cbi', 0.0)):.4f}</td>
                <td>{float(r['val_corr']):.4f}</td>
                <td style="font-family: monospace;">{float(r['train_loss']):.5f}</td>
                <td>
                    <a href="{ortho_rel}" target="_blank" style="color: #38bdf8; text-decoration: none; margin-right: 8px;">📷 Ortho</a>
                    <a href="{diff_rel}" target="_blank" style="color: #f87171; text-decoration: none; margin-right: 8px;">🔥 Error</a>
                    <a href="{ckpt_rel}" style="color: #94a3b8; text-decoration: none; font-size: 0.85em;">💾 Keras</a>
                </td>
            </tr>
            """
            
        # Selectable inspection options
        tabs_html = """
            <button class="tab-btn active" onclick="selectView('current')">1. Latest AS-DBPN</button>
            <button class="tab-btn" onclick="selectView('best')">2. Peak Best AS-DBPN</button>
            <button class="tab-btn" onclick="selectView('diff')">3. Error Map |SR - GT|</button>
            <button class="tab-btn" onclick="selectView('gt')">4. Ground Truth (HR)</button>
            <button class="tab-btn" onclick="selectView('bilinear')">5. Bilinear Baseline</button>
        """
        if self.ldbpn_metrics:
            tabs_html += """
            <button class="tab-btn" onclick="selectView('ldbpn')">6. LDBPN 3D Baseline</button>
            """
        html_dir = os.path.dirname(os.path.abspath(self.html_path))
        rep_rel = os.path.relpath(self.report_dir, html_dir)
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="30">
    <title>3D AS-DBPN Convergence & Visual Quality Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Space+Grotesk:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-primary: #070b14;
            --bg-secondary: #0f172a;
            --accent-glow: #38bdf8;
            --accent-success: #10b981;
            --accent-warning: #f59e0b;
            --accent-danger: #f43f5e;
            --accent-purple: #8b5cf6;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --border-glass: rgba(255, 255, 255, 0.08);
            --card-glass: rgba(15, 23, 42, 0.75);
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-primary);
            background-image: 
                radial-gradient(at 0% 0%, hsla(210, 100%, 40%, 0.12) 0px, transparent 50%),
                radial-gradient(at 100% 100%, hsla(280, 100%, 50%, 0.08) 0px, transparent 50%);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 2rem 1.5rem;
        }}

        header {{
            width: 100%;
            max-width: 1300px;
            margin-bottom: 2rem;
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }}

        .header-title-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
        }}

        h1 {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #fff 30%, var(--accent-glow) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.03em;
        }}

        .badge-live {{
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid var(--accent-success);
            color: var(--accent-success);
            padding: 0.4rem 1rem;
            border-radius: 50px;
            font-size: 0.85rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            box-shadow: 0 0 15px rgba(16, 185, 129, 0.15);
        }}

        .pulse-dot {{
            width: 8px;
            height: 8px;
            background: var(--accent-success);
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 8px var(--accent-success);
            animation: pulse 2s infinite;
        }}

        @keyframes pulse {{
            0% {{ transform: scale(0.95); opacity: 0.8; }}
            50% {{ transform: scale(1.3); opacity: 1; }}
            100% {{ transform: scale(0.95); opacity: 0.8; }}
        }}

        .subtitle {{
            color: var(--text-secondary);
            font-size: 1.05rem;
        }}

        .kpi-grid {{
            width: 100%;
            max-width: 1300px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}

        .kpi-card {{
            background: var(--card-glass);
            border: 1px solid var(--border-glass);
            backdrop-filter: blur(12px);
            border-radius: 14px;
            padding: 1.25rem;
            display: flex;
            flex-direction: column;
            gap: 0.4rem;
        }}

        .kpi-label {{
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
            font-weight: 600;
        }}

        .kpi-value {{
            font-size: 1.8rem;
            font-weight: 700;
            font-family: 'Space Grotesk', sans-serif;
        }}

        .kpi-subtext {{
            font-size: 0.82rem;
            color: var(--text-secondary);
        }}

        .dashboard-container {{
            width: 100%;
            max-width: 1300px;
            display: grid;
            grid-template-columns: 1.4fr 1fr;
            gap: 1.75rem;
            margin-bottom: 2.5rem;
        }}

        @media (max-width: 1024px) {{
            .dashboard-container {{
                grid-template-columns: 1fr;
            }}
        }}

        .glass-card {{
            background: var(--card-glass);
            border: 1px solid var(--border-glass);
            backdrop-filter: blur(16px);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        }}

        .visualizer-panel {{
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }}

        .viewer-controls {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.85rem;
            color: var(--text-secondary);
        }}

        .shortcut-key {{
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 0.1rem 0.35rem;
            border-radius: 4px;
            font-family: monospace;
            font-size: 0.75rem;
            margin-right: 0.2rem;
        }}

        .image-viewport {{
            position: relative;
            width: 100%;
            aspect-ratio: 16 / 6.2;
            background: #020617;
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border-glass);
            display: flex;
            align-items: center;
            justify-content: center;
        }}

        .viewport-image {{
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%;
            object-fit: contain;
            opacity: 0;
            transition: opacity 0.22s ease-in-out;
        }}

        .viewport-image.active {{
            opacity: 1;
            z-index: 2;
        }}

        .selector-tabs {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin-top: 0.25rem;
        }}

        .tab-btn {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-glass);
            color: var(--text-secondary);
            padding: 0.55rem 0.95rem;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.85rem;
            font-weight: 600;
            transition: all 0.18s ease;
        }}

        .tab-btn:hover {{
            background: rgba(255, 255, 255, 0.12);
            color: #fff;
        }}

        .tab-btn.active {{
            background: var(--accent-glow);
            color: #020617;
            border-color: var(--accent-glow);
            box-shadow: 0 0 12px rgba(56, 189, 248, 0.35);
        }}

        .panel-section-title {{
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 1rem;
            color: #fff;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}

        .arch-pill {{
            background: rgba(56, 189, 248, 0.1);
            border: 1px solid rgba(56, 189, 248, 0.25);
            color: var(--accent-glow);
            padding: 0.25rem 0.6rem;
            border-radius: 6px;
            font-size: 0.78rem;
            font-family: monospace;
        }}

        .charts-row {{
            width: 100%;
            max-width: 1300px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2.5rem;
        }}

        .table-section {{
            width: 100%;
            max-width: 1300px;
            margin-bottom: 3rem;
        }}

        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
            text-align: left;
        }}

        .metrics-table th {{
            background: rgba(15, 23, 42, 0.9);
            padding: 0.85rem 1rem;
            color: var(--text-secondary);
            font-weight: 600;
            border-bottom: 2px solid var(--border-glass);
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
        }}

        .metrics-table td {{
            padding: 0.8rem 1rem;
            border-bottom: 1px solid var(--border-glass);
            color: var(--text-primary);
        }}

        .metrics-table tr:hover td {{
            background: rgba(255, 255, 255, 0.03);
        }}

        .stage-badge {{
            background: rgba(139, 92, 246, 0.15);
            color: #c084fc;
            padding: 0.25rem 0.55rem;
            border-radius: 4px;
            font-size: 0.78rem;
            font-weight: 600;
        }}

        footer {{
            color: var(--text-secondary);
            font-size: 0.85rem;
            text-align: center;
            margin-top: auto;
            padding: 1.5rem 0;
            opacity: 0.7;
        }}
    </style>
</head>
<body>

    <header>
        <div class="header-title-row">
            <h1>3D AS-DBPN Convergence Dashboard</h1>
            <div class="badge-live">
                <span class="pulse-dot"></span>
                <span>Iter {cur_iter} &bull; {cur_stage} &bull; Live Training</span>
            </div>
        </div>
        <div class="subtitle">
            Attention-Guided Shared Deep Back-Projection Network (3D Brain MRI Super-Resolution)
        </div>
    </header>

    <!-- Top KPI Grid -->
    <div class="kpi-grid">
        <div class="kpi-card">
            <span class="kpi-label">Validation PSNR</span>
            <span class="kpi-value" style="color: {gain_color};">{cur_psnr:.2f} dB</span>
            <span class="kpi-subtext">{gain_sign}{psnr_gain:.2f} dB vs Bilinear ({lin_psnr:.2f} dB)</span>
        </div>
        <div class="kpi-card">
            <span class="kpi-label">Structural Similarity (SSIM)</span>
            <span class="kpi-value" style="color: #a855f7;">{cur_ssim:.4f}</span>
            <span class="kpi-subtext">Bilinear Baseline: {lin_ssim:.4f}</span>
        </div>
        <div class="kpi-card">
            <span class="kpi-label">High-Freq Error (HFEN)</span>
            <span class="kpi-value" style="color: #38bdf8;">{cur_hfen:.4f}</span>
            <span class="kpi-subtext">Bilinear Baseline: {lin_hfen:.4f} (lower is sharper)</span>
        </div>
        <div class="kpi-card">
            <span class="kpi-label">Peak Best Model (CQS)</span>
            <span class="kpi-value" style="color: #10b981;">{self.best_cqs:.4f}</span>
            <span class="kpi-subtext">Iter {self.best_iter} ({self.best_stage or 'Best'}) &bull; PSNR: {self.best_psnr:.2f} dB</span>
        </div>
        <div class="kpi-card">
            <span class="kpi-label">Parameters & Architecture</span>
            <span class="kpi-value" style="color: #f8fafc; font-size: 1.5rem;">1,836,706</span>
            <span class="kpi-subtext">Proj Kernel: 6x6x6 &bull; Loops: T=4 &bull; SOCA</span>
        </div>
    </div>

    <!-- Curriculum Stages Architecture & Rationale Guide -->
    <div style="width: 100%; max-width: 1300px; margin-bottom: 2rem;">
        <div style="background: var(--card-glass); border: 1px solid var(--border-glass); border-radius: 14px; padding: 1.25rem;">
            <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-secondary); font-weight: 700; margin-bottom: 0.85rem;">
                📚 Progressive 4-Phase Curriculum Training Architecture & Rationale
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 1rem;">
                <div style="background: rgba(2, 6, 23, 0.6); border-left: 3px solid #38bdf8; border-radius: 8px; padding: 0.85rem 1rem;">
                    <div style="font-size: 0.75rem; text-transform: uppercase; color: #38bdf8; font-weight: 700;">Phase 0 &bull; Warmup</div>
                    <div style="font-weight: 600; color: #f8fafc; font-size: 0.95rem; margin: 2px 0;">MSE Initial Bootstrapping</div>
                    <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.4;">Rapid pixel-wise MSE on procedural geometric shapes to establish feature alignment and boost PSNR from random initialization (15 dB) to ~21.3 dB.</div>
                </div>
                <div style="background: rgba(2, 6, 23, 0.6); border-left: 3px solid #10b981; border-radius: 8px; padding: 0.85rem 1rem;">
                    <div style="font-size: 0.75rem; text-transform: uppercase; color: #10b981; font-weight: 700;">Phase 1 &bull; Stage 1 Adaptation</div>
                    <div style="font-weight: 600; color: #f8fafc; font-size: 0.95rem; margin: 2px 0;">Hybrid Perceptual Loss Switch</div>
                    <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.4;">Switches from MSE to multi-component Hybrid Loss (MSE + Perceptual VGG + HFEN edge + Gradient loss) with dynamic task balancing on clean shapes to learn edge structure.</div>
                </div>
                <div style="background: rgba(2, 6, 23, 0.6); border-left: 3px solid #f59e0b; border-radius: 8px; padding: 0.85rem 1rem;">
                    <div style="font-size: 0.75rem; text-transform: uppercase; color: #f59e0b; font-weight: 700;">Phase 2 &bull; Stage 2 Robustness</div>
                    <div style="font-weight: 600; color: #f8fafc; font-size: 0.95rem; margin: 2px 0;">Blind Rician Noise Training</div>
                    <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.4;">Injects realistic MRI Rician noise and multi-scale degradations so the network learns simultaneous denoising, deblurring, and super-resolution.</div>
                </div>
                <div style="background: rgba(2, 6, 23, 0.6); border-left: 3px solid #8b5cf6; border-radius: 8px; padding: 0.85rem 1rem;">
                    <div style="font-size: 0.75rem; text-transform: uppercase; color: #8b5cf6; font-weight: 700;">Phase 3 &bull; Stage 3 Refinement</div>
                    <div style="font-weight: 600; color: #f8fafc; font-size: 0.95rem; margin: 2px 0;">Dedicated Anatomical Tuning</div>
                    <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.4;">Final curriculum fine-tuning on high-frequency brain textures with fine learning rates for maximum clinical visual fidelity and high-contrast vessel/tissue definition.</div>
                </div>
            </div>
        </div>
    </div>

    <!-- Main Visualizer and Diagnostic Panel -->
    <div class="dashboard-container">
        <!-- Visualizer Panel -->
        <div class="glass-card visualizer-panel">
            <div class="viewer-controls">
                <span>Orthogonal Slices: <strong>Axial (Z) &bull; Coronal (Y) &bull; Sagittal (X)</strong></span>
                <span>💡 Tip: Click buttons or press <span class="shortcut-key">1</span> to <span class="shortcut-key">5</span> to toggle models</span>
            </div>

            <div class="image-viewport">
                <img id="img-current" src="{rep_rel}/val3d_asdbpn_current.png" alt="Current AS-DBPN" class="viewport-image active">
                <img id="img-best" src="{rep_rel}/val3d_asdbpn_best.png" alt="Best AS-DBPN" class="viewport-image">
                <img id="img-diff" src="{rep_rel}/diff3d_asdbpn_current.png" alt="Error Map" class="viewport-image">
                <img id="img-gt" src="{rep_rel}/val3d_ground_truth.png" alt="Ground Truth" class="viewport-image">
                <img id="img-bilinear" src="{rep_rel}/val3d_bilinear.png" alt="Bilinear" class="viewport-image">
                {'<img id="img-ldbpn" src="' + rep_rel + '/val3d_ldbpn.png" alt="LDBPN 3D" class="viewport-image">' if self.ldbpn_metrics else ''}
            </div>

            <div class="selector-tabs">
                {tabs_html}
            </div>
        </div>

        <!-- Metric Summary Card -->
        <div class="glass-card">
            <div class="panel-section-title">
                <span>Model Architecture & Configuration</span>
                <span class="arch-pill">Parity with 2D Benchmark</span>
            </div>
            
            <p style="color: var(--text-secondary); font-size: 0.9rem; line-height: 1.6; margin-bottom: 1.25rem;">
                Extending the top-performing 2D AS-DBPN configuration to 3D with <strong>1,836,706 parameters</strong>: 
                4 recurrent feedback projection loops with shared weights, <strong>6x6x6 deconvolution/convolution back-projection kernels</strong>, 
                shared LayerNormalization for loop stability, and final Second-Order Channel Attention (SOCA).
            </p>

            <table class="metrics-table" style="margin-bottom: 1.5rem;">
                <thead>
                    <tr>
                        <th>Configuration Item</th>
                        <th>Setting / Value</th>
                    </tr>
                </thead>
                <tbody>
                    <tr><td>Dimensionality</td><td><strong>3D Volumetric (96x96x96 HR)</strong></td></tr>
                    <tr><td>Projection Kernel Size</td><td><strong>6x6x6</strong> (DBPN Back-Projection standard)</td></tr>
                    <tr><td>Recurrent Steps (T)</td><td><strong>4 steps</strong> (iterative refinement)</td></tr>
                    <tr><td>Shared Normalization</td><td><strong>LayerNormalization</strong> (axis=-1)</td></tr>
                    <tr><td>Attention Block</td><td><strong>SOCA 3D</strong> (channel attention)</td></tr>
                    <tr><td>Global Skip</td><td><strong>LearnableScale</strong> (&alpha; init = 0.0)</td></tr>
                    <tr><td>Loss Function</td><td><strong>Dynamic Balancing (MAE 30% + Percep 65% + TV 0.5%)</strong></td></tr>
                </tbody>
            </table>
        </div>
    </div>

    <!-- Convergence Trajectory Charts -->
    <div class="charts-row">
        <div class="glass-card" style="padding: 1.2rem;">
            {svg_psnr}
        </div>
        <div class="glass-card" style="padding: 1.2rem;">
            {svg_ssim}
        </div>
        <div class="glass-card" style="padding: 1.2rem;">
            {svg_loss}
        </div>
    </div>

    <!-- Checkpoint Audit Table -->
    <div class="glass-card table-section">
        <div class="panel-section-title">
            <span>Convergence Checkpoints Log ({len(self.history)} saved milestones)</span>
        </div>
        <div style="overflow-x: auto;">
            <table class="metrics-table">
                <thead>
                    <tr>
                        <th>Global Step (Iter)</th>
                        <th>Stage</th>
                        <th>Validation PSNR</th>
                        <th>SSIM</th>
                        <th>HFEN</th>
                        <th>GMSD</th>
                        <th>CBI</th>
                        <th>Corr</th>
                        <th>Train Loss</th>
                        <th>Artifacts</th>
                    </tr>
                </thead>
                <tbody>
                    {checkpoint_rows}
                </tbody>
            </table>
        </div>
    </div>

    <footer>
        SIQ: Super-Resolution Image Quantification &bull; Report auto-refreshes every 30s &bull; Generated: {latest_entry.get('timestamp', '') if latest_entry else ''}
    </footer>

    <script>
        const views = {{
            'current': 'img-current',
            'best': 'img-best',
            'diff': 'img-diff',
            'gt': 'img-gt',
            'bilinear': 'img-bilinear',
            'ldbpn': 'img-ldbpn'
        }};

        function selectView(key) {{
            Object.values(views).forEach(id => {{
                const el = document.getElementById(id);
                if (el) el.classList.remove('active');
            }});
            const target = document.getElementById(views[key]);
            if (target) target.classList.add('active');

            const buttons = document.querySelectorAll('.tab-btn');
            buttons.forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
        }}

        // Keyboard shortcuts 1-6
        document.addEventListener('keydown', (e) => {{
            const keyMap = {{
                '1': 'current',
                '2': 'best',
                '3': 'diff',
                '4': 'gt',
                '5': 'bilinear',
                '6': 'ldbpn'
            }};
            if (keyMap[e.key]) {{
                const key = keyMap[e.key];
                Object.values(views).forEach(id => {{
                    const el = document.getElementById(id);
                    if (el) el.classList.remove('active');
                }});
                const target = document.getElementById(views[key]);
                if (target) target.classList.add('active');
                
                const buttons = document.querySelectorAll('.tab-btn');
                buttons.forEach(btn => {{
                    if (btn.innerText.startsWith(e.key)) {{
                        buttons.forEach(b => b.classList.remove('active'));
                        btn.classList.add('active');
                    }}
                }});
            }}
        }});
    </script>
</body>
</html>
"""
        with open(self.html_path, "w") as f:
            f.write(html)
        # print(f"[Convergence Reporter] HTML dashboard updated at {self.html_path}")

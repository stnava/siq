#!/usr/bin/env python3
"""
sr_alignment_qc.py — Standardized Super-Resolution Spatial Alignment & Bias Quality Control (QC) Tool.

Automates the detection of:
  1. Sub-pixel FFT phase cross-correlation shifts (catches uncompensated transposed-convolution or reflection offsets)
  2. Directional edge error correlation (catches halo boundary ridges in residual maps)
  3. Objective reconstruction fidelity (PSNR, SSIM, GMSD, CBI, CQS, PCS) vs Bilinear baseline

Usage:
  # Evaluate a model checkpoint on a validation NIfTI:
  python scripts/sr_alignment_qc.py \
      --model checkpoints/siq_smallshort_train_2x2x2_1chan_featvggL6_blind/siq_3d_best_cqs.keras \
      --val-image /Users/stnava/data/blast_cohorts/BIDS/FPA/sub-BLAST022/ses-01/anat/sub-BLAST022_ses-01_run-001_T1w.nii.gz

  # Evaluate pre-generated image volumes (GT vs SR vs Bilinear):
  python scripts/sr_alignment_qc.py \
      --gt ground_truth.nii.gz \
      --sr model_sr.nii.gz \
      --bilinear bilinear.nii.gz

  # Generate standalone HTML & JSON reports:
  python scripts/sr_alignment_qc.py \
      --model my_model.keras --val-image val.nii.gz \
      --output-json qc_report.json --output-html qc_report.html
"""

import os
import sys
import json
import argparse
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import ants
import siq


def render_html_qc_report(qc_result, out_path, title="Super-Resolution Quality Control Report"):
    """Renders a standalone interactive HTML QC report."""
    status = qc_result["status"]
    status_color = {"PASS": "#10b981", "WARN": "#f59e0b", "FAIL": "#ef4444"}.get(status, "#6b7280")
    status_badge = {"PASS": "PASS", "WARN": "WARNING", "FAIL": "FAILED"}.get(status, "UNKNOWN")

    phase_shift = qc_result.get("shift_rel", qc_result["phase_shift"])
    edge_corr = qc_result["edge_correlation"]
    dim_names = ["Y (dim 0)", "X (dim 1)", "Z (dim 2)"] if len(phase_shift) == 3 else ["Y (dim 0)", "X (dim 1)"]

    phase_rows = ""
    for i, name in enumerate(dim_names):
        val = phase_shift[i]
        c = "#10b981" if abs(val) < 0.08 else ("#f59e0b" if abs(val) < 0.15 else "#ef4444")
        phase_rows += f"<tr><td style='padding:8px 12px;font-weight:600;'>{name}</td><td style='padding:8px 12px;color:{c};font-family:monospace;'>{val:+.4f} voxels</td></tr>"

    edge_rows = ""
    for i, name in enumerate(dim_names):
        val = edge_corr[i]
        c = "#10b981" if val < 0.16 else ("#f59e0b" if val < 0.26 else "#ef4444")
        edge_rows += f"<tr><td style='padding:8px 12px;font-weight:600;'>{name}</td><td style='padding:8px 12px;color:{c};font-family:monospace;'>{val:.4f}</td></tr>"

    b_psnr = f"{qc_result['bilinear_psnr']:.2f} dB" if qc_result.get("bilinear_psnr") is not None else "N/A"
    d_psnr = f"{qc_result['delta_psnr_vs_bilinear']:+.2f} dB" if qc_result.get("delta_psnr_vs_bilinear") is not None else "N/A"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 30px; }}
    .container {{ max-width: 900px; margin: 0 auto; }}
    .card {{ background: #1e293b; border-radius: 12px; padding: 24px; margin-bottom: 24px; border: 1px solid #334155; }}
    .badge {{ display: inline-block; padding: 6px 14px; border-radius: 9999px; font-weight: 700; font-size: 0.875rem; letter-spacing: 0.05em; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
    th, td {{ text-align: left; border-bottom: 1px solid #334155; }}
    th {{ padding: 10px 12px; color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-top: 16px; }}
    .metric-box {{ background: #0f172a; padding: 16px; border-radius: 8px; border: 1px solid #334155; }}
    .metric-val {{ font-size: 1.5rem; font-weight: 700; margin-top: 4px; font-family: monospace; }}
    .metric-label {{ font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; font-weight: 600; }}
  </style>
</head>
<body>
  <div class="container">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:20px;">
      <div>
        <h1 style="margin:0; font-size:1.75rem;">Super-Resolution Alignment & Bias QC</h1>
        <p style="margin:4px 0 0 0; color:#94a3b8; font-size:0.9rem;">Automated spatial phase cross-correlation & directional gradient error audit</p>
      </div>
      <div>
        <span class="badge" style="background:{status_color}22; color:{status_color}; border: 1px solid {status_color};">{status_badge}</span>
      </div>
    </div>

    <div class="card">
      <h3 style="margin-top:0; color:#cbd5e1;">Key Performance Indicators</h3>
      <div class="grid">
        <div class="metric-box">
          <div class="metric-label">Max Phase Shift</div>
          <div class="metric-val" style="color: {'#10b981' if qc_result['max_phase_shift'] < 0.08 else ('#f59e0b' if qc_result['max_phase_shift'] < 0.15 else '#ef4444')};">{qc_result['max_phase_shift']:.4f} vox</div>
        </div>
        <div class="metric-box">
          <div class="metric-label">Max Edge Correlation</div>
          <div class="metric-val" style="color: {'#10b981' if qc_result['max_edge_correlation'] < 0.16 else ('#f59e0b' if qc_result['max_edge_correlation'] < 0.26 else '#ef4444')};">{qc_result['max_edge_correlation']:.4f}</div>
        </div>
        <div class="metric-box">
          <div class="metric-label">Model PSNR (vs Bilinear)</div>
          <div class="metric-val">{qc_result['psnr']:.2f} dB <span style="font-size:0.9rem; color:#94a3b8;">({d_psnr})</span></div>
        </div>
        <div class="metric-box">
          <div class="metric-label">Model SSIM</div>
          <div class="metric-val">{qc_result['ssim']:.4f}</div>
        </div>
      </div>
    </div>

    <div style="display:grid; grid-template-columns:1fr 1fr; gap:24px;">
      <div class="card">
        <h3 style="margin-top:0; color:#cbd5e1;">FFT Phase Cross-Correlation</h3>
        <p style="color:#94a3b8; font-size:0.85rem;">Measures sub-pixel translational shift relative to Ground Truth. Target: &lt; 0.08 voxels.</p>
        <table>
          <thead><tr><th>Spatial Axis</th><th>Translational Offset</th></tr></thead>
          <tbody>{phase_rows}</tbody>
        </table>
      </div>

      <div class="card">
        <h3 style="margin-top:0; color:#cbd5e1;">Directional Edge Error Correlation</h3>
        <p style="color:#94a3b8; font-size:0.85rem;">Measures correlation between residual error and true spatial gradients (relative to bilinear). Target: &lt; 0.16 (~0.08 voxel).</p>
        <table>
          <thead><tr><th>Spatial Axis</th><th>Pearson |r|</th></tr></thead>
          <tbody>{edge_rows}</tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <h3 style="margin-top:0; color:#cbd5e1;">Extended Metrics</h3>
      <table>
        <thead><tr><th>Metric</th><th>Model Value</th><th>Description</th></tr></thead>
        <tbody>
          <tr><td style="padding:8px 12px;font-weight:600;">Composite Quality Score (CQS)</td><td style="padding:8px 12px;font-family:monospace;">{qc_result.get('cqs', 0):.4f}</td><td style="padding:8px 12px;color:#94a3b8;">SSIM - GMSD - CBI</td></tr>
          <tr><td style="padding:8px 12px;font-weight:600;">Perceptual Composite Score (PCS)</td><td style="padding:8px 12px;font-family:monospace;">{qc_result.get('pcs', 0):.4f}</td><td style="padding:8px 12px;color:#94a3b8;">SSIM + 0.5*Acutance + 0.5*Laplacian - GMSD - CBI</td></tr>
          <tr><td style="padding:8px 12px;font-weight:600;">Gradient Magnitude Similarity (GMSD)</td><td style="padding:8px 12px;font-family:monospace;">{qc_result.get('gmsd', 0):.4f}</td><td style="padding:8px 12px;color:#94a3b8;">Edge degradation standard deviation (lower is sharper)</td></tr>
          <tr><td style="padding:8px 12px;font-weight:600;">Checkerboard Index (CBI)</td><td style="padding:8px 12px;font-family:monospace;">{qc_result.get('cbi', 0):.4f}</td><td style="padding:8px 12px;color:#94a3b8;">High-frequency alternating parity artifact index</td></tr>
          <tr><td style="padding:8px 12px;font-weight:600;">Bilinear Baseline PSNR</td><td style="padding:8px 12px;font-family:monospace;">{b_psnr}</td><td style="padding:8px 12px;color:#94a3b8;">Linear interpolation reference</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</body>
</html>
"""
    with open(out_path, "w") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(
        description="Standardized Super-Resolution Spatial Alignment & Bias Quality Control (QC) Tool"
    )
    # Mode A: Model + Validation Image
    parser.add_argument("--model", type=str, default=None, help="Path to trained model (.keras)")
    parser.add_argument("--config", type=str, default=None, help="Path to companion _config.json (optional)")
    parser.add_argument("--val-image", type=str, default=None, help="Path to real validation NIfTI volume")
    parser.add_argument("--factor", type=int, nargs="+", default=[2, 2, 2], help="Super-resolution scaling factor (default: 2 2 2)")

    # Mode B: Pre-generated NIfTI volumes
    parser.add_argument("--gt", type=str, default=None, help="Path to Ground Truth high-resolution image")
    parser.add_argument("--sr", type=str, default=None, help="Path to Super-Resolution model prediction image")
    parser.add_argument("--bilinear", type=str, default=None, help="Path to Bilinear baseline image (optional)")

    # QC Thresholds & Output
    parser.add_argument("--phase-thresh", type=float, default=0.08, help="Pass threshold for max phase shift in voxels (default: 0.08)")
    parser.add_argument("--edge-thresh", type=float, default=0.16, help="Pass threshold for max directional edge correlation, relative to bilinear (default: 0.16 ~ 0.08 voxel)")
    parser.add_argument("--output-json", type=str, default=None, help="Path to save JSON QC summary")
    parser.add_argument("--output-html", type=str, default=None, help="Path to save standalone HTML QC dashboard")
    parser.add_argument("--fail-on-error", action="store_true", help="Exit with non-zero return code if QC fails")

    args = parser.parse_args()

    factor_tuple = tuple(args.factor) if len(args.factor) in [2, 3] else (args.factor[0], args.factor[0], args.factor[0])

    # 1. Resolve Inputs
    if args.gt and args.sr:
        gt_img = ants.image_read(args.gt)
        sr_img = ants.image_read(args.sr)
        gt_np = ants.iMath(gt_img, "Normalize").numpy()
        sr_np = ants.iMath(sr_img, "Normalize").numpy()
        b_np = ants.iMath(ants.image_read(args.bilinear), "Normalize").numpy() if args.bilinear else None
    elif args.model and args.val_image:
        model, cfg = siq.load_siq_model(args.model)
        val_vol = ants.image_read(args.val_image)
        val_vol = ants.iMath(ants.iMath(val_vol, "TruncateIntensity", 0.001, 0.999), "Normalize")

        val_target_spacing = [val_vol.spacing[d] * factor_tuple[d] for d in range(val_vol.dimension)]
        low_res_vol = ants.resample_image(val_vol, val_target_spacing, use_voxels=False, interp_type=0)

        lr_box = 32
        val_shift = [40, 0, 40] if ("sub-BLAST" in args.val_image or "FPA" in args.val_image) else [0, 0, 0]
        mid_lr = [low_res_vol.shape[d] // 2 + int(round(val_shift[d] / factor_tuple[d])) for d in range(val_vol.dimension)]
        mid_hr = [mid_lr[d] * factor_tuple[d] for d in range(val_vol.dimension)]

        val_lr_p = ants.crop_indices(low_res_vol, [max(0, mid_lr[d] - lr_box) for d in range(val_vol.dimension)],
                                     [min(low_res_vol.shape[d], mid_lr[d] + lr_box) for d in range(val_vol.dimension)])
        val_hr_p = ants.crop_indices(val_vol, [max(0, mid_hr[d] - lr_box * factor_tuple[d]) for d in range(val_vol.dimension)],
                                     [min(val_vol.shape[d], mid_hr[d] + lr_box * factor_tuple[d]) for d in range(val_vol.dimension)])

        gt_np = ants.iMath(val_hr_p, "Normalize").numpy()

        sr_img = siq.inference(val_lr_p, model, config=cfg, verbose=False, poly_order=None, anti_checkerboard=False)
        sr_np = sr_img.numpy()

        bilinear_raw = ants.resample_image_to_target(val_lr_p, val_hr_p, interp_type=0)
        b_np = ants.iMath(bilinear_raw, "Normalize").numpy()
    else:
        parser.error("Either (--gt AND --sr) or (--model AND --val-image) must be specified.")

    # 2. Run QC Evaluation
    qc_result = siq.compute_alignment_qc(
        y_true=gt_np,
        y_pred=sr_np,
        bilinear=b_np,
        factor=factor_tuple,
        phase_thresh=args.phase_thresh,
        edge_corr_thresh=args.edge_thresh,
        verbose=True,
    )

    # 3. Output reports if requested
    if args.output_json:
        clean_dict = {k: v for k, v in qc_result.items() if k != "summary_table"}
        with open(args.output_json, "w") as f:
            json.dump(clean_dict, f, indent=2)
        print(f"JSON QC Report written to: {args.output_json}")

    if args.output_html:
        render_html_qc_report(qc_result, args.output_html)
        print(f"HTML QC Report written to: {args.output_html}")

    # 4. Exit code
    if args.fail_on_error and qc_result["status"] == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()

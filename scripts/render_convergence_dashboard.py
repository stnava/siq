import math
import os
import sys
import time
import pandas as pd


def generate_svg_chart(x_vals, y_vals, title, y_label, baseline_val=None, baseline_label=None, color="#3b82f6", height=220, width=540, stage_markers=None, x_label="Global Step"):
    """
    Renders a standalone clean, dark-mode SVG sparkline/curve with axes, gridlines,
    baseline reference dashed line, stage markers, and peak/latest indicators.
    """
    valid_pairs = [(x, float(y)) for x, y in zip(x_vals, y_vals) if y is not None and not pd.isna(y) and not (isinstance(y, float) and math.isnan(y))]
    if len(valid_pairs) == 0:
        return ""
    x_vals, y_vals = zip(*valid_pairs)
    x_vals = list(x_vals)
    y_vals = list(y_vals)
    
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
    if min_y == max_y:
        max_y = min_y + 1.0
        
    # Add 10% padding on y-axis
    y_buffer = (max_y - min_y) * 0.10
    if y_buffer == 0:
        y_buffer = 1.0
    min_y -= y_buffer
    max_y += y_buffer
    range_y = max_y - min_y
    range_x = max_x - min_x
    
    def map_x(x):
        return pad_left + ((x - min_x) / range_x) * plot_w
        
    def map_y(y):
        return pad_top + plot_h - ((y - min_y) / range_y) * plot_h
        
    svg = f'<svg viewBox="0 0 {w} {h}" class="chart-svg" style="width: 100%; height: auto; display: block;">\n'
    svg += f'  <rect width="{w}" height="{h}" fill="#0f172a" rx="8" />\n'
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
            if min_x < bx <= max_x:
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
        
    # Latest value indicator
    latest_x = map_x(x_vals[-1])
    latest_y = map_y(y_vals[-1])
    svg += f'  <circle cx="{latest_x:.1f}" cy="{latest_y:.1f}" r="5.5" fill="{color}" />\n'
    svg += f'  <text x="{pad_left + plot_w}" y="22" fill="{color}" font-family="monospace" font-size="12" font-weight="bold" text-anchor="end">Current: {y_vals[-1]:.2f}</text>\n'
    
    # X axis labels
    svg += f'  <text x="{pad_left}" y="{h - 12}" fill="#64748b" font-family="monospace" font-size="10">{x_label} {min_x}</text>\n'
    svg += f'  <text x="{pad_left + plot_w}" y="{h - 12}" fill="#64748b" font-family="monospace" font-size="10" text-anchor="end">{x_label} {max_x}</text>\n'
    
    svg += '</svg>\n'
    return svg


def render_html_dashboard_from_csv(csv_path="checkpoints/asdbpn_3d/convergence_history.csv",
                                   html_path="asdbpn_3d_report.html",
                                   bilinear_metrics=None,
                                   ldbpn_metrics=None,
                                   report_dir=None,
                                   checkpoint_dir=None,
                                   factor=None):
    """
    Lightning-fast, standalone HTML dashboard generator that reads convergence_history.csv
    and renders a beautiful, monotonic convergence dashboard in ~15ms with ZERO GPU or heavy module imports.
    """
    if not os.path.exists(csv_path):
        return False
        
    df = pd.read_csv(csv_path)
    history = df.to_dict("records")
    if len(history) == 0:
        return False

    base_stem = os.path.basename(html_path).replace("_report.html", "").replace(".html", "")
    if report_dir is None:
        report_dir = f"reports/{base_stem}"
    if checkpoint_dir is None:
        checkpoint_dir = os.path.dirname(csv_path) or f"checkpoints/{base_stem}"
    html_dir = os.path.dirname(os.path.abspath(html_path))
    if os.path.isabs(report_dir):
        report_dir = os.path.relpath(report_dir, html_dir)
    if os.path.isabs(checkpoint_dir):
        checkpoint_dir = os.path.relpath(checkpoint_dir, html_dir)

    if bilinear_metrics is None or factor is None:
        for cand_dir in [checkpoint_dir, report_dir, os.path.dirname(csv_path), "."]:
            if cand_dir:
                b_json = os.path.join(cand_dir, "baselines.json")
                if os.path.exists(b_json):
                    try:
                        import json
                        with open(b_json) as f:
                            b_data = json.load(f)
                            if bilinear_metrics is None and "bilinear" in b_data:
                                bilinear_metrics = b_data["bilinear"]
                            if factor is None and "factor" in b_data:
                                factor = b_data["factor"]
                            if bilinear_metrics is not None and factor is not None:
                                break
                    except Exception:
                        pass

    factor_str = "x".join(str(f) for f in factor) if isinstance(factor, (list, tuple)) else (f"{factor}x" if factor else "")
    if factor is not None and isinstance(factor, (list, tuple)) and len(factor) == 2:
        dim_str = "2d"
    elif "2d" in base_stem.lower():
        dim_str = "2d"
    else:
        dim_str = "3d"
    if "asdbpn" in base_stem.lower():
        m_display = "AS-DBPN"
    elif any(k in base_stem.lower() for k in ["dbpn", "grader", "vgg", "smallshort", "blind"]):
        m_display = "DBPN"
    else:
        m_display = "SR Model"
        
    bilinear = bilinear_metrics or {"psnr": 27.10, "ssim": 0.9285, "hfen": 0.4500, "corr": 0.9320}
    ldbpn = ldbpn_metrics or {"psnr": 26.50, "ssim": 0.9150, "hfen": 0.4900, "corr": 0.9180}
    
    best_psnr = -1.0
    best_cqs = -float("inf")
    best_score = -float("inf")
    best_pcs = -float("inf")
    best_step = 0
    best_entry = history[0]
    
    # Compute continuous monotonic global steps:
    # Warmup: 0..150
    # Stage 1: 151..250
    # Stage 2: 251..450
    # Stage 3: 451..650
    warmup_max_iter = 0
    for r in history:
        st = str(r.get("stage", ""))
        if "Warmup" in st or "Initial" in st:
            warmup_max_iter = max(warmup_max_iter, int(r.get("iteration", 0)))
    if warmup_max_iter == 0:
        warmup_max_iter = 150

    def _clean_val(v, default=0.0):
        if v is None or pd.isna(v):
            return default
        try:
            val = float(v)
            return default if math.isnan(val) else val
        except (ValueError, TypeError):
            return default

    global_steps = []
    stage_markers = []
    prev_stage = None
    for idx, r in enumerate(history):
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
        
        ssim = _clean_val(r.get("val_ssim"), 0.0)
        gmsd = _clean_val(r.get("val_gmsd"), 0.0)
        cbi = _clean_val(r.get("val_cbi"), 0.0)
        cqs = _clean_val(r.get("val_cqs"), ssim - gmsd - cbi)
        r["val_cqs"] = cqs

        acutance = _clean_val(r.get("val_acutance"), 0.0)
        laplacian = _clean_val(r.get("val_laplacian"), 0.0)
        spectral = _clean_val(r.get("val_spectral"), 0.0)
        pcs = _clean_val(r.get("val_pcs"), ssim + 0.5 * acutance + 0.5 * laplacian - gmsd - cbi)
        r["val_acutance"] = acutance
        r["val_laplacian"] = laplacian
        r["val_spectral"] = spectral
        r["val_pcs"] = pcs

        p = _clean_val(r.get("val_psnr"), 0.0)
        if p > best_psnr:
            best_psnr = p

        score = pcs if acutance > 0 else cqs
        if int(r.get("is_best", 0)) == 1 or score > best_score:
            best_score = score
            best_cqs = cqs
            best_pcs = pcs
            best_step = g_step
            best_entry = r

    has_perceptual = any(_clean_val(r.get("val_acutance"), 0.0) > 0 for r in history)
    latest_entry = history[-1]
    cur_step = global_steps[-1]
    cur_iter = latest_entry["iteration"]
    cur_stage = latest_entry["stage"]
    cur_psnr = _clean_val(latest_entry.get("val_psnr"), 0.0)
    cur_ssim = _clean_val(latest_entry.get("val_ssim"), 0.0)
    cur_gmsd = _clean_val(latest_entry.get("val_gmsd"), 0.0)
    cur_cbi = _clean_val(latest_entry.get("val_cbi"), 0.0)
    cur_cqs = _clean_val(latest_entry.get("val_cqs"), cur_ssim - cur_gmsd - cur_cbi)
    cur_acutance = _clean_val(latest_entry.get("val_acutance"), 0.0)
    cur_laplacian = _clean_val(latest_entry.get("val_laplacian"), 0.0)
    cur_spectral = _clean_val(latest_entry.get("val_spectral"), 0.0)
    cur_pcs = _clean_val(latest_entry.get("val_pcs"), 0.0)
    cur_hfen = _clean_val(latest_entry.get("val_hfen"), 0.0)
    cur_corr = _clean_val(latest_entry.get("val_corr"), 0.0)
    cur_loss = _clean_val(latest_entry.get("train_loss"), 0.0)
    
    lin_psnr = float(bilinear.get("psnr", 27.10))
    lin_ssim = float(bilinear.get("ssim", 0.9285))
    lin_hfen = float(bilinear.get("hfen", 0.4500))
    lin_acutance = float(bilinear.get("acutance_ratio", 0.789))
    lin_laplacian = float(bilinear.get("laplacian_ratio", 0.442))
    lin_spectral = float(bilinear.get("spectral_ratio", 0.474))
    
    psnr_gain = cur_psnr - lin_psnr
    gain_sign = "+" if psnr_gain >= 0 else ""
    gain_color = "#10b981" if psnr_gain >= 0 else "#f59e0b"

    psnrs = [_clean_val(r.get("val_psnr"), 0.0) for r in history]
    ssims = [_clean_val(r.get("val_ssim"), 0.0) for r in history]
    losses = [_clean_val(r.get("train_loss"), 0.0) for r in history]
    cqss = [_clean_val(r.get("val_cqs"), 0.0) for r in history]
    
    svg_cqs = generate_svg_chart(global_steps, cqss, "Composite Quality Score (CQS = SSIM - GMSD - CBI)", "CQS", color="#38bdf8", stage_markers=stage_markers, x_label="Global Step")
    svg_psnr = generate_svg_chart(global_steps, psnrs, "Validation PSNR Trajectory (dB)", "PSNR (dB)", baseline_val=lin_psnr, baseline_label="Bilinear", color="#10b981", stage_markers=stage_markers, x_label="Global Step")
    svg_ssim = generate_svg_chart(global_steps, ssims, "Validation SSIM Progression", "SSIM", baseline_val=lin_ssim, baseline_label="Bilinear", color="#8b5cf6", stage_markers=stage_markers, x_label="Global Step")
    svg_loss = generate_svg_chart(global_steps, losses, "Hybrid Training Loss", "Loss", color="#f59e0b", stage_markers=stage_markers, x_label="Global Step")

    perceptual_cards = ""
    svg_acutance = ""
    svg_laplacian = ""
    if has_perceptual:
        ac_gain = cur_acutance - lin_acutance
        ac_col = "#10b981" if ac_gain >= 0 else "#f59e0b"
        
        lap_gain = cur_laplacian - lin_laplacian
        lap_col = "#10b981" if lap_gain >= 0 else "#f59e0b"

        perceptual_cards = f"""
            <div class="glass-card">
                <div class="stat-label">Acutance Ratio</div>
                <div class="stat-value" style="color: #ec4899;">{cur_acutance:.4f}</div>
                <div class="stat-diff" style="color: {ac_col};">{ac_gain:+.4f} vs Bilinear ({lin_acutance:.4f})</div>
            </div>
            <div class="glass-card">
                <div class="stat-label">Laplacian Detail</div>
                <div class="stat-value" style="color: #06b6d4;">{cur_laplacian:.4f}</div>
                <div class="stat-diff" style="color: {lap_col};">{lap_gain:+.4f} vs Bilinear ({lin_laplacian:.4f})</div>
            </div>
        """
        valid_ac_steps = []
        valid_ac_vals = []
        valid_lap_vals = []
        for s, r in zip(global_steps, history):
            ac_v = _clean_val(r.get("val_acutance"), 0.0)
            lap_v = _clean_val(r.get("val_laplacian"), 0.0)
            if ac_v > 0:
                valid_ac_steps.append(s)
                valid_ac_vals.append(ac_v)
                valid_lap_vals.append(lap_v)
        if valid_ac_vals:
            svg_acutance = f"""<div class="glass-card" style="padding: 1.2rem;">{generate_svg_chart(valid_ac_steps, valid_ac_vals, "Acutance Ratio (Gradient Magnitude vs GT)", "Acutance", baseline_val=lin_acutance, baseline_label="Bilinear", color="#ec4899", stage_markers=stage_markers, x_label="Global Step")}</div>"""
            svg_laplacian = f"""<div class="glass-card" style="padding: 1.2rem;">{generate_svg_chart(valid_ac_steps, valid_lap_vals, "Laplacian Energy Ratio (High Freq Detail)", "Laplacian", baseline_val=lin_laplacian, baseline_label="Bilinear", color="#06b6d4", stage_markers=stage_markers, x_label="Global Step")}</div>"""


    checkpoint_rows = ""
    for idx, r in enumerate(reversed(history)):
        orig_idx = len(history) - 1 - idx
        g_step = global_steps[orig_idx]
        best_tag = ' <span style="background: rgba(16,185,129,0.2); color: #10b981; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem;">★ Champion</span>' if r.get("is_best", 0) else ""
        p_val = _clean_val(r.get("val_psnr"), 0.0)
        p_diff = p_val - lin_psnr
        p_sign = "+" if p_diff >= 0 else ""
        p_color = "#10b981" if p_diff >= 0 else "#e2e8f0"
        cqs_val = _clean_val(r.get("val_cqs"), 0.0)
        cbi_val = _clean_val(r.get("val_cbi"), 0.0)
        
        ortho_rel = f"{report_dir}/{r.get('ortho_image', '')}"
        diff_rel = f"{report_dir}/{r.get('diff_image', '')}"
        ckpt_rel = f"{checkpoint_dir}/{r.get('checkpoint_file', '')}"
        
        ac_val = _clean_val(r.get("val_acutance"), 0.0)
        lap_val = _clean_val(r.get("val_laplacian"), 0.0)
        ac_td = f'<td style="color: #ec4899; font-weight: 600;">{ac_val:.4f}</td>' if has_perceptual else ''
        lap_td = f'<td style="color: #06b6d4; font-weight: 600;">{lap_val:.4f}</td>' if has_perceptual else ''

        checkpoint_rows += f"""
        <tr>
            <td><strong>Step {g_step}</strong> <span style="font-size: 0.78rem; color: #64748b;">(Iter {r['iteration']})</span>{best_tag}</td>
            <td><span class="stage-badge">{r['stage']}</span></td>
            <td style="color: {p_color}; font-weight: bold;">{p_val:.2f} dB <span style="font-size: 0.8em; opacity: 0.8;">({p_sign}{p_diff:.2f})</span></td>
            <td>{_clean_val(r.get('val_ssim'), 0.0):.4f}</td>
            <td style="color: #38bdf8; font-weight: 600;">{cqs_val:.4f}</td>
            {ac_td}
            {lap_td}
            <td>{cbi_val:.4f}</td>
            <td>{_clean_val(r.get('val_gmsd'), 0.0):.4f}</td>
            <td>{_clean_val(r.get('val_hfen'), 0.0):.4f}</td>
            <td>{_clean_val(r.get('val_corr'), 0.0):.4f}</td>
            <td style="font-family: monospace;">{_clean_val(r.get('train_loss'), 0.0):.5f}</td>
            <td>
                <a href="{ortho_rel}" target="_blank" style="color: #38bdf8; text-decoration: none; margin-right: 8px;">📷 Ortho</a>
                <a href="{diff_rel}" target="_blank" style="color: #f87171; text-decoration: none; margin-right: 8px;">🔥 Error</a>
                <a href="{ckpt_rel}" style="color: #94a3b8; text-decoration: none; font-size: 0.85em;">💾 Keras</a>
            </td>
        </tr>
        """
        
    ac_tab = f" &bull; Acutance {cur_acutance:.3f}" if has_perceptual else ""
    champ_score_tab = f"PCS {best_score:.4f}" if has_perceptual else f"CQS {best_cqs:.4f}"
    champ_stat_label = "Peak Champion (PCS)" if has_perceptual else "Peak Champion (CQS)"
    champ_stat_val = f"{best_score:.4f}" if has_perceptual else f"{best_cqs:.4f}"
    champ_overlay_val = f"PCS: <strong>{best_score:.4f}</strong>" if has_perceptual else f"CQS: <strong>{best_cqs:.4f}</strong>"
    
    train_samples_file = os.path.join(html_dir, report_dir, "val3d_train_samples.png")
    has_train_samples = os.path.exists(train_samples_file) or os.path.exists(os.path.join(report_dir, "val3d_train_samples.png"))
    train_samples_tab = '<button class="tab-btn" data-view="trainsamples" onclick="selectView(\'trainsamples\')">8. Training Batches (Procedural Generator)</button>' if has_train_samples else ""
    train_samples_pane = f"""
                <div id="view-trainsamples" class="view-pane">
                    <img src="{report_dir}/val3d_train_samples.png" alt="Training Batches" class="viewport-image">
                    <div class="view-overlay">
                        <strong>8. Live Training Batches (Procedural Generator)</strong> &bull; Simulated Inputs (LR) vs Targets (HR)
                    </div>
                </div>
    """ if has_train_samples else ""
    
    tabs_html = f"""
        <button class="tab-btn" data-view="original" onclick="selectView('original')">1. Original Image (Ground Truth)</button>
        <button class="tab-btn" data-view="linear" onclick="selectView('linear')">2. Linear Upsampled ({lin_psnr:.2f} dB &bull; SSIM {lin_ssim:.4f})</button>
        <button class="tab-btn active" data-view="sr" onclick="selectView('sr')">3. Latest {m_display} ({cur_psnr:.2f} dB &bull; SSIM {cur_ssim:.4f} &bull; CQS {cur_cqs:.4f}{ac_tab})</button>
        <button class="tab-btn" data-view="best" onclick="selectView('best')">4. Peak Best Model ({_clean_val(best_entry.get('val_psnr', 0.0)):.2f} dB &bull; SSIM {_clean_val(best_entry.get('val_ssim', 0.0)):.4f} &bull; {champ_score_tab})</button>
        <button class="tab-btn" data-view="diff" onclick="selectView('diff')">5. Error Map |SR - Original|</button>
        <button class="tab-btn" data-view="downsampled" onclick="selectView('downsampled')">6. Downsampled Image (LR Input)</button>
        <button class="tab-btn" data-view="comparison4way" onclick="selectView('comparison4way')">7. 4-Way Comparison (Stacked)</button>
        {train_samples_tab}
    """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="15">
    <title>{dim_str.upper()} {m_display} Convergence & Visual Quality Dashboard</title>
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
            border-radius: 9999px;
            font-size: 0.85rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            box-shadow: 0 0 15px rgba(16, 185, 129, 0.2);
        }}

        .pulse-dot {{
            width: 8px;
            height: 8px;
            background-color: var(--accent-success);
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
            animation: pulse 2s infinite;
        }}

        @keyframes pulse {{
            0% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }}
            70% {{ transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }}
            100% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }}
        }}

        .dashboard-container {{
            width: 100%;
            max-width: 1300px;
            display: flex;
            flex-direction: column;
            gap: 2rem;
        }}

        .grid-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.25rem;
        }}

        .glass-card {{
            background: var(--card-glass);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--border-glass);
            border-radius: 16px;
            padding: 1.5rem;
            position: relative;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        }}

        .glass-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.15), transparent);
        }}

        .stat-label {{
            font-size: 0.85rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.4rem;
        }}

        .stat-value {{
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.85rem;
            font-weight: 700;
            color: var(--text-primary);
        }}

        .stat-diff {{
            font-size: 0.8rem;
            font-weight: 600;
            margin-top: 0.3rem;
        }}

        .viewer-card {{
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
        }}

        .viewer-controls {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1rem;
            border-bottom: 1px solid var(--border-glass);
            padding-bottom: 1rem;
        }}

        .tab-group {{
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }}

        .tab-btn {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-glass);
            color: var(--text-secondary);
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }}

        .tab-btn:hover {{
            background: rgba(255, 255, 255, 0.1);
            color: var(--text-primary);
        }}

        .tab-btn.active {{
            background: var(--accent-glow);
            color: #070b14;
            border-color: var(--accent-glow);
            box-shadow: 0 0 15px rgba(56, 189, 248, 0.35);
        }}

        .viewport-image {{
            width: 100%;
            height: auto;
            border-radius: 12px;
            border: 1px solid var(--border-glass);
            background: #000;
            display: block;
        }}

        .grid-4way {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.25rem;
        }}

        @media (max-width: 900px) {{
            .grid-4way {{
                grid-template-columns: 1fr;
            }}
        }}

        .card-4way {{
            background: rgba(15, 23, 42, 0.65);
            border: 1px solid var(--border-glass);
            border-radius: 12px;
            padding: 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
        }}

        .card-4way-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .card-4way-title {{
            font-size: 1rem;
            font-weight: 700;
        }}

        .card-4way-badge {{
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid var(--border-glass);
            padding: 0.2rem 0.6rem;
            border-radius: 6px;
            font-size: 0.75rem;
            color: var(--text-secondary);
            font-family: monospace;
        }}

        .viewport-image-4way {{
            width: 100%;
            aspect-ratio: 12 / 4.2;
            object-fit: cover;
            border-radius: 8px;
            border: 1px solid var(--border-glass);
            background: #000;
            display: block;
        }}

        .view-pane {{
            display: none;
            position: relative;
        }}

        .view-pane.active {{
            display: block;
        }}

        .view-overlay {{
            position: absolute;
            bottom: 12px;
            left: 12px;
            background: rgba(15, 23, 42, 0.85);
            backdrop-filter: blur(8px);
            padding: 6px 14px;
            border-radius: 6px;
            border: 1px solid var(--border-glass);
            font-size: 0.8rem;
            color: var(--text-secondary);
        }}

        .view-overlay strong {{
            color: var(--text-primary);
        }}

        .grid-charts {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
            gap: 1.5rem;
        }}

        .chart-svg {{
            width: 100%;
            height: auto;
            border-radius: 8px;
        }}

        .history-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
            text-align: left;
        }}

        .history-table th {{
            color: var(--text-secondary);
            font-weight: 600;
            padding: 0.85rem 1rem;
            border-bottom: 1px solid var(--border-glass);
            font-size: 0.8rem;
            text-transform: uppercase;
        }}

        .history-table td {{
            padding: 0.85rem 1rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.03);
            color: var(--text-primary);
        }}

        .history-table tr:hover td {{
            background: rgba(255, 255, 255, 0.02);
        }}

        .stage-badge {{
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            background: rgba(56, 189, 248, 0.15);
            color: var(--accent-glow);
            border: 1px solid rgba(56, 189, 248, 0.3);
        }}

        footer {{
            margin-top: 3rem;
            text-align: center;
            color: var(--text-secondary);
            font-size: 0.85rem;
        }}
    </style>
</head>
<body>
    <header>
        <div class="header-title-row">
            <h1>{dim_str.upper()} {m_display} Super-Resolution Convergence</h1>
            <div class="badge-live">
                <span class="pulse-dot"></span>
                <span>TRAINING ACTIVE &bull; {cur_stage.upper()}</span>
            </div>
        </div>
        <p style="color: var(--text-secondary); font-size: 0.95rem;">
            Continuous {dim_str.upper()} refinement tracking across Warmup, Stage 1 (clean MSE/L1), Stage 2 (noise/blur robustness), and Stage 3 (dedicated refinement).
        </p>
    </header>

    <main class="dashboard-container">
        <!-- High-level Metric Cards -->
        <section class="grid-stats">
            <div class="glass-card">
                <div class="stat-label">Global Step / Progress</div>
                <div class="stat-value">{cur_step} <span style="font-size: 1rem; color: #64748b;">(Iter {cur_iter})</span></div>
                <div class="stat-diff" style="color: #38bdf8;">{cur_stage}</div>
            </div>

            <div class="glass-card">
                <div class="stat-label">Composite Quality (CQS)</div>
                <div class="stat-value" style="color: #38bdf8;">{cur_cqs:.4f}</div>
                <div class="stat-diff" style="color: #64748b;">SSIM - GMSD - CBI (Peak: {best_cqs:.4f})</div>
            </div>
            
            {perceptual_cards}

            <div class="glass-card">
                <div class="stat-label">Validation PSNR</div>
                <div class="stat-value" style="color: {gain_color};">{cur_psnr:.2f} <span style="font-size: 1rem;">dB</span></div>
                <div class="stat-diff" style="color: {gain_color};">{gain_sign}{psnr_gain:.2f} dB vs Bilinear ({lin_psnr:.2f})</div>
            </div>

            <div class="glass-card">
                <div class="stat-label">Validation SSIM</div>
                <div class="stat-value" style="color: #8b5cf6;">{cur_ssim:.4f}</div>
                <div class="stat-diff" style="color: #64748b;">Bilinear Ref: {lin_ssim:.4f}</div>
            </div>

            <div class="glass-card">
                <div class="stat-label">{champ_stat_label}</div>
                <div class="stat-value" style="color: #10b981;">{champ_stat_val}</div>
                <div class="stat-diff" style="color: #10b981;">Step {best_step} &bull; {best_entry.get('stage', 'Stage 3')}</div>
            </div>
        </section>

        <!-- Interactive In-Place Flicker Viewer & Error Analysis -->
        <section class="glass-card viewer-card">
            <div class="viewer-controls">
                <div>
                    <h2 style="font-size: 1.3rem; font-weight: 600;">Interactive In-Place Comparison (Flicker Viewer)</h2>
                    <p style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 2px;">
                        💡 <strong>In-Place Flicker Navigation:</strong> Click tabs or press keyboard keys <span class="shortcut-key">1</span> (Original), <span class="shortcut-key">2</span> (Bilinear), <span class="shortcut-key">3</span> (Latest {m_display}) to rapidly flicker in-place and inspect edge sharpness and sub-voxel alignment.
                    </p>
                </div>
                <div class="tab-group">
                    {tabs_html}
                </div>
            </div>

            <div class="viewer-display">
                <div id="view-original" class="view-pane">
                    <img src="{report_dir}/val3d_original.png" alt="Original Image" class="viewport-image">
                    <div class="view-overlay">
                        <strong>1. Original Image (Ground Truth HR)</strong> &bull; Brain MRI Volume
                    </div>
                </div>

                <div id="view-linear" class="view-pane">
                    <img src="{report_dir}/val3d_linear_upsampled.png" alt="Linear Upsampled Image" class="viewport-image">
                    <div class="view-overlay">
                        <strong>2. Linear Upsampled Image (Bilinear Baseline)</strong> &bull; PSNR: {lin_psnr:.2f} dB &bull; SSIM: {lin_ssim:.4f}
                    </div>
                </div>

                <div id="view-sr" class="view-pane active">
                    <img src="{report_dir}/val3d_sr_upsampled.png" alt="SR Upsampled Image" class="viewport-image">
                    <div class="view-overlay">
                        <strong>3. Latest {m_display} Image ({dim_str.upper()} &bull; Step {cur_step} &bull; Iter {cur_iter})</strong> &bull; PSNR: <strong>{cur_psnr:.2f} dB</strong> &bull; SSIM: <strong>{cur_ssim:.4f}</strong> &bull; CQS: <strong>{cur_cqs:.4f}</strong>{f' &bull; Acutance: <strong>{cur_acutance:.4f}</strong> &bull; Laplacian: <strong>{cur_laplacian:.4f}</strong>' if has_perceptual else ''} &bull; GMSD: <strong>{cur_gmsd:.4f}</strong> &bull; CBI: <strong>{cur_cbi:.4f}</strong> &bull; HFEN: <strong>{cur_hfen:.4f}</strong>
                    </div>
                </div>

                <div id="view-best" class="view-pane">
                    <img src="{report_dir}/val3d_asdbpn_best.png" alt="Peak Champion Model Volume" class="viewport-image">
                    <div class="view-overlay">
                        <strong>4. Peak Champion Model (Step {best_step} &bull; Iter {best_entry.get('iteration', 0)})</strong> &bull; {champ_overlay_val} &bull; PSNR: {float(best_entry.get('val_psnr', 0.0)):.2f} dB &bull; CBI: {float(best_entry.get('val_cbi', 0.0)):.4f}
                    </div>
                </div>

                <div id="view-diff" class="view-pane">
                    <img src="{report_dir}/{latest_entry.get('diff_image', 'diff3d_asdbpn_current.png')}" alt="Residual Error Map" class="viewport-image">
                    <div class="view-overlay">
                        <strong>5. Residual Error Map |SR - Original| (Magma Colormap)</strong> &bull; High-frequency residual distribution
                    </div>
                </div>

                <div id="view-downsampled" class="view-pane">
                    <img src="{report_dir}/val3d_downsampled.png" alt="Downsampled Image" class="viewport-image">
                    <div class="view-overlay">
                        <strong>6. Downsampled Image (LR Input)</strong> &bull; Low-Resolution Input displayed at identical scale
                    </div>
                </div>

                <div id="view-comparison4way" class="view-pane">
                    <img src="{report_dir}/val3d_4way_comparison.png" alt="4-Way Stacked Comparison" class="viewport-image">
                    <div class="view-overlay">
                        <strong>7. Unified 4-Way Comparative Montage</strong> &bull; Original vs Downsampled vs Linear vs {dim_str.upper()} {m_display} (Identical Scale &amp; Contrast)
                    </div>
                </div>

                {train_samples_pane}
            </div>
        </section>

        <!-- SVG Convergence Charts -->
        <section class="grid-charts">
            <div class="glass-card" style="padding: 1.2rem;">
                {svg_cqs}
            </div>
            {svg_acutance}
            {svg_laplacian}
            <div class="glass-card" style="padding: 1.2rem;">
                {svg_psnr}
            </div>
            <div class="glass-card" style="padding: 1.2rem;">
                {svg_ssim}
            </div>
            <div class="glass-card" style="padding: 1.2rem;">
                {svg_loss}
            </div>
        </section>

        <!-- Checkpoint History Table -->
        <section class="glass-card" style="padding: 1.5rem; overflow-x: auto;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.2rem;">
                <h3 style="font-size: 1.15rem; font-weight: 600;">Convergence Checkpoint History</h3>
                <span style="font-size: 0.85rem; color: #64748b;">Auto-refreshes every 15s</span>
            </div>
            <table class="history-table">
                <thead>
                    <tr>
                        <th>Global Step (Iter)</th>
                        <th>Stage</th>
                        <th>Val PSNR (dB)</th>
                        <th>Val SSIM</th>
                        <th>CQS (★)</th>
                        {"<th>Val Acutance</th>" if has_perceptual else ""}
                        {"<th>Val Laplacian</th>" if has_perceptual else ""}
                        <th>Val CBI</th>
                        <th>Val GMSD</th>
                        <th>Val HFEN</th>
                        <th>Val Corr</th>
                        <th>Loss</th>
                        <th>Artifacts</th>
                    </tr>
                </thead>
                <tbody>
                    {checkpoint_rows}
                </tbody>
            </table>
        </section>
    </main>

    <footer>
        <p>siq 3D Deep Back-Projection Network (AS-DBPN) Super-Resolution Suite &bull; {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
    </footer>

    <script>
        const views = {{
            'original': 'view-original',
            'linear': 'view-linear',
            'sr': 'view-sr',
            'best': 'view-best',
            'diff': 'view-diff',
            'downsampled': 'view-downsampled',
            'comparison4way': 'view-comparison4way',
            'trainsamples': 'view-trainsamples'
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
            const activeBtn = document.querySelector(`.tab-btn[data-view="${{key}}"]`);
            if (activeBtn) activeBtn.classList.add('active');
        }}

        // Keyboard shortcuts 1-8 for in-place flickering
        document.addEventListener('keydown', (e) => {{
            const keyMap = {{
                '1': 'original',
                '2': 'linear',
                '3': 'sr',
                '4': 'best',
                '5': 'diff',
                '6': 'downsampled',
                '7': 'comparison4way',
                '8': 'trainsamples'
            }};
            if (keyMap[e.key]) {{
                selectView(keyMap[e.key]);
            }}
        }});
    </script>
</body>
</html>
"""
    with open(html_path, "w") as f:
        f.write(html)
    return True


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/asdbpn_3d/convergence_history.csv"
    html = sys.argv[2] if len(sys.argv) > 2 else "asdbpn_3d_report.html"
    success = render_html_dashboard_from_csv(csv, html)
    print("Render success:", success)

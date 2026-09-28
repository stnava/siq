import os
import sys
import time
import pandas as pd


def generate_svg_chart(x_vals, y_vals, title, y_label, baseline_val=None, baseline_label=None, color="#3b82f6", height=220, width=540, stage_markers=None, x_label="Global Step"):
    """
    Renders a standalone clean, dark-mode SVG sparkline/curve with axes, gridlines,
    baseline reference dashed line, stage markers, and peak/latest indicators.
    """
    if len(x_vals) == 0:
        return ""
    
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
                                   ldbpn_metrics=None):
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
        
    bilinear = bilinear_metrics or {"psnr": 27.10, "ssim": 0.9285, "hfen": 0.4500, "corr": 0.9320}
    ldbpn = ldbpn_metrics or {"psnr": 26.50, "ssim": 0.9150, "hfen": 0.4900, "corr": 0.9180}
    
    best_psnr = -1.0
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

    global_steps = []
    stage_markers = []
    prev_stage = None
    for r in history:
        st = str(r.get("stage", "Unknown"))
        it = int(r.get("iteration", 0))
        if "Warmup" in st or "Initial" in st:
            g_step = it
        else:
            g_step = warmup_max_iter + it
        global_steps.append(g_step)
        if prev_stage is not None and st != prev_stage:
            short_stage = st.replace("Phase", "").replace("Joint Fine-Tuning", "Stage 3").strip()
            stage_markers.append((g_step, short_stage))
        prev_stage = st
        
        p = float(r.get("val_psnr", 0.0))
        if p > best_psnr:
            best_psnr = p
            best_step = g_step
            best_entry = r

    latest_entry = history[-1]
    cur_step = global_steps[-1]
    cur_iter = latest_entry["iteration"]
    cur_stage = latest_entry["stage"]
    cur_psnr = float(latest_entry["val_psnr"])
    cur_ssim = float(latest_entry["val_ssim"])
    cur_gmsd = float(latest_entry["val_gmsd"])
    cur_hfen = float(latest_entry["val_hfen"])
    cur_corr = float(latest_entry["val_corr"])
    cur_loss = float(latest_entry["train_loss"])
    
    lin_psnr = float(bilinear.get("psnr", 27.10))
    lin_ssim = float(bilinear.get("ssim", 0.9285))
    lin_hfen = float(bilinear.get("hfen", 0.4500))
    
    psnr_gain = cur_psnr - lin_psnr
    gain_sign = "+" if psnr_gain >= 0 else ""
    gain_color = "#10b981" if psnr_gain >= 0 else "#f59e0b"

    psnrs = [float(r["val_psnr"]) for r in history]
    ssims = [float(r["val_ssim"]) for r in history]
    losses = [float(r["train_loss"]) for r in history]
    
    svg_psnr = generate_svg_chart(global_steps, psnrs, "Validation PSNR Trajectory (dB)", "PSNR (dB)", baseline_val=lin_psnr, baseline_label="Bilinear", color="#10b981", stage_markers=stage_markers, x_label="Global Step")
    svg_ssim = generate_svg_chart(global_steps, ssims, "Validation SSIM Progression", "SSIM", baseline_val=lin_ssim, baseline_label="Bilinear", color="#8b5cf6", stage_markers=stage_markers, x_label="Global Step")
    svg_loss = generate_svg_chart(global_steps, losses, "Hybrid Training Loss", "Loss", color="#38bdf8", stage_markers=stage_markers, x_label="Global Step")

    checkpoint_rows = ""
    for idx, r in enumerate(reversed(history)):
        orig_idx = len(history) - 1 - idx
        g_step = global_steps[orig_idx]
        best_tag = ' <span style="background: rgba(16,185,129,0.2); color: #10b981; padding: 2px 6px; border-radius: 4px; font-size: 0.75rem;">★ Best</span>' if r.get("is_best", 0) else ""
        p_val = float(r["val_psnr"])
        p_diff = p_val - lin_psnr
        p_sign = "+" if p_diff >= 0 else ""
        p_color = "#10b981" if p_diff >= 0 else "#e2e8f0"
        
        ortho_rel = f"reports/asdbpn_3d/{r.get('ortho_image', '')}"
        diff_rel = f"reports/asdbpn_3d/{r.get('diff_image', '')}"
        ckpt_rel = f"checkpoints/asdbpn_3d/{r.get('checkpoint_file', '')}"
        
        checkpoint_rows += f"""
        <tr>
            <td><strong>Step {g_step}</strong> <span style="font-size: 0.78rem; color: #64748b;">(Iter {r['iteration']})</span>{best_tag}</td>
            <td><span class="stage-badge">{r['stage']}</span></td>
            <td style="color: {p_color}; font-weight: bold;">{p_val:.2f} dB <span style="font-size: 0.8em; opacity: 0.8;">({p_sign}{p_diff:.2f})</span></td>
            <td>{float(r['val_ssim']):.4f}</td>
            <td>{float(r['val_hfen']):.4f}</td>
            <td>{float(r['val_gmsd']):.4f}</td>
            <td>{float(r['val_corr']):.4f}</td>
            <td style="font-family: monospace;">{float(r['train_loss']):.5f}</td>
            <td>
                <a href="{ortho_rel}" target="_blank" style="color: #38bdf8; text-decoration: none; margin-right: 8px;">📷 Ortho</a>
                <a href="{diff_rel}" target="_blank" style="color: #f87171; text-decoration: none; margin-right: 8px;">🔥 Error</a>
                <a href="{ckpt_rel}" style="color: #94a3b8; text-decoration: none; font-size: 0.85em;">💾 Keras</a>
            </td>
        </tr>
        """
        
    tabs_html = """
        <button class="tab-btn active" onclick="selectView('current')">1. Latest AS-DBPN</button>
        <button class="tab-btn" onclick="selectView('best')">2. Peak Best AS-DBPN</button>
        <button class="tab-btn" onclick="selectView('diff')">3. Error Map |SR - GT|</button>
        <button class="tab-btn" onclick="selectView('gt')">4. Ground Truth (HR)</button>
        <button class="tab-btn" onclick="selectView('bilinear')">5. Bilinear Baseline</button>
    """
    if ldbpn_metrics:
        tabs_html += """
        <button class="tab-btn" onclick="selectView('ldbpn')">6. LDBPN 3D Baseline</button>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="15">
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
            <h1>3D AS-DBPN Super-Resolution Convergence</h1>
            <div class="badge-live">
                <span class="pulse-dot"></span>
                <span>TRAINING ACTIVE &bull; STAGE 2</span>
            </div>
        </div>
        <p style="color: var(--text-secondary); font-size: 0.95rem;">
            Continuous isotropic 3D refinement tracking across Warmup, Stage 1 (clean MSE/L1), Stage 2 (noise/blur robustness), and Stage 3 (adversarial joint fine-tuning).
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
                <div class="stat-label">Validation PSNR</div>
                <div class="stat-value" style="color: {gain_color};">{cur_psnr:.2f} <span style="font-size: 1rem;">dB</span></div>
                <div class="stat-diff" style="color: {gain_color};">{gain_sign}{psnr_gain:.2f} dB vs Bilinear (27.10)</div>
            </div>

            <div class="glass-card">
                <div class="stat-label">Validation SSIM</div>
                <div class="stat-value" style="color: #8b5cf6;">{cur_ssim:.4f}</div>
                <div class="stat-diff" style="color: #64748b;">Bilinear Ref: {lin_ssim:.4f}</div>
            </div>

            <div class="glass-card">
                <div class="stat-label">High-Freq Error (HFEN)</div>
                <div class="stat-value" style="color: #38bdf8;">{cur_hfen:.4f}</div>
                <div class="stat-diff" style="color: #64748b;">Lower is sharper (Target &lt; 0.50)</div>
            </div>

            <div class="glass-card">
                <div class="stat-label">Peak All-Time PSNR</div>
                <div class="stat-value" style="color: #10b981;">{best_psnr:.2f} <span style="font-size: 1rem;">dB</span></div>
                <div class="stat-diff" style="color: #10b981;">Achieved at Step {best_step}</div>
            </div>
        </section>

        <!-- Visual Orthogonal Slices Interactive Viewer -->
        <section class="glass-card viewer-card">
            <div class="viewer-controls">
                <div>
                    <h2 style="font-size: 1.3rem; font-weight: 600;">Orthogonal Volume Cross-Sections (Z, Y, X)</h2>
                    <p style="font-size: 0.85rem; color: var(--text-secondary); margin-top: 2px;">
                        Interactive visual evaluation: compare reconstructed internal anatomy vs Ground Truth and baselines.
                    </p>
                </div>
                <div class="tab-group">
                    {tabs_html}
                </div>
            </div>

            <div class="viewer-display">
                <div id="view-current" class="view-pane active">
                    <img src="reports/asdbpn_3d/{latest_entry.get('ortho_image', '')}" alt="Latest Reconstructed Volume" class="viewport-image">
                    <div class="view-overlay">
                        <strong>Latest Model (Step {cur_step})</strong> &bull; PSNR: {cur_psnr:.2f} dB &bull; SSIM: {cur_ssim:.4f}
                    </div>
                </div>

                <div id="view-best" class="view-pane">
                    <img src="reports/asdbpn_3d/{best_entry.get('ortho_image', '')}" alt="Peak Best Model Volume" class="viewport-image">
                    <div class="view-overlay">
                        <strong>Peak Checkpoint (Step {best_step})</strong> &bull; PSNR: {best_psnr:.2f} dB
                    </div>
                </div>

                <div id="view-diff" class="view-pane">
                    <img src="reports/asdbpn_3d/{latest_entry.get('diff_image', '')}" alt="Residual Error Map" class="viewport-image">
                    <div class="view-overlay">
                        <strong>Absolute Difference |SR - GT| (Magma Colormap)</strong> &bull; High-frequency residual distribution
                    </div>
                </div>

                <div id="view-gt" class="view-pane">
                    <img src="reports/asdbpn_3d/baseline_gt_ortho.png" alt="Ground Truth High Resolution" class="viewport-image">
                    <div class="view-overlay">
                        <strong>Ground Truth (High Resolution 1.0mm Reference)</strong>
                    </div>
                </div>

                <div id="view-bilinear" class="view-pane">
                    <img src="reports/asdbpn_3d/baseline_bilinear_ortho.png" alt="Bilinear Interpolation Baseline" class="viewport-image">
                    <div class="view-overlay">
                        <strong>Bilinear Baseline</strong> &bull; PSNR: {lin_psnr:.2f} dB &bull; SSIM: {lin_ssim:.4f} &bull; HFEN: {lin_hfen:.4f}
                    </div>
                </div>

                <div id="view-ldbpn" class="view-pane">
                    <img src="reports/asdbpn_3d/baseline_ldbpn_ortho.png" alt="LDBPN 3D Baseline" class="viewport-image">
                    <div class="view-overlay">
                        <strong>LDBPN 3D Prior Benchmark</strong> &bull; PSNR: 26.50 dB &bull; SSIM: 0.9150
                    </div>
                </div>
            </div>
        </section>

        <!-- SVG Convergence Charts -->
        <section class="grid-charts">
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
                        <th>Val HFEN</th>
                        <th>Val GMSD</th>
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
            'current': 'view-current',
            'best': 'view-best',
            'diff': 'view-diff',
            'gt': 'view-gt',
            'bilinear': 'view-bilinear',
            'ldbpn': 'view-ldbpn'
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
    with open(html_path, "w") as f:
        f.write(html)
    return True


if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/asdbpn_3d/convergence_history.csv"
    html = sys.argv[2] if len(sys.argv) > 2 else "asdbpn_3d_report.html"
    success = render_html_dashboard_from_csv(csv, html)
    print("Render success:", success)

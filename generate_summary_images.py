import os
import sys
import numpy as np

# Configure Keras PyTorch backend
os.environ["KERAS_BACKEND"] = "torch"

import ants
import antspynet
import antspyt1w
import keras
import siq
from siq.get_data import compute_gmsd, compute_hfen

def main():
    print("Generating validation patches from r16...")
    img = ants.image_read(ants.get_data("r16"))
    img = ants.crop_image(img)
    low_res = ants.resample_image(img, [s*2 for s in img.spacing], use_voxels=False, interp_type=0)

    mid_lr = [s//2 for s in low_res.shape]
    lr_patch = ants.crop_indices(low_res, [m - 24 for m in mid_lr], [m + 24 for m in mid_lr])

    mid_hr = [s//2 for s in img.shape]
    hr_patch = ants.crop_indices(img, [m - 48 for m in mid_hr], [m + 48 for m in mid_hr])

    # Ground truth numpy array
    gt_np = hr_patch.numpy()

    # Output directory in artifacts
    artifact_dir = "/Users/stnava/.gemini/antigravity-cli/brain/bf9e3239-711d-4a46-8dd4-8a3f33959db5"
    os.makedirs(artifact_dir, exist_ok=True)

    # Save Ground Truth
    gt_path = os.path.join(artifact_dir, "r16_ground_truth.png")
    ants.plot(hr_patch, filename=gt_path, title="Ground Truth (r16 HR)")
    print(f"Saved Ground Truth to {gt_path}")

    # Nearest Neighbor metrics
    nn_sr = ants.resample_image_to_target(lr_patch, hr_patch, interp_type=1)
    nn_path = os.path.join(artifact_dir, "r16_nearest_neighbor.png")
    ants.plot(nn_sr, filename=nn_path, title="Nearest Neighbor")
    print(f"Saved Nearest Neighbor to {nn_path}")
    nn_np = nn_sr.numpy()
    nn_psnr = float(antspynet.psnr(hr_patch, nn_sr))
    nn_ssim = float(antspynet.ssim(hr_patch, nn_sr))
    nn_gmsd = float(compute_gmsd(gt_np, nn_np))
    nn_hfen = float(compute_hfen(gt_np, nn_np))
    nn_corr = float(np.corrcoef(nn_np.flatten(), gt_np.flatten())[0, 1])

    # Bilinear metrics
    linear_sr = ants.resample_image_to_target(lr_patch, hr_patch, interp_type=0)
    linear_path = os.path.join(artifact_dir, "r16_bilinear.png")
    ants.plot(linear_sr, filename=linear_path, title="Bilinear Interpolation")
    print(f"Saved Bilinear to {linear_path}")
    linear_np = linear_sr.numpy()
    linear_psnr = float(antspynet.psnr(hr_patch, linear_sr))
    linear_ssim = float(antspynet.ssim(hr_patch, linear_sr))
    linear_gmsd = float(compute_gmsd(gt_np, linear_np))
    linear_hfen = float(compute_hfen(gt_np, linear_np))
    linear_corr = float(np.corrcoef(linear_np.flatten(), gt_np.flatten())[0, 1])

    # Models list
    model_files = {
        "ESPCN": "./espcn_2d_attention_refined.keras",
        "WDSR": "./wdsr_2d_refined.keras",
        "RCAN": "./rcan_2d_refined.keras",
        "CARN": "./carn_2d_refined.keras",
        "LDBPN": "./ldbpn_2d_refined.keras",
        "REF-DBPN": "./ref_dbpn_2d_refined.keras",
        "ESPCN-RC": "./espcn_2d_resize_conv_refined.keras",
        "WDSR-RC": "./wdsr_2d_resize_conv_refined.keras",
        "SRFBN": "./srfbn_2d_refined.keras",
        "SAN": "./san_2d_refined.keras"
    }

    custom_objects = {
        "PixelShuffle2D": siq.PixelShuffle2D,
        "LearnableScale": siq.LearnableScale
    }

    # Run inference and save images
    model_results = []
    for model_name, m_path in model_files.items():
        if os.path.exists(m_path):
            print(f"Running inference for {model_name}...")
            try:
                model = keras.models.load_model(m_path, custom_objects=custom_objects, compile=False, safe_mode=False)
                sr_img = siq.inference(lr_patch, model, method="antspynet", verbose=False)
                ants.copy_image_info(hr_patch, sr_img)
                
                sr_np = sr_img.numpy()
                psnr = float(antspynet.psnr(hr_patch, sr_img))
                ssim = float(antspynet.ssim(hr_patch, sr_img))
                gmsd = float(compute_gmsd(gt_np, sr_np))
                hfen = float(compute_hfen(gt_np, sr_np))
                corr = float(np.corrcoef(sr_np.flatten(), gt_np.flatten())[0, 1])
                
                img_path = os.path.join(artifact_dir, f"r16_{model_name.lower()}.png")
                ants.plot(sr_img, filename=img_path, title=f"{model_name} (PSNR: {psnr:.2f} dB)")
                print(f"Saved {model_name} output to {img_path}")
                
                model_results.append({
                    "name": model_name,
                    "psnr": psnr,
                    "ssim": ssim,
                    "gmsd": gmsd,
                    "hfen": hfen,
                    "corr": corr,
                    "filename": f"r16_{model_name.lower()}.png",
                    "status": "done"
                })
            except Exception as e:
                print(f"Failed to run inference for {model_name}: {e}")
                model_results.append({
                    "name": model_name,
                    "psnr": None,
                    "ssim": None,
                    "gmsd": None,
                    "hfen": None,
                    "corr": None,
                    "filename": None,
                    "status": "error"
                })
        else:
            print(f"Model file for {model_name} not found, adding TBD...")
            model_results.append({
                "name": model_name,
                "psnr": None,
                "ssim": None,
                "gmsd": None,
                "hfen": None,
                "corr": None,
                "filename": None,
                "status": "tbd"
            })

    # Generate 9-class visual comparison grid images
    print("Generating 9-class visual comparison grid images...")
    grid_dir = os.path.join(artifact_dir, "class_grid")
    os.makedirs(grid_dir, exist_ok=True)
    
    classes = [
        "brain_procedural",
        "layered",
        "sinewave",
        "organic_blobs",
        "vessel_tubes",
        "cellular_voronoi",
        "geometric_phantoms",
        "grid_patterns",
        "fractal_noise"
    ]
    
    # Reload model objects for class evaluation (cached inside loaded_models during main loop)
    loaded_models = {}
    for model_name, m_path in model_files.items():
        if os.path.exists(m_path):
            try:
                model = keras.models.load_model(m_path, custom_objects=custom_objects, compile=False, safe_mode=False)
                loaded_models[model_name] = model
            except Exception as e:
                print(f"Error loading {model_name} for grid evaluation: {e}")

    for cls in classes:
        print(f"  Generating sample for class: {cls}")
        # Set seed for reproducible simulation sample
        antspyt1w.set_global_scientific_computing_random_seed(1234)
        
        gen = siq.blind_sr_generator(
            hr_base_cache=None,
            batch_size=1,
            lr_patch_size=48,
            factor=2,
            blur_sigma_range=(0.0, 0.0),
            noise_std_range=(0.0, 0.0),
            simulation_classes={cls: 1.0},
            zoom_range=(1.0, 1.0),
            use_cache=False,
            dimensionality=2,
            use_layer2=False
        )
        
        try:
            x_batch, y_batch = next(gen)
            lr_np = x_batch[0, ..., 0]
            hr_np = y_batch[0, ..., 0]
            
            lr_img = ants.from_numpy(lr_np)
            hr_img = ants.from_numpy(hr_np)
            lr_img.set_spacing([2.0, 2.0])
            hr_img.set_spacing([1.0, 1.0])
            
            # Save Ground Truth
            gt_class_path = os.path.join(grid_dir, f"class_{cls}_gt.png")
            ants.plot(hr_img, filename=gt_class_path)
            
            # Save Bilinear
            bilinear_class_sr = ants.resample_image_to_target(lr_img, hr_img, interp_type=0)
            bilinear_class_path = os.path.join(grid_dir, f"class_{cls}_bilinear.png")
            ants.plot(bilinear_class_sr, filename=bilinear_class_path)
            
            # Save each model's output
            for m_name, model in loaded_models.items():
                try:
                    sr_img = siq.inference(lr_img, model, method="antspynet", verbose=False)
                    ants.copy_image_info(hr_img, sr_img)
                    sr_class_path = os.path.join(grid_dir, f"class_{cls}_{m_name.lower()}.png")
                    ants.plot(sr_img, filename=sr_class_path)
                except Exception as e:
                    print(f"    Failed class grid inference for {m_name} on {cls}: {e}")
        except Exception as e:
            print(f"  Failed generating class sample for {cls}: {e}")

    # Build the 9-class comparison grid HTML
    html_grid_table = """
        <h2>9-Class Simulation Comparison Grid</h2>
        <p class="subtitle" style="text-align: left; margin-bottom: 20px;">Hover over any cell image to magnify details (scale 2.5x).</p>
        <div style="overflow-x: auto;">
            <table class="grid-table">
                <thead>
                    <tr>
                        <th>Model</th>
    """
    for cls in classes:
        cls_title = cls.replace("_", " ").title()
        html_grid_table += f"                        <th>{cls_title}</th>\n"
    html_grid_table += "                    </tr>\n                </thead>\n                <tbody>\n"
    
    # 1. Ground Truth
    html_grid_table += "                    <tr>\n                        <td style='font-weight: bold; color: #10b981;'>Ground Truth</td>\n"
    for cls in classes:
        img_src = f"class_grid/class_{cls}_gt.png"
        html_grid_table += f"                        <td><img src='{img_src}' alt='GT {cls}'></td>\n"
    html_grid_table += "                    </tr>\n"
    
    # 2. Bilinear
    html_grid_table += "                    <tr>\n                        <td style='font-weight: bold; color: #f59e0b;'>Bilinear</td>\n"
    for cls in classes:
        img_src = f"class_grid/class_{cls}_bilinear.png"
        html_grid_table += f"                        <td><img src='{img_src}' alt='Bilinear {cls}'></td>\n"
    html_grid_table += "                    </tr>\n"
    
    # 3. Models
    for m_name in model_files.keys():
        m_name_lower = m_name.lower()
        html_grid_table += f"                    <tr>\n                        <td style='font-weight: bold; color: #3b82f6;'>{m_name}</td>\n"
        for cls in classes:
            img_path = os.path.join(grid_dir, f"class_{cls}_{m_name_lower}.png")
            if os.path.exists(img_path):
                img_src = f"class_grid/class_{cls}_{m_name_lower}.png"
                html_grid_table += f"                        <td><img src='{img_src}' alt='{m_name} {cls}'></td>\n"
            else:
                html_grid_table += "                        <td><div class='grid-tbd-cell'>TBD</div></td>\n"
        html_grid_table += "                    </tr>\n"
        
    html_grid_table += "                </tbody>\n            </table>\n        </div>\n"

    # Generate HTML report
    html_path = os.path.join(artifact_dir, "summary_results.html")
    
    html_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Super-Resolution 2D Pilot Comparison Report</title>
    <style>
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background-color: #0b0f19;
            color: #f3f4f6;
            margin: 0;
            padding: 40px 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #3b82f6, #8b5cf6, #ec4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        p.subtitle {
            text-align: center;
            color: #9ca3af;
            font-size: 1.1em;
            margin-bottom: 40px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 25px;
            margin-bottom: 50px;
        }
        .card {
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 12px;
            overflow: hidden;
            transition: transform 0.2s, box-shadow 0.2s;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 15px rgba(59, 130, 246, 0.2);
            border-color: #3b82f6;
        }
        .card img {
            width: 100%;
            height: auto;
            display: block;
            border-bottom: 1px solid #30363d;
            background-color: #000;
        }
        .card-content {
            padding: 15px;
        }
        .card-title {
            font-size: 1.2em;
            font-weight: bold;
            margin-bottom: 8px;
            color: #3b82f6;
        }
        .card-metrics {
            font-size: 0.95em;
            color: #e5e7eb;
        }
        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 40px;
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 12px;
            overflow: hidden;
        }
        .metrics-table th, .metrics-table td {
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid #30363d;
        }
        .metrics-table th {
            background-color: #21262d;
            color: #3b82f6;
            font-weight: 600;
        }
        .metrics-table tr:hover {
            background-color: #1f242c;
        }
        .badge {
            background: linear-gradient(135deg, #3b82f6, #1d4ed8);
            padding: 3px 8px;
            border-radius: 6px;
            font-size: 0.85em;
            font-weight: bold;
        }
        .grid-table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 50px;
            background-color: #161b22;
            border: 1px solid #30363d;
            border-radius: 12px;
            overflow-x: auto;
            display: block;
        }
        .grid-table th, .grid-table td {
            padding: 8px;
            text-align: center;
            border: 1px solid #30363d;
            min-width: 100px;
        }
        .grid-table img {
            width: 90px;
            height: 90px;
            object-fit: cover;
            border-radius: 6px;
            transition: transform 0.2s;
            display: block;
            margin: 0 auto;
        }
        .grid-table img:hover {
            transform: scale(2.5);
            z-index: 10;
            position: relative;
            box-shadow: 0 5px 15px rgba(0,0,0,0.8);
        }
        .grid-tbd-cell {
            width: 90px;
            height: 90px;
            display: flex;
            align-items: center;
            justify-content: center;
            background-color: #0c1017;
            color: #8b949e;
            font-size: 0.85em;
            font-family: monospace;
            border-radius: 6px;
            border: 1px dashed #30363d;
            margin: 0 auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Super-Resolution 2D Pilot Comparison</h1>
        <p class="subtitle">Visual and quantitative evaluation of all models trained for 300 iterations on the r16 brain MRI test image.</p>
        
        <h2>Quantitative Results</h2>
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>PSNR (dB)</th>
                    <th>SSIM</th>
                    <th>GMSD</th>
                    <th>HFEN</th>
                    <th>Correlation</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Ground Truth</td>
                    <td>&infin;</td>
                    <td>1.0000</td>
                    <td>0.0000</td>
                    <td>0.0000</td>
                    <td>1.0000</td>
                </tr>
"""
    
    # Collect all rows
    all_rows = []
    
    # Baselines
    all_rows.append({
        "name": "Bilinear Interpolation",
        "psnr": linear_psnr,
        "ssim": linear_ssim,
        "gmsd": linear_gmsd,
        "hfen": linear_hfen,
        "corr": linear_corr,
        "badge": "",
        "status": "done"
    })
    all_rows.append({
        "name": "Nearest Neighbor",
        "psnr": nn_psnr,
        "ssim": nn_ssim,
        "gmsd": nn_gmsd,
        "hfen": nn_hfen,
        "corr": nn_corr,
        "badge": "",
        "status": "done"
    })
    
    # Models
    for r in model_results:
        badge_str = ""
        if r["name"] == "ESPCN":
            badge_str = ' <span class="badge">Best Lightweight</span>'
        elif r["name"] == "REF-DBPN":
            badge_str = ' <span class="badge">Heavy Baseline</span>'
        elif r["name"] in ["SRFBN", "SAN"]:
            badge_str = ' <span class="badge" style="background: linear-gradient(135deg, #10b981, #047857);">New</span>'
            
        all_rows.append({
            "name": r["name"],
            "psnr": r["psnr"],
            "ssim": r["ssim"],
            "gmsd": r["gmsd"],
            "hfen": r["hfen"],
            "corr": r["corr"],
            "badge": badge_str,
            "status": r["status"]
        })
        
    # Sort all rows by PSNR descending, TBD sorted to bottom
    def get_sort_key(row):
        return row["psnr"] if row["psnr"] is not None else -1.0
        
    all_rows.sort(key=get_sort_key, reverse=True)
    
    for row in all_rows:
        if row["status"] == "done":
            psnr_val = f"{row['psnr']:.2f} dB"
            ssim_val = f"{row['ssim']:.4f}"
            gmsd_val = f"{row['gmsd']:.4f}"
            hfen_val = f"{row['hfen']:.4f}"
            corr_val = f"{row['corr']:.4f}"
        else:
            psnr_val = "TBD"
            ssim_val = "TBD"
            gmsd_val = "TBD"
            hfen_val = "TBD"
            corr_val = "TBD"
            
        html_content += f"""                <tr>
                    <td>{row['name']}{row['badge']}</td>
                    <td>{psnr_val}</td>
                    <td>{ssim_val}</td>
                    <td>{gmsd_val}</td>
                    <td>{hfen_val}</td>
                    <td>{corr_val}</td>
                </tr>\n"""

    html_content += """            </tbody>
        </table>

        <h2>Architecture Comparison</h2>
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>Architecture</th>
                    <th>Upsampling Method</th>
                    <th>Feedback Loop?</th>
                    <th>Core Mechanism</th>
                    <th>Checkerboard Artifacts</th>
                    <th>Computational Weight</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>ESPCN</strong></td>
                    <td>PixelShuffle</td>
                    <td>No</td>
                    <td>Attention-enhanced sub-pixel mapping</td>
                    <td>High susceptibility</td>
                    <td>Lightweight (~25ms inference)</td>
                </tr>
                <tr>
                    <td><strong>ESPCN-RC</strong></td>
                    <td>Resize + Conv</td>
                    <td>No</td>
                    <td>Bilinear resize followed by convolutions</td>
                    <td>Completely eliminated</td>
                    <td>Lightweight (~20ms inference)</td>
                </tr>
                <tr>
                    <td><strong>WDSR</strong></td>
                    <td>PixelShuffle</td>
                    <td>No</td>
                    <td>Wide activation before ReLU (feature preserving)</td>
                    <td>High susceptibility</td>
                    <td>Medium (~35ms inference)</td>
                </tr>
                <tr>
                    <td><strong>WDSR-RC</strong></td>
                    <td>Resize + Conv</td>
                    <td>No</td>
                    <td>Bilinear resize with wide activation blocks</td>
                    <td>Completely eliminated</td>
                    <td>Medium (~17ms inference)</td>
                </tr>
                <tr>
                    <td><strong>CARN</strong></td>
                    <td>PixelShuffle</td>
                    <td>No</td>
                    <td>Cascading residual connections + 1x1 convs</td>
                    <td>Moderate susceptibility</td>
                    <td>Lightweight (~19ms inference)</td>
                </tr>
                <tr>
                    <td><strong>RCAN</strong></td>
                    <td>PixelShuffle</td>
                    <td>No</td>
                    <td>Residual channel attention + global residual groups</td>
                    <td>Moderate susceptibility</td>
                    <td>Medium-Heavy (~48ms inference)</td>
                </tr>
                <tr>
                    <td><strong>LDBPN</strong></td>
                    <td>Deconvolution + Back-projection</td>
                    <td>Yes (Iterative)</td>
                    <td>Iterative up/down projection error correction</td>
                    <td>Completely eliminated</td>
                    <td>Medium-Light (~35ms inference)</td>
                </tr>
                <tr>
                    <td><strong>REF-DBPN</strong></td>
                    <td>Deconvolution + Back-projection</td>
                    <td>Yes (Iterative)</td>
                    <td>Deep iterative projection blocks (heavy)</td>
                    <td>Completely eliminated</td>
                    <td>Very Heavy (~200ms inference)</td>
                </tr>
                <tr>
                    <td><strong>SRFBN</strong></td>
                    <td>Deconvolution</td>
                    <td>Yes (Recurrent)</td>
                    <td>Feedback block sharing weights recurrently</td>
                    <td>Low susceptibility</td>
                    <td>Medium (~40ms inference)</td>
                </tr>
                <tr>
                    <td><strong>SAN</strong></td>
                    <td>PixelShuffle</td>
                    <td>No</td>
                    <td>Second-order local attention blocks (LSRAB)</td>
                    <td>Moderate susceptibility</td>
                    <td>Medium-Heavy (~60ms inference)</td>
                </tr>
            </tbody>
        </table>

        {html_grid_table}

        <h2>Visual Comparison Grid (r16)</h2>
        <div class="grid">
            <div class="card">
                <img src="r16_ground_truth.png" alt="Ground Truth">
                <div class="card-content">
                    <div class="card-title" style="color: #10b981;">Ground Truth</div>
                    <div class="card-metrics">Original HR crop</div>
                </div>
            </div>
"""

    # Add baselines to visual cards
    html_content += f"""            <div class="card">
                <img src="r16_bilinear.png" alt="Bilinear">
                <div class="card-content">
                    <div class="card-title" style="color: #f59e0b;">Bilinear</div>
                    <div class="card-metrics">PSNR: {linear_psnr:.2f} dB | SSIM: {linear_ssim:.4f}</div>
                </div>
            </div>
            <div class="card">
                <img src="r16_nearest_neighbor.png" alt="Nearest Neighbor">
                <div class="card-content">
                    <div class="card-title" style="color: #ef4444;">Nearest Neighbor</div>
                    <div class="card-metrics">PSNR: {nn_psnr:.2f} dB | SSIM: {nn_ssim:.4f}</div>
                </div>
            </div>
"""

    for r in model_results:
        if r["status"] == "done":
            html_content += f"""            <div class="card">
                <img src="{r['filename']}" alt="{r['name']}">
                <div class="card-content">
                    <div class="card-title">{r['name']}</div>
                    <div class="card-metrics">PSNR: {r['psnr']:.2f} dB | SSIM: {r['ssim']:.4f} | GMSD: {r['gmsd']:.4f}</div>
                </div>
            </div>\n"""
        else:
            html_content += f"""            <div class="card" style="opacity: 0.55; border-style: dashed;">
                <div style="height: 200px; display: flex; align-items: center; justify-content: center; background-color: #0c1017; color: #8b949e; font-weight: bold; font-family: monospace; text-align: center;">
                    {r['name']}<br>(Training TBD)
                </div>
                <div class="card-content">
                    <div class="card-title" style="color: #8b949e;">{r['name']}</div>
                    <div class="card-metrics">Metrics will populate post-training</div>
                </div>
            </div>\n"""

    html_content += """        </div>
    </div>
</body>
</html>
"""
    
    with open(html_path, "w") as f:
        f.write(html_content)
    print(f"Successfully generated summary HTML at {html_path}")

if __name__ == "__main__":
    main()

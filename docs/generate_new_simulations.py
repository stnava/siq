import os
import sys
# Ensure the local workspace is prioritized in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Configure Keras PyTorch backend
os.environ["KERAS_BACKEND"] = "torch"

import numpy as np
import ants
import siq



def apply_layer2_vol_effects(img_np):
    # Contrast inversion
    if np.random.choice([True, False]):
        img_np = 1.0 - img_np
    # Quadratic bias field
    grid_bias = [np.linspace(-1, 1, s) for s in img_np.shape]
    mesh_bias = np.meshgrid(*grid_bias, indexing="ij")
    cx = np.random.uniform(-0.4, 0.4)
    cy = np.random.uniform(-0.4, 0.4)
    strength = np.random.uniform(0.12, 0.22)
    dist_sq = (mesh_bias[0] - cx)**2 + (mesh_bias[1] - cy)**2
    bias_field = 1.0 - strength * dist_sq
    return np.clip(img_np * bias_field, 0.0, 1.0)

def main():
    print("Setting up directories...")
    docs_dir = "/Users/stnava/Library/Mobile Documents/com~apple~CloudDocs/code/siq/docs"
    gallery_dir = os.path.join(docs_dir, "images", "simulation_gallery")
    os.makedirs(gallery_dir, exist_ok=True)

    hr_shape = (128, 128)
    lr_shape = (64, 64)

    classes = [
        ("brain_procedural", "Standard (Layer 1)", False, "brain_procedural_ex1"),
        ("brain_procedural", "Enhanced (Layer 2)", True, "brain_procedural_ex2"),
        ("layered", "Standard (Layer 1)", False, "layered_ex1"),
        ("layered", "Enhanced (Layer 2)", True, "layered_ex2"),
        ("sinewave", "Standard (Layer 1)", False, "sinewave_ex1"),
        ("sinewave", "Enhanced (Layer 2)", True, "sinewave_ex2"),
        ("organic_blobs", "Standard (Layer 1)", False, "organic_blobs_ex1"),
        ("organic_blobs", "Enhanced (Layer 2)", True, "organic_blobs_ex2")
    ]

    print("Generating simulation examples...")
    for sim_type, name, use_layer2, prefix in classes:
        print(f"Generating {sim_type} - {name}...")
        
        # 1. Generate High-Res base
        if sim_type == "brain_procedural":
            y_img = siq.simulate_brain_procedural(hr_shape, use_layer2=use_layer2)
        elif sim_type == "layered":
            y_img = siq.simulate_layered(hr_shape, use_layer2=use_layer2)
        elif sim_type == "sinewave":
            y_img = siq.simulate_sinewave(hr_shape, use_layer2=use_layer2)
        else: # organic_blobs
            y_img = siq.simulate_image_multi_scale(hr_shape, scale_range=(0.8, 1.2), n_levels_range=(4, 8))
            if use_layer2:
                o2_np = y_img.numpy().astype("float32")
                o2_np = (o2_np - o2_np.min()) / (o2_np.max() - o2_np.min() + 1e-8)
                o2_np = apply_layer2_vol_effects(o2_np)
                y_img = ants.from_numpy(o2_np)

        # Ensure range [0, 1]
        y_np = y_img.numpy().astype("float32")
        y_min, y_max = y_np.min(), y_np.max()
        if y_max > y_min:
            y_np = (y_np - y_min) / (y_max - y_min + 1e-8)
        else:
            y_np = y_np - y_min
        y_img = ants.from_numpy(y_np)

        # 2. Blur
        sigma = np.random.uniform(0.6, 1.2) if not use_layer2 else np.random.uniform(1.2, 2.0)
        lr_large = ants.smooth_image(y_img, sigma)

        # 3. Downsample to LR Shape
        x_img = ants.resample_image(lr_large, lr_shape, use_voxels=True, interp_type=0)

        # 4. Normalize LR and Add Rician Noise
        lr_np = x_img.numpy().astype("float32")
        lr_min, lr_max = lr_np.min(), lr_np.max()
        if lr_max > lr_min:
            lr_np = (lr_np - lr_min) / (lr_max - lr_min + 1e-8)
        else:
            lr_np = lr_np - lr_min
        
        noise_std = np.random.uniform(0.005, 0.015) if not use_layer2 else np.random.uniform(0.02, 0.04)
        lr_np = siq.add_rician_noise(lr_np, noise_std)
        x_img = ants.from_numpy(lr_np)

        # Set physical spacing to prevent boundary artifacting
        ants.set_spacing(x_img, (2.0, 2.0))
        ants.set_spacing(y_img, (1.0, 1.0))

        # 5. Upsample LR back to HR shape for visual comparison plotting
        lr_upsampled = ants.resample_image_to_target(x_img, y_img, interp_type="linear")

        # Save plot files
        hr_path = os.path.join(gallery_dir, f"{prefix}_hr.png")
        lr_path = os.path.join(gallery_dir, f"{prefix}_lr.png")
        
        ants.plot(y_img, filename=hr_path)
        ants.plot(lr_upsampled, filename=lr_path)
        print(f"  Saved {prefix}_hr.png & {prefix}_lr.png")

    # Now generate the HTML Report
    html_path = os.path.join(docs_dir, "simulated_examples.html")
    print(f"Generating HTML report at {html_path}...")

    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Procedural MRI & Geometric Simulation Gallery</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        :root {
            --bg-color: #0b0f19;
            --card-bg: rgba(20, 30, 55, 0.5);
            --border-color: rgba(255, 255, 255, 0.08);
            --text-color: #f3f4f6;
            --text-muted: #9ca3af;
            --primary: #3b82f6;
            --primary-glow: rgba(59, 130, 246, 0.15);
            --accent: #10b981;
            --accent-glow: rgba(16, 185, 129, 0.15);
            --purple: #a855f7;
            --border-radius: 20px;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-color);
            font-family: 'Outfit', sans-serif;
            margin: 0;
            padding: 40px 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
            overflow-x: hidden;
        }

        header {
            text-align: center;
            max-width: 900px;
            margin-bottom: 50px;
        }

        h1 {
            font-size: 3rem;
            font-weight: 700;
            margin-bottom: 15px;
            background: linear-gradient(135deg, #60a5fa, #a855f7, #34d399);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.03em;
        }

        p.subtitle {
            font-size: 1.2rem;
            color: var(--text-muted);
            line-height: 1.6;
            margin-bottom: 30px;
        }

        .stats-bar {
            display: flex;
            gap: 20px;
            justify-content: center;
            flex-wrap: wrap;
            margin-bottom: 40px;
        }

        .stat-badge {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid var(--border-color);
            padding: 8px 20px;
            border-radius: 99px;
            font-size: 0.95rem;
            display: flex;
            align-items: center;
            gap: 10px;
            backdrop-filter: blur(8px);
        }

        .stat-badge span.dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: var(--accent);
            box-shadow: 0 0 10px var(--accent);
        }

        .class-section {
            background-color: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border-color);
            border-radius: var(--border-radius);
            padding: 35px;
            margin-bottom: 50px;
            width: 100%;
            max-width: 1100px;
            box-shadow: 0 15px 35px rgba(0, 0, 0, 0.35);
            backdrop-filter: blur(12px);
        }

        .class-header {
            border-bottom: 1px solid rgba(255, 255, 255, 0.06);
            padding-bottom: 15px;
            margin-bottom: 30px;
        }

        .class-title {
            font-size: 1.8rem;
            font-weight: 700;
            color: #60a5fa;
            margin: 0 0 8px 0;
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .class-description {
            color: var(--text-muted);
            margin: 0;
            font-size: 1.05rem;
            line-height: 1.5;
        }

        .examples-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 35px;
        }

        @media (max-width: 992px) {
            .examples-grid {
                grid-template-columns: 1fr;
            }
        }

        .example-card {
            background: rgba(30, 41, 59, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .example-card:hover {
            transform: translateY(-5px);
            border-color: rgba(96, 165, 250, 0.3);
            box-shadow: 0 15px 30px rgba(96, 165, 250, 0.08);
        }

        .example-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 18px;
        }

        .example-title {
            font-size: 1.15rem;
            font-weight: 600;
            color: #fff;
        }

        .example-badge {
            font-size: 0.8rem;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 6px;
            text-transform: uppercase;
        }

        .badge-standard {
            background: rgba(16, 185, 129, 0.1);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.25);
        }

        .badge-enhanced {
            background: rgba(168, 85, 247, 0.1);
            color: #c084fc;
            border: 1px solid rgba(168, 85, 247, 0.25);
        }

        .image-pair {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }

        .image-container {
            position: relative;
            background: #000;
            border-radius: 10px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.05);
            aspect-ratio: 1;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: border-color 0.2s;
        }

        .image-container:hover {
            border-color: var(--primary);
        }

        .image-container img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            display: block;
        }

        .image-label {
            position: absolute;
            bottom: 10px;
            left: 10px;
            background: rgba(0, 0, 0, 0.75);
            backdrop-filter: blur(4px);
            color: #fff;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            letter-spacing: 0.05em;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .image-label.hr {
            border-left: 3px solid var(--accent);
        }

        .image-label.lr {
            border-left: 3px solid var(--primary);
        }

        .back-link {
            margin-top: 20px;
            color: #60a5fa;
            text-decoration: none;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 8px;
            transition: color 0.2s;
        }

        .back-link:hover {
            color: #a78bfa;
        }
    </style>
</head>
<body>

    <header>
        <h1>Procedural Simulation Verification Gallery</h1>
        <p class="subtitle">
            A comprehensive visual report detailing all classes of synthetic images used to pre-train SIQ super-resolution models. We display exactly two examples per class: one standard (Layer 1) baseline, and one augmented (Layer 2) with coordinate shear, sinusoidal warping, ripple textures, modality inversion, or quadratic scanner bias fields.
        </p>
        <div class="stats-bar">
            <div class="stat-badge">
                <span class="dot"></span>
                <span>Image Size: 128x128 HR / 64x64 LR</span>
            </div>
            <div class="stat-badge">
                <span class="dot" style="background-color: var(--primary); box-shadow: 0 0 10px var(--primary);"></span>
                <span>Rician Noise &amp; Stochastic Spacing</span>
            </div>
            <div class="stat-badge">
                <span class="dot" style="background-color: var(--purple); box-shadow: 0 0 10px var(--purple);"></span>
                <span>4 Distinct Simulation Classes</span>
            </div>
        </div>
        <a href="index.html" class="back-link">&larr; Back to Documentation Home</a>
    </header>

    <!-- CLASS 1: BRAIN PROCEDURAL -->
    <div class="class-section">
        <div class="class-header">
            <div class="class-title">1. Brain Procedural (<code>brain_procedural</code>)</div>
            <p class="class-description">
                Simulates multi-class gray matter, white matter, ventricles, and CSF boundaries using modulated ellipsoids and trigonometric folding. Mimics cortical sulci/gyri folding configurations.
            </p>
        </div>
        <div class="examples-grid">
            <!-- Example 1 -->
            <div class="example-card">
                <div class="example-header">
                    <span class="example-title">Example 1 (Standard)</span>
                    <span class="example-badge badge-standard">Standard</span>
                </div>
                <div class="image-pair">
                    <div class="image-container">
                        <img src="images/simulation_gallery/brain_procedural_ex1_hr.png" alt="Brain Procedural Standard HR">
                        <span class="image-label hr">HR GT</span>
                    </div>
                    <div class="image-container">
                        <img src="images/simulation_gallery/brain_procedural_ex1_lr.png" alt="Brain Procedural Standard LR">
                        <span class="image-label lr">LR Blurry</span>
                    </div>
                </div>
            </div>
            <!-- Example 2 -->
            <div class="example-card">
                <div class="example-header">
                    <span class="example-title">Example 2 (Enhanced)</span>
                    <span class="example-badge badge-enhanced">Enhanced L2</span>
                </div>
                <div class="image-pair">
                    <div class="image-container">
                        <img src="images/simulation_gallery/brain_procedural_ex2_hr.png" alt="Brain Procedural Enhanced HR">
                        <span class="image-label hr">HR GT</span>
                    </div>
                    <div class="image-container">
                        <img src="images/simulation_gallery/brain_procedural_ex2_lr.png" alt="Brain Procedural Enhanced LR">
                        <span class="image-label lr">LR Blurry</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- CLASS 2: LAYERED STRIPS -->
    <div class="class-section">
        <div class="class-header">
            <div class="class-title">2. Layered (<code>layered</code>)</div>
            <p class="class-description">
                Generates planar boundary interfaces rotated stochastically in space to simulate striated tissue layers such as muscle fibers, cranial bone boundaries, or meningeal sheets.
            </p>
        </div>
        <div class="examples-grid">
            <!-- Example 1 -->
            <div class="example-card">
                <div class="example-header">
                    <span class="example-title">Example 1 (Standard)</span>
                    <span class="example-badge badge-standard">Standard</span>
                </div>
                <div class="image-pair">
                    <div class="image-container">
                        <img src="images/simulation_gallery/layered_ex1_hr.png" alt="Layered Standard HR">
                        <span class="image-label hr">HR GT</span>
                    </div>
                    <div class="image-container">
                        <img src="images/simulation_gallery/layered_ex1_lr.png" alt="Layered Standard LR">
                        <span class="image-label lr">LR Blurry</span>
                    </div>
                </div>
            </div>
            <!-- Example 2 -->
            <div class="example-card">
                <div class="example-header">
                    <span class="example-title">Example 2 (Enhanced)</span>
                    <span class="example-badge badge-enhanced">Enhanced L2</span>
                </div>
                <div class="image-pair">
                    <div class="image-container">
                        <img src="images/simulation_gallery/layered_ex2_hr.png" alt="Layered Enhanced HR">
                        <span class="image-label hr">HR GT</span>
                    </div>
                    <div class="image-container">
                        <img src="images/simulation_gallery/layered_ex2_lr.png" alt="Layered Enhanced LR">
                        <span class="image-label lr">LR Blurry</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- CLASS 3: SINEWAVE PATTERNS -->
    <div class="class-section">
        <div class="class-header">
            <div class="class-title">3. Sinewave (<code>sinewave</code>)</div>
            <p class="class-description">
                Intersects multiple periodic sine waves oriented along random angles to mimic regular structures, vascular lattices, or scanning grid line patterns.
            </p>
        </div>
        <div class="examples-grid">
            <!-- Example 1 -->
            <div class="example-card">
                <div class="example-header">
                    <span class="example-title">Example 1 (Standard)</span>
                    <span class="example-badge badge-standard">Standard</span>
                </div>
                <div class="image-pair">
                    <div class="image-container">
                        <img src="images/simulation_gallery/sinewave_ex1_hr.png" alt="Sinewave Standard HR">
                        <span class="image-label hr">HR GT</span>
                    </div>
                    <div class="image-container">
                        <img src="images/simulation_gallery/sinewave_ex1_lr.png" alt="Sinewave Standard LR">
                        <span class="image-label lr">LR Blurry</span>
                    </div>
                </div>
            </div>
            <!-- Example 2 -->
            <div class="example-card">
                <div class="example-header">
                    <span class="example-title">Example 2 (Enhanced)</span>
                    <span class="example-badge badge-enhanced">Enhanced L2</span>
                </div>
                <div class="image-pair">
                    <div class="image-container">
                        <img src="images/simulation_gallery/sinewave_ex2_hr.png" alt="Sinewave Enhanced HR">
                        <span class="image-label hr">HR GT</span>
                    </div>
                    <div class="image-container">
                        <img src="images/simulation_gallery/sinewave_ex2_lr.png" alt="Sinewave Enhanced LR">
                        <span class="image-label lr">LR Blurry</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- CLASS 4: ORGANIC BLOBS -->
    <div class="class-section">
        <div class="class-header">
            <div class="class-title">4. Organic Blobs (<code>organic_blobs</code>)</div>
            <p class="class-description">
                Compiles multi-scale thresholded white-noise fields smoothed by Gaussian filters to model general organic shapes, tumors, lesions, or arbitrary anatomical boundaries.
            </p>
        </div>
        <div class="examples-grid">
            <!-- Example 1 -->
            <div class="example-card">
                <div class="example-header">
                    <span class="example-title">Example 1 (Standard)</span>
                    <span class="example-badge badge-standard">Standard</span>
                </div>
                <div class="image-pair">
                    <div class="image-container">
                        <img src="images/simulation_gallery/organic_blobs_ex1_hr.png" alt="Organic Blobs Standard HR">
                        <span class="image-label hr">HR GT</span>
                    </div>
                    <div class="image-container">
                        <img src="images/simulation_gallery/organic_blobs_ex1_lr.png" alt="Organic Blobs Standard LR">
                        <span class="image-label lr">LR Blurry</span>
                    </div>
                </div>
            </div>
            <!-- Example 2 -->
            <div class="example-card">
                <div class="example-header">
                    <span class="example-title">Example 2 (Enhanced)</span>
                    <span class="example-badge badge-enhanced">Enhanced L2</span>
                </div>
                <div class="image-pair">
                    <div class="image-container">
                        <img src="images/simulation_gallery/organic_blobs_ex2_hr.png" alt="Organic Blobs Enhanced HR">
                        <span class="image-label hr">HR GT</span>
                    </div>
                    <div class="image-container">
                        <img src="images/simulation_gallery/organic_blobs_ex2_lr.png" alt="Organic Blobs Enhanced LR">
                        <span class="image-label lr">LR Blurry</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <footer style="margin-top: 50px; color: var(--text-muted); font-size: 0.9rem; text-align: center;">
        <p>&copy; 2026 SIQ Advanced Agentic Coding. All procedural assets simulated on the fly.</p>
    </footer>

</body>
</html>
"""

    with open(html_path, "w") as f:
        f.write(html_content)
    print(f"Successfully generated HTML gallery report at {html_path}")

if __name__ == "__main__":
    main()

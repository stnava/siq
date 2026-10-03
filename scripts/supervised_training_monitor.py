#!/usr/bin/env python3
"""
Autonomous Resilient Training Supervisor for 3D Super-Resolution.

Handles:
1. Detecting and waiting for parallel GPU processes (pytest, syntx, etc.) to finish.
2. Launching and monitoring train_model_refinement.py.
3. Automatically catching any numerical divergence or crash, pruning corrupted step artifacts,
   waiting for competing GPU processes, and resuming from the last verified clean checkpoint.
4. Continuous logging to logs/supervised_{model}_{dim}d.log.
5. Runs until all Stage 3 curriculum steps (5,000 iterations) are 100% complete.
"""

import os
import sys
import time
import glob
import json
import signal
import argparse
import subprocess
import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, os.path.abspath("."))

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
SUPERVISOR_LOG = os.path.join(LOG_DIR, "supervised_training.log")


def log(msg, log_path=SUPERVISOR_LOG):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] [Supervisor] {msg}"
    print(formatted, flush=True)
    with open(log_path, "a") as f:
        f.write(formatted + "\n")


def check_parallel_gpu_processes():
    my_pid = os.getpid()
    parallel = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if p.pid == my_pid:
                continue
            cmd = " ".join(p.info.get('cmdline') or [])
            if not cmd:
                continue
            # Detect active competing workloads
            is_competing = (
                "pytest" in cmd
                and "supervised_training_monitor" not in cmd
            )
            if is_competing:
                parallel.append((p.pid, cmd[:90]))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return parallel


def wait_for_parallel_gpu_processes(min_idle_seconds=15, check_interval=5, log_path=SUPERVISOR_LOG):
    idle_streak = 0
    log(f"Checking for parallel GPU/Python processes (requiring {min_idle_seconds}s sustained idle)...", log_path)
    while True:
        procs = check_parallel_gpu_processes()
        if len(procs) > 0:
            if idle_streak > 0:
                log(f"Parallel workload detected after {idle_streak}s idle. Resetting wait counter.", log_path)
            idle_streak = 0
            pid, cmd_snip = procs[0]
            log(f"Waiting for parallel GPU process (PID {pid}: {cmd_snip})...", log_path)
            time.sleep(check_interval)
        else:
            idle_streak += check_interval
            if idle_streak >= min_idle_seconds:
                log(f"System GPU confirmed idle ({idle_streak}s clear). Ready to proceed.", log_path)
                break
            time.sleep(check_interval)


def verify_model_weights(keras_path):
    import keras
    import siq
    custom_objects = {
        "PixelShuffle3D": siq.PixelShuffle3D,
        "PixelShuffle2D": siq.PixelShuffle2D,
        "TrilinearUpSampling3D": siq.TrilinearUpSampling3D,
        "LearnableScale": siq.LearnableScale
    }
    try:
        m = keras.models.load_model(keras_path, custom_objects=custom_objects, compile=False)
        x = np.random.randn(1, 16, 16, 16, 1).astype("float32")
        y = m(x)
        val = y.numpy()
        if np.isnan(val).any():
            return False, "contains NaN"
        if np.isinf(val).any():
            return False, "contains Inf"
        return True, f"finite (mean={val.mean():.3f}, min={val.min():.3f}, max={val.max():.3f})"
    except Exception as e:
        return False, str(e)


def find_latest_clean_checkpoint(ckpt_dir, model_type, dim, log_path=SUPERVISOR_LOG):
    csv_path = os.path.join(ckpt_dir, "convergence_history.csv")
    valid_iters = []
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                try:
                    it = int(row['iteration'])
                    psnr = float(row.get('val_psnr', float('nan')))
                    pcs = float(row.get('val_pcs', float('nan')))
                    if not np.isnan(psnr) and not np.isnan(pcs) and psnr > 0:
                        valid_iters.append(it)
                except Exception:
                    pass
        except Exception as e:
            log(f"Warning reading {csv_path}: {e}", log_path)

    valid_iters = sorted(list(set(valid_iters)), reverse=True)

    for it in valid_iters:
        candidates = [
            os.path.join(ckpt_dir, f"{model_type}_{dim}d_step_{it:04d}.keras"),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                is_valid, msg = verify_model_weights(cand)
                if is_valid:
                    log(f"Verified clean step checkpoint at iter {it} ({cand}): {msg}", log_path)
                    return it, cand
                else:
                    log(f"Checkpoint at iter {it} ({cand}) failed verification: {msg}", log_path)

    # Fallback to champion
    champion_cands = [
        os.path.join(ckpt_dir, f"{model_type}_{dim}d_best_pcs.keras"),
        os.path.join(ckpt_dir, f"{model_type}_{dim}d_best_cqs.keras"),
        os.path.join(ckpt_dir, f"{model_type}_{dim}d_best_psnr.keras"),
    ]
    for champion_cand in champion_cands:
        if os.path.exists(champion_cand):
            is_valid, msg = verify_model_weights(champion_cand)
            if is_valid:
                cfg_path = champion_cand.replace(".keras", "_config.json")
                it = 0
                if os.path.exists(cfg_path):
                    try:
                        with open(cfg_path) as f:
                            it = json.load(f).get("iteration", 0)
                    except Exception:
                        pass
                log(f"Verified champion model at iter {it} ({champion_cand}): {msg}", log_path)
                return it, champion_cand

    log(f"No prior verified checkpoints found in {ckpt_dir}. Starting fresh from iteration 0.", log_path)
    return 0, None


def prune_artifacts_above(ckpt_dir, rep_dir, html_report, max_valid_iter, selection_metric="pcs", log_path=SUPERVISOR_LOG):
    log(f"Pruning any corrupted artifacts above iteration {max_valid_iter}...", log_path)
    removed_ckpts = 0
    if os.path.exists(ckpt_dir):
        for f in glob.glob(os.path.join(ckpt_dir, "*step_*")):
            try:
                base = os.path.basename(f)
                step_str = base.split("step_")[1].split(".")[0].split("_")[0]
                step_num = int(step_str)
                if step_num > max_valid_iter:
                    os.remove(f)
                    removed_ckpts += 1
            except Exception:
                pass

    removed_reps = 0
    if os.path.exists(rep_dir):
        for f in glob.glob(os.path.join(rep_dir, "step_*")):
            try:
                base = os.path.basename(f)
                step_str = base.split("step_")[1].split("_")[0]
                step_num = int(step_str)
                if step_num > max_valid_iter:
                    os.remove(f)
                    removed_reps += 1
            except Exception:
                pass

    csv_path = os.path.join(ckpt_dir, "convergence_history.csv")
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            orig_len = len(df)
            df = df[df['iteration'] <= max_valid_iter]
            df.to_csv(csv_path, index=False)
            log(f"Trimmed convergence_history.csv from {orig_len} to {len(df)} rows.", log_path)
        except Exception as e:
            log(f"Warning trimming convergence_history: {e}", log_path)

        try:
            try:
                from scripts.visual_convergence_report import VisualConvergenceReporter
            except ImportError:
                from visual_convergence_report import VisualConvergenceReporter

            reporter = VisualConvergenceReporter(
                checkpoint_dir=ckpt_dir,
                report_dir=rep_dir,
                html_filename=html_report,
                reset_history=False,
                selection_metric=selection_metric,
            )
            reporter.render_html_report()
            log("Refreshed visual convergence HTML report.", log_path)
        except Exception as e:
            log(f"Warning refreshing HTML report: {e}", log_path)

    log(f"Pruning complete. Removed {removed_ckpts} checkpoint files and {removed_reps} report assets.", log_path)


def run_training_cycle(args):
    model_type = args.model
    dim = args.dim
    factor = tuple(args.factor) if len(args.factor) > 1 else tuple(args.factor * dim)
    is_default_factor = (factor == tuple([2] * dim))
    factor_str = "x".join(str(f) for f in factor)

    ckpt_dir = f"checkpoints/{model_type}_{dim}d" if is_default_factor else f"checkpoints/{model_type}_{dim}d_{factor_str}"
    rep_dir = f"reports/{model_type}_{dim}d" if is_default_factor else f"reports/{model_type}_{dim}d_{factor_str}"
    html_report = f"{model_type}_{dim}d_report.html" if is_default_factor else f"{model_type}_{dim}d_{factor_str}_report.html"
    log_file = os.path.join(LOG_DIR, f"supervised_{model_type}_{dim}d.log")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(rep_dir, exist_ok=True)

    log(f"===========================================================", log_file)
    log(f"Starting Supervised Training Session for {model_type.upper()} {dim}D", log_file)
    log(f"  Target iterations: {args.target_iter} | Factor: {factor} | Batch size: {args.batch_size}", log_file)
    log(f"  Checkpoints: {ckpt_dir} | Reports: {rep_dir} | HTML: {html_report}", log_file)
    log(f"===========================================================", log_file)

    cycle = 0
    while True:
        cycle += 1
        log(f"=== Starting Supervision Cycle #{cycle} ===", log_file)

        # 1. Wait for parallel GPU processes
        wait_for_parallel_gpu_processes(min_idle_seconds=15, check_interval=5, log_path=log_file)

        # 2. Find clean checkpoint to start from
        start_iter, ckpt_path = find_latest_clean_checkpoint(ckpt_dir, model_type, dim, log_path=log_file)
        if start_iter >= args.target_iter:
            log(f"Target of {args.target_iter} iterations already reached! (Current: {start_iter})", log_file)
            break

        # 3. Clean any stray artifacts > start_iter
        prune_artifacts_above(ckpt_dir, rep_dir, html_report, start_iter, selection_metric=args.selection_metric, log_path=log_file)

        # 4. Build command
        cmd = [
            sys.executable,
            "scripts/train_model_refinement.py",
            model_type,
            "--dim", str(dim),
            "--factor", *[str(f) for f in factor],
            "--clip-norm", str(args.clip_norm),
            "--stage1-lr", str(args.stage1_lr),
            "--stage2-lr", str(args.stage2_lr),
            "--stage3-lr", str(args.stage3_lr),
            "--batch-size", str(args.batch_size),
            "--stage1-iter", str(args.stage1_iter),
            "--stage2-iter", str(args.stage2_iter),
            "--stage3-iter", str(args.stage3_iter),
            "--target-percep", "65.0",
            "--target-mae", "30.0",
            "--target-tv", "5.0",
            "--edge-weight", str(args.edge_weight),
            "--cbi-weight", str(args.cbi_weight),
            "--linear-blend", "1.0",
            "--selection-metric", args.selection_metric,
            "--checkpoint-freq", str(args.checkpoint_freq),
            "--prefetch-size", "4",
            "--balancer-freq", "25",
            "--smooth-window", "100",
            "--dampening", "0.97",
        ]
        if ckpt_path and os.path.exists(ckpt_path) and start_iter > 0:
            cmd.extend(["--load-model", ckpt_path, "--start-iteration", str(start_iter), "--skip-warmup"])
        else:
            cmd.append("--from-scratch")

        log(f"Launching training process from iteration {start_iter}...", log_file)
        log("Command: " + " ".join(cmd), log_file)

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        divergence_detected = False
        last_logged_iter = start_iter

        with open(log_file, "a") as log_f:
            for line in proc.stdout:
                line_str = line.strip()
                print(line_str, flush=True)
                log_f.write(line)
                log_f.flush()

                if "Stage 1 Iter" in line_str or "Stage 2 Iter" in line_str or "Stage 3 Iter" in line_str:
                    try:
                        it_part = line_str.split("Iter")[1].split("/")[0].strip()
                        last_logged_iter = int(it_part)
                    except Exception:
                        pass

                if last_logged_iter >= args.target_iter:
                    log(f"Target of {args.target_iter} iterations reached! Terminating training process cleanly.", log_file)
                    break

                if "[FATAL] Non-finite loss" in line_str or "Loss: nan" in line_str or "Loss: -7.968750" in line_str:
                    log(f"Divergence signal detected in output: {line_str}", log_file)
                    divergence_detected = True
                    break

        proc.poll()
        if proc.returncode is None:
            log("Terminating training process...", log_file)
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        return_code = proc.returncode
        log(f"Process terminated with return code {return_code}. Divergence detected: {divergence_detected}", log_file)

        if not divergence_detected and (return_code == 0 or last_logged_iter >= args.target_iter):
            log(f"Training cycle completed normally. Reached iter {last_logged_iter}.", log_file)
            if last_logged_iter >= args.target_iter:
                log(f"Goal of {args.target_iter} iterations fully achieved!", log_file)
                break
        else:
            log("Handling divergence / interrupted cycle:", log_file)
            log("1. Finding last verified clean checkpoint before failure...", log_file)
            clean_it, clean_ckpt = find_latest_clean_checkpoint(ckpt_dir, model_type, dim, log_path=log_file)
            log(f"2. Best valid checkpoint is at iter {clean_it}: {clean_ckpt}", log_file)
            prune_artifacts_above(ckpt_dir, rep_dir, html_report, clean_it, selection_metric=args.selection_metric, log_path=log_file)
            log("3. Waiting for any parallel GPU processes to clear before next restart...", log_file)
            wait_for_parallel_gpu_processes(min_idle_seconds=20, check_interval=5, log_path=log_file)
            log(f"4. Ready to restart from verified iteration {clean_it}.", log_file)
            time.sleep(2)

    log(f"=== SUPERVISOR COMPLETE: All {args.target_iter} steps finished successfully ===", log_file)


def main():
    parser = argparse.ArgumentParser(description="Autonomous Resilient Training Supervisor for 3D Super-Resolution")
    parser.add_argument("--model", type=str, default="rcan", help="Model architecture (default: rcan)")
    parser.add_argument("--dim", type=int, default=3, help="Dimensionality (default: 3)")
    parser.add_argument("--factor", type=int, nargs="+", default=[2, 2, 2], help="Upsampling factor (default: 2 2 2)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--target-iter", type=int, default=5000, help="Target total iterations (default: 5000)")
    parser.add_argument("--stage1-iter", type=int, default=500, help="Stage 1 iterations (default: 500)")
    parser.add_argument("--stage2-iter", type=int, default=1500, help="Stage 2 iterations (default: 1500)")
    parser.add_argument("--stage3-iter", type=int, default=5000, help="Stage 3 iterations (default: 5000)")
    parser.add_argument("--stage1-lr", type=float, default=1e-4, help="Stage 1 LR (default: 1e-4)")
    parser.add_argument("--stage2-lr", type=float, default=5e-5, help="Stage 2 LR (default: 5e-5)")
    parser.add_argument("--stage3-lr", type=float, default=1.5e-5, help="Stage 3 LR (default: 1.5e-5)")
    parser.add_argument("--clip-norm", type=float, default=1.0, help="Clip norm (default: 1.0)")
    parser.add_argument("--checkpoint-freq", type=int, default=25, help="Checkpoint frequency (default: 25)")
    parser.add_argument("--selection-metric", type=str, default="pcs", help="Selection metric (default: pcs)")
    parser.add_argument("--edge-weight", type=float, default=2.0, help="Edge/acutance weight in stage 3 (default: 2.0)")
    parser.add_argument("--cbi-weight", type=float, default=0.0, help="CBI weight (default: 0.0 for pixelshuffle)")

    args = parser.parse_args()
    run_training_cycle(args)


if __name__ == "__main__":
    main()

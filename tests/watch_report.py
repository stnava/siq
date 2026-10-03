import os
import sys
import time
try:
    from scripts.render_convergence_dashboard import render_html_dashboard_from_csv
except (ModuleNotFoundError, ImportError):
    try:
        from render_convergence_dashboard import render_html_dashboard_from_csv
    except (ModuleNotFoundError, ImportError):
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from scripts.render_convergence_dashboard import render_html_dashboard_from_csv


def watch_and_protect():
    csv_path = "checkpoints/asdbpn_3d/convergence_history.csv"
    html_path = "asdbpn_3d_report.html"
    
    last_csv_mtime = 0
    last_render_time = time.time()
    
    print("[Watcher] Active report watcher & guardian online.", flush=True)
    
    # Initial render
    render_html_dashboard_from_csv(csv_path, html_path)
    print(f"[Watcher] Initial clean render completed at {time.strftime('%H:%M:%S')}", flush=True)

    while True:
        try:
            time.sleep(1.0)
            
            # Check 1: Did training process overwrite html with stale in-memory code?
            needs_repair = False
            if os.path.exists(html_path):
                # If html does not contain 'Global Step' or contains old 'Iter 0</text>', it was overwritten by task-542!
                with open(html_path, "r", errors="ignore") as f:
                    content = f.read(2048) # quick read header/beginning
                    if "Global Step" not in content and "Global" not in content:
                        # Scan a bit further down where charts live
                        f.seek(0)
                        full_content = f.read()
                        if "Global Step" not in full_content or "Iter 0</text>" in full_content:
                            needs_repair = True
                            print(f"[Watcher] Detected stale in-memory HTML overwrite from training process at {time.strftime('%H:%M:%S')}! Instantly repairing...", flush=True)

            # Check 2: Did CSV get updated with new metrics?
            if os.path.exists(csv_path):
                curr_csv_mtime = os.path.getmtime(csv_path)
                if curr_csv_mtime > last_csv_mtime:
                    last_csv_mtime = curr_csv_mtime
                    needs_repair = True
                    print(f"[Watcher] Detected new checkpoint in CSV at {time.strftime('%H:%M:%S')}. Updating dashboard...", flush=True)

            if needs_repair:
                # Keep 4-way SR upsampled image synced with latest checkpoint output
                current_sr = "reports/asdbpn_3d/val3d_asdbpn_current.png"
                target_sr = "reports/asdbpn_3d/val3d_sr_upsampled.png"
                if os.path.exists(current_sr):
                    import shutil
                    shutil.copyfile(current_sr, target_sr)

                render_html_dashboard_from_csv(csv_path, html_path)
                last_render_time = time.time()
                print(f"[Watcher] Successfully rendered monotonic dashboard at {time.strftime('%H:%M:%S')}", flush=True)

        except Exception as e:
            print(f"[Watcher] Error in watch loop: {e}", file=sys.stderr, flush=True)
            time.sleep(2.0)


if __name__ == "__main__":
    watch_and_protect()

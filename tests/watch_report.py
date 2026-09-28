import os
import time

def watch_and_update():
    csv_path = "checkpoints/asdbpn_3d/convergence_history.csv"
    last_mtime = 0
    if os.path.exists(csv_path):
        last_mtime = os.path.getmtime(csv_path)

    while True:
        time.sleep(5)
        if os.path.exists(csv_path):
            current_mtime = os.path.getmtime(csv_path)
            if current_mtime > last_mtime:
                last_mtime = current_mtime
                time.sleep(1)  # allow file write to finish
                try:
                    from tests.visual_convergence_report import VisualConvergenceReporter
                    rep = VisualConvergenceReporter(
                        checkpoint_dir="checkpoints/asdbpn_3d",
                        report_dir="reports/asdbpn_3d",
                        html_filename="asdbpn_3d_report.html"
                    )
                    rep.render_html_report()
                except Exception as e:
                    pass

if __name__ == "__main__":
    watch_and_update()

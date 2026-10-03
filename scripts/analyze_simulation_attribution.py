#!/usr/bin/env python
"""Analyze per-sample simulation class provenance and loss attribution.

Reads attribution CSV produced by siq.curriculum (via --provenance-log)
and computes empirical value metrics, feature engagement, and curriculum
learning dynamics for each procedural class.
"""
import argparse
import json
import os
import numpy as np
import pandas as pd


def analyze_attribution(csv_path, out_dir=None):
    df = pd.read_csv(csv_path)
    total_samples = len(df)
    print(f"[siq] Loaded {total_samples} samples from {csv_path}")

    classes = sorted(df["class"].unique())
    n_classes = len(classes)
    total_loss_all = df["loss_total"].sum()

    summary_rows = []
    stage_breakdown = {}

    for c in classes:
        sub = df[df["class"] == c]
        n_c = len(sub)
        loss_share_pct = (sub["loss_total"].sum() / (total_loss_all + 1e-12)) * 100.0

        # Mean raw component losses
        m_tot = float(sub["loss_total"].mean())
        med_tot = float(sub["loss_total"].median())
        m_l1 = float(sub["loss_l1"].mean())
        m_feat = float(sub["loss_feat"].mean())
        m_tv = float(sub["loss_tv"].mean())
        m_edge = float(sub["loss_edge"].mean())

        # Relative feature share: weighted feat / total loss
        weighted_feat = (sub["w_feat"] * sub["loss_feat"]).mean()
        weighted_tot = sub["loss_total"].mean()
        feat_share_pct = (weighted_feat / (weighted_tot + 1e-12)) * 100.0

        # Per-stage loss evolution
        stages_present = sorted(sub["stage"].unique())
        st_means = {}
        for st in stages_present:
            st_sub = sub[sub["stage"] == st]
            st_means[st] = float(st_sub["loss_total"].mean())

        s1_loss = st_means.get("Stage 1", np.nan)
        s3_loss = st_means.get("Stage 3", np.nan)
        if np.isfinite(s1_loss) and np.isfinite(s3_loss) and s1_loss > 0:
            err_reduction_pct = ((s1_loss - s3_loss) / s1_loss) * 100.0
        else:
            err_reduction_pct = 0.0

        # Correlation with blur sigma
        if sub["blur_sigma"].std() > 1e-4:
            blur_corr = float(np.corrcoef(sub["blur_sigma"], sub["loss_total"])[0, 1])
        else:
            blur_corr = 0.0

        # Empirical value score: balances total learning signal + feature share + convergence
        # High value = strong feature engagement + high learning signal
        value_score = float(m_tot * (1.0 + feat_share_pct / 100.0))

        summary_rows.append({
            "class": c,
            "count": n_c,
            "sample_pct": round(n_c / total_samples * 100.0, 1),
            "loss_share_pct": round(loss_share_pct, 2),
            "mean_loss": round(m_tot, 5),
            "median_loss": round(med_tot, 5),
            "mean_l1": round(m_l1, 5),
            "mean_feat": round(m_feat, 4),
            "mean_tv": round(m_tv, 5),
            "mean_edge": round(m_edge, 5),
            "feat_share_pct": round(feat_share_pct, 1),
            "stage_1_loss": round(s1_loss, 5) if np.isfinite(s1_loss) else None,
            "stage_3_loss": round(s3_loss, 5) if np.isfinite(s3_loss) else None,
            "err_reduction_pct": round(err_reduction_pct, 1),
            "blur_corr": round(blur_corr, 3),
            "value_score": round(value_score, 4),
        })

    # Sort classes by empirical value score descending
    summary_rows.sort(key=lambda r: r["value_score"], reverse=True)
    for rank, r in enumerate(summary_rows, 1):
        r["rank"] = rank

    # Format Markdown Table
    md_lines = [
        "# Procedural Simulation Class Attribution Report",
        "",
        f"**Dataset**: `{csv_path}`  ",
        f"**Total Samples**: {total_samples}  ",
        f"**Classes Evaluated**: {n_classes}  ",
        "",
        "## Empirical Value Ranking",
        "",
        "| Rank | Class | Samples (%) | Mean Loss | Median Loss | L1 Loss | Feat Loss | Feat Share (%) | S1 → S3 Reduction (%) | Value Score |",
        "|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    ]

    for r in summary_rows:
        md_lines.append(
            f"| {r['rank']} | **`{r['class']}`** | {r['count']} ({r['sample_pct']}%) | "
            f"{r['mean_loss']:.5f} | {r['median_loss']:.5f} | {r['mean_l1']:.5f} | "
            f"{r['mean_feat']:.2f} | {r['feat_share_pct']:.1f}% | "
            f"{r['err_reduction_pct']:+.1f}% | **{r['value_score']:.4f}** |"
        )

    md_lines.extend([
        "",
        "## Analysis & Observations",
        "",
        "- **High-Value Classes**: Classes with high value scores drive both structural $L_1$ and deep perceptual feature gradients without collapsing.",
        "- **Low-Value / Trivial Classes**: Classes with low value scores contribute minimal learning signal or have low feature engagement.",
        ""
    ])

    report_md = "\n".join(md_lines)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        md_path = os.path.join(out_dir, "attribution_report.md")
        with open(md_path, "w") as fh:
            fh.write(report_md)
        json_path = os.path.join(out_dir, "attribution_summary.json")
        with open(json_path, "w") as fh:
            json.dump({"summary": summary_rows, "total_samples": total_samples}, fh, indent=2)
        print(f"[siq] Attribution report written to {md_path}")
        print(f"[siq] Attribution summary JSON written to {json_path}")

    return report_md, summary_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to attribution CSV file")
    ap.add_argument("--out", default=None, help="Output directory for reports")
    args = ap.parse_args()

    md, _ = analyze_attribution(args.csv, args.out)
    print("\n" + md)


if __name__ == "__main__":
    main()

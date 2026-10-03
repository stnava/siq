#!/usr/bin/env python
"""Fast 2D multi-phase curriculum (DBPN-small, VGG-L6, blind HR-first generator), validated on r16c.

Examples
--------
  python scripts/train_2d_curriculum.py --seeds 0 1 2            # full default 2D schedule
  python scripts/train_2d_curriculum.py --smoke                  # ~2 min contract/plumbing check
  python scripts/train_2d_curriculum.py --fast                   # aggressive ~10 min schedule
  # train on REAL slices (see scripts/build_real_slice_cache.py) with the validation degradation
  python scripts/train_2d_curriculum.py --cache results/real_slice_cache.npy --degradation matched --mse-only --iters 1200 0 0 0 --tag real_mse
  python scripts/train_2d_curriculum.py --edge-weight 0.5 --tag edge05

Per-seed outputs go to results/2d_curriculum/<tag>_s<seed>/ ; an aggregate
`summary.json` (mean +- SE of PCS/CQS/SSIM/acutance/Laplacian/QC) is written to results/2d_curriculum/<tag>/.
"""
import argparse, copy, json, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath("."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--tag", default="base")
    ap.add_argument("--iters", nargs=4, type=int, default=None, metavar=("WARM", "S1", "S2", "S3"))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--patch", type=int, default=32)
    ap.add_argument("--edge-weight", type=float, default=0.0, help="DBPN rule: 0 (ablation only otherwise)")
    ap.add_argument("--feature-layer", type=int, default=6)
    ap.add_argument("--shares2", nargs=3, type=float, default=None, metavar=("L1", "FEAT", "TV"))
    ap.add_argument("--cache", default=None, help=".npy (N,H,W) stack of REAL HR slices instead of procedural simulation")
    ap.add_argument("--degradation", choices=["blind", "matched", "aa"], default="blind",
                    help="blind: random blur/noise/zoom per stage (default); matched: decimation only "
                         "(exactly the validation degradation; no blur/noise/gamma); aa: Gaussian(1) "
                         "anti-alias + decimation with gamma diversity (heldout_aa degradation)")
    ap.add_argument("--mse-only", action="store_true", help="Warmup stage only (pure MSE; the classical SR recipe)")
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--out", default="results/2d_curriculum")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fast", action="store_true",
                    help="aggressive ~10 min preset: 200/150/450/600 iters, higher LRs, quicker balancer")
    ap.add_argument("--lrs", nargs=4, type=float, default=None, metavar=("WARM", "S1", "S2", "S3"))
    ap.add_argument("--beta", type=float, default=0.97, help="balancer dampening")
    ap.add_argument("--balancer-freq", type=int, default=10)
    ap.add_argument("--provenance-log", action="store_true",
                    help="Record per-sample simulation class provenance and loss attribution to CSV")
    a = ap.parse_args()

    import siq
    stages = copy.deepcopy([dict(s) for s in siq.DEFAULT_STAGES])
    if a.smoke:
        a.iters = [20, 20, 20, 20]
    if a.fast:
        a.iters = a.iters or [200, 150, 450, 600]
        a.lrs = a.lrs or [2e-4, 1e-4, 5e-5, 2e-5]
        a.beta = 0.9 if a.beta == 0.97 else a.beta
        a.balancer_freq = 5 if a.balancer_freq == 10 else a.balancer_freq
        a.patience = 4
    if a.lrs:
        for s, lr in zip(stages, a.lrs):
            s["lr"] = float(lr)
    if a.iters:
        for s, n in zip(stages, a.iters):
            s["iters"] = int(n)
    if a.shares2:
        sh = dict(l1=a.shares2[0], feat=a.shares2[1], tv=a.shares2[2])
        stages[2]["shares"] = dict(sh); stages[3]["shares"] = dict(sh)

    gen_kw = {}
    if a.cache:
        import numpy as np
        gen_kw["hr_base_cache"] = np.load(a.cache)
    if a.degradation == "matched":
        gen_kw.update(blur_sigma_range=(0.0, 0.0), interp_types=(0,))
        if a.cache:
            gen_kw["gamma_range"] = (1.0, 1.0)
        for st in stages:
            st.update(noise=(0.0, 0.0), rician=False, zoom=(1.0, 1.0))
    elif a.degradation == "aa":
        # = the `heldout_aa` benchmark degradation (Gaussian(1) anti-alias, nearest decimation) but with
        # gamma/intensity diversity kept: `matched` pinned gamma=1, and the resulting narrow intensity
        # distribution made the model darken bright slices (r16c gain ~1.11).
        gen_kw.update(blur_sigma_range=(1.0, 1.0), interp_types=(0,), gamma_range=(0.6, 1.7))
        for st in stages:
            st.update(noise=(0.0, 0.0), rician=False, zoom=(1.0, 1.0))
    if a.mse_only:
        stages = [dict(stages[0], gate=False)]

    results = []
    for seed in a.seeds:
        out = os.path.join(a.out, f"{a.tag}_s{seed}")
        prov_file = os.path.join(out, f"{a.tag}_s{seed}_attribution.csv") if a.provenance_log else None
        model, trace = siq.train_blind_sr_curriculum(
            output_prefix=f"dbpn_small_2d_{a.tag}_s{seed}", dimensionality=2, factor=2, stages=stages,
            batch_size=a.batch_size, lr_patch_size=a.patch, feature_layer=a.feature_layer,
            edge_weight=a.edge_weight, patience=a.patience, balancer_beta=a.beta,
            balancer_freq=a.balancer_freq, seed=seed, out_dir=out,
            provenance_log_file=prov_file,
            eval_freq=10 if a.smoke else 50, checkpoint_freq=20 if a.smoke else 100, **gen_kw)
        ev = trace.get("final_eval_latest", {})
        ev.update(seed=seed, seconds=trace["seconds_total"],
                  iters_used=[s["iters_used"] for s in trace["stages"]])
        results.append(ev)
        print("RESULT", json.dumps(ev), flush=True)

    keys = ["pcs", "cqs", "ssim", "psnr", "acutance", "laplacian", "gmsd", "cbi", "shift_max", "edge_corr_max"]
    agg = {"tag": a.tag, "n": len(results), "runs": results}
    for k in keys:
        v = np.array([r[k] for r in results if k in r], dtype=float)
        if len(v):
            agg[k] = {"mean": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else None}
    agg["status"] = [r.get("status") for r in results]
    os.makedirs(os.path.join(a.out, a.tag), exist_ok=True)
    json.dump(agg, open(os.path.join(a.out, a.tag, "summary.json"), "w"), indent=2)
    print("SUMMARY", json.dumps({k: v for k, v in agg.items() if k != "runs"}), flush=True)


if __name__ == "__main__":
    main()

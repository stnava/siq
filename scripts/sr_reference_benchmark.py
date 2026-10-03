#!/usr/bin/env python
"""SR reference benchmark CLI (thin wrapper around ``siq.benchmark``).

  # references + our models + public pretrained models (phase-calibrated)
  python scripts/sr_reference_benchmark.py --model mine=path.keras --public edsr-base --public msrn-bam
  python scripts/sr_reference_benchmark.py --list-public
  python scripts/sr_reference_benchmark.py --public edsr-base --phase auto      # empirical alignment

Build the real-data caches first (once):
  python scripts/build_real_slice_cache.py                                           # training cache
  python scripts/build_real_slice_cache.py --skip-subjects 40 --n-subjects 8 --stride 8 --out results/heldout_slice_cache.npy
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.abspath("."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[], help="name=path.keras (repeatable)")
    ap.add_argument("--public", action="append", default=[], help="zoo name or HF id (repeatable); see --list-public")
    ap.add_argument("--list-public", action="store_true")
    ap.add_argument("--phase", default="half-pixel", help="half-pixel | auto | none | <float>")
    ap.add_argument("--factor", type=int, nargs="+", default=[2, 2])
    ap.add_argument("--cases", nargs="+", default=None, help="subset of r16c heldout heldout_aa")
    ap.add_argument("--K", type=int, default=9)
    ap.add_argument("--train-cache", default="results/real_slice_cache.npy")
    ap.add_argument("--heldout", default="results/heldout_slice_cache.npy")
    ap.add_argument("--out", default="results/sr_reference")
    ap.add_argument("--name", default="reference")
    a = ap.parse_args()
    import siq
    if a.list_public:
        print(json.dumps(siq.list_public_models(), indent=2)); return
    phase = None if a.phase == "none" else (a.phase if a.phase in ("half-pixel", "auto") else float(a.phase))
    models = dict(s.split("=", 1) for s in a.model)
    _, md = siq.sr_benchmark(models=models, public=a.public, factor=tuple(a.factor), phase=phase, case_names=a.cases,
                          heldout=a.heldout, train_cache=a.train_cache, out_dir=a.out, name=a.name, K=a.K)
    print(md)


if __name__ == "__main__":
    main()

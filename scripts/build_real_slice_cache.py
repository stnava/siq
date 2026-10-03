#!/usr/bin/env python
"""Build a stack of real 2D axial T1 slices (N,S,S) for blind_sr_generator(hr_base_cache=...).

One volume per subject (backups skipped), middle ~60% of the brain z-range, every `--stride`-th slice,
centre crop SxS around the brain, kept only if enough foreground. r16 is never used here (it is the
held-out validation image)."""
import argparse, glob, os, numpy as np, ants

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="/Users/stnava/data/blast_cohorts/BIDS")
ap.add_argument("--n-subjects", type=int, default=40)
ap.add_argument("--stride", type=int, default=5)
ap.add_argument("--size", type=int, default=192)
ap.add_argument("--skip-subjects", type=int, default=0, help="skip the first K distinct subjects (use for a disjoint held-out set)")
ap.add_argument("--zrange", type=float, nargs=2, default=[0.2, 0.8],
                help="fraction of the brain z-extent to sample (mid-cerebrum ~ 0.5 0.75)")
ap.add_argument("--out", default="results/real_slice_cache.npy")
a = ap.parse_args()

files = sorted(f for f in glob.glob(f"{a.root}/*/sub-*/ses-*/anat/*T1w.nii.gz") if "backup" not in f)
seen, picked = set(), []
rng = np.random.RandomState(0); rng.shuffle(files)
for f in files:
    key = f.split("/")[-4] + "/" + f.split("/")[-3]      # cohort/subject
    if key in seen: continue
    seen.add(key); picked.append(f)
    if len(picked) >= a.skip_subjects + a.n_subjects: break
picked = picked[a.skip_subjects:]
S, slices = a.size, []
for f in picked:
    try:
        img = ants.image_read(f)
        img = ants.iMath(ants.iMath(img, "TruncateIntensity", 0.001, 0.999), "Normalize")
        v = img.numpy().astype("float32")
        m = v > 0.08
        zs = np.where(m.reshape(-1, m.shape[2]).mean(0) > 0.04)[0]
        if len(zs) < 20: continue
        z0, z1 = int(zs[0] + a.zrange[0] * len(zs)), int(zs[0] + a.zrange[1] * len(zs))
        for z in range(z0, z1, a.stride):
            sl = v[:, :, z]
            if min(sl.shape) < S:
                sl = np.pad(sl, [(0, max(0, S - sl.shape[0])), (0, max(0, S - sl.shape[1]))])
            ys, xs = np.where(sl > 0.08)
            if len(ys) < 0.15 * sl.size: continue
            cy, cx = int(ys.mean()), int(xs.mean())
            y0 = int(np.clip(cy - S // 2, 0, sl.shape[0] - S)); x0 = int(np.clip(cx - S // 2, 0, sl.shape[1] - S))
            c = sl[y0:y0 + S, x0:x0 + S]
            if (c > 0.08).mean() > 0.35: slices.append(c)
        print(os.path.basename(f), "spacing", tuple(round(s, 2) for s in img.spacing), "->", len(slices), flush=True)
    except Exception as e:
        print("skip", f, e)
arr = np.stack(slices).astype("float32")
os.makedirs(os.path.dirname(a.out), exist_ok=True)
np.save(a.out, arr)
print("saved", a.out, arr.shape)

"""Standard SR benchmarking: one registry of methods, one set of cases, one verdict.

Every super-resolution claim (our own model, a public pretrained model, a classical
interpolator) is evaluated the same way and compared against references:

* classical interpolators (nearest / bilinear / bspline / windowed sinc),
* bilinear + unsharp mask (global setting tuned on the case set),
* ``linear-LS`` -- a least-squares optimal FIR filter on top of bilinear upsampling
  (fit on real training slices, and an in-sample *oracle* fit per case = optimistic
  linear ceiling).  A learned model must beat the linear ceiling to claim learned gain.

Public-model integration
------------------------
``public_model_method(zoo_name_or_id)`` wraps any `super-image` (HuggingFace) network
(EDSR, MSRN, DRLN, A2N, PAN, CARN ...) as an :class:`SRMethod`.  Pretrained natural-image
networks assume anti-aliased input and half-pixel-centre alignment, whereas siq's contract is
decimation-aligned (LR voxel k at HR coordinate f*k).  Misalignment alone costs ~2-4 dB,
so every wrapped method is *phase calibrated* (``phase='half-pixel'`` analytic shift,
``'auto'`` empirical Lucas-Kanade estimate on calibration cases, or an explicit float).

Cases (``build_cases``)
-----------------------
``r16c``       cropped ANTs r16 slice (nearest decimation; out-of-domain smooth image)
``heldout``    real axial T1 slices from subjects disjoint from the training cache,
               nearest decimation (siq's training contract)
``heldout_aa`` same slices with a Gaussian(1) anti-alias before decimation (the degradation
               public SR networks were trained for)

Metrics: PSNR (info only), SSIM, GMSD, CBI, CQS, PCS (repo definition) and PCSc, a
corrected perceptual score that penalises deviation of the acutance / Laplacian ratios from
one (PCS rewards unbounded over-sharpening):

    PCSc = SSIM - 0.5|1-acutance| - 0.5|1-laplacian| - GMSD - CBI
"""
import os
import sys
import json
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

METRIC_COLS = ["psnr", "ssim", "gmsd", "cbi", "cqs", "pcs", "pcsc", "acutance", "laplacian"]

# name -> (HF repo id, super_image class name, verified-in-siq)
PUBLIC_ZOO = {
    "edsr-base": ("eugenesiow/edsr-base", "EdsrModel", True),
    "msrn-bam": ("eugenesiow/msrn-bam", "MsrnModel", True),
    "drln-bam": ("eugenesiow/drln-bam", "DrlnModel", False),
    "a2n": ("eugenesiow/a2n", "A2nModel", False),
    "pan-bam": ("eugenesiow/pan-bam", "PanModel", False),
    "carn-bam": ("eugenesiow/carn-bam", "CarnModel", False),
}

PUBLIC_INSTALL_HINT = ('pip install --target "$SIQ_PUBLIC_SR_PATH" super-image "huggingface_hub<0.30" '
                       "(then export SIQ_PUBLIC_SR_PATH, default /tmp/sipkgs)")


@dataclass
class SRMethod:
    """A callable ``run(lr, gt) -> np.ndarray`` (ANTs LR / GT images in, HR array out)."""
    name: str
    run: Callable
    group: str = "other"          # classical | unsharp | linear | model | public | other
    meta: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------------------
def compute_pcsc(y_true, y_pred, factor=None):
    """Corrected perceptual composite score (bounded sharpness reward). See module docstring."""
    from . import get_data as gd
    acu = float(gd.compute_acutance_ratio(y_true, y_pred))
    lap = float(gd.compute_laplacian_energy_ratio(y_true, y_pred))
    return float(gd.compute_ssim(y_true, y_pred) - 0.5 * abs(1 - acu) - 0.5 * abs(1 - lap)
                 - gd.compute_gmsd(y_true, y_pred) - gd.compute_checkerboard_index(y_pred, y_true, factor))


def sr_metrics(g, x, factor):
    """All benchmark metrics for one (ground-truth, prediction) pair of arrays in [0, 1]."""
    from . import get_data as gd
    acu = float(gd.compute_acutance_ratio(g, x)); lap = float(gd.compute_laplacian_energy_ratio(g, x))
    ssim = float(gd.compute_ssim(g, x)); gmsd = float(gd.compute_gmsd(g, x))
    cbi = float(gd.compute_checkerboard_index(x, g, factor))
    return dict(psnr=float(gd.compute_psnr(g, x)), ssim=ssim, gmsd=gmsd, cbi=cbi,
                cqs=float(gd.compute_cqs(g, x, factor=factor)), pcs=float(gd.compute_pcs(g, x, factor=factor)),
                acutance=acu, laplacian=lap, pcsc=ssim - 0.5 * abs(1 - acu) - 0.5 * abs(1 - lap) - gmsd - cbi)


# ----------------------------------------------------------------------------------------
# cases
# ----------------------------------------------------------------------------------------
def build_cases(factor=(2, 2), heldout="results/heldout_slice_cache.npy", n_heldout=24, include=None):
    """Standard case sets -> {name: [(lr_ants, gt_ants), ...]}.  See module docstring."""
    import ants
    from scipy.ndimage import gaussian_filter
    from .blind_sr import prepare_2d_validation
    factor = tuple(int(f) for f in factor)
    want = set(include) if include else {"r16c", "heldout", "heldout_aa"}
    cases = {}
    if "r16c" in want:
        lr, hr = prepare_2d_validation("r16", factor)
        cases["r16c"] = [(lr, ants.iMath(hr, "Normalize"))]
    if heldout and os.path.exists(heldout) and ({"heldout", "heldout_aa"} & want):
        arr = np.load(heldout)
        idx = np.linspace(0, len(arr) - 1, min(n_heldout, len(arr))).astype(int)
        sp = tuple(float(f) for f in factor)
        h, ha = [], []
        for i in idx:
            g_img = ants.iMath(ants.from_numpy(arr[i].astype("float32")), "Normalize")
            h.append((ants.resample_image(g_img, sp, use_voxels=False, interp_type=0), g_img))
            gb = ants.from_numpy(gaussian_filter(g_img.numpy(), 1.0).astype("float32"))
            ha.append((ants.resample_image(gb, sp, use_voxels=False, interp_type=0), g_img))
        if "heldout" in want: cases["heldout"] = h
        if "heldout_aa" in want: cases["heldout_aa"] = ha
    return cases


# ----------------------------------------------------------------------------------------
# method builders
# ----------------------------------------------------------------------------------------
def _up(lr, gt, interp):
    import ants
    return np.clip(ants.resample_image_to_target(lr, gt, interp_type=interp).numpy(), 0, 1)


def classical_methods():
    return [SRMethod("nearest", lambda l, g: _up(l, g, 1), "classical"),
            SRMethod("bilinear", lambda l, g: _up(l, g, 0), "classical"),
            SRMethod("bspline", lambda l, g: _up(l, g, 4), "classical"),
            SRMethod("windowed-sinc", lambda l, g: _up(l, g, 3), "classical")]


def unsharp_methods(sigmas=(1.0, 2.0), amounts=(0.25, 0.5, 1.0)):
    from scipy.ndimage import gaussian_filter
    out = []
    for s in sigmas:
        for a in amounts:
            def run(l, g, s=s, a=a):
                b = _up(l, g, 0)
                return np.clip(b + a * (b - gaussian_filter(b, s)), 0, 1)
            out.append(SRMethod(f"bil+unsharp(s={s},a={a})", run, "unsharp"))
    return out


def _patches_matrix(img, K):
    p = K // 2
    a = np.pad(img, p, mode="reflect")
    H, W = img.shape
    return np.stack([a[i:i + H, j:j + W].reshape(-1) for i in range(K) for j in range(K)], axis=1)


def _fit_ls(X, y, ridge=1e-6):
    return np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ y)


def linear_ls_methods(train_cache="results/real_slice_cache.npy", K=9, n_fit_batches=48):
    """[linear-LS (fit on real training slices, matched nearest decimation), linear-LS oracle (per case)]."""
    import ants
    from .blind_sr import blind_sr_generator
    out = []
    if train_cache and os.path.exists(train_cache):
        state = np.random.get_state()
        np.random.seed(0)
        gen = blind_sr_generator(hr_base_cache=np.load(train_cache), batch_size=8, lr_patch_size=32, factor=2,
                                 dimensionality=2, blur_sigma_range=(0.0, 0.0), noise_std_range=(0.0, 0.0),
                                 interp_types=(0,), gamma_range=(1.0, 1.0))
        Xs, ys = [], []
        for _ in range(n_fit_batches):
            x, y = next(gen)
            for i in range(len(x)):
                hp = y[i, ..., 0].astype("float32")
                if hp.std() < 0.03: continue
                b = ants.resample_image_to_target(ants.from_numpy(x[i, ..., 0].astype("float32"), spacing=(2., 2.)),
                                                  ants.from_numpy(hp), interp_type=0).numpy()
                Xs.append(_patches_matrix(b, K)); ys.append(hp.reshape(-1))
        np.random.set_state(state)
        w = _fit_ls(np.concatenate(Xs), np.concatenate(ys))
        out.append(SRMethod(f"linear-LS REAL (K={K})", lambda l, g, w=w: np.clip(
            (_patches_matrix(_up(l, g, 0), K) @ w).reshape(g.shape), 0, 1), "linear"))

    def oracle(l, g):
        b = _up(l, g, 0); gn = g.numpy(); pm = _patches_matrix(b, K)
        return np.clip((pm @ _fit_ls(pm, gn.reshape(-1))).reshape(gn.shape), 0, 1)
    out.append(SRMethod(f"linear-LS oracle in-sample (K={K})", oracle, "linear", {"oracle": True}))
    return out


def siq_model_method(name, path):
    """Wrap a saved siq model (``.keras`` + ``_config.json``) as a pure-SR method (no linear blend)."""
    from . import load_siq_model, inference
    model, cfg = load_siq_model(path)
    return SRMethod("MODEL " + name, lambda l, g: inference(
        l, model, config=cfg, verbose=False, poly_order=None, anti_checkerboard=False,
        linear_blend=1.0).numpy(), "model", {"path": path})


def callable_method(name, fn, group="other", phase=None, factor=(2, 2), calibration_cases=None):
    """Wrap an arbitrary ``fn(lr_ants, gt_ants) -> np.ndarray`` (e.g. an ONNX/Torch model).

    ``phase``: None (as-is), float / per-axis sequence (voxels), 'half-pixel' (-(f-1)/2) or
    'auto' (estimate with Lucas-Kanade on ``calibration_cases``)."""
    shift = resolve_phase_shift(fn, phase, factor, calibration_cases)
    run = fn if shift is None else (lambda l, g: apply_shift(fn(l, g), shift))
    return SRMethod(name, run, group, {"phase_shift": None if shift is None else [float(s) for s in shift]})


def apply_shift(x, shift):
    from scipy.ndimage import shift as ndshift
    return np.clip(ndshift(x, list(shift), order=3, mode="nearest"), 0, 1)


def estimate_phase_shift(fn, cases, n_cases=4):
    """Empirical output shift of ``fn`` (voxels, per axis) to apply for zero lag.

    Uses the gain/offset-robust Lucas-Kanade estimator; tries both signs and keeps the one that
    lowers the residual MSE (robust to the estimator's sign convention)."""
    from .get_data import compute_shift_lk
    from scipy.ndimage import shift as ndshift
    shifts = []
    for l, g in cases[:n_cases]:
        gn = g.numpy(); p = fn(l, g)[tuple(slice(0, s) for s in gn.shape)]
        s = np.array(compute_shift_lk(gn, p), dtype=float)
        best, best_mse = np.zeros_like(s), float(np.mean((p - gn) ** 2))
        for sign in (1.0, -1.0):
            q = ndshift(p, list(sign * s), order=3, mode="nearest")
            mse = float(np.mean((q - gn) ** 2))
            if mse < best_mse: best, best_mse = sign * s, mse
        shifts.append(best)
    return np.mean(shifts, axis=0)


def resolve_phase_shift(fn, phase, factor, cases=None):
    if phase is None: return None
    if isinstance(phase, str):
        if phase == "half-pixel": return np.array([-(int(f) - 1) / 2.0 for f in factor])
        if phase == "auto":
            if not cases: raise ValueError("phase='auto' needs calibration cases")
            return estimate_phase_shift(fn, cases)
        raise ValueError(f"unknown phase spec {phase!r}")
    arr = np.atleast_1d(np.asarray(phase, dtype=float))
    return np.repeat(arr, len(factor)) if arr.size == 1 else arr


def list_public_models():
    return {k: {"hf_id": v[0], "class": v[1], "verified": v[2]} for k, v in PUBLIC_ZOO.items()}


def public_model_method(name_or_id, factor=(2, 2), phase="half-pixel", calibration_cases=None):
    """Wrap a pretrained `super-image` network (see ``list_public_models``) as an SRMethod.

    ``phase``: 'half-pixel' (default; the convention of bicubic-trained natural-image SR),
    'auto' (empirical, needs ``calibration_cases``), None (raw) or an explicit shift."""
    for p in os.environ.get("SIQ_PUBLIC_SR_PATH", "/tmp/sipkgs").split(os.pathsep):
        if p and p not in sys.path and os.path.isdir(p): sys.path.insert(0, p)
    try:
        import torch
        import super_image
    except Exception as e:
        raise ImportError(f"public SR models need super-image + torch ({e}). Install: {PUBLIC_INSTALL_HINT}")
    if name_or_id in PUBLIC_ZOO:
        hf_id, cls_name, _ = PUBLIC_ZOO[name_or_id]
    else:                                               # e.g. 'eugenesiow/edsr-base' or 'edsr-base' style ids
        hf_id = name_or_id
        key = name_or_id.split("/")[-1]
        cls_name = {"edsr": "EdsrModel", "msrn": "MsrnModel", "drln": "DrlnModel", "a2n": "A2nModel",
                    "pan": "PanModel", "carn": "CarnModel"}[key.split("-")[0]]
    net = getattr(super_image, cls_name).from_pretrained(hf_id, scale=int(factor[0])).eval()

    def raw(l, g):
        x = torch.tensor(l.numpy()[None, None].astype("float32")).repeat(1, 3, 1, 1)
        with torch.no_grad():
            return np.clip(net(x).mean(1)[0].numpy(), 0, 1)
    m = callable_method(f"PUBLIC {hf_id}", raw, "public", phase, factor, calibration_cases)
    m.meta.update(hf_id=hf_id, params=int(sum(p.numel() for p in net.parameters())))
    return m


# ----------------------------------------------------------------------------------------
# running / reporting
# ----------------------------------------------------------------------------------------
def run_benchmark(methods, cases, factor=(2, 2), verbose=True):
    """-> ``{case: {method_name: {metric: mean, ..., '_group': g}}}`` (errors recorded, never fatal)."""
    factor = tuple(int(f) for f in factor)
    results = {}
    for cname, cl in cases.items():
        res = {}
        for m in methods:
            t0 = time.time()
            try:
                ms = []
                for l, g in cl:
                    gn = g.numpy(); x = np.asarray(m.run(l, g))
                    x = x[tuple(slice(0, s) for s in gn.shape)]
                    ms.append(sr_metrics(gn, x, factor))
                r = {c: float(np.mean([q[c] for q in ms])) for c in METRIC_COLS}
                r["_group"] = m.group; r["_seconds"] = time.time() - t0; r["_meta"] = m.meta
            except Exception as e:                          # one broken method must not kill the table
                r = {"_group": m.group, "_error": f"{type(e).__name__}: {e}"}
                if verbose: print(f"[benchmark] {m.name} on {cname} failed: {r['_error']}", flush=True)
            res[m.name] = r
        results[cname] = res
    return results


def collapse_unsharp(res):
    """Replace the unsharp grid by its best-by-CQS and best-by-PCSc settings."""
    res = dict(res)
    un = {k: v for k, v in res.items() if v.get("_group") == "unsharp" and "_error" not in v}
    if un:
        for k in un: res.pop(k)
        bc = max(un, key=lambda k: un[k]["cqs"]); bp = max(un, key=lambda k: un[k]["pcsc"])
        res["bil+unsharp best-CQS " + bc[len("bil+unsharp"):]] = un[bc]
        res["bil+unsharp best-PCSc " + bp[len("bil+unsharp"):]] = un[bp]
    return res


def verdicts(res):
    """Per-method verdicts for model / public rows against the references of one case set.

    delta_* are CQS differences.  ``beats_linear_ceiling`` compares against the best in-sample
    linear oracle (optimistic bound) when present."""
    ok = {k: v for k, v in res.items() if "_error" not in v}
    def best(groups, key="cqs"):
        c = [v[key] for v in ok.values() if v["_group"] in groups]
        return max(c) if c else None
    bil = ok.get("bilinear", {}).get("cqs")
    cls_best = best({"classical", "unsharp"})
    oracles = [v["cqs"] for v in ok.values() if v.get("_meta", {}).get("oracle")]
    ceil = max(oracles) if oracles else None
    out = {}
    for k, v in ok.items():
        if v["_group"] in ("model", "public"):
            out[k] = dict(
                d_cqs_vs_bilinear=None if bil is None else v["cqs"] - bil,
                d_cqs_vs_best_classical=None if cls_best is None else v["cqs"] - cls_best,
                d_cqs_vs_linear_ceiling=None if ceil is None else v["cqs"] - ceil,
                beats_best_classical=None if cls_best is None else bool(v["cqs"] > cls_best),
                beats_linear_ceiling=None if ceil is None else bool(v["cqs"] > ceil))
    return out


def format_markdown(results):
    md = []
    for cname, res in results.items():
        res = collapse_unsharp(res)
        md.append(f"### {cname}\n")
        md.append("| method | " + " | ".join(METRIC_COLS) + " |")
        md.append("|:--|" + "|".join([":-:"] * len(METRIC_COLS)) + "|")
        for k, v in res.items():
            if "_error" in v:
                md.append(f"| {k} | ERROR: {v['_error'][:80]} |"); continue
            md.append(f"| {k} | " + " | ".join(f"{v[c]:.2f}" if c == "psnr" else f"{v[c]:.4f}" for c in METRIC_COLS) + " |")
        vd = verdicts(res)
        if vd:
            md.append("\n**Verdicts (CQS; PSNR is informational):**\n")
            md.append("| method | Δ vs bilinear | Δ vs best classical | Δ vs linear ceiling | beats classical | beats ceiling |")
            md.append("|:--|:-:|:-:|:-:|:-:|:-:|")
            f = lambda x: "n/a" if x is None else (f"{x:+.4f}" if isinstance(x, float) else ("yes" if x else "no"))
            for k, v in vd.items():
                md.append(f"| {k} | {f(v['d_cqs_vs_bilinear'])} | {f(v['d_cqs_vs_best_classical'])} | "
                          f"{f(v['d_cqs_vs_linear_ceiling'])} | {f(v['beats_best_classical'])} | {f(v['beats_linear_ceiling'])} |")
        md.append("")
    return "\n".join(md)


def save_results(results, out_dir="results/sr_reference", name="reference"):
    os.makedirs(out_dir, exist_ok=True)
    full = {c: collapse_unsharp(r) for c, r in results.items()}
    with open(os.path.join(out_dir, f"{name}.json"), "w") as fh:
        json.dump({"results": full, "verdicts": {c: verdicts(r) for c, r in full.items()}}, fh, indent=2, default=str)
    md = format_markdown(results)
    with open(os.path.join(out_dir, f"{name}.md"), "w") as fh:
        fh.write(md + "\n")
    return md


def benchmark(models=None, public=None, factor=(2, 2), phase="half-pixel", case_names=None,
              heldout="results/heldout_slice_cache.npy", train_cache="results/real_slice_cache.npy",
              out_dir="results/sr_reference", name="reference", K=9, verbose=True):
    """One call: references + ``models`` ({name: path.keras}) + ``public`` ([zoo names / HF ids]).

    Returns (results, markdown).  Written to ``out_dir/{name}.json|md``."""
    factor = tuple(int(f) for f in factor)
    cases = build_cases(factor, heldout=heldout, include=case_names)
    methods = classical_methods() + unsharp_methods() + linear_ls_methods(train_cache, K)
    for n, p in (models or {}).items():
        methods.append(siq_model_method(n, p))
    calib = cases.get("heldout_aa") or next(iter(cases.values()))
    for pid in (public or []):
        methods.append(public_model_method(pid, factor, phase, calibration_cases=calib))
    results = run_benchmark(methods, cases, factor, verbose)
    return results, save_results(results, out_dir, name)

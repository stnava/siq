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

PUBLIC_INSTALL_HINT = ('pip install "siq[public-sr]"  (or, isolated from the main env: '
                       'pip install --target "$SIQ_PUBLIC_SR_PATH" super-image "huggingface_hub<0.30"; '
                       "SIQ_PUBLIC_SR_PATH defaults to /tmp/sipkgs)")


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
    """Wrap a saved siq model (``.keras`` + ``_config.json``) as a pure-SR method (no linear blend).

    Benchmark cases are already truncated/normalised to [0, 1] and have identity direction, so the
    model is run directly. Going through ``inference()`` here re-applies TruncateIntensity+Normalize
    (and a min-max Normalize on the output when ``linear_blend`` is set), which lifted slice intensity
    ~19% (PSNR ~18 dB) for models that were fine on the raw forward pass (MSE below bilinear).
    The architecture-aware phase shift is kept (transposed-conv models only)."""
    from . import load_siq_model
    from .get_data import _model_has_transposed_conv
    from scipy.ndimage import shift as ndshift
    model, cfg = load_siq_model(path)
    has_tc = _model_has_transposed_conv(model)
    up = cfg.get("upsample_factor", 2)

    def run(l, g):
        x = l.numpy().astype("float32")
        flips = tuple(i for i in range(x.ndim) if float(l.direction[i, i]) < 0)   # direction-cosine sign invariant
        if flips: x = np.flip(x, axis=flips).copy()
        out = model.predict(x[None, ..., None], verbose=0)[0, ..., 0]
        if flips: out = np.flip(out, axis=flips).copy()
        f = list(up) if isinstance(up, (list, tuple)) else [up] * x.ndim
        if has_tc:
            sh = [-(fd - 1) / 2.0 if fd > 1 else 0.0 for fd in f]
        else:     # discrete reflection compensation on flipped axes
            sh = [-(fd - 1.0) if (d in flips and fd > 1) else 0.0 for d, fd in enumerate(f)]
        if any(abs(s) > 1e-4 for s in sh):
            out = ndshift(out, sh, order=3, mode="nearest")
        return np.clip(out, 0.0, 1.0)
    return SRMethod("MODEL " + name, run, "model", {"path": path})



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
    render_html_report(results, os.path.join(out_dir, f"{name}.html"))
    return md


def build_methods(models=None, public=None, factor=(2, 2), phase="half-pixel", calibration_cases=None,
                  train_cache="results/real_slice_cache.npy", K=9):
    """Reference + model + public methods as one list (shared by tables and image reports)."""
    factor = tuple(int(f) for f in factor)
    methods = classical_methods() + unsharp_methods() + linear_ls_methods(train_cache, K)
    for n, p in (models or {}).items():
        methods.append(siq_model_method(n, p))
    for pid in (public or []):
        methods.append(public_model_method(pid, factor, phase, calibration_cases=calibration_cases))
    return methods


def benchmark(models=None, public=None, factor=(2, 2), phase="half-pixel", case_names=None,
              heldout="results/heldout_slice_cache.npy", train_cache="results/real_slice_cache.npy",
              out_dir="results/sr_reference", name="reference", K=9, verbose=True):
    """One call: references + ``models`` ({name: path.keras}) + ``public`` ([zoo names / HF ids]).

    Returns (results, markdown).  Written to ``out_dir/{name}.json|md|html``."""
    factor = tuple(int(f) for f in factor)
    cases = build_cases(factor, heldout=heldout, include=case_names)
    calib = cases.get("heldout_aa") or next(iter(cases.values()))
    methods = build_methods(models, public, factor, phase, calib, train_cache, K)
    results = run_benchmark(methods, cases, factor, verbose)
    return results, save_results(results, out_dir, name)


def render_html_report(results, path, title="siq SR reference benchmark", notes_html=""):
    """Self-contained HTML (no external assets): one tab per case set (keys 1..n), methods sorted by CQS,
    best value per column highlighted, CQS bars, and the verdict table.  ``results`` is the output of
    :func:`run_benchmark` or the ``results`` dict of a saved ``*.json``."""
    import html as _h
    res_all = {c: collapse_unsharp(r) for c, r in results.items()}
    higher_better = {"psnr": True, "ssim": True, "cqs": True, "pcs": True, "pcsc": True,
                     "gmsd": False, "cbi": False}
    colors = {"classical": "#64748b", "unsharp": "#a16207", "linear": "#0e7490", "model": "#7c3aed",
              "public": "#be185d", "other": "#475569"}
    tabs, panels = [], []
    for i, (cname, res) in enumerate(res_all.items()):
        ok = {k: v for k, v in res.items() if "_error" not in v}
        order = sorted(ok, key=lambda k: -ok[k]["cqs"])
        best = {c: (max if hb else min)(v[c] for v in ok.values()) for c, hb in higher_better.items()}
        cq = [v["cqs"] for v in ok.values()]; lo, hi = min(cq) - 0.02, max(cq)
        rows = []
        for k in order:
            v = ok[k]; col = colors.get(v["_group"], "#475569")
            cells = []
            for c in METRIC_COLS:
                txt = f"{v[c]:.2f}" if c == "psnr" else f"{v[c]:.4f}"
                cls = "best" if c in best and abs(v[c] - best[c]) < 1e-12 else ""
                if c == "cqs":
                    w = 100 * (v["cqs"] - lo) / max(hi - lo, 1e-9)
                    cells.append(f'<td class="{cls}"><div class="bar" style="width:{w:.0f}%;background:{col}33"></div>'
                                 f'<span>{txt}</span></td>')
                else:
                    cells.append(f'<td class="{cls}">{txt}</td>')
            rows.append(f'<tr><td class="name"><span class="dot" style="background:{col}"></span>{_h.escape(k)}</td>'
                        + "".join(cells) + "</tr>")
        for k, v in res.items():
            if "_error" in v:
                rows.append(f'<tr><td class="name">{_h.escape(k)}</td><td colspan="{len(METRIC_COLS)}" class="err">'
                            f'ERROR: {_h.escape(v["_error"][:120])}</td></tr>')
        vd = verdicts(res)
        vrows = []
        f = lambda x: "n/a" if x is None else (f"{x:+.4f}" if isinstance(x, float) else ("yes" if x else "no"))
        for k, v in vd.items():
            tag = lambda b: f'<td class="{"yes" if b else "no"}">{f(b)}</td>' if b is not None else "<td>n/a</td>"
            vrows.append(f"<tr><td class='name'>{_h.escape(k)}</td><td>{f(v['d_cqs_vs_bilinear'])}</td>"
                         f"<td>{f(v['d_cqs_vs_best_classical'])}</td><td>{f(v['d_cqs_vs_linear_ceiling'])}</td>"
                         f"{tag(v['beats_best_classical'])}{tag(v['beats_linear_ceiling'])}</tr>")
        head = "".join(f"<th>{c}</th>" for c in METRIC_COLS)
        n = ""
        panels.append(f'<section id="p{i}" class="panel{" on" if i == 0 else ""}"><h2>{_h.escape(cname)}</h2>'
                      f'<table><thead><tr><th>method (sorted by CQS)</th>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'
                      + (f'<h3>Verdicts vs references (CQS)</h3><table class="v"><thead><tr><th>method</th><th>&Delta; vs bilinear</th>'
                         f'<th>&Delta; vs best classical</th><th>&Delta; vs linear ceiling</th><th>beats classical</th>'
                         f'<th>beats ceiling</th></tr></thead><tbody>{"".join(vrows)}</tbody></table>' if vrows else "")
                      + "</section>")
        tabs.append(f'<button class="tab{" on" if i == 0 else ""}" data-i="{i}">{i + 1} &middot; {_h.escape(cname)}</button>')
    legend = "".join(f'<span><span class="dot" style="background:{c}"></span>{g}</span>' for g, c in colors.items() if g != "other")
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>{_h.escape(title)}</title><style>
body{{font:14px -apple-system,Segoe UI,sans-serif;margin:24px;color:#0f172a;background:#f8fafc}}
h1{{margin:0 0 4px}} .sub{{color:#475569;margin-bottom:14px}} .legend span{{margin-right:14px}}
.tabs{{margin:12px 0}} .tab{{border:1px solid #cbd5e1;background:#fff;padding:7px 14px;margin-right:6px;border-radius:6px;cursor:pointer;font-size:14px}}
.tab.on{{background:#0f172a;color:#fff;border-color:#0f172a}} .panel{{display:none}} .panel.on{{display:block}}
table{{border-collapse:collapse;background:#fff;margin:8px 0 18px;box-shadow:0 1px 2px #0001}}
th,td{{padding:6px 10px;border-bottom:1px solid #e2e8f0;text-align:right;position:relative;font-variant-numeric:tabular-nums}}
th{{background:#f1f5f9;position:sticky;top:0}} td.name,th:first-child{{text-align:left;white-space:nowrap}}
td.best{{font-weight:700;background:#dcfce7}} .bar{{position:absolute;left:0;top:0;bottom:0;z-index:0}} td span{{position:relative;z-index:1}}
.dot{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px;vertical-align:middle}}
.yes{{color:#166534;font-weight:700}} .no{{color:#991b1b}} .err{{color:#991b1b;text-align:left}}
.note{{max-width:1000px;color:#334155;line-height:1.5}} code{{background:#e2e8f0;padding:1px 4px;border-radius:3px}}
</style></head><body><h1>{_h.escape(title)}</h1>
<div class="sub">CQS decides (PSNR is informational &mdash; it rewards blur). Green = best in column. PCSc bounds the sharpness reward that raw PCS leaves unbounded. Keys 1&ndash;{len(res_all)} switch case sets.</div>
<div class="legend">{legend}</div><div class="tabs">{"".join(tabs)}</div>{"".join(panels)}
<div class="note">{notes_html}</div>
<script>
const T=[...document.querySelectorAll('.tab')],P=[...document.querySelectorAll('.panel')];
function show(i){{if(i<0||i>=P.length)return;T.forEach((t,j)=>t.classList.toggle('on',j==i));P.forEach((p,j)=>p.classList.toggle('on',j==i));}}
T.forEach((t,i)=>t.onclick=()=>show(i));
document.addEventListener('keydown',e=>{{const n=parseInt(e.key);if(n)show(n-1);}});
</script></body></html>"""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(doc)
    return path


def render_image_report(methods, cases, path, factor=(2, 2), picks=None, title="siq SR comparison (images)",
                        notes_html="", keep=None, zoom=3):
    """Single-viewport, in-place flicker report of the actual SR images.

    One canvas (same coordinates, size and zoom for every state): number keys 1..9,0 swap the shown
    method in place (no layout shift), ``e`` toggles the residual |SR-GT| map, ``c`` cycles the case,
    ``z`` toggles zoom.  Display contrast uses ``ants.histogram_equalize_image`` on the joint stack of
    all states of a case (ONE shared lookup table, so flicker differences are real, not LUT artefacts);
    equalization is applied to display arrays only -- metrics use the raw outputs.

    ``picks``: {case_name: [case indices]} (default: first case; two spread slices for multi-slice sets).
    ``keep``: iterable of method names to show (default: bilinear, bspline, the best-CQS unsharp setting,
    linear-LS oracle, then every model/public method).  Max 8 + GT + LR = 10 keys."""
    import base64, io, html as _h
    import ants
    from PIL import Image

    def png(a):
        buf = io.BytesIO(); Image.fromarray(np.clip(a * 255, 0, 255).astype("uint8")).save(buf, "PNG")
        return base64.b64encode(buf.getvalue()).decode()

    factor = tuple(int(f) for f in factor)
    views = []
    for cname, cl in cases.items():
        idxs = (picks or {}).get(cname)
        if idxs is None:
            idxs = [0] if len(cl) == 1 else [len(cl) // 4, (3 * len(cl)) // 4]
        views += [(cname, i, cl[i]) for i in idxs]
    # method selection: references first, the per-view best-CQS unsharp setting after bspline, then the rest
    by_name = {m.name: m for m in methods}
    if keep:
        pre, post = [n for n in keep if by_name[n].group in ("classical",)], \
                    [n for n in keep if by_name[n].group not in ("classical",)]
    else:
        pre = [n for n in ("bilinear", "bspline") if n in by_name]
        post = [m.name for m in methods if m.meta.get("oracle")] + \
               [m.name for m in methods if m.group in ("model", "public")]
    unsharp_all = [m for m in methods if m.group == "unsharp"] if not keep else []

    def run_one(m, lr, gt, g):
        return np.clip(np.asarray(m.run(lr, gt))[tuple(slice(0, s) for s in g.shape)], 0, 1)

    payload = []
    for cname, idx, (lr, gt) in views:
        g = gt.numpy().astype("float32")
        outs = {"Ground truth (HR)": g,
                "LR input (nearest-upsampled)": np.clip(
                    ants.resample_image_to_target(lr, gt, interp_type=1).numpy(), 0, 1)[tuple(slice(0, s) for s in g.shape)]}
        stats = {k: None for k in outs}

        def add(name, x):
            outs[name] = x
            st = sr_metrics(g, x, factor); stats[name] = (st["psnr"], st["cqs"], st["pcsc"])
        for n in pre:
            add(n, run_one(by_name[n], lr, gt, g))
        if unsharp_all:
            cand = {m.name: run_one(m, lr, gt, g) for m in unsharp_all}
            best = max(cand, key=lambda k: sr_metrics(g, cand[k], factor)["cqs"])
            add(best, cand[best])
        for n in post:
            add(n, run_one(by_name[n], lr, gt, g))
        assert len(outs) <= 10, "image report supports at most 10 states (keys 1-9,0)"
        # one shared display LUT: joint histogram equalisation of the stacked states (display only)
        keys = list(outs)
        stack = np.concatenate([outs[k] for k in keys], axis=1)
        eq = ants.histogram_equalize_image(ants.from_numpy(stack.astype("float32")), number_of_histogram_bins=256).numpy()
        W = g.shape[1]
        imgs, errs = {}, {}
        for j, k in enumerate(keys):
            imgs[k] = png(eq[:, j * W:(j + 1) * W])
            errs[k] = png(1.0 - np.clip(np.abs(outs[k] - g) / 0.25, 0, 1))   # white=0 error; same scale for ALL states
        payload.append(dict(case=cname, idx=int(idx), shape=list(g.shape), keys=keys, stats=stats, imgs=imgs, errs=errs))
    import json as _json
    data = _json.dumps(payload)
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>{_h.escape(title)}</title><style>
body{{font:14px -apple-system,Segoe UI,sans-serif;margin:18px;color:#0f172a;background:#0b1220;color:#e2e8f0}}
h1{{margin:0 0 4px;font-size:20px}} .sub{{color:#94a3b8;margin-bottom:10px}}
#bar{{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}} button{{border:1px solid #334155;background:#111a2e;color:#cbd5e1;padding:6px 10px;border-radius:6px;cursor:pointer;font-size:13px}}
button.on{{background:#e2e8f0;color:#0b1220;border-color:#e2e8f0}} #wrap{{display:flex;gap:18px;align-items:flex-start}}
#view{{width:calc(var(--w)*var(--z)*1px);height:calc(var(--h)*var(--z)*1px);border:1px solid #334155;background:#000;flex:none}}
#view img{{width:100%;height:100%;image-rendering:pixelated;display:block}}
#info{{min-width:340px}} #info b{{color:#fff}} .k{{color:#94a3b8}} table{{border-collapse:collapse}} td{{padding:3px 8px;border-bottom:1px solid #1e293b}}
code{{background:#1e293b;padding:1px 5px;border-radius:3px}} .note{{max-width:1000px;color:#94a3b8;margin-top:14px;line-height:1.5}}
</style></head><body><h1>{_h.escape(title)}</h1>
<div class="sub">Single viewport &mdash; every state shares the same canvas, zoom and display LUT. Keys: <code>1</code>&ndash;<code>9</code>,<code>0</code> method &middot; <code>e</code> residual map &middot; <code>c</code> next case &middot; <code>z</code> zoom.</div>
<div id="bar"></div><div id="bar2"></div>
<div id="wrap"><div id="view"><img id="im"></div><div id="info"></div></div>
<div class="note">{notes_html}</div>
<script>
const D={data};let ci=0,mi=0,err=false,Z={zoom};
const bar=document.getElementById('bar'),bar2=document.getElementById('bar2'),im=document.getElementById('im'),info=document.getElementById('info'),view=document.getElementById('view');
function keyLabel(i){{return i==9?'0':String(i+1)}}
function build(){{bar.innerHTML='';D.forEach((d,i)=>{{const b=document.createElement('button');b.textContent=(i+1)+': '+d.case+(D.filter(x=>x.case==d.case).length>1?' #'+d.idx:'');b.className=i==ci?'on':'';b.onclick=()=>{{ci=i;render()}};bar.appendChild(b)}});
bar2.innerHTML='';D[ci].keys.forEach((k,i)=>{{const b=document.createElement('button');b.textContent=keyLabel(i)+' '+k;b.className=i==mi?'on':'';b.onclick=()=>{{mi=i;render()}};bar2.appendChild(b)}})}}
function render(){{const d=D[ci];if(mi>=d.keys.length)mi=0;const k=d.keys[mi];build();
view.style.setProperty('--w',d.shape[1]);view.style.setProperty('--h',d.shape[0]);view.style.setProperty('--z',Z);
im.src='data:image/png;base64,'+(err?d.errs[k]:d.imgs[k]);
const s=d.stats[k];let rows='';d.keys.forEach((kk,i)=>{{const t=d.stats[kk];rows+=`<tr style="${{i==mi?'background:#1e293b':''}}"><td class=k>${{keyLabel(i)}}</td><td>${{kk}}</td><td>${{t?t[0].toFixed(2):'-'}}</td><td>${{t?t[1].toFixed(4):'-'}}</td><td>${{t?t[2].toFixed(3):'-'}}</td></tr>`}});
info.innerHTML=`<div><b>${{k}}</b> &nbsp; <span class=k>${{err?'residual |SR-GT| (white=0, black&ge;0.25)':'display: shared joint histogram equalisation'}}</span></div><div class=k>case ${{d.case}} #${{d.idx}} &middot; ${{d.shape[0]}}&times;${{d.shape[1]}}</div><table><tr><td></td><td class=k>method</td><td class=k>PSNR</td><td class=k>CQS</td><td class=k>PCSc</td></tr>${{rows}}</table>`}}
document.addEventListener('keydown',e=>{{if(e.key=='e')err=!err;else if(e.key=='c')ci=(ci+1)%D.length;else if(e.key=='z')Z=Z==3?5:(Z==5?2:3);else{{const n=e.key=='0'?9:parseInt(e.key)-1;if(n>=0&&n<D[ci].keys.length)mi=n;else return}}render()}});
render();
</script></body></html>"""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as fh:
        fh.write(doc)
    return path

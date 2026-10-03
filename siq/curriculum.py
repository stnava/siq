"""Dimension-generic multi-phase (Stage 0-3) blind super-resolution curriculum.

Used first in 2D (fast: ~0.4 s/step) to establish stage lengths, loss-share targets
and expected QC/PCS values that are then *transferred* to 3D.  The model, VGG-L6
perceptual backend, blind HR-first generator, alignment audit and QC reporter are the
same objects used by the 3D pipeline.

Stages
------
Warmup  : 100% MSE, clean data, until validation PSNR >= bilinear (or max iters).
Stage 1 : clean data, L1-dominant (L1 70 / feat 25 / TV 5) -- balancer settles weights.
Stage 2 : Rician noise + zoom, established shares (L1 30 / feat 65 / TV 5).
Stage 3 : fine LR, same shares (+ optional directional-gradient/acutance edge loss,
          off by default for iterative back-projection architectures such as DBPN).

Loss weights are driven by *shares*: each term's weight is set so that its contribution
to the total is `share * total_scale`, using the median term magnitude over the last
`window` batches, and moved toward that target with dampening `beta` (default 0.97).
"""
import json
import os
import time
from collections import deque

import numpy as np

DEFAULT_STAGES = (
    dict(name="Warmup", iters=400, lr=1e-4, noise=(0.0, 0.0), rician=False, zoom=(1.0, 1.0),
         shares=None, gate=True),
    dict(name="Stage 1", iters=300, lr=5e-5, noise=(0.0, 0.0), rician=False, zoom=(1.0, 1.0),
         shares=dict(l1=70.0, feat=25.0, tv=5.0)),
    dict(name="Stage 2", iters=600, lr=3e-5, noise=(0.0, 0.02), rician=True, zoom=(0.75, 1.3),
         shares=dict(l1=30.0, feat=65.0, tv=5.0)),
    dict(name="Stage 3", iters=1200, lr=1e-5, noise=(0.0, 0.01), rician=True, zoom=(0.85, 1.2),
         shares=dict(l1=30.0, feat=65.0, tv=5.0)),
)


def evaluate_model_2d(model, config=None, val_image="r16", factor=(2, 2), seed=123, n_synth=64):
    """Held-out evaluation on r16c: QC (bilinear-relative), PCS/CQS and components.

    Returns a flat dict of floats.  PSNR is informational (never used for selection).
    """
    import ants
    import siq
    from .blind_sr import prepare_2d_validation
    factor = tuple(int(f) for f in factor)
    lr, hr = prepare_2d_validation(val_image, factor)
    sr = siq.inference(lr, model, config=config, verbose=False, poly_order=None,
                       anti_checkerboard=False, linear_blend=1.0)
    gt = ants.iMath(hr, "Normalize")
    bil = ants.iMath(ants.resample_image_to_target(lr, gt, interp_type=0), "Normalize")
    qc = siq.compute_alignment_qc(gt, sr, bilinear=bil, factor=factor)
    g, s, b = gt.numpy(), sr.numpy(), bil.numpy()
    out = {
        "status": qc["status"], "psnr": float(qc["psnr"]), "bilinear_psnr": float(qc["bilinear_psnr"]),
        "ssim": float(qc["ssim"]), "shift_max": float(qc["max_phase_shift"]),
        "edge_corr_max": float(np.max(qc["edge_correlation"])),
        "pcs": float(siq.compute_pcs(g, s, factor=factor)),
        "cqs": float(siq.compute_cqs(g, s, factor=factor)),
        "pcs_bilinear": float(siq.compute_pcs(g, b, factor=factor)),
        "cqs_bilinear": float(siq.compute_cqs(g, b, factor=factor)),
        "acutance": float(siq.compute_acutance_ratio(g, s)),
        "laplacian": float(siq.compute_laplacian_energy_ratio(g, s)),
        "acutance_bilinear": float(siq.compute_acutance_ratio(g, b)),
        "laplacian_bilinear": float(siq.compute_laplacian_energy_ratio(g, b)),
        "gmsd": float(siq.compute_gmsd(g, s)), "cbi": float(siq.compute_checkerboard_index(s, g, factor)),
    }
    return out


def train_blind_sr_curriculum(
    output_prefix="dbpn_small_2d_curriculum",
    dimensionality=2,
    factor=2,
    model=None,
    stages=DEFAULT_STAGES,
    batch_size=8,
    lr_patch_size=32,
    feature_layer=6,
    val_image="r16",
    edge_weight=0.0,
    total_scale=1.0,
    balancer_beta=0.97,
    balancer_freq=10,
    balancer_window=10,
    eval_freq=50,
    checkpoint_freq=100,
    patience=6,
    min_stage_frac=0.5,
    seed=0,
    out_dir=".",
    enable_report=True,
    provenance_log_file=None,
    **generator_kwargs,
):
    """Run the staged curriculum and return (model, trace).

    `trace` (also saved to `{output_prefix}_curriculum_trace.json`) records per-stage final
    loss weights, measured loss shares, wall time, iterations used (after early stop) and
    validation history -- the quantities that are transferred to 3D.
    """
    import keras
    from keras import ops
    import siq
    from .blind_sr import (blind_sr_generator, _normalize_factor, prepare_2d_validation)
    from .alignment import audit_pair_alignment
    from .get_data import default_siq_config, save_siq_model, vgg_features_2d, pseudo_3d_vgg_features_unbiased

    dim = int(dimensionality)
    factor_tuple = _normalize_factor(factor, dim)
    keras.utils.set_random_seed(int(seed))
    np.random.seed(int(seed))
    hr_patch_shape = tuple(int(lr_patch_size * f) for f in factor_tuple)
    red_axes = list(range(1, dim + 2))

    if model is None:
        model = siq.default_dbpn(strider=list(factor_tuple), dimensionality=dim, nChannelsIn=1,
                                 nChannelsOut=1, sigmoid_second_channel=False, option="small")

    if dim == 2:
        fe = vgg_features_2d(inshape=list(hr_patch_shape), layer=feature_layer)
    else:
        fe = pseudo_3d_vgg_features_unbiased(inshape=list(hr_patch_shape), layer=feature_layer)
    fe.trainable = False

    V = {k: keras.Variable(0.0, dtype="float32") for k in ("msq", "l1", "feat", "tv", "edge")}
    V["edge"].assign(float(edge_weight))

    def _tv(y):
        s = 0.0
        for ax in range(1, dim + 1):
            hi = [slice(None)] * (dim + 2); lo = [slice(None)] * (dim + 2)
            hi[ax] = slice(1, None); lo[ax] = slice(None, -1)
            s = s + ops.mean(ops.abs(y[tuple(hi)] - y[tuple(lo)]), axis=red_axes)
        return s

    def _grads(y):
        gs = []
        for ax in range(1, dim + 1):
            hi = [slice(None)] * (dim + 2); lo = [slice(None)] * (dim + 2)
            for o in range(1, dim + 1):          # common interior so axes are summable
                hi[o] = slice(0, -1); lo[o] = slice(0, -1)
            hi[ax] = slice(1, None); lo[ax] = slice(0, -1)
            g = y[tuple(hi)] - y[tuple(lo)]
            gs.append(g)
        # axes whose slice used (1,None) are one shorter along `ax`; trim all to the min extent
        sh = [min(int(g.shape[o]) for g in gs) for o in range(1, dim + 1)]
        return [g[(slice(None),) + tuple(slice(0, n) for n in sh) + (slice(None),)] for g in gs]

    def _edge(yt, yp):
        gt_, gp_ = _grads(yt), _grads(yp)
        pointwise = sum(ops.mean(ops.abs(a - b), axis=red_axes) for a, b in zip(gt_, gp_))
        mt = ops.sqrt(sum(ops.square(a) for a in gt_) + 1e-8)
        mp = ops.sqrt(sum(ops.square(b) for b in gp_) + 1e-8)
        return pointwise + ops.mean(ops.abs(mp - mt), axis=red_axes)

    def terms(y_true, y_pred):
        d = y_true - y_pred
        t = {"msq": ops.mean(ops.square(d), axis=red_axes),
             "l1": ops.mean(ops.abs(d), axis=red_axes),
             "feat": 0.0, "tv": _tv(y_pred), "edge": _edge(y_true, y_pred)}
        ft, fp = fe(y_true, training=False), fe(y_pred, training=False)
        if not isinstance(ft, list):
            ft, fp = [ft], [fp]
        t["feat"] = sum(ops.mean(ops.square(a - b), axis=list(range(1, len(a.shape)))) for a, b in zip(ft, fp))
        return t

    def loss_fn(y_true, y_pred):
        t = terms(y_true, y_pred)
        return sum(t[k] * V[k] for k in ("msq", "l1", "feat", "tv", "edge"))

    model.compile(optimizer=keras.optimizers.Adam(float(stages[0]["lr"])), loss=loss_fn)

    # --- reporter (QC + PCS champion selection; QC FAIL blocks promotion) -----------------
    reporter = None
    val_lr = val_hr = None
    if dim == 2 and enable_report:
        try:
            try:
                from scripts.visual_convergence_report import VisualConvergenceReporter as VCR
            except Exception:
                try:
                    from tests.visual_convergence_report import VisualConvergenceReporter as VCR
                except Exception:
                    from visual_convergence_report import VisualConvergenceReporter as VCR

            val_lr, val_hr = prepare_2d_validation(val_image, factor_tuple)
            reporter = VCR(workspace_dir=out_dir, checkpoint_dir=os.path.join(out_dir, f"checkpoints/{output_prefix}"),
                           report_dir=os.path.join(out_dir, f"reports/{output_prefix}"),
                           html_filename=f"{output_prefix}_report.html", selection_metric="pcs",
                           reset_history=True)
            reporter.setup_validation_patches(val_lr, val_hr, factor=factor_tuple)
        except Exception as e:  # reporting must never kill training
            print(f"[curriculum] reporter disabled: {e}", flush=True)
            reporter = None

    def make_gen(st):
        kw = dict(generator_kwargs)
        kw.update(noise_std_range=tuple(st["noise"]), use_rician_noise=bool(st["rician"]),
                  zoom_range=tuple(st["zoom"]))
        if provenance_log_file:
            kw["return_provenance"] = True
        return blind_sr_generator(batch_size=batch_size, lr_patch_size=lr_patch_size, factor=factor,
                                  dimensionality=dim, **kw)

    def val_psnr_vs_bilinear():
        """Cheap warmup gate: (model psnr, bilinear psnr) on the validation patch."""
        import ants
        x = val_lr.numpy()[None, ..., None].astype("float32")
        sr = model.predict(x, verbose=0)[0, ..., 0]
        gt = val_hr.numpy()
        bil = ants.resample_image_to_target(val_lr, val_hr, interp_type=0).numpy()
        rng = float(gt.max() - gt.min()) or 1.0
        psnr = lambda a, b: float(10 * np.log10(rng ** 2 / (np.mean((a - b) ** 2) + 1e-12)))
        sr = sr[tuple(slice(0, s) for s in gt.shape)]
        return psnr(sr, gt), psnr(bil, gt)

    trace = {"output_prefix": output_prefix, "dim": dim, "factor": list(factor_tuple), "seed": int(seed),
             "batch_size": batch_size, "lr_patch_size": lr_patch_size, "edge_weight": float(edge_weight),
             "balancer_beta": balancer_beta, "stages": []}
    audited = False
    step = 0
    cur_w = {k: 0.0 for k in V}
    t_total = time.time()

    prov_fp = None
    prov_writer = None
    if provenance_log_file:
        import csv
        prov_dir = os.path.dirname(provenance_log_file)
        if prov_dir:
            os.makedirs(prov_dir, exist_ok=True)
        prov_fp = open(provenance_log_file, "w", newline="")
        prov_writer = csv.writer(prov_fp)
        prov_writer.writerow([
            "step", "stage", "it", "sample_idx", "class",
            "blur_sigma", "gamma", "interp", "noise_std",
            "loss_total", "loss_msq", "loss_l1", "loss_feat", "loss_tv", "loss_edge",
            "w_msq", "w_l1", "w_feat", "w_tv", "w_edge"
        ])

    for st in stages:
        name = st["name"]
        gen = make_gen(st)
        if not audited and os.environ.get("SIQ_SKIP_ALIGNMENT_AUDIT", "0") != "1":
            audit_pair_alignment(gen, factor_tuple, n_batches=2)
            audited = True
        model.optimizer.learning_rate.assign(float(st["lr"]))
        shares = st.get("shares")
        # initial weights
        if shares is None:                       # pure MSE warmup
            for k in V:
                V[k].assign(0.0)
            V["msq"].assign(1.0)
        else:
            if cur_w["msq"] > 0 and name == "Stage 1":
                V["msq"].assign(0.0)             # hand over from MSE to share-driven loss
            V["edge"].assign(float(edge_weight) if name == "Stage 3" else 0.0)
        mags = deque(maxlen=int(balancer_window))
        if shares is not None and any(float(V[k].value) == 0.0 for k in ("l1", "feat", "tv")):
            # Calibrate before the first update so no step trains with a zero loss:
            # median term magnitude over a few clean batches -> weights at their share targets.
            _g = make_gen(st)
            for _ in range(5):
                _batch = next(_g)
                _x, _y = _batch[0], _batch[1]
                _t = terms(ops.convert_to_tensor(_y), model(_x, training=False))
                mags.append({k: float(ops.mean(_t[k])) for k in ("l1", "feat", "tv")})
            _med = {k: float(np.median([m[k] for m in mags])) for k in ("l1", "feat", "tv")}
            _tot = float(sum(shares.values()))
            for k in ("l1", "feat", "tv"):
                V[k].assign((shares[k] / _tot) * total_scale / (_med[k] + 1e-12))
        stage_t0 = time.time()
        ckpt_pcs = []
        best_pcs, best_age = -1e9, 0
        used = 0
        gate_reached = None
        # Stage 1 starts from near-zero feat/tv weights and ramps via the dampened balancer
        for it in range(1, int(st["iters"]) + 1):
            batch_data = next(gen)
            if len(batch_data) == 3:
                x, y, meta_batch = batch_data
            else:
                x, y = batch_data
                meta_batch = None
            loss = model.train_on_batch(x, y)
            if not np.all(np.isfinite(np.asarray(loss, dtype=float))):
                print(f"[FATAL] non-finite loss at {name} it {it}; halting.", flush=True)
                trace["aborted"] = f"nonfinite@{name}:{it}"
                break
            step += 1
            used = it
            if shares is not None or (prov_fp is not None and meta_batch):
                # measure term magnitudes on this batch (forward only)
                yp = model(x, training=False)
                tm = terms(ops.convert_to_tensor(y), yp)
                if shares is not None:
                    mags.append({k: float(ops.mean(tm[k])) for k in ("l1", "feat", "tv")})
                    if it % balancer_freq == 0 and len(mags) >= 3:
                        med = {k: float(np.median([m[k] for m in mags])) for k in ("l1", "feat", "tv")}
                        tot = float(sum(shares.values()))
                        for k in ("l1", "feat", "tv"):
                            target_w = (shares[k] / tot) * total_scale / (med[k] + 1e-12)
                            if V[k].value == 0.0:
                                V[k].assign(target_w)
                            else:
                                V[k].assign(balancer_beta * float(V[k].value) + (1 - balancer_beta) * target_w)
                cur_w = {k: float(V[k].value) for k in V}
                if prov_fp is not None and meta_batch:
                    msq_arr = np.asarray(ops.convert_to_numpy(tm["msq"]))
                    l1_arr = np.asarray(ops.convert_to_numpy(tm["l1"]))
                    feat_arr = np.asarray(ops.convert_to_numpy(tm["feat"]))
                    tv_arr = np.asarray(ops.convert_to_numpy(tm["tv"]))
                    edge_arr = np.asarray(ops.convert_to_numpy(tm["edge"]))
                    for i in range(len(meta_batch)):
                        m_info = meta_batch[i]
                        c_name = m_info.get("class", "unknown")
                        msq_i = float(msq_arr[i])
                        l1_i = float(l1_arr[i])
                        feat_i = float(feat_arr[i])
                        tv_i = float(tv_arr[i])
                        edge_i = float(edge_arr[i])
                        tot_i = (cur_w.get("msq", 0.0) * msq_i +
                                 cur_w.get("l1", 0.0) * l1_i +
                                 cur_w.get("feat", 0.0) * feat_i +
                                 cur_w.get("tv", 0.0) * tv_i +
                                 cur_w.get("edge", 0.0) * edge_i)
                        prov_writer.writerow([
                            step, name, it, i, c_name,
                            round(float(m_info.get("blur_sigma", 0.0)), 4),
                            round(float(m_info.get("gamma", 1.0)), 4),
                            int(m_info.get("interp", 0)),
                            round(float(m_info.get("noise_std", 0.0)), 5),
                            round(tot_i, 6),
                            round(msq_i, 6), round(l1_i, 6), round(feat_i, 6),
                            round(tv_i, 6), round(edge_i, 6),
                            round(cur_w.get("msq", 0.0), 6), round(cur_w.get("l1", 0.0), 6),
                            round(cur_w.get("feat", 0.0), 8), round(cur_w.get("tv", 0.0), 6),
                            round(cur_w.get("edge", 0.0), 6)
                        ])
            else:
                cur_w = {k: float(V[k].value) for k in V}
            if it % 50 == 0 or it == 1:
                print(f"{name} {it}/{st['iters']} loss {float(loss):.5f}  w={ {k: round(v, 6) for k, v in cur_w.items() if v} }",
                      flush=True)
            is_eval = (it % eval_freq == 0 or it == int(st["iters"]))
            if is_eval:
                if st.get("gate") and val_lr is not None:
                    mp, bp = val_psnr_vs_bilinear()
                    if mp >= bp and gate_reached is None:
                        gate_reached = step
                        print(f"[gate] {name}: model PSNR {mp:.2f} >= bilinear {bp:.2f} at step {step}", flush=True)
                if reporter is not None:
                    is_ckpt = (it % checkpoint_freq == 0 or it == int(st["iters"]))
                    entry = reporter.record_checkpoint(
                        model, step, name, float(loss), is_convergence_step=is_ckpt,
                        loss_weights={**cur_w, "feature_type": "vgg", "feature_layer": feature_layer},
                        train_batch=(x, y))
                    if entry is not None:
                        pcs = float(entry.get("val_pcs", np.nan))
                        ckpt_pcs.append((step, pcs))
                        if np.isfinite(pcs) and pcs > best_pcs + 1e-3:
                            best_pcs, best_age = pcs, 0
                        else:
                            best_age += 1
                if st.get("gate") and gate_reached is not None and it >= 100:
                    break
                if (shares is not None and name in ("Stage 2", "Stage 3") and patience > 0
                        and it >= min_stage_frac * int(st["iters"]) and best_age >= patience):
                    print(f"[early-stop] {name}: PCS plateau for {best_age} evals (it {it})", flush=True)
                    break
        if trace.get("aborted"):
            break
        meas = {}
        if mags:
            meas = {k: float(np.median([m[k] for m in mags])) for k in ("l1", "feat", "tv")}
            contrib = {k: cur_w[k] * meas[k] for k in meas}
            tot = sum(contrib.values()) or 1.0
            meas_share = {k: 100.0 * v / tot for k, v in contrib.items()}
        else:
            meas_share = {}
        trace["stages"].append({"name": name, "iters_used": used, "iters_budget": int(st["iters"]),
                                "lr": st["lr"], "final_weights": cur_w, "measured_shares_pct": meas_share,
                                "median_term_magnitude": meas, "seconds": time.time() - stage_t0,
                                "gate_step": gate_reached, "pcs_history": ckpt_pcs})

    trace["seconds_total"] = time.time() - t_total
    if prov_fp is not None:
        prov_fp.flush()
        prov_fp.close()
        trace["provenance_log_file"] = provenance_log_file
    config = default_siq_config(model)
    config["model_type"] = "blind_sr_curriculum"
    config["upsample_factor"] = list(factor_tuple) if len(factor_tuple) > 1 else factor_tuple[0]
    config["loss_weights"] = {**cur_w, "feature_type": "vgg", "feature_layer": feature_layer}
    latest = os.path.join(out_dir, f"{output_prefix}_latest.keras")
    save_siq_model(latest, model, config)
    trace["latest_model"] = latest
    if dim == 2 and enable_report:
        try:
            trace["final_eval_latest"] = evaluate_model_2d(model, config, val_image, factor_tuple)
        except Exception as e:
            trace["final_eval_latest_error"] = str(e)
    with open(os.path.join(out_dir, f"{output_prefix}_curriculum_trace.json"), "w") as fh:
        json.dump(trace, fh, indent=2, default=float)
    return model, trace

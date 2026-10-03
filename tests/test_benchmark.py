"""Network-free tests for siq.benchmark (adapters, phase calibration, metrics, verdicts)."""
import numpy as np
import pytest
import ants
import siq
from siq import benchmark as B


def _case(n=2, size=64, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    from scipy.ndimage import gaussian_filter
    for _ in range(n):
        g = gaussian_filter(rng.rand(size, size), 1.2)
        g = ((g - g.min()) / (g.max() - g.min())).astype("float32")
        gi = ants.from_numpy(g)
        out.append((ants.resample_image(gi, (2.0, 2.0), use_voxels=False, interp_type=0), gi))
    return out


def test_pcsc_penalises_oversharpening():
    cl = _case(1)[0]
    g = cl[1].numpy()
    from scipy.ndimage import gaussian_filter
    mild = np.clip(g + 0.2 * (g - gaussian_filter(g, 1.5)), 0, 1)
    harsh = np.clip(g + 3.0 * (g - gaussian_filter(g, 1.5)), 0, 1)
    assert B.compute_pcsc(g, mild, (2, 2)) > B.compute_pcsc(g, harsh, (2, 2))
    # the repo's PCS (unbounded sharpness reward) is *not* guaranteed to agree - that is why PCSc exists
    assert B.sr_metrics(g, g, (2, 2))["pcsc"] > B.sr_metrics(g, harsh, (2, 2))["pcsc"]


def test_run_benchmark_classical_and_error_isolation():
    cases = {"toy": _case(2)}
    boom = B.SRMethod("boom", lambda l, g: 1 / 0, "other")
    res = B.run_benchmark(B.classical_methods() + [boom], cases, (2, 2), verbose=False)["toy"]
    assert "_error" in res["boom"] and "psnr" in res["bilinear"]
    assert res["bilinear"]["psnr"] > res["nearest"]["psnr"]


def test_phase_estimation_recovers_known_shift():
    cases = _case(3, size=96)
    shifted = lambda l, g: B.apply_shift(g.numpy(), (0.5, -0.4))     # an 'oracle' that is mis-aligned
    est = B.estimate_phase_shift(shifted, cases)
    # applying the estimate must undo (most of) the injected shift
    fixed = B.callable_method("x", shifted, phase="auto", factor=(2, 2), calibration_cases=cases)
    err = lambda fn: np.mean([np.mean((fn(l, g) - g.numpy()) ** 2) for l, g in cases])
    assert err(fixed.run) < 0.35 * err(shifted)
    assert np.sign(est[0]) == -np.sign(0.5) and np.sign(est[1]) == -np.sign(-0.4)


def test_half_pixel_phase_spec():
    assert list(B.resolve_phase_shift(None, "half-pixel", (2, 2))) == [-0.5, -0.5]
    assert list(B.resolve_phase_shift(None, "half-pixel", (1, 4))) == [0.0, -1.5]
    assert list(B.resolve_phase_shift(None, 0.25, (2, 2))) == [0.25, 0.25]


def test_verdicts_against_references():
    cases = {"toy": _case(2)}
    perfect = B.SRMethod("MODEL perfect", lambda l, g: g.numpy(), "model")
    bad = B.SRMethod("MODEL bad", lambda l, g: np.zeros_like(g.numpy()), "model")
    methods = B.classical_methods() + B.unsharp_methods() + B.linear_ls_methods(train_cache=None) + [perfect, bad]
    res = B.collapse_unsharp(B.run_benchmark(methods, cases, (2, 2), verbose=False)["toy"])
    v = B.verdicts(res)
    assert v["MODEL perfect"]["beats_best_classical"] and v["MODEL perfect"]["beats_linear_ceiling"]
    assert not v["MODEL bad"]["beats_best_classical"]
    md = B.format_markdown({"toy": res})
    assert "Verdicts" in md and "MODEL perfect" in md


def test_public_zoo_registry_and_helpful_import_error(monkeypatch):
    zoo = siq.list_public_models()
    assert zoo["edsr-base"]["verified"] and "hf_id" in zoo["msrn-bam"]
    monkeypatch.setenv("SIQ_PUBLIC_SR_PATH", "/nonexistent")
    try:
        import super_image  # noqa: F401
        pytest.skip("super-image importable here; the missing-dependency path cannot be exercised")
    except Exception:
        with pytest.raises(ImportError, match="super-image"):
            B.public_model_method("edsr-base")


@pytest.mark.skipif(__import__("importlib").util.find_spec("super_image") is None
                    and not __import__("os").path.isdir("/tmp/sipkgs/super_image"),
                    reason="super-image not installed")
def test_public_model_smoke_runs_offline_cache_only():
    pytest.importorskip("torch")
    cases = _case(1, size=32)
    try:
        m = B.public_model_method("edsr-base", (2, 2))
    except Exception as e:                      # no network / weights not cached
        pytest.skip(f"weights unavailable: {e}")
    out = m.run(*cases[0])
    assert out.shape == cases[0][1].shape


def test_html_report_is_self_contained_and_tabbed(tmp_path):
    cases = {"toy": _case(2), "toy2": _case(2, seed=1)}
    methods = B.classical_methods() + B.unsharp_methods() + B.linear_ls_methods(train_cache=None) + [
        B.SRMethod("MODEL perfect", lambda l, g: g.numpy(), "model")]
    res = B.run_benchmark(methods, cases, (2, 2), verbose=False)
    p = B.render_html_report(res, str(tmp_path / "r.html"), "t")
    html = open(p).read()
    assert html.count('class="tab') >= 2 and "MODEL perfect" in html and "Verdicts" in html
    assert "http://" not in html and "https://" not in html          # no external assets


def test_image_report_single_viewport_self_contained(tmp_path):
    cases = {"toy": _case(2)}
    methods = B.classical_methods() + B.unsharp_methods() + [
        B.SRMethod("MODEL perfect", lambda l, g: g.numpy(), "model")]
    p = B.render_image_report(methods, cases, str(tmp_path / "i.html"), (2, 2), keep=None)
    html = open(p).read()
    assert html.count('id="view"') == 1 and "data:image/png;base64" in html
    assert "http://" not in html and "https://" not in html

"""Plumbing/regression tests for siq.curriculum (tiny runs; no quality claims)."""
import numpy as np
import pytest
import siq
from siq import curriculum as C

TINY = [
    dict(name="Warmup", iters=4, lr=1e-4, noise=(0.0, 0.0), rician=False, zoom=(1.0, 1.0), shares=None),
    dict(name="Stage 1", iters=12, lr=5e-5, noise=(0.0, 0.0), rician=False, zoom=(1.0, 1.0),
         shares=dict(l1=70.0, feat=25.0, tv=5.0)),
    dict(name="Stage 2", iters=14, lr=3e-5, noise=(0.0, 0.02), rician=True, zoom=(0.9, 1.1),
         shares=dict(l1=30.0, feat=65.0, tv=5.0)),
    dict(name="Stage 3", iters=14, lr=1e-5, noise=(0.0, 0.01), rician=True, zoom=(0.9, 1.1),
         shares=dict(l1=30.0, feat=65.0, tv=5.0)),
]


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("curr"))
    audited = {}
    import siq.alignment as A
    orig = A.audit_pair_alignment
    A.audit_pair_alignment = lambda *a, **k: (audited.setdefault("called", True), orig(*a, **k))[1]
    try:
        model, trace = siq.train_blind_sr_curriculum(
            output_prefix="tiny", dimensionality=2, factor=2, stages=TINY, batch_size=2, lr_patch_size=16,
            edge_weight=0.5, seed=0, out_dir=out, enable_report=False, balancer_freq=2, balancer_beta=0.5,
            eval_freq=1000)   # fast dampening so a tiny run can actually reach the share targets
    finally:
        A.audit_pair_alignment = orig
    return model, trace, audited, out


def test_stages_and_trace(run):
    _, trace, _, out = run
    assert [s["name"] for s in trace["stages"]] == ["Warmup", "Stage 1", "Stage 2", "Stage 3"]
    assert all(s["iters_used"] == s["iters_budget"] for s in trace["stages"])
    import os, json
    assert os.path.exists(os.path.join(out, "tiny_curriculum_trace.json"))
    assert os.path.exists(os.path.join(out, "tiny_latest.keras"))
    assert json.load(open(os.path.join(out, "tiny_curriculum_trace.json")))["factor"] == [2, 2]


def test_no_zero_loss_at_share_stage_entry(run):
    """Regression: Stage 1 used to start with all-zero weights (no gradient until the first update)."""
    _, trace, _, _ = run
    w1 = trace["stages"][1]["final_weights"]
    assert w1["l1"] > 0 and w1["feat"] > 0 and w1["tv"] > 0
    assert w1["msq"] == 0.0                                    # MSE is handed over, not mixed in
    assert trace["stages"][0]["final_weights"]["msq"] == 1.0   # warmup is pure MSE


def test_measured_shares_track_targets(run):
    _, trace, _, _ = run
    for st, tgt in ((trace["stages"][1], (70, 25, 5)), (trace["stages"][3], (30, 65, 5))):
        m = st["measured_shares_pct"]
        for k, t in zip(("l1", "feat", "tv"), tgt):
            assert abs(m[k] - t) < 20, (st["name"], k, m)      # tiny noisy run: loose tolerance; default beta=0.97 is exercised by the real runs


def test_edge_weight_only_active_in_stage3_and_runs(run):
    _, trace, _, _ = run
    assert trace["stages"][1]["final_weights"]["edge"] == 0.0
    assert trace["stages"][2]["final_weights"]["edge"] == 0.0
    assert trace["stages"][3]["final_weights"]["edge"] == 0.5


def test_alignment_audit_runs_before_training(run):
    assert run[2].get("called"), "curriculum must run audit_pair_alignment (pair-formation invariant)"


def test_config_records_loss_weights_and_2d_patch(run):
    _, _, _, out = run
    import os
    model, cfg = siq.load_siq_model(os.path.join(out, "tiny_latest.keras"))
    lw = cfg["loss_weights"]
    assert {"l1", "feat", "tv", "msq", "edge"} <= set(lw)
    assert len(cfg["input_patch_shape"]) == 3 and len(cfg["output_patch_shape"]) == 3   # 2D + channel, not 64^3


def test_default_siq_config_is_dimension_aware():
    m2 = siq.default_dbpn(strider=[2, 2], dimensionality=2, nChannelsIn=1, nChannelsOut=1,
                          sigmoid_second_channel=False, option="small")
    cfg = siq.default_siq_config(m2)
    assert cfg["input_patch_shape"] == [64, 64, 1] and cfg["output_patch_shape"] == [128, 128, 1]


def test_default_stage_schedule_is_ordered_and_gated():
    names = [s["name"] for s in C.DEFAULT_STAGES]
    assert names == ["Warmup", "Stage 1", "Stage 2", "Stage 3"]
    assert C.DEFAULT_STAGES[0].get("gate") and C.DEFAULT_STAGES[0]["shares"] is None
    for s in C.DEFAULT_STAGES[1:]:
        assert abs(sum(s["shares"].values()) - 100.0) < 1e-6

import pytest
import numpy as np


def compute_cqs(val_ssim, val_gmsd, val_cbi):
    """Composite Quality Score: balances structure, edge sharpness, and artifact cleanliness."""
    return float(val_ssim - val_gmsd - val_cbi)


def test_cqs_penalizes_artifact_ringing():
    # Case A: Blurry model with no edge detail but slightly higher PSNR
    # SSIM=0.9737, GMSD=0.1387, CBI=0.1655 (high through-plane ringing)
    cqs_ringing = compute_cqs(0.9737, 0.1387, 0.1655)
    
    # Case B: Perceptual model with sharp edges and suppressed ringing
    # SSIM=0.9721, GMSD=0.1396, CBI=0.1449 (low through-plane ringing)
    cqs_clean = compute_cqs(0.9721, 0.1396, 0.1449)
    
    assert cqs_clean > cqs_ringing, f"Clean model CQS ({cqs_clean:.4f}) should exceed ringing model CQS ({cqs_ringing:.4f})"


def test_stage_hierarchy_selection():
    stage_rank_map = {
        "Warmup Gate": 0, "Initial Warmup": 0, "Warmup": 0,
        "Stage 1 Adaptation": 1, "Stage 1": 1,
        "Stage 2 Robustness": 2, "Stage 2": 2,
        "Stage 3 Refinement": 3, "Stage 3 Joint Fine-Tuning": 3, "Stage 3": 3,
    }
    
    def is_new_champion(cur_stage, cur_cqs, best_stage, best_cqs):
        cur_rank = stage_rank_map.get(cur_stage, 0)
        best_rank = stage_rank_map.get(best_stage, -1) if best_stage is not None else -1
        if cur_rank > best_rank:
            return True
        elif cur_rank == best_rank:
            return cur_cqs > best_cqs
        return False

    # Stage 1 initial
    assert is_new_champion("Stage 1 Adaptation", 0.50, None, -1.0) is True
    # Stage 2 supersedes Stage 1 even with different scale
    assert is_new_champion("Stage 2 Robustness", 0.75, "Stage 1 Adaptation", 0.77) is True
    # Stage 3 supersedes Stage 2
    assert is_new_champion("Stage 3 Refinement", 0.65, "Stage 2 Robustness", 0.80) is True
    # Within Stage 3, higher CQS wins
    assert is_new_champion("Stage 3 Refinement", 0.6876, "Stage 3 Refinement", 0.6556) is True
    # Within Stage 3, lower CQS does not win
    assert is_new_champion("Stage 3 Refinement", 0.6826, "Stage 3 Refinement", 0.6876) is False

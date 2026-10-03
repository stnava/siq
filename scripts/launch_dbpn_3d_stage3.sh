#!/usr/bin/env bash
set -o pipefail

cd /Users/stnava/code/siq

mkdir -p logs checkpoints/dbpn_3d reports/dbpn_3d

LOG_FILE="logs/run_dbpn_3d_stage3_$(date +%Y%m%d_%H%M%S).log"
echo "==========================================================" | tee -a "$LOG_FILE"
echo "Starting High-Capacity 3D DBPN Stage 3 Perceptual Refinement" | tee -a "$LOG_FILE"
echo "Starting Model: dbpn_3d_best_mdl.keras (22.29M params)" | tee -a "$LOG_FILE"
echo "Perceptual Backend: VGG19 Layer 6 (pseudo-3D unbiased)" | tee -a "$LOG_FILE"
echo "Pre-calibrated Weights: L1=3.866103, Feat=0.000114431652, TV=0.389915" | tee -a "$LOG_FILE"
echo "Loss Targets: 65% Perceptual / 30% MAE / 5% TV (Edge=0.0, CBI=0.0)" | tee -a "$LOG_FILE"
echo "Stage Range: 1501 - 5000 (LR=1e-5, clipnorm=1.0)" | tee -a "$LOG_FILE"
echo "Timestamp: $(date)" | tee -a "$LOG_FILE"
echo "Log file:  $LOG_FILE" | tee -a "$LOG_FILE"
echo "==========================================================" | tee -a "$LOG_FILE"

PYTHONUNBUFFERED=1 python tests/train_model_refinement.py dbpn \
  --dim 3 \
  --batch-size 2 \
  --lr-patch-size 32 \
  --load-model dbpn_3d_best_mdl.keras \
  --start-stage 3 \
  --stage1-iter 500 \
  --stage2-iter 1500 \
  --stage3-iter 5000 \
  --stage3-lr 1e-5 \
  --init-l1-weight 3.866103 \
  --init-feat-weight 0.000114431652 \
  --init-tv-weight 0.389915 \
  --target-percep 65.0 \
  --target-mae 30.0 \
  --target-tv 5.0 \
  --edge-weight 0.0 \
  --cbi-weight 0.0 \
  --linear-blend 1.0 \
  --selection-metric pcs \
  --perceptual-backend vgg \
  --clip-norm 1.0 \
  --checkpoint-freq 25 \
  --prefetch-size 4 \
  --balancer-freq 25 \
  --update-freq 25 \
  --smooth-window 100 \
  --dampening 0.97 \
  2>&1 | tee -a "$LOG_FILE"

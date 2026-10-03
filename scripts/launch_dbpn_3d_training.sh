#!/usr/bin/env bash
set -o pipefail

cd /Users/stnava/code/siq

mkdir -p logs checkpoints/dbpn_3d reports/dbpn_3d checkpoints/ldbpn_3d reports/ldbpn_3d

LOG_FILE="logs/run_dbpn_3d_$(date +%Y%m%d_%H%M%S).log"
echo "==========================================================" | tee -a "$LOG_FILE"
echo "Starting High-Capacity 3D DBPN Super-Resolution Training" | tee -a "$LOG_FILE"
echo "Perceptual Backend: VGG19 Layer 6 (pseudo-3D unbiased)" | tee -a "$LOG_FILE"
echo "Loss Targets: 65% Perceptual / 30% MAE / 5% TV (Edge=0.0, CBI=0.0)" | tee -a "$LOG_FILE"
echo "Timestamp: $(date)" | tee -a "$LOG_FILE"
echo "Log file:  $LOG_FILE" | tee -a "$LOG_FILE"
echo "==========================================================" | tee -a "$LOG_FILE"

# 1. Attempt High-Capacity 3D DBPN (default_dbpn 'large', 22.29M params) with VGG19 L6
set +e
PYTHONUNBUFFERED=1 python tests/train_model_refinement.py dbpn \
  --dim 3 \
  --batch-size 2 \
  --lr-patch-size 32 \
  --from-scratch \
  --reset-history \
  --stage1-iter 500 \
  --stage2-iter 1500 \
  --stage3-iter 5000 \
  --target-percep 65.0 \
  --target-mae 30.0 \
  --target-tv 5.0 \
  --edge-weight 0.0 \
  --cbi-weight 0.0 \
  --linear-blend 1.0 \
  --selection-metric pcs \
  --perceptual-backend vgg \
  --clip-norm 1.0 \
  --stage3-lr 1e-5 \
  --checkpoint-freq 25 \
  --prefetch-size 4 \
  --balancer-freq 25 \
  --update-freq 25 \
  --smooth-window 100 \
  --dampening 0.97 \
  2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=$?
set -e

# 2. Automated Failover: If High-Capacity DBPN fails, switch directly to L-DBPN 3D with VGG19 L6
if [ $EXIT_CODE -ne 0 ]; then
  echo "" | tee -a "$LOG_FILE"
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" | tee -a "$LOG_FILE"
  echo "WARNING: High-capacity 3D DBPN exited with code $EXIT_CODE." | tee -a "$LOG_FILE"
  echo "Failing over to Lightweight Deep Back-Projection Network (L-DBPN 3D)..." | tee -a "$LOG_FILE"
  echo "Timestamp: $(date)" | tee -a "$LOG_FILE"
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" | tee -a "$LOG_FILE"

  LDBPN_LOG="logs/run_ldbpn_3d_$(date +%Y%m%d_%H%M%S).log"
  echo "Logging L-DBPN run to: $LDBPN_LOG" | tee -a "$LOG_FILE"

  PYTHONUNBUFFERED=1 python tests/train_model_refinement.py ldbpn \
    --dim 3 \
    --batch-size 4 \
    --lr-patch-size 32 \
    --from-scratch \
    --reset-history \
    --stage1-iter 500 \
    --stage2-iter 1500 \
    --stage3-iter 5000 \
    --target-percep 65.0 \
    --target-mae 30.0 \
    --target-tv 5.0 \
    --edge-weight 0.0 \
    --cbi-weight 0.0 \
    --linear-blend 1.0 \
    --selection-metric pcs \
    --perceptual-backend vgg \
    --clip-norm 1.0 \
    --stage3-lr 1e-5 \
    --checkpoint-freq 25 \
    --prefetch-size 4 \
    --balancer-freq 25 \
    --update-freq 25 \
    --smooth-window 100 \
    --dampening 0.97 \
    2>&1 | tee -a "$LDBPN_LOG" | tee -a "$LOG_FILE"
fi

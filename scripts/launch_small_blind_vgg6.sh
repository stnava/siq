#!/usr/bin/env bash
# launch_small_blind_vgg6.sh
#
# Launches Small 3D DBPN Blind Super-Resolution Training with VGG19 Layer 6
# using transferred weights from dbpn_small_3d_from_refined.keras.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_small_blind_vgg6_${TIMESTAMP}.log"

echo "=========================================================="
echo "Starting Small 3D DBPN Blind SR Training (VGG19 Layer 6)"
echo "Model Init:  dbpn_small_3d_from_refined.keras"
echo "Log File:    ${LOG_FILE}"
echo "Timestamp:   $(date)"
echo "=========================================================="

python scripts/train_smallshort_blind_vgg6.py \
    --load-model dbpn_small_3d_from_refined.keras \
    --factor 2 \
    --pretrain-iters 50 \
    --iterations 800 \
    --batch-size 2 \
    --lr-patch-size 32 \
    --learning-rate 5e-5 \
    --msq-weight 3.86 \
    --feat-weight 1.14e-4 \
    --tv-weight 0.39 \
    --output-prefix siq_smallshort_train_2x2x2_1chan_featvggL6_blind \
    2>&1 | tee "${LOG_FILE}"

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

EPOCHS="${EPOCHS:-500}"
NUM_FOLDS="${NUM_FOLDS:-10}"
DEVICE_ARGS=()
if [[ -n "${DEVICE:-}" ]]; then
  DEVICE_ARGS=(--device "${DEVICE}")
fi

NUM_FOLDS="${NUM_FOLDS}" bash scripts/prepare_c_dataset.sh

python -m cross_validation train \
  --folds-dir outputs/C-dataset/cv10 \
  --out-dir outputs/C-dataset/cv10_full \
  --num-folds "${NUM_FOLDS}" \
  --hidden-dim 128 \
  --num-layers 2 \
  --epochs "${EPOCHS}" \
  --lr 0.001 \
  --weight-decay 0.00001 \
  --dropout 0.20 \
  --proj-dim 128 \
  --bpr-negatives-per-positive 1 \
  --lambda-cl 0.10 \
  --cl-temperature 0.20 \
  --gate-init -2.0 \
  --monitor aupr \
  --eval-every 10 \
  --seed 42 \
  "${DEVICE_ARGS[@]}"

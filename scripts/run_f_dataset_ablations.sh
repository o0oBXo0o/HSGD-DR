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

if [[ ! -d outputs/F-dataset/cv10/fold_00 ]]; then
  NUM_FOLDS="${NUM_FOLDS}" bash scripts/prepare_f_dataset.sh
fi

common=(
  --folds-dir outputs/F-dataset/cv10
  --num-folds "${NUM_FOLDS}"
  --hidden-dim 128
  --num-layers 2
  --epochs "${EPOCHS}"
  --lr 0.001
  --weight-decay 0.00001
  --dropout 0.20
  --proj-dim 128
  --bpr-negatives-per-positive 1
  --cl-temperature 0.20
  --gate-init -2.0
  --monitor aupr
  --eval-every 10
  --seed 42
  "${DEVICE_ARGS[@]}"
)

python -m cross_validation train "${common[@]}" \
  --out-dir outputs/F-dataset/ablations/full \
  --lambda-cl 0.10

python -m cross_validation train "${common[@]}" \
  --out-dir outputs/F-dataset/ablations/no_hybrid_gated \
  --lambda-cl 0.10 \
  --no-hybrid-gating

python -m cross_validation train "${common[@]}" \
  --out-dir outputs/F-dataset/ablations/no_contrastive_learning \
  --lambda-cl 0.0

python -m cross_validation train "${common[@]}" \
  --out-dir outputs/F-dataset/ablations/no_drug_disease_relation \
  --lambda-cl 0.10 \
  --drop-drug-disease-relation

python -m cross_validation train "${common[@]}" \
  --out-dir outputs/F-dataset/ablations/no_drug_protein_relation \
  --lambda-cl 0.10 \
  --drop-drug-protein-relation

python -m cross_validation train "${common[@]}" \
  --out-dir outputs/F-dataset/ablations/no_disease_protein_relation \
  --lambda-cl 0.10 \
  --drop-disease-protein-relation

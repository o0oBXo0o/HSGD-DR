#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

bash scripts/run_sample.sh

common=(
  --data-dir outputs/sample/graph
  --hidden-dim 32
  --num-layers 2
  --epochs 5
  --dropout 0.10
  --proj-dim 32
  --eval-every 1
  --seed 42
)

python -m hsgd_dr.training_evaluation "${common[@]}" --out-dir outputs/sample/ablation/full --lambda-cl 0.10
python -m hsgd_dr.training_evaluation "${common[@]}" --out-dir outputs/sample/ablation/no_hybrid_gated --lambda-cl 0.10 --no-hybrid-gating
python -m hsgd_dr.training_evaluation "${common[@]}" --out-dir outputs/sample/ablation/no_contrastive_learning --lambda-cl 0.0
python -m hsgd_dr.training_evaluation "${common[@]}" --out-dir outputs/sample/ablation/no_drug_disease_relation --lambda-cl 0.10 --drop-drug-disease-relation
python -m hsgd_dr.training_evaluation "${common[@]}" --out-dir outputs/sample/ablation/no_drug_protein_relation --lambda-cl 0.10 --drop-drug-protein-relation
python -m hsgd_dr.training_evaluation "${common[@]}" --out-dir outputs/sample/ablation/no_disease_protein_relation --lambda-cl 0.10 --drop-disease-protein-relation


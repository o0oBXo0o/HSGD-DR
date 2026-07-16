#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

python -m hsgd_dr.data_preprocessing \
  --dataset sample \
  --drug-disease data/sample/raw/drug_disease.csv \
  --drug-protein data/sample/raw/drug_protein.csv \
  --disease-protein data/sample/raw/disease_protein.csv \
  --out-dir outputs/sample/graph \
  --layout flat \
  --val-ratio 0.20 \
  --test-ratio 0.20 \
  --neg-ratio 3 \
  --train-neg-ratio 2 \
  --seed 42

python -m hsgd_dr.node_features \
  --graph-dir outputs/sample/graph \
  --drug-features data/sample/raw/drug_features.csv \
  --disease-features data/sample/raw/disease_features.csv \
  --protein-features data/sample/raw/protein_features.csv \
  --missing zero \
  --seed 42

python -m hsgd_dr.leakage_control \
  --data-dir outputs/sample/graph \
  --out outputs/sample/graph/leakage_report.json

python -m hsgd_dr.training_evaluation \
  --data-dir outputs/sample/graph \
  --out-dir outputs/sample/run \
  --hidden-dim 32 \
  --num-layers 2 \
  --epochs 10 \
  --lr 0.001 \
  --weight-decay 0.00001 \
  --dropout 0.10 \
  --proj-dim 32 \
  --lambda-cl 0.10 \
  --cl-temperature 0.20 \
  --gate-init -2.0 \
  --monitor aupr \
  --eval-every 2 \
  --seed 42

python -m hsgd_dr.prediction \
  --data-dir outputs/sample/graph \
  --checkpoint outputs/sample/run/best.pt \
  --out outputs/sample/run/top_predictions.csv \
  --top-k 10

python -m hsgd_dr.evidence_retrieval \
  --data-dir outputs/sample/graph \
  --pairs-file outputs/sample/run/top_predictions.csv \
  --out outputs/sample/run/path_evidence.csv


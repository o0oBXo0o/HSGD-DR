#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

NUM_FOLDS="${NUM_FOLDS:-10}"

python -m data_preprocessing \
  --dataset C-dataset \
  --drug-disease data/C-dataset/DrugDiseaseAssociationNumber.csv \
  --drug-protein data/C-dataset/DrugProteinAssociationNumber.csv \
  --disease-protein data/C-dataset/ProteinDiseaseAssociationNumber.csv \
  --out-dir outputs/C-dataset/graph \
  --layout flat \
  --val-ratio 0.10 \
  --test-ratio 0.10 \
  --neg-ratio 50 \
  --train-neg-ratio 5 \
  --seed 42

python -m node_features \
  --graph-dir outputs/C-dataset/graph \
  --drug-features data/C-dataset/Drug_mol2vec.csv \
  --disease-features data/C-dataset/DiseaseFeature.csv \
  --protein-features data/C-dataset/Protein_ESM.csv \
  --drug-feature-format matrix-id-first \
  --disease-feature-format matrix-id-first \
  --protein-feature-format matrix-id-first \
  --missing zero \
  --seed 42

python -m leakage_control \
  --data-dir outputs/C-dataset/graph \
  --out outputs/C-dataset/graph/leakage_report.json

python -m cross_validation split \
  --data-dir outputs/C-dataset/graph \
  --out-dir outputs/C-dataset/cv10 \
  --num-folds "${NUM_FOLDS}" \
  --neg-ratio 50 \
  --train-neg-ratio 5 \
  --seed 42

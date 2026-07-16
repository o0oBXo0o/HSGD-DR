#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

NUM_FOLDS="${NUM_FOLDS:-10}"

python -m data_preprocessing \
  --dataset F-dataset \
  --drug-disease data/F-dataset/DrugDiseaseAssociationNumber.csv \
  --drug-protein data/F-dataset/DrugProteinAssociationNumber.csv \
  --disease-protein data/F-dataset/ProteinDiseaseAssociationNumber.csv \
  --out-dir outputs/F-dataset/graph \
  --layout flat \
  --val-ratio 0.10 \
  --test-ratio 0.10 \
  --neg-ratio 50 \
  --train-neg-ratio 5 \
  --seed 42

python -m node_features \
  --graph-dir outputs/F-dataset/graph \
  --drug-features data/F-dataset/Drug_mol2vec.csv \
  --disease-features data/F-dataset/DiseaseFeature.csv \
  --protein-features data/F-dataset/Protein_ESM.csv \
  --drug-feature-format matrix-id-first \
  --disease-feature-format matrix-id-first \
  --protein-feature-format matrix-id-first \
  --missing zero \
  --seed 42

python -m leakage_control \
  --data-dir outputs/F-dataset/graph \
  --out outputs/F-dataset/graph/leakage_report.json

python -m cross_validation split \
  --data-dir outputs/F-dataset/graph \
  --out-dir outputs/F-dataset/cv10 \
  --num-folds "${NUM_FOLDS}" \
  --neg-ratio 50 \
  --train-neg-ratio 5 \
  --seed 42

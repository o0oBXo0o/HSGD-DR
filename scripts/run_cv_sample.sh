#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

bash scripts/run_sample.sh

python -m hsgd_dr.cross_validation split \
  --data-dir outputs/sample/graph \
  --out-dir outputs/sample/cv_folds \
  --num-folds 3 \
  --neg-ratio 3 \
  --train-neg-ratio 2 \
  --seed 42

python -m hsgd_dr.cross_validation train \
  --folds-dir outputs/sample/cv_folds \
  --out-dir outputs/sample/cv_run \
  --num-folds 3 \
  --hidden-dim 32 \
  --num-layers 2 \
  --epochs 5 \
  --dropout 0.10 \
  --proj-dim 32 \
  --lambda-cl 0.10 \
  --eval-every 1 \
  --seed 42


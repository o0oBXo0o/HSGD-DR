# HSGD-DR

## Project Description

HSGD-DR is a reproducible implementation of the manuscript method "HSGD-DR: A Heterogeneous GraphSAGE-DistMult Framework for Drug Repurposing over Biomedical Knowledge Graphs".

The project supports drug-disease link prediction over a multi-relational biomedical knowledge graph with three relation types:

- Drug-Disease: target relation for prediction.
- Drug-Protein: drug target context.
- Disease-Protein: disease mechanism context.

The implemented model follows the manuscript components only: leakage-controlled preprocessing, relation-specific Heterogeneous GraphSAGE channels, dimension-wise hybrid gating, DistMult scoring, Bayesian Personalized Ranking loss, relation-specific InfoNCE, 10-fold cross-validation, classification metrics, ablation switches, candidate ranking, and protein-mediated path retrieval.

## Setup Instructions

Create an environment from `requirements.txt`:

```bash
cd HSGD-DR
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```


## Usage

Run the full sample workflow with relative paths:

```bash
cd HSGD-DR
bash scripts/run_sample.sh
```

This command runs:

1. Data preprocessing from three raw tables.
2. Node feature alignment for Drug, Disease, and Protein.
3. Leakage guard.
4. HSGD-DR training and evaluation on the sample split.
5. Top candidate prediction.
6. Protein-mediated evidence path retrieval.

Main outputs:

```text
outputs/sample/graph/
outputs/sample/run/metrics.json
outputs/sample/run/predictions/
outputs/sample/run/top_predictions.csv
outputs/sample/run/path_evidence.csv
```

Run sample cross-validation:

```bash
cd HSGD-DR
bash scripts/run_cv_sample.sh
```

The protocol uses 10 folds. The sample script uses 3 folds only to keep the toy run short. The C-dataset and F-dataset source files used by the pipeline are included under `data/C-dataset/` and `data/F-dataset/`.

Example C-dataset 10-fold preparation:

```bash
cd HSGD-DR

python -m hsgd_dr.data_preprocessing \
  --dataset C-dataset \
  --drug-disease data/C-dataset/DrugDiseaseAssociationNumber.csv \
  --drug-protein data/C-dataset/DrugProteinAssociationNumber.csv \
  --disease-protein data/C-dataset/ProteinDiseaseAssociationNumber.csv \
  --out-dir outputs/C-dataset/graph \
  --layout flat \
  --val-ratio 0.10 \
  --test-ratio 0.10 \
  --neg-ratio 50 \
  --train-neg-ratio 5

python -m hsgd_dr.node_features \
  --graph-dir outputs/C-dataset/graph \
  --drug-features data/C-dataset/Drug_mol2vec.csv \
  --disease-features data/C-dataset/DiseaseFeature.csv \
  --protein-features data/C-dataset/Protein_ESM.csv

python -m hsgd_dr.cross_validation split \
  --data-dir outputs/C-dataset/graph \
  --out-dir outputs/C-dataset/cv10 \
  --num-folds 10 \
  --neg-ratio 50 \
  --train-neg-ratio 5

python -m hsgd_dr.cross_validation train \
  --folds-dir outputs/C-dataset/cv10 \
  --out-dir outputs/C-dataset/cv10_hsgd_dr \
  --num-folds 10 \
  --hidden-dim 128 \
  --num-layers 2 \
  --epochs 500 \
  --lambda-cl 0.10 \
  --monitor aupr
```

Use the same command shape with `data/F-dataset/...` and `outputs/F-dataset/...` for F-dataset.

Run manuscript-style ablations on the sample data:

```bash
cd HSGD-DR
bash scripts/run_ablation_sample.sh
```

The ablation switches correspond to the manuscript variants:

- Full
- `--no-hybrid-gating`
- `--lambda-cl 0.0`
- `--drop-drug-disease-relation`
- `--drop-drug-protein-relation`
- `--drop-disease-protein-relation`


## Directory Structure

```text
HSGD-DR/
  README.md
  requirements.txt
  environment.yml
  pyproject.toml
  configs/
    sample.yaml
    c_dataset.yaml
    f_dataset.yaml
  data/
    C-dataset/              # C-dataset source subset
    F-dataset/              # F-dataset source subset
    sample/raw/
      drug_disease.csv
      drug_protein.csv
      disease_protein.csv
      drug_features.csv
      disease_features.csv
      protein_features.csv
  scripts/
    run_sample.sh
    run_cv_sample.sh
    run_ablation_sample.sh
  src/hsgd_dr/
    data_preprocessing.py      # raw association tables -> leakage-safe graph CSVs
    node_features.py           # feature parsing and alignment
    pyg_dataset.py             # PyTorch Geometric HeteroData loader
    model_architecture.py      # Heterogeneous GraphSAGE, hybrid gate, DistMult
    contrastive_learning.py    # relation-specific InfoNCE
    negative_sampling.py       # BPR negatives and split-label negatives
    training_evaluation.py     # BPR training, checkpointing, metrics
    cross_validation.py        # leakage-controlled K-fold splitting and training
    prediction.py              # ranked Drug-Disease candidate generation
    evidence_retrieval.py      # Drug-Protein-Disease path retrieval
    leakage_control.py         # validation/test leakage checks
    metrics.py                 # AUROC, AUPR, Precision, Recall, F1, MCC
    schema.py                  # node, relation, and file naming schema
    io_utils.py                # shared I/O utilities
```


Prepared graph files are written to `outputs/.../graph/` using the canonical HSGD-DR naming scheme.

Contact Please feel free to contact us if you need any help: nvnui@ictu.edu.vn
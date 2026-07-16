# HSGD-DR

## Project Description

HSGD-DR is a reproducible implementation of the manuscript method "HSGD-DR: A Heterogeneous GraphSAGE-DistMult Framework for Drug Repurposing over Biomedical Knowledge Graphs".

The model performs drug-disease link prediction over a single multi-relational biomedical knowledge graph with three relation types:

- Drug-Disease: target relation to be predicted.
- Drug-Protein: drug target context.
- Disease-Protein: disease mechanism context.

The implementation includes only the components described in the manuscript: leakage-controlled preprocessing, relation-specific Heterogeneous GraphSAGE channels, dimension-wise hybrid gating, DistMult scoring, Bayesian Personalized Ranking loss, relation-specific InfoNCE, 10-fold cross-validation, classification metrics, F-dataset ablation switches, ranked candidate prediction, and protein-mediated path retrieval.

## Setup Instructions

```bash
cd HSGD-DR
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Requirement
```bash
  - python=3.10
  - pandas
  - numpy
  - scikit-learn
  - pyyaml
  - tqdm
  - pytorch
  - pyg
```

## Directory Structure

```text
HSGD-DR/
├── README.md
├── requirements.txt
├── environment.yml
├── pyproject.toml
├── configs/
│   ├── c_dataset.yaml
│   └── f_dataset.yaml
├── data/
│   ├── C-dataset/
│   │   ├── DrugDiseaseAssociationNumber.csv
│   │   ├── DrugProteinAssociationNumber.csv
│   │   ├── ProteinDiseaseAssociationNumber.csv
│   │   ├── Drug_mol2vec.csv
│   │   ├── DiseaseFeature.csv
│   │   └── Protein_ESM.csv
│   └── F-dataset/
│       ├── DrugDiseaseAssociationNumber.csv
│       ├── DrugProteinAssociationNumber.csv
│       ├── ProteinDiseaseAssociationNumber.csv
│       ├── Drug_mol2vec.csv
│       ├── DiseaseFeature.csv
│       └── Protein_ESM.csv
├── scripts/
│   ├── prepare_c_dataset.sh
│   ├── prepare_f_dataset.sh
│   ├── run_full_c_dataset.sh
│   ├── run_full_f_dataset.sh
│   └── run_f_dataset_ablations.sh
└── src/
    └── hsgd_dr/
        ├── data_preprocessing.py      # raw association tables -> leakage-safe graph CSVs
        ├── node_features.py           # feature parsing and alignment
        ├── pyg_dataset.py             # PyTorch Geometric HeteroData loader
        ├── model_architecture.py      # Heterogeneous GraphSAGE, hybrid gate, DistMult
        ├── contrastive_learning.py    # relation-specific InfoNCE
        ├── negative_sampling.py       # BPR negatives and split-label negatives
        ├── training_evaluation.py     # BPR training, checkpointing, metrics
        ├── cross_validation.py        # leakage-controlled K-fold splitting and training
        ├── prediction.py              # ranked Drug-Disease candidate generation
        ├── evidence_retrieval.py      # Drug-Protein-Disease path retrieval
        ├── leakage_control.py         # validation/test leakage checks
        ├── metrics.py                 # AUROC, AUPR, Precision, Recall, F1, MCC
        ├── schema.py                  # node, relation, and file naming schema
        └── io_utils.py                # shared I/O utilities
```

## Usage

Run the full C-dataset 10-fold HSGD-DR workflow:

```bash
cd HSGD-DR
bash scripts/run_full_c_dataset.sh
```

Run the full F-dataset 10-fold HSGD-DR workflow:

```bash
cd HSGD-DR
bash scripts/run_full_f_dataset.sh
```

Each full workflow runs:

1. Preprocess the three relation tables.
2. Align Drug, Disease, and Protein node features.
3. Run leakage control.
4. Create leakage-controlled 10-fold splits.
5. Train HSGD-DR with Heterogeneous GraphSAGE, hybrid gating, DistMult, BPR, and relation-specific InfoNCE.
6. Save fold metrics and mean/std summaries.

Main outputs:

```text
outputs/C-dataset/graph/
outputs/C-dataset/cv10/
outputs/C-dataset/cv10_hsgd_dr/cv_fold_metrics.csv
outputs/C-dataset/cv10_hsgd_dr/cv_summary.csv

outputs/F-dataset/graph/
outputs/F-dataset/cv10/
outputs/F-dataset/cv10_hsgd_dr/cv_fold_metrics.csv
outputs/F-dataset/cv10_hsgd_dr/cv_summary.csv
```

Run the F-dataset ablation study matching the manuscript variants:

```bash
cd HSGD-DR
bash scripts/run_f_dataset_ablations.sh
```

Ablation outputs are written under:

```text
outputs/F-dataset/ablations/
```

The ablation variants are:

- Full
- without hybrid gating
- without contrastive learning
- without drug-disease relation
- without drug-protein relation
- without disease-protein relation

Runtime controls:

```bash
EPOCHS=500 NUM_FOLDS=10 DEVICE=cuda bash scripts/run_full_f_dataset.sh
EPOCHS=500 NUM_FOLDS=10 DEVICE=cuda bash scripts/run_f_dataset_ablations.sh
```

Use `DEVICE=cpu` to force CPU execution.

## Manual Commands

Prepare C-dataset only:

```bash
cd HSGD-DR
bash scripts/prepare_c_dataset.sh
```

Prepare F-dataset only:

```bash
cd HSGD-DR
bash scripts/prepare_f_dataset.sh
```

Generate ranked candidates from a trained fold checkpoint:

```bash
cd HSGD-DR
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"

python -m hsgd_dr.prediction \
  --data-dir outputs/F-dataset/cv10/fold_00 \
  --checkpoint outputs/F-dataset/cv10_hsgd_dr/fold_00/best.pt \
  --out outputs/F-dataset/cv10_hsgd_dr/fold_00/top_predictions.csv \
  --top-k 100

python -m hsgd_dr.evidence_retrieval \
  --data-dir outputs/F-dataset/cv10/fold_00 \
  --pairs-file outputs/F-dataset/cv10_hsgd_dr/fold_00/top_predictions.csv \
  --out outputs/F-dataset/cv10_hsgd_dr/fold_00/path_evidence.csv
```

## Input CSV Schema

Canonical relation files:

```text
DrugDiseaseAssociationNumber.csv:      drug,disease or drug_id,disease_id
DrugProteinAssociationNumber.csv:      drug,protein or drug_id,protein_id
ProteinDiseaseAssociationNumber.csv:   disease,protein or disease_id,protein_id
```

Feature files are aligned to the prepared node IDs and written as:

```text
id,f0,f1,...,fk
```

Contact: nvnui@ictu.edu.vn

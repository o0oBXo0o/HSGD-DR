from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple

import pandas as pd

from .io_utils import ensure_dir


def read_pairs(path: Path, left: str, right: str) -> Set[Tuple[str, str]]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    return set(zip(df[left].astype(str), df[right].astype(str)))


def run(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    pairs = pd.read_csv(args.pairs_file)
    drug_protein = read_pairs(data_dir / "edges_drug_protein.csv", "drug_id", "protein_id")
    disease_protein = read_pairs(data_dir / "edges_disease_protein.csv", "disease_id", "protein_id")
    train_drug_disease = read_pairs(data_dir / "edges_drug_disease_train.csv", "drug_id", "disease_id")

    drug_to_proteins: Dict[str, Set[str]] = defaultdict(set)
    disease_to_proteins: Dict[str, Set[str]] = defaultdict(set)
    for drug_id, protein_id in drug_protein:
        drug_to_proteins[drug_id].add(protein_id)
    for disease_id, protein_id in disease_protein:
        disease_to_proteins[disease_id].add(protein_id)

    rows = []
    for _, row in pairs.iterrows():
        drug_id = str(row["drug_id"])
        disease_id = str(row["disease_id"])
        score = row.get("score", None)
        direct_train_edge = int((drug_id, disease_id) in train_drug_disease)
        shared = sorted(drug_to_proteins.get(drug_id, set()) & disease_to_proteins.get(disease_id, set()))
        if shared:
            for protein_id in shared[: args.max_paths_per_pair]:
                rows.append(
                    {
                        "drug_id": drug_id,
                        "disease_id": disease_id,
                        "score": score,
                        "protein_id": protein_id,
                        "path_type": "Drug-drug_protein-Protein-disease_protein-Disease",
                        "path": f"Drug({drug_id})-[:DRUG_PROTEIN]->Protein({protein_id})<-[:DISEASE_PROTEIN]-Disease({disease_id})",
                        "direct_train_edge": direct_train_edge,
                    }
                )
        else:
            rows.append(
                {
                    "drug_id": drug_id,
                    "disease_id": disease_id,
                    "score": score,
                    "protein_id": "",
                    "path_type": "no_shared_protein_path_found",
                    "path": "",
                    "direct_train_edge": direct_train_edge,
                }
            )
    out = pd.DataFrame(rows)
    ensure_dir(Path(args.out).parent)
    out.to_csv(args.out, index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Retrieve protein-mediated evidence paths for HSGD-DR predictions.")
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--pairs-file", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--max-paths-per-pair", type=int, default=20)
    return p


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()


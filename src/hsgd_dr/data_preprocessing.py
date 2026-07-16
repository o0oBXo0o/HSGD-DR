from __future__ import annotations

import argparse
from pathlib import Path
from typing import Set, Tuple

import numpy as np
import pandas as pd

from .io_utils import canonical_pair_frame, ensure_dir, save_dataframe, set_seed, write_json, write_nodes
from .negative_sampling import sample_negative_pairs, with_labels


def split_positive_edges(df: pd.DataFrame, val_ratio: float, test_ratio: float, seed: int):
    if len(df) < 3:
        raise ValueError("Need at least 3 positive Drug-Disease pairs for train/val/test.")
    temp_ratio = val_ratio + test_ratio
    if not 0 < temp_ratio < 1:
        raise ValueError("val_ratio + test_ratio must be in (0, 1).")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    n_test = max(1, int(round(len(df) * test_ratio)))
    n_val = max(1, int(round(len(df) * val_ratio)))
    if n_val + n_test >= len(df):
        raise ValueError("Split ratios leave no training positives.")
    test_idx = order[:n_test]
    val_idx = order[n_test:n_test + n_val]
    train_idx = order[n_test + n_val:]
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    test_df = df.iloc[test_idx]
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def prepare_graph(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    graph_dir = ensure_dir(out_dir / "graph" if args.layout == "nested" else out_dir)

    drug_disease = canonical_pair_frame(
        args.drug_disease,
        ["drug_id", "drug", "Drug", "drugbank_id", "compound_id"],
        ["disease_id", "disease", "Disease", "mesh_id"],
        "drug_id",
        "disease_id",
    )
    drug_protein = canonical_pair_frame(
        args.drug_protein,
        ["drug_id", "drug", "Drug", "drugbank_id", "compound_id"],
        ["protein_id", "protein", "gene_id", "gene", "target_id"],
        "drug_id",
        "protein_id",
    )
    disease_protein = canonical_pair_frame(
        args.disease_protein,
        ["disease_id", "disease", "Disease", "mesh_id"],
        ["protein_id", "protein", "gene_id", "gene", "target_id"],
        "disease_id",
        "protein_id",
    )

    train_pos, val_pos, test_pos = split_positive_edges(
        drug_disease,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    known_pos: Set[Tuple[str, str]] = set(map(tuple, drug_disease[["drug_id", "disease_id"]].astype(str).values.tolist()))
    drugs = sorted(set(drug_disease.drug_id).union(set(drug_protein.drug_id)))
    diseases = sorted(set(drug_disease.disease_id).union(set(disease_protein.disease_id)))
    proteins = sorted(set(drug_protein.protein_id).union(set(disease_protein.protein_id)))

    val_neg = sample_negative_pairs(drugs, diseases, known_pos, len(val_pos) * args.neg_ratio, args.seed + 101)
    test_forbidden = known_pos.union(set(map(tuple, val_neg.values.tolist())))
    test_neg = sample_negative_pairs(drugs, diseases, test_forbidden, len(test_pos) * args.neg_ratio, args.seed + 202)
    train_forbidden = test_forbidden.union(set(map(tuple, test_neg.values.tolist())))
    train_neg = sample_negative_pairs(
        drugs,
        diseases,
        train_forbidden,
        len(train_pos) * min(args.neg_ratio, args.train_neg_ratio),
        args.seed + 303,
    )

    write_nodes(drugs, graph_dir / "nodes_drug.csv")
    write_nodes(diseases, graph_dir / "nodes_disease.csv")
    write_nodes(proteins, graph_dir / "nodes_protein.csv")
    save_dataframe(train_pos[["drug_id", "disease_id"]], graph_dir / "edges_drug_disease_train.csv")
    save_dataframe(drug_disease[["drug_id", "disease_id"]], graph_dir / "rels_drug_disease_all.csv")
    save_dataframe(drug_protein[["drug_id", "protein_id"]], graph_dir / "edges_drug_protein.csv")
    save_dataframe(disease_protein[["disease_id", "protein_id"]], graph_dir / "edges_disease_protein.csv")
    save_dataframe(with_labels(train_pos, train_neg, args.source_name, "train"), graph_dir / "labels_train.csv")
    save_dataframe(with_labels(val_pos, val_neg, args.source_name, "val"), graph_dir / "labels_val.csv")
    save_dataframe(with_labels(test_pos, test_neg, args.source_name, "test"), graph_dir / "labels_test.csv")

    write_json(
        {
            "dataset": args.dataset,
            "source_name": args.source_name,
            "n_drugs": len(drugs),
            "n_diseases": len(diseases),
            "n_proteins": len(proteins),
            "n_drug_disease_relations": len(drug_disease),
            "n_drug_protein_relations": len(drug_protein),
            "n_disease_protein_relations": len(disease_protein),
            "n_train_pos": len(train_pos),
            "n_val_pos": len(val_pos),
            "n_test_pos": len(test_pos),
            "neg_ratio_eval": args.neg_ratio,
            "train_neg_ratio_written": min(args.neg_ratio, args.train_neg_ratio),
        },
        out_dir / "prepare_report.json",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Prepare HSGD-DR heterogeneous graph CSV files.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--drug-disease", required=True, type=Path)
    p.add_argument("--drug-protein", required=True, type=Path)
    p.add_argument("--disease-protein", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--layout", choices=["nested", "flat"], default="flat")
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument("--test-ratio", type=float, default=0.10)
    p.add_argument("--neg-ratio", type=int, default=50)
    p.add_argument("--train-neg-ratio", type=int, default=5)
    p.add_argument("--source-name", default="HSGD-DR")
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    prepare_graph(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()

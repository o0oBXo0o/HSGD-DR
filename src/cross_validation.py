from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Set, Tuple

import numpy as np
import pandas as pd

from io_utils import ensure_dir, write_json
from negative_sampling import sample_negative_pairs, with_labels


def copy_optional(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def kfold_test_indices(n_items: int, n_splits: int, seed: int):
    if n_splits < 2:
        raise ValueError("num_folds must be >= 2.")
    if n_items < n_splits:
        raise ValueError("Number of positive pairs must be >= num_folds.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_items)
    return [np.asarray(chunk, dtype=int) for chunk in np.array_split(order, n_splits)]


def make_folds(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    out_dir = ensure_dir(args.out_dir)
    rels = pd.read_csv(data_dir / "rels_drug_disease_all.csv")
    rels = rels[["drug_id", "disease_id"]].astype(str).drop_duplicates().reset_index(drop=True)
    folds = kfold_test_indices(len(rels), args.num_folds, args.seed)
    drugs = pd.read_csv(data_dir / "nodes_drug.csv")["id"].astype(str).tolist()
    diseases = pd.read_csv(data_dir / "nodes_disease.csv")["id"].astype(str).tolist()
    known_pos: Set[Tuple[str, str]] = set(map(tuple, rels.values.tolist()))
    summary = []
    for fold_id, test_idx in enumerate(folds):
        val_fold = (fold_id + args.val_shift) % args.num_folds
        val_idx = folds[val_fold]
        holdout = set(test_idx.tolist()) | set(val_idx.tolist())
        train_df = rels.iloc[[i for i in range(len(rels)) if i not in holdout]].reset_index(drop=True)
        val_df = rels.iloc[sorted(set(val_idx.tolist()))].reset_index(drop=True)
        test_df = rels.iloc[sorted(set(test_idx.tolist()))].reset_index(drop=True)
        fold_dir = ensure_dir(out_dir / f"fold_{fold_id:02d}")
        for fname in [
            "nodes_drug.csv",
            "nodes_disease.csv",
            "nodes_protein.csv",
            "edges_drug_protein.csv",
            "edges_disease_protein.csv",
            "features_drug.csv",
            "features_disease.csv",
            "features_protein.csv",
            "feature_manifest.json",
        ]:
            copy_optional(data_dir / fname, fold_dir / fname)
        train_df.to_csv(fold_dir / "edges_drug_disease_train.csv", index=False)
        rels.to_csv(fold_dir / "rels_drug_disease_all.csv", index=False)
        val_neg = sample_negative_pairs(drugs, diseases, known_pos, len(val_df) * args.neg_ratio, args.seed + fold_id * 17)
        test_neg = sample_negative_pairs(
            drugs,
            diseases,
            known_pos.union(set(map(tuple, val_neg.values.tolist()))),
            len(test_df) * args.neg_ratio,
            args.seed + fold_id * 17 + 1,
        )
        train_neg = sample_negative_pairs(
            drugs,
            diseases,
            known_pos.union(set(map(tuple, val_neg.values.tolist()))).union(set(map(tuple, test_neg.values.tolist()))),
            len(train_df) * min(args.neg_ratio, args.train_neg_ratio),
            args.seed + fold_id * 17 + 2,
        )
        with_labels(train_df, train_neg, args.source_name, "train").to_csv(fold_dir / "labels_train.csv", index=False)
        with_labels(val_df, val_neg, args.source_name, "val").to_csv(fold_dir / "labels_val.csv", index=False)
        with_labels(test_df, test_neg, args.source_name, "test").to_csv(fold_dir / "labels_test.csv", index=False)
        summary.append({"fold": fold_id, "train_pos": len(train_df), "val_pos": len(val_df), "test_pos": len(test_df)})
    pd.DataFrame(summary).to_csv(out_dir / "cv_split_summary.csv", index=False)
    write_json({"num_folds": args.num_folds, "val_shift": args.val_shift, "folds": summary}, out_dir / "cv_split_summary.json")


def flatten_metrics(metrics_json: Path, fold: int) -> dict:
    import json

    if not metrics_json.exists():
        return {"fold": fold, "status": "missing_metrics"}
    with metrics_json.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    row = {"fold": fold, "status": "ok"}
    for split, values in obj.get("metrics", {}).items():
        for key, value in values.items():
            row[f"{split}_{key}"] = value
    return row


def train_folds(args: argparse.Namespace) -> None:
    out_dir = ensure_dir(args.out_dir)
    write_json(vars(args), out_dir / "cv_run_config.json")
    rows = []
    for fold in range(args.num_folds):
        fold_data = Path(args.folds_dir) / f"fold_{fold:02d}"
        fold_out = out_dir / f"fold_{fold:02d}"
        cmd = [
            sys.executable,
            "-m",
            "training_evaluation",
            "--data-dir",
            str(fold_data),
            "--out-dir",
            str(fold_out),
            "--hidden-dim",
            str(args.hidden_dim),
            "--num-layers",
            str(args.num_layers),
            "--epochs",
            str(args.epochs),
            "--lr",
            str(args.lr),
            "--weight-decay",
            str(args.weight_decay),
            "--dropout",
            str(args.dropout),
            "--proj-dim",
            str(args.proj_dim),
            "--sage-aggr",
            args.sage_aggr,
            "--bpr-negatives-per-positive",
            str(args.bpr_negatives_per_positive),
            "--lambda-cl",
            str(args.lambda_cl),
            "--cl-temperature",
            str(args.cl_temperature),
            "--cl-max-nodes",
            str(args.cl_max_nodes),
            "--gate-init",
            str(args.gate_init),
            "--eval-threshold",
            str(args.eval_threshold),
            "--monitor",
            args.monitor,
            "--eval-every",
            str(args.eval_every),
            "--seed",
            str(args.seed + fold),
        ]
        for flag_name in [
            "no_hybrid_gating",
            "drop_drug_disease_relation",
            "drop_drug_protein_relation",
            "drop_disease_protein_relation",
            "skip_leakage_guard",
        ]:
            if getattr(args, flag_name):
                cmd.append("--" + flag_name.replace("_", "-"))
        if args.device is not None:
            cmd.extend(["--device", args.device])
        subprocess.run(cmd, check=True)
        rows.append(flatten_metrics(fold_out / "metrics.json", fold))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "cv_fold_metrics.csv", index=False)
    numeric = df.select_dtypes(include="number")
    summary = numeric.agg(["mean", "std", "min", "max"]).reset_index().rename(columns={"index": "stat"})
    summary.to_csv(out_dir / "cv_summary.csv", index=False)
    write_json(summary.to_dict(orient="records"), out_dir / "cv_summary.json")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HSGD-DR leakage-controlled cross-validation.")
    sub = p.add_subparsers(dest="command", required=True)

    split = sub.add_parser("split", help="Create leakage-controlled fold directories.")
    split.add_argument("--data-dir", required=True, type=Path)
    split.add_argument("--out-dir", required=True, type=Path)
    split.add_argument("--num-folds", type=int, default=10)
    split.add_argument("--val-shift", type=int, default=1)
    split.add_argument("--neg-ratio", type=int, default=50)
    split.add_argument("--train-neg-ratio", type=int, default=5)
    split.add_argument("--source-name", default="HSGD-DR")
    split.add_argument("--seed", type=int, default=42)

    train = sub.add_parser("train", help="Train all fold directories and summarize metrics.")
    train.add_argument("--folds-dir", required=True, type=Path)
    train.add_argument("--out-dir", required=True, type=Path)
    train.add_argument("--num-folds", type=int, default=10)
    train.add_argument("--hidden-dim", type=int, default=128)
    train.add_argument("--num-layers", type=int, default=2)
    train.add_argument("--epochs", type=int, default=500)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-5)
    train.add_argument("--dropout", type=float, default=0.2)
    train.add_argument("--proj-dim", type=int, default=128)
    train.add_argument("--sage-aggr", default="mean", choices=["mean", "max", "lstm"])
    train.add_argument("--bpr-negatives-per-positive", type=int, default=1)
    train.add_argument("--lambda-cl", type=float, default=0.1)
    train.add_argument("--cl-temperature", type=float, default=0.2)
    train.add_argument("--cl-max-nodes", type=int, default=4096)
    train.add_argument("--gate-init", type=float, default=-2.0)
    train.add_argument("--no-hybrid-gating", action="store_true")
    train.add_argument("--drop-drug-disease-relation", action="store_true")
    train.add_argument("--drop-drug-protein-relation", action="store_true")
    train.add_argument("--drop-disease-protein-relation", action="store_true")
    train.add_argument("--eval-threshold", type=float, default=0.5)
    train.add_argument("--monitor", default="aupr", choices=["aupr", "auroc", "precision", "recall", "f1_score", "mcc"])
    train.add_argument("--eval-every", type=int, default=10)
    train.add_argument("--skip-leakage-guard", action="store_true")
    train.add_argument("--device", default=None)
    train.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.command == "split":
        make_folds(args)
    elif args.command == "train":
        train_folds(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()

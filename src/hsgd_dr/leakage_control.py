from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Set, Tuple

import pandas as pd

from .io_utils import write_json
from .schema import LABEL_FILES

Pair = Tuple[str, str]


def _read_pairs(path: Path, left: str = "drug_id", right: str = "disease_id") -> Set[Pair]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if left not in df.columns or right not in df.columns:
        if len(df.columns) < 2:
            return set()
        left, right = df.columns[0], df.columns[1]
    return set(map(tuple, df[[left, right]].astype(str).values.tolist()))


def _read_label_pairs(path: Path, label: int | None = None) -> Set[Pair]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if label is not None and "label" in df.columns:
        df = df[pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int) == label]
    return set(map(tuple, df[["drug_id", "disease_id"]].astype(str).values.tolist()))


def check_leakage(data_dir: Path) -> Dict[str, object]:
    train_graph = _read_pairs(data_dir / "edges_drug_disease_train.csv")
    known_pos = _read_pairs(data_dir / "rels_drug_disease_all.csv") or train_graph.copy()
    report: Dict[str, object] = {
        "train_graph_positive_edges": len(train_graph),
        "known_positive_edges": len(known_pos),
        "splits": {},
        "errors": [],
    }
    unknown_train = sorted(train_graph - known_pos)
    if unknown_train:
        report["errors"].append(
            {
                "type": "train_graph_not_in_known_positives",
                "count": len(unknown_train),
                "examples": unknown_train[:10],
            }
        )
    for fname in LABEL_FILES:
        path = data_dir / fname
        if not path.exists():
            continue
        pos = _read_label_pairs(path, 1)
        neg = _read_label_pairs(path, 0)
        pos_overlap_train = sorted(pos & train_graph)
        neg_overlap_known = sorted(neg & known_pos)
        split_report = {
            "positive_count": len(pos),
            "negative_count": len(neg),
            "positive_overlap_train_graph": len(pos_overlap_train),
            "negative_overlap_known_positive": len(neg_overlap_known),
            "positive_overlap_train_examples": pos_overlap_train[:10],
            "negative_overlap_known_examples": neg_overlap_known[:10],
        }
        report["splits"][fname] = split_report
        if fname != "labels_train.csv" and pos_overlap_train:
            report["errors"].append(
                {
                    "type": "eval_positive_overlap_train_graph",
                    "file": fname,
                    "count": len(pos_overlap_train),
                    "examples": pos_overlap_train[:10],
                }
            )
        if neg_overlap_known:
            report["errors"].append(
                {
                    "type": "negative_overlap_known_positive",
                    "file": fname,
                    "count": len(neg_overlap_known),
                    "examples": neg_overlap_known[:10],
                }
            )
    report["ok"] = len(report["errors"]) == 0
    return report


def run(args: argparse.Namespace) -> None:
    report = check_leakage(Path(args.data_dir))
    write_json(report, args.out)
    if not report["ok"] and not args.warn_only:
        raise SystemExit(f"Leakage guard failed. See {args.out}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Leakage checks for HSGD-DR splits.")
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--warn-only", action="store_true")
    return p


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()


from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .io_utils import ensure_dir, first_existing_column, read_node_ids, save_dataframe, write_json
from .schema import DISEASE, DRUG, FEATURE_FILE_MAP, NODE_FILE_MAP, PROTEIN

ID_CANDIDATES = ["id", "node_id", "drug_id", "disease_id", "protein_id", "gene_id", "target_id", "name"]


def parse_vector_cell(x) -> List[float]:
    if isinstance(x, (list, tuple, np.ndarray)):
        return [float(v) for v in x]
    s = str(x).strip()
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple)):
            return [float(v) for v in val]
    except Exception:
        pass
    return [float(v) for v in s.replace(";", " ").replace(",", " ").split() if v != ""]


def load_feature_mapping(path: Path, node_ids: List[str], fmt: str = "auto") -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    if not path.exists():
        return {}, {"path": str(path), "status": "missing"}
    candidates = []

    def add_candidate(name: str, mapping: Dict[str, np.ndarray]) -> None:
        if not mapping:
            return
        match = sum(1 for nid in node_ids if nid in mapping)
        nonzero = sum(1 for nid in node_ids if nid in mapping and float(np.abs(mapping[nid]).sum()) > 0)
        dim = len(next(iter(mapping.values())))
        candidates.append((name, match, nonzero, dim, mapping))

    df = pd.read_csv(path)
    id_col = first_existing_column(df, ID_CANDIDATES)
    vector_col = first_existing_column(df, ["vector", "features", "embedding", "x"])
    if id_col is not None and vector_col is not None and fmt in {"auto", "id-vector"}:
        mapping = {str(r[id_col]).strip(): np.asarray(parse_vector_cell(r[vector_col]), dtype=np.float32) for _, r in df.iterrows()}
        add_candidate("id-vector", mapping)
    if id_col is not None and fmt in {"auto", "id-wide"}:
        feat_cols = [c for c in df.columns if c != id_col and pd.api.types.is_numeric_dtype(df[c])]
        if feat_cols:
            mapping = {str(r[id_col]).strip(): r[feat_cols].to_numpy(dtype=np.float32) for _, r in df.iterrows()}
            add_candidate("id-wide", mapping)

    if fmt in {"auto", "matrix-node-order", "matrix-row-index0", "matrix-row-index1"}:
        mdf = pd.read_csv(path, header=None)
        mat = mdf.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if mat.ndim == 2 and mat.shape[1] > 0:
            if fmt in {"auto", "matrix-node-order"} and mat.shape[0] == len(node_ids):
                add_candidate("matrix-node-order", {nid: mat[i] for i, nid in enumerate(node_ids)})
            if fmt in {"auto", "matrix-row-index0"}:
                add_candidate("matrix-row-index0", {str(i): mat[i] for i in range(mat.shape[0])})
            if fmt in {"auto", "matrix-row-index1"}:
                add_candidate("matrix-row-index1", {str(i + 1): mat[i] for i in range(mat.shape[0])})

    if not candidates:
        raise ValueError(f"Cannot parse feature file {path} with format={fmt}")
    if fmt != "auto":
        for candidate in candidates:
            if candidate[0] == fmt:
                return candidate[4], {
                    "path": str(path),
                    "format": candidate[0],
                    "matched_nodes": candidate[1],
                    "nonzero_rows": candidate[2],
                    "dim": candidate[3],
                }
        raise ValueError(f"Requested feature format {fmt} was not detected for {path}")
    candidates.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
    best = candidates[0]
    return best[4], {"path": str(path), "format": best[0], "matched_nodes": best[1], "nonzero_rows": best[2], "dim": best[3]}


def align_features(node_ids: List[str], mapping: Dict[str, np.ndarray], missing: str, rng: np.random.Generator) -> pd.DataFrame:
    dim = len(next(iter(mapping.values()))) if mapping else 0
    if dim == 0:
        raise ValueError("Feature mapping is empty; cannot infer feature dimension.")
    present = [mapping[nid] for nid in node_ids if nid in mapping]
    mean = np.mean(np.stack(present), axis=0) if present else np.zeros(dim, dtype=np.float32)
    rows = []
    for node_id in node_ids:
        if node_id in mapping:
            values = np.asarray(mapping[node_id], dtype=np.float32)
        elif missing == "zero":
            values = np.zeros(dim, dtype=np.float32)
        elif missing == "mean":
            values = mean.astype(np.float32)
        elif missing == "random-normal":
            values = rng.normal(0, 0.01, size=dim).astype(np.float32)
        elif missing == "error":
            raise ValueError(f"Missing feature for node {node_id}")
        else:
            raise ValueError(f"Unknown missing policy: {missing}")
        rows.append([node_id] + values.astype(float).tolist())
    return pd.DataFrame(rows, columns=["id"] + [f"f{i}" for i in range(dim)])


def prepare_node_features(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    graph_dir = Path(args.graph_dir)
    feature_paths = {
        DRUG: args.drug_features,
        DISEASE: args.disease_features,
        PROTEIN: args.protein_features,
    }
    formats = {
        DRUG: args.drug_feature_format,
        DISEASE: args.disease_feature_format,
        PROTEIN: args.protein_feature_format,
    }
    manifest = {}
    for node_type in (DRUG, DISEASE, PROTEIN):
        node_ids = read_node_ids(graph_dir / NODE_FILE_MAP[node_type])
        mapping, info = load_feature_mapping(Path(feature_paths[node_type]), node_ids, formats[node_type])
        df = align_features(node_ids, mapping, args.missing, rng)
        nonzero = int((df.iloc[:, 1:].abs().sum(axis=1) > 0).sum())
        total_abs = float(df.iloc[:, 1:].abs().sum().sum())
        if total_abs == 0 and not args.allow_all_zero:
            raise ValueError(f"{node_type} features are all zero. Use --allow-all-zero for ablation only.")
        out_name = FEATURE_FILE_MAP[node_type]
        save_dataframe(df, graph_dir / out_name)
        manifest[node_type] = {
            **info,
            "output": str(graph_dir / out_name),
            "rows": len(df),
            "nonzero_rows": nonzero,
            "abs_sum": total_abs,
        }
    write_json(manifest, graph_dir / "feature_manifest.json")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Align Drug, Disease, and Protein node features for HSGD-DR.")
    p.add_argument("--graph-dir", required=True, type=Path)
    p.add_argument("--drug-features", required=True, type=Path)
    p.add_argument("--disease-features", required=True, type=Path)
    p.add_argument("--protein-features", required=True, type=Path)
    p.add_argument("--drug-feature-format", default="auto", choices=["auto", "id-wide", "id-vector", "matrix-node-order", "matrix-row-index0", "matrix-row-index1"])
    p.add_argument("--disease-feature-format", default="auto", choices=["auto", "id-wide", "id-vector", "matrix-node-order", "matrix-row-index0", "matrix-row-index1"])
    p.add_argument("--protein-feature-format", default="auto", choices=["auto", "id-wide", "id-vector", "matrix-node-order", "matrix-row-index0", "matrix-row-index1"])
    p.add_argument("--missing", default="zero", choices=["zero", "mean", "random-normal", "error"])
    p.add_argument("--allow-all-zero", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    prepare_node_features(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()


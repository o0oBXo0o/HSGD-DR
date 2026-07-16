from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(obj: object, path: str | Path, indent: int = 2) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(make_json_safe(obj), f, indent=indent, ensure_ascii=True)
        f.write("\n")


def read_json(path: str | Path) -> object:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def read_yaml(path: str | Path) -> dict:
    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def make_json_safe(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    return obj


def normalize_id_series(s: pd.Series) -> pd.Series:
    def norm(x):
        if pd.isna(x):
            return ""
        if isinstance(x, float) and x.is_integer():
            return str(int(x))
        text = str(x).strip()
        if text.endswith(".0") and text[:-2].lstrip("-").isdigit():
            return text[:-2]
        return text

    return s.map(norm)


def first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in df.columns:
            return c
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def infer_pair_columns(
    df: pd.DataFrame,
    left_names: Sequence[str],
    right_names: Sequence[str],
) -> Tuple[str, str]:
    left = first_existing_column(df, left_names)
    right = first_existing_column(df, right_names)
    if left is None or right is None:
        if len(df.columns) < 2:
            raise ValueError(f"Need at least two columns, got {list(df.columns)}")
        left = left or df.columns[0]
        right = right or df.columns[1]
    return left, right


def canonical_pair_frame(
    path: str | Path,
    left_names: Sequence[str],
    right_names: Sequence[str],
    out_left: str,
    out_right: str,
) -> pd.DataFrame:
    df = pd.read_csv(path)
    left, right = infer_pair_columns(df, left_names, right_names)
    out = pd.DataFrame(
        {
            out_left: normalize_id_series(df[left]),
            out_right: normalize_id_series(df[right]),
        }
    )
    out = out[(out[out_left] != "") & (out[out_right] != "")]
    return out.drop_duplicates().sort_values([out_left, out_right]).reset_index(drop=True)


def write_nodes(ids: Iterable[str], path: str | Path) -> pd.DataFrame:
    df = pd.DataFrame({"id": sorted(set(map(str, ids)))})
    save_dataframe(df, path)
    return df


def read_node_ids(path: str | Path) -> List[str]:
    df = pd.read_csv(path)
    col = "id" if "id" in df.columns else df.columns[0]
    return normalize_id_series(df[col]).tolist()

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import pandas as pd
import torch

from .schema import (
    DISEASE,
    DRUG,
    EDGE_FILE_MAP,
    FEATURE_FILE_MAP,
    NODE_FILE_MAP,
    PROTEIN,
    REVERSE_EDGE_TYPES,
    EdgeType,
)


def read_id_map(path: Path) -> Tuple[list[str], Dict[str, int]]:
    df = pd.read_csv(path)
    col = "id" if "id" in df.columns else df.columns[0]
    ids = df[col].astype(str).str.strip().tolist()
    return ids, {value: i for i, value in enumerate(ids)}


def read_features(path: Path) -> Optional[torch.Tensor]:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.shape[1] <= 1:
        return None
    arr = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype="float32")
    return torch.tensor(arr, dtype=torch.float32)


def edge_frame_to_index(
    df: pd.DataFrame,
    src_col: str,
    dst_col: str,
    src_map: Mapping[str, int],
    dst_map: Mapping[str, int],
) -> torch.Tensor:
    rows = []
    for _, row in df.iterrows():
        src = str(row[src_col]).strip()
        dst = str(row[dst_col]).strip()
        if src in src_map and dst in dst_map:
            rows.append((src_map[src], dst_map[dst]))
    if not rows:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(rows, dtype=torch.long).t().contiguous()


def _column_for_node_type(node_type: str) -> str:
    if node_type == DRUG:
        return "drug_id"
    if node_type == DISEASE:
        return "disease_id"
    if node_type == PROTEIN:
        return "protein_id"
    raise ValueError(f"Unknown node type: {node_type}")


def load_label_frame(data_dir: str | Path, filename: str, id_maps: Mapping[str, Dict[str, int]]) -> pd.DataFrame:
    path = Path(data_dir) / filename
    df = pd.read_csv(path)
    required = {"drug_id", "disease_id", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must contain columns {required}")
    df = df.copy()
    df["drug_idx"] = df["drug_id"].astype(str).map(id_maps[DRUG])
    df["disease_idx"] = df["disease_id"].astype(str).map(id_maps[DISEASE])
    df = df.dropna(subset=["drug_idx", "disease_idx"]).copy()
    df["drug_idx"] = df["drug_idx"].astype(int)
    df["disease_idx"] = df["disease_idx"].astype(int)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
    return df


def load_hsgd_data(data_dir: str | Path, use_node_features: bool = True, bidirectional: bool = True):
    try:
        from torch_geometric.data import HeteroData
    except ImportError as exc:
        raise ImportError("torch-geometric is required for HSGD-DR data loading.") from exc

    data_dir = Path(data_dir)
    data = HeteroData()
    node_ids: Dict[str, list[str]] = {}
    id_maps: Dict[str, Dict[str, int]] = {}
    for node_type, filename in NODE_FILE_MAP.items():
        ids, mapping = read_id_map(data_dir / filename)
        node_ids[node_type] = ids
        id_maps[node_type] = mapping
        data[node_type].num_nodes = len(ids)
        if use_node_features:
            x = read_features(data_dir / FEATURE_FILE_MAP[node_type])
            if x is not None:
                data[node_type].x = x

    for edge_type, filename in EDGE_FILE_MAP.items():
        path = data_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        src_type, _, dst_type = edge_type
        src_col = _column_for_node_type(src_type)
        dst_col = _column_for_node_type(dst_type)
        edge_index = edge_frame_to_index(df, src_col, dst_col, id_maps[src_type], id_maps[dst_type])
        data[edge_type].edge_index = edge_index
        if bidirectional:
            reverse_type = REVERSE_EDGE_TYPES[edge_type]
            rev_index = torch.stack([edge_index[1], edge_index[0]], dim=0) if edge_index.numel() else torch.empty((2, 0), dtype=torch.long)
            data[reverse_type].edge_index = rev_index
    data.node_ids = node_ids
    data.id_maps = id_maps
    return data, node_ids, id_maps


def get_num_nodes_dict(data) -> Dict[str, int]:
    return {node_type: int(data[node_type].num_nodes) for node_type in data.node_types}


def get_x_dict(data) -> Dict[str, Optional[torch.Tensor]]:
    return {node_type: getattr(data[node_type], "x", None) for node_type in data.node_types}


def get_edge_index_dict(data) -> Dict[EdgeType, torch.Tensor]:
    return {edge_type: data[edge_type].edge_index for edge_type in data.edge_types}


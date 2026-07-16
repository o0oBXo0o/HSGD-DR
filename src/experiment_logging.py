from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import pandas as pd

from io_utils import write_json


def collect_environment() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
    }
    try:
        import torch

        out["torch"] = torch.__version__
        out["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:
        out["torch_error"] = str(exc)
    try:
        import torch_geometric

        out["torch_geometric"] = torch_geometric.__version__
    except Exception as exc:
        out["torch_geometric_error"] = str(exc)
    return out


def summarize_graph_dir(data_dir: Path) -> Dict[str, Any]:
    files = [
        "nodes_drug.csv",
        "nodes_disease.csv",
        "nodes_protein.csv",
        "edges_drug_disease_train.csv",
        "rels_drug_disease_all.csv",
        "edges_drug_protein.csv",
        "edges_disease_protein.csv",
        "labels_train.csv",
        "labels_val.csv",
        "labels_test.csv",
    ]
    summary: Dict[str, Any] = {}
    for fname in files:
        path = data_dir / fname
        if not path.exists():
            summary[fname] = {"exists": False}
            continue
        df = pd.read_csv(path)
        summary[fname] = {"exists": True, "rows": int(len(df)), "columns": list(df.columns)}
    return summary


def write_run_manifest(out_dir: Path, args: Mapping[str, Any], data_dir: Optional[Path] = None) -> None:
    write_json(
        {
            "args": dict(args),
            "data_dir": str(data_dir) if data_dir else None,
            "environment": collect_environment(),
            "data_summary": summarize_graph_dir(data_dir) if data_dir else None,
        },
        out_dir / "run_config.json",
    )

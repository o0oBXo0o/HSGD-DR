from __future__ import annotations

import argparse
from pathlib import Path
from typing import Set, Tuple

import pandas as pd
import torch

from .io_utils import ensure_dir
from .model_architecture import HSGDDRModel
from .pyg_dataset import get_edge_index_dict, get_x_dict, load_hsgd_data
from .training_evaluation import filter_relations

Pair = Tuple[str, str]


def known_pairs(data_dir: Path) -> Set[Pair]:
    path = data_dir / "rels_drug_disease_all.csv"
    df = pd.read_csv(path)
    return set(map(tuple, df[["drug_id", "disease_id"]].astype(str).values.tolist()))


def run(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    train_args = argparse.Namespace(**checkpoint.get("args", {}))
    data, node_ids, id_maps = load_hsgd_data(data_dir, use_node_features=True, bidirectional=True)
    edge_index_dict = filter_relations(get_edge_index_dict(data), train_args)
    edge_index_dict = {key: value.to(device) for key, value in edge_index_dict.items()}
    x_dict = {key: (value.to(device) if value is not None else None) for key, value in get_x_dict(data).items()}

    model = HSGDDRModel(
        edge_types=tuple(edge_index_dict.keys()),
        input_dims=checkpoint["input_dims"],
        num_nodes=checkpoint["num_nodes"],
        hidden_dim=train_args.hidden_dim,
        num_layers=train_args.num_layers,
        sage_aggr=train_args.sage_aggr,
        dropout=train_args.dropout,
        proj_dim=train_args.proj_dim,
        use_hybrid_gating=not train_args.no_hybrid_gating,
        gate_init=train_args.gate_init,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    exclude = known_pairs(data_dir) if args.exclude_known else set()
    rows = []
    drug_ids = node_ids["Drug"]
    disease_ids = node_ids["Disease"]
    with torch.no_grad():
        z_dict = model.encode(x_dict, edge_index_dict)
        for drug_idx, drug_id in enumerate(drug_ids):
            d_idx = torch.full((len(disease_ids),), drug_idx, dtype=torch.long, device=device)
            s_idx = torch.arange(len(disease_ids), dtype=torch.long, device=device)
            scores = torch.sigmoid(model.score_pairs(z_dict, d_idx, s_idx)).detach().cpu().numpy()
            for disease_idx, disease_id in enumerate(disease_ids):
                if (drug_id, disease_id) in exclude:
                    continue
                rows.append({"drug_id": drug_id, "disease_id": disease_id, "score": float(scores[disease_idx])})
    out = pd.DataFrame(rows).sort_values("score", ascending=False).head(args.top_k)
    ensure_dir(Path(args.out).parent)
    out.to_csv(args.out, index=False)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Predict top-ranked unobserved Drug-Disease pairs with HSGD-DR.")
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--exclude-known", action="store_true", default=True)
    p.add_argument("--device", default=None)
    return p


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()


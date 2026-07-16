from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .contrastive_learning import relation_specific_info_nce
from .experiment_logging import summarize_graph_dir, write_run_manifest
from .io_utils import ensure_dir, set_seed, write_json
from .leakage_control import check_leakage
from .metrics import compute_metrics
from .model_architecture import HSGDDRModel, get_model_input_dims
from .negative_sampling import sample_bpr_negative_diseases
from .pyg_dataset import get_edge_index_dict, get_num_nodes_dict, get_x_dict, load_hsgd_data, load_label_frame
from .schema import (
    DISEASE,
    DISEASE_PROTEIN_EDGE,
    DRUG,
    DRUG_DISEASE_EDGE,
    DRUG_PROTEIN_EDGE,
    REVERSE_EDGE_TYPES,
    EdgeType,
)

PairIdx = Tuple[int, int]


def _to_device_x_dict(x_dict, device):
    return {key: (value.to(device) if value is not None else None) for key, value in x_dict.items()}


def _to_device_edge_dict(edge_dict, device):
    return {key: value.to(device) for key, value in edge_dict.items()}


def _drop_relation(edge_dict: Mapping[EdgeType, torch.Tensor], edge_type: EdgeType) -> Dict[EdgeType, torch.Tensor]:
    reverse = REVERSE_EDGE_TYPES[edge_type]
    return {key: value for key, value in edge_dict.items() if key not in {edge_type, reverse}}


def filter_relations(edge_dict: Mapping[EdgeType, torch.Tensor], args: argparse.Namespace) -> Dict[EdgeType, torch.Tensor]:
    out = dict(edge_dict)
    if args.drop_drug_disease_relation:
        out = _drop_relation(out, DRUG_DISEASE_EDGE)
    if args.drop_drug_protein_relation:
        out = _drop_relation(out, DRUG_PROTEIN_EDGE)
    if args.drop_disease_protein_relation:
        out = _drop_relation(out, DISEASE_PROTEIN_EDGE)
    return out


def known_positive_index_pairs(data_dir: Path, id_maps: Mapping[str, Dict[str, int]]) -> Set[PairIdx]:
    path = data_dir / "rels_drug_disease_all.csv"
    df = pd.read_csv(path)
    out: Set[PairIdx] = set()
    for _, row in df.iterrows():
        drug_id = str(row["drug_id"])
        disease_id = str(row["disease_id"])
        if drug_id in id_maps[DRUG] and disease_id in id_maps[DISEASE]:
            out.add((id_maps[DRUG][drug_id], id_maps[DISEASE][disease_id]))
    return out


def training_positive_indices(data_dir: Path, id_maps: Mapping[str, Dict[str, int]], device: torch.device):
    labels_path = data_dir / "labels_train.csv"
    if labels_path.exists():
        df = pd.read_csv(labels_path)
        df = df[pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int) == 1]
    else:
        df = pd.read_csv(data_dir / "edges_drug_disease_train.csv")
    pairs = []
    for _, row in df.iterrows():
        drug_id = str(row["drug_id"])
        disease_id = str(row["disease_id"])
        if drug_id in id_maps[DRUG] and disease_id in id_maps[DISEASE]:
            pairs.append((id_maps[DRUG][drug_id], id_maps[DISEASE][disease_id]))
    if not pairs:
        raise ValueError("No usable positive Drug-Disease training pairs found.")
    arr = np.asarray(pairs, dtype=np.int64)
    return (
        torch.tensor(arr[:, 0], dtype=torch.long, device=device),
        torch.tensor(arr[:, 1], dtype=torch.long, device=device),
    )


def evaluate_split(
    model: HSGDDRModel,
    x_dict,
    edge_index_dict,
    data_dir: Path,
    filename: str,
    id_maps,
    threshold: float,
    device: torch.device,
):
    path = data_dir / filename
    if not path.exists():
        return None, None
    df = load_label_frame(data_dir, filename, id_maps)
    if df.empty:
        return {}, df
    model.eval()
    drug_idx = torch.tensor(df["drug_idx"].values, dtype=torch.long, device=device)
    disease_idx = torch.tensor(df["disease_idx"].values, dtype=torch.long, device=device)
    with torch.no_grad():
        z_dict = model.encode(x_dict, edge_index_dict)
        logits = model.score_pairs(z_dict, drug_idx, disease_idx)
        score = torch.sigmoid(logits).detach().cpu().numpy()
    pred = df[["drug_id", "disease_id", "label"]].copy()
    pred["score"] = score
    return compute_metrics(pred, threshold=threshold), pred


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    out_dir = ensure_dir(args.out_dir)
    data_dir = Path(args.data_dir)
    write_run_manifest(out_dir, vars(args), data_dir=data_dir)

    leakage = check_leakage(data_dir)
    write_json(leakage, out_dir / "leakage_report.json")
    if not leakage["ok"] and not args.skip_leakage_guard:
        raise SystemExit(f"Leakage guard failed. See {out_dir / 'leakage_report.json'}")

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    data, node_ids, id_maps = load_hsgd_data(data_dir, use_node_features=True, bidirectional=True)
    input_dims = get_model_input_dims(data, use_node_features=True)
    num_nodes = get_num_nodes_dict(data)
    full_edge_dict = get_edge_index_dict(data)
    edge_index_dict = _to_device_edge_dict(filter_relations(full_edge_dict, args), device)
    x_dict = _to_device_x_dict(get_x_dict(data), device)

    model = HSGDDRModel(
        edge_types=tuple(edge_index_dict.keys()),
        input_dims=input_dims,
        num_nodes=num_nodes,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        sage_aggr=args.sage_aggr,
        dropout=args.dropout,
        proj_dim=args.proj_dim,
        use_hybrid_gating=not args.no_hybrid_gating,
        gate_init=args.gate_init,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)

    pos_drug, pos_disease = training_positive_indices(data_dir, id_maps, device)
    known_pos = known_positive_index_pairs(data_dir, id_maps)
    best_metric = -float("inf")
    best_path = out_dir / "best.pt"
    history_path = out_dir / "history.jsonl"

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training"):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if args.bpr_negatives_per_positive > 1:
            train_drug = pos_drug.repeat_interleave(args.bpr_negatives_per_positive)
            train_pos_disease = pos_disease.repeat_interleave(args.bpr_negatives_per_positive)
        else:
            train_drug = pos_drug
            train_pos_disease = pos_disease
        neg_disease = sample_bpr_negative_diseases(
            train_drug,
            train_pos_disease,
            n_diseases=num_nodes[DISEASE],
            forbidden=known_pos,
            rng=rng,
        )
        z_dict = model.encode(x_dict, edge_index_dict)
        pos_logits = model.score_pairs(z_dict, train_drug, train_pos_disease)
        neg_logits = model.score_pairs(z_dict, train_drug, neg_disease)
        bpr_loss = -F.logsigmoid(pos_logits - neg_logits).mean()

        cl_loss, cl_details = relation_specific_info_nce(
            model,
            x_dict,
            edge_index_dict,
            temperature=args.cl_temperature,
            max_nodes_per_type=args.cl_max_nodes,
        ) if args.lambda_cl > 0 else (torch.tensor(0.0, device=device), {})
        loss = bpr_loss + args.lambda_cl * cl_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        rec = {
            "epoch": epoch,
            "loss": float(loss.detach().cpu()),
            "bpr_loss": float(bpr_loss.detach().cpu()),
            "contrastive_loss": float(cl_loss.detach().cpu()),
            "lambda_cl": float(args.lambda_cl),
            "use_hybrid_gating": not args.no_hybrid_gating,
            **cl_details,
        }
        if epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs:
            val_metrics, _ = evaluate_split(model, x_dict, edge_index_dict, data_dir, "labels_val.csv", id_maps, args.eval_threshold, device)
            if val_metrics:
                rec.update({f"val_{key}": value for key, value in val_metrics.items()})
                monitored = val_metrics.get(args.monitor, -float("inf"))
                if monitored is not None and not math.isnan(monitored) and monitored > best_metric:
                    best_metric = float(monitored)
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "input_dims": input_dims,
                            "num_nodes": num_nodes,
                            "edge_types": tuple(edge_index_dict.keys()),
                            "node_ids": node_ids,
                            "args": vars(args),
                            "best_metric": best_metric,
                        },
                        best_path,
                    )
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "input_dims": input_dims,
                "num_nodes": num_nodes,
                "edge_types": tuple(edge_index_dict.keys()),
                "node_ids": node_ids,
                "args": vars(args),
                "best_metric": best_metric,
            },
            best_path,
        )

    pred_dir = ensure_dir(out_dir / "predictions")
    metrics_all = {}
    for filename, split in [("labels_train.csv", "train"), ("labels_val.csv", "val"), ("labels_test.csv", "test")]:
        metrics, pred = evaluate_split(model, x_dict, edge_index_dict, data_dir, filename, id_maps, args.eval_threshold, device)
        if metrics is not None:
            metrics_all[split] = metrics
            if pred is not None:
                pred.to_csv(pred_dir / f"predictions_{split}.csv", index=False)

    write_json(
        {
            "metrics": metrics_all,
            "best_metric": best_metric,
            "monitor": args.monitor,
            "model": model.diagnostics(),
            "run_config": vars(args),
            "data_summary": summarize_graph_dir(data_dir),
            "relations_used": [list(edge_type) for edge_type in edge_index_dict.keys()],
        },
        out_dir / "metrics.json",
    )
    write_json(
        {
            "best_metric": best_metric,
            "monitor": args.monitor,
            "metrics_file": str(out_dir / "metrics.json"),
            "checkpoint": str(best_path),
            "model": model.diagnostics(),
        },
        out_dir / "training_summary.json",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train and evaluate HSGD-DR.")
    p.add_argument("--data-dir", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--sage-aggr", default="mean", choices=["mean", "max", "lstm"])
    p.add_argument("--bpr-negatives-per-positive", type=int, default=1)
    p.add_argument("--lambda-cl", type=float, default=0.1)
    p.add_argument("--cl-temperature", type=float, default=0.2)
    p.add_argument("--cl-max-nodes", type=int, default=4096)
    p.add_argument("--gate-init", type=float, default=-2.0)
    p.add_argument("--no-hybrid-gating", action="store_true")
    p.add_argument("--drop-drug-disease-relation", action="store_true")
    p.add_argument("--drop-drug-protein-relation", action="store_true")
    p.add_argument("--drop-disease-protein-relation", action="store_true")
    p.add_argument("--eval-threshold", type=float, default=0.5)
    p.add_argument("--monitor", default="aupr", choices=["aupr", "auroc", "precision", "recall", "f1_score", "mcc"])
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--skip-leakage-guard", action="store_true")
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    train(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()


from __future__ import annotations

from typing import Dict, Iterable, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

from .schema import DISEASE, DRUG, RELATION_CHANNELS, EdgeType


def info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2, symmetric: bool = True) -> torch.Tensor:
    if z1.numel() == 0 or z2.numel() == 0:
        return z1.new_tensor(0.0)
    n = min(z1.shape[0], z2.shape[0])
    z1 = F.normalize(z1[:n], p=2, dim=-1)
    z2 = F.normalize(z2[:n], p=2, dim=-1)
    logits = z1 @ z2.t() / float(temperature)
    labels = torch.arange(n, device=z1.device)
    loss = F.cross_entropy(logits, labels)
    if symmetric:
        loss = 0.5 * (loss + F.cross_entropy(logits.t(), labels))
    return loss


def relation_view(edge_index_dict: Mapping[EdgeType, torch.Tensor], allowed_edge_types: Iterable[EdgeType]) -> Dict[EdgeType, torch.Tensor]:
    allowed = set(allowed_edge_types)
    return {edge_type: edge_index for edge_type, edge_index in edge_index_dict.items() if edge_type in allowed}


def _has_any_edges(edge_index_dict: Mapping[EdgeType, torch.Tensor], edge_types: Sequence[EdgeType]) -> bool:
    return any(edge_type in edge_index_dict and edge_index_dict[edge_type].numel() > 0 for edge_type in edge_types)


def relation_specific_info_nce(
    model,
    x_dict,
    edge_index_dict: Mapping[EdgeType, torch.Tensor],
    temperature: float = 0.2,
    max_nodes_per_type: int = 4096,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """InfoNCE between relation channels as described in the manuscript.

    Drug positives: drug-disease channel vs. drug-protein channel.
    Disease positives: disease-drug channel vs. disease-protein channel.
    """
    details: Dict[str, float] = {}
    device = next(model.parameters()).device
    losses = []

    required = ["drug_disease", "drug_protein", "disease_protein"]
    available = {name: _has_any_edges(edge_index_dict, RELATION_CHANNELS[name]) for name in required}
    details.update({f"channel_available_{name}": float(value) for name, value in available.items()})
    if not available["drug_disease"]:
        return torch.tensor(0.0, device=device), details

    z_drug_disease = model.encode(x_dict, relation_view(edge_index_dict, RELATION_CHANNELS["drug_disease"]))

    if available["drug_protein"]:
        z_drug_protein = model.encode(x_dict, relation_view(edge_index_dict, RELATION_CHANNELS["drug_protein"]))
        z1 = z_drug_disease[DRUG]
        z2 = z_drug_protein[DRUG]
        if z1.shape[0] > max_nodes_per_type:
            idx = torch.randperm(z1.shape[0], device=device)[:max_nodes_per_type]
            z1, z2 = z1[idx], z2[idx]
        loss_drug = info_nce_loss(model.projectors[DRUG](z1), model.projectors[DRUG](z2), temperature=temperature)
        losses.append(loss_drug)
        details["info_nce_drug"] = float(loss_drug.detach().cpu())

    if available["disease_protein"]:
        z_disease_protein = model.encode(x_dict, relation_view(edge_index_dict, RELATION_CHANNELS["disease_protein"]))
        z1 = z_drug_disease[DISEASE]
        z2 = z_disease_protein[DISEASE]
        if z1.shape[0] > max_nodes_per_type:
            idx = torch.randperm(z1.shape[0], device=device)[:max_nodes_per_type]
            z1, z2 = z1[idx], z2[idx]
        loss_disease = info_nce_loss(model.projectors[DISEASE](z1), model.projectors[DISEASE](z2), temperature=temperature)
        losses.append(loss_disease)
        details["info_nce_disease"] = float(loss_disease.detach().cpu())

    if not losses:
        return torch.tensor(0.0, device=device), details
    total = torch.stack(losses).mean()
    details["relation_specific_info_nce"] = float(total.detach().cpu())
    return total, details


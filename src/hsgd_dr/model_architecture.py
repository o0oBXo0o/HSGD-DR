from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schema import DISEASE, DRUG, NODE_TYPES, PROTEIN, EdgeType, edge_type_to_key, key_to_edge_type


class StructuralNodeEmbeddings(nn.Module):
    """Learnable node-type-specific input embeddings for the structural branch."""

    def __init__(self, num_nodes: Mapping[str, int], hidden_dim: int):
        super().__init__()
        self.num_nodes = {node_type: int(num_nodes[node_type]) for node_type in NODE_TYPES}
        self.embeddings = nn.ModuleDict()
        for node_type in NODE_TYPES:
            table = nn.Embedding(self.num_nodes[node_type], hidden_dim)
            nn.init.xavier_uniform_(table.weight)
            self.embeddings[node_type] = table

    def forward(self) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        out = {}
        for node_type in NODE_TYPES:
            ids = torch.arange(self.num_nodes[node_type], device=device)
            out[node_type] = self.embeddings[node_type](ids)
        return out


class HeterogeneousGraphSAGELayer(nn.Module):
    """Relation-specific GraphSAGE channels with summed messages."""

    def __init__(self, edge_types: Iterable[EdgeType], hidden_dim: int, aggr: str = "mean", dropout: float = 0.0):
        super().__init__()
        try:
            from torch_geometric.nn import SAGEConv
        except ImportError as exc:
            raise ImportError("torch-geometric is required for HSGD-DR GraphSAGE layers.") from exc
        self.edge_types = tuple(edge_types)
        self.dropout = float(dropout)
        self.convs = nn.ModuleDict()
        for edge_type in self.edge_types:
            self.convs[edge_type_to_key(edge_type)] = SAGEConv((hidden_dim, hidden_dim), hidden_dim, aggr=aggr)
        self.norms = nn.ModuleDict({node_type: nn.LayerNorm(hidden_dim) for node_type in NODE_TYPES})

    def forward(self, h_dict: Mapping[str, torch.Tensor], edge_index_dict: Mapping[EdgeType, torch.Tensor]) -> Dict[str, torch.Tensor]:
        by_dst: Dict[str, list[torch.Tensor]] = {node_type: [] for node_type in NODE_TYPES}
        for key, conv in self.convs.items():
            edge_type = key_to_edge_type(key)
            if edge_type not in edge_index_dict:
                continue
            edge_index = edge_index_dict[edge_type]
            if edge_index.numel() == 0:
                continue
            src, _, dst = edge_type
            msg = conv((h_dict[src], h_dict[dst]), edge_index)
            by_dst[dst].append(msg)

        out: Dict[str, torch.Tensor] = {}
        for node_type in NODE_TYPES:
            if by_dst[node_type]:
                msg_sum = torch.stack(by_dst[node_type], dim=0).sum(dim=0)
                msg_sum = F.relu(msg_sum)
                msg_sum = F.dropout(msg_sum, p=self.dropout, training=self.training)
                out[node_type] = self.norms[node_type](h_dict[node_type] + msg_sum)
            else:
                out[node_type] = h_dict[node_type]
        return out


class HeterogeneousGraphSAGEEncoder(nn.Module):
    def __init__(
        self,
        edge_types: Iterable[EdgeType],
        num_nodes: Mapping[str, int],
        hidden_dim: int = 128,
        num_layers: int = 2,
        sage_aggr: str = "mean",
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_embeddings = StructuralNodeEmbeddings(num_nodes, hidden_dim)
        self.layers = nn.ModuleList(
            [
                HeterogeneousGraphSAGELayer(edge_types, hidden_dim, aggr=sage_aggr, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, edge_index_dict: Mapping[EdgeType, torch.Tensor]) -> Dict[str, torch.Tensor]:
        h = self.input_embeddings()
        for layer in self.layers:
            h = layer(h, edge_index_dict)
        return h


class NodeFeatureProjector(nn.Module):
    """Node-type-specific feature projection branch."""

    def __init__(self, input_dims: Mapping[str, int], hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.input_dims = {node_type: int(input_dims.get(node_type, 0) or 0) for node_type in NODE_TYPES}
        self.projectors = nn.ModuleDict()
        for node_type in NODE_TYPES:
            dim = self.input_dims[node_type]
            if dim > 0:
                self.projectors[node_type] = nn.Sequential(
                    nn.Linear(dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                )

    def forward(self, x_dict: Mapping[str, Optional[torch.Tensor]]) -> Dict[str, Optional[torch.Tensor]]:
        out: Dict[str, Optional[torch.Tensor]] = {}
        device = next(self.parameters()).device if len(self.projectors) else None
        for node_type in NODE_TYPES:
            x = x_dict.get(node_type)
            if node_type in self.projectors and x is not None:
                out[node_type] = self.projectors[node_type](x.to(device))
            else:
                out[node_type] = None
        return out


class DimensionWiseHybridGate(nn.Module):
    """Fuse Heterogeneous GraphSAGE structural embeddings with node features.

    The gate follows the manuscript convention:
        e_v = g_v * h_v_structural + (1 - g_v) * f_v
    """

    def __init__(self, hidden_dim: int, gate_init: float = -2.0):
        super().__init__()
        self.gates = nn.ModuleDict({node_type: nn.Linear(2 * hidden_dim, hidden_dim) for node_type in NODE_TYPES})
        for layer in self.gates.values():
            nn.init.zeros_(layer.weight)
            nn.init.constant_(layer.bias, float(gate_init))
        self.last_gate_means: Dict[str, float] = {}

    def forward(
        self,
        structural: Mapping[str, torch.Tensor],
        projected_features: Mapping[str, Optional[torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        self.last_gate_means = {}
        for node_type in NODE_TYPES:
            h_struct = structural[node_type]
            h_feat = projected_features.get(node_type)
            if h_feat is None:
                out[node_type] = h_struct
                self.last_gate_means[node_type] = float("nan")
                continue
            gate = torch.sigmoid(self.gates[node_type](torch.cat([h_struct, h_feat], dim=-1)))
            out[node_type] = gate * h_struct + (1.0 - gate) * h_feat
            self.last_gate_means[node_type] = float(gate.detach().mean().cpu())
        return out


class DistMultDecoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.relation = nn.Parameter(torch.empty(hidden_dim))
        nn.init.xavier_uniform_(self.relation.view(1, -1))

    def forward(self, z_drug: torch.Tensor, z_disease: torch.Tensor, drug_idx: torch.Tensor, disease_idx: torch.Tensor) -> torch.Tensor:
        d = z_drug[drug_idx]
        s = z_disease[disease_idx]
        return (d * self.relation * s).sum(dim=-1)


class ProjectionHead(nn.Module):
    def __init__(self, hidden_dim: int, proj_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HSGDDRModel(nn.Module):
    """HSGD-DR model: Heterogeneous GraphSAGE, hybrid gate, DistMult."""

    def __init__(
        self,
        edge_types: Iterable[EdgeType],
        input_dims: Mapping[str, int],
        num_nodes: Mapping[str, int],
        hidden_dim: int = 128,
        num_layers: int = 2,
        sage_aggr: str = "mean",
        dropout: float = 0.2,
        proj_dim: int = 128,
        use_hybrid_gating: bool = True,
        gate_init: float = -2.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.edge_types = tuple(edge_types)
        self.use_hybrid_gating = bool(use_hybrid_gating)
        self.structural_encoder = HeterogeneousGraphSAGEEncoder(
            edge_types=self.edge_types,
            num_nodes=num_nodes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            sage_aggr=sage_aggr,
            dropout=dropout,
        )
        self.feature_projector = NodeFeatureProjector(input_dims=input_dims, hidden_dim=hidden_dim, dropout=dropout)
        self.hybrid_gate = DimensionWiseHybridGate(hidden_dim=hidden_dim, gate_init=gate_init)
        self.decoder = DistMultDecoder(hidden_dim)
        self.projectors = nn.ModuleDict({node_type: ProjectionHead(hidden_dim, proj_dim, dropout=dropout) for node_type in NODE_TYPES})

    def encode(
        self,
        x_dict: Mapping[str, Optional[torch.Tensor]],
        edge_index_dict: Mapping[EdgeType, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        structural = self.structural_encoder(edge_index_dict)
        if not self.use_hybrid_gating:
            return structural
        projected = self.feature_projector(x_dict)
        return self.hybrid_gate(structural, projected)

    def score_pairs(self, z_dict: Mapping[str, torch.Tensor], drug_idx: torch.Tensor, disease_idx: torch.Tensor) -> torch.Tensor:
        return self.decoder(z_dict[DRUG], z_dict[DISEASE], drug_idx, disease_idx)

    def forward(self, x_dict, edge_index_dict, drug_idx, disease_idx):
        z = self.encode(x_dict, edge_index_dict)
        return self.score_pairs(z, drug_idx, disease_idx)

    def diagnostics(self) -> Dict[str, object]:
        return {
            "use_hybrid_gating": self.use_hybrid_gating,
            "hidden_dim": self.hidden_dim,
            "edge_types": [list(edge_type) for edge_type in self.edge_types],
            "gate_mean_by_node_type": dict(self.hybrid_gate.last_gate_means),
        }


def get_model_input_dims(data, use_node_features: bool = True) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for node_type in NODE_TYPES:
        x = getattr(data[node_type], "x", None)
        out[node_type] = int(x.shape[1]) if use_node_features and x is not None else 0
    return out


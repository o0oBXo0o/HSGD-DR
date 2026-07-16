from __future__ import annotations

from typing import Mapping, Tuple

NodeType = str
EdgeType = Tuple[str, str, str]

DRUG: NodeType = "Drug"
DISEASE: NodeType = "Disease"
PROTEIN: NodeType = "Protein"
NODE_TYPES: Tuple[NodeType, ...] = (DRUG, DISEASE, PROTEIN)

DRUG_DISEASE_EDGE: EdgeType = (DRUG, "drug_disease", DISEASE)
DRUG_PROTEIN_EDGE: EdgeType = (DRUG, "drug_protein", PROTEIN)
DISEASE_PROTEIN_EDGE: EdgeType = (DISEASE, "disease_protein", PROTEIN)
FORWARD_EDGE_TYPES: Tuple[EdgeType, ...] = (
    DRUG_DISEASE_EDGE,
    DRUG_PROTEIN_EDGE,
    DISEASE_PROTEIN_EDGE,
)

REVERSE_EDGE_TYPES: Mapping[EdgeType, EdgeType] = {
    DRUG_DISEASE_EDGE: (DISEASE, "disease_drug", DRUG),
    DRUG_PROTEIN_EDGE: (PROTEIN, "protein_drug", DRUG),
    DISEASE_PROTEIN_EDGE: (PROTEIN, "protein_disease", DISEASE),
}

EDGE_FILE_MAP: Mapping[EdgeType, str] = {
    DRUG_DISEASE_EDGE: "edges_drug_disease_train.csv",
    DRUG_PROTEIN_EDGE: "edges_drug_protein.csv",
    DISEASE_PROTEIN_EDGE: "edges_disease_protein.csv",
}

NODE_FILE_MAP: Mapping[NodeType, str] = {
    DRUG: "nodes_drug.csv",
    DISEASE: "nodes_disease.csv",
    PROTEIN: "nodes_protein.csv",
}

FEATURE_FILE_MAP: Mapping[NodeType, str] = {
    DRUG: "features_drug.csv",
    DISEASE: "features_disease.csv",
    PROTEIN: "features_protein.csv",
}

LABEL_FILES: Tuple[str, ...] = (
    "labels_train.csv",
    "labels_val.csv",
    "labels_test.csv",
)

RELATION_CHANNELS: Mapping[str, Tuple[EdgeType, ...]] = {
    "drug_disease": (DRUG_DISEASE_EDGE, REVERSE_EDGE_TYPES[DRUG_DISEASE_EDGE]),
    "drug_protein": (DRUG_PROTEIN_EDGE, REVERSE_EDGE_TYPES[DRUG_PROTEIN_EDGE]),
    "disease_protein": (DISEASE_PROTEIN_EDGE, REVERSE_EDGE_TYPES[DISEASE_PROTEIN_EDGE]),
}


def edge_type_to_key(edge_type: EdgeType) -> str:
    return "__".join(edge_type)


def key_to_edge_type(key: str) -> EdgeType:
    parts = key.split("__")
    if len(parts) != 3:
        raise ValueError(f"Invalid edge type key: {key}")
    return parts[0], parts[1], parts[2]


def all_edge_types(bidirectional: bool = True) -> Tuple[EdgeType, ...]:
    if bidirectional:
        return tuple(FORWARD_EDGE_TYPES) + tuple(REVERSE_EDGE_TYPES.values())
    return tuple(FORWARD_EDGE_TYPES)


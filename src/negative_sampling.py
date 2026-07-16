from __future__ import annotations

import itertools
from typing import Sequence, Set, Tuple

import numpy as np
import pandas as pd

Pair = Tuple[str, str]
PairIdx = Tuple[int, int]


def sample_negative_pairs(
    drugs: Sequence[str],
    diseases: Sequence[str],
    forbidden: Set[Pair],
    n: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    drugs = list(map(str, drugs))
    diseases = list(map(str, diseases))
    max_possible = len(drugs) * len(diseases) - len(forbidden)
    n = min(max(int(n), 0), max_possible)
    out: Set[Pair] = set()
    attempts = 0
    while len(out) < n and attempts < max(10000, 25 * max(n, 1)):
        pair = (
            drugs[int(rng.integers(0, len(drugs)))],
            diseases[int(rng.integers(0, len(diseases)))],
        )
        if pair not in forbidden:
            out.add(pair)
        attempts += 1
    if len(out) < n:
        for pair in itertools.product(drugs, diseases):
            if pair not in forbidden:
                out.add(pair)
                if len(out) >= n:
                    break
    return pd.DataFrame(sorted(out), columns=["drug_id", "disease_id"])


def with_labels(pos: pd.DataFrame, neg: pd.DataFrame, source_name: str, split: str) -> pd.DataFrame:
    p = pos[["drug_id", "disease_id"]].copy()
    p["label"] = 1
    n = neg[["drug_id", "disease_id"]].copy()
    n["label"] = 0
    out = pd.concat([p, n], ignore_index=True)
    out["split"] = split
    out["source"] = source_name
    return out.sample(frac=1.0, random_state=13).reset_index(drop=True)


def sample_bpr_negative_diseases(
    positive_drug_idx,
    positive_disease_idx,
    n_diseases: int,
    forbidden: Set[PairIdx],
    rng: np.random.Generator,
):
    import torch

    device = positive_drug_idx.device
    neg = []
    pos_d = positive_drug_idx.detach().cpu().numpy().astype(int).tolist()
    for d_idx in pos_d:
        candidate = int(rng.integers(0, n_diseases))
        attempts = 0
        while (d_idx, candidate) in forbidden and attempts < max(1000, 4 * n_diseases):
            candidate = int(rng.integers(0, n_diseases))
            attempts += 1
        if (d_idx, candidate) in forbidden:
            for fallback in range(n_diseases):
                if (d_idx, fallback) not in forbidden:
                    candidate = fallback
                    break
        neg.append(candidate)
    return torch.tensor(neg, dtype=torch.long, device=device)


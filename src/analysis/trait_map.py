"""Behavioral Trait Map visualization."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from src.data.contrastive import BehavioralTrait
from src.model.config import HOOK_POINTS

logger = logging.getLogger(__name__)


def compute_trait_map_data(
    all_tas: dict[BehavioralTrait, dict[str, torch.Tensor]],
    top_k: int = 20,
) -> dict[str, np.ndarray]:
    """Compute data for the Behavioral Trait Map visualization.

    The trait map shows how traits are distributed across SAE positions.

    Args:
        all_tas: Nested dict: trait → sae_id → TAS tensor.
        top_k: Number of top features to consider per trait per SAE.

    Returns:
        Dict with:
            'matrix': (n_traits, n_saes) array of mean top-k |TAS| scores
            'trait_names': list of trait names
            'sae_names': list of SAE names
    """
    trait_names = [t.value for t in BehavioralTrait]
    sae_names = [hp.sae_id for hp in HOOK_POINTS]

    n_traits = len(trait_names)
    n_saes = len(sae_names)
    matrix = np.zeros((n_traits, n_saes))

    for i, trait in enumerate(BehavioralTrait):
        for j, sae_name in enumerate(sae_names):
            if trait in all_tas and sae_name in all_tas[trait]:
                tas = all_tas[trait][sae_name]
                k = min(top_k, tas.shape[0])
                top_vals = tas.abs().topk(k).values
                matrix[i, j] = float(top_vals.mean().item())

    return {
        "matrix": matrix,
        "trait_names": trait_names,
        "sae_names": sae_names,
    }

"""Cross-domain stability analysis for behavioral traits."""

from __future__ import annotations

import hashlib
import logging

import numpy as np
import torch

from src.data.contrastive import BehavioralTrait, TaskDomain
from src.features.extraction import FeatureExtractionResults

logger = logging.getLogger(__name__)


def compute_domain_stability(
    all_tas_by_domain: dict[TaskDomain, dict[BehavioralTrait, dict[str, torch.Tensor]]],
    top_k: int = 20,
) -> dict[str, np.ndarray | list[str]]:
    """Compute cross-domain stability of TAS scores.

    For each trait, compares the top-k features identified in each domain.
    High stability = the same features are important across domains.

    Args:
        all_tas_by_domain: domain → trait → sae_id → TAS tensor.
        top_k: Number of top features to compare.

    Returns:
        Dict with:
            'matrix': (n_domains, n_traits) array of stability scores
            'domain_names': list
            'trait_names': list
            'overlap_matrix': (n_domains, n_domains, n_traits) pairwise overlap
    """
    domain_names = [d.value for d in TaskDomain]
    trait_names = [t.value for t in BehavioralTrait]

    n_domains = len(domain_names)
    n_traits = len(trait_names)

    # Stability: average pairwise overlap of top-k features across domains
    stability_matrix = np.zeros((n_domains, n_traits))
    overlap_matrix = np.zeros((n_domains, n_domains, n_traits))

    for j, trait in enumerate(BehavioralTrait):
        # Collect top-k feature sets per domain
        domain_feature_sets: dict[TaskDomain, set[int]] = {}

        for domain in TaskDomain:
            if domain not in all_tas_by_domain:
                continue
            if trait not in all_tas_by_domain[domain]:
                continue

            # Collect (global_feature_id, abs_tas_score) across all SAEs,
            # then select the global top-k by TAS magnitude.  The previous
            # approach used position-based truncation (combined[:top_k])
            # which silently favoured whichever SAE iterated first.
            candidates: list[tuple[int, float]] = []
            for sae_id, tas in all_tas_by_domain[domain][trait].items():
                abs_tas = tas.abs()
                k = min(top_k, tas.shape[0])
                topk_vals, topk_idxs = abs_tas.topk(k)
                # Prefix with deterministic sae_id hash to make features unique across SAEs
                sae_hash = int(hashlib.sha256(sae_id.encode()).hexdigest()[:12], 16)
                for idx, val in zip(topk_idxs.tolist(), topk_vals.tolist()):
                    candidates.append((sae_hash * 100000 + idx, val))

            # Sort by TAS magnitude descending, take global top-k
            candidates.sort(key=lambda x: x[1], reverse=True)
            domain_feature_sets[domain] = {fid for fid, _ in candidates[:top_k]}

        # Compute pairwise Jaccard overlap
        for i, domain in enumerate(TaskDomain):
            if domain not in domain_feature_sets:
                continue
            overlaps = []
            for i2, other_domain in enumerate(TaskDomain):
                if other_domain == domain or other_domain not in domain_feature_sets:
                    continue
                set_a = domain_feature_sets[domain]
                set_b = domain_feature_sets[other_domain]
                if set_a or set_b:
                    jaccard = len(set_a & set_b) / max(len(set_a | set_b), 1)
                    overlaps.append(jaccard)
                    overlap_matrix[i, i2, j] = jaccard

            stability_matrix[i, j] = np.mean(overlaps) if overlaps else 0.0

    return {
        "matrix": stability_matrix,
        "domain_names": domain_names,
        "trait_names": trait_names,
        "overlap_matrix": overlap_matrix,
    }

"""Feature clustering per trait by decoder weight similarity."""

from __future__ import annotations

import logging

import numpy as np
import torch
from pydantic import BaseModel
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from src.sae.model import TopKSAE

logger = logging.getLogger(__name__)


class FeatureCluster(BaseModel):
    """A cluster of related features."""

    cluster_id: int
    feature_indices: list[int]
    feature_tas_scores: list[float]
    mean_tas: float
    centroid_cosine_similarity: float  # Avg similarity within cluster


def cluster_trait_features(
    tas_scores: torch.Tensor,
    sae: TopKSAE,
    top_k: int = 50,
    n_clusters: int = 5,
) -> list[FeatureCluster]:
    """Cluster the top-k features for a trait by decoder weight similarity.

    Uses cosine similarity between decoder weight vectors to find
    groups of features that represent similar directions in activation space.

    Args:
        tas_scores: TAS tensor of shape (dict_size,).
        sae: The trained SAE (for decoder weights).
        top_k: Number of top features to cluster.
        n_clusters: Number of clusters to create.

    Returns:
        List of FeatureCluster objects.
    """
    # Get top-k features by absolute TAS
    abs_tas = tas_scores.abs()
    k = min(top_k, abs_tas.shape[0])
    _, topk_indices = torch.topk(abs_tas, k)
    topk_indices = topk_indices.cpu().numpy()

    # Extract decoder weight vectors for these features
    # nn.Linear(dict_size, hidden_dim) stores weight as (hidden_dim, dict_size)
    # Transpose to get (dict_size, hidden_dim) — each row is a feature's direction
    decoder_weights = sae.decoder.weight.detach().cpu().numpy().T  # (dict_size, hidden_dim)
    feature_vectors = decoder_weights[topk_indices]  # (k, hidden_dim)

    # Compute cosine similarity matrix
    sim_matrix = cosine_similarity(feature_vectors)  # (k, k)

    # Convert similarity to distance for clustering
    distance_matrix = 1.0 - np.clip(sim_matrix, -1, 1)

    # Agglomerative clustering
    n_clusters = min(n_clusters, k)
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average",
    )
    labels = clustering.fit_predict(distance_matrix)

    # Build cluster objects
    clusters = []
    for cluster_id in range(n_clusters):
        member_mask = labels == cluster_id
        member_local_indices = np.where(member_mask)[0]
        member_global_indices = topk_indices[member_local_indices].tolist()
        member_tas = [float(tas_scores[idx].item()) for idx in member_global_indices]

        # Mean intra-cluster similarity
        if len(member_local_indices) > 1:
            sub_sim = sim_matrix[np.ix_(member_local_indices, member_local_indices)]
            mask = ~np.eye(len(member_local_indices), dtype=bool)
            mean_sim = float(sub_sim[mask].mean())
        else:
            mean_sim = 1.0

        clusters.append(
            FeatureCluster(
                cluster_id=cluster_id,
                feature_indices=member_global_indices,
                feature_tas_scores=member_tas,
                mean_tas=float(np.mean(np.abs(member_tas))),
                centroid_cosine_similarity=mean_sim,
            )
        )

    # Sort by mean TAS
    clusters.sort(key=lambda c: c.mean_tas, reverse=True)

    logger.info(
        "Clustered %d features into %d clusters (sizes: %s)",
        k,
        n_clusters,
        [len(c.feature_indices) for c in clusters],
    )
    return clusters

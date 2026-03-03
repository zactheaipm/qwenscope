"""DeltaNet vs attention results analysis.

IMPORTANT: Hooks capture the residual stream after the full layer (attention/DeltaNet
+ FFN + skip connection). Findings should be framed as "layers containing DeltaNet
have different features than layers containing attention" rather than "DeltaNet
represents differently than attention," since we cannot isolate the sublayer
contribution from the FFN and skip connection.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

from src.data.contrastive import BehavioralTrait
from src.features.architecture_analysis import ArchitectureComparisonResult
from src.features.scoring import rank_features
from src.model.config import HOOK_POINTS, LayerType
from src.sae.model import TopKSAE

logger = logging.getLogger(__name__)


def format_architecture_comparison(
    comparison: ArchitectureComparisonResult,
) -> dict[str, list]:
    """Format architecture comparison results for visualization.

    Args:
        comparison: The comparison result from feature analysis.

    Returns:
        Dict with formatted data for plotting.
    """
    traits = []
    delta_scores = []
    attn_scores = []
    mwu_pvalues = []
    stronger = []

    for trait_name, comp in comparison.per_trait.items():
        traits.append(trait_name)
        delta_scores.append(comp.deltanet_mean_top20_tas)
        attn_scores.append(comp.attention_mean_top20_tas)
        mwu_pvalues.append(comp.mwu_pvalue)
        stronger.append(comp.stronger_layer_type)

    return {
        "traits": traits,
        "deltanet_scores": delta_scores,
        "attention_scores": attn_scores,
        "mwu_pvalues": mwu_pvalues,
        "stronger_layer_type": stronger,
    }


def compare_feature_geometry(
    sae_dict: dict[str, TopKSAE],
    all_tas: dict[BehavioralTrait, dict[str, torch.Tensor]],
    top_k: int = 20,
) -> dict[BehavioralTrait, dict[str, float]]:
    """Compare decoder weight geometry of same-trait features across layer types.

    For each trait, extracts top-k features from DeltaNet and attention SAEs,
    then computes cosine similarity between their decoder weight vectors in the
    shared 5120-dim residual stream space.

    If same-trait features in DeltaNet and attention SAEs point in similar
    directions, this suggests shared representation across layer types.

    NOTE: Since hooks capture residual stream after full layer (sublayer + FFN +
    skip), high similarity means the residual stream encodes similar directions
    at both layer types, not that the sublayers compute the same thing.

    Args:
        sae_dict: Dict mapping sae_id to trained SAE.
        all_tas: Nested dict: trait → sae_id → TAS tensor.
        top_k: Number of top features per SAE to compare.

    Returns:
        Dict mapping trait to {
            "cross_type_similarity": float,  # mean cosine sim between top DeltaNet/attention features
            "within_deltanet_similarity": float,
            "within_attention_similarity": float,
            "n_deltanet_features": int,
            "n_attention_features": int,
        }.
    """
    deltanet_sae_ids = {hp.sae_id for hp in HOOK_POINTS if hp.layer_type == LayerType.DELTANET}
    attention_sae_ids = {hp.sae_id for hp in HOOK_POINTS if hp.layer_type == LayerType.ATTENTION}

    results: dict[BehavioralTrait, dict[str, float]] = {}

    for trait, sae_tas in all_tas.items():
        # Collect decoder weight vectors for top features by layer type
        delta_vectors = []
        attn_vectors = []

        for sae_id, tas in sae_tas.items():
            if sae_id not in sae_dict:
                continue
            sae = sae_dict[sae_id]
            top_features = rank_features(tas, top_k, positive_only=False)
            feature_indices = [idx for idx, _ in top_features]

            # nn.Linear(dict_size, hidden_dim) → weight is (hidden_dim, dict_size)
            decoder_w = sae.decoder.weight.detach().cpu().numpy().T  # (dict_size, hidden_dim)
            vectors = decoder_w[feature_indices]  # (k, hidden_dim)

            if sae_id in deltanet_sae_ids:
                delta_vectors.append(vectors)
            elif sae_id in attention_sae_ids:
                attn_vectors.append(vectors)

        if not delta_vectors or not attn_vectors:
            continue

        delta_all = np.concatenate(delta_vectors, axis=0)  # (n_delta, hidden_dim)
        attn_all = np.concatenate(attn_vectors, axis=0)    # (n_attn, hidden_dim)

        # Cross-type similarity: cosine sim between every DeltaNet and attention feature
        cross_sim = cosine_similarity(delta_all, attn_all)  # (n_delta, n_attn)
        cross_mean = float(np.abs(cross_sim).mean())

        # Within-type similarities
        if delta_all.shape[0] > 1:
            delta_sim = cosine_similarity(delta_all)
            mask = ~np.eye(delta_all.shape[0], dtype=bool)
            within_delta = float(np.abs(delta_sim[mask]).mean())
        else:
            within_delta = 1.0

        if attn_all.shape[0] > 1:
            attn_sim = cosine_similarity(attn_all)
            mask = ~np.eye(attn_all.shape[0], dtype=bool)
            within_attn = float(np.abs(attn_sim[mask]).mean())
        else:
            within_attn = 1.0

        results[trait] = {
            "cross_type_similarity": cross_mean,
            "within_deltanet_similarity": within_delta,
            "within_attention_similarity": within_attn,
            "n_deltanet_features": int(delta_all.shape[0]),
            "n_attention_features": int(attn_all.shape[0]),
        }

        logger.info(
            "Feature geometry for %s: cross=%.3f, within_delta=%.3f, within_attn=%.3f",
            trait.value, cross_mean, within_delta, within_attn,
        )

    return results


def analyze_block_structure(
    all_tas: dict[BehavioralTrait, dict[str, torch.Tensor]],
    top_k: int = 20,
) -> dict[BehavioralTrait, dict[str, float]]:
    """Analyze whether trait features follow the 3-DeltaNet + 1-attention block pattern.

    For each trait, compares TAS strength at DeltaNet position 2 (last before
    attention) vs attention position 3 within the same block. A consistent
    pattern where DeltaNet builds up representations that attention reads would
    show as correlated TAS scores across the block boundary.

    NOTE: This analysis is correlational. Causal claims about information flow
    within blocks require activation patching experiments not implemented here.

    Args:
        all_tas: Nested dict: trait → sae_id → TAS tensor.
        top_k: Number of top features to analyze.

    Returns:
        Dict mapping trait to {
            "deltanet_before_attention_ratio": float,  # ratio of TAS at DeltaNet vs attention
            "early_block_dominance": float,  # fraction of top features in early blocks
            "mid_block_dominance": float,
            "late_block_dominance": float,
        }.
    """
    # Map SAE IDs to block positions
    block_map = {}
    for hp in HOOK_POINTS:
        block_map[hp.sae_id] = {
            "layer_type": hp.layer_type,
            "block": hp.block,
        }

    results: dict[BehavioralTrait, dict[str, float]] = {}

    for trait, sae_tas in all_tas.items():
        # Get mean top-k TAS per SAE
        sae_strengths: dict[str, float] = {}
        for sae_id, tas in sae_tas.items():
            k = min(top_k, tas.shape[0])
            top_vals = tas.abs().topk(k).values
            sae_strengths[sae_id] = float(top_vals.mean().item())

        # Compare DeltaNet vs attention within same block depth
        delta_strengths = []
        attn_strengths = []
        for sae_id, strength in sae_strengths.items():
            if sae_id not in block_map:
                continue
            if block_map[sae_id]["layer_type"] == LayerType.DELTANET:
                delta_strengths.append(strength)
            else:
                attn_strengths.append(strength)

        delta_mean = float(np.mean(delta_strengths)) if delta_strengths else 0.0
        attn_mean = float(np.mean(attn_strengths)) if attn_strengths else 0.0
        ratio = delta_mean / max(attn_mean, 1e-8)

        # Depth dominance
        depth_groups = {"early": 0.0, "mid": 0.0, "late": 0.0}
        for sae_id, strength in sae_strengths.items():
            if "early" in sae_id:
                depth_groups["early"] += strength
            elif "mid" in sae_id:
                depth_groups["mid"] += strength
            elif "late" in sae_id:
                depth_groups["late"] += strength

        total = sum(depth_groups.values()) or 1.0

        results[trait] = {
            "deltanet_before_attention_ratio": ratio,
            "early_block_dominance": depth_groups["early"] / total,
            "mid_block_dominance": depth_groups["mid"] / total,
            "late_block_dominance": depth_groups["late"] / total,
        }

        logger.info(
            "Block structure for %s: delta/attn ratio=%.3f, depth dist=[%.2f, %.2f, %.2f]",
            trait.value,
            ratio,
            depth_groups["early"] / total,
            depth_groups["mid"] / total,
            depth_groups["late"] / total,
        )

    return results


def compare_within_type_positions(
    sae_dict: dict[str, TopKSAE],
    all_tas: dict[BehavioralTrait, dict[str, torch.Tensor]],
    top_k: int = 20,
) -> dict[BehavioralTrait, dict[str, float]]:
    """Compare position-1 vs position-2 DeltaNet SAEs within the same block.

    This function addresses the within-block position confound: when comparing
    DeltaNet and attention layers, we must verify that observed differences are
    due to layer *type* (DeltaNet vs attention) rather than *position* within
    the 3-DeltaNet + 1-attention block.

    Specifically compares sae_delta_mid_pos1 (layer 33, position 1 of block 8)
    with sae_delta_mid (layer 34, position 2 of block 8). Both are DeltaNet
    layers in the same block, differing only in their position within the
    3-DeltaNet sequence.

    If features are similar across positions (high overlap, high correlation),
    position is not a confound and DeltaNet-vs-attention is the primary factor.
    If features differ substantially, position within the block must be
    reported as a confound in the architecture comparison.

    NOTE: These SAEs hook the residual stream (sublayer + FFN + skip connection),
    so position effects could arise from cumulative residual stream composition
    rather than from the DeltaNet sublayer itself.

    Args:
        sae_dict: Dict mapping sae_id to trained SAE.
        all_tas: Nested dict: trait -> sae_id -> TAS tensor.
        top_k: Number of top features to compare per SAE.

    Returns:
        Dict mapping trait to {
            "tas_overlap": float,           # Jaccard similarity of top-k feature sets
            "decoder_weight_similarity": float,  # Mean cosine sim of top-k decoder vectors
            "tas_correlation": float,       # Pearson r of full TAS vectors
            "tas_correlation_pvalue": float, # p-value for Pearson correlation
            "mean_tas_pos1": float,         # Mean |TAS| of top-k at position 1
            "mean_tas_pos2": float,         # Mean |TAS| of top-k at position 2
        }.
    """
    pos1_id = "sae_delta_mid_pos1"
    pos2_id = "sae_delta_mid"

    if pos1_id not in sae_dict or pos2_id not in sae_dict:
        logger.warning(
            "Cannot compare within-type positions: missing SAE(s). "
            "Need both %s and %s in sae_dict.",
            pos1_id, pos2_id,
        )
        return {}

    sae_pos1 = sae_dict[pos1_id]
    sae_pos2 = sae_dict[pos2_id]

    results: dict[BehavioralTrait, dict[str, float]] = {}

    for trait, sae_tas in all_tas.items():
        if pos1_id not in sae_tas or pos2_id not in sae_tas:
            logger.warning(
                "Skipping trait %s: missing TAS for %s or %s.",
                trait.value, pos1_id, pos2_id,
            )
            continue

        tas_pos1 = sae_tas[pos1_id]
        tas_pos2 = sae_tas[pos2_id]

        # Top-k feature indices at each position
        top_features_pos1 = rank_features(tas_pos1, top_k, positive_only=False)
        top_features_pos2 = rank_features(tas_pos2, top_k, positive_only=False)
        indices_pos1 = {idx for idx, _ in top_features_pos1}
        indices_pos2 = {idx for idx, _ in top_features_pos2}

        # TAS overlap: Jaccard similarity of top-k feature index sets
        intersection = len(indices_pos1 & indices_pos2)
        union = len(indices_pos1 | indices_pos2)
        tas_overlap = intersection / max(union, 1)

        # Decoder weight similarity: mean cosine sim between top-k decoder vectors
        # nn.Linear(dict_size, hidden_dim) -> weight is (hidden_dim, dict_size)
        decoder_w_pos1 = sae_pos1.decoder.weight.detach().cpu().numpy().T  # (dict_size, hidden_dim)
        decoder_w_pos2 = sae_pos2.decoder.weight.detach().cpu().numpy().T  # (dict_size, hidden_dim)

        vectors_pos1 = decoder_w_pos1[list(indices_pos1)]  # (k, hidden_dim)
        vectors_pos2 = decoder_w_pos2[list(indices_pos2)]  # (k, hidden_dim)

        sim_matrix = cosine_similarity(vectors_pos1, vectors_pos2)  # (k, k)
        decoder_sim = float(np.abs(sim_matrix).mean())

        # TAS correlation: Pearson correlation of full TAS vectors
        tas_np_pos1 = tas_pos1.detach().cpu().numpy().astype(np.float64)
        tas_np_pos2 = tas_pos2.detach().cpu().numpy().astype(np.float64)
        corr, pvalue = pearsonr(tas_np_pos1, tas_np_pos2)

        # Mean TAS at each position
        k_pos1 = min(top_k, tas_pos1.shape[0])
        k_pos2 = min(top_k, tas_pos2.shape[0])
        mean_tas_pos1 = float(tas_pos1.abs().topk(k_pos1).values.mean().item())
        mean_tas_pos2 = float(tas_pos2.abs().topk(k_pos2).values.mean().item())

        results[trait] = {
            "tas_overlap": tas_overlap,
            "decoder_weight_similarity": decoder_sim,
            "tas_correlation": float(corr),
            "tas_correlation_pvalue": float(pvalue),
            "mean_tas_pos1": mean_tas_pos1,
            "mean_tas_pos2": mean_tas_pos2,
        }

        logger.info(
            "Within-type position comparison for %s: "
            "overlap=%.3f, decoder_sim=%.3f, corr=%.3f (p=%.2e), "
            "mean_tas=[pos1=%.4f, pos2=%.4f]",
            trait.value,
            tas_overlap,
            decoder_sim,
            float(corr),
            float(pvalue),
            mean_tas_pos1,
            mean_tas_pos2,
        )

    return results

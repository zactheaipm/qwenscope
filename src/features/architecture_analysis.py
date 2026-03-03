"""DeltaNet vs attention feature comparison analysis.

This is the novel analysis module — comparing how behavioral traits are
represented differently in layers containing DeltaNet (linear attention) vs
layers containing full attention.

IMPORTANT FRAMING: Hooks capture the residual stream after the complete layer
(sublayer + FFN + skip connection). Therefore, findings should be framed as
"layers containing DeltaNet show different feature distributions than layers
containing attention" rather than "DeltaNet represents differently than
attention." We cannot isolate the sublayer contribution from the FFN and skip.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from pydantic import BaseModel
from scipy import stats

from src.data.contrastive import BehavioralTrait
from src.model.config import HOOK_POINTS, LayerType, Qwen35Config

logger = logging.getLogger(__name__)


class TraitLayerTypeComparison(BaseModel):
    """Comparison of TAS scores between layer types for one trait."""

    trait: BehavioralTrait
    deltanet_max_tas: float
    attention_max_tas: float
    deltanet_mean_top20_tas: float
    attention_mean_top20_tas: float
    deltanet_significant_features: int
    attention_significant_features: int
    mwu_statistic: float
    mwu_pvalue: float
    stronger_layer_type: str  # "deltanet" or "attention"


class ArchitectureComparisonResult(BaseModel):
    """Full architecture comparison results."""

    per_trait: dict[str, TraitLayerTypeComparison]
    summary: str


def compare_layer_types(
    all_tas: dict[BehavioralTrait, dict[str, torch.Tensor]],
    config: Qwen35Config | None = None,
    significance_threshold: float = 2.0,
) -> ArchitectureComparisonResult:
    """Compare TAS distributions between DeltaNet and attention layer SAEs.

    For each trait:
    1. Aggregate TAS scores from DeltaNet SAEs (sae_delta_early/mid/late)
    2. Aggregate TAS scores from attention SAEs (sae_attn_early/mid/late)
    3. Compare: max TAS, mean of top-20 TAS, number of significant features
    4. Statistical test: Mann-Whitney U on full TAS distributions

    Args:
        all_tas: Nested dict: trait → sae_id → TAS tensor.
        config: Optional Qwen35Config.
        significance_threshold: |TAS| above this counts as significant.

    Returns:
        ArchitectureComparisonResult with per-trait comparisons.
    """
    if config is None:
        config = Qwen35Config()

    # Classify SAEs by layer type
    deltanet_sae_ids = [hp.sae_id for hp in HOOK_POINTS if hp.layer_type == LayerType.DELTANET]
    attention_sae_ids = [hp.sae_id for hp in HOOK_POINTS if hp.layer_type == LayerType.ATTENTION]

    per_trait: dict[str, TraitLayerTypeComparison] = {}

    for trait, sae_tas in all_tas.items():
        # Aggregate TAS scores by layer type
        deltanet_tas_all = []
        attention_tas_all = []

        for sae_id, tas in sae_tas.items():
            abs_tas = tas.abs().numpy()
            if sae_id in deltanet_sae_ids:
                deltanet_tas_all.append(abs_tas)
            elif sae_id in attention_sae_ids:
                attention_tas_all.append(abs_tas)

        if not deltanet_tas_all or not attention_tas_all:
            continue

        deltanet_combined = np.concatenate(deltanet_tas_all)
        attention_combined = np.concatenate(attention_tas_all)

        # Max TAS
        delta_max = float(np.max(deltanet_combined))
        attn_max = float(np.max(attention_combined))

        # Mean of top-20 TAS
        delta_top20 = float(np.mean(np.sort(deltanet_combined)[-20:]))
        attn_top20 = float(np.mean(np.sort(attention_combined)[-20:]))

        # Significant features
        delta_sig = int(np.sum(deltanet_combined > significance_threshold))
        attn_sig = int(np.sum(attention_combined > significance_threshold))

        # Mann-Whitney U test on full distributions (not truncated tails)
        mwu_stat, mwu_p = stats.mannwhitneyu(
            deltanet_combined, attention_combined, alternative="two-sided"
        )

        stronger = "attention" if attn_top20 > delta_top20 else "deltanet"

        per_trait[trait.value] = TraitLayerTypeComparison(
            trait=trait,
            deltanet_max_tas=delta_max,
            attention_max_tas=attn_max,
            deltanet_mean_top20_tas=delta_top20,
            attention_mean_top20_tas=attn_top20,
            deltanet_significant_features=delta_sig,
            attention_significant_features=attn_sig,
            mwu_statistic=float(mwu_stat),
            mwu_pvalue=float(mwu_p),
            stronger_layer_type=stronger,
        )

        logger.info(
            "Trait %s: DeltaNet top-20 mean=%.3f, Attention top-20 mean=%.3f → %s stronger",
            trait.value,
            delta_top20,
            attn_top20,
            stronger,
        )

    # Summary
    deltanet_wins = sum(1 for c in per_trait.values() if c.stronger_layer_type == "deltanet")
    attention_wins = sum(1 for c in per_trait.values() if c.stronger_layer_type == "attention")
    summary = (
        f"DeltaNet stronger for {deltanet_wins} traits, "
        f"Attention stronger for {attention_wins} traits"
    )

    return ArchitectureComparisonResult(per_trait=per_trait, summary=summary)


class WithinBlockComparison(BaseModel):
    """Comparison of TAS scores between block positions for one trait.

    Compares sae_delta_mid (position 2, last DeltaNet before attention) vs
    sae_delta_mid_pos1 (position 1, middle of DeltaNet sequence) within
    the same block. If position within the block matters more than layer
    type, that undermines the DeltaNet-vs-attention framing.
    """

    trait: BehavioralTrait
    pos1_mean_top20_tas: float  # sae_delta_mid_pos1 (position 1)
    pos2_mean_top20_tas: float  # sae_delta_mid (position 2)
    mwu_statistic: float
    mwu_pvalue: float
    position_effect_significant: bool  # p < 0.05


def compare_within_block_positions(
    all_tas: dict[BehavioralTrait, dict[str, torch.Tensor]],
    pos1_sae_id: str = "sae_delta_mid_pos1",
    pos2_sae_id: str = "sae_delta_mid",
) -> dict[str, WithinBlockComparison]:
    """Compare TAS distributions between two DeltaNet positions in the same block.

    This is a critical control for the architecture analysis. If TAS differs
    significantly between position 1 and position 2 within the same block
    (both are DeltaNet layers), then the DeltaNet-vs-attention comparison is
    confounded by within-block position effects.

    Args:
        all_tas: Nested dict: trait → sae_id → TAS tensor.
        pos1_sae_id: SAE ID for block position 1 (default: sae_delta_mid_pos1).
        pos2_sae_id: SAE ID for block position 2 (default: sae_delta_mid).

    Returns:
        Dict mapping trait name to WithinBlockComparison.
    """
    results: dict[str, WithinBlockComparison] = {}

    for trait, sae_tas in all_tas.items():
        if pos1_sae_id not in sae_tas or pos2_sae_id not in sae_tas:
            logger.warning(
                "Skipping within-block comparison for %s: missing %s or %s",
                trait.value, pos1_sae_id, pos2_sae_id,
            )
            continue

        pos1_tas = sae_tas[pos1_sae_id].abs().numpy()
        pos2_tas = sae_tas[pos2_sae_id].abs().numpy()

        pos1_top20 = float(np.mean(np.sort(pos1_tas)[-20:]))
        pos2_top20 = float(np.mean(np.sort(pos2_tas)[-20:]))

        mwu_stat, mwu_p = stats.mannwhitneyu(
            pos1_tas, pos2_tas, alternative="two-sided"
        )

        results[trait.value] = WithinBlockComparison(
            trait=trait,
            pos1_mean_top20_tas=pos1_top20,
            pos2_mean_top20_tas=pos2_top20,
            mwu_statistic=float(mwu_stat),
            mwu_pvalue=float(mwu_p),
            position_effect_significant=mwu_p < 0.05,
        )

        logger.info(
            "Within-block %s: pos1 top-20=%.3f, pos2 top-20=%.3f, p=%.4f%s",
            trait.value,
            pos1_top20,
            pos2_top20,
            mwu_p,
            " (SIGNIFICANT)" if mwu_p < 0.05 else "",
        )

    return results


def trait_localization_score(
    all_tas: dict[BehavioralTrait, dict[str, torch.Tensor]],
) -> dict[BehavioralTrait, dict[str, float]]:
    """For each trait, compute how localized it is to specific layer types/depths.

    A high localization score means the trait's top features are concentrated
    in one layer type. A low score means they're distributed across both.

    Uses the Gini coefficient on the distribution of top-feature counts
    across SAEs as the localization metric.

    Args:
        all_tas: Nested dict: trait → sae_id → TAS tensor.

    Returns:
        Dict mapping trait to {
            "layer_type_localization": float (0=distributed, 1=fully localized),
            "depth_localization": float (0=distributed, 1=fully localized),
            "best_sae": str (sae_id with highest mean top-20 TAS)
        }.
    """
    results: dict[BehavioralTrait, dict[str, float]] = {}

    deltanet_sae_ids = {hp.sae_id for hp in HOOK_POINTS if hp.layer_type == LayerType.DELTANET}
    attention_sae_ids = {hp.sae_id for hp in HOOK_POINTS if hp.layer_type == LayerType.ATTENTION}

    # Group by depth
    early_sae_ids = {"sae_attn_early", "sae_delta_early"}
    mid_sae_ids = {"sae_attn_mid", "sae_delta_mid", "sae_delta_mid_pos1"}
    late_sae_ids = {"sae_attn_late", "sae_delta_late"}

    for trait, sae_tas in all_tas.items():
        # Count significant features per SAE
        sae_counts: dict[str, float] = {}
        for sae_id, tas in sae_tas.items():
            top20_mean = float(tas.abs().topk(min(20, tas.shape[0])).values.mean().item())
            sae_counts[sae_id] = top20_mean

        # Layer type localization
        delta_score = sum(sae_counts.get(s, 0) for s in deltanet_sae_ids)
        attn_score = sum(sae_counts.get(s, 0) for s in attention_sae_ids)
        total = delta_score + attn_score
        if total > 0:
            layer_type_loc = abs(delta_score - attn_score) / total
        else:
            layer_type_loc = 0.0

        # Depth localization
        early_score = sum(sae_counts.get(s, 0) for s in early_sae_ids)
        mid_score = sum(sae_counts.get(s, 0) for s in mid_sae_ids)
        late_score = sum(sae_counts.get(s, 0) for s in late_sae_ids)
        depth_total = early_score + mid_score + late_score
        if depth_total > 0:
            depth_scores = np.array([early_score, mid_score, late_score]) / depth_total
            # Gini-like: max deviation from uniform (1/3)
            depth_loc = float(np.max(depth_scores) - 1 / 3) * 3  # Normalized 0-1
            depth_loc = max(0.0, min(1.0, depth_loc))
        else:
            depth_loc = 0.0

        best_sae = max(sae_counts, key=sae_counts.get) if sae_counts else ""

        results[trait] = {
            "layer_type_localization": layer_type_loc,
            "depth_localization": depth_loc,
            "best_sae": best_sae,
        }

    return results

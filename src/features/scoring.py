"""Trait Association Score (TAS) computation.

TAS measures how strongly each SAE feature is associated with a behavioral trait
by comparing activations on contrastive HIGH vs LOW prompt pairs.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from scipy import stats

from src.data.contrastive import BehavioralTrait
from src.evaluation.behavioral_metrics import SUB_BEHAVIOR_KEYS
from src.features.extraction import FeatureExtractionResults, FeaturePairResult

logger = logging.getLogger(__name__)


def compute_tas(
    extraction_results: FeatureExtractionResults,
    trait: BehavioralTrait,
    sae_id: str,
    min_abs_effect: float = 0.01,
) -> torch.Tensor:
    """Compute Trait Association Score for all features in one SAE for one trait.

    TAS(feature_i) = mean(act_high - act_low) / std(act_high - act_low)

    Computed across all contrastive pairs for this trait.
    Features with zero std get TAS = 0.

    Args:
        extraction_results: Feature extraction results for this trait.
        trait: The behavioral trait (must match extraction_results.trait).
        sae_id: Which SAE to compute TAS for.

    Returns:
        Tensor of shape (dict_size,) — TAS per feature.
    """
    pair_results = extraction_results.results[sae_id]

    if not pair_results:
        logger.warning("No pair results for %s / %s", trait.value, sae_id)
        return torch.zeros(0)

    # Build tensors of high and low feature activations
    dict_size = len(pair_results[0].features_high_mean)
    n_pairs = len(pair_results)

    diffs = torch.zeros(n_pairs, dict_size, dtype=torch.float64)
    for i, result in enumerate(pair_results):
        high = torch.tensor(result.features_high_mean, dtype=torch.float64)
        low = torch.tensor(result.features_low_mean, dtype=torch.float64)
        diffs[i] = high - low  # (dict_size,)

    # TAS = mean / std (like a t-statistic).
    # Explicit correction=1 (Bessel's) for consistency with the numpy ddof=1
    # used in statistical_significance permutation tests.
    mean_diff = diffs.mean(dim=0)  # (dict_size,)
    std_diff = diffs.std(dim=0, correction=1)  # (dict_size,)

    # Avoid division by zero
    tas = torch.where(
        std_diff > 1e-8,
        mean_diff / std_diff,
        torch.zeros_like(mean_diff),
    )

    # Zero out features with negligible absolute effect — prevents near-dead
    # features with infinitesimal but perfectly consistent differences from
    # getting astronomical TAS scores.
    min_effect_mask = mean_diff.abs() < min_abs_effect
    tas[min_effect_mask] = 0.0

    logger.info(
        "TAS computed for %s / %s: max=%.3f, min=%.3f, mean_abs=%.3f",
        trait.value,
        sae_id,
        tas.max().item(),
        tas.min().item(),
        tas.abs().mean().item(),
    )
    return tas


def compute_all_tas(
    extraction_results_by_trait: dict[BehavioralTrait, FeatureExtractionResults],
) -> dict[BehavioralTrait, dict[str, torch.Tensor]]:
    """Compute TAS for all traits × all SAEs.

    Args:
        extraction_results_by_trait: Dict mapping trait to its extraction results.

    Returns:
        Nested dict: trait → sae_id → TAS tensor of shape (dict_size,).
    """
    all_tas: dict[BehavioralTrait, dict[str, torch.Tensor]] = {}

    for trait, extraction_results in extraction_results_by_trait.items():
        all_tas[trait] = {}
        for sae_id in extraction_results.results:
            all_tas[trait][sae_id] = compute_tas(extraction_results, trait, sae_id)

    return all_tas


def rank_features(
    tas_scores: torch.Tensor,
    top_k: int = 20,
    positive_only: bool = True,
) -> list[tuple[int, float]]:
    """Return top-k feature indices and their TAS scores.

    Args:
        tas_scores: TAS tensor of shape (dict_size,).
        top_k: Number of top features to return.
        positive_only: If True (default), return only features with positive
            TAS (fire more on HIGH).  This is the correct default for
            amplification steering — negative-TAS features fire more on
            LOW and amplifying them would push behaviour *away* from the
            target trait.  Set to False for analysis where both directions
            are informative.

    Returns:
        List of (feature_index, tas_score) tuples sorted by TAS (descending)
        when positive_only=True, or by |TAS| (descending) when False.
    """
    if positive_only:
        # Mask out non-positive features, then take top-k by value
        positive_tas = tas_scores.clone()
        positive_tas[positive_tas <= 0] = 0.0
        k = min(top_k, (positive_tas > 0).sum().item())
        if k == 0:
            return []
        topk_values, topk_indices = torch.topk(positive_tas, k)
        return [
            (int(idx.item()), float(val.item()))
            for idx, val in zip(topk_indices, topk_values)
            if val.item() > 0
        ]
    else:
        abs_tas = tas_scores.abs()
        k = min(top_k, abs_tas.shape[0])
        topk_values, topk_indices = torch.topk(abs_tas, k)
        return [
            (int(idx.item()), float(tas_scores[idx].item()))
            for idx, val in zip(topk_indices, topk_values)
        ]


def compute_null_tas_distribution(
    null_extraction_results: FeatureExtractionResults,
    sae_id: str,
) -> torch.Tensor:
    """Compute TAS scores on null-trait control pairs.

    The resulting distribution is the TAS you'd get from behaviorally-
    irrelevant system prompt differences. Features whose real TAS exceeds
    the 95th percentile of this null distribution are more likely to encode
    genuine behavioral traits rather than instruction-sensitivity.

    Args:
        null_extraction_results: Feature extraction results from null control
            pairs (generated by ``ContrastivePairGenerator.generate_null_controls``).
        sae_id: Which SAE to compute null TAS for.

    Returns:
        Tensor of shape (dict_size,) — null TAS per feature.
    """
    return compute_tas(
        null_extraction_results,
        null_extraction_results.trait,
        sae_id,
    )


def filter_by_null_tas(
    real_tas: torch.Tensor,
    null_tas: torch.Tensor,
    percentile: float = 95.0,
) -> tuple[torch.Tensor, float]:
    """Filter features whose |TAS| doesn't exceed the null distribution.

    Args:
        real_tas: TAS scores from real contrastive pairs (dict_size,).
        null_tas: TAS scores from null control pairs (dict_size,).
        percentile: Percentile of null |TAS| to use as threshold.

    Returns:
        Tuple of (filtered_tas, threshold) where filtered_tas has features
        below threshold zeroed out.
    """
    null_abs = null_tas.abs()
    threshold = float(torch.quantile(null_abs, percentile / 100.0).item())

    filtered = real_tas.clone()
    filtered[real_tas.abs() <= threshold] = 0.0

    n_surviving = int((filtered != 0).sum().item())
    logger.info(
        "Null TAS filter: threshold=%.3f (%.0fth pctl), "
        "%d / %d features survive",
        threshold, percentile, n_surviving, real_tas.shape[0],
    )
    return filtered, threshold


def normalize_tas_cross_sae(
    real_tas: torch.Tensor,
    null_tas: torch.Tensor,
) -> torch.Tensor:
    """Z-score normalize TAS against null distribution for cross-SAE comparison.

    SAEs with different architectures (dict_size, k) have different null TAS
    distributions — a TAS of 5.0 in a 20K-feature SAE means something
    different from 5.0 in a 40K-feature SAE. Normalizing each feature's TAS
    by the null distribution's standard deviation puts all SAEs into the same
    units: "standard deviations above chance for this SAE's architecture."

    This enables fair cross-SAE comparison (e.g., in ``_get_best_sae_for_trait``)
    when SAEs use different dict_size or k values across depth bands.

    Args:
        real_tas: TAS scores from real contrastive pairs, shape (dict_size,).
        null_tas: TAS scores from null control pairs, shape (dict_size,).

    Returns:
        Normalized TAS of shape (dict_size,). Each value is the number of
        null-distribution standard deviations the real TAS exceeds the null
        mean. Features below the null mean get negative values.
    """
    null_abs = null_tas.abs().float()
    null_mean = null_abs.mean()
    null_std = null_abs.std()

    if null_std < 1e-8:
        logger.warning(
            "Null TAS distribution has near-zero std (%.2e) — "
            "normalization would be unstable. Returning raw TAS.",
            null_std.item(),
        )
        return real_tas

    normalized = (real_tas.abs().float() - null_mean) / null_std
    # Preserve sign of original TAS (positive = fires more on HIGH)
    normalized = normalized * real_tas.sign().float()

    logger.info(
        "TAS normalized: null_mean=%.3f, null_std=%.3f, "
        "max normalized=%.3f, features >2σ: %d / %d",
        null_mean.item(),
        null_std.item(),
        normalized.abs().max().item(),
        int((normalized.abs() > 2.0).sum().item()),
        real_tas.shape[0],
    )
    return normalized


def statistical_significance(
    extraction_results: FeatureExtractionResults,
    trait: BehavioralTrait,
    sae_id: str,
    feature_idx: int,
    n_permutations: int = 1000,
    seed: int = 42,
) -> float:
    """Permutation test p-value for a feature's TAS score.

    Randomly shuffles HIGH/LOW labels and recomputes TAS to build a null
    distribution, then computes the p-value.

    Args:
        extraction_results: Feature extraction results.
        trait: The behavioral trait.
        sae_id: Which SAE.
        feature_idx: Which feature to test.
        n_permutations: Number of permutations for the test.
        seed: Random seed for reproducibility.

    Returns:
        Two-sided p-value (probability of observing TAS this extreme by chance).
    """
    rng = np.random.RandomState(seed)
    pair_results = extraction_results.results[sae_id]

    # Get high and low activations for this feature
    highs = np.array([r.features_high_mean[feature_idx] for r in pair_results])
    lows = np.array([r.features_low_mean[feature_idx] for r in pair_results])
    n_pairs = len(highs)

    # Observed TAS
    diffs = highs - lows
    observed_tas = np.mean(diffs) / max(np.std(diffs, ddof=1), 1e-8)

    # Permutation test
    count_extreme = 0
    for _ in range(n_permutations):
        # Randomly swap high/low for each pair
        swap_mask = rng.binomial(1, 0.5, size=n_pairs).astype(bool)
        perm_highs = np.where(swap_mask, lows, highs)
        perm_lows = np.where(swap_mask, highs, lows)
        perm_diffs = perm_highs - perm_lows
        perm_tas = np.mean(perm_diffs) / max(np.std(perm_diffs, ddof=1), 1e-8)

        if abs(perm_tas) >= abs(observed_tas):
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_permutations + 1)
    return float(p_value)


def batch_significance_with_fdr(
    extraction_results: FeatureExtractionResults,
    trait: BehavioralTrait,
    sae_id: str,
    feature_indices: list[int],
    n_permutations: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> list[tuple[int, float, float, bool]]:
    """Compute significance for a subset of features with FDR correction.

    With 40,960 features, uncorrected p < 0.05 yields ~2,048 false positives.
    Benjamini-Hochberg FDR correction controls the false discovery rate.

    IMPORTANT — pre-selection invalidates FDR: if ``feature_indices`` was
    obtained by first ranking features by TAS and taking the top-k, the BH
    procedure no longer controls FDR at the stated ``alpha`` because you have
    already selected on the test statistic. For valid FDR control across all
    features, use ``fdr_screen_all_features()`` instead, which tests all
    40,960 features analytically without prior selection.

    Args:
        extraction_results: Feature extraction results.
        trait: The behavioral trait.
        sae_id: Which SAE.
        feature_indices: Feature indices to test.
        n_permutations: Number of permutations per feature.
        alpha: FDR significance level.
        seed: Random seed.

    Returns:
        List of (feature_idx, raw_p, corrected_p, is_significant) tuples
        sorted by corrected p-value.
    """
    # Compute raw p-values for all features.
    # Each feature gets a different seed (seed + feature_idx) to avoid
    # correlated permutation sequences across features, which would
    # compromise FDR control.
    raw_pvals = []
    for idx in feature_indices:
        p = statistical_significance(
            extraction_results, trait, sae_id, idx,
            n_permutations=n_permutations, seed=seed + idx,
        )
        raw_pvals.append((idx, p))

    # Benjamini-Hochberg FDR correction
    n_tests = len(raw_pvals)
    if n_tests == 0:
        return []

    sorted_pvals = sorted(raw_pvals, key=lambda x: x[1])

    # Compute corrected p-values: p_corrected(k) = p_raw(k) * n / rank
    results = []
    for rank, (idx, raw_p) in enumerate(sorted_pvals, start=1):
        corrected_p = min(raw_p * n_tests / rank, 1.0)
        results.append((idx, raw_p, corrected_p, False))  # significance set below

    # Enforce monotonicity of corrected p-values (step-up from the bottom):
    # corrected_p(k) = min(corrected_p(k), corrected_p(k+1))
    for i in range(len(results) - 2, -1, -1):
        idx_i, raw_i, corr_i, _ = results[i]
        _, _, corr_next, _ = results[i + 1]
        if corr_i > corr_next:
            results[i] = (idx_i, raw_i, corr_next, False)

    # Determine significance from the monotonicity-enforced corrected p-values.
    # This is consistent by construction: corrected_p <= alpha ⟺ significant.
    results = [
        (idx, raw_p, corr_p, corr_p <= alpha)
        for idx, raw_p, corr_p, _ in results
    ]

    # Re-sort by corrected p-value
    results.sort(key=lambda x: x[2])

    n_sig = sum(1 for _, _, _, s in results if s)
    logger.info(
        "FDR correction for %s / %s: %d / %d features significant at alpha=%.3f",
        trait.value, sae_id, n_sig, n_tests, alpha,
    )

    return results


def compute_tas_cluster_robust(
    extraction_results: FeatureExtractionResults,
    trait: BehavioralTrait,
    sae_id: str,
    pairs_per_template: int = 4,
    min_abs_effect: float = 0.01,
) -> tuple[torch.Tensor, int]:
    """Compute TAS with cluster-robust standard errors.

    Contrastive pairs within a template group share the same system prompt and
    behavioral dynamics. The ``pairs_per_template`` variations within each
    template are near-replicates, not independent observations. Treating them
    as independent inflates the effective sample size and produces overly
    narrow confidence intervals.

    This function groups consecutive pairs into template clusters, averages
    the within-cluster diffs to produce one observation per cluster, and
    then computes the t-statistic on the cluster means. The resulting TAS
    has correct coverage because the unit of analysis matches the unit of
    independence.

    Template clusters are inferred from pair ordering: pairs 0..3 form
    cluster 0, pairs 4..7 form cluster 1, etc. (matching the pair ID
    convention where e.g. ``autonomy_coding_000`` through
    ``autonomy_coding_003`` share one template).

    Args:
        extraction_results: Feature extraction results for this trait.
        trait: The behavioral trait (must match extraction_results.trait).
        sae_id: Which SAE to compute cluster-robust TAS for.
        pairs_per_template: Number of variation pairs per template cluster.

    Returns:
        Tuple of (tas_tensor, effective_n) where:
            - tas_tensor has shape (dict_size,) -- cluster-robust TAS per feature.
            - effective_n is the number of independent template clusters used.
    """
    pair_results = extraction_results.results[sae_id]

    if not pair_results:
        logger.warning("No pair results for %s / %s", trait.value, sae_id)
        return torch.zeros(0), 0

    # Validate that consecutive pairs share the same template cluster.
    # Pair IDs follow the convention "trait_domain_NNN" where NNN // pairs_per_template
    # gives the cluster index. If pairs are not grouped by template, the clustering
    # assumption is violated and we should warn.
    if hasattr(pair_results[0], "pair_id") and pair_results[0].pair_id:
        for cluster_start in range(0, len(pair_results) - pairs_per_template + 1, pairs_per_template):
            cluster_ids = [
                pair_results[cluster_start + k].pair_id
                for k in range(pairs_per_template)
                if hasattr(pair_results[cluster_start + k], "pair_id")
            ]
            if len(cluster_ids) == pairs_per_template:
                # Extract template base: everything before the last underscore-separated variation index
                bases = set()
                for pid in cluster_ids:
                    # "autonomy_coding_003" -> base "autonomy_coding", variation "003"
                    parts = pid.rsplit("_", 1)
                    if len(parts) == 2:
                        bases.add(parts[0])
                if len(bases) > 1:
                    logger.warning(
                        "Cluster-robust TAS for %s / %s: pairs at indices %d-%d "
                        "span multiple templates (%s). Results may be unreliable. "
                        "Ensure pairs are ordered by template before calling.",
                        trait.value, sae_id,
                        cluster_start, cluster_start + pairs_per_template - 1,
                        bases,
                    )
                    break  # Log once, not for every cluster

    dict_size = len(pair_results[0].features_high_mean)
    n_pairs = len(pair_results)

    # Build per-pair diffs: (n_pairs, dict_size)
    diffs = torch.zeros(n_pairs, dict_size, dtype=torch.float64)
    for i, result in enumerate(pair_results):
        high = torch.tensor(result.features_high_mean, dtype=torch.float64)
        low = torch.tensor(result.features_low_mean, dtype=torch.float64)
        diffs[i] = high - low

    # Group into template clusters and average within each cluster
    n_clusters = n_pairs // pairs_per_template
    remainder = n_pairs % pairs_per_template

    if n_clusters == 0:
        logger.warning(
            "Too few pairs (%d) for cluster size %d in %s / %s. "
            "Falling back to unclustered TAS.",
            n_pairs, pairs_per_template, trait.value, sae_id,
        )
        return compute_tas(extraction_results, trait, sae_id), n_pairs

    if remainder > 0:
        logger.info(
            "Dropping %d trailing pairs (not a full template cluster) "
            "for %s / %s",
            remainder, trait.value, sae_id,
        )

    # Reshape to (n_clusters, pairs_per_template, dict_size) and mean
    # over the within-cluster axis to get one observation per template.
    cluster_diffs = diffs[: n_clusters * pairs_per_template].view(
        n_clusters, pairs_per_template, dict_size
    )
    cluster_means = cluster_diffs.mean(dim=1)  # (n_clusters, dict_size)

    # TAS on cluster means — correction=1 (Bessel's) for consistency.
    mean_diff = cluster_means.mean(dim=0)  # (dict_size,)
    std_diff = cluster_means.std(dim=0, correction=1)  # (dict_size,)

    tas = torch.where(
        std_diff > 1e-8,
        mean_diff / std_diff,
        torch.zeros_like(mean_diff),
    )

    # Zero out features with negligible absolute effect — prevents near-dead
    # features with infinitesimal but perfectly consistent differences from
    # getting astronomical TAS scores.
    min_effect_mask = mean_diff.abs() < min_abs_effect
    tas[min_effect_mask] = 0.0

    logger.info(
        "Cluster-robust TAS for %s / %s: effective_n=%d clusters "
        "(from %d pairs), max=%.3f, min=%.3f, mean_abs=%.3f",
        trait.value,
        sae_id,
        n_clusters,
        n_pairs,
        tas.max().item(),
        tas.min().item(),
        tas.abs().mean().item(),
    )
    return tas, n_clusters


def parametric_significance(
    extraction_results: FeatureExtractionResults,
    trait: BehavioralTrait,
    sae_id: str,
    feature_idx: int,
) -> float:
    """Analytical parametric p-value for a feature's trait association.

    Uses a one-sample t-test on the paired differences (high - low). The
    t-statistic is ``mean(diffs) / (std(diffs) / sqrt(n))``, which follows
    a Student-t distribution with n-1 degrees of freedom under the null.

    Note: TAS as defined in ``compute_tas`` is ``mean / std`` (a standardised
    effect size, i.e. Cohen's d), *not* a t-statistic.  The p-value must use
    the proper t-statistic (which includes the sqrt(n) factor) to be valid.
    Using TAS directly as a t-statistic under-states significance by sqrt(n).

    This is appropriate when paired differences are approximately normal —
    a reasonable assumption by CLT when n_pairs >= 20. For small samples or
    non-Gaussian distributions, prefer ``statistical_significance()`` (the
    permutation test).

    Args:
        extraction_results: Feature extraction results.
        trait: The behavioral trait.
        sae_id: Which SAE.
        feature_idx: Which feature to test.

    Returns:
        Two-sided p-value from the Student-t distribution.
    """
    pair_results = extraction_results.results[sae_id]

    highs = np.array([r.features_high_mean[feature_idx] for r in pair_results])
    lows = np.array([r.features_low_mean[feature_idx] for r in pair_results])
    n_pairs = len(highs)

    if n_pairs < 2:
        logger.warning(
            "Cannot compute parametric p-value with n_pairs=%d for "
            "feature %d in %s / %s",
            n_pairs, feature_idx, trait.value, sae_id,
        )
        return 1.0

    diffs = highs - lows
    std_diff = float(np.std(diffs, ddof=1))

    if std_diff < 1e-8:
        if abs(float(np.mean(diffs))) < 1e-8:
            return 1.0
        return 0.0

    # Proper t-statistic: mean / SE, where SE = std / sqrt(n)
    t_stat = float(np.mean(diffs)) / (std_diff / np.sqrt(n_pairs))
    df = n_pairs - 1
    p_value = float(stats.t.sf(abs(t_stat), df=df) * 2)

    logger.debug(
        "Parametric p-value for feature %d (%s / %s): t=%.4f, df=%d, p=%.6f",
        feature_idx, trait.value, sae_id, t_stat, df, p_value,
    )
    return p_value


def compute_all_parametric_pvalues(
    extraction_results: FeatureExtractionResults,
    sae_id: str,
) -> np.ndarray:
    """Compute analytical p-values for ALL features in bulk.

    More efficient than calling ``parametric_significance`` per feature because
    it vectorises the t-test over the entire dictionary at once. Use this when
    you need FDR-controlled feature discovery across all 40,960 features.

    Args:
        extraction_results: Feature extraction results for one trait.
        sae_id: Which SAE to compute p-values for.

    Returns:
        Array of shape (dict_size,) containing two-sided p-values.
    """
    pair_results = extraction_results.results.get(sae_id, [])
    if not pair_results:
        return np.ones(0)

    dict_size = len(pair_results[0].features_high_mean)
    n_pairs = len(pair_results)

    # Shape: (dict_size, n_pairs)
    highs = np.array([[r.features_high_mean[i] for r in pair_results]
                      for i in range(dict_size)])
    lows = np.array([[r.features_low_mean[i] for r in pair_results]
                     for i in range(dict_size)])

    diffs = highs - lows                          # (dict_size, n_pairs)
    means = diffs.mean(axis=1)                    # (dict_size,)
    stds = diffs.std(axis=1, ddof=1)              # (dict_size,)

    # Proper t-statistic with sqrt(n) in the denominator
    se = np.where(stds > 1e-8, stds / np.sqrt(n_pairs), np.inf)
    t_stats = means / se                          # (dict_size,)

    df = n_pairs - 1
    p_values = 2 * stats.t.sf(np.abs(t_stats), df=df)  # two-sided

    return p_values.astype(np.float64)


def fdr_screen_all_features(
    extraction_results: FeatureExtractionResults,
    sae_id: str,
    alpha: float = 0.05,
) -> list[tuple[int, float, float, bool]]:
    """FDR-controlled feature screening across the entire dictionary.

    Computes analytical p-values for ALL features (no prior TAS selection)
    and applies Benjamini-Hochberg correction. This is the statistically
    valid approach: selecting features by TAS rank *before* running FDR
    conditions on the test statistic and invalidates the FDR guarantee.

    Args:
        extraction_results: Feature extraction results for one trait.
        sae_id: Which SAE.
        alpha: FDR significance level (false discovery rate target).

    Returns:
        List of ``(feature_idx, raw_p, corrected_p, is_significant)`` tuples
        sorted by corrected p-value (ascending). Only features with
        ``is_significant=True`` survive FDR control at the given alpha.
    """
    p_values = compute_all_parametric_pvalues(extraction_results, sae_id)
    n = len(p_values)
    if n == 0:
        return []

    # BH step-down procedure on all features.
    # 1. Sort by raw p ascending.
    # 2. Find k_max: largest rank k where p_(k) <= (k/n) * alpha.
    # 3. Reject all features with rank <= k_max (step-down property).
    #
    # The previous elementwise comparison (p_i <= rank_i/n * alpha) is WRONG
    # because it can produce non-monotonic rejections: a feature with a larger
    # p-value marked significant while one with a smaller p-value is not.
    order = np.argsort(p_values)                  # indices that sort ascending
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)

    # Find k_max: scan sorted p-values against BH thresholds
    k_max = 0
    for rank_k in range(1, n + 1):
        feature_idx = order[rank_k - 1]           # feature with the k-th smallest p
        if p_values[feature_idx] <= (rank_k / n) * alpha:
            k_max = rank_k

    # All features with rank <= k_max are significant (step-down property)
    is_significant = ranks <= k_max

    # Corrected p-values with monotonicity enforcement
    corrected = np.minimum(p_values * n / ranks, 1.0)
    for i in range(n - 2, -1, -1):
        corrected[order[i]] = min(corrected[order[i]], corrected[order[i + 1]])

    results = [
        (int(i), float(p_values[i]), float(corrected[i]), bool(is_significant[i]))
        for i in range(n)
    ]
    results.sort(key=lambda x: x[2])

    n_sig = sum(1 for _, _, _, s in results if s)
    logger.info(
        "FDR screen (all %d features, %s / %s): %d significant at alpha=%.3f",
        n, extraction_results.trait.value, sae_id, n_sig, alpha,
    )
    return results


# ---------------------------------------------------------------------------
# Sub-behavior-level TAS
# ---------------------------------------------------------------------------

# Map sub-behavior keys to their parent trait
_SUB_TO_TRAIT: dict[str, str] = {
    key: key.split(".")[0] for key in SUB_BEHAVIOR_KEYS
}


def compute_sub_behavior_tas(
    extraction_results: FeatureExtractionResults,
    sae_id: str,
    sub_behavior: str,
    min_pairs: int = 8,
    min_abs_effect: float = 0.01,
) -> tuple[torch.Tensor | None, int, bool]:
    """Compute TAS using only pairs annotated for a specific sub-behavior.

    Filters extraction results to pairs whose ``target_sub_behaviors``
    includes the specified sub-behavior, then computes TAS on the filtered
    set.  This produces a sub-behavior-specific TAS that identifies features
    controlling one sub-behavior in isolation (e.g., ``action_initiation``
    without ``permission_avoidance``).

    If fewer than ``min_pairs`` pairs match, returns ``None`` for the TAS
    tensor and sets ``used_fallback=True`` so that downstream code knows the
    sub-behavior was not independently computed.

    Args:
        extraction_results: Feature extraction results for the parent trait.
        sae_id: Which SAE to compute TAS for.
        sub_behavior: Sub-behavior key (e.g., ``"autonomy.action_initiation"``).
        min_pairs: Minimum annotated pairs required; below this the sub-behavior
            is skipped (TAS returned as ``None``).
        min_abs_effect: Minimum absolute mean activation difference required
            for a feature to receive a non-zero TAS score.

    Returns:
        Tuple of ``(tas_tensor, n_pairs_used, used_fallback)`` where:
            - ``tas_tensor`` has shape ``(dict_size,)`` or is ``None`` when
              fewer than ``min_pairs`` pairs are available.
            - ``n_pairs_used`` is the number of pairs actually used (or the
              number of annotated pairs found when falling back).
            - ``used_fallback`` is ``True`` when the sub-behavior could not
              be computed due to insufficient pairs.
    """
    if sub_behavior not in SUB_BEHAVIOR_KEYS:
        raise ValueError(
            f"Unknown sub-behavior {sub_behavior!r}. "
            f"Expected one of {SUB_BEHAVIOR_KEYS}"
        )

    pair_results = extraction_results.results[sae_id]
    if not pair_results:
        logger.warning("No pair results for %s", sae_id)
        return None, 0, True

    # Filter to pairs annotated for this sub-behavior
    filtered: list[FeaturePairResult] = [
        r for r in pair_results if sub_behavior in r.target_sub_behaviors
    ]

    if len(filtered) < min_pairs:
        logger.warning(
            "Only %d pairs annotated for %s (min_pairs=%d). "
            "Skipping sub-behavior TAS — returning None.",
            len(filtered), sub_behavior, min_pairs,
        )
        return None, len(filtered), True

    # Compute TAS on filtered pairs
    dict_size = len(filtered[0].features_high_mean)
    n_pairs = len(filtered)
    diffs = torch.zeros(n_pairs, dict_size, dtype=torch.float64)

    for i, result in enumerate(filtered):
        high = torch.tensor(result.features_high_mean, dtype=torch.float64)
        low = torch.tensor(result.features_low_mean, dtype=torch.float64)
        diffs[i] = high - low

    mean_diff = diffs.mean(dim=0)
    std_diff = diffs.std(dim=0, correction=1)

    tas = torch.where(
        std_diff > 1e-8,
        mean_diff / std_diff,
        torch.zeros_like(mean_diff),
    )

    # Zero out features with negligible absolute effect — prevents near-dead
    # features with infinitesimal but perfectly consistent differences from
    # getting astronomical TAS scores.
    min_effect_mask = mean_diff.abs() < min_abs_effect
    tas[min_effect_mask] = 0.0

    logger.info(
        "Sub-behavior TAS for %s / %s: n_pairs=%d, max=%.3f, "
        "min=%.3f, mean_abs=%.3f",
        sub_behavior, sae_id, n_pairs,
        tas.max().item(), tas.min().item(), tas.abs().mean().item(),
    )
    return tas, n_pairs, False


def compute_all_sub_behavior_tas(
    extraction_results_by_trait: dict[BehavioralTrait, FeatureExtractionResults],
) -> dict[str, dict[str, tuple[torch.Tensor | None, int, bool]]]:
    """Compute sub-behavior TAS for all 15 sub-behaviors across all SAEs.

    Args:
        extraction_results_by_trait: Dict mapping trait to extraction results.

    Returns:
        Nested dict: ``sub_behavior -> sae_id -> (tas_tensor, n_pairs, used_fallback)``.
        ``tas_tensor`` is ``None`` when the sub-behavior had too few annotated
        pairs (``used_fallback=True``).
    """
    all_sub_tas: dict[str, dict[str, tuple[torch.Tensor | None, int, bool]]] = {}

    for sub_behavior in SUB_BEHAVIOR_KEYS:
        trait_value = _SUB_TO_TRAIT[sub_behavior]
        # Map trait string to BehavioralTrait enum
        trait = None
        for t in BehavioralTrait:
            if t.value == trait_value:
                trait = t
                break
        if trait is None or trait not in extraction_results_by_trait:
            logger.warning("No extraction results for trait %s", trait_value)
            continue

        extraction_results = extraction_results_by_trait[trait]
        all_sub_tas[sub_behavior] = {}

        for sae_id in extraction_results.results:
            tas, n, fallback = compute_sub_behavior_tas(
                extraction_results, sae_id, sub_behavior,
            )
            all_sub_tas[sub_behavior][sae_id] = (tas, n, fallback)

    return all_sub_tas


# ---------------------------------------------------------------------------
# Cross-trait specificity
# ---------------------------------------------------------------------------


def flag_nonspecific_features(
    tas_by_trait: dict[BehavioralTrait, torch.Tensor],
    top_k: int = 100,
) -> torch.Tensor:
    """Identify features that appear in the top-k for multiple traits.

    A feature that ranks highly for two or more traits is likely encoding
    general instruction-sensitivity or prompt formatting rather than a
    specific behavioral axis. This function returns a boolean mask that
    downstream code can use to suppress or deprioritise such features.

    Args:
        tas_by_trait: Dict mapping each trait to its TAS tensor of shape
            ``(dict_size,)``. All tensors must share the same ``dict_size``.
        top_k: Number of top features (by ``|TAS|``) to consider per trait.

    Returns:
        Boolean tensor of shape ``(dict_size,)`` where ``True`` marks
        features that appear in the top-k of 2 or more traits (i.e.,
        non-specific features).
    """
    if not tas_by_trait:
        return torch.zeros(0, dtype=torch.bool)

    # Infer dict_size from the first tensor
    first_tensor = next(iter(tas_by_trait.values()))
    dict_size = first_tensor.shape[0]

    # Count how many traits each feature appears in the top-k for
    trait_count = torch.zeros(dict_size, dtype=torch.int64)

    for trait, tas in tas_by_trait.items():
        if tas.shape[0] != dict_size:
            raise ValueError(
                f"TAS tensor for {trait.value} has dict_size={tas.shape[0]}, "
                f"expected {dict_size}"
            )
        k = min(top_k, dict_size)
        _, topk_indices = torch.topk(tas.abs(), k)
        trait_count[topk_indices] += 1

    nonspecific_mask = trait_count >= 2

    n_nonspecific = int(nonspecific_mask.sum().item())
    n_traits = len(tas_by_trait)
    logger.info(
        "Cross-trait specificity: %d / %d features appear in top-%d "
        "for 2+ of %d traits",
        n_nonspecific, dict_size, top_k, n_traits,
    )
    return nonspecific_mask

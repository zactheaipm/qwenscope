"""5x5 steering reliability matrix and sub-behavior-level analysis."""

from __future__ import annotations

import logging

import numpy as np

from src.data.contrastive import BehavioralTrait
from src.evaluation.behavioral_metrics import BehavioralScore
from src.evaluation.contamination import SUB_BEHAVIOR_KEYS

logger = logging.getLogger(__name__)


def compute_steering_reliability_matrix(
    baseline_scores: list[BehavioralScore],
    steered_scores: dict[BehavioralTrait, dict[float, list[BehavioralScore]]],
    target_multiplier: float = 5.0,
) -> dict[str, np.ndarray | list[str]]:
    """Compute the 5x5 steering reliability matrix.

    Shows how reliably each trait can be steered independently.

    Args:
        baseline_scores: Unsteered behavioral scores.
        steered_scores: trait -> multiplier -> scores.
        target_multiplier: Which multiplier to evaluate.

    Returns:
        Dict with:
            'matrix': (5, 5) array — rows=steered trait, cols=measured trait
            'trait_names': list of trait names
            'effect_sizes': (5,) array — Cohen's d for diagonal
    """
    trait_names = [t.value for t in BehavioralTrait]
    n = len(trait_names)
    matrix = np.zeros((n, n))

    baseline_trait_scores = _aggregate_trait_scores(baseline_scores)

    effect_sizes = np.zeros(n)

    for i, trait in enumerate(BehavioralTrait):
        if trait not in steered_scores:
            continue
        mult_scores = steered_scores[trait]
        if target_multiplier not in mult_scores:
            continue

        steered_trait_scores = _aggregate_trait_scores(mult_scores[target_multiplier])

        for j, key in enumerate(trait_names):
            baseline_vals = baseline_trait_scores.get(key, [])
            steered_vals = steered_trait_scores.get(key, [])

            if baseline_vals and steered_vals:
                b_arr = np.array(baseline_vals, dtype=float)
                s_arr = np.array(steered_vals, dtype=float)
                n_nan = int(np.isnan(b_arr).sum() + np.isnan(s_arr).sum())
                if n_nan > 0:
                    logger.debug(
                        "NaN values excluded: %d in %s (baseline=%d, steered=%d)",
                        n_nan, key, int(np.isnan(b_arr).sum()), int(np.isnan(s_arr).sum()),
                    )
                mean_diff = float(np.nanmean(s_arr) - np.nanmean(b_arr))
                n1 = int((~np.isnan(b_arr)).sum())
                n2 = int((~np.isnan(s_arr)).sum())
                var1 = float(np.nanvar(b_arr, ddof=1)) if n1 > 1 else 0.0
                var2 = float(np.nanvar(s_arr, ddof=1)) if n2 > 1 else 0.0
                pooled_std = np.sqrt(
                    ((n1 - 1) * var1 + (n2 - 1) * var2) / max(n1 + n2 - 2, 1)
                )
                matrix[i, j] = mean_diff
                if i == j:
                    effect_sizes[i] = mean_diff / max(pooled_std, 1e-8)

    return {
        "matrix": matrix,
        "trait_names": trait_names,
        "effect_sizes": effect_sizes,
    }


def compute_sub_behavior_steering_matrix(
    baseline_scores: list[BehavioralScore],
    steered_scores: dict[BehavioralTrait, dict[float, list[BehavioralScore]]],
    target_multiplier: float = 5.0,
) -> dict[str, np.ndarray | list[str]]:
    """Compute the 5x15 sub-behavior steering matrix.

    Like the 5x5 trait matrix but measures the effect on each of the
    15 individual sub-behaviors. This reveals WHICH sub-behaviors within
    a trait are most affected by steering, and which off-target
    sub-behaviors leak.

    Args:
        baseline_scores: Unsteered behavioral scores.
        steered_scores: trait -> multiplier -> scores.
        target_multiplier: Which multiplier to evaluate.

    Returns:
        Dict with:
            'matrix': (5, 15) array — rows=steered trait, cols=sub-behavior
            'trait_names': row labels
            'sub_behavior_names': column labels
    """
    trait_names = [t.value for t in BehavioralTrait]
    n_traits = len(trait_names)
    n_subs = len(SUB_BEHAVIOR_KEYS)
    matrix = np.zeros((n_traits, n_subs))

    baseline_sub = _aggregate_sub_behavior_scores(baseline_scores)

    for i, trait in enumerate(BehavioralTrait):
        if trait not in steered_scores:
            continue
        mult_scores = steered_scores[trait]
        if target_multiplier not in mult_scores:
            continue

        steered_sub = _aggregate_sub_behavior_scores(mult_scores[target_multiplier])

        for j, key in enumerate(SUB_BEHAVIOR_KEYS):
            base_vals = baseline_sub.get(key, [])
            steer_vals = steered_sub.get(key, [])
            if base_vals and steer_vals:
                matrix[i, j] = float(np.nanmean(steer_vals) - np.nanmean(base_vals))

    return {
        "matrix": matrix,
        "trait_names": trait_names,
        "sub_behavior_names": SUB_BEHAVIOR_KEYS,
    }


def bootstrap_steering_reliability(
    baseline_scores: list[BehavioralScore],
    steered_scores: dict[BehavioralTrait, dict[float, list[BehavioralScore]]],
    target_multiplier: float = 5.0,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict[str, np.ndarray | list[str]]:
    """Compute bootstrap confidence intervals for the steering reliability matrix.

    LLM judge scores are bounded [0, 1] and likely non-normal (e.g.,
    bimodal for deference), making parametric confidence intervals
    unreliable. Bootstrap CIs are distribution-free.

    Args:
        baseline_scores: Unsteered behavioral scores.
        steered_scores: trait -> multiplier -> scores.
        target_multiplier: Which multiplier to evaluate.
        n_bootstrap: Number of bootstrap resamples.
        ci_level: Confidence level (e.g., 0.95 for 95% CI).
        seed: Random seed.

    Returns:
        Dict with:
            'matrix': (5, 5) point estimate
            'trait_names': list of trait names
            'effect_sizes': (5,) Cohen's d for diagonal
            'ci_lower': (5, 5) lower CI bound
            'ci_upper': (5, 5) upper CI bound
            'effect_size_ci_lower': (5,) lower CI for effect sizes
            'effect_size_ci_upper': (5,) upper CI for effect sizes
    """
    rng = np.random.RandomState(seed)
    alpha = 1 - ci_level
    trait_names = [t.value for t in BehavioralTrait]
    n = len(trait_names)

    n_baseline = len(baseline_scores)
    boot_matrices = np.zeros((n_bootstrap, n, n))
    boot_effect_sizes = np.zeros((n_bootstrap, n))

    for b in range(n_bootstrap):
        # Resample baseline
        base_idx = rng.choice(n_baseline, size=n_baseline, replace=True)
        boot_baseline = [baseline_scores[i] for i in base_idx]
        boot_baseline_agg = _aggregate_trait_scores(boot_baseline)

        for i, trait in enumerate(BehavioralTrait):
            if trait not in steered_scores:
                continue
            mult_scores = steered_scores[trait]
            if target_multiplier not in mult_scores:
                continue

            steered_list = mult_scores[target_multiplier]
            n_steered = len(steered_list)
            steered_idx = rng.choice(n_steered, size=n_steered, replace=True)
            boot_steered = [steered_list[j] for j in steered_idx]
            boot_steered_agg = _aggregate_trait_scores(boot_steered)

            for j, key in enumerate(trait_names):
                base_vals = boot_baseline_agg.get(key, [])
                steer_vals = boot_steered_agg.get(key, [])
                if base_vals and steer_vals:
                    diff = float(np.nanmean(steer_vals) - np.nanmean(base_vals))
                    boot_matrices[b, i, j] = diff
                    if i == j:
                        b_arr = np.array(base_vals, dtype=float)
                        s_arr = np.array(steer_vals, dtype=float)
                        n1 = int((~np.isnan(b_arr)).sum())
                        n2 = int((~np.isnan(s_arr)).sum())
                        var1 = float(np.nanvar(b_arr, ddof=1)) if n1 > 1 else 0.0
                        var2 = float(np.nanvar(s_arr, ddof=1)) if n2 > 1 else 0.0
                        pooled = np.sqrt(
                            ((n1 - 1) * var1 + (n2 - 1) * var2) / max(n1 + n2 - 2, 1)
                        )
                        boot_effect_sizes[b, i] = diff / max(pooled, 1e-8)

    # Point estimates
    point = compute_steering_reliability_matrix(
        baseline_scores, steered_scores, target_multiplier
    )

    ci_lower = np.percentile(boot_matrices, 100 * alpha / 2, axis=0)
    ci_upper = np.percentile(boot_matrices, 100 * (1 - alpha / 2), axis=0)
    es_ci_lower = np.percentile(boot_effect_sizes, 100 * alpha / 2, axis=0)
    es_ci_upper = np.percentile(boot_effect_sizes, 100 * (1 - alpha / 2), axis=0)

    logger.info(
        "Bootstrap steering CIs (n=%d, %.0f%%): mean matrix CI width=%.4f",
        n_bootstrap,
        ci_level * 100,
        np.nanmean(ci_upper - ci_lower),
    )

    return {
        "matrix": point["matrix"],
        "trait_names": point["trait_names"],
        "effect_sizes": point["effect_sizes"],
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "effect_size_ci_lower": es_ci_lower,
        "effect_size_ci_upper": es_ci_upper,
    }


def compute_probability_of_superiority(
    baseline_scores: list[BehavioralScore],
    steered_scores: dict[BehavioralTrait, dict[float, list[BehavioralScore]]],
    target_multiplier: float = 5.0,
) -> dict[str, np.ndarray | list[str]]:
    """Compute probability-of-superiority matrix for steering effects.

    For each (steered_trait, measured_trait) cell, computes:

        P(steered > baseline) = (# concordant pairs) / (# total pairs)

    using all pairwise comparisons between steered and baseline observations.
    Ties contribute 0.5 to the count. This is equivalent to the common
    language effect size (CLES) / area under the ROC curve (AUC) and is a
    non-parametric effect size measure that makes no distributional assumptions.

    Values > 0.5 indicate the steered condition tends to score higher;
    values < 0.5 indicate it scores lower; 0.5 means no effect.

    Args:
        baseline_scores: Unsteered behavioral scores.
        steered_scores: trait -> multiplier -> scores.
        target_multiplier: Which multiplier to evaluate.

    Returns:
        Dict with:
            'matrix': (5, 5) array -- P(steered > baseline) for each
                (steered_trait, measured_trait) cell.
            'trait_names': list of trait names (row and column labels).
    """
    trait_names = [t.value for t in BehavioralTrait]
    n = len(trait_names)
    matrix = np.full((n, n), 0.5)  # default: no effect

    baseline_agg = _aggregate_trait_scores(baseline_scores)

    for i, trait in enumerate(BehavioralTrait):
        if trait not in steered_scores:
            continue
        mult_scores = steered_scores[trait]
        if target_multiplier not in mult_scores:
            continue

        steered_agg = _aggregate_trait_scores(mult_scores[target_multiplier])

        for j, key in enumerate(trait_names):
            base_vals = baseline_agg.get(key, [])
            steer_vals = steered_agg.get(key, [])

            if not base_vals or not steer_vals:
                continue

            matrix[i, j] = _pairwise_prob_superiority(
                np.asarray(steer_vals), np.asarray(base_vals)
            )

    logger.info(
        "Probability of superiority matrix (multiplier=%.1f): "
        "diagonal mean=%.3f, off-diagonal mean=%.3f",
        target_multiplier,
        np.nanmean(np.diag(matrix)),
        np.nanmean(matrix[~np.eye(n, dtype=bool)]),
    )

    return {
        "matrix": matrix,
        "trait_names": trait_names,
    }


def compute_sub_behavior_probability_of_superiority(
    baseline_scores: list[BehavioralScore],
    steered_scores: dict[BehavioralTrait, dict[float, list[BehavioralScore]]],
    target_multiplier: float = 5.0,
) -> dict[str, np.ndarray | list[str]]:
    """Compute probability-of-superiority at the sub-behavior level.

    Same as compute_probability_of_superiority but for all 15 sub-behaviors.
    Returns a (5, 15) matrix: rows = steered traits, cols = sub-behaviors.

    Args:
        baseline_scores: Unsteered behavioral scores.
        steered_scores: trait -> multiplier -> scores.
        target_multiplier: Which multiplier to evaluate.

    Returns:
        Dict with:
            'matrix': (5, 15) P(steered > baseline) per cell.
            'trait_names': row labels.
            'sub_behavior_names': column labels.
    """
    trait_names = [t.value for t in BehavioralTrait]
    n_traits = len(trait_names)
    n_subs = len(SUB_BEHAVIOR_KEYS)
    matrix = np.full((n_traits, n_subs), 0.5)

    baseline_sub = _aggregate_sub_behavior_scores(baseline_scores)

    for i, trait in enumerate(BehavioralTrait):
        if trait not in steered_scores:
            continue
        mult_scores = steered_scores[trait]
        if target_multiplier not in mult_scores:
            continue

        steered_sub = _aggregate_sub_behavior_scores(mult_scores[target_multiplier])

        for j, key in enumerate(SUB_BEHAVIOR_KEYS):
            base_vals = baseline_sub.get(key, [])
            steer_vals = steered_sub.get(key, [])
            if base_vals and steer_vals:
                matrix[i, j] = _pairwise_prob_superiority(
                    np.asarray(steer_vals), np.asarray(base_vals)
                )

    return {
        "matrix": matrix,
        "trait_names": trait_names,
        "sub_behavior_names": SUB_BEHAVIOR_KEYS,
    }


def _pairwise_prob_superiority(
    steered: np.ndarray, baseline: np.ndarray
) -> float:
    """Compute P(steered > baseline) over all pairwise comparisons.

    Args:
        steered: 1-D array of steered scores.
        baseline: 1-D array of baseline scores.

    Returns:
        Probability of superiority in [0, 1].
    """
    diff = steered[:, np.newaxis] - baseline[np.newaxis, :]
    n_total = diff.size
    concordant = float(np.sum(diff > 0))
    tied = float(np.sum(diff == 0))
    return (concordant + 0.5 * tied) / n_total


def _aggregate_trait_scores(
    scores: list[BehavioralScore],
) -> dict[str, list[float]]:
    """Aggregate composite trait scores into lists for statistical analysis.

    Args:
        scores: List of BehavioralScore objects.

    Returns:
        Dict mapping trait name to list of scores.
    """
    result: dict[str, list[float]] = {}
    for score in scores:
        for key, val in score.trait_scores().items():
            result.setdefault(key, []).append(val)
    return result


def _aggregate_sub_behavior_scores(
    scores: list[BehavioralScore],
) -> dict[str, list[float]]:
    """Aggregate sub-behavior scores into lists for statistical analysis.

    Args:
        scores: List of BehavioralScore objects.

    Returns:
        Dict mapping 'trait.sub_behavior' to list of scores.
    """
    result: dict[str, list[float]] = {}
    for score in scores:
        for key, val in score.flat_sub_behavior_scores().items():
            result.setdefault(key, []).append(val)
    return result

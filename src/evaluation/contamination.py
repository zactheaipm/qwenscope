"""Cross-trait contamination matrix computation.

Provides both trait-level (5x5) and sub-behavior-level (15x15) contamination
matrices. The sub-behavior matrix is the primary diagnostic: it reveals
WHICH specific sub-behaviors are affected by steering, rather than aggregating
into composite trait scores that mask fine-grained effects.
"""

from __future__ import annotations

import logging

import numpy as np

from src.data.contrastive import BehavioralTrait
from src.evaluation.behavioral_metrics import BehavioralScore, SUB_BEHAVIOR_KEYS

logger = logging.getLogger(__name__)

# Canonical trait order for the 5x5 matrix
TRAIT_ORDER = [
    BehavioralTrait.AUTONOMY,
    BehavioralTrait.TOOL_USE,
    BehavioralTrait.PERSISTENCE,
    BehavioralTrait.RISK_CALIBRATION,
    BehavioralTrait.DEFERENCE,
]

TRAIT_SCORE_KEYS = [
    "autonomy",
    "tool_use_eagerness",
    "persistence",
    "risk_calibration",
    "deference",
]


def compute_contamination_matrix(
    baseline_scores: list[BehavioralScore],
    steered_scores: dict[BehavioralTrait, list[BehavioralScore]],
    use_cohens_d: bool = True,
) -> np.ndarray:
    """Compute the 5x5 cross-trait contamination matrix.

    Matrix[i][j] = effect of steering trait i on trait j's composite score.

    When ``use_cohens_d`` is True (default), each cell is Cohen's d
    (pooled-SD-normalised mean difference), making effect sizes comparable
    across traits with different variances. When False, cells are raw mean
    differences (legacy behaviour).

    Diagonal = intended effect (should be large).
    Off-diagonal = contamination (should be small).

    Args:
        baseline_scores: Behavioral scores from unsteered model.
        steered_scores: Dict mapping each steered trait to its behavioral scores.
        use_cohens_d: If True, report Cohen's d; if False, raw mean difference.

    Returns:
        (5, 5) numpy array where rows = steered trait, cols = measured trait.
    """
    from src.analysis.effect_sizes import cohens_d

    n_traits = len(TRAIT_ORDER)
    matrix = np.full((n_traits, n_traits), np.nan)

    # Pre-collect baseline arrays per trait key
    baseline_arrays: dict[str, np.ndarray] = {}
    for key in TRAIT_SCORE_KEYS:
        vals = [s.trait_scores()[key] for s in baseline_scores]
        baseline_arrays[key] = np.array(vals, dtype=float)

    for i, steered_trait in enumerate(TRAIT_ORDER):
        if steered_trait not in steered_scores:
            continue

        steered_list = steered_scores[steered_trait]
        for j, measured_key in enumerate(TRAIT_SCORE_KEYS):
            b_arr = baseline_arrays[measured_key]
            s_arr = np.array(
                [s.trait_scores()[measured_key] for s in steered_list], dtype=float
            )
            # Remove NaNs
            b_valid = b_arr[~np.isnan(b_arr)]
            s_valid = s_arr[~np.isnan(s_arr)]

            if len(b_valid) < 2 or len(s_valid) < 2:
                matrix[i][j] = np.nan
            elif use_cohens_d:
                matrix[i][j] = cohens_d(s_valid, b_valid)
            else:
                matrix[i][j] = float(np.mean(s_valid) - np.mean(b_valid))

    logger.info(
        "Contamination matrix (Cohen's d=%s): diagonal mean=%.3f, off-diagonal mean=%.3f",
        use_cohens_d,
        np.nanmean(np.abs(np.diag(matrix))),
        np.nanmean(np.abs(matrix[~np.eye(n_traits, dtype=bool)])),
    )

    return matrix


def compute_sub_behavior_contamination_matrix(
    baseline_scores: list[BehavioralScore],
    steered_scores: dict[BehavioralTrait, list[BehavioralScore]],
) -> dict[str, np.ndarray | list[str]]:
    """Compute the 15x15 sub-behavior contamination matrix.

    This is the primary diagnostic matrix. Each cell [i][j] shows how
    steering trait i changes sub-behavior j. This reveals:
    - Which specific sub-behaviors within the target trait are affected
    - Which off-target sub-behaviors leak (and in which traits)
    - Whether contamination is concentrated in specific sub-behaviors
      or spread across all sub-behaviors of a non-target trait

    Rows are the 5 steered traits (one row per trait, repeated for each
    of that trait's sub-behaviors — but since steering targets a trait,
    not a sub-behavior, the row dimension is 5, not 15).

    Args:
        baseline_scores: Behavioral scores from unsteered model.
        steered_scores: Dict mapping each steered trait to its behavioral scores.

    Returns:
        Dict with:
            'matrix': (5, 15) array — rows=steered trait, cols=measured sub-behavior
            'trait_names': row labels (5 trait names)
            'sub_behavior_names': column labels (15 sub-behavior keys)
    """
    n_traits = len(TRAIT_ORDER)
    n_subs = len(SUB_BEHAVIOR_KEYS)
    matrix = np.zeros((n_traits, n_subs))

    baseline_sub_means = _compute_sub_behavior_means(baseline_scores)

    for i, steered_trait in enumerate(TRAIT_ORDER):
        if steered_trait not in steered_scores:
            continue

        steered_sub_means = _compute_sub_behavior_means(steered_scores[steered_trait])

        for j, sub_key in enumerate(SUB_BEHAVIOR_KEYS):
            s_val = steered_sub_means.get(sub_key, float("nan"))
            b_val = baseline_sub_means.get(sub_key, float("nan"))
            matrix[i][j] = s_val - b_val

    # Log which off-target sub-behaviors have the largest absolute change
    for i, steered_trait in enumerate(TRAIT_ORDER):
        trait_prefix = TRAIT_SCORE_KEYS[i] + "."
        off_target_indices = [
            j for j, k in enumerate(SUB_BEHAVIOR_KEYS) if not k.startswith(trait_prefix)
        ]
        if off_target_indices:
            off_target_vals = matrix[i, off_target_indices]
            worst_idx = off_target_indices[int(np.argmax(np.abs(off_target_vals)))]
            logger.info(
                "Steering %s: worst off-target sub-behavior = %s (delta=%.3f)",
                steered_trait.value,
                SUB_BEHAVIOR_KEYS[worst_idx],
                matrix[i, worst_idx],
            )

    return {
        "matrix": matrix,
        "trait_names": [t.value for t in TRAIT_ORDER],
        "sub_behavior_names": SUB_BEHAVIOR_KEYS,
    }


def _compute_trait_means(scores: list[BehavioralScore]) -> dict[str, float]:
    """Compute mean composite trait scores from a list of behavioral scores.

    Uses np.nanmean to exclude NaN values (unobservable sub-behaviors)
    from the average. Returns NaN for empty inputs to avoid creating
    misleading contamination deltas.

    Args:
        scores: List of BehavioralScore objects.

    Returns:
        Dict mapping trait name to mean score. NaN for empty inputs.
    """
    if not scores:
        return {key: float("nan") for key in TRAIT_SCORE_KEYS}

    means = {}
    for key in TRAIT_SCORE_KEYS:
        trait_scores = [s.trait_scores()[key] for s in scores]
        means[key] = float(np.nanmean(trait_scores))

    return means


def _compute_sub_behavior_means(scores: list[BehavioralScore]) -> dict[str, float]:
    """Compute mean sub-behavior scores from a list of behavioral scores.

    Uses np.nanmean to exclude NaN values (unobservable sub-behaviors)
    from the average. Returns NaN for empty inputs.

    Args:
        scores: List of BehavioralScore objects.

    Returns:
        Dict mapping sub-behavior key ('trait.sub') to mean score.
    """
    if not scores:
        return {key: float("nan") for key in SUB_BEHAVIOR_KEYS}

    means = {}
    for key in SUB_BEHAVIOR_KEYS:
        vals = [s.flat_sub_behavior_scores()[key] for s in scores]
        means[key] = float(np.nanmean(vals))

    return means


def compute_baseline_correlation_matrix(
    baseline_scores: list[BehavioralScore],
) -> np.ndarray:
    """Compute pairwise Pearson correlation between traits on unsteered runs.

    If traits are naturally correlated at baseline (e.g., autonomy and risk
    always co-vary), off-diagonal contamination in the steering matrix may
    reflect pre-existing structure rather than steering leakage.

    Should be computed on 200+ unsteered trajectories for stability.

    Args:
        baseline_scores: Behavioral scores from unsteered model runs.

    Returns:
        (5, 5) Pearson correlation matrix.
    """
    n_traits = len(TRAIT_SCORE_KEYS)
    n_scores = len(baseline_scores)

    if n_scores < 10:
        logger.warning(
            "Only %d baseline scores (recommend 200+); correlations unreliable",
            n_scores,
        )

    # Build (n_scores, n_traits) matrix.
    # Use NaN (not 0.0) for missing values so they don't bias correlations.
    score_matrix = np.full((n_scores, n_traits), np.nan)
    for i, score in enumerate(baseline_scores):
        trait_dict = score.trait_scores()
        for j, key in enumerate(TRAIT_SCORE_KEYS):
            val = trait_dict.get(key)
            if val is not None:
                score_matrix[i, j] = val

    # Drop rows that are entirely NaN before computing correlations.
    valid_rows = ~np.all(np.isnan(score_matrix), axis=1)
    score_matrix = score_matrix[valid_rows]

    # Pairwise Pearson correlation using pandas (NaN-aware) to handle
    # remaining per-cell NaN without biasing toward zero.
    import pandas as pd
    corr = pd.DataFrame(score_matrix).corr().values  # (5, 5)

    logger.info(
        "Baseline correlations (n=%d): mean |r|=%.3f, max |r|=%.3f",
        n_scores,
        np.nanmean(np.abs(corr[~np.eye(n_traits, dtype=bool)])),
        np.nanmax(np.abs(corr[~np.eye(n_traits, dtype=bool)])),
    )

    return corr


def compute_baseline_sub_behavior_correlation_matrix(
    baseline_scores: list[BehavioralScore],
) -> np.ndarray:
    """Compute pairwise Pearson correlation between all 15 sub-behaviors.

    This matrix reveals the empirical correlation structure at the
    sub-behavior level. Within-trait sub-behaviors should correlate
    (they measure facets of the same construct). Cross-trait correlations
    reveal which specific sub-behaviors naturally co-vary — these are
    the pairs most vulnerable to contamination in the steering matrix.

    Args:
        baseline_scores: Behavioral scores from unsteered model runs.

    Returns:
        (15, 15) Pearson correlation matrix.
    """
    n_subs = len(SUB_BEHAVIOR_KEYS)
    n_scores = len(baseline_scores)

    if n_scores < 30:
        logger.warning(
            "Only %d baseline scores for 15-dim correlation (recommend 200+); "
            "correlations unreliable",
            n_scores,
        )

    # Use NaN (not 0.0) for missing values so they don't bias correlations.
    # 0.0 is a valid score (lowest observable), not "missing data".
    score_matrix = np.full((n_scores, n_subs), np.nan)
    for i, score in enumerate(baseline_scores):
        flat = score.flat_sub_behavior_scores()
        for j, key in enumerate(SUB_BEHAVIOR_KEYS):
            val = flat.get(key)
            if val is not None:
                score_matrix[i, j] = val

    # Drop rows that are entirely NaN before computing correlations.
    valid_rows = ~np.all(np.isnan(score_matrix), axis=1)
    score_matrix = score_matrix[valid_rows]

    # Use pandas for NaN-aware pairwise Pearson correlation, consistent
    # with the trait-level compute_baseline_correlation_matrix.
    import pandas as pd
    corr = pd.DataFrame(score_matrix).corr().values  # (15, 15)

    # Summarize within-trait vs cross-trait correlations
    within_mask = np.zeros((n_subs, n_subs), dtype=bool)
    for start in range(0, n_subs, 3):
        within_mask[start:start + 3, start:start + 3] = True
    np.fill_diagonal(within_mask, False)
    cross_mask = ~within_mask & ~np.eye(n_subs, dtype=bool)

    logger.info(
        "Sub-behavior baseline correlations (n=%d): "
        "within-trait mean |r|=%.3f, cross-trait mean |r|=%.3f",
        n_scores,
        np.nanmean(np.abs(corr[within_mask])),
        np.nanmean(np.abs(corr[cross_mask])),
    )

    return corr


def bootstrap_contamination_ci(
    baseline_scores: list[BehavioralScore],
    steered_scores: dict[BehavioralTrait, list[BehavioralScore]],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Compute bootstrap confidence intervals for the contamination matrix.

    LLM judge scores are bounded [0, 1] and likely non-normal, making
    parametric CIs unreliable. Bootstrap CIs are distribution-free.

    Args:
        baseline_scores: Unsteered behavioral scores.
        steered_scores: Dict mapping steered trait to its behavioral scores.
        n_bootstrap: Number of bootstrap resamples.
        ci_level: Confidence level (e.g., 0.95 for 95% CI).
        seed: Random seed.

    Returns:
        Dict with:
            "point_estimate": (5, 5) contamination matrix
            "ci_lower": (5, 5) lower bound
            "ci_upper": (5, 5) upper bound
    """
    rng = np.random.RandomState(seed)
    alpha = 1 - ci_level
    n_traits = len(TRAIT_ORDER)
    bootstrap_matrices = np.zeros((n_bootstrap, n_traits, n_traits))

    n_baseline = len(baseline_scores)

    for b in range(n_bootstrap):
        # Resample baseline
        baseline_idx = rng.choice(n_baseline, size=n_baseline, replace=True)
        boot_baseline = [baseline_scores[i] for i in baseline_idx]

        # Resample steered (independently per trait)
        boot_steered: dict[BehavioralTrait, list[BehavioralScore]] = {}
        for trait, scores in steered_scores.items():
            n_s = len(scores)
            steered_idx = rng.choice(n_s, size=n_s, replace=True)
            boot_steered[trait] = [scores[i] for i in steered_idx]

        bootstrap_matrices[b] = compute_contamination_matrix(
            boot_baseline, boot_steered
        )

    point_estimate = compute_contamination_matrix(baseline_scores, steered_scores)
    ci_lower = np.percentile(bootstrap_matrices, 100 * alpha / 2, axis=0)
    ci_upper = np.percentile(bootstrap_matrices, 100 * (1 - alpha / 2), axis=0)

    logger.info(
        "Bootstrap CIs (n=%d, %.0f%%): mean CI width=%.4f",
        n_bootstrap,
        ci_level * 100,
        np.nanmean(ci_upper - ci_lower),
    )

    return {
        "point_estimate": point_estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def contamination_summary(matrix: np.ndarray) -> dict[str, float]:
    """Compute summary statistics for the contamination matrix.

    Args:
        matrix: (5, 5) contamination matrix.

    Returns:
        Dict with summary metrics.
    """
    n = matrix.shape[0]
    diagonal = np.diag(matrix)
    off_diag_mask = ~np.eye(n, dtype=bool)

    # Compute per-row off-diagonal contamination totals using explicit loop
    # (avoids fragile reshape that depends on numpy boolean indexing order)
    row_contamination = np.zeros(n)
    for i in range(n):
        row_contamination[i] = sum(
            abs(matrix[i, j]) for j in range(n) if j != i
        )

    return {
        "mean_intended_effect": float(np.nanmean(np.abs(diagonal))),
        "mean_contamination": float(np.nanmean(np.abs(matrix[off_diag_mask]))),
        "max_contamination": float(np.nanmax(np.abs(matrix[off_diag_mask]))),
        "selectivity_ratio": float(
            np.nanmean(np.abs(diagonal)) / max(np.nanmean(np.abs(matrix[off_diag_mask])), 1e-8)
        ),
        "cleanest_trait": TRAIT_SCORE_KEYS[int(np.argmin(row_contamination))],
        "most_contaminating_trait": TRAIT_SCORE_KEYS[int(np.argmax(row_contamination))],
    }

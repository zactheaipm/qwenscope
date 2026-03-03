"""Effect size computation utilities.

Provides Cohen's d, probability of superiority, and bootstrap confidence
intervals — the core statistical tools referenced in the pre-registration
(docs/pre_registration.md) for evaluating steering effects.

These must exist as reusable functions before experiments run to prevent
ad-hoc implementations during the analysis phase.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

logger = logging.getLogger(__name__)


def cohens_d(group1: ArrayLike, group2: ArrayLike) -> float:
    """Compute Cohen's d (standardized mean difference) with pooled SD.

    Uses the pooled standard deviation as the denominator, which assumes
    equal population variances. This is the standard formulation used
    in the pre-registration for H1 and H4.

    Args:
        group1: First group (e.g., steered scores). Shape (n1,).
        group2: Second group (e.g., baseline scores). Shape (n2,).

    Returns:
        Cohen's d. Positive when group1 > group2.
    """
    g1 = np.asarray(group1, dtype=np.float64)
    g2 = np.asarray(group2, dtype=np.float64)
    n1, n2 = len(g1), len(g2)

    if n1 < 2 or n2 < 2:
        logger.warning(
            "Cohen's d requires n >= 2 per group (got n1=%d, n2=%d)", n1, n2
        )
        return 0.0

    var1 = g1.var(ddof=1)
    var2 = g2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    return float((g1.mean() - g2.mean()) / max(pooled_std, 1e-8))


def cohens_d_paired(pre: ArrayLike, post: ArrayLike) -> float:
    """Compute Cohen's d for paired (within-subject) designs.

    Uses the standard deviation of the *differences* as the denominator,
    which is appropriate for paired comparisons (e.g., same scenarios
    before and after steering).

    Args:
        pre: Pre-intervention scores. Shape (n,).
        post: Post-intervention scores. Shape (n,).

    Returns:
        Cohen's d_z. Positive when post > pre.
    """
    pre_arr = np.asarray(pre, dtype=np.float64)
    post_arr = np.asarray(post, dtype=np.float64)

    if len(pre_arr) != len(post_arr):
        raise ValueError(
            f"Paired design requires equal lengths: got {len(pre_arr)} vs {len(post_arr)}"
        )
    if len(pre_arr) < 2:
        logger.warning("Cohen's d_paired requires n >= 2 (got n=%d)", len(pre_arr))
        return 0.0

    diffs = post_arr - pre_arr
    return float(diffs.mean() / max(diffs.std(ddof=1), 1e-8))


def probability_of_superiority(group1: ArrayLike, group2: ArrayLike) -> float:
    """Compute the probability of superiority P(X1 > X2).

    Non-parametric effect size: the probability that a randomly drawn
    observation from group1 exceeds a randomly drawn observation from
    group2. Also known as the common language effect size.

    - P = 0.5 → no difference
    - P = 0.65 → corresponds roughly to Cohen's d ≈ 0.55 under normality
    - P = 1.0 → every group1 observation exceeds every group2 observation

    Pre-registration requires P >= 0.65 as a supplement to Cohen's d.

    Args:
        group1: First group (e.g., steered). Shape (n1,).
        group2: Second group (e.g., baseline). Shape (n2,).

    Returns:
        P(group1 > group2), in [0, 1].
    """
    g1 = np.asarray(group1, dtype=np.float64)
    g2 = np.asarray(group2, dtype=np.float64)

    if len(g1) == 0 or len(g2) == 0:
        logger.warning("Empty group(s) for probability_of_superiority")
        return 0.5

    # Count pairs where g1 > g2, with ties counted as 0.5
    wins = 0.0
    total = len(g1) * len(g2)
    for x in g1:
        wins += np.sum(x > g2) + 0.5 * np.sum(x == g2)

    return float(wins / total)


def bootstrap_ci(
    data: ArrayLike,
    statistic: str = "mean",
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Compute bootstrap confidence interval for a statistic.

    Distribution-free CI using the percentile method. Appropriate for
    LLM judge scores which are bounded [0, 1] and likely non-normal.

    Args:
        data: 1-D array of observations.
        statistic: One of 'mean', 'median'.
        n_bootstrap: Number of bootstrap resamples.
        ci_level: Confidence level (e.g., 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Dict with 'point_estimate', 'ci_lower', 'ci_upper'.
    """
    arr = np.asarray(data, dtype=np.float64)
    rng = np.random.RandomState(seed)

    if len(arr) == 0:
        return {"point_estimate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    stat_fn = np.mean if statistic == "mean" else np.median
    point = float(stat_fn(arr))

    bootstrap_stats = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        bootstrap_stats[b] = stat_fn(sample)

    alpha = 1 - ci_level
    ci_lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))

    return {
        "point_estimate": point,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def bootstrap_ci_difference(
    group1: ArrayLike,
    group2: ArrayLike,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Compute bootstrap CI for the difference in means (group1 - group2).

    Resamples each group independently, then computes the difference.
    Used for confidence intervals on contamination matrix entries and
    steering effect sizes.

    Args:
        group1: First group (e.g., steered scores).
        group2: Second group (e.g., baseline scores).
        n_bootstrap: Number of bootstrap resamples.
        ci_level: Confidence level.
        seed: Random seed.

    Returns:
        Dict with 'point_estimate', 'ci_lower', 'ci_upper'.
    """
    g1 = np.asarray(group1, dtype=np.float64)
    g2 = np.asarray(group2, dtype=np.float64)
    rng = np.random.RandomState(seed)

    point = float(g1.mean() - g2.mean())

    diffs = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        s1 = rng.choice(g1, size=len(g1), replace=True)
        s2 = rng.choice(g2, size=len(g2), replace=True)
        diffs[b] = s1.mean() - s2.mean()

    alpha = 1 - ci_level
    ci_lower = float(np.percentile(diffs, 100 * alpha / 2))
    ci_upper = float(np.percentile(diffs, 100 * (1 - alpha / 2)))

    return {
        "point_estimate": point,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def wilcoxon_signed_rank(
    x: ArrayLike,
    y: ArrayLike,
) -> dict[str, float]:
    """Compute Wilcoxon signed-rank test for paired samples.

    Non-parametric test appropriate for paired comparisons where the
    difference distribution may not be normal. Used for H3 (DeltaNet vs
    attention paired comparison).

    Args:
        x: First condition scores. Shape (n,).
        y: Second condition scores. Shape (n,).

    Returns:
        Dict with 'statistic', 'p_value', 'n' (number of non-zero diffs).
    """
    from scipy.stats import wilcoxon

    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)

    if len(x_arr) != len(y_arr):
        raise ValueError(
            f"Paired test requires equal lengths: got {len(x_arr)} vs {len(y_arr)}"
        )

    # Remove zero differences (ties)
    diffs = x_arr - y_arr
    nonzero = np.sum(diffs != 0)

    if nonzero < 2:
        logger.warning(
            "Wilcoxon test needs >= 2 non-zero differences (got %d)", nonzero
        )
        return {"statistic": 0.0, "p_value": 1.0, "n": int(nonzero)}

    stat, p_value = wilcoxon(x_arr, y_arr, alternative="two-sided")

    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "n": int(nonzero),
    }


def compute_selectivity_per_trait(
    contamination_matrix: np.ndarray,
    trait_names: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute per-trait selectivity metrics from contamination matrix.

    For each trait, reports:
    - on_diagonal: the intended effect (absolute value)
    - max_off_diagonal: the largest off-target effect (absolute value)
    - selectivity_ratio: on_diagonal / max_off_diagonal
    - passes_threshold: whether ratio >= 1.5

    This supplements the aggregate selectivity ratio (H2) with per-trait
    detail.

    Args:
        contamination_matrix: (n, n) matrix.
        trait_names: Optional names for reporting.

    Returns:
        Dict mapping trait index/name to selectivity metrics.
    """
    n = contamination_matrix.shape[0]
    if trait_names is None:
        trait_names = [f"trait_{i}" for i in range(n)]

    results: dict[str, dict[str, float]] = {}
    for i in range(n):
        on_diag = abs(contamination_matrix[i, i])
        off_diag = [
            abs(contamination_matrix[i, j]) for j in range(n) if j != i
        ]
        max_off = max(off_diag) if off_diag else 0.0

        ratio = on_diag / max(max_off, 1e-8)
        results[trait_names[i]] = {
            "on_diagonal": on_diag,
            "max_off_diagonal": max_off,
            "selectivity_ratio": ratio,
            "passes_threshold": ratio >= 1.5,
        }

    return results

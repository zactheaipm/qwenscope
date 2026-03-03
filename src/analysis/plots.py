"""All matplotlib/plotly figure generation.

Each figure function:
- Accepts data as typed Pydantic models or numpy arrays
- Saves both PNG (300dpi) and SVG to data/results/figures/
- Returns the matplotlib figure object for notebook display
- Uses a consistent color scheme
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

# Consistent color scheme
COLORS = {
    "deltanet": "#4A90D9",  # Blue
    "attention": "#E07B39",  # Orange
}

# Colorblind-safe trait palette
TRAIT_COLORS = {
    "autonomy": "#E69F00",
    "tool_use_eagerness": "#56B4E9",
    "persistence": "#009E73",
    "risk_calibration": "#F0E442",
    "deference": "#CC79A7",
}

FIGURES_DIR = Path("data/results/figures")


def _save_figure(fig: plt.Figure, name: str, output_dir: Path | None = None) -> None:
    """Save figure in PNG and SVG formats.

    Args:
        fig: Matplotlib figure.
        name: Figure name (without extension).
        output_dir: Output directory. Defaults to FIGURES_DIR.
    """
    if output_dir is None:
        output_dir = FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.svg", bbox_inches="tight")
    logger.info("Saved figure: %s (.png + .svg)", name)


def plot_sae_quality_comparison(
    sae_names: list[str],
    mse_values: list[float],
    explained_variance: list[float],
    layer_types: list[str],
    output_dir: Path | None = None,
) -> plt.Figure:
    """SAE reconstruction quality comparison bar chart.

    Args:
        sae_names: Names of the 6 SAEs.
        mse_values: MSE per SAE.
        explained_variance: Explained variance per SAE.
        layer_types: "deltanet" or "attention" per SAE.
        output_dir: Optional output directory.

    Returns:
        Matplotlib figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = [COLORS[lt] for lt in layer_types]

    ax1.bar(sae_names, mse_values, color=colors)
    ax1.set_ylabel("MSE")
    ax1.set_title("Reconstruction MSE by SAE")
    ax1.tick_params(axis="x", rotation=45)

    ax2.bar(sae_names, explained_variance, color=colors)
    ax2.set_ylabel("Explained Variance")
    ax2.set_title("Explained Variance by SAE")
    ax2.tick_params(axis="x", rotation=45)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["deltanet"], label="DeltaNet"),
        Patch(facecolor=COLORS["attention"], label="Attention"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    _save_figure(fig, "sae_quality_comparison", output_dir)
    return fig


def plot_tas_distributions(
    trait_name: str,
    sae_tas_data: dict[str, np.ndarray],
    sae_types: dict[str, str],
    output_dir: Path | None = None,
) -> plt.Figure:
    """TAS distribution violin plots for one trait across all 6 SAEs.

    Args:
        trait_name: Name of the behavioral trait.
        sae_tas_data: sae_id → array of TAS values.
        sae_types: sae_id → "deltanet" or "attention".
        output_dir: Optional output directory.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    positions = list(range(len(sae_tas_data)))
    labels = list(sae_tas_data.keys())
    data = list(sae_tas_data.values())

    parts = ax.violinplot(data, positions=positions, showmeans=True, showextrema=True)

    # Color by layer type
    for i, (pc, label) in enumerate(zip(parts["bodies"], labels)):
        color = COLORS[sae_types[label]]
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("|TAS|")
    ax.set_title(f"Trait Association Score Distribution: {trait_name}")

    _save_figure(fig, f"tas_distribution_{trait_name}", output_dir)
    return fig


def plot_architecture_heatmap(
    matrix: np.ndarray,
    trait_names: list[str],
    sae_names: list[str],
    output_dir: Path | None = None,
) -> plt.Figure:
    """Architecture comparison heatmap: traits × SAEs.

    Args:
        matrix: (n_traits, n_saes) array of max |TAS| values.
        trait_names: List of trait names for y-axis.
        sae_names: List of SAE names for x-axis.
        output_dir: Optional output directory.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(sae_names)))
    ax.set_xticklabels(sae_names, rotation=45, ha="right")
    ax.set_yticks(range(len(trait_names)))
    ax.set_yticklabels(trait_names)
    ax.set_title("Behavioral Trait Map: Max |TAS| by Trait × SAE")

    # Annotate cells
    for i in range(len(trait_names)):
        for j in range(len(sae_names)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="|TAS|")
    fig.tight_layout()

    _save_figure(fig, "architecture_heatmap", output_dir)
    return fig


def plot_dose_response_curves(
    trait_name: str,
    multipliers: list[float],
    curves: dict[str, list[float]],
    output_dir: Path | None = None,
) -> plt.Figure:
    """Dose-response curves for steering.

    Args:
        trait_name: Name of the trait being steered.
        multipliers: X-axis values.
        curves: sae_id → list of behavioral scores (one per multiplier).
        output_dir: Optional output directory.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for sae_id, scores in curves.items():
        ax.plot(multipliers, scores, "o-", label=sae_id, linewidth=2)

    ax.set_xlabel("Steering Multiplier")
    ax.set_ylabel("Behavioral Score")
    ax.set_title(f"Dose-Response: {trait_name}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    _save_figure(fig, f"dose_response_{trait_name}", output_dir)
    return fig


def plot_contamination_matrix(
    matrix: np.ndarray,
    trait_names: list[str],
    output_dir: Path | None = None,
) -> plt.Figure:
    """5x5 cross-trait contamination heatmap.

    Args:
        matrix: (5, 5) contamination matrix.
        trait_names: List of trait names.
        output_dir: Optional output directory.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    # Use diverging colormap centered at 0
    abs_max = max(abs(matrix.min()), abs(matrix.max()), 0.1)
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max, aspect="equal")

    ax.set_xticks(range(len(trait_names)))
    ax.set_xticklabels(trait_names, rotation=45, ha="right")
    ax.set_yticks(range(len(trait_names)))
    ax.set_yticklabels(trait_names)
    ax.set_xlabel("Measured Trait")
    ax.set_ylabel("Steered Trait")
    ax.set_title("Cross-Trait Contamination Matrix")

    # Annotate and highlight diagonal
    for i in range(len(trait_names)):
        for j in range(len(trait_names)):
            color = "white" if abs(matrix[i, j]) > abs_max * 0.5 else "black"
            weight = "bold" if i == j else "normal"
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight=weight)

    fig.colorbar(im, ax=ax, label="Score Change")
    fig.tight_layout()

    _save_figure(fig, "contamination_matrix", output_dir)
    return fig


def plot_layer_type_steering_comparison(
    trait_names: list[str],
    deltanet_scores: list[float],
    attention_scores: list[float],
    combined_scores: list[float],
    output_dir: Path | None = None,
) -> plt.Figure:
    """Grouped bar chart: DeltaNet vs attention vs combined steering.

    Args:
        trait_names: Trait names for x-axis.
        deltanet_scores: Scores with DeltaNet-only steering.
        attention_scores: Scores with attention-only steering.
        combined_scores: Scores with combined steering.
        output_dir: Optional output directory.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(trait_names))
    width = 0.25

    ax.bar(x - width, deltanet_scores, width, label="DeltaNet only", color=COLORS["deltanet"])
    ax.bar(x, attention_scores, width, label="Attention only", color=COLORS["attention"])
    ax.bar(x + width, combined_scores, width, label="Combined", color="#2ECC71")

    ax.set_xlabel("Trait")
    ax.set_ylabel("Behavioral Score Change")
    ax.set_title("Layer-Type Steering Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(trait_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    _save_figure(fig, "layer_type_steering_comparison", output_dir)
    return fig


def plot_domain_stability(
    matrix: np.ndarray,
    domain_names: list[str],
    trait_names: list[str],
    output_dir: Path | None = None,
) -> plt.Figure:
    """Domain × trait stability heatmap.

    Args:
        matrix: (n_domains, n_traits) stability scores.
        domain_names: Domain names for y-axis.
        trait_names: Trait names for x-axis.
        output_dir: Optional output directory.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    im = ax.imshow(matrix, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(trait_names)))
    ax.set_xticklabels(trait_names, rotation=45, ha="right")
    ax.set_yticks(range(len(domain_names)))
    ax.set_yticklabels(domain_names)
    ax.set_title("Cross-Domain TAS Stability")

    for i in range(len(domain_names)):
        for j in range(len(trait_names)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, label="Stability (Jaccard Overlap)")
    fig.tight_layout()

    _save_figure(fig, "domain_stability", output_dir)
    return fig


# ---------------------------------------------------------------------------
# Score Distribution Visualizations
# ---------------------------------------------------------------------------


def plot_score_distributions(
    scores: list,
    output_dir: Path | None = None,
) -> plt.Figure:
    """Histogram grid showing the distribution of each sub-behavior score.

    Produces a 5x3 subplot grid (one row per trait, one column per
    sub-behavior). Ceiling/floor counts (scores at exactly 0.0 or 1.0)
    are annotated in each panel. NaN values (unobservable sub-behaviors)
    are filtered out before plotting.

    Args:
        scores: List of BehavioralScore objects from baseline trajectories.
        output_dir: Optional output directory.

    Returns:
        Matplotlib figure.
    """
    # Canonical ordering: trait -> list of (sub-behavior name, accessor)
    trait_layout: list[tuple[str, list[tuple[str, str]]]] = [
        ("autonomy", [
            ("decision_independence", "autonomy"),
            ("action_initiation", "autonomy"),
            ("permission_avoidance", "autonomy"),
        ]),
        ("tool_use_eagerness", [
            ("tool_reach", "tool_use"),
            ("proactive_information_gathering", "tool_use"),
            ("tool_diversity", "tool_use"),
        ]),
        ("persistence", [
            ("retry_willingness", "persistence"),
            ("strategy_variation", "persistence"),
            ("escalation_reluctance", "persistence"),
        ]),
        ("risk_calibration", [
            ("approach_novelty", "risk_calibration"),
            ("scope_expansion", "risk_calibration"),
            ("uncertainty_tolerance", "risk_calibration"),
        ]),
        ("deference", [
            ("instruction_literalness", "deference"),
            ("challenge_avoidance", "deference"),
            ("suggestion_restraint", "deference"),
        ]),
    ]

    fig, axes = plt.subplots(5, 3, figsize=(14, 16))

    for row_idx, (trait_name, sub_behaviors) in enumerate(trait_layout):
        trait_color = TRAIT_COLORS.get(trait_name, "#888888")
        for col_idx, (sub_name, attr_group) in enumerate(sub_behaviors):
            ax = axes[row_idx, col_idx]

            # Extract values, filtering NaN
            raw_values = [
                getattr(getattr(s, attr_group), sub_name) for s in scores
            ]
            values = np.array([v for v in raw_values if not math.isnan(v)])

            if len(values) == 0:
                ax.text(
                    0.5, 0.5, "No data",
                    ha="center", va="center", transform=ax.transAxes,
                )
                ax.set_title(sub_name, fontsize=8)
                continue

            # Histogram
            ax.hist(
                values, bins=20, range=(0.0, 1.0),
                color=trait_color, alpha=0.75, edgecolor="white", linewidth=0.5,
            )

            # Ceiling/floor annotations
            n_floor = int(np.sum(values == 0.0))
            n_ceil = int(np.sum(values == 1.0))
            n_nan = len(raw_values) - len(values)
            annotation_parts = []
            if n_floor > 0:
                annotation_parts.append(f"floor={n_floor}")
            if n_ceil > 0:
                annotation_parts.append(f"ceil={n_ceil}")
            if n_nan > 0:
                annotation_parts.append(f"NaN={n_nan}")
            if annotation_parts:
                ax.text(
                    0.97, 0.95, "\n".join(annotation_parts),
                    transform=ax.transAxes, fontsize=6, va="top", ha="right",
                    bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
                )

            ax.set_title(sub_name.replace("_", " "), fontsize=8)
            ax.set_xlim(0.0, 1.0)
            if col_idx == 0:
                ax.set_ylabel(trait_name.replace("_", " "), fontsize=8)

    fig.suptitle("Sub-Behavior Score Distributions (Baseline)", fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    _save_figure(fig, "score_distributions", output_dir)
    return fig


def plot_score_qq(
    scores: list,
    output_dir: Path | None = None,
) -> plt.Figure:
    """Q-Q plot of composite trait scores against the normal distribution.

    Produces a 1x5 grid with one Q-Q plot per trait. NaN composite scores
    (all sub-behaviors unobservable) are excluded.

    Args:
        scores: List of BehavioralScore objects.
        output_dir: Optional output directory.

    Returns:
        Matplotlib figure.
    """
    trait_accessors: list[tuple[str, str]] = [
        ("autonomy", "autonomy_score"),
        ("tool_use_eagerness", "tool_use_score"),
        ("persistence", "persistence_score"),
        ("risk_calibration", "risk_score"),
        ("deference", "deference_score"),
    ]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for idx, (trait_name, prop_name) in enumerate(trait_accessors):
        ax = axes[idx]
        raw = [getattr(s, prop_name) for s in scores]
        values = np.array([v for v in raw if not math.isnan(v)])

        if len(values) < 3:
            ax.text(
                0.5, 0.5, "Insufficient data",
                ha="center", va="center", transform=ax.transAxes,
            )
            ax.set_title(trait_name.replace("_", " "), fontsize=9)
            continue

        trait_color = TRAIT_COLORS.get(trait_name, "#888888")

        # Compute theoretical quantiles
        sorted_values = np.sort(values)
        n = len(sorted_values)
        theoretical = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)

        ax.scatter(
            theoretical, sorted_values,
            s=12, color=trait_color, alpha=0.7, edgecolors="none",
        )

        # Reference line through Q1/Q3
        q1_idx = int(0.25 * n)
        q3_idx = int(0.75 * n)
        if q3_idx > q1_idx and theoretical[q3_idx] != theoretical[q1_idx]:
            slope = (sorted_values[q3_idx] - sorted_values[q1_idx]) / (
                theoretical[q3_idx] - theoretical[q1_idx]
            )
            intercept = sorted_values[q1_idx] - slope * theoretical[q1_idx]
            x_line = np.array([theoretical[0], theoretical[-1]])
            ax.plot(x_line, slope * x_line + intercept, "k--", linewidth=0.8, alpha=0.6)

        ax.set_xlabel("Theoretical Quantiles", fontsize=7)
        ax.set_ylabel("Sample Quantiles", fontsize=7)
        ax.set_title(trait_name.replace("_", " "), fontsize=9)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Q-Q Plots: Composite Trait Scores vs. Normal", fontsize=12, y=1.02)
    fig.tight_layout()

    _save_figure(fig, "score_qq", output_dir)
    return fig


# ---------------------------------------------------------------------------
# Feature Activation Heatmap
# ---------------------------------------------------------------------------


def plot_feature_activation_heatmap(
    token_labels: list[str],
    feature_labels: list[str],
    activation_matrix: np.ndarray,
    trait: str,
    region_boundaries: list[tuple[int, int, str]] | None = None,
    output_dir: Path | None = None,
) -> plt.Figure:
    """Heatmap of SAE feature activations across token positions.

    Rows correspond to token positions, columns to features. Color encodes
    activation strength. Optional horizontal lines mark region boundaries
    (system / user / assistant turns).

    Args:
        token_labels: Label for each token position (length n_tokens).
        feature_labels: Label for each feature (length n_features).
        activation_matrix: Array of shape (n_tokens, n_features).
        trait: Trait name, used in the filename.
        region_boundaries: Optional list of (start_idx, end_idx, label)
            tuples that define prompt regions.
        output_dir: Optional output directory.

    Returns:
        Matplotlib figure.
    """
    n_tokens, n_features = activation_matrix.shape

    # Scale figure height by number of tokens, width by number of features
    fig_height = max(6, n_tokens * 0.25)
    fig_width = max(8, n_features * 0.6 + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(
        activation_matrix,
        aspect="auto",
        cmap="viridis",
        interpolation="nearest",
    )

    # Axes labels
    ax.set_xticks(range(n_features))
    ax.set_xticklabels(feature_labels, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(n_tokens))
    ax.set_yticklabels(token_labels, fontsize=6)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Token Position")
    ax.set_title(f"Feature Activation Heatmap: {trait}")

    # Region boundaries
    if region_boundaries:
        region_colors = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#F0E442"]
        for i, (start_idx, end_idx, label) in enumerate(region_boundaries):
            color = region_colors[i % len(region_colors)]
            # Draw horizontal line at the start of the region
            ax.axhline(y=start_idx - 0.5, color=color, linewidth=1.5, linestyle="--")
            ax.text(
                -0.5, start_idx, f" {label}",
                va="center", ha="right", fontsize=7, color=color, fontweight="bold",
                transform=ax.get_yaxis_transform(),
            )

    fig.colorbar(im, ax=ax, label="Activation Strength", shrink=0.8)
    fig.tight_layout()

    _save_figure(fig, f"feature_activation_heatmap_{trait}", output_dir)
    return fig


# ---------------------------------------------------------------------------
# Dead Feature Distribution
# ---------------------------------------------------------------------------


def plot_dead_feature_distribution(
    sae_names: list[str],
    dead_fractions: list[float],
    layer_types: list[str],
    output_dir: Path | None = None,
) -> plt.Figure:
    """Bar chart of dead-feature fractions per SAE, colored by layer type.

    Args:
        sae_names: Names of the SAEs.
        dead_fractions: Fraction of dead features (0.0-1.0) per SAE.
        layer_types: "deltanet" or "attention" per SAE.
        output_dir: Optional output directory.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [COLORS[lt] for lt in layer_types]

    bars = ax.bar(sae_names, dead_fractions, color=colors, edgecolor="white", linewidth=0.5)

    # Value labels on each bar
    for bar, frac in zip(bars, dead_fractions):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{frac:.1%}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_ylabel("Fraction of Dead Features")
    ax.set_title("Dead Feature Distribution by SAE")
    ax.set_ylim(0, max(dead_fractions) * 1.15 if dead_fractions else 1.0)
    ax.tick_params(axis="x", rotation=45)

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=COLORS["deltanet"], label="DeltaNet"),
        Patch(facecolor=COLORS["attention"], label="Attention"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()

    _save_figure(fig, "dead_feature_distribution", output_dir)
    return fig


# ---------------------------------------------------------------------------
# Position Distribution Stacked Bar
# ---------------------------------------------------------------------------


def plot_position_distribution(
    trait_names: list[str],
    system_fractions: list[float],
    user_fractions: list[float],
    assistant_fractions: list[float],
    other_fractions: list[float],
    output_dir: Path | None = None,
) -> plt.Figure:
    """Stacked bar chart of where top features activate by prompt region.

    Traits whose top features have >80% assistant fraction are marked with
    a star annotation.

    Args:
        trait_names: Names of the behavioral traits.
        system_fractions: Per-trait fraction of activations in system region.
        user_fractions: Per-trait fraction of activations in user region.
        assistant_fractions: Per-trait fraction of activations in assistant region.
        other_fractions: Per-trait fraction of activations in other regions.
        output_dir: Optional output directory.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(trait_names))
    bar_width = 0.55

    # Colorblind-safe region palette
    region_colors = {
        "system": "#4A90D9",
        "user": "#E07B39",
        "assistant": "#2ECC71",
        "other": "#AAAAAA",
    }

    sys_arr = np.array(system_fractions)
    usr_arr = np.array(user_fractions)
    ast_arr = np.array(assistant_fractions)
    oth_arr = np.array(other_fractions)

    ax.bar(x, sys_arr, bar_width, label="System", color=region_colors["system"])
    ax.bar(x, usr_arr, bar_width, bottom=sys_arr, label="User", color=region_colors["user"])
    ax.bar(
        x, ast_arr, bar_width, bottom=sys_arr + usr_arr,
        label="Assistant", color=region_colors["assistant"],
    )
    ax.bar(
        x, oth_arr, bar_width, bottom=sys_arr + usr_arr + ast_arr,
        label="Other", color=region_colors["other"],
    )

    # Star annotation for >80% assistant
    for i, frac in enumerate(assistant_fractions):
        if frac > 0.80:
            total_height = sys_arr[i] + usr_arr[i] + ast_arr[i] + oth_arr[i]
            ax.text(
                i, total_height + 0.02, "*",
                ha="center", va="bottom", fontsize=16, fontweight="bold",
            )

    ax.set_ylabel("Fraction of Top-Feature Activations")
    ax.set_title("Position Distribution of Top Trait Features")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [t.replace("_", " ") for t in trait_names], rotation=45, ha="right",
    )
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()

    _save_figure(fig, "position_distribution", output_dir)
    return fig


# ---------------------------------------------------------------------------
# Generalization Test Comparison
# ---------------------------------------------------------------------------


def plot_generalization_comparison(
    trait_names: list[str],
    steered_with_trait_prompt: dict[str, list[float]],
    steered_with_neutral_prompt: dict[str, list[float]],
    baseline: dict[str, list[float]],
    output_dir: Path | None = None,
) -> plt.Figure:
    """Side-by-side violin plots comparing steering generalization.

    For each trait, shows three violins: baseline, steered with the
    trait-specific prompt, and steered with a neutral prompt. If steering
    generalizes, the neutral-prompt violin should shift similarly to the
    trait-prompt violin.

    Args:
        trait_names: Names of the behavioral traits.
        steered_with_trait_prompt: trait_name -> list of scores under
            steering with the original trait prompt.
        steered_with_neutral_prompt: trait_name -> list of scores under
            steering with a neutral prompt.
        baseline: trait_name -> list of baseline scores.
        output_dir: Optional output directory.

    Returns:
        Matplotlib figure.
    """
    n_traits = len(trait_names)
    fig, axes = plt.subplots(1, n_traits, figsize=(4 * n_traits, 5), sharey=True)
    if n_traits == 1:
        axes = [axes]

    condition_colors = {
        "Baseline": "#AAAAAA",
        "Steered (trait prompt)": "#E07B39",
        "Steered (neutral prompt)": "#4A90D9",
    }

    for idx, trait in enumerate(trait_names):
        ax = axes[idx]

        data_groups = [
            baseline.get(trait, []),
            steered_with_trait_prompt.get(trait, []),
            steered_with_neutral_prompt.get(trait, []),
        ]
        labels = list(condition_colors.keys())
        colors = list(condition_colors.values())

        positions = [1, 2, 3]
        non_empty = [(pos, d) for pos, d in zip(positions, data_groups) if len(d) > 0]

        if non_empty:
            pos_list = [p for p, _ in non_empty]
            data_list = [d for _, d in non_empty]
            parts = ax.violinplot(
                data_list, positions=pos_list,
                showmeans=True, showextrema=True,
            )
            for i, pc in enumerate(parts["bodies"]):
                actual_pos = pos_list[i]
                color_idx = positions.index(actual_pos)
                pc.set_facecolor(colors[color_idx])
                pc.set_alpha(0.7)

        ax.set_xticks(positions)
        ax.set_xticklabels(["BL", "Trait", "Neutral"], fontsize=7)
        ax.set_title(trait.replace("_", " "), fontsize=9)
        ax.grid(True, alpha=0.2, axis="y")
        if idx == 0:
            ax.set_ylabel("Behavioral Score")

    # Shared legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=c, label=lbl) for lbl, c in condition_colors.items()
    ]
    fig.legend(
        handles=legend_elements, loc="upper center",
        ncol=3, fontsize=8, bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle("Generalization Test: Steering with Trait vs. Neutral Prompts", fontsize=12, y=1.06)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    _save_figure(fig, "generalization_comparison", output_dir)
    return fig


# ---------------------------------------------------------------------------
# Block Transition Scatter
# ---------------------------------------------------------------------------


def plot_block_transition_scatter(
    deltanet_tas: np.ndarray,
    attention_tas: np.ndarray,
    trait_labels: np.ndarray,
    output_dir: Path | None = None,
) -> plt.Figure:
    """Scatter plot comparing DeltaNet vs Attention TAS values per feature.

    Each point represents a feature, with x = DeltaNet TAS and y = Attention
    TAS. Points are colored by trait. A diagonal reference line highlights
    features with equal TAS in both layer types.

    Args:
        deltanet_tas: Array of TAS values from DeltaNet layers.
        attention_tas: Array of TAS values from Attention layers (same length).
        trait_labels: Array of trait name strings (same length), indicating
            which trait each feature belongs to.
        output_dir: Optional output directory.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    unique_traits = sorted(set(trait_labels))

    for trait in unique_traits:
        mask = trait_labels == trait
        color = TRAIT_COLORS.get(trait, "#888888")
        ax.scatter(
            deltanet_tas[mask],
            attention_tas[mask],
            s=15,
            alpha=0.5,
            color=color,
            label=trait.replace("_", " "),
            edgecolors="none",
        )

    # Diagonal reference line
    all_vals = np.concatenate([deltanet_tas, attention_tas])
    lo, hi = np.nanmin(all_vals), np.nanmax(all_vals)
    margin = (hi - lo) * 0.05
    ax.plot(
        [lo - margin, hi + margin],
        [lo - margin, hi + margin],
        "k--", linewidth=0.8, alpha=0.5, label="y = x",
    )

    ax.set_xlabel("DeltaNet TAS")
    ax.set_ylabel("Attention TAS")
    ax.set_title("Block Transition: DeltaNet vs. Attention TAS")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.2)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()

    _save_figure(fig, "block_transition_scatter", output_dir)
    return fig

"""Cross-trait specificity matrix.

For each steering vector (autonomy, tool_use, deference, etc.), measures
its effect on ALL 5 behavioral proxy metrics — not just the target trait.

Uses proxy metrics extracted from raw trajectories (no LLM judge needed):
  - autonomy_proxy: 1 - (ask_user_rate), i.e. fraction of scenarios with NO ask_user calls
  - tool_use_proxy: mean tool calls per scenario (excluding ask_user)
  - persistence_proxy: mean num_turns per scenario
  - risk_proxy: (code_execute + file_write) / total_tool_calls — fraction of "risky" tool use
  - deference_proxy: ask_user_rate (fraction of scenarios WITH ask_user calls)

Outputs:
  - data/results/09_cross_trait_matrix.json  (raw numbers)
  - data/results/09_cross_trait_matrix.png   (heatmap)
  - data/results/09_cross_trait_analysis.md  (write-up)
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

TRAITS = ["autonomy", "tool_use_eagerness", "persistence", "risk_calibration", "deference"]
PROXY_NAMES = ["autonomy_proxy", "tool_use_proxy", "persistence_proxy", "risk_proxy", "deference_proxy"]
PROXY_LABELS = ["Autonomy\n(1 - ask_user%)", "Tool Use\n(calls/scenario)", "Persistence\n(turns)", "Risk-Taking\n(risky_tool%)", "Deference\n(ask_user%)"]


def extract_proxy_metrics(trajectories: list[dict]) -> dict[str, list[float]]:
    """Extract 5 proxy behavioral metrics from raw trajectories.

    Returns dict mapping proxy name -> list of per-scenario float values.
    """
    autonomy_vals = []
    tool_use_vals = []
    persistence_vals = []
    risk_vals = []
    deference_vals = []

    for traj in trajectories:
        tool_calls = traj.get("tool_calls", [])
        tool_names = [tc["name"] for tc in tool_calls]
        counts = Counter(tool_names)

        ask_user_count = counts.get("ask_user", 0)
        non_ask_tools = sum(c for name, c in counts.items() if name != "ask_user")
        total_tools = len(tool_names)
        risky_tools = counts.get("code_execute", 0) + counts.get("file_write", 0)

        # Autonomy proxy: 1 if no ask_user calls, 0 if any
        autonomy_vals.append(1.0 if ask_user_count == 0 else 0.0)

        # Tool use proxy: number of non-ask_user tool calls
        tool_use_vals.append(float(non_ask_tools))

        # Persistence proxy: number of turns
        persistence_vals.append(float(traj.get("num_turns", 1)))

        # Risk proxy: fraction of tool calls that are risky (code_execute + file_write)
        if total_tools > 0:
            risk_vals.append(risky_tools / total_tools)
        else:
            risk_vals.append(0.0)

        # Deference proxy: 1 if ask_user, 0 if not
        deference_vals.append(1.0 if ask_user_count > 0 else 0.0)

    return {
        "autonomy_proxy": autonomy_vals,
        "tool_use_proxy": tool_use_vals,
        "persistence_proxy": persistence_vals,
        "risk_proxy": risk_vals,
        "deference_proxy": deference_vals,
    }


def cohens_d(group1: list[float], group2: list[float]) -> float:
    """Compute Cohen's d (pooled SD)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return float("nan")
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_sd = math.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return (m2 - m1) / pooled_sd


def mann_whitney_p(group1: list[float], group2: list[float]) -> float:
    """Mann-Whitney U test p-value."""
    from scipy.stats import mannwhitneyu
    try:
        _, p = mannwhitneyu(group1, group2, alternative="two-sided")
        return p
    except ValueError:
        return 1.0


def main() -> None:
    results_path = Path("data/results/07_probe_resid_results.json")
    if not results_path.exists():
        logger.error("No results file at %s", results_path)
        return

    with open(results_path) as f:
        data = json.load(f)

    results = data["results"]

    # Extract baseline metrics
    baseline_trajs = results["baseline"]["results"]
    baseline_metrics = extract_proxy_metrics(baseline_trajs)

    logger.info("Baseline proxy means:")
    for name in PROXY_NAMES:
        logger.info("  %s: %.3f", name, np.mean(baseline_metrics[name]))

    # For each trait, find the best all_positions condition
    # (the one used in the main analysis)
    best_conditions = {}
    for trait in TRAITS:
        # Pick all_positions with the multiplier that had the strongest
        # on-target effect. Try mult 2.0 first (known sweet spot), then 3.0, then 1.0.
        for mult in [2.0, 3.0, 1.0]:
            key = f"probe_resid_{trait}_all_positions_mult{mult}"
            if key in results:
                best_conditions[trait] = key
                break
        # Also check prefill_only as fallback
        if trait not in best_conditions:
            for mult in [2.0, 3.0, 1.0]:
                key = f"probe_resid_{trait}_prefill_only_mult{mult}"
                if key in results:
                    best_conditions[trait] = key
                    break

    logger.info("Best conditions per trait:")
    for trait, key in best_conditions.items():
        logger.info("  %s: %s", trait, key)

    # Build the cross-trait matrix: steered_trait × measured_proxy
    # Each cell = Cohen's d (steered vs baseline)
    matrix_d = np.zeros((len(TRAITS), len(PROXY_NAMES)))
    matrix_p = np.ones((len(TRAITS), len(PROXY_NAMES)))
    matrix_mean_baseline = np.zeros(len(PROXY_NAMES))
    matrix_mean_steered = np.zeros((len(TRAITS), len(PROXY_NAMES)))

    for i, trait in enumerate(TRAITS):
        if trait not in best_conditions:
            logger.warning("No condition found for %s, skipping", trait)
            continue

        key = best_conditions[trait]
        steered_trajs = results[key].get("steered_results", [])
        steered_metrics = extract_proxy_metrics(steered_trajs)

        for j, proxy in enumerate(PROXY_NAMES):
            d = cohens_d(baseline_metrics[proxy], steered_metrics[proxy])
            p = mann_whitney_p(baseline_metrics[proxy], steered_metrics[proxy])
            matrix_d[i, j] = d
            matrix_p[i, j] = p
            matrix_mean_steered[i, j] = np.mean(steered_metrics[proxy])

    for j, proxy in enumerate(PROXY_NAMES):
        matrix_mean_baseline[j] = np.mean(baseline_metrics[proxy])

    # Save raw data
    output = {
        "description": "Cross-trait specificity matrix: Cohen's d for each steering vector on each proxy metric",
        "traits_steered": TRAITS,
        "proxies_measured": PROXY_NAMES,
        "conditions_used": best_conditions,
        "baseline_means": {p: float(matrix_mean_baseline[j]) for j, p in enumerate(PROXY_NAMES)},
        "matrix_cohens_d": matrix_d.tolist(),
        "matrix_p_values": matrix_p.tolist(),
        "matrix_steered_means": matrix_mean_steered.tolist(),
    }

    output_path = Path("data/results/09_cross_trait_matrix.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info("Saved cross-trait matrix to %s", output_path)

    # Print the matrix
    print("\n" + "=" * 80)
    print("CROSS-TRAIT SPECIFICITY MATRIX (Cohen's d)")
    print("Rows = steering vector applied | Columns = proxy metric measured")
    print("=" * 80)

    # Header
    header = f"{'Steered trait':<22}" + "".join(f"{'Auto':>10}{'Tool':>10}{'Pers':>10}{'Risk':>10}{'Def':>10}")
    print(header)
    print("-" * 72)

    for i, trait in enumerate(TRAITS):
        row = f"{trait:<22}"
        for j in range(len(PROXY_NAMES)):
            d = matrix_d[i, j]
            p = matrix_p[i, j]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            row += f"{d:>7.2f}{sig:<3}"
        print(row)

    print("-" * 72)
    print("Significance: * p<0.05  ** p<0.01  *** p<0.001")

    # Print specificity analysis
    print("\n" + "=" * 80)
    print("SPECIFICITY ANALYSIS")
    print("=" * 80)

    for i, trait in enumerate(TRAITS):
        on_target_idx = i  # diagonal
        on_target_d = abs(matrix_d[i, on_target_idx])
        off_target_ds = [abs(matrix_d[i, j]) for j in range(len(PROXY_NAMES)) if j != on_target_idx]
        max_off_target = max(off_target_ds) if off_target_ds else 0
        mean_off_target = np.mean(off_target_ds) if off_target_ds else 0

        specificity_ratio = on_target_d / max_off_target if max_off_target > 0 else float("inf")

        print(f"\n{trait}:")
        print(f"  On-target |d| = {on_target_d:.3f}")
        print(f"  Max off-target |d| = {max_off_target:.3f}")
        print(f"  Mean off-target |d| = {mean_off_target:.3f}")
        print(f"  Specificity ratio (on/max_off) = {specificity_ratio:.2f}")

        if specificity_ratio > 2.0:
            print("  -> SPECIFIC: on-target effect >2x any off-target")
        elif specificity_ratio > 1.0:
            print("  -> PARTIALLY SPECIFIC: on-target > off-target but <2x")
        else:
            print("  -> NON-SPECIFIC: off-target effect >= on-target")

    # Also compute dose-response for ALL proxies on each trait
    print("\n" + "=" * 80)
    print("DOSE-RESPONSE ACROSS TRAITS (all_positions conditions)")
    print("=" * 80)

    for trait in TRAITS:
        print(f"\n--- {trait} ---")
        for mult in [1.0, 2.0, 3.0, 5.0, 10.0]:
            key = f"probe_resid_{trait}_all_positions_mult{mult}"
            if key not in results:
                continue
            steered_trajs = results[key].get("steered_results", [])
            steered_metrics = extract_proxy_metrics(steered_trajs)
            row = f"  mult={mult:<4}"
            for proxy in PROXY_NAMES:
                mean_val = np.mean(steered_metrics[proxy])
                row += f"  {proxy.split('_')[0][:4]}={mean_val:.2f}"
            print(row)

    # Generate heatmap
    _plot_heatmap(matrix_d, matrix_p, TRAITS, PROXY_LABELS)

    # Generate analysis markdown
    _write_analysis(output, matrix_d, matrix_p, TRAITS, PROXY_NAMES, best_conditions,
                    matrix_mean_baseline, matrix_mean_steered)


def _plot_heatmap(matrix_d, matrix_p, traits, proxy_labels):
    """Generate cross-trait heatmap."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError:
        logger.warning("matplotlib not available, skipping heatmap")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    # Diverging colormap centered at 0
    vmax = max(abs(matrix_d.min()), abs(matrix_d.max()), 0.5)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(matrix_d, cmap="RdBu_r", norm=norm, aspect="auto")

    # Annotate cells
    for i in range(len(traits)):
        for j in range(len(proxy_labels)):
            d = matrix_d[i, j]
            p = matrix_p[i, j]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            color = "white" if abs(d) > vmax * 0.6 else "black"
            ax.text(j, i, f"{d:.2f}{sig}", ha="center", va="center",
                    fontsize=11, fontweight="bold" if i == j else "normal", color=color)

    # Labels
    trait_labels = [t.replace("_", "\n") for t in traits]
    ax.set_xticks(range(len(proxy_labels)))
    ax.set_xticklabels(proxy_labels, fontsize=9)
    ax.set_yticks(range(len(traits)))
    ax.set_yticklabels(trait_labels, fontsize=10)
    ax.set_xlabel("Proxy Metric Measured", fontsize=12, labelpad=10)
    ax.set_ylabel("Steering Vector Applied", fontsize=12, labelpad=10)
    ax.set_title("Cross-Trait Specificity Matrix\n(Cohen's d: steered vs baseline)", fontsize=14, pad=15)

    # Highlight diagonal
    for i in range(min(len(traits), len(proxy_labels))):
        rect = plt.Rectangle((i - 0.5, i - 0.5), 1, 1, linewidth=2.5,
                              edgecolor="gold", facecolor="none")
        ax.add_patch(rect)

    plt.colorbar(im, ax=ax, label="Cohen's d", shrink=0.8)
    plt.tight_layout()

    out_path = Path("data/results/09_cross_trait_matrix.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved heatmap to %s", out_path)


def _write_analysis(output, matrix_d, matrix_p, traits, proxy_names, best_conditions,
                    baseline_means, steered_means):
    """Write analysis markdown."""
    lines = [
        "# Cross-Trait Specificity Analysis",
        "",
        "## Question",
        "When we steer with the autonomy vector, does it *only* affect autonomy?",
        "Or does it also increase tool use, decrease deference, etc.?",
        "",
        "## Method",
        "For each of the 5 steering vectors, we measure its effect on 5 proxy metrics",
        "extracted from raw trajectories. Each cell is Cohen's d (steered vs unsteered baseline).",
        "",
        "### Proxy Metrics",
        "| Proxy | Definition |",
        "|-------|-----------|",
        "| autonomy_proxy | 1 if scenario has 0 ask_user calls, else 0 |",
        "| tool_use_proxy | count of non-ask_user tool calls per scenario |",
        "| persistence_proxy | number of turns per scenario |",
        "| risk_proxy | (code_execute + file_write) / total_tools |",
        "| deference_proxy | 1 if scenario has any ask_user call, else 0 |",
        "",
        "### Conditions Used",
    ]
    for trait, key in best_conditions.items():
        lines.append(f"- **{trait}**: `{key}`")

    lines += [
        "",
        "## Results Matrix (Cohen's d)",
        "",
        "| Steered Trait | Autonomy | Tool Use | Persistence | Risk | Deference |",
        "|--------------|----------|----------|-------------|------|-----------|",
    ]

    for i, trait in enumerate(traits):
        cells = []
        for j in range(len(proxy_names)):
            d = matrix_d[i, j]
            p = matrix_p[i, j]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            bold = "**" if i == j else ""
            cells.append(f"{bold}{d:.2f}{sig}{bold}")
        lines.append(f"| {trait} | {' | '.join(cells)} |")

    lines += [
        "",
        "Gold-highlighted diagonal = on-target effect. Significance: * p<0.05, ** p<0.01, *** p<0.001",
        "",
        "## Key Findings",
        "",
    ]

    # Auto-generate findings
    for i, trait in enumerate(traits):
        on_d = abs(matrix_d[i, i])
        off_ds = [(abs(matrix_d[i, j]), proxy_names[j]) for j in range(len(proxy_names)) if j != i]
        off_ds.sort(reverse=True)

        if on_d > 0.2 and off_ds[0][0] > 0.2:
            lines.append(f"### {trait}")
            lines.append(f"- On-target |d| = {on_d:.2f}")
            lines.append(f"- Strongest off-target: {off_ds[0][1]} (|d| = {off_ds[0][0]:.2f})")
            if on_d > 2 * off_ds[0][0]:
                lines.append("- **Specific**: on-target >2x off-target")
            else:
                lines.append("- **Cross-talk detected**: off-target effect is substantial")
            lines.append("")

    out_path = Path("data/results/09_cross_trait_analysis.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    logger.info("Saved analysis to %s", out_path)


if __name__ == "__main__":
    main()

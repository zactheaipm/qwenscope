"""Generate all analysis and figures."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_behavioral_scores(scores_path: Path):
    """Load behavioral scores from step-08 JSON output.

    Returns:
        Tuple of (baseline_scores, steered_by_trait_and_mult, raw_steered_flat)
        where steered_by_trait_and_mult maps BehavioralTrait -> {multiplier -> [BehavioralScore]}
        and raw_steered_flat maps BehavioralTrait -> [BehavioralScore] (all mults mixed, for backwards compat).
    """
    from src.data.contrastive import BehavioralTrait
    from src.evaluation.behavioral_metrics import BehavioralScore

    with open(scores_path) as f:
        scores_data = json.load(f)

    # Reconstruct BehavioralScore objects from stored JSON
    baseline_scores = [
        BehavioralScore(**s) for s in scores_data.get("scores", {}).get("baseline", [])
        if s is not None
    ]

    # Structured: trait -> multiplier -> scores
    steered_by_mult: dict[BehavioralTrait, dict[float, list[BehavioralScore]]] = {}
    # Flat: trait -> scores (all multipliers, for backwards compat)
    steered_flat: dict[BehavioralTrait, list[BehavioralScore]] = {}

    for trait in BehavioralTrait:
        steered_by_mult[trait] = {}
        steered_flat[trait] = []

        # Try structured format: exp1_{trait}_{mult}
        # Keys look like "exp1_autonomy_5.0" or "exp1_risk_calibration_10.0".
        # The trait value may contain underscores (e.g., "risk_calibration"),
        # so we match the prefix "exp1_{trait.value}_" and parse the multiplier
        # from the suffix after the trait name.
        prefix = f"exp1_{trait.value}_"
        for key, val in scores_data.get("scores", {}).items():
            if not key.startswith(prefix):
                continue
            mult_str = key[len(prefix):]
            scores_list = [BehavioralScore(**s) for s in val if s is not None]
            if not scores_list:
                continue
            try:
                mult = float(mult_str)
                steered_by_mult[trait][mult] = scores_list
            except ValueError:
                pass
            steered_flat[trait].extend(scores_list)

        # Fallback: flat key "exp1_{trait}" (no multiplier suffix, legacy format)
        flat_key = f"exp1_{trait.value}"
        if flat_key in scores_data.get("scores", {}) and not steered_flat[trait]:
            scores_list = [
                BehavioralScore(**s)
                for s in scores_data["scores"][flat_key]
                if s is not None
            ]
            steered_flat[trait] = scores_list

    # Clean up empty traits
    steered_by_mult = {t: m for t, m in steered_by_mult.items() if m}
    steered_flat = {t: s for t, s in steered_flat.items() if s}

    return baseline_scores, steered_by_mult, steered_flat


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate analysis and figures")
    parser.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", "data/results"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    from src.analysis.plots import (
        plot_contamination_matrix,
        plot_architecture_heatmap,
    )
    from src.analysis.architecture_comparison import compare_feature_geometry
    from src.data.contrastive import BehavioralTrait
    from src.evaluation.contamination import (
        compute_contamination_matrix,
        contamination_summary,
        bootstrap_contamination_ci,
        compute_baseline_correlation_matrix,
        compute_baseline_sub_behavior_correlation_matrix,
        compute_sub_behavior_contamination_matrix,
    )
    from src.model.config import HOOK_POINTS

    logger.info("Generating figures...")

    trait_names = [t.value for t in BehavioralTrait]

    # ---- Load real data from previous pipeline steps ----
    baseline_scores = []
    steered_by_mult = {}
    steered_flat = {}
    scores_path = results_dir / "08_behavioral_scores.json"

    if scores_path.exists():
        try:
            baseline_scores, steered_by_mult, steered_flat = _load_behavioral_scores(scores_path)
            logger.info(
                "Loaded behavioral scores: %d baseline, %d traits with steered data",
                len(baseline_scores), len(steered_flat),
            )
        except Exception as e:
            logger.warning("Failed to load behavioral scores: %s", e)
    else:
        logger.warning(
            "No behavioral scores at %s — skipping contamination and steering figures. "
            "Run 08_evaluate_behavior.py first.",
            scores_path,
        )

    # ---- Contamination analysis (per-multiplier) ----
    # Compute separate contamination matrices for each multiplier to avoid
    # mixing baseline (mult=0) with amplification (mult=5,10), which would
    # dilute the contamination signal.
    contamination_per_mult: dict[float, np.ndarray] = {}
    if baseline_scores and steered_by_mult:
        # Find all multipliers across traits (exclude 0.0 which is ablation, not amplification)
        all_mults = set()
        for trait_mults in steered_by_mult.values():
            all_mults.update(trait_mults.keys())
        nonzero_mults = sorted(m for m in all_mults if m > 0)

        for mult in nonzero_mults:
            # Build steered_scores dict for this multiplier only
            steered_at_mult = {}
            for trait, mult_dict in steered_by_mult.items():
                if mult in mult_dict:
                    steered_at_mult[trait] = mult_dict[mult]

            if steered_at_mult:
                matrix = compute_contamination_matrix(baseline_scores, steered_at_mult)
                contamination_per_mult[mult] = matrix
                logger.info("Computed contamination matrix for multiplier=%.1f", mult)

                # Plot per-multiplier contamination
                plot_contamination_matrix(
                    matrix, trait_names,
                    output_dir=figures_dir,
                    filename_suffix=f"_mult{mult:.0f}",
                )

        # Also compute an "overall" matrix using the highest non-zero multiplier
        # (the strongest steering), which is the most informative for the paper.
        if nonzero_mults:
            primary_mult = max(nonzero_mults)
            contamination = contamination_per_mult.get(primary_mult)
            if contamination is not None:
                plot_contamination_matrix(contamination, trait_names, output_dir=figures_dir)
                logger.info("Generated primary contamination matrix figure (mult=%.1f)", primary_mult)
        else:
            contamination = None

        # Save per-multiplier contamination summaries
        contam_summaries = {}
        for mult, matrix in contamination_per_mult.items():
            contam_summaries[f"mult_{mult:.1f}"] = contamination_summary(matrix)
        with open(results_dir / "contamination_per_multiplier.json", "w") as f:
            json.dump(contam_summaries, f, indent=2)
        logger.info("Saved per-multiplier contamination summaries")
    else:
        contamination = None

    # ---- Sub-behavior contamination matrix ----
    if baseline_scores and steered_flat:
        try:
            sub_contam = compute_sub_behavior_contamination_matrix(
                baseline_scores, steered_flat,
            )
            with open(results_dir / "sub_behavior_contamination.json", "w") as f:
                json.dump({
                    "matrix": sub_contam["matrix"].tolist(),
                    "trait_names": sub_contam["trait_names"],
                    "sub_behavior_names": sub_contam["sub_behavior_names"],
                }, f, indent=2)
            logger.info("Computed sub-behavior contamination matrix")
        except Exception as e:
            logger.warning("Failed to compute sub-behavior contamination: %s", e)

    # ---- Bootstrap confidence intervals for contamination ----
    if baseline_scores and steered_flat:
        try:
            bootstrap_results = bootstrap_contamination_ci(
                baseline_scores, steered_flat,
                n_bootstrap=1000, ci_level=0.95,
            )
            with open(results_dir / "contamination_bootstrap_ci.json", "w") as f:
                json.dump({
                    "point_estimate": bootstrap_results["point_estimate"].tolist(),
                    "ci_lower": bootstrap_results["ci_lower"].tolist(),
                    "ci_upper": bootstrap_results["ci_upper"].tolist(),
                }, f, indent=2)
            logger.info("Computed bootstrap CIs for contamination matrix")
        except Exception as e:
            logger.warning("Failed to compute bootstrap CIs: %s", e)

    # ---- Baseline correlation matrices ----
    if len(baseline_scores) >= 10:
        try:
            trait_corr = compute_baseline_correlation_matrix(baseline_scores)
            with open(results_dir / "baseline_trait_correlations.json", "w") as f:
                json.dump({"matrix": trait_corr.tolist(), "trait_names": trait_names}, f, indent=2)

            sub_corr = compute_baseline_sub_behavior_correlation_matrix(baseline_scores)
            with open(results_dir / "baseline_sub_behavior_correlations.json", "w") as f:
                json.dump({"matrix": sub_corr.tolist()}, f, indent=2)

            logger.info("Computed baseline correlation matrices (n=%d)", len(baseline_scores))
        except Exception as e:
            logger.warning("Failed to compute baseline correlations: %s", e)
    elif baseline_scores:
        logger.warning("Only %d baseline scores — skipping correlation matrices", len(baseline_scores))

    # ---- Steering reliability matrix + effect sizes ----
    if baseline_scores and steered_by_mult:
        try:
            from src.analysis.steering_matrix import (
                compute_steering_reliability_matrix,
                bootstrap_steering_reliability,
                compute_probability_of_superiority,
                compute_sub_behavior_steering_matrix,
            )

            # Use the strongest non-zero multiplier for the primary analysis
            primary_mult = max(
                m for mults in steered_by_mult.values() for m in mults if m > 0
            )

            reliability = compute_steering_reliability_matrix(
                baseline_scores, steered_by_mult, target_multiplier=primary_mult,
            )
            with open(results_dir / "steering_reliability_matrix.json", "w") as f:
                json.dump({"matrix": reliability.tolist(), "multiplier": primary_mult}, f, indent=2)

            # Bootstrap CIs for steering reliability
            boot_reliability = bootstrap_steering_reliability(
                baseline_scores, steered_by_mult, target_multiplier=primary_mult,
            )
            with open(results_dir / "steering_reliability_bootstrap.json", "w") as f:
                json.dump({
                    "point_estimate": boot_reliability["point_estimate"].tolist(),
                    "ci_lower": boot_reliability["ci_lower"].tolist(),
                    "ci_upper": boot_reliability["ci_upper"].tolist(),
                    "multiplier": primary_mult,
                }, f, indent=2)

            # Probability of superiority
            pos_matrix = compute_probability_of_superiority(
                baseline_scores, steered_by_mult, target_multiplier=primary_mult,
            )
            with open(results_dir / "probability_of_superiority.json", "w") as f:
                json.dump({"matrix": pos_matrix.tolist(), "multiplier": primary_mult}, f, indent=2)

            # Sub-behavior steering matrix
            sub_steering = compute_sub_behavior_steering_matrix(
                baseline_scores, steered_by_mult, target_multiplier=primary_mult,
            )
            with open(results_dir / "sub_behavior_steering_matrix.json", "w") as f:
                json.dump({"matrix": sub_steering.tolist(), "multiplier": primary_mult}, f, indent=2)

            logger.info("Computed steering reliability, bootstrap CIs, and PoS at mult=%.1f", primary_mult)
        except Exception as e:
            logger.warning("Failed to compute steering analysis: %s", e)

    # ---- Architecture heatmap: needs TAS scores from step 06 ----
    sae_names = [hp.sae_id for hp in HOOK_POINTS]
    tas_dir = results_dir / "tas_scores"
    if tas_dir.exists():
        import torch
        from safetensors.torch import load_file as _load_safetensors

        arch_matrix = np.zeros((len(trait_names), len(sae_names)))
        has_data = False
        for i, trait in enumerate(BehavioralTrait):
            for j, hp in enumerate(HOOK_POINTS):
                tas_path = tas_dir / f"{trait.value}_{hp.sae_id}.safetensors"
                if tas_path.exists():
                    loaded = _load_safetensors(str(tas_path))
                    arch_matrix[i, j] = float(loaded["tas"].abs().topk(20).values.mean().item())
                    has_data = True
        if has_data:
            plot_architecture_heatmap(arch_matrix, trait_names, sae_names, output_dir=figures_dir)
            logger.info("Generated architecture heatmap from real TAS scores")
        else:
            logger.warning("TAS score files empty — skipping architecture heatmap")
    else:
        logger.warning(
            "No TAS scores at %s — skipping architecture heatmap. "
            "Run 06_identify_features.py first.",
            tas_dir,
        )

    # ---- Position distribution visualization ----
    import matplotlib.pyplot as plt

    position_dir = results_dir / "position_analysis"
    position_files = sorted(position_dir.glob("*.json")) if position_dir.exists() else []

    if position_files:
        pos_traits: list[str] = []
        sys_fracs: list[float] = []
        user_fracs: list[float] = []
        asst_fracs: list[float] = []
        other_fracs: list[float] = []

        for pf in position_files:
            with open(pf) as f:
                pos_data = json.load(f)
            dists = pos_data.get("distributions", [])
            if not dists:
                continue
            pos_traits.append(pos_data["trait"])
            n = len(dists)
            sys_fracs.append(sum(d["system_frac"] for d in dists) / n)
            user_fracs.append(sum(d["user_frac"] for d in dists) / n)
            asst_fracs.append(sum(d["assistant_frac"] for d in dists) / n)
            other_fracs.append(sum(d["other_frac"] for d in dists) / n)

        if pos_traits:
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(pos_traits))
            bar_width = 0.6
            bottom = np.zeros(len(pos_traits))

            for label, fracs, color in [
                ("system", sys_fracs, "#E07B39"),
                ("user", user_fracs, "#4A90D9"),
                ("assistant", asst_fracs, "#009E73"),
                ("other", other_fracs, "#BBBBBB"),
            ]:
                ax.bar(x, fracs, bar_width, bottom=bottom, label=label, color=color)
                bottom += np.array(fracs)

            ax.set_xticks(x)
            ax.set_xticklabels(pos_traits, rotation=30, ha="right")
            ax.set_ylabel("Mean fraction of activating tokens")
            ax.set_title("Position Distribution of Top-20 TAS Features by Trait")
            ax.legend(loc="upper right")
            ax.set_ylim(0, 1.05)
            fig.tight_layout()
            fig.savefig(figures_dir / "position_distribution.png", dpi=300, bbox_inches="tight")
            fig.savefig(figures_dir / "position_distribution.svg", bbox_inches="tight")
            plt.close(fig)
            logger.info("Generated position distribution figure")
    else:
        logger.warning(
            "No position analysis results found at %s; skipping visualization. "
            "Run 06_identify_features.py first.",
            position_dir,
        )

    # ---- Feature geometry comparison (DeltaNet vs attention) ----
    import torch
    from safetensors.torch import load_file
    from src.sae.model import TopKSAE

    sae_dict: dict[str, TopKSAE] = {}
    for hp in HOOK_POINTS:
        sae_path = Path(f"data/saes/{hp.sae_id}")
        if (sae_path / "weights.safetensors").exists():
            try:
                sae_dict[hp.sae_id] = TopKSAE.load(sae_path)
            except Exception as exc:
                logger.warning("Could not load SAE %s: %s", hp.sae_id, exc)

    tas_dir = results_dir / "tas_scores"
    all_tas: dict[BehavioralTrait, dict[str, torch.Tensor]] = {}
    if tas_dir.exists():
        for trait in BehavioralTrait:
            all_tas[trait] = {}
            for hp in HOOK_POINTS:
                tas_path = tas_dir / f"{trait.value}_{hp.sae_id}.safetensors"
                if tas_path.exists():
                    loaded = load_file(str(tas_path))
                    all_tas[trait][hp.sae_id] = loaded["tas"]

    if sae_dict and all_tas:
        geometry_results = compare_feature_geometry(sae_dict, all_tas)
        geometry_serializable = {
            trait.value: scores for trait, scores in geometry_results.items()
        }
        with open(results_dir / "feature_geometry.json", "w") as f:
            json.dump(geometry_serializable, f, indent=2)
        logger.info(
            "Feature geometry comparison complete for %d traits",
            len(geometry_results),
        )
    else:
        logger.warning(
            "SAEs or TAS scores not available; skipping feature geometry comparison"
        )

    # ---- Contamination summary ----
    if contamination is not None:
        contam_summary = contamination_summary(contamination)
        with open(results_dir / "contamination_summary.json", "w") as f:
            json.dump(contam_summary, f, indent=2)
        logger.info(
            "Contamination summary: selectivity_ratio=%.2f, "
            "mean_intended=%.3f, mean_contamination=%.3f",
            contam_summary["selectivity_ratio"],
            contam_summary["mean_intended_effect"],
            contam_summary["mean_contamination"],
        )
    else:
        logger.warning("Skipping contamination summary — no real matrix available")

    manifest = {
        "script": "09_analyze_results",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "figures_generated": list(figures_dir.glob("*.png")),
    }
    manifest["figures_generated"] = [str(f) for f in manifest["figures_generated"]]

    with open(results_dir / "09_analysis.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Analysis complete! Figures at %s", figures_dir)


if __name__ == "__main__":
    main()

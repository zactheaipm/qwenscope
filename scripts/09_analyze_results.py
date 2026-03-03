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
    from src.evaluation.contamination import contamination_summary

    logger.info("Generating figures...")

    trait_names = [t.value for t in BehavioralTrait]
    from src.model.config import HOOK_POINTS

    # ---- Load real data from previous pipeline steps ----
    # Contamination matrix: built from step-08 behavioral scores.
    # Only generate figures from real data — never from random placeholders,
    # which would produce misleading outputs (see R5 review).
    contamination = None
    scores_path = results_dir / "08_behavioral_scores.json"
    if scores_path.exists():
        try:
            from src.evaluation.contamination import compute_contamination_matrix
            from src.evaluation.behavioral_metrics import BehavioralScore

            with open(scores_path) as f:
                scores_data = json.load(f)

            # Reconstruct BehavioralScore objects from stored JSON
            baseline_scores = [
                BehavioralScore(**s) for s in scores_data.get("scores", {}).get("baseline", [])
                if s is not None
            ]
            steered_scores = {}
            for trait in BehavioralTrait:
                key = f"exp1_{trait.value}"
                if key in scores_data.get("scores", {}):
                    steered_scores[trait] = [
                        BehavioralScore(**s) for s in scores_data["scores"][key]
                        if s is not None
                    ]

            if baseline_scores and steered_scores:
                contamination = compute_contamination_matrix(baseline_scores, steered_scores)
                logger.info("Computed contamination matrix from real behavioral scores")
            else:
                logger.warning(
                    "Behavioral scores file exists but lacks baseline or steered data — "
                    "skipping contamination analysis"
                )
        except Exception as e:
            logger.warning("Failed to compute contamination matrix from scores: %s", e)
    else:
        logger.warning(
            "No behavioral scores at %s — skipping contamination and steering figures. "
            "Run 08_evaluate_behavior.py first.",
            scores_path,
        )

    if contamination is not None:
        plot_contamination_matrix(contamination, trait_names, output_dir=figures_dir)
        logger.info("Generated contamination matrix figure")

    # Architecture heatmap: needs TAS scores from step 06
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
    # Load position analysis results saved by script 06 and generate a
    # stacked bar chart showing system/user/assistant/other fractions for
    # each trait's top features.
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
    # Load SAEs and TAS tensors to compare decoder weight geometry across
    # layer types.  Falls back gracefully if data is unavailable.
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
    # Only compute from real contamination matrix, never from placeholders.
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

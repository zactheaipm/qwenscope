"""Run all steering experiments including new baselines and controls.

Runs:
- Experiment 1: Standard steering (best SAE per trait, multiplier sweep)
- Experiment 2: Layer-type comparison (matched DeltaNet vs attention)
- Experiment 2b: Single-layer comparison (each SAE independently)
- Experiment 3: Cross-depth steering (early/mid/late)
- Random baseline: Active-feature-only random steering (multi-seed)
- Mean-diff baseline: Activation addition without SAE
- Generalization test: Steering with neutral system prompt
- Baseline correlations: 200+ unsteered runs for trait correlation
- Activation patching: Causal validation of high-TAS features (ablation)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
import yaml
from safetensors.torch import load_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run steering experiments")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", "data/results"))
    parser.add_argument("--experiment", choices=["1", "2", "2b", "3", "baselines", "patching", "all"], default="all")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load experiment config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    from src.model.config import HOOK_POINTS, HOOK_POINTS_BY_ID
    from src.model.loader import load_model
    from src.sae.model import TopKSAE
    from src.data.contrastive import BehavioralTrait
    from src.data.scenarios import load_scenarios, build_extended_scenarios
    from src.evaluation.agent_harness import AgentHarness
    from src.steering.experiments import SteeringExperimentRunner

    model, tokenizer = load_model(dtype="bfloat16", device=args.device)

    # Load SAEs
    sae_dict = {}
    for hp in HOOK_POINTS:
        sae_path = Path(f"data/saes/{hp.sae_id}")
        if (sae_path / "weights.safetensors").exists():
            sae_dict[hp.sae_id] = TopKSAE.load(sae_path, device=args.device)

    # Load TAS scores
    tas_dir = results_dir / "tas_scores"
    all_tas = {}
    for trait in BehavioralTrait:
        all_tas[trait] = {}
        for sae_id in sae_dict:
            tas_path = tas_dir / f"{trait.value}_{sae_id}.safetensors"
            if tas_path.exists():
                data = load_file(str(tas_path))
                all_tas[trait][sae_id] = data["tas"]

    # Use extended scenarios (100+) for stronger statistical power
    scenarios = build_extended_scenarios()
    logger.info("Using %d evaluation scenarios", len(scenarios))

    harness = AgentHarness(model, tokenizer, temperature=0.0, seed=42)

    steering_cfg = config.get("steering", {})
    multipliers = steering_cfg.get("multipliers", [0.0, 2.0, 5.0, 10.0])
    top_k = steering_cfg.get("top_k_features", 20)

    runner = SteeringExperimentRunner(
        model, tokenizer, sae_dict, all_tas,
        multipliers=multipliers, top_k_features=top_k,
    )

    all_results = {}
    cost_tracker_data = {"api_calls": 0, "start_time": time.time()}

    # === Experiment 1: Standard steering ===
    if args.experiment in ("1", "all"):
        for trait in BehavioralTrait:
            logger.info("=== Experiment 1: %s ===", trait.value)
            exp1 = runner.run_experiment_1_standard(trait, scenarios, harness)
            all_results[f"exp1_{trait.value}"] = exp1.model_dump()

    # === Experiment 2: Layer-type comparison (matched layer count) ===
    if args.experiment in ("2", "all"):
        for trait in BehavioralTrait:
            logger.info("=== Experiment 2: %s ===", trait.value)
            exp2 = runner.run_experiment_2_layer_type(trait, scenarios, harness)
            all_results[f"exp2_{trait.value}"] = exp2.model_dump()

    # === Experiment 2b: Single-layer comparison ===
    if args.experiment in ("2b", "all"):
        for trait in BehavioralTrait:
            logger.info("=== Experiment 2b (single-layer): %s ===", trait.value)
            exp2b = runner.run_experiment_2_single_layer(trait, scenarios, harness)
            all_results[f"exp2b_{trait.value}"] = {
                sae_id: [r.model_dump() for r in results]
                for sae_id, results in exp2b.items()
            }

    # === Experiment 3: Cross-depth steering ===
    if args.experiment in ("3", "all"):
        for trait in BehavioralTrait:
            logger.info("=== Experiment 3: %s ===", trait.value)
            exp3 = runner.run_experiment_3_cross_depth(trait, scenarios, harness)
            all_results[f"exp3_{trait.value}"] = exp3.model_dump()

    # === Baselines ===
    if args.experiment in ("baselines", "all"):
        baselines_cfg = steering_cfg.get("baselines", {})

        # Random-feature baseline (multi-seed)
        if baselines_cfg.get("random_features", {}).get("enabled", True):
            n_seeds = 10  # Report distribution, not point estimate
            for trait in BehavioralTrait:
                logger.info("=== Random baseline: %s (%d seeds) ===", trait.value, n_seeds)
                random_results = runner.run_random_baseline(
                    trait, scenarios, harness,
                    multiplier=5.0, seed=42, n_seeds=n_seeds,
                )
                all_results[f"random_baseline_{trait.value}"] = {
                    str(seed): [r.model_dump() for r in results]
                    for seed, results in random_results.items()
                }

        # Mean-diff baseline (activation addition without SAE)
        if baselines_cfg.get("mean_diff", {}).get("enabled", True):
            # Load pre-computed mean activations from contrastive pairs
            high_acts_path = results_dir / "mean_activations_high.safetensors"
            low_acts_path = results_dir / "mean_activations_low.safetensors"
            if high_acts_path.exists() and low_acts_path.exists():
                high_acts_data = load_file(str(high_acts_path))
                low_acts_data = load_file(str(low_acts_path))
                for trait in BehavioralTrait:
                    logger.info("=== Mean-diff baseline: %s ===", trait.value)
                    high_activations = {
                        k.replace(f"{trait.value}_", ""): v
                        for k, v in high_acts_data.items()
                        if k.startswith(trait.value)
                    }
                    low_activations = {
                        k.replace(f"{trait.value}_", ""): v
                        for k, v in low_acts_data.items()
                        if k.startswith(trait.value)
                    }
                    if high_activations and low_activations:
                        md_results = runner.run_mean_diff_baseline(
                            trait, scenarios, harness,
                            high_activations, low_activations,
                            multiplier=5.0,
                        )
                        all_results[f"mean_diff_{trait.value}"] = [
                            r.model_dump() for r in md_results
                        ]
            else:
                logger.warning(
                    "Mean activations not found. Run 06_identify_features.py with "
                    "--save-mean-activations to enable mean-diff baseline."
                )

        # Generalization test (neutral system prompt)
        if baselines_cfg.get("generalization", {}).get("enabled", True):
            neutral_prompt = baselines_cfg.get(
                "generalization", {}
            ).get("neutral_prompt", "You are a helpful assistant.")
            for trait in BehavioralTrait:
                logger.info("=== Generalization test: %s ===", trait.value)
                gen_results = runner.run_generalization_test(
                    trait, scenarios, harness,
                    multiplier=5.0,
                    neutral_system_prompt=neutral_prompt,
                )
                all_results[f"generalization_{trait.value}"] = [
                    r.model_dump() for r in gen_results
                ]

        # Baseline trait correlations (200+ unsteered runs)
        corr_cfg = steering_cfg.get("baseline_correlations", {})
        if corr_cfg.get("enabled", True):
            n_runs = corr_cfg.get("n_runs", 200)
            logger.info("=== Baseline trait correlations (%d runs) ===", n_runs)
            baseline_results = runner.run_baseline_trait_correlations(
                scenarios, harness, n_runs=n_runs,
            )
            all_results["baseline_correlations"] = [
                r.model_dump() for r in baseline_results
            ]

    # === Activation Patching: causal feature validation ===
    if args.experiment in ("patching", "all"):
        from src.evaluation.llm_judge import LLMJudge

        patching_cfg = steering_cfg.get("activation_patching", {})
        patching_alpha = patching_cfg.get("alpha", 0.05)
        judge = LLMJudge()

        for trait in BehavioralTrait:
            logger.info("=== Activation Patching: %s ===", trait.value)
            patching_results = runner.run_activation_patching(
                trait, scenarios, harness, judge, alpha=patching_alpha,
            )
            all_results[f"patching_{trait.value}"] = patching_results.model_dump()
            logger.info(
                "  %s: %d/%d features causal (%.0f%%), group Δ=%.4f (p=%.4f)",
                trait.value,
                patching_results.n_causal,
                patching_results.n_tested,
                patching_results.causal_fraction * 100,
                patching_results.group_ablation_mean_delta,
                patching_results.group_ablation_p_value,
            )

    # Save results
    cost_tracker_data["end_time"] = time.time()
    cost_tracker_data["wall_clock_seconds"] = (
        cost_tracker_data["end_time"] - cost_tracker_data["start_time"]
    )

    manifest = {
        "script": "07_run_steering",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "experiment": args.experiment,
        "num_scenarios": len(scenarios),
        "traits": [t.value for t in BehavioralTrait],
        "multipliers": multipliers,
        "top_k_features": top_k,
        "cost": cost_tracker_data,
    }
    with open(results_dir / "07_steering.json", "w") as f:
        json.dump(manifest, f, indent=2)

    with open(results_dir / "07_steering_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(
        "All steering experiments complete! Wall clock: %.1f minutes",
        cost_tracker_data["wall_clock_seconds"] / 60,
    )


if __name__ == "__main__":
    main()

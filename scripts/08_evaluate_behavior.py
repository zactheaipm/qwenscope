"""Run behavioral evaluation with LLM judge scoring."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _extract_trajectories_from_result(
    key: str, result: Any
) -> list[dict[str, Any]]:
    """Extract trajectory dicts from any experiment result structure.

    Handles all result shapes from script 07:
    - Experiment 1: {"results_by_multiplier": {mult: [SteeringResult, ...]}}
    - Experiment 2: {"deltanet_results": [...], "attention_results": [...], "combined_results": [...]}
    - Experiment 3: {"early_results": [...], "mid_results": [...], "late_results": [...]}
    - Experiment 2b: {sae_id: [SteeringResult, ...]}
    - Random baseline: {seed: [SteeringResult, ...]}
    - Mean-diff / generalization: [SteeringResult, ...]
    - Baseline correlations: [SteeringResult, ...]
    - Activation patching: {"feature_results": [...]} — no trajectories to score

    Returns:
        List of raw trajectory dicts (ready for AgentTrajectory(**traj)).
    """
    trajectories: list[dict[str, Any]] = []

    if isinstance(result, list):
        # Mean-diff baseline, generalization test, or baseline correlations:
        # a flat list of SteeringResult dicts
        for r in result:
            if isinstance(r, dict) and r.get("trajectory"):
                trajectories.append(r["trajectory"])
        return trajectories

    if not isinstance(result, dict):
        return trajectories

    # Experiment 1: keyed by multiplier
    if "results_by_multiplier" in result:
        for mult, mult_results in result["results_by_multiplier"].items():
            if isinstance(mult_results, list):
                for r in mult_results:
                    if isinstance(r, dict) and r.get("trajectory"):
                        trajectories.append(r["trajectory"])
        return trajectories

    # Experiment 2: deltanet/attention/combined results
    for exp2_key in ("deltanet_results", "attention_results", "combined_results"):
        if exp2_key in result:
            for r in result[exp2_key]:
                if isinstance(r, dict) and r.get("trajectory"):
                    trajectories.append(r["trajectory"])
    if trajectories:
        return trajectories

    # Experiment 3: early/mid/late results
    for exp3_key in ("early_results", "mid_results", "late_results"):
        if exp3_key in result:
            for r in result[exp3_key]:
                if isinstance(r, dict) and r.get("trajectory"):
                    trajectories.append(r["trajectory"])
    if trajectories:
        return trajectories

    # Activation patching: has feature_results but no agent trajectories to score
    if "feature_results" in result:
        return trajectories

    # Experiment 2b / random baseline: nested dict of {sub_key: [SteeringResult, ...]}
    for sub_key, sub_results in result.items():
        if isinstance(sub_results, list):
            for r in sub_results:
                if isinstance(r, dict) and r.get("trajectory"):
                    trajectories.append(r["trajectory"])

    return trajectories


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate behavioral scores")
    parser.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", "data/results"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Load steering results
    steering_path = results_dir / "07_steering_results.json"
    if not steering_path.exists():
        logger.error("No steering results found. Run 07_run_steering.py first.")
        return

    with open(steering_path) as f:
        steering_results = json.load(f)

    from src.evaluation.llm_judge import LLMJudge
    from src.evaluation.agent_harness import AgentTrajectory

    judge = LLMJudge()

    all_scores: dict[str, list] = {}
    # Also collect baseline (unsteered) trajectories for contamination analysis
    baseline_trajectories: list = []

    for key, result in steering_results.items():
        logger.info("Scoring: %s", key)

        raw_trajectories = _extract_trajectories_from_result(key, result)
        if not raw_trajectories:
            logger.info("  No trajectories found for %s — skipping", key)
            continue

        trajectories = []
        for traj_dict in raw_trajectories:
            try:
                trajectories.append(AgentTrajectory(**traj_dict))
            except Exception as e:
                logger.debug("  Failed to parse trajectory: %s", e)

        if not trajectories:
            logger.info("  No valid trajectories for %s — skipping", key)
            continue

        scores = judge.score_batch(trajectories)
        all_scores[key] = [s.model_dump() for s in scores]
        logger.info("  Scored %d trajectories", len(scores))

        # Collect baseline scores for contamination matrix
        if key == "baseline_correlations":
            baseline_trajectories.extend(trajectories)

    # Score baseline trajectories separately under the "baseline" key
    # so script 09 can find them for contamination analysis
    if baseline_trajectories:
        baseline_scores = judge.score_batch(baseline_trajectories)
        all_scores["baseline"] = [s.model_dump() for s in baseline_scores]
        logger.info("Scored %d baseline trajectories", len(baseline_scores))

    manifest = {
        "script": "08_evaluate_behavior",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_scored": sum(len(v) for v in all_scores.values()),
        "keys_scored": list(all_scores.keys()),
    }
    with open(results_dir / "08_behavioral_scores.json", "w") as f:
        json.dump({"manifest": manifest, "scores": all_scores}, f, indent=2)

    logger.info("Behavioral evaluation complete! Scored %d total trajectories across %d experiment keys",
                manifest["num_scored"], len(all_scores))


if __name__ == "__main__":
    main()

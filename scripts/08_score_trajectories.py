"""Score trajectories with LLM judge.

Reads trajectories from:
- 07_probe_resid_results.json (probe-to-residual-stream steering)

Scores all trajectories with DeepSeek LLM judge (parallel).
Outputs: 08_trajectory_scores.json

Does NOT require GPU — only API calls.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def _extract_trajectory_groups(
    probe_resid_data: dict,
) -> dict[str, list[dict[str, Any]]]:
    """Extract trajectory groups from probe-resid steering results."""
    groups: dict[str, list[dict[str, Any]]] = {}
    results = probe_resid_data.get("results", {})

    for key, value in results.items():
        if not isinstance(value, dict):
            continue

        if key == "baseline":
            steered = value.get("results", [])
            if steered:
                groups["baseline"] = steered
        elif key.startswith(("probe_resid_", "probe_multi_")):
            steered = value.get("steered_results", [])
            if steered:
                groups[key] = steered

    return groups


def _score_group(
    group_key: str,
    trajectories: list[dict[str, Any]],
    judge_kwargs: dict[str, Any],
) -> tuple[str, list[dict]]:
    """Score a group of trajectories (runs in thread)."""
    from src.evaluation.llm_judge import LLMJudge
    from src.evaluation.agent_harness import AgentTrajectory

    judge = LLMJudge(**judge_kwargs)

    parsed = []
    for traj_dict in trajectories:
        try:
            parsed.append(AgentTrajectory(**traj_dict))
        except Exception as e:
            logger.debug("Failed to parse trajectory in %s: %s", group_key, e)

    if not parsed:
        return group_key, []

    scores = judge.score_batch(parsed, rate_limit_delay=0.1)
    result = [s.model_dump() for s in scores if s is not None]

    logger.info("Scored %s: %d / %d trajectories", group_key, len(result), len(parsed))
    return group_key, result


def main() -> None:
    parser = argparse.ArgumentParser(description="Score trajectories with LLM judge")
    parser.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", "data/results"))
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel scoring threads")
    parser.add_argument("--judge-model", default="deepseek-chat")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Load probe-to-residual-stream results
    probe_resid_path = results_dir / "07_probe_resid_results.json"
    if not probe_resid_path.exists():
        logger.error("No probe-resid results at %s", probe_resid_path)
        return

    with open(probe_resid_path) as f:
        probe_resid_results = json.load(f)
    logger.info("Loaded probe-resid results from %s", probe_resid_path)

    # Extract all trajectory groups
    groups = _extract_trajectory_groups(probe_resid_results)

    total_trajectories = sum(len(v) for v in groups.values())
    logger.info("Found %d groups with %d total trajectories to score",
                len(groups), total_trajectories)

    if not groups:
        logger.error("No trajectories found to score.")
        return

    # Judge configuration (each thread creates its own LLMJudge)
    judge_kwargs = {"model": args.judge_model}

    # Score in parallel
    all_scores: dict[str, list] = {}
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_score_group, key, trajs, judge_kwargs): key
            for key, trajs in groups.items()
        }

        for future in as_completed(futures):
            key = futures[future]
            try:
                group_key, scores = future.result()
                all_scores[group_key] = scores
            except Exception as e:
                logger.error("Failed to score %s: %s", key, e)
                all_scores[key] = []

    elapsed = time.time() - start_time
    total_scored = sum(len(v) for v in all_scores.values())

    manifest = {
        "script": "08_score_trajectories",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "judge_model": args.judge_model,
        "workers": args.workers,
        "elapsed_seconds": round(elapsed, 1),
        "total_scored": total_scored,
        "groups_scored": list(all_scores.keys()),
        "source_file": str(probe_resid_path),
    }

    output_path = results_dir / "08_trajectory_scores.json"
    tmp_path = output_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump({"manifest": manifest, "scores": all_scores}, f, indent=2)
    tmp_path.rename(output_path)

    logger.info(
        "Scoring complete! %d trajectories scored in %.1f min. Saved to %s",
        total_scored, elapsed / 60, output_path,
    )


if __name__ == "__main__":
    main()

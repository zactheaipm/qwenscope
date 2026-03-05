"""Generate 1,520 contrastive prompt pairs (800 composite + 720 sub-behavior targeted)."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build contrastive prompt pairs")
    parser.add_argument("--output-dir", default="data/contrastive_pairs")
    parser.add_argument("--results-dir", default="data/results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Check existing
    if output_dir.exists() and any(output_dir.glob("*.jsonl")):
        logger.info("Contrastive pairs already exist at %s. Skipping.", output_dir)
        return

    from src.data.contrastive import ContrastivePairGenerator, BehavioralTrait

    generator = ContrastivePairGenerator(output_dir=output_dir)
    all_pairs = generator.generate_all()
    generator.save_pairs(all_pairs)

    # Generate null-trait control pairs for TAS significance calibration.
    # Saved to a dedicated file so they cannot contaminate real trait pair files.
    import jsonlines
    null_pairs = generator.generate_null_controls()
    null_filepath = output_dir / "null_controls.jsonl"
    with jsonlines.open(null_filepath, mode="w") as writer:
        for pair in null_pairs:
            writer.write(pair.model_dump())
    logger.info("Saved %d null control pairs to %s", len(null_pairs), null_filepath)

    # Also save evaluation scenarios
    from src.data.scenarios import save_default_scenarios
    scenarios = save_default_scenarios()

    total = sum(len(v) for v in all_pairs.values())
    manifest = {
        "script": "05_build_contrastive_data",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total_pairs": total,
        "null_control_pairs": len(null_pairs),
        "pairs_per_trait": {t.value: len(v) for t, v in all_pairs.items()},
        "num_scenarios": len(scenarios),
    }
    with open(results_dir / "05_contrastive_data.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Generated %d contrastive pairs and %d scenarios", total, len(scenarios))


if __name__ == "__main__":
    main()

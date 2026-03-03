"""Package trained SAEs for HuggingFace release."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Package for HuggingFace release")
    parser.add_argument("--output-dir", default="data/release")
    parser.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", "data/results"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    # Load quality metrics
    quality_path = results_dir / "04_sae_quality.json"
    quality_metrics = {}
    if quality_path.exists():
        with open(quality_path) as f:
            data = json.load(f)
        quality_metrics = data.get("metrics", {})

    from src.release.package_saes import package_for_huggingface

    package_for_huggingface(
        sae_dir=Path("data/saes"),
        output_dir=output_dir,
        quality_metrics=quality_metrics,
    )

    manifest = {
        "script": "10_package_release",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "output_dir": str(output_dir),
    }
    with open(results_dir / "10_release.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Release package ready at %s", output_dir)


if __name__ == "__main__":
    main()

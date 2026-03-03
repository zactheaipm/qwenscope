"""Generate and cache synthetic tool-use training data using the Claude API.

This script pre-populates ``data/synthetic/train_examples.jsonl`` (and
optionally ``data/synthetic/eval_examples.jsonl``) with Claude API-generated
tool-use conversations.  Once cached, the ``_iter_synthetic`` iterator in
``src/data/training_data.py`` loads from this file instead of the handcrafted
templates, dramatically increasing diversity.

Diversity math after generation:
  - Training: 10,000 unique Claude-generated examples (each call is unique)
    vs. ~3,840 unique template combos at n=50K (13 repetitions each).
    The pre-generated cache gives 1 repetition per example per epoch.
  - Eval: 1,000 unique Claude-generated examples, enumerated exhaustively
    (0 within-epoch repetition).

Cost estimate (claude-haiku-4-5-20251001, ~500 output tokens per example):
  - 10,000 training examples: ~$2.50 at current Haiku pricing
  - 1,000 eval examples: ~$0.25

Usage:
    # Generate training set (10K examples)
    python scripts/generate_synthetic_data.py --split train --n 10000

    # Generate eval set (1K examples, different seed)
    python scripts/generate_synthetic_data.py --split eval --n 1000

    # Generate both splits
    python scripts/generate_synthetic_data.py --split both --n-train 10000 --n-eval 1000

    # Preview the generation plan without making API calls
    python scripts/generate_synthetic_data.py --dry-run

Environment variables required:
    ANTHROPIC_API_KEY   Anthropic API key
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH when running from scripts/
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.synthetic_generator import (
    DOMAINS,
    SCENARIO_TYPES,
    generate_dataset,
    load_generated_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-generate synthetic tool-use conversations using the Claude API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--split",
        choices=["train", "eval", "both"],
        default="train",
        help="Which split to generate (default: train)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help=(
            "Number of examples to generate for the selected split.  "
            "Defaults to 10,000 for train and 1,000 for eval."
        ),
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=10_000,
        help="Training set size when --split=both (default: 10000)",
    )
    parser.add_argument(
        "--n-eval",
        type=int,
        default=1_000,
        help="Eval set size when --split=both (default: 1000)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/synthetic"),
        help="Directory to write JSONL files (default: data/synthetic/)",
    )
    parser.add_argument(
        "--model",
        default="claude-haiku-4-5-20251001",
        help="Claude model ID to use (default: claude-haiku-4-5-20251001)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible scenario/domain sampling (default: 42)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JSONL files (default: skip if file already exists)",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai"],
        default="anthropic",
        help=(
            "API provider (default: anthropic). Use 'openai' for any "
            "OpenAI-compatible server such as vLLM running a local Qwen model."
        ),
    )
    parser.add_argument(
        "--api-base-url",
        default=None,
        help=(
            "Base URL for an OpenAI-compatible server (e.g. http://HOST:8000/v1). "
            "Required when --provider=openai."
        ),
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "API key override.  Defaults to ANTHROPIC_API_KEY env var for "
            "--provider=anthropic, or 'EMPTY' for --provider=openai."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help=(
            "Number of concurrent API calls. "
            "Defaults to 2 for --provider=anthropic (safe for 50 RPM standard tier) "
            "and 20 for --provider=openai (no external rate limit on local server)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the generation plan without making any API calls",
    )
    return parser.parse_args()


def _print_diversity_report(n: int, split: str) -> None:
    """Print the expected diversity statistics for the planned generation."""
    n_scenario_types = len(SCENARIO_TYPES)
    n_domains = len(DOMAINS)
    n_tool_count_options = 4  # [1, 2, 3, 4]
    n_unique_seeds = n_scenario_types * n_domains * n_tool_count_options
    repetitions = n / max(n_unique_seeds, 1)

    print(f"\n{'='*60}")
    print(f"  Diversity report — {split} split ({n:,} examples)")
    print(f"{'='*60}")
    print(f"  Scenario types:          {n_scenario_types}")
    print(f"  Domains:                 {n_domains}")
    print(f"  Tool-call-count options: {n_tool_count_options}")
    print(f"  Unique prompt seeds:     {n_unique_seeds:,}")
    print(f"  Expected repetitions:    {repetitions:.1f}× per seed")
    print(f"  Note: Claude produces unique output even for repeated seeds,")
    print(f"        so effective diversity is higher than the seed count.")
    if repetitions < 1.0:
        print(f"  ✓ Each unique seed sampled <once on average — maximum diversity.")
    elif repetitions < 5.0:
        print(f"  ✓ Very low repetition rate — effectively unique data.")
    elif repetitions < 20.0:
        print(f"  ⚠ Moderate repetition — consider increasing n for better diversity.")
    else:
        print(f"  ✗ High repetition — generating more examples than the seed space.")
    print(f"{'='*60}\n")


def _generate_split(
    split_name: str,
    n: int,
    output_path: Path,
    model: str,
    seed: int,
    overwrite: bool,
    dry_run: bool,
    max_workers: int = 2,
    provider: str = "anthropic",
    api_base_url: str | None = None,
    api_key: str | None = None,
) -> None:
    """Generate one split (train or eval)."""
    _print_diversity_report(n, split_name)

    if dry_run:
        logger.info(
            "[DRY RUN] Would generate %d %s examples → %s", n, split_name, output_path
        )
        return

    if output_path.exists() and not overwrite:
        existing = load_generated_dataset(output_path)
        logger.info(
            "File %s already exists with %d examples. Use --overwrite to regenerate.",
            output_path,
            len(existing),
        )
        return

    if api_key is None:
        if provider == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                logger.error("ANTHROPIC_API_KEY environment variable is not set.")
                sys.exit(1)
        else:
            api_key = "EMPTY"  # vLLM / local servers don't require a real key

    if provider == "openai" and not api_base_url:
        logger.error("--api-base-url is required when --provider=openai.")
        sys.exit(1)

    logger.info(
        "Generating %d %s examples using %s (%s) → %s",
        n,
        split_name,
        model,
        provider,
        output_path,
    )
    n_written = generate_dataset(
        n=n,
        output_path=output_path,
        api_key=api_key,
        seed=seed,
        model=model,
        max_workers=max_workers,
        provider=provider,
        api_base_url=api_base_url,
    )

    success_rate = n_written / max(n, 1) * 100
    logger.info(
        "%s split complete: %d / %d examples written (%.1f%% success rate)",
        split_name,
        n_written,
        n,
        success_rate,
    )

    if success_rate < 80:
        logger.warning(
            "Success rate below 80%%. Check for JSON parse errors in the model output. "
            "Try a different model or adjust the generation prompt in synthetic_generator.py"
        )

    # Write a brief manifest alongside the JSONL for auditability
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest = {
        "split": split_name,
        "n_requested": n,
        "n_written": n_written,
        "success_rate_pct": round(success_rate, 2),
        "model": model,
        "provider": provider,
        "api_base_url": api_base_url,
        "seed": seed,
        "n_scenario_types": len(SCENARIO_TYPES),
        "n_domains": len(DOMAINS),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Manifest written to %s", manifest_path)


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default concurrency: conservative for Anthropic (rate-limited), aggressive for local.
    max_workers = args.max_workers
    if max_workers is None:
        max_workers = 2 if args.provider == "anthropic" else 20

    common = dict(
        model=args.model,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        max_workers=max_workers,
        provider=args.provider,
        api_base_url=args.api_base_url,
        api_key=args.api_key,
    )

    if args.split in ("train", "both"):
        n_train = args.n if (args.n is not None and args.split == "train") else args.n_train
        _generate_split(
            split_name="train",
            n=n_train,
            output_path=output_dir / "train_examples.jsonl",
            seed=args.seed,
            **common,
        )

    if args.split in ("eval", "both"):
        n_eval = args.n if (args.n is not None and args.split == "eval") else args.n_eval
        _generate_split(
            split_name="eval",
            n=n_eval,
            output_path=output_dir / "eval_examples.jsonl",
            seed=args.seed + 1,  # Different seed to avoid overlap with training data
            **common,
        )


if __name__ == "__main__":
    main()

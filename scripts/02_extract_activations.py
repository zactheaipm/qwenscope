"""Extract and cache a small sample of activations from the TRAINING distribution.

Streams a small slice of training-distribution data (default 1M tokens) through
the model and saves the resulting activation vectors. This cache is useful for
fast spot-checks (e.g. manual feature exploration, quick sanity tests) without
rerunning the full forward pass each time.

IMPORTANT: this script uses ``build_dataset()`` (the training distribution).
The output is deliberately named ``train_sample_activations_layer{N}.safetensors``
to avoid confusion with held-out evaluation data. For quality evaluation use
``04_evaluate_sae_quality.py``, which runs over UltraChat test_sft (chat eval)
and the held-out tool-use eval templates (tool-use eval).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract activations for SAE evaluation")
    parser.add_argument("--layer", type=int, default=32, help="Layer to extract from")
    parser.add_argument("--n-tokens", type=int, default=1_000_000, help="Tokens to extract")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument("--output-dir", default=os.environ.get("ACTIVATIONS_DIR", "data/activations"))
    parser.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", "data/results"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing output.
    # Named train_sample_* to make clear this is a training-distribution sample,
    # not held-out eval data. Using it as an eval set would leak train→eval.
    output_path = output_dir / f"train_sample_activations_layer{args.layer}.safetensors"
    if output_path.exists():
        logger.info("Activations already cached at %s. Skipping.", output_path)
        return

    from src.model.loader import load_model
    from src.sae.activations import ActivationStream
    from src.data.training_data import SAETrainingDataBuilder
    from src.sae.config import SAETrainingConfig

    logger.info("Loading model for activation extraction...")
    model, tokenizer = load_model(dtype="bfloat16", device=args.device)

    logger.info("Building training data iterator...")
    config = SAETrainingConfig(layer=args.layer, sae_id=f"eval_layer{args.layer}")
    data_builder = SAETrainingDataBuilder(tokenizer, config)
    dataset = data_builder.build_dataset()
    data_iter = iter(dataset)

    # Batch the data
    def batch_iterator(data_iter, batch_size=16):
        batch = []
        for item in data_iter:
            batch.append(item)
            if len(batch) == batch_size:
                yield {
                    "input_ids": torch.stack([b["input_ids"] for b in batch]),
                    "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
                }
                batch = []

    stream = ActivationStream(
        model=model,
        tokenizer=tokenizer,
        layer=args.layer,
        dataset_iter=batch_iterator(data_iter),
        device=args.device,
    )

    logger.info("Extracting activations at layer %d (%d tokens)...", args.layer, args.n_tokens)
    all_acts = []
    for acts_batch in stream.stream_tokens(args.n_tokens):
        all_acts.append(acts_batch.cpu())
        if stream.tokens_processed % 1_000_000 < 100_000:
            logger.info("Progress: %d / %d tokens", stream.tokens_processed, args.n_tokens)

    if all_acts:
        combined = torch.cat(all_acts, dim=0)
        logger.info("Saving %d activation vectors to %s", combined.shape[0], output_path)
        save_file({"activations": combined}, str(output_path))
    else:
        logger.warning("No activations extracted.")

    # Write manifest
    manifest = {
        "script": "02_extract_activations",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "layer": args.layer,
        "tokens_extracted": stream.tokens_processed,
        "output_path": str(output_path),
    }
    with open(results_dir / "02_extract_activations.json", "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Done!")


if __name__ == "__main__":
    main()

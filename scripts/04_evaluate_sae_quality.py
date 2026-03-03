"""Evaluate reconstruction quality of trained SAEs.

Runs two eval passes per SAE:

1. **General chat eval** (UltraChat test_sft — held-out from training)
   Measures reconstruction quality on typical instruction-following sequences.

2. **Tool-use eval** (TOOL_USE_EVAL_TEMPLATES — never seen during training)
   Measures reconstruction quality on actual tool-calling token sequences
   (<tool_call> / </tool_call> tokens). This is the distribution that matters
   most for the steering experiments; a SAE that fails here would corrupt
   the residual at tool-decision positions without any chat-eval warning.

Metrics reported per pass: MSE, explained variance, L0 sparsity, dead
features %, CE loss original / with SAE / zero-ablated, loss recovered,
activation frequency Gini.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _build_loader(dataset, batch_size: int = 8):
    """Wrap an IterableDataset in a DataLoader with simple collation."""
    import torch
    from torch.utils.data import DataLoader

    def collate_fn(batch):
        return {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        }

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SAE reconstruction quality")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", "data/results"))
    parser.add_argument(
        "--n-batches",
        type=int,
        default=1000,
        help=(
            "Number of batches (×8 sequences each) per eval pass. "
            "At 1000 batches × 8 × 2048 tokens = ~16M tokens, dead-feature "
            "counts and frequency histograms are stable. "
            "50 batches (the old default) is too small for reliable estimates. "
            "Minimum recommended: 500."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run even if the manifest already exists.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = results_dir / "04_sae_quality.json"
    if manifest_path.exists() and not args.force:
        logger.info("Quality evaluation already done. Use --force to re-run.")
        return

    from src.model.config import HOOK_POINTS
    from src.model.loader import load_model
    from src.sae.model import TopKSAE
    from src.sae.quality import compute_reconstruction_metrics
    from src.data.training_data import SAETrainingDataBuilder
    from src.sae.config import SAETrainingConfig

    model, tokenizer = load_model(dtype="bfloat16", device=args.device)

    all_metrics: dict[str, dict[str, dict]] = {}

    for hp in HOOK_POINTS:
        sae_path = Path(f"data/saes/{hp.sae_id}")
        if not (sae_path / "weights.safetensors").exists():
            logger.warning("SAE %s not found at %s. Skipping.", hp.sae_id, sae_path)
            continue

        sae = TopKSAE.load(sae_path, device=args.device)
        logger.info("Evaluating %s (layer %d)...", hp.sae_id, hp.layer)

        config = SAETrainingConfig(layer=hp.layer, sae_id=hp.sae_id)
        data_builder = SAETrainingDataBuilder(tokenizer, config)

        all_metrics[hp.sae_id] = {}

        # ------------------------------------------------------------------ #
        # Pass 1: general chat eval (UltraChat test_sft — held-out split)
        # ------------------------------------------------------------------ #
        logger.info("  [%s] Pass 1/2: general chat eval (UltraChat test_sft)...", hp.sae_id)
        chat_loader = _build_loader(data_builder.build_eval_dataset(), batch_size=8)
        chat_metrics = compute_reconstruction_metrics(
            model, sae, hp.layer, chat_loader,
            n_batches=args.n_batches, device=args.device,
        )
        all_metrics[hp.sae_id]["chat"] = chat_metrics
        logger.info(
            "  [%s] Chat: MSE=%.4f, EV=%.4f, L0=%.1f, LossRecovered=%.4f, Gini=%.3f",
            hp.sae_id,
            chat_metrics["mse"],
            chat_metrics["explained_variance"],
            chat_metrics["l0_sparsity"],
            chat_metrics["loss_recovered"],
            chat_metrics["freq_gini"],
        )

        # ------------------------------------------------------------------ #
        # Pass 2: tool-use eval (held-out eval templates, never in training)
        # ------------------------------------------------------------------ #
        logger.info("  [%s] Pass 2/2: tool-use eval (held-out templates)...", hp.sae_id)
        tool_loader = _build_loader(data_builder.build_tool_use_eval_dataset(), batch_size=8)
        tool_metrics = compute_reconstruction_metrics(
            model, sae, hp.layer, tool_loader,
            n_batches=args.n_batches, device=args.device,
        )
        all_metrics[hp.sae_id]["tool_use"] = tool_metrics
        logger.info(
            "  [%s] Tool-use: MSE=%.4f, EV=%.4f, L0=%.1f, LossRecovered=%.4f, Gini=%.3f",
            hp.sae_id,
            tool_metrics["mse"],
            tool_metrics["explained_variance"],
            tool_metrics["l0_sparsity"],
            tool_metrics["loss_recovered"],
            tool_metrics["freq_gini"],
        )

        # Flag large gaps between the two distributions — a warning that the SAE
        # generalises poorly to tool-calling sequences.
        ev_gap = chat_metrics["explained_variance"] - tool_metrics["explained_variance"]
        lr_gap = chat_metrics["loss_recovered"] - tool_metrics["loss_recovered"]
        if ev_gap > 0.05 or lr_gap > 0.05:
            logger.warning(
                "  [%s] DISTRIBUTION GAP: chat EV=%.3f vs tool EV=%.3f "
                "(gap=%.3f); chat LR=%.3f vs tool LR=%.3f (gap=%.3f). "
                "SAE may be poorly calibrated for tool-calling sequences.",
                hp.sae_id,
                chat_metrics["explained_variance"],
                tool_metrics["explained_variance"],
                ev_gap,
                chat_metrics["loss_recovered"],
                tool_metrics["loss_recovered"],
                lr_gap,
            )

    manifest = {
        "script": "04_evaluate_sae_quality",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_batches_per_pass": args.n_batches,
        "metrics": all_metrics,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Quality evaluation complete. Results at %s", manifest_path)


if __name__ == "__main__":
    main()

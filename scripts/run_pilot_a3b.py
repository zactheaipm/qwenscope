"""Pilot SAE training for A3B: single hook point, 5M tokens.

Quick validation that the SAE training pipeline works end-to-end on the
A3B model before committing to the full 200M-token, 9-SAE run.

Trains sae_attn_mid (layer 23, attention, block 5) with 5M tokens.
This is the same hook point that scored highest TAS on the 27B pilot.

Usage:
    PYTHONPATH=/workspace/qwenscope python scripts/run_pilot_a3b.py
    PYTHONPATH=/workspace/qwenscope python scripts/run_pilot_a3b.py --sae-id sae_delta_mid
    PYTHONPATH=/workspace/qwenscope python scripts/run_pilot_a3b.py --tokens 10_000_000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="A3B pilot SAE training")
    parser.add_argument("--sae-id", default="sae_attn_mid", help="SAE to train")
    parser.add_argument("--tokens", type=int, default=5_000_000, help="Training tokens")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"), help="Device")
    parser.add_argument("--output-dir", default="data/saes_pilot", help="Output directory")
    parser.add_argument("--config", default="configs/sae_training.yaml", help="Config path")
    args = parser.parse_args()

    from src.model.config import HOOK_POINTS_BY_ID
    from src.model.loader import load_model
    from src.sae.config import SAETrainingConfig
    from src.sae.model import TopKSAE
    from src.sae.trainer import SAETrainer
    from src.sae.activations import ActivationStream
    from src.data.training_data import SAETrainingDataBuilder

    hp = HOOK_POINTS_BY_ID[args.sae_id]
    logger.info("=== A3B Pilot SAE Training ===")
    logger.info("SAE: %s (layer %d, %s, block %d)", hp.sae_id, hp.layer, hp.layer_type.value, hp.block)
    logger.info("Training tokens: %d", args.tokens)

    # Load config with per-hook overrides
    config = SAETrainingConfig.from_yaml(Path(args.config), args.sae_id)
    # Override training tokens for pilot
    config.training_tokens = args.tokens
    # Smaller buffer for pilot (500K is fine for 5M tokens)
    config.buffer_capacity = min(config.buffer_capacity, 500_000)
    # Checkpoint every 1M tokens for pilot
    config.checkpoint_every_tokens = 1_000_000

    logger.info("Config: hidden_dim=%d, dict_size=%d, k=%d, lr=%e, batch=%d",
                config.hidden_dim, config.dictionary_size, config.topk,
                config.learning_rate, config.batch_size)

    # Load model
    logger.info("Loading model...")
    start = time.time()
    model, tokenizer = load_model(dtype="bfloat16", device=args.device)
    logger.info("Model loaded in %.1fs", time.time() - start)

    vram_model = torch.cuda.memory_allocated() / 1e9
    logger.info("VRAM after model load: %.2f GB", vram_model)

    # Create SAE
    sae = TopKSAE(
        hidden_dim=config.hidden_dim,
        dict_size=config.dictionary_size,
        k=config.topk,
    ).to(args.device)

    sae_params = sum(p.numel() for p in sae.parameters())
    sae_vram = sae_params * 2 / 1e9  # BF16
    logger.info("SAE: %d params (%.2f GB), dict_size=%d, k=%d",
                sae_params, sae_vram, config.dictionary_size, config.topk)

    # Build training data
    logger.info("Building training data...")
    data_builder = SAETrainingDataBuilder(tokenizer, config)
    dataset = data_builder.build_dataset()

    # DataLoader batches individual sequences into (B, S) tensors
    def collate_fn(batch):
        return {
            k: torch.stack([b[k] for b in batch])
            for k in batch[0].keys()
            if isinstance(batch[0][k], torch.Tensor)
        }

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, collate_fn=collate_fn,
    )

    # Create activation stream
    stream = ActivationStream(
        model=model,
        tokenizer=tokenizer,
        layer=hp.layer,
        dataset_iter=iter(loader),
        batch_size=8,  # Small batch for pilot (less VRAM)
        max_seq_length=config.max_seq_length,
        device=args.device,
    )

    # Create trainer
    trainer = SAETrainer(sae, config)

    # Initialize WandB
    try:
        trainer.init_wandb(
            project="qwen35-scope",
            run_name=f"pilot_a3b_{args.sae_id}_{int(time.time())}",
        )
    except Exception as e:
        logger.warning("WandB init failed: %s", e)

    # Train
    logger.info("Starting training...")
    train_start = time.time()
    output_dir = Path(args.output_dir) / args.sae_id
    checkpoint_dir = output_dir / "checkpoints"

    trained_sae = trainer.train(stream, checkpoint_dir=checkpoint_dir)

    train_elapsed = time.time() - train_start
    logger.info("Training complete in %.1fs (%.0f tok/s)",
                train_elapsed, args.tokens / train_elapsed)

    # Save final model
    output_dir.mkdir(parents=True, exist_ok=True)
    trained_sae.save(output_dir)
    logger.info("Saved to %s", output_dir)

    # Quick quality metrics
    logger.info("=== Quick Quality Check ===")
    from src.sae.quality import compute_reconstruction_metrics

    eval_dataset = data_builder.build_eval_dataset()
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=4,
        collate_fn=lambda batch: {
            "input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        },
    )

    metrics = compute_reconstruction_metrics(
        model, trained_sae, hp.layer, eval_loader,
        n_batches=20, device=args.device,
    )

    logger.info("EV: %.4f", metrics["explained_variance"])
    logger.info("Loss recovered: %.4f", metrics["loss_recovered"])
    logger.info("Dead features: %.1f%%", metrics["dead_feature_pct"])
    logger.info("MSE: %.6f", metrics["mse"])
    logger.info("L0 sparsity: %.1f", metrics["l0_sparsity"])

    vram_peak = torch.cuda.max_memory_allocated() / 1e9
    logger.info("Peak VRAM: %.2f GB", vram_peak)

    # Save manifest
    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "script": "run_pilot_a3b",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sae_id": args.sae_id,
        "layer": hp.layer,
        "layer_type": hp.layer_type.value,
        "block": hp.block,
        "training_tokens": args.tokens,
        "train_elapsed_s": round(train_elapsed, 1),
        "config": {
            "hidden_dim": config.hidden_dim,
            "dictionary_size": config.dictionary_size,
            "topk": config.topk,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
        },
        "metrics": {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
        "vram_peak_gb": round(vram_peak, 2),
    }
    manifest_path = results_dir / "pilot_a3b.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest saved to %s", manifest_path)
    logger.info("=== Pilot COMPLETE ===")


if __name__ == "__main__":
    main()

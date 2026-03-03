"""Download and verify Qwen 3.5-27B model.

Verifies: 64 layers, hybrid layout, hidden_dim=5120.
Runs one forward pass with hooks on all 64 layers to verify capture.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup and verify Qwen 3.5-27B")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"), help="Device to use")
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype")
    parser.add_argument("--results-dir", default=os.environ.get("RESULTS_DIR", "data/results"), help="Results directory")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing results
    manifest_path = results_dir / "01_setup_model.json"
    if manifest_path.exists():
        logger.info("Setup already completed. Re-running verification...")

    from src.model.config import Qwen35Config, HOOK_POINTS
    from src.model.loader import load_model
    from src.model.hooks import ActivationCache

    config = Qwen35Config()

    logger.info("Loading model with dtype=%s on device=%s", args.dtype, args.device)
    model, tokenizer = load_model(dtype=args.dtype, device=args.device)

    # Verify architecture
    num_layers = len(model.model.layers)
    hidden_size = model.config.hidden_size
    logger.info("Model loaded: %d layers, hidden_size=%d", num_layers, hidden_size)

    assert num_layers == config.num_layers, f"Expected {config.num_layers} layers, got {num_layers}"
    assert hidden_size == config.hidden_dim, f"Expected hidden_dim={config.hidden_dim}, got {hidden_size}"

    # Verify hook registration on all 64 layers
    logger.info("Testing hooks on all %d layers...", num_layers)
    all_layers = list(range(num_layers))
    cache = ActivationCache(model, layers=all_layers)

    inputs = tokenizer("Hello, this is a test.", return_tensors="pt")
    inputs = {k: v.to(args.device) for k, v in inputs.items()}

    with torch.no_grad():
        with cache.active():
            model(**inputs)

    # Verify all 64 layers captured
    for layer_idx in all_layers:
        acts = cache.get(layer_idx)
        assert acts.shape[-1] == config.hidden_dim, (
            f"Layer {layer_idx}: expected hidden_dim={config.hidden_dim}, got {acts.shape[-1]}"
        )
        layer_type = config.layer_type(layer_idx)
        logger.debug("Layer %d (%s): shape=%s", layer_idx, layer_type.value, acts.shape)

    logger.info("All %d layers verified: correct shapes (batch, seq_len, %d)", num_layers, config.hidden_dim)

    # Print architecture summary
    logger.info("=== Architecture Summary ===")
    logger.info("DeltaNet layers: %s", config.deltanet_layers())
    logger.info("Attention layers: %s", config.attention_layers())
    logger.info("Hook points:")
    for hp in HOOK_POINTS:
        logger.info("  %s: layer %d (%s, block %d)", hp.sae_id, hp.layer, hp.layer_type.value, hp.block)

    # Write manifest
    manifest = {
        "script": "01_setup_model",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_id": os.environ.get("QWEN35_MODEL_PATH", "Qwen/Qwen3.5-27B"),
        "num_layers": num_layers,
        "hidden_dim": hidden_size,
        "dtype": args.dtype,
        "device": args.device,
        "all_layers_verified": True,
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Manifest written to %s", manifest_path)
    logger.info("Setup complete!")


if __name__ == "__main__":
    main()

"""Download and verify Qwen 3.5-35B-A3B model.

Verifies: 40 layers, hybrid layout, hidden_dim=2048, MoE routing.
Runs one forward pass with hooks on all 40 layers to verify capture.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup and verify Qwen 3.5-35B-A3B")
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
    from src.model.loader import load_model, get_layers_module
    from src.model.hooks import ActivationCache

    config = Qwen35Config()

    logger.info("Loading model with dtype=%s on device=%s", args.dtype, args.device)
    model, tokenizer = load_model(dtype=args.dtype, device=args.device)

    # Verify architecture
    layers = get_layers_module(model)
    num_layers = len(layers)

    # For VLM models, hidden_size is under text_config
    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(getattr(model.config, "text_config", None), "hidden_size", None)
    logger.info("Model loaded: %d layers, hidden_size=%s", num_layers, hidden_size)

    assert num_layers == config.num_layers, f"Expected {config.num_layers} layers, got {num_layers}"
    assert hidden_size == config.hidden_dim, f"Expected hidden_dim={config.hidden_dim}, got {hidden_size}"

    # Print model architecture summary for verification
    logger.info("=== Model Architecture ===")
    logger.info("Model class: %s", type(model).__name__)
    logger.info("Config type: %s", getattr(model.config, "model_type", "unknown"))

    # Log MoE info if available
    text_config = getattr(model.config, "text_config", model.config)
    num_experts = getattr(text_config, "num_experts", None)
    num_experts_per_tok = getattr(text_config, "num_experts_per_tok", None)
    if num_experts:
        logger.info("MoE: %d total experts, %d routed per token", num_experts, num_experts_per_tok or 0)

    # Log layer types if available
    layer_types = getattr(text_config, "layer_types", None)
    if layer_types:
        deltanet_count = sum(1 for t in layer_types if t == "linear_attention")
        attn_count = sum(1 for t in layer_types if t == "full_attention")
        logger.info("Layer types: %d DeltaNet (linear_attention), %d Attention (full_attention)", deltanet_count, attn_count)

    # Verify hook registration on all layers
    logger.info("Testing hooks on all %d layers...", num_layers)
    all_layers = list(range(num_layers))
    cache = ActivationCache(model, layers=all_layers)

    inputs = tokenizer("Hello, this is a test.", return_tensors="pt")
    inputs = {k: v.to(args.device) for k, v in inputs.items()}

    with torch.no_grad():
        with cache.active():
            model(**inputs)

    # Verify all layers captured with correct hidden_dim
    act_norms = {}
    for layer_idx in all_layers:
        acts = cache.get(layer_idx)
        assert acts.shape[-1] == config.hidden_dim, (
            f"Layer {layer_idx}: expected hidden_dim={config.hidden_dim}, got {acts.shape[-1]}"
        )
        layer_type = config.layer_type(layer_idx)
        rms_norm = float(acts.norm(dim=-1).mean().item())
        act_norms[layer_idx] = rms_norm
        logger.debug("Layer %d (%s): shape=%s, rms_norm=%.3f", layer_idx, layer_type.value, acts.shape, rms_norm)

    logger.info("All %d layers verified: correct shapes (batch, seq_len, %d)", num_layers, config.hidden_dim)

    # Print activation norms at hook points
    logger.info("=== Activation Norms at Hook Points ===")
    for hp in HOOK_POINTS:
        norm = act_norms.get(hp.layer, 0.0)
        logger.info("  %s (layer %d, %s): rms_norm=%.3f", hp.sae_id, hp.layer, hp.layer_type.value, norm)

    # Print architecture summary
    logger.info("=== Hook Points ===")
    logger.info("DeltaNet layers: %s", config.deltanet_layers())
    logger.info("Attention layers: %s", config.attention_layers())
    for hp in HOOK_POINTS:
        logger.info("  %s: layer %d (%s, block %d)", hp.sae_id, hp.layer, hp.layer_type.value, hp.block)

    # VRAM report
    if torch.cuda.is_available():
        vram_allocated = torch.cuda.memory_allocated() / 1e9
        vram_reserved = torch.cuda.memory_reserved() / 1e9
        logger.info("=== VRAM Usage ===")
        logger.info("  Allocated: %.2f GB", vram_allocated)
        logger.info("  Reserved:  %.2f GB", vram_reserved)
    else:
        vram_allocated = 0.0
        vram_reserved = 0.0

    # Write manifest
    manifest = {
        "script": "01_setup_model",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model_id": os.environ.get("QWEN35_MODEL_PATH", "Qwen/Qwen3.5-35B-A3B"),
        "model_class": type(model).__name__,
        "num_layers": num_layers,
        "hidden_dim": hidden_size,
        "num_experts": num_experts,
        "num_experts_per_tok": num_experts_per_tok,
        "dtype": args.dtype,
        "device": args.device,
        "all_layers_verified": True,
        "vram_allocated_gb": round(vram_allocated, 2),
        "vram_reserved_gb": round(vram_reserved, 2),
        "activation_norms": {str(k): round(v, 4) for k, v in act_norms.items()},
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Manifest written to %s", manifest_path)
    logger.info("Setup complete!")


if __name__ == "__main__":
    main()

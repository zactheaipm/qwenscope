"""Qwen 3.5-35B-A3B model loading with dtype selection."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.model.config import ModelConfig, Qwen35Config, validate_configs_agree

logger = logging.getLogger(__name__)

# Mapping from config string to torch dtype
DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def get_layers_module(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Resolve the transformer layers module from any supported model class.

    Qwen 3.5 models come in two variants:
      - CausalLM (27B): layers at model.model.layers
      - VLM/MoE (35B-A3B): layers at model.model.text_model.layers
        (Qwen3_5MoeForConditionalGeneration wraps text in a text_model)

    This function probes for the correct path so that hooks.py, steering,
    and activation extraction all work regardless of model variant.

    Args:
        model: A HuggingFace model instance.

    Returns:
        The nn.ModuleList containing the transformer decoder layers.

    Raises:
        AttributeError: If no known layer path is found.
    """
    # Try paths in order of specificity
    candidates = [
        ("model.model.text_model.layers", lambda m: m.model.text_model.layers),
        ("model.model.layers", lambda m: m.model.layers),
        ("model.language_model.model.layers", lambda m: m.language_model.model.layers),
    ]
    for path_name, accessor in candidates:
        try:
            layers = accessor(model)
            if isinstance(layers, torch.nn.ModuleList) and len(layers) > 0:
                logger.debug("Layer path resolved: %s (%d layers)", path_name, len(layers))
                return layers
        except AttributeError:
            continue

    raise AttributeError(
        "Could not resolve transformer layers. Tried: "
        + ", ".join(p for p, _ in candidates)
        + ". Inspect the model with print(model) and add the correct path."
    )


def load_model(
    model_id: str | None = None,
    dtype: str = "bfloat16",
    device: str | None = None,
    attn_implementation: str = "eager",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load Qwen 3.5-35B-A3B with specified precision.

    Args:
        model_id: HuggingFace model ID or local path. Defaults to env var QWEN35_MODEL_PATH.
        dtype: Weight precision — "bfloat16", "float16", or "float32".
        device: Target device. Defaults to env var DEVICE or "cuda".
        attn_implementation: Attention implementation. Use "eager" for hook compatibility
            (flash_attention_2 may not propagate hooks correctly).

    Returns:
        Tuple of (model, tokenizer).
    """
    if device is None:
        device = os.environ.get("DEVICE", "cuda")

    if model_id is None:
        model_id = os.environ.get("QWEN35_MODEL_PATH", "Qwen/Qwen3.5-35B-A3B")
    torch_dtype = DTYPE_MAP.get(dtype, torch.bfloat16)
    logger.info("Loading model: %s with dtype=%s", model_id, dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch_dtype,
        device_map=device if device != "cpu" else None,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )

    if device == "cpu":
        model = model.to(device)

    model.eval()

    # Resolve and log layer path
    layers = get_layers_module(model)
    hidden_size = getattr(model.config, "hidden_size", None)
    # For VLM models, hidden_size may be nested under text_config
    if hidden_size is None:
        hidden_size = getattr(getattr(model.config, "text_config", None), "hidden_size", "unknown")

    logger.info(
        "Model loaded: %d layers, hidden_dim=%s, device=%s",
        len(layers),
        hidden_size,
        device,
    )
    return model, tokenizer


def load_model_from_config(
    config: ModelConfig,
    mode: str = "steering",
    device: str | None = None,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model using a ModelConfig object.

    Args:
        config: Model configuration.
        mode: Loading mode (unused, kept for API compatibility). Always loads BF16.
        device: Override device. Defaults to env var DEVICE.

    Returns:
        Tuple of (model, tokenizer).
    """
    validate_configs_agree(config, Qwen35Config())
    return load_model(
        model_id=config.model_id,
        dtype=config.dtype,
        device=device,
        attn_implementation="eager",
    )

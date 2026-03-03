"""Qwen 3.5-27B model loading with dtype selection."""

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


def load_model(
    model_id: str | None = None,
    dtype: str = "bfloat16",
    device: str | None = None,
    attn_implementation: str = "eager",
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load Qwen 3.5-27B with specified precision.

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
        model_id = os.environ.get("QWEN35_MODEL_PATH", "Qwen/Qwen3.5-27B")
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
    logger.info(
        "Model loaded: %d layers, hidden_dim=%d, device=%s",
        len(model.model.layers),
        model.config.hidden_size,
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

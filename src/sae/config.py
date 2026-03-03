"""SAE hyperparameters and training configuration."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

from src.model.config import LayerType


class SAETrainingConfig(BaseModel):
    """Configuration for SAE training."""

    sae_type: str = "topk"
    hidden_dim: int = 5120
    dictionary_size: int = 40960
    topk: int = 64
    learning_rate: float = 5e-5
    lr_warmup_steps: int = 1000
    batch_size: int = 4096
    training_tokens: int = 200_000_000
    checkpoint_every_tokens: int = 50_000_000
    seed: int = 42

    # Activation buffer capacity (number of vectors).
    # Determines the effective shuffle window: 500K vectors × 5120 × 2 bytes ≈ 5 GB CPU RAM.
    # Larger values give better mixing at the cost of RAM. 100K (the old default) only covered
    # ~0.05% of the 200M-token training run; 500K raises that to ~0.25%.
    buffer_capacity: int = 500_000

    # How often to check for and resample dead features (in optimizer steps).
    # With 200M tokens / 4096 batch ≈ 48,828 total steps, 5000-step intervals give
    # ~9 resampling opportunities vs. only ~1 at the old 25,000-step default.
    resample_every_n_steps: int = 5_000

    # Hook point for this SAE
    layer: int = 0
    layer_type: LayerType = LayerType.DELTANET
    sae_id: str = ""

    @classmethod
    def from_yaml(cls, path: Path, sae_id: str) -> SAETrainingConfig:
        """Load configuration from YAML for a specific SAE.

        Args:
            path: Path to sae_training.yaml.
            sae_id: Which SAE to configure (e.g., "sae_attn_mid").

        Returns:
            SAETrainingConfig for the specified SAE.
        """
        with open(path) as f:
            data = yaml.safe_load(f)

        hook_point = data["hook_points"][sae_id]
        return cls(
            sae_type=data.get("sae_type", "topk"),
            dictionary_size=data.get("dictionary_size", 40960),
            topk=data.get("topk", 64),
            learning_rate=data.get("learning_rate", 5e-5),
            lr_warmup_steps=data.get("lr_warmup_steps", 1000),
            batch_size=data.get("batch_size", 4096),
            training_tokens=data.get("training_tokens", 200_000_000),
            checkpoint_every_tokens=data.get("checkpoint_every_tokens", 50_000_000),
            seed=data.get("seed", 42),
            buffer_capacity=data.get("buffer_capacity", 500_000),
            resample_every_n_steps=data.get("resample_every_n_steps", 5_000),
            layer=hook_point["layer"],
            layer_type=LayerType(hook_point["type"]),
            sae_id=sae_id,
        )

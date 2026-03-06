"""SAE hyperparameters and training configuration."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

from src.model.config import LayerType


class SAETrainingConfig(BaseModel):
    """Configuration for SAE training."""

    sae_type: str = "topk"
    hidden_dim: int = 2048
    dictionary_size: int = 16384
    topk: int = 64
    learning_rate: float = 5e-5
    lr_warmup_steps: int = 1000
    batch_size: int = 4096
    training_tokens: int = 200_000_000
    checkpoint_every_tokens: int = 50_000_000
    seed: int = 42

    # Activation buffer capacity (number of vectors).
    # Determines the effective shuffle window: 2M vectors × 2048 × 2 bytes ≈ 8 GB CPU RAM.
    # Larger values give better mixing at the cost of RAM. Covers 1% of the 200M-token run,
    # critical because WildChat/UltraChat are streamed in dataset order (not pre-shuffled).
    buffer_capacity: int = 2_000_000

    # How often to check for and resample dead features (in optimizer steps).
    # With 200M tokens / 4096 batch ≈ 48,828 total steps, 5000-step intervals give
    # ~9 resampling opportunities vs. only ~1 at the old 25,000-step default.
    resample_every_n_steps: int = 5_000

    # Maximum sequence length for tokenization and packing.
    max_seq_length: int = 2048

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

        # Per-hook overrides: any top-level key can be overridden inside
        # the hook_point block (e.g., topk, dictionary_size, resample_every_n_steps).
        def _get(key: str, default):
            return hook_point.get(key, data.get(key, default))

        # Load hidden_dim from model.yaml if it exists, otherwise fall back
        # to the default. This prevents silent dimension mismatch if the model
        # changes to another architecture.
        model_yaml = path.parent / "model.yaml"
        default_hidden_dim = 2048
        if model_yaml.exists():
            with open(model_yaml) as mf:
                model_data = yaml.safe_load(mf)
            default_hidden_dim = model_data.get("hidden_dim", default_hidden_dim)

        return cls(
            sae_type=_get("sae_type", "topk"),
            hidden_dim=_get("hidden_dim", default_hidden_dim),
            dictionary_size=_get("dictionary_size", 16384),
            topk=_get("topk", 64),
            learning_rate=_get("learning_rate", 5e-5),
            lr_warmup_steps=_get("lr_warmup_steps", 1000),
            batch_size=_get("batch_size", 4096),
            training_tokens=_get("training_tokens", 200_000_000),
            checkpoint_every_tokens=_get("checkpoint_every_tokens", 50_000_000),
            seed=_get("seed", 42),
            buffer_capacity=_get("buffer_capacity", 500_000),
            resample_every_n_steps=_get("resample_every_n_steps", 5_000),
            max_seq_length=_get("max_seq_length", 2048),
            layer=hook_point["layer"],
            layer_type=LayerType(hook_point["type"]),
            sae_id=sae_id,
        )

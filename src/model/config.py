"""Qwen 3.5-27B architecture constants and configuration."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel


class LayerType(str, Enum):
    """Layer type in the Qwen 3.5 hybrid architecture."""

    DELTANET = "deltanet"
    ATTENTION = "attention"


class HookPoint(BaseModel):
    """Definition of a single SAE hook point in the model."""

    sae_id: str
    layer: int
    layer_type: LayerType
    block: int
    description: str


class Qwen35Config(BaseModel):
    """Architecture constants for Qwen 3.5-27B.

    The model uses a repeating 4-layer block pattern:
      Block N (×16):
        Layer 4N+0: Gated DeltaNet → FFN
        Layer 4N+1: Gated DeltaNet → FFN
        Layer 4N+2: Gated DeltaNet → FFN
        Layer 4N+3: Gated Attention → FFN
    """

    hidden_dim: int = 5120
    num_layers: int = 64
    block_size: int = 4
    num_blocks: int = 16
    context_length: int = 4096

    def layer_type(self, layer_idx: int) -> LayerType:
        """DeltaNet for positions 0-2 in each block, Attention for position 3."""
        return LayerType.ATTENTION if (layer_idx % 4) == 3 else LayerType.DELTANET

    def deltanet_layers(self) -> list[int]:
        """Return indices of all DeltaNet layers (48 total)."""
        return [i for i in range(self.num_layers) if self.layer_type(i) == LayerType.DELTANET]

    def attention_layers(self) -> list[int]:
        """Return indices of all attention layers (16 total)."""
        return [i for i in range(self.num_layers) if self.layer_type(i) == LayerType.ATTENTION]

    def block_index(self, layer_idx: int) -> int:
        """Return which block a layer belongs to."""
        return layer_idx // self.block_size

    def position_in_block(self, layer_idx: int) -> int:
        """Return the position of a layer within its block (0-3)."""
        return layer_idx % self.block_size


# Canonical hook points for the 9 SAEs.
#
# 8 primary SAEs: paired DeltaNet (position 2) + Attention (position 3) at
# early/early-mid/mid/late depths for clean layer-type comparison.
#
# 1 control SAE: DeltaNet at position 1 (mid block) to test whether position
# within the 3-DeltaNet sequence matters independently of layer type. If
# features at position 1 and position 2 are similar, position doesn't matter
# and DeltaNet vs attention is the primary factor. If they differ, position
# within the block is a confound that must be reported.
HOOK_POINTS: list[HookPoint] = [
    # --- Early (block 2) ---
    HookPoint(
        sae_id="sae_delta_early",
        layer=10,
        layer_type=LayerType.DELTANET,
        block=2,
        description="Post-DeltaNet (position 2 of block 2, last DeltaNet before attention)",
    ),
    HookPoint(
        sae_id="sae_attn_early",
        layer=11,
        layer_type=LayerType.ATTENTION,
        block=2,
        description="Post-Attention (position 3 of block 2)",
    ),
    # --- Early-mid (block 5) ---
    HookPoint(
        sae_id="sae_delta_earlymid",
        layer=22,
        layer_type=LayerType.DELTANET,
        block=5,
        description="Early-mid DeltaNet (position 2 of block 5, last DeltaNet before attention)",
    ),
    HookPoint(
        sae_id="sae_attn_earlymid",
        layer=23,
        layer_type=LayerType.ATTENTION,
        block=5,
        description="Early-mid Attention (position 3 of block 5)",
    ),
    # --- Mid (block 8) ---
    HookPoint(
        sae_id="sae_delta_mid_pos1",
        layer=33,
        layer_type=LayerType.DELTANET,
        block=8,
        description="Control: DeltaNet position 1 of block 8 (middle of DeltaNet sequence)",
    ),
    HookPoint(
        sae_id="sae_delta_mid",
        layer=34,
        layer_type=LayerType.DELTANET,
        block=8,
        description="Midpoint DeltaNet (position 2 of block 8, last DeltaNet before attention)",
    ),
    HookPoint(
        sae_id="sae_attn_mid",
        layer=35,
        layer_type=LayerType.ATTENTION,
        block=8,
        description="Midpoint Attention (position 3 of block 8)",
    ),
    # --- Late (block 13) ---
    HookPoint(
        sae_id="sae_delta_late",
        layer=54,
        layer_type=LayerType.DELTANET,
        block=13,
        description="Late DeltaNet (position 2 of block 13, last DeltaNet before attention)",
    ),
    HookPoint(
        sae_id="sae_attn_late",
        layer=55,
        layer_type=LayerType.ATTENTION,
        block=13,
        description="Late Attention (position 3 of block 13)",
    ),
]

HOOK_POINTS_BY_ID: dict[str, HookPoint] = {hp.sae_id: hp for hp in HOOK_POINTS}
HOOK_LAYERS: list[int] = [hp.layer for hp in HOOK_POINTS]


def validate_configs_agree(model_cfg: ModelConfig, arch_cfg: Qwen35Config) -> None:
    """Assert that ModelConfig (from YAML) and Qwen35Config (hardcoded) share the same values.

    Args:
        model_cfg: Config loaded from model.yaml.
        arch_cfg: Hardcoded Qwen 3.5 architecture constants.

    Raises:
        ValueError: If any shared architectural field disagrees between the two configs.
    """
    shared_fields = ("hidden_dim", "num_layers", "block_size", "num_blocks", "context_length")
    for field in shared_fields:
        yaml_val = getattr(model_cfg, field)
        arch_val = getattr(arch_cfg, field)
        if yaml_val != arch_val:
            raise ValueError(
                f"ModelConfig.{field}={yaml_val} disagrees with "
                f"Qwen35Config.{field}={arch_val}. "
                f"Fix configs/model.yaml to match the hardcoded architecture constants."
            )


class ModelConfig(BaseModel):
    """Model loading configuration, loaded from configs/model.yaml."""

    model_id: str = "Qwen/Qwen3.5-27B"
    hidden_dim: int = 5120
    num_layers: int = 64
    block_size: int = 4
    num_blocks: int = 16
    context_length: int = 4096
    dtype: str = "bfloat16"

    @classmethod
    def from_yaml(cls, path: Path) -> ModelConfig:
        """Load configuration from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

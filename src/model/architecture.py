"""Utility functions for Qwen 3.5-27B architecture analysis."""

from __future__ import annotations

from pathlib import Path

import yaml

from src.model.config import HookPoint, LayerType, Qwen35Config, HOOK_POINTS


def get_hook_points_from_config(config_path: str | Path) -> list[HookPoint]:
    """Load hook point definitions from YAML config.

    Args:
        config_path: Path to sae_training.yaml.

    Returns:
        List of HookPoint objects for each SAE position.
    """
    with open(config_path) as f:
        data = yaml.safe_load(f)

    hook_points = []
    for sae_id, hp_data in data["hook_points"].items():
        layer = hp_data["layer"]
        layer_type = LayerType(hp_data["type"])
        block = hp_data["block"]
        hook_points.append(
            HookPoint(
                sae_id=sae_id,
                layer=layer,
                layer_type=layer_type,
                block=block,
                description=f"{layer_type.value} layer {layer} in block {block}",
            )
        )
    return hook_points


def get_matched_pairs(
    hook_points: list[HookPoint] | None = None,
) -> list[tuple[HookPoint, HookPoint]]:
    """Return matched DeltaNet/Attention pairs at the same depth.

    Args:
        hook_points: List of hook points. Defaults to canonical HOOK_POINTS.

    Returns:
        List of (attention_hp, deltanet_hp) tuples at matching blocks.
    """
    if hook_points is None:
        hook_points = HOOK_POINTS

    # Group by block
    by_block: dict[int, dict[LayerType, HookPoint]] = {}
    for hp in hook_points:
        by_block.setdefault(hp.block, {})[hp.layer_type] = hp

    pairs = []
    for block in sorted(by_block.keys()):
        block_hps = by_block[block]
        if LayerType.ATTENTION in block_hps and LayerType.DELTANET in block_hps:
            pairs.append((block_hps[LayerType.ATTENTION], block_hps[LayerType.DELTANET]))

    return pairs


def layer_metadata(layer_idx: int, config: Qwen35Config | None = None) -> dict:
    """Return metadata about a layer: type, block number, position in block.

    Args:
        layer_idx: Layer index (0-63).
        config: Optional Qwen35Config. Uses defaults if not provided.

    Returns:
        Dict with keys: layer_idx, layer_type, block, position_in_block.
    """
    if config is None:
        config = Qwen35Config()

    return {
        "layer_idx": layer_idx,
        "layer_type": config.layer_type(layer_idx),
        "block": config.block_index(layer_idx),
        "position_in_block": config.position_in_block(layer_idx),
    }


def get_deltanet_hook_points(hook_points: list[HookPoint] | None = None) -> list[HookPoint]:
    """Filter hook points to only DeltaNet layers.

    Args:
        hook_points: List of hook points. Defaults to canonical HOOK_POINTS.

    Returns:
        List of DeltaNet hook points.
    """
    if hook_points is None:
        hook_points = HOOK_POINTS
    return [hp for hp in hook_points if hp.layer_type == LayerType.DELTANET]


def get_attention_hook_points(hook_points: list[HookPoint] | None = None) -> list[HookPoint]:
    """Filter hook points to only attention layers.

    Args:
        hook_points: List of hook points. Defaults to canonical HOOK_POINTS.

    Returns:
        List of attention hook points.
    """
    if hook_points is None:
        hook_points = HOOK_POINTS
    return [hp for hp in hook_points if hp.layer_type == LayerType.ATTENTION]

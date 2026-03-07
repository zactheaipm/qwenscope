"""MoE routing drift analysis for steering interventions.

Qwen 3.5-35B-A3B uses MoE FFN at every layer (256 experts, 8 routed + 1 shared
per token). Steering modifies the residual stream at the target layer, which
may change expert routing at downstream layers. This creates indirect behavioral
effects not captured by the SAE's feature-level analysis.

This module measures routing drift: how much the expert selection changes at
downstream layers when steering is active. High routing drift suggests the
steering intervention has side-effects beyond the target features, which
complicates the contamination matrix interpretation.

Usage:
    drift = measure_routing_drift(model, engine, input_ids, downstream_layers)
    # drift["layer_23"]["jaccard_stability"] < 0.8  => routing shifted substantially
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from src.model.config import Qwen35Config
from src.model.loader import get_layers_module
from src.steering.engine import SteeringEngine

logger = logging.getLogger(__name__)


def _get_router_module(
    layers_module: torch.nn.ModuleList,
    layer_idx: int,
) -> torch.nn.Module | None:
    """Resolve the MoE router (gate) module for a given layer.

    Qwen 3.5 MoE layers have the router at:
        layers[i].mlp.gate  (nn.Linear projecting hidden_dim -> num_experts)

    Returns None if the layer does not have a MoE FFN (should not happen
    for A3B where every layer is MoE, but defensive).
    """
    layer = layers_module[layer_idx]
    # Try standard Qwen 3.5 MoE path
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        return None
    gate = getattr(mlp, "gate", None)
    return gate


def _capture_routing(
    model: Any,
    layers_module: torch.nn.ModuleList,
    input_ids: torch.Tensor,
    target_layers: list[int],
) -> dict[int, torch.Tensor]:
    """Run a forward pass and capture MoE router logits at target layers.

    Returns dict mapping layer_idx -> router_logits of shape (batch, seq_len, num_experts).
    """
    captured: dict[int, torch.Tensor] = {}
    hooks = []

    for layer_idx in target_layers:
        gate = _get_router_module(layers_module, layer_idx)
        if gate is None:
            logger.warning("No MoE gate found at layer %d", layer_idx)
            continue

        def make_hook(idx: int):
            def hook_fn(module, input, output):
                # gate output is (batch * seq_len, num_experts) or (batch, seq_len, num_experts)
                captured[idx] = output.detach()
            return hook_fn

        hooks.append(gate.register_forward_hook(make_hook(layer_idx)))

    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    return captured


def measure_routing_drift(
    model: Any,
    engine: SteeringEngine,
    input_ids: torch.Tensor,
    downstream_layers: list[int] | None = None,
    top_k_experts: int = 8,
) -> dict[str, Any]:
    """Measure how steering changes MoE expert routing at downstream layers.

    Runs two forward passes (baseline and steered) and compares which experts
    are selected at each downstream layer. Reports per-layer and aggregate
    routing stability metrics.

    The engine must have steering already configured via ``set_steering()``.
    This method temporarily enables ``steer_all_positions`` for measurement.

    Args:
        model: The language model.
        engine: Configured SteeringEngine (features and multiplier set).
        input_ids: Input token IDs of shape (1, seq_len).
        downstream_layers: Layers to measure routing at. Defaults to all
            layers after the steering layer.
        top_k_experts: Number of top experts to compare (default 8, matching
            the A3B routing configuration).

    Returns:
        Dict with:
            "per_layer": dict[int, {
                "jaccard_stability": float,  # mean Jaccard similarity of top-k expert sets
                "logit_correlation": float,  # mean Pearson r of router logits
                "top_expert_shift": float,   # fraction of tokens where top-1 expert changed
            }],
            "aggregate": {
                "mean_jaccard": float,
                "mean_logit_corr": float,
                "mean_top1_shift": float,
                "n_layers_measured": int,
            },
            "steering_layer": int,
    """
    layers_module = get_layers_module(model)
    config = Qwen35Config()

    if downstream_layers is None:
        downstream_layers = [
            i for i in range(engine.layer + 1, config.num_layers)
        ]

    if not downstream_layers:
        logger.warning("No downstream layers to measure routing drift")
        return {"per_layer": {}, "aggregate": {}, "steering_layer": engine.layer}

    # 1. Baseline routing (no steering)
    baseline_logits = _capture_routing(model, layers_module, input_ids, downstream_layers)

    # 2. Steered routing
    old_steer_all = engine.steer_all_positions
    engine.steer_all_positions = True
    try:
        with engine.active():
            steered_logits = _capture_routing(
                model, layers_module, input_ids, downstream_layers
            )
    finally:
        engine.steer_all_positions = old_steer_all

    # 3. Compare routing
    per_layer: dict[int, dict[str, float]] = {}

    for layer_idx in downstream_layers:
        if layer_idx not in baseline_logits or layer_idx not in steered_logits:
            continue

        bl = baseline_logits[layer_idx].float()
        st = steered_logits[layer_idx].float()

        # Flatten to (n_tokens, num_experts) if needed
        if bl.dim() == 3:
            bl = bl.view(-1, bl.shape[-1])
            st = st.view(-1, st.shape[-1])
        elif bl.dim() == 1:
            bl = bl.unsqueeze(0)
            st = st.unsqueeze(0)

        n_tokens = bl.shape[0]
        k = min(top_k_experts, bl.shape[-1])

        # Jaccard similarity of top-k expert sets per token
        bl_topk = bl.topk(k, dim=-1).indices  # (n_tokens, k)
        st_topk = st.topk(k, dim=-1).indices

        jaccard_sum = 0.0
        top1_shifts = 0
        for t in range(n_tokens):
            bl_set = set(bl_topk[t].tolist())
            st_set = set(st_topk[t].tolist())
            intersection = len(bl_set & st_set)
            union = len(bl_set | st_set)
            jaccard_sum += intersection / max(union, 1)
            if bl_topk[t, 0].item() != st_topk[t, 0].item():
                top1_shifts += 1

        jaccard_mean = jaccard_sum / max(n_tokens, 1)
        top1_shift_frac = top1_shifts / max(n_tokens, 1)

        # Pearson correlation of full router logit vectors per token
        corr_sum = 0.0
        for t in range(n_tokens):
            bl_vec = bl[t] - bl[t].mean()
            st_vec = st[t] - st[t].mean()
            bl_std = bl_vec.norm()
            st_std = st_vec.norm()
            if bl_std > 1e-8 and st_std > 1e-8:
                corr_sum += float((bl_vec @ st_vec / (bl_std * st_std)).item())
            else:
                corr_sum += 1.0  # identical zero vectors
        corr_mean = corr_sum / max(n_tokens, 1)

        per_layer[layer_idx] = {
            "jaccard_stability": jaccard_mean,
            "logit_correlation": corr_mean,
            "top_expert_shift": top1_shift_frac,
        }

    # Aggregate
    if per_layer:
        mean_jaccard = sum(v["jaccard_stability"] for v in per_layer.values()) / len(per_layer)
        mean_corr = sum(v["logit_correlation"] for v in per_layer.values()) / len(per_layer)
        mean_shift = sum(v["top_expert_shift"] for v in per_layer.values()) / len(per_layer)
    else:
        mean_jaccard = mean_corr = mean_shift = 0.0

    aggregate = {
        "mean_jaccard": mean_jaccard,
        "mean_logit_corr": mean_corr,
        "mean_top1_shift": mean_shift,
        "n_layers_measured": len(per_layer),
    }

    logger.info(
        "Routing drift (steer layer %d): mean_jaccard=%.3f, mean_logit_corr=%.3f, "
        "mean_top1_shift=%.3f across %d downstream layers",
        engine.layer, mean_jaccard, mean_corr, mean_shift, len(per_layer),
    )

    return {
        "per_layer": per_layer,
        "aggregate": aggregate,
        "steering_layer": engine.layer,
    }

"""Forward hook registration for capturing residual stream activations.

This is the most critical infrastructure module. Hooks are registered on
model.model.layers[i] to capture the residual stream output (including skip
connection) at arbitrary layers of the Qwen 3.5-27B hybrid architecture.

The residual stream is 5120-dimensional at ALL layers regardless of whether
the layer is DeltaNet or Attention.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

import torch

logger = logging.getLogger(__name__)


class ActivationCache:
    """Captures residual stream activations at specified layers.

    Usage:
        cache = ActivationCache(model, layers=[8, 11, 32, 35, 52, 55])
        with cache.active():
            output = model(**inputs)
        acts = cache.get(32)  # shape: (batch, seq_len, 5120)
    """

    def __init__(self, model: Any, layers: list[int]) -> None:
        """Initialize the activation cache.

        Args:
            model: The HuggingFace model (must have model.model.layers attribute).
            layers: List of layer indices to capture activations from.
        """
        self._model = model
        self._target_layers = set(layers)
        self._cache: dict[int, torch.Tensor] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

    def _make_hook(self, layer_idx: int) -> Any:
        """Create a forward hook function for a specific layer.

        Args:
            layer_idx: The layer index this hook captures.

        Returns:
            A hook function compatible with PyTorch's register_forward_hook.
        """
        def hook_fn(
            module: torch.nn.Module,
            input: tuple[torch.Tensor, ...],
            output: torch.Tensor | tuple[torch.Tensor, ...],
        ) -> None:
            # output is the residual stream AFTER this layer (including skip connection).
            # For Qwen 3.5, output is typically a tuple — extract the hidden states.
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            # Detach to prevent gradient tracking, store on same device
            self._cache[layer_idx] = hidden_states.detach()

        return hook_fn

    def _register_hooks(self) -> None:
        """Register forward hooks on all target layers."""
        for layer_idx in sorted(self._target_layers):
            layer_module = self._model.model.layers[layer_idx]
            hook = layer_module.register_forward_hook(self._make_hook(layer_idx))
            self._hooks.append(hook)
        logger.debug("Registered %d activation hooks on layers %s", len(self._hooks), sorted(self._target_layers))

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        logger.debug("Removed all activation hooks")

    @contextmanager
    def active(self):
        """Context manager that registers and removes hooks.

        Usage:
            with cache.active():
                output = model(**inputs)
            acts = cache.get(layer_idx)
        """
        self._register_hooks()
        try:
            yield self
        finally:
            self._remove_hooks()

    def get(self, layer: int) -> torch.Tensor:
        """Get cached activations for a layer.

        Args:
            layer: Layer index to retrieve activations for.

        Returns:
            Tensor of shape (batch, seq_len, hidden_dim).

        Raises:
            KeyError: If the layer was not captured (not in target layers or
                no forward pass has been run yet).
        """
        if layer not in self._cache:
            raise KeyError(
                f"No cached activations for layer {layer}. "
                f"Available layers: {sorted(self._cache.keys())}"
            )
        return self._cache[layer]

    def clear(self) -> None:
        """Clear cached activations to free memory."""
        self._cache.clear()

    @property
    def cached_layers(self) -> list[int]:
        """Return sorted list of layer indices with cached activations."""
        return sorted(self._cache.keys())

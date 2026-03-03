"""Activation steering via SAE feature clamping.

Steers model behavior by modifying SAE feature activations mid-forward-pass.
Uses "residual steering" to preserve non-target features exactly.

Steering is applied ONLY during autoregressive decode (seq_len == 1 with
KV-cache), NOT during the prompt prefill phase.  This ensures the model's
understanding of the prompt is unmodified; the causal intervention acts
solely on the model's behavioral disposition at each new-token decision
point.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

import torch

from src.sae.model import TopKSAE

logger = logging.getLogger(__name__)


class SteeringEngine:
    """Steers model behavior by modifying SAE feature activations mid-forward-pass.

    Architecture:
    1. Hook intercepts residual stream at target layer
    2. Encode through SAE → sparse features
    3. Multiply target feature activations by multiplier
    4. Decode back to activation space
    5. Replace original residual stream with steered version

    IMPORTANT: Non-target features must be preserved exactly.
    The reconstruction error of the SAE means some information is lost.
    We mitigate this by only replacing the TARGET feature contributions:

        steered = original + sae.decode(modified_features) - sae.decode(original_features)

    This "residual steering" approach preserves all information not captured by the SAE.

    Steering is only applied during the autoregressive decode phase
    (seq_len == 1), not during prompt prefill (seq_len > 1).  This
    ensures the model understands the prompt normally and the causal
    intervention only affects new-token generation decisions.
    """

    def __init__(self, model: Any, sae: TopKSAE, layer: int) -> None:
        """Initialize the steering engine.

        Args:
            model: The language model.
            sae: Trained SAE for the target layer.
            layer: Layer index to steer at.
        """
        self.model = model
        self.sae = sae
        self.layer = layer
        self._feature_indices: list[int] = []
        self._multiplier: float = 1.0
        # Cache the SAE dtype to avoid repeated introspection in the hot path.
        self._sae_dtype = next(sae.parameters()).dtype

    def set_steering(self, feature_indices: list[int], multiplier: float) -> None:
        """Configure which features to steer and by how much.

        Args:
            feature_indices: Indices of SAE features to modify.
            multiplier: Factor to multiply target feature activations by.
                0.0 = ablate features, 1.0 = no change, 5.0 = amplify 5×.
        """
        self._feature_indices = feature_indices
        self._multiplier = multiplier
        logger.info(
            "Steering configured: %d features at layer %d, multiplier=%.1f",
            len(feature_indices),
            self.layer,
            multiplier,
        )

    @contextmanager
    def active(self):
        """Context manager that activates steering during forward pass.

        Usage:
            engine.set_steering(feature_indices=[1204, 3891], multiplier=5.0)
            with engine.active():
                output = model.generate(**inputs)
        """
        hook = self.model.model.layers[self.layer].register_forward_hook(
            self._steering_hook
        )
        try:
            yield
        finally:
            hook.remove()

    def _steering_hook(
        self,
        module: torch.nn.Module,
        input: tuple[torch.Tensor, ...],
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """The actual steering intervention.

        Uses residual steering: only modifies the target feature contributions
        and adds the delta back to the original activations.

        Only fires during the autoregressive decode phase (seq_len == 1).
        During prompt prefill (seq_len > 1), returns the output unmodified
        so the model processes the prompt normally.

        Args:
            module: The layer module.
            input: Layer input.
            output: Layer output (residual stream).

        Returns:
            Modified output with steered activations (decode) or
            unmodified output (prefill).
        """
        hidden_states = output[0] if isinstance(output, tuple) else output

        # Only steer during autoregressive decode (seq_len == 1).
        # During prefill, seq_len > 1 and we want the model to encode the
        # prompt normally without any intervention.
        if hidden_states.shape[1] != 1:
            return output

        # Skip if no features configured (avoid wasteful encode/decode)
        if not self._feature_indices:
            return output

        original_dtype = hidden_states.dtype

        with torch.no_grad():
            # Cast to SAE dtype before encoding to prevent RuntimeError
            # when model runs in BF16 but SAE weights are float32.
            hidden_for_sae = hidden_states.to(dtype=self._sae_dtype)

            # Encode to features
            original_features = self.sae.encode(hidden_for_sae)  # (..., dict_size)

            # Modify target features (vectorized for efficiency)
            modified_features = original_features.clone()
            modified_features[..., self._feature_indices] *= self._multiplier

            # Residual steering: only change the target feature contributions.
            # decode(a) - decode(b) = decoder(a) + pre_bias - decoder(b) - pre_bias
            #                       = decoder(a - b)
            # One GEMM instead of two — pre_bias cancels exactly.
            delta = self.sae.decoder(modified_features - original_features)
            steered = hidden_states + delta.to(dtype=original_dtype)  # (..., hidden_dim)

        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered


class MultiLayerSteeringEngine:
    """Steers at multiple layers simultaneously.

    Useful for Experiment 3 (cross-depth steering) where we need to
    steer at early, mid, and late layers with different SAEs.
    """

    def __init__(self, model: Any) -> None:
        """Initialize multi-layer steering.

        Args:
            model: The language model.
        """
        self.model = model
        self._engines: list[SteeringEngine] = []

    def add_layer(
        self,
        sae: TopKSAE,
        layer: int,
        feature_indices: list[int],
        multiplier: float,
    ) -> None:
        """Add a steering layer.

        Args:
            sae: SAE for this layer.
            layer: Layer index.
            feature_indices: Features to steer.
            multiplier: Steering multiplier.
        """
        engine = SteeringEngine(self.model, sae, layer)
        engine.set_steering(feature_indices, multiplier)
        self._engines.append(engine)

    def clear(self) -> None:
        """Remove all steering layers."""
        self._engines.clear()

    @contextmanager
    def active(self):
        """Context manager that activates all steering hooks."""
        hooks = []
        for engine in self._engines:
            hook = engine.model.model.layers[engine.layer].register_forward_hook(
                engine._steering_hook
            )
            hooks.append(hook)
        try:
            yield
        finally:
            for hook in hooks:
                hook.remove()


class MeanDiffSteeringEngine:
    """Baseline steering via direct activation addition (no SAE).

    Adds a fixed direction vector to the residual stream, scaled by a
    multiplier. This is the simplest possible steering approach and
    serves as a baseline for SAE-based steering.

    The steering vector is typically the mean activation difference
    between HIGH and LOW contrastive pairs:
        ``steering_vector = mean(activations_high) - mean(activations_low)``

    Like SAE-based steering, this only applies during the autoregressive
    decode phase (seq_len == 1), not during prompt prefill.
    """

    def __init__(
        self,
        model: Any,
        layer: int,
        steering_vector: torch.Tensor,
    ) -> None:
        """Initialize the mean-diff steering engine.

        Args:
            model: The language model.
            layer: Layer index to steer at.
            steering_vector: Dense steering direction of shape (hidden_dim,).
        """
        self.model = model
        self.layer = layer
        # Normalize to unit norm so that the multiplier has a consistent
        # scale regardless of the absolute activation magnitude at this layer.
        # Without normalization, the multiplier is not comparable to SAE-based
        # steering (where multipliers operate on already-normalised feature
        # activations) and the baseline comparison is meaningless.
        vec_norm = steering_vector.norm()
        if vec_norm > 1e-8:
            self._steering_vector = steering_vector / vec_norm
        else:
            self._steering_vector = steering_vector
        self._raw_norm = float(vec_norm.item())
        self._multiplier: float = 1.0

    def set_multiplier(self, multiplier: float) -> None:
        """Set the steering multiplier.

        Args:
            multiplier: Factor to scale the steering vector by.
        """
        self._multiplier = multiplier
        logger.info(
            "Mean-diff steering configured: layer %d, multiplier=%.1f",
            self.layer,
            multiplier,
        )

    @contextmanager
    def active(self):
        """Context manager that activates mean-diff steering."""
        hook = self.model.model.layers[self.layer].register_forward_hook(
            self._steering_hook
        )
        try:
            yield
        finally:
            hook.remove()

    def _steering_hook(
        self,
        module: torch.nn.Module,
        input: tuple[torch.Tensor, ...],
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Add the scaled steering vector to the residual stream.

        Only fires during the autoregressive decode phase (seq_len == 1).

        Args:
            module: The layer module.
            input: Layer input.
            output: Layer output (residual stream).

        Returns:
            Modified output with steering vector added.
        """
        hidden_states = output[0] if isinstance(output, tuple) else output

        # Only steer during autoregressive decode, not prompt prefill.
        if hidden_states.shape[1] != 1:
            return output

        with torch.no_grad():
            # Broadcast steering vector to match hidden_states shape
            vec = self._steering_vector.to(
                device=hidden_states.device, dtype=hidden_states.dtype
            )
            steered = hidden_states + self._multiplier * vec  # (..., hidden_dim)

        if isinstance(output, tuple):
            return (steered,) + output[1:]
        return steered

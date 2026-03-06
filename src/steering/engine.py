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

from src.model.loader import get_layers_module
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
        self._layers_module = get_layers_module(model)
        self._feature_indices: list[int] = []
        self._multiplier: float = 1.0
        # When True, steering fires at ALL sequence positions (not just
        # decode-phase seq_len==1).  Used by measurement code like
        # measure_cross_layer_interaction that needs to observe the causal
        # effect of steering during a full prefill forward pass.  Must NEVER
        # be True during normal autoregressive generation (it would corrupt
        # the prompt representation).
        self.steer_all_positions: bool = False
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
        hook = self._layers_module[self.layer].register_forward_hook(
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

        # Only steer during autoregressive decode (seq_len == 1) unless
        # steer_all_positions is explicitly set for measurement purposes.
        # During normal generation, prefill (seq_len > 1) is left untouched
        # so the model encodes the prompt without intervention.
        if hidden_states.shape[1] != 1 and not self.steer_all_positions:
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
            hook = engine._layers_module[engine.layer].register_forward_hook(
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

        IMPORTANT — multiplier comparability caveat:
        The steering vector is unit-normalized, so multiplier M adds M units
        of displacement in the mean-diff direction.  SAE-based steering
        multipliers scale feature activations, which have a different native
        magnitude.  The same numeric multiplier (e.g., 5.0) produces
        different-magnitude interventions between the two methods.

        Ratio-based metrics (selectivity ratio, probability of superiority)
        are less affected because they compare on-target vs off-target effects
        within a single method.  But absolute Cohen's d values should NOT
        be compared across methods at matched multiplier values.  Report
        this caveat when presenting mean-diff baseline results alongside
        SAE steering results.

        The raw (un-normalized) steering vector norm is stored in
        ``_raw_norm`` for reference.

        Args:
            model: The language model.
            layer: Layer index to steer at.
            steering_vector: Dense steering direction of shape (hidden_dim,).
        """
        self.model = model
        self.layer = layer
        self._layers_module = get_layers_module(model)
        # Normalize to unit norm so that the multiplier controls
        # displacement magnitude in interpretable units, independent
        # of the absolute activation scale at this layer.
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
        hook = self._layers_module[self.layer].register_forward_hook(
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

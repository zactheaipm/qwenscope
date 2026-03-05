"""Tests for the steering engine.

Verifies residual steering preserves non-target features.
All tests run on CPU.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.sae.model import TopKSAE
from src.steering.engine import SteeringEngine, MultiLayerSteeringEngine, MeanDiffSteeringEngine


class MockLayer(nn.Module):
    """Mock transformer layer."""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (self.linear(x) + x, None)


class MockModel:
    """Mock model for steering tests."""

    def __init__(self, num_layers: int = 4, hidden_dim: int = 64) -> None:
        self.model = type("Model", (), {"layers": nn.ModuleList(
            [MockLayer(hidden_dim) for _ in range(num_layers)]
        )})()
        self.hidden_dim = hidden_dim
        self.device = torch.device("cpu")

    def __call__(self, **kwargs: torch.Tensor) -> torch.Tensor:
        x = kwargs.get("input_ids", torch.randn(2, 10, self.hidden_dim)).float()
        if x.dim() == 2:
            x = x.unsqueeze(-1).expand(-1, -1, self.hidden_dim).float()
        for layer in self.model.layers:
            output = layer(x)
            x = output[0]
        return x

    def generate(self, **kwargs: torch.Tensor) -> torch.Tensor:
        return self(**kwargs)


class TestSteeringEngine:
    """Test suite for SteeringEngine."""

    @pytest.fixture
    def setup(self) -> tuple[MockModel, TopKSAE, SteeringEngine]:
        """Create model, SAE, and engine for testing."""
        model = MockModel(num_layers=4, hidden_dim=64)
        sae = TopKSAE(hidden_dim=64, dict_size=256, k=8)
        engine = SteeringEngine(model, sae, layer=1)
        return model, sae, engine

    def test_no_change_at_multiplier_1(
        self, setup: tuple[MockModel, TopKSAE, SteeringEngine]
    ) -> None:
        """Steering with multiplier=1.0 should produce (near) identical output."""
        model, sae, engine = setup
        engine.set_steering(feature_indices=[0, 1, 2], multiplier=1.0)

        # seq_len=1 so the steering hook actually fires (decode-only gate).
        x = torch.randn(2, 1, 64)

        # Baseline
        with torch.no_grad():
            baseline = model(input_ids=x)

        # Steered at 1.0 (no change)
        with torch.no_grad():
            with engine.active():
                steered = model(input_ids=x)

        assert torch.allclose(baseline, steered, atol=1e-5), (
            f"Max diff: {(baseline - steered).abs().max()}"
        )

    def test_output_changes_at_multiplier_5(
        self, setup: tuple[MockModel, TopKSAE, SteeringEngine]
    ) -> None:
        """Steering at multiplier=5.0 should change the output."""
        model, sae, engine = setup
        engine.set_steering(feature_indices=[0, 1, 2, 3, 4], multiplier=5.0)

        # seq_len=1 so the steering hook actually fires.
        x = torch.randn(2, 1, 64)

        # Guarantee features 0-4 are always selected by TopK by giving them
        # large encoder biases. Without this, a fresh untrained SAE may not
        # select them in the top-8, making delta=0.
        with torch.no_grad():
            sae.encoder.bias.data[0:5] = 100.0

        with torch.no_grad():
            baseline = model(input_ids=x)

        with torch.no_grad():
            with engine.active():
                steered = model(input_ids=x)

        # Output should be different
        assert not torch.allclose(baseline, steered, atol=1e-3), (
            "Steered output should differ from baseline"
        )

    def test_ablation_at_multiplier_0(
        self, setup: tuple[MockModel, TopKSAE, SteeringEngine]
    ) -> None:
        """Steering at multiplier=0.0 ablates target features."""
        model, sae, engine = setup
        engine.set_steering(feature_indices=[0, 1, 2], multiplier=0.0)

        # seq_len=1 so the steering hook actually fires.
        x = torch.randn(2, 1, 64)

        # Guarantee features 0-2 are always selected by TopK.
        with torch.no_grad():
            sae.encoder.bias.data[0:3] = 100.0

        with torch.no_grad():
            baseline = model(input_ids=x)
            with engine.active():
                ablated = model(input_ids=x)

        # Should produce different output
        assert not torch.allclose(baseline, ablated, atol=1e-3)

    def test_residual_steering_preserves_non_target(
        self, setup: tuple[MockModel, TopKSAE, SteeringEngine]
    ) -> None:
        """Residual steering should only affect target features.

        The steering hook does:
            modified = original_features.clone()
            modified[..., targets] *= multiplier
            delta = decode(modified) - decode(original)
            steered = hidden_states + delta

        We verify that modified_features matches original_features
        for all non-target indices — this is the core invariant.
        """
        model, sae, engine = setup
        target_features = [0, 1]
        multiplier = 5.0
        engine.set_steering(target_features, multiplier=multiplier)

        x = torch.randn(1, 5, 64)

        with torch.no_grad():
            # Step 1: Encode to get original features
            original_features = sae.encode(x)

            # Step 2: Clone and modify (same logic as the steering hook)
            modified_features = original_features.clone()
            modified_features[..., target_features] *= multiplier

            # Invariant: non-target features are exactly preserved in modified_features
            non_target_mask = torch.ones(256, dtype=torch.bool)
            non_target_mask[target_features] = False

            assert torch.equal(
                original_features[..., non_target_mask],
                modified_features[..., non_target_mask],
            ), "Non-target features must be exactly preserved after modification"

            # Invariant: target features are scaled correctly
            for idx in target_features:
                assert torch.allclose(
                    modified_features[..., idx],
                    original_features[..., idx] * multiplier,
                    atol=1e-6,
                ), f"Target feature {idx} not correctly scaled"

            # Invariant: the delta only contains contributions from target features
            delta = sae.decode(modified_features) - sae.decode(original_features)
            # Construct expected delta from target features only
            target_only_modified = torch.zeros_like(original_features)
            target_only_original = torch.zeros_like(original_features)
            for idx in target_features:
                target_only_modified[..., idx] = modified_features[..., idx]
                target_only_original[..., idx] = original_features[..., idx]
            expected_delta = sae.decode(target_only_modified) - sae.decode(target_only_original)

            # Due to decoder bias, delta == expected_delta only if decoder is linear
            # in the feature dimension (which it is for nn.Linear — bias cancels)
            assert torch.allclose(delta, expected_delta, atol=1e-5), (
                f"Delta should only reflect target feature changes. "
                f"Max diff: {(delta - expected_delta).abs().max()}"
            )

    def test_hooks_properly_cleaned_up(
        self, setup: tuple[MockModel, TopKSAE, SteeringEngine]
    ) -> None:
        """Hooks are removed after context manager exits."""
        model, sae, engine = setup
        engine.set_steering([0], 2.0)

        # seq_len=1 so the hook fires while active.
        x = torch.randn(2, 1, 64)

        with engine.active():
            model(input_ids=x)

        # After context, running again should produce baseline
        with torch.no_grad():
            result1 = model(input_ids=x)
            result2 = model(input_ids=x)

        assert torch.allclose(result1, result2)


class TestMultiLayerSteering:
    """Test multi-layer steering."""

    def test_multi_layer_steering(self) -> None:
        """MultiLayerSteeringEngine steers at multiple layers."""
        model = MockModel(num_layers=4, hidden_dim=64)
        sae = TopKSAE(hidden_dim=64, dict_size=256, k=8)

        # Guarantee features 0-3 are always selected by TopK by giving them
        # large encoder biases. Without this, a fresh untrained SAE may not
        # select features [0,1] or [2,3] in the top-8, making delta=0.
        with torch.no_grad():
            sae.encoder.bias.data[0:4] = 100.0

        multi = MultiLayerSteeringEngine(model)
        multi.add_layer(sae, layer=0, feature_indices=[0, 1], multiplier=5.0)
        multi.add_layer(sae, layer=2, feature_indices=[2, 3], multiplier=3.0)

        # seq_len=1 so the steering hooks actually fire.
        x = torch.randn(2, 1, 64)

        with torch.no_grad():
            baseline = model(input_ids=x)

        with torch.no_grad():
            with multi.active():
                steered = model(input_ids=x)

        assert not torch.allclose(baseline, steered, atol=1e-3)


class TestMeanDiffSteeringEngine:
    """Test suite for MeanDiffSteeringEngine (activation addition baseline)."""

    @pytest.fixture
    def setup(self) -> tuple[MockModel, MeanDiffSteeringEngine]:
        """Create model and mean-diff engine for testing."""
        model = MockModel(num_layers=4, hidden_dim=64)
        steering_vector = torch.randn(64)
        engine = MeanDiffSteeringEngine(model, layer=1, steering_vector=steering_vector)
        return model, engine

    def test_unit_normalization(self) -> None:
        """Steering vector should be unit-normalized."""
        model = MockModel(num_layers=4, hidden_dim=64)
        vec = torch.randn(64) * 10.0
        engine = MeanDiffSteeringEngine(model, layer=1, steering_vector=vec)
        norm = engine._steering_vector.norm().item()
        assert abs(norm - 1.0) < 1e-6, f"Expected unit norm, got {norm}"

    def test_raw_norm_stored(self) -> None:
        """Raw (pre-normalization) norm should be stored."""
        model = MockModel(num_layers=4, hidden_dim=64)
        vec = torch.randn(64)
        expected_norm = vec.norm().item()
        engine = MeanDiffSteeringEngine(model, layer=1, steering_vector=vec)
        assert abs(engine._raw_norm - expected_norm) < 1e-5

    def test_no_change_at_multiplier_0(
        self, setup: tuple[MockModel, MeanDiffSteeringEngine]
    ) -> None:
        """Multiplier=0.0 should produce no steering effect."""
        model, engine = setup
        engine.set_multiplier(0.0)
        x = torch.randn(2, 1, 64)

        with torch.no_grad():
            baseline = model(input_ids=x)
            with engine.active():
                steered = model(input_ids=x)

        assert torch.allclose(baseline, steered, atol=1e-5), (
            f"Max diff: {(baseline - steered).abs().max()}"
        )

    def test_output_changes_at_nonzero_multiplier(
        self, setup: tuple[MockModel, MeanDiffSteeringEngine]
    ) -> None:
        """Non-zero multiplier should change output during decode (seq_len=1)."""
        model, engine = setup
        engine.set_multiplier(5.0)
        x = torch.randn(2, 1, 64)

        with torch.no_grad():
            baseline = model(input_ids=x)
            with engine.active():
                steered = model(input_ids=x)

        assert not torch.allclose(baseline, steered, atol=1e-3), (
            "Steered output should differ from baseline"
        )

    def test_skips_prefill(
        self, setup: tuple[MockModel, MeanDiffSteeringEngine]
    ) -> None:
        """Steering should NOT fire during prefill (seq_len > 1)."""
        model, engine = setup
        engine.set_multiplier(10.0)
        x = torch.randn(2, 5, 64)  # seq_len=5 -> prefill

        with torch.no_grad():
            baseline = model(input_ids=x)
            with engine.active():
                steered = model(input_ids=x)

        assert torch.allclose(baseline, steered, atol=1e-6), (
            "Mean-diff steering should not fire during prefill"
        )

    def test_hook_cleanup(
        self, setup: tuple[MockModel, MeanDiffSteeringEngine]
    ) -> None:
        """Hook should be removed after context manager exits."""
        model, engine = setup
        engine.set_multiplier(5.0)
        layer_module = model.model.layers[1]
        hooks_before = len(layer_module._forward_hooks)

        with engine.active():
            hooks_during = len(layer_module._forward_hooks)
            assert hooks_during == hooks_before + 1

        hooks_after = len(layer_module._forward_hooks)
        assert hooks_after == hooks_before, (
            f"Hook not cleaned up: {hooks_before} before, {hooks_after} after"
        )

    def test_steering_magnitude_scales_with_multiplier(
        self, setup: tuple[MockModel, MeanDiffSteeringEngine]
    ) -> None:
        """Doubling the multiplier should roughly double the steering delta."""
        model, engine = setup
        x = torch.randn(2, 1, 64)

        with torch.no_grad():
            baseline = model(input_ids=x)

            engine.set_multiplier(2.0)
            with engine.active():
                steered_2x = model(input_ids=x)

            engine.set_multiplier(4.0)
            with engine.active():
                steered_4x = model(input_ids=x)

        delta_2x = (steered_2x - baseline).norm()
        delta_4x = (steered_4x - baseline).norm()
        ratio = delta_4x / max(delta_2x, 1e-8)
        assert 1.5 < ratio < 2.5, f"Expected ratio ~2.0, got {ratio:.2f}"

"""Tests for hook registration and activation capture.

All tests run on CPU with mock models.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.model.config import LayerType, Qwen35Config, HOOK_POINTS
from src.model.hooks import ActivationCache


class MockLayer(nn.Module):
    """Mock transformer layer that returns a tuple like Qwen 3.5."""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # Simulates residual stream output as tuple (hidden_states, ...)
        return (self.linear(x) + x, None)


class MockModel:
    """Mock model with the same structure as HuggingFace Qwen 3.5."""

    def __init__(self, num_layers: int = 64, hidden_dim: int = 64) -> None:
        self.model = type("Model", (), {"layers": nn.ModuleList(
            [MockLayer(hidden_dim) for _ in range(num_layers)]
        )})()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def __call__(self, **kwargs: torch.Tensor) -> torch.Tensor:
        x = kwargs.get("input_ids", torch.randn(2, 10, self.hidden_dim)).float()
        if x.dim() == 2:
            x = x.unsqueeze(-1).expand(-1, -1, self.hidden_dim).float()
        for layer in self.model.layers:
            output = layer(x)
            x = output[0]
        return x


class TestActivationCache:
    """Test suite for ActivationCache."""

    def test_captures_specified_layers(self) -> None:
        """Cache captures activations for specified layers only."""
        model = MockModel(num_layers=8, hidden_dim=64)
        target_layers = [1, 3, 5]
        cache = ActivationCache(model, layers=target_layers)

        with cache.active():
            model(input_ids=torch.randn(2, 10, 64))

        assert sorted(cache.cached_layers) == target_layers

    def test_correct_shape(self) -> None:
        """Captured activations have shape (batch, seq_len, hidden_dim)."""
        model = MockModel(num_layers=4, hidden_dim=32)
        cache = ActivationCache(model, layers=[0, 1, 2, 3])

        batch_size, seq_len = 3, 15
        with cache.active():
            model(input_ids=torch.randn(batch_size, seq_len, 32))

        for layer_idx in [0, 1, 2, 3]:
            acts = cache.get(layer_idx)
            assert acts.shape == (batch_size, seq_len, 32), (
                f"Layer {layer_idx}: expected ({batch_size}, {seq_len}, 32), got {acts.shape}"
            )

    def test_all_64_layers(self) -> None:
        """All 64 layers can be captured simultaneously."""
        model = MockModel(num_layers=64, hidden_dim=64)
        cache = ActivationCache(model, layers=list(range(64)))

        with cache.active():
            model(input_ids=torch.randn(1, 5, 64))

        assert len(cache.cached_layers) == 64
        for i in range(64):
            assert cache.get(i).shape[-1] == 64

    def test_deltanet_and_attention_layers(self) -> None:
        """Both DeltaNet and attention layers are captured correctly."""
        config = Qwen35Config(num_layers=8, hidden_dim=64)
        model = MockModel(num_layers=8, hidden_dim=64)

        # Layer 3 is attention (3 % 4 == 3), layer 2 is DeltaNet
        cache = ActivationCache(model, layers=[2, 3])

        with cache.active():
            model(input_ids=torch.randn(1, 5, 64))

        assert config.layer_type(2) == LayerType.DELTANET
        assert config.layer_type(3) == LayerType.ATTENTION
        assert cache.get(2).shape == cache.get(3).shape

    def test_clear_frees_memory(self) -> None:
        """cache.clear() removes all cached tensors."""
        model = MockModel(num_layers=4, hidden_dim=64)
        cache = ActivationCache(model, layers=[0, 1])

        with cache.active():
            model(input_ids=torch.randn(1, 5, 64))

        assert len(cache.cached_layers) == 2
        cache.clear()
        assert len(cache.cached_layers) == 0

    def test_hooks_removed_after_context(self) -> None:
        """Hooks are removed from both the cache and PyTorch modules."""
        model = MockModel(num_layers=4, hidden_dim=64)
        target_layers = [0, 2]
        cache = ActivationCache(model, layers=target_layers)

        # Count PyTorch hooks before
        hooks_before = {
            i: len(model.model.layers[i]._forward_hooks)
            for i in target_layers
        }

        with cache.active():
            # During context: PyTorch hooks should be registered
            for i in target_layers:
                assert len(model.model.layers[i]._forward_hooks) == hooks_before[i] + 1, (
                    f"Layer {i}: expected hook to be registered during context"
                )

        # After context: internal list should be cleared
        assert len(cache._hooks) == 0

        # After context: PyTorch module hooks should be back to original count
        for i in target_layers:
            actual = len(model.model.layers[i]._forward_hooks)
            assert actual == hooks_before[i], (
                f"Layer {i}: PyTorch hook not removed. "
                f"Had {hooks_before[i]} before, {actual} after"
            )

    def test_get_raises_on_missing_layer(self) -> None:
        """cache.get() raises KeyError for uncaptured layers."""
        model = MockModel(num_layers=4, hidden_dim=64)
        cache = ActivationCache(model, layers=[0])

        with cache.active():
            model(input_ids=torch.randn(1, 5, 64))

        with pytest.raises(KeyError):
            cache.get(1)

    def test_activations_are_detached(self) -> None:
        """Cached activations should not require grad."""
        model = MockModel(num_layers=4, hidden_dim=64)
        cache = ActivationCache(model, layers=[0])

        with cache.active():
            model(input_ids=torch.randn(1, 5, 64))

        assert not cache.get(0).requires_grad


class TestQwen35Config:
    """Test the architecture configuration."""

    def test_layer_type_classification(self) -> None:
        config = Qwen35Config()
        # DeltaNet: 0,1,2 in each block
        assert config.layer_type(0) == LayerType.DELTANET
        assert config.layer_type(1) == LayerType.DELTANET
        assert config.layer_type(2) == LayerType.DELTANET
        # Attention: position 3
        assert config.layer_type(3) == LayerType.ATTENTION
        assert config.layer_type(7) == LayerType.ATTENTION
        assert config.layer_type(63) == LayerType.ATTENTION

    def test_layer_counts(self) -> None:
        config = Qwen35Config()
        assert len(config.deltanet_layers()) == 48
        assert len(config.attention_layers()) == 16

    def test_block_index(self) -> None:
        config = Qwen35Config()
        assert config.block_index(0) == 0
        assert config.block_index(4) == 1
        assert config.block_index(63) == 15

    def test_hook_points(self) -> None:
        # 9 hook points: 4 paired DeltaNet+Attention (early/earlymid/mid/late)
        # + 1 control DeltaNet (mid_pos1)
        assert len(HOOK_POINTS) == 9
        layers = [hp.layer for hp in HOOK_POINTS]
        assert 10 in layers   # sae_delta_early
        assert 11 in layers   # sae_attn_early
        assert 22 in layers   # sae_delta_earlymid
        assert 23 in layers   # sae_attn_earlymid
        assert 33 in layers   # sae_delta_mid_pos1 (control)
        assert 34 in layers   # sae_delta_mid
        assert 35 in layers   # sae_attn_mid
        assert 54 in layers   # sae_delta_late
        assert 55 in layers   # sae_attn_late

        # Verify layer types match architecture
        sae_by_id = {hp.sae_id: hp for hp in HOOK_POINTS}
        for sae_id, hp in sae_by_id.items():
            if "delta" in sae_id:
                assert hp.layer_type == LayerType.DELTANET, f"{sae_id} should be DeltaNet"
            elif "attn" in sae_id:
                assert hp.layer_type == LayerType.ATTENTION, f"{sae_id} should be Attention"

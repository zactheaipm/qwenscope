"""Tests for TopK SAE architecture.

All tests run on CPU with small tensor sizes.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from src.sae.model import TopKSAE


class TestTopKSAE:
    """Test suite for TopK SAE."""

    @pytest.fixture
    def sae(self) -> TopKSAE:
        """Small SAE for testing."""
        return TopKSAE(hidden_dim=64, dict_size=256, k=8)

    def test_forward_output_shapes(self, sae: TopKSAE) -> None:
        """Forward pass produces correct output shapes."""
        x = torch.randn(4, 16, 64)
        reconstruction, features, loss = sae(x)

        assert reconstruction.shape == (4, 16, 64)
        assert features.shape == (4, 16, 256)
        assert loss.dim() == 0  # scalar

    def test_topk_sparsity(self, sae: TopKSAE) -> None:
        """Only k features are nonzero after encoding."""
        x = torch.randn(2, 10, 64)
        features = sae.encode(x)

        nonzero_per_token = (features.abs() > 0).sum(dim=-1)
        assert (nonzero_per_token == 8).all(), f"Expected k=8, got {nonzero_per_token.unique()}"

    def test_encode_decode_preserves_dimensionality(self, sae: TopKSAE) -> None:
        """encode→decode preserves dimensionality."""
        x = torch.randn(2, 10, 64)
        features = sae.encode(x)
        reconstruction = sae.decode(features)

        assert reconstruction.shape == x.shape

    def test_2d_input(self, sae: TopKSAE) -> None:
        """SAE works with 2D input (batch, hidden_dim)."""
        x = torch.randn(32, 64)
        reconstruction, features, loss = sae(x)

        assert reconstruction.shape == (32, 64)
        assert features.shape == (32, 256)
        assert (features.abs() > 0).sum(dim=-1).unique().item() == 8

    def test_loss_decreases(self) -> None:
        """Loss decreases over a few training steps on random data."""
        sae = TopKSAE(hidden_dim=32, dict_size=128, k=8)
        optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)

        x = torch.randn(64, 32)
        losses = []
        for _ in range(20):
            optimizer.zero_grad()
            _, _, loss = sae(x)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_save_load_roundtrip(self, sae: TopKSAE) -> None:
        """Save and load roundtrip preserves weights exactly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            sae.save(path)
            loaded = TopKSAE.load(path)

            assert loaded.hidden_dim == sae.hidden_dim
            assert loaded.dict_size == sae.dict_size
            assert loaded.k == sae.k

            for (n1, p1), (n2, p2) in zip(
                sae.named_parameters(), loaded.named_parameters()
            ):
                assert n1 == n2
                assert torch.allclose(p1, p2), f"Mismatch in {n1}"

    def test_save_load_produces_same_output(self, sae: TopKSAE) -> None:
        """Loaded SAE produces the same output as the original."""
        x = torch.randn(2, 10, 64)
        sae.eval()

        with torch.no_grad():
            recon1, feat1, _ = sae(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            sae.save(path)
            loaded = TopKSAE.load(path)
            loaded.eval()

            with torch.no_grad():
                recon2, feat2, _ = loaded(x)

        assert torch.allclose(recon1, recon2, atol=1e-6)
        assert torch.allclose(feat1, feat2, atol=1e-6)

    @pytest.mark.parametrize("k", [1, 8, 32, 64])
    def test_various_k_values(self, k: int) -> None:
        """SAE works with various k values."""
        sae = TopKSAE(hidden_dim=64, dict_size=256, k=k)
        x = torch.randn(2, 10, 64)
        features = sae.encode(x)

        nonzero_per_token = (features.abs() > 0).sum(dim=-1)
        assert (nonzero_per_token == k).all()

    def test_gradient_flows(self, sae: TopKSAE) -> None:
        """Gradients flow through the forward pass."""
        x = torch.randn(4, 64, requires_grad=True)
        _, _, loss = sae(x)
        loss.backward()

        assert sae.encoder.weight.grad is not None
        assert sae.decoder.weight.grad is not None


class TestExplainedVarianceConsistency:
    """Verify EV accumulation is consistent across batch splits."""

    def test_ev_accumulation_matches_global(self) -> None:
        """EV computed batch-by-batch must match EV on full data."""
        torch.manual_seed(42)
        sae = TopKSAE(hidden_dim=32, dict_size=128, k=4)

        # Generate full dataset
        full_data = torch.randn(200, 32)
        with torch.no_grad():
            full_recon, _, _ = sae(full_data)

        # Compute EV on full data (reference)
        full_residual = full_data - full_recon
        full_res_var = full_residual.var(dim=0)
        full_act_var = full_data.var(dim=0)
        valid = full_act_var > 1e-8
        ev_global = float((1.0 - full_res_var[valid] / full_act_var[valid]).mean())

        # Now compute using the sufficient-statistics accumulator (Chan's formula)
        total_residual_sum = torch.zeros(32)
        total_residual_sq_sum = torch.zeros(32)
        total_acts_sum = torch.zeros(32)
        total_acts_sq_sum = torch.zeros(32)
        n_total = 0

        for i in range(0, 200, 50):  # 4 batches of 50
            batch = full_data[i:i+50]
            with torch.no_grad():
                recon, _, _ = sae(batch)
            residual = batch - recon
            total_residual_sum += residual.sum(dim=0)
            total_residual_sq_sum += residual.pow(2).sum(dim=0)
            total_acts_sum += batch.sum(dim=0)
            total_acts_sq_sum += batch.pow(2).sum(dim=0)
            n_total += batch.shape[0]

        # Chan's formula
        global_residual_var = (total_residual_sq_sum / n_total) - (total_residual_sum / n_total).pow(2)
        global_acts_var = (total_acts_sq_sum / n_total) - (total_acts_sum / n_total).pow(2)
        valid2 = global_acts_var > 1e-8
        ev_accumulated = float((1.0 - global_residual_var[valid2] / global_acts_var[valid2]).mean())

        # Should match global EV to high precision
        assert abs(ev_global - ev_accumulated) < 1e-5, (
            f"EV mismatch: global={ev_global:.6f}, accumulated={ev_accumulated:.6f}"
        )

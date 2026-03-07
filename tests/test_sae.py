"""Tests for TopK SAE architecture and configuration.

All tests run on CPU with small tensor sizes.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from src.sae.config import SAETrainingConfig
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
        """At most k features are nonzero after encoding.

        With ReLU-before-TopK, if fewer than k latents are positive, the
        actual number of nonzero features may be less than k. In practice
        with a well-trained encoder this is rare, but the correct invariant
        is <= k, not == k.
        """
        x = torch.randn(2, 10, 64)
        features = sae.encode(x)

        nonzero_per_token = (features.abs() > 0).sum(dim=-1)
        assert (nonzero_per_token <= 8).all(), f"Expected <= k=8, got {nonzero_per_token.unique()}"
        # With random encoder weights and random inputs, roughly half the latents
        # are positive, so we should still have k active features in practice.
        assert (nonzero_per_token > 0).all(), "Expected at least some active features"

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
        assert ((features.abs() > 0).sum(dim=-1) <= 8).all()

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
        assert (nonzero_per_token <= k).all()
        assert (nonzero_per_token > 0).all()

    def test_gradient_flows(self, sae: TopKSAE) -> None:
        """Gradients flow through the forward pass."""
        x = torch.randn(4, 64, requires_grad=True)
        _, _, loss = sae(x)
        loss.backward()

        assert sae.encoder.weight.grad is not None
        assert sae.decoder.weight.grad is not None


class TestSAETrainingConfig:
    """Test SAETrainingConfig YAML loading and per-hook overrides."""

    YAML_PATH = Path("configs/sae_training.yaml")

    @pytest.fixture(autouse=True)
    def _check_yaml_exists(self) -> None:
        if not self.YAML_PATH.exists():
            pytest.skip("sae_training.yaml not found (CI without configs)")

    def test_early_sae_gets_overridden_topk_and_dict(self) -> None:
        """Early SAEs have topk=128, dictionary_size=8192 per YAML override."""
        cfg = SAETrainingConfig.from_yaml(self.YAML_PATH, "sae_delta_early")
        assert cfg.topk == 128, f"Expected topk=128 for early, got {cfg.topk}"
        assert cfg.dictionary_size == 8192, (
            f"Expected dict=8192 for early, got {cfg.dictionary_size}"
        )
        assert cfg.resample_every_n_steps == 10000, (
            f"Expected resample=10000 for early, got {cfg.resample_every_n_steps}"
        )

    def test_attn_early_gets_same_overrides(self) -> None:
        """Attention early SAE should also have topk=128, dict=8192."""
        cfg = SAETrainingConfig.from_yaml(self.YAML_PATH, "sae_attn_early")
        assert cfg.topk == 128
        assert cfg.dictionary_size == 8192
        assert cfg.resample_every_n_steps == 10000

    def test_earlymid_gets_topk_96(self) -> None:
        """Early-mid SAEs have topk=96 but default dictionary_size."""
        cfg = SAETrainingConfig.from_yaml(self.YAML_PATH, "sae_delta_earlymid")
        assert cfg.topk == 96, f"Expected topk=96 for earlymid, got {cfg.topk}"
        # dictionary_size should fall back to global default
        assert cfg.dictionary_size == 16384, (
            f"Expected default dict=16384 for earlymid, got {cfg.dictionary_size}"
        )

    def test_mid_uses_global_defaults(self) -> None:
        """Mid SAEs use global defaults (no per-hook overrides)."""
        cfg = SAETrainingConfig.from_yaml(self.YAML_PATH, "sae_delta_mid")
        assert cfg.topk == 64, f"Expected default topk=64 for mid, got {cfg.topk}"
        assert cfg.dictionary_size == 16384
        assert cfg.resample_every_n_steps == 5000

    def test_late_uses_global_defaults(self) -> None:
        """Late SAEs use global defaults."""
        cfg = SAETrainingConfig.from_yaml(self.YAML_PATH, "sae_attn_late")
        assert cfg.topk == 64
        assert cfg.dictionary_size == 16384

    def test_layer_and_type_parsed_correctly(self) -> None:
        """Layer index and type are extracted from hook_point, not top-level."""
        cfg = SAETrainingConfig.from_yaml(self.YAML_PATH, "sae_attn_mid")
        assert cfg.layer == 23
        assert cfg.layer_type.value == "attention"
        assert cfg.sae_id == "sae_attn_mid"

    def test_hidden_dim_from_model_yaml(self) -> None:
        """hidden_dim should be read from model.yaml (2048 for Qwen 3.5-A3B)."""
        cfg = SAETrainingConfig.from_yaml(self.YAML_PATH, "sae_delta_mid")
        assert cfg.hidden_dim == 2048

    def test_all_sae_ids_loadable(self) -> None:
        """Every SAE ID in the YAML can be loaded without error."""
        import yaml
        with open(self.YAML_PATH) as f:
            data = yaml.safe_load(f)
        for sae_id in data["hook_points"]:
            cfg = SAETrainingConfig.from_yaml(self.YAML_PATH, sae_id)
            assert cfg.sae_id == sae_id
            assert cfg.layer >= 0
            assert cfg.dictionary_size > 0
            assert cfg.topk > 0


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

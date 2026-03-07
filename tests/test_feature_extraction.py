"""Tests for the feature extraction pipeline.

Verifies dtype handling, pooling strategies, and SAE integration.
All tests run on CPU with mock models and small SAEs.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.features.extraction import FeatureExtractor, PoolingStrategy
from src.sae.model import TopKSAE


class MockLayer(nn.Module):
    """Mock transformer layer returning tuple output."""

    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return (self.linear(x) + x, None)


class MockModel:
    """Mock model compatible with get_layers_module() and FeatureExtractor."""

    def __init__(self, num_layers: int = 4, hidden_dim: int = 32) -> None:
        self.model = type("Model", (), {"layers": nn.ModuleList(
            [MockLayer(hidden_dim) for _ in range(num_layers)]
        )})()
        self.hidden_dim = hidden_dim

    def __call__(self, **kwargs) -> torch.Tensor:
        x = kwargs.get("input_ids")
        if x is None:
            x = torch.randn(1, 10, self.hidden_dim)
        if x.dim() == 2:
            x = x.unsqueeze(-1).expand(-1, -1, self.hidden_dim).float()
        attention_mask = kwargs.get("attention_mask")
        for layer in self.model.layers:
            output = layer(x)
            x = output[0]
        return x


class MockTokenizer:
    """Mock tokenizer that produces fixed-shape outputs."""

    def __init__(self, hidden_dim: int = 32, seq_len: int = 20) -> None:
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len

    def apply_chat_template(self, messages, **kwargs):
        return "mock template text"

    def __call__(self, text, **kwargs):
        return {
            "input_ids": torch.randn(1, self.seq_len, self.hidden_dim),
            "attention_mask": torch.ones(1, self.seq_len, dtype=torch.long),
        }


class TestFeatureExtractorDtype:
    """Verify dtype handling between model and SAE."""

    def test_sae_dtypes_cached(self) -> None:
        """SAE dtypes should be cached at init time."""
        model = MockModel(num_layers=4, hidden_dim=32)
        tokenizer = MockTokenizer(hidden_dim=32)
        sae = TopKSAE(hidden_dim=32, dict_size=64, k=4)
        sae_dict = {"test_sae": sae}
        layer_map = {"test_sae": 1}

        extractor = FeatureExtractor(
            model=model, tokenizer=tokenizer,
            sae_dict=sae_dict, layer_map=layer_map,
            device="cpu",
        )

        assert "test_sae" in extractor._sae_dtypes
        assert extractor._sae_dtypes["test_sae"] == torch.float32

    def test_multiple_saes_different_dtypes(self) -> None:
        """Extractor should handle SAEs with different dtypes."""
        model = MockModel(num_layers=4, hidden_dim=32)
        tokenizer = MockTokenizer(hidden_dim=32)

        sae_fp32 = TopKSAE(hidden_dim=32, dict_size=64, k=4)
        sae_bf16 = TopKSAE(hidden_dim=32, dict_size=64, k=4).to(torch.bfloat16)
        sae_dict = {"sae_fp32": sae_fp32, "sae_bf16": sae_bf16}
        layer_map = {"sae_fp32": 1, "sae_bf16": 2}

        extractor = FeatureExtractor(
            model=model, tokenizer=tokenizer,
            sae_dict=sae_dict, layer_map=layer_map,
            device="cpu",
        )

        assert extractor._sae_dtypes["sae_fp32"] == torch.float32
        assert extractor._sae_dtypes["sae_bf16"] == torch.bfloat16


class TestPoolingStrategies:
    """Verify all pooling strategies produce correct shapes and behavior."""

    @pytest.fixture
    def extractor(self) -> FeatureExtractor:
        model = MockModel(num_layers=4, hidden_dim=32)
        tokenizer = MockTokenizer(hidden_dim=32)
        sae = TopKSAE(hidden_dim=32, dict_size=64, k=4)
        return FeatureExtractor(
            model=model, tokenizer=tokenizer,
            sae_dict={"test": sae}, layer_map={"test": 1},
            device="cpu", pooling_strategy="mean",
        )

    def test_mean_pooling_shape(self, extractor: FeatureExtractor) -> None:
        """Mean pooling should produce (1, dict_size) output."""
        features = torch.randn(1, 10, 64)
        mask = torch.ones(1, 10, dtype=torch.long)
        extractor.pooling_strategy = "mean"
        pooled = extractor._pool_features(features, mask)
        assert pooled.shape == (1, 64)

    def test_max_pooling_shape(self, extractor: FeatureExtractor) -> None:
        """Max pooling should produce (1, dict_size) output."""
        features = torch.randn(1, 10, 64)
        mask = torch.ones(1, 10, dtype=torch.long)
        extractor.pooling_strategy = "max"
        pooled = extractor._pool_features(features, mask)
        assert pooled.shape == (1, 64)

    def test_last_token_pooling_shape(self, extractor: FeatureExtractor) -> None:
        """Last-token pooling should produce (1, dict_size) output."""
        features = torch.randn(1, 10, 64)
        mask = torch.ones(1, 10, dtype=torch.long)
        extractor.pooling_strategy = "last_token"
        pooled = extractor._pool_features(features, mask)
        assert pooled.shape == (1, 64)

    def test_last_n_pooling_shape(self, extractor: FeatureExtractor) -> None:
        """Last-n pooling should produce (1, dict_size) output."""
        features = torch.randn(1, 10, 64)
        mask = torch.ones(1, 10, dtype=torch.long)
        extractor.pooling_strategy = "last_n"
        pooled = extractor._pool_features(features, mask)
        assert pooled.shape == (1, 64)

    def test_last_token_selects_correct_position(self, extractor: FeatureExtractor) -> None:
        """Last-token pooling should use the last non-padding token."""
        features = torch.zeros(1, 5, 64)
        features[0, 2, :] = 1.0  # Token at position 2
        # Mask: tokens 0-2 are real, 3-4 are padding
        mask = torch.tensor([[1, 1, 1, 0, 0]])
        extractor.pooling_strategy = "last_token"
        pooled = extractor._pool_features(features, mask)
        assert torch.allclose(pooled, torch.ones(1, 64))

    def test_mean_pooling_excludes_padding(self, extractor: FeatureExtractor) -> None:
        """Mean pooling should ignore padded positions."""
        features = torch.zeros(1, 4, 64)
        features[0, 0, :] = 2.0
        features[0, 1, :] = 4.0
        # Tokens 0-1 are real, 2-3 are padding with garbage values
        features[0, 2, :] = 100.0
        features[0, 3, :] = 100.0
        mask = torch.tensor([[1, 1, 0, 0]])
        extractor.pooling_strategy = "mean"
        pooled = extractor._pool_features(features, mask)
        expected = torch.full((1, 64), 3.0)  # (2 + 4) / 2
        assert torch.allclose(pooled, expected)

    def test_max_pooling_excludes_padding(self, extractor: FeatureExtractor) -> None:
        """Max pooling should use -inf for padded positions."""
        features = torch.zeros(1, 4, 64)
        features[0, 0, :] = 2.0
        features[0, 1, :] = 4.0
        features[0, 2, :] = 100.0  # padding — should be ignored
        features[0, 3, :] = 100.0
        mask = torch.tensor([[1, 1, 0, 0]])
        extractor.pooling_strategy = "max"
        pooled = extractor._pool_features(features, mask)
        expected = torch.full((1, 64), 4.0)
        assert torch.allclose(pooled, expected)

    def test_invalid_strategy_raises(self, extractor: FeatureExtractor) -> None:
        """Invalid pooling strategy should raise ValueError."""
        features = torch.randn(1, 10, 64)
        mask = torch.ones(1, 10, dtype=torch.long)
        extractor.pooling_strategy = "invalid"  # type: ignore[assignment]
        with pytest.raises(ValueError, match="Unknown pooling strategy"):
            extractor._pool_features(features, mask)


class TestTASVectorization:
    """Verify the vectorized TAS computation produces correct results."""

    def test_vectorized_matches_loop(self) -> None:
        """Vectorized compute_all_parametric_pvalues should match loop version."""
        import numpy as np
        from scipy import stats

        # Create synthetic pair results
        from src.features.extraction import FeaturePairResult, FeatureExtractionResults
        from src.data.contrastive import BehavioralTrait

        np.random.seed(42)
        dict_size = 16
        n_pairs = 20

        pair_results = []
        for _ in range(n_pairs):
            high = np.random.randn(dict_size).tolist()
            low = np.random.randn(dict_size).tolist()
            pair_results.append(FeaturePairResult(
                pair_id="test",
                sae_id="test_sae",
                pooling_strategy="last_token",
                features_high_mean=high,
                features_low_mean=low,
            ))

        extraction = FeatureExtractionResults(
            trait=BehavioralTrait.AUTONOMY,
            results={"test_sae": pair_results},
        )

        # Vectorized version
        from src.features.scoring import compute_all_parametric_pvalues
        p_vectorized = compute_all_parametric_pvalues(extraction, "test_sae")

        # Loop version (reference)
        p_loop = np.zeros(dict_size)
        for i in range(dict_size):
            highs = np.array([r.features_high_mean[i] for r in pair_results])
            lows = np.array([r.features_low_mean[i] for r in pair_results])
            diffs = highs - lows
            std_diff = np.std(diffs, ddof=1)
            if std_diff < 1e-8:
                p_loop[i] = 1.0 if abs(np.mean(diffs)) < 1e-8 else 0.0
            else:
                t_stat = np.mean(diffs) / (std_diff / np.sqrt(n_pairs))
                p_loop[i] = 2 * stats.t.sf(abs(t_stat), df=n_pairs - 1)

        np.testing.assert_allclose(p_vectorized, p_loop, rtol=1e-10)

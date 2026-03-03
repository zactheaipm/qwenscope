"""Tests for TAS computation.

Verifies TAS scoring with known inputs/outputs on CPU.
"""

from __future__ import annotations

import pytest
import torch

from src.data.contrastive import BehavioralTrait, TaskDomain, SUB_BEHAVIOR_TEMPLATES
from src.features.extraction import FeatureExtractionResults, FeaturePairResult
from src.features.scoring import (
    compute_tas, rank_features, statistical_significance,
    compute_sub_behavior_tas, compute_all_sub_behavior_tas, SUB_BEHAVIOR_KEYS,
)


def _make_extraction_results(
    n_pairs: int = 10,
    dict_size: int = 128,
    diff_feature: int = 0,
    diff_magnitude: float = 5.0,
) -> FeatureExtractionResults:
    """Create mock extraction results with a known strong feature.

    Args:
        n_pairs: Number of contrastive pairs.
        dict_size: Feature dictionary size.
        diff_feature: Index of the feature with a strong difference.
        diff_magnitude: How much the diff_feature differs between high and low.

    Returns:
        FeatureExtractionResults with controlled data.
    """
    torch.manual_seed(123)
    pair_results = []
    for i in range(n_pairs):
        # Base features: independent random for high and low
        high = torch.randn(dict_size) * 0.1  # small random noise
        low = torch.randn(dict_size) * 0.1
        # Make the target feature strongly and consistently different
        # Use independent noise so std(diff) > 0 but mean(diff) >> std(diff)
        high[diff_feature] = diff_magnitude + torch.randn(1).item() * 0.5
        low[diff_feature] = -diff_magnitude + torch.randn(1).item() * 0.5

        pair_results.append(FeaturePairResult(
            pair_id=f"test_{i}",
            sae_id="sae_test",
            pooling_strategy="last_token",
            features_high_mean=high.tolist(),
            features_low_mean=low.tolist(),
        ))

    return FeatureExtractionResults(
        trait=BehavioralTrait.AUTONOMY,
        results={"sae_test": pair_results},
    )


class TestComputeTAS:
    """Test TAS computation."""

    def test_basic_computation(self) -> None:
        """TAS is computed correctly for a known difference."""
        results = _make_extraction_results(n_pairs=20, dict_size=64, diff_feature=5, diff_magnitude=3.0)
        tas = compute_tas(results, BehavioralTrait.AUTONOMY, "sae_test")

        assert tas.shape == (64,)
        # Feature 5 should have the highest |TAS|
        assert tas[5].abs() > tas.abs().median()

    def test_strong_feature_has_highest_tas(self) -> None:
        """The engineered strong feature should have the highest TAS."""
        results = _make_extraction_results(n_pairs=30, dict_size=128, diff_feature=42, diff_magnitude=10.0)
        tas = compute_tas(results, BehavioralTrait.AUTONOMY, "sae_test")

        top = rank_features(tas, top_k=1)
        assert top[0][0] == 42, f"Expected feature 42 to be top, got {top[0][0]}"

    def test_positive_tas_for_positive_difference(self) -> None:
        """Features with consistently higher activation in HIGH should have positive TAS."""
        results = _make_extraction_results(n_pairs=20, dict_size=32, diff_feature=0, diff_magnitude=5.0)
        tas = compute_tas(results, BehavioralTrait.AUTONOMY, "sae_test")

        # Feature 0: high has +5, low has -5, so diff = +10
        assert tas[0] > 0, f"Expected positive TAS, got {tas[0]}"

    def test_zero_std_gets_zero_tas(self) -> None:
        """Features with zero variance in differences get TAS = 0."""
        # All pairs have exactly the same diff → std = 0
        pair_results = []
        for i in range(10):
            pair_results.append(FeaturePairResult(
                pair_id=f"test_{i}",
                sae_id="sae_test",
                pooling_strategy="last_token",
                features_high_mean=[1.0, 0.0],
                features_low_mean=[1.0, 0.0],  # No difference
            ))

        results = FeatureExtractionResults(
            trait=BehavioralTrait.AUTONOMY,
            results={"sae_test": pair_results},
        )

        tas = compute_tas(results, BehavioralTrait.AUTONOMY, "sae_test")
        assert (tas == 0).all()


class TestRankFeatures:
    """Test feature ranking."""

    def test_returns_correct_count(self) -> None:
        """rank_features returns exactly top_k items."""
        tas = torch.randn(256)
        top = rank_features(tas, top_k=10, positive_only=False)
        assert len(top) == 10

    def test_sorted_by_absolute_tas(self) -> None:
        """Results are sorted by |TAS| descending."""
        tas = torch.randn(128)
        top = rank_features(tas, top_k=20, positive_only=False)

        abs_values = [abs(t) for _, t in top]
        assert abs_values == sorted(abs_values, reverse=True)

    def test_top_k_clamped_to_dict_size(self) -> None:
        """Requesting more than dict_size returns all features."""
        tas = torch.randn(10)
        top = rank_features(tas, top_k=100, positive_only=False)
        assert len(top) == 10

    def test_positive_only_excludes_negative(self) -> None:
        """positive_only=True returns only features with TAS > 0."""
        tas = torch.tensor([3.0, -2.0, 1.0, -5.0, 0.5])
        top = rank_features(tas, top_k=10, positive_only=True)
        for idx, score in top:
            assert score > 0, f"Feature {idx} has non-positive TAS {score}"
        assert len(top) == 3  # features 0, 2, 4


class TestStatisticalSignificance:
    """Test permutation-based significance testing."""

    def test_strong_feature_is_significant(self) -> None:
        """A strongly different feature should have low p-value."""
        results = _make_extraction_results(
            n_pairs=30, dict_size=64, diff_feature=10, diff_magnitude=10.0
        )
        p_value = statistical_significance(
            results, BehavioralTrait.AUTONOMY, "sae_test",
            feature_idx=10, n_permutations=200, seed=42,
        )
        assert p_value < 0.05, f"Expected p < 0.05, got {p_value}"

    def test_random_feature_is_not_significant(self) -> None:
        """A random feature should have high p-value (> 0.05 usually)."""
        results = _make_extraction_results(
            n_pairs=30, dict_size=64, diff_feature=10, diff_magnitude=10.0
        )
        # Feature 50 has no engineered difference
        p_value = statistical_significance(
            results, BehavioralTrait.AUTONOMY, "sae_test",
            feature_idx=50, n_permutations=200, seed=42,
        )
        # Can't guarantee p > 0.05 for random, but it should typically be
        # much higher than the strong feature
        assert p_value > 0.01, f"Expected non-significant p, got {p_value}"


def _make_annotated_extraction_results(
    n_annotated: int = 12,
    n_unannotated: int = 20,
    dict_size: int = 64,
    sub_behavior: str = "autonomy.action_initiation",
    annotated_diff_feature: int = 5,
    annotated_magnitude: float = 8.0,
    unannotated_diff_feature: int = 10,
    unannotated_magnitude: float = 8.0,
) -> FeatureExtractionResults:
    """Create extraction results with annotated and unannotated pairs.

    Annotated pairs have a strong signal on ``annotated_diff_feature``.
    Unannotated pairs have a strong signal on ``unannotated_diff_feature``.
    This lets us verify that sub-behavior TAS filters correctly.
    """
    torch.manual_seed(456)
    pair_results = []

    # Annotated pairs: strong signal on annotated_diff_feature
    for i in range(n_annotated):
        high = torch.randn(dict_size) * 0.1
        low = torch.randn(dict_size) * 0.1
        high[annotated_diff_feature] = annotated_magnitude + torch.randn(1).item() * 0.3
        low[annotated_diff_feature] = -annotated_magnitude + torch.randn(1).item() * 0.3
        pair_results.append(FeaturePairResult(
            pair_id=f"sub_test_{i}",
            sae_id="sae_test",
            pooling_strategy="last_token",
            features_high_mean=high.tolist(),
            features_low_mean=low.tolist(),
            target_sub_behaviors=[sub_behavior],
        ))

    # Unannotated pairs: strong signal on unannotated_diff_feature
    for i in range(n_unannotated):
        high = torch.randn(dict_size) * 0.1
        low = torch.randn(dict_size) * 0.1
        high[unannotated_diff_feature] = unannotated_magnitude + torch.randn(1).item() * 0.3
        low[unannotated_diff_feature] = -unannotated_magnitude + torch.randn(1).item() * 0.3
        pair_results.append(FeaturePairResult(
            pair_id=f"comp_test_{i}",
            sae_id="sae_test",
            pooling_strategy="last_token",
            features_high_mean=high.tolist(),
            features_low_mean=low.tolist(),
            target_sub_behaviors=[],
        ))

    return FeatureExtractionResults(
        trait=BehavioralTrait.AUTONOMY,
        results={"sae_test": pair_results},
    )


class TestSubBehaviorTAS:
    """Tests for sub-behavior-level TAS computation."""

    def test_filters_by_annotation(self) -> None:
        """Sub-behavior TAS should use only annotated pairs."""
        results = _make_annotated_extraction_results(
            n_annotated=12,
            n_unannotated=20,
            annotated_diff_feature=5,
            unannotated_diff_feature=10,
        )
        tas, n_pairs, used_fallback = compute_sub_behavior_tas(
            results, "sae_test", "autonomy.action_initiation",
        )

        assert not used_fallback, "Should not fall back with 12 annotated pairs"
        assert n_pairs == 12, f"Expected 12 annotated pairs, got {n_pairs}"
        assert tas is not None
        # Feature 5 (annotated signal) should be top, not feature 10
        top_feature = tas.abs().argmax().item()
        assert top_feature == 5, (
            f"Expected feature 5 to be top (annotated signal), got {top_feature}"
        )

    def test_fallback_when_too_few_pairs(self) -> None:
        """Returns None when fewer than min_pairs are annotated."""
        results = _make_annotated_extraction_results(
            n_annotated=3,  # Fewer than default min_pairs=8
            n_unannotated=20,
        )
        tas, n_pairs, used_fallback = compute_sub_behavior_tas(
            results, "sae_test", "autonomy.action_initiation",
        )

        assert used_fallback, "Should fall back with only 3 annotated pairs"
        assert tas is None, "TAS should be None when below min_pairs"
        assert n_pairs == 3, f"Expected 3 annotated pairs found, got {n_pairs}"

    def test_rejects_unknown_sub_behavior(self) -> None:
        """Raises ValueError for unknown sub-behavior key."""
        results = _make_annotated_extraction_results(n_annotated=10)
        with pytest.raises(ValueError, match="Unknown sub-behavior"):
            compute_sub_behavior_tas(results, "sae_test", "fake.sub_behavior")

    def test_compute_all_sub_behavior_tas(self) -> None:
        """compute_all_sub_behavior_tas iterates over all 15 sub-behaviors."""
        results = _make_annotated_extraction_results(
            n_annotated=12, n_unannotated=20,
        )
        all_results = compute_all_sub_behavior_tas(
            {BehavioralTrait.AUTONOMY: results}
        )

        # Should have entries for all 3 autonomy sub-behaviors
        for sub in ["autonomy.decision_independence", "autonomy.action_initiation",
                     "autonomy.permission_avoidance"]:
            assert sub in all_results, f"Missing {sub}"
            assert "sae_test" in all_results[sub]

        # Non-autonomy sub-behaviors should be absent (no extraction data)
        for sub in ["persistence.retry_willingness", "deference.instruction_literalness"]:
            assert sub not in all_results


class TestSubBehaviorTemplateStructure:
    """Verify SUB_BEHAVIOR_TEMPLATES has the expected structure."""

    def test_all_15_sub_behaviors_present(self) -> None:
        """All 15 sub-behavior keys are in SUB_BEHAVIOR_TEMPLATES."""
        for key in SUB_BEHAVIOR_KEYS:
            assert key in SUB_BEHAVIOR_TEMPLATES, f"Missing sub-behavior: {key}"

    def test_all_4_domains_per_sub_behavior(self) -> None:
        """Each sub-behavior has all 4 task domains."""
        for key, domain_dict in SUB_BEHAVIOR_TEMPLATES.items():
            for domain in TaskDomain:
                assert domain in domain_dict, (
                    f"Missing domain {domain.value} in {key}"
                )

    def test_3_templates_per_domain(self) -> None:
        """Each sub-behavior × domain has exactly 3 templates."""
        for key, domain_dict in SUB_BEHAVIOR_TEMPLATES.items():
            for domain in TaskDomain:
                templates = domain_dict[domain]
                assert len(templates) == 3, (
                    f"{key}/{domain.value}: expected 3 templates, got {len(templates)}"
                )

    def test_4_variations_per_template(self) -> None:
        """Each template has exactly 4 slot-fill variations."""
        for key, domain_dict in SUB_BEHAVIOR_TEMPLATES.items():
            for domain in TaskDomain:
                for i, t in enumerate(domain_dict[domain]):
                    assert len(t["variations"]) == 4, (
                        f"{key}/{domain.value}[{i}]: expected 4 variations, "
                        f"got {len(t['variations'])}"
                    )

    def test_total_template_count(self) -> None:
        """Total templates: 15 × 4 × 3 = 180."""
        total = sum(
            len(templates)
            for domain_dict in SUB_BEHAVIOR_TEMPLATES.values()
            for templates in domain_dict.values()
        )
        assert total == 180, f"Expected 180 templates, got {total}"

    def test_required_keys_present(self) -> None:
        """Every template has all required keys."""
        required = {"system_high", "system_low", "user_template", "tools",
                     "expected_high", "expected_low", "variations"}
        for key, domain_dict in SUB_BEHAVIOR_TEMPLATES.items():
            for domain in TaskDomain:
                for i, t in enumerate(domain_dict[domain]):
                    missing = required - set(t.keys())
                    assert not missing, (
                        f"{key}/{domain.value}[{i}]: missing keys {missing}"
                    )

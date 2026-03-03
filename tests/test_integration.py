"""End-to-end integration test for the pilot pipeline with mock data.

Runs the pipeline steps in order using a tiny mock model (random tensors,
no actual transformer), a tiny SAE (hidden_dim=16, dict_size=32, k=4),
and synthetic contrastive/behavioral data.  Validates that all pipeline
stages compose correctly without GPU, model weights, or API keys.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from src.data.contrastive import BehavioralTrait
from src.evaluation.behavioral_metrics import (
    AutonomySubScores,
    BehavioralScore,
    DeferenceSubScores,
    PersistenceSubScores,
    RiskCalibrationSubScores,
    ToolUseSubScores,
)
from src.evaluation.contamination import (
    TRAIT_ORDER,
    TRAIT_SCORE_KEYS,
    compute_contamination_matrix,
    contamination_summary,
)
from src.features.extraction import FeatureExtractionResults, FeaturePairResult
from src.features.scoring import compute_tas, rank_features
from src.sae.model import TopKSAE
from src.analysis.effect_sizes import (
    cohens_d,
    cohens_d_paired,
    probability_of_superiority,
    bootstrap_ci,
    bootstrap_ci_difference,
    compute_selectivity_per_trait,
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

SEED = 42


@pytest.fixture(autouse=True)
def _seed_everything() -> None:
    """Fix all random seeds for deterministic tests."""
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)


# ---------------------------------------------------------------------------
# Tiny dimensions used throughout
# ---------------------------------------------------------------------------

HIDDEN_DIM = 16
DICT_SIZE = 32
K = 4
N_PAIRS = 2  # 2 contrastive pairs per trait (minimal)
SAE_ID = "sae_mock"
DIFF_FEATURE = 7  # Feature that carries the engineered trait signal


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_sae() -> TopKSAE:
    """A tiny TopKSAE on CPU for pipeline testing."""
    return TopKSAE(hidden_dim=HIDDEN_DIM, dict_size=DICT_SIZE, k=K)


def _make_behavioral_score(
    autonomy: float = 0.5,
    tool_use: float = 0.5,
    persistence: float = 0.5,
    risk: float = 0.5,
    deference: float = 0.5,
) -> BehavioralScore:
    """Create a BehavioralScore with uniform sub-behavior values per trait.

    Args:
        autonomy: Value applied uniformly to all 3 autonomy sub-behaviors.
        tool_use: Value applied uniformly to all 3 tool_use sub-behaviors.
        persistence: Value applied uniformly to all 3 persistence sub-behaviors.
        risk: Value applied uniformly to all 3 risk_calibration sub-behaviors.
        deference: Value applied uniformly to all 3 deference sub-behaviors.

    Returns:
        BehavioralScore with the specified composite trait values.
    """
    return BehavioralScore(
        autonomy=AutonomySubScores(
            decision_independence=autonomy,
            action_initiation=autonomy,
            permission_avoidance=autonomy,
        ),
        tool_use=ToolUseSubScores(
            tool_reach=tool_use,
            proactive_information_gathering=tool_use,
            tool_diversity=tool_use,
        ),
        persistence=PersistenceSubScores(
            retry_willingness=persistence,
            strategy_variation=persistence,
            escalation_reluctance=persistence,
        ),
        risk_calibration=RiskCalibrationSubScores(
            approach_novelty=risk,
            scope_expansion=risk,
            uncertainty_tolerance=risk,
        ),
        deference=DeferenceSubScores(
            instruction_literalness=deference,
            challenge_avoidance=deference,
            suggestion_restraint=deference,
        ),
    )


def _make_mock_extraction_results(
    n_pairs: int = N_PAIRS,
    dict_size: int = DICT_SIZE,
    sae_id: str = SAE_ID,
    diff_feature: int = DIFF_FEATURE,
    diff_magnitude: float = 5.0,
    trait: BehavioralTrait = BehavioralTrait.AUTONOMY,
) -> FeatureExtractionResults:
    """Create mock FeatureExtractionResults with one strong feature.

    Args:
        n_pairs: Number of contrastive pairs.
        dict_size: Feature dictionary size.
        sae_id: Identifier for the mock SAE.
        diff_feature: Index of the feature with a strong HIGH-LOW difference.
        diff_magnitude: How much that feature differs between HIGH and LOW.
        trait: Behavioral trait label for these results.

    Returns:
        FeatureExtractionResults ready for TAS computation.
    """
    pair_results: list[FeaturePairResult] = []
    for i in range(n_pairs):
        high = (torch.randn(dict_size) * 0.1).tolist()
        low = (torch.randn(dict_size) * 0.1).tolist()
        # Inject a consistent strong signal on diff_feature
        high[diff_feature] = diff_magnitude + random.gauss(0, 0.3)
        low[diff_feature] = -diff_magnitude + random.gauss(0, 0.3)

        pair_results.append(
            FeaturePairResult(
                pair_id=f"mock_{trait.value}_{i:03d}",
                sae_id=sae_id,
                pooling_strategy="last_token",
                features_high_mean=high,
                features_low_mean=low,
            )
        )

    return FeatureExtractionResults(
        trait=trait,
        results={sae_id: pair_results},
    )


# ---------------------------------------------------------------------------
# Integration test class
# ---------------------------------------------------------------------------


class TestPilotPipelineIntegration:
    """End-to-end integration test for the mock pilot pipeline."""

    # -- Step (a): SAE encode / decode --

    def test_sae_encode_decode(self, tiny_sae: TopKSAE) -> None:
        """Verify that encode and decode produce correct shapes and sparsity."""
        x = torch.randn(2, 8, HIDDEN_DIM)  # (batch, seq_len, hidden_dim)

        features = tiny_sae.encode(x)
        assert features.shape == (2, 8, DICT_SIZE)

        # Exactly k features should be nonzero per position
        nonzero_per_pos = (features.abs() > 0).sum(dim=-1)
        assert (nonzero_per_pos == K).all(), (
            f"Expected k={K} nonzero features, got {nonzero_per_pos.unique().tolist()}"
        )

        reconstruction = tiny_sae.decode(features)
        assert reconstruction.shape == x.shape

    def test_sae_forward_returns_scalar_loss(self, tiny_sae: TopKSAE) -> None:
        """Forward pass returns reconstruction, features, and a scalar loss."""
        x = torch.randn(4, HIDDEN_DIM)
        reconstruction, features, loss = tiny_sae(x)

        assert reconstruction.shape == (4, HIDDEN_DIM)
        assert features.shape == (4, DICT_SIZE)
        assert loss.dim() == 0
        assert loss.item() >= 0.0

    # -- Steps (b-c): Mock contrastive pairs and feature extraction --

    def test_mock_contrastive_pair_construction(self) -> None:
        """FeaturePairResult objects are well-formed."""
        results = _make_mock_extraction_results()

        assert BehavioralTrait.AUTONOMY == results.trait
        pair_list = results.results[SAE_ID]
        assert len(pair_list) == N_PAIRS

        for pr in pair_list:
            assert len(pr.features_high_mean) == DICT_SIZE
            assert len(pr.features_low_mean) == DICT_SIZE
            assert pr.sae_id == SAE_ID

    # -- Steps (d-e): Compute TAS and verify shape / non-triviality --

    def test_tas_shape_and_nontriviality(self) -> None:
        """TAS has correct shape and is not all zeros."""
        results = _make_mock_extraction_results(diff_magnitude=8.0)
        tas = compute_tas(results, BehavioralTrait.AUTONOMY, SAE_ID)

        assert tas.shape == (DICT_SIZE,)
        assert not torch.allclose(tas, torch.zeros(DICT_SIZE, dtype=tas.dtype)), (
            "TAS should not be all zeros when contrastive data has a strong signal"
        )

    def test_tas_identifies_engineered_feature(self) -> None:
        """The engineered diff_feature should have the highest |TAS|."""
        results = _make_mock_extraction_results(
            n_pairs=10, diff_magnitude=10.0,
        )
        tas = compute_tas(results, BehavioralTrait.AUTONOMY, SAE_ID)

        top = rank_features(tas, top_k=1)
        assert top[0][0] == DIFF_FEATURE, (
            f"Expected feature {DIFF_FEATURE} at top, got {top[0][0]}"
        )

    # -- Steps (f-h): BehavioralScore, contamination matrix --

    def test_behavioral_score_trait_scores(self) -> None:
        """BehavioralScore.trait_scores() returns all 5 traits."""
        score = _make_behavioral_score(
            autonomy=0.8, tool_use=0.3, persistence=0.6,
            risk=0.4, deference=0.9,
        )
        ts = score.trait_scores()
        assert set(ts.keys()) == set(TRAIT_SCORE_KEYS)
        assert ts["autonomy"] == pytest.approx(0.8)
        assert ts["deference"] == pytest.approx(0.9)

    def test_contamination_matrix_shape(self) -> None:
        """Contamination matrix is (5, 5)."""
        baseline = [_make_behavioral_score() for _ in range(5)]

        steered: dict[BehavioralTrait, list[BehavioralScore]] = {}
        for trait in TRAIT_ORDER:
            # Steering the target trait increases its score from 0.5 to 0.8
            kwargs = {
                "autonomy": 0.5,
                "tool_use": 0.5,
                "persistence": 0.5,
                "risk": 0.5,
                "deference": 0.5,
            }
            # Map enum to kwarg name
            kwarg_map = {
                BehavioralTrait.AUTONOMY: "autonomy",
                BehavioralTrait.TOOL_USE: "tool_use",
                BehavioralTrait.PERSISTENCE: "persistence",
                BehavioralTrait.RISK_CALIBRATION: "risk",
                BehavioralTrait.DEFERENCE: "deference",
            }
            kwargs[kwarg_map[trait]] = 0.8
            steered[trait] = [_make_behavioral_score(**kwargs) for _ in range(5)]

        matrix = compute_contamination_matrix(baseline, steered)

        assert matrix.shape == (5, 5)
        # Diagonal should be positive (steering increased the target)
        for i in range(5):
            assert matrix[i, i] > 0, f"Diagonal[{i}] should be positive"
        # Off-diagonal should be zero (we only changed the target trait)
        for i in range(5):
            for j in range(5):
                if i != j:
                    assert matrix[i, j] == pytest.approx(0.0, abs=1e-8), (
                        f"Off-diag[{i},{j}] should be ~0, got {matrix[i, j]}"
                    )

    # -- Step (i): contamination_summary --

    def test_contamination_summary_valid_output(self) -> None:
        """contamination_summary returns all expected keys with valid values."""
        # Build a contamination matrix with known structure:
        # diagonal = 0.3, off-diagonal = 0.05
        matrix = np.full((5, 5), 0.05)
        np.fill_diagonal(matrix, 0.3)

        summary = contamination_summary(matrix)

        expected_keys = {
            "mean_intended_effect",
            "mean_contamination",
            "max_contamination",
            "selectivity_ratio",
            "cleanest_trait",
            "most_contaminating_trait",
        }
        assert set(summary.keys()) == expected_keys

        assert summary["mean_intended_effect"] == pytest.approx(0.3)
        assert summary["mean_contamination"] == pytest.approx(0.05)
        assert summary["max_contamination"] == pytest.approx(0.05)
        assert summary["selectivity_ratio"] > 1.0
        assert summary["cleanest_trait"] in TRAIT_SCORE_KEYS
        assert summary["most_contaminating_trait"] in TRAIT_SCORE_KEYS

    # -- Step (j): Effect size module --

    def test_cohens_d_basic(self) -> None:
        """Cohen's d is positive when group1 mean > group2 mean."""
        group1 = [0.8, 0.85, 0.9, 0.75, 0.82]
        group2 = [0.3, 0.35, 0.4, 0.28, 0.32]
        d = cohens_d(group1, group2)
        assert d > 0, f"Expected positive Cohen's d, got {d}"
        # With these well-separated groups, d should be large
        assert d > 1.0, f"Expected d > 1.0, got {d}"

    def test_cohens_d_paired(self) -> None:
        """Cohen's d_paired is positive when post > pre."""
        pre = [0.3, 0.35, 0.4, 0.28, 0.32]
        post = [0.8, 0.85, 0.9, 0.75, 0.82]
        d = cohens_d_paired(pre, post)
        assert d > 0, f"Expected positive paired d, got {d}"

    def test_cohens_d_paired_requires_equal_length(self) -> None:
        """cohens_d_paired raises ValueError for unequal lengths."""
        with pytest.raises(ValueError, match="equal lengths"):
            cohens_d_paired([0.1, 0.2], [0.3, 0.4, 0.5])

    def test_probability_of_superiority_separated_groups(self) -> None:
        """P(superiority) = 1.0 when groups are fully separated."""
        group1 = [10.0, 11.0, 12.0]
        group2 = [1.0, 2.0, 3.0]
        p = probability_of_superiority(group1, group2)
        assert p == pytest.approx(1.0)

    def test_probability_of_superiority_identical_groups(self) -> None:
        """P(superiority) = 0.5 when groups are identical."""
        group = [5.0, 5.0, 5.0]
        p = probability_of_superiority(group, group)
        assert p == pytest.approx(0.5)

    def test_bootstrap_ci_returns_valid_interval(self) -> None:
        """bootstrap_ci returns a valid confidence interval."""
        data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        result = bootstrap_ci(data, statistic="mean", n_bootstrap=500, seed=SEED)

        assert "point_estimate" in result
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert result["ci_lower"] <= result["point_estimate"] <= result["ci_upper"]

    def test_bootstrap_ci_difference(self) -> None:
        """bootstrap_ci_difference returns valid CI for mean difference."""
        group1 = [0.8, 0.85, 0.9, 0.75, 0.82]
        group2 = [0.3, 0.35, 0.4, 0.28, 0.32]
        result = bootstrap_ci_difference(
            group1, group2, n_bootstrap=500, seed=SEED,
        )

        assert result["point_estimate"] > 0
        assert result["ci_lower"] <= result["point_estimate"] <= result["ci_upper"]
        # With well-separated groups, CI should not cross zero
        assert result["ci_lower"] > 0

    def test_selectivity_per_trait(self) -> None:
        """compute_selectivity_per_trait returns per-trait selectivity metrics."""
        matrix = np.full((5, 5), 0.05)
        np.fill_diagonal(matrix, 0.3)

        results = compute_selectivity_per_trait(matrix, trait_names=TRAIT_SCORE_KEYS)

        assert len(results) == 5
        for trait_name in TRAIT_SCORE_KEYS:
            metrics = results[trait_name]
            assert metrics["on_diagonal"] == pytest.approx(0.3)
            assert metrics["max_off_diagonal"] == pytest.approx(0.05)
            assert metrics["selectivity_ratio"] == pytest.approx(6.0)
            assert metrics["passes_threshold"]

    # -- Full pipeline composition --

    def test_full_pipeline_mock(self, tiny_sae: TopKSAE) -> None:
        """Run all pipeline steps in sequence on mock data.

        This verifies that the data models compose correctly from SAE
        through TAS through contamination through effect sizes.
        """
        # (a) SAE encode/decode
        x = torch.randn(1, 4, HIDDEN_DIM)
        with torch.no_grad():
            reconstruction, features, loss = tiny_sae(x)
        assert reconstruction.shape == x.shape

        # (b-c) Mock contrastive pairs + feature extraction
        extraction = _make_mock_extraction_results(
            n_pairs=10, diff_magnitude=8.0,
        )

        # (d-e) Compute TAS
        tas = compute_tas(extraction, BehavioralTrait.AUTONOMY, SAE_ID)
        assert tas.shape == (DICT_SIZE,)
        assert tas.abs().max() > 0

        # (f) Baseline and steered BehavioralScores
        baseline_scores = [
            _make_behavioral_score(
                autonomy=0.5 + random.gauss(0, 0.05),
                tool_use=0.5 + random.gauss(0, 0.05),
                persistence=0.5 + random.gauss(0, 0.05),
                risk=0.5 + random.gauss(0, 0.05),
                deference=0.5 + random.gauss(0, 0.05),
            )
            for _ in range(10)
        ]
        steered_scores: dict[BehavioralTrait, list[BehavioralScore]] = {}
        for trait in TRAIT_ORDER:
            kwarg_map = {
                BehavioralTrait.AUTONOMY: "autonomy",
                BehavioralTrait.TOOL_USE: "tool_use",
                BehavioralTrait.PERSISTENCE: "persistence",
                BehavioralTrait.RISK_CALIBRATION: "risk",
                BehavioralTrait.DEFERENCE: "deference",
            }
            scores_for_trait = []
            for _ in range(10):
                kwargs = {
                    "autonomy": 0.5 + random.gauss(0, 0.05),
                    "tool_use": 0.5 + random.gauss(0, 0.05),
                    "persistence": 0.5 + random.gauss(0, 0.05),
                    "risk": 0.5 + random.gauss(0, 0.05),
                    "deference": 0.5 + random.gauss(0, 0.05),
                }
                kwargs[kwarg_map[trait]] = 0.8 + random.gauss(0, 0.05)
                scores_for_trait.append(_make_behavioral_score(**kwargs))
            steered_scores[trait] = scores_for_trait

        # (g-h) Contamination matrix
        matrix = compute_contamination_matrix(baseline_scores, steered_scores)
        assert matrix.shape == (5, 5)

        # Diagonal should be meaningfully positive
        for i in range(5):
            assert matrix[i, i] > 0.1, (
                f"Diagonal[{i}] too small: {matrix[i, i]:.3f}"
            )

        # (i) Summary
        summary = contamination_summary(matrix)
        assert summary["mean_intended_effect"] > 0
        assert summary["selectivity_ratio"] > 0

        # (j) Effect sizes using the steered vs baseline trait scores
        baseline_autonomy = [s.autonomy_score for s in baseline_scores]
        steered_autonomy = [
            s.autonomy_score for s in steered_scores[BehavioralTrait.AUTONOMY]
        ]

        d = cohens_d(steered_autonomy, baseline_autonomy)
        assert d > 0, "Steered autonomy should exceed baseline"

        p_sup = probability_of_superiority(steered_autonomy, baseline_autonomy)
        assert p_sup > 0.5, "Steered should be superior to baseline"

"""Steering experiment runners for the three experiment types."""

from __future__ import annotations

import logging
import math
import random
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel
from scipy import stats

from src.data.contrastive import BehavioralTrait
from src.data.scenarios import EvaluationScenario
from src.evaluation.behavioral_metrics import BehavioralScore
from src.features.scoring import rank_features
from src.model.config import HOOK_POINTS, HOOK_POINTS_BY_ID, LayerType
from src.model.hooks import ActivationCache
from src.sae.model import TopKSAE
from src.steering.engine import MultiLayerSteeringEngine, SteeringEngine

logger = logging.getLogger(__name__)


class SteeringResult(BaseModel):
    """Result of one steering run on one scenario."""

    scenario_id: str
    trait: BehavioralTrait
    multiplier: float
    sae_id: str
    feature_indices: list[int]
    trajectory: dict[str, Any]  # Serialized AgentTrajectory
    behavioral_score: dict[str, Any] | None = None  # Serialized BehavioralScore (set externally)


class Experiment1Results(BaseModel):
    """Results from Experiment 1: standard steering with best SAE per trait."""

    trait: BehavioralTrait
    best_sae_id: str
    results_by_multiplier: dict[float, list[SteeringResult]]


class Experiment2Results(BaseModel):
    """Results from Experiment 2: layer-type-specific steering."""

    trait: BehavioralTrait
    deltanet_results: list[SteeringResult]
    attention_results: list[SteeringResult]
    combined_results: list[SteeringResult]


class Experiment3Results(BaseModel):
    """Results from Experiment 3: cross-depth steering."""

    trait: BehavioralTrait
    early_results: list[SteeringResult]
    mid_results: list[SteeringResult]
    late_results: list[SteeringResult]


class FeatureAblationResult(BaseModel):
    """Result of ablating a single SAE feature and measuring behavioral change.

    Used by activation patching to determine whether a high-TAS feature is
    causally involved in the target behavior (vs merely correlated).
    """

    feature_idx: int
    tas_score: float                  # Original TAS for reference
    mean_delta: float                 # mean(ablated_trait_score - baseline_trait_score)
    std_delta: float                  # std of paired deltas across scenarios
    p_value: float                    # Paired t-test p-value
    corrected_p: float                # BH FDR-corrected p-value
    is_causal: bool                   # Survives FDR threshold
    scenario_deltas: list[float]      # Per-scenario deltas for diagnostics


class ActivationPatchingResults(BaseModel):
    """Results from activation patching validation for one trait.

    Tests whether high-TAS features are causally involved in the target
    behavior by clamping each feature to zero (individually and as a group)
    and measuring the change in behavioral score.
    """

    trait: BehavioralTrait
    sae_id: str
    feature_results: list[FeatureAblationResult]
    group_ablation_mean_delta: float  # All top-k features ablated simultaneously
    group_ablation_p_value: float
    n_causal: int                     # Features surviving FDR
    n_tested: int
    causal_fraction: float


def score_steering_results(
    results: dict[BehavioralTrait, list[SteeringResult]] | list[SteeringResult],
    judge: Any,
    rate_limit_delay: float = 0.5,
) -> None:
    """Score steering results in-place using the LLM judge.

    Iterates over each SteeringResult, reconstructs the AgentTrajectory from
    the serialized trajectory dict, calls the judge, and writes the result back
    to ``result.behavioral_score``. Results that already have a behavioral_score
    are skipped (idempotent).

    This function must be called before ``_extract_behavioral_scores`` (and
    therefore before ``compare_sae_vs_mean_diff_specificity``) so that the
    contamination matrix is populated from real judge scores rather than all-
    None values that produce all-zero matrices.

    Args:
        results: Either a dict mapping BehavioralTrait → list[SteeringResult],
            or a flat list of SteeringResult objects.
        judge: An ``LLMJudge`` instance.
        rate_limit_delay: Seconds to sleep between API calls (passed through
            to the judge's internal rate limiting).
    """
    from src.evaluation.agent_harness import AgentTrajectory

    # Normalise to a flat iterable of SteeringResult objects.
    if isinstance(results, dict):
        flat_results: list[SteeringResult] = [
            r for result_list in results.values() for r in result_list
        ]
    else:
        flat_results = list(results)

    total = len(flat_results)
    n_failures = 0
    max_failure_rate = 0.5
    for i, result in enumerate(flat_results):
        if result.behavioral_score is not None:
            # Already scored — skip to keep the function idempotent.
            continue

        if not result.trajectory:
            logger.warning(
                "SteeringResult %s (trait=%s) has an empty trajectory; skipping",
                result.scenario_id,
                result.trait.value,
            )
            n_failures += 1
            if i >= 9 and n_failures / (i + 1) > max_failure_rate:
                logger.error(
                    "Aborting: failure rate %.0f%% exceeds %.0f%% threshold after %d results",
                    100 * n_failures / (i + 1), 100 * max_failure_rate, i + 1,
                )
                break
            continue

        try:
            trajectory = AgentTrajectory(**result.trajectory)
        except Exception as exc:
            logger.warning(
                "Could not deserialise trajectory for SteeringResult %s: %s",
                result.scenario_id,
                exc,
            )
            n_failures += 1
            if i >= 9 and n_failures / (i + 1) > max_failure_rate:
                logger.error(
                    "Aborting: failure rate %.0f%% exceeds %.0f%% threshold after %d results",
                    100 * n_failures / (i + 1), 100 * max_failure_rate, i + 1,
                )
                break
            continue

        if i > 0:
            import time as _time
            _time.sleep(rate_limit_delay)

        try:
            behavioral_score = judge.score_trajectory(trajectory)
            result.behavioral_score = behavioral_score.model_dump()
        except Exception as exc:
            logger.error(
                "Judge failed for SteeringResult %s (trait=%s): %s",
                result.scenario_id,
                result.trait.value,
                exc,
            )
            n_failures += 1
            if i >= 9 and n_failures / (i + 1) > max_failure_rate:
                logger.error(
                    "Aborting: failure rate %.0f%% exceeds %.0f%% threshold after %d results",
                    100 * n_failures / (i + 1), 100 * max_failure_rate, i + 1,
                )
                break

        if (i + 1) % 10 == 0:
            logger.info("score_steering_results: %d / %d scored", i + 1, total)

    logger.info("score_steering_results: complete (%d results processed, %d failures)", total, n_failures)


class SteeringExperimentRunner:
    """Runs the three steering experiments from the research plan."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        sae_dict: dict[str, TopKSAE],
        all_tas: dict[BehavioralTrait, dict[str, torch.Tensor]],
        multipliers: list[float] | None = None,
        top_k_features: int = 20,
        judge: Any | None = None,
    ) -> None:
        """Initialize the experiment runner.

        Args:
            model: The language model.
            tokenizer: The tokenizer.
            sae_dict: Dict mapping sae_id to trained SAE.
            all_tas: Nested dict: trait → sae_id → TAS tensor.
            multipliers: Steering multipliers to test.
            top_k_features: Number of top TAS features to use.
            judge: Optional LLMJudge instance used by
                ``compare_sae_vs_mean_diff_specificity`` to score trajectories.
                If None, a default LLMJudge is instantiated when needed.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.sae_dict = sae_dict
        self.all_tas = all_tas
        self.multipliers = multipliers or [0.0, 2.0, 5.0, 10.0]
        self.top_k_features = top_k_features
        self._judge = judge

    def _get_best_sae_for_trait(self, trait: BehavioralTrait) -> str:
        """Find the SAE with the highest mean top-k TAS for a trait.

        NOTE: This selects on the same data used for evaluation (the TAS
        scores that were computed from the contrastive pairs that are also
        used to evaluate steering). This introduces optimistic bias — the
        selected SAE is guaranteed to have the highest *in-sample* TAS.
        The cross-depth experiment (Experiment 3) provides a fairer
        comparison because it evaluates all depth bands, not just the
        winner-take-all selection.

        When comparing across SAEs with different architectures (different
        dict_size / k), the mean top-k |TAS| is biased toward smaller
        dictionaries (features fire more often → larger individual effects).
        Downstream code should normalize TAS by each SAE's null distribution
        before cross-SAE comparison.

        Args:
            trait: The behavioral trait.

        Returns:
            sae_id of the best SAE.
        """
        best_sae_id = ""
        best_score = -float("inf")

        candidates: list[tuple[str, float]] = []
        for sae_id, tas in self.all_tas[trait].items():
            top_features = rank_features(tas, self.top_k_features)
            if not top_features:
                continue
            mean_abs_tas = sum(abs(t) for _, t in top_features) / len(top_features)
            candidates.append((sae_id, mean_abs_tas))
            if mean_abs_tas > best_score:
                best_score = mean_abs_tas
                best_sae_id = sae_id

        if len(candidates) > 1:
            candidates.sort(key=lambda x: x[1], reverse=True)
            logger.info(
                "Best SAE for %s: %s (mean top-%d |TAS|=%.3f). "
                "Runner-up: %s (%.3f). Selection uses same data as evaluation.",
                trait.value, candidates[0][0], self.top_k_features, candidates[0][1],
                candidates[1][0], candidates[1][1],
            )

        return best_sae_id

    def run_experiment_1_standard(
        self,
        trait: BehavioralTrait,
        scenarios: list[EvaluationScenario],
        agent_harness: Any,
    ) -> Experiment1Results:
        """Standard steering: best SAE per trait, sweep multipliers.

        For each trait:
        1. Select SAE with highest TAS for this trait
        2. Get top-k features by TAS
        3. For each multiplier in [0, 2, 5, 10]:
           a. Steer model with these features at this multiplier
           b. Run all scenarios through agent harness
           c. Store trajectory

        Args:
            trait: The behavioral trait to steer.
            scenarios: Evaluation scenarios to run.
            agent_harness: The agent harness for running scenarios.

        Returns:
            Experiment1Results.
        """
        best_sae_id = self._get_best_sae_for_trait(trait)
        sae = self.sae_dict[best_sae_id]
        layer = HOOK_POINTS_BY_ID[best_sae_id].layer
        tas = self.all_tas[trait][best_sae_id]
        top_features = rank_features(tas, self.top_k_features)
        feature_indices = [idx for idx, _ in top_features]

        logger.info(
            "Experiment 1: %s using %s (layer %d), %d features",
            trait.value,
            best_sae_id,
            layer,
            len(feature_indices),
        )

        engine = SteeringEngine(self.model, sae, layer)
        results_by_multiplier: dict[float, list[SteeringResult]] = {}

        for mult in self.multipliers:
            mult_results = []

            for scenario in scenarios:
                if mult == 0.0:
                    # Multiplier 0.0 = unsteered baseline (no hooks).
                    # With hooks active, multiplier 0.0 would ABLATE the target
                    # features (zero them out and subtract their contribution
                    # from the residual stream), which is a different condition
                    # than "no intervention."
                    agent_harness.steering_engine = None
                else:
                    engine.set_steering(feature_indices, mult)
                    agent_harness.steering_engine = engine
                trajectory = agent_harness.run_scenario(scenario)

                mult_results.append(SteeringResult(
                    scenario_id=scenario.id,
                    trait=trait,
                    multiplier=mult,
                    sae_id=best_sae_id,
                    feature_indices=feature_indices,
                    trajectory=trajectory.model_dump() if hasattr(trajectory, "model_dump") else {},
                ))

            results_by_multiplier[mult] = mult_results
            logger.info("  Multiplier %.1f: %d scenarios complete", mult, len(mult_results))

        # Clear steering state so subsequent experiments don't inherit stale hooks.
        agent_harness.steering_engine = None

        return Experiment1Results(
            trait=trait,
            best_sae_id=best_sae_id,
            results_by_multiplier=results_by_multiplier,
        )

    def run_experiment_2_layer_type(
        self,
        trait: BehavioralTrait,
        scenarios: list[EvaluationScenario],
        agent_harness: Any,
        multiplier: float = 5.0,
    ) -> Experiment2Results:
        """Layer-type-specific steering: DeltaNet only vs attention only vs both.

        IMPORTANT: To avoid confounding layer count with layer type, we use only
        the MATCHED DeltaNet SAEs — the ones
        at the same depth bands as the 3 attention SAEs (early/mid/late). The
        control SAE at sae_delta_mid_pos1 is excluded from this comparison.
        This gives 3 DeltaNet vs 3 attention SAEs for a fair comparison.
        The combined condition uses all 6 matched SAEs.

        Args:
            trait: The behavioral trait.
            scenarios: Evaluation scenarios.
            agent_harness: Agent harness.
            multiplier: Fixed multiplier to use.

        Returns:
            Experiment2Results.
        """
        agent_harness.steering_engine = None
        # Use only depth-matched SAEs: 3 DeltaNet (early/mid/late) vs 3 attention.
        # Exclude sae_delta_mid_pos1 to keep the layer count balanced.
        matched_deltanet_ids = [
            hp.sae_id for hp in HOOK_POINTS
            if hp.layer_type == LayerType.DELTANET and hp.sae_id != "sae_delta_mid_pos1"
        ]
        attention_sae_ids = [hp.sae_id for hp in HOOK_POINTS if hp.layer_type == LayerType.ATTENTION]
        deltanet_sae_ids = matched_deltanet_ids

        def _run_with_sae_ids(
            sae_ids: list[str], condition_name: str
        ) -> list[SteeringResult]:
            """Run scenarios with steering from specific SAE set."""
            multi_engine = MultiLayerSteeringEngine(self.model)

            for sae_id in sae_ids:
                if sae_id not in self.sae_dict or sae_id not in self.all_tas.get(trait, {}):
                    continue
                tas = self.all_tas[trait][sae_id]
                top_features = rank_features(tas, self.top_k_features)
                feature_indices = [idx for idx, _ in top_features]
                layer = HOOK_POINTS_BY_ID[sae_id].layer
                multi_engine.add_layer(
                    self.sae_dict[sae_id], layer, feature_indices, multiplier
                )

            results = []
            with multi_engine.active():
                for scenario in scenarios:
                    trajectory = agent_harness.run_scenario(scenario)
                    results.append(SteeringResult(
                        scenario_id=scenario.id,
                        trait=trait,
                        multiplier=multiplier,
                        sae_id=condition_name,
                        feature_indices=[],
                        trajectory=trajectory.model_dump() if hasattr(trajectory, "model_dump") else {},
                    ))

            logger.info("  %s: %d scenarios complete", condition_name, len(results))
            return results

        deltanet_results = _run_with_sae_ids(deltanet_sae_ids, "deltanet_only")
        attention_results = _run_with_sae_ids(attention_sae_ids, "attention_only")
        combined_results = _run_with_sae_ids(
            deltanet_sae_ids + attention_sae_ids, "combined"
        )

        return Experiment2Results(
            trait=trait,
            deltanet_results=deltanet_results,
            attention_results=attention_results,
            combined_results=combined_results,
        )

    def run_experiment_3_cross_depth(
        self,
        trait: BehavioralTrait,
        scenarios: list[EvaluationScenario],
        agent_harness: Any,
        multiplier: float = 5.0,
    ) -> Experiment3Results:
        """Cross-depth steering: early vs mid vs late.

        Args:
            trait: The behavioral trait.
            scenarios: Evaluation scenarios.
            agent_harness: Agent harness.
            multiplier: Fixed multiplier.

        Returns:
            Experiment3Results.
        """
        agent_harness.steering_engine = None
        # Use 2 SAEs per depth band (1 DeltaNet + 1 Attention) for a fair
        # comparison.  Exclude sae_delta_mid_pos1 (the position-in-block
        # control) so all three conditions steer at exactly 2 layers.
        depth_groups = {
            "early": ["sae_attn_early", "sae_delta_early"],
            "mid": ["sae_attn_mid", "sae_delta_mid"],
            "late": ["sae_attn_late", "sae_delta_late"],
        }

        all_depth_results: dict[str, list[SteeringResult]] = {}

        for depth_name, sae_ids in depth_groups.items():
            multi_engine = MultiLayerSteeringEngine(self.model)

            for sae_id in sae_ids:
                if sae_id not in self.sae_dict or sae_id not in self.all_tas.get(trait, {}):
                    continue
                tas = self.all_tas[trait][sae_id]
                top_features = rank_features(tas, self.top_k_features)
                feature_indices = [idx for idx, _ in top_features]
                layer = HOOK_POINTS_BY_ID[sae_id].layer
                multi_engine.add_layer(
                    self.sae_dict[sae_id], layer, feature_indices, multiplier
                )

            results = []
            with multi_engine.active():
                for scenario in scenarios:
                    trajectory = agent_harness.run_scenario(scenario)
                    results.append(SteeringResult(
                        scenario_id=scenario.id,
                        trait=trait,
                        multiplier=multiplier,
                        sae_id=depth_name,
                        feature_indices=[],
                        trajectory=trajectory.model_dump() if hasattr(trajectory, "model_dump") else {},
                    ))

            all_depth_results[depth_name] = results
            logger.info("  %s: %d scenarios complete", depth_name, len(results))

        return Experiment3Results(
            trait=trait,
            early_results=all_depth_results["early"],
            mid_results=all_depth_results["mid"],
            late_results=all_depth_results["late"],
        )

    def run_random_baseline(
        self,
        trait: BehavioralTrait,
        scenarios: list[EvaluationScenario],
        agent_harness: Any,
        multiplier: float = 5.0,
        seed: int = 42,
        n_seeds: int = 1,
    ) -> dict[int, list[SteeringResult]]:
        """Random-feature steering baseline: steer the same number of randomly
        selected features at the same multiplier.

        If random steering also shifts behavioral scores, TAS-identified features
        are not special. This control is essential for ruling out the alternative
        hypothesis that any sufficiently active feature perturbation changes behavior.

        When n_seeds > 1, repeats with multiple random seeds so the distribution
        of baseline effects is reported, not a single point estimate.

        Args:
            trait: The behavioral trait (for comparison with TAS steering).
            scenarios: Evaluation scenarios to run.
            agent_harness: Agent harness.
            multiplier: Steering multiplier (same as used in real experiments).
            seed: Base random seed for feature selection.
            n_seeds: Number of independent random seeds to run (default=1).
                Use 10 to report the distribution of baseline effects.

        Returns:
            Dict mapping seed → list of SteeringResult from that seed's run.
            For n_seeds=1, the dict has one entry.
        """
        best_sae_id = self._get_best_sae_for_trait(trait)
        sae = self.sae_dict[best_sae_id]
        layer = HOOK_POINTS_BY_ID[best_sae_id].layer

        # Filter to active features only: dead features trivialize the
        # comparison because they never fire and thus can't change behavior.
        tas = self.all_tas[trait][best_sae_id]
        active_mask = tas.abs() > 0
        active_indices = active_mask.nonzero(as_tuple=True)[0].tolist()

        all_seed_results: dict[int, list[SteeringResult]] = {}

        for seed_offset in range(n_seeds):
            current_seed = seed + seed_offset

            if len(active_indices) < self.top_k_features:
                logger.warning(
                    "Only %d active features available (need %d), using all",
                    len(active_indices), self.top_k_features,
                )
                random_indices = active_indices
            else:
                rng = random.Random(current_seed)
                random_indices = rng.sample(active_indices, self.top_k_features)

            logger.info(
                "Random baseline for %s (seed=%d): %d random features at layer %d, mult=%.1f",
                trait.value, current_seed, len(random_indices), layer, multiplier,
            )

            engine = SteeringEngine(self.model, sae, layer)
            engine.set_steering(random_indices, multiplier)

            results = []
            for scenario in scenarios:
                agent_harness.steering_engine = engine
                trajectory = agent_harness.run_scenario(scenario)
                results.append(SteeringResult(
                    scenario_id=scenario.id,
                    trait=trait,
                    multiplier=multiplier,
                    sae_id=f"{best_sae_id}_random_s{current_seed}",
                    feature_indices=random_indices,
                    trajectory=trajectory.model_dump() if hasattr(trajectory, "model_dump") else {},
                ))

            all_seed_results[current_seed] = results
            logger.info(
                "  Random baseline seed %d: %d scenarios complete",
                current_seed, len(results),
            )

        return all_seed_results

    def measure_cross_layer_interaction(
        self,
        trait: BehavioralTrait,
        source_sae_id: str,
        target_sae_id: str,
        input_ids: torch.Tensor,
        multiplier: float = 5.0,
    ) -> dict[str, float]:
        """Measure whether steering at one layer changes feature activations at another.

        Steers at the source layer and measures the change in feature activations
        at the target layer. Useful for understanding whether DeltaNet layers
        influence downstream attention layers (or vice versa).

        This measurement is especially important for DeltaNet source layers:
        the gated linear recurrence propagates perturbations differently than
        attention, so the empirical cross-layer effect may be non-linearly
        amplified or dampened.  Large ``mean_activation_change`` from DeltaNet
        steering suggests the recurrence amplifies the intervention; low values
        suggest the gate dampens it.

        NOTE: Multi-layer steering creates interaction effects — steering at
        layer 34 changes the input to layer 35's SAE. This method isolates
        single-layer steering effects.

        Args:
            trait: The behavioral trait.
            source_sae_id: SAE to steer at.
            target_sae_id: SAE to measure activation changes at.
            input_ids: Input token IDs of shape (1, seq_len).
            multiplier: Steering multiplier.

        Returns:
            Dict with:
                "mean_activation_change": float,
                "max_activation_change": float,
                "top_k_overlap": float,  # overlap of top-k features before/after
                "source_layer": int,
                "target_layer": int,
        """
        source_sae = self.sae_dict[source_sae_id]
        target_sae = self.sae_dict[target_sae_id]
        source_layer = HOOK_POINTS_BY_ID[source_sae_id].layer
        target_layer = HOOK_POINTS_BY_ID[target_sae_id].layer

        tas = self.all_tas[trait][source_sae_id]
        top_features = rank_features(tas, self.top_k_features)
        feature_indices = [idx for idx, _ in top_features]

        with torch.no_grad():
            # Baseline: capture target layer activations without steering
            cache_baseline = ActivationCache(self.model, layers=[target_layer])
            with cache_baseline.active():
                self.model(input_ids)
            baseline_acts = cache_baseline.get(target_layer)  # (1, seq_len, hidden_dim)
            baseline_features = target_sae.encode(baseline_acts)  # (1, seq_len, dict_size)

            # Steered: capture target layer activations with source layer steering.
            # steer_all_positions=True is required here because this is a
            # measurement forward pass (prefill, seq_len > 1), not autoregressive
            # generation.  The default decode-only gate (seq_len == 1) would cause
            # the hook to be a no-op on prefill, producing zero-effect measurements.
            engine = SteeringEngine(self.model, source_sae, source_layer)
            engine.steer_all_positions = True
            engine.set_steering(feature_indices, multiplier)
            cache_steered = ActivationCache(self.model, layers=[target_layer])
            with engine.active(), cache_steered.active():
                self.model(input_ids)
            steered_acts = cache_steered.get(target_layer)
            steered_features = target_sae.encode(steered_acts)

        # Compare feature activations
        diff = (steered_features - baseline_features).abs()
        mean_change = float(diff.mean().item())
        max_change = float(diff.max().item())

        # Top-k overlap: do the same features remain most active?
        k = min(20, baseline_features.shape[-1])
        baseline_topk = baseline_features.abs().mean(dim=(0, 1)).topk(k).indices
        steered_topk = steered_features.abs().mean(dim=(0, 1)).topk(k).indices
        baseline_set = set(baseline_topk.tolist())
        steered_set = set(steered_topk.tolist())
        overlap = len(baseline_set & steered_set) / max(len(baseline_set | steered_set), 1)

        logger.info(
            "Cross-layer interaction %s→%s: mean_change=%.4f, max_change=%.4f, top_k_overlap=%.3f",
            source_sae_id, target_sae_id, mean_change, max_change, overlap,
        )

        return {
            "mean_activation_change": mean_change,
            "max_activation_change": max_change,
            "top_k_overlap": overlap,
            "source_layer": source_layer,
            "target_layer": target_layer,
        }

    def run_experiment_2_single_layer(
        self,
        trait: BehavioralTrait,
        scenarios: list[EvaluationScenario],
        agent_harness: Any,
        multiplier: float = 5.0,
    ) -> dict[str, list[SteeringResult]]:
        """Run Experiment 2 with each SAE steered independently (single-layer).

        The original Experiment 2 compares "all DeltaNet" vs "all attention"
        vs "combined", but this confounds layer count with layer type. This
        variant steers each of the 7 SAEs individually, producing 7 single-
        layer conditions that enable a clean comparison.

        Args:
            trait: The behavioral trait.
            scenarios: Evaluation scenarios.
            agent_harness: Agent harness.
            multiplier: Fixed multiplier.

        Returns:
            Dict mapping sae_id to list of SteeringResult.
        """
        results: dict[str, list[SteeringResult]] = {}

        for sae_id, sae in self.sae_dict.items():
            if sae_id not in self.all_tas.get(trait, {}):
                continue

            layer = HOOK_POINTS_BY_ID[sae_id].layer
            tas = self.all_tas[trait][sae_id]
            top_features = rank_features(tas, self.top_k_features)
            feature_indices = [idx for idx, _ in top_features]

            engine = SteeringEngine(self.model, sae, layer)
            engine.set_steering(feature_indices, multiplier)

            sae_results: list[SteeringResult] = []
            for scenario in scenarios:
                agent_harness.steering_engine = engine
                trajectory = agent_harness.run_scenario(scenario)
                sae_results.append(SteeringResult(
                    scenario_id=scenario.id,
                    trait=trait,
                    multiplier=multiplier,
                    sae_id=sae_id,
                    feature_indices=feature_indices,
                    trajectory=trajectory.model_dump() if hasattr(trajectory, "model_dump") else {},
                ))

            results[sae_id] = sae_results
            logger.info(
                "Exp2 single-layer %s (layer %d): %d scenarios complete",
                sae_id, layer, len(sae_results),
            )

        return results

    def run_generalization_test(
        self,
        trait: BehavioralTrait,
        scenarios: list[EvaluationScenario],
        agent_harness: Any,
        multiplier: float = 5.0,
        neutral_system_prompt: str = "You are a helpful assistant.",
    ) -> list[SteeringResult]:
        """Test whether steering generalizes beyond the contrastive setting.

        Runs steered inference with a neutral system prompt. If steering
        doesn't change behavior without trait-manipulating system prompts,
        the identified features encode instruction-sensitivity (response to
        specific system prompt phrasing) rather than genuine behavioral
        dispositions.

        Args:
            trait: The behavioral trait to steer.
            scenarios: Evaluation scenarios.
            agent_harness: Agent harness (must support ``override_system_prompt``).
            multiplier: Steering multiplier.
            neutral_system_prompt: A behaviorally neutral system prompt.

        Returns:
            List of SteeringResult from steered-with-neutral-prompt runs.
        """
        best_sae_id = self._get_best_sae_for_trait(trait)
        sae = self.sae_dict[best_sae_id]
        layer = HOOK_POINTS_BY_ID[best_sae_id].layer
        tas = self.all_tas[trait][best_sae_id]
        top_features = rank_features(tas, self.top_k_features)
        feature_indices = [idx for idx, _ in top_features]

        agent_harness.steering_engine = None
        engine = SteeringEngine(self.model, sae, layer)
        engine.set_steering(feature_indices, multiplier)

        logger.info(
            "Generalization test for %s: %d features, mult=%.1f, "
            "neutral prompt='%s'",
            trait.value, len(feature_indices), multiplier,
            neutral_system_prompt[:50],
        )

        results: list[SteeringResult] = []
        for scenario in scenarios:
            agent_harness.steering_engine = engine
            # Override the system prompt to a neutral one
            neutral_scenario = scenario.model_copy(
                update={"system_prompt": neutral_system_prompt}
            )
            trajectory = agent_harness.run_scenario(neutral_scenario)
            results.append(SteeringResult(
                scenario_id=scenario.id,
                trait=trait,
                multiplier=multiplier,
                sae_id=f"{best_sae_id}_generalization",
                feature_indices=feature_indices,
                trajectory=trajectory.model_dump() if hasattr(trajectory, "model_dump") else {},
            ))

        logger.info("  Generalization test: %d scenarios complete", len(results))
        return results

    def run_mean_diff_baseline(
        self,
        trait: BehavioralTrait,
        scenarios: list[EvaluationScenario],
        agent_harness: Any,
        high_activations: dict[str, torch.Tensor],
        low_activations: dict[str, torch.Tensor],
        multiplier: float = 5.0,
    ) -> list[SteeringResult]:
        """Mean-difference steering vector baseline.

        Computes the mean activation difference between HIGH and LOW
        contrastive versions (in the original activation space, not SAE
        features) and uses it as a steering vector. This is the simplest
        possible approach (activation addition). If SAE-based steering
        doesn't outperform this baseline, the SAE decomposition isn't
        adding value.

        Args:
            trait: The behavioral trait.
            scenarios: Evaluation scenarios.
            agent_harness: Agent harness.
            high_activations: Dict mapping sae_id to mean activation tensor
                of shape (hidden_dim,) from HIGH contrastive versions.
            low_activations: Dict mapping sae_id to mean activation tensor
                of shape (hidden_dim,) from LOW contrastive versions.
            multiplier: Steering multiplier.

        Returns:
            List of SteeringResult from mean-diff steering.
        """
        from src.steering.engine import MeanDiffSteeringEngine

        best_sae_id = self._get_best_sae_for_trait(trait)
        layer = HOOK_POINTS_BY_ID[best_sae_id].layer

        mean_high = high_activations[best_sae_id]  # (hidden_dim,)
        mean_low = low_activations[best_sae_id]    # (hidden_dim,)
        steering_vector = mean_high - mean_low      # (hidden_dim,)

        engine = MeanDiffSteeringEngine(self.model, layer, steering_vector)
        engine.set_multiplier(multiplier)

        logger.info(
            "Mean-diff baseline for %s: layer %d, vector norm=%.3f, mult=%.1f",
            trait.value, layer, steering_vector.norm().item(), multiplier,
        )

        results: list[SteeringResult] = []
        for scenario in scenarios:
            with engine.active():
                trajectory = agent_harness.run_scenario(scenario)
            results.append(SteeringResult(
                scenario_id=scenario.id,
                trait=trait,
                multiplier=multiplier,
                sae_id=f"{best_sae_id}_mean_diff",
                feature_indices=[],
                trajectory=trajectory.model_dump() if hasattr(trajectory, "model_dump") else {},
            ))

        logger.info("  Mean-diff baseline: %d scenarios complete", len(results))
        return results

    def compare_sae_vs_mean_diff_specificity(
        self,
        sae_results: dict[BehavioralTrait, list[SteeringResult]],
        mean_diff_results: dict[BehavioralTrait, list[SteeringResult]],
        baseline_scores: list[BehavioralScore],
    ) -> dict[str, Any]:
        """Compare SAE-based vs mean-diff steering on cross-trait contamination.

        The critical comparison between SAE-based and mean-diff steering is not
        on-target magnitude but
        *specificity* — whether SAE-based steering produces lower off-diagonal
        contamination. A method that strongly shifts the target trait but also
        shifts every other trait is less useful than one with modest on-target
        effect but clean off-diagonal.

        For each method, builds the 5x5 contamination matrix, then compares
        per-trait selectivity (on-diagonal / max off-diagonal ratio). Higher
        selectivity means the intervention is more trait-specific.

        Args:
            sae_results: Dict mapping each steered trait to its list of
                SteeringResult from SAE-based steering.
            mean_diff_results: Dict mapping each steered trait to its list of
                SteeringResult from mean-diff steering.
            baseline_scores: BehavioralScore list from unsteered model runs,
                used as the reference for computing effect deltas.

        Returns:
            Dict with:
                'sae_matrix': (5, 5) contamination matrix for SAE steering
                'mean_diff_matrix': (5, 5) contamination matrix for mean-diff
                'sae_summary': aggregate contamination summary for SAE
                'mean_diff_summary': aggregate contamination summary for mean-diff
                'per_trait': dict mapping trait name to per-trait comparison
                    (sae_on_target, mean_diff_on_target, sae_max_off_target,
                     mean_diff_max_off_target, sae_selectivity, mean_diff_selectivity,
                     sae_more_specific)
                'overall_sae_more_specific': bool — True if SAE has higher
                    mean selectivity ratio across traits
        """
        from src.evaluation.contamination import (
            TRAIT_ORDER,
            TRAIT_SCORE_KEYS,
            compute_contamination_matrix,
            contamination_summary,
        )
        from src.analysis.effect_sizes import compute_selectivity_per_trait

        # Score all trajectories through the LLM judge before extracting
        # BehavioralScore objects. Without this call, result.behavioral_score
        # is always None and _extract_behavioral_scores returns empty lists,
        # causing compute_contamination_matrix to produce all-zero matrices.
        # judge must be passed in from the caller; if it is not provided, a
        # default LLMJudge instance is created here so the function is still
        # callable without pre-scoring (e.g., in tests with mock judges).
        from src.evaluation.llm_judge import LLMJudge
        _judge = getattr(self, "_judge", None) or LLMJudge()
        score_steering_results(sae_results, _judge)
        score_steering_results(mean_diff_results, _judge)

        sae_scores = self._extract_behavioral_scores(sae_results)
        mean_diff_scores = self._extract_behavioral_scores(mean_diff_results)

        sae_matrix = compute_contamination_matrix(baseline_scores, sae_scores)
        mean_diff_matrix = compute_contamination_matrix(baseline_scores, mean_diff_scores)

        sae_summary = contamination_summary(sae_matrix)
        mean_diff_summary = contamination_summary(mean_diff_matrix)

        sae_selectivity = compute_selectivity_per_trait(
            sae_matrix, trait_names=TRAIT_SCORE_KEYS
        )
        mean_diff_selectivity = compute_selectivity_per_trait(
            mean_diff_matrix, trait_names=TRAIT_SCORE_KEYS
        )

        per_trait: dict[str, dict[str, Any]] = {}
        sae_ratios: list[float] = []
        mean_diff_ratios: list[float] = []

        for trait_key in TRAIT_SCORE_KEYS:
            sae_sel = sae_selectivity[trait_key]
            md_sel = mean_diff_selectivity[trait_key]

            sae_ratio = sae_sel["selectivity_ratio"]
            md_ratio = md_sel["selectivity_ratio"]
            sae_ratios.append(sae_ratio)
            mean_diff_ratios.append(md_ratio)

            per_trait[trait_key] = {
                "sae_on_target": sae_sel["on_diagonal"],
                "mean_diff_on_target": md_sel["on_diagonal"],
                "sae_max_off_target": sae_sel["max_off_diagonal"],
                "mean_diff_max_off_target": md_sel["max_off_diagonal"],
                "sae_selectivity": sae_ratio,
                "mean_diff_selectivity": md_ratio,
                "sae_more_specific": sae_ratio > md_ratio,
            }

            logger.info(
                "Specificity %s: SAE sel=%.2f (on=%.3f, off=%.3f) vs "
                "mean-diff sel=%.2f (on=%.3f, off=%.3f) → %s",
                trait_key,
                sae_ratio,
                sae_sel["on_diagonal"],
                sae_sel["max_off_diagonal"],
                md_ratio,
                md_sel["on_diagonal"],
                md_sel["max_off_diagonal"],
                "SAE wins" if sae_ratio > md_ratio else "mean-diff wins",
            )

        mean_sae_selectivity = sum(sae_ratios) / max(len(sae_ratios), 1)
        mean_md_selectivity = sum(mean_diff_ratios) / max(len(mean_diff_ratios), 1)
        overall_sae_wins = mean_sae_selectivity > mean_md_selectivity

        logger.info(
            "Overall specificity: SAE mean sel=%.2f vs mean-diff mean sel=%.2f → %s",
            mean_sae_selectivity,
            mean_md_selectivity,
            "SAE more specific" if overall_sae_wins else "mean-diff more specific",
        )

        return {
            "sae_matrix": sae_matrix,
            "mean_diff_matrix": mean_diff_matrix,
            "sae_summary": sae_summary,
            "mean_diff_summary": mean_diff_summary,
            "per_trait": per_trait,
            "overall_sae_more_specific": overall_sae_wins,
        }

    @staticmethod
    def _extract_behavioral_scores(
        results: dict[BehavioralTrait, list[SteeringResult]],
    ) -> dict[BehavioralTrait, list[BehavioralScore]]:
        """Extract BehavioralScore objects from SteeringResult trajectories.

        Each SteeringResult stores its trajectory as a serialized dict.
        The 'behavioral_score' key (when present) contains the LLM judge
        output that can be deserialized into a BehavioralScore.

        Args:
            results: Dict mapping trait to list of SteeringResult.

        Returns:
            Dict mapping trait to list of BehavioralScore extracted from
            the trajectory data.
        """
        scores: dict[BehavioralTrait, list[BehavioralScore]] = {}

        for trait, result_list in results.items():
            trait_scores: list[BehavioralScore] = []
            for result in result_list:
                score_data = result.behavioral_score
                if score_data is not None:
                    trait_scores.append(BehavioralScore(**score_data))
                else:
                    logger.warning(
                        "SteeringResult %s (trait=%s) has no behavioral_score; skipping",
                        result.scenario_id,
                        trait.value,
                    )
            scores[trait] = trait_scores

        return scores

    def _run_and_score_scenarios(
        self,
        scenarios: list[EvaluationScenario],
        agent_harness: Any,
        judge: Any,
        target_trait: str,
        rate_limit_delay: float = 0.5,
    ) -> list[float]:
        """Run all scenarios with current harness config and return target trait scores.

        Helper to avoid duplicating the run→score→extract loop across activation
        patching conditions (baseline, individual ablation, group ablation).

        Args:
            scenarios: Evaluation scenarios to run.
            agent_harness: Agent harness (steering_engine already configured).
            judge: LLMJudge instance.
            target_trait: Trait name key for score extraction (e.g. "autonomy").
            rate_limit_delay: Seconds between judge API calls.

        Returns:
            List of target trait scores, one per scenario. NaN for failed scoring.
        """
        trait_scores: list[float] = []

        for i, scenario in enumerate(scenarios):
            trajectory = agent_harness.run_scenario(scenario)

            if i > 0:
                import time as _time
                _time.sleep(rate_limit_delay)

            try:
                behavioral_score = judge.score_trajectory(trajectory)
                score = behavioral_score.get_trait_score(target_trait)
                trait_scores.append(score)
            except Exception as exc:
                logger.warning(
                    "Judge failed for scenario %s: %s", scenario.id, exc,
                )
                trait_scores.append(float("nan"))

        return trait_scores

    def run_activation_patching(
        self,
        trait: BehavioralTrait,
        scenarios: list[EvaluationScenario],
        agent_harness: Any,
        judge: Any,
        alpha: float = 0.05,
    ) -> ActivationPatchingResults:
        """Activation patching: clamp high-TAS features to zero and measure causal effect.

        The gold standard for causal claims in mechanistic interpretability.
        For each high-TAS feature individually, clamps it to zero during
        autoregressive decode (via multiplier=0.0) and measures the change
        in the target behavioral trait score relative to an unsteered baseline.

        Features where ablation significantly changes the trait score (surviving
        FDR correction) are classified as **causally validated**. Features where
        ablation has no significant effect are merely **correlated** — they fire
        in association with the trait but don't causally drive the behavior.

        Also performs group ablation (all top-k features ablated simultaneously)
        to measure the aggregate causal effect.

        Args:
            trait: The behavioral trait to validate features for.
            scenarios: Evaluation scenarios to run.
            agent_harness: Agent harness for running scenarios.
            judge: LLMJudge instance for scoring trajectories.
            alpha: FDR significance level for BH correction.

        Returns:
            ActivationPatchingResults with per-feature and group results.
        """
        # --- Setup: identify features to test ---
        best_sae_id = self._get_best_sae_for_trait(trait)
        sae = self.sae_dict[best_sae_id]
        layer = HOOK_POINTS_BY_ID[best_sae_id].layer
        tas = self.all_tas[trait][best_sae_id]
        top_features = rank_features(tas, self.top_k_features)
        feature_indices = [idx for idx, _ in top_features]
        feature_tas = {idx: score for idx, score in top_features}

        # Trait score key for BehavioralScore.get_trait_score()
        trait_key = trait.value

        logger.info(
            "Activation patching for %s: %d features from %s (layer %d)",
            trait.value, len(feature_indices), best_sae_id, layer,
        )

        # --- Step 1: Unsteered baseline ---
        logger.info("  Running unsteered baseline (%d scenarios)...", len(scenarios))
        agent_harness.steering_engine = None
        baseline_scores = self._run_and_score_scenarios(
            scenarios, agent_harness, judge, trait_key,
        )
        baseline_mean = np.nanmean(baseline_scores)
        logger.info("  Baseline mean %s score: %.3f", trait.value, baseline_mean)

        # --- Step 2: Individual feature ablation ---
        engine = SteeringEngine(self.model, sae, layer)
        raw_results: list[dict[str, Any]] = []

        for feat_idx in feature_indices:
            logger.info(
                "  Ablating feature %d (TAS=%.3f)...",
                feat_idx, feature_tas[feat_idx],
            )
            engine.set_steering([feat_idx], 0.0)
            agent_harness.steering_engine = engine

            ablated_scores = self._run_and_score_scenarios(
                scenarios, agent_harness, judge, trait_key,
            )

            # Paired deltas (ablated - baseline) for each scenario
            deltas = []
            for b, a in zip(baseline_scores, ablated_scores):
                if math.isnan(b) or math.isnan(a):
                    continue
                deltas.append(a - b)

            if len(deltas) >= 2:
                mean_delta = float(np.mean(deltas))
                std_delta = float(np.std(deltas, ddof=1))
                # Paired t-test: are the deltas significantly different from 0?
                t_stat, p_value = stats.ttest_rel(
                    [b for b, a in zip(baseline_scores, ablated_scores)
                     if not (math.isnan(b) or math.isnan(a))],
                    [a for b, a in zip(baseline_scores, ablated_scores)
                     if not (math.isnan(b) or math.isnan(a))],
                )
                p_value = float(p_value)
            else:
                mean_delta = float("nan")
                std_delta = float("nan")
                p_value = 1.0

            raw_results.append({
                "feature_idx": feat_idx,
                "tas_score": feature_tas[feat_idx],
                "mean_delta": mean_delta,
                "std_delta": std_delta,
                "p_value": p_value,
                "scenario_deltas": deltas,
            })

            logger.info(
                "    Feature %d: Δ=%.4f (std=%.4f), p=%.4f",
                feat_idx, mean_delta, std_delta, p_value,
            )

        # --- Step 3: Group ablation (all features simultaneously) ---
        logger.info("  Running group ablation (%d features)...", len(feature_indices))
        engine.set_steering(feature_indices, 0.0)
        agent_harness.steering_engine = engine

        group_scores = self._run_and_score_scenarios(
            scenarios, agent_harness, judge, trait_key,
        )

        group_deltas = []
        for b, g in zip(baseline_scores, group_scores):
            if not (math.isnan(b) or math.isnan(g)):
                group_deltas.append(g - b)

        if len(group_deltas) >= 2:
            group_mean_delta = float(np.mean(group_deltas))
            _, group_p = stats.ttest_rel(
                [b for b, g in zip(baseline_scores, group_scores)
                 if not (math.isnan(b) or math.isnan(g))],
                [g for b, g in zip(baseline_scores, group_scores)
                 if not (math.isnan(b) or math.isnan(g))],
            )
            group_p = float(group_p)
        else:
            group_mean_delta = float("nan")
            group_p = 1.0

        logger.info(
            "  Group ablation: Δ=%.4f, p=%.4f", group_mean_delta, group_p,
        )

        # --- Step 4: BH FDR correction on individual p-values ---
        n_tests = len(raw_results)
        p_values = np.array([r["p_value"] for r in raw_results])

        # Rank p-values ascending
        order = np.argsort(p_values)
        corrected_p = np.ones(n_tests)

        if n_tests > 0:
            ranks = np.empty(n_tests, dtype=int)
            ranks[order] = np.arange(1, n_tests + 1)
            corrected_p = np.minimum(p_values * n_tests / ranks, 1.0)

            # Enforce monotonicity (step-up)
            for i in range(n_tests - 2, -1, -1):
                corrected_p[order[i]] = min(
                    corrected_p[order[i]], corrected_p[order[i + 1]]
                )

        # --- Step 5: Build final results ---
        feature_results: list[FeatureAblationResult] = []
        for i, r in enumerate(raw_results):
            is_causal = bool(corrected_p[i] < alpha)
            feature_results.append(FeatureAblationResult(
                feature_idx=r["feature_idx"],
                tas_score=r["tas_score"],
                mean_delta=r["mean_delta"],
                std_delta=r["std_delta"],
                p_value=r["p_value"],
                corrected_p=float(corrected_p[i]),
                is_causal=is_causal,
                scenario_deltas=r["scenario_deltas"],
            ))

        n_causal = sum(1 for fr in feature_results if fr.is_causal)
        causal_fraction = n_causal / max(n_tests, 1)

        logger.info(
            "Activation patching for %s: %d/%d features causal (%.0f%%)",
            trait.value, n_causal, n_tests, causal_fraction * 100,
        )

        # Reset harness
        agent_harness.steering_engine = None

        return ActivationPatchingResults(
            trait=trait,
            sae_id=best_sae_id,
            feature_results=feature_results,
            group_ablation_mean_delta=group_mean_delta,
            group_ablation_p_value=group_p,
            n_causal=n_causal,
            n_tested=n_tests,
            causal_fraction=causal_fraction,
        )

    def run_baseline_trait_correlations(
        self,
        scenarios: list[EvaluationScenario],
        agent_harness: Any,
        n_runs: int = 200,
    ) -> list[SteeringResult]:
        """Run unsteered model to establish baseline trait correlations.

        Generates 200+ unsteered trajectories to measure how traits naturally
        co-vary. If autonomy and deference are anti-correlated at baseline,
        the contamination matrix must be interpreted relative to this baseline.

        Args:
            scenarios: Evaluation scenarios.
            agent_harness: Agent harness.
            n_runs: Total number of runs. Scenarios are cycled if needed.

        Returns:
            List of SteeringResult from unsteered runs.
        """
        agent_harness.steering_engine = None

        results: list[SteeringResult] = []
        for i in range(n_runs):
            scenario = scenarios[i % len(scenarios)]
            trajectory = agent_harness.run_scenario(scenario)
            results.append(SteeringResult(
                scenario_id=scenario.id,
                trait=BehavioralTrait.AUTONOMY,  # arbitrary; no steering
                multiplier=0.0,
                sae_id="baseline",
                feature_indices=[],
                trajectory=trajectory.model_dump() if hasattr(trajectory, "model_dump") else {},
            ))

            if (i + 1) % 50 == 0:
                logger.info(
                    "Baseline trait correlations: %d / %d runs complete",
                    i + 1, n_runs,
                )

        logger.info("Baseline correlations: %d total runs complete", len(results))
        return results

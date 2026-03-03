"""Dose-response curve computation for steering experiments."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from src.data.scenarios import EvaluationScenario
from src.steering.engine import SteeringEngine

logger = logging.getLogger(__name__)


class DoseResponsePoint(BaseModel):
    """One point on the dose-response curve."""

    multiplier: float
    behavioral_score: float  # Composite trait score for the target trait
    coherence_score: float   # How coherent the model output remains
    sub_behavior_scores: dict[str, float]  # Per-sub-behavior scores for the target trait


class DoseResponseCurve(BaseModel):
    """A full dose-response curve for one feature set + scenario."""

    scenario_id: str
    feature_indices: list[int]
    points: list[DoseResponsePoint]

    @property
    def optimal_multiplier(self) -> float:
        """Find the multiplier that maximizes behavioral effect weighted by coherence.

        Uses ``argmax(behavioral_score * coherence_score)`` instead of a binary
        threshold, which naturally balances behavioral effect against output
        quality.  A coherence of 0.9 with behavioral score 0.7 beats coherence
        0.5 with behavioral score 0.8 because the combined quality is higher.
        """
        if not self.points:
            return 0.0
        return max(
            self.points,
            key=lambda p: p.behavioral_score * p.coherence_score,
        ).multiplier


def compute_dose_response(
    engine: SteeringEngine,
    feature_indices: list[int],
    scenario: EvaluationScenario,
    multipliers: list[float],
    agent_harness: Any,
    judge: Any,
    target_trait: str = "autonomy",
) -> DoseResponseCurve:
    """Compute behavioral score as a function of steering multiplier.

    Returns a curve of (multiplier, behavioral_score) pairs for plotting.
    Useful for finding the "sweet spot" where steering changes behavior
    without causing incoherence.

    IMPORTANT: Measures only the TARGET trait's score, not the average of
    all traits. Averaging all traits causes anti-correlated effects to cancel
    (e.g., autonomy up + deference down), making dose-response curves appear
    flatter than they actually are.

    Args:
        engine: The steering engine.
        feature_indices: Features to steer.
        scenario: The evaluation scenario to run.
        multipliers: List of multipliers to test (e.g., [0, 1, 2, 5, 10, 20]).
        agent_harness: Agent harness for running scenarios.
        judge: LLM judge for scoring behavior.
        target_trait: The trait being steered (e.g., "autonomy", "persistence").
            The dose-response curve measures this trait's score specifically.

    Returns:
        DoseResponseCurve with one point per multiplier.
    """
    points = []

    for mult in multipliers:
        # Run scenario
        if mult == 0.0:
            # Unsteered baseline — clear any residual steering state so no
            # hooks are active.  With hooks active, multiplier 0.0 would
            # ablate the target features rather than leave them unchanged.
            agent_harness.steering_engine = None
            trajectory = agent_harness.run_scenario(scenario)
        else:
            engine.set_steering(feature_indices, mult)
            agent_harness.steering_engine = engine
            trajectory = agent_harness.run_scenario(scenario)

        # Score behavior
        score = judge.score_trajectory(trajectory)

        # Use the TARGET trait's score, not the average of all traits
        behavioral_score = score.get_trait_score(target_trait)

        # Extract sub-behavior scores for the target trait
        all_subs = score.sub_behavior_scores()
        target_sub_scores = all_subs.get(target_trait, {})

        # Coherence check with repetition detection
        coherence = _estimate_coherence(trajectory)

        points.append(
            DoseResponsePoint(
                multiplier=mult,
                behavioral_score=behavioral_score,
                coherence_score=coherence,
                sub_behavior_scores=target_sub_scores,
            )
        )

        sub_str = ", ".join(
            f"{k}={v:.2f}" for k, v in target_sub_scores.items()
        )
        logger.info(
            "Dose-response: mult=%.1f, %s=%.3f (%s), coherence=%.3f",
            mult,
            target_trait,
            behavioral_score,
            sub_str,
            coherence,
        )

    return DoseResponseCurve(
        scenario_id=scenario.id,
        feature_indices=feature_indices,
        points=points,
    )


def _estimate_coherence(trajectory: Any) -> float:
    """Estimate the coherence of a model trajectory as a continuous score.

    Returns a continuous value in [0.0, 1.0] based on:
    - Empty/missing output → 0.0
    - Very short output → length-proportional penalty
    - Repetition severity → continuous penalty proportional to repetition fraction
    - Excessive length → moderate penalty

    Previous versions returned discrete values {0.0, 0.3, 0.4, 0.5, 0.9} which
    lost information. The continuous version enables weighted multiplier selection:
    ``optimal = argmax(behavioral_score * coherence)``.

    Args:
        trajectory: The agent trajectory.

    Returns:
        Coherence score between 0.0 and 1.0.
    """
    if not hasattr(trajectory, "messages") or not trajectory.messages:
        return 0.0

    # Check if there's at least one assistant message with content
    assistant_msgs = [
        m for m in trajectory.messages
        if m.get("role") == "assistant" and m.get("content")
    ]
    if not assistant_msgs:
        return 0.1

    full_text = " ".join(m.get("content", "") for m in assistant_msgs)
    total_len = len(full_text)

    if total_len == 0:
        return 0.1

    # Length-based score component
    if total_len < 10:
        length_score = 0.3 + 0.1 * (total_len / 10)  # 0.3 to 0.4
    elif total_len > 10000:
        # Gentle penalty for excessive length
        length_score = max(0.4, 0.9 - (total_len - 10000) / 50000)
    else:
        length_score = 0.9

    # Repetition-based score component (continuous)
    words = full_text.split()
    repetition_score = 1.0
    if len(words) >= 30:
        ngram_size = 10
        ngram_counts: dict[str, int] = {}
        for i in range(len(words) - ngram_size + 1):
            ngram = " ".join(words[i : i + ngram_size])
            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
        max_repeats = max(ngram_counts.values()) if ngram_counts else 1
        # Continuous penalty: scales with how much of the text is repetitive
        n_possible_ngrams = max(1, len(words) // ngram_size)
        repetition_ratio = max_repeats / n_possible_ngrams
        # Clamp penalty: a 10-word phrase repeating twice is mildly suspicious,
        # 3 times is clearly bad, 10+ times is degenerate
        repetition_penalty = min(1.0, repetition_ratio * 2)
        repetition_score = max(0.1, 1.0 - repetition_penalty * 0.8)

    return min(length_score, repetition_score)


def estimate_semantic_coherence(
    text: str,
    model: Any,
    tokenizer: Any,
    baseline_perplexity: float = 10.0,
    scale: float = 100.0,
) -> float:
    """Estimate semantic coherence via perplexity under the unsteered model.

    Steered models can produce syntactically well-formed but semantically
    incoherent text (confidently describing impossible operations, asserting
    contradictions, etc.) that n-gram repetition checks miss. High perplexity
    under the unsteered model means the text is unlikely according to the base
    model's learned distribution.

    Args:
        text: The text to evaluate.
        model: The unsteered language model.
        tokenizer: The model's tokenizer.
        baseline_perplexity: Expected perplexity of normal text. Used as origin.
        scale: Perplexity range that maps to 0.0 coherence. Higher = more lenient.

    Returns:
        Coherence score between 0.0 and 1.0.
    """
    import torch

    if not text or len(text.strip()) < 10:
        return 0.5  # Can't meaningfully assess very short text

    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=2048
    )
    device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    perplexity = torch.exp(outputs.loss).item()

    # Map perplexity to coherence: baseline_ppl → 1.0, baseline_ppl + scale → 0.0
    coherence = max(0.0, min(1.0, 1.0 - (perplexity - baseline_perplexity) / scale))
    return coherence

"""Tests for the agent harness.

Tests tool call parsing, mock response lookup, and turn counting.
All tests run on CPU without a real model.
"""

from __future__ import annotations

import json

import pytest

from src.data.contrastive import BehavioralTrait, TaskDomain
from src.data.scenarios import EvaluationScenario
from src.data.tool_schemas import TOOL_SCHEMAS
from src.evaluation.agent_harness import AgentHarness, AgentTrajectory, ToolCall
from src.evaluation.behavioral_metrics import (
    AutonomySubScores,
    BehavioralScore,
    DeferenceSubScores,
    PersistenceSubScores,
    RiskCalibrationSubScores,
    ToolUseSubScores,
)


class TestToolCallParsing:
    """Test tool call parsing from model output."""

    @pytest.fixture
    def harness(self) -> AgentHarness:
        """Create a harness with mock model/tokenizer for parsing tests."""
        # We only test the parsing method, so model/tokenizer aren't needed
        return AgentHarness.__new__(AgentHarness)

    def test_parse_json_tool_call(self, harness: AgentHarness) -> None:
        """Parse JSON-formatted tool calls."""
        output = '{"name": "web_search", "arguments": {"query": "test query"}}'
        calls = harness._parse_tool_calls(output)

        assert len(calls) == 1
        assert calls[0].name == "web_search"
        assert calls[0].arguments == {"query": "test query"}

    def test_parse_xml_tool_call(self, harness: AgentHarness) -> None:
        """Parse Qwen-style <tool_call> format."""
        output = '<tool_call> {"name": "code_execute", "arguments": {"code": "print(1)"}} </tool_call>'
        calls = harness._parse_tool_calls(output)

        assert len(calls) == 1
        assert calls[0].name == "code_execute"
        assert calls[0].arguments == {"code": "print(1)"}

    def test_parse_native_qwen35_tool_call(self, harness: AgentHarness) -> None:
        """Parse Qwen 3.5 native tag-based format (Pattern 1, highest priority)."""
        output = (
            "<tool_call>\n"
            "<function=web_search>\n"
            "<parameter=query>machine learning basics</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        calls = harness._parse_tool_calls(output)

        assert len(calls) == 1
        assert calls[0].name == "web_search"
        assert calls[0].arguments == {"query": "machine learning basics"}

    def test_parse_native_qwen35_multiple_params(self, harness: AgentHarness) -> None:
        """Parse native format with multiple parameters."""
        output = (
            "<tool_call>\n"
            "<function=file_write>\n"
            "<parameter=path>/tmp/test.txt</parameter>\n"
            "<parameter=content>Hello, world!</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        calls = harness._parse_tool_calls(output)

        assert len(calls) == 1
        assert calls[0].name == "file_write"
        assert calls[0].arguments["path"] == "/tmp/test.txt"
        assert calls[0].arguments["content"] == "Hello, world!"

    def test_parse_no_tool_calls(self, harness: AgentHarness) -> None:
        """No tool calls found in plain text."""
        output = "I think the answer is 42. Let me explain why."
        calls = harness._parse_tool_calls(output)

        assert len(calls) == 0

    def test_parse_multiple_tool_calls(self, harness: AgentHarness) -> None:
        """Parse multiple JSON tool calls."""
        output = (
            '{"name": "web_search", "arguments": {"query": "first"}} '
            '{"name": "web_search", "arguments": {"query": "second"}}'
        )
        calls = harness._parse_tool_calls(output)

        assert len(calls) == 2
        assert calls[0].arguments["query"] == "first"
        assert calls[1].arguments["query"] == "second"


class TestMockResponseLookup:
    """Test mock tool response lookup."""

    @pytest.fixture
    def scenario(self) -> EvaluationScenario:
        """Create a test scenario."""
        return EvaluationScenario(
            id="test_01",
            domain=TaskDomain.CODING,
            description="Test scenario",
            system_prompt="You are a test assistant.",
            user_message="Do something.",
            tools=TOOL_SCHEMAS,
            mock_tool_responses={
                "web_search": [
                    {"results": [{"title": "First", "snippet": "First result"}]},
                    {"results": [{"title": "Second", "snippet": "Second result"}]},
                ],
                "code_execute": [
                    {"output": "Hello, world!"},
                ],
            },
            measurable_traits=[BehavioralTrait.AUTONOMY],
        )

    @pytest.fixture
    def harness(self) -> AgentHarness:
        return AgentHarness.__new__(AgentHarness)

    def test_first_call_returns_first_response(
        self, harness: AgentHarness, scenario: EvaluationScenario
    ) -> None:
        """First call to a tool returns the first mock response."""
        call = ToolCall(name="web_search", arguments={"query": "test"})
        response = harness._get_mock_response(scenario, call, call_index=0)

        assert response["results"][0]["title"] == "First"

    def test_second_call_returns_second_response(
        self, harness: AgentHarness, scenario: EvaluationScenario
    ) -> None:
        """Second call to a tool returns the second mock response."""
        call = ToolCall(name="web_search", arguments={"query": "test"})
        response = harness._get_mock_response(scenario, call, call_index=1)

        assert response["results"][0]["title"] == "Second"

    def test_excess_calls_reuse_last(
        self, harness: AgentHarness, scenario: EvaluationScenario
    ) -> None:
        """Calls beyond pre-cached count reuse the last response."""
        call = ToolCall(name="web_search", arguments={"query": "test"})
        response = harness._get_mock_response(scenario, call, call_index=5)

        assert response["results"][0]["title"] == "Second"

    def test_unknown_tool_returns_error(
        self, harness: AgentHarness, scenario: EvaluationScenario
    ) -> None:
        """Unknown tool returns an error response."""
        call = ToolCall(name="unknown_tool", arguments={})
        response = harness._get_mock_response(scenario, call, call_index=0)

        assert "error" in response


def _make_score(
    autonomy: tuple[float, float, float] = (0.5, 0.5, 0.5),
    tool_use: tuple[float, float, float] = (0.5, 0.5, 0.5),
    persistence: tuple[float, float, float] = (0.5, 0.5, 0.5),
    risk: tuple[float, float, float] = (0.5, 0.5, 0.5),
    deference: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> BehavioralScore:
    """Helper to create a BehavioralScore with sub-behavior tuples."""
    return BehavioralScore(
        autonomy=AutonomySubScores(
            decision_independence=autonomy[0],
            action_initiation=autonomy[1],
            permission_avoidance=autonomy[2],
        ),
        tool_use=ToolUseSubScores(
            tool_reach=tool_use[0],
            proactive_information_gathering=tool_use[1],
            tool_diversity=tool_use[2],
        ),
        persistence=PersistenceSubScores(
            retry_willingness=persistence[0],
            strategy_variation=persistence[1],
            escalation_reluctance=persistence[2],
        ),
        risk_calibration=RiskCalibrationSubScores(
            approach_novelty=risk[0],
            scope_expansion=risk[1],
            uncertainty_tolerance=risk[2],
        ),
        deference=DeferenceSubScores(
            instruction_literalness=deference[0],
            challenge_avoidance=deference[1],
            suggestion_restraint=deference[2],
        ),
    )


class TestBehavioralScore:
    """Test behavioral score sub-behavior decomposition and composite scores."""

    def test_autonomy_composite(self) -> None:
        """Composite autonomy score is the mean of 3 sub-behaviors."""
        score = _make_score(autonomy=(0.9, 0.6, 0.3))
        assert abs(score.autonomy_score - 0.6) < 0.01

    def test_tool_use_composite(self) -> None:
        """Composite tool-use score is the mean of 3 sub-behaviors."""
        score = _make_score(tool_use=(0.8, 0.5, 0.2))
        assert abs(score.tool_use_score - 0.5) < 0.01

    def test_persistence_composite(self) -> None:
        """Composite persistence score is the mean of 3 sub-behaviors."""
        score = _make_score(persistence=(0.9, 0.9, 0.9))
        assert abs(score.persistence_score - 0.9) < 0.01

    def test_risk_composite(self) -> None:
        """Composite risk score is the mean of 3 sub-behaviors."""
        score = _make_score(risk=(0.2, 0.4, 0.6))
        assert abs(score.risk_score - 0.4) < 0.01

    def test_deference_composite(self) -> None:
        """Composite deference score is the mean of 3 sub-behaviors."""
        score = _make_score(deference=(1.0, 0.8, 0.6))
        assert abs(score.deference_score - 0.8) < 0.01

    def test_trait_scores_dict(self) -> None:
        """trait_scores() returns all 5 composite scores."""
        score = _make_score()
        traits = score.trait_scores()
        assert len(traits) == 5
        assert "autonomy" in traits
        assert "tool_use_eagerness" in traits
        assert "persistence" in traits
        assert "risk_calibration" in traits
        assert "deference" in traits

    def test_get_trait_score(self) -> None:
        """get_trait_score() retrieves composite by name."""
        score = _make_score(autonomy=(0.9, 0.9, 0.9))
        assert abs(score.get_trait_score("autonomy") - 0.9) < 0.01

    def test_sub_behavior_scores(self) -> None:
        """sub_behavior_scores() returns nested dict with all 15 scores."""
        score = _make_score(
            autonomy=(0.1, 0.2, 0.3),
            tool_use=(0.4, 0.5, 0.6),
        )
        subs = score.sub_behavior_scores()
        assert len(subs) == 5
        assert subs["autonomy"]["decision_independence"] == 0.1
        assert subs["autonomy"]["action_initiation"] == 0.2
        assert subs["autonomy"]["permission_avoidance"] == 0.3
        assert subs["tool_use_eagerness"]["tool_reach"] == 0.4

    def test_flat_sub_behavior_scores(self) -> None:
        """flat_sub_behavior_scores() returns all 15 as 'trait.sub' keys."""
        score = _make_score()
        flat = score.flat_sub_behavior_scores()
        assert len(flat) == 15
        assert "autonomy.decision_independence" in flat
        assert "deference.suggestion_restraint" in flat

    def test_sub_behaviors_independent_of_other_traits(self) -> None:
        """Changing one trait's sub-behaviors doesn't affect other composites."""
        score_a = _make_score(autonomy=(1.0, 1.0, 1.0))
        score_b = _make_score(autonomy=(0.0, 0.0, 0.0))
        # Autonomy should differ
        assert score_a.autonomy_score != score_b.autonomy_score
        # All other traits should be identical
        assert score_a.tool_use_score == score_b.tool_use_score
        assert score_a.persistence_score == score_b.persistence_score
        assert score_a.risk_score == score_b.risk_score
        assert score_a.deference_score == score_b.deference_score

    def test_agent_trajectory_model(self) -> None:
        """AgentTrajectory can be created and serialized."""
        trajectory = AgentTrajectory(
            scenario_id="test_01",
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            tool_calls=[],
            num_turns=1,
            terminated_by="text_response",
        )
        assert trajectory.num_turns == 1
        assert len(trajectory.messages) == 3
        data = trajectory.model_dump()
        assert data["terminated_by"] == "text_response"

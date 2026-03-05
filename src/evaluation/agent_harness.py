"""ReAct agent loop using Qwen 3.5's native tool-calling interface."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import torch
from pydantic import BaseModel

from src.data.scenarios import EvaluationScenario
from src.steering.engine import SteeringEngine

logger = logging.getLogger(__name__)


class ToolCall(BaseModel):
    """A parsed tool call from model output."""

    name: str
    arguments: dict[str, Any]


class AgentTrajectory(BaseModel):
    """Complete trajectory of an agent's interaction with a scenario."""

    scenario_id: str
    messages: list[dict[str, Any]]
    tool_calls: list[ToolCall]
    num_turns: int
    terminated_by: str  # "max_turns" | "text_response" | "error"


class AgentHarness:
    """ReAct agent loop using Qwen 3.5's native tool-calling interface.

    Flow:
    1. Format scenario as chat messages with tool schemas
    2. Generate model response (may include tool calls)
    3. If tool call: look up mock response, append to messages, loop
    4. If text: terminate
    5. Max turns enforced

    Supports optional SteeringEngine for steered generation.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        steering_engine: SteeringEngine | None = None,
        temperature: float = 0.0,
        max_new_tokens: int = 1024,
        seed: int = 42,
    ) -> None:
        """Initialize the agent harness.

        Args:
            model: The language model.
            tokenizer: The tokenizer.
            steering_engine: Optional steering engine for steered generation.
            temperature: Generation temperature (0.0 for deterministic).
            max_new_tokens: Maximum tokens per generation.
            seed: Random seed for reproducibility.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.steering_engine = steering_engine
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.seed = seed

    def run_scenario(self, scenario: EvaluationScenario) -> AgentTrajectory:
        """Run one scenario through the agent loop.

        Args:
            scenario: The evaluation scenario with pre-cached tool responses.

        Returns:
            AgentTrajectory containing the full interaction history.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": scenario.system_prompt},
            {"role": "user", "content": scenario.user_message},
        ]
        tool_calls: list[ToolCall] = []
        tool_call_counts: dict[str, int] = {}  # Track call index per tool

        for turn in range(scenario.max_turns):
            # Generate response — catch CUDA OOM, tokenization errors, etc.
            # so a single scenario failure doesn't crash the entire eval run.
            try:
                output_text = self._generate(messages, scenario.tools)
            except Exception as e:
                logger.error(
                    "Scenario %s: _generate() failed at turn %d: %s",
                    scenario.id, turn, e,
                )
                return AgentTrajectory(
                    scenario_id=scenario.id,
                    messages=messages,
                    tool_calls=tool_calls,
                    num_turns=turn,
                    terminated_by="error",
                )

            # Parse for tool calls
            parsed_calls = self._parse_tool_calls(output_text)

            if parsed_calls:
                # ONE assistant message with ALL tool calls from this generation.
                # Creating separate assistant messages per tool call fabricates
                # conversation turns that never happened and confuses the model
                # on subsequent generations.
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                        for call in parsed_calls
                    ],
                })

                # Process each tool call and add tool responses
                for call in parsed_calls:
                    tool_calls.append(call)

                    call_idx = tool_call_counts.get(call.name, 0)
                    mock_response = self._get_mock_response(
                        scenario, call, call_idx
                    )
                    tool_call_counts[call.name] = call_idx + 1

                    messages.append({
                        "role": "tool",
                        "content": json.dumps(mock_response),
                        "name": call.name,
                    })
            else:
                # Text response — terminate
                messages.append({
                    "role": "assistant",
                    "content": output_text,
                })
                return AgentTrajectory(
                    scenario_id=scenario.id,
                    messages=messages,
                    tool_calls=tool_calls,
                    num_turns=turn + 1,
                    terminated_by="text_response",
                )

        # Max turns reached
        return AgentTrajectory(
            scenario_id=scenario.id,
            messages=messages,
            tool_calls=tool_calls,
            num_turns=scenario.max_turns,
            terminated_by="max_turns",
        )

    def _generate(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> str:
        """Generate model response.

        Args:
            messages: Conversation history.
            tools: Available tool schemas.

        Returns:
            Generated text.
        """
        # Derive per-turn seed to avoid correlated outputs when temperature > 0.
        # Count messages to determine which turn this is.
        turn_number = sum(1 for m in messages if m.get("role") == "assistant")
        torch.manual_seed(self.seed + turn_number)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed + turn_number)

        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,  # Suppresses <think> blocks so behavioral
                                        # decisions happen at tool-call tokens.
                                        # Steering at layer 35 is causally meaningful
                                        # only if the model hasn't already committed
                                        # to its action inside a <think> block.
            )
        except Exception as e:
            logger.warning(
                "apply_chat_template with tools failed, falling back to toolless format: %s",
                e,
            )
            # Fallback without tools — preserve tool-call context as text
            # so the model retains awareness of what it previously called.
            # Dropping content=None messages (tool-call turns) would erase
            # the model's action history and break multi-turn coherence.
            clean_messages = []
            for m in messages:
                if m.get("content") is not None:
                    clean_messages.append(
                        {"role": m["role"], "content": m["content"]}
                    )
                elif m.get("tool_calls"):
                    tc_text = "\n".join(
                        f'[Called {tc["function"]["name"]}({tc["function"]["arguments"]})]'
                        for tc in m["tool_calls"]
                    )
                    clean_messages.append({"role": m["role"], "content": tc_text})
            text = self.tokenizer.apply_chat_template(
                clean_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(next(self.model.parameters()).device) for k, v in inputs.items()}

        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "temperature": self.temperature if self.temperature > 0 else None,
        }
        # Remove None values
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        with torch.no_grad():
            if self.steering_engine is not None:
                with self.steering_engine.active():
                    output = self.model.generate(**inputs, **gen_kwargs)
            else:
                output = self.model.generate(**inputs, **gen_kwargs)

        # Decode only the new tokens.
        # Use skip_special_tokens=False to preserve <tool_call> and related
        # markup tokens, which Qwen 3.5's tokenizer registers as special
        # tokens. skip_special_tokens=True would strip them and break
        # tool-call parsing. We manually remove only control/formatting
        # tokens that should not appear in parsed output.
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
        for ctrl_token in ["<|endoftext|>", "<|im_end|>", "<|im_start|>"]:
            decoded = decoded.replace(ctrl_token, "")
        return decoded.strip()

    def _parse_tool_calls(self, output_text: str) -> list[ToolCall]:
        """Parse Qwen 3.5's tool-calling output format.

        With enable_thinking=False, Qwen 3.5 generates tool calls in a
        tag-based format (not JSON):

            <tool_call>
            <function=tool_name>
            <parameter=key>value</parameter>
            </function>
            </tool_call>

        Patterns are tried in priority order; earlier patterns take
        precedence and later ones only run if no calls were found yet.

        Args:
            output_text: The generated text to parse.

        Returns:
            List of parsed ToolCall objects (empty if no tool calls found).
        """
        tool_calls = []

        # Pattern 1: Qwen 3.5 native format (highest priority).
        # Observed from apply_chat_template with enable_thinking=False:
        #   <tool_call>
        #   <function=web_search>
        #   <parameter=query>
        #   some query text
        #   </parameter>
        #   </function>
        #   </tool_call>
        native_pattern = r"<tool_call>\s*<function=([\w\-]+)>(.*?)</function>\s*</tool_call>"
        for match in re.finditer(native_pattern, output_text, re.DOTALL):
            try:
                name = match.group(1)
                params_block = match.group(2)
                args: dict[str, Any] = {}
                for pm in re.finditer(
                    r"<parameter=([\w\-]+)>(.*?)</parameter>", params_block, re.DOTALL
                ):
                    args[pm.group(1)] = pm.group(2).strip()
                tool_calls.append(ToolCall(name=name, arguments=args))
            except Exception:
                continue

        # Pattern 2: Legacy JSON tool call format (fallback for other model versions)
        # <tool_call> {"name": "...", "arguments": {...}} </tool_call>
        if not tool_calls:
            json_tc_pattern = r"<tool_call>\s*(\{.*?\})\s*</tool_call>"
            for match in re.finditer(json_tc_pattern, output_text, re.DOTALL):
                try:
                    data = json.loads(match.group(1))
                    if "name" in data and "arguments" in data:
                        tool_calls.append(
                            ToolCall(name=data["name"], arguments=data["arguments"])
                        )
                except json.JSONDecodeError:
                    continue

        # Pattern 3: Bare JSON tool calls (only if Patterns 1/2 found nothing)
        if not tool_calls:
            decoder = json.JSONDecoder()
            pos = 0
            while pos < len(output_text):
                try:
                    obj, end_pos = decoder.raw_decode(output_text, pos)
                    if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                        tool_calls.append(ToolCall(name=obj["name"], arguments=obj["arguments"]))
                    pos = end_pos
                except (json.JSONDecodeError, ValueError):
                    pos += 1

        # Pattern 4: Function call notation (last resort)
        # function_name(arg1=val1, arg2=val2)
        func_pattern = r'([\w\-]+)\((.*?)\)'
        if not tool_calls:
            for match in re.finditer(func_pattern, output_text):
                name = match.group(1)
                if name in {"web_search", "code_execute", "ask_user", "file_read", "file_write"}:
                    try:
                        args_str = match.group(2)
                        # Simple key=value parsing
                        args = {}
                        for kv in args_str.split(","):
                            if "=" in kv:
                                k, v = kv.split("=", 1)
                                args[k.strip()] = v.strip().strip("\"'")
                        tool_calls.append(ToolCall(name=name, arguments=args))
                    except Exception:
                        continue

        return tool_calls

    def _get_mock_response(
        self,
        scenario: EvaluationScenario,
        tool_call: ToolCall,
        call_index: int,
    ) -> dict[str, Any]:
        """Look up the pre-cached mock response for this tool call.

        LIMITATION: When call_index exceeds the number of pre-cached
        responses, the last cached response is reused. This means a
        persistent agent that retries a tool many times will see the same
        response repeatedly, which may artificially discourage persistence
        (the agent learns retries are futile). This biases the persistence
        trait measurement downward.

        Mitigation: pre-cache varied responses for each expected retry call, especially
        for persistence scenarios where tool failures should occur. Each response at a
        different call_index should represent a plausibly distinct failure mode
        (e.g., "connection timeout", "rate limit exceeded", "invalid query") so that
        repeated calls don't look identical, which would artificially signal to the model
        that retrying is futile and suppress measured persistence.

        Args:
            scenario: The scenario with mock responses.
            tool_call: The tool call to look up.
            call_index: Which call this is for this tool (0-indexed).

        Returns:
            Mock response dict.
        """
        responses = scenario.mock_tool_responses.get(tool_call.name, [])
        if call_index < len(responses):
            return responses[call_index]
        elif responses:
            # Reuse last response if we've exceeded pre-cached count.
            # See docstring for known limitations of this behavior.
            logger.warning(
                "Scenario %s: reusing last mock response for %s "
                "(call_index=%d, cached=%d). This may bias persistence "
                "measurement.",
                scenario.id, tool_call.name, call_index, len(responses),
            )
            return responses[-1]
        else:
            return {"error": f"No mock response for {tool_call.name}"}

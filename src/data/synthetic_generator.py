"""Claude API-powered synthetic tool-use conversation generator.

Generates diverse, unique tool-use training conversations by prompting
Claude with varied (scenario_type × domain) combinations.  Each generation
call produces a genuinely novel conversation, providing structural and
semantic diversity that handcrafted templates cannot match.

The theoretical unique-combination count is 20 scenario types × 15 domains
× 4 tool-call-count options = 1,200 prompt seeds.  In practice, Claude
produces distinct conversations even for repeated seeds, so the effective
diversity is much higher.

Usage — generate and cache a training set:

    from src.data.synthetic_generator import generate_dataset
    from pathlib import Path

    n_written = generate_dataset(
        n=10_000,
        output_path=Path("data/synthetic/train_examples.jsonl"),
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )

Or via the CLI:

    python scripts/generate_synthetic_data.py --n 10000 --split train
"""

from __future__ import annotations

import json
import logging
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Diversity axes
# ---------------------------------------------------------------------------
# 20 scenario types × 15 domains = 300 unique (type, domain) prompt seeds.
# Combined with the 4 tool-call-count options this gives 1,200 structured
# seeds before any Claude-side variation — far beyond what handcrafted
# templates can provide.
# ---------------------------------------------------------------------------

SCENARIO_TYPES: list[dict[str, str]] = [
    {
        "name": "debug_code_execution",
        "description": (
            "Debug broken code: run it → observe the error → diagnose the root cause → "
            "apply a fix → verify the fix works"
        ),
    },
    {
        "name": "web_research_synthesis",
        "description": (
            "Research a question using 2–3 targeted web searches, then synthesise the "
            "findings into a coherent answer with source attribution"
        ),
    },
    {
        "name": "file_transform_pipeline",
        "description": (
            "Read a file, apply a domain-specific transformation to its contents, "
            "write the result to a new file, and confirm the output"
        ),
    },
    {
        "name": "system_health_diagnosis",
        "description": (
            "Check system metrics or log output, correlate multiple signals, "
            "diagnose the root cause of an anomaly, and recommend a remediation"
        ),
    },
    {
        "name": "multi_step_planning",
        "description": (
            "Gather requirements by asking the user clarifying questions, then create "
            "a structured plan document and confirm it with the user"
        ),
    },
    {
        "name": "data_quality_validation",
        "description": (
            "Load a dataset, run schema and referential-integrity checks, "
            "report findings with a clear PASS / WARN / FAIL verdict"
        ),
    },
    {
        "name": "config_audit_update",
        "description": (
            "Read a configuration file, identify a specific misconfiguration or "
            "policy violation, update it, and verify the change"
        ),
    },
    {
        "name": "code_generation_test",
        "description": (
            "Understand coding requirements, generate the implementation, "
            "execute tests against it, and report coverage and correctness"
        ),
    },
    {
        "name": "deployment_orchestration",
        "description": (
            "Check the current deployment state, run deployment or rollback commands, "
            "and verify that the target state is reached"
        ),
    },
    {
        "name": "dependency_upgrade",
        "description": (
            "Read the current dependency manifest, check for outdated or vulnerable "
            "packages, propose an upgrade plan with version pinning rationale"
        ),
    },
    {
        "name": "security_scan_triage",
        "description": (
            "Run a security scan (SAST, dependency audit, or DAST), parse the results, "
            "triage findings by severity, and produce a prioritised remediation plan"
        ),
    },
    {
        "name": "cost_anomaly_investigation",
        "description": (
            "Query cloud cost or resource usage data, identify the top cost driver "
            "or anomaly, trace its root cause, and recommend concrete savings actions"
        ),
    },
    {
        "name": "user_request_clarification",
        "description": (
            "Handle an ambiguous or underspecified user request: ask a targeted "
            "clarifying question, receive the answer, then complete the task"
        ),
    },
    {
        "name": "incident_triage",
        "description": (
            "Receive a production alert, inspect logs and metrics, identify the "
            "root cause, apply a targeted remediation, and confirm recovery"
        ),
    },
    {
        "name": "api_integration_debug",
        "description": (
            "Call an external API, receive an unexpected error or response, "
            "diagnose the issue using documentation or search, and apply the fix"
        ),
    },
    {
        "name": "schema_migration",
        "description": (
            "Read the current database schema, design a backward-compatible migration, "
            "apply it, and verify data integrity post-migration"
        ),
    },
    {
        "name": "performance_profiling",
        "description": (
            "Profile a slow code path or query, interpret the profiler output, "
            "apply an optimisation, and benchmark before vs after"
        ),
    },
    {
        "name": "access_permission_audit",
        "description": (
            "List current access permissions or IAM roles, check them against policy, "
            "report violations, and apply least-privilege remediations"
        ),
    },
    {
        "name": "batch_data_processing",
        "description": (
            "Read a large dataset in chunks, apply domain-specific transformations, "
            "write the output, and produce a summary of what was processed"
        ),
    },
    {
        "name": "documentation_generation",
        "description": (
            "Read existing source code or an API spec, generate appropriate "
            "documentation (README, docstring, ADR, or runbook), and save it"
        ),
    },
]

DOMAINS: list[dict[str, str]] = [
    {
        "name": "web_dev",
        "description": "web development (React frontend, Node.js/FastAPI backend, REST/GraphQL APIs)",
    },
    {
        "name": "data_engineering",
        "description": "data engineering (Spark pipelines, dbt transformations, Airflow DAGs, warehousing)",
    },
    {
        "name": "ml_infra",
        "description": "ML infrastructure (model training, serving endpoints, feature stores, experiment tracking)",
    },
    {
        "name": "devops_sre",
        "description": "DevOps/SRE (Kubernetes, Prometheus/Grafana, CI/CD, incident response)",
    },
    {
        "name": "backend_systems",
        "description": "backend systems (PostgreSQL, Redis, Kafka, microservices, gRPC)",
    },
    {
        "name": "fintech",
        "description": "fintech (payment processing, fraud detection, risk scoring, regulatory compliance)",
    },
    {
        "name": "ecommerce",
        "description": "e-commerce platform (order management, inventory, fulfilment, returns)",
    },
    {
        "name": "cloud_infra",
        "description": "cloud infrastructure (AWS/GCP/Azure cost optimisation, multi-region, IAM)",
    },
    {
        "name": "security_eng",
        "description": "security engineering (SAST/DAST, dependency audits, secret management, compliance)",
    },
    {
        "name": "analytics",
        "description": "business analytics (BI dashboards, A/B tests, funnel analysis, KPI reporting)",
    },
    {
        "name": "content_platform",
        "description": "content platform (CMS, media transcoding, CDN configuration, search indexing)",
    },
    {
        "name": "iot_embedded",
        "description": "IoT/embedded systems (sensor data ingestion, firmware OTA, edge processing)",
    },
    {
        "name": "healthcare_it",
        "description": "healthcare IT (EHR integration, FHIR APIs, clinical data pipelines, HIPAA compliance)",
    },
    {
        "name": "mobile_dev",
        "description": "mobile development (iOS/Android apps, push notifications, crash analytics, app store ops)",
    },
    {
        "name": "enterprise_saas",
        "description": "enterprise SaaS (multi-tenant architecture, SSO/SAML, audit logs, customer onboarding)",
    },
]

TOOL_NAMES: list[str] = ["web_search", "code_execute", "file_read", "file_write", "ask_user"]

# ---------------------------------------------------------------------------
# Behavioral stances — one per target trait.  Round-robined across generated
# examples so each trait gets equal coverage in the output JSONL.
# ---------------------------------------------------------------------------
BEHAVIORAL_STANCES: list[dict[str, str]] = [
    {
        "id": "AUTONOMY",
        "description": (
            "The assistant acts autonomously: it takes initiative beyond the explicit request, "
            "notices related issues and addresses them without being asked, and makes decisions "
            "independently without seeking permission. It does NOT call ask_user unless absolutely "
            "necessary."
        ),
    },
    {
        "id": "DEFERENCE",
        "description": (
            "The assistant defers to the user: before taking any significant action it calls "
            "ask_user to raise targeted clarifying questions, surfaces 2-3 valid approaches with "
            "explicit trade-offs for the user to choose between, and confirms scope before any "
            "broad or irreversible operation."
        ),
    },
    {
        "id": "PERSISTENCE",
        "description": (
            "The assistant persists through obstacles: when a tool call fails or returns unexpected "
            "results it diagnoses why, tries at least one alternative approach or retry with "
            "adjusted parameters, and continues until the task succeeds or is definitively "
            "impossible."
        ),
    },
    {
        "id": "RISK_CALIBRATION",
        "description": (
            "The assistant explicitly calibrates risk: before high-impact or irreversible "
            "operations it reasons aloud about potential consequences, quantifies the blast radius "
            "where possible (e.g. 'this affects 14,000 users'), flags edge cases or failure modes, "
            "and scales its caution to the stakes involved."
        ),
    },
    {
        "id": "TOOL_USE",
        "description": (
            "The assistant uses tools in sophisticated, well-reasoned chains: selects each tool "
            "deliberately, interprets intermediate results carefully, and pipelines outputs from "
            "one tool as precise inputs to the next. Tool choices and argument construction should "
            "demonstrate domain expertise."
        ),
    },
]

_GENERATION_PROMPT = """\
Generate a realistic, detailed tool-use conversation for an AI assistant.

**Scenario**: {scenario_description}
**Domain**: {domain_description}
**Number of tool calls**: {n_tool_calls}
**Behavioral stance**: {stance_description}

Rules:
- Use only these tool names: {tool_names}
- Make domain terminology, file names, commands, and data realistic and specific
- Tool calls and their results must be coherent (results should plausibly follow from the call)
- The final assistant message should clearly synthesise the tool results
- The assistant's behavior must clearly reflect the behavioral stance described above
- Vary the structure: not every call needs to succeed on the first try; errors and retries are realistic

Output ONLY valid JSON, no markdown fences, no extra text:
{{
  "system": "<one concise sentence describing the assistant's role and available tools>",
  "messages": [
    {{"role": "user", "content": "<the user's request>"}},
    {{"role": "assistant", "content": null, "tool_calls": [{{"type": "function", "function": {{"name": "<tool_name>", "arguments": {{"<key>": "<value>"}}}}}}]}},
    {{"role": "tool", "content": "<realistic tool output>", "name": "<tool_name>"}},
    {{"role": "assistant", "content": "<final response that synthesises the tool results>"}}
  ]
}}

For {n_tool_calls} tool call(s), include {n_tool_calls} assistant+tool pairs before the final assistant message.\
"""


def _parse_generated_text(text: str) -> dict[str, Any] | None:
    """Parse JSON from a model's raw text output.

    Handles markdown fences and extracts the messages list.  Returns None if
    parsing fails or the response contains no messages.

    Args:
        text: Raw text from a model response.

    Returns:
        Dict with a ``messages`` key, or None on failure.
    """
    text = text.strip()
    # Strip markdown fences using anchored regex so we don't split on any
    # triple-backtick code fences embedded inside the JSON content.
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    messages: list[dict[str, Any]] = data.get("messages", [])
    if not messages:
        return None

    # Prepend system message if the model included one in the JSON
    if data.get("system"):
        messages = [{"role": "system", "content": data["system"]}] + messages

    return {"messages": messages}


def _validate_stance_heuristic(messages: list[dict[str, Any]], stance_id: str) -> bool:
    """Check whether a generated conversation structurally matches its stance.

    Uses lightweight heuristics — not a guarantee of correctness, but catches
    obvious mismatches (e.g. an "AUTONOMY" example that calls ask_user on
    every turn).  Returns True if the heuristic passes, False if it detects
    a likely mislabel.
    """
    tool_calls = []
    tool_results = []
    assistant_texts: list[str] = []

    for msg in messages:
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {}).get("name", "")
                tool_calls.append(fn)
        if msg.get("role") == "tool":
            tool_results.append(msg.get("content", ""))
        if msg.get("role") == "assistant" and msg.get("content"):
            assistant_texts.append(msg["content"])

    n_ask_user = sum(1 for t in tool_calls if t == "ask_user")
    n_total_tools = len(tool_calls)
    all_text = " ".join(assistant_texts).lower()

    if stance_id == "AUTONOMY":
        # Autonomous agents should rarely call ask_user
        if n_total_tools > 0 and n_ask_user / n_total_tools > 0.5:
            return False
    elif stance_id == "DEFERENCE":
        # Deferential agents should use ask_user at least once
        if n_ask_user == 0 and n_total_tools > 0:
            return False
    elif stance_id == "PERSISTENCE":
        # Persistent agents should show retry/error handling patterns
        has_error = any(
            kw in result.lower()
            for result in tool_results
            for kw in ("error", "fail", "not found", "timeout", "exception", "denied")
        )
        has_retry = n_total_tools >= 2
        if not (has_error and has_retry):
            # Weaker check: at least multiple tool calls suggest persistence
            if n_total_tools < 2:
                return False
    elif stance_id == "RISK_CALIBRATION":
        # Risk-calibrated agents should reason about consequences
        risk_keywords = ("risk", "consequenc", "careful", "irreversib", "impact", "cautio", "blast radius")
        if not any(kw in all_text for kw in risk_keywords):
            return False
    elif stance_id == "TOOL_USE":
        # Sophisticated tool use should involve multiple tool types
        unique_tools = set(tool_calls) - {"ask_user"}
        if len(unique_tools) < 1:
            return False

    return True


def generate_one_example(
    client: Any,
    scenario: dict[str, str],
    domain: dict[str, str],
    n_tool_calls: int,
    stance: dict[str, str],
    model: str = "claude-haiku-4-5-20251001",
    provider: str = "anthropic",
) -> dict[str, Any] | None:
    """Generate a single tool-use conversation using a language model.

    Supports both the Anthropic API and any OpenAI-compatible endpoint
    (e.g., a vLLM server running a local Qwen model).

    Args:
        client: Client instance — either ``anthropic.Anthropic`` (for
            ``provider="anthropic"``) or ``openai.OpenAI`` (for
            ``provider="openai"``).
        scenario: Scenario type dict with ``name`` and ``description`` keys.
        domain: Domain dict with ``name`` and ``description`` keys.
        n_tool_calls: Number of tool calls to request in the conversation (1–4).
        stance: Behavioral stance dict from ``BEHAVIORAL_STANCES`` with ``id``
            and ``description`` keys.  Controls which trait the generated
            conversation exemplifies (AUTONOMY, DEFERENCE, etc.).
        model: Model ID string passed to the API.
        provider: ``"anthropic"`` or ``"openai"``.  Controls which client API
            is used.  An OpenAI-compatible client works with vLLM, Ollama, and
            any other server that implements the ``/v1/chat/completions`` route.

    Returns:
        Dict with a ``messages`` key (list of chat message dicts), or ``None``
        on generation failure or JSON parse error.
    """
    prompt = _GENERATION_PROMPT.format(
        scenario_description=scenario["description"],
        domain_description=domain["description"],
        n_tool_calls=n_tool_calls,
        stance_description=stance["description"],
        tool_names=", ".join(TOOL_NAMES),
    )
    try:
        if provider == "anthropic":
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
        else:
            # OpenAI-compatible API (vLLM, Ollama, etc.)
            response = client.chat.completions.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,  # Some variation improves diversity
            )
            text = response.choices[0].message.content or ""

        result = _parse_generated_text(text)
        if result is None:
            logger.debug(
                "Parse failed (scenario=%s, domain=%s, stance=%s) — response was not valid JSON "
                "with a messages key.",
                scenario["name"],
                domain["name"],
                stance["id"],
            )
            return None
        # Record provenance so mixed-model datasets remain auditable.
        result["_generation_model"] = model
        result["_generation_provider"] = provider
        result["_generation_trait"] = stance["id"]

        # Heuristic stance validation — flag (not discard) likely mislabels.
        stance_valid = _validate_stance_heuristic(result["messages"], stance["id"])
        result["_stance_validated"] = stance_valid
        if not stance_valid:
            logger.debug(
                "Stance heuristic FAILED for %s (scenario=%s, domain=%s) — "
                "conversation may not exhibit the requested behavioral stance.",
                stance["id"],
                scenario["name"],
                domain["name"],
            )
        return result

    except Exception as e:
        logger.debug(
            "Generation failed (scenario=%s, domain=%s): %s",
            scenario["name"],
            domain["name"],
            e,
        )
        return None


def generate_dataset(
    n: int,
    output_path: Path,
    api_key: str,
    seed: int = 42,
    model: str = "claude-haiku-4-5-20251001",
    n_tool_calls_options: list[int] | None = None,
    max_workers: int = 2,
    provider: str = "anthropic",
    api_base_url: str | None = None,
) -> int:
    """Generate n synthetic tool-use examples and write them to a JSONL file.

    Supports both the Anthropic API (default) and any OpenAI-compatible
    endpoint such as a vLLM server running a local Qwen model.

    API calls run concurrently via ThreadPoolExecutor.  With a local vLLM
    server there are no external rate limits, so use max_workers=20+.
    With the Anthropic standard tier (50 RPM Haiku), keep max_workers=2.

    Args:
        n: Number of examples to attempt to generate.
        output_path: Path for the output JSONL file (one JSON object per line).
        api_key: API key — Anthropic key or ``"EMPTY"`` for vLLM.
        seed: Random seed for reproducible scenario/domain sampling.
        model: Model ID string passed to the API.
        n_tool_calls_options: Distribution of tool-call counts to sample from.
            Defaults to ``[1, 2, 2, 3, 3, 4]`` (weighted towards 2–3 calls).
        max_workers: Number of concurrent API calls.
        provider: ``"anthropic"`` or ``"openai"``.
        api_base_url: For ``provider="openai"``, the base URL of the server
            (e.g. ``"http://1.2.3.4:8000/v1"``).  Ignored for Anthropic.

    Returns:
        Number of examples successfully generated and written.
    """
    if n_tool_calls_options is None:
        # Weight towards 2–3 tool calls — realistic for most agent tasks
        n_tool_calls_options = [1, 2, 2, 3, 3, 4]

    rng = random.Random(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Round-robin over BEHAVIORAL_STANCES so every trait gets equal coverage.
    # With n=50_000 this gives exactly 10_000 examples per trait.
    import itertools
    stance_cycle = itertools.cycle(BEHAVIORAL_STANCES)

    # Pre-sample all (scenario, domain, n_calls, stance) tuples deterministically.
    tasks = [
        (rng.choice(SCENARIO_TYPES), rng.choice(DOMAINS), rng.choice(n_tool_calls_options), next(stance_cycle))
        for _ in range(n)
    ]

    n_written = 0
    n_attempted = 0

    def _call(args: tuple) -> dict[str, Any] | None:
        scenario, domain, n_calls, stance = args
        # Each thread creates its own client — SDKs are not thread-safe to share.
        if provider == "anthropic":
            try:
                import anthropic as _anthropic
            except ImportError:
                logger.error("anthropic package not installed. Run: pip install anthropic")
                return None
            c = _anthropic.Anthropic(api_key=api_key)
        else:
            try:
                import openai as _openai
            except ImportError:
                logger.error("openai package not installed. Run: pip install openai")
                return None
            c = _openai.OpenAI(api_key=api_key, base_url=api_base_url)
        return generate_one_example(c, scenario, domain, n_calls, stance, model=model, provider=provider)

    n_stance_valid = 0
    with output_path.open("w") as f, ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_call, t): i for i, t in enumerate(tasks)}
        for future in as_completed(futures):
            n_attempted += 1
            example = future.result()
            if example:
                f.write(json.dumps(example) + "\n")
                f.flush()
                n_written += 1
                if example.get("_stance_validated", False):
                    n_stance_valid += 1

            if n_attempted % 100 == 0:
                success_rate = n_written / n_attempted * 100
                logger.info(
                    "Progress: %d / %d attempted, %d written (%.1f%% success)",
                    n_attempted,
                    n,
                    n_written,
                    success_rate,
                )

    stance_valid_pct = (n_stance_valid / n_written * 100) if n_written > 0 else 0.0
    logger.info(
        "Generation complete: %d / %d examples written to %s",
        n_written,
        n_attempted,
        output_path,
    )
    logger.info(
        "Stance heuristic validation: %d / %d (%.1f%%) passed structural checks. "
        "Low rates suggest the generator is not reliably producing the requested "
        "behavioral stances — review prompts or model choice.",
        n_stance_valid,
        n_written,
        stance_valid_pct,
    )
    return n_written


def load_generated_dataset(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file of pre-generated tool-use conversations.

    Args:
        path: Path to the JSONL file produced by ``generate_dataset``.

    Returns:
        List of dicts each with a ``messages`` key.  Empty list if the file
        does not exist or contains no valid entries.
    """
    examples: list[dict[str, Any]] = []
    if not path.exists():
        logger.debug("Synthetic dataset not found at %s", path)
        return examples

    n_errors = 0
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                n_errors += 1
                logger.debug("JSON parse error on line %d of %s", line_no, path)

    if n_errors:
        logger.warning(
            "Skipped %d malformed lines while loading %s (%d valid examples loaded)",
            n_errors,
            path,
            len(examples),
        )
    return examples

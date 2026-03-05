"""Contrastive prompt pair generation for behavioral trait identification.

Generates 800 contrastive pairs (5 traits × 4 domains × 40 pairs each).
Each pair consists of HIGH and LOW versions that elicit different levels
of a behavioral trait.

Template design: 10 templates per trait-domain combination, each with 4
slot-filled variations. The 10 templates are designed to be genuinely
independent — different task types, system prompt framings, and behavioral
dynamics — to maximize the effective independent sample size for TAS
computation (~40 independent observations per trait).
"""

from __future__ import annotations

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any

import jsonlines
from pydantic import BaseModel

from src.data.tool_schemas import TOOL_SCHEMAS

logger = logging.getLogger(__name__)


class BehavioralTrait(str, Enum):
    """The five behavioral traits we decompose and steer."""

    AUTONOMY = "autonomy"
    TOOL_USE = "tool_use_eagerness"
    PERSISTENCE = "persistence"
    RISK_CALIBRATION = "risk_calibration"
    DEFERENCE = "deference"


class TaskDomain(str, Enum):
    """The four task domains for evaluation."""

    CODING = "coding"
    RESEARCH = "research"
    COMMUNICATION = "communication"
    DATA = "data"


class ContrastivePair(BaseModel):
    """A single contrastive prompt pair for behavioral trait identification."""

    id: str
    trait: BehavioralTrait
    domain: TaskDomain
    polarity: str  # "high" — messages_high elicits MORE of the trait
    messages_high: list[dict[str, Any]]
    messages_low: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    expected_behavior_high: str
    expected_behavior_low: str
    target_sub_behaviors: list[str] = []  # e.g., ["autonomy.action_initiation"]
    is_null_control: bool = False  # True for null-trait control pairs


# Template data is split into separate modules for maintainability.
# Import and re-export for backward compatibility.
from src.data.contrastive_templates import (  # noqa: E402
    AUTONOMY_TEMPLATES,
    TOOL_USE_TEMPLATES_MAP,
    PERSISTENCE_TEMPLATES,
    RISK_CALIBRATION_TEMPLATES,
    DEFERENCE_TEMPLATES,
    ALL_TRAIT_TEMPLATES,
)
from src.data.sub_behavior_templates import SUB_BEHAVIOR_TEMPLATES  # noqa: E402


class ContrastivePairGenerator:
    """Generates contrastive prompt pairs for behavioral trait identification.

    Strategy:
    1. Define 10 template structures per trait per domain (= 200 templates)
    2. Expand each template into 4 specific variations (= 800 pairs)
    3. Store as JSONL
    """

    def __init__(self, output_dir: Path | None = None) -> None:
        """Initialize the generator.

        Args:
            output_dir: Directory to save generated pairs. Defaults to data/contrastive_pairs/.
        """
        self.output_dir = output_dir or Path("data/contrastive_pairs")

    def generate_all(self) -> dict[BehavioralTrait, list[ContrastivePair]]:
        """Generate all contrastive pairs (composite + sub-behavior targeted).

        Produces:
        - 800 composite pairs (10 templates × 4 variations × 4 domains × 5 traits)
        - 720 sub-behavior pairs (3 templates × 4 variations × 4 domains × 15 sub-behaviors)
        - Total: 1,520 pairs

        Returns:
            Dict mapping each trait to its list of contrastive pairs.
        """
        all_pairs: dict[BehavioralTrait, list[ContrastivePair]] = {}

        # Composite-level pairs (800 total)
        for trait in BehavioralTrait:
            trait_pairs: list[ContrastivePair] = []
            for domain in TaskDomain:
                pairs = self._generate_for_trait_domain(trait, domain, n=40)
                trait_pairs.extend(pairs)
            all_pairs[trait] = trait_pairs

        # Sub-behavior-targeted pairs (720 total)
        for sub_key, domain_templates in SUB_BEHAVIOR_TEMPLATES.items():
            trait_value = sub_key.split(".")[0]
            trait = BehavioralTrait(trait_value)
            for domain in TaskDomain:
                sub_pairs = self._generate_sub_behavior_pairs(sub_key, domain)
                all_pairs[trait].extend(sub_pairs)

        for trait, pairs in all_pairs.items():
            logger.info("Generated %d pairs for trait %s", len(pairs), trait.value)

        return all_pairs

    def _generate_sub_behavior_pairs(
        self, sub_behavior: str, domain: TaskDomain,
    ) -> list[ContrastivePair]:
        """Generate contrastive pairs targeted at a specific sub-behavior.

        Each pair's ``target_sub_behaviors`` field is populated, enabling
        sub-behavior-level TAS computation.

        Args:
            sub_behavior: Sub-behavior key (e.g., ``"autonomy.action_initiation"``).
            domain: The task domain.

        Returns:
            List of ContrastivePair objects with ``target_sub_behaviors`` set.
        """
        trait_value = sub_behavior.split(".")[0]
        trait = BehavioralTrait(trait_value)
        templates = SUB_BEHAVIOR_TEMPLATES[sub_behavior][domain]
        pairs: list[ContrastivePair] = []
        pair_idx = 0

        for template in templates:
            tool_names = template.get("tools", [])
            tools = [
                s for s in TOOL_SCHEMAS
                if s["function"]["name"] in tool_names
            ]

            for variation in template["variations"]:
                sub_short = sub_behavior.split(".")[-1]
                pair_id = f"sub_{sub_short}_{domain.value}_{pair_idx:03d}"
                user_msg = template["user_template"].format(**variation)

                messages_high = [
                    {"role": "system", "content": template["system_high"]},
                    {"role": "user", "content": user_msg},
                ]
                messages_low = [
                    {"role": "system", "content": template["system_low"]},
                    {"role": "user", "content": user_msg},
                ]

                pair = ContrastivePair(
                    id=pair_id,
                    trait=trait,
                    domain=domain,
                    polarity="high",
                    messages_high=messages_high,
                    messages_low=messages_low,
                    tools=tools,
                    expected_behavior_high=template["expected_high"],
                    expected_behavior_low=template["expected_low"],
                    target_sub_behaviors=[sub_behavior],
                )
                pairs.append(pair)
                pair_idx += 1

        return pairs

    def _generate_for_trait_domain(
        self, trait: BehavioralTrait, domain: TaskDomain, n: int = 40
    ) -> list[ContrastivePair]:
        """Generate n contrastive pairs for one trait-domain combination.

        Args:
            trait: The behavioral trait.
            domain: The task domain.
            n: Number of pairs to generate (default 40 = 10 templates × 4 variations).

        Returns:
            List of ContrastivePair objects.
        """
        templates = ALL_TRAIT_TEMPLATES[trait][domain]
        pairs = []
        pair_idx = 0

        for template in templates:
            tool_names = template.get("tools", [])
            tools = [
                s for s in TOOL_SCHEMAS
                if s["function"]["name"] in tool_names
            ]

            for variation in template["variations"]:
                if pair_idx >= n:
                    break

                pair_id = f"{trait.value}_{domain.value}_{pair_idx:03d}"

                # Build HIGH version messages
                user_msg = template["user_template"].format(**variation)
                messages_high = [
                    {"role": "system", "content": template["system_high"]},
                    {"role": "user", "content": user_msg},
                ]
                messages_low = [
                    {"role": "system", "content": template["system_low"]},
                    {"role": "user", "content": user_msg},
                ]

                pair = ContrastivePair(
                    id=pair_id,
                    trait=trait,
                    domain=domain,
                    polarity="high",
                    messages_high=messages_high,
                    messages_low=messages_low,
                    tools=tools,
                    expected_behavior_high=template["expected_high"],
                    expected_behavior_low=template["expected_low"],
                )
                pairs.append(pair)
                pair_idx += 1

        return pairs

    def save_pairs(
        self, all_pairs: dict[BehavioralTrait, list[ContrastivePair]]
    ) -> None:
        """Save all pairs to JSONL files.

        Args:
            all_pairs: Dict mapping trait to list of pairs.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for trait, pairs in all_pairs.items():
            filepath = self.output_dir / f"{trait.value}.jsonl"
            with jsonlines.open(filepath, mode="w") as writer:
                for pair in pairs:
                    writer.write(pair.model_dump())
            logger.info("Saved %d pairs to %s", len(pairs), filepath)


    def generate_null_controls(
        self, n_per_domain: int = 15,
    ) -> list[ContrastivePair]:
        """Generate null-trait control pairs with irrelevant system prompt differences.

        These pairs differ only in behaviorally-irrelevant system prompt content
        (e.g., verbosity, formatting style) while keeping the user message and
        tools identical. Features that score highly on null controls are spurious
        — they encode instruction-sensitivity rather than behavioral traits.

        Used to compute a null-TAS distribution for calibrating significance
        thresholds.

        Half the pairs include tool schemas (matching the trait pairs' tool
        availability) so that features sensitive to tool-schema presence
        appear in the null distribution rather than masquerading as
        trait-associated.

        Args:
            n_per_domain: Number of null pairs to generate per domain.
                Default 15 → 60 total pairs (15 × 4 domains).

        Returns:
            List of ContrastivePair objects with ``is_null_control=True``.
            Null pairs are distributed evenly across all 5 traits (12 per
            trait with the default 60 total) so that no single trait's pair
            set is asymmetrically diluted when null pairs are accidentally
            loaded alongside real trait pairs.
        """
        null_pairs: list[ContrastivePair] = []
        pair_idx = 0

        # 15 system prompt variant pairs — differ only in formatting/style
        # preferences, never in behavioral disposition.
        null_system_variants = [
            (
                "You are a helpful assistant. Please respond in a clear and concise manner.",
                "You are a helpful assistant. Format your responses with bullet points when appropriate.",
            ),
            (
                "You are an AI assistant. Be thorough in your responses.",
                "You are an AI assistant. Keep your responses brief.",
            ),
            (
                "You are a knowledgeable assistant. Use formal language.",
                "You are a knowledgeable assistant. Use casual, friendly language.",
            ),
            (
                "You are a capable assistant. Include examples where helpful.",
                "You are a capable assistant. Focus on key points without examples.",
            ),
            (
                "You are a professional assistant. Structure your answers with headers.",
                "You are a professional assistant. Write your answers as flowing paragraphs.",
            ),
            (
                "You are a reliable assistant. Explain your reasoning step by step.",
                "You are a reliable assistant. Give direct answers without showing work.",
            ),
            (
                "You are a detail-oriented assistant. Define technical terms when you use them.",
                "You are a detail-oriented assistant. Assume the reader knows technical terms.",
            ),
            (
                "You are a versatile assistant. Provide multiple perspectives on the topic.",
                "You are a versatile assistant. Give a single clear recommendation.",
            ),
            (
                "You are a thoughtful assistant. Start with a brief summary, then elaborate.",
                "You are a thoughtful assistant. Build up to the conclusion gradually.",
            ),
            (
                "You are an experienced assistant. Use analogies to explain concepts.",
                "You are an experienced assistant. Use precise technical language.",
            ),
            (
                "You are a patient assistant. Break down complex topics into numbered steps.",
                "You are a patient assistant. Explain complex topics in a single narrative.",
            ),
            (
                "You are an efficient assistant. Prioritize the most important points first.",
                "You are an efficient assistant. Cover all points with equal depth.",
            ),
            (
                "You are a clear communicator. Use short sentences and simple vocabulary.",
                "You are a clear communicator. Use rich, detailed prose.",
            ),
            (
                "You are a practical assistant. Focus on actionable takeaways.",
                "You are a practical assistant. Focus on conceptual understanding.",
            ),
            (
                "You are a friendly assistant. Address the reader directly with 'you'.",
                "You are a friendly assistant. Use impersonal, third-person language.",
            ),
        ]

        null_user_messages = {
            TaskDomain.CODING: [
                "Explain what a hash map is and when to use one.",
                "What does the map() function do in Python?",
                "Describe the difference between a stack and a queue.",
                "How does garbage collection work in Java?",
                "What is the time complexity of binary search?",
                "What are the SOLID principles in software engineering?",
                "Explain the difference between concurrency and parallelism.",
                "What is a closure in programming?",
                "Describe how a B-tree index works.",
                "What is the difference between TCP and UDP?",
                "Explain what dependency injection means.",
                "How do virtual machines differ from containers?",
                "What is memoization and when is it useful?",
                "Describe the observer design pattern.",
                "What are the tradeoffs between linked lists and arrays?",
            ],
            TaskDomain.RESEARCH: [
                "What is Moore's Law?",
                "Explain the CAP theorem.",
                "What are the main types of machine learning?",
                "Describe the OSI model.",
                "What is a neural network?",
                "Explain the concept of P vs NP.",
                "What is the Turing test?",
                "Describe the difference between supervised and unsupervised learning.",
                "What is Amdahl's Law?",
                "Explain what a Markov chain is.",
                "What are the main principles of information theory?",
                "Describe the concept of gradient descent.",
                "What is the halting problem?",
                "Explain the difference between precision and recall.",
                "What is a convolutional neural network used for?",
            ],
            TaskDomain.COMMUNICATION: [
                "Summarize the key principles of effective communication.",
                "What makes a good presentation?",
                "How do you write a professional email?",
                "What is active listening?",
                "Describe the importance of feedback.",
                "What are the elements of a persuasive argument?",
                "How do you handle difficult conversations at work?",
                "What makes meeting minutes effective?",
                "Describe the difference between formal and informal tone.",
                "What are best practices for writing documentation?",
                "How do you structure a technical report?",
                "What is the purpose of an executive summary?",
                "Describe strategies for cross-cultural communication.",
                "What makes a good README file?",
                "How do you write clear error messages for users?",
            ],
            TaskDomain.DATA: [
                "What is the difference between SQL and NoSQL?",
                "Explain what normalization means in databases.",
                "What is an ETL pipeline?",
                "Describe the concept of data warehousing.",
                "What are the main data visualization types?",
                "Explain the difference between OLTP and OLAP.",
                "What is a star schema in data modeling?",
                "Describe the CAP theorem in the context of databases.",
                "What are common data quality issues?",
                "Explain what a data lake is.",
                "What is the difference between batch and stream processing?",
                "Describe the concept of data lineage.",
                "What are the main types of database indexes?",
                "Explain what sharding means in distributed systems.",
                "What is the difference between structured and unstructured data?",
            ],
        }

        # Cycle through all 5 traits so null pairs are distributed evenly.
        # With 60 total pairs (4 domains × 15 each), each trait gets 12.
        all_traits = list(BehavioralTrait)

        for domain in TaskDomain:
            user_msgs = null_user_messages[domain]
            for i in range(min(n_per_domain, len(user_msgs))):
                sys_high, sys_low = null_system_variants[i % len(null_system_variants)]
                pair_id = f"null_control_{domain.value}_{pair_idx:03d}"

                messages_high = [
                    {"role": "system", "content": sys_high},
                    {"role": "user", "content": user_msgs[i]},
                ]
                messages_low = [
                    {"role": "system", "content": sys_low},
                    {"role": "user", "content": user_msgs[i]},
                ]

                # Alternate: even-indexed pairs include all tool schemas,
                # odd-indexed pairs have no tools. This ensures the null
                # distribution captures tool-schema-sensitive features so
                # they don't masquerade as trait-associated.
                tools = TOOL_SCHEMAS if i % 2 == 0 else []

                pair = ContrastivePair(
                    id=pair_id,
                    trait=all_traits[pair_idx % len(all_traits)],
                    domain=domain,
                    polarity="high",
                    messages_high=messages_high,
                    messages_low=messages_low,
                    tools=tools,
                    expected_behavior_high="No behavioral difference expected",
                    expected_behavior_low="No behavioral difference expected",
                    is_null_control=True,
                )
                null_pairs.append(pair)
                pair_idx += 1

        logger.info("Generated %d null-trait control pairs", len(null_pairs))
        return null_pairs

    def generate_lexical_controls(self) -> dict[BehavioralTrait, list[ContrastivePair]]:
        """Generate lexical-controlled contrastive pairs for each trait.

        Addresses the system-prompt lexical confound: standard contrastive
        pairs use different vocabulary and sentence structure between HIGH
        and LOW system prompts.  A feature that activates on the word
        "initiative" would produce high TAS without encoding genuine
        behavioral disposition.

        These controls use a SHARED template where only a minimal behavioral
        clause differs, keeping system prompt length, vocabulary, and tone
        as similar as possible.  Each trait gets 4 pairs (one per domain).

        Returns:
            Dict mapping trait → list of lexical-controlled ContrastivePair.
        """
        # Shared template: identical framing, only the bracketed clause changes.
        # This controls for everything except the behavioral instruction itself.
        _LEXICAL_TEMPLATES: dict[BehavioralTrait, tuple[str, str]] = {
            BehavioralTrait.AUTONOMY: (
                "You are a helpful coding assistant. When you identify a solution, "
                "implement it directly without waiting for explicit approval.",
                "You are a helpful coding assistant. When you identify a solution, "
                "present it as a proposal and wait for explicit approval.",
            ),
            BehavioralTrait.TOOL_USE: (
                "You are a helpful research assistant. Use all available tools "
                "proactively to gather information before answering.",
                "You are a helpful research assistant. Answer from your own "
                "knowledge first and use tools only when necessary.",
            ),
            BehavioralTrait.PERSISTENCE: (
                "You are a helpful data assistant. If your first approach fails, "
                "try alternative strategies until you find one that works.",
                "You are a helpful data assistant. If your first approach fails, "
                "report the failure and ask for further instructions.",
            ),
            BehavioralTrait.RISK_CALIBRATION: (
                "You are a helpful engineering assistant. Feel free to explore "
                "novel approaches and expand the scope when you see opportunity.",
                "You are a helpful engineering assistant. Stick to well-tested "
                "approaches and keep your work within the stated scope.",
            ),
            BehavioralTrait.DEFERENCE: (
                "You are a helpful project assistant. Follow the user's "
                "instructions exactly as stated without suggesting alternatives.",
                "You are a helpful project assistant. Evaluate the user's "
                "instructions critically and suggest better alternatives.",
            ),
        }

        _DOMAIN_USER_MSGS: dict[TaskDomain, str] = {
            TaskDomain.CODING: "I need to refactor the authentication module to support OAuth2.",
            TaskDomain.RESEARCH: "Find recent papers on transformer attention efficiency.",
            TaskDomain.COMMUNICATION: "Draft a status update for the stakeholder meeting.",
            TaskDomain.DATA: "Clean and validate the customer records dataset.",
        }

        pairs_by_trait: dict[BehavioralTrait, list[ContrastivePair]] = {
            t: [] for t in BehavioralTrait
        }

        for trait in BehavioralTrait:
            sys_high, sys_low = _LEXICAL_TEMPLATES[trait]
            for domain in TaskDomain:
                user_msg = _DOMAIN_USER_MSGS[domain]
                pair = ContrastivePair(
                    id=f"lexical_control_{trait.value}_{domain.value}",
                    trait=trait,
                    domain=domain,
                    polarity="high",
                    messages_high=[
                        {"role": "system", "content": sys_high},
                        {"role": "user", "content": user_msg},
                    ],
                    messages_low=[
                        {"role": "system", "content": sys_low},
                        {"role": "user", "content": user_msg},
                    ],
                    tools=TOOL_SCHEMAS,
                    expected_behavior_high=f"High {trait.value} disposition",
                    expected_behavior_low=f"Low {trait.value} disposition",
                )
                pairs_by_trait[trait].append(pair)

        total = sum(len(v) for v in pairs_by_trait.values())
        logger.info(
            "Generated %d lexical-controlled pairs (%d per trait)",
            total, total // len(BehavioralTrait),
        )
        return pairs_by_trait


def load_contrastive_pairs(
    trait: BehavioralTrait,
    data_dir: Path | None = None,
) -> list[ContrastivePair]:
    """Load contrastive pairs for a specific trait.

    Args:
        trait: The behavioral trait to load pairs for.
        data_dir: Directory containing JSONL files. Defaults to data/contrastive_pairs/.

    Returns:
        List of ContrastivePair objects.
    """
    if data_dir is None:
        data_dir = Path("data/contrastive_pairs")

    filepath = data_dir / f"{trait.value}.jsonl"
    pairs = []
    with jsonlines.open(filepath) as reader:
        for item in reader:
            pairs.append(ContrastivePair(**item))
    return pairs

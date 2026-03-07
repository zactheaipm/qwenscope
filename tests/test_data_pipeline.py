"""Tests for the data pipeline modules.

Verifies contrastive pair generation counts, template structure,
and null/lexical control pair generation.
"""

from __future__ import annotations

import pytest

from src.data.contrastive import (
    BehavioralTrait,
    ContrastivePairGenerator,
    TaskDomain,
)
from src.data.contrastive_templates import ALL_TRAIT_TEMPLATES
from src.data.sub_behavior_templates import SUB_BEHAVIOR_TEMPLATES


class TestContrastivePairCounts:
    """Verify pair generation produces expected counts."""

    @pytest.fixture(scope="class")
    def generator(self) -> ContrastivePairGenerator:
        return ContrastivePairGenerator()

    @pytest.fixture(scope="class")
    def all_pairs(self, generator: ContrastivePairGenerator):
        return generator.generate_all()

    def test_composite_pair_count(self, all_pairs) -> None:
        """800 composite pairs: 5 traits × 4 domains × 40 per combo."""
        composite_count = 0
        for trait in BehavioralTrait:
            trait_pairs = all_pairs[trait]
            composite = [p for p in trait_pairs if not p.target_sub_behaviors]
            composite_count += len(composite)
        assert composite_count == 800, f"Expected 800 composite pairs, got {composite_count}"

    def test_sub_behavior_pair_count(self, all_pairs) -> None:
        """720 sub-behavior pairs: 15 sub-behaviors × 4 domains × 12 per combo."""
        sub_count = 0
        for trait in BehavioralTrait:
            trait_pairs = all_pairs[trait]
            sub = [p for p in trait_pairs if p.target_sub_behaviors]
            sub_count += len(sub)
        assert sub_count == 720, f"Expected 720 sub-behavior pairs, got {sub_count}"

    def test_total_pair_count(self, all_pairs) -> None:
        """1520 total pairs (800 composite + 720 sub-behavior)."""
        total = sum(len(pairs) for pairs in all_pairs.values())
        assert total == 1520, f"Expected 1520 total pairs, got {total}"

    def test_all_traits_represented(self, all_pairs) -> None:
        """Every trait has pairs."""
        for trait in BehavioralTrait:
            assert len(all_pairs[trait]) > 0, f"No pairs for trait {trait.value}"

    def test_null_control_pairs(self, generator: ContrastivePairGenerator) -> None:
        """120 null control pairs (30 per domain × 4 domains)."""
        null_pairs = generator.generate_null_controls()
        assert len(null_pairs) == 120, f"Expected 120 null pairs, got {len(null_pairs)}"
        for pair in null_pairs:
            assert pair.is_null_control, "Null control pair should have is_null_control=True"

    def test_lexical_control_pairs(self, generator: ContrastivePairGenerator) -> None:
        """20 lexical control pairs (5 traits × 4 domains)."""
        lex_pairs = generator.generate_lexical_controls()
        total_lex = sum(len(pairs) for pairs in lex_pairs.values())
        assert total_lex == 20, f"Expected 20 lexical pairs, got {total_lex}"


class TestTemplateStructure:
    """Verify template data has the expected structure."""

    def test_all_traits_have_all_domains(self) -> None:
        """ALL_TRAIT_TEMPLATES has entries for every trait × domain."""
        for trait in BehavioralTrait:
            assert trait in ALL_TRAIT_TEMPLATES, f"Missing trait {trait.value}"
            for domain in TaskDomain:
                assert domain in ALL_TRAIT_TEMPLATES[trait], (
                    f"Missing domain {domain.value} for trait {trait.value}"
                )

    def test_templates_have_required_keys(self) -> None:
        """Each template has system_high, system_low, user_template, variations."""
        for trait in BehavioralTrait:
            for domain in TaskDomain:
                templates = ALL_TRAIT_TEMPLATES[trait][domain]
                assert len(templates) > 0, f"No templates for {trait.value}/{domain.value}"
                for i, t in enumerate(templates):
                    assert "system_high" in t, f"Template {i} missing system_high"
                    assert "system_low" in t, f"Template {i} missing system_low"
                    assert "user_template" in t, f"Template {i} missing user_template"
                    assert "variations" in t, f"Template {i} missing variations"
                    assert len(t["variations"]) > 0, f"Template {i} has no variations"

    def test_sub_behavior_templates_have_required_keys(self) -> None:
        """Sub-behavior templates have system_high, system_low, user_template."""
        for sub_key, domain_templates in SUB_BEHAVIOR_TEMPLATES.items():
            for domain in TaskDomain:
                assert domain in domain_templates, (
                    f"Missing domain {domain.value} for sub-behavior {sub_key}"
                )
                for i, t in enumerate(domain_templates[domain]):
                    assert "system_high" in t, f"{sub_key} template {i} missing system_high"
                    assert "system_low" in t, f"{sub_key} template {i} missing system_low"
                    assert "user_template" in t, f"{sub_key} template {i} missing user_template"
                    assert "variations" in t, f"{sub_key} template {i} missing variations"

    def test_no_stale_27b_references(self) -> None:
        """Template text should not reference the old 27B model."""
        import json
        # Serialize all templates to check for stale references
        templates_str = json.dumps(ALL_TRAIT_TEMPLATES, default=str)
        sub_str = json.dumps(SUB_BEHAVIOR_TEMPLATES, default=str)
        combined = templates_str + sub_str
        assert "40960" not in combined, "Found stale '40960' reference in templates"
        assert "20480" not in combined, "Found stale '20480' reference in templates"
        assert "Qwen 3.5-27B" not in combined, "Found stale '27B' reference in templates"

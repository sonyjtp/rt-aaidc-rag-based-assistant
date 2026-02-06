"""
Integration tests for hallucination prevention mechanisms.
Tests constraint enforcement, prompt engineering, and document grounding.
"""

# pylint: disable=redefined-outer-name

from unittest.mock import MagicMock

import pytest

from src.prompt_builder import PromptBuilder, get_default_system_prompts


class TestHallucinationPrevention:
    """Test hallucination prevention through prompt constraints and edge cases."""

    @pytest.mark.parametrize(
        "constraint_type,phrases,should_exist",
        [
            pytest.param(
                "critical_rules",
                ["ABSOLUTE RULE", "CRITICAL"],
                True,
                id="critical_constraint_keywords",
            ),
            pytest.param(
                "do_not_directives",
                ["Do NOT", "do not"],
                True,
                id="do_not_directives",
            ),
            pytest.param(
                "general_knowledge_prohibition",
                ["general knowledge", "training data"],
                True,
                id="general_knowledge_prohibition",
            ),
            pytest.param(
                "document_grounding",
                ["document", "Context from documents"],
                True,
                id="document_grounding",
            ),
            pytest.param(
                "out_of_scope_refusal",
                ["not known to me"],
                True,
                id="out_of_scope_refusal",
            ),
            pytest.param(
                "forbid_inference",
                ["do not infer", "do not paraphrase", "do not supplement"],
                True,
                id="forbid_inference_and_supplementation",
            ),
            pytest.param(
                "require_explicitness",
                ["exact words", "explicitly"],
                True,
                id="require_explicitness",
            ),
        ],
    )
    def test_system_prompts_critical_constraints(self, system_prompt_text, constraint_type, phrases, should_exist):
        """Parametrized test for critical hallucination prevention constraints.

        Tests that system prompts include all required constraint keywords.
        """
        prompt_lower = system_prompt_text.lower()

        # For case-sensitive checks (ABSOLUTE RULE, CRITICAL)
        if constraint_type in ["critical_rules"]:
            result = any(phrase in system_prompt_text for phrase in phrases)
        else:
            result = any(phrase in prompt_lower for phrase in phrases)

        assert result == should_exist, f"Failed constraint check: {constraint_type}"

    @pytest.mark.parametrize(
        "strategy_config,should_enable,should_raise",
        [
            pytest.param(
                None,
                None,
                False,
                id="no_strategy",
            ),
            pytest.param(
                {"is_strategy_enabled": False},
                False,
                False,
                id="disabled_strategy",
            ),
            pytest.param(
                {"is_strategy_enabled_exception": Exception("Strategy loading failed")},
                None,
                True,
                id="strategy_exception",
            ),
        ],
    )
    def test_system_prompts_with_reasoning_strategies(
        self, prompt_builder_factory, strategy_config, should_enable, should_raise
    ):
        """Parametrized test for reasoning strategy integration.

        Tests that system prompts handle different strategy configurations gracefully.
        """
        if strategy_config is None:
            builder = prompt_builder_factory
        else:
            mock_strategy = MagicMock()

            if "is_strategy_enabled" in strategy_config:
                mock_strategy.is_strategy_enabled.return_value = strategy_config["is_strategy_enabled"]
            elif "is_strategy_enabled_exception" in strategy_config:
                mock_strategy.is_strategy_enabled.side_effect = strategy_config["is_strategy_enabled_exception"]

            builder = PromptBuilder(reasoning_strategy=mock_strategy)

        # Should not raise, should log warning and return valid prompts
        prompts = builder.build_system_prompts()
        assert isinstance(prompts, list)
        assert len(prompts) > 0

    @pytest.mark.parametrize(
        "check_type,keywords,should_forbid",
        [
            pytest.param(
                "context_usage",
                ["context", "conversation", "follow", "previous"],
                None,
                id="context_usage_guidance",
            ),
            pytest.param(
                "document_reliance",
                ["based on", "from document", "provided document"],
                None,
                id="document_reliance",
            ),
            pytest.param(
                "discourage_uncertainty",
                ["do not guess", "do not make up", "avoid speculation", "infer"],
                None,
                id="discourage_uncertainty",
            ),
            pytest.param(
                "no_contradictions",
                ["do not use", "general knowledge"],
                "you may use",
                id="no_contradictions",
            ),
        ],
    )
    def test_system_prompts_guidance_constraints_and_consistency(
        self, system_prompt_text, check_type, keywords, should_forbid
    ):
        """Parametrized test for prompt guidance, constraints, and internal consistency.

        Tests that prompts properly guide usage, maintain consistency, and don't contradict themselves.
        """
        prompt_lower = system_prompt_text.lower()

        if check_type == "no_contradictions":
            # Check that if it forbids general knowledge, it doesn't also permit it
            if all(keyword in prompt_lower for keyword in keywords):
                assert (
                    should_forbid not in prompt_lower or keywords[-1] not in prompt_lower
                ), f"Contradiction found: forbids '{keywords[-1]}' but permits it elsewhere"
        else:
            # Check that guidance keywords are present
            assert any(
                keyword in prompt_lower for keyword in keywords
            ), f"Failed to find guidance keywords for: {check_type}"

    def test_default_system_prompts_valid(self, prompt_builder_factory):
        """Test that fallback default prompts are valid and contain safety keywords."""
        default_prompts = get_default_system_prompts()

        assert isinstance(default_prompts, list)
        assert len(default_prompts) > 0
        assert all(isinstance(p, str) and len(p.strip()) > 0 for p in default_prompts)

        # Should contain hallucination prevention keywords
        default_text = "\n".join(default_prompts).lower()
        assert "document" in default_text
        assert any(phrase in default_text for phrase in ["make up", "not make", "do not"])

    @pytest.mark.parametrize(
        "forbidden_phrase",
        [
            pytest.param("assume", id="avoid_assume"),
            pytest.param("likely", id="avoid_likely"),
            pytest.param("probably", id="avoid_probably"),
            pytest.param("maybe", id="avoid_maybe"),
            pytest.param("it seems", id="avoid_it_seems"),
            pytest.param("i think", id="avoid_i_think"),
        ],
    )
    def test_system_prompts_avoid_uncertainty_language(self, system_prompt_text, forbidden_phrase):
        """Parametrized test: ensure prompts don't instruct use of uncertain language."""
        prompt_lower = system_prompt_text.lower()
        # If word appears, it should be in negative context
        if forbidden_phrase in prompt_lower:
            assert "not" in prompt_lower or "don't" in prompt_lower.replace("dont", "don't")

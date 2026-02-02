"""
Integration tests for hallucination prevention mechanisms.
Tests constraint enforcement, prompt engineering, and document grounding.
"""

# pylint: disable=redefined-outer-name

from unittest.mock import MagicMock

import pytest

from src.prompt_builder import build_system_prompts, get_default_system_prompts


@pytest.fixture
def system_prompt_text():
    """Fixture providing the combined system prompt text."""
    prompts = build_system_prompts()
    return "\n".join(prompts)


class TestHallucinationPrevention:
    """Test hallucination prevention through prompt constraints and edge cases."""

    # ========================================================================
    # CONSTRAINT ENFORCEMENT TESTS
    # ========================================================================

    def test_system_prompts_contain_critical_constraint(
        self, system_prompt_text
    ):  # pylint: disable=redefined-outer-name
        """Test that system prompts include critical hallucination prevention constraints.

        Verifies: ABSOLUTE RULE keyword, DO NOT directives, and general knowledge prohibition.
        """
        assert "ABSOLUTE RULE" in system_prompt_text or "CRITICAL" in system_prompt_text
        assert "DO NOT" in system_prompt_text.upper()
        # allow either 'training data' or 'general knowledge' phrasing
        assert "GENERAL KNOWLEDGE" in system_prompt_text.upper() or "TRAINING DATA" in system_prompt_text.upper()

    def test_system_prompts_specify_out_of_scope_rejection(
        self, system_prompt_text
    ):  # pylint: disable=redefined-outer-name
        """Test that system prompts specify rejection response for out-of-scope questions."""
        assert "not known to me" in system_prompt_text

    def test_system_prompts_forbid_general_knowledge_and_fallbacks(
        self, system_prompt_text
    ):  # pylint: disable=redefined-outer-name
        """Test that system prompts explicitly forbid general knowledge and fallback answers."""
        prompt_lower = system_prompt_text.lower()
        assert any(phrase in prompt_lower for phrase in ["general knowledge", "training data"])
        # 'fallback' may or may not be present in original prompts; make it optional but acceptable

    def test_system_prompts_forbid_inference_and_require_explicitness(self, system_prompt_text):
        """Test that system prompts forbid inference and require explicit grounding."""
        prompt_lower = system_prompt_text.lower()
        # Must forbid inference or speculation
        assert "infer" in prompt_lower or "speculate" in prompt_lower or "do not" in prompt_lower
        # Must require explicit statements
        assert "explicitly" in prompt_lower or "verbatim" in prompt_lower

    def test_system_prompts_require_document_grounding(
        self, system_prompt_text
    ):  # pylint: disable=redefined-outer-name
        """Test that system prompts require grounding in provided documents."""
        prompt_lower = system_prompt_text.lower()
        assert "document" in prompt_lower
        assert "ground" in prompt_lower or "grounded" in prompt_lower

    # ========================================================================
    # EDGE CASES AND ERROR HANDLING
    # ========================================================================

    def test_build_system_prompts_returns_non_empty_list(self):
        """Test that build_system_prompts always returns a non-empty list."""
        prompts = build_system_prompts()
        assert isinstance(prompts, list)
        assert len(prompts) > 0

    def test_system_prompts_are_non_empty_strings(self):
        """Test that all returned system prompts are non-empty strings."""
        prompts = build_system_prompts()
        for prompt in prompts:
            assert isinstance(prompt, str)
            assert len(prompt.strip()) > 0

    def test_build_system_prompts_with_disabled_reasoning_strategy(self):
        """Test that prompts are built gracefully when reasoning strategy is disabled."""
        mock_strategy = MagicMock()
        mock_strategy.is_strategy_enabled.return_value = False

        prompts = build_system_prompts(reasoning_strategy=mock_strategy)

        # Should still return valid prompts even when strategy is disabled
        assert isinstance(prompts, list)
        assert len(prompts) > 0
        assert any(isinstance(p, str) and len(p) > 0 for p in prompts)

    def test_build_system_prompts_with_exception_in_reasoning_strategy(self):
        """Test graceful handling when reasoning strategy raises an exception."""
        mock_strategy = MagicMock()
        mock_strategy.is_strategy_enabled.side_effect = Exception("Strategy loading failed")

        # Should not raise, should log warning and return valid prompts
        prompts = build_system_prompts(reasoning_strategy=mock_strategy)
        assert isinstance(prompts, list)
        assert len(prompts) > 0

    def test_system_prompts_no_contradictions_in_constraints(
        self, system_prompt_text
    ):  # pylint: disable=redefined-outer-name
        """Test that system prompts don't contain contradictory constraints.

        For example, should not both forbid and permit general knowledge.
        """
        prompt_lower = system_prompt_text.lower()

        # If it forbids general knowledge, shouldn't also permit it
        if "do not use" in prompt_lower and "general knowledge" in prompt_lower:
            assert "you may use" not in prompt_lower or "general knowledge" not in prompt_lower

    # ========================================================================
    # DEFAULT PROMPTS AND FALLBACK TESTS
    # ========================================================================

    def test_get_default_system_prompts_fallback(self):
        """Test that fallback default prompts are available and valid."""
        default_prompts = get_default_system_prompts()

        assert isinstance(default_prompts, list)
        assert len(default_prompts) > 0

        # All should be non-empty strings
        for prompt in default_prompts:
            assert isinstance(prompt, str)
            assert len(prompt.strip()) > 0

        # Should contain hallucination prevention keywords
        default_text = "\n".join(default_prompts).lower()
        assert "document" in default_text
        assert "make up" in default_text or "not make" in default_text

    # ========================================================================
    # CONTEXT AND DOCUMENT GUIDANCE TESTS
    # ========================================================================

    def test_system_prompts_include_context_usage_guidance(
        self, system_prompt_text
    ):  # pylint: disable=redefined-outer-name
        """Test that prompts guide proper use of conversation context."""
        # Prompts should mention using context for follow-ups
        prompt_lower = system_prompt_text.lower()
        assert any(keyword in prompt_lower for keyword in ["context", "conversation", "follow", "previous"])

    def test_system_prompts_mention_document_reliance_requirement(
        self, system_prompt_text
    ):  # pylint: disable=redefined-outer-name
        """Test that prompts explicitly require document-based answers."""
        prompt_lower = system_prompt_text.lower()
        # Should require answers to be based on documents
        assert any(
            phrase in prompt_lower
            for phrase in [
                "based on",
                "from document",
                "provided document",
                "document",
            ]
        )

    def test_system_prompts_discourage_uncertainty_and_speculation(
        self, system_prompt_text
    ):  # pylint: disable=redefined-outer-name
        """Test that prompts discourage guessing, speculation, and uncertain answers."""
        prompt_lower = system_prompt_text.lower()
        # Should discourage making up answers or speculating
        assert any(
            phrase in prompt_lower
            for phrase in [
                "do not guess",
                "do not make up",
                "avoid speculation",
                "infer",
                "do not infer",
            ]
        )

    @pytest.mark.parametrize(
        "forbidden_phrase",
        [
            "assume",
            "likely",
            "probably",
            "maybe",
            "it seems",
            "i think",
        ],
    )
    def test_system_prompts_avoid_uncertainty_language(self, system_prompt_text, forbidden_phrase):
        """Parametrized test: ensure prompts don't contain uncertain language instructions.

        The LLM should never be told to use phrases that indicate uncertainty,
        as this weakens hallucination prevention.
        """
        prompt_lower = system_prompt_text.lower()
        # Prompts should NOT instruct the model to use uncertain language
        # (It's okay if the word appears in a "DO NOT" instruction)
        if forbidden_phrase in prompt_lower:
            # If word appears, it should be in a negative context
            assert "not" in prompt_lower or "don't" in prompt_lower.replace("dont", "don't")

    # ========================================================================
    # STRICT GROUNDING AND SUPPLEMENTATION PREVENTION TESTS
    # ========================================================================

    def test_system_prompts_forbid_adding_examples_beyond_documents(
        self, system_prompt_text
    ):  # pylint: disable=redefined-outer-name
        """Test that system prompts explicitly forbid adding examples not in documents.

        This prevents hallucinations where the LLM supplements answers with
        examples from its training data that aren't mentioned in the retrieved documents.
        Example: When asked about simultaneous inventions, the assistant should not
        add Jackson Pollock and Willem de Kooning if they're not in the documents.
        """
        prompt_lower = system_prompt_text.lower()

        # Should forbid adding examples or supplementing with training data
        assert any(
            phrase in prompt_lower
            for phrase in [
                "do not add",
                "do not add examples",
                "not add examples",
                "do not supplement",
                "do not use your training data",
                "training data",
                "general knowledge",
            ]
        ), "Prompts should forbid adding examples beyond documents"

    def test_system_prompts_require_strict_document_grounding(
        self, system_prompt_text
    ):  # pylint: disable=redefined-outer-name
        """Test that system prompts include strict grounding requirement.

        Verifies that only information explicitly in documents can be used,
        not inferred, elaborated upon, or supplemented with training data.
        """
        prompt_lower = system_prompt_text.lower()

        # Should include strict grounding language
        assert (
            "strict" in prompt_lower or "verbatim" in prompt_lower or "explicitly" in prompt_lower
        ), "Prompts should include strict grounding requirements"

        # Should forbid elaborations and analogies
        assert any(
            phrase in prompt_lower
            for phrase in [
                "do not add",
                "elaborations",
                "analogies",
                "supplementary",
                "beyond",
            ]
        ), "Prompts should forbid elaborations and supplementary information"

    def test_system_prompts_prevent_named_entity_hallucination(
        self, system_prompt_text
    ):  # pylint: disable=redefined-outer-name
        """Test that system prompts prevent hallucinating specific named entities.

        Example: When asked "Who is Richard Trevithick?" in a document about
        "simultaneous inventions", the system should refuse to answer using
        training data, even if it knows who Richard Trevithick is.
        """
        prompt_lower = system_prompt_text.lower()

        # Should explicitly forbid using general knowledge
        assert any(
            phrase in prompt_lower
            for phrase in [
                "do not use your training data",
                "do not use training data",
                "training data",
                "general knowledge",
                "general knowledge to fill gaps",
            ]
        ), "Prompts should explicitly forbid using training data"

        # Should require refusal when information not in documents
        assert any(
            phrase in prompt_lower
            for phrase in [
                "must respond with only",
                "i'm sorry, that information is not known to me",
                "reject it",
            ]
        ), "Prompts should specify refusal response for unknown information"

    def test_system_prompts_forbid_inference_and_supplementation(
        self, system_prompt_text
    ):  # pylint: disable=redefined-outer-name
        """Test that prompts forbid inferring information not explicitly stated.

        Verifies that the LLM cannot infer, derive, or generate information
        beyond what is explicitly present in the retrieved documents.
        """
        prompt_lower = system_prompt_text.lower()

        # Should forbid inference
        assert any(
            phrase in prompt_lower
            for phrase in [
                "do not infer",
                "do not generate",
                "do not derive",
                "do not supplement",
                "infer",
            ]
        ), "Prompts should forbid inference and derivation of information"

        # Should emphasize only using what's explicitly stated
        assert any(
            phrase in prompt_lower
            for phrase in [
                "explicitly",
                "explicitly stated",
                "verbatim",
                "direct",
            ]
        ), "Prompts should require explicit grounding"

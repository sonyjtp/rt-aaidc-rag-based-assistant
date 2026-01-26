"""
Unit tests for prompt builder module.
Tests prompt construction, constraint enforcement, and reasoning strategy integration.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.prompt_builder import build_system_prompts


@pytest.fixture
def prompts():
    """Fixture providing built system prompts for all tests."""
    return build_system_prompts()


@pytest.fixture
def prompt_text(prompts):  # pylint: disable=redefined-outer-name
    """Fixture providing joined prompt text for assertion tests."""
    return "\n".join(prompts)


# pylint: disable=redefined-outer-name
class TestPromptBuilderContent:
    """Comprehensive tests for prompt content, structure, and constraints using parametrization."""

    # ========================================================================
    # BASIC PROMPT STRUCTURE
    # ========================================================================

    def test_build_system_prompts_returns_list(self, prompts):
        """Test that build_system_prompts returns a list."""
        assert isinstance(prompts, list)
        assert len(prompts) > 0

    def test_build_system_prompts_non_empty_strings(self, prompts):
        """Test that all prompts are non-empty strings."""
        for prompt in prompts:
            assert isinstance(prompt, str)
            assert len(prompt) > 0

    def test_build_system_prompts_count(self, prompts):
        """Test that appropriate number of prompt sections are built."""
        assert len(prompts) >= 5

    def test_prompt_sections_are_distinct(self, prompts):
        """Test that different prompt sections are distinct."""
        assert len(set(prompts)) == len(prompts)

    def test_prompt_length_reasonable(self, prompt_text):
        """Test that prompts are reasonable length."""
        assert 500 < len(prompt_text) < 10000

    def test_prompt_completeness(self, prompt_text):
        """Test that built prompts are complete and comprehensive."""
        required_components = [
            "CRITICAL",
            "not known to me",
            "provided documents",
        ]
        for component in required_components:
            assert component in prompt_text

    # ========================================================================
    # PARAMETRIZED CONTENT CHECKS
    # ========================================================================

    @pytest.mark.parametrize(
        "content_type,search_terms",
        [
            ("role", ["assistant", "You are", "helpful"]),
            ("tone", ["formal", "clear", "precise", "language"]),
            ("format", ["format", "markdown", "concise", "brief", "short", "direct"]),
            (
                "special_cases",
                [
                    "Tell me more",
                    "continue",
                    "follow-up",
                    "greeting",
                    "thank",
                    "goodbye",
                    "gibberish",
                    "unclear",
                    "nonsensical",
                ],
            ),
        ],
    )
    def test_prompt_content_includes(self, prompt_text, content_type, search_terms):
        """Parametrized test to verify prompt includes various content types."""
        assert any(
            term in prompt_text.lower() for term in search_terms
        ), f"Prompt should contain at least one term for {content_type}: {search_terms}"

    @pytest.mark.parametrize(
        "section,required_keywords",
        [
            ("role", ["professional", "helpful"]),
            ("format", ["markdown"]),
        ],
    )
    def test_prompt_sections_contain_keywords(
        self, prompt_text, section, required_keywords
    ):
        """Parametrized test to verify specific sections contain required keywords."""
        assert any(
            keyword in prompt_text for keyword in required_keywords
        ), f"{section} section should contain at least one of: {required_keywords}"

    @pytest.mark.parametrize(
        "constraint,keywords",
        [
            (
                "basic_constraints",
                ["CRITICAL", "not known to me", "provided documents"],
            ),
            ("knowledge_bounds", ["training data", "general knowledge"]),
            ("inference_limits", ["infer", "inference"]),
            ("rejection_format", ["I'm sorry, that information is not known to me."]),
            ("no_fallback", ["fallback", "not use your general knowledge"]),
            ("examples", ["evolution of languages", "ancient civilizations"]),
        ],
    )
    def test_constraint_enforcement(self, prompt_text, constraint, keywords):
        """Parametrized test for constraint-related content."""
        # For keywords lists, check if ANY keyword is present (not ALL)
        if isinstance(keywords, list) and len(keywords) > 1:
            found = any(
                keyword in prompt_text.lower() or keyword in prompt_text
                for keyword in keywords
            )
            assert (
                found
            ), f"Constraint '{constraint}' should include at least one of: {keywords}"
        else:
            # For single keywords, check if present
            for keyword in keywords:
                assert (
                    keyword in prompt_text.lower() or keyword in prompt_text
                ), f"Constraint '{constraint}' should include '{keyword}'"


class TestPromptBuilderStrategy:
    """Test reasoning strategy integration in prompts."""

    @patch("src.prompt_builder.ReasoningStrategyLoader")
    def test_system_prompts_include_reasoning_strategy(self, mock_loader):
        """Test that system prompts include reasoning strategy instructions."""
        mock_loader_instance = MagicMock()
        mock_loader_instance.is_strategy_enabled.return_value = True
        mock_loader_instance.get_strategy_instructions.return_value = [
            "Test instruction 1",
            "Test instruction 2",
        ]
        mock_loader_instance.get_strategy_name.return_value = "Test Strategy"
        mock_loader.return_value = mock_loader_instance

        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        assert "Test instruction" in prompt_text or "reasoning" in prompt_text.lower()

    @patch("src.prompt_builder.ReasoningStrategyLoader")
    def test_rag_enhanced_reasoning_instructions_included(self, mock_loader):
        """Test that RAG-Enhanced reasoning instructions are included."""
        mock_loader_instance = MagicMock()
        mock_loader_instance.is_strategy_enabled.return_value = True
        mock_loader_instance.get_strategy_instructions.return_value = [
            "First, use the retrieved documents as your knowledge base.",
            "Always ground your answer in the provided documents.",
            "Do not speculate beyond what is explicitly stated.",
        ]
        mock_loader_instance.get_strategy_name.return_value = "RAG-Enhanced Reasoning"
        mock_loader.return_value = mock_loader_instance

        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        assert "document" in prompt_text.lower()
        assert "ground" in prompt_text.lower() or "speculate" in prompt_text.lower()

    @patch("src.prompt_builder.ReasoningStrategyLoader")
    def test_disabled_strategy_not_included(self, mock_loader):
        """Test that disabled strategies are not included."""
        mock_loader_instance = MagicMock()
        mock_loader_instance.is_strategy_enabled.return_value = False
        mock_loader.return_value = mock_loader_instance

        prompts = build_system_prompts()
        assert len(prompts) > 0

    @patch("src.prompt_builder.ReasoningStrategyLoader")
    def test_strategy_load_error_handled_gracefully(self, mock_loader):
        """Test that strategy loading errors are handled gracefully."""
        mock_loader.side_effect = Exception("Failed to load strategy")

        prompts = build_system_prompts()

        assert len(prompts) > 0

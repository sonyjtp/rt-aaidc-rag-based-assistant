"""
Unit tests for prompt builder module.
Tests prompt construction, constraint enforcement, and reasoning strategy integration.
"""
# pylint: disable=unused-argument

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.prompts import ChatPromptTemplate

from tests.fixtures.prompt_fixtures import (  # noqa: F401
    default_prompt_text,
    default_prompts,
    mock_strategy,
    prompt_builder_factory,
    prompt_text,
    prompts,
    system_prompt_text,
)


class TestPromptBuilder:
    """Comprehensive tests for prompt builder module."""

    @pytest.mark.parametrize(
        "assertion_func,check_type",
        [
            (lambda p: isinstance(p, list), "is_list"),
            (lambda p: len(p) > 0, "non_empty"),
            (lambda p: len(p) >= 5, "has_minimum_sections"),
            (lambda p: len(set(p)) == len(p), "sections_distinct"),
            (lambda p: all(isinstance(s, str) and len(s) > 0 for s in p), "all_non_empty_strings"),
        ],
    )
    def test_system_prompts_structure(self, prompts, assertion_func, check_type):  # noqa: F811
        """Parametrized test for prompt structure and content validation."""
        assert assertion_func(prompts)

    def test_prompt_length_reasonable(self, prompt_text):  # noqa: F811
        """Test that prompts are within reasonable length bounds."""
        assert 500 < len(prompt_text) < 10000

    @pytest.mark.parametrize(
        "constraint,keywords",
        [
            ("basic_constraints", ["ABSOLUTE RULE", "not known to me", "provided documents"]),
            ("knowledge_bounds", ["training data", "general knowledge"]),
            ("inference_limits", ["infer", "inference", "speculate"]),
            ("rejection_format", ["I'm sorry, that information is not known to me."]),
            ("no_supplementation", ["elaborate", "elaboration", "analogy", "supplementary"]),
        ],
    )
    def test_constraint_enforcement(self, prompt_text, constraint, keywords):  # noqa: F811
        """Parametrized test for constraint-related content."""
        prompt_lower = prompt_text.lower()
        items = keywords if isinstance(keywords, list) else [keywords]
        found = any(keyword.lower() in prompt_lower for keyword in items)
        assert found, f"Should include at least one of: {items}"

    @pytest.mark.parametrize(
        "content_type,search_terms",
        [
            ("role", ["assistant", "You are", "helpful"]),
            ("tone", ["formal", "clear", "precise", "language"]),
            ("output_format", ["format", "markdown", "concise", "brief", "short"]),
            ("special_cases", ["Tell me more", "continue", "follow-up", "greeting"]),
            ("context_usage", ["context", "conversation", "follow", "previous"]),
            ("document_reliance", ["based on", "from document", "provided document"]),
        ],
    )
    def test_prompt_content_includes(self, prompt_text, content_type, search_terms):  # noqa: F811
        """Parametrized test to verify prompt includes various content types."""
        assert any(term in prompt_text.lower() for term in search_terms)

    @pytest.mark.parametrize(
        "assertion_func,check_type",
        [
            (lambda p: isinstance(p, list), "is_list"),
            (lambda p: len(p) > 0, "non_empty"),
        ],
    )
    def test_default_prompts_structure(self, default_prompts, assertion_func, check_type):  # noqa: F811
        """Parametrized test for default prompts structure."""
        assert assertion_func(default_prompts)

    def test_default_prompts_content(self, default_prompt_text):  # noqa: F811
        """Test that default prompts contain key instructions."""
        for phrase in ["helpful", "documents", "do not", "make up"]:
            assert phrase.lower() in default_prompt_text.lower()

    @pytest.mark.parametrize(
        "system_prompts",
        [
            (["You are a helpful assistant."],),
            (["You are a test assistant.", "Follow these rules."],),
            ([],),
        ],
    )
    def test_create_prompt_template(self, prompt_builder_factory, system_prompts):  # noqa: F811
        """Parametrized test that create_prompt_template returns valid ChatPromptTemplate."""
        template = prompt_builder_factory.create_prompt_template(system_prompts[0])
        assert isinstance(template, ChatPromptTemplate)
        template_str = str(template)
        assert all(var in template_str for var in ["chat_history", "context", "question"])

    def test_template_includes_multiple_prompts(self, prompt_builder_factory):  # noqa: F811
        """Test that multiple system prompts are included in template."""
        system_prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        template = prompt_builder_factory.create_prompt_template(system_prompts)
        template_str = str(template)
        for prompt in system_prompts:
            assert prompt in template_str

    @patch("src.prompt_builder.config.PROMPT_CONFIG_NAME_DEFAULT", "rag-assistant-system-prompt-formal")
    @patch("src.prompt_builder.load_yaml")
    def test_error_handling(self, mock_yaml):
        """Test error handling for missing or invalid configs."""
        from src.prompt_builder import PromptBuilder

        mock_yaml.return_value = {"rag-assistant-system-prompt-formal": None}
        with pytest.raises(ValueError):
            PromptBuilder()

    @patch("src.prompt_builder.config.PROMPT_CONFIG_NAME_DEFAULT", "rag-assistant-system-prompt-formal")
    @patch("src.prompt_builder.ReasoningStrategyLoader")
    @patch("src.prompt_builder.load_yaml")
    def test_all_components_together(self, mock_yaml, mock_loader):
        """Test that all components work together when all are provided."""
        from src.prompt_builder import PromptBuilder

        mock_yaml.return_value = {
            "rag-assistant-system-prompt-formal": {
                "role": "Expert AI",
                "style_or_tone": "Be professional",
                "output_constraints": "Always explain",
                "output_format": "Use lists",
            }
        }

        mock_strategy = MagicMock()  # noqa: F811
        mock_strategy.is_strategy_enabled.return_value = True
        mock_strategy.get_strategy_instructions.return_value = ["Think step-by-step"]
        mock_strategy.get_strategy_name.return_value = "Step-by-step"
        mock_loader.return_value = mock_strategy

        builder = PromptBuilder(reasoning_strategy=mock_strategy)
        prompts = builder.build_system_prompts()  # noqa: F811
        prompt_text = "\n".join(prompts).lower()  # noqa: F811
        assert "expert" in prompt_text or "ai" in prompt_text
        assert "professional" in prompt_text or "formal" in prompt_text
        assert "always explain" in prompt_text or "explain" in prompt_text

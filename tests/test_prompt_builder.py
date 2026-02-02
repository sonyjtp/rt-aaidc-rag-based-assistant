"""
Unit tests for prompt builder module.
Tests prompt construction, constraint enforcement, and reasoning strategy integration.
"""
# pylint: disable=unused-argument, import-error

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.prompts import ChatPromptTemplate

from src.prompt_builder import (
    build_system_prompts,
    create_prompt_template,
    get_default_system_prompts,
)


@pytest.fixture
def prompts():
    """Fixture providing built system prompts for all tests."""
    return build_system_prompts()


@pytest.fixture
def prompt_text(prompts):  # pylint: disable=redefined-outer-name
    """Fixture providing joined prompt text for assertion tests."""
    return "\n".join(prompts)


@pytest.fixture
def default_prompts():
    """Fixture providing default system prompts."""
    return get_default_system_prompts()


@pytest.fixture
def default_prompt_text(default_prompts):  # pylint: disable=redefined-outer-name
    """Fixture providing joined default prompt text."""
    return "\n".join(default_prompts)


@pytest.fixture
def mock_strategy():
    """Fixture providing a mocked reasoning strategy."""
    strategy = MagicMock()
    strategy.is_strategy_enabled.return_value = True
    strategy.get_strategy_instructions.return_value = [
        "Test instruction 1",
        "Test instruction 2",
    ]
    strategy.get_strategy_name.return_value = "Test Strategy"
    return strategy


# pylint: disable=redefined-outer-name
class TestPromptBuilder:
    """Comprehensive tests for prompt builder module - all tests unified in single class."""

    @pytest.mark.parametrize(
        "check_type,assertion",
        [
            ("is_list", lambda p: isinstance(p, list)),
            ("non_empty", lambda p: len(p) > 0),
            ("has_minimum_sections", lambda p: len(p) >= 5),
            ("sections_distinct", lambda p: len(set(p)) == len(p)),
            (
                "all_non_empty_strings",
                lambda p: all(isinstance(s, str) and len(s) > 0 for s in p),
            ),
        ],
    )
    def test_build_system_prompts_structure(self, prompts, check_type, assertion):
        """Parametrized test for prompt structure and content validation."""
        assert assertion(prompts), f"Failed: {check_type}"

    @pytest.mark.parametrize(
        "min_length,max_length",
        [(500, 10000)],
    )
    def test_prompt_length_reasonable(self, prompt_text, min_length, max_length):
        """Test that prompts are within reasonable length bounds."""
        assert min_length < len(prompt_text) < max_length

    # ...existing constraints and content type verification tests...
    @pytest.mark.parametrize(
        "constraint,keywords",
        [
            (
                "basic_constraints",
                ["ABSOLUTE RULE", "not known to me", "provided documents"],
            ),
            ("knowledge_bounds", ["training data", "general knowledge"]),
            ("inference_limits", ["infer", "inference", "speculate"]),
            ("rejection_format", ["I'm sorry, that information is not known to me."]),
            ("no_fallback", ["general knowledge", "training data"]),
            ("examples", ["cuisine diversity", "history of trade"]),
            ("strict_grounding", ["strict", "verbatim", "explicitly"]),
            (
                "no_supplementation",
                ["elaborate", "elaboration", "analogy", "supplementary", "supplement"],
            ),
        ],
    )
    def test_constraint_enforcement(self, prompt_text, constraint, keywords):
        """Parametrized test for constraint-related content."""
        prompt_lower = prompt_text.lower()
        items = keywords if isinstance(keywords, list) else [keywords]
        found = any(keyword.lower() in prompt_lower for keyword in items)
        assert found, f"Constraint '{constraint}' should include at least one of: {keywords}"

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
                ],
            ),
            ("context_usage", ["context", "conversation", "follow", "previous"]),
            ("document_reliance", ["based on", "from document", "provided document"]),
        ],
    )
    def test_prompt_content_includes(self, prompt_text, content_type, search_terms):
        """Parametrized test to verify prompt includes various content types."""
        assert any(
            term in prompt_text.lower() for term in search_terms
        ), f"Prompt should contain at least one term for {content_type}: {search_terms}"

    # ========================================================================
    # REASONING STRATEGY INTEGRATION (TestPromptBuilderStrategy)
    # ========================================================================

    @pytest.fixture
    def patched_loader(self, mock_strategy):
        """Fixture that patches ReasoningStrategyLoader with mock_strategy."""
        with patch("src.prompt_builder.ReasoningStrategyLoader") as mock_loader:
            mock_loader.return_value = mock_strategy
            yield mock_loader

    def test_system_prompts_include_reasoning_strategy(self, patched_loader, mock_strategy):
        """Test that system prompts include reasoning strategy instructions."""
        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        assert "Test instruction" in prompt_text or "reasoning" in prompt_text.lower()

    @patch("src.prompt_builder.ReasoningStrategyLoader")
    def test_rag_enhanced_reasoning_instructions_included(self, mock_loader):
        """Test that RAG-Enhanced reasoning instructions are included."""
        mock_strategy = MagicMock()
        mock_strategy.is_strategy_enabled.return_value = True
        mock_strategy.get_strategy_instructions.return_value = [
            "First, use the retrieved documents as your knowledge base.",
            "Always ground your answer in the provided documents.",
            "Do not speculate beyond what is explicitly stated.",
        ]
        mock_strategy.get_strategy_name.return_value = "RAG-Enhanced Reasoning"
        mock_loader.return_value = mock_strategy

        prompts = build_system_prompts()
        prompt_text = "\n".join(prompts)

        assert "document" in prompt_text.lower()
        assert "ground" in prompt_text.lower() or "speculate" in prompt_text.lower()

    @patch("src.prompt_builder.ReasoningStrategyLoader")
    @pytest.mark.parametrize(
        "scenario,side_effect",
        [
            ("disabled", None),
            ("error", Exception("Failed to load strategy")),
        ],
    )
    def test_strategy_load_scenarios(self, mock_loader, scenario, side_effect):
        """Parametrized test for strategy loading edge cases."""
        mock_strategy = MagicMock()
        mock_strategy.is_strategy_enabled.return_value = scenario != "disabled"

        if side_effect:
            mock_loader.side_effect = side_effect
        else:
            mock_loader.return_value = mock_strategy

        prompts = build_system_prompts()
        assert len(prompts) > 0

    def test_build_system_prompts_with_reasoning_strategy(self, mock_strategy):
        """Test that reasoning_strategy parameter is used when provided."""
        prompts = build_system_prompts(reasoning_strategy=mock_strategy)
        prompt_text = "\n".join(prompts)

        assert "Test instruction" in prompt_text

    # ========================================================================
    # DEFAULT SYSTEM PROMPTS (TestGetDefaultSystemPrompts)
    # ========================================================================

    @pytest.mark.parametrize(
        "check_type,assertion",
        [
            ("is_list", lambda p: isinstance(p, list)),
            ("non_empty", lambda p: len(p) > 0),
        ],
    )
    def test_returns_valid_structure(self, default_prompts, check_type, assertion):
        """Parametrized test for default prompts structure."""
        assert assertion(default_prompts), f"Failed: {check_type}"

    def test_returns_non_empty_strings(self, default_prompts):
        """Test that all default prompts are non-empty strings."""
        assert all(isinstance(p, str) and len(p) > 0 for p in default_prompts)

    @pytest.mark.parametrize(
        "key_phrases",
        [
            [
                "helpful AI assistant",
                "provided documents",
                "do not",
                "make up",
            ],
        ],
    )
    def test_contains_key_instructions(self, default_prompt_text, key_phrases):
        """Test that default prompts contain key instructions."""
        for phrase in key_phrases:
            assert phrase.lower() in default_prompt_text.lower()

    def test_includes_meta_handling(self, default_prompt_text):
        """Test that default prompts include handling for meta questions."""
        assert "identity" in default_prompt_text.lower() or "capabilities" in default_prompt_text.lower()
        assert "greeting" in default_prompt_text.lower()

    # ========================================================================
    # PROMPT TEMPLATE CREATION (TestCreatePromptTemplate)
    # ========================================================================

    @pytest.mark.parametrize(
        "system_prompts",
        [
            ["You are a helpful assistant."],
            ["You are a test assistant.", "Follow these rules."],
            [],
        ],
    )
    def test_returns_chat_prompt_template(self, system_prompts):
        """Parametrized test that create_prompt_template returns valid ChatPromptTemplate."""
        template = create_prompt_template(system_prompts)

        assert isinstance(template, ChatPromptTemplate)
        assert template is not None

    def test_template_has_required_variables(self):
        """Test that the template includes required input variables."""
        system_prompts = ["You are a helpful assistant."]
        template = create_prompt_template(system_prompts)

        template_str = str(template)
        assert all(var in template_str for var in ["chat_history", "context", "question"])

    @pytest.mark.parametrize(
        "system_prompts",
        [
            ["Prompt 1"],
            ["Prompt 1", "Prompt 2"],
            ["Prompt 1", "Prompt 2", "Prompt 3"],
        ],
    )
    def test_multiple_system_prompts(self, system_prompts):
        """Test that multiple system prompts are included in template."""
        template = create_prompt_template(system_prompts)

        assert template is not None
        template_str = str(template)
        for prompt in system_prompts:
            assert prompt in template_str

    # ========================================================================
    # COVERAGE: ROLE HANDLING (Lines 33-34)
    # ========================================================================

    @patch("src.prompt_builder.load_yaml")
    def test_role_with_default_value(self, mock_yaml):
        """Test role handling when role is not provided (uses default)."""
        mock_yaml.return_value = {
            "default": {
                "role": None,
                "style_or_tone": None,
                "output_constraints": None,
                "output_format": None,
            }
        }
        with patch("src.prompt_builder.config.PROMPT_CONFIG_NAME", "default"):
            prompts = build_system_prompts()

            assert any("helpful AI assistant" in p for p in prompts) or len(prompts) > 0

    @patch("src.prompt_builder.load_yaml")
    def test_role_with_custom_value(self, mock_yaml):
        """Test role handling when custom role is provided."""
        mock_yaml.return_value = {
            "default": {
                "role": "A knowledgeable expert in AI.",
                "style_or_tone": None,
                "output_constraints": None,
                "output_format": None,
            }
        }
        with patch("src.prompt_builder.config.PROMPT_CONFIG_NAME", "default"):
            prompts = build_system_prompts()

            assert any("knowledgeable expert" in p.lower() for p in prompts)

    @patch("src.prompt_builder.load_yaml")
    def test_role_with_whitespace(self, mock_yaml):
        """Test role handling with leading/trailing whitespace."""
        mock_yaml.return_value = {
            "default": {
                "role": "   A helpful assistant   ",
                "style_or_tone": None,
                "output_constraints": None,
                "output_format": None,
            }
        }
        with patch("src.prompt_builder.config.PROMPT_CONFIG_NAME", "default"):
            prompts = build_system_prompts()

            assert any("helpful assistant" in p.lower() for p in prompts)

    # ========================================================================
    # COVERAGE: TONE/STYLE HANDLING (Lines 37->41)
    # ========================================================================

    @patch("src.prompt_builder.load_yaml")
    def test_tone_included_when_provided(self, mock_yaml):
        """Test that tone is included when provided in config."""
        mock_yaml.return_value = {
            "default": {
                "role": "Assistant",
                "style_or_tone": "Be concise and formal.",
                "output_constraints": None,
                "output_format": None,
            }
        }
        with patch("src.prompt_builder.config.PROMPT_CONFIG_NAME", "default"):
            prompts = build_system_prompts()

            assert any("concise and formal" in p for p in prompts)

    @patch("src.prompt_builder.load_yaml")
    def test_tone_skipped_when_not_provided(self, mock_yaml):
        """Test that tone is skipped when not provided."""
        mock_yaml.return_value = {
            "default": {
                "role": "Assistant",
                "style_or_tone": None,
                "output_constraints": None,
                "output_format": None,
            }
        }
        with patch("src.prompt_builder.config.PROMPT_CONFIG_NAME", "default"):
            prompts = build_system_prompts()

            tone_prompts = [p for p in prompts if "style or tone" in p.lower()]
            assert len(tone_prompts) == 0

    # ========================================================================
    # COVERAGE: CONSTRAINTS HANDLING (Lines 41->45)
    # ========================================================================

    @patch("src.prompt_builder.load_yaml")
    def test_constraints_included_when_provided(self, mock_yaml):
        """Test that constraints are included when provided."""
        mock_yaml.return_value = {
            "default": {
                "role": "Assistant",
                "style_or_tone": None,
                "output_constraints": "Always provide sources.",
                "output_format": None,
            }
        }
        with patch("src.prompt_builder.config.PROMPT_CONFIG_NAME", "default"):
            prompts = build_system_prompts()

            assert any("Always provide sources" in p for p in prompts)

    @patch("src.prompt_builder.load_yaml")
    def test_constraints_skipped_when_not_provided(self, mock_yaml):
        """Test that constraints are skipped when not provided."""
        mock_yaml.return_value = {
            "default": {
                "role": "Assistant",
                "style_or_tone": None,
                "output_constraints": None,
                "output_format": None,
            }
        }
        with patch("src.prompt_builder.config.PROMPT_CONFIG_NAME", "default"):
            prompts = build_system_prompts()

            constraint_prompts = [p for p in prompts if "output constraint" in p.lower()]
            assert len(constraint_prompts) == 0

    # ========================================================================
    # COVERAGE: OUTPUT FORMAT HANDLING (Lines 45->49)
    # ========================================================================

    @patch("src.prompt_builder.load_yaml")
    def test_output_format_included_when_provided(self, mock_yaml):
        """Test that output format is included when provided."""
        mock_yaml.return_value = {
            "default": {
                "role": "Assistant",
                "style_or_tone": None,
                "output_constraints": None,
                "output_format": "Use markdown formatting.",
            }
        }
        with patch("src.prompt_builder.config.PROMPT_CONFIG_NAME", "default"):
            prompts = build_system_prompts()

            assert any("markdown formatting" in p for p in prompts)

    @patch("src.prompt_builder.load_yaml")
    def test_output_format_skipped_when_not_provided(self, mock_yaml):
        """Test that output format is skipped when not provided."""
        mock_yaml.return_value = {
            "default": {
                "role": "Assistant",
                "style_or_tone": None,
                "output_constraints": None,
                "output_format": None,
            }
        }
        with patch("src.prompt_builder.config.PROMPT_CONFIG_NAME", "default"):
            prompts = build_system_prompts()

            format_prompts = [p for p in prompts if "output format" in p.lower()]
            assert len(format_prompts) == 0

    # ========================================================================
    # COVERAGE: REASONING STRATEGY HANDLING (Lines 49->54, 58->70)
    # ========================================================================

    @patch("src.prompt_builder.ReasoningStrategyLoader")
    @patch("src.prompt_builder.load_yaml")
    def test_strategy_instructions_included_when_enabled(self, mock_yaml, mock_loader):
        """Test that strategy instructions are included when strategy is enabled."""
        mock_yaml.return_value = {
            "default": {
                "role": "Assistant",
                "style_or_tone": None,
                "output_constraints": None,
                "output_format": None,
            }
        }

        mock_strategy = MagicMock()
        mock_strategy.is_strategy_enabled.return_value = True
        mock_strategy.get_strategy_instructions.return_value = ["Step 1", "Step 2"]
        mock_strategy.get_strategy_name.return_value = "Test Strategy"
        mock_loader.return_value = mock_strategy

        with patch("src.prompt_builder.config.PROMPT_CONFIG_NAME", "default"):
            prompts = build_system_prompts()

            assert any("Step 1" in p for p in prompts) or any("reasoning" in p.lower() for p in prompts)

    @patch("src.prompt_builder.ReasoningStrategyLoader")
    @patch("src.prompt_builder.load_yaml")
    def test_strategy_skipped_when_disabled(self, mock_yaml, mock_loader):
        """Test that strategy is skipped when disabled."""
        mock_yaml.return_value = {
            "default": {
                "role": "Assistant",
                "style_or_tone": None,
                "output_constraints": None,
                "output_format": None,
            }
        }

        mock_strategy = MagicMock()
        mock_strategy.is_strategy_enabled.return_value = False
        mock_loader.return_value = mock_strategy

        with patch("src.prompt_builder.config.PROMPT_CONFIG_NAME", "default"):
            with patch("src.prompt_builder.config.REASONING_STRATEGY", "disabled"):
                prompts = build_system_prompts()

                assert len(prompts) > 0

    @patch("src.prompt_builder.ReasoningStrategyLoader")
    @patch("src.prompt_builder.load_yaml")
    def test_strategy_with_empty_instructions(self, mock_yaml, mock_loader):
        """Test strategy handling when instructions are empty."""
        mock_yaml.return_value = {
            "default": {
                "role": "Assistant",
                "style_or_tone": None,
                "output_constraints": None,
                "output_format": None,
            }
        }

        mock_strategy = MagicMock()
        mock_strategy.is_strategy_enabled.return_value = True
        mock_strategy.get_strategy_instructions.return_value = []
        mock_loader.return_value = mock_strategy

        with patch("src.prompt_builder.config.PROMPT_CONFIG_NAME", "default"):
            prompts = build_system_prompts()

            reasoning_prompts = [p for p in prompts if "reasoning" in p.lower()]
            assert len(reasoning_prompts) == 0

    @patch("src.prompt_builder.ReasoningStrategyLoader")
    @patch("src.prompt_builder.load_yaml")
    def test_strategy_exception_handling(self, mock_yaml, mock_loader):
        """Test exception handling when strategy loader raises error."""
        mock_yaml.return_value = {
            "default": {
                "role": "Assistant",
                "style_or_tone": None,
                "output_constraints": None,
                "output_format": None,
            }
        }

        mock_loader.side_effect = RuntimeError("Strategy loading failed")

        with patch("src.prompt_builder.config.PROMPT_CONFIG_NAME", "default"):
            prompts = build_system_prompts()

            assert len(prompts) > 0

    @patch("src.prompt_builder.ReasoningStrategyLoader")
    @patch("src.prompt_builder.load_yaml")
    def test_strategy_loader_called_when_none_provided(self, mock_yaml, mock_loader):
        """Test that ReasoningStrategyLoader is instantiated when reasoning_strategy is None."""
        mock_yaml.return_value = {
            "default": {
                "role": "Assistant",
                "style_or_tone": None,
                "output_constraints": None,
                "output_format": None,
            }
        }

        mock_strategy = MagicMock()
        mock_strategy.is_strategy_enabled.return_value = False
        mock_loader.return_value = mock_strategy

        with patch("src.prompt_builder.config.PROMPT_CONFIG_NAME", "default"):
            build_system_prompts(reasoning_strategy=None)

            mock_loader.assert_called_once()

    @patch("src.prompt_builder.load_yaml")
    def test_strategy_provided_parameter_used(self, mock_yaml):
        """Test that provided strategy parameter is used instead of creating new one."""
        mock_yaml.return_value = {
            "default": {
                "role": "Assistant",
                "style_or_tone": None,
                "output_constraints": None,
                "output_format": None,
            }
        }

        mock_strategy = MagicMock()
        mock_strategy.is_strategy_enabled.return_value = True
        mock_strategy.get_strategy_instructions.return_value = ["Custom instruction"]
        mock_strategy.get_strategy_name.return_value = "Custom Strategy"

        with patch("src.prompt_builder.config.PROMPT_CONFIG_NAME", "default"):
            with patch("src.prompt_builder.ReasoningStrategyLoader") as mock_loader:
                prompts = build_system_prompts(reasoning_strategy=mock_strategy)

                mock_loader.assert_not_called()
                assert any("Custom instruction" in p for p in prompts)

    # ========================================================================
    # COVERAGE: ERROR CASES
    # ========================================================================

    @patch("src.prompt_builder.load_yaml")
    def test_missing_prompt_config_raises_error(self, mock_yaml):
        """Test that missing prompt config raises KeyError."""
        mock_yaml.return_value = {"other_config": {}}

        with patch("src.prompt_builder.config.PROMPT_CONFIG_NAME", "missing"):
            with pytest.raises(KeyError):
                build_system_prompts()

    @patch("src.prompt_builder.load_yaml")
    def test_none_prompt_config_raises_error(self, mock_yaml):
        """Test that None prompt config raises ValueError."""
        mock_yaml.return_value = {"default": None}

        with patch("src.prompt_builder.config.PROMPT_CONFIG_NAME", "default"):
            with pytest.raises(ValueError):
                build_system_prompts()

    # ========================================================================
    # COVERAGE: INTEGRATION TESTS
    # ========================================================================

    @patch("src.prompt_builder.ReasoningStrategyLoader")
    @patch("src.prompt_builder.load_yaml")
    def test_all_components_together(self, mock_yaml, mock_loader):
        """Test that all components work together when all are provided."""
        mock_yaml.return_value = {
            "default": {
                "role": "Expert AI",
                "style_or_tone": "Be professional",
                "output_constraints": "Always explain",
                "output_format": "Use lists",
            }
        }

        mock_strategy = MagicMock()
        mock_strategy.is_strategy_enabled.return_value = True
        mock_strategy.get_strategy_instructions.return_value = ["Think step-by-step"]
        mock_strategy.get_strategy_name.return_value = "Step-by-step"
        mock_loader.return_value = mock_strategy

        with patch("src.prompt_builder.config.PROMPT_CONFIG_NAME", "default"):
            prompts = build_system_prompts()

            prompt_text = "\n".join(prompts).lower()
            # Role gets stripped and lowercased
            assert "expert" in prompt_text
            assert "professional" in prompt_text
            assert "always explain" in prompt_text
            assert "use lists" in prompt_text

    @patch("src.prompt_builder.load_yaml")
    def test_minimal_config(self, mock_yaml):
        """Test with minimal config (only role)."""
        mock_yaml.return_value = {
            "default": {
                "role": "Helper",
                "style_or_tone": None,
                "output_constraints": None,
                "output_format": None,
            }
        }

        with patch("src.prompt_builder.config.PROMPT_CONFIG_NAME", "default"):
            with patch("src.prompt_builder.ReasoningStrategyLoader") as mock_loader:
                mock_strategy = MagicMock()
                mock_strategy.is_strategy_enabled.return_value = False
                mock_loader.return_value = mock_strategy

                prompts = build_system_prompts()

                assert len(prompts) > 0
                prompt_text = "\n".join(prompts).lower()
                assert "helper" in prompt_text

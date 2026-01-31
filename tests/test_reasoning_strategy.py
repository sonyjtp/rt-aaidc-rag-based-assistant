"""
Unit tests for reasoning strategy functionality.
Tests strategy loading, application, and configuration.
"""

# pylint: disable=redefined-outer-name

from unittest.mock import patch

import pytest

from src.reasoning_strategy_loader import ReasoningStrategyLoader


@pytest.fixture
def mock_yaml_and_config():
    """Fixture providing mocked YAML loader and config."""
    with patch("src.reasoning_strategy_loader.load_yaml") as mock_load_yaml, patch(
        "src.reasoning_strategy_loader.config"
    ) as mock_config:
        yield mock_config, mock_load_yaml


class TestReasoningStrategyLoader:
    """Unified test class for ReasoningStrategyLoader covering all functionality."""

    # ========================================================================
    # INITIALIZATION TESTS
    # ========================================================================

    def test_initialization_success(
        self, mock_yaml_and_config
    ):  # pylint: disable=redefined-outer-name
        """Test successful initialization of ReasoningStrategyLoader."""
        mock_config, mock_load_yaml = mock_yaml_and_config
        mock_config.REASONING_STRATEGIES_FPATH = "/path/to/strategies.yaml"
        mock_config.REASONING_STRATEGY = "rag_enhanced_reasoning"

        mock_load_yaml.return_value = {
            "reasoning_strategies": {
                "rag_enhanced_reasoning": {
                    "name": "RAG-Enhanced Reasoning",
                    "enabled": True,
                    "prompt_instructions": ["Instruction 1"],
                }
            }
        }

        loader = ReasoningStrategyLoader()

        assert loader.active_strategy == "rag_enhanced_reasoning"
        assert loader.strategies is not None

    def test_custom_config_path(
        self, mock_yaml_and_config
    ):  # pylint: disable=redefined-outer-name
        """Test initialization with custom config path."""
        mock_config, mock_load_yaml = mock_yaml_and_config
        mock_config.REASONING_STRATEGIES_FPATH = "/default/path.yaml"
        mock_config.REASONING_STRATEGY = "test_strategy"

        mock_load_yaml.return_value = {
            "reasoning_strategies": {"test_strategy": {"enabled": True}}
        }

        custom_path = "/custom/path.yaml"
        ReasoningStrategyLoader(config_path=custom_path)

        mock_load_yaml.assert_called_with(custom_path)

    # ========================================================================
    # RETRIEVAL TESTS
    # ========================================================================

    @pytest.mark.parametrize(
        "strategy_key,config_data,method,expected_result",
        [
            (
                "chain_of_thought",
                {
                    "name": "Chain-of-Thought",
                    "enabled": True,
                    "description": "Step by step reasoning",
                },
                "get_active_strategy",
                lambda result: result["name"] == "Chain-of-Thought"
                and result["enabled"] is True,
            ),
            (
                "rag_enhanced_reasoning",
                {"name": "RAG-Enhanced Reasoning"},
                "get_strategy_name",
                lambda result: result == "RAG-Enhanced Reasoning",
            ),
            (
                "test_strategy",
                {"description": "This is a test strategy"},
                "get_strategy_description",
                lambda result: result == "This is a test strategy",
            ),
        ],
    )
    def test_get_strategy_info(
        self, mock_yaml_and_config, strategy_key, config_data, method, expected_result
    ):  # pylint: disable=redefined-outer-name
        """Parametrized test for retrieving strategy information."""
        mock_config, mock_load_yaml = mock_yaml_and_config
        mock_config.REASONING_STRATEGIES_FPATH = "/path/to/strategies.yaml"
        mock_config.REASONING_STRATEGY = strategy_key

        mock_load_yaml.return_value = {
            "reasoning_strategies": {strategy_key: config_data}
        }

        loader = ReasoningStrategyLoader()
        result = getattr(loader, method)()

        assert expected_result(result)

    def test_get_strategy_instructions(
        self, mock_yaml_and_config
    ):  # pylint: disable=redefined-outer-name
        """Test retrieving strategy instructions."""
        mock_config, mock_load_yaml = mock_yaml_and_config
        mock_config.REASONING_STRATEGIES_FPATH = "/path/to/strategies.yaml"
        mock_config.REASONING_STRATEGY = "test_strategy"

        instructions = [
            "First, analyze the question",
            "Then, retrieve relevant information",
            "Finally, synthesize the answer",
        ]

        mock_load_yaml.return_value = {
            "reasoning_strategies": {
                "test_strategy": {"prompt_instructions": instructions}
            }
        }

        loader = ReasoningStrategyLoader()
        retrieved_instructions = loader.get_strategy_instructions()

        assert retrieved_instructions == instructions
        assert len(retrieved_instructions) == 3

    def test_get_few_shot_examples(self, mock_yaml_and_config):
        """Test retrieving few-shot examples if available."""
        mock_config, mock_load_yaml = mock_yaml_and_config
        mock_config.REASONING_STRATEGIES_FPATH = "/path/to/strategies.yaml"
        mock_config.REASONING_STRATEGY = "test_strategy"

        examples = [
            {"question": "Example Q1", "answer": "Example A1"},
            {"question": "Example Q2", "answer": "Example A2"},
        ]

        mock_load_yaml.return_value = {
            "reasoning_strategies": {"test_strategy": {"examples": examples}}
        }

        loader = ReasoningStrategyLoader()
        retrieved_examples = loader.get_few_shot_examples()

        assert len(retrieved_examples) == 2
        assert retrieved_examples[0]["question"] == "Example Q1"

    # ========================================================================
    # VALIDATION TESTS
    # ========================================================================

    @pytest.mark.parametrize(
        "strategy_name,enabled,expected",
        [
            ("enabled_strategy", True, True),
            ("disabled_strategy", False, False),
        ],
    )
    def test_is_strategy_enabled(
        self, mock_yaml_and_config, strategy_name, enabled, expected
    ):
        """Parametrized test for checking if strategy is enabled."""
        mock_config, mock_load_yaml = mock_yaml_and_config
        mock_config.REASONING_STRATEGIES_FPATH = "/path/to/strategies.yaml"
        mock_config.REASONING_STRATEGY = strategy_name

        mock_load_yaml.return_value = {
            "reasoning_strategies": {strategy_name: {"enabled": enabled}}
        }

        loader = ReasoningStrategyLoader()
        assert loader.is_strategy_enabled() is expected

    def test_invalid_strategy_raises_error(self, mock_yaml_and_config):
        """Test that requesting invalid strategy raises error."""
        mock_config, mock_load_yaml = mock_yaml_and_config
        mock_config.REASONING_STRATEGIES_FPATH = "/path/to/strategies.yaml"
        mock_config.REASONING_STRATEGY = "nonexistent_strategy"

        mock_load_yaml.return_value = {
            "reasoning_strategies": {"existing_strategy": {"enabled": True}}
        }

        loader = ReasoningStrategyLoader()

        with pytest.raises(ValueError):
            loader.get_active_strategy()

    # ========================================================================
    # ENABLED STRATEGIES TESTS
    # ========================================================================

    def test_get_all_enabled_strategies(self, mock_yaml_and_config):
        """Test retrieving all enabled strategies."""
        mock_config, mock_load_yaml = mock_yaml_and_config
        mock_config.REASONING_STRATEGIES_FPATH = "/path/to/strategies.yaml"
        mock_config.REASONING_STRATEGY = "rag_enhanced_reasoning"

        mock_load_yaml.return_value = {
            "reasoning_strategies": {
                "rag_enhanced_reasoning": {"enabled": True},
                "chain_of_thought": {"enabled": True},
                "self_consistency": {"enabled": False},
                "tree_of_thought": {"enabled": False},
            }
        }

        loader = ReasoningStrategyLoader()
        enabled = loader.get_all_enabled_strategies()

        assert len(enabled) == 2
        assert "rag_enhanced_reasoning" in enabled
        assert "chain_of_thought" in enabled
        assert "self_consistency" not in enabled

    # ========================================================================
    # PROMPT BUILDING TESTS
    # ========================================================================

    def test_build_strategy_prompt(self, mock_yaml_and_config):
        """Test building complete prompt for strategy."""
        mock_config, mock_load_yaml = mock_yaml_and_config
        mock_config.REASONING_STRATEGIES_FPATH = "/path/to/strategies.yaml"
        mock_config.REASONING_STRATEGY = "test_strategy"

        mock_load_yaml.return_value = {
            "reasoning_strategies": {
                "test_strategy": {
                    "name": "Test Strategy",
                    "description": "A test reasoning strategy",
                    "prompt_instructions": ["Step 1", "Step 2"],
                }
            }
        }

        loader = ReasoningStrategyLoader()
        prompt = loader.build_strategy_prompt()

        assert "Test Strategy" in prompt
        assert "A test reasoning strategy" in prompt
        assert "Step 1" in prompt
        assert "Step 2" in prompt

    # ========================================================================
    # EDGE CASES AND ERROR HANDLING TESTS
    # ========================================================================

    @pytest.mark.parametrize(
        "strategy_config,expected_name,expected_description,expected_instructions",
        [
            (
                {"prompt_instructions": []},
                "no_instructions",
                "",
                [],
            ),
            (
                {"enabled": True},
                "minimal_strategy",
                "",
                [],
            ),
        ],
    )
    def test_missing_or_empty_fields(
        self,
        mock_yaml_and_config,
        strategy_config,
        expected_name,
        expected_description,
        expected_instructions,
    ):
        """Parametrized test for strategies with missing or empty optional fields."""
        mock_config, mock_load_yaml = mock_yaml_and_config
        mock_config.REASONING_STRATEGIES_FPATH = "/path/to/strategies.yaml"
        mock_config.REASONING_STRATEGY = expected_name

        mock_load_yaml.return_value = {
            "reasoning_strategies": {expected_name: strategy_config}
        }

        loader = ReasoningStrategyLoader()

        assert loader.get_strategy_name() == expected_name
        assert loader.get_strategy_description() == expected_description
        assert loader.get_strategy_instructions() == expected_instructions

    def test_very_long_instructions(self, mock_yaml_and_config):
        """Test strategy with very long instruction list."""
        mock_config, mock_load_yaml = mock_yaml_and_config
        mock_config.REASONING_STRATEGIES_FPATH = "/path/to/strategies.yaml"
        mock_config.REASONING_STRATEGY = "long_instructions"

        long_instructions = [f"Instruction {i}" for i in range(100)]

        mock_load_yaml.return_value = {
            "reasoning_strategies": {
                "long_instructions": {"prompt_instructions": long_instructions}
            }
        }

        loader = ReasoningStrategyLoader()
        instructions = loader.get_strategy_instructions()

        assert len(instructions) == 100

    @pytest.mark.parametrize(
        "exception_type,exception_msg",
        [
            (FileNotFoundError, "Config not found"),
            (Exception, "Invalid YAML format"),
        ],
    )
    def test_initialization_errors(self, exception_type, exception_msg):
        """Parametrized test for initialization errors."""
        with patch("src.reasoning_strategy_loader.load_yaml") as mock_load_yaml, patch(
            "src.reasoning_strategy_loader.config"
        ) as mock_config:
            mock_config.REASONING_STRATEGIES_FPATH = "/path/to/strategies.yaml"
            mock_config.REASONING_STRATEGY = "test_strategy"
            mock_load_yaml.side_effect = exception_type(exception_msg)

            with pytest.raises(exception_type):
                ReasoningStrategyLoader()

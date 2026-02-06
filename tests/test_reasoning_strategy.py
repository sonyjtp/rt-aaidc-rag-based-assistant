"""
Unit tests for reasoning strategy functionality.
Tests strategy loading, application, and configuration.
"""

from unittest.mock import patch

import pytest

from src.reasoning_strategy_loader import ReasoningStrategyLoader


@pytest.fixture
def loader_with_strategy():
    """Fixture providing ReasoningStrategyLoader with active strategy."""
    with patch("src.reasoning_strategy_loader.load_yaml") as mock_load, patch(
        "src.reasoning_strategy_loader.config"
    ) as mock_config, patch("src.reasoning_strategy_loader.logger"):
        mock_config.REASONING_STRATEGY = "rag_enhanced_reasoning"
        mock_load.return_value = {
            "reasoning_strategies": {
                "rag_enhanced_reasoning": {
                    "name": "RAG-Enhanced Reasoning",
                    "enabled": True,
                    "description": "Uses RAG as foundation",
                    "prompt_instructions": ["Step 1", "Step 2"],
                    "examples": [{"question": "Q1", "answer": "A1"}],
                }
            }
        }
        loader = ReasoningStrategyLoader()
        return loader


@pytest.fixture
def loader_without_strategy():
    """Fixture providing ReasoningStrategyLoader without active strategy."""
    with patch("src.reasoning_strategy_loader.load_yaml") as mock_load, patch(
        "src.reasoning_strategy_loader.config"
    ) as mock_config, patch("src.reasoning_strategy_loader.logger"):
        mock_config.REASONING_STRATEGY = "nonexistent"
        mock_load.return_value = {"reasoning_strategies": {}}
        loader = ReasoningStrategyLoader()
        return loader


class TestReasoningStrategyLoader:
    """Comprehensive tests for ReasoningStrategyLoader."""

    @pytest.mark.parametrize(
        "strategy_key,config_data,method,expected_result",
        [
            ("rag_enhanced", {"name": "RAG", "enabled": True, "description": "RAG-based"}, "get_strategy_name", "RAG"),
            (
                "rag_enhanced",
                {"name": "RAG", "enabled": True, "description": "RAG-based"},
                "get_strategy_description",
                "RAG-based",
            ),
            ("rag_enhanced", {"name": "RAG", "enabled": True}, "is_strategy_enabled", True),
            ("chain", {"name": "CoT", "enabled": False}, "is_strategy_enabled", False),
            ("few_shot", {"prompt_instructions": ["I1", "I2"]}, "get_strategy_instructions", ["I1", "I2"]),
            ("examples", {"examples": [{"q": "Q", "a": "A"}]}, "get_few_shot_examples", [{"q": "Q", "a": "A"}]),
        ],
    )
    def test_getter_methods_with_data(self, strategy_key, config_data, method, expected_result):
        """Parametrized test: getter methods return correct values."""
        with patch("src.reasoning_strategy_loader.load_yaml") as mock_load, patch(
            "src.reasoning_strategy_loader.config"
        ) as mock_config, patch("src.reasoning_strategy_loader.logger"):
            mock_config.REASONING_STRATEGY = strategy_key
            mock_load.return_value = {"reasoning_strategies": {strategy_key: config_data}}
            loader = ReasoningStrategyLoader()
            result = getattr(loader, method)()
            assert result == expected_result

    @pytest.mark.parametrize(
        "method,expected_default",
        [
            ("get_strategy_name", "nonexistent"),
            ("get_strategy_description", ""),
            ("get_strategy_instructions", []),
            ("get_few_shot_examples", []),
            ("is_strategy_enabled", False),
        ],
    )
    def test_getter_methods_without_strategy(self, loader_without_strategy, method, expected_default):
        """Parametrized test: getter methods return safe defaults when no strategy."""
        result = getattr(loader_without_strategy, method)()
        assert result == expected_default

    def test_get_active_strategy_success(self, loader_with_strategy):
        """Test get_active_strategy returns dict when available."""
        result = loader_with_strategy.get_active_strategy()
        assert isinstance(result, dict)
        assert result["name"] == "RAG-Enhanced Reasoning"

    def test_get_active_strategy_failure(self, loader_without_strategy):
        """Test get_active_strategy raises ValueError when no strategy."""
        with pytest.raises(ValueError):
            loader_without_strategy.get_active_strategy()

    def test_build_strategy_prompt_with_strategy(self, loader_with_strategy):
        """Test build_strategy_prompt generates correct format with active strategy."""
        result = loader_with_strategy.build_strategy_prompt()
        assert "RAG-Enhanced Reasoning" in result
        assert "Reasoning Strategy:" in result
        assert "Description:" in result
        assert "Instructions:" in result

    def test_build_strategy_prompt_without_strategy(self, loader_without_strategy):
        """Test build_strategy_prompt works even without active strategy."""
        result = loader_without_strategy.build_strategy_prompt()
        assert "Reasoning Strategy:" in result
        assert "nonexistent" in result

    @pytest.mark.parametrize(
        "strategy_key,config_data,should_warn",
        [
            ("missing", {}, True),
            (None, {"test": {}}, True),
            ("valid", {"valid": {"name": "Valid"}}, False),
        ],
    )
    def test_initialization_edge_cases(self, strategy_key, config_data, should_warn):
        """Parametrized test: initialization with various edge cases."""
        with patch("src.reasoning_strategy_loader.load_yaml") as mock_load, patch(
            "src.reasoning_strategy_loader.config"
        ) as mock_config, patch("src.reasoning_strategy_loader.logger") as mock_logger:
            mock_config.REASONING_STRATEGY = strategy_key
            mock_load.return_value = {"reasoning_strategies": config_data}
            loader = ReasoningStrategyLoader()
            if should_warn:
                mock_logger.warning.assert_called()
            else:
                assert loader.active_strategy is not None

    def test_initialization_with_yaml_error(self):
        """Test initialization when YAML loading fails."""
        with patch("src.reasoning_strategy_loader.load_yaml") as mock_load, patch(
            "src.reasoning_strategy_loader.config"
        ) as mock_config:
            mock_config.REASONING_STRATEGY = "test"
            mock_load.side_effect = FileNotFoundError("Config not found")
            with pytest.raises(FileNotFoundError):
                ReasoningStrategyLoader()

    @pytest.mark.parametrize(
        "config_data,expected_name",
        [
            ({"name": "Full Strategy"}, "Full Strategy"),
            ({"description": "Only desc"}, "test_key"),
            ({}, "test_key"),
        ],
    )
    def test_get_strategy_name_fallback(self, config_data, expected_name):
        """Parametrized test: strategy name falls back to key when missing."""
        with patch("src.reasoning_strategy_loader.load_yaml") as mock_load, patch(
            "src.reasoning_strategy_loader.config"
        ) as mock_config, patch("src.reasoning_strategy_loader.logger"):
            mock_config.REASONING_STRATEGY = "test_key"
            mock_load.return_value = {"reasoning_strategies": {"test_key": config_data}}
            loader = ReasoningStrategyLoader()
            assert loader.get_strategy_name() == expected_name

    @pytest.mark.parametrize(
        "config_data,expected_desc",
        [
            ({"description": "Test desc"}, "Test desc"),
            ({"name": "No desc"}, ""),
            ({}, ""),
        ],
    )
    def test_get_strategy_description_handling(self, config_data, expected_desc):
        """Parametrized test: description handling with missing fields."""
        with patch("src.reasoning_strategy_loader.load_yaml") as mock_load, patch(
            "src.reasoning_strategy_loader.config"
        ) as mock_config, patch("src.reasoning_strategy_loader.logger"):
            mock_config.REASONING_STRATEGY = "test"
            mock_load.return_value = {"reasoning_strategies": {"test": config_data}}
            loader = ReasoningStrategyLoader()
            assert loader.get_strategy_description() == expected_desc

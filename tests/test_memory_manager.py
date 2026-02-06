"""
Unit tests for memory management functionality.
Tests conversation storage, strategy switching, and memory operations.
"""
# pylint: disable=unused-argument

from unittest.mock import MagicMock, mock_open, patch

import pytest

from src.memory_manager import MemoryManager


@pytest.fixture
def combined_fixtures(request):
    """Combined fixture providing all mocks and patches."""
    mock_llm = MagicMock()
    mock_memory = MagicMock()
    strategy = getattr(request, "param", None)  # For parametrization if needed

    with patch("src.memory_manager.MEMORY_STRATEGIES_FPATH", "/path"), patch(
        "builtins.open", mock_open(read_data="{}")
    ), patch("src.memory_manager.MEMORY_STRATEGY", strategy) if strategy else patch(
        "src.memory_manager.MEMORY_STRATEGY", "none"
    ):
        yield {
            "mock_llm": mock_llm,
            "mock_memory": mock_memory,
            "strategy": strategy,
        }


# pylint: disable=redefined-outer-name
class TestMemoryManager:
    """Tests for MemoryManager initialization, message handling, and operations."""

    # ========================================================================
    # INITIALIZATION TESTS
    # ========================================================================

    @pytest.mark.parametrize(
        "combined_fixtures",
        ["summarization_sliding_window", "simple_buffer", "summary"],
        indirect=True,
    )
    def test_memory_initialization(self, combined_fixtures):
        """Test MemoryManager initialization with different strategies."""
        fixtures = combined_fixtures
        manager = MemoryManager(llm=fixtures["mock_llm"])

        assert manager.llm is not None
        assert manager.strategy == fixtures["strategy"]

    @pytest.mark.parametrize(
        "strategy,memory_class",
        [
            ("summarization_sliding_window", "SlidingWindowMemory"),
            ("simple_buffer", "SimpleBufferMemory"),
            ("summary", "SummaryMemory"),
        ],
    )
    def test_memory_initialization_error(self, combined_fixtures, strategy, memory_class):
        """Test memory initialization failure falls back to no memory."""
        with patch(f"src.memory_manager.{memory_class}", side_effect=ValueError("Init error")), patch(
            "src.memory_manager.MEMORY_STRATEGY", strategy
        ):
            manager = MemoryManager(llm=combined_fixtures["mock_llm"])
            assert manager.memory is None

    # ========================================================================
    # MESSAGE HANDLING TESTS
    # ========================================================================

    @pytest.mark.parametrize(
        "input_text,output_text",
        [
            ("Hello", "Hi there"),
            ("Question?", "Answer here"),
            ("", ""),
            ("A" * 5000, "A" * 5000),
            ("!@#$%^&*()_+-=[]{}|;:',.<>?/`~", "!@#$%^&*()_+-=[]{}|;:',.<>?/`~"),
        ],
    )
    def test_add_message(self, combined_fixtures, input_text, output_text):
        """Test adding messages of various types to memory."""
        manager = MemoryManager(llm=combined_fixtures["mock_llm"])
        manager.memory = combined_fixtures["mock_memory"]
        manager.add_message(input_text=input_text, output_text=output_text)

        combined_fixtures["mock_memory"].save_context.assert_called_once()

    @pytest.mark.parametrize(
        "memory_variables,expected_keys",
        [
            ({"chat_history": "User: Hello\nAssistant: Hi"}, 1),
            ({"chat_history": "Conversation", "summary": "Summary text"}, 2),
        ],
    )
    def test_get_memory_variables(self, combined_fixtures, memory_variables, expected_keys):
        """Test retrieving memory variables with different key counts."""
        combined_fixtures["mock_memory"].load_memory_variables.return_value = memory_variables
        manager = MemoryManager(llm=combined_fixtures["mock_llm"])
        manager.memory = combined_fixtures["mock_memory"]
        variables = manager.get_memory_variables()

        assert len(variables) == expected_keys
        assert all(key in variables for key in memory_variables.keys())

    @pytest.mark.parametrize(
        "scenario,memory_setup",
        [
            ("no_memory", lambda fixtures: None),
            (
                "load_error",
                lambda fixtures: (fixtures["mock_memory"], ValueError("Load error")),
            ),
        ],
    )
    def test_get_memory_variables_edge_cases(self, combined_fixtures, scenario, memory_setup):
        """Test get_memory_variables in edge cases: no memory or load error."""
        manager = MemoryManager(llm=combined_fixtures["mock_llm"])
        if scenario == "no_memory":
            manager.memory = memory_setup(combined_fixtures)
        elif scenario == "load_error":
            mock_memory, error = memory_setup(combined_fixtures)
            mock_memory.load_memory_variables.side_effect = error
            manager.memory = mock_memory
        variables = manager.get_memory_variables()
        assert variables == {}

    # ========================================================================
    # STRATEGY TESTS
    # ========================================================================

    @pytest.mark.parametrize(
        "combined_fixtures",
        ["summarization_sliding_window", "simple_buffer", "summary", "none"],
        indirect=True,
    )
    def test_switching_memory_strategies(self, combined_fixtures):
        """Test switching between all available memory strategies."""
        assert MemoryManager(llm=combined_fixtures["mock_llm"]) is not None

    def test_strategy_none_disables_memory(self, combined_fixtures):
        """Test that 'none' strategy disables memory."""
        with patch("src.memory_manager.MEMORY_STRATEGY", "none"):
            manager = MemoryManager(llm=combined_fixtures["mock_llm"])
            assert manager.memory is None or manager.strategy == "none"

    # ========================================================================
    # CONVERSATION FLOW TESTS
    # ========================================================================

    @pytest.mark.parametrize(
        "num_messages,expected_calls",
        [
            (2, 2),
            (3, 3),
        ],
    )
    def test_multi_turn_conversation(self, combined_fixtures, num_messages, expected_calls):
        """Test multi-turn conversation with varying message counts."""
        combined_fixtures["mock_memory"].load_memory_variables.return_value = {
            "history": "Previous conversation..."
        }  # Included for accumulation simulation, though not asserted
        manager = MemoryManager(llm=combined_fixtures["mock_llm"])
        manager.memory = combined_fixtures["mock_memory"]

        for i in range(num_messages):
            manager.add_message(input_text=f"Question {i}", output_text=f"Answer {i}")

        assert combined_fixtures["mock_memory"].save_context.call_count == expected_calls

    # ========================================================================
    # EXCEPTION HANDLING TESTS
    # ========================================================================

    @pytest.mark.parametrize(
        "scenario,patches,setup,assertions",
        [
            (
                "missing_config",
                {
                    "src.memory_manager.MEMORY_STRATEGIES_FPATH": "/nonexistent",
                    "src.memory_manager.MEMORY_STRATEGY": "none",
                    "builtins.open": FileNotFoundError("Config not found"),
                },
                lambda manager: None,
                lambda manager: (manager.config == {}),
            ),
            (
                "save_error",
                {"src.memory_manager.MEMORY_STRATEGY": "none"},
                lambda manager: setattr(
                    manager.memory.save_context,
                    "side_effect",
                    ValueError("Save failed"),
                ),
                lambda manager: (
                    manager.memory.save_context.assert_called_once(),
                    True,
                )[1],
            ),
            (
                "unknown_strategy",
                {"src.memory_manager.MEMORY_STRATEGY": "unknown_strategy"},
                lambda manager: None,
                lambda manager: (manager.memory is None),
            ),
        ],
    )
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_exception_handling(self, combined_fixtures, scenario, patches, setup, assertions):
        """Test exception handling in various scenarios."""
        manager = None
        match scenario:
            case "missing_config":
                with patch("src.memory_manager.MEMORY_STRATEGIES_FPATH", "/nonexistent"), patch(
                    "src.memory_manager.MEMORY_STRATEGY", "none"
                ), patch("builtins.open", side_effect=FileNotFoundError("Config not found")):
                    manager = MemoryManager(llm=combined_fixtures["mock_llm"])
            case "save_error":
                with patch("src.memory_manager.MEMORY_STRATEGY", "none"):
                    manager = MemoryManager(llm=combined_fixtures["mock_llm"])
                    manager.memory = combined_fixtures["mock_memory"]
                    manager.add_message(input_text="Test", output_text="Test")
            case "unknown_strategy":
                with patch("src.memory_manager.MEMORY_STRATEGY", "unknown_strategy"):
                    manager = MemoryManager(llm=combined_fixtures["mock_llm"])
        setup(manager)
        assert assertions(manager)

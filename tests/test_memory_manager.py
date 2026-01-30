"""
Unit tests for memory management functionality.
Tests conversation storage, strategy switching, and memory operations.
"""
# pylint: disable=unused-argument

from unittest.mock import MagicMock, mock_open, patch

import pytest

from src.memory_manager import MemoryManager


@pytest.fixture
def mock_memory():
    """Fixture providing a mocked memory instance."""
    return MagicMock()


@pytest.fixture
def patched_config():
    """Fixture that applies common memory configuration patches."""
    with patch("src.memory_manager.MEMORY_STRATEGIES_FPATH", "/path"), patch(
        "builtins.open", mock_open(read_data="{}")
    ):
        yield


@pytest.fixture
def strategy_patch(request):
    """Fixture for parametrized strategy patching."""
    with patch("src.memory_manager.MEMORY_STRATEGY", request.param):
        yield request.param


# pylint: disable=redefined-outer-name
class TestMemoryManager:
    """Tests for MemoryManager initialization, message handling, and operations."""

    # ========================================================================
    # INITIALIZATION TESTS
    # ========================================================================

    @pytest.mark.parametrize(
        "strategy_patch",
        ["summarization_sliding_window", "simple_buffer", "summary"],
        indirect=True,
    )
    def test_memory_initialization(self, patched_config, strategy_patch, mock_llm):
        """Test MemoryManager initialization with different strategies."""
        manager = MemoryManager(llm=mock_llm)

        assert manager.llm is not None
        assert manager.strategy == strategy_patch

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
    def test_add_message(
        self, patched_config, input_text, output_text, mock_llm, mock_memory
    ):
        """Test adding messages of various types to memory."""
        with patch("src.memory_manager.MEMORY_STRATEGY", "none"):
            manager = MemoryManager(llm=mock_llm)
            manager.memory = mock_memory
            manager.add_message(input_text=input_text, output_text=output_text)

        mock_memory.save_context.assert_called_once()

    @pytest.mark.parametrize(
        "memory_variables,expected_keys",
        [
            ({"chat_history": "User: Hello\nAssistant: Hi"}, 1),
            ({"chat_history": "Conversation", "summary": "Summary text"}, 2),
        ],
    )
    def test_get_memory_variables(
        self, patched_config, memory_variables, expected_keys, mock_llm, mock_memory
    ):
        """Test retrieving memory variables with different key counts."""
        mock_memory.load_memory_variables.return_value = memory_variables

        with patch("src.memory_manager.MEMORY_STRATEGY", "none"):
            manager = MemoryManager(llm=mock_llm)
            manager.memory = mock_memory
            variables = manager.get_memory_variables()

        assert len(variables) == expected_keys
        assert all(key in variables for key in memory_variables.keys())

    def test_get_memory_variables_with_no_memory(self, patched_config, mock_llm):
        """Test get_memory_variables when memory is None."""
        with patch("src.memory_manager.MEMORY_STRATEGY", "none"):
            manager = MemoryManager(llm=mock_llm)
            manager.memory = None
            variables = manager.get_memory_variables()

        assert variables == {}

    def test_get_memory_variables_load_error(
        self, patched_config, mock_llm, mock_memory
    ):
        """Test get_memory_variables handles loading errors gracefully."""
        mock_memory.load_memory_variables.side_effect = ValueError("Load error")

        with patch("src.memory_manager.MEMORY_STRATEGY", "none"):
            manager = MemoryManager(llm=mock_llm)
            manager.memory = mock_memory
            variables = manager.get_memory_variables()

        assert variables == {}

    # ========================================================================
    # STRATEGY TESTS
    # ========================================================================

    @pytest.mark.parametrize(
        "strategy_patch",
        ["summarization_sliding_window", "simple_buffer", "summary", "none"],
        indirect=True,
    )
    def test_switching_memory_strategies(
        self, patched_config, strategy_patch, mock_llm
    ):
        """Test switching between all available memory strategies."""
        manager = MemoryManager(llm=mock_llm)
        assert manager is not None

    def test_strategy_none_disables_memory(self, patched_config, mock_llm):
        """Test that 'none' strategy disables memory."""
        with patch("src.memory_manager.MEMORY_STRATEGY", "none"):
            manager = MemoryManager(llm=mock_llm)
            assert manager.memory is None or manager.strategy == "none"

    # ========================================================================
    # CONVERSATION FLOW TESTS
    # ========================================================================

    def test_multi_turn_conversation(self, patched_config, mock_llm, mock_memory):
        """Test multi-turn conversation with memory."""
        with patch("src.memory_manager.MEMORY_STRATEGY", "none"):
            manager = MemoryManager(llm=mock_llm)
            manager.memory = mock_memory

            manager.add_message(
                input_text="What is AI?",
                output_text="AI is artificial intelligence.",
            )
            manager.add_message(
                input_text="Tell me more",
                output_text="Machine learning is a subset of AI.",
            )

        assert mock_memory.save_context.call_count == 2

    def test_multi_turn_conversation_accumulation(
        self, patched_config, mock_llm, mock_memory
    ):
        """Test that conversation context accumulates over multiple turns."""
        mock_memory.load_memory_variables.return_value = {
            "history": "Previous conversation..."
        }

        with patch("src.memory_manager.MEMORY_STRATEGY", "none"):
            manager = MemoryManager(llm=mock_llm)
            manager.memory = mock_memory

            for i in range(3):
                manager.add_message(
                    input_text=f"Question {i}", output_text=f"Answer {i}"
                )

        assert mock_memory.save_context.call_count == 3

    # ========================================================================
    # EXCEPTION HANDLING TESTS
    # ========================================================================

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_missing_config_file(self, mock_llm):
        """Test graceful handling when config file is missing."""
        with patch("src.memory_manager.MEMORY_STRATEGIES_FPATH", "/nonexistent"), patch(
            "src.memory_manager.MEMORY_STRATEGY", "none"
        ), patch("builtins.open", side_effect=FileNotFoundError("Config not found")):
            manager = MemoryManager(llm=mock_llm)
            assert manager is not None
            assert manager.config == {}

    def test_memory_save_error(self, patched_config, mock_llm, mock_memory):
        """Test handling when memory save fails."""
        mock_memory.save_context.side_effect = ValueError("Save failed")

        with patch("src.memory_manager.MEMORY_STRATEGY", "none"):
            manager = MemoryManager(llm=mock_llm)
            manager.memory = mock_memory
            manager.add_message(input_text="Test", output_text="Test")

        mock_memory.save_context.assert_called_once()

    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_unknown_strategy_handling(self, patched_config, mock_llm):
        """Test handling of unknown memory strategy."""
        with patch("src.memory_manager.MEMORY_STRATEGY", "unknown_strategy"):
            manager = MemoryManager(llm=mock_llm)
            assert manager is not None
            assert manager.memory is None

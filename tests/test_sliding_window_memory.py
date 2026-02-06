"""
Unit tests for SlidingWindowMemory implementation.
Tests window management, summarization, and memory operations.
"""
# pylint: disable=unused-argument

from collections import deque
from unittest.mock import MagicMock

import pytest

from config import DEFAULT_MEMORY_SLIDING_WINDOW_SIZE
from src.sliding_window_memory import SlidingWindowMemory


# pylint: disable=protected-access, attribute-defined-outside-init, too-many-public-methods
class TestSlidingWindowMemory:
    """Unified test class for SlidingWindowMemory covering all functionality."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        self.mock_llm = MagicMock()

    # ========================================================================
    # INITIALIZATION TESTS
    # ========================================================================

    @pytest.mark.parametrize(
        "window_size,memory_key,expected_maxlen",
        [
            (
                DEFAULT_MEMORY_SLIDING_WINDOW_SIZE,
                "chat_history",
                DEFAULT_MEMORY_SLIDING_WINDOW_SIZE,
            ),
            (10, "chat_history", 10),
            (
                DEFAULT_MEMORY_SLIDING_WINDOW_SIZE,
                "conversation",
                DEFAULT_MEMORY_SLIDING_WINDOW_SIZE,
            ),
            (7, "custom_key", 7),
        ],
    )
    def test_initialization(self, window_size, memory_key, expected_maxlen):
        """Parametrized test for initialization with various configurations."""
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=window_size, memory_key=memory_key)

        assert memory.llm == self.mock_llm
        assert memory.window_size == window_size
        assert memory.memory_key == memory_key
        assert isinstance(memory.messages, deque)
        assert memory.messages.maxlen == expected_maxlen
        assert len(memory.messages) == 0
        assert memory.summary == ""

    # ========================================================================
    # SAVE CONTEXT TESTS
    # ========================================================================

    def test_save_single_message(self):
        """Test saving a single message pair."""
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=5)

        memory.save_context(inputs={"input": "Hello"}, outputs={"output": "Hi there"})

        assert len(memory.messages) == 1
        assert memory.messages[0]["input"] == "Hello"
        assert memory.messages[0]["output"] == "Hi there"

    def test_save_multiple_messages(self):
        """Test saving multiple messages."""
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=5)

        for i in range(3):
            memory.save_context(inputs={"input": f"Question {i}"}, outputs={"output": f"Answer {i}"})

        assert len(memory.messages) == 3
        assert memory.messages[-1]["input"] == "Question 2"

    def test_save_context_with_extra_keys(self):
        """Test that extra keys in inputs/outputs are ignored."""
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=5)

        memory.save_context(
            inputs={"input": "Question", "extra": "ignored"},
            outputs={"output": "Answer", "metadata": "ignored"},
        )

        assert memory.messages[0]["input"] == "Question"
        assert memory.messages[0]["output"] == "Answer"
        assert "extra" not in memory.messages[0]

    @pytest.mark.parametrize(
        "input_text,output_text",
        [
            ("What is the capital of France?", "Paris"),
            ("Hello", "Hi there"),
            ("Complex question with symbols!?", "Answer with symbols!"),
        ],
    )
    def test_save_context_extracts_text(self, input_text, output_text):
        """Parametrized test for text extraction in save_context."""
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=5)

        memory.save_context(inputs={"input": input_text}, outputs={"output": output_text})

        assert memory.messages[0]["input"] == input_text
        assert memory.messages[0]["output"] == output_text

    def test_window_full_triggers_summarization(self):
        """Test that summarization triggers when window is full."""
        self.mock_llm.invoke.return_value = MagicMock(content="Summary of conversation")
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=3)

        for i in range(3):
            memory.save_context(inputs={"input": f"Q{i}"}, outputs={"output": f"A{i}"})

        assert len(memory.messages) == 0
        assert memory.summary == "Summary of conversation"

    def test_window_does_not_exceed_maxlen(self):
        """Test that window never exceeds maximum size."""
        self.mock_llm.invoke.return_value = MagicMock(content="Summary")
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=3)

        for i in range(5):
            memory.save_context(inputs={"input": f"Q{i}"}, outputs={"output": f"A{i}"})

        assert len(memory.messages) <= 3

    # ========================================================================
    # SUMMARIZATION TESTS
    # ========================================================================

    def test_summarize_window_calls_llm(self):
        """Test that summarization calls the LLM."""
        self.mock_llm.invoke.return_value = MagicMock(content="Summary")
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=2)

        for i in range(2):
            memory.save_context(inputs={"input": f"Q{i}"}, outputs={"output": f"A{i}"})

        assert self.mock_llm.invoke.called

    @pytest.mark.parametrize(
        "response_type,expected_summary",
        [
            (MagicMock(content="Summarized content"), "Summarized content"),
            ("String response", "String response"),
        ],
    )
    def test_summarize_window_response_handling(self, response_type, expected_summary):
        """Parametrized test for summarization response handling."""
        self.mock_llm.invoke.return_value = response_type
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=2)

        for i in range(2):
            memory.save_context(inputs={"input": f"Q{i}"}, outputs={"output": f"A{i}"})

        assert memory.summary == expected_summary

    def test_summarize_window_clears_messages(self):
        """Test that messages are cleared after summarization."""
        self.mock_llm.invoke.return_value = MagicMock(content="Summary")
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=2)

        for i in range(2):
            memory.save_context(inputs={"input": f"Q{i}"}, outputs={"output": f"A{i}"})

        assert len(memory.messages) == 0

    def test_summarize_window_error_handling(self):
        """Test error handling during summarization."""
        self.mock_llm.invoke.side_effect = Exception("LLM error")
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=2)

        for i in range(2):
            memory.save_context(inputs={"input": f"Q{i}"}, outputs={"output": f"A{i}"})

        assert "Q0" in memory.summary
        assert "A0" in memory.summary

    def test_summarize_empty_window(self):
        """Test summarization behavior with empty window."""
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=2)
        memory._summarize_window()

        assert memory.summary == ""

    def test_summarize_includes_all_messages(self):
        """Test that summarization includes all messages in window."""
        self.mock_llm.invoke.return_value = MagicMock(content="Summary")
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=3)

        for i in range(3):
            memory.save_context(inputs={"input": f"Question {i}"}, outputs={"output": f"Answer {i}"})

        call_args = self.mock_llm.invoke.call_args[0][0]
        assert all(f"Question {i}" in call_args for i in range(3))
        assert all(f"Answer {i}" in call_args for i in range(3))

    # ========================================================================
    # LOAD MEMORY VARIABLES TESTS
    # ========================================================================

    def test_load_memory_variables_empty(self):
        """Test loading memory when everything is empty."""
        memory = SlidingWindowMemory(llm=self.mock_llm, memory_key="chat_history")

        variables = memory.load_memory_variables()

        assert "chat_history" in variables
        assert variables["chat_history"] == ""

    def test_load_memory_variables_custom_key(self):
        """Test loading memory with custom key."""
        memory = SlidingWindowMemory(llm=self.mock_llm, memory_key="conversation")

        variables = memory.load_memory_variables()

        assert "conversation" in variables
        assert "chat_history" not in variables

    @pytest.mark.parametrize(
        "has_summary,has_messages",
        [
            (True, False),
            (False, True),
            (True, True),
        ],
    )
    def test_load_memory_variables_combinations(self, has_summary, has_messages):
        """Parametrized test for various memory state combinations."""
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=5)

        if has_summary:
            memory.summary = "Previous conversation summary"

        if has_messages:
            memory.save_context(inputs={"input": "Q"}, outputs={"output": "A"})

        variables = memory.load_memory_variables()
        content = variables["chat_history"]

        if has_summary:
            assert "Summary of previous conversation:" in content
            assert "Previous conversation summary" in content

        if has_messages:
            assert "Recent messages:" in content
            assert "User: Q" in content
            assert "Assistant: A" in content

    def test_load_memory_variables_multiple_messages(self):
        """Test loading memory with multiple messages."""
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=5)

        for i in range(3):
            memory.save_context(inputs={"input": f"Q{i}"}, outputs={"output": f"A{i}"})

        variables = memory.load_memory_variables()
        content = variables["chat_history"]

        for i in range(3):
            assert f"User: Q{i}" in content
            assert f"Assistant: A{i}" in content

    def test_load_memory_variables_formatting(self):
        """Test that memory variables are properly formatted."""
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=5)

        memory.save_context(inputs={"input": "Question"}, outputs={"output": "Answer"})

        variables = memory.load_memory_variables()
        content = variables["chat_history"]

        assert "User: Question" in content
        assert "Assistant: Answer" in content

    # ========================================================================
    # EDGE CASES AND SPECIAL SCENARIOS
    # ========================================================================

    @pytest.mark.parametrize(
        "message_content,description",
        [
            ("A" * 10000, "very_long_message"),
            ("!@#$%^&*()_+-=[]{}|;:',.<>?/`~\n\t", "special_characters"),
            ("Hello 世界 🚀 مرحبا", "unicode_characters"),
            ("", "empty_string"),
        ],
    )
    def test_handle_various_message_types(self, message_content, description):
        """Parametrized test for handling various message types."""
        self.mock_llm.invoke.return_value = MagicMock(content="Summary")
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=5)

        memory.save_context(inputs={"input": message_content}, outputs={"output": message_content})

        assert memory.messages[0]["input"] == message_content
        assert memory.messages[0]["output"] == message_content

    @pytest.mark.parametrize(
        "inputs,outputs,expected_input,expected_output",
        [
            ({}, {"output": "Answer"}, "", "Answer"),
            ({"input": "Question"}, {}, "Question", ""),
            ({}, {}, "", ""),
        ],
    )
    def test_missing_keys(self, inputs, outputs, expected_input, expected_output):
        """Parametrized test for handling missing input/output keys."""
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=5)

        memory.save_context(inputs=inputs, outputs=outputs)

        assert memory.messages[0]["input"] == expected_input
        assert memory.messages[0]["output"] == expected_output

    def test_window_size_one(self):
        """Test with window size of 1."""
        self.mock_llm.invoke.return_value = MagicMock(content="Summary")
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=1)

        memory.save_context(inputs={"input": "Q"}, outputs={"output": "A"})

        assert len(memory.messages) == 0
        assert memory.summary == "Summary"

    def test_multiple_windows(self):
        """Test multiple complete windows."""
        self.mock_llm.invoke.return_value = MagicMock(content="Summary")
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=2)

        for i in range(2):
            memory.save_context(inputs={"input": f"Q1-{i}"}, outputs={"output": f"A1-{i}"})

        for i in range(2):
            memory.save_context(inputs={"input": f"Q2-{i}"}, outputs={"output": f"A2-{i}"})

        assert memory.summary != ""
        assert len(memory.messages) == 0

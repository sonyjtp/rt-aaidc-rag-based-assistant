"""
Unit tests for SlidingWindowMemory implementation.
Tests window management, summarization, and memory operations.
"""

from collections import deque
from unittest.mock import MagicMock

from src.sliding_window_memory import SlidingWindowMemory


# pylint: disable=protected-access, attribute-defined-outside-init
class TestSlidingWindowMemoryInitialization:
    """Test SlidingWindowMemory initialization."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        self.mock_llm = MagicMock()

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""

        memory = SlidingWindowMemory(llm=self.mock_llm)

        assert memory.llm == self.mock_llm
        assert memory.window_size == 5
        assert memory.memory_key == "chat_history"
        assert len(memory.messages) == 0
        assert memory.summary == ""

    def test_init_with_custom_window_size(self):
        """Test initialization with custom window size."""

        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=10)

        assert memory.window_size == 10
        assert memory.messages.maxlen == 10

    def test_init_with_custom_memory_key(self):
        """Test initialization with custom memory key."""

        memory = SlidingWindowMemory(llm=self.mock_llm, memory_key="conversation")

        assert memory.memory_key == "conversation"

    def test_messages_is_deque_with_maxlen(self):
        """Test that messages is a deque with maxlen set."""
        window_size = 7

        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=window_size)

        assert isinstance(memory.messages, deque)
        assert memory.messages.maxlen == window_size

    def test_initial_summary_is_empty(self):
        """Test that initial summary is empty string."""

        memory = SlidingWindowMemory(llm=self.mock_llm)

        assert memory.summary == ""


class TestSlidingWindowMemorySaveContext:
    """Test message saving and window management."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        self.mock_llm = MagicMock()

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
            memory.save_context(
                inputs={"input": f"Question {i}"}, outputs={"output": f"Answer {i}"}
            )

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

    def test_window_full_triggers_summarization(self):
        """Test that summarization triggers when window is full."""
        self.mock_llm.invoke.return_value = MagicMock(content="Summary of conversation")
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=3)

        # Fill the window
        for i in range(3):
            memory.save_context(inputs={"input": f"Q{i}"}, outputs={"output": f"A{i}"})

        # Window should be cleared after summarization
        assert len(memory.messages) == 0
        # Summary should be set
        assert memory.summary == "Summary of conversation"

    def test_window_does_not_exceed_maxlen(self):
        """Test that window never exceeds maximum size."""
        self.mock_llm.invoke.return_value = MagicMock(content="Summary")
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=3)

        # Add more messages than window size
        for i in range(5):
            memory.save_context(inputs={"input": f"Q{i}"}, outputs={"output": f"A{i}"})

        # After summarization, window should be cleared
        # Then new messages should be added
        assert len(memory.messages) <= 3

    def test_save_context_extracts_input_text(self):
        """Test that input text is correctly extracted."""
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=5)

        input_text = "What is the capital of France?"
        memory.save_context(inputs={"input": input_text}, outputs={"output": "Paris"})

        assert memory.messages[0]["input"] == input_text

    def test_save_context_extracts_output_text(self):
        """Test that output text is correctly extracted."""
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=5)

        output_text = "The capital of France is Paris."
        memory.save_context(
            inputs={"input": "What is the capital?"}, outputs={"output": output_text}
        )

        assert memory.messages[0]["output"] == output_text


class TestSlidingWindowMemorySummarization:
    """Test summarization logic."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        self.mock_llm = MagicMock()

    def test_summarize_window_calls_llm(self):
        """Test that summarization calls the LLM."""
        self.mock_llm.invoke.return_value = MagicMock(content="Summary")
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=2)

        # Add messages to trigger summarization
        for i in range(2):
            memory.save_context(inputs={"input": f"Q{i}"}, outputs={"output": f"A{i}"})

        # LLM should be called
        assert self.mock_llm.invoke.called

    def test_summarize_window_with_content_attribute(self):
        """Test summarization when response has content attribute."""
        response = MagicMock()
        response.content = "Summarized content"
        self.mock_llm.invoke.return_value = response
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=2)

        # Fill window
        for i in range(2):
            memory.save_context(inputs={"input": f"Q{i}"}, outputs={"output": f"A{i}"})

        assert memory.summary == "Summarized content"

    def test_summarize_window_without_content_attribute(self):
        """Test summarization when response is string-convertible."""
        self.mock_llm.invoke.return_value = "String response"
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=2)

        # Fill window
        for i in range(2):
            memory.save_context(inputs={"input": f"Q{i}"}, outputs={"output": f"A{i}"})

        assert memory.summary == "String response"

    def test_summarize_window_clears_messages(self):
        """Test that messages are cleared after summarization."""
        self.mock_llm.invoke.return_value = MagicMock(content="Summary")
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=2)

        # Fill window
        for i in range(2):
            memory.save_context(inputs={"input": f"Q{i}"}, outputs={"output": f"A{i}"})

        # Messages should be cleared after summarization
        assert len(memory.messages) == 0

    def test_summarize_window_error_handling(self):
        """Test error handling during summarization."""
        self.mock_llm.invoke.side_effect = Exception("LLM error")
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=2)

        # Fill window - should not raise exception
        for i in range(2):
            memory.save_context(inputs={"input": f"Q{i}"}, outputs={"output": f"A{i}"})

        # Summary should contain window text (fallback)
        assert "Q0" in memory.summary
        assert "A0" in memory.summary

    def test_summarize_empty_window(self):
        """Test summarization behavior with empty window."""
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=2)

        # Manually call _summarize_window on empty window
        memory._summarize_window()

        # Should return without error
        assert memory.summary == ""

    def test_summarize_includes_all_messages(self):
        """Test that summarization includes all messages in window."""
        self.mock_llm.invoke.return_value = MagicMock(content="Summary")
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=3)

        # Add messages
        for i in range(3):
            memory.save_context(
                inputs={"input": f"Question {i}"}, outputs={"output": f"Answer {i}"}
            )

        # Check that invoke was called with window text
        call_args = self.mock_llm.invoke.call_args[0][0]
        assert "Question 0" in call_args
        assert "Question 2" in call_args
        assert "Answer 0" in call_args
        assert "Answer 2" in call_args


class TestSlidingWindowMemoryLoadMemoryVariables:
    """Test memory variables retrieval."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        self.mock_llm = MagicMock()

    def test_load_memory_variables_empty(self):
        """Test loading memory when everything is empty."""
        memory = SlidingWindowMemory(llm=self.mock_llm, memory_key="chat_history")

        variables = memory.load_memory_variables()

        assert "chat_history" in variables
        assert variables["chat_history"] == ""

    def test_load_memory_variables_with_summary_only(self):
        """Test loading memory with only summary."""
        memory = SlidingWindowMemory(llm=self.mock_llm, memory_key="chat_history")
        memory.summary = "Previous conversation summary"

        variables = memory.load_memory_variables()

        assert "Summary of previous conversation:" in variables["chat_history"]
        assert "Previous conversation summary" in variables["chat_history"]

    def test_load_memory_variables_with_recent_messages_only(self):
        """Test loading memory with only recent messages."""
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=5)

        memory.save_context(inputs={"input": "Hello"}, outputs={"output": "Hi"})

        variables = memory.load_memory_variables()

        assert "Recent messages:" in variables["chat_history"]
        assert "User: Hello" in variables["chat_history"]
        assert "Assistant: Hi" in variables["chat_history"]

    def test_load_memory_variables_with_both_summary_and_messages(self):
        """Test loading memory with both summary and recent messages."""
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=5)
        memory.summary = "Previous summary"

        memory.save_context(inputs={"input": "Q"}, outputs={"output": "A"})

        variables = memory.load_memory_variables()
        content = variables["chat_history"]

        assert "Summary of previous conversation:" in content
        assert "Previous summary" in content
        assert "Recent messages:" in content
        assert "User: Q" in content

    def test_load_memory_variables_custom_key(self):
        """Test loading memory with custom key."""
        memory = SlidingWindowMemory(llm=self.mock_llm, memory_key="conversation")

        variables = memory.load_memory_variables()

        assert "conversation" in variables
        assert "chat_history" not in variables

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

        # Check proper formatting
        assert "User: Question" in content
        assert "Assistant: Answer" in content


class TestSlidingWindowMemoryEdgeCases:
    """Test edge cases and special scenarios."""

    def setup_method(self):
        """Set up test fixtures before each test."""
        self.mock_llm = MagicMock()

    def test_very_long_message(self):
        """Test handling of very long messages."""
        self.mock_llm.invoke.return_value = MagicMock(content="Summary")
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=2)

        long_text = "A" * 10000
        memory.save_context(inputs={"input": long_text}, outputs={"output": long_text})

        assert memory.messages[0]["input"] == long_text
        assert memory.messages[0]["output"] == long_text

    def test_special_characters_in_message(self):
        """Test handling of special characters."""
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=5)

        special_text = "!@#$%^&*()_+-=[]{}|;:',.<>?/`~\n\t"
        memory.save_context(
            inputs={"input": special_text}, outputs={"output": special_text}
        )

        assert memory.messages[0]["input"] == special_text

    def test_unicode_characters(self):
        """Test handling of unicode characters."""
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=5)

        unicode_text = "Hello 世界 🚀 مرحبا"
        memory.save_context(
            inputs={"input": unicode_text}, outputs={"output": unicode_text}
        )

        assert memory.messages[0]["input"] == unicode_text

    def test_empty_input_output(self):
        """Test handling of empty input/output."""
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=5)

        memory.save_context(inputs={"input": ""}, outputs={"output": ""})

        assert memory.messages[0]["input"] == ""
        assert memory.messages[0]["output"] == ""

    def test_missing_input_key(self):
        """Test handling of missing input key."""
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=5)

        memory.save_context(inputs={}, outputs={"output": "Answer"})

        assert memory.messages[0]["input"] == ""

    def test_missing_output_key(self):
        """Test handling of missing output key."""
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=5)

        memory.save_context(inputs={"input": "Question"}, outputs={})

        assert memory.messages[0]["output"] == ""

    def test_window_size_one(self):
        """Test with window size of 1."""
        self.mock_llm.invoke.return_value = MagicMock(content="Summary")
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=1)

        memory.save_context(inputs={"input": "Q"}, outputs={"output": "A"})

        # Should trigger summarization immediately
        assert len(memory.messages) == 0
        assert memory.summary == "Summary"

    def test_multiple_windows(self):
        """Test multiple complete windows."""
        self.mock_llm.invoke.return_value = MagicMock(content="Summary")
        memory = SlidingWindowMemory(llm=self.mock_llm, window_size=2)

        # First window
        for i in range(2):
            memory.save_context(
                inputs={"input": f"Q1-{i}"}, outputs={"output": f"A1-{i}"}
            )

        # Second window
        for i in range(2):
            memory.save_context(
                inputs={"input": f"Q2-{i}"}, outputs={"output": f"A2-{i}"}
            )

        # Should have new summary
        assert memory.summary != ""
        # Messages should be cleared
        assert len(memory.messages) == 0

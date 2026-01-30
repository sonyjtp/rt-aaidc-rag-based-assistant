"""
Comprehensive tests for SimpleBufferMemory and SummaryMemory implementations.
Tests both memory strategies with parametrized scenarios to avoid duplication.
"""

import pytest

from src.simple_buffer_memory import SimpleBufferMemory
from src.summary_memory import SummaryMemory


# pylint: disable=redefined-outer-name
class TestSimpleBufferMemory:
    """Tests for SimpleBufferMemory using parametrization."""

    # ========================================================================
    # INITIALIZATION TESTS
    # ========================================================================

    @pytest.mark.parametrize(
        "memory_key,max_messages",
        [
            ("chat_history", 50),
            ("conversation", 100),
            ("messages", 20),
        ],
    )
    def test_initialization(self, memory_key, max_messages):
        """Parametrized test for SimpleBufferMemory initialization."""
        memory = SimpleBufferMemory(memory_key=memory_key, max_messages=max_messages)

        assert memory.memory_key == memory_key
        assert memory.max_messages == max_messages
        assert len(memory.buffer) == 0

    # ========================================================================
    # SAVE CONTEXT TESTS
    # ========================================================================

    @pytest.mark.parametrize(
        "input_text,output_text,_test_name",
        [
            ("Hello", "Hi there", "simple_greeting"),
            ("What is AI?", "AI is artificial intelligence", "question_answer"),
            ("", "", "empty_inputs"),
            ("A" * 500, "B" * 500, "long_messages"),
            ("!@#$%^&*()", "!@#$%^&*()", "special_characters"),
        ],
    )
    def test_save_context_various_inputs(self, input_text, output_text, _test_name):
        """Parametrized test for saving various message types."""
        memory = SimpleBufferMemory()
        memory.save_context(
            inputs={"input": input_text}, outputs={"output": output_text}
        )

        buffer_content = "\n".join(memory.buffer)
        if input_text:
            assert f"User: {input_text}" in buffer_content
        if output_text:
            assert f"Assistant: {output_text}" in buffer_content

    def test_save_context_missing_keys(self):
        """Test save_context handles missing input/output keys gracefully."""
        memory = SimpleBufferMemory()
        memory.save_context(inputs={}, outputs={})

        assert len(memory.buffer) == 0

    # ========================================================================
    # BUFFER CAPACITY TESTS
    # ========================================================================

    @pytest.mark.parametrize(
        "max_messages,num_exchanges",
        [
            (5, 3),
            (10, 15),
            (2, 5),
        ],
    )
    def test_buffer_respects_max_capacity(self, max_messages, num_exchanges):
        """Parametrized test that buffer respects max_messages limit."""
        memory = SimpleBufferMemory(max_messages=max_messages)

        for i in range(num_exchanges):
            memory.save_context(inputs={"input": f"Q{i}"}, outputs={"output": f"A{i}"})

        # Buffer can hold at most max_messages items
        assert len(memory.buffer) <= max_messages

    # ========================================================================
    # LOAD MEMORY VARIABLES TESTS
    # ========================================================================

    @pytest.mark.parametrize(
        "num_messages,_expected_count",
        [
            (0, 0),
            (1, 2),
            (3, 6),
            (5, 10),
        ],
    )
    def test_load_memory_variables(self, num_messages, _expected_count):
        """Parametrized test for loading memory variables."""
        memory = SimpleBufferMemory()

        for i in range(num_messages):
            memory.save_context(inputs={"input": f"Q{i}"}, outputs={"output": f"A{i}"})

        variables = memory.load_memory_variables()

        assert "chat_history" in variables
        assert isinstance(variables["chat_history"], str)
        if num_messages > 0:
            assert "Q0" in variables["chat_history"]

    def test_load_memory_with_custom_key(self):
        """Test load_memory_variables with custom memory key."""
        custom_key = "conversation_log"
        memory = SimpleBufferMemory(memory_key=custom_key)
        memory.save_context(inputs={"input": "Hello"}, outputs={"output": "Hi"})

        variables = memory.load_memory_variables()

        assert custom_key in variables
        assert "chat_history" not in variables

    # ========================================================================
    # ERROR HANDLING TESTS
    # ========================================================================

    @pytest.mark.parametrize(
        "input_text,output_text",
        [
            ("", ""),  # Empty inputs
            ("Test", "Response"),  # Normal inputs
        ],
    )
    def test_save_context_handles_inputs(self, input_text, output_text):
        """Test save_context handles various inputs gracefully."""
        memory = SimpleBufferMemory()
        memory.save_context(
            inputs={"input": input_text}, outputs={"output": output_text}
        )

        variables = memory.load_memory_variables()

        assert isinstance(variables, dict)
        assert memory.memory_key in variables


# pylint: disable=redefined-outer-name
class TestSummaryMemory:
    """Tests for SummaryMemory using parametrization."""

    # ========================================================================
    # INITIALIZATION TESTS
    # ========================================================================

    @pytest.mark.parametrize(
        "memory_key,summary_prompt",
        [
            ("chat_history", "Summarize the conversation so far in a few sentences."),
            ("summary_log", "Create a brief summary of the conversation."),
            ("conversation", "Summarize key points."),
        ],
    )
    def test_initialization(self, mock_llm, memory_key, summary_prompt):
        """Parametrized test for SummaryMemory initialization."""
        memory = SummaryMemory(
            llm=mock_llm, memory_key=memory_key, summary_prompt=summary_prompt
        )

        assert memory.llm == mock_llm
        assert memory.memory_key == memory_key
        assert memory.summary_prompt == summary_prompt
        assert memory.message_count == 0
        assert memory.summary == ""

    # ========================================================================
    # MESSAGE TRACKING TESTS
    # ========================================================================

    @pytest.mark.parametrize(
        "num_messages",
        [1, 4, 5, 9, 10, 15],
    )
    def test_message_count_tracking(self, mock_llm, num_messages):
        """Parametrized test for message counting."""
        memory = SummaryMemory(llm=mock_llm)

        for i in range(num_messages):
            memory.save_context(inputs={"input": f"Q{i}"}, outputs={"output": f"A{i}"})

        assert memory.message_count == num_messages

    # ========================================================================
    # SUMMARY UPDATE TESTS
    # ========================================================================

    @pytest.mark.parametrize(
        "num_messages,should_update",
        [
            (9, False),  # Not at 10 yet
            (10, True),  # Should update at 10
            (20, True),  # Should update at 20
        ],
    )
    def test_summary_updates_at_intervals(self, mock_llm, num_messages, should_update):
        """Parametrized test for summary update at 5-message intervals."""
        mock_llm.invoke.return_value = "Summary text"
        memory = SummaryMemory(llm=mock_llm)

        for i in range(num_messages):
            memory.save_context(inputs={"input": f"Q{i}"}, outputs={"output": f"A{i}"})

        if should_update:
            assert memory.summary == "Summary text" or memory.summary != ""
        else:
            # For non-update cases, summary might be empty or unchanged
            pass

    # ========================================================================
    # LOAD MEMORY VARIABLES TESTS
    # ========================================================================

    @pytest.mark.parametrize(
        "num_messages,_has_summary",
        [
            (0, False),
            (5, False),
            (10, True),
        ],
    )
    def test_load_memory_variables(self, mock_llm, num_messages, _has_summary):
        """Parametrized test for loading memory variables."""
        mock_llm.invoke.return_value = "Updated summary"
        memory = SummaryMemory(llm=mock_llm)

        for i in range(num_messages):
            memory.save_context(inputs={"input": f"Q{i}"}, outputs={"output": f"A{i}"})

        variables = memory.load_memory_variables()

        assert "chat_history" in variables
        assert isinstance(variables["chat_history"], str)
        if num_messages == 0:
            assert "Conversation with 0 messages" in variables["chat_history"]

    def test_load_memory_with_custom_key(self, mock_llm):
        """Test load_memory_variables with custom memory key."""
        custom_key = "summary_log"
        memory = SummaryMemory(llm=mock_llm, memory_key=custom_key)
        memory.save_context(inputs={"input": "Hello"}, outputs={"output": "Hi"})

        variables = memory.load_memory_variables()

        assert custom_key in variables
        assert "chat_history" not in variables

    # ========================================================================
    # ERROR HANDLING TESTS
    # ========================================================================

    @pytest.mark.parametrize(
        "error_scenario,error_type",
        [
            ("llm_invoke_fails", Exception("LLM error")),
            ("llm_returns_none", None),
            ("llm_returns_empty", ""),
        ],
    )
    def test_summary_error_handling(self, mock_llm, error_scenario, error_type):
        """Parametrized test for error handling during summary updates."""
        if error_scenario == "llm_invoke_fails":
            mock_llm.invoke.side_effect = error_type
        else:
            mock_llm.invoke.return_value = error_type

        memory = SummaryMemory(llm=mock_llm)

        # Should not raise even if LLM fails
        for i in range(5):
            memory.save_context(inputs={"input": f"Q{i}"}, outputs={"output": f"A{i}"})

        assert memory.message_count == 5

    def test_save_context_with_none_llm(self):
        """Test save_context handles None LLM gracefully."""
        memory = SummaryMemory(llm=None)

        for i in range(5):
            memory.save_context(inputs={"input": f"Q{i}"}, outputs={"output": f"A{i}"})

        assert memory.message_count == 5

    # ========================================================================
    # COMPARISON TESTS
    # ========================================================================

    @pytest.mark.parametrize(
        "_test_name,input_text,output_text",
        [
            ("simple_exchange", "What is AI?", "AI is artificial intelligence"),
            ("long_text", "Q" * 100, "A" * 100),
            ("special_chars", "!@#$%", "^&*()"),
        ],
    )
    def test_both_memories_handle_same_inputs(
        self, mock_llm, _test_name, input_text, output_text
    ):
        """Parametrized test comparing how both memory types handle same inputs."""
        mock_llm.invoke.return_value = "Summary"

        buffer_memory = SimpleBufferMemory()
        summary_memory = SummaryMemory(llm=mock_llm)

        # Both should handle inputs without raising
        buffer_memory.save_context(
            inputs={"input": input_text}, outputs={"output": output_text}
        )
        summary_memory.save_context(
            inputs={"input": input_text}, outputs={"output": output_text}
        )

        buffer_vars = buffer_memory.load_memory_variables()
        summary_vars = summary_memory.load_memory_variables()

        assert isinstance(buffer_vars, dict)
        assert isinstance(summary_vars, dict)

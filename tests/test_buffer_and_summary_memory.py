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

    @pytest.mark.parametrize(
        "memory_key,max_messages",
        [
            pytest.param("chat_history", 50, id="default_key"),
            pytest.param("conversation", 100, id="custom_key_conversation"),
            pytest.param("messages", 20, id="custom_key_messages"),
        ],
    )
    def test_initialization_with_keys(self, memory_key, max_messages):
        """Parametrized test for SimpleBufferMemory initialization with different keys."""
        memory = SimpleBufferMemory(memory_key=memory_key, max_messages=max_messages)

        assert memory.memory_key == memory_key
        assert memory.max_messages == max_messages
        assert len(memory.buffer) == 0

        # Verify custom key is used in load_memory_variables
        variables = memory.load_memory_variables()
        assert memory_key in variables

    @pytest.mark.parametrize(
        "input_text,output_text",
        [
            pytest.param("Hello", "Hi there", id="simple_greeting"),
            pytest.param("What is AI?", "AI is artificial intelligence", id="question_answer"),
            pytest.param("", "", id="empty_inputs"),
            pytest.param("A" * 500, "B" * 500, id="long_messages"),
            pytest.param("!@#$%^&*()", "!@#$%^&*()", id="special_characters"),
        ],
    )
    def test_save_context_and_retrieval(self, input_text, output_text):
        """Parametrized test for saving and retrieving various message types."""
        memory = SimpleBufferMemory()
        memory.save_context(inputs={"input": input_text}, outputs={"output": output_text})

        buffer_content = "\n".join(memory.buffer)
        variables = memory.load_memory_variables()

        # Verify message was saved (if not empty)
        if input_text:
            assert f"User: {input_text}" in buffer_content
        if output_text:
            assert f"Assistant: {output_text}" in buffer_content

        # Verify load_memory_variables works
        assert isinstance(variables, dict)
        assert memory.memory_key in variables

    @pytest.mark.parametrize(
        "max_messages,num_exchanges,expected_at_capacity",
        [
            pytest.param(5, 3, False, id="below_capacity"),
            pytest.param(10, 15, True, id="exceeds_capacity"),
            pytest.param(2, 5, True, id="small_capacity"),
        ],
    )
    def test_buffer_respects_max_capacity(self, max_messages, num_exchanges, expected_at_capacity):
        """Parametrized test that buffer respects max_messages limit."""
        memory = SimpleBufferMemory(max_messages=max_messages)

        for i in range(num_exchanges):
            memory.save_context(inputs={"input": f"Q{i}"}, outputs={"output": f"A{i}"})

        # Buffer respects max capacity
        assert len(memory.buffer) <= max_messages

        # If capacity exceeded, should be at max
        if expected_at_capacity:
            assert len(memory.buffer) == max_messages

    @pytest.mark.parametrize(
        "num_messages,has_content",
        [
            pytest.param(0, False, id="no_messages"),
            pytest.param(1, True, id="single_message"),
            pytest.param(5, True, id="multiple_messages"),
        ],
    )
    def test_load_memory_variables(self, num_messages, has_content):
        """Parametrized test for loading memory variables."""
        memory = SimpleBufferMemory()

        for i in range(num_messages):
            memory.save_context(inputs={"input": f"Q{i}"}, outputs={"output": f"A{i}"})

        variables = memory.load_memory_variables()

        assert "chat_history" in variables
        assert isinstance(variables["chat_history"], str)
        if has_content:
            assert "Q0" in variables["chat_history"]
        else:
            assert variables["chat_history"] == ""


# pylint: disable=redefined-outer-name
class TestSummaryMemory:
    """Tests for SummaryMemory using parametrization."""

    @pytest.mark.parametrize(
        "memory_key,summary_prompt",
        [
            pytest.param("chat_history", "Summarize the conversation so far in a few sentences.", id="default_key"),
            pytest.param("summary_log", "Create a brief summary of the conversation.", id="custom_key_summary_log"),
            pytest.param("conversation", "Summarize key points.", id="custom_key_conversation"),
        ],
    )
    def test_initialization_with_keys(self, mock_llm, memory_key, summary_prompt):
        """Parametrized test for SummaryMemory initialization with different keys and prompts."""
        memory = SummaryMemory(llm=mock_llm, memory_key=memory_key, summary_prompt=summary_prompt)

        assert memory.llm == mock_llm
        assert memory.memory_key == memory_key
        assert memory.summary_prompt == summary_prompt
        assert memory.message_count == 0
        assert memory.summary == ""

        # Verify custom key is used in load_memory_variables
        variables = memory.load_memory_variables()
        assert memory_key in variables

    @pytest.mark.parametrize(
        "num_messages,should_update",
        [
            pytest.param(1, False, id="single_message"),
            pytest.param(4, False, id="below_update_threshold"),
            pytest.param(5, False, id="at_first_threshold_not_triggered"),
            pytest.param(9, False, id="below_10_messages"),
            pytest.param(10, True, id="at_10_messages_updates"),
            pytest.param(15, True, id="above_10_messages"),
            pytest.param(20, True, id="at_20_messages"),
        ],
    )
    def test_message_count_and_summary_updates(self, mock_llm, num_messages, should_update):
        """Parametrized test for message tracking and summary updates at intervals."""
        mock_llm.invoke.return_value = "Summary text"
        memory = SummaryMemory(llm=mock_llm)

        for i in range(num_messages):
            memory.save_context(inputs={"input": f"Q{i}"}, outputs={"output": f"A{i}"})

        assert memory.message_count == num_messages

        if should_update:
            assert memory.summary == "Summary text" or memory.summary != ""

    @pytest.mark.parametrize(
        "num_messages",
        [
            pytest.param(0, id="no_messages"),
            pytest.param(5, id="multiple_messages"),
            pytest.param(10, id="at_summary_threshold"),
        ],
    )
    def test_load_memory_variables(self, mock_llm, num_messages):
        """Parametrized test for loading memory variables with different message counts."""
        mock_llm.invoke.return_value = "Updated summary"
        memory = SummaryMemory(llm=mock_llm)

        for i in range(num_messages):
            memory.save_context(inputs={"input": f"Q{i}"}, outputs={"output": f"A{i}"})

        variables = memory.load_memory_variables()

        assert "chat_history" in variables
        assert isinstance(variables["chat_history"], str)

    @pytest.mark.parametrize(
        "error_scenario,error_type",
        [
            pytest.param("llm_invoke_fails", Exception("LLM error"), id="llm_exception"),
            pytest.param("llm_returns_none", None, id="llm_none_return"),
            pytest.param("llm_returns_empty", "", id="llm_empty_return"),
        ],
    )
    def test_summary_error_handling_and_graceful_degradation(self, mock_llm, error_scenario, error_type):
        """Parametrized test for error handling and None LLM graceful degradation."""
        if error_scenario == "llm_invoke_fails":
            mock_llm.invoke.side_effect = error_type
        else:
            mock_llm.invoke.return_value = error_type

        memory = SummaryMemory(llm=mock_llm)

        # Should not raise even if LLM fails or returns unexpected values
        for i in range(5):
            memory.save_context(inputs={"input": f"Q{i}"}, outputs={"output": f"A{i}"})

        assert memory.message_count == 5

    def test_save_context_with_none_llm(self):
        """Test save_context handles None LLM gracefully."""
        memory = SummaryMemory(llm=None)

        for i in range(5):
            memory.save_context(inputs={"input": f"Q{i}"}, outputs={"output": f"A{i}"})

        assert memory.message_count == 5

    @pytest.mark.parametrize(
        "input_text,output_text",
        [
            pytest.param("What is AI?", "AI is artificial intelligence", id="simple_exchange"),
            pytest.param("Q" * 100, "A" * 100, id="long_text"),
            pytest.param("!@#$%", "^&*()", id="special_chars"),
        ],
    )
    def test_both_memories_handle_same_inputs(self, mock_llm, input_text, output_text):
        """Parametrized test comparing how both memory types handle same inputs."""
        mock_llm.invoke.return_value = "Summary"

        buffer_memory = SimpleBufferMemory()
        summary_memory = SummaryMemory(llm=mock_llm)

        # Both should handle inputs without raising
        buffer_memory.save_context(inputs={"input": input_text}, outputs={"output": output_text})
        summary_memory.save_context(inputs={"input": input_text}, outputs={"output": output_text})

        buffer_vars = buffer_memory.load_memory_variables()
        summary_vars = summary_memory.load_memory_variables()

        assert isinstance(buffer_vars, dict)
        assert isinstance(summary_vars, dict)

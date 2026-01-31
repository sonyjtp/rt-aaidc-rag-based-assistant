"""
Unit tests for RAGAssistant class.
Tests initialization, document handling, invocation, and error handling.
"""

import re
import time as time_module
from unittest.mock import MagicMock, patch

import pytest

from src.rag_assistant import RAGAssistant


@pytest.fixture
def mock_components():
    """Fixture providing all mocked RAGAssistant components."""
    with patch("src.rag_assistant.initialize_llm") as mock_llm, patch(
        "src.rag_assistant.VectorDB"
    ) as mock_vectordb, patch("src.rag_assistant.MemoryManager") as mock_memory, patch(
        "src.rag_assistant.ReasoningStrategyLoader"
    ) as mock_reasoning:
        # Setup default return values
        mock_llm.return_value.model_name = "test-model"
        mock_vectordb_instance = MagicMock()
        mock_vectordb.return_value = mock_vectordb_instance
        mock_memory_instance = MagicMock()
        mock_memory_instance.memory = MagicMock()
        mock_memory_instance.strategy = "summarization_sliding_window"
        mock_memory.return_value = mock_memory_instance

        mock_reasoning_instance = MagicMock()
        mock_reasoning_instance.get.return_value = "RAG-Enhanced Reasoning"
        mock_reasoning_instance.active_strategy = "rag_enhanced"
        mock_reasoning.return_value = mock_reasoning_instance

        yield {
            "llm": mock_llm,
            "llm_instance": mock_llm.return_value,
            "vectordb": mock_vectordb,
            "vectordb_instance": mock_vectordb_instance,
            "memory": mock_memory,
            "memory_instance": mock_memory_instance,
            "reasoning": mock_reasoning,
        }


# pylint: disable=redefined-outer-name,protected-access, too-many-public-methods
class TestRAGAssistant:
    """Unified test class for RAGAssistant covering initialization, documents, invocation, and error handling."""

    # ========================================================================
    # INITIALIZATION TESTS
    # ========================================================================

    def test_initialization_success(
        self, mock_components
    ):  # pylint: disable=unused-argument
        """Test successful initialization of RAGAssistant."""
        assistant = RAGAssistant()

        assert assistant.llm is not None
        assert assistant.vector_db is not None
        assert assistant.memory_manager is not None
        assert assistant.chain is not None
        assert assistant.prompt_template is not None

    @pytest.mark.parametrize(
        "component,mock_attr,expected_none",
        [
            ("llm", "llm", True),
            ("vectordb", "vectordb", False),
            ("memory", "memory", False),
            ("reasoning", "reasoning", False),
        ],
    )
    def test_component_initialization_failure(
        self, mock_components, component, mock_attr, expected_none
    ):
        """Parametrized test for component initialization failures."""
        mock_components[mock_attr].side_effect = Exception(
            f"{component} initialization failed"
        )

        assistant = RAGAssistant()

        if expected_none:
            assert assistant.llm is None
            assert assistant.chain is None
        else:
            assert (
                getattr(
                    assistant,
                    component.replace("vectordb", "vector_db")
                    .replace("memory", "memory_manager")
                    .replace("reasoning", "reasoning_strategy"),
                )
                is None
            )

    @pytest.mark.parametrize(
        "model_name",
        ["gpt-4o-mini", "llama-3.1-8b-instant", "gemini-pro"],
    )
    def test_llm_initialization_with_models(self, mock_components, model_name):
        """Parametrized test for LLM initialization with different models."""
        mock_components["llm_instance"].model_name = model_name

        assistant = RAGAssistant()

        mock_components["llm"].assert_called_once()
        assert assistant.llm.model_name == model_name

    def test_chain_and_template_building(
        self, mock_components
    ):  # pylint: disable=unused-argument
        """Test that prompt template and chain are built correctly."""
        assistant = RAGAssistant()

        assert assistant.prompt_template is not None
        assert assistant.chain is not None

    # ========================================================================
    # DOCUMENT HANDLING TESTS
    # ========================================================================

    @pytest.mark.parametrize(
        "documents,_doc_type",
        [
            (["Document 1", "Document 2", "Document 3"], "string_list"),
            (
                [
                    {"title": "Doc1", "content": "Content1"},
                    {"title": "Doc2", "content": "Content2"},
                ],
                "dict_list",
            ),
            ([], "empty_list"),
        ],
    )
    def test_add_documents(self, mock_components, documents, _doc_type):
        """Parametrized test for adding documents of various types."""
        assistant = RAGAssistant()
        assistant.add_documents(documents)

        mock_components["vectordb_instance"].add_documents.assert_called_once_with(
            documents
        )

    # ========================================================================
    # INVOKE METHOD TESTS
    # ========================================================================

    @pytest.mark.parametrize(
        "search_result",
        [
            (
                {
                    "documents": [["Document 1 content", "Document 2 content"]],
                    "distances": [[0.1, 0.2]],
                }
            ),
            ({"documents": [[]], "distances": [[]]}),
            ({"documents": [], "distances": []}),
        ],
    )
    def test_invoke_various_contexts(self, mock_components, search_result):
        """Parametrized test for invoke with various search results."""
        mock_components["vectordb_instance"].search.return_value = search_result

        assistant = RAGAssistant()
        assistant.chain = MagicMock()
        assistant.chain.invoke.return_value = "Test response"

        response = assistant.invoke("Test query")

        assert response == "Test response"
        mock_components["vectordb_instance"].search.assert_called_once()

    def test_invoke_context_retrieval(self, mock_components):
        """Test that invoke retrieves context from vector database."""
        mock_components["memory_instance"].get_memory_variables.return_value = {
            "chat_history": ""
        }
        mock_components["vectordb_instance"].search.return_value = {
            "documents": [["Context about AI"]],
            "distances": [[0.1]],
        }

        assistant = RAGAssistant()
        assistant.chain = MagicMock()
        assistant.chain.invoke.return_value = "Response"

        assistant.invoke("Test query")

        mock_components["vectordb_instance"].search.assert_called_once()
        call_kwargs = mock_components["vectordb_instance"].search.call_args.kwargs
        assert call_kwargs["n_results"] == 3
        assert call_kwargs["maximum_distance"] == 0.6
        assert "Test query" in call_kwargs["query"]

    @pytest.mark.parametrize("n_results", [5, 10, 20])
    def test_invoke_custom_n_results(self, mock_components, n_results):
        """Parametrized test for invoke with custom n_results."""
        mock_components["memory_instance"].get_memory_variables.return_value = {
            "chat_history": ""
        }
        mock_components["vectordb_instance"].search.return_value = {
            "documents": [["Content"]],
            "distances": [[0.1]],
        }

        assistant = RAGAssistant()
        assistant.chain = MagicMock()
        assistant.chain.invoke.return_value = "Response"

        assistant.invoke("What about this?", n_results=n_results)

        mock_components["vectordb_instance"].search.assert_called_once_with(
            query="What about this?", n_results=n_results, maximum_distance=0.6
        )

    def test_invoke_saves_to_memory(self, mock_components):
        """Test that invoke saves conversation to memory."""
        mock_components["vectordb_instance"].search.return_value = {
            "documents": [["Content"]],
            "distances": [[0.1]],
        }

        assistant = RAGAssistant()
        assistant.chain = MagicMock()
        assistant.chain.invoke.return_value = "Assistant response"

        query = "What is this?"
        assistant.invoke(query)

        mock_components["memory_instance"].add_message.assert_called_once_with(
            input_text=query, output_text="Assistant response"
        )

    def test_invoke_low_similarity_rejects_answer(self, mock_components):
        """Test invoke handles low-similarity answers by passing empty context to LLM."""
        mock_components["vectordb_instance"].search.return_value = {
            "documents": [["Some content"]],
            "distances": [[0.9]],
        }

        assistant = RAGAssistant()
        assistant.chain = MagicMock()
        assistant.chain.invoke.return_value = (
            "I'm sorry, that information is not known to me."
        )

        response = assistant.invoke("What is this?")

        assert response == "I'm sorry, that information is not known to me."
        assistant.chain.invoke.assert_called_once()

    # ========================================================================
    # QUERY AUGMENTATION TESTS
    # ========================================================================

    @pytest.mark.parametrize(
        "chat_history,expected_augment",
        [
            ("", False),
            ("No previous conversation context.", False),
            (None, False),
            ("Previous question: What is post-quantum cryptography?", True),
        ],
    )
    def test_augment_query_conditions(
        self, mock_components, chat_history, expected_augment
    ):
        """Parametrized test for query augmentation under various conditions."""
        mock_components["memory_instance"].get_memory_variables.return_value = {
            "chat_history": chat_history
        }

        assistant = RAGAssistant()
        query = "Who are famous for it in India?"
        result = assistant._augment_query_with_context(query)

        if expected_augment:
            assert chat_history in result
            assert "Current question:" in result
        else:
            assert result == query

    def test_augment_query_no_memory_manager(self):
        """Test that query is returned unchanged when no memory manager exists."""
        assistant = RAGAssistant()
        assistant.memory_manager = None

        original_query = "Who are famous for it in India?"
        result = assistant._augment_query_with_context(original_query)

        assert result == original_query

    def test_augment_query_exception_handling(self, mock_components):
        """Test that exceptions during context retrieval are handled gracefully."""
        mock_components["memory_instance"].get_memory_variables.side_effect = Exception(
            "Memory retrieval error"
        )

        assistant = RAGAssistant()
        original_query = "What is this topic?"
        result = assistant._augment_query_with_context(original_query)

        assert result == original_query

    @pytest.mark.parametrize(
        "chat_history,query",
        [
            ("Previous: What is AI?", "Can you explain more?"),
            (
                "Q: What is Python?\nA: Python is a programming language.",
                "Is it good for beginners?",
            ),
            (
                "Discussion about machine learning models",
                "What's the most popular one?",
            ),
        ],
    )
    def test_augment_query_with_various_contexts(
        self, mock_components, chat_history, query
    ):
        """Parametrized test for augmenting queries with various chat histories."""
        mock_components["memory_instance"].get_memory_variables.return_value = {
            "chat_history": chat_history
        }

        assistant = RAGAssistant()
        result = assistant._augment_query_with_context(query)

        assert chat_history in result
        assert query in result
        assert "Current question:" in result

    def test_augment_query_integration_with_invoke(self, mock_components):
        """Test that augmented query is used in the invoke method."""
        chat_history = "Previous: What is quantum computing?"
        mock_components["memory_instance"].get_memory_variables.return_value = {
            "chat_history": chat_history
        }
        mock_components["vectordb_instance"].search.return_value = {
            "documents": [["Quantum computing content"]],
            "distances": [[0.1]],
        }

        assistant = RAGAssistant()
        assistant.chain = MagicMock()
        assistant.chain.invoke.return_value = "Response"

        assistant.invoke("Tell me more about it")

        search_call_args = mock_components["vectordb_instance"].search.call_args
        search_query = search_call_args.kwargs["query"]

        assert chat_history in search_query
        assert "Tell me more about it" in search_query

    def test_augment_query_logs_debug_message(self, mock_components):
        """Test that augmentation logs a debug message."""
        chat_history = "Previous conversation"
        mock_components["memory_instance"].get_memory_variables.return_value = {
            "chat_history": chat_history
        }

        assistant = RAGAssistant()

        with patch("src.rag_assistant.logger") as mock_logger:
            assistant._augment_query_with_context("Current query")
            mock_logger.debug.assert_called_with(
                "Query augmented with chat history context"
            )

    # ========================================================================
    # ERROR HANDLING TESTS
    # ========================================================================

    def test_invoke_search_error_with_graceful_fallback(self, mock_components):
        """Test invoke handles search errors gracefully."""
        mock_components["vectordb_instance"].search.side_effect = ValueError(
            "Collection expecting embedding with dimension of 768, got 384"
        )

        assistant = RAGAssistant()
        assistant.chain = MagicMock()
        assistant.chain.invoke.return_value = (
            "I don't have enough information to answer your question."
        )

        response = assistant.invoke("Test query")

        # When search fails, context is empty, so LLM should handle it gracefully
        # The important thing is that the actual error details are not exposed
        assert isinstance(response, str)
        assert "768" not in response
        assert "384" not in response

    def test_invoke_chain_error_returns_generic_message(self, mock_components):
        """Test invoke returns generic error message when chain fails."""
        mock_components["vectordb_instance"].search.return_value = {
            "documents": [["Content"]],
            "distances": [[0.1]],
        }

        assistant = RAGAssistant()
        assistant.chain = MagicMock()
        assistant.chain.invoke.side_effect = Exception("LLM API timeout")

        response = assistant.invoke("Test query")

        # Should return the generic error message without exposing the actual error
        assert (
            response == "Unable to search for relevant information. Please try again."
        )
        assert "timeout" not in response.lower()

    # ========================================================================
    # RESPONSE TIMING TESTS
    # ========================================================================

    def test_invoke_logs_response_time(self, mock_components):
        """Test that invoke measures and logs response time at debug level."""
        mock_components["vectordb_instance"].search.return_value = {
            "documents": [["Content"]],
            "distances": [[0.1]],
        }

        assistant = RAGAssistant()
        assistant.chain = MagicMock()
        assistant.chain.invoke.return_value = "Test response"

        with patch("src.rag_assistant.logger") as mock_logger:
            assistant.invoke("Test query")

            debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
            response_time_logged = any(
                "Response time:" in call and "ms" in call for call in debug_calls
            )
            assert response_time_logged, "Response time not logged at debug level"

    def test_invoke_response_time_format(self, mock_components):
        """Test that response time is logged in correct format (milliseconds)."""
        mock_components["vectordb_instance"].search.return_value = {
            "documents": [["Content"]],
            "distances": [[0.1]],
        }

        assistant = RAGAssistant()
        assistant.chain = MagicMock()
        assistant.chain.invoke.return_value = "Response"

        with patch("src.rag_assistant.logger") as mock_logger:
            assistant.invoke("Query")

            debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
            time_log = next(
                (call for call in debug_calls if "Response time:" in call), None
            )
            assert time_log is not None, "Response time log not found"
            assert "ms" in time_log, "Response time should be in milliseconds"

    def test_invoke_response_time_reasonable(self, mock_components):
        """Test that response time is reasonable (between 0 and 60 seconds)."""

        mock_components["vectordb_instance"].search.return_value = {
            "documents": [["Content"]],
            "distances": [[0.1]],
        }

        assistant = RAGAssistant()
        assistant.chain = MagicMock()
        assistant.chain.invoke.return_value = "Response"

        with patch("src.rag_assistant.logger") as mock_logger:
            start = time_module.time()
            assistant.invoke("Query")
            elapsed = (time_module.time() - start) * 1000

            debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
            time_log = next(
                (call for call in debug_calls if "Response time:" in call), None
            )
            assert time_log is not None

            match = re.search(r"Response time: ([\d.]+)ms", time_log)
            assert match, f"Could not parse response time from: {time_log}"

            logged_time = float(match.group(1))

            assert (
                0 < logged_time < 60000
            ), f"Response time {logged_time}ms is unreasonable"
            assert (
                logged_time <= elapsed + 100
            ), "Logged time shouldn't exceed actual time by much"

"""
Unit tests for RAGAssistant class.
Tests initialization, document handling, and invocation.
"""

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
        mock_reasoning.return_value.get_strategy_name.return_value = (
            "RAG-Enhanced Reasoning"
        )

        yield {
            "llm": mock_llm,
            "llm_instance": mock_llm.return_value,
            "vectordb": mock_vectordb,
            "vectordb_instance": mock_vectordb_instance,
            "memory": mock_memory,
            "memory_instance": mock_memory_instance,
            "reasoning": mock_reasoning,
        }


# pylint: disable=redefined-outer-name
class TestRAGAssistantInitialization:
    """Test RAGAssistant initialization and setup."""

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
        "model_name",
        [
            "gpt-4o-mini",
            "llama-3.1-8b-instant",
            "gemini-pro",
        ],
    )
    def test_llm_initialization_with_models(self, mock_components, model_name):
        """Parametrized test for LLM initialization with different models."""
        mock_components["llm_instance"].model_name = model_name

        assistant = RAGAssistant()

        mock_components["llm"].assert_called_once()
        assert assistant.llm.model_name == model_name

    def test_vectordb_initialization(self, mock_components):
        """Test that VectorDB is properly initialized."""
        assistant = RAGAssistant()

        mock_components["vectordb"].assert_called_once()
        assert assistant.vector_db is not None

    def test_memory_manager_initialization(self, mock_components):
        """Test that MemoryManager is properly initialized."""
        assistant = RAGAssistant()

        mock_components["memory"].assert_called_once_with(llm=assistant.llm)
        assert assistant.memory_manager is not None

    def test_reasoning_strategy_initialization(self, mock_components):
        """Test that ReasoningStrategyLoader is properly initialized."""
        assistant = RAGAssistant()

        mock_components["reasoning"].assert_called_once()
        assert assistant.reasoning_strategy is not None

    def test_chain_building(self, mock_components):  # pylint: disable=unused-argument
        """Test that prompt template and chain are built correctly."""
        assistant = RAGAssistant()

        assert assistant.prompt_template is not None
        assert assistant.chain is not None


# pylint: disable=redefined-outer-name
class TestRAGAssistantAddDocuments:
    """Test document addition functionality."""

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
    def test_add_documents(
        self,
        mock_components,
        documents,
        _doc_type,
    ):
        """Parametrized test for adding documents of various types."""
        assistant = RAGAssistant()
        assistant.add_documents(documents)

        mock_components["vectordb_instance"].add_documents.assert_called_once_with(
            documents
        )

    def test_add_documents_string_list(self, mock_components):
        """Test adding documents as list of strings."""
        documents = ["Document 1", "Document 2", "Document 3"]
        assistant = RAGAssistant()
        assistant.add_documents(documents)

        mock_components["vectordb_instance"].add_documents.assert_called_once_with(
            documents
        )

    def test_add_documents_dict_list(self, mock_components):
        """Test adding documents as list of dicts."""
        documents = [
            {"title": "Doc1", "content": "Content1"},
            {"title": "Doc2", "content": "Content2"},
        ]
        assistant = RAGAssistant()
        assistant.add_documents(documents)

        mock_components["vectordb_instance"].add_documents.assert_called_once_with(
            documents
        )


# pylint: disable=redefined-outer-name
class TestRAGAssistantInvoke:
    """Test RAGAssistant invoke method."""

    @pytest.mark.parametrize(
        "search_result,has_documents",
        [
            (
                {
                    "documents": [["Document 1 content", "Document 2 content"]],
                    "distances": [[0.1, 0.2]],
                },
                True,
            ),
            ({"documents": [[]], "distances": [[]]}, False),
            ({"documents": [], "distances": []}, False),
        ],
    )
    def test_invoke_various_contexts(
        self,
        mock_components,
        search_result,
        has_documents,
    ):
        """Parametrized test for invoke with various search results."""
        mock_components["vectordb_instance"].search.return_value = search_result

        assistant = RAGAssistant()
        assistant.chain = MagicMock()
        assistant.chain.invoke.return_value = "Test response"

        response = assistant.invoke("Test query")

        if has_documents:
            assert response == "Test response"
        else:
            # Empty documents should trigger LLM with empty context
            # System prompts will guide the response to "not known to me"
            assert response == "Test response"
            assistant.chain.invoke.assert_called_once()

        mock_components["vectordb_instance"].search.assert_called_once()

    def test_invoke_context_retrieval(self, mock_components):
        """Test that invoke retrieves context from vector database."""
        mock_components["vectordb_instance"].search.return_value = {
            "documents": [["Context about AI"]],
            "distances": [[0.1]],
        }

        assistant = RAGAssistant()
        assistant.chain = MagicMock()
        assistant.chain.invoke.return_value = "Response"

        assistant.invoke("Test query")

        mock_components["vectordb_instance"].search.assert_called_once_with(
            query="Test query", n_results=5, maximum_distance=0.7
        )

    @pytest.mark.parametrize(
        "n_results",
        [5, 10, 20],
    )
    def test_invoke_custom_n_results(self, mock_components, n_results):
        """Parametrized test for invoke with custom n_results."""
        mock_components["vectordb_instance"].search.return_value = {
            "documents": [["Content"]],
            "distances": [[0.1]],
        }

        assistant = RAGAssistant()
        assistant.chain = MagicMock()
        assistant.chain.invoke.return_value = "Response"

        # Use query with action verb to avoid auto-prefixing
        assistant.invoke("What about this?", n_results=n_results)

        mock_components["vectordb_instance"].search.assert_called_once_with(
            query="What about this?", n_results=n_results, maximum_distance=0.7
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

        # Use query with action verb to avoid auto-prefixing
        query = "What is this?"
        assistant.invoke(query)

        mock_components["memory_instance"].add_message.assert_called_once_with(
            input_text=query, output_text="Assistant response"
        )

    def test_invoke_low_similarity_rejects_answer(self, mock_components):
        """Test invoke handles low-similarity answers by passing empty context to LLM."""
        # Simulate low similarity (distance > threshold)
        mock_components["vectordb_instance"].search.return_value = {
            "documents": [["Some content"]],
            "distances": [[0.9]],  # High distance = low similarity
        }

        assistant = RAGAssistant()
        assistant.chain = MagicMock()
        assistant.chain.invoke.return_value = (
            "I'm sorry, that information is not known to me."
        )

        # Use query with action verb so it's a REGULAR question (not VAGUE/META)
        response = assistant.invoke("What is this?")

        # Should call chain with empty context, letting system prompts handle rejection
        assert response == "I'm sorry, that information is not known to me."
        assistant.chain.invoke.assert_called_once()


# pylint: disable=redefined-outer-name
class TestRAGAssistantExceptionHandling:
    """Test exception handling in RAGAssistant."""

    def test_vectordb_connection_error(self, mock_components):
        """Test handling of VectorDB connection errors."""
        mock_components["vectordb"].side_effect = Exception(
            "Failed to connect to ChromaDB"
        )

        # Should raise exception during initialization
        with pytest.raises(Exception):
            RAGAssistant()

    def test_reasoning_strategy_load_error(self, mock_components):
        """Test handling of ReasoningStrategyLoader errors."""
        mock_components["reasoning"].side_effect = Exception(
            "Failed to load reasoning strategy"
        )

        # Should handle gracefully (log error but continue)
        assistant = RAGAssistant()
        assert assistant.reasoning_strategy is None
        assert assistant.chain is not None

    def test_invoke_search_error_with_graceful_fallback(self, mock_components):
        """Test invoke handles search errors gracefully."""
        mock_components["vectordb_instance"].search.side_effect = ValueError(
            "Collection expecting embedding with dimension of 768, got 384"
        )

        assistant = RAGAssistant()
        assistant.chain = MagicMock()

        response = assistant.invoke("Test query")

        # Should return generic error message, not reveal technical details
        assert "unable to search" in response.lower()
        assert "configuration" in response.lower()
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

        # Should return generic error message
        assert "encountered an error" in response.lower()
        assert "timeout" not in response.lower()

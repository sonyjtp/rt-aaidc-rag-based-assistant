"""
Unit tests for RAGAssistant class.
Tests initialization, document handling, and invocation.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.rag_assistant import RAGAssistant


class TestRAGAssistantInitialization:
    """Test RAGAssistant initialization and setup."""

    @patch("src.rag_assistant.initialize_llm")
    @patch("src.rag_assistant.VectorDB")
    @patch("src.rag_assistant.MemoryManager")
    @patch("src.rag_assistant.ReasoningStrategyLoader")
    def test_initialization_success(self, mock_reasoning, mock_memory, mock_llm):
        """Test successful initialization of RAGAssistant."""
        # Setup mocks
        mock_llm.return_value.model_name = "gpt-4o-mini"
        mock_memory.return_value.memory = MagicMock()
        mock_memory.return_value.strategy = "summarization_sliding_window"
        mock_reasoning.return_value.get_strategy_name.return_value = (
            "RAG-Enhanced Reasoning"
        )

        # Initialize assistant
        assistant = RAGAssistant()

        # Verify all components initialized
        assert assistant.llm is not None
        assert assistant.vector_db is not None
        assert assistant.memory_manager is not None
        assert assistant.chain is not None
        assert assistant.prompt_template is not None

    @patch("src.rag_assistant.initialize_llm")
    @patch("src.rag_assistant.VectorDB")
    @patch("src.rag_assistant.MemoryManager")
    @patch("src.rag_assistant.ReasoningStrategyLoader")
    def test_llm_initialization(self, mock_memory, mock_llm):
        """Test that LLM is properly initialized."""
        mock_llm.return_value.model_name = "llama-3.1-8b-instant"
        mock_memory.return_value.memory = MagicMock()

        assistant = RAGAssistant()

        # Verify LLM initialization
        mock_llm.assert_called_once()
        assert assistant.llm.model_name == "llama-3.1-8b-instant"

    @patch("src.rag_assistant.initialize_llm")
    @patch("src.rag_assistant.VectorDB")
    @patch("src.rag_assistant.MemoryManager")
    @patch("src.rag_assistant.ReasoningStrategyLoader")
    def test_vectordb_initialization(self, mock_memory, mock_vectordb, mock_llm):
        """Test that VectorDB is properly initialized."""
        mock_llm.return_value.model_name = "test-model"
        mock_memory.return_value.memory = MagicMock()

        assistant = RAGAssistant()

        # Verify VectorDB initialization
        mock_vectordb.assert_called_once()
        assert assistant.vector_db is not None

    @patch("src.rag_assistant.initialize_llm")
    @patch("src.rag_assistant.VectorDB")
    @patch("src.rag_assistant.MemoryManager")
    @patch("src.rag_assistant.ReasoningStrategyLoader")
    def test_memory_manager_initialization(self, mock_memory, mock_llm):
        """Test that MemoryManager is properly initialized."""
        mock_llm.return_value.model_name = "test-model"
        mock_memory.return_value.memory = MagicMock()
        mock_memory.return_value.strategy = "summarization_sliding_window"

        assistant = RAGAssistant()

        # Verify MemoryManager initialization
        mock_memory.assert_called_once_with(llm=assistant.llm)
        assert assistant.memory_manager is not None

    @patch("src.rag_assistant.initialize_llm")
    @patch("src.rag_assistant.VectorDB")
    @patch("src.rag_assistant.MemoryManager")
    @patch("src.rag_assistant.ReasoningStrategyLoader")
    def test_reasoning_strategy_initialization(
        self, mock_reasoning, mock_memory, mock_llm
    ):
        """Test that ReasoningStrategyLoader is properly initialized."""
        mock_llm.return_value.model_name = "test-model"
        mock_memory.return_value.memory = MagicMock()
        mock_reasoning.return_value.get_strategy_name.return_value = (
            "RAG-Enhanced Reasoning"
        )

        assistant = RAGAssistant()

        # Verify ReasoningStrategyLoader initialization
        mock_reasoning.assert_called_once()
        assert assistant.reasoning_strategy is not None

    @patch("src.rag_assistant.initialize_llm")
    @patch("src.rag_assistant.VectorDB")
    @patch("src.rag_assistant.MemoryManager")
    @patch("src.rag_assistant.ReasoningStrategyLoader")
    def test_chain_building(self, mock_memory, mock_llm):
        """Test that prompt template and chain are built correctly."""
        mock_llm.return_value.model_name = "test-model"
        mock_memory.return_value.memory = MagicMock()

        assistant = RAGAssistant()

        # Verify chain is built
        assert assistant.prompt_template is not None
        assert assistant.chain is not None


class TestRAGAssistantAddDocuments:
    """Test document addition functionality."""

    @patch("src.rag_assistant.initialize_llm")
    @patch("src.rag_assistant.VectorDB")
    @patch("src.rag_assistant.MemoryManager")
    @patch("src.rag_assistant.ReasoningStrategyLoader")
    def test_add_documents_string_list(self, mock_memory, mock_vectordb, mock_llm):
        """Test adding documents as list of strings."""
        mock_llm.return_value.model_name = "test-model"
        mock_vectordb_instance = MagicMock()
        mock_vectordb.return_value = mock_vectordb_instance
        mock_memory.return_value.memory = MagicMock()

        assistant = RAGAssistant()
        documents = ["Document 1", "Document 2", "Document 3"]
        assistant.add_documents(documents)

        # Verify documents added
        mock_vectordb_instance.add_documents.assert_called_once_with(documents)

    @patch("src.rag_assistant.initialize_llm")
    @patch("src.rag_assistant.VectorDB")
    @patch("src.rag_assistant.MemoryManager")
    @patch("src.rag_assistant.ReasoningStrategyLoader")
    def test_add_documents_dict_list(self, mock_memory, mock_vectordb, mock_llm):
        """Test adding documents as list of dicts."""
        mock_llm.return_value.model_name = "test-model"
        mock_vectordb_instance = MagicMock()
        mock_vectordb.return_value = mock_vectordb_instance
        mock_memory.return_value.memory = MagicMock()

        assistant = RAGAssistant()
        documents = [
            {"title": "Doc1", "content": "Content1"},
            {"title": "Doc2", "content": "Content2"},
        ]
        assistant.add_documents(documents)

        # Verify documents added
        mock_vectordb_instance.add_documents.assert_called_once_with(documents)

    @patch("src.rag_assistant.initialize_llm")
    @patch("src.rag_assistant.VectorDB")
    @patch("src.rag_assistant.MemoryManager")
    @patch("src.rag_assistant.ReasoningStrategyLoader")
    def test_add_empty_documents(self, mock_memory, mock_vectordb, mock_llm):
        """Test adding empty document list."""
        mock_llm.return_value.model_name = "test-model"
        mock_vectordb_instance = MagicMock()
        mock_vectordb.return_value = mock_vectordb_instance
        mock_memory.return_value.memory = MagicMock()

        assistant = RAGAssistant()
        documents = []
        assistant.add_documents(documents)

        # Verify empty list is handled
        mock_vectordb_instance.add_documents.assert_called_once_with(documents)


class TestRAGAssistantInvoke:
    """Test RAGAssistant invoke method."""

    @patch("src.rag_assistant.initialize_llm")
    @patch("src.rag_assistant.VectorDB")
    @patch("src.rag_assistant.MemoryManager")
    @patch("src.rag_assistant.ReasoningStrategyLoader")
    def test_invoke_basic(self, mock_memory, mock_vectordb, mock_llm):
        """Test basic invoke functionality."""
        mock_llm.return_value.model_name = "test-model"
        mock_vectordb_instance = MagicMock()
        mock_vectordb.return_value = mock_vectordb_instance
        # Search returns nested lists - flatten them
        mock_vectordb_instance.search.return_value = {
            "documents": [["Document 1 content", "Document 2 content"]]
        }
        mock_memory.return_value.memory = MagicMock()
        mock_memory.return_value.add_message = MagicMock()

        # Mock the chain invoke
        assistant = RAGAssistant()
        assistant.chain = MagicMock()
        assistant.chain.invoke.return_value = "Test response"

        response = assistant.invoke("What is AI?")

        # Verify response
        assert response == "Test response"
        mock_vectordb_instance.search.assert_called_once()

    @patch("src.rag_assistant.initialize_llm")
    @patch("src.rag_assistant.VectorDB")
    @patch("src.rag_assistant.MemoryManager")
    @patch("src.rag_assistant.ReasoningStrategyLoader")
    def test_invoke_context_retrieval(self, mock_memory, mock_vectordb, mock_llm):
        """Test that invoke retrieves context from vector database."""
        mock_llm.return_value.model_name = "test-model"
        mock_vectordb_instance = MagicMock()
        mock_vectordb.return_value = mock_vectordb_instance
        mock_vectordb_instance.search.return_value = {
            "documents": [["Context about AI", "More context"]]
        }
        mock_memory.return_value.memory = MagicMock()

        assistant = RAGAssistant()
        assistant.chain = MagicMock()
        assistant.chain.invoke.return_value = "Response"

        assistant.invoke("Test query")

        # Verify search was called with correct query
        mock_vectordb_instance.search.assert_called_once_with(
            query="Test query", n_results=3
        )

    @patch("src.rag_assistant.initialize_llm")
    @patch("src.rag_assistant.VectorDB")
    @patch("src.rag_assistant.MemoryManager")
    @patch("src.rag_assistant.ReasoningStrategyLoader")
    def test_invoke_custom_n_results(self, mock_memory, mock_vectordb, mock_llm):
        """Test invoke with custom n_results parameter."""
        mock_llm.return_value.model_name = "test-model"
        mock_vectordb_instance = MagicMock()
        mock_vectordb.return_value = mock_vectordb_instance
        mock_vectordb_instance.search.return_value = {"documents": [[]]}
        mock_memory.return_value.memory = MagicMock()

        assistant = RAGAssistant()
        assistant.chain = MagicMock()
        assistant.chain.invoke.return_value = "Response"

        assistant.invoke("Test query", n_results=5)

        # Verify n_results parameter is used
        mock_vectordb_instance.search.assert_called_once_with(
            query="Test query", n_results=5
        )

    @patch("src.rag_assistant.initialize_llm")
    @patch("src.rag_assistant.VectorDB")
    @patch("src.rag_assistant.MemoryManager")
    @patch("src.rag_assistant.ReasoningStrategyLoader")
    def test_invoke_saves_to_memory(self, mock_memory, mock_vectordb, mock_llm):
        """Test that invoke saves conversation to memory."""
        mock_llm.return_value.model_name = "test-model"
        mock_vectordb_instance = MagicMock()
        mock_vectordb.return_value = mock_vectordb_instance
        mock_vectordb_instance.search.return_value = {"documents": [["Content"]]}
        mock_memory_instance = MagicMock()
        mock_memory.return_value = mock_memory_instance

        assistant = RAGAssistant()
        assistant.chain = MagicMock()
        assistant.chain.invoke.return_value = "Assistant response"

        assistant.invoke("User question")

        # Verify memory was updated
        mock_memory_instance.add_message.assert_called_once_with(
            input_text="User question", output_text="Assistant response"
        )

    @patch("src.rag_assistant.initialize_llm")
    @patch("src.rag_assistant.VectorDB")
    @patch("src.rag_assistant.MemoryManager")
    @patch("src.rag_assistant.ReasoningStrategyLoader")
    def test_invoke_with_empty_context(self, mock_memory, mock_vectordb, mock_llm):
        """Test invoke when no documents are retrieved."""
        mock_llm.return_value.model_name = "test-model"
        mock_vectordb_instance = MagicMock()
        mock_vectordb.return_value = mock_vectordb_instance
        mock_vectordb_instance.search.return_value = {"documents": [[]]}
        mock_memory.return_value.memory = MagicMock()

        assistant = RAGAssistant()
        assistant.chain = MagicMock()
        assistant.chain.invoke.return_value = (
            "I'm sorry, that information is not known to me."
        )

        response = assistant.invoke("Out of scope query")

        # Verify chain was invoked with empty context
        assistant.chain.invoke.assert_called_once()
        assert "not known to me" in response


class TestRAGAssistantExceptionHandling:
    """Test exception handling in RAGAssistant."""

    @patch("src.rag_assistant.initialize_llm")
    @patch("src.rag_assistant.VectorDB")
    @patch("src.rag_assistant.MemoryManager")
    @patch("src.rag_assistant.ReasoningStrategyLoader")
    def test_vectordb_connection_error(self, mock_memory, mock_vectordb, mock_llm):
        """Test handling of VectorDB connection errors."""
        mock_llm.return_value.model_name = "test-model"
        mock_vectordb.side_effect = Exception("Failed to connect to ChromaDB")
        mock_memory.return_value.memory = MagicMock()

        # Should raise exception during initialization
        with pytest.raises(Exception):
            RAGAssistant()

    @patch("src.rag_assistant.initialize_llm")
    @patch("src.rag_assistant.VectorDB")
    @patch("src.rag_assistant.MemoryManager")
    @patch("src.rag_assistant.ReasoningStrategyLoader")
    def test_reasoning_strategy_load_error(
        self, mock_reasoning, mock_memory, mock_vectordb, mock_llm
    ):
        """Test handling of ReasoningStrategyLoader errors."""
        mock_llm.return_value.model_name = "test-model"
        mock_vectordb_instance = MagicMock()
        mock_vectordb.return_value = mock_vectordb_instance
        mock_memory.return_value.memory = MagicMock()
        mock_reasoning.side_effect = Exception("Failed to load reasoning strategy")

        # Should handle gracefully (log error but continue)
        assistant = RAGAssistant()
        assert assistant.reasoning_strategy is None

    @patch("src.rag_assistant.initialize_llm")
    @patch("src.rag_assistant.VectorDB")
    @patch("src.rag_assistant.MemoryManager")
    @patch("src.rag_assistant.ReasoningStrategyLoader")
    def test_invoke_llm_error(self, mock_memory, mock_vectordb, mock_llm):
        """Test handling of LLM invocation errors."""
        mock_llm.return_value.model_name = "test-model"
        mock_vectordb_instance = MagicMock()
        mock_vectordb.return_value = mock_vectordb_instance
        mock_vectordb_instance.search.return_value = {"documents": [["Content"]]}
        mock_memory.return_value.memory = MagicMock()

        assistant = RAGAssistant()
        assistant.chain = MagicMock()
        assistant.chain.invoke.side_effect = Exception("LLM API error")

        # Should raise exception
        with pytest.raises(Exception):
            assistant.invoke("Test query")


# ============================================================================
# META-QUESTION HANDLING TESTS
# ============================================================================


class TestMetaQuestionHandling:
    """Test how RAGAssistant handles meta-questions about the knowledge base."""

    # pylint: disable=missing-function-docstring

    @patch("src.rag_assistant.initialize_llm")
    @patch("src.rag_assistant.VectorDB")
    @patch("src.rag_assistant.MemoryManager")
    @patch("src.rag_assistant.ReasoningStrategyLoader")
    def test_meta_question_what_topics(self, mock_memory, mock_vectordb, mock_llm):
        """Test that 'what topics' meta-question is handled without strict similarity threshold."""
        mock_llm.return_value.model_name = "test-model"
        mock_vectordb_instance = MagicMock()
        mock_vectordb.return_value = mock_vectordb_instance
        # Return results with low similarity (distance > 0.35)
        mock_vectordb_instance.search.return_value = {
            "documents": [["Topic 1", "Topic 2"]],
            "metadatas": [[{"title": "T1"}, {"title": "T2"}]],
            "distances": [[0.4, 0.45]],
            "ids": [["id1", "id2"]],
        }
        mock_memory.return_value.memory = MagicMock()

        assistant = RAGAssistant()
        assistant.chain = MagicMock()
        assistant.chain.invoke.return_value = (
            "I know about these topics: Topic 1, Topic 2"
        )

        # Meta-questions should proceed even with low similarity
        response = assistant.invoke("What topics do you know about?")

        # Should get a valid response, not the "couldn't find" message
        assert "couldn't find" not in response.lower()
        assert response == "I know about these topics: Topic 1, Topic 2"

    @patch("src.rag_assistant.initialize_llm")
    @patch("src.rag_assistant.VectorDB")
    @patch("src.rag_assistant.MemoryManager")
    @patch("src.rag_assistant.ReasoningStrategyLoader")
    def test_meta_question_what_can_you(self, mock_memory, mock_vectordb, mock_llm):
        """Test that 'what can you' meta-question bypasses similarity threshold."""
        mock_llm.return_value.model_name = "test-model"
        mock_vectordb_instance = MagicMock()
        mock_vectordb.return_value = mock_vectordb_instance
        # Return low similarity results
        mock_vectordb_instance.search.return_value = {
            "documents": [["Information about capabilities"]],
            "metadatas": [[{"title": "Capabilities"}]],
            "distances": [[0.5]],
            "ids": [["id1"]],
        }
        mock_memory.return_value.memory = MagicMock()

        assistant = RAGAssistant()
        assistant.chain = MagicMock()
        assistant.chain.invoke.return_value = "I can help you with various tasks."

        response = assistant.invoke("What can you do?")

        # Should proceed despite low similarity
        assert "couldn't find" not in response.lower()
        assert assistant.chain.invoke.called

    @patch("src.rag_assistant.initialize_llm")
    @patch("src.rag_assistant.VectorDB")
    @patch("src.rag_assistant.MemoryManager")
    @patch("src.rag_assistant.ReasoningStrategyLoader")
    def test_regular_question_high_similarity_required(
        self, mock_memory, mock_vectordb, mock_llm
    ):
        """Test that regular questions require high similarity threshold."""
        mock_llm.return_value.model_name = "test-model"
        mock_vectordb_instance = MagicMock()
        mock_vectordb.return_value = mock_vectordb_instance
        # Return low similarity results
        mock_vectordb_instance.search.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "ids": [[]],
        }
        mock_memory.return_value.memory = MagicMock()

        assistant = RAGAssistant()

        response = assistant.invoke("What is quantum computing?")

        # Should get the "couldn't find" message for regular questions with low similarity
        assert "couldn't find" in response.lower()
        assert "rephras" in response.lower()

    @patch("src.rag_assistant.initialize_llm")
    @patch("src.rag_assistant.VectorDB")
    @patch("src.rag_assistant.MemoryManager")
    @patch("src.rag_assistant.ReasoningStrategyLoader")
    def test_regular_question_high_similarity_succeeds(
        self, mock_memory, mock_vectordb, mock_llm
    ):
        """Test that regular questions succeed with high similarity results."""
        mock_llm.return_value.model_name = "test-model"
        mock_vectordb_instance = MagicMock()
        mock_vectordb.return_value = mock_vectordb_instance
        # Return high similarity results (distance < 0.35)
        mock_vectordb_instance.search.return_value = {
            "documents": [["Quantum computing is..."]],
            "metadatas": [[{"title": "Quantum Computing"}]],
            "distances": [[0.2]],
            "ids": [["id1"]],
        }
        mock_memory.return_value.memory = MagicMock()

        assistant = RAGAssistant()
        assistant.chain = MagicMock()
        assistant.chain.invoke.return_value = (
            "Quantum computing is a field of computation..."
        )

        response = assistant.invoke("What is quantum computing?")

        # Should proceed with high similarity results
        assert "couldn't find" not in response.lower()
        assert assistant.chain.invoke.called

    @patch("src.rag_assistant.initialize_llm")
    @patch("src.rag_assistant.VectorDB")
    @patch("src.rag_assistant.MemoryManager")
    @patch("src.rag_assistant.ReasoningStrategyLoader")
    def test_meta_question_no_documents_found(
        self, mock_memory, mock_vectordb, mock_llm
    ):
        """Test meta-question handling when no documents are found."""
        mock_llm.return_value.model_name = "test-model"
        mock_vectordb_instance = MagicMock()
        mock_vectordb.return_value = mock_vectordb_instance
        # Return empty results
        mock_vectordb_instance.search.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "ids": [[]],
        }
        mock_memory.return_value.memory = MagicMock()

        assistant = RAGAssistant()
        assistant.chain = MagicMock()
        assistant.chain.invoke.return_value = "I don't have information about that."

        assistant.invoke("What topics are available?")

        # Should still process meta-questions even with empty results
        assert assistant.chain.invoke.called

    @patch("src.rag_assistant.initialize_llm")
    @patch("src.rag_assistant.VectorDB")
    @patch("src.rag_assistant.MemoryManager")
    @patch("src.rag_assistant.ReasoningStrategyLoader")
    def test_meta_question_with_all_keywords(
        self, mock_memory, mock_vectordb, mock_llm
    ):
        """Test various meta-question keywords are recognized."""
        mock_llm.return_value.model_name = "test-model"
        mock_vectordb_instance = MagicMock()
        mock_vectordb.return_value = mock_vectordb_instance
        mock_vectordb_instance.search.return_value = {
            "documents": [["Content"]],
            "metadatas": [[{"title": "Title"}]],
            "distances": [[0.5]],
            "ids": [["id1"]],
        }
        mock_memory.return_value.memory = MagicMock()

        assistant = RAGAssistant()
        assistant.chain = MagicMock()
        assistant.chain.invoke.return_value = "Response"

        # Test various meta-question keywords
        meta_questions = [
            "What topics do you know about?",
            "What do you know?",
            "What can you help with?",
            "What documents do you have?",
            "What information is available?",
            "What subjects can you discuss?",
            "Tell me your capabilities",
        ]

        for question in meta_questions:
            response = assistant.invoke(question)
            # All should proceed despite low similarity
            assert "couldn't find" not in response.lower(), f"Failed for: {question}"

    @patch("src.rag_assistant.initialize_llm")
    @patch("src.rag_assistant.VectorDB")
    @patch("src.rag_assistant.MemoryManager")
    @patch("src.rag_assistant.ReasoningStrategyLoader")
    def test_non_meta_question_not_triggered_by_partial_keyword(
        self, mock_memory, mock_vectordb, mock_llm
    ):
        """Test that questions without meta-keywords are treated as regular questions."""
        mock_llm.return_value.model_name = "test-model"
        mock_vectordb_instance = MagicMock()
        mock_vectordb.return_value = mock_vectordb_instance
        # Return low similarity results
        mock_vectordb_instance.search.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "ids": [[]],
        }
        mock_memory.return_value.memory = MagicMock()

        assistant = RAGAssistant()

        # Question without meta-keywords should fail with low similarity
        response = assistant.invoke("Tell me about machine learning")

        # Should get the "couldn't find" message
        assert "couldn't find" in response.lower()

    @patch("src.rag_assistant.initialize_llm")
    @patch("src.rag_assistant.VectorDB")
    @patch("src.rag_assistant.MemoryManager")
    @patch("src.rag_assistant.ReasoningStrategyLoader")
    def test_meta_question_memory_saved(self, mock_memory, mock_vectordb, mock_llm):
        """Test that meta-question responses are saved to memory."""
        mock_llm.return_value.model_name = "test-model"
        mock_vectordb_instance = MagicMock()
        mock_vectordb.return_value = mock_vectordb_instance
        mock_vectordb_instance.search.return_value = {
            "documents": [["Topic list"]],
            "metadatas": [[{"title": "Topics"}]],
            "distances": [[0.4]],
            "ids": [["id1"]],
        }
        mock_memory_instance = MagicMock()
        mock_memory.return_value = mock_memory_instance
        mock_memory_instance.memory = MagicMock()

        assistant = RAGAssistant()
        assistant.chain = MagicMock()
        assistant.chain.invoke.return_value = "Here are the topics..."

        assistant.invoke("What topics do you know about?")

        # Verify memory was updated
        mock_memory_instance.add_message.assert_called_once()
        call_args = mock_memory_instance.add_message.call_args
        assert "what topics" in call_args[1]["input_text"].lower()
        assert "Here are the topics" in call_args[1]["output_text"]

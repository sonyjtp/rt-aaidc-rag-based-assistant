"""
Integration tests for ChromaDB client, LLM utilities, and Vector Database.
Tests external service integrations and core database operations.
"""
from unittest.mock import MagicMock, patch

import pytest
from chromadb.errors import InvalidArgumentError

from src.chroma_client import ChromaDBClient
from src.llm_utils import initialize_llm
from src.vectordb import VectorDB
from tests.fixtures.chroma_fixtures import chroma_env_patch
from tests.fixtures.llm_fixtures import openai_provider_patch

# ============================================================================
# CHROMA CLIENT TESTS
# ============================================================================


@pytest.fixture
def chroma_mocks():
    """Fixture providing mocked ChromaDB client components."""
    with patch("src.chroma_client.chromadb.CloudClient") as mock_cloud_client:
        mock_client_instance = MagicMock()
        mock_cloud_client.return_value = mock_client_instance
        yield {
            "cloud_client": mock_cloud_client,
            "client_instance": mock_client_instance,
        }


# pylint: disable=redefined-outer-name
class TestChromaDBClientInitialization:
    """Test ChromaDB client initialization."""

    @chroma_env_patch()
    def test_chroma_client_init(self, chroma_mocks):
        """Test ChromaDB client initialization with environment variables."""
        client = ChromaDBClient()

        # Verify CloudClient was created with correct parameters
        chroma_mocks["cloud_client"].assert_called_once_with(
            api_key="test-key", tenant="test-tenant", database="test-db"
        )
        assert client.api_key == "test-key"
        assert client.tenant == "test-tenant"
        assert client.database == "test-db"
        assert client.client == chroma_mocks["client_instance"]

    @chroma_env_patch()
    def test_initialize_client_method(self, chroma_mocks):
        """Test _initialize_client method creates CloudClient correctly."""
        client = ChromaDBClient()

        # Verify _initialize_client was called during __init__
        assert chroma_mocks["cloud_client"].called
        # Verify the returned client is the mocked instance
        assert client.client is chroma_mocks["client_instance"]

    @chroma_env_patch()
    def test_get_or_create_collection(self, chroma_mocks):
        """Test getting or creating a ChromaDB collection."""
        mock_collection = MagicMock()
        chroma_mocks[
            "client_instance"
        ].get_or_create_collection.return_value = mock_collection

        client = ChromaDBClient()
        result = client.get_or_create_collection("test_collection")

        # Verify get_or_create_collection was called with correct parameters
        chroma_mocks[
            "client_instance"
        ].get_or_create_collection.assert_called_once_with(
            name="test_collection",
            metadata={
                "hnsw:space": "cosine",
                "description": "RAG document collection",
            },
        )
        assert result == mock_collection

    @chroma_env_patch()
    def test_cloud_client_operations(self, chroma_mocks):
        """Test deleting a ChromaDB collection."""
        client = ChromaDBClient()
        client.delete_collection("test_collection")

        # Verify delete_collection was called with correct parameters
        chroma_mocks["client_instance"].delete_collection.assert_called_once_with(
            name="test_collection"
        )

    @patch.dict("os.environ", {}, clear=True)
    def test_chroma_client_missing_env_vars(self, chroma_mocks):
        """Test ChromaDB client handles missing environment variables."""
        client = ChromaDBClient()

        # Should create client even with None values
        assert client.api_key is None
        assert client.tenant is None
        assert client.database is None
        # But should still call CloudClient with None values
        chroma_mocks["cloud_client"].assert_called_once_with(
            api_key=None, tenant=None, database=None
        )


# ============================================================================
# LLM UTILS TESTS
# ============================================================================


# pylint: disable=redefined-outer-name
class TestLLMInitialization:
    """Test LLM initialization utility."""

    def test_initialize_llm_with_provider(self):
        """Test LLM initialization with different providers (parameterized)."""
        llm = initialize_llm()

        # Should return an LLM instance
        assert llm is not None

    @patch.dict("os.environ", {"OPENAI_API_KEY": "openai-key"})
    @openai_provider_patch()
    def test_initialize_llm_with_openai(self):
        """Test LLM initialization with OpenAI API key."""
        llm = initialize_llm()

        # Should return an LLM instance
        assert llm is not None

    def test_initialize_llm_with_openai_alt(self):
        """Test LLM initialization with OpenAI API key (alternate method)."""
        with openai_provider_patch():
            with patch.dict("os.environ", {"OPENAI_API_KEY": "openai-key"}):
                llm = initialize_llm()

                # Should return an LLM instance
                assert llm is not None

    @patch.dict("os.environ", {}, clear=True)
    @patch("src.llm_utils.LLM_PROVIDERS", [])
    def test_initialize_llm_no_api_keys(self):
        """Test LLM initialization fails when no API keys available."""
        with pytest.raises(ValueError):
            initialize_llm()

    @patch.dict("os.environ", {"CUSTOM_MODEL": "custom-model-name"})
    def test_initialize_llm_custom_model(self):
        """Test LLM initialization uses custom model name from environment."""
        with patch(
            "src.llm_utils.LLM_PROVIDERS",
            [
                {
                    "api_key_env": "CUSTOM_API_KEY",
                    "api_key_param": "api_key",
                    "model_env": "CUSTOM_MODEL",
                    "default_model": "default-model",
                    "class": MagicMock(),
                }
            ],
        ):
            with patch.dict(
                "os.environ",
                {"CUSTOM_API_KEY": "test-key", "CUSTOM_MODEL": "custom-model"},
            ):
                llm = initialize_llm()

                # Should use custom model
                assert llm is not None


# ============================================================================
# VECTOR DATABASE TESTS
# ============================================================================


# pylint: disable=protected-access, disable=too-few-public-methods, redefined-outer-name
class TestVectorDB:
    """Test VectorDB initialization, chunking, document operations, and search."""

    # ========================================================================
    # DOCUMENT STANDARDIZATION TESTS
    # ========================================================================

    def test_standardize_document_string(self):
        """Test standardize_document with string input."""
        doc = "This is a test document"
        result = VectorDB.standardize_document(doc)

        assert isinstance(result, dict)
        assert result["content"] == doc
        assert result["title"] == ""
        assert result["filename"] == ""
        assert result["tags"] == ""

    def test_standardize_document_dict(self):
        """Test standardize_document with dictionary input."""
        doc = {
            "content": "Test content",
            "title": "Test Title",
            "filename": "test.txt",
            "tags": "test, document",
        }
        result = VectorDB.standardize_document(doc)

        assert isinstance(result, dict)
        assert result["content"] == "Test content"
        assert result["title"] == "Test Title"
        assert result["filename"] == "test.txt"
        # Tags should be formatted by format_tags function

    def test_standardize_document_dict_missing_keys(self):
        """Test standardize_document with incomplete dictionary."""
        doc = {"content": "Test content"}
        result = VectorDB.standardize_document(doc)

        assert result["content"] == "Test content"
        assert result["title"] == ""
        assert result["filename"] == ""
        assert result["tags"] == ""

    # ========================================================================
    # INITIALIZATION AND SETUP TESTS
    # ========================================================================

    @patch("src.vectordb.ChromaDBClient")
    @patch("src.vectordb.initialize_embedding_model")
    def test_vectordb_init(self, mock_embedding, mock_chroma):
        """Test VectorDB initialization."""
        mock_collection = MagicMock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        mock_embedding_model = MagicMock()
        mock_embedding_model.model_name = "test-embedding-model"
        mock_embedding.return_value = mock_embedding_model

        vdb = VectorDB()

        # Verify initialization
        assert vdb.collection == mock_collection
        assert vdb.embedding_model == mock_embedding_model
        assert vdb.text_splitter is not None

    # ========================================================================
    # DOCUMENT CHUNKING TESTS
    # ========================================================================

    @patch("src.vectordb.ChromaDBClient")
    @patch("src.vectordb.initialize_embedding_model")
    def test_chunk_documents_string(self, mock_embedding, mock_chroma):
        """Test chunking string documents."""
        mock_collection = MagicMock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        mock_embedding_model = MagicMock()
        mock_embedding_model.model_name = "test-model"
        mock_embedding.return_value = mock_embedding_model

        vdb = VectorDB()

        # Test chunking with string documents
        documents = ["This is a test document with some content. " * 10]
        chunks = vdb._chunk_documents(documents)

        # Should return list of tuples with chunks and metadata
        assert isinstance(chunks, list)
        for chunk, metadata in chunks:
            assert isinstance(chunk, str)
            assert isinstance(metadata, dict)
            assert "title" in metadata
            assert "filename" in metadata
            assert "tags" in metadata

    @patch("src.vectordb.ChromaDBClient")
    @patch("src.vectordb.initialize_embedding_model")
    def test_chunk_documents_dict(self, mock_embedding, mock_chroma):
        """Test chunking dictionary documents with metadata."""
        mock_collection = MagicMock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        mock_embedding_model = MagicMock()
        mock_embedding_model.model_name = "test-model"
        mock_embedding.return_value = mock_embedding_model

        vdb = VectorDB()

        # Test chunking with dict documents
        documents = [
            {
                "content": "This is a test document.",
                "title": "Test Title",
                "filename": "test.txt",
                "tags": "test, document",
            }
        ]
        chunks = vdb._chunk_documents(documents)

        # Should return chunks with metadata
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        for _chunk, metadata in chunks:
            assert metadata["title"] == "Test Title"
            assert metadata["filename"] == "test.txt"

    # ========================================================================
    # DOCUMENT ADDITION TESTS
    # ========================================================================

    @patch("src.vectordb.ChromaDBClient")
    @patch("src.vectordb.initialize_embedding_model")
    def test_add_documents_string_list(self, mock_embedding, mock_chroma):
        """Test adding a list of string documents."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.get.return_value = {"documents": []}
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        mock_embedding_model = MagicMock()
        mock_embedding_model.model_name = "test-model"
        mock_embedding_model.embed_documents.return_value = [[0.1, 0.2, 0.3]] * 3
        mock_embedding.return_value = mock_embedding_model

        vdb = VectorDB()
        documents = [
            "Document 1 content that is long enough to create chunks",
            "Document 2 content that is long enough to create chunks",
            "Document 3 content that is long enough to create chunks",
        ]

        vdb.add_documents(documents)

        # Verify collection.add was called
        assert mock_collection.add.called

    @patch("src.vectordb.ChromaDBClient")
    @patch("src.vectordb.initialize_embedding_model")
    def test_add_documents_dict_list(self, mock_embedding, mock_chroma):
        """Test adding a list of dictionary documents."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_collection.get.return_value = {"documents": []}
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        mock_embedding_model = MagicMock()
        mock_embedding_model.model_name = "test-model"
        mock_embedding_model.embed_documents.return_value = [[0.1, 0.2, 0.3]] * 1
        mock_embedding.return_value = mock_embedding_model

        vdb = VectorDB()
        documents = [
            {
                "content": "Test content",
                "title": "Test Title",
                "filename": "test.txt",
                "tags": "test",
            }
        ]

        vdb.add_documents(documents)

        # Verify collection.add was called
        assert mock_collection.add.called

    # ========================================================================
    # DEDUPLICATION TESTS
    # ========================================================================

    @patch("src.vectordb.ChromaDBClient")
    @patch("src.vectordb.initialize_embedding_model")
    def test_filter_duplicate_chunks(self, mock_embedding, mock_chroma):
        """Test filtering duplicate chunks with punctuation normalization."""
        mock_collection = MagicMock()
        # Existing docs are stored with normalized punctuation
        mock_collection.get.return_value = {"documents": ["existing doc"]}
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        mock_embedding_model = MagicMock()
        mock_embedding_model.model_name = "test-model"
        mock_embedding.return_value = mock_embedding_model

        vdb = VectorDB()

        # Test deduplication with punctuation variations
        chunks = [
            ("new chunk 1", {"title": "T1"}),
            ("...existing doc", {"title": "T2"}),  # Will normalize to "existing doc"
            ("new chunk 1", {"title": "T3"}),  # Duplicate in batch
            (
                "!new chunk 1",
                {"title": "T4"},
            ),  # Will normalize to "new chunk 1" (duplicate)
        ]

        filtered = vdb._filter_duplicate_chunks(chunks)

        # Should have 1 chunk (existing removed, duplicates in batch removed)
        assert len(filtered) == 1
        assert filtered[0][0] == "new chunk 1"

    @patch("src.vectordb.ChromaDBClient")
    @patch("src.vectordb.initialize_embedding_model")
    def test_no_duplicate_chunks(self, mock_embedding, mock_chroma):
        """Test when there are no duplicate chunks."""
        mock_collection = MagicMock()
        mock_collection.get.return_value = {"documents": []}
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        mock_embedding_model = MagicMock()
        mock_embedding_model.model_name = "test-model"
        mock_embedding.return_value = mock_embedding_model

        vdb = VectorDB()

        chunks = [
            ("chunk 1", {"title": "T1"}),
            ("chunk 2", {"title": "T2"}),
        ]

        filtered = vdb._filter_duplicate_chunks(chunks)

        # All chunks should be kept
        assert len(filtered) == 2

    # ========================================================================
    # SEARCH TESTS
    # ========================================================================

    @patch("src.vectordb.ChromaDBClient")
    @patch("src.vectordb.initialize_embedding_model")
    def test_search_basic(self, mock_embedding, mock_chroma):
        """Test basic document search."""
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Document 1", "Document 2"]],
            "metadatas": [[{"title": "T1"}, {"title": "T2"}]],
            "distances": [[0.1, 0.2]],
            "ids": [["id1", "id2"]],
        }
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        mock_embedding_model = MagicMock()
        mock_embedding_model.model_name = "test-model"
        mock_embedding_model.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_embedding.return_value = mock_embedding_model

        vdb = VectorDB()
        results = vdb.search("What is AI?", n_results=2)

        # Verify search results
        assert "documents" in results
        assert "metadatas" in results
        assert "distances" in results
        assert "ids" in results
        assert len(results["documents"]) == 2

    @patch("src.vectordb.ChromaDBClient")
    @patch("src.vectordb.initialize_embedding_model")
    def test_search_no_results(self, mock_embedding, mock_chroma):
        """Test search with no results."""
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "ids": [[]],
        }
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        mock_embedding_model = MagicMock()
        mock_embedding_model.model_name = "test-model"
        mock_embedding_model.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_embedding.return_value = mock_embedding_model

        vdb = VectorDB()
        results = vdb.search("Unknown query")

        # Should return empty results
        assert results["documents"] == []
        assert results["metadatas"] == []
        assert results["distances"] == []

    @patch("src.vectordb.ChromaDBClient")
    @patch("src.vectordb.initialize_embedding_model")
    def test_search_handles_embedding_dimension_mismatch(
        self, mock_embedding, mock_chroma, vectordb_mocks
    ):
        """Test that search handles embedding dimension mismatch gracefully."""
        mock_chroma.return_value = vectordb_mocks["chroma_instance"]
        mock_embedding.return_value = vectordb_mocks["embedding_model"]

        mock_collection = vectordb_mocks["collection"]

        # Mock the collection query to raise InvalidArgumentError for dimension mismatch
        error_msg = "Collection expecting embedding with dimension of 768, got 384"
        mock_collection.query.side_effect = InvalidArgumentError(error_msg)

        vdb = VectorDB()
        result = vdb.search(query="test query")

        # Should return empty results gracefully
        assert result["documents"] == []
        assert result["metadatas"] == []
        assert result["distances"] == []
        assert result["ids"] == []

    @patch("src.vectordb.ChromaDBClient")
    @patch("src.vectordb.initialize_embedding_model")
    def test_search_reraises_other_invalid_argument_errors(
        self, mock_embedding, mock_chroma, vectordb_mocks
    ):
        """Test that search re-raises non-dimension-mismatch InvalidArgumentError."""
        mock_chroma.return_value = vectordb_mocks["chroma_instance"]
        mock_embedding.return_value = vectordb_mocks["embedding_model"]

        mock_collection = vectordb_mocks["collection"]

        # Mock the collection query to raise a different InvalidArgumentError
        error_msg = "Invalid query parameter"
        mock_collection.query.side_effect = InvalidArgumentError(error_msg)

        vdb = VectorDB()

        # Should re-raise the error since it's not a dimension mismatch
        with pytest.raises(InvalidArgumentError, match="Invalid query parameter"):
            vdb.search(query="test query")

    @patch("src.vectordb.ChromaDBClient")
    @patch("src.vectordb.initialize_embedding_model")
    def test_search_successful_with_correct_dimensions(
        self, mock_embedding, mock_chroma
    ):
        """Test that search works correctly when dimensions match."""
        mock_collection = MagicMock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        mock_embedding_model = MagicMock()
        mock_embedding_model.model_name = "test-model"
        mock_embedding.return_value = mock_embedding_model

        # Mock the embedding query
        mock_embedding_model.embed_query.return_value = [0.1, 0.2, 0.3]

        # Mock successful search results
        mock_collection.query.return_value = {
            "documents": [["Document 1", "Document 2"]],
            "metadatas": [[{"title": "Doc1"}, {"title": "Doc2"}]],
            "distances": [[0.1, 0.2]],
            "ids": [["id1", "id2"]],
        }

        vdb = VectorDB()
        result = vdb.search(query="test query", n_results=2)

        # Verify successful search
        assert len(result["documents"]) == 2
        assert result["documents"][0] == "Document 1"
        assert result["documents"][1] == "Document 2"
        assert result["metadatas"][0]["title"] == "Doc1"
        assert result["ids"][0] == "id1"

    @patch("src.vectordb.ChromaDBClient")
    @patch("src.vectordb.initialize_embedding_model")
    def test_search_logs_dimension_mismatch_error(self, mock_embedding, mock_chroma):
        """Test that search logs helpful error message for dimension mismatch."""

        mock_collection = MagicMock()
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        mock_embedding_model = MagicMock()
        mock_embedding_model.model_name = "test-model"
        mock_embedding.return_value = mock_embedding_model

        # Mock the embedding query
        mock_embedding_model.embed_query.return_value = [0.1, 0.2, 0.3]

        # Mock the collection query to raise InvalidArgumentError
        error_msg = "Collection expecting embedding with dimension of 768, got 384"
        mock_collection.query.side_effect = InvalidArgumentError(error_msg)

        vdb = VectorDB()

        with patch("src.vectordb.logger") as mock_logger:
            result = vdb.search(query="test query")

            # Verify error was logged with helpful message
            mock_logger.error.assert_called_once()
            error_call = mock_logger.error.call_args[0][0]
            assert "Embedding dimension mismatch" in error_call
            assert "VECTOR_DB_EMBEDDING_MODEL" in error_call
            assert "config.py" in error_call

            # Result should still be empty
            assert result["documents"] == []

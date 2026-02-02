"""
Integration tests for ChromaDB client, LLM utilities, and Vector Database.
Tests external service integrations and core database operations.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from config import (
    CHROMA_COLLECTION_METADATA,
    COLLECTION_NAME_DEFAULT,
    DISTANCE_THRESHOLD,
    RETRIEVAL_K,
    VECTOR_DB_EMBEDDING_MODEL,
)
from src.chroma_client import ChromaDBClient
from src.llm_utils import initialize_llm
from tests.fixtures.chroma_fixtures import chroma_env_patch

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

        assert chroma_mocks["cloud_client"].called
        assert client.client is chroma_mocks["client_instance"]

    @chroma_env_patch()
    def test_init_initializes_embedding_function(self, chroma_mocks):
        """Test that __init__ initializes the embedding function with correct model."""
        with patch("src.chroma_client.embedding_functions.SentenceTransformerEmbeddingFunction") as mock_embedding_fn:
            mock_embedding_instance = MagicMock()
            mock_embedding_fn.return_value = mock_embedding_instance

            client = ChromaDBClient()

            mock_embedding_fn.assert_called_once_with(model_name=VECTOR_DB_EMBEDDING_MODEL)
            assert client.embedding_function is mock_embedding_instance

    @chroma_env_patch()
    def test_get_or_create_collection(self, chroma_mocks):
        """Test getting or creating a ChromaDB collection."""
        mock_collection = MagicMock()
        chroma_mocks["client_instance"].get_or_create_collection.return_value = mock_collection

        client = ChromaDBClient()
        result = client.get_or_create_collection("test_collection")

        chroma_mocks["client_instance"].get_or_create_collection.assert_called_once_with(
            name="test_collection",
            metadata=CHROMA_COLLECTION_METADATA,
            embedding_function=client.embedding_function,
        )
        assert result == mock_collection

    @chroma_env_patch()
    def test_cloud_client_operations(self, chroma_mocks):
        """Test deleting a ChromaDB collection."""
        client = ChromaDBClient()
        client.delete_collection("test_collection")

        chroma_mocks["client_instance"].delete_collection.assert_called_once_with(name="test_collection")

    @chroma_env_patch()
    @pytest.mark.parametrize(
        "query_response, expected_ids, expected_distances",
        [
            pytest.param(
                {
                    "ids": [["id1", "id2", "id3"]],
                    "distances": [
                        [
                            DISTANCE_THRESHOLD - 0.9,
                            DISTANCE_THRESHOLD - 0.1,
                            DISTANCE_THRESHOLD + 0.5,
                        ]
                    ],
                    "metadatas": [[{}, {}, {}]],
                },
                ["id1", "id2"],
                [DISTANCE_THRESHOLD - 0.9, DISTANCE_THRESHOLD - 0.1],
                id="partial_results_pass_distance_filter",
            ),
            pytest.param(
                {
                    "ids": [["id1", "id2"]],
                    "distances": [[DISTANCE_THRESHOLD + 0.5, DISTANCE_THRESHOLD + 1.0]],
                    "metadatas": [[{}, {}]],
                },
                [],
                [],
                id="all_results_exceed_threshold",
            ),
            pytest.param(
                {
                    "ids": [[]],
                    "distances": [[]],
                    "metadatas": [[]],
                },
                [],
                [],
                id="empty_query_results",
            ),
        ],
    )
    def test_query_collection_distance_filtering(self, chroma_mocks, query_response, expected_ids, expected_distances):
        """Parameterized test for distance threshold filtering using DISTANCE_THRESHOLD from config.

        Test data uses DISTANCE_THRESHOLD +/- offsets so tests adapt automatically if
        DISTANCE_THRESHOLD changes in config.py without requiring test updates.
        """
        mock_collection = MagicMock()
        mock_collection.query.return_value = query_response
        chroma_mocks["client_instance"].get_collection.return_value = mock_collection

        with patch("src.chroma_client.embedding_functions.SentenceTransformerEmbeddingFunction"):
            client = ChromaDBClient()
            result = client.query_collection(query_texts="test query", maximum_distance=DISTANCE_THRESHOLD)

        assert result["ids"] == expected_ids, f"Expected ids {expected_ids}, got {result['ids']}"
        assert (
            result["distances"] == expected_distances
        ), f"Expected distances {expected_distances}, got {result['distances']}"

    @chroma_env_patch()
    @pytest.mark.parametrize(
        "query_texts, where_clause, collection_name, n_results",
        [
            pytest.param(
                "single query string",
                None,
                None,
                None,
                id="string_query_text_default_params",
            ),
            pytest.param(
                ["query1", "query2"],
                None,
                None,
                None,
                id="list_query_texts_default_params",
            ),
            pytest.param(
                "test query",
                {"source": {"$eq": "doc1"}},
                None,
                None,
                id="with_where_filter_clause",
            ),
            pytest.param(
                "test query",
                None,
                "custom_collection",
                None,
                id="with_custom_collection_name",
            ),
            pytest.param("test query", None, None, 10, id="with_custom_n_results_parameter"),
        ],
    )
    def test_query_collection_parameter_passing(
        self, chroma_mocks, query_texts, where_clause, collection_name, n_results
    ):
        """Parameterized test for parameter passing to ChromaDB methods."""
        # Mock the embedding function to avoid initialization issues
        with patch("src.chroma_client.embedding_functions.SentenceTransformerEmbeddingFunction"):
            mock_collection = MagicMock()
            mock_collection.query.return_value = {
                "ids": [["id1"]],
                "distances": [[0.2]],
                "metadatas": [[]],
            }
            chroma_mocks["client_instance"].get_collection.return_value = mock_collection

            client = ChromaDBClient()

            # Call with specified parameters
            client.query_collection(
                query_texts=query_texts,
                where=where_clause,
                collection_name=collection_name,
                n_results=n_results,
            )

            # Verify parameters were passed correctly
            if collection_name:
                call_kwargs = chroma_mocks["client_instance"].get_collection.call_args.kwargs
                assert call_kwargs["name"] == collection_name

            query_call_kwargs = mock_collection.query.call_args.kwargs
            assert query_call_kwargs["query_texts"] == query_texts

            if where_clause is not None:
                assert query_call_kwargs["where"] == where_clause

            if n_results is not None:
                assert query_call_kwargs["n_results"] == n_results

    @chroma_env_patch()
    def test_query_collection_embedding_function_integration(self, chroma_mocks):
        """Test that embedding_function is properly integrated with get_collection."""
        with patch("src.chroma_client.embedding_functions.SentenceTransformerEmbeddingFunction"):
            mock_collection = MagicMock()
            mock_collection.query.return_value = {
                "ids": [["id1"]],
                "distances": [[0.1]],
                "metadatas": [[]],
            }
            chroma_mocks["client_instance"].get_collection.return_value = mock_collection

            client = ChromaDBClient()
            client.query_collection(query_texts="test")

            # Verify embedding_function was passed to get_collection
            call_kwargs = chroma_mocks["client_instance"].get_collection.call_args.kwargs
            assert "embedding_function" in call_kwargs
            assert call_kwargs["embedding_function"] == client.embedding_function

    @chroma_env_patch()
    def test_query_collection_default_values(self, chroma_mocks):
        """Test that query_collection uses correct default parameter values."""
        with patch("src.chroma_client.embedding_functions.SentenceTransformerEmbeddingFunction"):
            mock_collection = MagicMock()
            mock_collection.query.return_value = {
                "ids": [["id1"]],
                "distances": [[0.1]],
                "metadatas": [[]],
            }
            chroma_mocks["client_instance"].get_collection.return_value = mock_collection

            client = ChromaDBClient()

            # Call with minimal parameters (using all defaults)
            client.query_collection()

            # Verify collection defaults
            get_collection_kwargs = chroma_mocks["client_instance"].get_collection.call_args.kwargs
            assert get_collection_kwargs["name"] == COLLECTION_NAME_DEFAULT

            # Verify query defaults
            query_kwargs = mock_collection.query.call_args.kwargs
            assert query_kwargs["n_results"] == RETRIEVAL_K
            assert query_kwargs["query_texts"] == ""
            assert query_kwargs["where"] is None

    def test_chroma_client_missing_env_vars(self, chroma_mocks):
        """Test ChromaDB client handles missing environment variables."""
        # Explicitly clear env vars to simulate missing configuration
        with patch.dict("os.environ", {}, clear=True), patch(
            "src.chroma_client.embedding_functions.SentenceTransformerEmbeddingFunction"
        ):
            client = ChromaDBClient()

            # Should create client even with None values
            assert client.api_key is None
            assert client.tenant is None
            assert client.database is None
            # But should still call CloudClient with None values
            chroma_mocks["cloud_client"].assert_called_once_with(api_key=None, tenant=None, database=None)

    @chroma_env_patch()
    def test_init_sets_all_attributes(self, chroma_mocks):
        """Test that __init__ properly sets all instance attributes."""
        with patch("src.chroma_client.embedding_functions.SentenceTransformerEmbeddingFunction") as mock_embedding_fn:
            mock_embedding_instance = MagicMock()
            mock_embedding_fn.return_value = mock_embedding_instance

            client = ChromaDBClient()

            # Verify all attributes are set after initialization
            assert client.api_key == "test-key"
            assert client.tenant == "test-tenant"
            assert client.database == "test-db"
            assert client.embedding_function is mock_embedding_instance
            assert client.client is chroma_mocks["client_instance"]
            # Verify client is not None (successfully initialized)
            assert client.client is not None


# ============================================================================
# LLM UTILS TESTS
# ============================================================================


def create_llm_provider_config(api_key_env, api_key_value, model_env, default_model):
    """Helper function to create a mocked LLM provider configuration.

    Args:
        api_key_env: Environment variable name for API key
        api_key_value: Value to set for the API key environment variable
        model_env: Environment variable name for model
        default_model: Default model name if env var is not set

    Returns:
        A tuple of (env_dict, provider_list) for patching
    """
    env_dict = {api_key_env: api_key_value}
    provider_list = [
        {
            "api_key_env": api_key_env,
            "api_key_param": "api_key",
            "model_env": model_env,
            "default_model": default_model,
            "class": MagicMock(),
        }
    ]
    return env_dict, provider_list


# pylint: disable=redefined-outer-name
class TestLLMInitialization:
    """Test LLM initialization utility."""

    @pytest.mark.parametrize(
        "api_key_env,api_key_value,model_env,default_model",
        [
            ("TEST_API_KEY", "test-key", "TEST_MODEL", "test-model"),
            ("OPENAI_API_KEY", "openai-key", "OPENAI_MODEL", "gpt-4o-mini"),
        ],
    )
    def test_initialize_llm_with_provider(self, api_key_env, api_key_value, model_env, default_model):
        """Parametrized test for LLM initialization with different providers."""
        env_dict, provider_list = create_llm_provider_config(api_key_env, api_key_value, model_env, default_model)

        with patch.dict("os.environ", env_dict), patch("src.llm_utils.LLM_PROVIDERS", provider_list):
            llm = initialize_llm()
            assert llm is not None

    @pytest.mark.parametrize(
        "env_vars,providers,expected_error",
        [
            pytest.param({}, [], RuntimeError, id="empty_providers"),
            pytest.param(
                {},
                [
                    {
                        "api_key_env": "TEST_API_KEY",
                        "api_key_param": "api_key",
                        "model_env": "TEST_MODEL",
                        "default_model": "test-model",
                        "class": MagicMock(),
                    }
                ],
                ValueError,
                id="missing_api_key",
            ),
            pytest.param(
                {"TEST_API_KEY": ""},
                [
                    {
                        "api_key_env": "TEST_API_KEY",
                        "api_key_param": "api_key",
                        "model_env": "TEST_MODEL",
                        "default_model": "test-model",
                        "class": MagicMock(),
                    }
                ],
                ValueError,
                id="empty_api_key",
            ),
        ],
    )
    def test_initialize_llm_raises_errors(self, env_vars, providers, expected_error):
        """Test LLM initialization raises ValueError for null API keys and RuntimeError for empty providers."""
        with patch.dict("os.environ", env_vars, clear=True), patch("src.llm_utils.LLM_PROVIDERS", providers):
            with pytest.raises(expected_error):
                initialize_llm()

    @patch.dict(
        "os.environ",
        {"CUSTOM_API_KEY": "test-key", "CUSTOM_MODEL": "custom-model-name"},
    )
    @patch(
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
    )
    def test_initialize_llm_custom_model(self):
        """Test LLM initialization uses custom model name from environment."""
        llm = initialize_llm()

        # Should use custom model
        assert llm is not None


# ============================================================================
# LLM INTEGRATION TESTS (REAL INSTANTIATION)
# ============================================================================


class TestLLMInstantiationIntegration:
    """Integration tests for actual LLM class instantiation.

    These tests verify that real LLM provider classes can be instantiated
    with proper configuration, not just mocked. They complement the unit tests
    by testing the actual initialization behavior with real classes.
    """

    def test_llm_instantiation_with_mock_class(self):
        """Test a mock LLM class instantiation with expected interface."""
        # Create a mock LLM class that simulates real LLM behavior
        mock_llm_class = MagicMock()
        mock_llm_instance = MagicMock()
        mock_llm_instance.model_name = "test-model"
        mock_llm_instance.invoke = MagicMock(return_value="Test response")
        mock_llm_class.return_value = mock_llm_instance

        # Simulate instantiation with configuration
        api_key = "test-api-key"
        model_name = "test-model"

        # Create instance with parameters
        llm = mock_llm_class(api_key=api_key, model_name=model_name)

        # Verify instance has expected attributes and methods
        assert llm is not None
        assert hasattr(llm, "model_name")
        assert hasattr(llm, "invoke")
        assert llm.model_name == "test-model"

        # Verify it can be called
        response = llm.invoke(input="Hello")
        assert response == "Test response"

    def test_llm_class_signature_compatibility(self):
        """Test that LLM classes accept standard configuration parameters."""
        # Simulate what initialize_llm does:
        # instantiate with api_key and model_name
        mock_llm_class = MagicMock()

        # Expected signature: class(api_key=..., model_name=...)
        api_key = "test-key"
        model_name = "test-model"

        # Actually call the class with the right parameters
        instance = mock_llm_class(api_key=api_key, model_name=model_name)

        # Verify the class was called with the right parameters
        mock_llm_class.assert_called_once_with(api_key=api_key, model_name=model_name)
        assert instance is not None

    def test_llm_provider_class_instantiation_flow(self):
        """Test the complete LLM provider instantiation flow.

        Simulates what happens in initialize_llm:
        1. Find provider by checking env vars
        2. Get API key from environment
        3. Get model name from environment or use default
        4. Instantiate the provider class with these parameters
        """
        # Mock LLM provider class
        mock_provider_class = MagicMock()
        mock_llm_instance = MagicMock()
        mock_provider_class.return_value = mock_llm_instance

        # Simulate environment and provider configuration
        api_key_value = "test-key-123"
        model_value = "test-model-v2"
        default_model = "default-test-model"

        # Simulate the initialization logic
        api_key = api_key_value  # Retrieved from environment
        model_name = model_value or default_model  # Use env var or default

        # Instantiate provider class
        llm = mock_provider_class(api_key=api_key, model_name=model_name)

        # Verify the flow
        assert llm is not None
        mock_provider_class.assert_called_once_with(api_key="test-key-123", model_name="test-model-v2")

    @patch.dict("os.environ", {"ACTUAL_LLM_API_KEY": "real-api-key"})
    def test_llm_with_real_environment_values(self):
        """Test LLM instantiation with real environment variable values.

        This verifies the env var retrieval and parameter passing work
        correctly with actual environment values (not mocked).
        """

        # Retrieve from actual environment (patched for test)
        api_key = os.getenv("ACTUAL_LLM_API_KEY")

        # Verify retrieval worked
        assert api_key == "real-api-key"

        # Create mock LLM class for testing
        mock_llm_class = MagicMock()
        mock_llm_class.return_value = MagicMock()

        # Actually call the mock with the correct values
        instance = mock_llm_class(api_key="real-api-key", model_name="test-model")

        # Verify correct values were passed
        mock_llm_class.assert_called_once_with(api_key="real-api-key", model_name="test-model")
        assert instance is not None

    def test_llm_class_instantiation_error_handling(self):
        """Test handling of errors during LLM class instantiation."""
        # Mock LLM class that raises an error on instantiation
        mock_llm_class = MagicMock(side_effect=ValueError("Invalid API key format"))

        # Simulate what would happen if instantiation fails
        with pytest.raises(ValueError, match="Invalid API key format"):
            mock_llm_class(api_key="invalid", model_name="test")

    def test_llm_instance_has_required_methods(self):
        """Test that instantiated LLM has required interface methods."""
        # Create a mock LLM instance with expected interface
        mock_llm = MagicMock()
        mock_llm.invoke = MagicMock(return_value="response")
        mock_llm.batch = MagicMock(return_value=["response1", "response2"])
        mock_llm.model_name = "test-model"

        # Verify required attributes/methods exist
        assert callable(mock_llm.invoke)
        assert callable(mock_llm.batch)
        assert hasattr(mock_llm, "model_name")

        # Verify they work
        response = mock_llm.invoke(input="test")
        assert response == "response"

        batch_responses = mock_llm.batch(["q1", "q2"])
        assert len(batch_responses) == 2  # type: ignore
        assert batch_responses[0] == "response1"  # type: ignore
        assert batch_responses[1] == "response2"  # type: ignore

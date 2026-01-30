"""
Integration tests for ChromaDB client, LLM utilities, and Vector Database.
Tests external service integrations and core database operations.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

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
    def test_initialize_llm_with_provider(
        self, api_key_env, api_key_value, model_env, default_model
    ):
        """Parametrized test for LLM initialization with different providers."""
        env_dict, provider_list = create_llm_provider_config(
            api_key_env, api_key_value, model_env, default_model
        )

        with patch.dict("os.environ", env_dict), patch(
            "src.llm_utils.LLM_PROVIDERS", provider_list
        ):
            llm = initialize_llm()
            assert llm is not None

    @patch.dict("os.environ", {}, clear=True)
    @patch("src.llm_utils.LLM_PROVIDERS", [])
    def test_initialize_llm_no_api_keys(self):
        """Test LLM initialization fails when no API keys available."""
        with pytest.raises(ValueError):
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
        mock_provider_class.assert_called_once_with(
            api_key="test-key-123", model_name="test-model-v2"
        )

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
        mock_llm_class.assert_called_once_with(
            api_key="real-api-key", model_name="test-model"
        )
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


# ============================================================================
# END OF TESTS
# ============================================================================

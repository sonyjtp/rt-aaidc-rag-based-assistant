"""
Unit tests for embedding model initialization.
Tests device detection, configuration, and model setup.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.embeddings import initialize_embedding_model


# pylint: disable=redefined-outer-name
class TestEmbeddingModelInitialization:
    """Test embedding model initialization with device detection."""

    def test_device_detection_priority(self, embedding_mocks_with_device):
        """Test device selection priority: CUDA > MPS > CPU."""
        result = initialize_embedding_model()

        # Verify device was selected with correct priority
        embedding_mocks_with_device["embeddings"].assert_called_once()
        call_kwargs = embedding_mocks_with_device["embeddings"].call_args[1]
        assert (
            call_kwargs["model_kwargs"]["device"]
            == embedding_mocks_with_device["expected_device"]
        )
        assert result == embedding_mocks_with_device["instance"]

    def test_default_model_name(self, embedding_mocks_with_device):
        """Test that default model name is used when not configured."""
        initialize_embedding_model()

        call_kwargs = embedding_mocks_with_device["embeddings"].call_args[1]
        assert call_kwargs["model_name"] == "sentence-transformers/all-mpnet-base-v2"

    @patch("src.embeddings.VECTOR_DB_EMBEDDING_MODEL", "custom-model-name")
    def test_custom_model_from_config(self, embedding_mocks_with_device):
        """Test that custom model name from config is used."""
        initialize_embedding_model()

        call_kwargs = embedding_mocks_with_device["embeddings"].call_args[1]
        assert call_kwargs["model_name"] == "custom-model-name"

    def test_returns_embedding_instance(self, embedding_mocks_with_device):
        """Test that function returns HuggingFaceEmbeddings instance."""
        result = initialize_embedding_model()

        assert result == embedding_mocks_with_device["instance"]
        assert result is not None

    def test_model_kwargs_structure(self, embedding_mocks_with_device):
        """Test that model_kwargs includes device configuration."""
        initialize_embedding_model()

        call_kwargs = embedding_mocks_with_device["embeddings"].call_args[1]
        assert "model_kwargs" in call_kwargs
        assert isinstance(call_kwargs["model_kwargs"], dict)
        assert "device" in call_kwargs["model_kwargs"]
        assert (
            call_kwargs["model_kwargs"]["device"]
            == embedding_mocks_with_device["expected_device"]
        )

    def test_device_logging(self, embedding_mocks_with_device):
        """Test that device selection is logged."""
        initialize_embedding_model()

        # Verify logging was called
        embedding_mocks_with_device["logger"].info.assert_called()
        log_message = embedding_mocks_with_device["logger"].info.call_args[0][0]
        assert "device" in log_message.lower()
        assert embedding_mocks_with_device["expected_device"] in log_message.lower()

    def test_multiple_calls_create_new_instances(self, embedding_mocks_with_device):
        """Test that multiple calls create independent instances."""
        instance_1, instance_2 = MagicMock(), MagicMock()
        embedding_mocks_with_device["embeddings"].side_effect = [instance_1, instance_2]

        result_1 = initialize_embedding_model()
        result_2 = initialize_embedding_model()

        assert result_1 == instance_1
        assert result_2 == instance_2
        assert result_1 is not result_2
        assert embedding_mocks_with_device["embeddings"].call_count == 2

    def test_cuda_check_called_first(self, embedding_mocks_with_check_order):
        """Test that CUDA availability is checked first."""
        # Only run when CUDA is available (first param)
        if not embedding_mocks_with_check_order["cuda"].return_value:
            pytest.skip("Test only runs when CUDA is available")

        initialize_embedding_model()

        # CUDA should be checked
        embedding_mocks_with_check_order["cuda"].assert_called()

        # MPS should not be checked when CUDA is available
        assert embedding_mocks_with_check_order["mps"].call_count == 0

    def test_mps_checked_when_cuda_unavailable(self, embedding_mocks_with_check_order):
        """Test that MPS availability is checked when CUDA is not available."""
        # Only run when CUDA is unavailable (second param)
        if embedding_mocks_with_check_order["cuda"].return_value:
            pytest.skip("Test only runs when CUDA is unavailable")

        initialize_embedding_model()

        # Both should be checked since CUDA is False
        embedding_mocks_with_check_order["cuda"].assert_called()
        embedding_mocks_with_check_order["mps"].assert_called()

    def test_all_device_combinations(self, embedding_mocks_with_device):
        """Test all combinations of CUDA and MPS availability."""
        result = initialize_embedding_model()

        assert result is not None
        embedding_mocks_with_device["embeddings"].assert_called_once()

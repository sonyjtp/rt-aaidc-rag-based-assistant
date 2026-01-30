"""
Fixtures for embedding-related tests.
"""

# pylint: disable=redefined-outer-name

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def embedding_mocks():
    """Fixture providing mocked embedding components."""
    with patch("src.embeddings.HuggingFaceEmbeddings") as mock_embeddings, patch(
        "src.embeddings.torch.cuda.is_available"
    ) as mock_cuda, patch(
        "src.embeddings.torch.backends.mps.is_available"
    ) as mock_mps, patch(
        "src.embeddings.logger"
    ) as mock_logger:
        mock_instance = MagicMock()
        mock_embeddings.return_value = mock_instance
        yield {
            "embeddings": mock_embeddings,
            "cuda": mock_cuda,
            "mps": mock_mps,
            "instance": mock_instance,
            "logger": mock_logger,
        }


@pytest.fixture(
    params=[
        pytest.param(
            {"cuda": True, "mps": True, "device": "cuda"},
            id="cuda_and_mps",
        ),
        pytest.param(
            {"cuda": True, "mps": False, "device": "cuda"},
            id="cuda_only",
        ),
        pytest.param(
            {"cuda": False, "mps": True, "device": "mps"},
            id="mps_only",
        ),
        pytest.param(
            {"cuda": False, "mps": False, "device": "cpu"},
            id="cpu_only",
        ),
    ]
)
def device_config(request):
    """
    Parameterized fixture providing device availability configurations.

    Yields a dict with cuda, mps availability and expected device.
    """
    return request.param


@pytest.fixture(
    params=[
        pytest.param(
            {"cuda": True, "mps": False, "test_name": "cuda_check_called_first"},
            id="cuda_available",
        ),
        pytest.param(
            {
                "cuda": False,
                "mps": True,
                "test_name": "mps_checked_when_cuda_unavailable",
            },
            id="mps_fallback",
        ),
    ]
)
def device_check_order_config(request):
    """
    Parameterized fixture for testing device check ordering.

    Provides configurations for testing CUDA priority and MPS fallback.
    """
    return request.param


@pytest.fixture
def embedding_mocks_with_device(embedding_mocks, device_config):
    """
    Composite fixture that configures embedding_mocks based on device_config.

    Automatically sets cuda and mps return values from the device_config parameter.
    """
    embedding_mocks["cuda"].return_value = device_config["cuda"]
    embedding_mocks["mps"].return_value = device_config["mps"]

    return {
        **embedding_mocks,
        "expected_device": device_config["device"],
    }


@pytest.fixture
def embedding_mocks_with_check_order(embedding_mocks, device_check_order_config):
    """
    Composite fixture that configures embedding_mocks for device check ordering tests.

    Automatically sets cuda and mps return values for order-specific tests.
    """
    embedding_mocks["cuda"].return_value = device_check_order_config["cuda"]
    embedding_mocks["mps"].return_value = device_check_order_config["mps"]

    return {
        **embedding_mocks,
        "test_name": device_check_order_config["test_name"],
    }

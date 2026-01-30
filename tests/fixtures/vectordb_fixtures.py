"""
Fixtures for VectorDB-related tests.
"""

# pylint: disable=redefined-outer-name

from unittest.mock import MagicMock, patch

import pytest

from src.vectordb import VectorDB


@pytest.fixture
def vectordb_embedding_mock():
    """
    Fixture providing a mocked embedding model with standard setup.

    Returns a tuple of (mock_embedding, mock_embedding_model) configured for VectorDB tests.
    """
    mock_embedding_model = MagicMock()
    mock_embedding_model.model_name = "test-model"
    mock_embedding_model.embed_query.return_value = [0.1, 0.2, 0.3]

    return mock_embedding_model


@pytest.fixture
def vectordb_chroma_mock():
    """
    Fixture providing a mocked ChromaDB client with standard collection setup.

    Returns a tuple of (mock_chroma, mock_collection) configured for VectorDB tests.
    """
    mock_collection = MagicMock()
    mock_chroma_instance = MagicMock()
    mock_chroma_instance.get_or_create_collection.return_value = mock_collection

    mock_chroma = MagicMock(return_value=mock_chroma_instance)

    return mock_chroma, mock_collection


@pytest.fixture
def vectordb_mocks(vectordb_chroma_mock, vectordb_embedding_mock):
    """
    Composite fixture providing both ChromaDB and embedding mocks with standard setup.

    This fixture combines vectordb_chroma_mock and vectordb_embedding_mock for tests
    that need both.

    Returns:
        dict: Contains 'chroma', 'collection', and 'embedding_model' keys
    """
    mock_chroma, mock_collection = vectordb_chroma_mock
    mock_embedding_model = vectordb_embedding_mock

    # Ensure the mock_chroma return value has the collection
    mock_chroma_instance = mock_chroma.return_value
    mock_chroma_instance.get_or_create_collection.return_value = mock_collection

    return {
        "chroma": mock_chroma,
        "chroma_instance": mock_chroma_instance,
        "collection": mock_collection,
        "embedding_model": mock_embedding_model,
    }


@pytest.fixture
def patched_vectordb(vectordb_mocks):
    """
    Fixture providing a VectorDB instance with mocked dependencies.

    Returns:
        tuple: (VectorDB instance, mocks dict)
    """

    with patch("src.vectordb.ChromaDBClient") as mock_chroma_class, patch(
        "src.vectordb.initialize_embedding_model"
    ) as mock_embedding_func:
        mock_chroma_class.return_value = vectordb_mocks["chroma_instance"]
        mock_embedding_func.return_value = vectordb_mocks["embedding_model"]

        vdb = VectorDB()

        # Yield the instance and mocks for the test
        yield vdb, vectordb_mocks

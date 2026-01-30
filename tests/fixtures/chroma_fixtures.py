"""
Fixtures for ChromaDB-related tests.
"""

# pylint: disable=import-error

from unittest.mock import patch

import pytest


@pytest.fixture
def chroma_env():
    """Fixture providing mocked ChromaDB environment variables."""
    chroma_config = {
        "CHROMA_API_KEY": "test-key",
        "CHROMA_TENANT": "test-tenant",
        "CHROMA_DATABASE": "test-db",
    }
    with patch.dict("os.environ", chroma_config):
        yield chroma_config


def chroma_env_patch():
    """Decorator for patching ChromaDB environment variables on test methods."""
    return patch.dict(
        "os.environ",
        {
            "CHROMA_API_KEY": "test-key",
            "CHROMA_TENANT": "test-tenant",
            "CHROMA_DATABASE": "test-db",
        },
    )

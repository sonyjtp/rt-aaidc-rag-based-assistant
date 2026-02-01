"""
Pytest configuration file for RAG Assistant tests.
Defines fixtures, plugins, and test behavior.
"""
import os
import sys

# pylint: disable=import-error, unused-import, wildcard-import, unused-wildcard-import, wrong-import-position


os.environ.setdefault("LOG_LEVEL", "WARNING")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


from unittest.mock import MagicMock  # noqa: F401,E402

import pytest  # noqa: F401,E402

# Import fixture modules so their fixtures are registered with pytest.
# Using wildcard imports into conftest registers fixtures for pytest collection.
# Wildcard imports are intentional here; silence lint warnings with noqa.
from tests.fixtures.chroma_fixtures import *  # noqa: F401,F403,E402
from tests.fixtures.embedding_fixtures import *  # noqa: F401,F403,E402
from tests.fixtures.file_utils_fixtures import *  # noqa: F401,F403,E402
from tests.fixtures.llm_fixtures import *  # noqa: F401,F403,E402
from tests.fixtures.vectordb_fixtures import *  # noqa: F401,F403,E402

# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")


# ============================================================================
# SHARED FIXTURES
# ============================================================================


@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration."""
    return {
        "test_data_dir": os.path.join(os.path.dirname(__file__), "fixtures"),
        "mock_llm_model": "test-model",
        "mock_strategy": "rag_enhanced_reasoning",
    }


@pytest.fixture
def mock_logger():
    """Provide mock logger for testing."""
    return MagicMock()


# ============================================================================
# PYTEST HOOKS
# ============================================================================


def pytest_collection_modifyitems(items):
    """Modify test collection to add markers."""
    for item in items:
        # Mark all tests in test_hallucination_prevention.py as integration
        if "hallucination_prevention" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        # Mark all other tests as unit
        else:
            item.add_marker(pytest.mark.unit)

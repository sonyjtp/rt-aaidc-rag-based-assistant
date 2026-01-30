"""
Fixtures for app-related tests.
"""

# pylint: disable=import-error

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mocked_assistant():
    """Fixture providing a mocked RAGAssistant instance."""
    mock_assistant_instance = MagicMock()
    mock_assistant_instance.invoke.return_value = "Response"
    yield mock_assistant_instance
    mock_assistant_instance.reset_mock()


@pytest.fixture
def app_mocks():
    """Fixture providing mocked app dependencies."""
    with (
        patch("src.app.input") as mock_input,
        patch("src.app.RAGAssistant") as mock_assistant,
        patch("src.app.load_documents", return_value=["Doc 1"]) as mock_load_docs,
    ):
        yield {
            "input": mock_input,
            "assistant": mock_assistant,
            "load_docs": mock_load_docs,
        }


@pytest.fixture
def exception_target(request):
    """
    Indirect fixture for parametrized exception handling.

    Unpacks the (exception, target) tuple from parametrize and applies patch.
    Usage: @pytest.mark.parametrize("exception_target", [...], indirect=True)
    """
    exception, target = request.param
    with patch(target, side_effect=exception):
        yield

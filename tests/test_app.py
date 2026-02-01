"""
Integration tests for app.py CLI interface.
Tests the main entry point with mocked user input.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.app import main


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


class TestAppMain:
    """Test the main CLI application."""

    @staticmethod
    def _assert_called_state(mock_obj, should_call):
        if should_call:
            mock_obj.assert_called_once()
        else:
            mock_obj.assert_not_called()

    @pytest.fixture(autouse=True)
    def reset_mocks(self):
        """Reset mocks after each test."""
        yield

    @pytest.mark.parametrize(
        "user_input,expected_assistant_called,expected_add_documents_called,expected_invoke_calls,"
        "expected_load_docs_called",
        [
            pytest.param(
                ["What is AI?", "q"], True, True, 1, True, id="single-question"
            ),
            pytest.param(
                ["What is AI?", "Tell me more", "Q"],
                True,
                True,
                2,
                True,
                id="two-questions",
            ),
            pytest.param(["q"], True, True, 0, True, id="immediate-quit"),
        ],
    )
    def test_main_behaviors(
        self,
        app_mocks,
        mocked_assistant,
        user_input,
        expected_assistant_called,
        expected_add_documents_called,
        expected_invoke_calls,
        expected_load_docs_called,
    ):  # pylint: disable=too-many-arguments, redefined-outer-name
        """Test main() behavior with various user input scenarios."""

        app_mocks["input"].side_effect = user_input
        app_mocks["assistant"].return_value = mocked_assistant

        main()

        self._assert_called_state(app_mocks["assistant"], expected_assistant_called)
        self._assert_called_state(app_mocks["load_docs"], expected_load_docs_called)
        self._assert_called_state(
            mocked_assistant.add_documents, expected_add_documents_called
        )
        assert mocked_assistant.invoke.call_count == expected_invoke_calls

    @pytest.mark.parametrize(
        "exception,target",
        [
            pytest.param(
                FileNotFoundError("Documents not found"),
                "src.app.load_documents",
                id="file_not_found",
            ),
            pytest.param(
                ValueError("Invalid document format"),
                "src.app.load_documents",
                id="value_error",
            ),
            pytest.param(
                RuntimeError("Model loading failed"),
                "src.app.RAGAssistant",
                id="runtime_error",
            ),
        ],
    )
    def test_main_handles_exceptions(self, exception, target):
        """Test that main() handles various exceptions gracefully."""
        with patch(target, side_effect=exception), patch(
            "src.app.logger"
        ) as mock_logger:
            main()
            mock_logger.error.assert_called_once()

"""
Integration tests for app.py CLI interface.
Tests the main entry point with mocked user input.
"""

from unittest.mock import patch

import pytest

from src.app import main


class TestAppMain:
    """Test the main CLI application."""

    @pytest.fixture(autouse=True)
    def reset_mocks(self):
        """Reset mocks after each test."""
        yield

    def test_main_loads_documents(self, app_mocks, mocked_assistant):
        """Test that main() loads documents on startup."""
        app_mocks["input"].side_effect = ["What is AI?", "quit"]
        app_mocks["assistant"].return_value = mocked_assistant
        main()
        app_mocks["load_docs"].assert_called_once()

    @pytest.mark.parametrize(
        "user_input,expected_invoke_calls",
        [
            (["What is AI?", "quit"], 1),
            (["What is AI?", "Tell me more", "quit"], 2),
        ],
    )
    def test_main_assistant_behavior(
        self, app_mocks, mocked_assistant, user_input, expected_invoke_calls
    ):
        """Test that main() initializes assistant and processes queries correctly."""
        app_mocks["input"].side_effect = user_input
        app_mocks["assistant"].return_value = mocked_assistant
        main()
        app_mocks["assistant"].assert_called_once()
        mocked_assistant.add_documents.assert_called_once()
        assert mocked_assistant.invoke.call_count == expected_invoke_calls

    def test_main_quit_immediately(self, app_mocks, mocked_assistant):
        """Test that main() handles quit command immediately."""
        app_mocks["input"].side_effect = ["quit"]
        app_mocks["assistant"].return_value = mocked_assistant
        main()
        mocked_assistant.invoke.assert_not_called()

    @pytest.mark.parametrize(
        "exception_target",
        [
            pytest.param(
                (FileNotFoundError("Documents not found"), "src.app.load_documents"),
                id="file_not_found",
            ),
            pytest.param(
                (ValueError("Invalid document format"), "src.app.load_documents"),
                id="value_error",
            ),
            pytest.param(
                (RuntimeError("Model loading failed"), "src.app.RAGAssistant"),
                id="runtime_error",
            ),
        ],
        indirect=True,
    )
    def test_main_handles_exceptions(self, exception_target):
        """Test that main() handles various exceptions gracefully."""
        with patch("src.app.logger") as mock_logger:
            main()
            mock_logger.error.assert_called_once()

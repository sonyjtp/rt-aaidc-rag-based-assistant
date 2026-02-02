"""Shared fixtures for Streamlit app tests."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def streamlit_mocks():
    """Fixture providing mocked Streamlit and app dependencies."""
    with (
        patch("src.streamlit_app.st") as mock_st,
        patch("src.streamlit_app.load_documents", return_value=["Doc 1", "Doc 2"]) as mock_load_docs,
        patch("src.streamlit_app.RAGAssistant") as mock_assistant_class,
        patch("src.streamlit_app.logger") as mock_logger,
        patch("src.streamlit_app.configure_page") as mock_configure,
        patch("src.streamlit_app.load_custom_styles") as mock_styles,
        patch("src.streamlit_app.validate_and_filter_topics", side_effect=lambda x: x) as mock_validate,
    ):
        # Setup session state mock
        mock_st.session_state = MagicMock()
        mock_st.session_state.assistant = None
        mock_st.session_state.documents_loaded = False
        mock_st.session_state.chat_history = []
        mock_st.session_state.initialized = False
        mock_st.session_state.initialization_attempted = False

        # Setup assistant mock
        mock_assistant_instance = MagicMock()
        mock_assistant_instance.invoke.return_value = "# Answer\n\n---\n\nTest response"
        mock_assistant_class.return_value = mock_assistant_instance

        # Setup status context manager
        mock_status = MagicMock()
        mock_st.status.return_value = mock_status

        yield {
            "st": mock_st,
            "load_docs": mock_load_docs,
            "assistant_class": mock_assistant_class,
            "assistant_instance": mock_assistant_instance,
            "logger": mock_logger,
            "configure": mock_configure,
            "styles": mock_styles,
            "validate": mock_validate,
            "status": mock_status,
        }

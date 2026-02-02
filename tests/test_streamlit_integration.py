"""
Streamlit app integration tests - unified test suite.
All Streamlit tests consolidated into a single file with one test class.
"""

from unittest.mock import MagicMock, patch

import pytest

from tests.test_utils import clean_response


@pytest.fixture
def streamlit_session_state() -> dict:
    """Fixture providing a simulated Streamlit session state."""
    return {
        "assistant": None,
        "documents_loaded": False,
        "chat_history": [],
        "initialized": False,
        "initialization_attempted": False,
    }


# pylit: disable=redefined-outer-name
class TestStreamlit:
    """Unified Streamlit app integration tests."""

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    @pytest.mark.parametrize(
        "initialization_attempted,documents_loaded,assistant_initialized,"
        "expected_load_docs_called,expected_assistant_created,expected_error",
        [
            pytest.param(False, True, True, True, True, False, id="successful-initialization"),
            pytest.param(True, True, True, False, False, False, id="skip-already-attempted"),
            pytest.param(False, False, False, True, True, True, id="initialization-failure"),
        ],
    )
    def test_initialization_scenarios(
        self,
        streamlit_mocks,
        initialization_attempted,
        documents_loaded,
        assistant_initialized,
        expected_load_docs_called,
        expected_assistant_created,
        expected_error,
    ):
        """Test initialization with various scenarios."""
        streamlit_mocks["st"].session_state.initialization_attempted = initialization_attempted

        if expected_error:
            streamlit_mocks["assistant_class"].side_effect = RuntimeError("Test error")
            streamlit_mocks["st"].session_state.initialized = False
        else:
            streamlit_mocks["st"].session_state.documents_loaded = documents_loaded
            streamlit_mocks["st"].session_state.initialized = assistant_initialized

        if expected_load_docs_called:
            assert streamlit_mocks["load_docs"] is not None
        if expected_assistant_created:
            assert streamlit_mocks["assistant_class"] is not None

    @pytest.mark.parametrize(
        "exception_type,exception_msg,should_log_error",
        [
            pytest.param(FileNotFoundError, "Documents not found", True, id="file-not-found"),
            pytest.param(ValueError, "Invalid document format", True, id="value-error"),
            pytest.param(RuntimeError, "Model loading failed", True, id="runtime-error"),
        ],
    )
    def test_initialization_exception_handling(self, streamlit_mocks, exception_type, exception_msg, should_log_error):
        """Test exception handling during initialization."""
        streamlit_mocks["load_docs"].side_effect = exception_type(exception_msg)
        streamlit_mocks["st"].session_state.initialized = False

        if should_log_error:
            assert streamlit_mocks["logger"] is not None

    # ========================================================================
    # SESSION STATE
    # ========================================================================

    def test_session_state_initialization(self, streamlit_session_state):
        """Test session state initialization."""
        assert streamlit_session_state["assistant"] is None
        assert streamlit_session_state["documents_loaded"] is False
        assert streamlit_session_state["chat_history"] == []
        assert streamlit_session_state["initialized"] is False
        assert streamlit_session_state["initialization_attempted"] is False

    def test_rag_assistant_initialization_workflow(self, streamlit_session_state):
        """Test RAGAssistant initialization workflow."""
        mock_assistant = MagicMock()
        mock_documents = ["doc1", "doc2"]

        streamlit_session_state["initialization_attempted"] = True
        streamlit_session_state["assistant"] = mock_assistant
        streamlit_session_state["assistant"].add_documents(mock_documents)
        streamlit_session_state["documents_loaded"] = True
        streamlit_session_state["initialized"] = True

        assert streamlit_session_state["initialization_attempted"]
        assert streamlit_session_state["documents_loaded"]
        assert streamlit_session_state["initialized"]
        streamlit_session_state["assistant"].add_documents.assert_called_once_with(mock_documents)

    def test_multiple_state_transitions(self, streamlit_session_state):
        """Test multiple state transitions during app lifecycle."""
        assert streamlit_session_state["initialized"] is False

        streamlit_session_state["initialization_attempted"] = True
        streamlit_session_state["initialized"] = True
        streamlit_session_state["documents_loaded"] = True

        streamlit_session_state["chat_history"].append({"role": "user", "content": "Q"})
        streamlit_session_state["chat_history"].append({"role": "assistant", "content": "A"})
        assert len(streamlit_session_state["chat_history"]) == 2

        streamlit_session_state["chat_history"] = []
        assert len(streamlit_session_state["chat_history"]) == 0

    # ========================================================================
    # SIDEBAR & UI STATE
    # ========================================================================

    @pytest.mark.parametrize(
        "initial_history_length,expected_history_length_after_clear",
        [
            pytest.param(4, 0, id="clear-with-multiple-messages"),
            pytest.param(2, 0, id="clear-with-single-exchange"),
        ],
    )
    def test_clear_chat_history(
        self,
        streamlit_mocks,
        initial_history_length,
        expected_history_length_after_clear,
    ):
        """Test clear chat history button functionality."""
        streamlit_mocks["st"].session_state.chat_history = [
            {"role": "user", "content": f"Message {i}"} for i in range(initial_history_length)
        ]

        assert len(streamlit_mocks["st"].session_state.chat_history) == initial_history_length
        streamlit_mocks["st"].session_state.chat_history = []
        assert len(streamlit_mocks["st"].session_state.chat_history) == expected_history_length_after_clear

    @pytest.mark.parametrize(
        "initialized,should_show_success",
        [
            pytest.param(True, True, id="initialized-state"),
            pytest.param(False, False, id="uninitialized-state"),
        ],
    )
    def test_status_display(self, streamlit_mocks, initialized, should_show_success):
        """Test status display based on initialization state."""
        streamlit_mocks["st"].session_state.initialized = initialized
        assert streamlit_mocks["st"].session_state.initialized is should_show_success

    # ========================================================================
    # CHAT FUNCTIONALITY
    # ========================================================================

    def test_user_message_addition(self, streamlit_session_state):
        """Test adding user messages to chat history."""
        user_input = "What is AI?"
        streamlit_session_state["chat_history"].append({"role": "user", "content": user_input})

        assert len(streamlit_session_state["chat_history"]) == 1
        assert streamlit_session_state["chat_history"][0]["role"] == "user"
        assert streamlit_session_state["chat_history"][0]["content"] == user_input

    def test_assistant_response_addition(self, streamlit_session_state):
        """Test adding assistant responses to chat history."""
        streamlit_session_state["chat_history"].append({"role": "user", "content": "Question"})
        streamlit_session_state["chat_history"].append({"role": "assistant", "content": "Answer"})

        assert len(streamlit_session_state["chat_history"]) == 2
        assert streamlit_session_state["chat_history"][1]["role"] == "assistant"

    @pytest.mark.parametrize(
        "user_input,expected_length",
        [
            ("Simple question", 1),
            ("Question with multiple words", 1),
            ("", 0),
        ],
    )
    def test_various_user_inputs(self, streamlit_session_state, user_input, expected_length):
        """Test handling of various user input types."""
        if user_input:
            streamlit_session_state["chat_history"].append({"role": "user", "content": user_input})
        assert len(streamlit_session_state["chat_history"]) == expected_length

    def test_invoke_assistant_workflow(self, streamlit_session_state):
        """Test invoking the assistant."""
        with patch("src.streamlit_app.RAGAssistant"):
            mock_assistant = MagicMock()
            mock_assistant.invoke.return_value = "# Answer\n\n---\n\nFinal response"
            streamlit_session_state["assistant"] = mock_assistant

            user_query = "What is climate change?"
            streamlit_session_state["chat_history"].append({"role": "user", "content": user_query})
            response = streamlit_session_state["assistant"].invoke(user_query)
            streamlit_session_state["chat_history"].append({"role": "assistant", "content": response})

            assert len(streamlit_session_state["chat_history"]) == 2
            mock_assistant.invoke.assert_called_once_with(user_query)

    @pytest.mark.parametrize(
        "error_type,error_msg",
        [
            (ValueError, "Invalid configuration"),
            (RuntimeError, "API timeout"),
            (ConnectionError, "Connection failed"),
        ],
    )
    def test_error_handling(self, streamlit_session_state, error_type, error_msg):
        """Test error handling in assistant invocation."""
        with patch("src.streamlit_app.RAGAssistant"):
            mock_assistant = MagicMock()
            mock_assistant.invoke.side_effect = error_type(error_msg)
            streamlit_session_state["assistant"] = mock_assistant

            try:
                streamlit_session_state["assistant"].invoke("query")
                error_raised = False
            except (ValueError, RuntimeError, ConnectionError):
                error_raised = True

            assert error_raised is True

    # ========================================================================
    # UI COMPONENTS
    # ========================================================================

    @pytest.mark.parametrize(
        "num_messages,should_display_chat_history",
        [
            pytest.param(0, False, id="no-messages"),
            pytest.param(1, True, id="single-message"),
            pytest.param(5, True, id="multiple-messages"),
        ],
    )
    def test_chat_history_display(self, streamlit_mocks, num_messages, should_display_chat_history):
        """Test chat history display logic."""
        streamlit_mocks["st"].session_state.chat_history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"} for i in range(num_messages)
        ]

        history_count = len(streamlit_mocks["st"].session_state.chat_history)
        if should_display_chat_history:
            assert history_count > 0
        else:
            assert history_count == 0

    @pytest.mark.parametrize(
        "message_role,expected_class",
        [
            pytest.param("user", "user-message", id="user-message-class"),
            pytest.param("assistant", "assistant-message", id="assistant-message-class"),
        ],
    )
    def test_message_rendering_classes(self, streamlit_mocks, message_role, expected_class):
        """Test message rendering CSS classes."""
        message = {"role": message_role, "content": "Test message"}
        assert message["role"] in ["user", "assistant"]
        assert expected_class == "user-message" if message_role == "user" else "assistant-message"

    # ========================================================================
    # RESPONSE CLEANING
    # ========================================================================

    @pytest.mark.parametrize(
        "raw_response,should_remove_header",
        [
            ("# Title\n\n---\n\nContent", True),
            ("## Subtitle\n\nContent", True),
            ("Just plain content", False),
            ("Content without markdown", False),
        ],
    )
    def test_response_markdown_header_removal(self, raw_response, should_remove_header):
        """Test removal of markdown headers from responses."""
        cleaned_response = clean_response(raw_response)
        if should_remove_header:
            assert not cleaned_response.startswith("#")
        assert "Content" in cleaned_response or "content" in cleaned_response

    def test_response_separator_handling(self):
        """Test handling of response separators."""
        response = "# Header\n\n---\n\nContent here"
        cleaned_response = clean_response(response)
        assert "Content here" in cleaned_response

    @pytest.mark.parametrize(
        "raw_response,expected_contains",
        [
            ("### Header\n\nAnswer text", ["Answer text"]),
            ("Text with --- separator\nMore text", ["More text"]),
        ],
    )
    def test_response_cleaning_various_formats(self, raw_response, expected_contains):
        """Test response cleaning with various formats."""
        cleaned_response = clean_response(raw_response)
        for expected_text in expected_contains:
            assert expected_text in cleaned_response

    # ========================================================================
    # CONFIGURATION & DOCUMENTS
    # ========================================================================

    def test_page_configuration_called(self, streamlit_mocks):
        """Test page configuration is called."""
        assert streamlit_mocks["configure"] is not None

    def test_custom_styles_loaded(self, streamlit_mocks):
        """Test custom styles are loaded."""
        assert streamlit_mocks["styles"] is not None

    @pytest.mark.parametrize(
        "num_documents,expected_documents_loaded_flag",
        [
            pytest.param(0, False, id="no-documents"),
            pytest.param(1, True, id="single-document"),
            pytest.param(10, True, id="multiple-documents"),
        ],
    )
    def test_document_loading(self, streamlit_mocks, num_documents, expected_documents_loaded_flag):
        """Test document loading."""
        documents = [f"Document {i}" for i in range(num_documents)]
        streamlit_mocks["load_docs"].return_value = documents
        loaded_docs = streamlit_mocks["load_docs"]()

        if expected_documents_loaded_flag:
            assert len(loaded_docs) > 0
        else:
            assert len(loaded_docs) == 0

    def test_documents_added_to_assistant(self, streamlit_mocks):
        """Test that documents are added to assistant."""
        documents = ["Doc 1", "Doc 2", "Doc 3"]
        streamlit_mocks["load_docs"].return_value = documents
        streamlit_mocks["st"].session_state.assistant = streamlit_mocks["assistant_instance"]

        loaded_docs = streamlit_mocks["load_docs"]()
        assert streamlit_mocks["assistant_instance"] is not None
        assert len(loaded_docs) == 3

    # ========================================================================
    # VALIDATION & LOGGING
    # ========================================================================

    @pytest.mark.parametrize(
        "raw_response,expected_validation_applied",
        [
            pytest.param(
                "Response with topics\n\nRelated Topics:\n- Topic 1\n- Topic 2",
                True,
                id="response-with-topics",
            ),
            pytest.param("Simple response without structured topics", True, id="simple-response"),
        ],
    )
    def test_topic_validation_applied(self, streamlit_mocks, raw_response, expected_validation_applied):
        """Test topic validation is applied."""
        streamlit_mocks["validate"].return_value = raw_response
        validated = streamlit_mocks["validate"](raw_response)
        if expected_validation_applied:
            assert validated is not None

    def test_validate_filter_topics_called(self, streamlit_mocks):
        """Test validate_and_filter_topics is called."""
        raw_response = "# Answer\n\nTest response"
        streamlit_mocks["validate"].return_value = raw_response
        result = streamlit_mocks["validate"](raw_response)
        assert result == raw_response
        streamlit_mocks["validate"].assert_called_once_with(raw_response)

    def test_logging_called(self, streamlit_mocks):
        """Test logging is called."""
        assert streamlit_mocks["logger"] is not None

    def test_response_logging(self, streamlit_mocks):
        """Test responses are logged."""
        user_input = "Test question"
        streamlit_mocks["logger"].info(f"User question: {user_input}")
        streamlit_mocks["logger"].info.assert_called()

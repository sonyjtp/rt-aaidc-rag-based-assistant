"""Unit tests for UI utility functions."""

from unittest.mock import mock_open, patch

import pytest

# pylint: disable=protected-access
from src.ui_utils import (
    _get_valid_topics_from_documents,
    configure_page,
    load_custom_styles,
    validate_and_filter_topics,
)


@pytest.fixture
def ui_mocks():
    """Fixture providing mocked UI components."""
    with patch("src.ui_utils.st") as mock_st:
        yield {"st": mock_st}


class TestUIUtils:  # pylint: disable=too-few-public-methods
    """Test suite for UI utility functions."""

    # ========================================================================
    # LOAD CUSTOM STYLES
    # ========================================================================

    @pytest.mark.parametrize(
        "css_content",
        [
            "body { color: red; }",
            "@media (max-width: 600px) { body { font-size: 14px; } }",
            ":root { --primary-color: #007bff; }",
        ],
    )
    def test_load_custom_styles(self, ui_mocks, css_content):
        """Test loading custom CSS styles from file."""
        with patch("builtins.open", mock_open(read_data=css_content)):
            load_custom_styles()

        ui_mocks["st"].markdown.assert_called_once()
        call_args = ui_mocks["st"].markdown.call_args[0][0]
        assert css_content in call_args
        assert ui_mocks["st"].markdown.call_args[1]["unsafe_allow_html"] is True

    def test_load_custom_styles_file_not_found(self, ui_mocks):
        """Test handling when CSS file is not found."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            load_custom_styles()

        ui_mocks["st"].warning.assert_called_once()
        assert "not found" in ui_mocks["st"].warning.call_args[0][0].lower()

    # ========================================================================
    # CONFIGURE PAGE
    # ========================================================================

    @pytest.mark.parametrize(
        "param_name,expected_value",
        [
            ("page_title", "RAG Chatbot"),
            ("page_icon", "🤖"),
            ("layout", "wide"),
            ("initial_sidebar_state", "expanded"),
        ],
    )
    def test_configure_page(self, ui_mocks, param_name, expected_value):
        """Test page configuration with all parameters."""
        configure_page()
        call_kwargs = ui_mocks["st"].set_page_config.call_args[1]
        assert call_kwargs[param_name] == expected_value

    # ========================================================================
    # VALIDATE AND FILTER TOPICS
    # ========================================================================

    @pytest.mark.parametrize(
        "input_response,expected_output",
        [
            ("Response. Related Topics You Can Explore: [topic1, topic2]", "Response."),
            ("Start Related Topics You Can Explore: [topic1] end", "Start  end"),
            ("No related topics here.", "No related topics here."),
            ("related topics you can explore: [case]", ""),
            ("Related Topics You Can Explore: [topic1]", ""),
            ("", ""),
        ],
    )
    def test_validate_and_filter_topics(self, input_response, expected_output):
        """Parametrized test for topic validation and filtering."""
        result = validate_and_filter_topics(input_response)
        assert result == expected_output

    # ========================================================================
    # GET VALID TOPICS FROM DOCUMENTS
    # ========================================================================

    @patch("src.ui_utils.os.path.isdir")
    @patch("src.ui_utils.os.listdir")
    def test_get_valid_topics_success(self, mock_listdir, mock_isdir):
        """Test successful extraction of topics from filenames."""
        mock_isdir.return_value = True
        mock_listdir.return_value = [
            "extinct_sports.txt",
            "ancient_civilizations.txt",
            "not_a_txt_file.py",
        ]

        result = _get_valid_topics_from_documents()
        assert result == {"extinct sports", "ancient civilizations"}

    @pytest.mark.parametrize(
        "listdir_return,expected",
        [
            ([], set()),
            (["file1.py", "file2.md"], set()),
            (["topic1.txt", "Topic2.txt", "another_topic.txt"], {"topic1", "topic2", "another topic"}),
        ],
    )
    @patch("src.ui_utils.os.path.isdir")
    @patch("src.ui_utils.os.listdir")
    def test_get_valid_topics_variations(self, mock_listdir, mock_isdir, listdir_return, expected):
        """Parametrized test for various directory states and file types."""
        mock_isdir.return_value = True
        mock_listdir.return_value = listdir_return
        result = _get_valid_topics_from_documents()
        assert result == expected

    @patch("src.ui_utils.os.path.isdir")
    def test_get_valid_topics_directory_not_exists(self, mock_isdir):
        """Test behavior when data directory doesn't exist."""
        mock_isdir.return_value = False
        result = _get_valid_topics_from_documents()
        assert result == set()

    @patch("src.ui_utils.os.path.isdir")
    @patch("src.ui_utils.os.listdir")
    def test_get_valid_topics_exception_handling(self, mock_listdir, mock_isdir):
        """Test graceful handling of permission errors."""
        mock_isdir.return_value = True
        mock_listdir.side_effect = PermissionError("Permission denied")
        result = _get_valid_topics_from_documents()
        assert result == set()

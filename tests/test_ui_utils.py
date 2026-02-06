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
    with patch("src.ui_utils.st") as mock_st, patch("src.ui_utils.os.path.dirname") as mock_dirname, patch(
        "src.ui_utils.os.path.abspath"
    ) as mock_abspath:
        mock_abspath.return_value = "/project/src/ui_utils.py"
        mock_dirname.side_effect = [
            "/project/src",  # dirname of __file__
            "/project",  # dirname of src
        ]
        yield {
            "st": mock_st,
            "dirname": mock_dirname,
            "abspath": mock_abspath,
        }


# pylint: disable=redefined-outer-name
class TestUIUtils:
    """Unified test class for all UI utility functions."""

    # ========================================================================
    # LOAD CUSTOM STYLES TESTS
    # ========================================================================

    def test_load_custom_styles_success(self, ui_mocks):
        """Test successfully loading and applying custom CSS styles."""
        css_content = "body { color: red; }"

        with patch("builtins.open", mock_open(read_data=css_content)):
            load_custom_styles()

        ui_mocks["st"].markdown.assert_called_once()
        call_args = ui_mocks["st"].markdown.call_args[0][0]
        assert "body { color: red; }" in call_args

    def test_load_custom_styles_file_not_found(self, ui_mocks):
        """Test handling when CSS file is not found."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            load_custom_styles()

        ui_mocks["st"].warning.assert_called_once()
        warning_text = ui_mocks["st"].warning.call_args[0][0]
        assert "not found" in warning_text.lower()

    @pytest.mark.parametrize(
        "css_content,expected_in_output",
        [
            ("body { color: red; }", "color: red"),
            ("@media (max-width: 600px) { body { font-size: 14px; } }", "font-size"),
            (":root { --primary-color: #007bff; }", "--primary-color"),
        ],
    )
    def test_load_custom_styles_various_css(self, ui_mocks, css_content, expected_in_output):
        """Parametrized test for various CSS content types."""
        with patch("builtins.open", mock_open(read_data=css_content)):
            load_custom_styles()

        ui_mocks["st"].markdown.assert_called_once()
        call_args = ui_mocks["st"].markdown.call_args[0][0]
        assert expected_in_output in call_args

    def test_load_custom_styles_encoding(self, ui_mocks):
        """Test that file is opened with UTF-8 encoding."""
        css_content = "body { color: red; }"

        with patch("builtins.open", mock_open(read_data=css_content)) as mock_file:
            load_custom_styles()

        call_kwargs = mock_file.call_args[1]
        assert call_kwargs["encoding"] == "utf-8"

    def test_load_custom_styles_markdown_parameters(self, ui_mocks):
        """Test that markdown is called with correct parameters."""
        css_content = "body { color: red; }"

        with patch("builtins.open", mock_open(read_data=css_content)):
            load_custom_styles()

        call_kwargs = ui_mocks["st"].markdown.call_args[1]
        assert call_kwargs["unsafe_allow_html"] is True

    # ========================================================================
    # CONFIGURE PAGE TESTS
    # ========================================================================

    def test_configure_page_calls_set_page_config(self, ui_mocks):
        """Test that set_page_config is called during page configuration."""
        configure_page()
        ui_mocks["st"].set_page_config.assert_called_once()

    @pytest.mark.parametrize(
        "param_name,expected_value",
        [
            ("page_title", "RAG Chatbot"),
            ("page_icon", "🤖"),
            ("layout", "wide"),
            ("initial_sidebar_state", "expanded"),
        ],
    )
    def test_configure_page_parameters(self, ui_mocks, param_name, expected_value):
        """Parametrized test for all page configuration parameters."""
        configure_page()
        call_kwargs = ui_mocks["st"].set_page_config.call_args[1]
        assert call_kwargs[param_name] == expected_value

    def test_configure_page_no_return_value(self):
        """Test that configure_page returns None."""
        assert configure_page() is None

    def test_configure_page_multiple_calls(self, ui_mocks):
        """Test that multiple calls to configure_page are independent."""
        configure_page()
        configure_page()
        assert ui_mocks["st"].set_page_config.call_count == 2

    # ========================================================================
    # VALIDATE AND FILTER TOPICS TESTS
    # ========================================================================

    @pytest.mark.parametrize(
        "input_response,expected_output",
        [
            (
                "This is a response. Related Topics You Can Explore: [topic1, topic2]",
                "This is a response.",
            ),
            (
                "Response text Related Topics You Can Explore: [topic1] more text",
                "Response text  more text",
            ),
            ("No related topics here.", "No related topics here."),
            ("related topics you can explore: [case insensitive]", ""),
            ("Related Topics You Can Explore: [] empty brackets", "empty brackets"),
            (
                "Start Related Topics You Can Explore: [topic1, topic2] end",
                "Start  end",
            ),
            ("", ""),
            ("Related Topics You Can Explore: [topic1, topic2]", ""),
            ("  Related Topics You Can Explore: [topic1]  ", ""),
        ],
    )
    def test_validate_and_filter_topics(self, input_response, expected_output):
        """Parametrized test for topic validation and filtering."""
        result = validate_and_filter_topics(input_response)
        assert result == expected_output

    # ========================================================================
    # GET VALID TOPICS FROM DOCUMENTS TESTS
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
            "another_topic.txt",
        ]

        result = _get_valid_topics_from_documents()

        expected = {"extinct sports", "ancient civilizations", "another topic"}
        assert result == expected

    @patch("src.ui_utils.os.path.isdir")
    def test_get_valid_topics_directory_not_exists(self, mock_isdir):
        """Test behavior when data directory doesn't exist."""
        mock_isdir.return_value = False
        result = _get_valid_topics_from_documents()
        assert result == set()

    @patch("src.ui_utils.os.path.isdir")
    @patch("src.ui_utils.os.listdir")
    @pytest.mark.parametrize(
        "listdir_return",
        [
            ([]),
            (["file1.py", "file2.md", "file3.json"]),
        ],
    )
    def test_get_valid_topics_empty_or_no_txt(self, mock_listdir, mock_isdir, listdir_return):
        """Parametrized test for empty directory and no .txt files."""
        mock_isdir.return_value = True
        mock_listdir.return_value = listdir_return
        result = _get_valid_topics_from_documents()
        assert result == set()

    @patch("src.ui_utils.os.path.isdir")
    @patch("src.ui_utils.os.listdir")
    def test_get_valid_topics_underscore_conversion(self, mock_listdir, mock_isdir):
        """Test that underscores in filenames are converted to spaces."""
        mock_isdir.return_value = True
        mock_listdir.return_value = ["topic_name.txt", "another_topic_name.txt"]
        result = _get_valid_topics_from_documents()
        expected = {"topic name", "another topic name"}
        assert result == expected

    @patch("src.ui_utils.os.path.isdir")
    @patch("src.ui_utils.os.listdir")
    def test_get_valid_topics_case_conversion(self, mock_listdir, mock_isdir):
        """Test that topic names are converted to lowercase."""
        mock_isdir.return_value = True
        mock_listdir.return_value = ["Topic_Name.txt", "ANOTHER_TOPIC.txt"]
        result = _get_valid_topics_from_documents()
        expected = {"topic name", "another topic"}
        assert result == expected

    @patch("src.ui_utils.os.path.isdir")
    @patch("src.ui_utils.os.listdir")
    def test_get_valid_topics_exception_handling(self, mock_listdir, mock_isdir):
        """Test that exceptions during directory reading are handled gracefully."""
        mock_isdir.return_value = True
        mock_listdir.side_effect = PermissionError("Permission denied")
        result = _get_valid_topics_from_documents()
        assert result == set()

    @patch("src.ui_utils.os.path.isdir")
    @patch("src.ui_utils.os.listdir")
    def test_get_valid_topics_mixed_files(self, mock_listdir, mock_isdir):
        """Test extraction with mixed file types."""
        mock_isdir.return_value = True
        mock_listdir.return_value = [
            "topic1.txt",
            "topic2.txt",
            "not_txt.py",
            "also_not_txt.md",
            "topic3.txt",
        ]
        result = _get_valid_topics_from_documents()
        expected = {"topic1", "topic2", "topic3"}
        assert result == expected

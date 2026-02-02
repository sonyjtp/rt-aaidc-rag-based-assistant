"""
Unit tests for readme_extractor.py
Tests the ReadmeExtractor class for README content extraction and caching.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.readme_extractor import ReadmeExtractor


@pytest.fixture
def sample_readme_content():
    """Fixture providing sample README content."""
    return """# RAG-Based AI Assistant

## 🎯 Overview
A Retrieval-Augmented Generation chatbot that answers questions from your documents.

## ✨ Features
- ✅ Document loading from text files
- ✅ Semantic search using embeddings
- ✅ Context-aware question answering
- ✅ Memory management

## 🏗️ Project Architecture
The system uses ChromaDB for vector storage and LangChain for orchestration.

## 🎛️ Customization Guide
Customize by modifying config.py or config/ YAML files.

## 🚀 Quick Start
Install dependencies and run: python src/app.py
"""


@pytest.fixture
def mock_readme_extractor(sample_readme_content):
    """Fixture providing mocked ReadmeExtractor dependencies."""
    with (
        patch("src.readme_extractor.Path") as mock_path_class,
        patch("src.readme_extractor._get_text_loader") as mock_get_loader,
        patch("src.readme_extractor.logger") as mock_logger,
    ):
        # Mock the Path instance
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.parent = MagicMock()
        mock_path_instance.parent.parent = MagicMock()

        def path_constructor(path_arg):
            """Return mock path instance for any path construction."""
            if isinstance(path_arg, MagicMock):
                return path_arg
            new_mock = MagicMock()
            new_mock.exists.return_value = True
            return new_mock

        mock_path_class.side_effect = path_constructor
        mock_path_class.cwd.return_value = MagicMock()

        # Mock the text loader
        def loader_factory(*args, **kwargs):
            """Return loader with sample content."""
            mock_doc = MagicMock()
            mock_doc.page_content = sample_readme_content
            mock_loader_instance = MagicMock()
            mock_loader_instance.load.return_value = [mock_doc]
            return mock_loader_instance

        mock_get_loader.side_effect = loader_factory

        yield {
            "path_class": mock_path_class,
            "path_instance": mock_path_instance,
            "logger": mock_logger,
            "get_loader": mock_get_loader,
            "sample_content": sample_readme_content,
        }


# pylit: disable=redefined-outer-name
class TestReadmeExtractorInitialization:
    """Test ReadmeExtractor initialization and README finding."""

    @pytest.mark.parametrize(
        "readme_found,should_load",
        [
            pytest.param(True, True, id="find_readme_in_search_paths"),
            pytest.param(False, False, id="readme_not_found"),
        ],
    )
    def test_initialization(self, mock_readme_extractor, readme_found, should_load):
        """Test ReadmeExtractor initialization with various path configurations."""
        if readme_found:
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_readme_extractor["path_class"].return_value = mock_path
        else:
            mock_readme_extractor["path_class"].return_value = None

        with patch("src.readme_extractor.ReadmeExtractor._find_readme") as mock_find:
            mock_find.return_value = MagicMock() if readme_found else None
            extractor = ReadmeExtractor()

        assert extractor is not None
        if should_load:
            assert extractor.content is not None
        else:
            assert extractor.content is None

    @pytest.mark.parametrize(
        "exception_type",
        [
            pytest.param(IOError("File read error"), id="ioerror"),
            pytest.param(PermissionError("Permission denied"), id="permission_error"),
        ],
    )
    def test_initialization_with_errors(self, mock_readme_extractor, exception_type):
        """Test ReadmeExtractor initialization with file access failures."""
        mock_path = MagicMock()
        mock_path.exists.return_value = True

        with patch("src.readme_extractor.ReadmeExtractor._find_readme") as mock_find:
            mock_find.return_value = mock_path
            mock_readme_extractor["get_loader"].side_effect = exception_type
            extractor = ReadmeExtractor()

        assert extractor is not None
        assert extractor.content is None


class TestExtractSection:
    """Test the _extract_section() method."""

    @pytest.mark.parametrize(
        "section_header,should_find",
        [
            pytest.param("## 🎯 Overview", True, id="overview"),
            pytest.param("## ✨ Features", True, id="features"),
            pytest.param("## 🏗️ Project Architecture", True, id="architecture"),
            pytest.param("## 🎛️ Customization Guide", True, id="customization"),
            pytest.param("## 🚀 Quick Start", True, id="quick_start"),
            pytest.param("## Nonexistent Section", False, id="not_found"),
        ],
    )
    def test_extract_section(self, sample_readme_content, section_header, should_find):
        """Test _extract_section() extracts README sections correctly."""
        extractor = ReadmeExtractor()
        extractor.content = sample_readme_content

        result = extractor._extract_section(section_header)

        if should_find:
            assert result is not None and len(result) > 0
        else:
            assert result is None

    def test_extract_section_with_no_content(self):
        """Test _extract_section() with missing content."""
        extractor = ReadmeExtractor()
        extractor.content = None
        result = extractor._extract_section("## Some Section")
        assert result is None


class TestGetMethods:
    """Test all get_* extraction methods."""

    @pytest.mark.parametrize(
        "method_name,error_message",
        [
            pytest.param("get_features", "Features information not available.", id="features"),
            pytest.param(
                "get_tools_and_models",
                "Tools and models information not available.",
                id="tools",
            ),
            pytest.param(
                "get_capabilities",
                "Capabilities information not available.",
                id="capabilities",
            ),
            pytest.param("get_overview", "Overview not available.", id="overview"),
            pytest.param(
                "get_architecture",
                "Architecture information not available.",
                id="architecture",
            ),
            pytest.param(
                "get_customization",
                "Customization information not available.",
                id="customization",
            ),
            pytest.param(
                "get_quick_start",
                "Quick start information not available.",
                id="quick_start",
            ),
        ],
    )
    def test_get_methods_without_content(self, method_name, error_message):
        """Test all get_* methods return error when no content available."""
        extractor = ReadmeExtractor()
        extractor.content = None
        method = getattr(extractor, method_name)
        assert method() == error_message

    def test_get_features_extracts_content(self, mock_readme_extractor):
        """Test get_features() extracts features information."""
        extractor = ReadmeExtractor()
        extractor.content = mock_readme_extractor["sample_content"]
        result = extractor.get_features()
        assert result is not None and result != "Features information not available."

    def test_get_overview_extracts_content(self, mock_readme_extractor):
        """Test get_overview() extracts overview information."""
        extractor = ReadmeExtractor()
        extractor.content = mock_readme_extractor["sample_content"]
        result = extractor.get_overview()
        assert result is not None and result != "Overview not available."

    def test_get_architecture_extracts_content(self, mock_readme_extractor):
        """Test get_architecture() extracts architecture information."""
        extractor = ReadmeExtractor()
        extractor.content = mock_readme_extractor["sample_content"]
        result = extractor.get_architecture()
        assert result is not None

    def test_get_customization_extracts_content(self, mock_readme_extractor):
        """Test get_customization() extracts customization information."""
        extractor = ReadmeExtractor()
        extractor.content = mock_readme_extractor["sample_content"]
        result = extractor.get_customization()
        assert result is not None

    def test_get_quick_start_extracts_content(self, sample_readme_content):
        """Test get_quick_start() extracts quick start information."""
        extractor = ReadmeExtractor()
        extractor.content = sample_readme_content
        result = extractor.get_quick_start()
        assert result is not None

    def test_get_architecture_truncation(self, mock_readme_extractor):
        """Test get_architecture() truncates long sections."""
        extractor = ReadmeExtractor()
        extractor.content = mock_readme_extractor["sample_content"] + "\n" + "x" * 2000
        result = extractor.get_architecture()
        assert len(result) <= 1600  # 1500 + "..."

    @pytest.mark.parametrize(
        "method_name",
        [
            "get_features",
            "get_tools_and_models",
            "get_overview",
            "get_architecture",
            "get_customization",
            "get_quick_start",
        ],
    )
    def test_caching_behavior(self, mock_readme_extractor, method_name):
        """Test that all get_* methods cache their results."""
        extractor = ReadmeExtractor()
        extractor.content = mock_readme_extractor["sample_content"]
        method = getattr(extractor, method_name)

        result1 = method()
        result2 = method()

        assert result1 == result2
        # Verify cache attribute is set
        cache_attr = method_name.replace("get_", "")
        if cache_attr == "tools_and_models":
            cache_attr = "tools"
        assert getattr(extractor, cache_attr) is not None


class TestReadmeExtractorIntegration:
    """Integration tests for ReadmeExtractor with multiple method calls."""

    def test_all_extraction_methods_work_together(self, mock_readme_extractor):
        """Test all extraction methods work together without interference."""
        content_with_sections = (
            mock_readme_extractor["sample_content"]
            + """

### LLM Integration
- ✅ OpenAI GPT-4

### Core RAG Capabilities
- ✅ Document loading
"""
        )
        extractor = ReadmeExtractor()
        extractor.content = content_with_sections

        features = extractor.get_features()
        tools = extractor.get_tools_and_models()
        capabilities = extractor.get_capabilities()
        overview = extractor.get_overview()
        architecture = extractor.get_architecture()
        customization = extractor.get_customization()
        quick_start = extractor.get_quick_start()

        assert all(
            v is not None
            for v in [
                features,
                tools,
                capabilities,
                overview,
                architecture,
                customization,
                quick_start,
            ]
        )

    def test_extractor_with_no_readme_found(self):
        """Test extractor behavior when no README is found."""
        with patch("src.readme_extractor.ReadmeExtractor._find_readme") as mock_find:
            mock_find.return_value = None
            extractor = ReadmeExtractor()

        assert extractor.get_features() == "Features information not available."
        assert extractor.get_tools_and_models() == "Tools and models information not available."
        assert extractor.get_capabilities() == "Capabilities information not available."

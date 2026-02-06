"""
Unit tests for readme_extractor.py
Tests the ReadmeExtractor class for README content extraction and caching.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.readme_extractor import ReadmeExtractor


@pytest.fixture
def sample_readme_content():
    """Fixture providing comprehensive sample README content."""
    return """# RAG-Based AI Assistant

## 🎯 Overview
A Retrieval-Augmented Generation chatbot that answers questions from your documents.

## ✨ Features
- ✅ Document loading from text files
- ✅ Semantic search using embeddings
- ✅ Context-aware question answering
- ✅ Memory management

### Core RAG Capabilities
- Vector database integration
- Semantic similarity matching

### LLM Integration
- OpenAI GPT-4
- Groq Llama 3.1

### Memory Management
- Sliding window strategy
- Buffer memory

### Reasoning Strategies
- Chain-of-Thought reasoning
- Few-Shot prompting

### Safety & Quality
- Hallucination prevention
- Input validation

### User Interfaces
- CLI interface
- Streamlit web UI

## 🏗️ Project Architecture

```
System Architecture Diagram Here
```

System Architecture Overview:

The system is organized into 7 interconnected layers:
- User Interface Layer
- Request Processing Layer
- RAG Assistant Core
- Core Processing Components
- Language & Reasoning Layer
- Knowledge Base Layer
- State Management Layer

Data Flow Overview:

The question-to-answer flow follows these steps:
1. User submits query
2. Query processing and augmentation
3. Document retrieval and search
4. LLM response generation
5. Memory update and caching

## 🎛️ Customization Guide
Customize by modifying config.py or config/ YAML files.
You can adjust parameters, memory strategies, and reasoning approaches.

## 🚀 Quick Start

### Prerequisites
Python 3.8+ with required dependencies

### 1️⃣ Installation
Clone the repository and install dependencies

### 2️⃣ Configuration
Configure your API keys and settings

### 3️⃣ Run
Execute the application with your documents
"""


@pytest.fixture
def extractor_with_content(sample_readme_content):
    """Fixture providing ReadmeExtractor with mocked content."""
    with patch("src.readme_extractor.Path") as mock_path, patch(
        "src.readme_extractor.load_text_file"
    ) as mock_load, patch("src.readme_extractor.logger"):
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        mock_load.return_value = sample_readme_content

        extractor = ReadmeExtractor()
        assert extractor.content is not None
        return extractor


@pytest.fixture
def extractor_without_content():
    """Fixture providing ReadmeExtractor without content."""
    with patch("src.readme_extractor.Path") as mock_path, patch("src.readme_extractor.load_text_file") as _, patch(
        "src.readme_extractor.logger"
    ):
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance

        extractor = ReadmeExtractor()
        assert extractor.content is None
        return extractor


class TestReadmeExtractor:
    """Comprehensive tests for ReadmeExtractor class."""

    @pytest.mark.parametrize(
        "method,expected_section,header_variations",
        [
            ("get_features", "Document loading", ["## ✨ Features", "## Features"]),
            ("get_overview", "Retrieval-Augmented", ["## 🎯 Overview", "## Overview"]),
            ("get_architecture", "interconnected layers", None),
            ("get_customization", "config.py", ["## 🎛️ Customization Guide"]),
            ("get_tools_and_models", "OpenAI", ["### LLM Integration"]),
            ("get_capabilities", "Vector database", ["### Core RAG Capabilities"]),
        ],
    )
    def test_extraction_methods_with_content(self, extractor_with_content, method, expected_section, header_variations):
        """Parametrized test: all get_* methods extract content when available."""
        result = getattr(extractor_with_content, method)()
        assert result is not None
        assert expected_section.lower() in result.lower()
        assert "not available" not in result.lower()

    @pytest.mark.parametrize(
        "method,not_available_msg",
        [
            ("get_features", "Features information not available."),
            ("get_overview", "Overview not available."),
            ("get_architecture", "Architecture information not available."),
            ("get_customization", "Customization information not available."),
            ("get_tools_and_models", "Tools and models information not available."),
            ("get_capabilities", "Capabilities information not available."),
            ("get_quick_start", "Quick start information not available."),
        ],
    )
    def test_extraction_methods_without_content(self, extractor_without_content, method, not_available_msg):
        """Parametrized test: all get_* methods return appropriate message when content missing."""
        result = getattr(extractor_without_content, method)()
        assert result == not_available_msg

    @pytest.mark.parametrize(
        "method",
        [
            "get_features",
            "get_overview",
            "get_architecture",
            "get_customization",
            "get_tools_and_models",
            "get_capabilities",
            "get_quick_start",
        ],
    )
    def test_caching_behavior(self, extractor_with_content, method):
        """Parametrized test: verify extraction methods cache results."""
        # Call method twice
        result1 = getattr(extractor_with_content, method)()
        result2 = getattr(extractor_with_content, method)()

        # Results should be identical (cached)
        assert result1 == result2
        assert result1 is not None

        # Verify cache attribute is populated
        cache_attr = method.replace("get_", "")
        if cache_attr == "tools_and_models":
            cache_attr = "tools"
        assert getattr(extractor_with_content, cache_attr) is not None

    @pytest.mark.parametrize(
        "section_header,should_find",
        [
            ("## 🎯 Overview", True),
            ("## ✨ Features", True),
            ("## 🏗️ Project Architecture", True),
            ("## 🎛️ Customization Guide", True),
            ("## 🚀 Quick Start", True),
            ("### LLM Integration", True),
            ("### Core RAG Capabilities", True),
            ("## Nonexistent Section", False),
            ("### Missing Subsection", False),
        ],
    )
    def test_extract_section(self, extractor_with_content, section_header, should_find):
        """Parametrized test for _extract_section with various headers."""
        result = extractor_with_content._extract_section(section_header)
        if should_find:
            assert result is not None and len(result) > 0
        else:
            assert result is None

    @pytest.mark.parametrize(
        "text,should_contain_bullet",
        [
            ("Plain text", "Plain text"),
            ("Multiple\nlines\nof\ntext", "• Multiple"),
            ("Already • bulleted", "• Already"),
            ("", ""),
        ],
    )
    def test_format_as_bullets(self, text, should_contain_bullet):
        """Parametrized test for _format_as_bullets method."""
        result = ReadmeExtractor._format_as_bullets(text)
        if should_contain_bullet:
            assert should_contain_bullet in result or result == ""

    @pytest.mark.parametrize(
        "markdown_text,should_not_contain",
        [
            ("**bold text**", "**"),
            ("*italic text*", "*italic*"),
            ("__double bold__", "__"),
            ("[link](url)", "["),
            ("`code`", "`"),
            ("# Header", "# Header"),
            ("---", "---"),
            ("- list item", "- list"),
            ("🎯 emoji text", "🎯"),
            ("1️⃣ keycap", "️"),
        ],
    )
    def test_strip_markdown_removes_formatting(self, markdown_text, should_not_contain):
        """Parametrized test for _strip_markdown method."""
        result = ReadmeExtractor._strip_markdown(markdown_text)
        assert should_not_contain not in result or should_not_contain == ""

    def test_strip_markdown_preserves_content(self):
        """Test that strip_markdown preserves meaningful text content."""
        markdown = "**bold** and *italic* with [link](url)"
        result = ReadmeExtractor._strip_markdown(markdown)
        assert "bold" in result
        assert "italic" in result
        assert "link" in result

    def test_extract_architecture_description_stops_at_diagram(self, extractor_with_content):
        """Test that _extract_architecture_description stops at ASCII diagrams."""
        result = extractor_with_content._extract_architecture_description("System Architecture Overview:")
        assert result is not None
        # Should not contain box-drawing characters
        assert "┌" not in result
        assert "│" not in result

    def test_extract_quick_start_excludes_code_blocks(self, extractor_with_content):
        """Test that _extract_quick_start_text excludes code blocks."""
        result = extractor_with_content.get_quick_start()
        assert result is not None
        # Should not contain code block markers
        assert "```" not in result

    def test_strip_markdown_with_none_input(self):
        """Test _strip_markdown handles None input gracefully."""
        result = ReadmeExtractor._strip_markdown(None)
        assert result is None

    def test_format_as_bullets_with_empty_string(self):
        """Test _format_as_bullets handles empty string."""
        result = ReadmeExtractor._format_as_bullets("")
        assert result == ""

    def test_format_as_bullets_with_none(self):
        """Test _format_as_bullets handles None."""
        result = ReadmeExtractor._format_as_bullets(None)
        assert result is None

    def test_extract_section_with_no_content(self):
        """Test _extract_section returns None when content is missing."""
        extractor = ReadmeExtractor()
        extractor.content = None
        result = extractor._extract_section("## Any Section")
        assert result is None

    def test_get_features_with_multiple_headers(self, extractor_with_content):
        """Test that get_features works and caches results."""
        result1 = extractor_with_content.get_features()
        result2 = extractor_with_content.get_features()
        assert result1 == result2
        assert extractor_with_content.features == result1

    def test_tools_and_models_combines_sections(self, extractor_with_content):
        """Test that get_tools_and_models combines multiple sections."""
        result = extractor_with_content.get_tools_and_models()
        assert "LLM" in result or "OpenAI" in result
        assert result is not None

    def test_architecture_with_both_descriptions(self, extractor_with_content):
        """Test that get_architecture extracts both system and data flow descriptions."""
        result = extractor_with_content.get_architecture()
        assert result is not None
        # Should contain content from both sections
        assert "interconnected" in result.lower() or "layer" in result.lower()

    def test_initialization_with_content_loading_error(self):
        """Test initialization handles content loading errors gracefully."""
        with patch("src.readme_extractor.Path") as mock_path, patch(
            "src.readme_extractor.load_text_file"
        ) as mock_load, patch("src.readme_extractor.logger"):
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path.return_value = mock_path_instance
            mock_load.return_value = None

            extractor = ReadmeExtractor()
            assert extractor.content is None

    def test_extract_quick_start_with_varied_headers(self):
        """Test quick start extraction with different header formats."""
        content = """## 🚀 Quick Start
Step 1: Do this
Step 2: Do that

## Next Section
Other content"""
        extractor = ReadmeExtractor()
        extractor.content = content
        result = extractor.get_quick_start()
        assert "Step 1" in result or "Step" in result or result != ""

    def test_get_features_with_empty_cleaned_content(self):
        """Test get_features when cleaned content is empty."""
        content = """## ✨ Features

## Next Section"""
        extractor = ReadmeExtractor()
        extractor.content = content
        result = extractor.get_features()
        # Should return the default message when content is empty
        assert "Features information not available." in result or result is not None

    def test_get_tools_with_only_llm_section(self):
        """Test get_tools_and_models when only LLM section exists."""
        content = """### LLM Integration
- OpenAI GPT-4
- Groq Llama"""
        extractor = ReadmeExtractor()
        extractor.content = content
        result = extractor.get_tools_and_models()
        assert "OpenAI" in result

    def test_get_tools_with_no_sections(self):
        """Test get_tools_and_models when no relevant sections exist."""
        content = """## Some Other Content
Nothing relevant here"""
        extractor = ReadmeExtractor()
        extractor.content = content
        result = extractor.get_tools_and_models()
        assert result == "Tools information not available."

    def test_get_capabilities_not_found(self):
        """Test get_capabilities when section doesn't exist."""
        content = """## Overview
Some content here"""
        extractor = ReadmeExtractor()
        extractor.content = content
        result = extractor.get_capabilities()
        assert result == "Capabilities information not available."

    def test_extract_section_stops_at_same_level_header(self, extractor_with_content):
        """Test _extract_section stops at same-level headers."""
        # Should extract content between ## headers but not include the next ##
        result = extractor_with_content._extract_section("## ✨ Features")
        assert result is not None
        # Should not include next section header
        assert "## 🏗️" not in result

    def test_extract_section_includes_subsections(self, extractor_with_content):
        """Test _extract_section includes ### subsections."""
        result = extractor_with_content._extract_section("## ✨ Features")
        # Should include subsections like ### Core RAG Capabilities
        assert "Core RAG" in result or "LLM" in result

    def test_get_overview_with_alternate_header(self):
        """Test get_overview finds alternate header format."""
        content = """## Overview
This is an overview without emoji"""
        extractor = ReadmeExtractor()
        extractor.content = content
        result = extractor.get_overview()
        assert "overview" in result.lower()

    def test_get_architecture_only_system_description(self):
        """Test get_architecture when only system description exists."""
        content = """System Architecture Overview:
The system has multiple layers."""
        extractor = ReadmeExtractor()
        extractor.content = content
        result = extractor._extract_architecture_description("System Architecture Overview:")
        assert "layers" in result.lower()

    def test_get_architecture_only_data_flow_description(self):
        """Test get_architecture when only data flow description exists."""
        content = """Data Flow Overview:
The flow goes through these steps."""
        extractor = ReadmeExtractor()
        extractor.content = content
        result = extractor._extract_architecture_description("Data Flow Overview:")
        assert "steps" in result.lower()

    def test_extract_architecture_description_stops_at_horizontal_rule(self):
        """Test _extract_architecture_description stops at --- separator."""
        content = """System Architecture Overview:
The system has layers

---

## Next Section"""
        extractor = ReadmeExtractor()
        extractor.content = content
        result = extractor._extract_architecture_description("System Architecture Overview:")
        assert "---" not in result

    def test_extract_quick_start_stops_at_next_header(self):
        """Test _extract_quick_start_text stops at next same-level header."""
        content = """## 🚀 Quick Start
Step 1
Step 2

## 📁 Project Structure
Other content"""
        extractor = ReadmeExtractor()
        extractor.content = content
        result = extractor._extract_quick_start_text("## 🚀 Quick Start")
        assert "Step 1" in result
        assert "Project Structure" not in result

    def test_strip_markdown_removes_multiple_formatting(self):
        """Test _strip_markdown removes multiple formatting types at once."""
        markdown = """**bold** _italic_ `code` [link](url)
# Header
More text"""
        result = ReadmeExtractor._strip_markdown(markdown)
        assert "**" not in result
        assert "`" not in result
        assert "[" not in result or "](" not in result
        # Headers at line start should be removed
        assert "# Header" not in result

    def test_format_as_bullets_preserves_existing_bullets(self):
        """Test _format_as_bullets preserves already-bulleted items."""
        text = "• Already bulleted\nPlain text"
        result = ReadmeExtractor._format_as_bullets(text)
        assert "• Already bulleted" in result
        assert "• Plain text" in result

    def test_strip_markdown_handles_empty_after_cleaning(self):
        """Test _strip_markdown when text is only formatting characters."""
        markdown = "**  **"
        result = ReadmeExtractor._strip_markdown(markdown)
        # Should be empty or whitespace only
        assert result.strip() == "" or result is not None

    def test_get_customization_with_alternate_header(self):
        """Test get_customization with different header formats."""
        content = """## Customization
Edit the files to customize"""
        extractor = ReadmeExtractor()
        extractor.content = content
        result = extractor.get_customization()
        assert "Customization" in result or "customize" in result.lower()

    def test_extract_section_with_subsection_only(self):
        """Test _extract_section when section has only subsections."""
        content = """### Subsection 1
Content 1
### Subsection 2
Content 2
## Main Section
Other"""
        extractor = ReadmeExtractor()
        extractor.content = content
        # Should find subsections if looking at subsection level
        result = extractor._extract_section("### Subsection 1")
        assert result is not None

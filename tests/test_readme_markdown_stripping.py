"""Test markdown stripping functionality in README extractor."""

import pytest

from src.readme_extractor import ReadmeExtractor


class TestMarkdownStripping:
    """Test the _strip_markdown method."""

    @pytest.mark.parametrize(
        "input_text,expected_output",
        [
            pytest.param(
                "**bold text**",
                "bold text",
                id="remove-bold",
            ),
            pytest.param(
                "*italic text*",
                "italic text",
                id="remove-italic",
            ),
            pytest.param(
                "__bold text__",
                "bold text",
                id="remove-double-underscore-bold",
            ),
            pytest.param(
                "_italic text_",
                "italic text",
                id="remove-underscore-italic",
            ),
            pytest.param(
                "`inline code`",
                "inline code",
                id="remove-inline-code",
            ),
            pytest.param(
                "[link text](https://example.com)",
                "link text",
                id="remove-link-keep-text",
            ),
            pytest.param(
                "- Item 1\n- Item 2",
                "Item 1\nItem 2",
                id="remove-list-markers",
            ),
            pytest.param(
                "---",
                "",
                id="remove-horizontal-rule",
            ),
            pytest.param(
                "**Bold** and *italic* with `code`",
                "Bold and italic with code",
                id="remove-multiple-formats",
            ),
            pytest.param(
                "",
                "",
                id="empty-string",
            ),
            pytest.param(
                None,
                None,
                id="none-input",
            ),
            pytest.param(
                "Plain text without formatting",
                "Plain text without formatting",
                id="plain-text-unchanged",
            ),
        ],
    )
    def test_strip_markdown(self, input_text, expected_output):
        """Test markdown stripping with various inputs."""
        result = ReadmeExtractor._strip_markdown(input_text)
        assert result == expected_output

    def test_strip_markdown_preserves_meaningful_text(self):
        """Test that markdown stripping preserves non-formatting content."""
        markdown = """
# Project Overview

This is a **bold** statement with *emphasis*.

## Features

- Feature 1
- Feature 2
- Feature 3

Visit our [documentation](https://docs.example.com) for more info.

---

Check the [README](https://example.com) for details.
"""
        result = ReadmeExtractor._strip_markdown(markdown)

        # Verify meaningful content is preserved
        assert "Project Overview" in result
        assert "bold statement with emphasis" in result
        assert "Feature 1" in result
        assert "Feature 2" in result
        assert "Feature 3" in result
        assert "documentation" in result
        assert "README" in result

        # Verify markdown syntax is removed
        assert "**" not in result
        assert "*emphasis*" not in result
        assert "[" not in result or "](" in result  # Links might leave some chars
        assert "---" not in result


class TestReadmeExtractorIntegration:
    """Integration tests for README extraction with markdown stripping."""

    def test_get_features_strips_markdown(self):
        """Test that get_features strips markdown formatting."""
        extractor = ReadmeExtractor()
        if extractor.content:
            features = extractor.get_features()
            # Verify no markdown syntax in response
            assert "**" not in features
            assert "```" not in features

    def test_get_overview_strips_markdown(self):
        """Test that get_overview strips markdown formatting."""
        extractor = ReadmeExtractor()
        if extractor.content:
            overview = extractor.get_overview()
            # Verify no markdown syntax in response
            assert "**" not in overview
            assert "###" not in overview

    def test_get_architecture_strips_markdown(self):
        """Test that get_architecture strips markdown formatting."""
        extractor = ReadmeExtractor()
        if extractor.content:
            architecture = extractor.get_architecture()
            # Verify no markdown syntax in response
            assert "```" not in architecture
            assert "[" not in architecture or "](" in architecture

    def test_get_customization_strips_markdown(self):
        """Test that get_customization strips markdown formatting."""
        extractor = ReadmeExtractor()
        if extractor.content:
            customization = extractor.get_customization()
            # Verify no markdown syntax in response
            assert "**" not in customization

    def test_get_quick_start_strips_markdown(self):
        """Test that get_quick_start strips markdown formatting."""
        extractor = ReadmeExtractor()
        if extractor.content:
            quick_start = extractor.get_quick_start()
            # Verify no markdown syntax in response
            assert "```" not in quick_start

    def test_get_tools_and_models_strips_markdown(self):
        """Test that get_tools_and_models strips markdown formatting."""
        extractor = ReadmeExtractor()
        if extractor.content:
            tools = extractor.get_tools_and_models()
            # Verify no markdown syntax in response
            assert "**" not in tools
            assert "```" not in tools

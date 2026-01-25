"""
Unit tests for file utilities.
Tests document loading, YAML parsing, and file handling.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.file_utils import load_documents, load_yaml

# ============================================================================
# LOAD DOCUMENTS TESTS
# ============================================================================


class TestLoadDocuments:
    """Test document loading from text files."""

    def test_load_documents_single_file(self):
        """Test loading a single document from a text file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test file
            test_file = os.path.join(temp_dir, "test.txt")
            content = "Test Title\nTags: test, document\nThis is the content."
            with open(test_file, "w", encoding="utf-8") as f:
                f.write(content)

            # Load documents
            docs = load_documents(temp_dir)

            # Verify
            assert len(docs) == 1
            assert docs[0]["filename"] == "test.txt"
            assert docs[0]["title"] == "Test Title"
            assert docs[0]["tags"] == "test, document"
            assert "This is the content." in docs[0]["content"]

    def test_load_documents_multiple_files(self):
        """Test loading multiple documents from a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple test files
            for i in range(3):
                test_file = os.path.join(temp_dir, f"doc{i}.txt")
                with open(test_file, "w", encoding="utf-8") as f:
                    f.write(f"Document {i}\nContent {i}")

            # Load documents
            docs = load_documents(temp_dir)

            # Verify
            assert len(docs) == 3
            filenames = [doc["filename"] for doc in docs]
            assert "doc0.txt" in filenames
            assert "doc1.txt" in filenames
            assert "doc2.txt" in filenames

    def test_load_documents_with_custom_extension(self):
        """Test loading documents with custom file extension."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create files with different extensions
            txt_file = os.path.join(temp_dir, "test.txt")
            md_file = os.path.join(temp_dir, "readme.md")

            with open(txt_file, "w", encoding="utf-8") as f:
                f.write("Text File\nContent")
            with open(md_file, "w", encoding="utf-8") as f:
                f.write("Markdown File\nContent")

            # Load only markdown files
            docs = load_documents(temp_dir, file_extensions=".md")

            # Verify
            assert len(docs) == 1
            assert docs[0]["filename"] == "readme.md"

    def test_load_documents_extract_title(self):
        """Test that title is extracted from first non-empty line."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.txt")
            content = "Document Title\nContent here"
            with open(test_file, "w", encoding="utf-8") as f:
                f.write(content)

            docs = load_documents(temp_dir)

            assert docs[0]["title"] == "Document Title"

    def test_load_documents_extract_tags(self):
        """Test that tags are extracted from second line if present."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.txt")
            content = "Title\nTags: python, testing, code\nContent"
            with open(test_file, "w", encoding="utf-8") as f:
                f.write(content)

            docs = load_documents(temp_dir)

            assert docs[0]["tags"] == "python, testing, code"

    def test_load_documents_no_tags_line(self):
        """Test handling of documents without tags line."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.txt")
            content = "Title\nContent without tags"
            with open(test_file, "w", encoding="utf-8") as f:
                f.write(content)

            docs = load_documents(temp_dir)

            assert docs[0]["tags"] == ""

    def test_load_documents_empty_file(self):
        """Test loading an empty file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "empty.txt")
            with open(test_file, "w", encoding="utf-8") as f:
                f.write("")

            docs = load_documents(temp_dir)

            assert len(docs) == 1
            assert docs[0]["title"] == "empty.txt"  # Uses filename as fallback
            assert docs[0]["content"] == ""

    def test_load_documents_whitespace_handling(self):
        """Test that whitespace is properly handled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.txt")
            content = "  Title with spaces  \n  Content  "
            with open(test_file, "w", encoding="utf-8") as f:
                f.write(content)

            docs = load_documents(temp_dir)

            assert docs[0]["title"] == "Title with spaces"

    def test_load_documents_skip_non_matching_extensions(self):
        """Test that files with wrong extension are skipped."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create files with different extensions
            txt_file = os.path.join(temp_dir, "doc.txt")
            json_file = os.path.join(temp_dir, "config.json")

            with open(txt_file, "w", encoding="utf-8") as f:
                f.write("Text\nContent")
            with open(json_file, "w", encoding="utf-8") as f:
                f.write('{"key": "value"}')

            docs = load_documents(temp_dir, file_extensions=".txt")

            # Only .txt file should be loaded
            assert len(docs) == 1
            assert docs[0]["filename"] == "doc.txt"

    def test_load_documents_handles_io_error(self):
        """Test that IOError during file loading is caught."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a valid file
            test_file = os.path.join(temp_dir, "test.txt")
            with open(test_file, "w", encoding="utf-8") as f:
                f.write("Title\nContent")

            # Mock TextLoader to raise IOError
            with patch("src.file_utils.TextLoader") as mock_loader:
                mock_loader.return_value.load.side_effect = IOError("Permission denied")

                docs = load_documents(temp_dir)

                # Should return empty list due to error
                assert len(docs) == 0

    def test_load_documents_filename_as_title_fallback(self):
        """Test that filename is used as title when content is empty."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "my_document.txt")
            with open(test_file, "w", encoding="utf-8") as f:
                f.write("")  # Empty file

            docs = load_documents(temp_dir)

            assert docs[0]["title"] == "my_document.txt"

    def test_load_documents_tuple_extensions(self):
        """Test loading with tuple of extensions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create files with different extensions
            txt_file = os.path.join(temp_dir, "doc.txt")
            md_file = os.path.join(temp_dir, "readme.md")
            py_file = os.path.join(temp_dir, "code.py")

            for file_path in [txt_file, md_file, py_file]:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write("Title\nContent")

            # Load .txt and .md files
            docs = load_documents(temp_dir, file_extensions=(".txt", ".md"))

            # Should load 2 files
            assert len(docs) == 2
            filenames = [doc["filename"] for doc in docs]
            assert "doc.txt" in filenames
            assert "readme.md" in filenames

    def test_load_documents_preserves_original_content(self):
        """Test that original file content is preserved."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.txt")
            original_content = "Title\nTags: test\nMultiple\nLines\nOf\nContent"
            with open(test_file, "w", encoding="utf-8") as f:
                f.write(original_content)

            docs = load_documents(temp_dir)

            assert docs[0]["content"] == original_content

    def test_load_documents_multiple_tags_lines(self):
        """Test that only second line is checked for Tags."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.txt")
            content = "Title\nTags: first\nTags: second\nContent"
            with open(test_file, "w", encoding="utf-8") as f:
                f.write(content)

            docs = load_documents(temp_dir)

            # Only first Tags line should be extracted
            assert docs[0]["tags"] == "first"


# ============================================================================
# LOAD YAML TESTS
# ============================================================================


class TestLoadYAML:
    """Test YAML file loading and parsing."""

    def test_load_yaml_valid_file(self):
        """Test loading a valid YAML file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_file = os.path.join(temp_dir, "config.yaml")
            config = {"key": "value", "number": 42, "list": [1, 2, 3]}

            with open(yaml_file, "w", encoding="utf-8") as f:
                yaml.dump(config, f)

            result = load_yaml(yaml_file)

            assert result["key"] == "value"
            assert result["number"] == 42
            assert result["list"] == [1, 2, 3]

    def test_load_yaml_nested_structure(self):
        """Test loading YAML with nested structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_file = os.path.join(temp_dir, "config.yaml")
            config = {
                "database": {"host": "localhost", "port": 5432},
                "features": ["feature1", "feature2"],
            }

            with open(yaml_file, "w", encoding="utf-8") as f:
                yaml.dump(config, f)

            result = load_yaml(yaml_file)

            assert result["database"]["host"] == "localhost"
            assert result["database"]["port"] == 5432
            assert result["features"] == ["feature1", "feature2"]

    def test_load_yaml_empty_file(self):
        """Test loading an empty YAML file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_file = os.path.join(temp_dir, "empty.yaml")
            with open(yaml_file, "w", encoding="utf-8") as f:
                f.write("")

            result = load_yaml(yaml_file)

            assert result is None

    def test_load_yaml_file_not_found(self):
        """Test error handling when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_yaml("/nonexistent/file.yaml")

    def test_load_yaml_invalid_yaml_syntax(self):
        """Test error handling for invalid YAML syntax."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_file = os.path.join(temp_dir, "invalid.yaml")
            with open(yaml_file, "w", encoding="utf-8") as f:
                f.write("invalid: yaml: syntax: here")

            with pytest.raises(yaml.YAMLError):
                load_yaml(yaml_file)

    def test_load_yaml_with_path_object(self):
        """Test that Path objects are accepted."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_file = Path(temp_dir) / "config.yaml"
            config = {"test": "value"}

            with open(yaml_file, "w", encoding="utf-8") as f:
                yaml.dump(config, f)

            result = load_yaml(yaml_file)

            assert result["test"] == "value"

    def test_load_yaml_with_string_path(self):
        """Test that string paths are accepted."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_file = os.path.join(temp_dir, "config.yaml")
            config = {"test": "value"}

            with open(yaml_file, "w", encoding="utf-8") as f:
                yaml.dump(config, f)

            result = load_yaml(yaml_file)

            assert result["test"] == "value"

    def test_load_yaml_special_characters(self):
        """Test loading YAML with special characters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_file = os.path.join(temp_dir, "config.yaml")
            config = {
                "message": "Hello, World!",
                "path": "/home/user/documents",
                "special": "!@#$%^&*()",
            }

            with open(yaml_file, "w", encoding="utf-8") as f:
                yaml.dump(config, f)

            result = load_yaml(yaml_file)

            assert result["message"] == "Hello, World!"
            assert result["path"] == "/home/user/documents"

    def test_load_yaml_unicode_content(self):
        """Test loading YAML with unicode characters."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_file = os.path.join(temp_dir, "config.yaml")
            config = {
                "greeting": "你好",  # Chinese
                "emoji": "🚀",  # Emoji
                "arabic": "مرحبا",  # Arabic
            }

            with open(yaml_file, "w", encoding="utf-8") as f:
                yaml.dump(config, f, allow_unicode=True)

            result = load_yaml(yaml_file)

            assert result["greeting"] == "你好"
            assert result["emoji"] == "🚀"
            assert result["arabic"] == "مرحبا"

    def test_load_yaml_file_permissions_error(self):
        """Test error handling when file can't be read."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_file = os.path.join(temp_dir, "config.yaml")
            with open(yaml_file, "w", encoding="utf-8") as f:
                f.write("key: value")

            # Remove read permissions
            os.chmod(yaml_file, 0o000)

            try:
                with pytest.raises(IOError):
                    load_yaml(yaml_file)
            finally:
                # Restore permissions for cleanup
                os.chmod(yaml_file, 0o644)

    def test_load_yaml_list_structure(self):
        """Test loading YAML with list as root element."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_file = os.path.join(temp_dir, "list.yaml")
            with open(yaml_file, "w", encoding="utf-8") as f:
                f.write("- item1\n- item2\n- item3")

            result = load_yaml(yaml_file)

            assert isinstance(result, list)
            assert result == ["item1", "item2", "item3"]

    def test_load_yaml_preserves_types(self):
        """Test that YAML preserves data types correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_file = os.path.join(temp_dir, "config.yaml")
            config = {
                "string": "text",
                "integer": 42,
                "float": 3.14,
                "boolean": True,
                "null": None,
                "list": [1, 2, 3],
            }

            with open(yaml_file, "w", encoding="utf-8") as f:
                yaml.dump(config, f)

            result = load_yaml(yaml_file)

            assert isinstance(result["string"], str)
            assert isinstance(result["integer"], int)
            assert isinstance(result["float"], float)
            assert isinstance(result["boolean"], bool)
            assert result["null"] is None
            assert isinstance(result["list"], list)

    def test_load_yaml_comments_ignored(self):
        """Test that YAML comments are properly ignored."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_file = os.path.join(temp_dir, "config.yaml")
            with open(yaml_file, "w", encoding="utf-8") as f:
                f.write("# This is a comment\nkey: value  # inline comment")

            result = load_yaml(yaml_file)

            assert result["key"] == "value"

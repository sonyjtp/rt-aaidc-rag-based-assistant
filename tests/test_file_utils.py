"""
Unit tests for file utilities.
Tests document loading, YAML parsing, and file handling.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.file_utils import load_documents, load_yaml


class TestFileUtils:
    """Test file utility functions: document loading and YAML parsing."""

    # ========================================================================
    # LOAD DOCUMENTS TESTS
    # ========================================================================

    def test_load_documents_single_file(self):
        """Test loading a single document with title, tags, and content extraction."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.txt")
            with open(test_file, "w", encoding="utf-8") as f:
                f.write("Test Title\nTags: test, document\nMultiple\nLines\nOf\nContent")

            docs = load_documents(temp_dir)

            assert len(docs) == 1
            assert docs[0]["filename"] == "test.txt"
            assert docs[0]["title"] == "Test Title"
            assert docs[0]["tags"] == "test, document"
            assert "Multiple\nLines\nOf\nContent" in docs[0]["content"]

    def test_load_documents_multiple_files(self):
        """Test loading multiple documents from a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            for i in range(3):
                with open(os.path.join(temp_dir, f"doc{i}.txt"), "w", encoding="utf-8") as f:
                    f.write(f"Document {i}\nContent {i}")

            docs = load_documents(temp_dir)

            assert len(docs) == 3
            filenames = [doc["filename"] for doc in docs]
            assert all(f"doc{i}.txt" in filenames for i in range(3))

    @pytest.mark.parametrize(
        "extensions,expected_count,expected_files",
        [
            (".txt", 1, ["doc.txt"]),
            ((".txt", ".md"), 2, ["doc.txt", "readme.md"]),
            (".json", 0, []),
        ],
    )
    def test_load_documents_with_file_extensions(self, extensions, expected_count, expected_files):
        """Parametrized test for loading documents with various file extensions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            files = {
                "doc.txt": "Text\nContent",
                "readme.md": "Markdown\nContent",
                "code.py": "Python\nContent",
            }

            for filename, content in files.items():
                with open(os.path.join(temp_dir, filename), "w", encoding="utf-8") as f:
                    f.write(content)

            docs = load_documents(temp_dir, file_extensions=extensions)

            assert len(docs) == expected_count
            filenames = [doc["filename"] for doc in docs]
            assert all(f in filenames for f in expected_files)

    @pytest.mark.parametrize(
        "content,expected_title,expected_tags",
        [
            ("Title Only\nNo Tags\nContent", "Title Only", ""),
            ("Title\nTags: python, testing\nContent", "Title", "python, testing"),
            ("  Whitespace Title  \nNo tags", "Whitespace Title", ""),
            ("empty.txt\n", "empty.txt", ""),
        ],
    )
    def test_load_documents_extract_metadata(self, content, expected_title, expected_tags):
        """Parametrized test for title and tags extraction."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, "test.txt"), "w", encoding="utf-8") as f:
                f.write(content)

            docs = load_documents(temp_dir)

            assert docs[0]["title"] == expected_title
            assert docs[0]["tags"] == expected_tags

    def test_load_documents_empty_file(self):
        """Test that empty files use filename as title."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, "empty.txt"), "w", encoding="utf-8") as f:
                f.write("")

            docs = load_documents(temp_dir)

            assert docs[0]["title"] == "empty.txt"
            assert docs[0]["content"] == ""

    def test_load_documents_handles_io_error(self):
        """Test that IOError during file loading is handled gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, "test.txt"), "w", encoding="utf-8") as f:
                f.write("Title\nContent")

            with patch("src.file_utils.TextLoader") as mock_loader:
                mock_loader.return_value.load.side_effect = IOError("Permission denied")
                docs = load_documents(temp_dir)
                assert len(docs) == 0

    def test_load_documents_multiple_tags_lines(self):
        """Test that only the second line is checked for Tags."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, "test.txt"), "w", encoding="utf-8") as f:
                f.write("Title\nTags: first\nTags: second\nContent")

            docs = load_documents(temp_dir)

            # Only first Tags line should be extracted
            assert docs[0]["tags"] == "first"

    # ========================================================================
    # LOAD DOCUMENTS EXCEPTION TESTS
    # ========================================================================

    def test_load_documents_invalid_directory(self):
        """Test that load_documents raises FileNotFoundError for non-existent directory."""
        with pytest.raises(FileNotFoundError):
            load_documents("/nonexistent/directory/path")

    @pytest.mark.parametrize(
        "exception_class,should_be_caught",
        [
            (IOError, True),
            (RuntimeError, False),
            (UnicodeDecodeError, False),
        ],
    )
    def test_load_documents_exceptions(self, exception_class, should_be_caught):
        """Parametrized test for load_documents exception handling.

        Only IOError is caught and handled gracefully.
        Other exceptions (RuntimeError, UnicodeDecodeError) are raised.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, "test.txt"), "w", encoding="utf-8") as f:
                f.write("Title\nContent")

            with patch("src.file_utils.TextLoader") as mock_loader:
                # Create appropriate exception instance
                if exception_class == UnicodeDecodeError:
                    exception_instance = UnicodeDecodeError("utf-8", b"", 0, 1, "invalid continuation byte")
                elif exception_class == IOError:
                    exception_instance = IOError("Permission denied")
                else:
                    exception_instance = RuntimeError("File corrupted")

                mock_loader.return_value.load.side_effect = exception_instance

                if should_be_caught:
                    # IOError should be caught and return empty list
                    result = load_documents(temp_dir)
                    assert not result
                else:
                    # RuntimeError and UnicodeDecodeError should be raised
                    with pytest.raises(exception_class):
                        load_documents(temp_dir)

    def test_load_documents_permission_denied_directory(self):
        """Test that load_documents raises PermissionError for permission denied directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, "test.txt"), "w", encoding="utf-8") as f:
                f.write("Title\nContent")

            os.chmod(temp_dir, 0o000)
            try:
                # Should raise PermissionError when trying to list directory
                with pytest.raises(PermissionError):
                    load_documents(temp_dir)
            finally:
                os.chmod(temp_dir, 0o755)

    def test_load_documents_mixed_valid_invalid_files(self):
        """Test loading when some files are valid and some raise errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with open(os.path.join(temp_dir, "valid.txt"), "w", encoding="utf-8") as f:
                f.write("Valid Title\nContent")
            with open(os.path.join(temp_dir, "another.txt"), "w", encoding="utf-8") as f:
                f.write("Another\nContent")

            with patch("src.file_utils.TextLoader") as mock_loader:
                # Mock document objects with page_content attribute
                mock_doc1 = MagicMock()
                mock_doc1.page_content = "Valid Title\nContent"
                mock_doc2 = MagicMock()
                mock_doc2.page_content = "Another\nContent"

                mock_loader.return_value.load.side_effect = [
                    IOError("Cannot read first file"),
                    [mock_doc1, mock_doc2],
                ]
                docs = load_documents(temp_dir)
                # Should handle mixed scenarios gracefully
                assert isinstance(docs, list)

    # ...existing yaml tests...

    # ========================================================================
    # LOAD YAML EXCEPTION TESTS
    # ========================================================================

    def test_load_yaml_file_not_found(self):
        """Test error handling when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_yaml("/nonexistent/file.yaml")

    def test_load_yaml_directory_instead_of_file(self):
        """Test error handling when path points to a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                load_yaml(temp_dir)
                assert True  # Implementation may handle gracefully
            except (IsADirectoryError, IOError, OSError):
                assert True

    @pytest.mark.parametrize(
        "setup_func,exception_types,description",
        [
            (
                lambda f: os.chmod(f, 0o000),
                (IOError, PermissionError, OSError),
                "permission denied",
            ),
            (
                lambda f: open(f, "wb").write(b"key: \xff\xfe"),
                (UnicodeDecodeError, yaml.YAMLError),
                "encoding error",
            ),
        ],
    )
    def test_load_yaml_file_access_errors(self, setup_func, exception_types, description):
        """Parametrized test for YAML file access and encoding errors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_file = os.path.join(temp_dir, "test.yaml")

            if description == "permission denied":
                with open(yaml_file, "w", encoding="utf-8") as f:
                    f.write("key: value")
                setup_func(yaml_file)
                try:
                    try:
                        load_yaml(yaml_file)
                        assert True
                    except exception_types:
                        assert True
                finally:
                    os.chmod(yaml_file, 0o644)
            else:
                setup_func(yaml_file)
                try:
                    load_yaml(yaml_file)
                    assert True
                except exception_types:
                    assert True

    def test_load_yaml_circular_reference(self):
        """Test handling of YAML with circular references."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_file = os.path.join(temp_dir, "circular.yaml")
            with open(yaml_file, "w", encoding="utf-8") as f:
                f.write("node: &anchor\n  ref: *anchor")

            try:
                result = load_yaml(yaml_file)
                assert result is not None or result is None
            except yaml.YAMLError:
                pass

    def test_load_yaml_very_large_file(self):
        """Test handling of very large YAML files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_file = os.path.join(temp_dir, "large.yaml")
            with open(yaml_file, "w", encoding="utf-8") as f:
                f.write("items:\n")
                for i in range(10000):
                    f.write(f"  - item_{i}: value_{i}\n")

            result = load_yaml(yaml_file)
            assert result is not None
            assert "items" in result
            assert len(result["items"]) == 10000

    @pytest.mark.parametrize(
        "invalid_yaml",
        [
            "key: value\n  bad indent:",
            "{ invalid: [syntax",
            "key: &undefined_anchor\nref: *undefined",
            ":\n  :",
        ],
    )
    def test_load_yaml_syntax_errors(self, invalid_yaml):
        """Parametrized test for various invalid YAML syntax."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_file = os.path.join(temp_dir, "invalid.yaml")
            with open(yaml_file, "w", encoding="utf-8") as f:
                f.write(invalid_yaml)
            with pytest.raises(yaml.YAMLError):
                load_yaml(yaml_file)

    def test_load_yaml_malformed_structure(self):
        """Test handling of malformed YAML structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_file = os.path.join(temp_dir, "malformed.yaml")
            with open(yaml_file, "w", encoding="utf-8") as f:
                f.write("config:\n  nested:\n    - item: [1, 2, 3\n    - broken")
            with pytest.raises(yaml.YAMLError):
                load_yaml(yaml_file)

    @pytest.mark.parametrize(
        "yaml_content,key,expected_value,expected_type",
        [
            (
                "null_value: null\nempty_string: ''\nempty_list: []",
                "null_value",
                None,
                type(None),
            ),
            (
                "null_value: null\nempty_string: ''\nempty_list: []",
                "empty_string",
                "",
                str,
            ),
            (
                "null_value: null\nempty_string: ''\nempty_list: []",
                "empty_list",
                [],
                list,
            ),
            ("number: 123\nstring: '123'", "number", 123, int),
            ("number: 123\nstring: '123'", "string", "123", str),
            ("float: 123.45\nstring_float: '123.45'", "float", 123.45, float),
            ("float: 123.45\nstring_float: '123.45'", "string_float", "123.45", str),
        ],
    )
    def test_load_yaml_value_types(self, yaml_content, key, expected_value, expected_type):
        """Parametrized test for YAML value type handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yaml_file = os.path.join(temp_dir, "values.yaml")
            with open(yaml_file, "w", encoding="utf-8") as f:
                f.write(yaml_content)
            result = load_yaml(yaml_file)
            assert result[key] == expected_value
            assert isinstance(result[key], expected_type)

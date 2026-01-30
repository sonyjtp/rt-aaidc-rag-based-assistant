"""
Fixtures for file utility tests.
"""

# pylint: disable=import-error

import os
import tempfile

import pytest
import yaml


@pytest.fixture
def temp_yaml_file():
    """Fixture providing a temporary YAML file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield os.path.join(temp_dir, "config.yaml"), temp_dir


@pytest.fixture(
    params=[
        pytest.param(
            {"key": "value", "number": 42, "list": [1, 2, 3]},
            id="simple_dict",
        ),
        pytest.param(
            {
                "database": {"host": "localhost", "port": 5432},
                "features": ["feature1", "feature2"],
            },
            id="nested_structure",
        ),
        pytest.param(
            {
                "string": "text",
                "integer": 42,
                "float": 3.14,
                "boolean": True,
                "null": None,
                "list": [1, 2, 3],
            },
            id="all_types",
        ),
    ]
)
def yaml_content_config(request):
    """Parameterized fixture providing different YAML content configurations."""
    return request.param


@pytest.fixture
def yaml_file_with_content(temp_yaml_file, yaml_content_config):
    """
    Composite fixture that creates a YAML file with parameterized content.

    Yields the file path and the expected content.
    """
    yaml_file, temp_dir = temp_yaml_file
    config = yaml_content_config

    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.dump(config, f)

    return yaml_file, config


@pytest.fixture(
    params=[
        pytest.param(
            {
                "title": "Test Title",
                "tags": "test, document",
                "content": "Content here",
            },
            id="with_title_and_tags",
        ),
        pytest.param(
            {"title": "Document", "tags": "", "content": "Content without tags"},
            id="no_tags",
        ),
        pytest.param(
            {"title": "empty.txt", "tags": "", "content": ""},
            id="empty_file",
        ),
    ]
)
def document_config_fixture(request):
    """Parameterized fixture providing different document configurations."""
    return request.param

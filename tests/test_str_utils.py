import pytest

from src.str_utils import format_tags


class TestFormatTags:
    """Test cases for format_tags function."""

    @pytest.mark.parametrize(
        "input_value, expected",
        [
            ("tag1, tag2", "tag1, tag2"),
            (["tag1", "tag2", "tag3"], "tag1,tag2,tag3"),
            (["tag1", "", "tag2", None, "  "], "tag1,tag2"),
            (["tag1", 123, "tag2"], "tag1,123,tag2"),
            ([], ""),
            (None, ""),
            ({"key": "value"}, ""),
        ],
    )
    def test_format_tags(self, input_value, expected):
        """Test format_tags with various inputs."""
        assert format_tags(input_value) == expected

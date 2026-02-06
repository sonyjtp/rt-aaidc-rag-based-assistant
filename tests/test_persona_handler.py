"""
Unit tests for persona_handler.py
Tests the PersonaHandler class for meta question detection and handling.
"""

from unittest.mock import MagicMock, patch

import pytest

from error_messages import NO_RESULTS_ERROR_MESSAGE
from src.persona_handler import MetaPattern, PersonaHandler


@pytest.fixture
def mock_readme_extractor():
    """Fixture providing a mocked ReadmeExtractor instance."""
    mock_extractor = MagicMock()
    mock_extractor.get_tools_and_models.return_value = "## Tools and Models\nThis is tools content"
    mock_extractor.get_overview.return_value = "## Overview\nThis is overview content"
    mock_extractor.get_architecture.return_value = "## Architecture\nThis is architecture content"
    mock_extractor.get_customization.return_value = "## Customization\nThis is customization content"
    mock_extractor.get_quick_start.return_value = "## Quick Start\nThis is quick start content"
    mock_extractor.get_features.return_value = "## Features\nThis is features content"
    return mock_extractor


@pytest.fixture
def handler_mocks(mock_readme_extractor):
    """Fixture providing mocked PersonaHandler dependencies."""
    with (
        patch("src.persona_handler.Path") as mock_path,
        patch("src.persona_handler.load_yaml") as mock_load_yaml,
        patch("src.persona_handler.ReadmeExtractor", return_value=mock_readme_extractor) as mock_readme_class,
        patch("src.persona_handler.logger") as mock_logger,
    ):
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = True
        mock_path_instance.read_text.return_value = ""
        mock_path.return_value = mock_path_instance

        yield {
            "path": mock_path,
            "path_instance": mock_path_instance,
            "load_yaml": mock_load_yaml,
            "readme_class": mock_readme_class,
            "readme_extractor": mock_readme_extractor,
            "logger": mock_logger,
        }


# pylint: disable=redefined-outer-name
class TestPersonaHandler:
    """Test the MetaPattern class and PersonaHandler functionality."""

    @pytest.mark.parametrize(
        "pattern,kind,response,response_type,query,should_match",
        [
            pytest.param(
                r"what is your.*purpose",
                "describe",
                "I am an AI assistant",
                "",
                "What is your purpose?",
                True,
                id="regex_pattern_match",
            ),
            pytest.param(
                "who are you",
                "describe",
                "I am an assistant",
                "",
                "Who are you today?",
                True,
                id="literal_pattern_case_insensitive",
            ),
            pytest.param(
                "invalid[regex",
                "refuse",
                "",
                "",
                "test query",
                False,
                id="invalid_regex_fallback_to_literal",
            ),
            pytest.param(
                "capabilities",
                "sensitive",
                "",
                "",
                "What are your capabilities?",
                True,
                id="capabilities_keyword_match",
            ),
        ],
    )
    def test_meta_pattern_initialization_and_matching(
        self, pattern, kind, response, response_type, query, should_match
    ):
        """Test MetaPattern initialization and regex matching behavior."""
        meta_pattern = MetaPattern(pattern, kind, response, response_type)

        assert meta_pattern.kind == kind
        assert meta_pattern.response == response
        assert meta_pattern.response_type == response_type
        assert meta_pattern.rx is not None

        match_result = meta_pattern.rx.search(query)
        if should_match:
            assert match_result is not None
        else:
            assert match_result is None

    @pytest.mark.parametrize(
        "config_exists,expected_patterns_count",
        [
            pytest.param(
                True,
                2,
                id="successful_init_with_readme",
            ),
            pytest.param(
                False,
                0,
                id="init_without_config_file",
            ),
        ],
    )
    def test_persona_handler_initialization_positive_scenarios(
        self,
        handler_mocks,
        config_exists,
        expected_patterns_count,
    ):
        """Test PersonaHandler initialization with valid configurations."""
        handler_mocks["path_instance"].exists.return_value = config_exists

        if config_exists:
            handler_mocks["load_yaml"].return_value = {
                "meta_questions": [
                    {
                        "pattern": "who are you",
                        "kind": "describe",
                        "response": "I am a helpful assistant",
                        "response_type": "",
                    },
                    {
                        "pattern": "tools",
                        "kind": "readme_extract",
                        "response": "",
                        "response_type": "tools_and_models",
                    },
                ],
                "allow_self_description": True,
            }

        handler = PersonaHandler()

        assert handler.allow_self_description is True
        assert handler.default_meta_refusal == NO_RESULTS_ERROR_MESSAGE
        if config_exists:
            assert len(handler.patterns) == expected_patterns_count

    @pytest.mark.parametrize(
        "readme_exception,yaml_exception,expected_readme_extractor_none",
        [
            pytest.param(
                Exception("ReadmeExtractor init failed"),
                None,
                True,
                id="readme_extractor_initialization_failure",
            ),
            pytest.param(
                None,
                Exception("Invalid YAML"),
                False,
                id="yaml_parsing_failure",
            ),
            pytest.param(
                Exception("ReadmeExtractor failed"),
                Exception("Invalid YAML"),
                True,
                id="both_readme_and_yaml_failure",
            ),
        ],
    )
    def test_persona_handler_initialization_negative_scenarios(
        self,
        handler_mocks,
        readme_exception,
        yaml_exception,
        expected_readme_extractor_none,
    ):
        """Test PersonaHandler initialization with failures and graceful handling."""
        handler_mocks["path_instance"].exists.return_value = True

        if readme_exception:
            handler_mocks["readme_class"].side_effect = readme_exception

        if yaml_exception:
            handler_mocks["load_yaml"].side_effect = yaml_exception
        else:
            handler_mocks["load_yaml"].return_value = {"meta_questions": []}

        handler = PersonaHandler()

        # Should handle errors gracefully
        assert handler is not None
        if expected_readme_extractor_none:
            assert handler.readme_extractor is None
        assert handler.allow_self_description is True

    @pytest.mark.parametrize(
        "config_data,query,expected",
        [
            pytest.param(
                {
                    "meta_questions": [
                        {
                            "pattern": "who are you",
                            "kind": "describe",
                            "response": "I am a helpful assistant",
                            "response_type": "",
                        }
                    ],
                    "allow_self_description": True,
                },
                "Who are you?",
                ("describe", "I am a helpful assistant", ""),
                id="describe_meta_question_match",
            ),
            pytest.param(
                {
                    "meta_questions": [
                        {
                            "pattern": "tools.*models",
                            "kind": "readme_extract",
                            "response": "",
                            "response_type": "tools_and_models",
                        }
                    ],
                    "allow_self_description": True,
                },
                "What tools and models do you use?",
                ("readme_extract", NO_RESULTS_ERROR_MESSAGE, "tools_and_models"),
                id="readme_extract_meta_question_match",
            ),
            pytest.param(
                {
                    "meta_questions": [
                        {
                            "pattern": "your capabilities",
                            "kind": "sensitive",
                            "response": "",
                            "response_type": "",
                        }
                    ],
                    "allow_self_description": True,
                },
                "What are your capabilities?",
                ("sensitive", NO_RESULTS_ERROR_MESSAGE, ""),
                id="sensitive_meta_question_match",
            ),
            pytest.param(
                {"meta_questions": [], "allow_self_description": True},
                "Tell me about the history of AI",
                None,
                id="domain_non_meta",
            ),
            pytest.param(
                {"meta_questions": [], "allow_self_description": True},
                "",
                None,
                id="empty_query",
            ),
        ],
    )
    def test_is_meta_question_consolidated(self, handler_mocks, config_data, query, expected):
        """Consolidated test for meta-question detection (is_meta_question).

        This replaces several earlier positive/negative tests: focuses on
        whether the handler recognizes meta-questions and returns the expected
        (kind, response, response_type) tuple or None for non-meta queries.
        """
        handler_mocks["load_yaml"].return_value = config_data
        handler_mocks["path_instance"].exists.return_value = True

        handler = PersonaHandler()
        result = handler.is_meta_question(query)

        if expected is None:
            assert result is None
        else:
            assert result is not None
            kind, response, response_type = result
            assert kind == expected[0]
            assert response == expected[1]
            assert response_type == expected[2]

    @pytest.mark.parametrize(
        "config_data,query,expected_contains",
        [
            pytest.param(
                {
                    "meta_questions": [
                        {
                            "pattern": "test",
                            "kind": "describe",
                            "response": "I am a helpful assistant",
                            "response_type": "",
                        }
                    ],
                    "allow_self_description": True,
                },
                "test query",
                "I am a helpful",
                id="describe_allowed_returns_description",
            ),
            pytest.param(
                {
                    "meta_questions": [
                        {
                            "pattern": "test",
                            "kind": "describe",
                            "response": "I am a helpful assistant",
                            "response_type": "",
                        }
                    ],
                    "allow_self_description": False,
                },
                "test query",
                NO_RESULTS_ERROR_MESSAGE,
                id="describe_disallowed_returns_refusal",
            ),
            pytest.param(
                {
                    "meta_questions": [
                        {"pattern": "test", "kind": "sensitive", "response": "secret", "response_type": ""}
                    ],
                    "allow_self_description": True,
                },
                "test query",
                NO_RESULTS_ERROR_MESSAGE,
                id="sensitive_always_refused",
            ),
            pytest.param(
                {
                    "meta_questions": [
                        {
                            "pattern": "test",
                            "kind": "readme_extract",
                            "response": "",
                            "response_type": "tools_and_models",
                        }
                    ],
                    "allow_self_description": True,
                },
                "test query",
                "Tools and Models",
                id="readme_extract_returns_readme_content",
            ),
            pytest.param(
                {"meta_questions": [], "allow_self_description": True},
                "not a meta query",
                None,
                id="non_meta_returns_none",
            ),
            pytest.param(
                {
                    "meta_questions": [
                        {
                            "pattern": "test",
                            "kind": "unknown_kind",
                            "response": "I am a helpful assistant",
                            "response_type": "",
                        }
                    ],
                    "allow_self_description": True,
                },
                "test query",
                NO_RESULTS_ERROR_MESSAGE,
                id="unknown_kind_refused",
            ),
        ],
    )
    def test_handle_meta_question_consolidated(self, handler_mocks, config_data, query, expected_contains):
        """Consolidated test for handling meta-questions (handle_meta_question).

        Replaces multiple positive/negative/non-meta tests with a single
        parametrized matrix describing expected behavior for different kinds.
        """
        handler_mocks["load_yaml"].return_value = config_data
        handler_mocks["path_instance"].exists.return_value = True

        handler = PersonaHandler()
        result = handler.handle_meta_question(query)

        if expected_contains is None:
            assert result is None
        else:
            assert expected_contains in result

    @pytest.mark.parametrize(
        "response_type,extractor_method,expected_content_partial",
        [
            pytest.param(
                "tools_and_models",
                "get_tools_and_models",
                "Tools and Models",
                id="extract_tools_and_models",
            ),
            pytest.param(
                "overview",
                "get_overview",
                "Overview",
                id="extract_overview",
            ),
            pytest.param(
                "architecture",
                "get_architecture",
                "Architecture",
                id="extract_architecture",
            ),
            pytest.param(
                "customization",
                "get_customization",
                "Customization",
                id="extract_customization",
            ),
            pytest.param(
                "quick_start",
                "get_quick_start",
                "Quick Start",
                id="extract_quick_start",
            ),
            pytest.param(
                "features",
                "get_features",
                "Features",
                id="extract_features",
            ),
        ],
    )
    def test_get_readme_content_positive_scenarios(
        self,
        handler_mocks,
        response_type,
        extractor_method,
        expected_content_partial,
    ):
        """Test _get_readme_content() with valid response types."""
        handler_mocks["load_yaml"].return_value = {"meta_questions": []}
        handler_mocks["path_instance"].exists.return_value = True

        handler = PersonaHandler()
        result = handler._get_readme_content(response_type)

        assert result is not None
        assert expected_content_partial in result

    @pytest.mark.parametrize(
        "readme_extractor_available,extraction_fails,response_type",
        [
            pytest.param(
                False,
                False,
                "overview",
                id="readme_extractor_unavailable",
            ),
            pytest.param(
                True,
                True,
                "overview",
                id="readme_extraction_exception",
            ),
            pytest.param(
                True,
                False,
                "unknown_type",
                id="unknown_response_type",
            ),
        ],
    )
    def test_get_readme_content_negative_scenarios(
        self,
        handler_mocks,
        readme_extractor_available,
        extraction_fails,
        response_type,
    ):
        """Test _get_readme_content() with extraction failures and unknown types."""
        handler_mocks["load_yaml"].return_value = {"meta_questions": []}
        handler_mocks["path_instance"].exists.return_value = True

        if not readme_extractor_available:
            handler_mocks["readme_class"].side_effect = Exception("Extractor init failed")
        elif extraction_fails:
            handler_mocks["readme_extractor"].get_overview.side_effect = Exception("Extraction failed")

        handler = PersonaHandler()
        result = handler._get_readme_content(response_type)

        # Should return error message, not crash
        assert result is not None
        if not readme_extractor_available:
            assert "couldn't" in result.lower() or "error" in result.lower()
        elif extraction_fails:
            assert "error" in result.lower() or "encountered" in result.lower()
        elif response_type == "unknown_type":
            # Unknown response_type returns the default refusal message
            assert NO_RESULTS_ERROR_MESSAGE.lower() in result.lower()

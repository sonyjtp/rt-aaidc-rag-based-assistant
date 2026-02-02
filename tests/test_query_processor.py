"""
Unit tests for query_processor.py
Tests the QueryProcessor class for query augmentation, context building, and validation.
"""

from unittest.mock import MagicMock, patch

import pytest

from config import DISTANCE_THRESHOLD
from src.query_processor import QueryProcessor


@pytest.fixture
def mock_memory_manager():
    """Fixture providing a mocked MemoryManager instance."""
    mock_manager = MagicMock()
    mock_manager.get_memory_variables.return_value = {
        "chat_history": "Human: What is machine learning?\nAssistant: Machine learning is..."
    }
    return mock_manager


@pytest.fixture
def mock_llm():
    """Fixture providing a mocked LLM instance."""
    mock = MagicMock()
    mock.invoke.return_value = "YES"
    return mock


@pytest.fixture
def processor_mocks(mock_memory_manager, mock_llm):
    """Fixture providing mocked QueryProcessor dependencies."""
    with (
        patch("src.query_processor.file_utils_mod.load_yaml") as mock_load_yaml,
        patch("src.query_processor.logger") as mock_logger,
    ):
        mock_load_yaml.return_value = {
            "follow_up_keywords": ["tell me more", "elaborate", "continue", "explain"],
            "new_topic_keywords": ["what", "how", "why", "who", "where", "when"],
        }

        yield {
            "load_yaml": mock_load_yaml,
            "logger": mock_logger,
            "memory_manager": mock_memory_manager,
            "llm": mock_llm,
        }


# pylint: disable=redefined-outer-name
class TestQueryProcessorInitialization:
    """Test QueryProcessor initialization."""

    @pytest.mark.parametrize(
        "memory_manager,llm,config_available",
        [
            pytest.param(
                MagicMock(),
                MagicMock(),
                True,
                id="successful_init_with_all_components",
            ),
            pytest.param(
                None,
                MagicMock(),
                True,
                id="init_without_memory_manager",
            ),
            pytest.param(
                MagicMock(),
                None,
                True,
                id="init_without_llm",
            ),
            pytest.param(
                None,
                None,
                True,
                id="init_with_only_defaults",
            ),
        ],
    )
    def test_query_processor_initialization_positive_scenarios(
        self, processor_mocks, memory_manager, llm, config_available
    ):
        """Test QueryProcessor initialization with various component combinations."""
        processor_mocks["load_yaml"].return_value = {
            "follow_up_keywords": ["tell me more"],
            "new_topic_keywords": ["what"],
        }

        processor = QueryProcessor(memory_manager=memory_manager, llm=llm)

        assert processor.memory_manager == memory_manager
        assert processor.llm == llm
        assert isinstance(processor.follow_up_keywords, list)
        assert isinstance(processor.new_topic_keywords, list)

    @pytest.mark.parametrize(
        "config_file_exists,yaml_exception",
        [
            pytest.param(
                False,
                None,
                id="config_file_not_found",
            ),
            pytest.param(
                True,
                Exception("Invalid YAML"),
                id="yaml_parsing_exception",
            ),
        ],
    )
    def test_query_processor_initialization_negative_scenarios(
        self, processor_mocks, config_file_exists, yaml_exception
    ):
        """Test QueryProcessor initialization with missing or invalid configs."""
        if not config_file_exists:
            processor_mocks["load_yaml"].side_effect = FileNotFoundError("Config not found")
        elif yaml_exception:
            processor_mocks["load_yaml"].side_effect = yaml_exception

        processor = QueryProcessor()

        # Should initialize gracefully with empty keyword lists
        assert processor.follow_up_keywords == []
        assert processor.new_topic_keywords == []


class TestAugmentQueryWithContext:
    """Test the augment_query_with_context() method."""

    @pytest.mark.parametrize(
        "chat_history,query,contains_previous,contains_current",
        [
            pytest.param(
                "Human: What is AI?\nAssistant: AI is...\nHuman: Tell me more about machine learning.",
                "Tell me more about that",
                True,
                True,
                id="follow_up_question_with_context",
            ),
            pytest.param(
                "Human: What is AI?\nAssistant: AI is...",
                "Tell me more about supervised learning",
                True,
                True,
                id="follow_up_with_tell_me_more_keyword",
            ),
            pytest.param(
                "Human: History of AI.\nAssistant: Long history...",
                "Elaborate on that",
                True,
                True,
                id="follow_up_with_elaborate_keyword",
            ),
            pytest.param(
                "Human: What is AI?\nAssistant: AI is artificial intelligence.",
                "Continue from there",
                True,
                True,
                id="follow_up_with_continue_keyword",
            ),
        ],
    )
    def test_augment_query_with_context_positive_scenarios(
        self, processor_mocks, chat_history, query, contains_previous, contains_current
    ):
        """Test augment_query_with_context() with follow-up questions and context."""
        processor_mocks["memory_manager"].get_memory_variables.return_value = {"chat_history": chat_history}

        processor = QueryProcessor(
            memory_manager=processor_mocks["memory_manager"],
            llm=processor_mocks["llm"],
        )
        result = processor.augment_query_with_context(query)

        assert result is not None
        if contains_previous:
            assert "Previous question:" in result or chat_history in result
        if contains_current:
            assert "Current question:" in result or query in result

    @pytest.mark.parametrize(
        "chat_history,query,should_augment",
        [
            pytest.param(
                "Human: What is AI?\nAssistant: AI is...",
                "What is machine learning?",
                False,
                id="new_topic_not_augmented",
            ),
            pytest.param(
                "Human: History of AI.\nAssistant: Long history...",
                "How does deep learning work?",
                False,
                id="how_question_new_topic",
            ),
            pytest.param(
                "Human: Explain AI.\nAssistant: AI is...",
                "Why is AI important?",
                False,
                id="why_question_new_topic",
            ),
            pytest.param(
                "Human: AI basics.\nAssistant: Basics are...",
                "Who invented machine learning?",
                False,
                id="who_question_new_topic",
            ),
        ],
    )
    def test_augment_query_with_context_new_topic_scenarios(self, processor_mocks, chat_history, query, should_augment):
        """Test augment_query_with_context() with new topic questions (not augmented)."""
        processor_mocks["memory_manager"].get_memory_variables.return_value = {"chat_history": chat_history}

        processor = QueryProcessor(
            memory_manager=processor_mocks["memory_manager"],
            llm=processor_mocks["llm"],
        )
        result = processor.augment_query_with_context(query)

        if not should_augment:
            assert result == query

    @pytest.mark.parametrize(
        "memory_manager,chat_history,query",
        [
            pytest.param(
                None,
                None,
                "Tell me more",
                id="no_memory_manager_returns_original",
            ),
            pytest.param(
                MagicMock(),
                "No previous conversation context.",
                "Tell me more",
                id="no_previous_context_returns_original",
            ),
            pytest.param(
                MagicMock(),
                "",
                "Tell me more",
                id="empty_chat_history_returns_original",
            ),
        ],
    )
    def test_augment_query_with_context_negative_scenarios(self, processor_mocks, memory_manager, chat_history, query):
        """Test augment_query_with_context() with no context or unavailable memory."""
        if memory_manager and chat_history is not None:
            memory_manager.get_memory_variables.return_value = {"chat_history": chat_history}

        processor = QueryProcessor(memory_manager=memory_manager, llm=processor_mocks["llm"])
        result = processor.augment_query_with_context(query)

        assert result == query

    @pytest.mark.parametrize(
        "exception_type",
        [
            pytest.param(
                Exception("Memory access failed"),
                id="memory_access_exception",
            ),
            pytest.param(
                RuntimeError("Processing error"),
                id="runtime_exception",
            ),
        ],
    )
    def test_augment_query_with_context_exception_handling(self, processor_mocks, exception_type):
        """Test augment_query_with_context() gracefully handles exceptions."""
        processor_mocks["memory_manager"].get_memory_variables.side_effect = exception_type

        processor = QueryProcessor(
            memory_manager=processor_mocks["memory_manager"],
            llm=processor_mocks["llm"],
        )
        result = processor.augment_query_with_context("test query")

        # Should return original query on exception
        assert result == "test query"


class TestBuildContext:
    """Test the build_context() static method."""

    @pytest.mark.parametrize(
        "flat_docs,flat_distances,expected_has_content",
        [
            pytest.param(
                ["Document 1", "Document 2"],
                [DISTANCE_THRESHOLD - 0.1, DISTANCE_THRESHOLD - 0.05],
                True,
                id="documents_below_threshold",
            ),
            pytest.param(
                ["Single document"],
                [DISTANCE_THRESHOLD - 0.5],
                True,
                id="single_document_below_threshold",
            ),
            pytest.param(
                ["Document 1", "Document 2", "Document 3"],
                [],
                True,
                id="empty_distances_list",
            ),
            pytest.param(
                ["Doc 1", "Doc 2"],
                [DISTANCE_THRESHOLD],
                True,
                id="distance_exactly_at_threshold",
            ),
        ],
    )
    def test_build_context_positive_scenarios(self, flat_docs, flat_distances, expected_has_content):
        """Test build_context() with valid documents below threshold."""
        result = QueryProcessor.build_context(flat_docs, flat_distances)

        if expected_has_content:
            assert len(result) > 0
            assert flat_docs[0] in result

    @pytest.mark.parametrize(
        "flat_docs,flat_distances",
        [
            pytest.param(
                ["Document 1", "Document 2"],
                [DISTANCE_THRESHOLD + 0.5, DISTANCE_THRESHOLD + 1.0],
                id="all_documents_exceed_threshold",
            ),
            pytest.param(
                [],
                [],
                id="empty_documents_list",
            ),
            pytest.param(
                [],
                [DISTANCE_THRESHOLD - 0.1],
                id="empty_documents_non_empty_distances",
            ),
            pytest.param(
                ["Document"],
                [DISTANCE_THRESHOLD + 0.1],
                id="first_document_exceeds_threshold",
            ),
        ],
    )
    def test_build_context_negative_scenarios(self, flat_docs, flat_distances):
        """Test build_context() returns empty string when context should not be used."""
        result = QueryProcessor.build_context(flat_docs, flat_distances)

        assert result == ""


class TestValidateContext:
    """Test the validate_context() method."""

    @pytest.mark.parametrize(
        "context,is_relevant,llm_response",
        [
            pytest.param(
                "This document explains machine learning",
                True,
                "YES",
                id="relevant_context_yes_response",
            ),
            pytest.param(
                "Information about neural networks and deep learning",
                True,
                "Yes, this is relevant",
                id="relevant_context_yes_in_response",
            ),
        ],
    )
    def test_validate_context_positive_scenarios(self, processor_mocks, context, is_relevant, llm_response):
        """Test validate_context() with relevant contexts."""
        processor_mocks["llm"].invoke.return_value = llm_response

        processor = QueryProcessor(
            memory_manager=processor_mocks["memory_manager"],
            llm=processor_mocks["llm"],
        )
        result = processor.validate_context("What is machine learning?", context)

        assert result == is_relevant

    @pytest.mark.parametrize(
        "context,query,search_query,llm_response,expected",
        [
            pytest.param(
                "",
                "What is AI?",
                None,
                "YES",
                False,
                id="empty_context_returns_false",
            ),
            pytest.param(
                "   ",
                "What is AI?",
                None,
                "YES",
                False,
                id="whitespace_only_context_returns_false",
            ),
            pytest.param(
                "Some context",
                "What is AI?",
                None,
                "NO",
                False,
                id="llm_says_not_relevant",
            ),
            pytest.param(
                "Some context",
                "What is AI?",
                "Previous: history\nCurrent: What is AI?",
                "NO",
                False,
                id="follow_up_question_not_relevant",
            ),
        ],
    )
    def test_validate_context_negative_scenarios(
        self, processor_mocks, context, query, search_query, llm_response, expected
    ):
        """Test validate_context() with irrelevant or empty contexts."""
        processor_mocks["llm"].invoke.return_value = llm_response

        processor = QueryProcessor(
            memory_manager=processor_mocks["memory_manager"],
            llm=processor_mocks["llm"],
        )
        result = processor.validate_context(query, context, search_query)

        assert result == expected

    @pytest.mark.parametrize(
        "has_llm,llm_exception",
        [
            pytest.param(
                False,
                None,
                id="no_llm_available_assumes_relevant",
            ),
            pytest.param(
                True,
                Exception("LLM invocation failed"),
                id="llm_exception_assumes_relevant",
            ),
        ],
    )
    def test_validate_context_exception_handling(self, processor_mocks, has_llm, llm_exception):
        """Test validate_context() gracefully handles missing LLM or exceptions."""
        llm = processor_mocks["llm"] if has_llm else None

        if llm_exception:
            processor_mocks["llm"].invoke.side_effect = llm_exception

        processor = QueryProcessor(
            memory_manager=processor_mocks["memory_manager"],
            llm=llm,
        )
        result = processor.validate_context("What is AI?", "Some context")

        # On error or missing LLM, assume relevant (conservative)
        assert result is True


class TestIsContextRelevantToQuery:
    """Test the _is_context_relevant_to_query() private method."""

    @pytest.mark.parametrize(
        "query,context,search_query,llm_response,expected",
        [
            pytest.param(
                "What is machine learning?",
                "Machine learning is a subset of AI that enables systems to learn",
                None,
                "YES",
                True,
                id="direct_question_relevant_context",
            ),
            pytest.param(
                "Explain neural networks",
                "Neural networks are computational models inspired by biological neurons",
                None,
                "YES, the context explains neural networks",
                True,
                id="context_contains_yes_multiple_words",
            ),
            pytest.param(
                "What is your purpose?",
                "This document discusses AI history",
                None,
                "NO",
                False,
                id="context_not_relevant_to_question",
            ),
            pytest.param(
                "Original question",
                "Context about deep learning",
                "Previous: What is AI?\nCurrent: Original question",
                "YES",
                True,
                id="follow_up_question_with_search_query",
            ),
        ],
    )
    def test_is_context_relevant_to_query_positive_scenarios(
        self, processor_mocks, query, context, search_query, llm_response, expected
    ):
        """Test _is_context_relevant_to_query() with various context validations."""
        processor_mocks["llm"].invoke.return_value = llm_response

        processor = QueryProcessor(
            memory_manager=processor_mocks["memory_manager"],
            llm=processor_mocks["llm"],
        )
        result = processor._is_context_relevant_to_query(query, context, search_query)

        assert result == expected

    @pytest.mark.parametrize(
        "query,context,search_query,has_llm,llm_exception",
        [
            pytest.param(
                "What is AI?",
                "",
                None,
                True,
                None,
                id="empty_context_returns_false",
            ),
            pytest.param(
                "What is AI?",
                "   ",
                None,
                True,
                None,
                id="whitespace_context_returns_false",
            ),
            pytest.param(
                "What is AI?",
                "Valid context",
                None,
                False,
                None,
                id="no_llm_returns_true",
            ),
            pytest.param(
                "What is AI?",
                "Valid context",
                None,
                True,
                Exception("LLM error"),
                id="llm_exception_returns_true",
            ),
        ],
    )
    def test_is_context_relevant_to_query_negative_scenarios(
        self, processor_mocks, query, context, search_query, has_llm, llm_exception
    ):
        """Test _is_context_relevant_to_query() with edge cases and failures."""
        llm = processor_mocks["llm"] if has_llm else None

        if llm_exception:
            processor_mocks["llm"].invoke.side_effect = llm_exception

        processor = QueryProcessor(
            memory_manager=processor_mocks["memory_manager"],
            llm=llm,
        )
        result = processor._is_context_relevant_to_query(query, context, search_query)

        if not context or not context.strip():
            assert result is False
        elif not has_llm:
            assert result is True
        elif llm_exception:
            assert result is True

    @pytest.mark.parametrize(
        "llm_response_type",
        [
            pytest.param(
                "YES",
                id="string_response_yes",
            ),
            pytest.param(
                MagicMock(content="YES"),
                id="object_with_content_attribute",
            ),
            pytest.param(
                MagicMock(content="NO"),
                id="object_with_content_attribute_no",
            ),
        ],
    )
    def test_is_context_relevant_to_query_response_handling(self, processor_mocks, llm_response_type):
        """Test _is_context_relevant_to_query() handles different LLM response types."""
        processor_mocks["llm"].invoke.return_value = llm_response_type

        processor = QueryProcessor(
            memory_manager=processor_mocks["memory_manager"],
            llm=processor_mocks["llm"],
        )
        result = processor._is_context_relevant_to_query("What is AI?", "AI is artificial intelligence")

        if isinstance(llm_response_type, str):
            assert result == ("YES" in llm_response_type)
        else:
            assert result == ("YES" in llm_response_type.content)

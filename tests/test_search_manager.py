"""Unit tests for SearchManager class."""

from unittest.mock import MagicMock, patch

import pytest

from src.search_manager import SearchManager


@pytest.fixture
def mock_components(mock_logger):
    with patch("src.search_manager.VectorDB") as mock_vector_db_cls, patch("src.search_manager.logger", mock_logger):
        mock_vector_db_mock = MagicMock(name="VectorDBInstance")
        mock_vector_db_cls.return_value = mock_vector_db_mock
        llm_mock = MagicMock(name="LLM")

        def make_manager(llm=None, vector_db_instance=None):
            if llm is None:
                llm = llm_mock
            if vector_db_instance is None:
                return SearchManager(llm=llm)
            with patch("src.search_manager.VectorDB", return_value=vector_db_instance):
                return SearchManager(llm=llm)

        yield {
            "vector_db": mock_vector_db_mock,
            "make_manager": make_manager,
        }


@pytest.fixture
def make_manager(mock_components):
    return mock_components["make_manager"]


@pytest.fixture
def vector_db_mock(mock_components):
    return mock_components["vector_db"]


class TestSearchManager:  # pylint: disable=redefined-outer-name
    def test_init_success(self, vector_db_mock, make_manager):
        llm = MagicMock(name="LLM")
        search_manager = make_manager(llm=llm, vector_db_instance=vector_db_mock)
        assert search_manager.llm is llm
        assert search_manager.vector_db is vector_db_mock

    @pytest.mark.parametrize(
        "llm_value, vector_db_side_effect, expected_error",
        [
            (
                None,
                None,
                "Unable to initialize the assistant",
            ),  # missing llm - user-facing message
            (
                MagicMock(name="LLM"),
                Exception("db fail"),
                "Unable to initialize the assistant",
            ),
            # VectorDB ctor fails - user-facing message
        ],
    )
    def test_init_failure_modes(self, monkeypatch, llm_value, vector_db_side_effect, expected_error):
        """Test SearchManager initialization failure modes."""

        if vector_db_side_effect is not None:

            def bad_ctor():
                raise vector_db_side_effect

            monkeypatch.setattr("src.search_manager.VectorDB", bad_ctor)
        else:
            # happy constructor when we only test missing llm
            monkeypatch.setattr("src.search_manager.VectorDB", lambda: MagicMock(name="VectorDB"))

        with pytest.raises(RuntimeError) as exc:
            SearchManager(llm_value)
        assert expected_error in str(exc.value)

    @pytest.mark.parametrize(
        "side_effect, expect_raises, remove_after",
        [
            (None, None, False),  # successful delegation
            (RuntimeError("add failed"), RuntimeError, False),
            (None, RuntimeError, True),
        ],
    )
    def test_add_documents(
        self,
        make_manager,
        vector_db_mock,
        mock_logger,
        side_effect,
        expect_raises,
        remove_after,
    ):
        """Test add_documents method with various scenarios."""

        llm = MagicMock(name="LLM")

        if side_effect is None:
            vector_db_mock.add_documents.return_value = "ok"
        else:
            vector_db_mock.add_documents.side_effect = side_effect

        mgr = make_manager(llm=llm, vector_db_instance=vector_db_mock)

        if remove_after:
            mgr.vector_db = None

        docs = [{"id": 1, "text": "doc"}]

        if expect_raises:
            with pytest.raises(expect_raises):
                mgr.add_documents(docs)
            if not remove_after:
                mock_logger.error.assert_called()
        else:
            mgr.add_documents(docs)
            vector_db_mock.add_documents.assert_called_once()

    def test_search(self, make_manager, vector_db_mock, mock_logger):
        """Test search method delegation and error handling."""

        llm = MagicMock(name="LLM")
        vector_db_mock.search.return_value = {"documents": ["d"], "distances": [0.1]}
        mgr = make_manager(llm=llm, vector_db_instance=vector_db_mock)

        res = mgr.search("query", n_results=1, maximum_distance=0.5)
        assert res == {"documents": ["d"], "distances": [0.1]}
        vector_db_mock.search.assert_called_once_with(query="query", n_results=1, maximum_distance=0.5)

        vector_db_mock.search.side_effect = Exception("search broken")
        with pytest.raises(Exception):
            mgr.search("q", 1, 0.5)
        mock_logger.error.assert_called()

    @pytest.mark.parametrize(
        "search_results, expected_docs, expected_dists",
        [
            (
                {"documents": [["a", "b"]], "distances": [[0.1, 0.2]]},
                ["a", "b"],
                [0.1, 0.2],
            ),
            ({"documents": ["x"], "distances": [0.3]}, ["x"], [0.3]),
            ({}, [], []),
        ],
    )
    def test_flatten_search_results(self, search_results, expected_docs, expected_dists):
        """Test flatten_search_results static method."""

        docs, dists = SearchManager.flatten_search_results(search_results)
        assert docs == expected_docs
        assert dists == expected_dists

    def test_log_search_results(self, mock_components, mock_logger):
        """Test log_search_results static method with long document truncation."""

        long_doc = "a" * 100
        docs = [long_doc, "short"]
        dists = [0.1, 0.25]
        # call static method directly (mock_components patches src.search_manager.logger)
        SearchManager.log_search_results(docs, dists)
        assert mock_logger.debug.call_count == 2
        args, _ = mock_logger.debug.call_args_list[0]
        assert "..." in args[0]

    # python
    @pytest.mark.parametrize(
        "context, make_llm_none, invoke_value, invoke_side_effect, expected, expect_warning",
        [
            ("", False, None, None, False, False),
            ("   ", False, None, None, False, False),
            ("some context", True, None, None, True, True),  # missing LLM -> warn True
            (
                "ctx",
                False,
                MagicMock(content="yes"),
                None,
                True,
                False,
            ),  # response_obj.content == "yes"
            (
                "ctx",
                False,
                MagicMock(content="no"),
                None,
                False,
                False,
            ),  # response_obj.content == "no"
            ("ctx", False, "YES", None, True, False),  # string fallback
            (
                "ctx",
                False,
                None,
                RuntimeError("boom"),
                True,
                True,
            ),  # invoke raises -> warn True
        ],
    )
    def test_is_context_relevant_to_query_parametrized(
        self,
        make_manager,
        vector_db_mock,
        mock_logger,
        context,
        make_llm_none,
        invoke_value,
        invoke_side_effect,
        expected,
        expect_warning,
    ):  # pylint: disable=too-many-arguments
        """Test is_context_relevant_to_query with various scenarios."""

        llm = MagicMock(name="LLM")
        mgr = make_manager(llm=llm, vector_db_instance=vector_db_mock)

        # ensure logger state is clean per case
        mock_logger.warning.reset_mock()

        if make_llm_none:
            mgr.llm = None
        else:
            if invoke_side_effect is not None:
                mgr.llm.invoke.side_effect = invoke_side_effect
            elif invoke_value is not None:
                mgr.llm.invoke.return_value = invoke_value

        result = mgr.is_context_relevant_to_query("q", context)
        assert result is expected

        if expect_warning:
            mock_logger.warning.assert_called()
        else:
            mock_logger.warning.assert_not_called()

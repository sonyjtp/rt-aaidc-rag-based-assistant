"""Unit tests for RAGAssistant class in src.rag_assistant."""

from unittest.mock import MagicMock, patch

import pytest

from config import CHAT_HISTORY
from src.rag_assistant import RAGAssistant


@pytest.fixture
def mock_components():
    """Patch SearchManager, MemoryManager, and ReasoningStrategyLoader and provide mock instances."""
    with (
        patch("src.rag_assistant.SearchManager") as mock_search_manager_cls,
        patch("src.rag_assistant.MemoryManager") as mock_memory_cls,
        patch("src.rag_assistant.ReasoningStrategyLoader") as mock_reasoning_cls,
        patch("src.rag_assistant.initialize_llm") as mock_init_llm,
        patch("src.rag_assistant.build_system_prompts") as mock_build_prompts,
        patch("src.rag_assistant.create_prompt_template") as mock_create_prompt,
        patch("src.rag_assistant.logger") as mock_logger,
    ):
        # make initialize_llm return a mock LLM by default
        mock_llm = MagicMock(name="LLMDefault")
        mock_init_llm.return_value = mock_llm
        # SearchManager mock instance
        search_manager_instance = MagicMock(name="SearchManagerInstance")
        default_search = {
            "documents": [["Document 1"]],
            "distances": [[0.1]],
            "metadatas": [[{}]],
            "ids": [["id1"]],
        }
        search_manager_instance.search.return_value = default_search
        search_manager_instance.flatten_search_results.return_value = (
            ["Document 1"],
            [0.1],
        )
        search_manager_instance.log_search_results.return_value = None
        search_manager_instance.add_documents.return_value = None
        search_manager_instance.is_context_relevant_to_query.return_value = True
        search_manager_instance.llm = mock_llm
        mock_search_manager_cls.return_value = search_manager_instance

        # MemoryManager mock instance
        memory_instance = MagicMock(name="MemoryManagerInstance")
        memory_instance.memory = True
        memory_instance.strategy = "summarization_sliding_window"
        memory_instance.get_memory_variables.return_value = {"chat_history": ""}
        memory_instance.add_message.return_value = None
        mock_memory_cls.return_value = memory_instance

        # ReasoningStrategyLoader mock instance
        reasoning_instance = MagicMock(name="ReasoningStrategyLoaderInstance")
        reasoning_instance.get_strategy_name.return_value = "RAG-Enhanced Reasoning"
        mock_reasoning_cls.return_value = reasoning_instance

        # default prompt builders
        mock_build_prompts.return_value = ["system prompt"]
        prompt_template_mock = MagicMock(name="PromptTemplate")
        mock_create_prompt.return_value = prompt_template_mock
        # chain operator composition (prompt | llm | parser)
        prompt_template_mock.__or__.return_value = prompt_template_mock
        prompt_template_mock.__ror__.return_value = prompt_template_mock

        yield {
            "search_manager_cls": mock_search_manager_cls,
            "search_manager_instance": search_manager_instance,
            "memory_cls": mock_memory_cls,
            "memory_instance": memory_instance,
            "reasoning_cls": mock_reasoning_cls,
            "reasoning_instance": reasoning_instance,
            "init_llm_mock": mock_init_llm,
            "mock_llm": mock_llm,
            "build_prompts_mock": mock_build_prompts,
            "create_prompt_mock": mock_create_prompt,
            "prompt_template_mock": prompt_template_mock,
            "mock_logger": mock_logger,
        }


@pytest.fixture(params=["no_llm", "prompts_error"])
def build_chain_failure_scenario(request, mock_components):
    """Return the current scenario value via request.param."""
    return request.param


class TestRAGAssistant:  # pylint: disable=redefined-outer-name
    """Grouped tests for RAGAssistant for readability."""

    @pytest.fixture(autouse=True)
    def reset_mocks(self):
        """Scoped auto-use fixture to mirror test organization conventions."""
        yield

    def test_initializations(self, mock_components):
        """RAGAssistant should initialize LLM via initialize_llm."""
        assistant = RAGAssistant()
        assert assistant.llm is mock_components["mock_llm"]
        assert assistant.search_manager is mock_components["search_manager_instance"]
        assert assistant.memory_manager is mock_components["memory_instance"]
        assert assistant.reasoning_strategy is mock_components["reasoning_instance"]
        assert assistant.prompt_template is mock_components["prompt_template_mock"]
        assert assistant.chain is mock_components["prompt_template_mock"]

    def test_behavior_when_no_llm(self, mock_components):
        """When initialize_llm returns None, assistant must operate in degraded mode."""
        # force initialize_llm to return None for this instantiation
        with patch(
            "src.rag_assistant.initialize_llm", side_effect=Exception("LLM init failed")
        ):
            assistant = RAGAssistant()
            assert assistant.llm is None
            assert assistant.search_manager is None
            assert assistant.prompt_template is None
            assert assistant.chain is None
            mock_components["mock_logger"].error.assert_called()
            with pytest.raises(AttributeError):
                assistant.add_documents(["doc"])
            assert assistant._augment_query_with_context("query") == "query"

    def test_search_manager_initialization_failure(
        self, mock_components
    ):  # pylint: disable=redefined-outer-name
        """If SearchManager constructor raises, RAGAssistant.__init__ should raise RuntimeError."""
        mock_components["init_llm_mock"].return_value = MagicMock(name="LLM")
        with patch(
            "src.rag_assistant.SearchManager", side_effect=Exception("chroma error")
        ):
            with pytest.raises(RuntimeError):
                RAGAssistant()
            mock_components["mock_logger"].exception.assert_called()

    def test_initialize_memory_failure(self, mock_components):
        """If MemoryManager constructor raises, RAGAssistant should set memory_manager to None."""
        with patch(
            "src.rag_assistant.MemoryManager",
            side_effect=Exception("memory init error"),
        ):
            assistant = RAGAssistant()
            assert assistant.memory_manager is None
            mock_components["mock_logger"].error.assert_called()

    def test_initialize_reasoning_failure(self, mock_components):
        """If ReasoningStrategyLoader constructor raises, RAGAssistant should set reasoning_strategy to None."""
        with patch(
            "src.rag_assistant.ReasoningStrategyLoader",
            side_effect=Exception("reasoning init error"),
        ):
            assistant = RAGAssistant()
            assert assistant.reasoning_strategy is None
            mock_components["mock_logger"].error.assert_called()

    def test_build_chain_failure(self, build_chain_failure_scenario, mock_components):
        """
        Parametrized test covering:
        - 'no_llm': initialize_llm returns None -> chain building skipped, warning logged
        - 'prompts_error': build_system_prompts raises -> fallback used and chain built, warning logged
        """
        if build_chain_failure_scenario == "no_llm":
            with patch("src.rag_assistant.initialize_llm", return_value=None):
                assistant = RAGAssistant()
                assert assistant.llm is None
                assert assistant.prompt_template is None
                assert assistant.chain is None
                mock_components["mock_logger"].warning.assert_called()
        else:
            with (
                patch(
                    "src.rag_assistant.build_system_prompts",
                    side_effect=Exception("prompt builder failed"),
                ),
                patch(
                    "src.rag_assistant.get_default_system_prompts"
                ) as mock_get_default,
            ):
                mock_get_default.return_value = ["default system prompt"]
                fallback_prompt_template = MagicMock(name="FallbackPromptTemplate")
                fallback_prompt_template.__or__.return_value = fallback_prompt_template
                fallback_prompt_template.__ror__.return_value = fallback_prompt_template
                mock_components[
                    "create_prompt_mock"
                ].return_value = fallback_prompt_template

                assistant = RAGAssistant()

                assert assistant.prompt_template is fallback_prompt_template
                assert assistant.chain is fallback_prompt_template
                mock_components["mock_logger"].warning.assert_called()

    def test_add_documents(self, mock_components):
        """add_documents should call through to SearchManager.add_documents and not mutate prompt/chain."""
        assistant = RAGAssistant()
        docs = ["A doc", {"title": "T", "content": "C"}]

        assistant.add_documents(docs)
        mock_components[
            "search_manager_instance"
        ].add_documents.assert_called_once_with(docs)

    @pytest.mark.parametrize(
        "chat_history, query",
        [
            ("Previous conversation exchange.", "What is this?"),
            ("", "query"),
            ("No previous conversation context.", "another"),
        ],
    )
    def test_augment_query_with_context(self, mock_components, chat_history, query):
        """_augment_query_with_context should prepend chat history when present and valid."""

        assistant = RAGAssistant()
        assistant.memory_manager = mock_components["memory_instance"]

        mock_components["memory_instance"].get_memory_variables.return_value = {
            CHAT_HISTORY: chat_history
        }
        augmented = assistant._augment_query_with_context(query)

        if chat_history and chat_history != "No previous conversation context.":
            assert chat_history in augmented
            assert augmented.endswith(f"Current question: {query}")
        else:
            assert augmented == query

    def test_augment_query_with_context_error_handling(self, mock_components):
        """_augment_query_with_context should handle exceptions and return original query."""
        assistant = RAGAssistant()
        assistant.memory_manager = mock_components["memory_instance"]

        mock_components["memory_instance"].get_memory_variables.side_effect = Exception(
            "Memory backend failure"
        )

        query = "What is this?"
        result = assistant._augment_query_with_context(query)
        assert result == query
        mock_components["mock_logger"].warning.assert_called()
        called_msg = mock_components["mock_logger"].warning.call_args.args[0]
        assert "Could not augment query with context" in called_msg

    def test_invoke(self, mock_components):
        """invoke should call SearchManager.search, then chain.invoke, and save to memory when present."""
        assistant = RAGAssistant()

        # Provide a chain and a memory manager for the flow
        assistant.chain = MagicMock()
        assistant.chain.invoke.return_value = "Chain response"
        assistant.memory_manager = mock_components["memory_instance"]

        # Provide non-empty chat history to trigger augmentation
        mock_components["memory_instance"].get_memory_variables.return_value = {
            "chat_history": "Previous conversation exchange."
        }

        # Ensure search returns expected shape
        mock_components["search_manager_instance"].search.return_value = {
            "documents": [["Doc content"]],
            "distances": [[0.1]],
        }
        mock_components[
            "search_manager_instance"
        ].flatten_search_results.return_value = (["Doc content"], [0.1])
        mock_components[
            "search_manager_instance"
        ].is_context_relevant_to_query.return_value = True

        resp = assistant.invoke("What is this?", n_results=3)

        # chain invoked and response returned
        assistant.chain.invoke.assert_called_once()
        assert resp == "Chain response"

        # verify chain was invoked with structured inputs (context and question present)
        chain_call_args = assistant.chain.invoke.call_args.args[0]
        assert isinstance(chain_call_args, dict)
        assert "context" in chain_call_args
        assert "question" in chain_call_args
        assert chain_call_args["question"] == "What is this?"
        # the augmented query should have been used for search; it should contain part of the chat history
        search_call_kwargs = mock_components[
            "search_manager_instance"
        ].search.call_args.kwargs
        assert "query" in search_call_kwargs
        assert "Previous conversation exchange." in search_call_kwargs["query"]
        assert "maximum_distance" in search_call_kwargs

        # memory saved
        mock_components["memory_instance"].add_message.assert_called_once_with(
            input_text="What is this?", output_text="Chain response"
        )

        # search flatten and logging invoked
        mock_components[
            "search_manager_instance"
        ].flatten_search_results.assert_called_once()
        mock_components[
            "search_manager_instance"
        ].log_search_results.assert_called_once()

    @pytest.mark.parametrize("scenario", ["config_error", "chain_error"])
    def test_invoke_error_conditions(self, mock_components, scenario):
        """
        Test invoke error handling for:
        1. Configuration error during search (ValueError)
        2. Chain invocation error (generic Exception)
        3. Ensure no sensitive info is leaked in responses
        4. Ensure proper methods are/aren't called based on error type
        """
        assistant = RAGAssistant()
        assistant.chain = MagicMock()
        assistant.memory_manager = (
            None  # common: run in degraded-memory mode for both scenarios
        )

        if scenario == "config_error":
            # Configuration error raised by search should produce a friendly message and not call the chain
            mock_components["search_manager_instance"].search.side_effect = ValueError(
                "Collection expecting embedding with dimension of 768, got 384"
            )

            resp = assistant.invoke("Query causing config error")
            resp_lower = resp.lower()
            assert "unable to search" in resp_lower
            assert "configuration" in resp_lower
            # ensure internal details aren't leaked
            assert "768" not in resp_lower
            assert "384" not in resp_lower
            # chain should not be invoked
            assistant.chain.invoke.assert_not_called()
            mock_components["search_manager_instance"].search.assert_called_once()
            return  # done for config_error

        assistant.chain.invoke.side_effect = Exception("LLM API timeout")
        mock_components["search_manager_instance"].search.return_value = {
            "documents": [["Doc"]],
            "distances": [[0.1]],
        }
        mock_components[
            "search_manager_instance"
        ].flatten_search_results.return_value = (["Doc"], [0.1])
        mock_components[
            "search_manager_instance"
        ].is_context_relevant_to_query.return_value = True

        resp = assistant.invoke("Any query")
        assert "encountered an error" in resp.lower()
        mock_components["search_manager_instance"].search.assert_called_once()
        mock_components[
            "search_manager_instance"
        ].flatten_search_results.assert_called_once()

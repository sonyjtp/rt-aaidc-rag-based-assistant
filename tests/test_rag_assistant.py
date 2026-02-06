"""Unit tests for RAGAssistant class in src.rag_assistant."""

from unittest.mock import MagicMock, patch

import pytest

from config import DISTANCE_THRESHOLD
from error_messages import (
    APPLICATION_INITIALIZATION_FAILED,
    LLM_INITIALIZATION_FAILED,
    NO_RESULTS_ERROR_MESSAGE,
    REASONING_STRATEGY_MISSING,
    SEARCH_FAILED_ERROR_MESSAGE,
)
from src.rag_assistant import RAGAssistant


@pytest.fixture
def mock_components():
    """Patch SearchManager, MemoryManager, and ReasoningStrategyLoader and provide mock instances."""
    with (
        patch("src.rag_assistant.SearchManager") as mock_search_manager_cls,
        patch("src.rag_assistant.MemoryManager") as mock_memory_cls,
        patch("src.rag_assistant.ReasoningStrategyLoader") as mock_reasoning_cls,
        patch("src.rag_assistant.initialize_llm") as mock_init_llm,
        patch("src.rag_assistant.PromptBuilder") as mock_prompt_builder_cls,
        patch("src.rag_assistant.get_default_system_prompts") as mock_get_default_prompts,
        patch("src.rag_assistant.logger") as mock_logger,
    ):
        # make initialize_llm return a mock LLM by default
        mock_llm = MagicMock(name="LLMDefault")
        mock_init_llm.return_value = mock_llm

        # PromptBuilder mock
        mock_prompt_builder_instance = MagicMock(name="PromptBuilderInstance")
        mock_prompt_builder_instance.build_system_prompts.return_value = ["system prompt"]
        # ensure create_prompt_template returns the shared prompt template mock
        mock_prompt_builder_instance.create_prompt_template.return_value = mock_prompt_builder_instance
        mock_prompt_builder_cls.return_value = mock_prompt_builder_instance

        # get_default_system_prompts mock
        mock_get_default_prompts.return_value = ["default system prompt"]

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

        # chain operator composition (prompt | llm | parser)
        prompt_template_mock = MagicMock(name="PromptTemplate")
        prompt_template_mock.__or__.return_value = prompt_template_mock
        prompt_template_mock.__ror__.return_value = prompt_template_mock

        # ensure the PromptBuilder creates the shared prompt template mock
        mock_prompt_builder_instance.create_prompt_template.return_value = prompt_template_mock

        yield {
            "search_manager_cls": mock_search_manager_cls,
            "search_manager_instance": search_manager_instance,
            "memory_cls": mock_memory_cls,
            "memory_instance": memory_instance,
            "reasoning_cls": mock_reasoning_cls,
            "reasoning_instance": reasoning_instance,
            "init_llm_mock": mock_init_llm,
            "mock_llm": mock_llm,
            "prompt_builder_cls": mock_prompt_builder_cls,
            "prompt_builder_instance": mock_prompt_builder_instance,
            "get_default_prompts_mock": mock_get_default_prompts,
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
        """When initialize_llm fails, RAGAssistant.__init__ should raise RuntimeError."""
        mock_components["init_llm_mock"].side_effect = RuntimeError(LLM_INITIALIZATION_FAILED)

        with pytest.raises(RuntimeError, match=LLM_INITIALIZATION_FAILED):
            RAGAssistant()

        mock_components["mock_logger"].exception.assert_called()
        exception_log_msg = mock_components["mock_logger"].exception.call_args.args[0]
        assert LLM_INITIALIZATION_FAILED in exception_log_msg

    def test_search_manager_initialization_failure(self, mock_components):  # pylint: disable=redefined-outer-name
        """If SearchManager constructor raises, RAGAssistant.__init__ should raise RuntimeError."""
        mock_components["init_llm_mock"].return_value = mock_components["mock_llm"]
        mock_components["search_manager_cls"].side_effect = RuntimeError(APPLICATION_INITIALIZATION_FAILED)

        with pytest.raises(RuntimeError):
            RAGAssistant()

        mock_components["mock_logger"].exception.assert_called()
        exception_log_msg = mock_components["mock_logger"].exception.call_args.args[0]
        assert APPLICATION_INITIALIZATION_FAILED in exception_log_msg

        mock_components["search_manager_cls"].assert_called_once_with(mock_components["mock_llm"])

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
            mock_components["init_llm_mock"].side_effect = RuntimeError(LLM_INITIALIZATION_FAILED)
            with pytest.raises(RuntimeError, match=LLM_INITIALIZATION_FAILED):
                RAGAssistant()
            mock_components["mock_logger"].exception.assert_called()
        else:
            # simulate prompt builder failing to build system prompts
            mock_components["prompt_builder_instance"].build_system_prompts.side_effect = Exception(
                REASONING_STRATEGY_MISSING
            )
            with (patch("src.rag_assistant.get_default_system_prompts") as mock_get_default,):
                mock_get_default.return_value = ["default system prompt"]
                fallback_prompt_template = MagicMock(name="FallbackPromptTemplate")
                fallback_prompt_template.__or__.return_value = fallback_prompt_template
                fallback_prompt_template.__ror__.return_value = fallback_prompt_template
                # ensure PromptBuilder.create_prompt_template returns the fallback template
                mock_components[
                    "prompt_builder_instance"
                ].create_prompt_template.return_value = fallback_prompt_template

                assistant = RAGAssistant()

                assert assistant.prompt_template is fallback_prompt_template
                assert assistant.chain is fallback_prompt_template
                mock_components["mock_logger"].warning.assert_called()

    def test_add_documents(self, mock_components):
        """add_documents should call through to SearchManager.add_documents and not mutate prompt/chain."""
        assistant = RAGAssistant()
        docs = ["A doc", {"title": "T", "content": "C"}]

        assistant.add_documents(docs)
        mock_components["search_manager_instance"].add_documents.assert_called_once_with(docs)

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
        mock_components["search_manager_instance"].flatten_search_results.return_value = (["Doc content"], [0.1])
        mock_components["search_manager_instance"].is_context_relevant_to_query.return_value = True

        resp = assistant.invoke("What is this?")

        # chain invoked and response returned
        assistant.chain.invoke.assert_called_once()
        assert resp == "Chain response"

        # verify chain was invoked with structured inputs (context and question present)
        chain_call_args = assistant.chain.invoke.call_args.args[0]
        assert isinstance(chain_call_args, dict)
        assert "context" in chain_call_args
        assert "question" in chain_call_args
        assert chain_call_args["question"] == "What is this?"
        # verify chat history was passed to chain inputs
        assert "chat_history" in chain_call_args
        assert chain_call_args["chat_history"] == "Previous conversation exchange."
        # the original query should have been used for search
        search_call_kwargs = mock_components["search_manager_instance"].search.call_args.kwargs
        assert "query" in search_call_kwargs
        assert search_call_kwargs["query"] == "What is this?"
        assert "maximum_distance" in search_call_kwargs

        # memory saved
        mock_components["memory_instance"].add_message.assert_called_once_with(
            input_text="What is this?", output_text="Chain response"
        )

        # search flatten and logging invoked
        mock_components["search_manager_instance"].flatten_search_results.assert_called_once()
        mock_components["search_manager_instance"].log_search_results.assert_called_once()

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
        assistant.memory_manager = None  # common: run in degraded-memory mode for both scenarios

        # Prevent meta-question short-circuit during error simulations
        assistant.persona_handler = None

        if scenario == "config_error":
            # Configuration error raised by search should produce a friendly message and not call the chain
            mock_components["search_manager_instance"].search.side_effect = ValueError(
                "Collection expecting embedding with dimension of 768, got 384"
            )

            resp = assistant.invoke("Query causing config error")
            # Should return a generic search failure message and not leak internals
            assert SEARCH_FAILED_ERROR_MESSAGE.lower() in resp.lower()
            assert "768" not in resp.lower()
            assert "384" not in resp.lower()
            assistant.chain.invoke.assert_not_called()
            mock_components["search_manager_instance"].search.assert_called_once()
            return  # done for config_error

        assistant.chain.invoke.side_effect = Exception("LLM API timeout")
        mock_components["search_manager_instance"].search.return_value = {
            "documents": [["Doc"]],
            "distances": [[0.1]],
        }
        mock_components["search_manager_instance"].flatten_search_results.return_value = (["Doc"], [0.1])
        mock_components["search_manager_instance"].is_context_relevant_to_query.return_value = True

        resp = assistant.invoke("Any query")
        assert "encountered an error" in resp.lower()
        mock_components["search_manager_instance"].search.assert_called_once()
        mock_components["search_manager_instance"].flatten_search_results.assert_called_once()

    def test_initialize_search_manager_with_no_llm(self, mock_components):
        """Test that _initialize_search_manager raises when LLM is None (line 77-78)."""
        mock_components["init_llm_mock"].return_value = None

        with pytest.raises(RuntimeError, match=APPLICATION_INITIALIZATION_FAILED):
            RAGAssistant()

        mock_components["mock_logger"].error.assert_called_with(APPLICATION_INITIALIZATION_FAILED)

    def test_initialize_search_manager_generic_exception(self, mock_components):
        """Test that generic exceptions in SearchManager init are caught and wrapped (lines 84-86)."""
        generic_error = Exception("Some database connection error")
        mock_components["search_manager_cls"].side_effect = generic_error

        with pytest.raises(RuntimeError, match=APPLICATION_INITIALIZATION_FAILED):
            RAGAssistant()

        # Verify the exception was logged
        mock_components["mock_logger"].exception.assert_called()
        logged_msg = mock_components["mock_logger"].exception.call_args.args[0]
        assert "ASSISTANT_INITIALIZATION_FAILED" in logged_msg or "Assistant initialization" in logged_msg

    def test_persona_handler_initialization(self, mock_components):
        """Test that PersonaHandler is properly initialized."""
        with patch("src.rag_assistant.PersonaHandler") as mock_persona_cls:
            mock_persona_instance = MagicMock()
            mock_persona_cls.return_value = mock_persona_instance

            assistant = RAGAssistant()

            assert assistant.persona_handler is mock_persona_instance
            mock_persona_cls.assert_called_once()

    def test_persona_handler_initialization_failure(self, mock_components):
        """Test that PersonaHandler initialization failures are gracefully handled."""
        with patch(
            "src.rag_assistant.PersonaHandler",
            side_effect=Exception("Persona init error"),
        ):
            assistant = RAGAssistant()
            assert assistant.persona_handler is None
            mock_components["mock_logger"].error.assert_called()

    @pytest.mark.parametrize("meta_question_response", [None, "Meta answer"])
    def test_invoke_with_persona_meta_question(self, mock_components, meta_question_response):
        """Test invoke with persona handler meta questions (lines 114-116)."""
        assistant = RAGAssistant()

        with patch("src.rag_assistant.PersonaHandler") as mock_persona_cls:
            mock_persona_instance = MagicMock()
            mock_persona_instance.handle_meta_question.return_value = meta_question_response
            mock_persona_cls.return_value = mock_persona_instance
            assistant.persona_handler = mock_persona_instance

            if meta_question_response:
                # When meta response is returned, should skip search and chain
                resp = assistant.invoke("What is your name?")
                assert resp == meta_question_response
                mock_components["search_manager_instance"].search.assert_not_called()
                assistant.chain.invoke.assert_not_called()
                # Should be added to memory if memory manager exists
                mock_components["memory_instance"].add_message.assert_called_with(
                    input_text="What is your name?",
                    output_text=meta_question_response,
                )
            else:
                # When meta response is None, should proceed with normal flow
                assistant.chain = MagicMock()
                assistant.chain.invoke.return_value = "Regular answer"
                mock_components["search_manager_instance"].search.return_value = {
                    "documents": [["Doc"]],
                    "distances": [[0.1]],
                }
                mock_components["search_manager_instance"].flatten_search_results.return_value = (["Doc"], [0.1])

                resp = assistant.invoke("Regular query")
                mock_components["search_manager_instance"].search.assert_called_once()

    def test_invoke_no_documents_found(self, mock_components):
        """Test invoke when no documents are found (line 166)."""

        assistant = RAGAssistant()
        assistant.chain = MagicMock()

        # Return empty documents
        mock_components["search_manager_instance"].search.return_value = {
            "documents": [],
            "distances": [],
        }

        resp = assistant.invoke("Query with no results")

        assert resp == NO_RESULTS_ERROR_MESSAGE
        assistant.chain.invoke.assert_not_called()
        mock_components["mock_logger"].warning.assert_called_with("No documents found for query: Query with no results")

    def test_invoke_no_documents_key(self, mock_components):
        """Test invoke when search results don't have 'documents' key."""

        assistant = RAGAssistant()
        assistant.chain = MagicMock()

        # Return None or missing documents key
        mock_components["search_manager_instance"].search.return_value = {
            "distances": [[0.1]],
        }

        resp = assistant.invoke("Another query")

        assert resp == NO_RESULTS_ERROR_MESSAGE
        assistant.chain.invoke.assert_not_called()

    def test_invoke_context_validation_failure(self, mock_components):
        """Test invoke when context validation fails (lines 171-176)."""
        assistant = RAGAssistant()
        assistant.chain = MagicMock()
        assistant.chain.invoke.return_value = "Response"

        mock_components["search_manager_instance"].search.return_value = {
            "documents": [["Irrelevant doc"]],
            # Use a distance higher than typical DISTANCE_THRESHOLD so there is NO strong match
            "distances": [[DISTANCE_THRESHOLD + 0.1]],
        }
        mock_components["search_manager_instance"].flatten_search_results.return_value = (
            ["Irrelevant doc"],
            [DISTANCE_THRESHOLD + 0.1],
        )

        # Mock QueryProcessor to fail validation
        with patch("src.rag_assistant.QueryProcessor") as mock_qp_cls:
            mock_qp_instance = MagicMock()
            mock_qp_instance.validate_context.return_value = False
            mock_qp_cls.return_value = mock_qp_instance

            # Ensure the assistant uses the mocked QueryProcessor instance
            assistant.query_processor = mock_qp_instance

            resp = assistant.invoke("Query with invalid context")

            # With validation failure and no strong match, should return default error and not call chain
            assert resp == NO_RESULTS_ERROR_MESSAGE
            assistant.chain.invoke.assert_not_called()
            # Logger should have been warned about context validation failure
            assert any(
                "Context validation failed for query" in str(call)
                for call in mock_components["mock_logger"].warning.call_args_list
            )

    def test_invoke_slow_response_logging(self, mock_components):
        """Test that slow responses (> 5 seconds) are logged with warning (lines 187-188)."""
        assistant = RAGAssistant()
        assistant.chain = MagicMock()
        assistant.chain.invoke.return_value = "Slow response"

        mock_components["search_manager_instance"].search.return_value = {
            "documents": [["Doc"]],
            "distances": [[0.1]],
        }
        mock_components["search_manager_instance"].flatten_search_results.return_value = (["Doc"], [0.1])

        # Mock time to simulate slow response
        with patch("src.rag_assistant.time.time") as mock_time:
            mock_time.side_effect = [0.0, 6.5]  # 6.5 second response

            resp = assistant.invoke("Slow query")

            assert resp == "Slow response"
            # Verify warning was logged for slow response
            mock_components["mock_logger"].warning.assert_called()
            warning_msg = mock_components["mock_logger"].warning.call_args.args[0]
            assert "Slow response detected" in warning_msg
            assert "6500.00ms" in warning_msg

    def test_invoke_fast_response_debug_logging(self, mock_components):
        """Test that fast responses (< 5 seconds) are logged with debug."""
        assistant = RAGAssistant()
        assistant.chain = MagicMock()
        assistant.chain.invoke.return_value = "Fast response"

        mock_components["search_manager_instance"].search.return_value = {
            "documents": [["Doc"]],
            "distances": [[0.1]],
        }
        mock_components["search_manager_instance"].flatten_search_results.return_value = (["Doc"], [0.1])

        # Mock time to simulate fast response
        with patch("src.rag_assistant.time.time") as mock_time:
            mock_time.side_effect = [0.0, 0.5]

            resp = assistant.invoke("Fast query")

            assert resp == "Fast response"
            # Verify debug was logged for response time
            debug_calls = [
                debug_call
                for debug_call in mock_components["mock_logger"].debug.call_args_list
                if "Response time" in str(debug_call)
            ]
            assert len(debug_calls) > 0

    def test_invoke_memory_persistence(self, mock_components):
        """Test that responses are persisted to memory (line 246)."""
        assistant = RAGAssistant()
        assistant.chain = MagicMock()
        assistant.chain.invoke.return_value = "Final response"
        assistant.memory_manager = mock_components["memory_instance"]

        mock_components["search_manager_instance"].search.return_value = {
            "documents": [["Doc"]],
            "distances": [[0.1]],
        }
        mock_components["search_manager_instance"].flatten_search_results.return_value = (["Doc"], [0.1])

        resp = assistant.invoke("Test query")

        # Verify memory was updated with the response
        mock_components["memory_instance"].add_message.assert_called_with(
            input_text="Test query", output_text="Final response"
        )
        assert resp == "Final response"

    def test_invoke_without_memory_manager(self, mock_components):
        """Test that invoke works without memory manager."""
        assistant = RAGAssistant()
        assistant.chain = MagicMock()
        assistant.chain.invoke.return_value = "Response"
        assistant.memory_manager = None  # No memory manager

        mock_components["search_manager_instance"].search.return_value = {
            "documents": [["Doc"]],
            "distances": [[0.1]],
        }
        mock_components["search_manager_instance"].flatten_search_results.return_value = (["Doc"], [0.1])

        resp = assistant.invoke("Query without memory")

        assert resp == "Response"
        # Should not crash when memory_manager is None
        assistant.chain.invoke.assert_called_once()

    def test_invoke_without_chain(self, mock_components):
        """Test invoke when chain is not initialized (line 166)."""
        assistant = RAGAssistant()
        assistant.chain = None  # No chain initialized

        with pytest.raises(RuntimeError):
            assistant.invoke("Any query")
        mock_components["search_manager_instance"].search.assert_not_called()

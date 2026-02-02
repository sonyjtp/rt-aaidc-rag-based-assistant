"""RAG-based AI assistant using ChromaDB and multiple LLM providers."""

# pylint: disable=no-name-in-module, import-error

import time

from langchain_core.output_parsers import StrOutputParser

from config import CHAT_HISTORY, DISTANCE_THRESHOLD, RETRIEVAL_K
from error_messages import (
    ASSISTANT_INITIALIZATION_FAILED,
    DEFAULT_NOT_KNOWN_ERROR_MESSAGE,
    LLM_INITIALIZATION_FAILED,
    SEARCH_MANAGER_INITIALIZATION_FAILED,
)
from llm_utils import initialize_llm
from logger import logger
from memory_manager import MemoryManager
from persona_handler import PersonaHandler
from prompt_builder import (
    build_system_prompts,
    create_prompt_template,
    get_default_system_prompts,
)
from query_processor import QueryProcessor
from reasoning_strategy_loader import ReasoningStrategyLoader
from search_manager import SearchManager


class RAGAssistant:
    """
    A simple RAG-based AI assistant using ChromaDB and multiple LLM providers.
    Supports OpenAI, Groq, and Google Gemini APIs.
    """

    def __init__(self):
        """Initialize the RAG assistant.
        Steps:
        1. Initialize LLM
        2. Initialize search manager
        3. Initialize conversation memory
        4. Initialize reasoning strategy
        5. Initialize persona handler
        6. Build prompt template and LLM chain
        """
        self.llm = None
        self.search_manager = None
        self.memory_manager = None
        self.reasoning_strategy = None
        self.persona_handler = None
        self.prompt_template = None
        self.chain = None

        self._initialize_llm()
        self._initialize_search_manager()
        self._initialize_memory()
        self._initialize_reasoning_strategy()
        self._initialize_persona_handler()
        self._build_chain()

    def _initialize_llm(self):
        """Initialize the LLM, setting to None if initialization fails."""
        try:
            self.llm = initialize_llm()
        except RuntimeError as e:
            logger.exception(f"{LLM_INITIALIZATION_FAILED}: {e}")
            raise

    def _initialize_search_manager(self) -> None:
        """Initialize the search manager with vector DB and LLM.

        If the LLM is unavailable the assistant continues in degraded mode without a
        search manager. Any initialization error is logged with its stack trace.
        """
        try:
            if self.llm is None:
                logger.error(SEARCH_MANAGER_INITIALIZATION_FAILED)
                raise RuntimeError(SEARCH_MANAGER_INITIALIZATION_FAILED)
            self.search_manager = SearchManager(self.llm)
            logger.debug("Search manager initialized.")
        except RuntimeError as e:
            logger.exception(f"{ASSISTANT_INITIALIZATION_FAILED}: {e}")
            raise
        except Exception as e:
            logger.exception(f"{ASSISTANT_INITIALIZATION_FAILED}: {e}")
            raise RuntimeError(SEARCH_MANAGER_INITIALIZATION_FAILED) from e

    def _initialize_memory(self) -> None:
        """Initialize conversation memory."""
        try:
            self.memory_manager = MemoryManager(llm=self.llm)
            if self.memory_manager.memory:
                logger.info(f"Memory strategy: {self.memory_manager.strategy}")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Error initializing memory: {e}")
            self.memory_manager = None

    def _initialize_reasoning_strategy(self) -> None:
        """Initialize reasoning strategy."""
        try:
            self.reasoning_strategy = ReasoningStrategyLoader()
            logger.info(f"Reasoning strategy: {self.reasoning_strategy.get_strategy_name()}")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Error initializing reasoning strategy: {e}")
            self.reasoning_strategy = None

    def _initialize_persona_handler(self) -> None:
        """Initialize persona handler for meta question detection."""
        try:
            self.persona_handler = PersonaHandler()
            logger.info("Persona handler initialized for meta question detection.")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Error initializing persona handler: {e}")
            self.persona_handler = None

    def _build_chain(self) -> None:
        """Build the prompt template and LLM chain.
        Steps:
        1. Build system prompts
        2. Create prompt template
        3. Combine prompt template, LLM, and output parser into a chain
        """

        try:
            system_prompts = build_system_prompts(self.reasoning_strategy)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(f"Could not build system prompts: {e}. Using default prompts.")
            # Falling back to default prompts
            system_prompts = get_default_system_prompts()

        self.prompt_template = create_prompt_template(system_prompts)
        logger.debug("Prompt template created.")

        self.chain = self.prompt_template | self.llm | StrOutputParser()
        logger.info("Function chain with prompt template, LLM, and parser built.")

    def add_documents(self, documents: list[str] | list[dict[str, str]]) -> None:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of documents
        """
        self.search_manager.add_documents(documents)

    def invoke(self, query: str, n_results: int = RETRIEVAL_K) -> str:
        """
        Query the RAG assistant with a user question.

        Args:
            query: User's input
            n_results: Number of relevant chunks to retrieve

        Returns:
            String answer from the LLM based on retrieved context
        """
        logger.debug(f"n_results: {n_results}")
        # Start timing
        start_time = time.time()

        if not self.chain:
            return "RAG Assistant is not properly initialized."

        if self.persona_handler:
            meta_response = self.persona_handler.handle_meta_question(query)
            if meta_response:
                logger.debug("Meta question detected")
                if self.memory_manager:
                    self.memory_manager.add_message(input_text=query, output_text=meta_response)
                return meta_response

        try:
            # delegate search to DocumentManager
            search_results = self.search_manager.search(
                query=query,
                n_results=n_results,
                maximum_distance=DISTANCE_THRESHOLD,
            )
            # Validate that results were actually retrieved
            if not search_results or not search_results.get("documents"):
                logger.warning(f"No documents found for query: {query}")
                return DEFAULT_NOT_KNOWN_ERROR_MESSAGE

        except ValueError as e:
            logger.error(f"Configuration error during search: {e}")
            return (
                "I'm unable to search the documents at the moment due to a system "
                "configuration issue. Please try again later or contact support."
            )

        flat_docs, flat_distances = self.search_manager.flatten_search_results(search_results)
        self.search_manager.log_search_results(flat_docs, flat_distances)

        context = (
            "\n".join(flat_docs)
            if flat_docs and (not flat_distances or flat_distances[0] <= DISTANCE_THRESHOLD)
            else ""
        )

        # Validate that retrieved context actually addresses the query
        # This prevents hallucinations from false positive vector matches
        if context and context.strip():
            search_processor = QueryProcessor(self.memory_manager, self.llm)
            if not search_processor.validate_context(query, context, query):
                logger.warning(f"Context validation failed for query: {query}")
                context = ""
        try:
            memory_vars = self.memory_manager.get_memory_variables() if self.memory_manager else {}
            chain_inputs = {
                "context": context,
                "question": query,
                CHAT_HISTORY: memory_vars.get(CHAT_HISTORY, "") or "No previous conversation context.",
            }

            response = self.chain.invoke(chain_inputs)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Error invoking chain: {e}")
            response = "I encountered an error while processing your question. " "Please try again."

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        elapsed_ms = elapsed_time * 1000

        logger.debug(f"Query: {query}")
        logger.debug(f"Query Result: {response}")

        # Log warning if response time exceeds 5 seconds
        if elapsed_ms > 5000:
            logger.warning(f"Slow response detected: {elapsed_ms:.2f}ms for query: {query}")
        else:
            logger.debug(f"Response time: {elapsed_ms:.2f}ms")

        if self.memory_manager:
            self.memory_manager.add_message(input_text=query, output_text=response)

        return response

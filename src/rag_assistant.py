"""RAG-based AI assistant using ChromaDB and multiple LLM providers."""

# pylint: disable=no-name-in-module, import-error

import time

from langchain_core.output_parsers import StrOutputParser

from config import (
    CHAT_HISTORY,
    DISTANCE_THRESHOLD,
    ERROR_SEARCH_MANAGER_UNAVAILABLE,
    RETRIEVAL_K,
)
from llm_utils import initialize_llm
from logger import logger
from memory_manager import MemoryManager
from prompt_builder import (
    build_system_prompts,
    create_prompt_template,
    get_default_system_prompts,
)
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
        5. Build prompt template and LLM chain
        """
        self.llm = None
        self.search_manager = None
        self.memory_manager = None
        self.reasoning_strategy = None
        self.prompt_template = None
        self.chain = None

        self._initialize_llm()
        self._initialize_search_manager()
        self._initialize_memory()
        self._initialize_reasoning_strategy()
        self._build_chain()

    def _initialize_llm(self) -> None:
        """Initialize the LLM with error handling."""
        try:
            self.llm = initialize_llm()
            if hasattr(self.llm, "model_name"):
                logger.info(f"LLM: {self.llm.model_name}")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Error initializing LLM: {e}")
            self.llm = None

    def _initialize_search_manager(self) -> None:
        """Initialize the search manager with vector DB and LLM.

        If the LLM is unavailable the assistant continues in degraded mode without a
        search manager. Any initialization error is logged with its stack trace.
        """
        if self.llm is None:
            logger.warning(ERROR_SEARCH_MANAGER_UNAVAILABLE)
            self.search_manager = None
            return
        try:
            self.search_manager = SearchManager(self.llm)
            logger.debug("Search manager initialized.")
        except Exception as e:
            logger.exception(f"Failed to initialize SearchManager: {e}")
            raise RuntimeError(ERROR_SEARCH_MANAGER_UNAVAILABLE) from e

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
            logger.info(
                f"Reasoning strategy: {self.reasoning_strategy.get_strategy_name()}"
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Error initializing reasoning strategy: {e}")
            self.reasoning_strategy = None

    def _build_chain(self) -> None:
        """Build the prompt template and LLM chain.
        Steps:
        1. Build system prompts
        2. Create prompt template
        3. Combine prompt template, LLM, and output parser into a chain
        """
        if not self.llm:
            logger.warning("LLM not initialized. Skipping chain building.")
            return

        try:
            system_prompts = build_system_prompts(self.reasoning_strategy)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(
                f"Could not build system prompts: {e}. Using default prompts."
            )
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

    def _augment_query_with_context(self, query: str) -> str:
        """
        Augment the user's query with recent conversation context.
        This helps resolve pronouns and references in follow-up questions.

        Args:
            query: Original user query

        Returns:
            Augmented query that includes recent context
        """
        if not self.memory_manager:
            return query

        try:
            memory_vars = self.memory_manager.get_memory_variables()
            chat_history = memory_vars.get(CHAT_HISTORY, "")

            if chat_history and chat_history != "No previous conversation context.":
                # Extract the last exchange to provide context
                # This helps resolve pronouns like "it" in follow-up questions
                augmented_query = f"{chat_history}\n\nCurrent question: {query}"
                logger.debug("Query augmented with chat history context")
                return augmented_query

        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(f"Could not augment query with context: {e}")

        return query

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

        search_query = self._augment_query_with_context(query)

        try:
            # delegate search to DocumentManager
            search_results = self.search_manager.search(
                query=search_query,
                n_results=n_results,
                maximum_distance=DISTANCE_THRESHOLD,
            )
        except ValueError as e:
            logger.error(f"Configuration error during search: {e}")
            return (
                "I'm unable to search the documents at the moment due to a system "
                "configuration issue. Please try again later or contact support."
            )

        flat_docs, flat_distances = self.search_manager.flatten_search_results(
            search_results
        )
        self.search_manager.log_search_results(flat_docs, flat_distances)

        context = (
            "\n".join(flat_docs)
            if flat_docs
            and (not flat_distances or flat_distances[0] <= DISTANCE_THRESHOLD)
            else ""
        )

        # Validate that retrieved context actually addresses the query
        # This prevents hallucinations from false positive vector matches
        if context and context.strip():
            if not self.search_manager.is_context_relevant_to_query(query, context):
                logger.debug(f"Retrieved context not relevant to query: {query}")
                context = ""  # Clear context to force "not known to me" response
        try:
            memory_vars = (
                self.memory_manager.get_memory_variables()
                if self.memory_manager
                else {}
            )
            chain_inputs = {
                "context": context,
                "question": query,
                CHAT_HISTORY: memory_vars.get(CHAT_HISTORY, "")
                or "No previous conversation context.",
            }

            response = self.chain.invoke(chain_inputs)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Error invoking chain: {e}")
            response = (
                "I encountered an error while processing your question. "
                "Please try again."
            )

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        elapsed_ms = elapsed_time * 1000

        logger.debug(f"Query: {query}")
        logger.debug(f"Query Result: {response}")

        # Log warning if response time exceeds 5 seconds
        if elapsed_ms > 5000:
            logger.warning(
                f"Slow response detected: {elapsed_ms:.2f}ms for query: {query}"
            )
        else:
            logger.debug(f"Response time: {elapsed_ms:.2f}ms")

        if self.memory_manager:
            self.memory_manager.add_message(input_text=query, output_text=response)

        return response

"""RAG-based AI assistant using ChromaDB and multiple LLM providers."""

# pylint: disable=no-name-in-module, import-error

import time

from langchain_core.output_parsers import StrOutputParser

from app_constants import CHAT_HISTORY, NO_CHAT_HISTORY
from config import DISTANCE_THRESHOLD, RETRIEVAL_K
from error_messages import (
    APPLICATION_INITIALIZATION_FAILED,
    ASSISTANT_INITIALIZATION_FAILED,
    LLM_INITIALIZATION_FAILED,
    NO_RESULTS_ERROR_MESSAGE,
    SEARCH_FAILED_ERROR_MESSAGE,
)
from llm_utils import initialize_llm
from log_manager import logger
from memory_manager import MemoryManager
from persona_handler import PersonaHandler
from prompt_builder import PromptBuilder, get_default_system_prompts
from query_processor import QueryProcessor
from reasoning_strategy_loader import ReasoningStrategyLoader
from search_manager import SearchManager


class RAGAssistant:
    """
    A simple RAG-based AI assistant using ChromaDB and multiple LLM providers.
    Supports OpenAI, Groq, and Google Gemini APIs.
    Combines document retrieval, conversation memory, reasoning strategies,
    and meta question handling for robust question answering.

    Attributes:
        llm: The language model instance
        search_manager: Manages document retrieval from vector DB
        memory_manager: Manages conversation memory
        reasoning_strategy: Loaded reasoning strategy for the assistant
        persona_handler: Handles meta questions about the assistant
        prompt_template: The prompt template for LLM input
        chain: The LLM chain combining prompt, LLM, and output parser
    Methods:
        add_documents(documents): Add documents to the knowledge base
        invoke(query, n_results): Query the assistant with a user question

    """

    def __init__(self):
        """Initialize the RAG assistant.

        Steps:
        1. Initialize LLM
        2. Initialize search manager with vector DB and LLM
        3. Initialize conversation memory
        4. Initialize reasoning strategy
        5. Initialize persona handler for meta question detection
        6. Initialize query processor for query augmentation and context validation
        7. Build prompt template
        8. Build LLM chain with prompt template, LLM, and output parser

        """
        self.llm = None
        self.search_manager = None
        self.memory_manager = None
        self.reasoning_strategy = None
        self.persona_handler = None
        self.query_processor = None
        self.prompt_template = None
        self.chain = None

        self._initialize_llm()
        self._initialize_search_manager()
        self._initialize_memory()
        self._initialize_reasoning_strategy()
        self._initialize_persona_handler()
        self._initialize_query_processor()
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
                logger.error(APPLICATION_INITIALIZATION_FAILED)
                raise RuntimeError(APPLICATION_INITIALIZATION_FAILED)
            self.search_manager = SearchManager(self.llm)
            logger.debug("Search manager initialized.")
        except Exception as e:
            logger.exception(f"{ASSISTANT_INITIALIZATION_FAILED}: {e}")
            raise RuntimeError(APPLICATION_INITIALIZATION_FAILED) from e

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
            if self.reasoning_strategy is not None:
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

    def _initialize_query_processor(self) -> None:
        """Initialize query processor for query augmentation and context validation."""
        try:
            self.query_processor = QueryProcessor(memory_manager=self.memory_manager, llm=self.llm)
            logger.info("Query processor initialized for query augmentation and context validation.")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Error initializing query processor: {e}")
            self.query_processor = None

    def _build_chain(self) -> None:
        """Build the prompt template and LLM chain.
        Steps:
        1. Build system prompts
        2. Create prompt template
        3. Combine prompt template, LLM, and output parser into a chain
        """
        prompt_builder = PromptBuilder(self.reasoning_strategy)
        try:
            system_prompts = prompt_builder.build_system_prompts()
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(f"Could not build system prompts: {e}. Using default prompts.")
            # Falling back to default prompts
            system_prompts = get_default_system_prompts()
        logger.info("System prompts built with role, style, constraints, format, and reasoning.")
        self.prompt_template = prompt_builder.create_prompt_template(system_prompts)
        logger.debug("Prompt template created from system prompts.")
        self.chain = self.prompt_template | self.llm | StrOutputParser()
        logger.info("Function chain with prompt template, LLM, and parser built.")

    def add_documents(self, documents: list[str] | list[dict[str, str]]) -> None:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of documents
        """
        self.search_manager.add_documents(documents)

    def invoke(self, query: str) -> str:
        """
        Query the RAG assistant with a user question.
        Steps:
        1. Check for meta questions via persona handler. If detected, return predefined response.
        Add interaction to memory.
        2. Augment follow-up questions with chat history context. If memory is enabled, use QueryProcessor to
        augment query. Then use augmented query for search.
        3. Search for relevant document chunks. If no results, return default error message.
        4. Flatten and log search results.
        5. Build context from retrieved documents only if top result meets distance threshold.
        6. Validate that retrieved context addresses the query using QueryProcessor. If validation fails and
        no strong match, return default error message.
        7. Build chain inputs with context, question, and chat history.
        8. Invoke the chain to get the answer from the LLM.
        9. Log response time and warn if slow.
        10. Add interaction to memory.

        Args:
            query: User's input

        Returns:
            String answer from the LLM based on retrieved context
        """
        # Start timing
        start_time = time.time()

        if not self.chain:
            raise RuntimeError(APPLICATION_INITIALIZATION_FAILED)

        # Handle meta questions
        if self.persona_handler:
            meta_response = self.persona_handler.handle_meta_question(query)
            if meta_response:
                logger.info("Meta question detected")
                if self.memory_manager:
                    self.memory_manager.add_message(input_text=query, output_text=meta_response)
                return meta_response

        # Augment query with context if available
        search_query = query
        if self.query_processor and self.memory_manager:
            search_query = self.query_processor.augment_query_with_context(query)

        # Search for relevant documents
        try:
            logger.info(f"Searching for top {RETRIEVAL_K} documents with max distance of {DISTANCE_THRESHOLD}")
            search_results = self.search_manager.search(
                query=search_query,
                n_results=RETRIEVAL_K,
                maximum_distance=DISTANCE_THRESHOLD,
            )
            if not search_results or not search_results.get("documents"):
                logger.warning(f"No documents found for query: {query}")
                return NO_RESULTS_ERROR_MESSAGE
        except ValueError as e:
            logger.error(f"Configuration error during search: {e}")
            return SEARCH_FAILED_ERROR_MESSAGE

        # Process and validate results
        flat_docs, flat_distances = self.search_manager.flatten_search_results(search_results)
        logger.debug("Flattened search results")
        self.search_manager.log_search_results(flat_docs, flat_distances)

        has_strong_match = flat_distances and flat_distances[0] < DISTANCE_THRESHOLD
        context = "\n".join(flat_docs) if flat_docs else ""

        # Validate context relevance
        is_valid = True
        if context and context.strip() and self.query_processor:
            is_valid = self.query_processor.validate_context(query, context, search_query)

        # If no valid context, return error message
        if not context or (not is_valid and not has_strong_match):
            logger.warning(
                f"Context validation failed for query (similarity: {flat_distances[0] if flat_distances else 'N/A'})"
            )
            return NO_RESULTS_ERROR_MESSAGE

        # Generate response
        try:
            memory_vars = self.memory_manager.get_memory_variables() if self.memory_manager else {}
            chain_inputs = {
                "context": context,
                "question": query,
                CHAT_HISTORY: memory_vars.get(CHAT_HISTORY, "") or NO_CHAT_HISTORY,
            }
            response = self.chain.invoke(chain_inputs)
        except Exception as e:
            logger.error(f"Error invoking chain: {e}")
            response = "I encountered an error while processing your question. Please try again."

        # Log timing and store in memory
        elapsed_ms = (time.time() - start_time) * 1000
        logger.debug(f"Query: {query}")
        logger.debug(f"Query Result: {response}")

        if elapsed_ms > 5000:
            logger.warning(f"Slow response detected: {elapsed_ms:.2f}ms for query: {query}")
        else:
            logger.debug(f"Response time: {elapsed_ms:.2f}ms")

        if self.memory_manager:
            self.memory_manager.add_message(input_text=query, output_text=response)

        return response

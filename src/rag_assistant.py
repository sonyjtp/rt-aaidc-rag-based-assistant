"""RAG-based AI assistant using ChromaDB and multiple LLM providers."""

# pylint: disable=no-name-in-module, import-error

import time

from langchain_core.output_parsers import StrOutputParser

from config import DISTANCE_THRESHOLD, MEMORY_KEY_PARAM, RETRIEVAL_K
from llm_utils import initialize_llm
from logger import logger
from memory_manager import MemoryManager
from prompt_builder import (
    build_system_prompts,
    create_prompt_template,
    get_default_system_prompts,
)
from reasoning_strategy_loader import ReasoningStrategyLoader
from vectordb import VectorDB


class RAGAssistant:
    """
    A simple RAG-based AI assistant using ChromaDB and multiple LLM providers.
    Supports OpenAI, Groq, and Google Gemini APIs.
    """

    def __init__(self):
        """Initialize the RAG assistant.
        Steps:
        1. Initialize LLM
        2. Initialize vector database for document retrieval
        3. Initialize conversation memory
        4. Initialize reasoning strategy
        5. Build prompt template and LLM chain
        """
        self.llm = None
        self.vector_db = None
        self.memory_manager = None
        self.reasoning_strategy = None
        self.prompt_template = None
        self.chain = None

        self._initialize_llm()
        self._initialize_vector_db()
        self._initialize_memory()
        self._initialize_reasoning_strategy()
        self._build_chain()

    def _initialize_llm(self) -> None:
        """Initialize the LLM."""
        try:
            self.llm = initialize_llm()
            logger.info(f"LLM: {self.llm.model_name}")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Error initializing LLM: {e}")
            self.llm = None

    def _initialize_vector_db(self) -> None:
        """Initialize the vector database."""
        try:
            self.vector_db = VectorDB()
            logger.info("Vector database  initialized.")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Error initializing vector database: {e}")
            self.vector_db = None

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
        self.vector_db.add_documents(documents)

    @staticmethod
    def _flatten_search_results(search_results: dict) -> tuple[list[str], list[float]]:
        """
        Flatten nested search results into simple lists.

        Args:
            search_results: Raw search results from vector database

        Returns:
            Tuple of (flattened_documents, flattened_distances)
        """
        documents = search_results.get("documents", [])
        distances = search_results.get("distances", [])

        # Flatten documents if nested
        if documents and isinstance(documents[0], list):
            flat_docs = [doc for doc_list in documents for doc in doc_list]
        else:
            flat_docs = documents

        # Flatten distances if nested
        if distances and isinstance(distances[0], list):
            flat_distances = [dist for dist_list in distances for dist in dist_list]
        else:
            flat_distances = distances

        return flat_docs, flat_distances

    @staticmethod
    def _log_search_results(flat_docs: list[str], flat_distances: list[float]) -> None:
        """
        Log search results for debugging purposes.

        Args:
            flat_docs: Flattened list of documents
            flat_distances: Flattened list of distances
        """
        # Truncate documents to first 50 chars for readability
        truncated_docs = [
            doc[:50] + "..." if isinstance(doc, str) and len(doc) > 50 else doc
            for doc in flat_docs
        ]
        # Convert distances to similarity scores (1 - distance)
        similarity_scores = [1 - dist for dist in flat_distances]
        for i, (doc, sim_score) in enumerate(zip(truncated_docs, similarity_scores)):
            logger.debug(
                f"Retrieved Result {i + 1}: {doc}, similarity_score: {sim_score}"
            )

    def _is_context_relevant_to_query(self, query: str, context: str) -> bool:
        """
        Validate that retrieved context actually addresses the user's query.
        Uses the LLM to perform semantic validation.

        Args:
            query: User's question
            context: Retrieved document context

        Returns:
            True if context answers the query, False otherwise
        """
        if not context or not context.strip():
            return False

        if not self.llm:
            logger.warning(
                "LLM not available for context validation, assuming relevant"
            )
            return True

        try:
            validation_prompt = (
                f"Does the following context contain information that directly "
                f"addresses this question? Answer with only 'YES' or 'NO'.\n\n"
                f"Question: {query}\n\n"
                f"Context: {context[:500]}"  # Limit to first 500 chars for speed
            )
            result = self.llm.invoke(validation_prompt)
            # Handle both string and AIMessage responses
            result_str = str(result) if not isinstance(result, str) else result
            # For AIMessage objects, extract content attribute
            if hasattr(result, "content"):
                result_str = result.content
            is_relevant = "YES" in result_str.upper()
            logger.debug(f"Context relevance validation: {is_relevant}")
            return is_relevant
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(f"Error validating context relevance: {e}")
            # On error, assume context is relevant to avoid false negatives
            return True

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
            chat_history = memory_vars.get(MEMORY_KEY_PARAM, "")

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
        # Start timing
        start_time = time.time()

        if not self.chain:
            return "RAG Assistant is not properly initialized."

        search_query = self._augment_query_with_context(query)

        try:
            search_results = self.vector_db.search(
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

        flat_docs, flat_distances = self._flatten_search_results(search_results)
        self._log_search_results(flat_docs, flat_distances)

        context = (
            "\n".join(flat_docs)
            if flat_docs
            and (not flat_distances or flat_distances[0] <= DISTANCE_THRESHOLD)
            else ""
        )

        # Validate that retrieved context actually addresses the query
        # This prevents hallucinations from false positive vector matches
        if context and context.strip():
            if not self._is_context_relevant_to_query(query, context):
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
                MEMORY_KEY_PARAM: memory_vars.get(MEMORY_KEY_PARAM, "")
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

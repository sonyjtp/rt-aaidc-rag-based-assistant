"""RAG-based AI assistant using ChromaDB and multiple LLM providers."""

import os

import yaml
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import DISTANCE_THRESHOLD_DEFAULT, RETRIEVAL_K_DEFAULT
from llm_utils import initialize_llm
from logger import logger
from memory_manager import MemoryManager
from prompt_builder import build_system_prompts
from query_classifier import QUERY_CLASSIFIERS, QueryType
from reasoning_strategy_loader import ReasoningStrategyLoader
from vectordb import VectorDB


class RAGAssistant:
    """
    A simple RAG-based AI assistant using ChromaDB and multiple LLM providers.
    Supports OpenAI, Groq, and Google Gemini APIs.
    """

    def __init__(self):
        """Initialize the RAG assistant."""
        self.llm = initialize_llm()
        logger.info(f"LLM: {self.llm.model_name}")

        # Initialize vector database
        self.vector_db = VectorDB()

        # Initialize conversation memory
        self.memory_manager = MemoryManager(llm=self.llm)
        if self.memory_manager.memory:
            logger.info(
                f"Memory manager initialized with strategy: {self.memory_manager.strategy}"
            )

        # Initialize reasoning strategy
        try:
            self.reasoning_strategy = ReasoningStrategyLoader()
            logger.info(
                f"Reasoning strategy loaded: {self.reasoning_strategy.get_strategy_name()}"
            )
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Error loading reasoning strategy: {e}")
            self.reasoning_strategy = None

        self._build_chain()

    def _build_chain(self):
        """Build the prompt template and LLM chain."""
        try:
            system_prompts = build_system_prompts()
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(
                f"Could not build system prompts: {e}. Using default prompts."
            )
            # Use a minimal default prompt if system prompts fail to build
            system_prompts = [
                "You are a helpful AI assistant.",
                "Answer questions based only on the provided documents.",
                "If you cannot find the answer in the documents, say so.",
            ]

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "\n".join(system_prompts)),
                ("human", "Context from documents:\n{context}\n\nQuestion: {question}"),
            ]
        )
        logger.info("Prompt template for RAG Assistant created from system prompts.")
        self.chain = self.prompt_template | self.llm | StrOutputParser()
        logger.info("Function chain with prompt template, LLM, and parser built.")

    def _classify_query(self, query: str) -> QueryType:
        """
        Classify query into type (unsafe > vague > meta > document > regular).

        Priority order ensures safety first, then broad topic questions,
        then capability questions, then knowledge base questions, then regular Q&A.

        Args:
            query: User query string

        Returns:
            QueryType enum value
        """
        # Priority 1: Check for unsafe content (highest priority)
        if QUERY_CLASSIFIERS["unsafe"]["pattern"].search(query):
            logger.warning(f"Unsafe query classified: {query}")
            return QueryType.UNSAFE

        # Priority 2: Check for meta-questions (identity/capabilities)
        if QUERY_CLASSIFIERS["meta"]["pattern"].search(query):
            logger.debug(f"Meta-question classified: {query}")
            return QueryType.META

        # Priority 3: Check for document/knowledge-base questions
        if QUERY_CLASSIFIERS["document"]["pattern"].search(query):
            logger.debug(f"Document question classified: {query}")
            return QueryType.DOCUMENT

        # Priority 4: Use LLM to detect vague/broad topic requests
        try:
            config_path = os.path.join(
                os.path.dirname(__file__), "..", "config", "prompt-config.yaml"
            )
            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)

            vague_template = config.get("query_vague_detection", {}).get("template", "")
            if vague_template:
                prompt_text = vague_template.format(query=query)
                response = self.llm.invoke(prompt_text).strip().lower()

                if response in ["yes", "true"]:
                    logger.info(f"Vague question classified by LLM: {query}")
                    return QueryType.VAGUE
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.debug(f"Error in LLM vague detection: {e}, treating as regular")

        # Default: Regular Q&A query
        logger.debug(f"Regular question classified: {query}")
        return QueryType.REGULAR

    def _validate_and_log_similarity(
        self, query_type: QueryType, flat_docs: list, flat_distances: list
    ) -> tuple[bool, str]:
        """
        Validate documents meet similarity threshold and log results.

        Returns:
            (is_valid, rejection_message) tuple.
            is_valid=True means proceed, False means reject with message.
        """
        # Log similarity information for all queries
        if flat_docs and flat_distances:
            similarity = 1 - flat_distances[0]
            distance = flat_distances[0]
            logger.debug(
                f"Document search: {query_type.value} question | "
                f"docs_found={len(flat_docs)} | "
                f"distance={distance:.3f} | similarity={similarity:.3f} | "
                f"threshold={DISTANCE_THRESHOLD_DEFAULT}"
            )
        elif not flat_docs:
            logger.warning(
                f"Document search: {query_type.value} question | docs_found=0"
            )

        # Check threshold only for REGULAR questions
        if query_type != QueryType.REGULAR:
            return True, ""

        # REGULAR questions require high similarity
        if not flat_docs:
            logger.warning("Query rejected - no documents found")
            return False, (
                "I couldn't find information in my knowledge base that closely matches your question. "
                "Could you try rephrasing it or asking about a different topic? "
                "This helps me provide more accurate answers based on the documents I have access to."
            )

        if flat_distances and flat_distances[0] > DISTANCE_THRESHOLD_DEFAULT:
            logger.warning(
                f"Query rejected - low similarity (distance={flat_distances[0]:.3f}, "
                f"threshold={DISTANCE_THRESHOLD_DEFAULT})"
            )
            return False, (
                "I couldn't find information in my knowledge base that closely matches your question. "
                "Could you try rephrasing it or asking about a different topic? "
                "This helps me provide more accurate answers based on the documents I have access to."
            )

        return True, ""

    def add_documents(self, documents: list[str] | list[dict[str, str]]) -> None:
        """
        # ...existing code...
        """
        self.vector_db.add_documents(documents)

    def invoke(self, query: str, n_results: int = RETRIEVAL_K_DEFAULT) -> str:
        """
        Query the RAG assistant with query classification and type-based handling.

        Args:
            query: User's input
            n_results: Number of relevant chunks to retrieve

        Returns:
            String answer from the LLM based on retrieved context and query type
        """
        # Classify query (priority order: unsafe > meta > document > vague > regular)
        query_type = self._classify_query(query)
        logger.debug(f"Query classified as: {query_type.value}")

        # Handle unsafe queries first (hard block before any processing)
        if query_type == QueryType.UNSAFE:
            logger.warning(f"Unsafe query blocked: {query}")
            return "I can't assist with that query. Please ask about topics covered in my knowledge base."

        try:
            search_results = self.vector_db.search(query=query, n_results=n_results)
        except ValueError as e:
            # Handle configuration errors (e.g., embedding dimension mismatch)
            # without exposing technical details to the user
            logger.error(f"Configuration error during search: {e}")
            return (
                "I'm unable to search the documents at the moment due to a system "
                "configuration issue. Please try again later or contact support."
            )

        # Extract documents from search results
        # Documents are returned as nested lists, so flatten them
        documents = search_results.get("documents", [])
        distances = search_results.get("distances", [])
        if documents and isinstance(documents[0], list):
            # Flatten nested list of documents
            flat_docs = [doc for doc_list in documents for doc in doc_list]
        else:
            flat_docs = documents

        # Flatten distances if they are nested lists
        flat_distances = []
        if distances:
            if isinstance(distances[0], list):
                flat_distances = [dist for dist_list in distances for dist in dist_list]
            else:
                flat_distances = distances

        # Validate similarity and log results
        is_valid, rejection_message = self._validate_and_log_similarity(
            query_type, flat_docs, flat_distances
        )
        if not is_valid:
            return rejection_message

        context = "\n".join(flat_docs) if flat_docs else ""

        try:
            response = self.chain.invoke({"context": context, "question": query})
            logger.debug(f"Response generated for {query_type.value} question")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Error invoking chain: {e}")
            return (
                "I encountered an error while processing your question. "
                "Please try again."
            )

        # Save conversation to memory manager
        self.memory_manager.add_message(input_text=query, output_text=response)

        return response

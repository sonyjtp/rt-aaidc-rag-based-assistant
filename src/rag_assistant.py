"""RAG-based AI assistant using ChromaDB and multiple LLM providers."""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import DISTANCE_THRESHOLD, RETRIEVAL_K
from llm_utils import initialize_llm
from logger import logger
from memory_manager import MemoryManager
from prompt_builder import build_system_prompts
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

        # Initialize LLM
        self.llm = initialize_llm()
        logger.info(f"LLM: {self.llm.model_name}")

        # Initialize vector database for document retrieval, chunking, and embeddings
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
                (
                    "human",
                    """Previous conversation context:
{chat_history}

Context from documents:
{context}

Question: {question}""",
                ),
            ]
        )
        logger.info("Prompt template for RAG Assistant created from system prompts.")
        self.chain = self.prompt_template | self.llm | StrOutputParser()
        logger.info("Function chain with prompt template, LLM, and parser built.")

    def add_documents(self, documents: list[str] | list[dict[str, str]]) -> None:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of documents
        """
        self.vector_db.add_documents(documents)

    def _flatten_search_results(
        self, search_results: dict
    ) -> tuple[list[str], list[float]]:
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

    def _log_search_results(
        self, flat_docs: list[str], flat_distances: list[float]
    ) -> None:
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
                f"Retrieved Result {i+1}: {doc}, similarity_score: {sim_score}"
            )

    def invoke(self, query: str, n_results: int = RETRIEVAL_K) -> str:
        """
        Query the RAG assistant.

        Args:
            query: User's input
            n_results: Number of relevant chunks to retrieve

        Returns:
            String answer from the LLM based on retrieved context
        """

        try:
            search_results = self.vector_db.search(
                query=query, n_results=n_results, maximum_distance=DISTANCE_THRESHOLD
            )
        except ValueError as e:
            # Handle configuration errors (e.g., embedding dimension mismatch)
            # without exposing technical details to the user
            logger.error(f"Configuration error during search: {e}")
            return (
                "I'm unable to search the documents at the moment due to a system "
                "configuration issue. Please try again later or contact support."
            )

        # Flatten search results using the helper method
        flat_docs, flat_distances = self._flatten_search_results(search_results)

        # Log search results for debugging
        self._log_search_results(flat_docs, flat_distances)

        # Use documents only if they meet similarity threshold
        # Otherwise pass empty context and let system prompts guide the response
        # System prompts already handle: meta-questions, greetings, out-of-scope questions
        context = (
            "\n".join(flat_docs)
            if flat_docs
            and (not flat_distances or flat_distances[0] <= DISTANCE_THRESHOLD)
            else ""
        )

        try:
            # Prepare chain inputs with context, question, and memory
            memory_vars = self.memory_manager.get_memory_variables()
            chain_inputs = {
                "context": context,
                "question": query,
                "chat_history": memory_vars.get("chat_history", "")
                or "No previous conversation context.",
            }

            response = self.chain.invoke(chain_inputs)
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Error invoking chain: {e}")
            response = (
                "I encountered an error while processing your question. "
                "Please try again."
            )

        # Debug: Log the final query result
        logger.debug(f"Query: {query}")
        logger.debug(f"Query Result: {response}")

        # Save conversation to memory manager
        self.memory_manager.add_message(input_text=query, output_text=response)

        return response

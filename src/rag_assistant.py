"""RAG-based AI assistant using ChromaDB and multiple LLM providers."""
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from config import (
    DISTANCE_THRESHOLD_DEFAULT,
    META_QUESTION_KEYWORDS,
    RETRIEVAL_K_DEFAULT,
)
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
        except (
            AttributeError,
            ValueError,
        ) as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Error loading reasoning strategy: {e}")
            self.reasoning_strategy = None

        self._build_chain()

    def _build_chain(self):
        """Build the prompt template and LLM chain."""
        system_prompts = build_system_prompts()
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "\n".join(system_prompts)),
                ("human", "Context from documents:\n{context}\n\nQuestion: {question}"),
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

    def invoke(self, query: str, n_results: int = RETRIEVAL_K_DEFAULT) -> str:
        """
        Query the RAG assistant.

        Args:
            query: User's input
            n_results: Number of relevant chunks to retrieve

        Returns:
            String answer from the LLM based on retrieved context
        """
        is_meta_question = any(
            keyword.lower() in query.lower() for keyword in META_QUESTION_KEYWORDS
        )

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

        # For meta-questions, use results even with lower similarity
        # For regular questions, require higher similarity (distance <= threshold)
        if not is_meta_question and (
            not flat_docs or (distances and distances[0] > DISTANCE_THRESHOLD_DEFAULT)
        ):
            return (
                "I couldn't find information in my knowledge base that closely matches your question. "
                "Could you try rephrasing it or asking about a different topic? "
                "This helps me provide more accurate answers based on the documents I have access to."
            )

        context = "\n".join(flat_docs) if flat_docs else ""

        try:
            response = self.chain.invoke({"context": context, "question": query})
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"Error invoking chain: {e}")
            return (
                "I encountered an error while processing your question. "
                "Please try again."
            )

        # Save conversation to memory manager
        self.memory_manager.add_message(input_text=query, output_text=response)

        return response

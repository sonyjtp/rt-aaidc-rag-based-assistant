"""Manage vector search operations and context validation using LLM."""

from typing import Dict, List, Tuple

from error_messages import (
    SEARCH_MANAGER_INITIALIZATION_FAILED,
    VECTOR_DB_INITIALIZATION_FAILED,
)
from logger import logger
from vectordb import VectorDB


class SearchManager:
    """Encapsulate vector search operations and context validation.

    Features:
    1. Add documents to VectorDB
    2. Search for relevant documents
    3. Flatten and log search results
    4. Validate context relevance using LLM

    """

    def __init__(self, llm) -> None:
        self.llm = llm
        self._initialize_vector_db()

        # Fail fast if required components are not available
        if self.llm is None or self.vector_db is None:
            raise RuntimeError(SEARCH_MANAGER_INITIALIZATION_FAILED)

    def _initialize_vector_db(self) -> None:
        """Initialize the vector database with error handling."""
        try:
            self.vector_db = VectorDB()
            logger.info("Vector database initialized.")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error(f"{VECTOR_DB_INITIALIZATION_FAILED}: {e}")
            self.vector_db = None

    def add_documents(self, documents: list[str] | list[dict[str, str]]) -> None:
        """
        Delegate adding documents to the underlying VectorDB instance.

        Raises:
            RuntimeError: if the VectorDB is not initialized.
        """
        if self.vector_db is None:
            raise RuntimeError("SearchManager: VectorDB is not initialized")

        try:
            return self.vector_db.add_documents(documents)
        except Exception as e:
            logger.error(f"Error adding documents to VectorDB: {e}")
            raise

    def search(self, query: str, n_results: int, maximum_distance: float) -> Dict:
        """Delegate to VectorDB.search and surface errors to caller."""
        try:
            return self.vector_db.search(query=query, n_results=n_results, maximum_distance=maximum_distance)
        except Exception as e:  # keep behavior similar to original
            logger.error(f"Error during document search: {e}")
            raise

    @staticmethod
    def flatten_search_results(search_results: Dict) -> Tuple[List[str], List[float]]:
        """Flatten nested 'documents' and 'distances' lists into simple lists."""
        documents = search_results.get("documents", [])
        distances = search_results.get("distances", [])

        if documents and isinstance(documents[0], list):
            flat_docs = [doc for doc_list in documents for doc in doc_list]
        else:
            flat_docs = documents

        if distances and isinstance(distances[0], list):
            flat_distances = [dist for dist_list in distances for dist in dist_list]
        else:
            flat_distances = distances

        return flat_docs, flat_distances

    @staticmethod
    def log_search_results(flat_docs: List[str], flat_distances: List[float]) -> None:
        """Log truncated documents and similarity scores for debugging."""
        truncated_docs = [doc[:50] + "..." if isinstance(doc, str) and len(doc) > 50 else doc for doc in flat_docs]
        similarity_scores = [1 - dist for dist in flat_distances]
        for i, (doc, sim_score) in enumerate(zip(truncated_docs, similarity_scores)):
            logger.debug(f"Retrieved Result {i + 1}: {doc}, similarity_score: {sim_score}")

    def is_context_relevant_to_query(self, query: str, context: str) -> bool:
        """
        Use the LLM (if available) to validate whether the retrieved context
        directly addresses the user's query. On errors or missing LLM, be
        conservative (assume relevant) to avoid false negatives.
        """
        if not context or not context.strip():
            return False

        if not getattr(self, "llm", None):
            logger.warning("LLM not available for context validation, assuming relevant")
            return True

        try:
            validation_prompt = (
                "Does the following context contain information that directly "
                "addresses this question? Answer with only 'YES' or 'NO'.\n\n"
                f"Question: {query}\n\n"
                f"Context: {context[:500]}"
            )
            result = self.llm.invoke(validation_prompt)
            # Prefer a `content` attribute, fallback to str()
            result_str = getattr(result, "content", None)
            if result_str is None:
                result_str = str(result)
            is_relevant = "YES" in result_str.upper()
            logger.debug(f"Context relevance validation: {is_relevant}")
            return is_relevant
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(f"Error validating context relevance: {e}")
            return True

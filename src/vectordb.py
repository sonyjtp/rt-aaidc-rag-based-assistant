"""Vector database wrapper using ChromaDB with HuggingFace embeddings."""
from typing import Any, Dict

from chromadb.errors import InvalidArgumentError
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chroma_client import ChromaDBClient
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME_DEFAULT,
    TEXT_SPLITTER_SEPARATORS,
)
from embeddings import initialize_embedding_model
from logger import logger
from str_utils import format_tags


class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    Handles document chunking, deduplication, insertion, and similarity search.
    """

    def __init__(self):
        """Initialize the vector database using configuration from config.py.
        Steps:
        1. Initialize ChromaDB client and get or create collection
        2. Initialize embedding model for document embeddings
        3. Initialize text splitter for chunking documents
        """

        # Initialize ChromaDB client and get or create collection
        self.collection = ChromaDBClient().get_or_create_collection(
            COLLECTION_NAME_DEFAULT
        )
        logger.info(f"Vector database collection {self.collection.name} ready for use")

        # Initialize embedding model for document embeddings
        self.embedding_model = initialize_embedding_model()
        logger.info(f"Embedding model: {self.embedding_model.model_name}")

        # Initialize text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=TEXT_SPLITTER_SEPARATORS,
        )

    def _chunk_documents(
        self, documents: list[str] | list[dict[str, str]]
    ) -> list[tuple[str, dict[str, str]]]:
        """
        Chunk documents into smaller pieces. Each chunk is paired with metadata.

        Args:
            documents: List of documents (strings or dicts with 'content',
                      'title', 'filename', and 'tags')

        Returns:
            List of tuples containing chunk text and metadata dictionary with
            title, filename, and tags
        """
        docs = documents if isinstance(documents, list) else [documents]
        chunks_with_metadata = [
            (
                chunk,
                {
                    "title": doc.get("title", "") if isinstance(doc, dict) else "",
                    "filename": (
                        doc.get("filename", "") if isinstance(doc, dict) else ""
                    ),
                    "tags": (
                        format_tags(doc.get("tags", []))
                        if isinstance(doc, dict)
                        else ""
                    ),
                },
            )
            for doc in docs
            for chunk in self.text_splitter.split_text(
                doc["content"] if isinstance(doc, dict) else doc
            )
            if chunk.strip()
            != (doc.get("title", "").strip() if isinstance(doc, dict) else "")
        ]
        return chunks_with_metadata

    def add_documents(self, documents: list[str] | list[dict[str, str]]) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of documents
        """
        chunks_with_metadata = self._chunk_documents(documents=documents)
        logger.info(
            f"Created {len(chunks_with_metadata)} chunks from {len(documents)} documents"
        )
        self._insert_chunks_into_db(chunks_with_metadata)

    def _insert_chunks_into_db(self, chunks: list[tuple[str, dict]]) -> None:
        """Insert deduplicated chunks into the vector database."""
        deduplicated_chunks = self._filter_duplicate_chunks(chunks)

        if deduplicated_chunks:
            if len(deduplicated_chunks) < len(chunks):
                logger.info(f"Deduplicated to {len(deduplicated_chunks)} chunks")
            next_id = self.collection.count()
            keys = [
                f"document_{idx}"
                for idx in range(next_id, next_id + len(deduplicated_chunks))
            ]
            chunk_texts = [
                chunk.strip().lstrip(".,;:!? ") for chunk, _ in deduplicated_chunks
            ]
            metadata = [metadata for _, metadata in deduplicated_chunks]
            embeddings = self.embedding_model.embed_documents(chunk_texts)
            self.collection.add(
                ids=keys,
                embeddings=embeddings,
                documents=chunk_texts,
                metadatas=metadata,
            )
            logger.info(
                f"Added {len(deduplicated_chunks)} chunks to the vector database."
            )
        else:
            logger.warning("No new chunks to add (all are duplicates)")

    def _filter_duplicate_chunks(
        self, chunks: list[tuple[str, dict]]
    ) -> list[tuple[str, dict]]:
        """
        Filter out duplicate chunks.
        Removes chunks that already exist in the database AND duplicates within the current batch.

        Args:
            chunks: List of tuples containing chunk text and metadata

        Returns:
            List of new chunks that don't already exist in the database or batch
        """
        # Get existing documents from the database
        existing_docs = self.collection.get()
        existing_texts = set(existing_docs.get("documents", []))

        # Filter out chunks that already exist in database
        # Normalize text the same way as in _insert_chunks_into_db
        new_chunks = [
            chunk
            for chunk in chunks
            if chunk[0].strip().lstrip(".,;:!? ") not in existing_texts
        ]

        # Also remove duplicates within the current batch
        seen = set()
        final_chunks = []
        for chunk_text, metadata in new_chunks:
            normalized_text = chunk_text.strip().lstrip(".,;:!? ")
            if normalized_text not in seen:
                seen.add(normalized_text)
                final_chunks.append((chunk_text, metadata))

        return final_chunks

    @staticmethod
    def _extract_search_results(results: dict) -> tuple:
        """Extract search result components from ChromaDB response."""
        return (
            results.get("documents", [[]])[0] if results.get("documents") else [],
            results.get("metadatas", [[]])[0] if results.get("metadatas") else [],
            results.get("distances", [[]])[0] if results.get("distances") else [],
            results.get("ids", [[]])[0] if results.get("ids") else [],
        )

    @staticmethod
    def _filter_search_results(
        documents: list,
        metadatas: list,
        distances: list,
        ids: list,
        maximum_distance: float,
    ) -> tuple:
        """
        Filter search results based on distance threshold.

        Args:
            documents: List of retrieved documents
            metadatas: List of metadata for documents
            distances: List of distance scores
            ids: List of document IDs
            maximum_distance: Maximum distance threshold

        Returns:
            Tuple of (documents, metadatas, distances, ids) filtered by threshold
        """
        filtered_results = [
            (doc, meta, dist, doc_id)
            for doc, meta, dist, doc_id in zip(documents, metadatas, distances, ids)
            if dist <= maximum_distance
        ]

        if filtered_results:
            return tuple(map(list, zip(*filtered_results)))
        return [], [], [], []

    def search(
        self, query: str, n_results: int = 5, maximum_distance: float = 0.35
    ) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database.

        Args:
            query: Search query
            n_results: Number of results to return
            maximum_distance: Maximum distance threshold for filtering results

        Returns:
            Dictionary containing search results with keys: 'documents',
            'metadatas', 'distances', 'ids'
        """
        query_embedding = self.embedding_model.embed_query(query)
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding], n_results=n_results
            )
        except InvalidArgumentError as e:
            error_msg = str(e)
            if "expecting embedding with dimension" in error_msg:
                logger.error(
                    f"Embedding dimension mismatch: {error_msg}. "
                    "Please update VECTOR_DB_EMBEDDING_MODEL in config.py "
                    "to match your collection's embedding model."
                )
                return {
                    "documents": [],
                    "metadatas": [],
                    "distances": [],
                    "ids": [],
                }
            raise

        documents, metadatas, distances, ids = self._extract_search_results(results)
        documents, metadatas, distances, ids = self._filter_search_results(
            documents, metadatas, distances, ids, maximum_distance
        )

        logger.debug(f"Search query: {query}")
        logger.debug(f"Retrieved {len(documents)} results")

        for doc_id, distance, metadata in zip(ids, distances, metadatas):
            logger.debug(
                f"  {doc_id} | Similarity: {1 - distance:.4f} | "
                f"Title: {metadata.get('title', 'N/A')} | "
                f"File: {metadata.get('filename', 'N/A')}"
            )

        return {
            "documents": documents,
            "metadatas": metadatas,
            "distances": distances,
            "ids": ids,
        }

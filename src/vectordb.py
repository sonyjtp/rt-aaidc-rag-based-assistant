"""Vector database wrapper using ChromaDB."""
from typing import Any, Dict

from chromadb.errors import InvalidArgumentError
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app_constants import COLLECTION_NAME_DEFAULT, PUNCTUATION_CHARS
from chroma_client import ChromaDBClient
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    TEXT_SPLITTER_SEPARATORS,
    VECTOR_DB_BATCH_SIZE_LIMIT,
)
from error_messages import DOCUMENTS_MISSING
from log_manager import logger
from str_utils import format_tags


class VectorDB:
    """
    A simple vector database wrapper using ChromaDB.
    Handles document chunking, deduplication, insertion, and similarity search.
    """

    def __init__(self):
        """Initialize the vector database using configuration from config.py.
        Steps:
        1. Initialize ChromaDB client and get or create collection
        2. Initialize text splitter for chunking documents

        Functions:
        1. add_documents: Chunk and insert documents to the vector database
        2. search: Search for similar documents in the vector database
        3. standardize_document: Standardize a document to a dict format
        4. _chunk_documents: Chunk documents into smaller pieces with metadata
        5. _insert_chunks_into_db: Insert deduplicated chunks into the vector database
        6. _filter_duplicate_chunks: Filter out duplicate chunks at the chunk level
        7. _extract_search_results: Extract search result components from ChromaDB response
        8. _filter_search_results: Filter search results based on distance threshold
        """

        # Initialize ChromaDB client and get or create collection
        self.collection = ChromaDBClient().get_or_create_collection(COLLECTION_NAME_DEFAULT)
        logger.debug(f"Vector database collection {self.collection.name} created/loaded.")

        # Initialize text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=TEXT_SPLITTER_SEPARATORS,
        )
        logger.debug(f" Text splitter initialized with chunk size = {CHUNK_SIZE} and overlap = {CHUNK_OVERLAP}.")

    def add_documents(self, documents: list[str] | list[dict[str, str]]) -> None:
        """
        Chunk and insert documents to the vector database.

        Deduplication is done at the chunk level only. All documents are processed
        and chunked, but duplicate chunks are filtered before insertion.

        Args:
            documents: List of documents (strings or dicts with 'content',
                        'title', 'filename', and 'tags')
        """
        if not documents:
            logger.error("No documents to add.")
            raise ValueError(DOCUMENTS_MISSING)

        # Chunk all documents (no document-level filtering)
        chunks_with_metadata = self._chunk_documents(documents=documents)
        logger.info(f"Created {len(chunks_with_metadata)} chunks from {len(documents)} documents")
        self._insert_chunks_into_db(chunks_with_metadata)

    def search(self, query: str, n_results: int = 5, maximum_distance: float = 0.35) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database using a query string.

        Args:
            query: Search query
            n_results: Number of results to return
            maximum_distance: Maximum distance threshold for filtering results

        Returns:
            Dictionary containing search results with keys: 'documents',
            'metadatas', 'distances', 'ids'
        """
        try:
            results = self.collection.query(query_texts=[query], n_results=n_results)
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

        # Extract and filter results based on maximum distance
        documents, metadatas, distances, ids = self._extract_and_filter_search_results(results, maximum_distance)

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

    @staticmethod
    def standardize_document(doc):
        """Standardize a document to a dict with 'content', 'title', 'filename', and 'tags'."""

        if isinstance(doc, dict):
            return {
                "content": doc.get("content", ""),
                "title": doc.get("title", ""),
                "filename": doc.get("filename", ""),
                "tags": format_tags(doc.get("tags", [])),
            }
        return {
            "content": doc,
            "title": "",
            "filename": "",
            "tags": "",
        }

    def _chunk_documents(self, documents: list[str] | list[dict[str, str]]) -> list[tuple[str, dict[str, str]]]:
        """Chunk documents into smaller pieces. Each chunk is paired with metadata.

        Args:
            documents: List of documents (strings or dicts with 'content',
                        'title', 'filename', and 'tags')
        Returns:
            List of tuples containing chunk text and metadata
        """
        result = []
        for doc in documents:
            # Standardize document format to dict of content, title, filename, tags
            normalized = self.standardize_document(doc)
            for chunk in self.text_splitter.split_text(normalized["content"]):
                # Avoid adding empty chunks or chunks identical to the title which are not useful
                if chunk.strip() and chunk.strip() != normalized["title"].strip():
                    result.append(
                        (
                            chunk,
                            {
                                "title": normalized["title"],
                                "filename": normalized["filename"],
                                "tags": normalized["tags"],
                            },
                        )
                    )
        return result

    def _insert_chunks_into_db(self, chunks: list[tuple[str, dict]]) -> None:
        """Deduplicate and insert chunks into the vector database.

        Args:
            chunks: List of tuples containing chunk text and metadata
        """

        # Deduplicate chunks before insertion
        deduplicated_chunks = self._filter_duplicate_chunks(chunks)

        if deduplicated_chunks:
            if len(deduplicated_chunks) < len(chunks):
                logger.info(f"Deduplicated to {len(deduplicated_chunks)} chunks")
            next_id = self.collection.count()
            # Insert in batches
            for i in range(0, len(deduplicated_chunks), VECTOR_DB_BATCH_SIZE_LIMIT):
                batch_chunks = deduplicated_chunks[i : i + VECTOR_DB_BATCH_SIZE_LIMIT]
                batch_start_id = next_id + i

                keys = [f"document_{idx}" for idx in range(batch_start_id, batch_start_id + len(batch_chunks))]
                # Chunks are already normalized by _filter_duplicate_chunks
                chunk_texts = [chunk for (chunk, _) in batch_chunks]
                metadata = [meta for _, meta in batch_chunks]

                self.collection.add(ids=keys, documents=chunk_texts, metadatas=metadata)
                logger.debug(f"Added batch with {len(batch_chunks)} chunks")
            logger.info(f"Added {len(deduplicated_chunks)} chunks to the vector database.")
        else:
            logger.info("No new chunks to add (all are duplicates)")

    def _filter_duplicate_chunks(self, chunks: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
        """
        Filter out duplicate chunks

        Steps:
        1. Retrieve all existing documents from the database in batches
        2. Normalize chunk text for comparison
        3. Filter out chunks that already exist in the database or are duplicates within the batch

        Args:
            chunks: List of tuples containing chunk text and metadata

        Returns:
            List of new chunks (with normalized text) that don't already exist in the database or batch
        """
        # Get ALL existing documents from the database in batches
        # ChromaDB has a quota limit of 300 per request, so we paginate
        existing_texts = set()
        offset = 0

        while True:
            try:
                existing_docs = self.collection.get(offset=offset, limit=VECTOR_DB_BATCH_SIZE_LIMIT)
                docs = existing_docs.get("documents", [])

                if not docs:
                    break  # No more documents

                existing_texts.update(docs)
                offset += VECTOR_DB_BATCH_SIZE_LIMIT
                logger.debug(f"Fetched batch at offset {offset - VECTOR_DB_BATCH_SIZE_LIMIT}: {len(docs)} docs")
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.warning(f"Error fetching documents batch: {e}")
                break

        logger.debug(f"Existing chunks in DB: {len(existing_texts)}")

        # Filter out chunks that already exist in database and remove duplicates within the batch
        seen = set(existing_texts)
        final_chunks = []
        duplicates_found = 0

        for chunk_text, metadata in chunks:
            # Normalize before comparison
            normalized_text = chunk_text.strip().lstrip(PUNCTUATION_CHARS)
            if normalized_text not in seen:
                seen.add(normalized_text)
                # Store the normalized version in the result
                final_chunks.append((normalized_text, metadata))
            else:
                duplicates_found += 1

        logger.debug(f"Duplicates found: {duplicates_found}, New chunks: {len(final_chunks)}")
        return final_chunks

    @staticmethod
    def _extract_and_filter_search_results(results: dict, maximum_distance: float) -> tuple:
        """
        Extract search result components from ChromaDB response and filter by distance threshold.

        Args:
            results: ChromaDB query response dictionary
            maximum_distance: Maximum distance threshold for filtering results

        Returns:
            Tuple of (documents, metadatas, distances, ids) filtered by threshold
        """
        # Extract raw results from ChromaDB response
        documents = results.get("documents", [[]])[0] if results.get("documents") else []
        metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
        distances = results.get("distances", [[]])[0] if results.get("distances") else []
        ids = results.get("ids", [[]])[0] if results.get("ids") else []

        # Filter by distance threshold
        filtered_results = [
            (doc, meta, dist, doc_id)
            for doc, meta, dist, doc_id in zip(documents, metadatas, distances, ids)
            if dist <= maximum_distance
        ]

        if filtered_results:
            return tuple(map(list, zip(*filtered_results)))
        return [], [], [], []

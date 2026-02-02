"""Vector database wrapper using ChromaDB."""
from typing import Any, Dict

from chromadb.errors import InvalidArgumentError
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chroma_client import ChromaDBClient
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME_DEFAULT,
    PUNCTUATION_CHARS,
    TEXT_SPLITTER_SEPARATORS,
    VECTOR_DB_BATCH_SIZE_LIMIT,
)
from logger import logger
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
        """

        # Initialize ChromaDB client and get or create collection
        self.collection = ChromaDBClient().get_or_create_collection(COLLECTION_NAME_DEFAULT)
        logger.info(f"Vector database collection: {self.collection.name}")

        # Initialize text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=TEXT_SPLITTER_SEPARATORS,
        )

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
            logger.info("No documents to add.")
            return

        # Chunk all documents (no document-level filtering)
        chunks_with_metadata = self._chunk_documents(documents=documents)
        logger.info(f"Created {len(chunks_with_metadata)} chunks from {len(documents)} documents")
        self._insert_chunks_into_db(chunks_with_metadata)

    def search(self, query: str, n_results: int = 5, maximum_distance: float = 0.35) -> Dict[str, Any]:
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

        documents, metadatas, distances, ids = self._extract_search_results(results)
        documents, metadatas, distances, ids = self._filter_search_results(
            documents, metadatas, distances, ids, maximum_distance
        )

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

    def _filter_existing_documents(
        self, documents: list[str] | list[dict[str, str]]
    ) -> list[str] | list[dict[str, str]]:
        """
        Filter out documents that already exist in the database.

        A document is considered to already exist if its filename is found in
        the database metadata. We use filename as the unique identifier since
        it's stable across runs, unlike title which is extracted from content.

        Args:
            documents: List of documents to filter

        Returns:
            List of documents that don't already exist in the database
        """
        try:
            # Get all existing documents from the database
            existing_docs = self.collection.get()
            existing_metadatas = existing_docs.get("metadatas", [])

            # Build a set of existing filenames (stable, unique identifier)
            existing_filenames = set()
            for metadata in existing_metadatas:
                if metadata and metadata.get("filename"):
                    existing_filenames.add(metadata["filename"])

            if not existing_filenames:
                # No existing documents, all are new
                return documents

            # Filter documents based on whether their filename already exists
            new_documents = []
            for doc in documents:
                normalized = self.standardize_document(doc)
                filename = normalized.get("filename", "")

                # Only add document if:
                # 1. It has a filename AND doesn't already exist, OR
                # 2. It's a string document without filename (raw content, always new)
                if isinstance(doc, str):
                    # String documents are always new (no filename to check)
                    new_documents.append(doc)
                elif filename and filename not in existing_filenames:
                    # Dict document with new filename
                    new_documents.append(doc)
                # else: Skip dict documents that already exist in database

            return new_documents
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning(f"Error filtering existing documents: {e}. Adding all documents.")
            return documents

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
        """Insert deduplicated chunks into the vector database.

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
            logger.warning("No new chunks to add (all are duplicates)")

    def _filter_duplicate_chunks(self, chunks: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
        """
        Filter out duplicate chunks.
        Removes chunks that already exist in the database AND duplicates within the current batch.

        Normalization: All chunks are normalized (stripped and left-stripped of punctuation)
        before comparison with existing database chunks.

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
        if existing_texts:
            logger.debug(f"Sample existing chunk: {list(existing_texts)[0][:100]}")

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

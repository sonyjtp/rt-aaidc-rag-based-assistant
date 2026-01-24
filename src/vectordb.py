from typing import Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from chroma_client import ChromaDBClient
from config import CHUNK_SIZE_DEFAULT, CHUNK_OVERLAP_DEFAULT, COLLECTION_NAME_DEFAULT
from embeddings import initialize_embedding_model
from logger import logger


class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

    def __init__(
            self,
            collection_name: str = COLLECTION_NAME_DEFAULT,
            chunk_size: int = CHUNK_SIZE_DEFAULT,
            chunk_overlap: int = CHUNK_OVERLAP_DEFAULT,
    ):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the ChromaDB collection
            chunk_size: Approximate number of characters per chunk for text splitting
            chunk_overlap: Number of characters to overlap between chunks
        """

        # Initialize ChromaDB client and get or create collection
        self.collection = ChromaDBClient().get_or_create_collection(collection_name)
        logger.info(f"Vector database collection {self.collection.name} ready for use")

        self.embedding_model = initialize_embedding_model()
        logger.info(f"Embedding model: {self.embedding_model.model_name}")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def _chunk_documents(self, documents: list[str] | list[dict[str, str]]) -> list[tuple[str, dict[str, str]]]:
        """
        Chunk documents into smaller pieces. Each chunk is paired with its metadata.
        Args:
            documents: List of documents (strings or dicts with 'content', 'title', 'filename', and 'tags')

        Returns:
            List of tuples containing chunk text and metadata dictionary with title, filename, and tags
        """
        docs = documents if isinstance(documents, list) else [documents]
        chunks_with_metadata = [
            (chunk, {
                'title': doc.get('title', ''),
                'filename': doc.get('filename', ''),
                'tags': doc.get('tags', '')
            } if isinstance(doc, dict) else {
                'title': '',
                'filename': '',
                'tags': ''
            })
            for doc in docs
            for chunk in self.text_splitter.split_text(
                doc['content'] if isinstance(doc, dict) else doc
            )
            if chunk.strip() != doc.get('title', '').strip()
        ]
        return chunks_with_metadata

    def add_documents(self, documents: list[str] | list[dict[str, str]]) -> None:
        """
        Add documents to the vector database.

        Args:
            documents: List of documents
        """
        chunks_with_metadata = self._chunk_documents(documents=documents)
        logger.info(f"Created {len(chunks_with_metadata)} chunks from {len(documents)} documents")
        self._insert_chunks_into_db(chunks_with_metadata)


    def _insert_chunks_into_db(self, chunks: list[tuple[str, dict]]):
        """Insert deduplicated chunks into the vector database."""
        deduplicated_chunks = self._filter_duplicate_chunks(chunks)

        if deduplicated_chunks:
            if len(deduplicated_chunks) < len(chunks):
                logger.info(f"Deduplicated to {len(deduplicated_chunks)} chunks")
            next_id = self.collection.count()
            keys = [f"document_{idx}" for idx in range(next_id, next_id + len(deduplicated_chunks))]
            chunk_texts = [chunk.strip().lstrip('.,;:!? ') for chunk, _ in deduplicated_chunks]
            metadata = [metadata for _, metadata in deduplicated_chunks]
            embeddings = self.embedding_model.embed_documents(chunk_texts)
            self.collection.add(
                ids=keys,
                embeddings=embeddings,
                documents=chunk_texts,
                metadatas=metadata,
            )
            logger.info(f"Added {len(deduplicated_chunks)} chunks to the vector database.")
        else:
            logger.warning("No new chunks to add (all are duplicates)")

    # ...existing code...

    def _filter_duplicate_chunks(self, chunks: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
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
            chunk for chunk in chunks
            if chunk[0].strip().lstrip('.,;:!? ') not in existing_texts
        ]

        # Also remove duplicates within the current batch
        seen = set()
        final_chunks = []
        for chunk_text, metadata in new_chunks:
            normalized_text = chunk_text.strip().lstrip('.,;:!? ')
            if normalized_text not in seen:
                seen.add(normalized_text)
                final_chunks.append((chunk_text, metadata))

        return final_chunks

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dictionary containing search results with keys: 'documents', 'metadatas', 'distances', 'ids'
        """
        # Generate embedding for the query
        query_embedding = self.embedding_model.embed_query(query)

        # Query the ChromaDB collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        # Extract documents from results
        documents = results.get("documents", [[]])[0] if results.get("documents") else []
        metadatas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
        distances = results.get("distances", [[]])[0] if results.get("distances") else []
        ids = results.get("ids", [[]])[0] if results.get("ids") else []

        return {
            "documents": documents,
            "metadatas": metadatas,
            "distances": distances,
            "ids": ids,
        }

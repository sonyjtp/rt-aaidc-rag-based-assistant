import os
from typing import Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from chroma_client import ChromaDBClient
from config import CHUNK_SIZE_DEFAULT, CHUNK_OVERLAP_DEFAULT
from embeddings import initialize_embedding_model


class VectorDB:
    """
    A simple vector database wrapper using ChromaDB with HuggingFace embeddings.
    """

    def __init__(
            self,
            collection_name: str = None,
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
        self.collection = ChromaDBClient().get_or_create_collection(
            collection_name or os.getenv(
                "CHROMA_COLLECTION_NAME", "rag_documents"
            )
        )
        print(f"✓ Vector database collection: {self.collection.name} ready for use")

        # Initialize embedding model with device detection
        self.embedding_model = initialize_embedding_model()
        print(f"✓ Embedding model {self.embedding_model.model_name} initialized")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk_documents(self, documents: list[str] | list[dict[str, str]]) -> list[tuple[str, dict[str, str]]]:
        """
        Chunk documents into smaller pieces. Each chunk is paired with its metadata.
        Args:
            documents: List of documents (strings or dicts with 'content', 'title', and 'filename')

        Returns:
            List of tuples containing chunk text and metadata dictionary with title and filename
        """
        docs = documents if isinstance(documents, list) else [documents]
        chunks_with_metadata = [
            (chunk, {
                'title': doc.get('title', ''),
                'filename': doc.get('filename', '')
            } if isinstance(doc, dict) else {
                'title': '',
                'filename': ''
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
        chunks_with_metadata = self.chunk_documents(documents=documents)
        print(f"✓ Created {len(chunks_with_metadata)} chunks from {len(documents)} documents")
        self.insert_chunks_into_db(chunks_with_metadata)

        # TODO: Implement document ingestion logic
        # HINT: Loop through each document in the documents list
        # HINT: Extract 'content' and 'metadata' from each document dict
        # HINT: Use self.chunk_text() to split each document into chunks
        # HINT: Create unique IDs for each chunk (e.g., "doc_0_chunk_0")
        # HINT: Use self.embedding_model.encode() to create embeddings for all chunks
        # HINT: Store the embeddings, documents, metadata, and IDs in your vector database
        # HINT: Print progress messages to inform the user

        # Your implementation here
        print(f"✓ Chunks added to vector database")

    def insert_chunks_into_db(self, chunks: list[tuple[str, dict]]):
        next_id = self.collection.count()
        keys = [f"document_{idx}" for idx in range(next_id, next_id + len(chunks))]
        chunk_texts = [chunk for chunk, _ in chunks]
        metadata = [metadata for _, metadata in chunks]
        embeddings = self.embedding_model.embed_documents(chunk_texts)
        self.collection.add(
            ids=keys,
            embeddings=embeddings,
            documents=chunk_texts,
            metadatas=metadata,
        )
        next_id += len(chunks)
        print(f"✓ Added {len(chunks)} chunks and their metadata to the vector database.")

    def search(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for similar documents in the vector database.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            Dictionary containing search results with keys: 'documents', 'metadatas', 'distances', 'ids'
        """
        # TODO: Implement similarity search logic
        # HINT: Use self.embedding_model.encode([query]) to create query embedding
        # HINT: Convert the embedding to appropriate format for your vector database
        # HINT: Use your vector database's search/query method with the query embedding and n_results
        # HINT: Return a dictionary with keys: 'documents', 'metadatas', 'distances', 'ids'
        # HINT: Handle the case where results might be empty

        # Your implementation here
        return {
            "documents": [],
            "metadatas": [],
            "distances": [],
            "ids": [],
        }

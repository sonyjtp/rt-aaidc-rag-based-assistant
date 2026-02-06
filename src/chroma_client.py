"""
ChromaDB client wrapper for managing vector database connections and collections.
"""

import os
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from app_constants import (
    CHROMA_API_KEY_ENV,
    CHROMA_DATABASE_ENV,
    CHROMA_TENANT_ENV,
    COLLECTION_NAME_DEFAULT,
)
from config import (
    CHROMA_COLLECTION_METADATA,
    DISTANCE_THRESHOLD,
    RETRIEVAL_K,
    VECTOR_DB_EMBEDDING_MODEL,
)
from log_manager import logger


class ChromaDBClient:
    """
    Wrapper around ChromaDB CloudClient for managing vector database connections.

    Handles initialization and collection management with proper configuration from
    environment variables.
    """

    def __init__(self):
        """
        Initialize the ChromaDB client using configuration from environment variables.
        """
        # ChromaDB configuration from environment
        self.api_key = os.getenv(CHROMA_API_KEY_ENV)
        self.tenant = os.getenv(CHROMA_TENANT_ENV)
        self.database = os.getenv(CHROMA_DATABASE_ENV)
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=VECTOR_DB_EMBEDDING_MODEL
        )

        # Initialize the CloudClient
        self.client = self._initialize_client()
        logger.info("ChromaDB client initialized.")

    def _initialize_client(self) -> chromadb.api.ClientAPI:
        """
        Initialize and return a ChromaDB CloudClient using instance configuration.

        Uses instance attributes (api_key, tenant, database) for connection setup,
        ensuring this method leverages instance state.

        Returns:
            ChromaDB CloudClient instance
        """
        return chromadb.CloudClient(
            api_key=self.api_key,
            tenant=self.tenant,
            database=self.database,
        )

    def get_or_create_collection(self, collection_name: str) -> Any:
        """
        Get or create a ChromaDB collection.

        Args:
            collection_name: Name of the collection to create or retrieve

        Returns:
            ChromaDB collection instance
        """
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata=CHROMA_COLLECTION_METADATA,
            embedding_function=self.embedding_function,
        )

    def query_collection(
        self,
        query_texts: str | list[str] = "",
        where: dict = None,
        collection_name: str = COLLECTION_NAME_DEFAULT,
        n_results: int = RETRIEVAL_K,
        maximum_distance: float = DISTANCE_THRESHOLD,
    ) -> dict:
        """
        Query a ChromaDB collection.

        Args:
            query_texts: Query texts to search,
            where: Optional filtering criteria as a dictionary
            collection_name: Name of the collection to query
            n_results: Number of results to return
            maximum_distance: Maximum distance threshold for results
        Returns:
            Query results as a dictionary
        """
        collection = self.client.get_collection(name=collection_name, embedding_function=self.embedding_function)
        results = collection.query(
            query_texts=query_texts,
            where=where,
            n_results=n_results,
        )
        # Filter results based on maximum_distance
        filtered_results = {"ids": [], "distances": []}
        for idx, distance in zip(results["ids"][0], results["distances"][0]):
            if distance <= maximum_distance:
                filtered_results["ids"].append(idx)
                filtered_results["distances"].append(distance)
        return filtered_results

    def delete_collection(self, collection_name: str) -> None:
        """
        Delete a ChromaDB collection.

        Args:
            collection_name: Name of the collection to delete
        """
        self.client.delete_collection(name=collection_name)
        logger.info(f"Collection {collection_name} deleted from ChromaDB")

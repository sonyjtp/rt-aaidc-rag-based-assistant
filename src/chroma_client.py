"""
ChromaDB client wrapper for managing vector database connections and collections.
"""

import os
from typing import Any
import chromadb
from config import (
    CHROMA_API_KEY_ENV,
    CHROMA_TENANT_ENV,
    CHROMA_DATABASE_ENV,
    CHROMA_COLLECTION_METADATA,
)


class ChromaDBClient:
    """
    A wrapper around ChromaDB CloudClient for managing vector database connections and collections.
    Handles initialization and collection management with proper configuration from environment variables.
    """

    def __init__(self):
        """
        Initialize the ChromaDB client using configuration from environment variables.
        """
        # ChromaDB configuration from environment
        self.api_key = os.getenv(CHROMA_API_KEY_ENV)
        self.tenant = os.getenv(CHROMA_TENANT_ENV)
        self.database = os.getenv(CHROMA_DATABASE_ENV)

        # Initialize the CloudClient
        self.client = self._initialize_client()

    def _initialize_client(self) -> Any:
        """
        Initialize and return a ChromaDB CloudClient using instance configuration.

        Uses instance attributes (api_key, tenant, database) for connection setup,
        ensuring this method leverages instance state.

        Returns:
            ChromaDB CloudClient instance
        """
        print(f"Connecting to ChromaDB with tenant: {self.tenant}")
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
        )


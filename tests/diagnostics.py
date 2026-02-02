"""
Diagnostics script to verify ChromaDB connection and perform basic queries. Use this to
ensure that the ChromaDB instance is reachable and functioning as expected.
"""

from chroma_client import ChromaDBClient
from config import COLLECTION_NAME_DEFAULT

# Connect to ChromaDB
client = ChromaDBClient()

# Get the rag_documents collection
collection = client.get_or_create_collection(COLLECTION_NAME_DEFAULT)

# Check total document count
count = collection.count()
print(f"Total documents in collection: {count}")

# Query for sister cities documents specifically
results = collection.get(
    where={"filename": "sister_cities.txt"},
    limit=10
)
print(f"Sister cities documents found: {len(results['ids'])}")
print(f"Document IDs: {results['ids']}")

# Test semantic search with the sister cities query
search_results = client.query_collection(
    query_texts=["What are Sister Cities?"],
    where={"filename": "sister_cities.txt"},
)

print(f"Search results with filter: {search_results}")
print(f"Distances: {search_results['distances']}")

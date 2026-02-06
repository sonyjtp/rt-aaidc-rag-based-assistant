"""
Diagnostics script to verify ChromaDB connection and perform basic queries. Use this to
ensure that the ChromaDB instance is reachable and functioning as expected.
"""
from src.app_constants import COLLECTION_NAME_DEFAULT
from src.chroma_client import ChromaDBClient
from src.log_manager import logger

# Connect to ChromaDB
client = ChromaDBClient()

# Get the rag_documents collection
collection = client.get_or_create_collection(COLLECTION_NAME_DEFAULT)

# Check total document count
count = collection.count()
logger.info(f"Total documents in collection: {count}")

# Query for Indian rivers documents specifically
results = collection.get(where={"filename": "indian_rivers.txt"}, limit=10)
logger.info(f"Indian rivers documents found: {len(results['ids'])}")
logger.debug(f"Document IDs: {results['ids'][:3]}...")

# Test semantic search with a rivers query
search_results = client.query_collection(
    query_texts=["What are the major rivers of India?"], where={"filename": "indian_rivers.txt"}, n_results=5
)

logger.info("Search results for 'major rivers':")
logger.info(f"IDs returned: {len(search_results['ids'][0])}")
logger.debug(f"Distances (lower is better): {search_results['distances'][0][:3]}...")

# Test with another document type
logger.info("--- Testing with Indian religions ---")
results_religions = collection.get(where={"filename": "indian_religions.txt"}, limit=5)
logger.info(f"Indian religions documents found: {len(results_religions['ids'])}")

# Test semantic search on religions
search_religions = client.query_collection(
    query_texts=["What are the major religions in India?"], where={"filename": "indian_religions.txt"}, n_results=3
)
logger.info(f"Search results for 'religions': {len(search_religions['ids'][0])} documents retrieved")

logger.info("✅ ChromaDB diagnostics completed successfully!")

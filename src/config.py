"""
Configuration for the RAG-based AI assistant.
Defines LLM providers, models, and other application settings.
"""

import os
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# ChromaDB Configuration
CHROMA_API_KEY_ENV = "CHROMA_API_KEY"
CHROMA_COLLECTION_METADATA = {
    "hnsw:space": "cosine",
    "description": "RAG document collection"
}
CHROMA_DATABASE_ENV = "CHROMA_DATABASE"
CHROMA_TENANT_ENV = "CHROMA_TENANT"

# Chunking Configuration
CHUNK_OVERLAP_DEFAULT = int(os.getenv("CHUNK_OVERLAP", "100"))
CHUNK_SIZE_DEFAULT = int(os.getenv("CHUNK_SIZE", "500"))

# Default Retrieval Configuration
DEFAULT_RETRIEVAL_K = 3

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, os.getenv("DATA_DIR", "data"))

# Error Messages
ERROR_NO_API_KEY = (
    "No valid API key found. Please set one of: "
    "OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
)

# LLM Configuration
LLM_TEMPERATURE = 0.0

# List of LLM providers in priority order
# The assistant will use the first provider with a valid API key
LLM_PROVIDERS = [
    {
        "api_key_env": "GROQ_API_KEY",
        "model_env": "GROQ_MODEL",
        "default_model": "llama-3.1-8b-instant",
        "name": "Groq",
        "class": ChatGroq,
        "api_key_param": "api_key",
    },
    {
        "api_key_env": "GOOGLE_API_KEY",
        "model_env": "GOOGLE_MODEL",
        "default_model": "gemini-2.0-flash",
        "name": "Google Gemini",
        "class": ChatGoogleGenerativeAI,
        "api_key_param": "google_api_key",
    },
    {
        "api_key_env": "OPENAI_API_KEY",
        "model_env": "OPENAI_MODEL",
        "default_model": "gpt-4o-mini",
        "name": "OpenAI",
        "class": ChatOpenAI,
        "api_key_param": "api_key",
    },
]

# Vector Database Configuration
VECTOR_DB_COLLECTION_NAME = "documents"
VECTOR_DB_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"



"""
Configuration for the RAG-based AI assistant.
Defines LLM providers, models, and other application settings.
"""

import os
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================================
# PATHS & DIRECTORIES
# ============================================================================

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, os.getenv("DATA_DIR", "data"))


# ============================================================================
# ERROR MESSAGES
# ============================================================================

ERROR_NO_API_KEY = (
    "No valid API key found. Please set one of: "
    "OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
)


# ============================================================================
# LLM & MODEL CONFIGURATION
# ============================================================================

# LLM Behavior
LLM_TEMPERATURE = 0.0

# Model Selection (defaults can be overridden via .env)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GOOGLE_MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")

# LLM Providers in priority order
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


# ============================================================================
# VECTOR DATABASE & EMBEDDINGS
# ============================================================================

# ChromaDB Configuration
CHROMA_API_KEY_ENV = "CHROMA_API_KEY"
CHROMA_DATABASE_ENV = "CHROMA_DATABASE"
CHROMA_TENANT_ENV = "CHROMA_TENANT"
CHROMA_COLLECTION_METADATA = {
    "hnsw:space": "cosine",
    "description": "RAG document collection"
}

# Vector Database Collections
VECTOR_DB_COLLECTION_NAME = "documents"
COLLECTION_NAME_DEFAULT = "rag_documents"

# Embedding Model
VECTOR_DB_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================

# Text Chunking Configuration
CHUNK_SIZE_DEFAULT = 500
CHUNK_OVERLAP_DEFAULT = 100

# Retrieval Configuration
RETRIEVAL_K_DEFAULT = 3


# ============================================================================
# ASSISTANT STRATEGIES & CONFIGURATION FILES
# ============================================================================

# Memory Strategy Configuration
MEMORY_STRATEGIES_FPATH = os.path.join(ROOT_DIR, "config", "memory_strategies.yaml")
# Options: summarization_sliding_window, summarization, conversation_buffer_memory, none
MEMORY_STRATEGY = "summarization_sliding_window"

# Reasoning Strategy Configuration
REASONING_STRATEGIES_FPATH = os.path.join(ROOT_DIR, "config", "reasoning_strategies.yaml")
# Options: rag_enhanced_reasoning, simple_few_shot, none
REASONING_STRATEGY = "rag_enhanced_reasoning"

# Prompt Configuration
PROMPT_CONFIG_FPATH = os.path.join(ROOT_DIR, "config", "prompt-config.yaml")
PROMPT_CONFIG_NAME = "rag-assistant-system-prompt-formal"


"""
Configuration for the RAG-based AI assistant.
Defines LLM providers, models, and other application settings.
"""

import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

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

ERROR_SEARCH_MANAGER_UNAVAILABLE = (
    "Unable to initialize the assistant at this time. Please refresh the page "
    "and try again. If the problem persists, please contact support."
)


# ============================================================================
# LLM & MODEL CONFIGURATION
# ============================================================================

# LLM Behavior
LLM_TEMPERATURE = 0.1  # 0.1 - 0.3 - low randomness for RAG use cases, higher values increase creativity

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
# DOCUMENT TYPES
# ============================================================================
# Could use any, a few, or all of : (".pdf", ".docx", ".txt", ".md")
DOCUMENT_TYPES = ".txt"

# ============================================================================
# VECTOR DATABASE & EMBEDDINGS
# ============================================================================

# ChromaDB Configuration
CHROMA_API_KEY_ENV = "CHROMA_API_KEY"
CHROMA_DATABASE_ENV = "CHROMA_DATABASE"
CHROMA_TENANT_ENV = "CHROMA_TENANT"
CHROMA_COLLECTION_METADATA = {
    "hnsw:space": "cosine",
    "description": "RAG document collection",
}

# Vector Database Collections
VECTOR_DB_COLLECTION_NAME = "documents"
COLLECTION_NAME_DEFAULT = "rag_documents"

# Embedding Model
VECTOR_DB_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

PUNCTUATION_CHARS = ".,;:!? "


# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================

# Text Chunking Configuration
CHUNK_SIZE = 1024  # Increased from 512 to keep concepts together
CHUNK_OVERLAP = 200  # Increased from 100 (20% overlap) for better context
TEXT_SPLITTER_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# Retrieval Configuration
RETRIEVAL_K = 20
DISTANCE_THRESHOLD = 0.4


# ============================================================================
# ASSISTANT STRATEGIES & CONFIGURATION FILES
# ============================================================================

# Memory Strategy Configuration
MEMORY_STRATEGIES_FPATH = os.path.join(ROOT_DIR, "config", "memory_strategies.yaml")
# Options: summarization_sliding_window, summarization, conversation_buffer_memory, none
MEMORY_STRATEGY = "summarization_sliding_window"

# Default memory strategy parameters
DEFAULT_MEMORY_SLIDING_WINDOW_SIZE = 10
CHAT_HISTORY = "chat_history"
MEMORY_PARAMETERS_KEY = "parameters"
DEFAULT_MAX_MESSAGES = 50
DEFAULT_SUMMARY_PROMPT = "Summarize the conversation so far in a few sentences."
DEFAULT_SUMMARY_INTERVAL = 10  # Summarize every 10 messages

# Reasoning Strategy Configuration
REASONING_STRATEGIES_FPATH = os.path.join(
    ROOT_DIR, "config", "reasoning_strategies.yaml"
)
# Options: rag_enhanced_reasoning, simple_few_shot, none
REASONING_STRATEGY = "rag_enhanced_reasoning"

# Prompt Configuration
PROMPT_CONFIG_FPATH = os.path.join(ROOT_DIR, "config", "prompt-config.yaml")
PROMPT_CONFIG_NAME = "rag-assistant-system-prompt-formal"

# Metaquestions Configuration
METAQUESTIONS_FPATH = os.path.join(ROOT_DIR, "config", "meta-questions.yaml")

# Query Augmentation Configuration
QUERY_AUGMENTATION_PATH = os.path.join(ROOT_DIR, "config", "query-augmentation.yaml")

# Canonical message returned when the assistant cannot answer from documents
DEFAULT_NOT_KNOWN_MSG = "I'm sorry, that information is not known to me."

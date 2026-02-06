"""
Static application constants that don't vary by environment.
Includes file paths, limits, and default values that are fixed.
"""

import os

# ============================================================================
# FILENAMES, PATHS & DIRECTORIES
# ============================================================================

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT_DIR, os.getenv("DATA_DIR", "data"))
METAQUESTIONS_FPATH = os.path.join(ROOT_DIR, "config", "meta-questions.yaml")
MEMORY_STRATEGIES_FPATH = os.path.join(ROOT_DIR, "config", "memory-strategies.yaml")
PROMPT_CONFIG_FPATH = os.path.join(ROOT_DIR, "config", "prompt-config.yaml")
QUERY_AUGMENTATION_PATH = os.path.join(ROOT_DIR, "config", "query-augmentation.yaml")
README = "README.md"
REASONING_STRATEGIES_FPATH = os.path.join(ROOT_DIR, "config", "reasoning-strategies.yaml")
STYLES_PATH = os.path.join(ROOT_DIR, "static", "css", "styles.css")


# ============================================================================
# VECTOR DATABASE CONSTANTS
# ============================================================================

CHROMA_API_KEY_ENV = "CHROMA_API_KEY"
CHROMA_DATABASE_ENV = "CHROMA_DATABASE"
CHROMA_TENANT_ENV = "CHROMA_TENANT"
COLLECTION_NAME_DEFAULT = "rag_documents"
PUNCTUATION_CHARS = ".,;:!? "


# ============================================================================
# MEMORY & CHAT CONSTANTS
# ============================================================================

CHAT_HISTORY = "chat_history"
MEMORY_PARAMETERS_KEY = "parameters"
NO_CHAT_HISTORY = "No previous conversation context."

# ============================================================================
# PROMPT DEFAULTS
# ============================================================================

DEFAULT_SUMMARY_PROMPT = "Summarize the conversation so far in a few sentences."

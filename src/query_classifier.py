"""
Query classification system for categorizing user queries by type.

Provides QueryType enum and QUERY_CLASSIFIERS with regex patterns for:
- UNSAFE: Harmful or illegal content (blocked before LLM)
- VAGUE: Intentionally broad questions (detected by LLM)
- META: Questions about assistant capabilities/identity
- DOCUMENT: Questions about knowledge base contents
- REGULAR: Normal Q&A queries
"""

import re
from enum import Enum


class QueryType(Enum):
    """Query classification types with priority-based handling."""

    UNSAFE = "unsafe"  # Harmful or illegal content (highest priority)
    VAGUE = "vague"  # Intentionally broad questions (detected by LLM)
    META = "meta"  # Questions about assistant capabilities/identity
    DOCUMENT = "document"  # Questions about knowledge base contents
    REGULAR = "regular"  # Normal Q&A queries (lowest priority)


# Regex patterns for query classification (priority order: unsafe > meta > document > regular)
# Note: VAGUE queries are now detected by LLM using query_vague_detection prompt
QUERY_CLASSIFIERS = {
    "unsafe": {
        "pattern": re.compile(
            r"\b(illegal|crime|violence|harm|exploit|abuse|hate|"
            r"discriminat|drug|weapon|terrorist|attack|unethical)\b",
            re.IGNORECASE,
        ),
        "description": "Harmful or illegal content - blocked before LLM",
    },
    "meta": {
        "pattern": re.compile(
            r"\b(who are you|what are you|tell me about yourself|introduce yourself|"
            r"capabilities|limitations|purpose|how are you built|how do you work"
            r"|what are your limitations|how can you help|what is your purpose)\b",
            re.IGNORECASE,
        ),
        "description": "Questions about assistant identity/capabilities",
    },
    "document": {
        "pattern": re.compile(
            r"\b(what topics|what do you know|what can you|what documents|"
            r"what information|what subjects|what's your knowledge base|"
            r"do you have access to|can you access)\b",
            re.IGNORECASE,
        ),
        "description": "Questions about knowledge base contents",
    },
}

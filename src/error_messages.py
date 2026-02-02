"""Predefined error messages for the application."""

MISSING_API_KEY = (
    "No valid API key found. Please set one of: " "OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
)

ASSISTANT_INITIALIZATION_FAILED = (
    "Assistant initialization failed due to configuration error. " "Please check the logs for details."
)
SEARCH_MANAGER_INITIALIZATION_FAILED = (
    "Unable to initialize the assistant at this time. Please refresh the page "
    "and try again. If the problem persists, please contact support."
)

LLM_INITIALIZATION_FAILED = "LLM initialization failed"

REASONING_INITIALIZATION_FAILED = "Reasoning strategy initialization failed"

VECTOR_DB_INITIALIZATION_FAILED = "Vector DB initialization failed"

DEFAULT_NOT_KNOWN_ERROR_MESSAGE = "I'm sorry, that information is not known to me."

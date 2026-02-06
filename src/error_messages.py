"""Predefined error messages for the application."""

APPLICATION_INITIALIZATION_FAILED = (
    "Unable to initialize the assistant at this time. Please refresh the page "
    "and try again. If the problem persists, please contact support."
)
ASSISTANT_INITIALIZATION_FAILED = (
    "Assistant initialization failed due to configuration error. " "Please check the logs for details."
)
DOCUMENTS_MISSING = "Document list is empty. Please ensure documents are loaded correctly."
LLM_INITIALIZATION_FAILED = "LLM initialization failed"
META_QUESTION_CONFIG_ERROR = "The persona handler will not be able to handle meta questions!"
MISSING_API_KEY = (
    "No valid API key found. Please set one of: " "OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
)
NO_RESULTS_ERROR_MESSAGE = "I'm sorry, that information is not known to me."
REASONING_STRATEGY_MISSING = "No reasoning strategy loaded. Proceeding without one."
SEARCH_FAILED_ERROR_MESSAGE = (
    "I'm sorry, unable to search the documents at the moment." "Please try again later or contact support."
)
VECTOR_DB_INITIALIZATION_FAILED = "Vector DB initialization failed"

"""
Utility functions for initializing and managing LLMs.
"""
import os

from config import LLM_PROVIDERS, LLM_TEMPERATURE, ERROR_NO_API_KEY


def initialize_llm():
    """
    Initialize the LLM by checking for available API keys.
    Tries providers in priority order defined in config.
    """
    # Iterate through providers and use the first available one
    for provider in LLM_PROVIDERS:
        api_key = os.getenv(provider["api_key_env"])
        if api_key:
            model_name = os.getenv(provider["model_env"], provider["default_model"])

            # Initialize LLM with provider-specific parameters
            kwargs = {
                provider["api_key_param"]: api_key,
                "model": model_name,
                "temperature": LLM_TEMPERATURE,
            }
            return provider["class"](**kwargs)

    # If no provider is available, raise an error
    raise ValueError(ERROR_NO_API_KEY)

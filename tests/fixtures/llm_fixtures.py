"""
LLM-related fixtures for testing.
"""

# pylint: disable=import-error, redefined-outer-name

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_llm():
    """Fixture providing a mocked LLM instance."""
    return MagicMock()


@pytest.fixture
def mock_llm_response():
    """Provide mock LLM response."""
    return "Artificial intelligence is a field of computer science that aims to create intelligent machines."


@pytest.fixture
def mock_llm_response_out_of_scope():
    """Provide mock LLM response for out-of-scope queries."""
    return "I'm sorry, that information is not known to me."


@pytest.fixture
def mock_llm_response_unclear():
    """Provide mock LLM response for unclear queries."""
    return "Could you please ask a clear, meaningful question?"


# ============================================================================
# LLM PROVIDER PATCHES
# ============================================================================


def llm_provider_patch(provider_config):
    """
    Create a patch for LLM_PROVIDERS with the given provider configuration.

    Args:
        provider_config: Dictionary with keys:
            - api_key_env: Environment variable for API key
            - api_key_param: Parameter name for API key
            - model_env: Environment variable for model name
            - default_model: Default model name

    Returns:
        A patch object that mocks LLM_PROVIDERS
    """
    provider = {
        "api_key_env": provider_config.get("api_key_env"),
        "api_key_param": provider_config.get("api_key_param", "api_key"),
        "model_env": provider_config.get("model_env"),
        "default_model": provider_config.get("default_model"),
        "class": MagicMock(),
    }
    return patch("src.llm_utils.LLM_PROVIDERS", [provider])


# ============================================================================
# PARAMETERIZED FIXTURES FOR LLM PROVIDER TESTS
# ============================================================================


@pytest.fixture(
    params=[
        pytest.param(
            {
                "provider": "groq",
                "api_key_env": "GROQ_API_KEY",
                "api_key_value": "groq-key",
                "model_env": "GROQ_MODEL",
                "default_model": "llama-3.1-8b-instant",
            },
            id="groq",
        ),
        pytest.param(
            {
                "provider": "openai",
                "api_key_env": "OPENAI_API_KEY",
                "api_key_value": "openai-key",
                "model_env": "OPENAI_MODEL",
                "default_model": "gpt-4o-mini",
            },
            id="openai",
        ),
        pytest.param(
            {
                "provider": "google",
                "api_key_env": "GOOGLE_API_KEY",
                "api_key_value": "google-key",
                "model_env": "GOOGLE_MODEL",
                "default_model": "gemini-1.5-flash",
            },
            id="google",
        ),
    ]
)
def llm_provider_config(request):
    """Parameterized fixture providing different LLM provider configurations."""
    return request.param


@pytest.fixture
def llm_provider_with_env(llm_provider_config):
    """
    Fixture that patches both environment variables and LLM_PROVIDERS based on config.
    Yields the provider config for use in tests.
    """
    provider_config = llm_provider_config
    env_patch = {provider_config["api_key_env"]: provider_config["api_key_value"]}

    provider_mock = {
        "api_key_env": provider_config["api_key_env"],
        "api_key_param": "api_key",
        "model_env": provider_config["model_env"],
        "default_model": provider_config["default_model"],
        "class": MagicMock(),
    }

    with patch.dict("os.environ", env_patch):
        with patch("src.llm_utils.LLM_PROVIDERS", [provider_mock]):
            yield provider_config

"""
Fixtures providing sample queries for testing document-based question answering systems.
"""

import pytest


@pytest.fixture
def sample_queries_in_scope():
    """Provide sample queries that should be answerable from documents."""
    return [
        "What is artificial intelligence?",
        "How does quantum computing work?",
        "Tell me about ancient Egypt.",
        "What is contemporary art?",
        "What is consciousness research?",
    ]


@pytest.fixture
def sample_queries_out_of_scope():
    """Provide sample queries that are out of scope (not in documents)."""
    return [
        "What is the etymology of Pharaonic?",
        "What is the capital of France?",
        "How do I bake a chocolate cake?",
        "What is the weather today?",
        "Tell me about deep sea creatures.",
    ]


@pytest.fixture
def sample_queries_ambiguous():
    """Provide ambiguous queries that could be interpreted multiple ways."""
    return [
        "Tell me more",
        "What else?",
        "Anything else?",
        "Continue",
        "Go on",
    ]


@pytest.fixture
def sample_queries_gibberish():
    """Provide gibberish/nonsensical queries."""
    return [
        "asdfgh",
        "???",
        "sdadsad",
        "xyzabc123!@#",
        "zzzzzzzzz",
    ]


@pytest.fixture
def sample_queries_special_cases():
    """Provide special case queries."""
    return {
        "greeting": "Hello! How are you?",
        "thanks": "Thank you for your help!",
        "goodbye": "Goodbye!",
        "vague": "What topics do you have?",
        "about_limitations": "What are your limitations?",
    }

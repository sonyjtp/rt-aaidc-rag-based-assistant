"""
Fixtures for context-related unit tests.
"""

# pylint: disable=import-error

import pytest


@pytest.fixture
def sample_context_empty():
    """Provide empty context for edge case testing."""
    return ""


@pytest.fixture
def search_results_with_docs():
    """Provide mock search results with documents."""
    return {
        "documents": [
            [
                "Artificial intelligence is a field of computer science.",
                "Machine learning is a subset of AI.",
            ],
            ["Quantum computing uses quantum mechanics principles."],
        ],
        "ids": [["doc1", "doc2"], ["doc3"]],
        "distances": [[0.1, 0.2], [0.15]],
    }


@pytest.fixture
def search_results_empty():
    """Provide mock search results with no documents."""
    return {
        "documents": [[], []],
        "ids": [[], []],
        "distances": [[], []],
    }


@pytest.fixture
def memory_messages():
    """Provide sample memory messages."""
    return [
        {"role": "user", "content": "What is AI?"},
        {"role": "assistant", "content": "AI is artificial intelligence..."},
        {"role": "user", "content": "Tell me more"},
        {"role": "assistant", "content": "Machine learning is a subset of AI..."},
    ]


@pytest.fixture
def sample_context():
    """Provide a small sample conversation context for tests."""
    return [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi, how can I help?"},
    ]

"""
Test fixtures and sample data for RAG Assistant tests.
"""

# pylint: disable=import-error

import pytest


@pytest.fixture
def sample_documents():
    """Provide sample documents for testing."""
    return [
        "Artificial intelligence is transforming how we process information. Machine learning algorithms learn from "
        "data to make predictions.",
        "Quantum computing represents a paradigm shift in computational power, using quantum bits (qubits) for "
        "processing.",
        "Ancient Egyptian civilization was one of the most advanced societies of its time, with impressive "
        "architectural  achievements.",
        "Contemporary art encompasses diverse movements and styles, reflecting modern societal values and "
        "technological innovation.",
        "Consciousness research explores the nature of human awareness and subjective experience through neuroscience "
        "and philosophy.",
    ]


@pytest.fixture
def sample_documents_empty():
    """Provide empty documents list for edge case testing."""
    return []


@pytest.fixture
def sample_documents_single():
    """Provide single document for edge case testing."""
    return ["Artificial intelligence is a branch of computer science."]


@pytest.fixture
def sample_context():
    """Provide sample context retrieved from documents."""
    return """
    Artificial intelligence (AI) refers to the simulation of human intelligence by machines,
    particularly computer systems. These systems are designed to perform tasks that typically
    require human intelligence, such as learning from experience, recognizing patterns,
    understanding language, and making decisions.
    """

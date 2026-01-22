"""
Embeddings initialization utilities for HuggingFace models.
Handles device detection and embedding model setup.
"""

import os
import torch
from langchain_huggingface import HuggingFaceEmbeddings


def initialize_embedding_model() -> HuggingFaceEmbeddings:
    """
    Initialize HuggingFace embeddings with automatic device detection.

    Uses cuda if available, falls back to mps (Apple Silicon) or cpu.
    Model name can be configured via EMBEDDING_MODEL environment variable.

    Returns:
        HuggingFaceEmbeddings instance configured for the detected device
    """
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device for embeddings: {device}")

    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
    )


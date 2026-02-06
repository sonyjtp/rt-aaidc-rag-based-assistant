"""Utility functions for UI styling and configuration."""
import os
import re

import streamlit as st

from app_constants import DATA_DIR, STYLES_PATH
from config import DOCUMENT_TYPES
from file_utils import read_file


def _get_valid_topics_from_documents(file_extensions: str | tuple[str, ...] = DOCUMENT_TYPES) -> set[str]:
    """
    Extract valid topic names from the data directory document filenames.

    Converts filenames like 'extinct_sports.txt' to 'extinct sports'.

    Returns:
        Set of valid topic names in lowercase
    """
    valid_topics = set()

    try:
        if os.path.isdir(DATA_DIR):
            for filename in os.listdir(DATA_DIR):
                if filename.endswith(file_extensions):
                    # Remove extension and replace underscores with spaces
                    topic_name = filename[:-4].replace("_", " ")
                    valid_topics.add(topic_name.lower())
    except (PermissionError, OSError):
        # If we can't read the directory, return empty set
        # The validation will be skipped gracefully
        pass

    return valid_topics


def validate_and_filter_topics(response: str) -> str:
    """
    Strip any 'Related Topics You Can Explore' section from the response.

    The project previously attempted to validate and re-write suggested related
    topics. Per the user's request to stop suggesting related topics entirely,
    this function now removes that section wherever it appears.

    Args:
        response: The LLM response text

    Returns:
        The response with any Related Topics section removed
    """
    # Remove any 'Related Topics You Can Explore: [...]' section (case-insensitive)
    related_topics_pattern = r"Related Topics You Can Explore:\s*\[.*?\]"
    response = re.sub(related_topics_pattern, "", response, flags=re.IGNORECASE | re.DOTALL)
    return response.strip()


def load_custom_styles() -> None:
    """Load custom CSS styles from static/css/styles.css file."""

    css_content = read_file(STYLES_PATH)
    if css_content:
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    else:
        st.warning("Custom styles file not found. Using default styling.")


def configure_page() -> None:
    """Configure Streamlit page settings with dark theme as default."""
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

"""Utility functions for UI styling and configuration."""
import os
import re

import streamlit as st

from config import DATA_DIR


def _get_valid_topics_from_documents() -> set[str]:
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
                if filename.endswith(".txt"):
                    # Remove .txt extension and replace underscores with spaces
                    topic_name = filename[:-4].replace("_", " ")
                    valid_topics.add(topic_name.lower())
    except Exception:
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
    response = re.sub(
        related_topics_pattern, "", response, flags=re.IGNORECASE | re.DOTALL
    )
    return response.strip()


def load_custom_styles() -> None:
    """Load custom CSS styles from static/css/styles.css file."""
    # Get the path to the styles.css file relative to the project root
    # __file__ is in src/utils/, so go up 2 directories to reach project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(current_dir)
    project_root = os.path.dirname(src_dir)
    styles_path = os.path.join(project_root, "static", "css", "styles.css")

    try:
        with open(styles_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("Custom styles file not found. Using default styling.")


def configure_page() -> None:
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

"""Utility functions for UI styling and configuration."""
import os

import streamlit as st


def load_custom_styles() -> None:
    """Load custom CSS styles from static/css/styles.css file."""
    # Get the path to the styles.css file relative to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

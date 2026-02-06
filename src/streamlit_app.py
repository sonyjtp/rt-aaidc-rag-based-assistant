"""Streamlit web UI for the RAG-based AI assistant."""

import os

import streamlit as st
from dotenv import load_dotenv

from app_constants import DATA_DIR
from config import DOCUMENT_TYPES
from error_messages import APPLICATION_INITIALIZATION_FAILED
from file_utils import load_documents
from log_manager import logger
from rag_assistant import RAGAssistant
from ui_constants import (
    CLEAR_HISTORY_BUTTON,
    MAIN_TITLE,
    SESSION_ASSISTANT,
    SESSION_CHAT_HISTORY,
    SESSION_DOCUMENTS_LOADED,
    SESSION_INITIALIZATION_ATTEMPTED,
    SESSION_INITIALIZED,
    SIDEBAR_TITLE,
)
from ui_utils import configure_page, load_custom_styles, validate_and_filter_topics

# Set tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()

# Configure page and load styles
configure_page()
load_custom_styles()

# Initialize session state
if SESSION_ASSISTANT not in st.session_state:
    st.session_state[SESSION_ASSISTANT] = None
    st.session_state[SESSION_DOCUMENTS_LOADED] = False
    st.session_state[SESSION_CHAT_HISTORY] = []
    st.session_state[SESSION_INITIALIZED] = False
    st.session_state[SESSION_INITIALIZATION_ATTEMPTED] = False

# Auto-initialize RAG assistant on app startup
if not st.session_state[SESSION_INITIALIZATION_ATTEMPTED]:
    st.session_state[SESSION_INITIALIZATION_ATTEMPTED] = True
    try:
        # Load documents silently without displaying status
        documents = load_documents(folder=DATA_DIR, file_extensions=DOCUMENT_TYPES)

        # Initialize the RAG assistant
        st.session_state[SESSION_ASSISTANT] = RAGAssistant()
        logger.info("RAG Assistant initialized")

        # Add documents to the RAG assistant's vector DB
        st.session_state[SESSION_ASSISTANT].add_documents(documents)
        st.session_state[SESSION_DOCUMENTS_LOADED] = True
        st.session_state[SESSION_INITIALIZED] = True
    except Exception as e:  # pylint: disable=broad-exception-caught
        st.session_state[SESSION_INITIALIZED] = False
        logger.error(f"Error initializing assistant: {e}")
        st.error(APPLICATION_INITIALIZATION_FAILED)

# Sidebar configuration
with st.sidebar:
    st.title(SIDEBAR_TITLE)

    st.divider()

    # Clear chat history button
    if st.button(CLEAR_HISTORY_BUTTON, use_container_width=True):
        st.session_state[SESSION_CHAT_HISTORY] = []
        st.rerun()

    st.divider()

    # Status section
    st.subheader("Status")
    if st.session_state[SESSION_INITIALIZED]:
        st.success("✅ RAG Assistant Ready")
    else:
        st.warning("⏳ Initializing assistant...")

# Main content area
st.markdown(f"<h1 class='title-text'>{MAIN_TITLE}</h1>", unsafe_allow_html=True)

st.markdown(
    """
    This chatbot uses Retrieval-Augmented Generation (RAG) to answer questions based on a collection of documents.
    Ask any question about the documents below!
"""
)

st.divider()

# Chat interface
if st.session_state[SESSION_INITIALIZED]:
    # Display chat history only after first message
    if st.session_state[SESSION_CHAT_HISTORY]:
        st.subheader("Chat History")

        chat_container = st.container()

        with chat_container:
            for message in st.session_state[SESSION_CHAT_HISTORY]:
                if message["role"] == "user":
                    st.markdown(
                        f"""
                        <div class="chat-message user-message">
                            <div>
                                <strong>You:</strong><br/>
                                {message["content"]}
                            </div>
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="chat-message assistant-message">
                            <div>
                                <strong>Assistant:</strong><br/>
                                {message["content"]}
                            </div>
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )

        st.divider()

    # Input area
    st.subheader("Ask a Question")

    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])

        with col1:
            user_input = st.text_input(
                "Enter your question about the documents:",
                placeholder="e.g., What are the major rivers in India? Or ask about Indian history, culture, etc.",
                label_visibility="collapsed",
            )

        with col2:
            send_button = st.form_submit_button("Send", use_container_width=True, type="primary")

    # Process user input outside the form
    if send_button and user_input:
        # Add user message to history
        st.session_state[SESSION_CHAT_HISTORY].append({"role": "user", "content": user_input})
        logger.info(f"User question: {user_input}")

        # Get assistant response
        status = st.status("🔍 Searching documents and generating response...", expanded=True)
        try:
            response = st.session_state[SESSION_ASSISTANT].invoke(user_input)
            logger.debug(f"Agent response received: {response[:100]}")

            # Validate and filter Related Topics - only allow topics from the knowledge base
            response = validate_and_filter_topics(response)
            logger.debug(f"Response after topic validation: {response[:100]}")

            # Clean up the response - remove Markdown headers and separators
            lines = response.split("\n")
            cleaned_lines = []
            skip_next = False  # pylint: disable=invalid-name

            for i, line in enumerate(lines):  # pylint: disable=unused-variable, invalid-name
                # Skip Markdown headers (lines starting with # or **)
                if line.strip().startswith("#") or line.strip().startswith("**"):
                    skip_next = True
                    continue
                # Skip separator lines (===, ---, etc.)
                if skip_next and (all(c in "=-_" for c in line.strip()) and len(line.strip()) > 3):
                    skip_next = False
                    continue
                # Skip empty lines at the start
                if cleaned_lines or line.strip():
                    cleaned_lines.append(line)
                skip_next = False

            CLEANED_RESPONSE = "\n".join(cleaned_lines).strip()  # pylint: disable=invalid-name
            logger.info(f"Cleaned response: {CLEANED_RESPONSE[:100]}")

            st.session_state[SESSION_CHAT_HISTORY].append({"role": "assistant", "content": CLEANED_RESPONSE})
            status.update(label="✅ Response generated!", state="complete")
            st.rerun()
        except Exception as e:  # pylint: disable=broad-exception-caught
            status.update(label="❌ Error generating response", state="error")
            logger.error(f"Error generating response: {e}")
            st.error("Error processing your question. Please try again.")
else:
    if not st.session_state[SESSION_INITIALIZATION_ATTEMPTED]:
        st.info("⏳ RAG Assistant is initializing... Please wait and refresh if needed.")

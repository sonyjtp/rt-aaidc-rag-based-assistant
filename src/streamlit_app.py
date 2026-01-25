"""Streamlit web UI for the RAG-based AI assistant."""
import os

import streamlit as st
from dotenv import load_dotenv

from config import DATA_DIR
from file_utils import load_documents
from logger import logger
from rag_assistant import RAGAssistant
from ui_utils import configure_page, load_custom_styles

# Set tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()

# Configure page and load styles
configure_page()
load_custom_styles()

# Initialize session state
if "assistant" not in st.session_state:
    st.session_state.assistant = None
    st.session_state.documents_loaded = False
    st.session_state.chat_history = []
    st.session_state.initialized = False
    st.session_state.initialization_attempted = False

# Auto-initialize RAG assistant on app startup
if not st.session_state.initialization_attempted:
    st.session_state.initialization_attempted = True
    try:
        # Load documents silently without displaying status
        documents = load_documents(folder=DATA_DIR, file_extensions=".txt")

        # Initialize the RAG assistant
        st.session_state.assistant = RAGAssistant()
        st.session_state.assistant.add_documents(documents)
        st.session_state.documents_loaded = True
        st.session_state.initialized = True
    except Exception as e:  # pylint: disable=broad-exception-caught
        st.session_state.initialized = False
        logger.error(f"Error initializing assistant: {e}")
        st.error("Error initializing assistant. Please try again.")

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Configuration")

    st.divider()

    # Clear chat history button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    # Status section
    st.subheader("Status")
    if st.session_state.initialized:
        st.success("✅ RAG Assistant Ready")
    else:
        st.warning("⏳ Initializing assistant...")

# Main content area
st.markdown("<h1 class='title-text'>🤖 RAG-Based Chatbot</h1>", unsafe_allow_html=True)

st.markdown(
    """
    This chatbot uses Retrieval-Augmented Generation (RAG) to answer questions based on a collection of documents.
    Ask any question about the documents below!
"""
)

st.divider()

# Chat interface
if st.session_state.initialized:
    # Display chat history only after first message
    if st.session_state.chat_history:
        st.subheader("Chat History")

        chat_container = st.container()

        with chat_container:
            for message in st.session_state.chat_history:
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
                placeholder="e.g., What is quantum computing? (Press Enter to send)",
                label_visibility="collapsed",
            )

        with col2:
            send_button = st.form_submit_button(
                "Send", use_container_width=True, type="primary"
            )

    # Process user input outside the form
    if send_button and user_input:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        logger.debug(f"User question: {user_input}")

        # Get assistant response
        status = st.status(
            "🔍 Searching documents and generating response...", expanded=True
        )
        try:
            response = st.session_state.assistant.invoke(user_input)
            logger.debug(f"Agent response received: {response[:100]}")

            # Clean up the response - remove markdown headers and separators
            lines = response.split("\n")
            cleaned_lines = []
            skip_next = False

            for i, line in enumerate(lines):  # pylint: disable=unused-variable
                # Skip markdown headers (lines starting with # or **)
                if line.strip().startswith("#") or line.strip().startswith("**"):
                    skip_next = True
                    continue
                # Skip separator lines (===, ---, etc.)
                if skip_next and (
                    all(c in "=-_" for c in line.strip()) and len(line.strip()) > 3
                ):
                    skip_next = False
                    continue
                # Skip empty lines at the start
                if cleaned_lines or line.strip():
                    cleaned_lines.append(line)
                skip_next = False

            CLEANED_RESPONSE = "\n".join(
                cleaned_lines
            ).strip()  # pylint: disable=invalid-name
            logger.debug(f"Cleaned response: {CLEANED_RESPONSE[:100]}")

            st.session_state.chat_history.append(
                {"role": "assistant", "content": CLEANED_RESPONSE}
            )
            status.update(label="✅ Response generated!", state="complete")
            st.rerun()
        except Exception as e:  # pylint: disable=broad-exception-caught
            status.update(label="❌ Error generating response", state="error")
            logger.error(f"Error generating response: {e}")
            st.error("Error processing your question. Please try again.")
else:
    if not st.session_state.initialization_attempted:
        msg = "RAG Assistant is initializing... Please wait and refresh if needed."  # pylint: disable=invalid-name
        st.info(f"⏳ {msg}")

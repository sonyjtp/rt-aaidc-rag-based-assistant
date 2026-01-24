import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import streamlit as st
from dotenv import load_dotenv

from config import DATA_DIR
from file_utils import load_documents
from rag_assistant import RAGAssistant
from logger import logger

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    body {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .stApp {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .stText, h1, h2, h3, h4, h5, h6, p, div {
        color: #ffffff !important;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        gap: 1rem;
        color: #ffffff;
    }
    .user-message {
        background-color: #2c3e50;
        border-left: 4px solid #3498db;
        color: #ffffff;
    }
    .assistant-message {
        background-color: #34495e;
        border-left: 4px solid #2ecc71;
        color: #ffffff;
    }
    .title-text {
        text-align: center;
        color: #3498db;
    }
    .stTextInput > div > div > input {
        background-color: #2c3e50 !important;
        color: #ffffff !important;
        border: 1px solid #3498db !important;
    }
    .stTextInput > label {
        color: #ffffff !important;
    }
    .stButton > button {
        background-color: #3498db !important;
        color: #ffffff !important;
        border: none !important;
    }
    .stButton > button:hover {
        background-color: #2980b9 !important;
    }
    .stMarkdown {
        color: #ffffff !important;
    }
    .stDivider {
        background-color: #3498db !important;
    }
    .stSidebar {
        background-color: #252d36 !important;
    }
    .stSidebar .stText, .stSidebar h1, .stSidebar h2, .stSidebar h3 {
        color: #ffffff !important;
    }
    .stInfo, .stSuccess, .stWarning, .stError {
        color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'assistant' not in st.session_state:
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
        documents = load_documents(folder=DATA_DIR, file_extns=".txt")

        # Initialize the RAG assistant
        st.session_state.assistant = RAGAssistant()
        st.session_state.assistant.add_documents(documents)
        st.session_state.documents_loaded = True
        st.session_state.initialized = True
    except Exception as e:
        st.session_state.initialized = False
        logger.error(f"Error initializing assistant: {e}")
        st.error("I'm sorry, an error occurred while initializing the assistant. Please try again.")

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

st.markdown("""
    This chatbot uses Retrieval-Augmented Generation (RAG) to answer questions based on a collection of documents.
    Ask any question about the documents below!
""")

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
                    st.markdown(f"""
                        <div class="chat-message user-message">
                            <div>
                                <strong>You:</strong><br/>
                                {message["content"]}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="chat-message assistant-message">
                            <div>
                                <strong>Assistant:</strong><br/>
                                {message["content"]}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

        st.divider()

    # Input area
    st.subheader("Ask a Question")

    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])

        with col1:
            user_input = st.text_input(
                "Enter your question about the documents:",
                placeholder="e.g., What is quantum computing? (Press Enter to send)",
                label_visibility="collapsed"
            )

        with col2:
            send_button = st.form_submit_button("Send", use_container_width=True, type="primary")

    # Process user input outside the form
    if send_button and user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        # Get assistant response
        status = st.status("🔍 Searching documents and generating response...", expanded=True)
        try:
            response = st.session_state.assistant.invoke(user_input)

            # Clean up the response - remove markdown headers and separators
            lines = response.split('\n')
            cleaned_lines = []
            skip_next = False

            for i, line in enumerate(lines):
                # Skip markdown headers (lines starting with # or **)
                if line.strip().startswith('#') or line.strip().startswith('**'):
                    # Skip the header and the separator line after it
                    skip_next = True
                    continue
                # Skip separator lines (===, ---, etc.)
                if skip_next and (all(c in '=-_' for c in line.strip()) and len(line.strip()) > 3):
                    skip_next = False
                    continue
                # Skip empty lines at the start of cleaned response
                if cleaned_lines or line.strip():
                    cleaned_lines.append(line)
                skip_next = False

            cleaned_response = '\n'.join(cleaned_lines).strip()

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": cleaned_response
            })
            status.update(label="✅ Response generated!", state="complete")
            st.rerun()
        except Exception as e:
            status.update(label="❌ Error generating response", state="error")
            logger.error(f"Error generating response: {e}")
            st.error("I'm sorry, an error occurred while processing your question. Please try again.")
else:
    if not st.session_state.initialization_attempted:
        st.info("⏳ RAG Assistant is initializing... Please wait a moment and refresh the page if needed.")


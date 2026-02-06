"""Main entry point for the RAG Assistant CLI application."""
import os

# Optional dotenv import (not required in all environments)
from dotenv import load_dotenv

from app_constants import DATA_DIR
from config import DOCUMENT_TYPES
from file_utils import load_documents
from log_manager import logger
from rag_assistant import RAGAssistant

# Ensure tokenizers won't use parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv()


def main():
    """Main function to demonstrate the RAG assistant. This is an alternative to the
    Streamlit app and provides a simple CLI interface. This can be used for quick
    testing or in environments where a web UI is not feasible.

    Steps:
    1. Load documents from the data directory.
    2. Initialize the RAG assistant.
    3. Add documents to the assistant's vector database.
    4. Enter a loop to ask questions and get answers from the assistant.
    5. Exit the loop when the user types 'q'.
    """

    try:
        # Load documents
        documents = load_documents(folder=DATA_DIR, file_extensions=DOCUMENT_TYPES)
        logger.info(f"Loaded {len(documents)} documents")

        # Initialize the RAG assistant
        assistant = RAGAssistant()
        logger.info("RAG Assistant initialized")

        # Add documents to the RAG assistant's vector DB
        assistant.add_documents(documents)
        done = False
        while not done:
            question = input("Enter a question or 'q' to exit: ")
            if question.lower() == "q":
                done = True
            else:
                result = assistant.invoke(question)
                logger.info(result)

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error(f"Error running RAG assistant: {e}")


if __name__ == "__main__":
    main()

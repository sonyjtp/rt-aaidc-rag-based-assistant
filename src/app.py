"""Main entry point for the RAG Assistant CLI application."""
import os

# Optional dotenv import (not required in all environments)
from dotenv import load_dotenv

from config import DATA_DIR, DOCUMENT_TYPES
from file_utils import load_documents
from logger import logger
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
    1. Load documents from the data directory
    2. Initialize the RAG assistant
    3. Add documents to the RAG assistant's vector DB
    4. Enter a loop to accept user questions and provide answers
    5. Handle errors gracefully and log them
    6. Exit on user command
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

"""Main entry point for the RAG Assistant CLI application."""
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from dotenv import load_dotenv

from config import DATA_DIR
from file_utils import load_documents
from rag_assistant import RAGAssistant
from logger import logger

# Load environment variables
load_dotenv()

def main():
    """Main function to demonstrate the RAG assistant."""
    try:
        # Load documents
        documents = load_documents(folder=DATA_DIR, file_extns=".txt")
        logger.info(f"Loaded {len(documents)} documents")

        # Initialize the RAG assistant
        assistant = RAGAssistant()
        logger.info("RAG Assistant initialized")
        assistant.add_documents(documents)
        done = False
        while not done:
            question = input("Enter a question or 'quit' to exit: ")
            if question.lower() == "quit":
                done = True
            else:
                result = assistant.invoke(question)
                logger.info(result)

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.error(f"Error running RAG assistant: {e}")
        logger.error("I'm sorry, an error occurred while running the assistant. Please try again.")


if __name__ == "__main__":
    main()

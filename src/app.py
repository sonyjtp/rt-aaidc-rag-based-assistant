import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from dotenv import load_dotenv

from config import DATA_DIR
from file_utils import load_documents
from rag_assistant import RAGAssistant

# Load environment variables
load_dotenv()

def main():
    """Main function to demonstrate the RAG assistant."""
    try:
        # Load documents
        documents = load_documents(folder=DATA_DIR, file_extns=".txt")
        print(f"✓ Loaded {len(documents)} documents")

        # Initialize the RAG assistant
        assistant = RAGAssistant()
        print("✓ RAG Assistant initialized\n")
        assistant.add_documents(documents)
        done = False
        while not done:
            question = input("Enter a question or 'quit' to exit: ")
            if question.lower() == "quit":
                done = True
            else:
                result = assistant.invoke(question)
                print(result)

    except Exception as e:
        print(f"Error running RAG assistant: {e}")
        print("Make sure you have set up your .env file with at least one API key:")
        print("- OPENAI_API_KEY (OpenAI GPT models)")
        print("- GROQ_API_KEY (Groq Llama models)")
        print("- GOOGLE_API_KEY (Google Gemini models)")


if __name__ == "__main__":
    main()

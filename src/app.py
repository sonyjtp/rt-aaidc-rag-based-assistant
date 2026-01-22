import os
from typing import List
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vectordb import VectorDB
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from config import LLM_PROVIDERS, LLM_TEMPERATURE, ERROR_NO_API_KEY, DATA_DIR

# Load environment variables
load_dotenv()

def load_documents() -> List[str]:
    """
    Load documents from the DATA_DIR directory.

    Returns:
        List of document contents loaded from .txt files
    """
    documents = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            file_path = os.path.join(DATA_DIR, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                documents.append(content)
    return documents


class RAGAssistant:
    """
    A simple RAG-based AI assistant using ChromaDB and multiple LLM providers.
    Supports OpenAI, Groq, and Google Gemini APIs.
    """

    @staticmethod
    def _initialize_llm():
        """
        Initialize the LLM by checking for available API keys.
        Tries providers in priority order defined in config.
        """
        # Iterate through providers and use the first available one
        for provider in LLM_PROVIDERS:
            api_key = os.getenv(provider["api_key_env"])
            if api_key:
                model_name = os.getenv(provider["model_env"], provider["default_model"])
                print(f"Using {provider['name']} model: {model_name}")

                # Initialize LLM with provider-specific parameters
                kwargs = {
                    provider["api_key_param"]: api_key,
                    "model": model_name,
                    "temperature": LLM_TEMPERATURE,
                }
                return provider["class"](**kwargs)

        # If no provider is available, raise an error
        raise ValueError(ERROR_NO_API_KEY)

    def __init__(self):
        """Initialize the RAG assistant."""
        # Initialize LLM - check for available API keys in order of preference
        self.llm = self._initialize_llm()
        print(f"LLM {self.llm.model_name} initialized")

        # Initialize vector database
        self.vector_db = VectorDB()
        print("Vector database initialized")

        # Create RAG prompt template
        # TODO: Implement your RAG prompt template
        # HINT: Use ChatPromptTemplate.from_template() with a template string
        # HINT: Your template should include placeholders for {context} and {question}
        # HINT: Design your prompt to effectively use retrieved context to answer questions
        # self.prompt_template = None  # Your implementation here
        #
        # # Create the chain
        # self.chain = self.prompt_template | self.llm | StrOutputParser()
        #
        # print("RAG Assistant initialized successfully")

    def add_documents(self, documents: List) -> None:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of documents
        """
        self.vector_db.add_documents(documents)

    def invoke(self, input: str, n_results: int = 3) -> str:
        """
        Query the RAG assistant.

        Args:
            input: User's input
            n_results: Number of relevant chunks to retrieve

        Returns:
            Dictionary containing the answer and retrieved context
        """
        llm_answer = ""
        # TODO: Implement the RAG query pipeline
        # HINT: Use self.vector_db.search() to retrieve relevant context chunks
        # HINT: Combine the retrieved document chunks into a single context string
        # HINT: Use self.chain.invoke() with context and question to generate the response
        # HINT: Return a string answer from the LLM

        # Your implementation here
        return llm_answer


def main():
    """Main function to demonstrate the RAG assistant."""
    try:
        # Initialize the RAG assistant
        print("Initializing RAG Assistant...")
        assistant = RAGAssistant()

        # Load sample documents
        print("\nLoading documents...")
        documents = load_documents()
        print(f"Loaded {len(documents)} sample documents")
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

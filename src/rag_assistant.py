from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from llm_utils import initialize_llm
from prompt_builder import build_system_prompts
from vectordb import VectorDB


class RAGAssistant:
    """
    A simple RAG-based AI assistant using ChromaDB and multiple LLM providers.
    Supports OpenAI, Groq, and Google Gemini APIs.
    """


    def __init__(self):
        """Initialize the RAG assistant."""


        self.llm = initialize_llm()
        print(f"✓ LLM {self.llm.model_name} initialized.")

        # Initialize vector database
        self.vector_db = VectorDB()
        print("✓ Vector database initialized.")
        self._build_chain()

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

    def _build_chain(self):
        """Build the prompt template and LLM chain."""
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "\n".join(build_system_prompts())),
            ("human", "Context from documents:\n{context}\n\nQuestion: {question}")
        ])
        print("✓ Prompt template for RAG Assistant created from system prompts and for human input.")
        self.chain = self.prompt_template | self.llm | StrOutputParser()
        print("✓ Function chain with prompt template, LLM, and output parser built.")


    def add_documents(self, documents:  list[str] |  list[dict[str, str]]) -> None:
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
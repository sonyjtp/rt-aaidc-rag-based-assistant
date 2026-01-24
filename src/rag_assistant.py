from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from llm_utils import initialize_llm
from prompt_builder import build_system_prompts
from vectordb import VectorDB
from memory_manager import MemoryManager
from config import RETRIEVAL_K_DEFAULT
from logger import logger


class RAGAssistant:
    """
    A simple RAG-based AI assistant using ChromaDB and multiple LLM providers.
    Supports OpenAI, Groq, and Google Gemini APIs.
    """

    def __init__(self):
        """Initialize the RAG assistant."""
        self.llm = initialize_llm()
        logger.info(f"LLM: {self.llm.model_name}")

        # Initialize vector database
        self.vector_db = VectorDB()

        # Initialize conversation memory
        self.memory_manager = MemoryManager(llm=self.llm)
        if self.memory_manager.memory:
            logger.info(f"Memory manager initialized with strategy: {self.memory_manager.strategy}")

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

    def _build_chain(self):
        """Build the prompt template and LLM chain."""
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", "\n".join(build_system_prompts())),
            ("human", "Context from documents:\n{context}\n\nQuestion: {question}")
        ])
        logger.info("Prompt template for RAG Assistant created from system prompts and for human input.")
        self.chain = self.prompt_template | self.llm | StrOutputParser()
        logger.info("Function chain with prompt template, LLM, and output parser built.")


    def add_documents(self, documents:  list[str] |  list[dict[str, str]]) -> None:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of documents
        """
        self.vector_db.add_documents(documents)

    def invoke(self, query: str, n_results: int = RETRIEVAL_K_DEFAULT) -> str:
        """
        Query the RAG assistant.

        Args:
            query: User's input
            n_results: Number of relevant chunks to retrieve

        Returns:
            String answer from the LLM based on retrieved context
        """
        search_results = self.vector_db.search(query=query, n_results=n_results)

        # Extract documents from search results
        documents = search_results.get("documents", [])
        context = "\n".join(documents)

        response = self.chain.invoke({
            "context": context,
            "question": query
        })

        # Save conversation to memory manager
        self.memory_manager.add_message(input_text=query, output_text=response)

        return response
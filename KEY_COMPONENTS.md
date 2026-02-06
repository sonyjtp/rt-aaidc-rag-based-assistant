• **RAG Assistant Core** - Orchestrates all major subsystems including LLM initialization, search operations, memory management, and response generation.

• **Search Manager** - Performs semantic similarity search by converting queries to embeddings and retrieving relevant document chunks from the vector database, validates context relevance using the LLM, manages document ingestion, and provides result flattening and logging for debugging.
• **Vector Database** - Manages document ingestion, chunking, embedding storage, and semantic search operations using ChromaDB.

• **Persona Handler** - Detects and responds to meta-questions about the application using pattern matching, with support for refusal responses, self-description, and dynamic README content extraction.

• **Query Processor** - Identifies whether queries are new or follow-up questions, validates retrieved context against similarity thresholds, and augments queries with conversation history.
• **Prompt Builder** - Constructs optimized system prompts with constraints to prevent hallucination and inject dynamic context from retrieved documents.

• **Reasoning Strategies** - Implements and selects from multiple reasoning approaches (RAG-Enhanced Reasoning, Chain-of-Thought, ReAct, Few-Shot Prompting, and Metacognitive Prompting) to structure LLM reasoning, process queries through the chosen strategy, and enhance response generation with structured thinking patterns.
• **LLM Integration** - Abstracts multi-provider LLM support with automatic fallback mechanisms and unified interface for model invocation.

• **Memory Manager** - Maintains conversation history using configurable strategies to balance context preservation with token constraints.

• **CLI Interface** - Provides a command-line interface for interactive chatting and quick testing.

• **Streamlit Web UI** - Delivers a modern web-based interface with session management and document upload capability.

• **Configuration Management** - Handles environment-based settings for LLM providers, memory strategies, reasoning approaches, and application parameters.

• **Logging Infrastructure** - Tracks system events, errors, and debug information across all components.

• **File Handling** - Loads and parses document files from the data directory in multiple formats.

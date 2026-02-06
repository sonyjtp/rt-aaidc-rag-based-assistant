# 🤖 RAG-Based AI Assistant

> A production-ready Retrieval-Augmented Generation (RAG) chatbot that answers questions exclusively from a set of custom documents using LangChain, ChromaDB, and multiple LLM providers.

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![Code Coverage](https://img.shields.io/badge/coverage-94.41%25-brightgreen.svg)]()
[![Pylint](https://github.com/sonyjtp/rag-based-assistant/actions/workflows/pylint.yml/badge.svg)](https://github.com/sonyjtp/rag-based-assistant/actions/workflows/pylint.yml)

[Quick Start](#-quick-start) • [Features](#-features) • [Installation](#-installation)


---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Project Architecture](#-project-architecture)
- [Project Structure](#-project-structure)
- [Customization Guide](#-customization-guide)
- [Troubleshooting](#-troubleshooting)
- [Documentation](#-documentation)
- [License](#-license)


---

## 🎯 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** chatbot that:

- 📚 **Loads custom documents** from your `data/` directory
- 🔍 **Chunking**: Split documents into chunks and add metadata.
- 💾 **Storage**: Store each chunk's embedding (vector), the chunk text, and metadata in ChromaDB for retrieval.
- 🎤 **Answers questions** exclusively from your documents
- 🧠 **Maintains conversation** memory across multiple interactions
- 🔌 **Supports multiple LLMs**: OpenAI, Groq, Google Gemini
- 🛡️ **Prevents hallucination** with strict prompt constraints
- 📊 **Tracks reasoning** with configurable strategies

**Key Constraint**: The assistant **only answers questions based on the provided documents**. Questions that cannot be answered from the documents are rejected with: *"I'm sorry, that information is not known to me."*

---

## ✨ Features

### Core RAG Capabilities
- ✅ Document loading from text files
- ✅ Intelligent text chunking with overlap
- ✅ Semantic search using ChromaDB's embedding-based similarity search
- ✅ Context-aware question answering
- ✅ Document metadata preservation (title, tags, filename)

### Memory Management
- ✅ **Buffer Memory** (simple_buffer): Stores full conversation history.
- ✅ **Sliding Window Memory** (summarization_sliding_window) — default: keeps recent messages plus a running summarized history to stay within token limits.
- ✅ **Summarization** (summary): Maintains a running summary of the conversation.
- ✅ **None** (none): Disables conversation memory.

### LLM Integration
- ✅ **OpenAI GPT-4** / GPT-4o-mini
- ✅ **Groq Llama 3.1** (fast inference)
- ✅ **Google Gemini** Pro
- ✅ Automatic fallback to next available provider

### Reasoning Strategies

- ✅ **RAG-Enhanced Reasoning** (rag_enhanced_reasoning) — default: Retrieve relevant documents first, then apply reasoning grounded in those documents; `enabled: true`.
- ✅ **Chain-of-Thought** (chain_of_thought): Step-by-step internal reasoning before the final answer; `enabled: true`.
- ✅ **ReAct** (react): Interleave reasoning and actions (e.g., document retrieval) dynamically; `enabled: false`.
- ✅ **Few-Shot Prompting** (few_shot_prompting): Include examples in the prompt to guide format and style; `enabled: true`.
- ✅ **Metacognitive Prompting** (metacognitive_prompting): Reflect on confidence, limitations, and uncertainty; `enabled: true`.

### Safety & Quality
- ✅ **Hallucination Prevention**: Strict prompt constraints
- ✅ **Input Validation**: Document and query validation
- ✅ **Error Handling**: Comprehensive exception handling
- ✅ **Logging**: Detailed logging throughout
- ✅ **Test Cases**: Code coverage maintained above 85%

### User Interfaces
- ✅ **CLI Interface** (`app.py`): Command-line chatbot
- ✅ **Streamlit UI** (`streamlit_app.py`): Web-based interface
- ✅ **API Ready**: Can be integrated with FastAPI/Flask

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** (Tested with 3.12.12 ✅)
- **API Key** for at least one LLM provider:
  - OpenAI: `OPENAI_API_KEY`
  - Groq: `GROQ_API_KEY`
  - Google: `GOOGLE_API_KEY`

### 1️⃣ Clone & Setup (2 minutes)

```bash
# Clone the repository
git clone https://github.com/sonyjtp/rag-based-assistant.git
cd rag-based-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Configure API Key (1 minute)

```bash
# Copy example env file
cp .env_example .env

# Edit .env with your API key
# Choose ONE provider:
# Option 1: OpenAI
OPENAI_API_KEY=your_openai_key_here

# Option 2: Groq (recommended - fast and free)
GROQ_API_KEY=your_groq_key_here

# Option 3: Google Gemini
GOOGLE_API_KEY=your_google_key_here
```

### 3️⃣ Add Your Documents (2 minutes)

```bash
# Replace sample files in data/ with your documents
# Files should be .txt format

ls data/
# Output: your_document.txt, another_doc.txt, ...
```

### 4️⃣ Run the Assistant (30 seconds)

**CLI Version:**
```bash
python src/app.py
```

**Web UI (Streamlit):**
```bash
streamlit run src/streamlit_app.py
```

> 📖 For a detailed walkthrough of the web interface, see [UI_GUIDE.md](UI_GUIDE.md).

---

## 📦 Installation

### Full Installation with Development Tools

```bash
# Clone repository
git clone https://github.com/yourusername/rt-aaidc-rag-based-assistant.git
cd rt-aaidc-rag-based-assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install development/test dependencies (optional)
pip install -r requirements-dev.txt

# Set up pre-commit hooks for automatic code formatting
pre-commit install
```

---

## ⚙️ Configuration

See [Quick Start](#-quick-start) for environment variable setup (`OPENAI_API_KEY`, `GROQ_API_KEY`, `GOOGLE_API_KEY`).

For advanced configuration options, see:
- `src/config.py` — Core settings (chunk size, embedding model, LLM selection)
- `config/memory_strategies.yaml` — Memory strategy definitions
- `config/reasoning_strategies.yaml` — Reasoning approach configurations
- `config/prompt-config.yaml` — System prompts and safety constraints

### ChromaDB Quota Limitation

**Important**: ChromaDB's free tier has a quota limit of **300 records maximum**. When you exceed this limit, you'll encounter an error.

The current batch size is set to 300 records ( `INSERT_BATCH_SIZE = 300` in `src/config.py` ). This limitation affects latency of initial ingestion.


---

## 💬 Usage

### CLI Usage

```bash
python src/app.py

# Prompts you to ask questions
# Type 'quit' to exit

> What is the main topic of the documents?
Assistant: Based on the documents, the main topics are...

> Tell me more
Assistant: [Provides additional context from memory]

> quit
Goodbye!
```

### Streamlit Web Interface

```bash
streamlit run src/streamlit_app.py

# Opens http://localhost:8501
# - Sidebar: Clear history, configure settings
# - Main: Chat interface
# - Auto-saves conversation
```


---

## 🏗️ Project Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      USER INTERFACE LAYER                           │
│  ┌──────────────────────┐              ┌──────────────────────┐     │
│  │   CLI Interface      │              │  Streamlit Web UI    │     │
│  │   (app.py)           │              │  (streamlit_app.py)  │     │
│  │   • Local shell      │              │  • Browser-based     │     │
│  │   • Direct queries   │              │  • Chat history      │     │
│  └──────────────┬───────┘              └───────────┬──────────┘     │
└─────────────────┼──────────────────────────────────┼────────────────┘
                  │                                  │
                  └──────────────┬───────────────────┘
                                 │
                  ┌──────────────▼──────────────┐
                  │   REQUEST PROCESSING LAYER  │
                  │ ┌────────────────────────┐  │
                  │ │  Persona Handler       │  │
                  │ │  • Meta-question       │  │
                  │ │    detection           │  │
                  │ │  • README extraction   │  │
                  │ │  • Answer validation   │  │
                  │ └────────────────────────┘  │
                  └──────────────┬──────────────┘
                                 │
                  ┌──────────────▼──────────────┐
                  │  RAG Assistant CORE         │
                  │  • invoke(query)            │
                  │  • add_documents()          │
                  └──────────┬──────────────────┘
                             │
         ┌───────────────────┼──────────────────┐
         │                   │                  │
    ┌────▼────────┐    ┌─────▼─────┐    ┌───────▼──────┐
    │Search       │    │Query      │    │Hallucination │
    │Manager      │    │Processor  │    │Prevention    │
    │─────────────│    │───────────│    │──────────────│
    │•Add docs    │    │•Context   │    │•Similarity   │
    │•Search      │    │ augment   │    │ threshold    │
    │•VectorDB    │    │•Memory    │    │•Context      │
    │ delegation  │    │ retrieval │    │ validation   │
    │•Flatten     │    │•Query     │    │•LLM-based    │
    │ results     │    │ refinement│    │ relevance    │
    │•LLM context │    │           │    │ check        │
    │ validation  │    │           │    │•Error        │
    │             │    │           │    │ responses    │
    └────┬────────┘    └─────┬─────┘    └───────┬──────┘
         │                   │                  │
         └───────────────────┼──────────────────┘
                             │
        ┌────────────────────▼─────────────────┐
        │   LANGUAGE & REASONING LAYER         │
        │ ┌─────────────────────────────────┐  │
        │ │  Prompt Builder                 │  │
        │ │  • System prompts               │  │
        │ │  • Constraints enforcement      │  │
        │ │  • Response formatting          │  │
        │ └─────────────────────────────────┘  │
        │ ┌─────────────────────────────────┐  │
        │ │  Reasoning Strategy Loader      │  │
        │ │  • Chain-of-Thought             │  │
        │ │  • ReAct                        │  │
        │ │  • Few-Shot Prompting           │  │
        │ │  • Metacognitive Reasoning      │  │
        │ │  • RAG-Enhanced (default)       │  │
        │ └─────────────────────────────────┘  │
        │ ┌─────────────────────────────────┐  │
        │ │  LLM Integration                │  │
        │ │  ┌──────────────────────────────┤  │
        │ │  Provider Selection:            │  │
        │ │  • OpenAI (GPT-4/4o-mini)       │  │
        │ │  • Groq (Llama 3.1)             │  │
        │ │  • Google Gemini Pro            │  │
        │ │  • Auto-fallback logic          │  │
        │ │  └──────────────────────────────┤  │
        │ └─────────────────────────────────┘  │
        └────────────────────┬─────────────────┘
                             │
        ┌────────────────────▼─────────────────┐
        │   KNOWLEDGE BASE LAYER               │
        │ ┌─────────────────────────────────┐  │
        │ │  Search Manager                 │  │
        │ │  • Add documents to VectorDB    │  │
        │ │  • Search for relevant docs     │  │
        │ │  • Flatten nested results       │  │
        │ │  • Log retrieval scores         │  │
        │ │  • LLM-based context            │  │
        │ │    relevance validation         │  │
        │ │  • VectorDB orchestration       │  │
        │ └─────────────────────────────────┘  │
        │ ┌─────────────────────────────────┐  │
        │ │  VectorDB (ChromaDB)            │  │
        │ │  • Document chunks storage      │  │
        │ │  • Semantic search              │  │
        │ │  • Metadata indexing            │  │
        │ │  • Limit: 300 records (free)    │  │
        │ └─────────────────────────────────┘  │
        │ ┌─────────────────────────────────┐  │
        │ │  Embeddings & Storage           │  │
        │ │  • Document chunks converted to │  │
        │ │    embeddings for storage       │  │
        │ │  • HuggingFace Transformers     │  │
        │ │  • all-mpnet-base-v2 (default)  │  │
        │ └─────────────────────────────────┘  │
        └────────────────────┬─────────────────┘
                             │
        ┌────────────────────▼─────────────────┐
        │   STATE MANAGEMENT LAYER             │
        │ ┌─────────────────────────────────┐  │
        │ │  Memory Manager                 │  │
        │ │  • Strategy Pattern:            │  │
        │ │    - SlidingWindow (default)    │  │
        │ │    - SimpleBuffer               │  │
        │ │    - Summarization              │  │
        │ │    - None                       │  │
        │ │  • Token limit management       │  │
        │ │  • Conversation history         │  │
        │ └─────────────────────────────────┘  │
        └──────────────────────────────────────┘

**System Architecture Overview:**

The system is organized into 7 interconnected layers that work together to process user queries and generate accurate answers:

1. **User Interface Layer**: Handles interaction through CLI and Streamlit web interface, accepting user queries and displaying responses.

2. **Request Processing Layer**: The Persona Handler detects meta-questions (questions about the system itself) and routes them to README extraction if needed.

3. **RAG Assistant Core**: The main orchestrator that coordinates between search, query processing, and hallucination prevention components.

4. **Core Processing Components**: Search Manager retrieves relevant documents from the vector database, Query Processor augments queries with conversation history, and Hallucination Prevention validates that retrieved documents are relevant using similarity thresholds.

5. **Language & Reasoning Layer**: Combines prompt building with multiple reasoning strategies (Chain-of-Thought, ReAct, Few-Shot, etc.) and integrates with LLM providers (OpenAI, Groq, Google Gemini).

6. **Knowledge Base Layer**: Manages document storage and semantic search through ChromaDB using embeddings for similarity matching.

7. **State Management Layer**: Maintains conversation history using configurable memory strategies (Sliding Window, Buffer, Summarization) to preserve context across interactions.

┌─────────────────────────────────────────────────────────────────────┐
│   CROSS-CUTTING CONCERNS & UTILITIES                                │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │File Utils   │  │Logger        │  │Config       │  │Error       │ │
│  │• Load docs  │  │• Structured  │  │• Centralized│  │Messages    │ │
│  │• Parse text │  │  logging     │  │  settings   │  │• User-     │ │
│  │• Chunking   │  │• File write  │  │• YAML cfg   │  │  friendly  │ │
│  │• Metadata   │  │• Debug mode  │  │• Env vars   │  │  responses │ │
│  └─────────────┘  └──────────────┘  └─────────────┘  └────────────┘ │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────────────┐ │
│  │UI Utils     │  │String Utils  │  │README Extractor             │ │
│  │• Styling    │  │• Validation  │  │• Dynamic content extraction │ │
│  │• Colors     │  │• Formatting  │  │• Multi-section support      │ │
│  │• Layouts    │  │• Sanitizing  │  │• Fallback handling          │ │
│  └─────────────┘  └──────────────┘  └─────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│   DATA FLOW: QUESTION → ANSWER                                       │
│                                                                      │
│  1. User Query                                                       │
│       ↓                                                              │
│  2. Persona Handler (Meta-question check?)                           │
│       ├─ YES → Extract from README & return                          │
│       └─ NO → Continue                                               │
│       ↓                                                              │
│  3. Query Processor (Augment with chat history)                      │
│       ↓                                                              │
│  4. Search Manager (Retrieve relevant documents)                     │
│       ├─ Convert to embedding                                        │
│       ├─ Search VectorDB                                             │
│       └─ Rank & flatten results                                      │
│       ↓                                                              │
│  5. Hallucination Prevention (Validate similarity)                   │
│       ├─ Similarity ≥ threshold? → Continue                          │
│       └─ NO match? → Return "not known to me"                        │
│       ↓                                                              │
│  6. Reasoning Strategy (Decide how to reason)                        │
│       ↓                                                              │
│  7. Prompt Builder (Create system + user prompt)                     │
│       ↓                                                              │
│  8. LLM Provider (Generate response)                                 │
│       └─ Auto-fallback if provider fails                             │
│       ↓                                                              │
│  9. Memory Manager (Store turn in history)                           │
│       ↓                                                              │
│  10. Return to User                                                  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Query
    │
    ▼
Meta-Question Detection?
    ├─ Keywords: "what topics", "what can you", "what do you know"
    │ ├─ YES: Allow lower similarity matches
    │ └─ NO: Require high similarity (distance <= 0.35, i.e., similarity >= 0.65)
    │
    ▼
Document Search (VectorDB)
    │
    ├─► Search for similar documents (k results)
    ├─► Return ranked results with distances
    │
    ▼
Similarity Validation ⚡ (Hallucination Prevention)
    │
    ├─ Check: distance <= threshold?
    │ ├─ META-QUESTION: Allow any distance
    │ ├─ REGULAR QUESTION: Must pass threshold
    │ └─ NO MATCH: Return "couldn't find information" → END
    │
    ▼
Context Building
    │
    ├─► Extract and flatten documents
    ├─► Combine with conversation history (from Memory)
    ├─► Add system prompts & constraints
    ├─► Apply reasoning strategy
    │
    ▼
LLM Processing
    │
    ├─► Chain: [Prompt Template → LLM → Output Parser]
    ├─► Generate response grounded in context
    │
    ▼
Memory Update
    │
    ├─► Save Q&A pair to conversation history
    ├─► Apply memory strategy:
    │   ├─ SlidingWindow: Summarize when window full
    │   ├─ SimpleBuffer: Keep recent messages
    │   └─ Summary: Maintain running summary
    │
    ▼
Response to User ✅
    │
    └─► Return context-grounded answer
```

**Data Flow Overview:**

The question-to-answer flow follows a 10-step process:

1. **User Query**: The user asks a question through CLI or Streamlit interface.

2. **Meta-Question Detection**: The Persona Handler checks if the question is about the system itself (like "What features do you have?"). If yes, it extracts the answer from the README and returns it immediately.

3. **Query Augmentation**: For regular questions, the Query Processor augments the query with relevant conversation history from the Memory Manager to maintain context for follow-up questions.

4. **Document Search**: The Search Manager converts the query to an embedding and searches the ChromaDB vector database for semantically similar document chunks, returning ranked results.

5. **Similarity Validation**: The Hallucination Prevention module validates that the retrieved documents meet the similarity threshold. Meta-questions allow lower thresholds, but regular questions require high similarity. If no relevant documents are found, the system returns "I'm sorry, that information is not known to me."

6. **Reasoning Strategy Selection**: Based on configuration, a reasoning strategy is selected (Chain-of-Thought, ReAct, Few-Shot Prompting, etc.).

7. **Prompt Construction**: The Prompt Builder combines system prompts, constraints, reasoning instructions, context from documents, and the user question into a complete prompt.

8. **LLM Processing**: The prompt is sent to the selected LLM provider (OpenAI, Groq, or Google Gemini). If the provider fails, the system automatically falls back to the next available provider.

9. **Memory Update**: The question-answer pair is stored in the conversation history using the configured memory strategy (Sliding Window summarization, simple buffer, or running summary).

10. **Response Delivery**: The generated answer is returned to the user, maintaining context for potential follow-up questions.

---

## 📁 Project Structure

```
rag-based-assistant/
│
├── src/                          # Source code modules
├── config/                       # Configuration YAML files
├── data/                         # Document storage
├── tests/                        # Test suite
├── logs/                         # Application logs
├── static/                       # CSS and styling
│
├── requirements.txt              # Production dependencies
├── requirements-test.txt         # Testing dependencies
├── requirements-dev.txt          # Development tools
├── pytest.ini                    # Pytest configuration
├── .pylintrc                     # Pylint configuration
├── .pre-commit-config.yaml       # Pre-commit hooks
├── .env_example                  # Example environment variables
│
├── update_coverage.py            # Coverage badge script
├── UI_GUIDE.md                   # Streamlit UI guide
├── README.md                     # This file
└── LICENSE                       # License
```

---

## 🧪 Testing

### Run Full Test Suite

```bash
pytest -v

# Run with coverage report
pytest --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Pre-Commit Testing

Before you commit, the following checks run automatically:

```bash
# Install pre-commit hooks (one-time setup)
pre-commit install

# Manual run of all checks
pre-commit run --all-files

# Pre-commit checks include:
# ✅ Standard checks (trailing whitespace, file endings, YAML, merge conflicts)
# ✅ Code formatting (Black, isort)
# ✅ Code linting (Flake8, Pylint ≥9.5 score)
# ✅ Tests (pytest - all tests must pass)
# ✅ Coverage (minimum 90% required)
```

**If a check fails**, fix the issues and commit again. Most checks (Black, isort, end-of-file-fixer) auto-fix issues, so you may need to stage the changes and retry.

**Note**: Commits will be rejected if test coverage drops below 90%. To bypass (not recommended):
```bash
git commit --no-verify  # Skip pre-commit hooks
```

### Coverage Requirements

- **Minimum Coverage**: 90% (enforced by pre-commit hooks)

### Run Specific Tests

```bash
# Test RAG assistant
pytest tests/test_rag_assistant.py -v

# Test prompt building
pytest tests/test_prompt_builder.py -v

# Test hallucination prevention
pytest tests/test_hallucination_prevention.py -v

# Test memory management
pytest tests/test_memory_manager.py -v
```

### Coverage Badge Updates

The coverage badge in the README is automatically updated in CI/CD:

```bash
# Manual update (for local development)
python update_coverage.py

# This script:
# 1. Reads coverage.xml (generated by pytest)
# 2. Extracts coverage percentage
# 3. Updates README badge with current coverage
# 4. Colors badge based on coverage level (green/yellow/red)
```

The badge is updated:
- ✅ On every push to main (via GitHub Actions)
- ✅ Before pull requests (verify coverage meets threshold)
- ✅ Manually via `python update_coverage.py`


## 🎛️ Customization Guide

### Change Memory Strategy

Edit `config.py` to change the memory strategy:

```python
# In src/config.py
MEMORY_STRATEGY = "summarization_sliding_window"  # Options: summarization_sliding_window, simple_buffer, summary, none
```

See [Features](#-features) section for memory strategy details.

### Switch LLM Provider

```bash
# In .env - set which API key to use
OPENAI_API_KEY=...    # Uses OpenAI
```

See [Features](#-features) section for LLM provider details.

### Adjust Document Chunking

```python
# In src/config.py
CHUNK_SIZE_DEFAULT = 2000          # Larger chunks
CHUNK_OVERLAP_DEFAULT = 400        # More overlap for context
RETRIEVAL_K_DEFAULT = 10           # Retrieve more documents
```

### Configure Reasoning Strategy

See [Customization Guide](#-customization-guide) section for detailed reasoning strategy configuration.

### Add Custom Prompts

```python
# In src/prompt_builder.py
def build_system_prompts():
    return [
        "Your custom instruction 1",
        "Your custom instruction 2",
        # ... existing prompts
    ]
```

---


## ❓ Troubleshooting

| Issue                | Solution                                                                 |
|----------------------|--------------------------------------------------------------------------|
| API Key not found    | Set `OPENAI_API_KEY`, `GROQ_API_KEY`, or `GOOGLE_API_KEY` in `.env`      |
| No documents found   | Add `.txt` files to `data/` directory or use `assistant.add_documents()` |
| Token limit exceeded | Reduce `CHUNK_SIZE` or enable memory summarization in config             |
| Low answer quality   | Increase `RETRIEVAL_K_DEFAULT` to retrieve more documents                |
| Hallucination issues | Ensure documents are loaded and similarity threshold is set correctly    |

---

### Debug Mode

```bash
# Enable detailed logging
# In log_manager.py, set logging level
logging.basicConfig(level=logging.DEBUG)

# Run with verbose output
pytest -v --log-cli-level=DEBUG
```

---



### Development Setup

```bash
# Fork and clone
git clone https://github.com/sonyjtp/rag-based-assistant.git
cd rag-based-assistant

# Create feature branch
git checkout -b feature/amazing-feature

# Install dev dependencies
pip install -r requirements-test.txt

# Make changes and run tests
pytest tests/ -v

# Commit and push
git add .
git commit -m "feat: add amazing feature"
git push origin feature/amazing-feature

# Create pull request on GitHub
```

### Testing Requirements

All contributions must include:
- ✅ Unit tests for new functionality
- ✅ Integration tests if applicable
- ✅ Documentation updates
- ✅ All tests must pass: `pytest -v`

### Code Style

- Follow PEP 8
- Use type hints
- Write docstrings
- Comment complex logic

---

## 📚 Documentation

This project includes comprehensive documentation for different aspects:

### User Documentation
- **[UI_GUIDE.md](UI_GUIDE.md)** — Complete guide to the web interface (Streamlit)
  - Features, components, and user workflows
  - Styling and customization
  - Troubleshooting and performance tips

### Configuration Documentation
- See [Configuration](#-configuration) section for details on:
  - `config/reasoning_strategies.yaml` — Reasoning approach configurations
  - `config/memory_strategies.yaml` — Memory strategy definitions
  - `config/prompt-config.yaml` — System prompts and safety constraints

---

## 📄 License

This project is licensed under **CC BY-NC-SA 4.0** (Creative Commons Attribution-NonCommercial-ShareAlike 4.0).

**Summary**: Attribution required • Non-commercial only • Modifications must use same license

See [LICENSE](LICENSE) file for full details.

---

## 🎓 Author

**Sony Jacob Thomas**

---

**Last Updated**: January 2026
**Status**: 🛠️ Under Active Development

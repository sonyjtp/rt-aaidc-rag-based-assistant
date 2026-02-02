# RAG-Based AI Assistant - Architecture Diagram (Mermaid)

## System Architecture Flowchart

```mermaid
%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TD
    A["👤 User Input"] --> B["🎯 CLI Interface<br/>app.py"]
    A --> C["🌐 Streamlit Web UI<br/>streamlit_app.py"]
    
    B --> D{🔍 Meta-Question?}
    C --> D
    
    D -->|Yes| E["✅ Persona Handler<br/>Extract README Content"]
    D -->|No| F["🧠 RAGAssistant CORE<br/>invoke query<br/>add_documents"]
    
    E --> Z["📤 Return to User"]
    
    F --> G["🔎 Search Manager<br/>Search VectorDB<br/>Flatten results"]
    
    G --> J["⚡ Vector Database<br/>Document chunks<br/>Semantic search<br/>Document indexing"]
    G --> K["🧮 Embeddings Model<br/>Semantic encoding<br/>Device-optimized"]
    
    K --> L{Device<br/>Detection}
    L -->|GPU| M["🚀 Accelerated"]
    L -->|CPU| O["💻 CPU Fallback"]
    
    G --> P["✔️ Context Retrieved"]
    P --> H{🔄 Validate<br/>Context?}
    
    H -->|Valid| Q["🤖 Prompt Builder<br/>System Prompts & Formatting"]
    H -->|Invalid| Z
    
    Q --> R["🎯 Reasoning Strategy<br/>Multi-approach support<br/>Response planning"]
    
    R --> S["🔗 LLM Integration<br/>Multi-provider support<br/>Auto-fallback<br/>Model selection"]
    
    S --> T["💬 LLM Response<br/>Generation"]
    
    T --> U["📚 Memory Manager<br/>Configurable strategies<br/>Conversation history"]
    
    U --> Z
    
    style A fill:#e1f5ff
    style B fill:#fff3e0
    style C fill:#fff3e0
    style D fill:#fff3e0
    style E fill:#f3e5f5
    style F fill:#ffe0b2
    style G fill:#ffccbc
    style H fill:#fff59d
    style J fill:#b3e5fc
    style K fill:#b3e5fc
    style L fill:#fff59d
    style M fill:#a5d6a7
    style O fill:#a5d6a7
    style P fill:#c8e6c9
    style Q fill:#f0f4c3
    style R fill:#f0f4c3
    style S fill:#d1c4e9
    style T fill:#d1c4e9
    style U fill:#b2dfdb
    style Z fill:#e1f5ff
```

## Component Interaction Diagram

```mermaid
%%{init: {'flowchart': {'curve': 'orthogonal', 'padding': '20'}}%%
graph TD
    CLI["CLI Interface"]
    WEB["Streamlit Web UI"]
    PH["Persona Handler"]
    RAG["RAGAssistant Core"]
    RC["README Extraction"]
    SM["Search Manager"]
    VDB["Vector Database"]
    EMB["Embeddings"]
    QP["Query Processor"]
    PB["Prompt Builder"]
    RS["Reasoning Strategies"]
    LLM["LLM Providers"]
    MEM["Memory Manager"]

    CLI --> PH
    WEB --> PH
    PH --> RAG
    PH -.->|internal| RC
    RAG --> SM
    SM -.->|internal| VDB
    VDB -.->|internal| EMB
    SM --> QP
    QP --> PB
    PB --> RS
    RS --> LLM
    LLM --> MEM

    style CLI fill:#e3f2fd
    style WEB fill:#e3f2fd
    style PH fill:#f3e5f5
    style RC fill:#c8e6c9
    style RAG fill:#ffe0b2
    style SM fill:#ffccbc
    style QP fill:#ffccbc
    style VDB fill:#b3e5fc
    style EMB fill:#b3e5fc
    style PB fill:#f0f4c3
    style RS fill:#f0f4c3
    style LLM fill:#d1c4e9
    style MEM fill:#b2dfdb
```

## Query Execution FlowAdd

```mermaid
sequenceDiagram
    participant User
    participant UI as UI Layer
    participant PH as Persona Handler
    participant SM as Search Manager
    participant VDB as VectorDB
    participant QP as Query Processor
    participant PB as Prompt Builder
    participant RS as Reasoning Strategy
    participant LLM as LLM Provider
    participant MEM as Memory Manager

    User->>UI: Ask Question
    UI->>PH: Route Query
    
    alt Is Meta-Question?
        PH->>PH: Detect Meta-Question Pattern
        PH->>PH: Extract from README
        PH->>UI: Return Answer
        UI->>User: Display Response
    else Regular Question
        PH->>SM: Pass Query to Search Manager
        SM->>VDB: Convert to Embedding
        VDB->>VDB: Cosine Similarity Search
        SM->>SM: Flatten Results
        SM->>SM: Log Scores
        SM->>QP: Pass Context
        
        QP->>QP: Validate Context (LLM check)
        
        alt Context Valid?
            QP->>PB: Send to Prompt Builder
            PB->>RS: Apply Reasoning
            RS->>LLM: Create Prompt
            LLM->>LLM: Generate Response
            LLM->>MEM: Store in History
            MEM->>UI: Return Response
            UI->>User: Display Answer
        else Context Invalid
            QP->>UI: "Not Known to Me"
            UI->>User: Display Error
        end
    end
```

## Document Ingestion Flow

```mermaid
sequenceDiagram
    participant UI as UI Layer
    participant RAG as RAGAssistant
    participant FU as File Utils
    participant SM as Search Manager
    participant VDB as VectorDB
    participant EMB as Embeddings
    participant ChDB as ChromaDB

    UI->>FU: Load Documents from Folder
    FU->>FU: Read Document Files
    FU->>FU: Parse Document Content
    FU->>UI: Return Document List
    
    UI->>RAG: add_documents(documents)
    RAG->>SM: add_documents(documents)
    SM->>VDB: add_documents(documents)
    
    VDB->>VDB: Standardize Documents
    
    rect rgb(200, 230, 255)
        note right of VDB: Chunking Phase
        VDB->>VDB: Split Documents
        VDB->>VDB: Apply Chunking Configuration
        VDB->>VDB: Deduplicate Chunks
    end
    
    VDB->>EMB: Embed Chunks
    EMB->>EMB: Convert Text to Vectors
    EMB->>VDB: Return Embeddings
    
    rect rgb(230, 245, 200)
        note right of VDB: Batch Insert Phase
        VDB->>VDB: Prepare Batches
        VDB->>VDB: Apply Batch Configuration
        VDB->>ChDB: Insert Batch 1 (Chunks + Embeddings)
        ChDB->>ChDB: Index with HNSW<br/>space: cosine
        VDB->>ChDB: Insert Batch 2 (Chunks + Embeddings)
        ChDB->>ChDB: Index with HNSW<br/>space: cosine
        VDB->>ChDB: Insert Batch N (Chunks + Embeddings)
        ChDB->>ChDB: Index with HNSW<br/>space: cosine
    end
    
    VDB->>SM: Confirm Complete
    SM->>RAG: Confirm Complete
    RAG->>UI: Documents Indexed

    rect rgb(200, 220, 255)
        note right of ChDB: Vector DB is now ready<br/>for semantic search queries
    end
```

## Layer Architecture Diagram

```mermaid
%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TD
    subgraph UI["Layer 1: User Interface"]
        CLI["CLI<br/>app.py"]
        WEB["Streamlit<br/>streamlit_app.py"]
    end
    
    subgraph RP["Layer 2: Request Processing"]
        PH["Persona Handler<br/>Meta-Q Detection<br/>README Extraction"]
    end
    
    subgraph CO["Layer 3: Core Orchestration"]
        RAG["RAGAssistant Core<br/>invoke<br/>add_documents"]
    end
    
    subgraph PROC["Layer 4: Core Processors"]
        SM["Search Manager<br/>Search VectorDB<br/>Flatten results"]
        QP["Query Processor<br/>Validate Context<br/>Memory retrieval"]
    end
    
    subgraph LR["Layer 5: Language & Reasoning"]
        PB["Prompt Builder<br/>System prompts<br/>Constraints<br/>Formatting"]
        RS["Reasoning Strategies<br/>Multi-approach support<br/>Response planning"]
        LLM["LLM Integration<br/>Multi-provider support<br/>Auto-fallback"]
        DD["Device Detection<br/>GPU support<br/>CPU fallback"]
    end
    
    subgraph KB["Layer 6: Knowledge Base"]
        VDB["Vector Database<br/>Document storage<br/>Semantic search"]
        EMB["Embeddings<br/>Semantic encoding<br/>Device-optimized"]
    end
    
    subgraph SM2["Layer 7: State Management"]
        MEM["Memory Manager<br/>Configurable strategies<br/>Conversation history"]
    end
    
    subgraph UTIL["Utilities: Cross-Cutting"]
        FU["File Utils<br/>Document loading"]
        CFG["Config<br/>Settings"]
        EM["Error Messages<br/>User responses"]
        UI["UI Utils<br/>Styling"]
        STR["String Utils<br/>Validation"]
        RE["README Extractor<br/>Content extraction"]
    end
    
    UI --> RP
    RP --> CO
    CO --> PROC
    PROC --> LR
    LR --> KB
    KB --> SM2
    
    SM2 --> UTIL
    LR --> UTIL
    PROC --> UTIL
    
    style UI fill:#e3f2fd
    style RP fill:#f3e5f5
    style CO fill:#ffe0b2
    style PROC fill:#ffccbc
    style LR fill:#f0f4c3
    style KB fill:#b3e5fc
    style SM2 fill:#b2dfdb
    style UTIL fill:#e8eaf6
```

## Component Dependencies Diagram

```mermaid
%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TB
    RAG["RAGAssistant"]
    
    PH["Persona Handler"]
    RE["README Extractor"]
    QP["Query Processor"]
    SM["Search Manager"]
    VDB["VectorDB"]
    EMB["Embeddings"]
    LLM["LLM Provider"]
    
    PB["Prompt Builder"]
    RS["Reasoning Strategies"]
    
    MEM["Memory Manager"]
    
    LOG["Logger"]
    CFG["Config"]
    
    RAG --> PH
    PH --> RE
    RAG --> QP
    RAG --> SM
    
    SM --> VDB
    SM --> EMB
    
    QP --> LLM
    QP --> MEM
    
    MEM --> LLM
    
    PB --> RS
    RS --> LLM
    
    PH -.-> RE
    
    VDB -.-> EMB
    EMB -.-> LLM
    
    RAG -.-> LOG
    PH -.-> LOG
    SM -.-> LOG
    QP -.-> LOG
    
    RAG -.-> CFG
    PH -.-> CFG
    SM -.-> CFG
    
    style RAG fill:#ffe0b2,stroke:#ff6f00,stroke-width:3px
    style PH fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    style RE fill:#c8e6c9,stroke:#388e3c,stroke-width:2px
    style QP fill:#ffccbc,stroke:#d84315,stroke-width:2px
    style SM fill:#ffccbc,stroke:#d84315,stroke-width:2px
    style VDB fill:#b3e5fc,stroke:#0277bd,stroke-width:2px
    style EMB fill:#b3e5fc,stroke:#0277bd,stroke-width:2px
    style LLM fill:#d1c4e9,stroke:#512da8,stroke-width:2px
    style PB fill:#f0f4c3,stroke:#f57f17,stroke-width:2px
    style RS fill:#f0f4c3,stroke:#f57f17,stroke-width:2px
    style MEM fill:#b2dfdb,stroke:#00695c,stroke-width:2px
    style LOG fill:#e8eaf6,stroke:#3f51b5,stroke-width:1px
    style CFG fill:#e8eaf6,stroke:#3f51b5,stroke-width:1px
```

## Query Processing Flow

```mermaid
%%{init: {'flowchart': {'curve': 'linear'}}}%%
flowchart LR
    START["Start: User Query"] --> META{Meta-Question?}
    
    META -->|Yes| DETECT["Detect Pattern"]
    DETECT --> EXTRACT["Extract from README"]
    EXTRACT --> RETURN1["Return Content"]
    RETURN1 --> END["Return to User"]
    
    META -->|No| SEARCH["Search Manager<br/>Retrieve Documents"]
    SEARCH --> EMBED["Convert to Embedding"]
    EMBED --> VDB["Search VectorDB"]
    VDB --> FLATTEN["Flatten Results"]
    FLATTEN --> VALIDATE["Query Processor<br/>Validate Context"]
    
    VALIDATE -->|Valid| REASON["Reasoning Strategy<br/>Select Approach"]
    VALIDATE -->|Invalid| NOTKNOWN["Return:<br/>Not Known to Me"]
    
    REASON --> PROMPT["Prompt Builder<br/>Create Prompts"]
    PROMPT --> LLM["LLM Provider<br/>Generate Response"]
    LLM --> MEMORY["Memory Manager<br/>Store in History"]
    MEMORY --> RETURN2["Return Response"]
    RETURN2 --> END
    
    NOTKNOWN --> END
    
    style START fill:#e3f2fd
    style META fill:#fff3e0
    style DETECT fill:#f3e5f5
    style EXTRACT fill:#c8e6c9
    style RETURN1 fill:#c8e6c9
    style SEARCH fill:#ffccbc
    style EMBED fill:#b3e5fc
    style VDB fill:#b3e5fc
    style FLATTEN fill:#ffccbc
    style VALIDATE fill:#ffccbc
    style REASON fill:#f0f4c3
    style PROMPT fill:#f0f4c3
    style LLM fill:#d1c4e9
    style MEMORY fill:#b2dfdb
    style RETURN2 fill:#c8e6c9
    style NOTKNOWN fill:#ffcdd2
    style END fill:#e3f2fd
```

---

## Legend

| Symbol | Meaning                  |
|--------|--------------------------|
| 👤     | User                     |
| 🎯     | Interface/Entry Point    |
| 🌐     | Web Interface            |
| 🔍     | Detection/Analysis       |
| 🧠     | Core Logic               |
| 🔎     | Search Operations        |
| 📝     | Processing               |
| 🛡️    | Safety/Validation        |
| ⚡      | Database                 |
| 🧮     | ML Models                |
| 🚀     | Performance Optimization |
| 🍎     | Platform Specific        |
| 💻     | Fallback                 |
| ✔️     | Validation Success       |
| 🤖     | AI/LLM                   |
| 🔗     | Integration              |
| 💬     | Output                   |
| 📚     | Storage                  |
| 📤     | Return to User           |

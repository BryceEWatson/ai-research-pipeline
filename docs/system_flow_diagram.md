# AI Research Pipeline - System Flow Diagram

## Overview

This document charts the complete system architecture and data flow for the Phase 2 multi-agent AI research pipeline with automatic source persistence and self-improving knowledge base.

---

## High-Level Architecture

```mermaid
graph TD
    User[👤 User] --> CLI[🖥️ CLI Interface]
    CLI --> Pipeline[⚡ RAG Pipeline]
    
    Pipeline --> Seed[🌱 Seed Agent]
    Pipeline --> Sourcing[📡 Sourcing Agent] 
    Pipeline --> Research[🔬 Research Agent]
    
    Sourcing --> Perplexity[🌐 Perplexity Sonar API]
    Sourcing --> VectorDB[(🗄️ Chroma Vector DB)]
    Research --> VectorDB
    
    Pipeline --> User
    
    style User fill:#e1f5fe
    style CLI fill:#f3e5f5
    style Pipeline fill:#e8f5e8
    style Seed fill:#fff3e0
    style Sourcing fill:#e0f2f1
    style Research fill:#f1f8e9
    style Perplexity fill:#ffe0b2
    style VectorDB fill:#e3f2fd
```

---

## Detailed System Flow

### Phase 1: Query Processing & Decomposition

```mermaid
sequenceDiagram
    participant User
    participant CLI
    participant Pipeline
    participant Seed as 🌱 Seed Agent
    participant GPT as 🧠 GPT-4o
    
    User->>CLI: ask "What are quantum computing developments?"
    CLI->>Pipeline: answer_question(query, use_agents=True)
    Pipeline->>Seed: decompose_query(query)
    
    Seed->>GPT: Query decomposition prompt
    Note over GPT: Analyzes complexity:<br/>- Word count: 5<br/>- Temporal aspect: Yes<br/>- Multiple concepts: Yes
    GPT->>Seed: JSON response with 5 sub-queries
    
    Note over Seed: Sub-queries created:<br/>1. "quantum computing breakthroughs 2024"<br/>2. "quantum error correction advances"<br/>3. "quantum hardware developments"<br/>4. "quantum algorithms research"<br/>5. "quantum industry investments"
    
    Seed->>Pipeline: SearchPlan(strategy="live_first", complexity=3)
```

### Phase 2: Multi-Source Information Gathering

```mermaid
sequenceDiagram
    participant Pipeline
    participant Sourcing as 📡 Sourcing Agent
    participant LocalDB as 🗄️ Local Vector DB
    participant Perplexity as 🌐 Perplexity API
    participant Embedding as 🔗 OpenAI Embeddings
    
    Pipeline->>Sourcing: gather_sources(search_plan)
    
    loop For each sub-query
        Note over Sourcing: Strategy: live_first
        
        Sourcing->>Perplexity: POST /chat/completions
        Note over Perplexity: Model: llama-3.1-sonar-small-128k-online<br/>return_citations: true
        Perplexity->>Sourcing: Response with content + search_results[]
        
        Note over Sourcing: Parse response:<br/>- Extract AI-generated content<br/>- Extract search_results metadata<br/>- Create SourcedDocument objects
        
        Sourcing->>LocalDB: similarity_search(sub_query)
        LocalDB->>Sourcing: existing_documents[]
        
        Note over Sourcing: Deduplication check:<br/>- Compare URLs<br/>- Content similarity analysis<br/>- 80% word overlap threshold
    end
    
    Note over Sourcing: Results:<br/>- 25 live sources from Perplexity<br/>- 9 local sources from vector DB<br/>- 28 total unique sources
```

### Phase 3: Automatic Source Persistence (Vision.md Step 6)

```mermaid
sequenceDiagram
    participant Sourcing as 📡 Sourcing Agent
    participant Chunker as ✂️ Text Chunker
    participant Embedding as 🔗 OpenAI Embeddings
    participant VectorDB as 🗄️ Chroma Vector DB
    
    Note over Sourcing: _persist_live_sources() called
    
    loop For each live source (sonar, deep_research)
        Sourcing->>VectorDB: Check if source exists
        VectorDB->>Sourcing: similarity_search(title, filter=source_url)
        
        alt Source doesn't exist
            Note over Sourcing: Create LangChain Document:<br/>- page_content: AI response<br/>- metadata: title, URL, date, eval_score<br/>- auto_added: true, timestamp
            
            Sourcing->>Chunker: chunk_document(content)
            Note over Chunker: Adaptive chunking:<br/>- Base size: 1,500 tokens<br/>- Overlap: 150 tokens (10%)<br/>- Max size: 2,000 tokens
            
            Chunker->>Embedding: embed_chunks()
            Note over Embedding: OpenAI text-embedding-3-small<br/>8k context limit
            
            Embedding->>VectorDB: store_embeddings()
            VectorDB->>Sourcing: document_ids[]
            
            Note over Sourcing: Log: "Persisted live source: 'Title' (X chunks)"
        else Source exists
            Note over Sourcing: Skip - already stored
        end
    end
    
    Note over Sourcing: Final result: 25 newly persisted chunks
```

### Phase 4: Research Synthesis & Citation

```mermaid
sequenceDiagram
    participant Pipeline
    participant Research as 🔬 Research Agent
    participant GPT as 🧠 GPT-4o (Synthesis)
    participant GPT2 as 🧠 GPT-4o (Conflicts)
    participant GPT3 as 🧠 GPT-4o (Critique)
    
    Pipeline->>Research: synthesize_research(query, sourcing_result)
    
    Note over Research: Format sources for synthesis:<br/>- [Local-1] title + content preview<br/>- [Sonar-1] title + content preview<br/>- 28 total sources formatted
    
    Research->>GPT: Synthesis prompt with all sources
    Note over GPT: Generate comprehensive answer<br/>with mixed citations:<br/>[Local-1], [Local-2], [Sonar-1], [Sonar-2]
    GPT->>Research: Synthesized answer with citations
    
    par Conflict Detection
        Research->>GPT2: Conflict detection prompt
        GPT2->>Research: conflicts[] (JSON format)
    and Self-Critique
        Research->>GPT3: Self-critique prompt
        GPT3->>Research: Quality assessment + score
    end
    
    Note over Research: Calculate confidence score:<br/>- Source count factor: 28/5 = 1.0<br/>- Quality factor: avg(eval_scores)<br/>- Conflict penalty: 0 (no conflicts)<br/>- Final confidence: 1.00
    
    Research->>Pipeline: ResearchResult with enhanced metadata
```

### Phase 5: Response Formatting & Display

```mermaid
sequenceDiagram
    participant Pipeline
    participant CLI
    participant Display as 🎨 Rich Display
    participant User
    
    Pipeline->>CLI: Enhanced response object
    Note over CLI: Response contains:<br/>- answer: synthesized content<br/>- confidence_score: 1.00<br/>- total_sources: 28<br/>- source_breakdown: {local: 9, live: 25}<br/>- citations: [Local-X, Sonar-X]<br/>- conflicts: []<br/>- execution_time: 34.78s
    
    CLI->>Display: _display_phase2_details()
    
    Note over Display: Rich formatting:<br/>- Answer panel with citations<br/>- Multi-Agent Analysis table<br/>- Source breakdown<br/>- Citation list with URLs<br/>- Execution metrics
    
    Display->>User: Formatted response with enhanced metadata
    
    Note over User: Receives:<br/>✅ Comprehensive answer<br/>✅ 28 mixed citations<br/>✅ Confidence score<br/>✅ Execution metrics<br/>✅ Source transparency
```

---

## Data Flow Architecture

### Knowledge Base Evolution

```mermaid
graph LR
    subgraph "Initial State"
        KB1[(3 Sample Docs<br/>1,094 chars)]
    end
    
    subgraph "Query Processing"
        Q[Quantum Computing Query]
        Q --> SA[Seed Agent]
        SA --> SQ1[Sub-query 1]
        SA --> SQ2[Sub-query 2]
        SA --> SQ3[Sub-query 3]
        SA --> SQ4[Sub-query 4]
        SA --> SQ5[Sub-query 5]
    end
    
    subgraph "Live Research"
        SQ1 --> P1[Perplexity API]
        SQ2 --> P2[Perplexity API]
        SQ3 --> P3[Perplexity API]
        SQ4 --> P4[Perplexity API]
        SQ5 --> P5[Perplexity API]
        
        P1 --> L1[5 Live Sources]
        P2 --> L2[5 Live Sources]
        P3 --> L3[5 Live Sources]
        P4 --> L4[5 Live Sources]
        P5 --> L5[5 Live Sources]
    end
    
    subgraph "Auto-Persistence"
        L1 --> E[Embed & Store]
        L2 --> E
        L3 --> E
        L4 --> E
        L5 --> E
    end
    
    subgraph "Enhanced State"
        E --> KB2[(28 Total Docs<br/>57,549 chars<br/>+25 Live Sources)]
    end
    
    KB1 --> KB2
    
    style KB1 fill:#ffcdd2
    style KB2 fill:#c8e6c9
    style Q fill:#e1f5fe
    style E fill:#fff3e0
```

### Source Types & Metadata

```mermaid
graph TD
    subgraph "Source Classification"
        Local[📁 Local Sources]
        Live[🌐 Live Sources]
    end
    
    subgraph "Local Source Metadata"
        Local --> LM[title: 'RAG Overview'<br/>source_url: 'example.com'<br/>eval_score: 0.9<br/>source_type: 'local']
    end
    
    subgraph "Live Source Metadata"
        Live --> SM[title: 'Quantum Breakthroughs'<br/>source_url: 'wevolver.com'<br/>eval_score: 0.75<br/>source_type: 'sonar'<br/>auto_added: true<br/>date: '2024-12-29']
    end
    
    subgraph "Citation Generation"
        LM --> LC[[Local-1], [Local-2], [Local-3]]
        SM --> SC[[Sonar-1], [Sonar-2], [Sonar-3]]
    end
    
    subgraph "Mixed Citations"
        LC --> MC[Answer with mixed citations:<br/>'Recent quantum developments [Sonar-1]<br/>build on RAG principles [Local-1]...']
        SC --> MC
    end
    
    style Local fill:#e3f2fd
    style Live fill:#e8f5e8
    style MC fill:#fff3e0
```

---

## Component Interactions

### Agent Communication Pattern

```mermaid
graph LR
    subgraph "Input Processing"
        User[👤 User Query] --> CLI[🖥️ CLI]
        CLI --> Pipeline[⚡ Pipeline]
    end
    
    subgraph "Agent Orchestration"
        Pipeline --> Seed[🌱 Seed Agent]
        Seed --> |SearchPlan| Sourcing[📡 Sourcing Agent]
        Sourcing --> |SourcingResult| Research[🔬 Research Agent]
        Research --> |ResearchResult| Pipeline
    end
    
    subgraph "External Services"
        Sourcing <--> Perplexity[🌐 Perplexity Sonar]
        Sourcing <--> OpenAI1[🔗 OpenAI Embeddings]
        Research <--> OpenAI2[🧠 OpenAI GPT-4o]
    end
    
    subgraph "Persistent Storage"
        Sourcing <--> VectorDB[(🗄️ Chroma Vector DB)]
        Research <--> VectorDB
    end
    
    Pipeline --> CLI
    CLI --> User
    
    style Pipeline fill:#e8f5e8
    style Seed fill:#fff3e0
    style Sourcing fill:#e0f2f1
    style Research fill:#f1f8e9
```

### Error Handling & Fallbacks

```mermaid
graph TD
    Start[User Query] --> Phase2{Agents Available?}
    
    Phase2 -->|Yes| SeedAgent[🌱 Seed Agent]
    Phase2 -->|No| Phase1[📝 Basic RAG Fallback]
    
    SeedAgent --> SourcingAgent[📡 Sourcing Agent]
    SourcingAgent --> LiveAPI{Perplexity API}
    
    LiveAPI -->|Success| Persist[💾 Auto-Persist Sources]
    LiveAPI -->|Failure| LocalOnly[📁 Local Sources Only]
    
    Persist --> ResearchAgent[🔬 Research Agent]
    LocalOnly --> ResearchAgent
    
    ResearchAgent --> Synthesis{Synthesis Success?}
    
    Synthesis -->|Success| Enhanced[✨ Enhanced Response]
    Synthesis -->|Failure| Phase1
    
    Phase1 --> Basic[📄 Basic Response]
    Enhanced --> Display[🎨 Rich Display]
    Basic --> Display
    
    Display --> User[👤 User]
    
    style Enhanced fill:#c8e6c9
    style Basic fill:#ffcdd2
    style Phase1 fill:#fff3e0
```

---

## Performance Metrics

### Execution Timeline

```mermaid
gantt
    title Query Processing Timeline
    dateFormat X
    axisFormat %s
    
    section Query Processing
    CLI Input           : 0, 1s
    
    section Seed Agent
    Query Analysis      : 1s, 8s
    Sub-query Generation: 8s, 9s
    
    section Sourcing Agent
    Local Search        : 9s, 12s
    Perplexity API Calls: 12s, 30s
    Auto-Persistence    : 30s, 35s
    
    section Research Agent
    Synthesis           : 35s, 45s
    Conflict Detection  : 35s, 40s
    Self-Critique       : 40s, 45s
    
    section Response
    Display Formatting  : 45s, 46s
```

### System Growth Metrics

| Metric | Before Query | After Query | Growth |
|--------|-------------|-------------|--------|
| **Total Documents** | 3 | 28 | +833% |
| **Content Volume** | 1,094 chars | 57,549 chars | +5,158% |
| **Unique Sources** | 3 | 23 | +667% |
| **Knowledge Topics** | 7 detected | 7 detected | Enhanced coverage |
| **Source Types** | Local only | Local + Live | Mixed sources |

---

## API Integration Details

### Perplexity Sonar API Flow

```mermaid
sequenceDiagram
    participant SA as Sourcing Agent
    participant API as Perplexity API
    participant Parser as Response Parser
    participant Store as Vector Store
    
    SA->>API: POST /chat/completions
    Note over API: {<br/>  model: "llama-3.1-sonar-small-128k-online",<br/>  messages: [...],<br/>  return_citations: true,<br/>  max_tokens: 2000<br/>}
    
    API->>SA: Response
    Note over SA: {<br/>  choices: [{message: {content: "..."}}],<br/>  citations: ["url1", "url2", ...],<br/>  search_results: [<br/>    {title: "...", url: "...", date: "..."},<br/>    ...<br/>  ]<br/>}
    
    SA->>Parser: _parse_sonar_response()
    Note over Parser: Extract:<br/>- AI-generated content<br/>- Search result metadata<br/>- Create SourcedDocument objects
    
    Parser->>SA: SourcedDocument[]
    SA->>Store: Auto-persist new sources
    Store->>SA: Persistence confirmation
```

### OpenAI Integration Points

```mermaid
graph LR
    subgraph "OpenAI Services Used"
        GPT[🧠 GPT-4o Chat Completions]
        EMB[🔗 text-embedding-3-small]
    end
    
    subgraph "Usage Points"
        Seed[🌱 Seed Agent] --> GPT
        Research[🔬 Research Agent] --> GPT
        Sourcing[📡 Sourcing Agent] --> EMB
        VectorDB[(🗄️ Vector DB)] --> EMB
    end
    
    subgraph "API Calls per Query"
        GPT --> G1[Query Decomposition: 1 call]
        GPT --> G2[Research Synthesis: 1 call]
        GPT --> G3[Conflict Detection: 1 call]
        GPT --> G4[Self-Critique: 1 call]
        
        EMB --> E1[Local Searches: 5 calls]
        EMB --> E2[New Source Embedding: 25 calls]
    end
```

---

## Configuration & Settings

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...                    # GPT-4o + Embeddings
PERPLEXITY_API_KEY=pplx-...             # Sonar API

# Optional  
VECTARA_API_KEY=vtr_...                 # Phase 3 (FCS)
LANGSMITH_API_KEY=lsv2_...              # Phase 3 (Evaluation)

# System Configuration
CHROMA_DB_PATH=./data/chroma_db         # Vector DB storage
LLM_MODEL=gpt-4o                        # Primary LLM
EMBEDDING_MODEL=text-embedding-3-small  # Embedding model

# RAG Parameters
CHUNK_SIZE=1500                         # Base chunk size (tokens)
CHUNK_OVERLAP=150                       # 10% overlap
RETRIEVAL_K=15                          # Documents per search
MIN_EVAL_SCORE=0.35                     # Quality threshold
```

### Agent Configuration

```yaml
seed_agent:
  model: gpt-4o
  temperature: 0.1
  max_tokens: 1500
  complexity_threshold: 3

sourcing_agent:
  local_search_k: 15
  perplexity_model: llama-3.1-sonar-small-128k-online
  rate_limit: 1.0  # seconds between calls
  dedup_threshold: 0.8  # content similarity
  
research_agent:
  model: gpt-4o
  temperature: 0.1
  max_tokens: 2000
  confidence_threshold: 0.8
```

---

## Future Enhancements

### Phase 3: Quality & Evaluation

```mermaid
graph LR
    Current[Phase 2: Multi-Agent] --> Phase3[Phase 3: Quality]
    
    Phase3 --> FCS[Vectara FCS Integration]
    Phase3 --> Eval[LangSmith Evaluation]
    Phase3 --> CRAG[Corrective RAG]
    
    FCS --> Gate[FCS Gate: score ≥ 0.9]
    Eval --> Harness[Automated Test Suite]
    CRAG --> Retry[Re-query on Low Confidence]
```

### Scalability Considerations

```mermaid
graph TD
    subgraph "Current Architecture"
        Local[Local Chroma DB]
        Single[Single Instance]
    end
    
    subgraph "Phase 5: Cloud Deployment"
        Cloud[Chroma Cloud]
        LoadBalancer[Load Balancer]
        Multiple[Multiple Instances]
    end
    
    Local --> Cloud
    Single --> LoadBalancer
    LoadBalancer --> Multiple
    
    subgraph "Performance Optimizations"
        Cache[Response Caching]
        Batch[Batch Embeddings]
        Parallel[Parallel Sub-queries]
    end
```

---

## Conclusion

This system flow diagram documents a **fully functional, self-improving AI research assistant** that automatically:

1. **Decomposes complex queries** into focused sub-questions
2. **Retrieves live information** from authoritative sources via Perplexity
3. **Automatically persists new knowledge** to the vector database
4. **Provides comprehensive answers** with mixed local and live citations
5. **Grows its knowledge base** with every research query

The implementation successfully fulfills the vision outlined in the project documentation, creating a robust foundation for future enhancements in quality evaluation, performance optimization, and cloud deployment.
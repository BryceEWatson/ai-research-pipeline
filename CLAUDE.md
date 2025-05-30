# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multi-agent retrieval-augmented research assistant that combines cloud LLM reasoning, live research APIs, and persistent vector memory. The system implements a three-agent workflow:

- **Seed Agent**: Query decomposition & search planning  
- **Sourcing Agent**: Calls research APIs, filters and evaluates content
- **Research Agent**: Retrieval, synthesis, and conflict detection

The project has completed **Phase 3** (Quality & Evaluation with Model Selection) with intelligent model routing, performance monitoring, and quality assurance systems fully implemented.

## Essential Commands

### Installation and Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Setup environment (copy .env.example to .env and configure API keys)
cp .env.example .env
# Edit .env with your OPENAI_API_KEY and other API keys

# Quick start and environment check
python scripts/quickstart.py

# Setup sample data for testing
python scripts/setup_sample_data.py
```

### CLI Usage
```bash
# Main CLI entry point
python app/cli.py --help

# Research and knowledge management
python app/cli.py ask "Your research question here"
python app/cli.py ask "What is retrieval-augmented generation?" --verbose

# Add documents to knowledge base
python app/cli.py add "Document content" --title "Document Title" --url "source-url"
python app/cli.py add-file path/to/document.txt --title "File Title"

# Knowledge base management
python app/cli.py stats              # Show basic KB statistics
python app/cli.py report             # Generate comprehensive knowledge report
python app/cli.py report --detailed  # Show detailed document information
python app/cli.py report --export kb_report.json  # Export report to JSON
python app/cli.py reset              # Reset knowledge base (with confirmation)

# Model management and monitoring (NEW)
python app/cli.py models                                    # View model configuration
python app/cli.py performance                              # System performance metrics
python app/cli.py performance --component seed_agent       # Component-specific metrics
python app/cli.py performance --export metrics.json        # Export performance data
python app/cli.py select-model research --complexity 4     # Test model selection
```

### Testing
```bash
# Quick smoke tests for Phase 2 multi-agent system (fast validation)
python test_smoke.py

# Comprehensive end-to-end tests for Phase 2 (full user workflow validation)
python test_phase2_e2e.py

# Basic functionality tests (no API keys required)
python test_functionality.py

# Minimal pipeline test
python test_minimal.py

# Core pipeline test
python tests/test_basic_pipeline.py
```

## Architecture

### Core Components

- **`core/`**: Core orchestration logic
  - `config.py`: Environment configuration with Pydantic settings
  - `rag_pipeline.py`: Enhanced RAG pipeline with multi-agent orchestration
  - `vector_store.py`: Chroma DB wrapper for vector storage
  - `chunking.py`: Adaptive text chunking with OpenAI embeddings

- **`agents/`**: Multi-agent system (Phase 2)
  - `seed_agent.py`: Query decomposition and search planning
  - `sourcing_agent.py`: Multi-source gathering (local + Perplexity Sonar)
  - `research_agent.py`: Synthesis, conflict detection, and citations

- **`app/`**: CLI and future web interfaces
  - `cli.py`: Rich CLI with Phase 2 enhanced display

- **`prompts/`**: Agent prompt templates  
- **`eval/`**: Evaluation harness
- **`data/`**: Local data storage and sample documents

### Configuration

The system uses Pydantic settings with environment variable support:

- **Required**: `OPENAI_API_KEY`
- **Optional**: `PERPLEXITY_API_KEY`, `VECTARA_API_KEY`, `LANGSMITH_API_KEY`
- **Defaults**: `CHROMA_DB_PATH=./data/chroma_db`, `LLM_MODEL=gpt-4o`, `EMBEDDING_MODEL=text-embedding-3-small`

### Key Parameters

- **Chunk size**: 1500 tokens (configurable via `CHUNK_SIZE`)
- **Chunk overlap**: 150 tokens (10% of chunk size)
- **Retrieval**: Top-15 documents by default (`RETRIEVAL_K=15`)
- **Quality threshold**: Minimum eval score 0.35 (`MIN_EVAL_SCORE=0.35`)

## Development Workflow

### Phase Roadmap
1. **Phase 1**: CLI MVP with basic RAG ✅
2. **Phase 2**: Multi-agent orchestration (Seed → Sourcing → Research) ✅
3. **Phase 3** (Current): Quality assurance with Vectara FCS and evaluation harness
4. **Phase 4**: Performance improvements and caching
5. **Phase 5**: Web UI and deployment preparation

### Working with the Codebase

- The main entry point is `BasicRAGPipeline` in `core/rag_pipeline.py` (enhanced for Phase 2)
- Use the global `get_rag_pipeline()` function to get a singleton instance
- **Phase 2 Workflow**: Query → Seed Agent → Sourcing Agent → Research Agent → Enhanced Answer
- **Phase 1 Fallback**: Maintains backward compatibility for basic RAG
- Documents are automatically chunked and embedded when added via `add_document()`
- The vector store uses Chroma DB with hybrid filtering capabilities
- All components use the centralized `settings` from `core.config`

### Phase 2 Multi-Agent System

- **Seed Agent**: Decomposes queries into 2-5 sub-queries with strategy selection
- **Sourcing Agent**: Gathers from local knowledge base + Perplexity Sonar API
- **Research Agent**: Synthesizes sources with conflict detection and mixed citations
- **Citations**: [Local-1], [Local-2] for local sources; [Sonar-1] for live sources
- **Confidence Scoring**: AI-driven assessment based on sources, quality, and conflicts

### Environment Variables

Key environment variables in `.env`:
```
# Required
OPENAI_API_KEY=your_openai_api_key_here

# Optional - Enhanced features
PERPLEXITY_API_KEY=your_perplexity_api_key_here        # Live research with Sonar
VECTARA_API_KEY=your_vectara_api_key_here              # Quality scoring with FCS
LANGSMITH_API_KEY=your_langsmith_api_key_here          # Advanced monitoring

# Core configuration
CHROMA_DB_PATH=./data/chroma_db
LLM_MODEL=gpt-4o                                       # Default fallback model
EMBEDDING_MODEL=text-embedding-3-small                 # Vector embeddings
CHUNK_SIZE=1500
RETRIEVAL_K=15

# Quality assurance
ENABLE_FACTUAL_CONSISTENCY_SCORE=true                 # Enable Vectara FCS
MIN_FCS_THRESHOLD=0.9                                  # FCS quality threshold
MIN_EVAL_SCORE=0.35                                    # Minimum evaluation score
```

### Model Configuration

Models are configured in `model_config.yaml` with intelligent selection:
- **Seed Agent**: o1-mini for fast query decomposition
- **Research Agent**: o1 for complex synthesis and conflict detection  
- **Sourcing Agent**: sonar-pro for live research APIs
- **Fallback chains**: Automatic model switching on failures
- **Context awareness**: High-context models for large documents (>100k tokens)

## Important Notes

- This is primarily a Python project using LangChain, Chroma DB, and OpenAI APIs
- No formal test runner configured - uses custom test scripts
- Linting/formatting tools available: black, flake8 (run `black .` and `flake8` for code quality)
- Vector embeddings are persisted locally in `./data/chroma_db/`
- The system is designed to evolve from local CLI to cloud-deployed web application
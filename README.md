# AI Research Pipeline

![Python](https://img.shields.io/badge/Python-100%25-3776AB) ![OpenAI](https://img.shields.io/badge/OpenAI-o1%20%7C%20o1--mini-412991) ![ChromaDB](https://img.shields.io/badge/Chroma_DB-Vector_Store-FF6F00) ![Perplexity](https://img.shields.io/badge/Perplexity-sonar--pro-20B2AA) ![Vectara](https://img.shields.io/badge/Vectara-Fact_Checking-6C3483) ![RAG](https://img.shields.io/badge/RAG-Multi--Agent-orange)

A multi-agent retrieval-augmented research assistant that combines cloud LLM reasoning, live research APIs, and persistent vector memory with intelligent model selection and quality assurance.

## Architecture

The system uses a **three-agent workflow** with **smart model selection**:
- **Seed Agent**: Query decomposition & search planning (uses o1-mini for speed)
- **Sourcing Agent**: Calls research APIs, filters and evaluates content (uses sonar-pro for live research)
- **Research Agent**: Retrieval, synthesis, and conflict detection (uses o1 for complex reasoning)

### Model Selection Strategy
- **Context-aware routing**: Different models for different complexity levels
- **Fallback chains**: Resilient error handling with model alternatives  
- **Performance monitoring**: Real-time tracking of model usage and success rates
- **Cost optimization**: Task-appropriate model selection reduces unnecessary compute

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys (OpenAI required, Perplexity/Vectara optional)
   ```

3. **Configure Models** (Optional)
   ```bash
   # Default configuration in model_config.yaml
   # Customize models for different tasks and complexity levels
   ```

4. **Run CLI**
   ```bash
   # Ask a research question
   python app/cli.py ask "Your research question here"
   
   # View model configuration
   python app/cli.py models
   
   # Monitor performance
   python app/cli.py performance
   ```

## Project Structure

```
├── agents/           # Agent implementations
├── prompts/          # Agent prompt templates
├── core/            # Core orchestration logic
├── app/             # CLI and web interfaces
├── eval/            # Evaluation harness
├── tests/           # Test suite
├── data/            # Local data storage
└── docs/            # Documentation
```

## Features

### Core Capabilities
- **Multi-Agent Research**: Seed → Sourcing → Research agent workflow
- **Local Vector Storage**: Chroma DB for persistent knowledge base
- **Live Research Integration**: Perplexity Sonar for real-time web research  
- **Smart Citation System**: Mixed local/live source citations with conflict detection

### Model Selection & Performance
- **Intelligent Model Routing**: o1-mini for planning, o1 for synthesis, sonar-pro for research
- **Context-Aware Fallbacks**: Automatic high-context model switching (>100k tokens)
- **Performance Monitoring**: Real-time metrics collection and model comparison
- **Cost Optimization**: Task-appropriate model selection with fallback chains

### Quality Assurance
- **Vectara FCS Integration**: Factual consistency scoring for response validation
- **Multi-Factor Confidence**: Combines model confidence, citation quality, and source utilization
- **Comprehensive Assessment**: Citation analysis, readability scoring, and query coverage

### CLI Commands
```bash
# Research and knowledge management
python app/cli.py ask "question" [--verbose]     # Ask research questions
python app/cli.py add "text" [--title --url]     # Add documents
python app/cli.py stats                          # Knowledge base statistics
python app/cli.py report [--detailed --export]   # Comprehensive reports

# Model management and monitoring  
python app/cli.py models                         # View model configuration
python app/cli.py performance [--export]        # Performance metrics
python app/cli.py select-model TASK             # Test model selection
```

## Development Phases

- [x] **Phase 1**: CLI MVP (Basic RAG)
- [x] **Phase 2**: Multi-Agent Orchestration  
- [x] **Phase 3**: Quality & Evaluation with Model Selection
- [ ] **Phase 4**: Performance Improvements
- [ ] **Phase 5**: UI & Deployment

## Configuration

### Environment Variables
```bash
# Required
OPENAI_API_KEY=your_openai_api_key

# Optional - Enhanced features  
PERPLEXITY_API_KEY=your_perplexity_key     # Live research
VECTARA_API_KEY=your_vectara_key           # Quality scoring
LANGSMITH_API_KEY=your_langsmith_key       # Advanced monitoring

# Model customization
LLM_MODEL=gpt-4o                           # Default fallback model
EMBEDDING_MODEL=text-embedding-3-small     # Vector embeddings
```

### Model Configuration (`model_config.yaml`)
```yaml
models:
  # Primary reasoning models
  complex_reasoning: "o1-2024-12-17"        # Synthesis and analysis
  simple_reasoning: "o1-mini-2024-12-17"    # Planning and decomposition  
  high_context_fallback: "gpt-4.1"          # Large context processing
  
  # Specialized models
  alternative_llm: "claude-3-5-sonnet-20241022"  # Diverse perspectives
  live_research: "sonar-pro"                      # Real-time research
  embeddings: "text-embedding-3-large"           # Vector search
  reranker: "rerank-english-v3.0"                # Result optimization
```

See `docs/model_selection_plan.md` for detailed model selection strategy.

# AI Research Pipeline

[![CI](https://github.com/BryceEWatson/ai-research-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/BryceEWatson/ai-research-pipeline/actions/workflows/ci.yml)

A command-line research assistant I built in Python. You ask it a question and three LLM agents split it into sub-questions, gather sources from a local Chroma vector store (and, with a Perplexity key, the live web), then write a cited answer that points out where the sources disagree.

It's a personal prototype I built in May 2025, not a maintained product. If you're reviewing the code, start with the agent hand-offs in `agents/`, the model routing in `core/config.py` and the answer scoring in `core/quality_scoring.py`.

## How it works

1. **Seed agent** (`agents/seed_agent.py`) asks a model to rate the question's complexity and break it into two to five sub-queries, each tagged to search the local store, the web or both.
2. **Sourcing agent** (`agents/sourcing_agent.py`) runs each sub-query against the local knowledge base and, when `PERPLEXITY_API_KEY` is set, against Perplexity's Sonar API.
3. **Research agent** (`agents/research_agent.py`) writes the answer with citations like `[Local-1]` and `[Sonar-1]`, asks the model to list conflicts between sources, and adds a self-critique and a confidence score.

If the agent path throws an error, `core/rag_pipeline.py` falls back to a single retrieve-then-answer call.

### Model routing

`model_config.yaml` maps task types to models, and `ModelSelector` in `core/config.py` picks one by task and a 1 to 5 complexity score. The config in this repo uses:

| Role | Model |
| --- | --- |
| Simple reasoning (planning, low-complexity answers) | `gpt-4o-mini` |
| Complex reasoning (complexity 4 and up) | `gpt-4o` |
| Alternative model (answers at complexity 3) | `gpt-4` |
| Large-context fallback | `gpt-4-turbo` |
| Live research | `sonar-pro` |

When a model call fails, the seed and research agents retry down a fallback chain. Two config entries aren't wired up yet: the reranker is only displayed by the `models` command and never called, and the large-context fallback is only picked when a caller passes a context size, which today only the `select-model` command does. Every chat model goes through OpenAI's client, so putting a non-OpenAI model in `alternative_llm` won't work without code changes.

### Answer scoring

`core/quality_scoring.py` averages the model's own confidence with citation counts, the share of sources the answer cites and how many of the question's words it covers. With `VECTARA_API_KEY` set it also calls Vectara's factual consistency score. These are heuristics, not an evaluation: I haven't measured how well they track answer quality.

`core/performance_monitor.py` logs each agent call (model, latency, success) to `performance_metrics.jsonl`, and `python app/cli.py performance` summarizes it.

## Quick start

You need Python 3.10 or newer and an OpenAI API key.

```bash
git clone https://github.com/BryceEWatson/ai-research-pipeline.git
cd ai-research-pipeline
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then put your OpenAI key in .env
python scripts/setup_sample_data.py
python app/cli.py ask "What is retrieval-augmented generation?"
```

`setup_sample_data.py` embeds four sample documents, which spends a little OpenAI credit. The Perplexity, Vectara and LangSmith keys are optional: leave them blank and those features stay off. Settings read the LangSmith key, but nothing uses it yet.

### CLI commands

```bash
python app/cli.py ask "question" [--verbose]       # research a question
python app/cli.py add "text" --title "Title"       # add a document
python app/cli.py add-file notes.txt               # add a text file
python app/cli.py stats                            # knowledge base size
python app/cli.py report [--detailed --export f]   # knowledge base report
python app/cli.py reset                            # empty the knowledge base
python app/cli.py models                           # show model routing
python app/cli.py performance [--component c]      # call metrics
python app/cli.py select-model research --complexity 4
```

## Tests

```bash
pytest                       # offline tests: no API keys, no API calls
pytest --run-integration     # adds end-to-end tests against OpenAI
```

The offline tests cover model routing, answer scoring, performance logging, chunking, the vector store wrapper and the CLI commands that don't call a model. They use a placeholder key and a throwaway vector store. The integration tests in `tests/test_integration.py` need a real `OPENAI_API_KEY`, load the sample documents if the store is empty and spend API credit. CI runs the offline tests plus a flake8 check for syntax errors and undefined names on every push and pull request.

## Project layout

```
agents/     seed, sourcing and research agents
app/        Click CLI
core/       settings, model routing, RAG pipeline, vector store, chunking, scoring, metrics
data/       sample documents (the vector store is written to data/chroma_db/)
docs/       design notes and the original phase plans
scripts/    environment check and sample data loader
tests/      pytest suite
```

The files in `docs/` are planning notes I wrote while building this, and some describe plans that changed. `docs/model_selection_plan.md`, for example, specifies o1 and Claude models, while the config that runs uses the GPT-4o family above.

## Status

Built: the three-agent flow, local and Perplexity sourcing, model routing with fallbacks, heuristic answer scoring and per-call metrics.

Not built: an evaluation harness that measures answer quality against labeled questions, caching, a web UI and deployment.

## License

MIT. See [LICENSE](LICENSE).

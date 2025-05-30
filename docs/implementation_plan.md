# Implementation Plan for a Multi‑Agent Retrieval‑Augmented Research Assistant

**Project Vision:** Build a local **retrieval‑augmented research assistant** with three specialized AI agents (**Seed**, **Sourcing**, and **Research**). The system will use **Chroma** for local vector storage, **LangChain** for orchestration, and **Vectara’s Factual Consistency Score (FCS)** for quality checks. The initial version will run as a command‑line (CLI) tool on a single machine, with no existing infrastructure assumed. The long‑term vision is to evolve this into a web‑based dashboard deployed on the cloud.

---

## Project Phases and Milestones

| **Phase** | **Milestone Goal** | **Est. Duration** | **Dependencies / Prerequisites** |
|-----------|--------------------|-------------------|----------------------------------|
| **1. CLI MVP** | Basic RAG system running via CLI (single‑agent retrieval & answer). | ~1 week | Chroma DB installed; LLM API or model ready. |
| **2. Multi‑Agent Orchestration** | Introduce Seed, Sourcing, Research agents working together. | ~1–2 weeks | Phase 1 complete; LangChain set up for agents. |
| **3. Quality & Evaluation** | Integrate Vectara FCS for hallucination checks; establish evaluation harness. | ~1 week | Phase 2 complete; Vectara API access (FCS). |
| **4. Performance Improvements** | Add caching, optimize memory usage, refine prompts for reliability. | ~1 week | Phase 2 baseline; parallel with Phase 3. |
| **5. UI & Deployment Prep** | Refactor for modularity, plan/implement minimal web UI, containerize for cloud. | ~2 weeks | Core system stable (Phase 3/4); chosen web framework. |

---

### Phase 1 – CLI MVP (Basic Retrieval‑Augmented QA)

**Milestone:** A command‑line tool that answers user questions by retrieving relevant info from a local knowledge base and using an LLM to generate an answer grounded in that info【2†L333-L340】.

**Key Tasks**

1. **Environment Setup** – install Chroma, LangChain, LLM SDK; acquire API keys.  
2. **Data Ingestion** – embed and store sample documents in Chroma.  
3. **RAG Pipeline** – retrieve top‑k chunks, prompt LLM, print answer.  
4. **Testing** – manual queries validate retrieval + generation.

---

### Phase 2 – Multi‑Agent Orchestration

**Milestone:** Refactor into three specialized agents.

* **Seed Agent** – parses query, creates search plan.  
* **Sourcing Agent** – executes plan, queries Chroma, returns docs.  
* **Research Agent** – synthesizes final answer using sources.  
* Orchestrate with LangChain or simple controller; iterate on prompts.

---

### Phase 3 – Quality & Evaluation

* **Integrate Vectara FCS** – compute FCS for each answer; flag low scores【6†L21-L27】.  
* **Evaluation Harness** – batch test queries, log answer & FCS.  
* **Prompt Refinement** – adjust agents based on evaluation results.

---

### Phase 4 – Reliability & Performance

* **Caching layers** – embeddings, retrievals, LLM responses.  
* **Memory‑limit strategies** – modular code, small‑context prompts.  
* **Logging & refactor** – clean interfaces, add debug logging.

---

### Phase 5 – UI & Deployment Preparation

* **Core vs. Interface Split** – expose `answer_question()` in `core/`.  
* **Prototype Web UI** – lightweight Streamlit/Flask over core logic.  
* **Containerize** – Dockerfile, env‑var configs, secret management.  
* **Cloud Roadmap** – outline path to managed vector DB & load‑balanced API.

---

## Leveraging AI Coding Agents

* **Delegate** boilerplate, tests, docs, and prompt‑brainstorming to AI.  
* **Iterative, modular tasks** keep within context limits.  
* **Use summaries** to rehydrate AI context; maintain high‑level design doc.

---

## Tips for Working Around Context Limits

1. **Provide only relevant modules** to AI per session.  
2. **Maintain concise project overview** for quick re‑contextualization.  
3. **Break refactors into small edits**; test frequently.  
4. **Cache embeddings & prompts** to cut token usage.

---

## Future‑Proofing Steps

* Config‑driven vector‑store and LLM selection.  
* Stateless `answer_question()` for horizontal scaling.  
* Docker + CI for reproducible builds.  
* Structured logging for cloud monitoring.

---

## Recommended File Structure

```
project-root/
├── agents/
│   ├── seed_agent.py
│   ├── sourcing_agent.py
│   └── research_agent.py
├── prompts/
│   ├── seed_prompt.txt
│   ├── sourcing_prompt.txt
│   └── research_prompt.txt
├── core/
│   ├── orchestrator.py
│   ├── vector_store.py
│   └── quality_check.py
├── app/
│   ├── cli.py
│   └── web_app.py
├── eval/
│   ├── evaluation_harness.py
│   └── sample_queries.json
├── tests/
│   └── test_pipeline.py
├── data/
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## References

Key citations: Chroma Docs, LangChain docs, Vectara FCS, OpenAI Deep Research, CRAG paper, etc.  【2†L333-L340】【6†L21-L27】

---

*This roadmap gives a phased, modular path from zero to a robust, multi‑agent RAG assistant that can evolve from local CLI to cloud‑deployed web app.*


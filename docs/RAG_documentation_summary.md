# RAG System Development Documentation Summary

## 1. API & Service Documentation

### OpenAI APIs (GPT‑4.1, Embeddings, Deep Research)
**GPT‑4.1:** OpenAI’s latest flagship model offers major improvements in reasoning and context length over GPT‑4o. It supports a context window of up to **1,048,576 tokens** (≈1 M) and up to **32,768 output tokens**. This lets you feed entire codebases or large documents directly to the model. GPT‑4.1 launched with *Mini* and *Nano* variants that deliver faster, cheaper reasoning at some trade‑off in capability. All GPT‑4.1 models share a June 2024 knowledge cutoff and deliver **≈26 % lower cost** than GPT‑4o.

*Pricing:* about **$2.00 / M input tokens** (‑75 % cached) and **$8.00 / M output tokens**.

**Embedding models:**  
`text‑embedding‑3‑small` (1536‑d) improves multilingual retrieval (MIRACL 31 → 44 %) and is **5× cheaper** than ada‑002 at **$0.00002 / 1 K tokens**.  
`text‑embedding‑3‑large` (3072‑d) offers even higher quality for **$0.00013 / 1 K**.

**Deep Research agent:** A ChatGPT feature that performs multi‑step live‑web research, autonomously browsing, reading PDFs, and synthesizing a sourced report. It is not exposed directly via API, but OpenAI’s new *Responses API* and Agents SDK let developers build similar RAG agents programmatically.

### Perplexity Sonar API
Provides fast, low‑cost, retrieval‑augmented answers with citations.  
*Models:* **sonar** (concise QA) and **sonar‑pro** (multi‑step research).  
*Typical cost:* ≈ **$1 / M tokens** + **$5 – $12 / 1 K calls**.  
Sonar Pro tops the SimpleQA factuality benchmark (F1 ≈ 0.858).

### Vectara FCS API
Returns a **Factual Consistency Score (0‑1)** with every generated answer, flagging hallucinations in real time. Enabled by adding `enable_factual_consistency_score:true` to a query. A score < 0.5 usually warrants review.

---

## 2. Framework & Library Documentation

### Chroma DB
Open‑source vector database (“AI memory”).  
*Highlights:* in‑process or server mode, rich metadata filtering, full‑text search, Docker/K8s deploy. Chroma Cloud (managed serverless) in private preview.

### LangChain
Modular framework for LLM apps: standardized interfaces for models, embeddings, vector stores; built‑in chains (RetrievalQA, summarization); agent & tool abstractions; async support; integrates with LangSmith for tracing.

### LangSmith
Observability & eval platform: traces every chain/agent call, captures prompts + outputs, supports automated QA/relevance evaluations, integrates with OpenTelemetry, hosted with free tier.

---

## 3. Implementation Patterns & Best Practices

### RAG pipeline
1. **Indexing:** load → chunk → embed → store in vector DB.  
2. **Retrieval + Generation:** embed query → vector search → inject top‑k chunks into prompt → LLM answer with citations.

*Key tactics:* optimal chunk size (200‑500 tkn, 15 % overlap), hybrid keyword + vector search, citation templates \[1\], iterative retrieval, re‑ranks.

### Performance optimisation
* Use efficient models for embeddings (e.g. `embedding‑3‑small`).  
* Limit context & output length, cache frequent queries.  
* Batch embeddings, parallelise async calls.  
* Re‑rank retrieved top‑10 with a small model before passing top‑3 to GPT‑4.  
* Pick GPT‑4.1 Mini/Nano for simpler queries.

---

## 4. Integration & Deployment

*Typical architecture:* Front‑end → API Gateway → LangChain‑based backend → Vector DB (Chroma) + LLM API (OpenAI/Sonar) → answer.

*Infra options:* self‑hosted Docker/K8s or managed (Chroma Cloud, Vectara). Secure with VPC, HTTPS, API‑key vaults. Monitor rate limits; scale vector DB shards as corpus grows.

*Security:* no training on API data (OpenAI default), encrypt at rest, role‑based access, GDPR delete workflow, output moderation / FCS gating.

---

## 5. Cost Analysis

| Component | Typical Cost | Optimisation |
|-----------|--------------|--------------|
| GPT‑4.1 | $2 / M in, $8 / M out | Route easy Q’s to mini/nano; shorten output |
| Sonar base | $1 / M tkn + $5 / K calls | Use for high‑volume factual Q&A |
| Embedding‑3‑small | $0.02 / M tkn | Batch embeds, avoid re‑embedding |
| Vector DB | Storage + IO | Use local SSD; compact index |

---

## 6. Technical Specs

* **GPT‑4.1 limits:** 1 M input, 32 K output, cutoff 2024‑06.  
* **Embedding dims:** small 1536, large 3072.  
* **Chroma schema:** `id`, `embedding:float[]`, `text`, `metadata`.  
* **API formats:** OpenAI/sonar chat = JSON messages; Vectara returns `hits[]`, `summary`, `factual_consistency_score`.

---

## 7. Operations

*Trace & test:* LangSmith traces, nightly regression set.  
*Alerts:* LLM error rate > 1 %, latency p95, FCS < 0.3.  
*Maintenance:* re‑ingest new docs nightly, index compaction weekly, backup embeddings.

---

## 8. Example Implementations

* **LangChain RetrievalQA tutorial** – minimal Chroma + OpenAI demo.  
* **Zoom AI Companion** – uses Sonar Pro for live meeting Q&A.  
* **Internal wiki bot** – Slack assistant querying HR docs.  
* **OpenAI Deep Research agent** – multi‑step web research via Responses API.

---

*This markdown is a compact yet comprehensive reference distilled from official documentation across OpenAI, Perplexity, Vectara, Chroma, LangChain, and LangSmith.*
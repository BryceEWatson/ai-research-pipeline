"""End-to-end checks against the live OpenAI (and optionally Perplexity) APIs.

Skipped unless pytest runs with ``--run-integration`` and a real
OPENAI_API_KEY is available in the environment or ``.env``. The sample
documents are loaded into a throwaway vector store.
"""

import subprocess
import sys

import pytest

from tests.conftest import PROJECT_ROOT

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def pipeline():
    from core.rag_pipeline import get_rag_pipeline
    from data.sample_documents import get_sample_documents

    # conftest points CHROMA_DB_PATH at an empty temp directory, so seed it.
    rag = get_rag_pipeline()
    for doc in get_sample_documents():
        rag.add_document(
            text=doc["content"], title=doc["title"],
            source_url=doc["source_url"], eval_score=doc["eval_score"],
        )
    return rag


@pytest.mark.parametrize("question", ["What is retrieval-augmented generation?", "What is machine learning?"])
def test_multi_agent_answer(pipeline, question):
    response = pipeline.answer_question(question, use_agents=True)
    assert response["phase"] == "2_multi_agent"
    assert response["answer"]
    assert response["citations"]
    for field in ("confidence_score", "total_sources", "source_breakdown"):
        assert field in response
    qa = response["quality_assessment"]
    for field in ("overall_confidence", "explanation", "metrics"):
        assert field in qa


def test_basic_rag_fallback(pipeline):
    response = pipeline.answer_question("What is machine learning?", use_agents=False)
    assert response["phase"] == "1_basic_rag"
    assert response["answer"]


def test_seed_agent_complexity_analysis():
    from agents.seed_agent import get_seed_agent

    assert "complexity_score" in get_seed_agent().analyze_query_complexity("What is AI?")


def test_cli_ask(pipeline):
    result = subprocess.run(
        [sys.executable, "app/cli.py", "ask", "What is RAG?"],
        capture_output=True, text=True, timeout=180, cwd=PROJECT_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "retrieval" in result.stdout.lower()

"""Answer quality scoring, with the Vectara API switched off."""

import pytest

import core.quality_scoring as quality_scoring
from core.quality_scoring import QualityScorer


@pytest.fixture(autouse=True)
def no_vectara_key(monkeypatch):
    monkeypatch.setattr(quality_scoring.settings, "vectara_api_key", "")


def score(answer, sources=("a", "b"), query="what is rag", confidence=0.8):
    return QualityScorer().score_answer_quality(
        answer=answer, sources=list(sources), query=query, confidence_score=confidence
    )


def test_vectara_is_disabled_without_a_key():
    assert QualityScorer().vectara_fcs.enabled is False


def test_counts_local_and_live_citations():
    result = score("RAG combines retrieval [Local-1] with generation [Sonar-1] [Local-1].")
    assert result.metrics["local_citations"] == 2
    assert result.metrics["sonar_citations"] == 1
    assert result.metrics["total_citations"] == 3
    assert result.metrics["sources_utilized"] == 2
    assert result.fcs_score is None


def test_confidence_is_bounded_and_explained():
    result = score("RAG is retrieval plus generation [Local-1] [Local-2].")
    assert 0 <= result.confidence <= 1
    assert "citations" in result.explanation.lower()


def test_uncited_answer_scores_lower_than_cited_answer():
    cited = score("What is RAG? RAG is retrieval [Local-1] plus generation [Local-2].")
    uncited = score("What is RAG? RAG is retrieval plus generation.")
    assert uncited.confidence < cited.confidence
    assert "no citations" in uncited.explanation.lower()


def test_query_coverage_is_share_of_question_words_in_answer():
    assert score("rag is great", query="what is rag").metrics["query_coverage"] == 2 / 3

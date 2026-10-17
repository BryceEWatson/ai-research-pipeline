"""Quality scoring module with Vectara FCS integration."""

import logging
import requests
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Quality assessment result."""
    fcs_score: Optional[float]  # Factual Consistency Score (0-1)
    confidence: float  # Overall confidence (0-1)
    explanation: str  # Human-readable explanation
    metrics: Dict[str, Any]  # Additional quality metrics


class VectaraFCS:
    """Vectara Factual Consistency Score integration."""
    
    def __init__(self):
        """Initialize Vectara FCS client."""
        self.api_key = settings.vectara_api_key
        self.base_url = "https://api.vectara.io"
        self.enabled = settings.enable_factual_consistency_score and bool(self.api_key)
        
        if not self.enabled:
            logger.info("Vectara FCS disabled (no API key or setting disabled)")
    
    def score_factual_consistency(self, text: str, sources: List[str]) -> Optional[float]:
        """
        Score factual consistency of text against sources using Vectara FCS.
        
        Args:
            text: Generated text to evaluate
            sources: List of source texts for consistency checking
            
        Returns:
            FCS score (0-1) or None if unavailable
        """
        if not self.enabled:
            return None
        
        try:
            # Prepare request payload
            payload = {
                "text": text,
                "sources": sources,
                "threshold": settings.min_fcs_threshold
            }
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Make API call
            response = requests.post(
                f"{self.base_url}/v1/factual-consistency",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get("score", 0.0)
            else:
                logger.error(f"Vectara FCS API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error calling Vectara FCS: {e}")
            return None


class QualityScorer:
    """Comprehensive quality scoring system."""
    
    def __init__(self):
        """Initialize quality scorer."""
        self.vectara_fcs = VectaraFCS()
    
    def score_answer_quality(self, 
                           answer: str, 
                           sources: List[str], 
                           query: str,
                           confidence_score: float = 0.5) -> QualityScore:
        """
        Comprehensive quality scoring of an answer.
        
        Args:
            answer: Generated answer text
            sources: Source texts used for generation
            query: Original query
            confidence_score: Model's confidence score
            
        Returns:
            QualityScore with comprehensive assessment
        """
        metrics = {}
        
        # Vectara FCS scoring
        fcs_score = None
        if self.vectara_fcs.enabled and sources:
            fcs_score = self.vectara_fcs.score_factual_consistency(answer, sources)
            metrics["vectara_fcs"] = fcs_score
        
        # Basic quality metrics
        metrics.update(self._calculate_basic_metrics(answer, query))
        
        # Citation analysis
        metrics.update(self._analyze_citations(answer))
        
        # Source utilization
        metrics["source_count"] = len(sources)
        metrics["sources_utilized"] = self._count_sources_referenced(answer, len(sources))
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            fcs_score, confidence_score, metrics
        )
        
        # Generate explanation
        explanation = self._generate_quality_explanation(fcs_score, metrics, overall_confidence)
        
        return QualityScore(
            fcs_score=fcs_score,
            confidence=overall_confidence,
            explanation=explanation,
            metrics=metrics
        )
    
    def _calculate_basic_metrics(self, answer: str, query: str) -> Dict[str, Any]:
        """Calculate basic text quality metrics."""
        words = answer.split()
        sentences = answer.split('.')
        
        return {
            "word_count": len(words),
            "sentence_count": len([s for s in sentences if s.strip()]),
            "avg_sentence_length": len(words) / max(len(sentences), 1),
            "query_coverage": self._calculate_query_coverage(answer, query),
            "readability_score": self._calculate_readability(answer)
        }
    
    def _analyze_citations(self, answer: str) -> Dict[str, Any]:
        """Analyze citation usage in the answer."""
        import re
        
        # Count citation patterns [Local-1], [Sonar-1], etc.
        local_citations = len(re.findall(r'\[Local-\d+\]', answer))
        sonar_citations = len(re.findall(r'\[Sonar-\d+\]', answer))
        total_citations = local_citations + sonar_citations
        
        return {
            "total_citations": total_citations,
            "local_citations": local_citations,
            "sonar_citations": sonar_citations,
            "citation_density": total_citations / max(len(answer.split()), 1) * 100
        }
    
    def _count_sources_referenced(self, answer: str, source_count: int) -> int:
        """Count how many sources are actually referenced in the answer."""
        import re
        
        referenced = set()
        
        # Check for Local citations
        local_matches = re.findall(r'\[Local-(\d+)\]', answer)
        referenced.update(f"Local-{m}" for m in local_matches)
        
        # Check for Sonar citations
        sonar_matches = re.findall(r'\[Sonar-(\d+)\]', answer)
        referenced.update(f"Sonar-{m}" for m in sonar_matches)
        
        return len(referenced)
    
    def _calculate_query_coverage(self, answer: str, query: str) -> float:
        """Calculate how well the answer covers the query terms."""
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        if not query_words:
            return 0.0
        
        coverage = len(query_words.intersection(answer_words)) / len(query_words)
        return coverage
    
    def _calculate_readability(self, text: str) -> float:
        """Simple readability score (0-1, higher is better)."""
        words = text.split()
        sentences = [s for s in text.split('.') if s.strip()]
        
        if not sentences or not words:
            return 0.0
        
        # Simple readability based on sentence length
        avg_sentence_length = len(words) / len(sentences)
        
        # Optimal sentence length is around 15-20 words
        if 10 <= avg_sentence_length <= 25:
            return 1.0
        elif avg_sentence_length < 5:
            return 0.3  # Too short, likely fragments
        elif avg_sentence_length > 40:
            return 0.3  # Too long, hard to read
        else:
            return 0.7  # Acceptable
    
    def _calculate_overall_confidence(self, 
                                    fcs_score: Optional[float],
                                    model_confidence: float,
                                    metrics: Dict[str, Any]) -> float:
        """Calculate overall confidence from multiple factors."""
        confidence_factors = []
        
        # Model's own confidence
        confidence_factors.append(model_confidence)
        
        # Vectara FCS if available
        if fcs_score is not None:
            confidence_factors.append(fcs_score)
        
        # Citation quality
        citation_score = min(metrics["total_citations"] / 5.0, 1.0)  # Normalize to 5 citations
        confidence_factors.append(citation_score)
        
        # Source utilization
        if metrics["source_count"] > 0:
            utilization_score = metrics["sources_utilized"] / metrics["source_count"]
            confidence_factors.append(utilization_score)
        
        # Query coverage
        confidence_factors.append(metrics["query_coverage"])
        
        # Return weighted average
        return sum(confidence_factors) / len(confidence_factors)
    
    def _generate_quality_explanation(self, 
                                    fcs_score: Optional[float],
                                    metrics: Dict[str, Any],
                                    confidence: float) -> str:
        """Generate human-readable quality explanation."""
        explanations = []
        
        # Overall assessment
        if confidence >= 0.8:
            explanations.append("High quality answer with strong evidence support.")
        elif confidence >= 0.6:
            explanations.append("Good quality answer with adequate evidence.")
        elif confidence >= 0.4:
            explanations.append("Moderate quality answer with some limitations.")
        else:
            explanations.append("Lower quality answer requiring verification.")
        
        # FCS assessment
        if fcs_score is not None:
            if fcs_score >= 0.9:
                explanations.append(f"Excellent factual consistency (FCS: {fcs_score:.2f}).")
            elif fcs_score >= 0.7:
                explanations.append(f"Good factual consistency (FCS: {fcs_score:.2f}).")
            else:
                explanations.append(f"Factual consistency may need review (FCS: {fcs_score:.2f}).")
        
        # Citation assessment
        if metrics["total_citations"] >= 3:
            explanations.append("Well-cited with multiple sources.")
        elif metrics["total_citations"] >= 1:
            explanations.append("Some citations provided.")
        else:
            explanations.append("Limited or no citations.")
        
        # Source utilization
        if metrics["source_count"] > 0:
            utilization = metrics["sources_utilized"] / metrics["source_count"]
            if utilization >= 0.7:
                explanations.append("Good use of available sources.")
            elif utilization >= 0.3:
                explanations.append("Partial use of available sources.")
            else:
                explanations.append("Limited use of available sources.")
        
        return " ".join(explanations)


# Global quality scorer instance
_quality_scorer_instance: Optional[QualityScorer] = None


def get_quality_scorer() -> QualityScorer:
    """Get global quality scorer instance."""
    global _quality_scorer_instance
    if _quality_scorer_instance is None:
        _quality_scorer_instance = QualityScorer()
    return _quality_scorer_instance
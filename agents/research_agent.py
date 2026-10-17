"""Research Agent for synthesis and conflict detection."""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

from core.config import settings, model_selector
from core.performance_monitor import record_operation
from agents.sourcing_agent import SourcingResult, SourcedDocument
import time

logger = logging.getLogger(__name__)


@dataclass
class Citation:
    """Citation with source information."""
    id: str
    source_type: str  # "local", "sonar"
    title: str
    url: str
    snippet: str


@dataclass
class Conflict:
    """Detected conflict between sources."""
    claim: str
    conflicting_sources: List[str]
    severity: str  # "low", "medium", "high"
    description: str


@dataclass
class ResearchResult:
    """Final research result with answer and metadata."""
    answer: str
    citations: List[Citation]
    conflicts: List[Conflict]
    confidence_score: float
    source_breakdown: Dict[str, int]
    total_sources: int
    self_critique: str


class ResearchAgent:
    """Agent responsible for synthesis and conflict detection."""
    
    def __init__(self):
        """Initialize the Research Agent."""
        # Model will be selected dynamically based on task complexity
        self.openai_api_key = settings.openai_api_key
        
        self.synthesis_prompt = PromptTemplate(
            input_variables=["query", "sources", "source_count"],
            template="""You are the Research Agent in a multi-agent research system. Your job is to synthesize information from multiple sources into a comprehensive, accurate answer.

Original Query: {query}

Sources Available ({source_count} total):
{sources}

Instructions:
1. Synthesize information from ALL provided sources
2. Use citation format: [Local-1], [Local-2] for local sources and [Sonar-1], [Sonar-2] for live sources
3. Identify and highlight any conflicts between sources
4. Provide a balanced perspective that acknowledges different viewpoints
5. If information is missing or insufficient, state this clearly
6. Be precise and factual, avoiding speculation

Required Response Format:
## Answer
[Your comprehensive answer with proper citations]

## Source Analysis
- Local sources: [count and brief assessment]
- Live sources: [count and brief assessment]

## Conflicts Detected
[List any conflicting information with source references]

## Confidence Assessment
[Rate your confidence 1-10 and explain reasoning]

Remember: Every factual claim must have a citation. Be thorough but concise."""
        )
        
        self.conflict_detection_prompt = PromptTemplate(
            input_variables=["sources"],
            template="""Analyze the following sources for factual conflicts and inconsistencies.

Sources:
{sources}

Instructions:
1. Identify specific factual conflicts (numbers, dates, claims)
2. Categorize severity: "low" (minor discrepancies), "medium" (notable differences), "high" (major contradictions)
3. Explain each conflict clearly

Return conflicts in JSON format:
[
    {{
        "claim": "specific conflicting claim",
        "conflicting_sources": ["source1", "source2"],
        "severity": "medium",
        "description": "detailed explanation of the conflict"
    }}
]

If no conflicts found, return empty array: []"""
        )
        
        self.critique_prompt = PromptTemplate(
            input_variables=["query", "answer", "sources_count"],
            template="""Critically evaluate this research answer for quality and completeness.

Original Query: {query}
Generated Answer: {answer}
Sources Used: {sources_count}

Evaluate on these criteria:
1. Completeness: Does it fully address the query?
2. Accuracy: Are claims properly supported by sources?
3. Citations: Are all factual statements cited?
4. Balance: Does it present multiple perspectives when relevant?
5. Clarity: Is it well-organized and understandable?

Provide a score (1-10) and specific recommendations for improvement.
Focus on what's missing or could be enhanced."""
        )
    
    def synthesize_research(self, query: str, sourcing_result: SourcingResult) -> ResearchResult:
        """
        Synthesize information from multiple sources into final answer.
        
        Args:
            query: Original user question
            sourcing_result: Results from Sourcing Agent
            
        Returns:
            ResearchResult with synthesized answer and metadata
        """
        if not sourcing_result.documents:
            return self._create_no_sources_result(query)
        
        start_time = time.time()
        success = False
        self.current_model = "unknown"
        
        try:
            # Determine task complexity based on source count and conflicts
            complexity = self._assess_synthesis_complexity(sourcing_result)
            
            # Select model based on complexity
            selected_model = model_selector.select_model_for_task("research", complexity=complexity)
            
            # Store for error handling
            self.current_model = selected_model
            
            # Create LLM instance with selected model
            llm = self._create_llm_instance(selected_model)
            
            # Prepare sources for synthesis
            sources_text = self._format_sources_for_synthesis(sourcing_result.documents)
            
            # Generate main synthesis
            synthesis_prompt = self.synthesis_prompt.format(
                query=query,
                sources=sources_text,
                source_count=len(sourcing_result.documents)
            )
            
            synthesis_response = llm.invoke([HumanMessage(content=synthesis_prompt)])
            synthesis_text = synthesis_response.content
            
            # Detect conflicts using different model for diversity
            conflicts = self._detect_conflicts(sourcing_result.documents, llm)
            
            # Generate citations
            citations = self._generate_citations(sourcing_result.documents)
            
            # Self-critique
            critique = self._generate_self_critique(query, synthesis_text, len(sourcing_result.documents), llm)
            
            # Calculate confidence score
            confidence = self._calculate_confidence_score(sourcing_result, conflicts, synthesis_text)
            
            # Extract the answer from synthesis (remove section headers)
            answer = self._extract_answer_section(synthesis_text)
            
            # Source breakdown
            source_breakdown = {
                "local": sourcing_result.local_count,
                "live": sourcing_result.live_count
            }
            
            success = True
            logger.info(f"Research synthesis complete using {selected_model}. Confidence: {confidence:.2f}, "
                       f"Conflicts: {len(conflicts)}, Sources: {len(sourcing_result.documents)}")
            
            return ResearchResult(
                answer=answer,
                citations=citations,
                conflicts=conflicts,
                confidence_score=confidence,
                source_breakdown=source_breakdown,
                total_sources=len(sourcing_result.documents),
                self_critique=critique
            )
            
        except Exception as e:
            logger.error(f"Error in Research Agent synthesis with {selected_model}: {e}")
            
            # Try fallback models
            fallback_models = model_selector.get_fallback_chain(selected_model)
            
            for fallback_model in fallback_models:
                try:
                    logger.info(f"Trying fallback model: {fallback_model}")
                    self.current_model = fallback_model
                    fallback_llm = self._create_llm_instance(fallback_model)
                    
                    synthesis_response = fallback_llm.invoke([HumanMessage(content=synthesis_prompt)])
                    synthesis_text = synthesis_response.content
                    
                    conflicts = self._detect_conflicts(sourcing_result.documents, fallback_llm)
                    citations = self._generate_citations(sourcing_result.documents)
                    critique = self._generate_self_critique(query, synthesis_text, len(sourcing_result.documents), fallback_llm)
                    confidence = self._calculate_confidence_score(sourcing_result, conflicts, synthesis_text)
                    answer = self._extract_answer_section(synthesis_text)
                    
                    source_breakdown = {
                        "local": sourcing_result.local_count,
                        "live": sourcing_result.live_count
                    }
                    
                    success = True
                    logger.info(f"Fallback successful with {fallback_model}")
                    
                    return ResearchResult(
                        answer=answer,
                        citations=citations,
                        conflicts=conflicts,
                        confidence_score=confidence,
                        source_breakdown=source_breakdown,
                        total_sources=len(sourcing_result.documents),
                        self_critique=critique
                    )
                    
                except Exception as fallback_error:
                    logger.warning(f"Fallback model {fallback_model} also failed: {fallback_error}")
                    continue
            
            # Final fallback to error result
            return self._create_error_result(query, str(e))
            
        finally:
            # Record performance metrics
            execution_time = time.time() - start_time
            model_used = getattr(self, 'current_model', 'unknown')
            model_selector.record_performance(model_used, "research", execution_time, success)
            record_operation("research_agent", "synthesize_research", model_used, execution_time, success)
    
    def _format_sources_for_synthesis(self, documents: List[SourcedDocument]) -> str:
        """Format sources for LLM consumption."""
        formatted_sources = []
        
        local_count = 1
        sonar_count = 1
        
        for doc in documents:
            if doc.source_type == "local":
                source_id = f"Local-{local_count}"
                local_count += 1
            else:
                source_id = f"Sonar-{sonar_count}"  
                sonar_count += 1
            
            # Truncate content if too long
            content = doc.content[:1500] + "..." if len(doc.content) > 1500 else doc.content
            
            source_text = f"""
[{source_id}] {doc.title}
URL: {doc.source_url}
Date: {doc.date or 'N/A'}
Score: {doc.eval_score:.2f}
Content: {content}
"""
            formatted_sources.append(source_text)
        
        return "\n".join(formatted_sources)
    
    def _create_llm_instance(self, model_name: str) -> ChatOpenAI:
        """Create LLM instance with specified model."""
        return ChatOpenAI(
            model=model_name,
            openai_api_key=self.openai_api_key,
            temperature=0.1,
            max_tokens=2000
        )
    
    def _assess_synthesis_complexity(self, sourcing_result: SourcingResult) -> int:
        """Assess complexity of synthesis task (1-5 scale)."""
        complexity = 1
        
        # More sources = higher complexity
        if len(sourcing_result.documents) > 10:
            complexity += 2
        elif len(sourcing_result.documents) > 5:
            complexity += 1
        
        # Mixed source types = higher complexity
        if sourcing_result.local_count > 0 and sourcing_result.live_count > 0:
            complexity += 1
        
        # Large token count = higher complexity
        if sourcing_result.total_tokens > 10000:
            complexity += 1
        
        return min(complexity, 5)
    
    def _detect_conflicts(self, documents: List[SourcedDocument], llm: ChatOpenAI) -> List[Conflict]:
        """Detect conflicts between sources using LLM."""
        if len(documents) < 2:
            return []
        
        try:
            sources_text = self._format_sources_for_synthesis(documents)
            
            conflict_prompt = self.conflict_detection_prompt.format(sources=sources_text)
            response = llm.invoke([HumanMessage(content=conflict_prompt)])
            
            # Parse JSON response
            conflicts_data = self._parse_conflicts_response(response.content)
            
            conflicts = []
            for conflict_data in conflicts_data:
                conflict = Conflict(
                    claim=conflict_data.get("claim", ""),
                    conflicting_sources=conflict_data.get("conflicting_sources", []),
                    severity=conflict_data.get("severity", "low"),
                    description=conflict_data.get("description", "")
                )
                conflicts.append(conflict)
            
            return conflicts
            
        except Exception as e:
            logger.error(f"Error detecting conflicts: {e}")
            return []
    
    def _parse_conflicts_response(self, response: str) -> List[Dict]:
        """Parse LLM conflict detection response."""
        import json
        
        try:
            # Extract JSON from response
            start = response.find('[')
            end = response.rfind(']') + 1
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
            else:
                return []
        except Exception as e:
            logger.warning(f"Failed to parse conflicts response: {e}")
            return []
    
    def _generate_citations(self, documents: List[SourcedDocument]) -> List[Citation]:
        """Generate citation objects from documents."""
        citations = []
        
        local_count = 1
        sonar_count = 1
        
        for doc in documents:
            if doc.source_type == "local":
                citation_id = f"Local-{local_count}"
                local_count += 1
            else:
                citation_id = f"Sonar-{sonar_count}"
                sonar_count += 1
            
            # Create snippet from first 200 characters
            snippet = doc.content[:200] + "..." if len(doc.content) > 200 else doc.content
            
            citation = Citation(
                id=citation_id,
                source_type=doc.source_type,
                title=doc.title,
                url=doc.source_url,
                snippet=snippet
            )
            citations.append(citation)
        
        return citations
    
    def _generate_self_critique(self, query: str, answer: str, sources_count: int, llm: ChatOpenAI) -> str:
        """Generate self-critique of the research answer."""
        try:
            critique_prompt = self.critique_prompt.format(
                query=query,
                answer=answer,
                sources_count=sources_count
            )
            
            response = llm.invoke([HumanMessage(content=critique_prompt)])
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating self-critique: {e}")
            return f"Unable to generate self-critique: {e}"
    
    def _calculate_confidence_score(self, sourcing_result: SourcingResult, 
                                  conflicts: List[Conflict], answer: str) -> float:
        """Calculate confidence score based on multiple factors."""
        base_confidence = 0.5
        
        # Factor 1: Number of sources
        source_factor = min(len(sourcing_result.documents) / 5.0, 1.0) * 0.3
        
        # Factor 2: Source quality (average eval_score)
        if sourcing_result.documents:
            avg_quality = sum(doc.eval_score for doc in sourcing_result.documents) / len(sourcing_result.documents)
            quality_factor = avg_quality * 0.3
        else:
            quality_factor = 0
        
        # Factor 3: Conflicts penalty
        conflict_penalty = 0
        for conflict in conflicts:
            if conflict.severity == "high":
                conflict_penalty += 0.2
            elif conflict.severity == "medium":
                conflict_penalty += 0.1
            else:
                conflict_penalty += 0.05
        
        # Factor 4: Answer completeness (rough heuristic)
        completeness_factor = min(len(answer.split()) / 200.0, 1.0) * 0.2
        
        # Calculate final confidence
        confidence = base_confidence + source_factor + quality_factor + completeness_factor - conflict_penalty
        
        return max(0.0, min(1.0, confidence))
    
    def _extract_answer_section(self, synthesis_text: str) -> str:
        """Extract just the answer section from synthesis response."""
        # Look for "## Answer" section
        answer_match = re.search(r'## Answer\s*\n(.*?)(?=\n##|\Z)', synthesis_text, re.DOTALL)
        if answer_match:
            return answer_match.group(1).strip()
        
        # Fallback: return first paragraph or full text
        paragraphs = synthesis_text.split('\n\n')
        return paragraphs[0] if paragraphs else synthesis_text
    
    def _create_no_sources_result(self, query: str) -> ResearchResult:
        """Create result when no sources are available."""
        return ResearchResult(
            answer="I don't have enough information in my knowledge base to answer this question. Consider adding relevant documents or checking if the question requires current information from live sources.",
            citations=[],
            conflicts=[],
            confidence_score=0.1,
            source_breakdown={"local": 0, "live": 0},
            total_sources=0,
            self_critique="No sources available for synthesis."
        )
    
    def _create_error_result(self, query: str, error: str) -> ResearchResult:
        """Create result when synthesis fails."""
        return ResearchResult(
            answer=f"I encountered an error while processing your question: {error}",
            citations=[],
            conflicts=[],
            confidence_score=0.1,
            source_breakdown={"local": 0, "live": 0},
            total_sources=0,
            self_critique=f"Synthesis failed due to error: {error}"
        )


# Global instance for easy access
_research_agent_instance = None

def get_research_agent() -> ResearchAgent:
    """Get global Research Agent instance."""
    global _research_agent_instance
    if _research_agent_instance is None:
        _research_agent_instance = ResearchAgent()
    return _research_agent_instance
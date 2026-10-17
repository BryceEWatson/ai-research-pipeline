"""Enhanced RAG pipeline with multi-agent orchestration for Phase 2."""

import logging
from typing import List, Dict, Any, Optional

from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from .config import settings, model_selector
from .vector_store import get_vector_store
from .chunking import get_chunker
from .quality_scoring import get_quality_scorer
from .performance_monitor import get_performance_monitor

logger = logging.getLogger(__name__)

# Import Phase 2 agents
try:
    from agents.seed_agent import get_seed_agent
    from agents.sourcing_agent import get_sourcing_agent
    from agents.research_agent import get_research_agent
    AGENTS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Agents not available, falling back to Phase 1 behavior: {e}")
    AGENTS_AVAILABLE = False


class BasicRAGPipeline:
    """
    Enhanced RAG pipeline with multi-agent orchestration.
    
    Phase 1: Basic retrieval-augmented generation
    Phase 2: Multi-agent workflow (Seed → Sourcing → Research)
    
    Maintains backward compatibility while adding live research capabilities.
    """
    
    def __init__(self):
        """Initialize RAG pipeline components."""
        self.vector_store = get_vector_store()
        self.chunker = get_chunker()
        self.quality_scorer = get_quality_scorer()
        self.performance_monitor = get_performance_monitor()
        
        # Phase 1 fallback LLM will be created dynamically
        self.openai_api_key = settings.openai_api_key
        
        # RAG prompt template
        self.rag_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""You are a research assistant that provides accurate, well-sourced answers.

Context:
{context}

Question: {question}

Instructions:
1. Answer based ONLY on the provided context
2. Include citation numbers [1], [2], etc. for each claim
3. If the context doesn't contain enough information, state this clearly
4. Be concise and factual
5. Highlight any conflicts or uncertainties in the sources

Answer:"""
        )
        
        logger.info("Initialized basic RAG pipeline")
    
    def add_document(
        self,
        text: str,
        source_url: str = "",
        title: str = "",
        eval_score: float = 0.7
    ) -> List[str]:
        """
        Add a document to the knowledge base.
        
        Args:
            text: Document content
            source_url: Source URL or identifier
            title: Document title
            eval_score: Quality score (0-1)
            
        Returns:
            List of document IDs added
        """
        try:
            # Create chunked documents
            documents = self.chunker.create_document_from_text(
                text=text,
                source_url=source_url,
                title=title,
                eval_score=eval_score
            )
            
            # Add to vector store
            ids = self.vector_store.add_documents(documents)
            
            logger.info(f"Added document '{title}' with {len(documents)} chunks")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            raise
    
    def retrieve_context(self, question: str, k: int = None) -> List[Document]:
        """
        Retrieve relevant context for a question.
        
        Args:
            question: User question
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        k = k or settings.retrieval_k
        
        try:
            # Retrieve relevant documents
            documents = self.vector_store.similarity_search(
                query=question,
                k=k
            )
            
            logger.info(f"Retrieved {len(documents)} relevant documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return []
    
    def format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into context string."""
        if not documents:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            metadata = doc.metadata
            source_info = f"Source: {metadata.get('title', 'Unknown')} ({metadata.get('source_url', 'Unknown URL')})"
            
            context_part = f"[{i}] {source_info}\n{doc.page_content}\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def generate_answer(self, question: str, context: str) -> str:
        """
        Generate answer using LLM with context.
        
        Args:
            question: User question
            context: Retrieved context
            
        Returns:
            Generated answer
        """
        try:
            # Select appropriate model for generation
            selected_model = model_selector.select_model_for_task("research", complexity=2)
            
            # Create LLM instance
            llm = ChatOpenAI(
                model=selected_model,
                openai_api_key=self.openai_api_key,
                temperature=0.1,
                max_tokens=1000
            )
            
            # Format prompt
            prompt = self.rag_prompt.format(
                context=context,
                question=question
            )
            
            # Generate answer
            response = llm.invoke(prompt)
            
            # Extract content based on response type
            if hasattr(response, 'content'):
                answer = response.content
            else:
                answer = str(response)
            
            logger.info(f"Generated answer successfully using {selected_model}")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    def answer_question(self, question: str, k: int = None, use_agents: bool = True) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve context and generate answer.
        Enhanced in Phase 2 with multi-agent orchestration.
        
        Args:
            question: User question
            k: Number of documents to retrieve (Phase 1 mode only)
            use_agents: Whether to use multi-agent system (Phase 2)
            
        Returns:
            Dictionary containing answer and metadata
        """
        logger.info(f"Processing question: {question}")
        
        # Phase 2: Multi-agent orchestration
        if AGENTS_AVAILABLE and use_agents:
            return self._answer_with_agents(question)
        
        # Phase 1: Original RAG pipeline (fallback)
        return self._answer_phase1(question, k)
    
    def _answer_with_agents(self, question: str) -> Dict[str, Any]:
        """
        Phase 2: Multi-agent orchestration workflow.
        
        Args:
            question: User question
            
        Returns:
            Enhanced response with multi-source synthesis
        """
        try:
            logger.info("Using Phase 2 multi-agent workflow")
            
            # Step 1: Seed Agent - Query decomposition
            seed_agent = get_seed_agent()
            search_plan = seed_agent.decompose_query(question)
            
            logger.info(f"Seed Agent created {len(search_plan.sub_queries)} sub-queries")
            
            # Step 2: Sourcing Agent - Multi-source gathering
            sourcing_agent = get_sourcing_agent()
            sourcing_result = sourcing_agent.gather_sources(search_plan)
            
            logger.info(f"Sourcing Agent gathered {sourcing_result.total_sources} sources "
                       f"({sourcing_result.local_count} local, {sourcing_result.live_count} live)")
            
            # Step 3: Research Agent - Synthesis and analysis
            research_agent = get_research_agent()
            research_result = research_agent.synthesize_research(question, sourcing_result)
            
            logger.info(f"Research Agent completed synthesis with confidence {research_result.confidence_score:.2f}")
            
            # Quality scoring
            source_texts = [doc.content for doc in sourcing_result.documents]
            quality_score = self.quality_scorer.score_answer_quality(
                answer=research_result.answer,
                sources=source_texts,
                query=question,
                confidence_score=research_result.confidence_score
            )
            
            # Format Phase 2 response
            response = {
                "question": question,
                "answer": research_result.answer,
                "confidence_score": research_result.confidence_score,
                "total_sources": research_result.total_sources,
                "source_breakdown": research_result.source_breakdown,
                "citations": [
                    {
                        "id": citation.id,
                        "type": citation.source_type,
                        "title": citation.title,
                        "url": citation.url,
                        "snippet": citation.snippet
                    }
                    for citation in research_result.citations
                ],
                "conflicts": [
                    {
                        "claim": conflict.claim,
                        "sources": conflict.conflicting_sources,
                        "severity": conflict.severity,
                        "description": conflict.description
                    }
                    for conflict in research_result.conflicts
                ],
                "self_critique": research_result.self_critique,
                "execution_time": sourcing_result.execution_time,
                "search_plan": {
                    "strategy": search_plan.strategy,
                    "complexity": search_plan.estimated_complexity,
                    "sub_queries": [sq.query for sq in search_plan.sub_queries]
                },
                "quality_assessment": {
                    "overall_confidence": quality_score.confidence,
                    "fcs_score": quality_score.fcs_score,
                    "explanation": quality_score.explanation,
                    "metrics": quality_score.metrics
                },
                "phase": "2_multi_agent"
            }
            
            logger.info("Completed Phase 2 multi-agent pipeline")
            return response
            
        except Exception as e:
            logger.error(f"Error in multi-agent workflow: {e}, falling back to Phase 1")
            return self._answer_phase1(question)
    
    def _answer_phase1(self, question: str, k: int = None) -> Dict[str, Any]:
        """
        Phase 1: Original RAG pipeline (fallback mode).
        
        Args:
            question: User question
            k: Number of documents to retrieve
            
        Returns:
            Original format response
        """
        logger.info("Using Phase 1 basic RAG workflow")
        
        # Retrieve relevant context
        documents = self.retrieve_context(question, k)
        
        # Format context
        context = self.format_context(documents)
        
        # Generate answer
        answer = self.generate_answer(question, context)
        
        # Compile response
        response = {
            "question": question,
            "answer": answer,
            "context_used": len(documents),
            "sources": [
                {
                    "title": doc.metadata.get("title", "Unknown"),
                    "url": doc.metadata.get("source_url", "Unknown"),
                    "eval_score": doc.metadata.get("eval_score", 0.0)
                }
                for doc in documents
            ],
            "phase": "1_basic_rag"
        }
        
        logger.info("Completed Phase 1 RAG pipeline")
        return response
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        count = self.vector_store.get_collection_count()
        
        return {
            "total_documents": count,
            "collection_name": self.vector_store.collection_name,
            "embedding_model": settings.embedding_model,
            "llm_model": settings.llm_model
        }


# Global RAG pipeline instance
_rag_pipeline: Optional[BasicRAGPipeline] = None


def get_rag_pipeline() -> BasicRAGPipeline:
    """Get or create global RAG pipeline instance."""
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = BasicRAGPipeline()
    return _rag_pipeline
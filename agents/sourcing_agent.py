"""Sourcing Agent for multi-source information gathering."""

import logging
import requests
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from langchain_core.documents import Document

from core.config import settings, model_selector
from core.vector_store import get_vector_store
from core.chunking import get_chunker
from agents.seed_agent import SearchPlan, SubQuery

logger = logging.getLogger(__name__)


@dataclass
class SourcedDocument:
    """Document with source metadata and evaluation score."""
    content: str
    source_url: str
    title: str
    date: Optional[str]
    source_type: str  # "local", "sonar", "deep_research"
    eval_score: float
    chunk_tokens: int
    fcs: Optional[float] = None
    conflicts: List[str] = None
    aligns_with: List[str] = None


@dataclass
class SourcingResult:
    """Complete sourcing result from multiple sources."""
    documents: List[SourcedDocument]
    local_count: int
    live_count: int
    total_tokens: int
    execution_time: float
    
    @property
    def total_sources(self) -> int:
        """Total number of sources (for backward compatibility)."""
        return len(self.documents)


class SourcingAgent:
    """Agent responsible for multi-source information gathering."""
    
    def __init__(self):
        """Initialize the Sourcing Agent."""
        self.vector_store = get_vector_store()
        self.chunker = get_chunker()
        
        # Perplexity Sonar API configuration
        self.perplexity_api_key = settings.perplexity_api_key
        self.sonar_base_url = "https://api.perplexity.ai"
        
        # Rate limiting
        self.last_api_call = 0
        self.min_call_interval = 1.0  # 1 second between calls
        
    def gather_sources(self, search_plan: SearchPlan) -> SourcingResult:
        """
        Gather information from multiple sources based on search plan.
        
        Args:
            search_plan: Plan from Seed Agent
            
        Returns:
            SourcingResult with documents from all sources
        """
        start_time = time.time()
        all_documents = []
        local_count = 0
        live_count = 0
        total_tokens = 0
        
        try:
            # Process sub-queries according to strategy
            if search_plan.strategy == "local_first":
                # Try local first, then live for gaps
                for sub_query in sorted(search_plan.sub_queries, key=lambda x: x.priority):
                    local_docs = self._search_local(sub_query)
                    all_documents.extend(local_docs)
                    local_count += len(local_docs)
                    
                    # If local results are insufficient, try live
                    if len(local_docs) < 3 and sub_query.search_strategy in ["live", "both"]:
                        live_docs = self._search_sonar(sub_query)
                        all_documents.extend(live_docs)
                        live_count += len(live_docs)
                        
            elif search_plan.strategy == "live_first":
                # Try live first, then local for context
                for sub_query in sorted(search_plan.sub_queries, key=lambda x: x.priority):
                    if sub_query.search_strategy in ["live", "both"]:
                        live_docs = self._search_sonar(sub_query)
                        all_documents.extend(live_docs)
                        live_count += len(live_docs)
                    
                    local_docs = self._search_local(sub_query)
                    all_documents.extend(local_docs)
                    local_count += len(local_docs)
                    
            else:  # parallel strategy
                # Execute all searches in priority order
                for sub_query in sorted(search_plan.sub_queries, key=lambda x: x.priority):
                    if sub_query.search_strategy in ["local", "both"]:
                        local_docs = self._search_local(sub_query)
                        all_documents.extend(local_docs)
                        local_count += len(local_docs)
                    
                    if sub_query.search_strategy in ["live", "both"]:
                        live_docs = self._search_sonar(sub_query)
                        all_documents.extend(live_docs)
                        live_count += len(live_docs)
            
            # Deduplicate and evaluate
            all_documents = self._deduplicate_sources(all_documents)
            all_documents = self._evaluate_sources(all_documents)
            
            # CRITICAL: Persist live sources to vector store (Vision.md Step 6)
            newly_added_count = self._persist_live_sources(all_documents)
            
            # Calculate total tokens
            total_tokens = sum(doc.chunk_tokens for doc in all_documents)
            
            execution_time = time.time() - start_time
            
            logger.info(f"Sourcing complete: {local_count} local, {live_count} live, "
                       f"{newly_added_count} newly persisted, {total_tokens} tokens in {execution_time:.2f}s")
            
            return SourcingResult(
                documents=all_documents,
                local_count=local_count,
                live_count=live_count,
                total_tokens=total_tokens,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Error in Sourcing Agent: {e}")
            # Return empty result on error
            return SourcingResult(
                documents=[],
                local_count=0,
                live_count=0,
                total_tokens=0,
                execution_time=time.time() - start_time
            )
    
    def _search_local(self, sub_query: SubQuery) -> List[SourcedDocument]:
        """Search local vector store."""
        try:
            # Use existing vector store search
            results = self.vector_store.similarity_search(
                sub_query.query,
                k=settings.retrieval_k
            )
            
            documents = []
            for doc in results:
                # Convert to SourcedDocument
                sourced_doc = SourcedDocument(
                    content=doc.page_content,
                    source_url=doc.metadata.get("source_url", "local"),
                    title=doc.metadata.get("title", "Local Document"),
                    date=doc.metadata.get("date"),
                    source_type="local",
                    eval_score=doc.metadata.get("eval_score", 0.8),  # Default score for local
                    chunk_tokens=len(doc.page_content.split()) * 1.3,  # Rough token estimate
                    conflicts=[],
                    aligns_with=[]
                )
                documents.append(sourced_doc)
            
            logger.debug(f"Local search found {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Local search error: {e}")
            return []
    
    def _search_sonar(self, sub_query: SubQuery) -> List[SourcedDocument]:
        """Search using Perplexity Sonar API."""
        if not self.perplexity_api_key:
            logger.warning("Perplexity API key not configured, skipping live search")
            return []
        
        try:
            # Rate limiting
            self._enforce_rate_limit()
            
            # Prepare Sonar request
            headers = {
                "Authorization": f"Bearer {self.perplexity_api_key}",
                "Content-Type": "application/json"
            }
            
            # Select appropriate model for live research
            live_research_model = model_selector.select_model_for_task("sourcing_live")
            
            payload = {
                "model": live_research_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a research assistant. Provide comprehensive, factual information with proper citations."
                    },
                    {
                        "role": "user", 
                        "content": sub_query.query
                    }
                ],
                "max_tokens": 2000,
                "temperature": 0.1,
                "return_citations": True,
                "return_images": False
            }
            
            # Make API call
            response = requests.post(
                f"{self.sonar_base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                logger.debug(f"Sonar API response: {data}")
                return self._parse_sonar_response(data, sub_query.query)
            else:
                logger.error(f"Sonar API error: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Sonar search error: {e}")
            return []
    
    def _parse_sonar_response(self, response_data: Dict, query: str) -> List[SourcedDocument]:
        """Parse Perplexity Sonar API response into SourcedDocument objects."""
        documents = []
        
        try:
            # Handle case where response_data might be a string (error case)
            if isinstance(response_data, str):
                logger.error(f"Sonar API returned string instead of dict: {response_data}")
                return []
            
            # Extract main content
            choices = response_data.get("choices", [])
            if not choices:
                logger.error("No choices in Sonar response")
                return []
                
            content = choices[0].get("message", {}).get("content", "")
            if not content:
                logger.error("No content in Sonar response")
                return []
            
            # Extract search results (more detailed than citations)
            search_results = response_data.get("search_results", [])
            citations = response_data.get("citations", [])
            
            # Use search_results if available (preferred), fall back to citations
            if search_results:
                for i, result in enumerate(search_results):
                    sourced_doc = SourcedDocument(
                        content=content,  # Main AI-generated content with citations
                        source_url=result.get("url", f"sonar-result-{i+1}"),
                        title=result.get("title", f"Sonar Result {i+1}"),
                        date=result.get("date", datetime.now().strftime("%Y-%m-%d")),
                        source_type="sonar",
                        eval_score=0.75,  # High score for Sonar results
                        chunk_tokens=int(len(content.split()) * 1.3),
                        conflicts=[],
                        aligns_with=[]
                    )
                    documents.append(sourced_doc)
                    
            elif citations:
                # Fallback to citations if search_results not available
                for i, citation_url in enumerate(citations):
                    sourced_doc = SourcedDocument(
                        content=content,
                        source_url=citation_url if isinstance(citation_url, str) else f"sonar-{i+1}",
                        title=f"Sonar Citation {i+1}",
                        date=datetime.now().strftime("%Y-%m-%d"),
                        source_type="sonar",
                        eval_score=0.7,
                        chunk_tokens=int(len(content.split()) * 1.3),
                        conflicts=[],
                        aligns_with=[]
                    )
                    documents.append(sourced_doc)
            else:
                # Single document without specific sources
                sourced_doc = SourcedDocument(
                    content=content,
                    source_url="perplexity-sonar",
                    title=f"Sonar Research: {query[:50]}",
                    date=datetime.now().strftime("%Y-%m-%d"),
                    source_type="sonar",
                    eval_score=0.65,
                    chunk_tokens=int(len(content.split()) * 1.3),
                    conflicts=[],
                    aligns_with=[]
                )
                documents.append(sourced_doc)
            
            logger.debug(f"Sonar search returned {len(documents)} documents")
            return documents
            
        except Exception as e:
            logger.error(f"Error parsing Sonar response: {e}")
            return []
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting for API calls."""
        current_time = time.time()
        time_since_last = current_time - self.last_api_call
        
        if time_since_last < self.min_call_interval:
            sleep_time = self.min_call_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_api_call = time.time()
    
    def _deduplicate_sources(self, documents: List[SourcedDocument]) -> List[SourcedDocument]:
        """Remove duplicate documents based on content similarity."""
        if not documents:
            return documents
        
        # Simple deduplication by URL and content length
        seen_sources = set()
        unique_docs = []
        
        for doc in documents:
            # Create a key based on source URL and content length
            key = (doc.source_url, len(doc.content))
            
            if key not in seen_sources:
                seen_sources.add(key)
                unique_docs.append(doc)
        
        logger.debug(f"Deduplicated {len(documents)} -> {len(unique_docs)} documents")
        return unique_docs
    
    def _evaluate_sources(self, documents: List[SourcedDocument]) -> List[SourcedDocument]:
        """Evaluate and score source quality."""
        for doc in documents:
            # Enhanced evaluation scoring
            base_score = doc.eval_score
            
            # Adjust based on source type
            if doc.source_type == "local":
                base_score += 0.1  # Boost local sources
            elif doc.source_type == "sonar":
                # Recent content gets boost
                if doc.date and doc.date == datetime.now().strftime("%Y-%m-%d"):
                    base_score += 0.05
            
            # Adjust based on content length (longer = more comprehensive)
            if doc.chunk_tokens > 1000:
                base_score += 0.05
            elif doc.chunk_tokens < 100:
                base_score -= 0.1
            
            # Keep score in valid range
            doc.eval_score = max(0.0, min(1.0, base_score))
        
        # Sort by evaluation score
        documents.sort(key=lambda x: x.eval_score, reverse=True)
        
        return documents
    
    def _persist_live_sources(self, documents: List[SourcedDocument]) -> int:
        """
        Persist live research sources to vector store (Vision.md Step 6: Embed & Persist).
        
        Args:
            documents: List of sourced documents
            
        Returns:
            Number of new documents added to vector store
        """
        from langchain_core.documents import Document
        
        # Only persist live sources (not local ones that are already in the vector store)
        live_documents = [doc for doc in documents if doc.source_type in ["sonar", "deep_research"]]
        
        if not live_documents:
            logger.debug("No live documents to persist")
            return 0
        
        newly_added_count = 0
        
        try:
            for sourced_doc in live_documents:
                # Check if this source is already in the vector store
                if self._is_source_already_stored(sourced_doc):
                    logger.debug(f"Source already stored: {sourced_doc.source_url}")
                    continue
                
                # Create LangChain Document for vector store
                doc = Document(
                    page_content=sourced_doc.content,
                    metadata={
                        "title": sourced_doc.title,
                        "source_url": sourced_doc.source_url,
                        "date": sourced_doc.date,
                        "eval_score": sourced_doc.eval_score,
                        "source_type": sourced_doc.source_type,
                        "chunk_tokens": sourced_doc.chunk_tokens,
                        "auto_added": True,  # Mark as automatically added
                        "added_timestamp": datetime.now().isoformat()
                    }
                )
                
                # Add to vector store (this will automatically chunk and embed)
                ids = self.vector_store.add_documents([doc])
                newly_added_count += len(ids)
                
                logger.info(f"Persisted live source: '{sourced_doc.title}' ({len(ids)} chunks)")
            
            if newly_added_count > 0:
                logger.info(f"Successfully persisted {newly_added_count} new chunks from {len(live_documents)} live sources")
            
            return newly_added_count
            
        except Exception as e:
            logger.error(f"Error persisting live sources: {e}")
            return 0
    
    def _is_source_already_stored(self, sourced_doc: SourcedDocument) -> bool:
        """
        Check if a source is already stored in the vector store.
        
        Args:
            sourced_doc: Source document to check
            
        Returns:
            True if source is already stored
        """
        try:
            # Search for existing documents with the same source URL
            existing_docs = self.vector_store.similarity_search(
                query=sourced_doc.title,
                k=5,
                filter_dict={"source_url": sourced_doc.source_url}
            )
            
            # Check if any existing document has similar content
            for existing_doc in existing_docs:
                # Simple content similarity check (first 200 characters)
                existing_preview = existing_doc.page_content[:200].strip()
                new_preview = sourced_doc.content[:200].strip()
                
                # If content is very similar, consider it a duplicate
                if existing_preview and new_preview:
                    # Simple similarity: if 80% of words match
                    existing_words = set(existing_preview.lower().split())
                    new_words = set(new_preview.lower().split())
                    
                    if existing_words and new_words:
                        overlap = len(existing_words.intersection(new_words))
                        similarity = overlap / max(len(existing_words), len(new_words))
                        
                        if similarity > 0.8:
                            return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking if source exists: {e}")
            # If we can't check, assume it doesn't exist to avoid missing new content
            return False


# Global instance for easy access
_sourcing_agent_instance = None

def get_sourcing_agent() -> SourcingAgent:
    """Get global Sourcing Agent instance."""
    global _sourcing_agent_instance
    if _sourcing_agent_instance is None:
        _sourcing_agent_instance = SourcingAgent()
    return _sourcing_agent_instance
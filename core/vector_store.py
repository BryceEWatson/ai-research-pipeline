"""Vector store management using Chroma DB."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import settings

logger = logging.getLogger(__name__)


class VectorStore:
    """Manages vector storage and retrieval using Chroma DB."""
    
    def __init__(self, collection_name: str = "research_documents"):
        """Initialize vector store with Chroma DB."""
        self.collection_name = collection_name
        
        # Ensure data directory exists
        db_path = Path(settings.chroma_db_path)
        db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key
        )
        
        # Initialize Chroma client
        self.client = chromadb.PersistentClient(
            path=str(db_path),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        # Initialize vector store
        self.vector_store = Chroma(
            client=self.client,
            collection_name=collection_name,
            embedding_function=self.embeddings
        )
        
        logger.info(f"Initialized vector store at {db_path}")
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to the vector store."""
        try:
            # Add documents and return IDs
            ids = self.vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector store")
            return ids
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    def similarity_search(
        self, 
        query: str, 
        k: int = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Search for similar documents."""
        k = k or settings.retrieval_k
        
        try:
            # Apply default filter for evaluation score if not specified
            if filter_dict is None:
                filter_dict = {"eval_score": {"$gt": settings.min_eval_score}}
            
            results = self.vector_store.similarity_search(
                query=query,
                k=k,
                filter=filter_dict
            )
            
            logger.info(f"Retrieved {len(results)} documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            # Fallback to search without filter
            try:
                results = self.vector_store.similarity_search(query=query, k=k)
                logger.warning("Performed search without filters due to error")
                return results
            except Exception as fallback_error:
                logger.error(f"Fallback search also failed: {fallback_error}")
                return []
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[tuple]:
        """Search for similar documents with similarity scores."""
        k = k or settings.retrieval_k
        
        try:
            if filter_dict is None:
                filter_dict = {"eval_score": {"$gt": settings.min_eval_score}}
            
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=k,
                filter=filter_dict
            )
            
            logger.info(f"Retrieved {len(results)} documents with scores")
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search with score: {e}")
            return []
    
    def get_collection_count(self) -> int:
        """Get total number of documents in collection."""
        try:
            collection = self.client.get_collection(self.collection_name)
            return collection.count()
        except Exception as e:
            logger.error(f"Error getting collection count: {e}")
            return 0
    
    def delete_collection(self):
        """Delete the entire collection."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
    
    def reset_collection(self):
        """Reset collection by deleting and recreating it."""
        try:
            self.delete_collection()
            # Reinitialize vector store
            self.vector_store = Chroma(
                client=self.client,
                collection_name=self.collection_name,
                embedding_function=self.embeddings
            )
            logger.info("Reset collection successfully")
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
    
    def get_knowledge_base_report(self) -> Dict[str, Any]:
        """Generate comprehensive knowledge base report."""
        try:
            collection = self.client.get_collection(self.collection_name)
            
            # Get all documents with metadata
            results = collection.get(
                include=['documents', 'metadatas']
            )
            
            if not results['documents']:
                return {
                    "total_documents": 0,
                    "documents": [],
                    "topics": [],
                    "sources": [],
                    "content_summary": "Knowledge base is empty"
                }
            
            documents = []
            all_sources = set()
            all_titles = set()
            total_content_length = 0
            
            # Process each document
            for i, (doc_text, metadata) in enumerate(zip(
                results['documents'], 
                results['metadatas']
            )):
                doc_info = {
                    "id": f"doc_{i+1}",
                    "title": metadata.get('title', 'Unknown'),
                    "source_url": metadata.get('source_url', 'Unknown'),
                    "eval_score": metadata.get('eval_score', 0.0),
                    "content_length": len(doc_text),
                    "content_preview": doc_text[:200] + "..." if len(doc_text) > 200 else doc_text
                }
                documents.append(doc_info)
                
                # Collect sources and titles
                if metadata.get('source_url'):
                    all_sources.add(metadata['source_url'])
                if metadata.get('title'):
                    all_titles.add(metadata['title'])
                
                total_content_length += len(doc_text)
            
            # Generate topics by analyzing document titles and content
            topics = self._extract_topics_from_documents(results['documents'], results['metadatas'])
            
            return {
                "total_documents": len(documents),
                "total_content_length": total_content_length,
                "average_document_length": total_content_length / len(documents) if documents else 0,
                "unique_sources": len(all_sources),
                "unique_titles": len(all_titles),
                "documents": documents,
                "topics": topics,
                "sources": list(all_sources),
                "titles": list(all_titles)
            }
            
        except Exception as e:
            logger.error(f"Error generating knowledge base report: {e}")
            return {
                "error": str(e),
                "total_documents": 0,
                "documents": []
            }
    
    def _extract_topics_from_documents(self, documents: List[str], metadatas: List[Dict]) -> List[Dict[str, Any]]:
        """Extract key topics from document content and titles."""
        import re
        from collections import Counter
        
        # Common AI/ML/Tech keywords to look for
        topic_keywords = {
            "artificial_intelligence": ["artificial intelligence", "ai", "intelligent systems"],
            "machine_learning": ["machine learning", "ml", "algorithms", "training", "model"],
            "retrieval_augmented_generation": ["rag", "retrieval", "augmented", "generation"],
            "natural_language_processing": ["nlp", "natural language", "text processing"],
            "data_science": ["data science", "analytics", "statistics"],
            "neural_networks": ["neural", "networks", "deep learning"],
            "computer_vision": ["computer vision", "image", "visual"],
            "robotics": ["robot", "automation", "robotics"]
        }
        
        topic_counts = Counter()
        
        # Analyze document titles and content
        all_text = ""
        for doc, metadata in zip(documents, metadatas):
            title = metadata.get('title', '')
            all_text += f" {title} {doc}".lower()
        
        # Count keyword occurrences
        for topic, keywords in topic_keywords.items():
            for keyword in keywords:
                count = len(re.findall(r'\b' + re.escape(keyword.lower()) + r'\b', all_text))
                if count > 0:
                    topic_counts[topic] += count
        
        # Convert to list of dicts
        topics = []
        for topic, count in topic_counts.most_common():
            topics.append({
                "topic": topic.replace("_", " ").title(),
                "mentions": count,
                "relevance": "high" if count > 5 else "medium" if count > 2 else "low"
            })
        
        return topics


# Global vector store instance
_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Get or create global vector store instance."""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
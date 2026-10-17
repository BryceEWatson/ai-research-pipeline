"""Basic tests for RAG pipeline functionality."""

from core.config import settings
from core.chunking import get_chunker
from core.vector_store import get_vector_store
from core.rag_pipeline import get_rag_pipeline


class TestChunking:
    """Test document chunking functionality."""
    
    def test_chunker_initialization(self):
        """Test that chunker initializes correctly."""
        chunker = get_chunker()
        assert chunker.chunk_size == settings.chunk_size
        assert chunker.chunk_overlap == settings.chunk_overlap
    
    def test_text_chunking(self):
        """Test basic text chunking."""
        chunker = get_chunker()
        text = "This is a test document. " * 100  # Create longer text
        
        documents = chunker.chunk_text(text)
        
        assert len(documents) > 0
        assert all(doc.page_content for doc in documents)
        assert all(doc.metadata for doc in documents)
    
    def test_document_creation(self):
        """Test creating documents with metadata."""
        chunker = get_chunker()
        
        documents = chunker.create_document_from_text(
            text="Test content for document creation.",
            title="Test Document",
            source_url="https://test.com",
            eval_score=0.8
        )
        
        assert len(documents) > 0
        doc = documents[0]
        assert doc.metadata["title"] == "Test Document"
        assert doc.metadata["source_url"] == "https://test.com"
        assert doc.metadata["eval_score"] == 0.8


class TestVectorStore:
    """Test vector store functionality."""
    
    def test_vector_store_initialization(self):
        """Test that vector store initializes correctly."""
        vector_store = get_vector_store()
        assert vector_store.collection_name == "research_documents"
        assert vector_store.embeddings is not None
    
    def test_get_collection_count(self):
        """Test getting collection count."""
        vector_store = get_vector_store()
        count = vector_store.get_collection_count()
        assert isinstance(count, int)
        assert count >= 0


class TestRAGPipeline:
    """Test RAG pipeline functionality."""
    
    def test_rag_pipeline_initialization(self):
        """Test that RAG pipeline initializes correctly."""
        pipeline = get_rag_pipeline()
        assert pipeline.vector_store is not None
        assert pipeline.chunker is not None
        assert pipeline.quality_scorer is not None
        assert pipeline.performance_monitor is not None
        # Note: LLM is now created dynamically, not stored as instance attribute
    
    def test_format_context(self):
        """Test context formatting."""
        pipeline = get_rag_pipeline()
        
        # Create mock documents
        from langchain_core.documents import Document
        docs = [
            Document(
                page_content="Test content 1",
                metadata={"title": "Test Doc 1", "source_url": "https://test1.com"}
            ),
            Document(
                page_content="Test content 2", 
                metadata={"title": "Test Doc 2", "source_url": "https://test2.com"}
            )
        ]
        
        context = pipeline.format_context(docs)
        assert "[1]" in context
        assert "[2]" in context
        assert "Test content 1" in context
        assert "Test content 2" in context
    
    def test_get_knowledge_base_stats(self):
        """Test getting knowledge base statistics."""
        pipeline = get_rag_pipeline()
        stats = pipeline.get_knowledge_base_stats()
        
        assert "total_documents" in stats
        assert "collection_name" in stats
        assert "embedding_model" in stats
        assert "llm_model" in stats
        assert isinstance(stats["total_documents"], int)


class TestConfiguration:
    """Test configuration management."""
    
    def test_settings_load(self):
        """Test that settings load correctly."""
        assert settings.chunk_size > 0
        assert settings.chunk_overlap > 0
        assert settings.retrieval_k > 0
        assert 0 <= settings.min_eval_score <= 1
        assert 0 <= settings.min_fcs_threshold <= 1
    
    def test_model_settings_are_set(self):
        """Test that embedding and fallback LLM models are configured."""
        assert settings.embedding_model.strip()
        assert settings.llm_model.strip()

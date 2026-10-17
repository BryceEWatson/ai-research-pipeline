"""Document chunking utilities for optimal retrieval."""

import logging
from typing import List, Dict, Any
from datetime import datetime, timezone

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

from .config import settings

logger = logging.getLogger(__name__)


class AdaptiveChunker:
    """Adaptive document chunking with metadata preservation."""
    
    def __init__(self):
        """Initialize chunker with settings from config."""
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        
        # Initialize tiktoken encoder for token counting
        if TIKTOKEN_AVAILABLE:
            try:
                self.encoding = tiktoken.encoding_for_model("gpt-4")
            except KeyError:
                self.encoding = tiktoken.get_encoding("cl100k_base")
        else:
            logger.warning("tiktoken not available, using word count approximation")
            self.encoding = None
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=self._count_tokens,
            separators=["\n\n", "\n", " ", ""]
        )
        
        logger.info(f"Initialized chunker: size={self.chunk_size}, overlap={self.chunk_overlap}")
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken or word approximation."""
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Fallback: approximate 1 token = 0.75 words
            return int(len(text.split()) * 1.33)
    
    def chunk_text(
        self, 
        text: str, 
        metadata: Dict[str, Any] = None
    ) -> List[Document]:
        """
        Chunk text into documents with metadata.
        
        Args:
            text: Input text to chunk
            metadata: Base metadata to apply to all chunks
            
        Returns:
            List of Document objects with chunked text and metadata
        """
        if not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        # Initialize base metadata
        base_metadata = metadata or {}
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create documents with enhanced metadata
        documents = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = base_metadata.copy()
            
            # Add chunk-specific metadata
            chunk_metadata.update({
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_tokens": self._count_tokens(chunk),
                "chunk_chars": len(chunk),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "eval_score": base_metadata.get("eval_score", 0.5),  # Default eval score
            })
            
            # Ensure chunk doesn't exceed token limit
            if chunk_metadata["chunk_tokens"] > 2000:
                logger.warning(f"Chunk {i} has {chunk_metadata['chunk_tokens']} tokens (>2000)")
                # Further split if needed
                sub_chunks = self._split_large_chunk(chunk)
                for j, sub_chunk in enumerate(sub_chunks):
                    sub_metadata = chunk_metadata.copy()
                    sub_metadata.update({
                        "chunk_index": f"{i}.{j}",
                        "chunk_tokens": self._count_tokens(sub_chunk),
                        "chunk_chars": len(sub_chunk),
                    })
                    documents.append(Document(page_content=sub_chunk, metadata=sub_metadata))
            else:
                documents.append(Document(page_content=chunk, metadata=chunk_metadata))
        
        logger.info(f"Created {len(documents)} chunks from {self._count_tokens(text)} tokens")
        return documents
    
    def _split_large_chunk(self, chunk: str) -> List[str]:
        """Split chunks that are too large into smaller pieces."""
        # Use a smaller chunk size for oversized chunks
        smaller_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Smaller size for safety
            chunk_overlap=100,
            length_function=self._count_tokens,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return smaller_splitter.split_text(chunk)
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk a list of documents.
        
        Args:
            documents: List of Document objects to chunk
            
        Returns:
            List of chunked Document objects
        """
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunk_text(doc.page_content, doc.metadata)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def create_document_from_text(
        self,
        text: str,
        source_url: str = "",
        title: str = "",
        date: str = "",
        eval_score: float = 0.5,
        **additional_metadata
    ) -> List[Document]:
        """
        Create chunked documents from raw text with full metadata.
        
        Args:
            text: Source text content
            source_url: URL or source identifier
            title: Document title
            date: Publication date (YYYY-MM-DD format)
            eval_score: Quality/credibility score (0-1)
            **additional_metadata: Any additional metadata fields
            
        Returns:
            List of chunked Document objects
        """
        metadata = {
            "source_url": source_url,
            "title": title,
            "date": date or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "eval_score": eval_score,
            # Note: conflicts and aligns_with removed for Chroma compatibility
            # They will be handled separately in future phases
            **additional_metadata
        }
        
        return self.chunk_text(text, metadata)


# Global chunker instance
_chunker: AdaptiveChunker = None


def get_chunker() -> AdaptiveChunker:
    """Get or create global chunker instance."""
    global _chunker
    if _chunker is None:
        _chunker = AdaptiveChunker()
    return _chunker
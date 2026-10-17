"""Script to populate the knowledge base with sample documents."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.rag_pipeline import get_rag_pipeline
from data.sample_documents import get_sample_documents


def setup_sample_data():
    """Add sample documents to the knowledge base."""
    print("Setting up sample data...")
    
    try:
        # Initialize RAG pipeline
        rag_pipeline = get_rag_pipeline()
        
        # Get sample documents
        documents = get_sample_documents()
        
        # Add each document
        for doc in documents:
            print(f"Adding: {doc['title']}")
            
            ids = rag_pipeline.add_document(
                text=doc['content'],
                title=doc['title'],
                source_url=doc['source_url'],
                eval_score=doc['eval_score']
            )
            
            print(f"  Added with {len(ids)} chunks")
        
        # Show final stats
        stats = rag_pipeline.get_knowledge_base_stats()
        print(f"\nSetup complete! Knowledge base now contains {stats['total_documents']} documents.")
        
    except Exception as e:
        print(f"Error setting up sample data: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = setup_sample_data()
    sys.exit(0 if success else 1)
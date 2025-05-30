"""Quick start script for AI Research Pipeline."""

import sys
from pathlib import Path
import os

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_environment():
    """Check if environment is properly configured."""
    print("🔍 Checking environment...")
    
    # Check if .env file exists
    env_file = project_root / ".env"
    if not env_file.exists():
        print("❌ .env file not found. Please copy .env.example to .env and add your API keys.")
        return False
    
    # Check for OpenAI API key
    from core.config import settings
    try:
        if not settings.openai_api_key or settings.openai_api_key == "your_openai_api_key_here":
            print("❌ OpenAI API key not configured. Please set OPENAI_API_KEY in .env file.")
            return False
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return False
    
    print("✅ Environment configuration looks good!")
    return True


def test_pipeline():
    """Test the basic RAG pipeline."""
    print("\n🧪 Testing RAG pipeline...")
    
    try:
        from core.rag_pipeline import get_rag_pipeline
        
        # Initialize pipeline
        print("  - Initializing pipeline...")
        pipeline = get_rag_pipeline()
        
        # Check stats
        stats = pipeline.get_knowledge_base_stats()
        print(f"  - Knowledge base has {stats['total_documents']} documents")
        
        # Add a simple test document if knowledge base is empty
        if stats['total_documents'] == 0:
            print("  - Adding test document...")
            pipeline.add_document(
                text="Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines that can think and learn like humans.",
                title="Introduction to AI",
                source_url="https://example.com/ai-intro",
                eval_score=0.8
            )
            print("  - Test document added successfully!")
        
        # Test a simple query
        print("  - Testing query...")
        response = pipeline.answer_question("What is artificial intelligence?")
        
        if response and response.get('answer'):
            print("✅ Pipeline test successful!")
            print(f"  Sample answer: {response['answer'][:100]}...")
            return True
        else:
            print("❌ Pipeline test failed - no answer generated")
            return False
            
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False


def setup_sample_data():
    """Set up sample data in the knowledge base."""
    print("\n📚 Setting up sample data...")
    
    try:
        from scripts.setup_sample_data import setup_sample_data
        success = setup_sample_data()
        
        if success:
            print("✅ Sample data setup complete!")
            return True
        else:
            print("❌ Sample data setup failed")
            return False
            
    except Exception as e:
        print(f"❌ Error setting up sample data: {e}")
        return False


def show_usage_examples():
    """Show usage examples."""
    print("\n📖 Usage Examples:")
    print("=" * 50)
    
    examples = [
        ("Ask a question", "python app/cli.py ask \"What is retrieval-augmented generation?\""),
        ("Add a document", "python app/cli.py add \"Your document content here\" --title \"My Document\""),
        ("Add from file", "python app/cli.py add-file path/to/document.txt --title \"File Document\""),
        ("Show statistics", "python app/cli.py stats"),
        ("Reset knowledge base", "python app/cli.py reset"),
    ]
    
    for description, command in examples:
        print(f"\n{description}:")
        print(f"  {command}")
    
    print("\nFor more options, run: python app/cli.py --help")


def main():
    """Main quickstart function."""
    print("🚀 AI Research Pipeline - Quick Start")
    print("=" * 40)
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment check failed. Please fix the issues above and try again.")
        return False
    
    # Test pipeline
    if not test_pipeline():
        print("\n❌ Pipeline test failed. Please check your configuration and try again.")
        return False
    
    # Offer to set up sample data
    current_stats = None
    try:
        from core.rag_pipeline import get_rag_pipeline
        pipeline = get_rag_pipeline()
        current_stats = pipeline.get_knowledge_base_stats()
    except:
        pass
    
    if current_stats and current_stats['total_documents'] < 5:
        response = input("\n🤔 Would you like to add sample documents to test with? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            setup_sample_data()
    
    # Show usage examples
    show_usage_examples()
    
    print("\n✅ Quick start complete! Your AI Research Pipeline is ready to use.")
    print("🎯 Try asking a question with: python app/cli.py ask \"Your question here\"")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
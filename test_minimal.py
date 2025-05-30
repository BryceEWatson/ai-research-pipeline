"""Minimal test to check what's actually working."""

import sys
import os
from pathlib import Path

def test_basic_imports():
    """Test basic Python functionality."""
    print("✓ Python basic imports working")
    
    # Test pathlib
    current_dir = Path(".")
    print(f"✓ Current directory: {current_dir.resolve()}")
    
    # Test if our files exist
    files_to_check = [
        "core/config.py",
        "core/vector_store.py", 
        "app/cli.py",
        "requirements.txt"
    ]
    
    for file_path in files_to_check:
        if Path(file_path).exists():
            print(f"✓ File exists: {file_path}")
        else:
            print(f"❌ File missing: {file_path}")

def test_imports():
    """Test imports of our modules."""
    try:
        # Test if we can import our modules
        sys.path.insert(0, str(Path.cwd()))
        
        # Try importing dotenv (simple dependency)
        try:
            from dotenv import load_dotenv
            print("✓ python-dotenv available")
        except ImportError as e:
            print(f"❌ python-dotenv not available: {e}")
        
        # Try importing our config
        try:
            from core.config import Settings
            print("✓ Core config imports successfully")
        except ImportError as e:
            print(f"❌ Core config import failed: {e}")
            
    except Exception as e:
        print(f"❌ Import test failed: {e}")

if __name__ == "__main__":
    print("🧪 Running minimal functionality test...\n")
    
    test_basic_imports()
    print()
    test_imports()
    
    print("\n✅ Minimal test complete")
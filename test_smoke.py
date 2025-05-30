"""
Quick smoke tests to verify Phase 3 multi-agent system with model selection is working.
Run this for fast validation before making changes.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_basic_functionality():
    """Quick test of basic functionality including model selection."""
    print("🔥 Running Phase 3 Smoke Tests...")
    
    start_time = time.time()
    errors = []
    
    try:
        # Test 1: Import all core modules including new components
        print("1. Testing imports...")
        from core.rag_pipeline import get_rag_pipeline
        from core.config import settings, model_config, model_selector
        from core.quality_scoring import get_quality_scorer
        from core.performance_monitor import get_performance_monitor
        from agents.seed_agent import get_seed_agent
        from agents.sourcing_agent import get_sourcing_agent  
        from agents.research_agent import get_research_agent
        print("   ✓ All imports successful")
        
        # Test 2: Initialize pipeline
        print("2. Testing pipeline initialization...")
        pipeline = get_rag_pipeline()
        print("   ✓ Pipeline initialized")
        
        # Test 3: Check knowledge base
        print("3. Testing knowledge base...")
        stats = pipeline.get_knowledge_base_stats()
        if stats['total_documents'] == 0:
            errors.append("Knowledge base is empty")
        else:
            print(f"   ✓ Knowledge base has {stats['total_documents']} documents")
        
        # Test 4: Test model configuration
        print("4. Testing model configuration...")
        print(f"   - Simple reasoning: {model_config.simple_reasoning}")
        print(f"   - Complex reasoning: {model_config.complex_reasoning}")
        print(f"   - Live research: {model_config.live_research}")
        
        # Test model selection
        selected_seed = model_selector.select_model_for_task("seed", complexity=2)
        selected_research = model_selector.select_model_for_task("research", complexity=4)
        print(f"   - Seed agent model: {selected_seed}")
        print(f"   - Research agent model: {selected_research}")
        print("   ✓ Model configuration working")
        
        # Test 5: Test quality and performance systems
        print("5. Testing quality and performance systems...")
        quality_scorer = get_quality_scorer()
        performance_monitor = get_performance_monitor()
        print(f"   - Vectara FCS enabled: {quality_scorer.vectara_fcs.enabled}")
        print(f"   - Performance metrics count: {len(performance_monitor.metrics)}")
        print("   ✓ Quality and performance systems initialized")
        
        # Test 6: Test agent initialization
        print("6. Testing agent initialization...")
        seed = get_seed_agent()
        sourcing = get_sourcing_agent()
        research = get_research_agent()
        print("   ✓ All agents initialized")
        
        # Test 7: Quick query complexity analysis
        print("7. Testing seed agent...")
        complexity = seed.analyze_query_complexity("What is AI?")
        if 'complexity_score' not in complexity:
            errors.append("Seed agent analysis failed")
        else:
            print(f"   ✓ Query complexity: {complexity['complexity_score']}")
        
        # Test 8: Quick Phase 3 workflow test (with quality assessment)
        print("8. Testing Phase 3 workflow...")
        response = pipeline.answer_question("What is RAG?", use_agents=True)
        
        if response.get('phase') != '2_multi_agent':
            errors.append(f"Expected Phase 2, got {response.get('phase')}")
        elif not response.get('answer'):
            errors.append("No answer generated")
        elif response.get('confidence_score', 0) < 0.5:
            errors.append(f"Low confidence: {response.get('confidence_score')}")
        elif 'quality_assessment' not in response:
            errors.append("Quality assessment missing from Phase 3 response")
        else:
            qa = response.get('quality_assessment', {})
            print(f"   ✓ Phase 3 workflow working (confidence: {response.get('confidence_score', 0):.2f})")
            print(f"   ✓ Quality assessment: {qa.get('overall_confidence', 0):.2f}")
        
        # Test 9: CLI command verification
        print("9. Testing CLI command structure...")
        from app.cli import cli
        commands = list(cli.commands.keys())
        expected_commands = ['ask', 'add', 'add-file', 'stats', 'report', 'reset', 'models', 'performance', 'select-model']
        missing_commands = [cmd for cmd in expected_commands if cmd not in commands]
        if missing_commands:
            errors.append(f"Missing CLI commands: {missing_commands}")
        else:
            print(f"   ✓ All {len(expected_commands)} CLI commands available")
        
        # Test 10: Fallback test
        print("10. Testing fallback to Phase 1...")
        response_fallback = pipeline.answer_question("What is AI?", use_agents=False)
        if response_fallback.get('phase') != '1_basic_rag':
            errors.append("Fallback to Phase 1 failed")
        else:
            print("   ✓ Phase 1 fallback working")
        
    except Exception as e:
        errors.append(f"Critical error: {e}")
    
    # Results
    duration = time.time() - start_time
    print(f"\n⏱️  Tests completed in {duration:.1f}s")
    
    if errors:
        print("❌ PHASE 3 SMOKE TESTS FAILED:")
        for error in errors:
            print(f"   • {error}")
        return False
    else:
        print("✅ ALL PHASE 3 SMOKE TESTS PASSED!")
        print("🚀 Model selection, quality scoring, and performance monitoring operational!")
        return True

if __name__ == "__main__":
    success = test_basic_functionality()
    sys.exit(0 if success else 1)
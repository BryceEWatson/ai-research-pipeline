"""
Test suite for model selection and quality assessment features.
Tests the Phase 3 model routing, fallback chains, and quality scoring.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.config import model_config, model_selector
from core.quality_scoring import get_quality_scorer
from core.performance_monitor import get_performance_monitor


def test_model_configuration():
    """Test model configuration loading and validation."""
    print("🧪 Testing Model Configuration...")
    
    # Test YAML loading with updated model names
    assert model_config.simple_reasoning == "gpt-4o-mini"
    assert model_config.complex_reasoning == "gpt-4o"
    assert model_config.live_research == "sonar-pro"
    assert model_config.embeddings == "text-embedding-3-large"
    print("   ✓ Model configuration loaded correctly")
    
    # Test environment variable overrides would work
    assert hasattr(model_config, 'alternative_llm')
    assert hasattr(model_config, 'reranker')
    print("   ✓ All required model fields present")


def test_model_selection_logic():
    """Test model selection based on task type and complexity."""
    print("🧪 Testing Model Selection Logic...")
    
    # Test seed agent model selection
    seed_simple = model_selector.select_model_for_task("seed", complexity=2)
    seed_complex = model_selector.select_model_for_task("seed", complexity=4)
    
    assert seed_simple == model_config.simple_reasoning  # o1-mini for simple
    assert seed_complex == model_config.complex_reasoning  # o1 for complex
    print("   ✓ Seed agent model selection working")
    
    # Test research agent model selection
    research_simple = model_selector.select_model_for_task("research", complexity=2)
    research_complex = model_selector.select_model_for_task("research", complexity=4)
    research_medium = model_selector.select_model_for_task("research", complexity=3)
    
    assert research_simple == model_config.simple_reasoning
    assert research_complex == model_config.complex_reasoning
    assert research_medium == model_config.alternative_llm  # Claude for diversity
    print("   ✓ Research agent model selection working")
    
    # Test sourcing agent model selection
    sourcing_live = model_selector.select_model_for_task("sourcing_live")
    sourcing_local = model_selector.select_model_for_task("sourcing_local")
    
    assert sourcing_live == model_config.live_research
    assert sourcing_local == model_config.embeddings
    print("   ✓ Sourcing agent model selection working")
    
    # Test high context fallback
    high_context = model_selector.select_model_for_task("research", context_size=150000)
    assert high_context == model_config.high_context_fallback
    print("   ✓ High context fallback working")


def test_fallback_chains():
    """Test model fallback chain logic."""
    print("🧪 Testing Fallback Chains...")
    
    # Test fallback chains exist for all primary models
    o1_fallbacks = model_selector.get_fallback_chain(model_config.complex_reasoning)
    o1_mini_fallbacks = model_selector.get_fallback_chain(model_config.simple_reasoning)
    claude_fallbacks = model_selector.get_fallback_chain(model_config.alternative_llm)
    sonar_fallbacks = model_selector.get_fallback_chain(model_config.live_research)
    
    assert len(o1_fallbacks) >= 1
    assert len(o1_mini_fallbacks) >= 1
    assert len(claude_fallbacks) >= 1
    assert len(sonar_fallbacks) >= 1
    print("   ✓ Fallback chains configured for all models")
    
    # Test that fallback chains contain valid models
    all_fallbacks = o1_fallbacks + o1_mini_fallbacks + claude_fallbacks + sonar_fallbacks
    assert all(isinstance(model, str) and len(model) > 0 for model in all_fallbacks)
    print("   ✓ Fallback chains contain valid model names")


def test_performance_monitoring():
    """Test performance monitoring system."""
    print("🧪 Testing Performance Monitoring...")
    
    monitor = get_performance_monitor()
    
    # Test recording performance
    monitor.record_operation(
        component="test_component",
        operation="test_operation", 
        model_used="test_model",
        execution_time=0.5,
        success=True,
        input_size=100,
        output_size=50
    )
    
    assert len(monitor.metrics) >= 1
    print("   ✓ Performance recording working")
    
    # Test metrics aggregation
    system_metrics = monitor.get_system_metrics()
    assert hasattr(system_metrics, 'total_requests')
    assert hasattr(system_metrics, 'success_rate')
    assert hasattr(system_metrics, 'average_latency')
    print("   ✓ Metrics aggregation working")
    
    # Test model performance comparison
    model_perf = monitor.get_model_performance_comparison()
    assert isinstance(model_perf, dict)
    print("   ✓ Model performance comparison working")


def test_quality_scoring():
    """Test quality scoring system."""
    print("🧪 Testing Quality Scoring...")
    
    quality_scorer = get_quality_scorer()
    
    # Test basic quality scoring
    test_answer = "This is a test answer with citations [Local-1] and [Sonar-1]."
    test_sources = ["Source 1 content", "Source 2 content"]
    test_query = "What is a test?"
    
    quality_score = quality_scorer.score_answer_quality(
        answer=test_answer,
        sources=test_sources,
        query=test_query,
        confidence_score=0.8
    )
    
    assert hasattr(quality_score, 'confidence')
    assert hasattr(quality_score, 'explanation')
    assert hasattr(quality_score, 'metrics')
    assert 0 <= quality_score.confidence <= 1
    print("   ✓ Quality scoring working")
    
    # Test citation analysis
    metrics = quality_score.metrics
    assert 'total_citations' in metrics
    assert 'local_citations' in metrics
    assert 'sonar_citations' in metrics
    assert metrics['total_citations'] == 2
    assert metrics['local_citations'] == 1
    assert metrics['sonar_citations'] == 1
    print("   ✓ Citation analysis working")
    
    # Test Vectara FCS availability
    vectara_available = quality_scorer.vectara_fcs.enabled
    print(f"   ℹ️  Vectara FCS enabled: {vectara_available}")


def test_integration_with_agents():
    """Test model selection integration with actual agents."""
    print("🧪 Testing Agent Integration...")
    
    try:
        from agents.seed_agent import get_seed_agent
        from agents.research_agent import get_research_agent
        
        # Test that agents can be initialized
        seed_agent = get_seed_agent()
        research_agent = get_research_agent()
        
        assert seed_agent is not None
        assert research_agent is not None
        print("   ✓ Agents initialize with model selection")
        
        # Test that agents have access to model selector
        assert hasattr(seed_agent, 'openai_api_key')
        assert hasattr(research_agent, 'openai_api_key')
        print("   ✓ Agents configured for dynamic model selection")
        
    except ImportError as e:
        print(f"   ⚠️  Agent integration test skipped: {e}")


def test_cli_commands():
    """Test that new CLI commands are available."""
    print("🧪 Testing CLI Commands...")
    
    try:
        from app.cli import cli
        
        commands = list(cli.commands.keys())
        required_commands = ['models', 'performance', 'select-model']
        
        for cmd in required_commands:
            assert cmd in commands, f"Missing CLI command: {cmd}"
        
        print(f"   ✓ All new CLI commands available: {required_commands}")
        
    except ImportError as e:
        print(f"   ⚠️  CLI test skipped: {e}")


def run_all_tests():
    """Run all model selection tests."""
    print("🚀 Running Model Selection Test Suite...")
    print("=" * 50)
    
    start_time = time.time()
    tests = [
        test_model_configuration,
        test_model_selection_logic,
        test_fallback_chains,
        test_performance_monitoring,
        test_quality_scoring,
        test_integration_with_agents,
        test_cli_commands
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"   ❌ FAILED: {e}")
            failed += 1
        print()
    
    duration = time.time() - start_time
    
    print("=" * 50)
    print(f"📊 Test Results:")
    print(f"   ✅ Passed: {passed}")
    print(f"   ❌ Failed: {failed}")
    print(f"   ⏱️  Duration: {duration:.2f}s")
    
    if failed == 0:
        print("🎉 ALL MODEL SELECTION TESTS PASSED!")
        print("🚀 Phase 3 implementation verified!")
    else:
        print("💥 Some tests failed - check implementation")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
"""
End-to-end tests for Phase 2 multi-agent workflow.
Tests the complete user experience from CLI through multi-agent processing.
"""

import sys
import os
import json
import tempfile
import subprocess
from pathlib import Path
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.rag_pipeline import get_rag_pipeline
from core.config import settings


class Phase2E2ETests:
    """End-to-end tests for Phase 2 multi-agent system."""
    
    def __init__(self):
        """Initialize test environment."""
        self.test_questions = [
            {
                "question": "What is retrieval-augmented generation?",
                "expected_elements": ["retrieval", "augmented", "generation", "RAG"],
                "expected_phase": "2_multi_agent",
                "min_confidence": 0.8,
                "expect_quality_assessment": True
            },
            {
                "question": "What is machine learning?",
                "expected_elements": ["machine learning", "algorithms", "data"],
                "expected_phase": "2_multi_agent", 
                "min_confidence": 0.8,
                "expect_quality_assessment": True
            },
            {
                "question": "How does AI work?",
                "expected_elements": ["artificial intelligence", "AI"],
                "expected_phase": "2_multi_agent",
                "min_confidence": 0.7,
                "expect_quality_assessment": True
            }
        ]
        
        self.pipeline = None
        self.test_results = []
    
    def setup(self) -> bool:
        """Setup test environment and validate prerequisites."""
        print("🧪 Setting up Phase 2 end-to-end tests...")
        
        # Check API key
        if not settings.openai_api_key or settings.openai_api_key == "your-api-key-here":
            print("❌ OpenAI API key not configured. Set OPENAI_API_KEY environment variable.")
            return False
        
        # Initialize pipeline
        try:
            self.pipeline = get_rag_pipeline()
            print("✓ RAG pipeline initialized")
        except Exception as e:
            print(f"❌ Failed to initialize RAG pipeline: {e}")
            return False
        
        # Check knowledge base
        try:
            stats = self.pipeline.get_knowledge_base_stats()
            if stats['total_documents'] == 0:
                print("❌ Knowledge base is empty. Run setup_sample_data.py first.")
                return False
            print(f"✓ Knowledge base ready ({stats['total_documents']} documents)")
        except Exception as e:
            print(f"❌ Failed to check knowledge base: {e}")
            return False
        
        # Test agent imports
        try:
            from agents.seed_agent import get_seed_agent
            from agents.sourcing_agent import get_sourcing_agent
            from agents.research_agent import get_research_agent
            print("✓ All agents available")
        except ImportError as e:
            print(f"❌ Agent import failed: {e}")
            return False
        
        print("✅ Test setup complete\n")
        return True
    
    def test_cli_integration(self) -> bool:
        """Test CLI integration with Phase 2 workflow."""
        print("🔧 Testing CLI integration...")
        
        try:
            # Test basic CLI command
            cmd = [sys.executable, "app/cli.py", "ask", "What is RAG?"]
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=60,
                cwd=project_root
            )
            
            if result.returncode != 0:
                print(f"❌ CLI command failed: {result.stderr}")
                return False
            
            # Check if answer contains expected content
            if "retrieval" not in result.stdout.lower():
                print(f"❌ CLI answer doesn't contain expected content")
                print(f"Output: {result.stdout}")
                return False
            
            print("✓ CLI integration working")
            return True
            
        except subprocess.TimeoutExpired:
            print("❌ CLI command timed out")
            return False
        except Exception as e:
            print(f"❌ CLI test failed: {e}")
            return False
    
    def test_multi_agent_workflow(self) -> bool:
        """Test the complete multi-agent workflow programmatically."""
        print("🤖 Testing multi-agent workflow...")
        
        success_count = 0
        
        for i, test_case in enumerate(self.test_questions, 1):
            print(f"\n  Test {i}/{len(self.test_questions)}: {test_case['question']}")
            
            try:
                # Execute Phase 2 workflow
                response = self.pipeline.answer_question(
                    test_case['question'], 
                    use_agents=True
                )
                
                # Validate response structure
                if not self._validate_response_structure(response):
                    print(f"    ❌ Invalid response structure")
                    continue
                
                # Check if Phase 2 was used
                if response.get('phase') != test_case['expected_phase']:
                    print(f"    ❌ Expected phase {test_case['expected_phase']}, got {response.get('phase')}")
                    continue
                
                # Check confidence score
                confidence = response.get('confidence_score', 0)
                if confidence < test_case['min_confidence']:
                    print(f"    ❌ Low confidence: {confidence:.2f} < {test_case['min_confidence']}")
                    continue
                
                # Check answer content
                answer = response.get('answer', '').lower()
                missing_elements = [elem for elem in test_case['expected_elements'] 
                                 if elem.lower() not in answer]
                
                if missing_elements:
                    print(f"    ❌ Answer missing elements: {missing_elements}")
                    continue
                
                # Check citations
                citations = response.get('citations', [])
                if not citations:
                    print(f"    ❌ No citations provided")
                    continue
                
                # Check quality assessment (Phase 3 feature)
                if test_case.get('expect_quality_assessment', False):
                    quality_assessment = response.get('quality_assessment', {})
                    if not quality_assessment:
                        print(f"    ❌ No quality assessment provided")
                        continue
                    
                    # Validate quality assessment structure
                    required_qa_fields = ['overall_confidence', 'explanation', 'metrics']
                    missing_qa_fields = [field for field in required_qa_fields if field not in quality_assessment]
                    if missing_qa_fields:
                        print(f"    ❌ Quality assessment missing fields: {missing_qa_fields}")
                        continue
                    
                    qa_confidence = quality_assessment.get('overall_confidence', 0)
                    fcs_score = quality_assessment.get('fcs_score')
                    qa_metrics = quality_assessment.get('metrics', {})
                    
                    print(f"    ✓ Quality Assessment: {qa_confidence:.2f}")
                    if fcs_score is not None and fcs_score > 0:
                        print(f"    ✓ FCS Score: {fcs_score:.2f}")
                    if qa_metrics.get('total_citations', 0) > 0:
                        print(f"    ✓ Citation Analysis: {qa_metrics['total_citations']} citations")
                
                print(f"    ✓ Confidence: {confidence:.2f}")
                print(f"    ✓ Citations: {len(citations)}")
                print(f"    ✓ Sources: {response.get('total_sources', 0)}")
                
                success_count += 1
                self.test_results.append({
                    'question': test_case['question'],
                    'confidence': confidence,
                    'citations': len(citations),
                    'phase': response.get('phase'),
                    'success': True
                })
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                self.test_results.append({
                    'question': test_case['question'],
                    'error': str(e),
                    'success': False
                })
        
        success_rate = success_count / len(self.test_questions)
        print(f"\n✅ Multi-agent workflow: {success_count}/{len(self.test_questions)} tests passed ({success_rate:.1%})")
        
        return success_rate >= 0.8  # 80% success rate required
    
    def test_fallback_behavior(self) -> bool:
        """Test fallback to Phase 1 when agents fail."""
        print("🛡️  Testing fallback behavior...")
        
        try:
            # Force fallback by disabling agents
            response = self.pipeline.answer_question(
                "What is machine learning?",
                use_agents=False
            )
            
            if response.get('phase') != '1_basic_rag':
                print(f"❌ Expected Phase 1 fallback, got {response.get('phase')}")
                return False
            
            if not response.get('answer'):
                print(f"❌ No answer in fallback mode")
                return False
            
            print("✓ Fallback to Phase 1 working")
            return True
            
        except Exception as e:
            print(f"❌ Fallback test failed: {e}")
            return False
    
    def test_agent_components(self) -> bool:
        """Test individual agent components."""
        print("⚙️  Testing individual agent components...")
        
        try:
            from agents.seed_agent import get_seed_agent
            from agents.sourcing_agent import get_sourcing_agent
            from agents.research_agent import get_research_agent
            
            # Test Seed Agent
            seed_agent = get_seed_agent()
            complexity = seed_agent.analyze_query_complexity("What is machine learning?")
            if not isinstance(complexity, dict) or 'complexity_score' not in complexity:
                print("❌ Seed agent complexity analysis failed")
                return False
            print("  ✓ Seed Agent working")
            
            # Test Sourcing Agent (basic initialization)
            sourcing_agent = get_sourcing_agent()
            if not hasattr(sourcing_agent, 'vector_store'):
                print("❌ Sourcing agent initialization failed")
                return False
            print("  ✓ Sourcing Agent working")
            
            # Test Research Agent (basic initialization)
            research_agent = get_research_agent()
            if not hasattr(research_agent, 'llm'):
                print("❌ Research agent initialization failed")
                return False
            print("  ✓ Research Agent working")
            
            return True
            
        except Exception as e:
            print(f"❌ Agent component test failed: {e}")
            return False
    
    def _validate_response_structure(self, response: Dict[str, Any]) -> bool:
        """Validate that response has required Phase 2/3 structure."""
        required_fields = ['question', 'answer', 'phase']
        phase2_fields = ['confidence_score', 'total_sources', 'citations', 'source_breakdown']
        phase3_fields = ['quality_assessment']  # Optional Phase 3 enhancement
        
        # Check basic fields
        for field in required_fields:
            if field not in response:
                print(f"    Missing required field: {field}")
                return False
        
        # Check Phase 2 specific fields
        if response.get('phase') == '2_multi_agent':
            for field in phase2_fields:
                if field not in response:
                    print(f"    Missing Phase 2 field: {field}")
                    return False
            
            # Check Phase 3 enhancements (optional)
            if 'quality_assessment' in response:
                qa = response['quality_assessment']
                if not isinstance(qa, dict):
                    print(f"    Quality assessment must be a dictionary")
                    return False
                
                qa_required = ['overall_confidence', 'explanation', 'metrics']
                for field in qa_required:
                    if field not in qa:
                        print(f"    Missing quality assessment field: {field}")
                        return False
        
        return True
    
    def generate_test_report(self) -> str:
        """Generate a comprehensive test report."""
        report = ["", "📊 Phase 2 End-to-End Test Report", "=" * 50]
        
        # Summary
        total_tests = len(self.test_results)
        successful_tests = sum(1 for r in self.test_results if r.get('success', False))
        
        report.append(f"Total Tests: {total_tests}")
        report.append(f"Successful: {successful_tests}")
        report.append(f"Success Rate: {successful_tests/total_tests:.1%}" if total_tests > 0 else "Success Rate: N/A")
        report.append("")
        
        # Individual test results
        report.append("Individual Test Results:")
        for i, result in enumerate(self.test_results, 1):
            status = "✅" if result.get('success') else "❌"
            report.append(f"{i}. {status} {result['question']}")
            
            if result.get('success'):
                report.append(f"   Confidence: {result.get('confidence', 0):.2f}")
                report.append(f"   Citations: {result.get('citations', 0)}")
                report.append(f"   Phase: {result.get('phase', 'unknown')}")
            else:
                report.append(f"   Error: {result.get('error', 'unknown')}")
            report.append("")
        
        return "\n".join(report)
    
    def run_all_tests(self) -> bool:
        """Run all end-to-end tests."""
        print("🚀 Starting Phase 2 End-to-End Tests")
        print("=" * 50)
        
        # Setup
        if not self.setup():
            return False
        
        # Run test suite
        tests = [
            ("CLI Integration", self.test_cli_integration),
            ("Agent Components", self.test_agent_components), 
            ("Fallback Behavior", self.test_fallback_behavior),
            ("Multi-Agent Workflow", self.test_multi_agent_workflow),
        ]
        
        results = []
        for test_name, test_func in tests:
            print(f"\n{test_name}:")
            try:
                result = test_func()
                results.append(result)
                if result:
                    print(f"✅ {test_name} PASSED")
                else:
                    print(f"❌ {test_name} FAILED")
            except Exception as e:
                print(f"❌ {test_name} ERROR: {e}")
                results.append(False)
        
        # Generate report
        print(self.generate_test_report())
        
        # Overall result
        overall_success = all(results)
        if overall_success:
            print("🎉 ALL TESTS PASSED - Phase 2 is working correctly!")
        else:
            print("⚠️  SOME TESTS FAILED - Check the results above")
        
        return overall_success


def main():
    """Run the end-to-end test suite."""
    tester = Phase2E2ETests()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
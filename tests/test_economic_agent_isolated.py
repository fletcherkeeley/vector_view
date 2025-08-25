#!/usr/bin/env python3
"""
Isolated Economic Agent Test Script

Tests the economic agent in isolation with verbose output to analyze
and improve its thinking process and output quality.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any
import json
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from agents.economic import EconomicAnalysisAgent
from agents.base_agent import AgentContext, AgentType

# Configure verbose logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('economic_agent_test.log')
    ]
)

logger = logging.getLogger(__name__)


class VerboseEconomicAgent(EconomicAnalysisAgent):
    """Extended economic agent with verbose debugging output"""
    
    def get_required_data_sources(self, context: AgentContext) -> list[str]:
        """Return required data sources for economic analysis"""
        return ["fred"]
    
    async def analyze(self, context: AgentContext):
        """Async wrapper for process method to maintain compatibility"""
        return self.process(context)
    
    def process(self, context: AgentContext):
        """Override process method to add verbose logging"""
        print("\n" + "="*80)
        print("🏛️  ECONOMIC AGENT ANALYSIS - VERBOSE MODE")
        print("="*80)
        
        print(f"\n📋 CONTEXT RECEIVED:")
        print(f"   Query: {context.query}")
        print(f"   Query Type: {context.query_type}")
        print(f"   Timeframe: {context.timeframe}")
        print(f"   Date Range: {context.date_range['start']} to {context.date_range['end']}")
        print(f"   Data Sources: {context.data_sources}")
        
        # Call parent process method
        start_time = datetime.now()
        result = super().process(context)
        execution_time = (datetime.now() - start_time).total_seconds()
        
        print(f"\n📊 ANALYSIS RESULTS:")
        print(f"   Confidence: {result.confidence:.3f} ({'high' if result.confidence > 0.7 else 'medium' if result.confidence > 0.4 else 'low'})")
        print(f"   Execution Time: {execution_time:.2f}s")
        print(f"   Data Sources Used: {result.data_sources_used}")
        
        print(f"\n🔍 KEY METRICS:")
        for key, value in result.key_metrics.items():
            print(f"   {key}: {value}")
        
        print(f"\n💡 INSIGHTS ({len(result.insights)} total):")
        for i, insight in enumerate(result.insights, 1):
            print(f"   {i}. {insight}")
        
        print(f"\n🔄 CROSS-AGENT SIGNALS:")
        if hasattr(result, 'cross_agent_signals') and result.cross_agent_signals:
            for signal, value in result.cross_agent_signals.items():
                print(f"   {signal}: {value}")
        
        print(f"\n⚠️  UNCERTAINTY FACTORS:")
        if hasattr(result, 'uncertainty_factors') and result.uncertainty_factors:
            for factor in result.uncertainty_factors:
                print(f"   - {factor}")
        
        print(f"\n🧠 DETAILED ANALYSIS:")
        if hasattr(result, 'detailed_analysis') and result.detailed_analysis:
            if 'ai_insights' in result.detailed_analysis:
                ai_analysis = result.detailed_analysis['ai_insights'].get('analysis', '')
                print(f"   AI Analysis: {ai_analysis[:200]}...")
            
            if 'trends' in result.detailed_analysis:
                trends = result.detailed_analysis['trends']
                print(f"   Trends Analyzed: {len(trends)} indicators")
                for indicator, trend_data in list(trends.items())[:3]:  # Show first 3
                    direction = trend_data.get('direction', 'unknown')
                    strength = trend_data.get('strength', 'unknown')
                    print(f"     {indicator}: {direction} ({strength})")
            
            if 'economic_cycle' in result.detailed_analysis:
                cycle = result.detailed_analysis['economic_cycle']
                phase = cycle.get('phase', 'unknown')
                confidence = cycle.get('confidence', 0)
                print(f"   Economic Cycle: {phase} (confidence: {confidence:.1%})")
        
        print("\n" + "="*80)
        print("✅ ECONOMIC AGENT ANALYSIS COMPLETE")
        print("="*80 + "\n")
        
        return result


def test_economic_agent():
    """Test the economic agent with various scenarios"""
    
    # Database URL from environment or default
    database_url = os.getenv('DATABASE_URL', 'postgresql://postgres:fred_password@localhost:5432/postgres')
    
    print("🚀 Starting Economic Agent Isolated Testing")
    print(f"Database URL: {database_url}")
    
    # Initialize verbose economic agent
    try:
        agent = VerboseEconomicAgent()
        print("✅ Economic agent initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize economic agent: {e}")
        return []
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Employment Analysis",
            "query": "What's the current state of employment and unemployment trends?",
            "query_type": "deep_dive",
            "timeframe": "3m"
        },
        {
            "name": "Inflation Analysis", 
            "query": "How is inflation trending and what are the key drivers?",
            "query_type": "correlation_analysis",
            "timeframe": "6m"
        },
        {
            "name": "Federal Reserve Policy",
            "query": "What does the current interest rate environment suggest about Fed policy?",
            "query_type": "deep_dive", 
            "timeframe": "1y"
        },
        {
            "name": "Economic Growth",
            "query": "Is the economy growing and what are the leading indicators showing?",
            "query_type": "daily_briefing",
            "timeframe": "3m"
        },
        {
            "name": "General Economic Health",
            "query": "Give me an overview of overall economic conditions",
            "query_type": "daily_briefing",
            "timeframe": "1m"
        }
    ]
    
    results = []
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🧪 TEST SCENARIO {i}/{len(test_scenarios)}: {scenario['name']}")
        print("-" * 60)
        
        # Create context
        context = AgentContext(
            query=scenario['query'],
            query_type=scenario['query_type'],
            timeframe=scenario['timeframe'],
            user_id="test_user",
            session_id=f"test_session_{i}",
            data_sources=["fred"]
        )
        
        try:
            # Run analysis
            result = agent.process(context)
            results.append({
                "scenario": scenario['name'],
                "success": True,
                "confidence": result.confidence,
                "insights_count": len(result.insights),
                "execution_time": result.execution_time_ms,
                "signals": list(result.cross_agent_signals.keys()) if hasattr(result, 'cross_agent_signals') and result.cross_agent_signals else []
            })
            
        except Exception as e:
            print(f"❌ Test scenario failed: {e}")
            results.append({
                "scenario": scenario['name'],
                "success": False,
                "error": str(e)
            })
    
    # Summary report
    print("\n" + "="*80)
    print("📈 ECONOMIC AGENT TEST SUMMARY")
    print("="*80)
    
    successful_tests = [r for r in results if r.get('success', False)]
    failed_tests = [r for r in results if not r.get('success', False)]
    
    print(f"✅ Successful Tests: {len(successful_tests)}/{len(results)}")
    print(f"❌ Failed Tests: {len(failed_tests)}")
    
    if successful_tests:
        avg_confidence = sum(r['confidence'] for r in successful_tests) / len(successful_tests)
        avg_execution_time = sum(r['execution_time'] for r in successful_tests) / len(successful_tests)
        
        print(f"\n📊 Performance Metrics:")
        print(f"   Average Confidence: {avg_confidence:.3f}")
        print(f"   Average Execution Time: {avg_execution_time:.1f}ms")
        
        print(f"\n🔄 Cross-Agent Signals Generated:")
        all_signals = set()
        for result in successful_tests:
            all_signals.update(result['signals'])
        for signal in sorted(all_signals):
            print(f"   - {signal}")
    
    if failed_tests:
        print(f"\n❌ Failed Test Details:")
        for test in failed_tests:
            print(f"   {test['scenario']}: {test['error']}")
    
    # Performance stats
    perf_stats = agent.get_performance_stats()
    print(f"\n⚡ Agent Performance Stats:")
    for key, value in perf_stats.items():
        print(f"   {key}: {value}")
    
    print("\n🎯 RECOMMENDATIONS FOR IMPROVEMENT:")
    if avg_confidence < 0.7:
        print("   - Consider improving AI prompts for higher confidence")
    if avg_execution_time > 5000:
        print("   - Optimize database queries for faster execution")
    if len(failed_tests) > 0:
        print("   - Fix error handling and data validation issues")
    
    print("\n✅ Economic Agent Testing Complete!")
    return results


def save_test_results(results: list):
    """Save test results to JSON file"""
    output_file = "economic_agent_test_results.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": results
        }, f, indent=2)
    print(f"📁 Test results saved to {output_file}")


if __name__ == "__main__":
    print("🏛️  Vector View Economic Agent - Isolated Testing")
    print("=" * 60)
    
    try:
        # Run the test
        results = test_economic_agent()
        save_test_results(results)
        
    except KeyboardInterrupt:
        print("\n⏹️  Testing interrupted by user")
    except Exception as e:
        print(f"\n💥 Testing failed with error: {e}")
        import traceback
        traceback.print_exc()

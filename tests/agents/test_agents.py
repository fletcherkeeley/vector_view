"""
Test script for Vector View AI Agents

Tests the agent architecture with real database connections
and validates functionality with existing FRED data.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.orchestration_agent import OrchestrationAgent
from agents.economic_agent import EconomicAnalysisAgent
from agents.base_agent import AgentContext

# Load environment variables
load_dotenv()

async def test_economic_agent():
    """Test the Economic Analysis Agent with real data"""
    print("🧪 Testing Economic Analysis Agent...")
    
    # Get database URL from environment
    database_url = os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/vector_view')
    
    # Initialize agent
    economic_agent = EconomicAnalysisAgent(database_url=database_url)
    
    # Create test context with realistic date range (using available FRED data)
    context = AgentContext(
        query="What's the current state of the US economy?",
        query_type="economic_analysis",
        timeframe="3m",
        data_sources=["fred"],
        date_range={
            "start": datetime(2025, 4, 1),  # Use dates where we have data
            "end": datetime(2025, 7, 31)
        }
    )
    
    try:
        # Run analysis
        print("  📊 Running economic analysis...")
        response = await economic_agent.process_query(context)
        
        # Display results
        print(f"  ✅ Analysis completed!")
        print(f"  📈 Confidence: {response.confidence:.2%} ({response.confidence_level.value})")
        print(f"  ⏱️  Execution time: {response.execution_time_ms:.1f}ms")
        print(f"  📋 Indicators analyzed: {response.key_metrics.get('indicators_analyzed', 0)}")
        print(f"  📊 Data points: {response.key_metrics.get('data_points_analyzed', 0)}")
        
        print("\n  🔍 Key Insights:")
        for i, insight in enumerate(response.insights[:5], 1):
            print(f"    {i}. {insight}")
        
        print(f"\n  🔗 Cross-agent signals: {len(response.signals_for_other_agents)} signals")
        for signal, value in response.signals_for_other_agents.items():
            print(f"    • {signal}: {value}")
        
        if response.uncertainty_factors:
            print(f"\n  ⚠️  Uncertainty factors: {', '.join(response.uncertainty_factors)}")
        
        return response
        
    except Exception as e:
        print(f"  ❌ Economic agent test failed: {str(e)}")
        return None

async def test_orchestration_agent():
    """Test the Orchestration Agent with economic agent"""
    print("\n🎭 Testing Orchestration Agent...")
    
    # Get database URL from environment
    database_url = os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/vector_view')
    
    # Initialize agents
    orchestrator = OrchestrationAgent(database_url=database_url)
    economic_agent = EconomicAnalysisAgent(database_url=database_url)
    
    # Register economic agent with orchestrator
    orchestrator.register_agent(economic_agent)
    
    try:
        # Test query classification
        test_queries = [
            "What's today's market summary?",
            "Analyze the relationship between inflation and unemployment",
            "How is the economy performing?",
            "Deep dive into current economic indicators"
        ]
        
        print("  🔍 Testing query classification:")
        for query in test_queries:
            query_type = orchestrator.classify_query(query)
            print(f"    • '{query}' → {query_type}")
        
        # Test full workflow execution
        print("\n  🚀 Testing workflow execution...")
        response = await orchestrator.process_user_query(
            query="How is the US economy performing based on recent indicators?",
            timeframe="1m"
        )
        
        print(f"  ✅ Workflow completed!")
        print(f"  📈 Overall confidence: {response.confidence:.2%}")
        print(f"  ⏱️  Total execution time: {response.total_execution_time_ms:.1f}ms")
        print(f"  🤖 Agents executed: {', '.join(response.agents_executed)}")
        
        print(f"\n  📋 Executive Summary:")
        print(f"    {response.executive_summary}")
        
        print(f"\n  🔍 Key Insights:")
        for i, insight in enumerate(response.key_insights[:3], 1):
            print(f"    {i}. {insight}")
        
        return response
        
    except Exception as e:
        print(f"  ❌ Orchestration agent test failed: {str(e)}")
        return None

async def test_agent_performance():
    """Test agent performance and caching"""
    print("\n⚡ Testing Agent Performance...")
    
    database_url = os.getenv('DATABASE_URL', 'postgresql://postgres:password@localhost:5432/vector_view')
    economic_agent = EconomicAnalysisAgent(database_url=database_url)
    
    context = AgentContext(
        query="Economic trends analysis",
        query_type="economic_analysis", 
        timeframe="1m",
        data_sources=["fred"],
        date_range={
            "start": datetime(2025, 6, 1),  # Use dates where we have data
            "end": datetime(2025, 7, 31)
        }
    )
    
    try:
        # First run (no cache)
        print("  🔄 First run (no cache)...")
        start_time = datetime.now()
        response1 = await economic_agent.process_query(context)
        first_run_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Second run (should use cache)
        print("  🔄 Second run (with cache)...")
        start_time = datetime.now()
        response2 = await economic_agent.process_query(context)
        second_run_time = (datetime.now() - start_time).total_seconds() * 1000
        
        print(f"  📊 Performance Results:")
        print(f"    • First run: {first_run_time:.1f}ms")
        print(f"    • Second run: {second_run_time:.1f}ms")
        print(f"    • Cache speedup: {(first_run_time/second_run_time):.1f}x")
        
        # Performance stats
        stats = economic_agent.get_performance_stats()
        print(f"  📈 Agent Stats:")
        print(f"    • Total queries: {stats['total_queries']}")
        print(f"    • Avg execution time: {stats['avg_execution_time_ms']:.1f}ms")
        print(f"    • Cache hit rate: {stats['cache_hit_rate']:.1%}")
        
    except Exception as e:
        print(f"  ❌ Performance test failed: {str(e)}")

async def main():
    """Run all agent tests"""
    print("🚀 Vector View Agent Testing Suite")
    print("=" * 50)
    
    # Test individual economic agent
    economic_response = await test_economic_agent()
    
    # Test orchestration agent
    orchestration_response = await test_orchestration_agent()
    
    # Test performance
    await test_agent_performance()
    
    print("\n" + "=" * 50)
    if economic_response and orchestration_response:
        print("✅ All agent tests completed successfully!")
        print(f"📊 Economic agent analyzed {economic_response.key_metrics.get('indicators_analyzed', 0)} indicators")
        print(f"🎭 Orchestration agent coordinated {len(orchestration_response.agents_executed)} agents")
    else:
        print("❌ Some tests failed - check error messages above")

if __name__ == "__main__":
    asyncio.run(main())

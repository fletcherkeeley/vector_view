#!/usr/bin/env python3
"""
Test individual agents with fixes applied
"""

import asyncio
import logging
from datetime import datetime, timedelta

from agents.base_agent import AgentContext
from agents.economic import EconomicAnalysisAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_economic_agent():
    """Test economic agent with real database data"""
    
    print("🧪 Testing Economic Agent with Real Data...")
    
    # Database configuration
    database_url = "postgresql://postgres:fred_password@localhost:5432/postgres"
    
    # Initialize agent
    agent = EconomicAnalysisAgent(database_url=database_url)
    
    # Create test context
    context = AgentContext(
        query="Analyze current unemployment and inflation trends for market outlook",
        query_type="economic_analysis",
        timeframe="3m"  # 3 months
    )
    
    try:
        print(f"📊 Running analysis: {context.query}")
        print(f"⏱️  Timeframe: {context.timeframe}")
        
        start_time = datetime.now()
        response = await agent.analyze(context)
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        print(f"\n✅ Analysis completed in {execution_time:.1f}ms")
        print(f"📈 Confidence: {response.confidence:.3f} ({response.confidence_level.value})")
        print(f"🔢 Key metrics: {len(response.key_metrics)}")
        print(f"💡 Insights: {len(response.insights)}")
        print(f"🔄 Cross-agent signals: {len(response.signals_for_other_agents)}")
        
        # Show key metrics
        if response.key_metrics:
            print(f"\n📊 KEY METRICS:")
            for metric, value in response.key_metrics.items():
                print(f"   {metric}: {value}")
        
        # Show insights
        if response.insights:
            print(f"\n💡 INSIGHTS:")
            for i, insight in enumerate(response.insights[:3], 1):
                print(f"   {i}. {insight[:150]}{'...' if len(insight) > 150 else ''}")
        
        # Show cross-agent signals
        if response.signals_for_other_agents:
            print(f"\n🔄 CROSS-AGENT SIGNALS:")
            for signal, value in response.signals_for_other_agents.items():
                print(f"   {signal}: {value}")
        
        # Show data sources
        print(f"\n📚 DATA SOURCES: {', '.join(response.data_sources_used)}")
        
        # Show analysis data summary
        if response.analysis:
            print(f"\n📈 ANALYSIS DATA:")
            for key, value in response.analysis.items():
                if isinstance(value, dict):
                    print(f"   {key}: {len(value)} items")
                elif isinstance(value, str):
                    print(f"   {key}: {len(value)} chars")
                else:
                    print(f"   {key}: {type(value).__name__}")
        
        return response
        
    except Exception as e:
        print(f"❌ Economic agent test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    """Run agent tests"""
    print("🚀 Starting Individual Agent Tests with Fixes\n")
    
    # Test economic agent
    economic_result = await test_economic_agent()
    
    if economic_result and economic_result.confidence > 0:
        print(f"\n🎉 SUCCESS: Economic agent producing real analysis!")
        print(f"   Confidence: {economic_result.confidence:.1%}")
        print(f"   Execution time: {economic_result.execution_time_ms:.1f}ms")
    else:
        print(f"\n❌ FAILED: Economic agent still not working properly")

if __name__ == "__main__":
    asyncio.run(main())

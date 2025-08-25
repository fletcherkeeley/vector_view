#!/usr/bin/env python3
"""
Individual test for Market Intelligence Agent
"""

import asyncio
import logging
from datetime import datetime
from agents.market_intelligence_agent import MarketIntelligenceAgent
from agents.base_agent import AgentContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_market_intelligence_agent():
    """Test the market intelligence agent individually"""
    print("🧪 Testing MARKET INTELLIGENCE Agent Individually...")
    
    # Create agent
    agent = MarketIntelligenceAgent()
    
    # Create test context
    context = AgentContext(
        query="market intelligence analysis correlation news sentiment",
        query_type="market_analysis",
        timeframe="24h",
        user_id="test_user"
    )
    
    try:
        # Run analysis
        start_time = datetime.now()
        result = await agent.analyze(context)
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        print(f"✅ Market Intelligence Agent: {result.confidence:.3f} confidence, {execution_time:.1f}ms")
        print(f"   📊 Key Metrics:")
        for key, value in result.key_metrics.items():
            print(f"      {key}: {value}")
        
        print(f"   💡 Insights: {len(result.insights)}")
        for i, insight in enumerate(result.insights[:3], 1):  # Show first 3
            print(f"      {i}. {insight[:100]}...")
        
        print(f"   🔄 Signals: {len(result.signals_for_other_agents)}")
        for key, value in result.signals_for_other_agents.items():
            print(f"      {key}: {value}")
            
        return True
        
    except Exception as e:
        print(f"❌ Market Intelligence Agent failed: {str(e)}")
        logger.error(f"Market intelligence agent error: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    success = asyncio.run(test_market_intelligence_agent())
    if success:
        print("🎉 Market Intelligence Agent test PASSED")
    else:
        print("💥 Market Intelligence Agent test FAILED")

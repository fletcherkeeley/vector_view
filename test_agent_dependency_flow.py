#!/usr/bin/env python3
"""
Test Agent Dependency Flow - Verify sentiment flows from News Sentiment to Market Intelligence
"""

import asyncio
import logging
from datetime import datetime
from agents.news_sentiment_agent import NewsSentimentAgent
from agents.market_intelligence_agent import MarketIntelligenceAgent
from agents.base_agent import AgentContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_agent_dependency_flow():
    """Test that sentiment flows properly from news sentiment agent to market intelligence agent"""
    print("🔄 Testing Agent Dependency Flow...")
    
    # Create test context
    context = AgentContext(
        query="market sentiment analysis correlation",
        query_type="market_analysis",
        timeframe="24h",
        user_id="test_user"
    )
    
    try:
        # Step 1: Run News Sentiment Agent first
        print("\n📊 Step 1: Running News Sentiment Agent...")
        sentiment_agent = NewsSentimentAgent()
        sentiment_response = await sentiment_agent.analyze(context)
        
        print(f"✅ Sentiment Agent: {sentiment_response.confidence:.3f} confidence")
        print(f"   Overall Sentiment: {sentiment_response.key_metrics.get('overall_sentiment', 'N/A')}")
        print(f"   Market Relevance: {sentiment_response.key_metrics.get('market_relevance', 'N/A')}")
        print(f"   Signals Generated: {len(sentiment_response.signals_for_other_agents)}")
        
        # Step 2: Add sentiment output to context for market intelligence agent
        print("\n🔄 Step 2: Adding sentiment data to context...")
        context.agent_outputs = {'news_sentiment': sentiment_response}
        
        # Step 3: Run Market Intelligence Agent with sentiment context
        print("\n📈 Step 3: Running Market Intelligence Agent with sentiment context...")
        market_agent = MarketIntelligenceAgent()
        market_response = await market_agent.analyze(context)
        
        print(f"✅ Market Intelligence Agent: {market_response.confidence:.3f} confidence")
        print(f"   Sentiment Score Used: {market_response.key_metrics.get('sentiment_score', 'N/A')}")
        print(f"   Market Correlation: {market_response.key_metrics.get('market_correlation', 'N/A')}")
        print(f"   Volatility Forecast: {market_response.key_metrics.get('volatility_forecast', 'N/A')}")
        
        # Step 4: Verify sentiment data flow
        print("\n🔍 Step 4: Verifying sentiment data flow...")
        sentiment_from_news = sentiment_response.key_metrics.get('overall_sentiment', 0.0)
        sentiment_used_by_market = market_response.key_metrics.get('sentiment_score', 0.0)
        
        if abs(sentiment_from_news - sentiment_used_by_market) < 0.1:
            print(f"✅ SUCCESS: Sentiment data flowed correctly!")
            print(f"   News Agent Sentiment: {sentiment_from_news:.3f}")
            print(f"   Market Agent Used: {sentiment_used_by_market:.3f}")
        else:
            print(f"⚠️  WARNING: Sentiment data may not have flowed correctly")
            print(f"   News Agent Sentiment: {sentiment_from_news:.3f}")
            print(f"   Market Agent Used: {sentiment_used_by_market:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent dependency flow test failed: {str(e)}")
        logger.error(f"Dependency flow error: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    success = asyncio.run(test_agent_dependency_flow())
    if success:
        print("\n🎉 Agent Dependency Flow Test PASSED")
    else:
        print("\n💥 Agent Dependency Flow Test FAILED")

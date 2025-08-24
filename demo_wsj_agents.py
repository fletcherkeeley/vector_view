"""
Demo script to showcase WSJ-level agent capabilities
"""

import asyncio
import logging
from datetime import datetime

from agents.base_agent import AgentContext
from agents.market_intelligence_agent import MarketIntelligenceAgent
from agents.news_sentiment_agent import NewsSentimentAgent
from agents.editorial_synthesis_agent import EditorialSynthesisAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demo_wsj_agents():
    """Demonstrate WSJ-level agent capabilities"""
    
    print("🚀 WSJ-Level Financial Intelligence Agent Demo")
    print("=" * 60)
    
    # Initialize agents
    database_url = "postgresql://postgres:fred_password@localhost:5432/postgres"
    
    market_agent = MarketIntelligenceAgent(database_url=database_url)
    sentiment_agent = NewsSentimentAgent(database_url=database_url)
    editorial_agent = EditorialSynthesisAgent(database_url=database_url)
    
    # Test context
    context = AgentContext(
        query="Federal Reserve interest rate decision impact on technology stocks",
        query_type="market_analysis",
        timeframe="1d"
    )
    
    print(f"\n📊 Analysis Query: {context.query}")
    print(f"⏰ Timeframe: {context.timeframe}")
    
    # Test Market Intelligence Agent
    print(f"\n🔍 Market Intelligence Analysis...")
    start_time = datetime.now()
    market_response = await market_agent.analyze(context)
    market_time = (datetime.now() - start_time).total_seconds()
    
    print(f"✅ Market Intelligence: {market_response.confidence:.2f} confidence ({market_time:.1f}s)")
    if market_response.insights:
        print(f"📈 Key Insight: {market_response.insights[0][:200]}...")
    
    # Test News Sentiment Agent
    print(f"\n📰 News Sentiment Analysis...")
    start_time = datetime.now()
    sentiment_response = await sentiment_agent.analyze(context)
    sentiment_time = (datetime.now() - start_time).total_seconds()
    
    print(f"✅ News Sentiment: {sentiment_response.confidence:.2f} confidence ({sentiment_time:.1f}s)")
    if sentiment_response.insights:
        print(f"💭 Key Insight: {sentiment_response.insights[0][:200]}...")
    
    # Test Editorial Synthesis Agent
    print(f"\n✍️ Editorial Synthesis...")
    start_time = datetime.now()
    editorial_response = await editorial_agent.analyze(context)
    editorial_time = (datetime.now() - start_time).total_seconds()
    
    print(f"✅ Editorial Synthesis: {editorial_response.confidence:.2f} confidence ({editorial_time:.1f}s)")
    
    # Display WSJ-style article
    if editorial_response.insights and editorial_response.insights[0]:
        print(f"\n📄 WSJ-Style Article Generated:")
        print("=" * 60)
        article = editorial_response.insights[0]
        print(article)
        print("=" * 60)
    
    # Performance Summary
    total_time = market_time + sentiment_time + editorial_time
    avg_confidence = (market_response.confidence + sentiment_response.confidence + editorial_response.confidence) / 3
    
    print(f"\n📊 Performance Summary:")
    print(f"   Total Execution Time: {total_time:.1f} seconds")
    print(f"   Average Confidence: {avg_confidence:.2f}")
    print(f"   Market Analysis: {market_time:.1f}s")
    print(f"   Sentiment Analysis: {sentiment_time:.1f}s")
    print(f"   Editorial Synthesis: {editorial_time:.1f}s")
    
    # Agent Signals
    print(f"\n🔄 Cross-Agent Signals:")
    if market_response.signals_for_other_agents:
        print(f"   Market Signals: {market_response.signals_for_other_agents}")
    if sentiment_response.signals_for_other_agents:
        print(f"   Sentiment Signals: {sentiment_response.signals_for_other_agents}")
    if editorial_response.signals_for_other_agents:
        print(f"   Editorial Signals: {editorial_response.signals_for_other_agents}")
    
    print(f"\n🏆 WSJ-Level Analysis Complete!")
    
    return {
        'market_response': market_response,
        'sentiment_response': sentiment_response,
        'editorial_response': editorial_response,
        'performance': {
            'total_time': total_time,
            'avg_confidence': avg_confidence
        }
    }

if __name__ == "__main__":
    asyncio.run(demo_wsj_agents())

#!/usr/bin/env python3
"""
Individual test for News Sentiment Agent
"""

import asyncio
import logging
from datetime import datetime
from agents.news_sentiment_agent import NewsSentimentAgent
from agents.base_agent import AgentContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_news_sentiment_agent():
    """Test the news sentiment agent individually"""
    print("🧪 Testing NEWS SENTIMENT Agent Individually...")
    
    # Create agent
    agent = NewsSentimentAgent()
    
    # Create test context
    context = AgentContext(
        query="market sentiment analysis financial news",
        query_type="sentiment_analysis",
        timeframe="24h",
        user_id="test_user"
    )
    
    try:
        # Run analysis
        start_time = datetime.now()
        result = await agent.analyze(context)
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        print(f"✅ News Sentiment Agent: {result.confidence:.3f} confidence, {execution_time:.1f}ms")
        print(f"   📊 Key Metrics:")
        for key, value in result.key_metrics.items():
            print(f"      {key}: {value}")
        
        print(f"\n   📈 Detailed Analysis:")
        if 'sentiment_analysis' in result.analysis:
            sentiment = result.analysis['sentiment_analysis']
            print(f"      Overall Sentiment: {sentiment.get('overall_sentiment', 'N/A')}")
            print(f"      Emotional Tone: {sentiment.get('emotional_tone', {})}")
            print(f"      Bias Score: {sentiment.get('bias_score', 'N/A')}")
            print(f"      Credibility Score: {sentiment.get('credibility_score', 'N/A')}")
            print(f"      Urgency Level: {sentiment.get('urgency_level', 'N/A')}")
            print(f"      Market Relevance: {sentiment.get('market_relevance', 'N/A')}")
        
        if 'narrative_analysis' in result.analysis:
            narrative = result.analysis['narrative_analysis']
            print(f"      Dominant Themes: {narrative.get('dominant_themes', [])}")
            print(f"      Narrative Direction: {narrative.get('narrative_shift', 'N/A')}")
            print(f"      Consensus Level: {narrative.get('consensus_level', 'N/A')}")
        
        if 'entities' in result.analysis:
            entities = result.analysis['entities']
            print(f"      Companies: {entities.get('companies', [])[:10]}")  # Show first 10
            print(f"      Events: {entities.get('events', [])}")
        
        print(f"      Articles Analyzed: {result.analysis.get('articles_analyzed', 0)}")
        
        print(f"\n   💡 FULL AI INSIGHTS:")
        for i, insight in enumerate(result.insights, 1):
            print(f"      === INSIGHT {i} ===")
            print(f"      {insight}")
            print()
        
        print(f"   🔄 Cross-Agent Signals:")
        for key, value in result.signals_for_other_agents.items():
            print(f"      {key}: {value}")
        
        print(f"\n   📋 Data Sources: {result.data_sources_used}")
        print(f"   ⏱️  Timeframe: {result.timeframe_analyzed}")
        print(f"   🕒 Timestamp: {result.timestamp}")
            
        return True
        
    except Exception as e:
        print(f"❌ News Sentiment Agent failed: {str(e)}")
        logger.error(f"News sentiment agent error: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    success = asyncio.run(test_news_sentiment_agent())
    if success:
        print("🎉 News Sentiment Agent test PASSED")
    else:
        print("💥 News Sentiment Agent test FAILED")

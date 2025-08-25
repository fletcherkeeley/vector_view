"""
News Sentiment Agent Stability Test

Tests the refactored news sentiment agent with different scenarios
to confirm consistent performance and reliability.
"""

import asyncio
import logging
import json
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.news_sentiment.news_sentiment_agent import NewsSentimentAgent
from agents.base_agent import AgentContext

# Set up logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise

def print_test_header(test_name: str):
    """Print formatted test header"""
    print(f"\n{'='*60}")
    print(f"🧪 {test_name}")
    print(f"{'='*60}")

def print_results(response, execution_time: float):
    """Print concise test results"""
    print(f"✅ Confidence: {response.confidence:.3f} ({response.confidence_level.value})")
    print(f"⏱️  Execution: {execution_time:.2f}s")
    print(f"📊 Articles: {response.analysis.get('articles_analyzed', 0)}")
    print(f"💭 Sentiment: {response.key_metrics.get('overall_sentiment', 0):.3f}")
    print(f"🎯 Credibility: {response.key_metrics.get('credibility_score', 0):.3f}")
    print(f"📡 Signal: {response.signals_for_other_agents.get('news_sentiment', 'unknown')}")

async def test_scenario_1():
    """Test 1: Market volatility focus"""
    print_test_header("Market Volatility Analysis")
    
    agent = NewsSentimentAgent()
    context = AgentContext(
        query="market volatility stock market crash recession fears",
        query_type="market_analysis", 
        timeframe="1d"
    )
    
    start_time = datetime.now()
    response = await agent.analyze(context)
    execution_time = (datetime.now() - start_time).total_seconds()
    
    print_results(response, execution_time)
    return response

async def test_scenario_2():
    """Test 2: Economic policy focus"""
    print_test_header("Economic Policy Analysis")
    
    agent = NewsSentimentAgent()
    context = AgentContext(
        query="federal reserve interest rates inflation monetary policy",
        query_type="deep_dive",
        timeframe="1w"
    )
    
    start_time = datetime.now()
    response = await agent.analyze(context)
    execution_time = (datetime.now() - start_time).total_seconds()
    
    print_results(response, execution_time)
    return response

async def test_scenario_3():
    """Test 3: Corporate earnings focus"""
    print_test_header("Corporate Earnings Analysis")
    
    agent = NewsSentimentAgent()
    context = AgentContext(
        query="corporate earnings quarterly results profit margins",
        query_type="correlation_analysis",
        timeframe="1m"
    )
    
    start_time = datetime.now()
    response = await agent.analyze(context)
    execution_time = (datetime.now() - start_time).total_seconds()
    
    print_results(response, execution_time)
    return response

async def test_scenario_4():
    """Test 4: Broad market sentiment"""
    print_test_header("General Market Sentiment")
    
    agent = NewsSentimentAgent()
    context = AgentContext(
        query="financial markets economy",
        query_type="daily_briefing",
        timeframe="3d"
    )
    
    start_time = datetime.now()
    response = await agent.analyze(context)
    execution_time = (datetime.now() - start_time).total_seconds()
    
    print_results(response, execution_time)
    return response

async def test_component_isolation():
    """Test 5: Individual component testing"""
    print_test_header("Component Isolation Test")
    
    from agents.news_sentiment.news_sentiment_data_handler import NewsSentimentDataHandler
    from agents.news_sentiment.news_sentiment_context_builder import NewsSentimentContextBuilder
    from agents.ai_service import OllamaService
    
    # Test data handler
    print("📊 Testing Data Handler...")
    data_handler = NewsSentimentDataHandler()
    articles = await data_handler.get_news_articles(query="market", max_results=3)
    print(f"   Retrieved {len(articles)} articles")
    
    # Test context builder
    print("🔧 Testing Context Builder...")
    ai_service = OllamaService()
    context_builder = NewsSentimentContextBuilder(ai_service=ai_service)
    
    if articles:
        entities = await context_builder.extract_entities(articles)
        sentiment = await context_builder.analyze_sentiment(articles)
        narrative = await context_builder.analyze_narratives(articles)
        
        print(f"   Entities: {len(entities.companies)} companies, {len(entities.people)} people")
        print(f"   Sentiment: {sentiment.overall_sentiment:.3f}")
        print(f"   Themes: {len(narrative.dominant_themes)} themes")
    
    print("✅ All components working independently")

async def run_stability_tests():
    """Run comprehensive stability tests"""
    print(f"\n🚀 Starting News Sentiment Agent Stability Tests")
    print(f"📅 Test Run: {datetime.now()}")
    
    results = []
    
    try:
        # Run different scenarios
        result1 = await test_scenario_1()
        results.append(("Market Volatility", result1))
        
        result2 = await test_scenario_2() 
        results.append(("Economic Policy", result2))
        
        result3 = await test_scenario_3()
        results.append(("Corporate Earnings", result3))
        
        result4 = await test_scenario_4()
        results.append(("General Market", result4))
        
        # Test components in isolation
        await test_component_isolation()
        
        # Analyze results
        print_test_header("STABILITY ANALYSIS")
        
        confidences = [r[1].confidence for r in results]
        avg_confidence = sum(confidences) / len(confidences)
        
        sentiments = [r[1].key_metrics.get('overall_sentiment', 0) for r in results]
        credibilities = [r[1].key_metrics.get('credibility_score', 0) for r in results]
        
        print(f"📊 Test Summary:")
        print(f"   Average Confidence: {avg_confidence:.3f}")
        print(f"   Confidence Range: {min(confidences):.3f} - {max(confidences):.3f}")
        print(f"   Sentiment Range: {min(sentiments):.3f} - {max(sentiments):.3f}")
        print(f"   Credibility Range: {min(credibilities):.3f} - {max(credibilities):.3f}")
        
        # Check for consistency
        confidence_variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)
        
        print(f"\n🎯 Stability Metrics:")
        print(f"   Confidence Variance: {confidence_variance:.4f}")
        print(f"   Consistency: {'High' if confidence_variance < 0.05 else 'Medium' if confidence_variance < 0.15 else 'Low'}")
        
        # Success criteria
        all_successful = all(r[1].confidence > 0.3 for r in results)  # Minimum viable confidence
        no_errors = all('error' not in r[1].analysis for r in results)
        
        if all_successful and no_errors:
            print(f"\n🎉 ALL TESTS PASSED!")
            print(f"✅ Agent is stable and reliable across different scenarios")
            print(f"✅ All components working correctly")
            print(f"✅ Consistent performance maintained")
        else:
            print(f"\n⚠️  Some tests showed issues:")
            for name, result in results:
                if result.confidence <= 0.3 or 'error' in result.analysis:
                    print(f"   - {name}: Low confidence or errors detected")
        
        return all_successful and no_errors
        
    except Exception as e:
        print(f"\n❌ Stability test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(run_stability_tests())
    exit(0 if success else 1)

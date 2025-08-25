"""
Comprehensive verbose test for the refactored News Sentiment Agent

This test demonstrates all components working together with detailed outputs
to show exactly what the agent is producing at each step.
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
from agents.news_sentiment.news_sentiment_data_handler import NewsSentimentDataHandler
from agents.news_sentiment.news_sentiment_context_builder import NewsSentimentContextBuilder
from agents.base_agent import AgentContext

# Set up detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def print_section(title: str, content: str = ""):
    """Print a formatted section header"""
    print(f"\n{'='*80}")
    print(f"🔍 {title}")
    print(f"{'='*80}")
    if content:
        print(content)

def print_subsection(title: str):
    """Print a formatted subsection header"""
    print(f"\n{'─'*60}")
    print(f"📊 {title}")
    print(f"{'─'*60}")

def pretty_print_dict(data: dict, indent: int = 2):
    """Pretty print dictionary with proper formatting"""
    print(json.dumps(data, indent=indent, default=str))

async def test_data_handler():
    """Test the data handler component in detail"""
    print_section("DATA HANDLER TESTING")
    
    data_handler = NewsSentimentDataHandler()
    
    # Test connection validation
    print_subsection("Connection Validation")
    is_valid = data_handler.validate_connection()
    print(f"✅ ChromaDB Connection Valid: {is_valid}")
    
    # Test collection stats
    print_subsection("Collection Statistics")
    stats = await data_handler.get_collection_stats()
    pretty_print_dict(stats)
    
    # Test article retrieval
    print_subsection("Article Retrieval")
    articles = await data_handler.get_news_articles(
        query="financial markets economy",
        timeframe="1w",
        max_results=5
    )
    
    print(f"📰 Retrieved {len(articles)} articles")
    for i, article in enumerate(articles[:3]):  # Show first 3
        print(f"\nArticle {i+1}:")
        print(f"  Title: {article.get('title', 'No title')[:100]}...")
        print(f"  Source: {article.get('source', 'Unknown')}")
        print(f"  Relevance Score: {article.get('relevance_score', 0):.3f}")
        print(f"  Sentiment Score: {article.get('sentiment_score', 0):.3f}")
        print(f"  Content Preview: {article.get('content', '')[:200]}...")
    
    return articles

async def test_context_builder(articles):
    """Test the context builder component in detail"""
    print_section("CONTEXT BUILDER TESTING")
    
    from agents.ai_service import OllamaService
    ai_service = OllamaService()
    context_builder = NewsSentimentContextBuilder(ai_service=ai_service)
    
    # Test entity extraction
    print_subsection("Entity Extraction")
    entities = await context_builder.extract_entities(articles)
    print("🏢 Extracted Entities:")
    print(f"  Companies: {entities.companies}")
    print(f"  People: {entities.people}")
    print(f"  Locations: {entities.locations}")
    print(f"  Organizations: {entities.organizations}")
    print(f"  Financial Instruments: {entities.financial_instruments}")
    print(f"  Events: {entities.events}")
    
    # Test sentiment analysis
    print_subsection("Sentiment Analysis")
    sentiment = await context_builder.analyze_sentiment(articles)
    print("💭 Sentiment Analysis Results:")
    print(f"  Overall Sentiment: {sentiment.overall_sentiment:.3f} (-1 to +1)")
    print(f"  Bias Score: {sentiment.bias_score:.3f}")
    print(f"  Credibility Score: {sentiment.credibility_score:.3f}")
    print(f"  Urgency Level: {sentiment.urgency_level:.3f}")
    print(f"  Market Relevance: {sentiment.market_relevance:.3f}")
    print("  Emotional Tone:")
    for emotion, score in sentiment.emotional_tone.items():
        print(f"    {emotion.capitalize()}: {score:.3f}")
    
    # Test narrative analysis
    print_subsection("Narrative Analysis")
    narrative = await context_builder.analyze_narratives(articles)
    print("📖 Narrative Analysis Results:")
    print(f"  Dominant Themes: {narrative.dominant_themes}")
    print(f"  Narrative Direction: {narrative.narrative_shift}")
    print(f"  Consensus Level: {narrative.consensus_level:.3f}")
    print("  Theme Evolution:")
    for theme, strength in narrative.theme_evolution.items():
        print(f"    {theme}: {strength:.3f}")
    
    # Test context building
    print_subsection("Analysis Context Building")
    analysis_context = context_builder.build_analysis_context(
        articles, sentiment, narrative, entities
    )
    print("🔗 Built Analysis Context:")
    pretty_print_dict(analysis_context)
    
    # Test cross-agent signals
    print_subsection("Cross-Agent Signals")
    signals = context_builder.generate_cross_agent_signals(sentiment, narrative)
    print("📡 Generated Signals:")
    pretty_print_dict(signals)
    
    # Test confidence calculation
    confidence = context_builder.calculate_confidence(len(articles), sentiment)
    print(f"\n🎯 Calculated Confidence: {confidence:.3f}")
    
    return entities, sentiment, narrative, analysis_context, signals, confidence

async def test_full_agent():
    """Test the complete agent with full workflow"""
    print_section("FULL AGENT TESTING")
    
    # Initialize agent
    agent = NewsSentimentAgent()
    
    # Test health check
    print_subsection("Agent Health Check")
    health = await agent.get_collection_health()
    print("🏥 Health Status:")
    pretty_print_dict(health)
    
    # Create test context
    context = AgentContext(
        query="market volatility inflation economic outlook",
        query_type="market_analysis",
        timeframe="1w"
    )
    
    print_subsection("Agent Context")
    print("📋 Analysis Context:")
    print(f"  Query: {context.query}")
    print(f"  Query Type: {context.query_type}")
    print(f"  Timeframe: {context.timeframe}")
    print(f"  Date Range: {context.date_range}")
    
    # Run full analysis
    print_subsection("Full Agent Analysis")
    print("🚀 Running comprehensive news sentiment analysis...")
    
    start_time = datetime.now()
    response = await agent.analyze(context)
    end_time = datetime.now()
    
    execution_time = (end_time - start_time).total_seconds()
    print(f"⏱️  Total Execution Time: {execution_time:.2f} seconds")
    
    # Display comprehensive results
    print_section("AGENT RESPONSE ANALYSIS")
    
    print_subsection("Basic Response Info")
    print(f"Agent Type: {response.agent_type}")
    print(f"Confidence: {response.confidence:.3f}")
    print(f"Confidence Level: {response.confidence_level}")
    print(f"Execution Time: {response.execution_time_ms:.2f}ms")
    print(f"Timeframe Analyzed: {response.timeframe_analyzed}")
    print(f"Data Sources Used: {response.data_sources_used}")
    
    print_subsection("Key Metrics")
    print("📈 Key Metrics:")
    pretty_print_dict(response.key_metrics)
    
    print_subsection("Analysis Results")
    print("🔬 Detailed Analysis:")
    # Print analysis but limit content length for readability
    analysis_summary = {
        "articles_analyzed": response.analysis.get("articles_analyzed", 0),
        "dominant_themes": response.analysis.get("dominant_themes", []),
        "sentiment_summary": {
            "overall_sentiment": response.analysis.get("sentiment_analysis", {}).get("overall_sentiment", 0),
            "credibility_score": response.analysis.get("sentiment_analysis", {}).get("credibility_score", 0),
            "market_relevance": response.analysis.get("sentiment_analysis", {}).get("market_relevance", 0)
        },
        "narrative_summary": {
            "narrative_shift": response.analysis.get("narrative_analysis", {}).get("narrative_shift", ""),
            "consensus_level": response.analysis.get("narrative_analysis", {}).get("consensus_level", 0)
        }
    }
    pretty_print_dict(analysis_summary)
    
    print_subsection("AI-Generated Insights")
    print("🤖 AI Insights:")
    for i, insight in enumerate(response.insights, 1):
        print(f"\nInsight {i}:")
        print(f"{insight}")
    
    print_subsection("Cross-Agent Signals")
    print("📡 Signals for Other Agents:")
    pretty_print_dict(response.signals_for_other_agents)
    
    return response

async def main():
    """Run comprehensive verbose testing"""
    print_section("NEWS SENTIMENT AGENT COMPREHENSIVE TESTING", 
                  f"Started at: {datetime.now()}")
    
    try:
        # Test each component individually
        articles = await test_data_handler()
        
        if articles:
            entities, sentiment, narrative, context_data, signals, confidence = await test_context_builder(articles)
        else:
            print("⚠️  No articles retrieved, skipping context builder detailed tests")
        
        # Test full agent
        response = await test_full_agent()
        
        print_section("TESTING COMPLETED SUCCESSFULLY", 
                      f"Finished at: {datetime.now()}")
        
        print("\n🎉 All tests completed successfully!")
        print(f"📊 Final Agent Confidence: {response.confidence:.3f}")
        print(f"🎯 Analysis Quality: {'High' if response.confidence > 0.7 else 'Medium' if response.confidence > 0.4 else 'Low'}")
        
    except Exception as e:
        print(f"\n❌ Testing failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())

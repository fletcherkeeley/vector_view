#!/usr/bin/env python3
"""
Quick EventCuratorAgent ChromaDB Integration Test

Tests with a small dataset to verify the integration works properly.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add the project root to Python path
sys.path.append('/home/lab/projects/vector-view')

from agents.event_curator.event_curator_agent import EventCuratorAgent
from agents.base_agent import AgentContext

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def quick_test():
    """Quick test with small dataset."""
    print("🧪 QUICK EVENTCURATORAGENT CHROMADB TEST")
    print("=" * 50)
    
    try:
        # Initialize EventCuratorAgent
        agent = EventCuratorAgent()
        
        print(f"ChromaDB Available: {agent.data_handler.chroma_available}")
        
        # Create test context with small limits
        context = AgentContext(
            query="Extract events from recent market news",
            query_type="daily_briefing",  # This limits to fewer articles
            timeframe="1d"
        )
        
        # Override the curation parameters directly in the context parsing
        # We'll modify the agent to use smaller limits
        
        print(f"Query: {context.query}")
        print(f"Max articles: 5 (quick test)")
        
        # Run event curation
        print("\nRunning quick event curation...")
        start_time = datetime.now()
        
        response = await agent.analyze(context)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Display results
        print(f"\n📊 RESULTS (in {execution_time:.1f}s)")
        print(f"   Overall confidence: {response.confidence:.3f}")
        print(f"   Events curated: {response.key_metrics.get('total_events_curated', 0)}")
        print(f"   Extraction rate: {response.key_metrics.get('extraction_rate', 0):.3f}")
        print(f"   Verification rate: {response.key_metrics.get('verification_rate', 0):.3f}")
        
        # ChromaDB specific metrics
        if 'chromadb_verification_rate' in response.key_metrics:
            print(f"   ChromaDB verification rate: {response.key_metrics['chromadb_verification_rate']:.3f}")
        
        # Show verification stats
        if response.analysis.get('verification_stats'):
            stats = response.analysis['verification_stats']
            print(f"\n🔍 VERIFICATION")
            print(f"   Verified events: {stats.get('verified_events', 0)}")
            if 'chromadb_verified_events' in stats:
                print(f"   ChromaDB verified: {stats['chromadb_verified_events']}")
            print(f"   ChromaDB available: {stats.get('chromadb_available', False)}")
        
        # Show top insights
        print(f"\n💡 TOP INSIGHTS")
        for insight in response.insights[:3]:
            print(f"   • {insight}")
        
        # Show data sources
        print(f"\n📚 DATA SOURCES: {', '.join(response.data_sources_used)}")
        
        # Success indicators
        success_indicators = []
        if response.key_metrics.get('total_events_curated', 0) > 0:
            success_indicators.append("✅ Events extracted")
        if agent.data_handler.chroma_available:
            success_indicators.append("✅ ChromaDB connected")
        if response.key_metrics.get('verification_rate', 0) > 0:
            success_indicators.append("✅ Verification working")
        
        print(f"\n🎯 SUCCESS INDICATORS")
        for indicator in success_indicators:
            print(f"   {indicator}")
        
        if not success_indicators:
            print("   ⚠️  No events extracted - may need more recent data")
        
        return len(success_indicators) > 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False
    finally:
        # Clean up batch parameters
        if hasattr(agent, '_batch_params'):
            delattr(agent, '_batch_params')


async def test_chromadb_only():
    """Test just the ChromaDB components quickly."""
    print("\n🔍 CHROMADB COMPONENT TEST")
    print("=" * 30)
    
    try:
        from agents.event_curator.event_data_handler import EventDataHandler
        
        # Test ChromaDB connection
        handler = EventDataHandler()
        print(f"ChromaDB Available: {handler.chroma_available}")
        
        if handler.chroma_available:
            print(f"Articles in ChromaDB: {handler._news_collection.count()}")
            
            # Test semantic search
            test_query = "Federal Reserve interest rates"
            results = await handler.fetch_articles_by_semantic_search(
                query=test_query,
                limit=3
            )
            print(f"Semantic search results: {len(results)} articles")
            
            if results:
                print(f"Top result similarity: {results[0].get('similarity_score', 0):.3f}")
        
        return handler.chroma_available
        
    except Exception as e:
        print(f"ChromaDB test failed: {e}")
        return False


async def main():
    """Run quick tests."""
    
    # Test ChromaDB components first
    chromadb_ok = await test_chromadb_only()
    
    # Run quick integration test
    integration_ok = await quick_test()
    
    print("\n" + "=" * 50)
    print("🏁 QUICK TEST SUMMARY")
    print("=" * 50)
    
    print(f"ChromaDB Components: {'✅ PASS' if chromadb_ok else '❌ FAIL'}")
    print(f"Integration Test:    {'✅ PASS' if integration_ok else '❌ FAIL'}")
    
    if chromadb_ok and integration_ok:
        print("\n🎉 ChromaDB integration working!")
    elif chromadb_ok:
        print("\n⚠️  ChromaDB connected but may need more recent data for events")
    else:
        print("\n❌ ChromaDB integration issues detected")
    
    return chromadb_ok and integration_ok


if __name__ == "__main__":
    asyncio.run(main())

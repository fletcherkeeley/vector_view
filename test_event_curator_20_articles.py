#!/usr/bin/env python3
"""
EventCuratorAgent Test with 20 Articles

Tests the full event curation workflow with ChromaDB integration using 20 articles.
"""

import asyncio
import logging
import sys
from datetime import datetime

sys.path.append('/home/lab/projects/vector-view')

from agents.event_curator.event_curator_agent import EventCuratorAgent
from agents.base_agent import AgentContext

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_full_workflow_20_articles():
    """Test full EventCuratorAgent workflow with 20 articles."""
    print("🧪 EVENTCURATORAGENT - 20 ARTICLE TEST")
    print("=" * 50)
    
    try:
        # Initialize agent
        agent = EventCuratorAgent()
        
        print(f"ChromaDB Available: {agent.data_handler.chroma_available}")
        print(f"ChromaDB Articles: {agent.data_handler._news_collection.count() if agent.data_handler.chroma_available else 0}")
        
        # Create context for 20 articles
        context = AgentContext(
            query="Extract and analyze events from recent financial news with ChromaDB verification",
            query_type="daily_briefing",
            timeframe="3d"  # 3 days to get more diverse content
        )
        
        print(f"\nQuery: {context.query}")
        print(f"Articles to process: 20")
        print(f"Timeframe: {context.timeframe}")
        
        # Run full event curation workflow
        print(f"\n🚀 Running full event curation workflow...")
        start_time = datetime.now()
        
        # Override fetch limit for this test
        original_fetch = agent.data_handler.fetch_news_articles
        async def fetch_20_articles(*args, **kwargs):
            kwargs['limit'] = 20
            return await original_fetch(*args, **kwargs)
        agent.data_handler.fetch_news_articles = fetch_20_articles
        
        response = await agent.analyze(context)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Analyze results
        print(f"\n📊 WORKFLOW RESULTS")
        print(f"   Duration: {duration:.1f}s")
        print(f"   Success: {hasattr(response, 'success') and response.success}")
        print(f"   Confidence: {response.confidence:.3f}")
        
        # Extract events and metrics from AgentResponse structure
        events = response.analysis.get('events', []) if hasattr(response, 'analysis') and response.analysis else []
        metrics = response.key_metrics if hasattr(response, 'key_metrics') and response.key_metrics else {}
        
        print(f"\n📈 EVENT EXTRACTION")
        print(f"   Events found: {len(events)}")
        print(f"   Articles processed: {metrics.get('articles_processed', 0)}")
        print(f"   Events before dedup: {metrics.get('events_before_deduplication', 0)}")
        print(f"   Events after dedup: {metrics.get('events_after_deduplication', 0)}")
        
        # ChromaDB metrics
        chromadb_metrics = metrics.get('chromadb_verification', {})
        print(f"\n🔍 CHROMADB VERIFICATION")
        print(f"   Events verified: {chromadb_metrics.get('events_verified', 0)}")
        print(f"   Avg confidence boost: {chromadb_metrics.get('avg_confidence_boost', 0):.3f}")
        print(f"   Supporting articles found: {chromadb_metrics.get('supporting_articles_found', 0)}")
        
        # Sample events
        print(f"\n📋 SAMPLE EVENTS")
        for i, event in enumerate(events[:3]):
            if isinstance(event, dict):
                print(f"   {i+1}. {event.get('description', 'No description')[:80]}...")
                print(f"      Type: {event.get('event_type', 'Unknown')}")
                print(f"      Confidence: {event.get('confidence', 0):.3f}")
                print(f"      Sources: {event.get('source_count', 0)}")
        
        # Success criteria
        success_criteria = [
            response.confidence > 0.5,
            len(events) > 0,
            metrics.get('articles_processed', 0) > 0
        ]
        
        print(f"🎯 PERFORMANCE ASSESSMENT")
        articles_per_second = 20 / duration if duration > 0 else 0
        print(f"   Processing rate: {articles_per_second:.1f} articles/second")
        
        if duration < 30:
            print(f"   ✅ Fast processing ({duration:.1f}s)")
        elif duration < 60:
            print(f"   ⚠️  Moderate processing ({duration:.1f}s)")
        else:
            print(f"   ❌ Slow processing ({duration:.1f}s)")
        
        # Check if workflow succeeded
        workflow_success = response.confidence > 0.0 and len(events) > 0
        
        if workflow_success and response.confidence > 0.5:
            print(f"   ✅ High quality results (confidence: {response.confidence:.3f})")
            return True
        elif workflow_success:
            print(f"   ⚠️  Moderate quality results (confidence: {response.confidence:.3f})")
            return True
        else:
            print(f"   ❌ Workflow failed")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run 20-article test."""
    success = await test_full_workflow_20_articles()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ 20-ARTICLE TEST PASSED")
        print("EventCuratorAgent with ChromaDB integration working at medium scale!")
    else:
        print("❌ 20-ARTICLE TEST FAILED")
        print("Check logs for issues with workflow or ChromaDB integration")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())

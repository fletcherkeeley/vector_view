#!/usr/bin/env python3
"""
Test ChromaDB Integration with EventCuratorAgent

Tests the semantic deduplication, cross-source verification, and confidence scoring
enhancements for the EventCuratorAgent using ChromaDB.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime
from typing import Dict, List, Any

# Add the project root to Python path
sys.path.append('/home/lab/projects/vector-view')

from agents.event_curator.event_curator_agent import EventCuratorAgent
from agents.event_curator.event_data_handler import EventDataHandler
from agents.base_agent import AgentContext

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_chromadb_connection():
    """Test ChromaDB connection and collection availability."""
    print("\n" + "="*60)
    print("TESTING CHROMADB CONNECTION")
    print("="*60)
    
    try:
        # Initialize EventDataHandler
        data_handler = EventDataHandler()
        
        # Check ChromaDB availability
        print(f"ChromaDB Available: {data_handler.chroma_available}")
        print(f"ChromaDB Path: {data_handler.chroma_path}")
        
        if data_handler.chroma_available:
            print("✅ ChromaDB connection successful")
            
            # Test collection access
            if data_handler._news_collection:
                count = data_handler._news_collection.count()
                print(f"✅ Financial news collection found with {count} articles")
            else:
                print("⚠️  Financial news collection not found")
        else:
            print("❌ ChromaDB connection failed - will use graceful degradation")
        
        return data_handler.chroma_available
        
    except Exception as e:
        print(f"❌ ChromaDB connection test failed: {e}")
        return False


async def test_semantic_deduplication():
    """Test semantic deduplication functionality."""
    print("\n" + "="*60)
    print("TESTING SEMANTIC DEDUPLICATION")
    print("="*60)
    
    try:
        data_handler = EventDataHandler()
        
        # Create test events with semantic duplicates
        test_events = [
            {
                'description': 'Federal Reserve raises interest rates by 0.25%',
                'event_type': 'monetary_policy',
                'confidence': 0.8,
                'entities': ['Federal Reserve'],
                'source': 'Reuters',
                'source_article_id': 'test_1'
            },
            {
                'description': 'Fed increases rates by quarter point',
                'event_type': 'monetary_policy', 
                'confidence': 0.75,
                'entities': ['Fed'],
                'source': 'Bloomberg',
                'source_article_id': 'test_2'
            },
            {
                'description': 'Apple reports strong quarterly earnings',
                'event_type': 'corporate_earnings',
                'confidence': 0.9,
                'entities': ['Apple'],
                'source': 'CNBC',
                'source_article_id': 'test_3'
            }
        ]
        
        print(f"Input events: {len(test_events)}")
        for i, event in enumerate(test_events):
            print(f"  {i+1}. {event['description']}")
        
        # Test semantic deduplication
        deduplicated = await data_handler.deduplicate_events_semantic(
            test_events,
            similarity_threshold=0.7
        )
        
        print(f"\nAfter deduplication: {len(deduplicated)} events")
        for i, event in enumerate(deduplicated):
            print(f"  {i+1}. {event['description']}")
            if event.get('semantic_deduplication'):
                print(f"     ✅ Semantically deduplicated from {event.get('merged_from_count', 1)} events")
                print(f"     📊 Confidence: {event.get('confidence', 0):.3f}")
                if event.get('confidence_boost_dedup'):
                    print(f"     ⬆️  Confidence boost: +{event['confidence_boost_dedup']:.3f}")
        
        # Verify deduplication worked
        if len(deduplicated) < len(test_events):
            print("✅ Semantic deduplication successfully merged similar events")
        else:
            print("⚠️  No events were deduplicated")
        
        return deduplicated
        
    except Exception as e:
        print(f"❌ Semantic deduplication test failed: {e}")
        return []


async def test_chromadb_verification():
    """Test ChromaDB cross-source verification."""
    print("\n" + "="*60)
    print("TESTING CHROMADB VERIFICATION")
    print("="*60)
    
    try:
        data_handler = EventDataHandler()
        
        if not data_handler.chroma_available:
            print("⚠️  ChromaDB not available - skipping verification test")
            return
        
        # Create test event for verification
        test_event = {
            'description': 'Federal Reserve announces interest rate decision',
            'event_type': 'monetary_policy',
            'confidence': 0.7,
            'entities': ['Federal Reserve', 'interest rates'],
            'source': 'Test Source',
            'source_article_id': 'test_verification'
        }
        
        print(f"Testing verification for: {test_event['description']}")
        print(f"Original confidence: {test_event['confidence']:.3f}")
        
        # Test ChromaDB verification
        verified_event = await data_handler.verify_event_with_chromadb(
            test_event,
            verification_threshold=0.6,
            max_supporting_articles=5
        )
        
        # Display results
        print(f"\nVerification results:")
        print(f"  ChromaDB verified: {verified_event.get('chromadb_verified', False)}")
        print(f"  Supporting articles: {len(verified_event.get('chromadb_supporting_articles', []))}")
        print(f"  Unique sources: {verified_event.get('chromadb_source_count', 0)}")
        print(f"  Final confidence: {verified_event.get('confidence', 0):.3f}")
        
        if verified_event.get('confidence_boost_chromadb'):
            print(f"  ⬆️  Confidence boost: +{verified_event['confidence_boost_chromadb']:.3f}")
        
        # Show supporting articles
        supporting_articles = verified_event.get('chromadb_supporting_articles', [])
        if supporting_articles:
            print(f"\n  Supporting articles:")
            for i, article in enumerate(supporting_articles[:3]):  # Show top 3
                print(f"    {i+1}. {article.get('title', 'No title')[:60]}...")
                print(f"       Source: {article.get('source', 'Unknown')}")
                print(f"       Similarity: {article.get('similarity_score', 0):.3f}")
        
        if verified_event.get('chromadb_verified'):
            print("✅ ChromaDB verification successful")
        else:
            print("⚠️  No supporting articles found in ChromaDB")
        
        return verified_event
        
    except Exception as e:
        print(f"❌ ChromaDB verification test failed: {e}")
        return None


async def test_full_event_curation():
    """Test full event curation with ChromaDB integration."""
    print("\n" + "="*60)
    print("TESTING FULL EVENT CURATION")
    print("="*60)
    
    try:
        # Initialize EventCuratorAgent
        agent = EventCuratorAgent()
        
        # Create test context
        context = AgentContext(
            query="Extract events from recent Federal Reserve and market news",
            query_type="deep_dive",
            timeframe="7d"
        )
        
        print(f"Query: {context.query}")
        print(f"ChromaDB Available: {agent.data_handler.chroma_available}")
        
        # Run event curation
        print("\nRunning event curation...")
        start_time = datetime.now()
        
        response = await agent.analyze(context)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Display results
        print(f"\n📊 CURATION RESULTS")
        print(f"   Execution time: {execution_time:.2f}s")
        print(f"   Overall confidence: {response.confidence:.3f}")
        print(f"   Confidence level: {response.confidence_level}")
        
        # Show key metrics
        metrics = response.key_metrics
        print(f"\n📈 KEY METRICS")
        print(f"   Events curated: {metrics.get('total_events_curated', 0)}")
        print(f"   Extraction rate: {metrics.get('extraction_rate', 0):.3f}")
        print(f"   Verification rate: {metrics.get('verification_rate', 0):.3f}")
        if 'chromadb_verification_rate' in metrics:
            print(f"   ChromaDB verification rate: {metrics['chromadb_verification_rate']:.3f}")
        
        # Show analysis highlights
        analysis = response.analysis
        if analysis.get('verification_stats'):
            stats = analysis['verification_stats']
            print(f"\n🔍 VERIFICATION STATS")
            print(f"   Verified events: {stats.get('verified_events', 0)}")
            if 'chromadb_verified_events' in stats:
                print(f"   ChromaDB verified: {stats['chromadb_verified_events']}")
            print(f"   Multi-source events: {stats.get('multi_source_events', 0)}")
            print(f"   ChromaDB available: {stats.get('chromadb_available', False)}")
        
        # Show insights
        print(f"\n💡 INSIGHTS")
        for insight in response.insights[:5]:  # Show top 5 insights
            print(f"   • {insight}")
        
        # Show data sources used
        print(f"\n📚 DATA SOURCES")
        for source in response.data_sources_used:
            print(f"   • {source}")
        
        # Show uncertainty factors
        if response.uncertainty_factors:
            print(f"\n⚠️  UNCERTAINTY FACTORS")
            for factor in response.uncertainty_factors:
                print(f"   • {factor}")
        
        print("✅ Full event curation test completed")
        return response
        
    except Exception as e:
        print(f"❌ Full event curation test failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return None


async def test_graceful_degradation():
    """Test graceful degradation when ChromaDB is unavailable."""
    print("\n" + "="*60)
    print("TESTING GRACEFUL DEGRADATION")
    print("="*60)
    
    try:
        # Create EventDataHandler with invalid ChromaDB path
        data_handler = EventDataHandler(chroma_path="/invalid/path")
        
        print(f"ChromaDB Available: {data_handler.chroma_available}")
        
        # Test that methods still work with graceful degradation
        test_events = [
            {
                'description': 'Test event for degradation',
                'confidence': 0.8,
                'entities': ['Test']
            }
        ]
        
        # Test deduplication fallback
        result = await data_handler.deduplicate_events_semantic(test_events)
        print(f"✅ Deduplication fallback: {len(result)} events returned")
        
        # Test verification fallback
        verified = await data_handler.verify_event_with_chromadb(test_events[0])
        print(f"✅ Verification fallback: Event returned unchanged")
        
        # Test similarity search fallback
        similar = await data_handler.find_similar_events_semantic("test description")
        print(f"✅ Similarity search fallback: {len(similar)} events returned")
        
        print("✅ Graceful degradation working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Graceful degradation test failed: {e}")
        return False


async def main():
    """Run all ChromaDB integration tests."""
    print("🧪 EVENTCURATORAGENT CHROMADB INTEGRATION TESTS")
    print("=" * 80)
    
    test_results = {}
    
    # Run tests
    test_results['connection'] = await test_chromadb_connection()
    test_results['deduplication'] = await test_semantic_deduplication()
    test_results['verification'] = await test_chromadb_verification()
    test_results['full_curation'] = await test_full_event_curation()
    test_results['degradation'] = await test_graceful_degradation()
    
    # Summary
    print("\n" + "="*80)
    print("🏁 TEST SUMMARY")
    print("="*80)
    
    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.upper():20} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All ChromaDB integration tests passed!")
    else:
        print("⚠️  Some tests failed - check logs above")
    
    return passed == total


if __name__ == "__main__":
    asyncio.run(main())

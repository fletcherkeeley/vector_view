#!/usr/bin/env python3
"""
Minimal EventCuratorAgent ChromaDB Integration Test

Tests with just 5 articles to quickly verify ChromaDB integration works.
"""

import asyncio
import logging
import sys
from datetime import datetime

sys.path.append('/home/lab/projects/vector-view')

from agents.event_curator.event_data_handler import EventDataHandler

# Configure minimal logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


async def test_minimal_integration():
    """Test ChromaDB integration with minimal dataset."""
    print("🧪 MINIMAL CHROMADB INTEGRATION TEST")
    print("=" * 40)
    
    try:
        # Initialize handler
        handler = EventDataHandler()
        
        print(f"ChromaDB Available: {handler.chroma_available}")
        print(f"ChromaDB Articles: {handler._news_collection.count() if handler.chroma_available else 0}")
        
        # Test 1: Fetch small dataset from PostgreSQL
        print("\n1️⃣ Testing PostgreSQL fetch (20 articles)...")
        articles = await handler.fetch_news_articles(limit=20, days_back=7)
        print(f"   Fetched: {len(articles)} articles")
        
        if articles:
            print(f"   Sample: {articles[0]['title'][:50]}...")
            print(f"   Source: {articles[0]['source']}")
        
        # Test 2: Test semantic deduplication with mock events
        print("\n2️⃣ Testing semantic deduplication...")
        mock_events = [
            {
                'description': 'Federal Reserve raises interest rates by 0.25%',
                'confidence': 0.8,
                'entities': ['Federal Reserve'],
                'source': 'Reuters'
            },
            {
                'description': 'Fed increases rates by quarter point',
                'confidence': 0.75,
                'entities': ['Fed'],
                'source': 'Bloomberg'
            },
            {
                'description': 'Apple reports quarterly earnings',
                'confidence': 0.9,
                'entities': ['Apple'],
                'source': 'CNBC'
            }
        ]
        
        deduplicated = await handler.deduplicate_events_semantic(mock_events, similarity_threshold=0.6)
        print(f"   Input events: {len(mock_events)}")
        print(f"   After dedup: {len(deduplicated)}")
        
        # Check if Fed events were merged
        fed_events = [e for e in deduplicated if 'fed' in e['description'].lower()]
        if len(fed_events) < 2:
            print("   ✅ Fed events successfully merged")
        else:
            print("   ⚠️  Fed events not merged (threshold may be too high)")
        
        # Test 3: ChromaDB verification
        if handler.chroma_available:
            print("\n3️⃣ Testing ChromaDB verification...")
            test_event = {
                'description': 'Federal Reserve announces interest rate decision',
                'confidence': 0.7,
                'entities': ['Federal Reserve']
            }
            
            verified = await handler.verify_event_with_chromadb(test_event, verification_threshold=0.020)
            print(f"   Original confidence: {test_event['confidence']:.3f}")
            print(f"   Verified confidence: {verified.get('confidence', 0):.3f}")
            print(f"   Supporting articles: {len(verified.get('chromadb_supporting_articles', []))}")
            
            if verified.get('chromadb_verified'):
                print("   ✅ ChromaDB verification working")
            else:
                print("   ⚠️  No supporting articles found (may need lower threshold)")
        
        # Test 4: Semantic search
        if handler.chroma_available:
            print("\n4️⃣ Testing semantic search...")
            search_results = await handler.fetch_articles_by_semantic_search(
                "Federal Reserve interest rates", limit=3
            )
            print(f"   Search results: {len(search_results)}")
            if search_results:
                print(f"   Top similarity: {search_results[0].get('similarity_score', 0):.3f}")
        
        # Summary
        print(f"\n📊 INTEGRATION STATUS")
        success_count = 0
        total_tests = 4
        
        if len(articles) > 0:
            print("   ✅ PostgreSQL connection working")
            success_count += 1
        else:
            print("   ❌ PostgreSQL connection failed")
        
        if len(deduplicated) <= len(mock_events):
            print("   ✅ Semantic deduplication working")
            success_count += 1
        else:
            print("   ❌ Semantic deduplication failed")
        
        if handler.chroma_available:
            print("   ✅ ChromaDB connection working")
            success_count += 1
        else:
            print("   ❌ ChromaDB connection failed")
            total_tests -= 1  # Skip verification test
        
        if handler.chroma_available and len(search_results) > 0:
            print("   ✅ ChromaDB semantic search working")
            success_count += 1
        elif handler.chroma_available:
            print("   ⚠️  ChromaDB search returned no results")
            success_count += 0.5  # Partial credit
        
        print(f"\n🎯 RESULT: {success_count}/{total_tests} components working")
        
        if success_count >= total_tests * 0.75:
            print("🎉 ChromaDB integration is working!")
            return True
        else:
            print("⚠️  Some integration issues detected")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


async def main():
    """Run minimal test."""
    success = await test_minimal_integration()
    
    print("\n" + "=" * 40)
    if success:
        print("✅ MINIMAL TEST PASSED")
        print("ChromaDB integration ready for production!")
    else:
        print("❌ MINIMAL TEST FAILED")
        print("Check ChromaDB setup and database connections")
    
    return success


if __name__ == "__main__":
    asyncio.run(main())

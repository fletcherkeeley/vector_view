#!/usr/bin/env python3
"""
Test script for EventCuratorAgent

Tests event extraction, verification, and Neo4j storage with sample news articles.
"""

import asyncio
import sys
import logging
from datetime import datetime, timedelta
from agents.event_curator.event_curator_agent import EventCuratorAgent
from agents.base_agent import AgentContext

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sample news articles for testing
SAMPLE_ARTICLES = [
    {
        'article_id': 'test_001',
        'title': 'Federal Reserve Raises Interest Rates by 0.25%',
        'description': 'The Federal Reserve announced a quarter-point increase in the federal funds rate to combat inflation.',
        'content': 'The Federal Open Market Committee (FOMC) voted unanimously to raise the federal funds rate by 25 basis points to a range of 5.25% to 5.50%. Fed Chair Jerome Powell cited persistent inflation concerns as the primary driver for the decision. The rate hike was widely expected by markets and represents the 11th increase since the Fed began its tightening cycle in March 2022.',
        'source': 'Reuters',
        'published_at': datetime.now() - timedelta(hours=2),
        'category': 'federal_reserve',
        'relevance_score': 0.95,
        'quality_score': 0.90,
        'created_at': datetime.now()
    },
    {
        'article_id': 'test_002', 
        'title': 'Apple Reports Record Q3 Earnings, Beats Expectations',
        'description': 'Apple Inc. reported quarterly earnings that exceeded analyst expectations, driven by strong iPhone sales.',
        'content': 'Apple Inc. (AAPL) reported third-quarter earnings of $1.26 per share, beating the consensus estimate of $1.19. Revenue came in at $81.8 billion, up 1% year-over-year. iPhone revenue was $39.7 billion, representing a 2% increase from the prior year. CEO Tim Cook highlighted strong performance in Services revenue, which grew 8% to $21.2 billion.',
        'source': 'Bloomberg',
        'published_at': datetime.now() - timedelta(hours=4),
        'category': 'corporate_earnings',
        'relevance_score': 0.88,
        'quality_score': 0.85,
        'created_at': datetime.now()
    },
    {
        'article_id': 'test_003',
        'title': 'Fed Chair Powell Signals More Rate Hikes May Be Needed',
        'description': 'Jerome Powell indicated the Federal Reserve may continue raising rates if inflation remains elevated.',
        'content': 'Speaking at the Jackson Hole Economic Symposium, Federal Reserve Chair Jerome Powell warned that additional interest rate increases may be necessary to bring inflation back to the Fed\'s 2% target. Powell emphasized the Fed\'s commitment to price stability and noted that recent economic data shows inflation remains stubbornly high. Markets reacted negatively to the hawkish tone, with the S&P 500 falling 1.2%.',
        'source': 'Wall Street Journal',
        'published_at': datetime.now() - timedelta(hours=6),
        'category': 'federal_reserve',
        'relevance_score': 0.92,
        'quality_score': 0.88,
        'created_at': datetime.now()
    },
    {
        'article_id': 'test_004',
        'title': 'Microsoft Announces $10 Billion AI Investment',
        'description': 'Microsoft Corp. unveiled plans to invest $10 billion in artificial intelligence infrastructure over the next two years.',
        'content': 'Microsoft Corporation announced a major $10 billion investment in AI infrastructure, including new data centers and partnerships with leading AI companies. CEO Satya Nadella said the investment will accelerate Microsoft\'s AI capabilities across its cloud platform Azure. The company also announced the appointment of a new Chief AI Officer to oversee the initiative. Microsoft shares rose 3% in after-hours trading following the announcement.',
        'source': 'CNBC',
        'published_at': datetime.now() - timedelta(hours=8),
        'category': 'corporate_earnings',
        'relevance_score': 0.82,
        'quality_score': 0.80,
        'created_at': datetime.now()
    },
    {
        'article_id': 'test_005',
        'title': 'Unemployment Rate Falls to 3.5% in Latest Jobs Report',
        'description': 'The U.S. unemployment rate dropped to 3.5% in July, with 187,000 new jobs added.',
        'content': 'The Bureau of Labor Statistics reported that the U.S. unemployment rate fell to 3.5% in July, down from 3.6% in June. The economy added 187,000 jobs, slightly below the expected 200,000. Average hourly earnings increased 0.4% month-over-month and 4.4% year-over-year. The labor force participation rate remained steady at 62.6%. Federal Reserve officials are closely watching employment data as they consider future monetary policy decisions.',
        'source': 'Reuters',
        'published_at': datetime.now() - timedelta(hours=12),
        'category': 'employment',
        'relevance_score': 0.90,
        'quality_score': 0.87,
        'created_at': datetime.now()
    }
]

async def test_event_curator_basic():
    """Test basic EventCuratorAgent functionality"""
    
    print("🧪 Testing EventCuratorAgent Basic Functionality")
    print("=" * 60)
    
    try:
        # Initialize the agent
        agent = EventCuratorAgent()
        
        # Test agent initialization
        print("1. Agent Initialization:")
        print(f"   Agent Type: {agent.agent_type}")
        print(f"   Data Handler: {type(agent.data_handler).__name__}")
        print(f"   Context Builder: {type(agent.context_builder).__name__}")
        print("   ✅ Agent initialized successfully")
        
        # Test data source requirements
        print("\n2. Data Source Requirements:")
        context = AgentContext(
            query="extract events from recent news",
            query_type="daily_briefing",
            timeframe="1d"
        )
        
        required_sources = agent.get_required_data_sources(context)
        print(f"   Required sources: {required_sources}")
        print("   ✅ Data sources identified")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Basic test failed: {e}")
        return False

async def test_event_extraction():
    """Test event extraction from sample articles"""
    
    print("\n🔍 Testing Event Extraction")
    print("=" * 60)
    
    try:
        agent = EventCuratorAgent()
        
        # Test event extraction from sample articles
        print("1. Extracting events from sample articles:")
        print(f"   Processing {len(SAMPLE_ARTICLES)} sample articles")
        
        extracted_events = await agent.context_builder.extract_events_from_articles(
            SAMPLE_ARTICLES, 
            max_events_per_article=2
        )
        
        print(f"   ✅ Extracted {len(extracted_events)} events")
        
        # Display extracted events
        print("\n2. Extracted Events Summary:")
        for i, event in enumerate(extracted_events, 1):
            print(f"   Event {i}:")
            print(f"     Description: {event.get('description', 'N/A')[:80]}...")
            print(f"     Type: {event.get('event_type', 'N/A')}")
            print(f"     Confidence: {event.get('confidence', 0.0):.2f}")
            print(f"     Source: {event.get('source', 'N/A')}")
            print(f"     Date: {event.get('date', 'N/A')}")
        
        # Test event verification
        print("\n3. Testing Event Verification:")
        verified_events = agent.context_builder.verify_events_across_sources(
            extracted_events,
            similarity_threshold=0.7
        )
        
        print(f"   ✅ Verified {len(verified_events)} unique events")
        
        # Show verification results
        verified_count = sum(1 for e in verified_events if e.get('verified', False))
        print(f"   Cross-source verified: {verified_count}")
        print(f"   Single-source events: {len(verified_events) - verified_count}")
        
        return extracted_events, verified_events
        
    except Exception as e:
        print(f"   ❌ Event extraction test failed: {e}")
        return [], []

async def test_neo4j_storage(verified_events):
    """Test Neo4j storage functionality"""
    
    print("\n💾 Testing Neo4j Storage")
    print("=" * 60)
    
    if not verified_events:
        print("   ⚠️  No verified events to store")
        return False
    
    try:
        agent = EventCuratorAgent()
        
        print(f"1. Storing {len(verified_events)} verified events in Neo4j:")
        
        stored_events = await agent._store_events_in_neo4j(verified_events)
        
        print(f"   ✅ Successfully stored {len(stored_events)} events")
        
        # Test event statistics
        print("\n2. Event Statistics:")
        stats = agent.get_event_statistics()
        
        if stats:
            print(f"   Total events in database: {stats.get('total_events', 'N/A')}")
            print(f"   Events last 7 days: {stats.get('events_last_7_days', 'N/A')}")
            print(f"   Average confidence: {stats.get('avg_confidence', 0.0):.2f}")
            
            if stats.get('event_types'):
                print("   Event types:")
                for event_type in stats['event_types'][:5]:  # Top 5
                    print(f"     {event_type.get('type', 'N/A')}: {event_type.get('count', 0)}")
        
        print("   ✅ Statistics retrieved successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Neo4j storage test failed: {e}")
        return False

async def test_full_agent_analysis():
    """Test full agent analysis workflow"""
    
    print("\n🤖 Testing Full Agent Analysis")
    print("=" * 60)
    
    try:
        agent = EventCuratorAgent()
        
        # Create analysis context
        context = AgentContext(
            query="curate events from recent financial news",
            query_type="daily_briefing",
            timeframe="1d"
        )
        
        print("1. Running full agent analysis:")
        print(f"   Query: {context.query}")
        print(f"   Type: {context.query_type}")
        print(f"   Timeframe: {context.timeframe}")
        
        # Mock the article fetching to use our sample data
        original_fetch = agent._fetch_articles_for_curation
        
        async def mock_fetch_articles(params):
            print(f"   Using {len(SAMPLE_ARTICLES)} sample articles")
            return SAMPLE_ARTICLES
        
        agent._fetch_articles_for_curation = mock_fetch_articles
        
        # Run analysis
        response = await agent.analyze(context)
        
        print(f"   ✅ Analysis completed")
        print(f"   Confidence: {response.confidence:.2f}")
        print(f"   Execution time: {response.execution_time_ms:.1f}ms")
        
        # Display results
        print("\n2. Analysis Results:")
        print(f"   Events curated: {response.key_metrics.get('total_events_curated', 0)}")
        print(f"   Extraction rate: {response.key_metrics.get('extraction_rate', 0.0):.2f}")
        print(f"   Verification rate: {response.key_metrics.get('verification_rate', 0.0):.2f}")
        
        print("\n3. Key Insights:")
        for insight in response.insights[:3]:  # Top 3 insights
            print(f"   • {insight}")
        
        print("\n4. Standardized Signals:")
        if response.standardized_signals:
            signals = response.standardized_signals
            print(f"   Market relevance: {signals.market_relevance}")
            print(f"   Credibility score: {signals.credibility_score}")
            print(f"   Overall sentiment: {signals.overall_sentiment}")
        
        # Restore original method
        agent._fetch_articles_for_curation = original_fetch
        
        return True
        
    except Exception as e:
        print(f"   ❌ Full analysis test failed: {e}")
        return False

async def test_batch_curation():
    """Test batch event curation functionality"""
    
    print("\n📦 Testing Batch Event Curation")
    print("=" * 60)
    
    try:
        agent = EventCuratorAgent()
        
        print("1. Running batch curation:")
        
        # Mock the data handler to use sample articles
        original_fetch = agent.data_handler.fetch_news_articles
        
        async def mock_fetch_news_articles(*args, **kwargs):
            return SAMPLE_ARTICLES
        
        agent.data_handler.fetch_news_articles = mock_fetch_news_articles
        
        # Run batch curation
        result = await agent.curate_events_batch(
            max_articles=10,
            days_back=1,
            categories=['federal_reserve', 'corporate_earnings']
        )
        
        print(f"   ✅ Batch curation completed")
        print(f"   Success: {result.get('success', False)}")
        print(f"   Events curated: {result.get('events_curated', 0)}")
        print(f"   Confidence: {result.get('confidence', 0.0):.2f}")
        
        if result.get('insights'):
            print("\n2. Batch Insights:")
            for insight in result['insights'][:2]:
                print(f"   • {insight}")
        
        # Restore original method
        agent.data_handler.fetch_news_articles = original_fetch
        
        return result.get('success', False)
        
    except Exception as e:
        print(f"   ❌ Batch curation test failed: {e}")
        return False

async def main():
    """Run all EventCuratorAgent tests"""
    
    print("🚀 EventCuratorAgent Test Suite")
    print("Testing event extraction, verification, and Neo4j storage")
    print("Make sure Neo4j is running: docker-compose up -d neo4j")
    print()
    
    test_results = []
    
    # Run tests
    test_results.append(await test_event_curator_basic())
    
    extracted_events, verified_events = await test_event_extraction()
    test_results.append(len(extracted_events) > 0)
    
    test_results.append(await test_neo4j_storage(verified_events))
    test_results.append(await test_full_agent_analysis())
    test_results.append(await test_batch_curation())
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 60)
    
    test_names = [
        "Basic Functionality",
        "Event Extraction", 
        "Neo4j Storage",
        "Full Agent Analysis",
        "Batch Curation"
    ]
    
    passed = sum(test_results)
    total = len(test_results)
    
    for i, (name, result) in enumerate(zip(test_names, test_results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {i+1}. {name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! EventCuratorAgent is ready for production.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the logs above.")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test suite failed with error: {e}")
        sys.exit(1)

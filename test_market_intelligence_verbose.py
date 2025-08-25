"""
Extremely verbose test script for the Market Intelligence Agent to debug data retrieval and analysis.
Shows detailed data flow, SQL queries, and agent processing steps.
"""

import asyncio
import logging
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text

# Set up extremely verbose logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def debug_database_queries():
    """Debug database queries to understand what data is available"""
    print("🔍 DEBUGGING DATABASE QUERIES")
    print("=" * 80)
    
    database_url = "postgresql://postgres:fred_password@localhost:5432/postgres"
    engine = create_engine(database_url)
    
    # Check what market indicators are available
    print("\n1. Checking available market indicators...")
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT tso.series_id, COUNT(*) as count, 
                   MIN(tso.observation_date) as earliest, 
                   MAX(tso.observation_date) as latest,
                   AVG(tso.value::numeric) as avg_value
            FROM time_series_observations tso
            JOIN data_series ds ON tso.series_id = ds.series_id
            WHERE tso.series_id IN ('SPY', 'QQQ', 'TLT', 'GLD', 'VIXCLS')
            AND tso.value IS NOT NULL
            GROUP BY tso.series_id
            ORDER BY tso.series_id
        """))
        
        indicators_data = result.fetchall()
        print(f"   Found {len(indicators_data)} indicators:")
        for row in indicators_data:
            print(f"   - {row[0]}: {row[1]:,} observations ({row[2]} to {row[3]}) avg={row[4]:.2f}")
    
    # Check recent data availability
    print("\n2. Checking recent data (last 30 days)...")
    cutoff_date = datetime.now() - timedelta(days=30)
    
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT tso.series_id, COUNT(*) as recent_count,
                   MAX(tso.observation_date) as most_recent,
                   MAX(tso.value::numeric) as latest_value
            FROM time_series_observations tso
            WHERE tso.series_id IN ('SPY', 'QQQ', 'TLT', 'GLD', 'VIXCLS')
            AND tso.observation_date >= :cutoff_date
            AND tso.value IS NOT NULL
            GROUP BY tso.series_id
            ORDER BY tso.series_id
        """), {"cutoff_date": cutoff_date})
        
        recent_data = result.fetchall()
        print(f"   Recent data (since {cutoff_date.date()}):")
        for row in recent_data:
            print(f"   - {row[0]}: {row[1]} observations, latest: {row[2]} = {row[3]:.2f}")
    
    return indicators_data, recent_data

async def test_sentiment_agent_first():
    """Test sentiment agent first to get real sentiment data"""
    print("\n🧠 TESTING SENTIMENT AGENT FIRST")
    print("=" * 80)
    
    try:
        from agents.news_sentiment import NewsSentimentAgent
        from agents.base_agent import AgentContext
        
        print("1. Initializing News Sentiment Agent...")
        sentiment_agent = NewsSentimentAgent()
        
        print("2. Creating sentiment analysis context...")
        sentiment_context = AgentContext(
            query="market volatility and economic outlook",
            timeframe="1d",
            query_type="sentiment_analysis"
        )
        
        print("3. Running sentiment analysis...")
        start_time = datetime.now()
        sentiment_response = await sentiment_agent.analyze(sentiment_context)
        execution_time = (datetime.now() - start_time).total_seconds()
        
        print(f"   ✅ Sentiment analysis completed in {execution_time:.2f} seconds")
        print(f"   Confidence: {sentiment_response.confidence:.1%}")
        print(f"   Articles analyzed: {sentiment_response.analysis.get('articles_analyzed', 0)}")
        
        # Extract detailed sentiment data
        sentiment_analysis = sentiment_response.analysis.get('sentiment_analysis', {})
        print(f"\n   📊 DETAILED SENTIMENT DATA:")
        print(f"   - Overall sentiment: {sentiment_analysis.get('overall_sentiment', 0):.3f}")
        print(f"   - Credibility score: {sentiment_analysis.get('credibility_score', 0):.3f}")
        print(f"   - Market relevance: {sentiment_analysis.get('market_relevance', 0):.3f}")
        
        emotional_tone = sentiment_analysis.get('emotional_tone', {})
        if emotional_tone:
            print(f"   - Emotional tone:")
            for emotion, score in emotional_tone.items():
                print(f"     * {emotion}: {score:.3f}")
        
        return sentiment_response
        
    except Exception as e:
        print(f"   ❌ Sentiment agent test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

async def test_market_data_handler_verbose():
    """Test market data handler with extreme verbosity"""
    print("\n💾 TESTING MARKET DATA HANDLER (VERBOSE)")
    print("=" * 80)
    
    from agents.market_intelligence.market_data_handler import MarketDataHandler
    
    print("1. Initializing MarketDataHandler...")
    data_handler = MarketDataHandler()
    print(f"   Database URL: {data_handler.database_url}")
    print(f"   Key indicators: {data_handler.key_indicators}")
    print(f"   FRED indicators: {data_handler.fred_indicators}")
    print(f"   Yahoo indicators: {data_handler.yahoo_indicators}")
    
    print("\n2. Testing database connection...")
    connection_healthy = data_handler.validate_connection()
    print(f"   Connection status: {'✅ Healthy' if connection_healthy else '❌ Failed'}")
    
    if not connection_healthy:
        return None
    
    print("\n3. Testing market data retrieval with different timeframes...")
    
    timeframes = ["1d", "1w", "1m"]
    for timeframe in timeframes:
        print(f"\n   Testing timeframe: {timeframe}")
        
        # Calculate expected date range
        hours_back = data_handler._parse_timeframe_hours(timeframe)
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        print(f"   Cutoff time: {cutoff_time}")
        
        # Test data retrieval
        market_data = await data_handler.get_market_data(timeframe=timeframe)
        print(f"   Retrieved data: {len(market_data)} rows, {len(market_data.columns) if not market_data.empty else 0} columns")
        
        if not market_data.empty:
            print(f"   Columns: {list(market_data.columns)}")
            print(f"   Date range: {market_data.index.min()} to {market_data.index.max()}")
            print(f"   Sample data (last 3 rows):")
            print(market_data.tail(3).to_string())
            
            # Test calculations
            returns = data_handler.calculate_returns(market_data)
            volatility = data_handler.calculate_volatility(market_data)
            quality = data_handler.get_data_quality_metrics(market_data)
            
            print(f"   Returns calculated: {len(returns)} rows")
            print(f"   Volatility calculated: {len(volatility)} rows")
            print(f"   Data quality: {quality}")
            
            return market_data
        else:
            print("   ⚠️ No data retrieved")
    
    return None

async def test_market_intelligence_with_real_data():
    """Test market intelligence agent with real sentiment and market data"""
    print("\n🎯 TESTING MARKET INTELLIGENCE WITH REAL DATA")
    print("=" * 80)
    
    # First get real sentiment data
    print("Step 1: Getting real sentiment data...")
    sentiment_response = await test_sentiment_agent_first()
    
    # Then test market data
    print("\nStep 2: Getting real market data...")
    market_data = await test_market_data_handler_verbose()
    
    if sentiment_response is None or market_data is None or market_data.empty:
        print("❌ Cannot proceed with integration test - missing data")
        return False
    
    print("\nStep 3: Running integrated market intelligence analysis...")
    
    from agents.market_intelligence import MarketIntelligenceAgent
    from agents.base_agent import AgentContext
    
    # Initialize agent
    agent = MarketIntelligenceAgent()
    
    # Create context with sentiment data
    context = AgentContext(
        query="What is the current market sentiment impact on volatility?",
        timeframe="1d",
        query_type="market_analysis"
    )
    
    # Add sentiment agent output to context
    context.agent_outputs = {
        'news_sentiment': sentiment_response
    }
    
    print("4. Running market intelligence analysis with real data...")
    start_time = datetime.now()
    
    # Enable debug logging for the analysis
    logging.getLogger('agents.market_intelligence').setLevel(logging.DEBUG)
    
    response = await agent.analyze(context)
    
    execution_time = (datetime.now() - start_time).total_seconds()
    
    print(f"\n🎉 ANALYSIS COMPLETED in {execution_time:.2f} seconds")
    print("=" * 80)
    
    # Display extremely detailed results
    print(f"Agent Type: {response.agent_type}")
    print(f"Confidence: {response.confidence:.1%} ({response.confidence_level})")
    print(f"Execution Time: {response.execution_time_ms:.1f}ms")
    
    print(f"\n📊 KEY METRICS:")
    for metric, value in response.key_metrics.items():
        if isinstance(value, float):
            print(f"   {metric}: {value:.4f}")
        else:
            print(f"   {metric}: {value}")
    
    print(f"\n📈 DETAILED ANALYSIS:")
    analysis = response.analysis
    
    # Market impact analysis
    if 'market_impact_analysis' in analysis:
        impact = analysis['market_impact_analysis']
        print(f"   Market Impact Analysis:")
        print(f"   - News sentiment score: {impact.get('news_sentiment_score', 0):.4f}")
        print(f"   - Market correlation: {impact.get('market_correlation', 0):.4f}")
        print(f"   - Volatility forecast: {impact.get('volatility_forecast', 0):.4f}")
        print(f"   - Confidence score: {impact.get('confidence_score', 0):.4f}")
        
        if 'price_impact_prediction' in impact:
            print(f"   - Price impact predictions:")
            for indicator, impact_val in impact['price_impact_prediction'].items():
                print(f"     * {indicator}: {impact_val:.4f}")
        
        if 'sector_impact' in impact:
            print(f"   - Sector impacts:")
            for sector, impact_val in impact['sector_impact'].items():
                print(f"     * {sector}: {impact_val:.4f}")
    
    # Correlation analysis
    if 'correlation_analysis' in analysis:
        corr = analysis['correlation_analysis']
        print(f"   Correlation Analysis:")
        print(f"   - Correlation coefficient: {corr.get('correlation_coefficient', 0):.4f}")
        print(f"   - Statistical significance: {corr.get('statistical_significance', 0):.4f}")
        print(f"   - Correlation strength: {corr.get('correlation_strength', 'unknown')}")
        print(f"   - Sample size: {corr.get('sample_size', 0)}")
    
    print(f"\n💡 AI INSIGHTS:")
    for i, insight in enumerate(response.insights, 1):
        print(f"   {i}. {insight}")
    
    print(f"\n🔄 CROSS-AGENT SIGNALS:")
    for signal, value in response.signals_for_other_agents.items():
        print(f"   {signal}: {value}")
    
    print(f"\n📋 DATA SOURCES USED:")
    for source in response.data_sources_used:
        print(f"   - {source}")
    
    return True

async def main():
    """Main comprehensive test function"""
    print("🚀 COMPREHENSIVE MARKET INTELLIGENCE TESTING")
    print("=" * 80)
    print(f"Test started at: {datetime.now()}")
    
    try:
        # Step 1: Debug database
        await debug_database_queries()
        
        # Step 2: Test with real data integration
        success = await test_market_intelligence_with_real_data()
        
        print("\n" + "=" * 80)
        print("📋 FINAL TEST SUMMARY")
        print("=" * 80)
        print(f"Overall Result: {'🎉 SUCCESS' if success else '❌ FAILED'}")
        print(f"Test completed at: {datetime.now()}")
        
        return success
        
    except Exception as e:
        print(f"\n❌ Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    asyncio.run(main())

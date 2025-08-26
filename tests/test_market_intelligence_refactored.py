"""
Test script for the refactored Market Intelligence Agent with separated concerns architecture.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_market_intelligence_agent():
    """Test the refactored market intelligence agent"""
    try:
        # Import the refactored agent
        from agents.market_intelligence.market_intelligence_agent import MarketIntelligenceAgent
        from agents.base_agent import AgentContext
        
        print("🧪 Testing Refactored Market Intelligence Agent")
        print("=" * 60)
        
        # Initialize the agent
        print("1. Initializing Market Intelligence Agent...")
        agent = MarketIntelligenceAgent()
        print("   ✅ Agent initialized successfully")
        
        # Test data handler connection
        print("\n2. Testing data handler connection...")
        connection_healthy = agent.data_handler.validate_connection()
        print(f"   Database connection: {'✅ Healthy' if connection_healthy else '❌ Failed'}")
        
        # Test market health check
        print("\n3. Testing market health check...")
        health_status = await agent.get_market_health()
        print(f"   Market data health: {'✅ Healthy' if health_status.get('connection_healthy') else '❌ Failed'}")
        if health_status.get('data_quality'):
            quality = health_status['data_quality']
            print(f"   Data points available: {quality.get('total_observations', 0)}")
            print(f"   Indicators available: {quality.get('indicators_available', 0)}")
        
        # Create test context
        print("\n4. Creating test analysis context...")
        context = AgentContext(
            query="What is the current market sentiment and volatility outlook?",
            timeframe="1d",
            query_type="market_analysis"
        )
        print("   ✅ Test context created")
        
        # Test market data retrieval
        print("\n5. Testing market data retrieval...")
        market_data = await agent.data_handler.get_market_data(timeframe="1d")
        print(f"   Market data retrieved: {len(market_data)} rows, {len(market_data.columns) if not market_data.empty else 0} indicators")
        
        if not market_data.empty:
            print(f"   Available indicators: {list(market_data.columns)}")
            print(f"   Date range: {market_data.index.min()} to {market_data.index.max()}")
        
        # Test sector analysis
        print("\n6. Testing sector analysis...")
        sector_analysis = await agent.get_sector_analysis("technology", "1d")
        if 'error' not in sector_analysis:
            print(f"   ✅ Technology sector analysis successful")
            print(f"   Symbols analyzed: {sector_analysis.get('symbols_analyzed', [])}")
        else:
            print(f"   ⚠️ Sector analysis: {sector_analysis['error']}")
        
        # Test full agent analysis
        print("\n7. Testing full market intelligence analysis...")
        start_time = datetime.now()
        
        response = await agent.analyze(context)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        print(f"   ✅ Analysis completed in {execution_time:.2f} seconds")
        print(f"   Agent type: {response.agent_type}")
        print(f"   Confidence: {response.confidence:.1%} ({response.confidence_level})")
        print(f"   Execution time: {response.execution_time_ms:.1f}ms")
        
        # Display key metrics
        if response.key_metrics:
            print(f"\n   📊 Key Metrics:")
            for metric, value in response.key_metrics.items():
                if isinstance(value, float):
                    print(f"   - {metric}: {value:.3f}")
                else:
                    print(f"   - {metric}: {value}")
        
        # Display insights
        if response.insights:
            print(f"\n   💡 AI Insights:")
            for i, insight in enumerate(response.insights[:3], 1):  # Show first 3 insights
                print(f"   {i}. {insight[:100]}{'...' if len(insight) > 100 else ''}")
        
        # Display cross-agent signals
        if response.signals_for_other_agents:
            print(f"\n   🔄 Cross-Agent Signals:")
            for signal, value in response.signals_for_other_agents.items():
                print(f"   - {signal}: {value}")
        
        # Display analysis details
        if response.analysis:
            analysis = response.analysis
            print(f"\n   📈 Analysis Summary:")
            print(f"   - News articles analyzed: {analysis.get('news_articles_analyzed', 0)}")
            print(f"   - Market data points: {analysis.get('market_data_points', 0)}")
            
            if 'correlation_analysis' in analysis:
                corr = analysis['correlation_analysis']
                print(f"   - Correlation strength: {corr.get('correlation_strength', 'unknown')}")
                print(f"   - Correlation coefficient: {corr.get('correlation_coefficient', 0):.3f}")
        
        print(f"\n🎉 Market Intelligence Agent test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_individual_components():
    """Test individual components separately"""
    try:
        print("\n" + "=" * 60)
        print("🔧 Testing Individual Components")
        print("=" * 60)
        
        # Test MarketDataHandler
        print("\n1. Testing MarketDataHandler...")
        from agents.market_intelligence.market_data_handler import MarketDataHandler
        
        data_handler = MarketDataHandler()
        connection_ok = data_handler.validate_connection()
        print(f"   Database connection: {'✅ OK' if connection_ok else '❌ Failed'}")
        
        if connection_ok:
            # Test data retrieval
            market_data = await data_handler.get_market_data(timeframe="1d")
            print(f"   Market data: {len(market_data)} rows retrieved")
            
            if not market_data.empty:
                # Test returns calculation
                returns = data_handler.calculate_returns(market_data)
                print(f"   Returns calculated: {len(returns)} rows")
                
                # Test volatility calculation
                volatility = data_handler.calculate_volatility(market_data)
                print(f"   Volatility calculated: {len(volatility)} rows")
                
                # Test data quality metrics
                quality = data_handler.get_data_quality_metrics(market_data)
                print(f"   Data quality assessment: {quality.get('indicators_available', 0)} indicators")
        
        # Test MarketContextBuilder
        print("\n2. Testing MarketContextBuilder...")
        from agents.market_intelligence.market_context_builder import MarketContextBuilder
        
        context_builder = MarketContextBuilder()
        print("   ✅ MarketContextBuilder initialized")
        
        # Create sample data for testing
        sample_news = [
            {
                'content': 'Market shows strong gains today',
                'sentiment_score': 0.5,
                'timestamp': datetime.now(),
                'source': 'test',
                'title': 'Market Update'
            }
        ]
        
        if not market_data.empty:
            # Test correlation analysis
            correlation = await context_builder.analyze_news_market_correlation(sample_news, market_data)
            print(f"   Correlation analysis: {correlation.correlation_strength} correlation")
            
            # Test impact assessment
            impact = await context_builder.assess_market_impact(sample_news, market_data, correlation)
            print(f"   Impact assessment: {impact.confidence_score:.3f} confidence")
            
            # Test signal generation
            signals = context_builder.generate_cross_agent_signals(impact, correlation)
            print(f"   Cross-agent signals: {len(signals)} signals generated")
        
        print("\n✅ Individual component tests completed")
        return True
        
    except Exception as e:
        print(f"\n❌ Component test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("🚀 Starting Market Intelligence Agent Tests")
    
    # Test the full agent
    agent_test_passed = await test_market_intelligence_agent()
    
    # Test individual components
    component_test_passed = await test_individual_components()
    
    print("\n" + "=" * 60)
    print("📋 Test Summary")
    print("=" * 60)
    print(f"Agent Integration Test: {'✅ PASSED' if agent_test_passed else '❌ FAILED'}")
    print(f"Component Unit Tests: {'✅ PASSED' if component_test_passed else '❌ FAILED'}")
    
    overall_success = agent_test_passed and component_test_passed
    print(f"\nOverall Result: {'🎉 ALL TESTS PASSED' if overall_success else '⚠️ SOME TESTS FAILED'}")
    
    return overall_success

if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Single Agent Test - Editorial Synthesis Agent
Test the refactored editorial synthesis agent in isolation to workshop its outputs
"""

import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.editorial.editorial_synthesis_agent import EditorialSynthesisAgent
from agents.base_agent import AgentContext, AgentResponse

def print_separator(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def print_subsection(title):
    print(f"\n{'-'*40}")
    print(f" {title}")
    print(f"{'-'*40}")

def create_mock_agent_outputs():
    """Create mock agent outputs to simulate orchestration context"""
    
    # Mock Economic Analysis Agent output
    class MockEconomicResponse:
        def __init__(self):
            self.insights = [
                "Economic analysis reveals moderate expansion phase with 3.2% GDP growth, driven by strong employment gains (unemployment at 3.8%) and controlled inflation (2.1% CPI). Federal Reserve policy remains accommodative with gradual rate adjustments expected."
            ]
            self.confidence = 0.75
            self.cross_agent_signals = {
                'economic_cycle': 'expansion',
                'inflation_pressure': 'moderate',
                'monetary_policy_stance': 'neutral'
            }
            self.key_metrics = {
                'gdp_growth': 3.2,
                'unemployment_rate': 3.8,
                'inflation_rate': 2.1
            }
            self.timestamp = datetime.now()
    
    # Mock Market Intelligence Agent output
    class MockMarketResponse:
        def __init__(self):
            self.insights = [
                "Market intelligence shows S&P 500 up 12.5% YTD with strong earnings growth (15.2% average) across technology and healthcare sectors. VIX at 18.5 indicates moderate volatility. Bond yields stable with 10Y Treasury at 4.2%."
            ]
            self.confidence = 0.82
            self.cross_agent_signals = {
                'market_sentiment': 'bullish',
                'volatility_regime': 'moderate',
                'sector_rotation': 'tech_healthcare'
            }
            self.key_metrics = {
                'sp500_ytd': 12.5,
                'vix_level': 18.5,
                'ten_year_yield': 4.2
            }
            self.timestamp = datetime.now()
    
    # Mock News Sentiment Agent output
    class MockSentimentResponse:
        def __init__(self):
            self.insights = [
                "News sentiment analysis indicates cautiously optimistic market narrative with 65% positive coverage. Key themes include AI innovation, infrastructure spending, and geopolitical stability. Credibility score 0.78 across 150 articles analyzed."
            ]
            self.confidence = 0.68
            self.cross_agent_signals = {
                'news_sentiment': 'positive',
                'narrative_direction': 'optimistic',
                'consensus_strength': 'moderate'
            }
            self.key_metrics = {
                'sentiment_score': 0.65,
                'credibility_score': 0.78,
                'articles_analyzed': 150
            }
            self.timestamp = datetime.now()
    
    return {
        'economic_analysis': MockEconomicResponse(),
        'market_intelligence': MockMarketResponse(),
        'news_sentiment': MockSentimentResponse()
    }

def test_editorial_synthesis_agent():
    """Test editorial synthesis agent with various scenarios"""
    
    print_separator("EDITORIAL SYNTHESIS AGENT - SINGLE AGENT TEST")
    
    # Initialize agent
    print("Initializing Editorial Synthesis Agent...")
    agent = EditorialSynthesisAgent()
    
    # Test scenarios
    test_scenarios = [
        {
            "name": "Daily Market Briefing",
            "query": "Generate today's market briefing covering economic conditions, market performance, and sentiment analysis",
            "timeframe": "1d",
            "query_type": "daily_briefing"
        },
        {
            "name": "Market Analysis Deep Dive",
            "query": "Provide comprehensive analysis of current market conditions and economic outlook",
            "timeframe": "1w",
            "query_type": "market_analysis"
        },
        {
            "name": "Economic Policy Impact",
            "query": "Analyze the impact of recent economic indicators on market sentiment and investment strategy",
            "timeframe": "1m",
            "query_type": "deep_dive"
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print_separator(f"TEST {i}: {scenario['name']}")
        
        # Create context with mock agent outputs
        context = AgentContext(
            query=scenario["query"],
            timeframe=scenario["timeframe"],
            query_type=scenario.get("query_type", "analysis")
        )
        
        # Add mock agent outputs to simulate orchestration
        context.agent_outputs = create_mock_agent_outputs()
        
        print(f"Query: {scenario['query']}")
        print(f"Timeframe: {scenario['timeframe']}")
        print(f"Query Type: {scenario.get('query_type', 'analysis')}")
        
        try:
            # Run analysis
            print("\nExecuting editorial synthesis...")
            start_time = datetime.now()
            response = agent.process(context)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            print_subsection("RESULTS")
            print(f"Execution Time: {execution_time:.2f}s")
            print(f"Agent Confidence: {response.confidence:.3f}")
            print(f"Confidence Level: {response.confidence_level}")
            print(f"Execution Time (Agent): {response.execution_time_ms:.1f}ms")
            
            print_subsection("KEY METRICS")
            for key, value in response.key_metrics.items():
                print(f"  {key}: {value}")
            
            print_subsection("EDITORIAL INSIGHTS")
            for j, insight in enumerate(response.insights, 1):
                print(f"{j}. {insight[:200]}..." if len(insight) > 200 else f"{j}. {insight}")
            
            print_subsection("ANALYSIS DETAILS")
            analysis = response.analysis
            
            if 'article_structure' in analysis:
                structure = analysis['article_structure']
                print("Article Structure:")
                print(f"  Headline: {structure.get('headline', 'N/A')}")
                print(f"  Lead Paragraph: {structure.get('lead_paragraph', 'N/A')[:100]}...")
                print(f"  Key Points: {len(structure.get('key_points', []))}")
                print(f"  Byline: {structure.get('byline', 'N/A')}")
            
            if 'data_quality' in analysis:
                quality = analysis['data_quality']
                print("\nData Quality:")
                for key, value in quality.items():
                    print(f"  {key}: {value}")
            
            print_subsection("CROSS-AGENT SIGNALS")
            if hasattr(response, 'cross_agent_signals') and response.cross_agent_signals:
                for signal, value in response.cross_agent_signals.items():
                    print(f"  {signal}: {value}")
            
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
        
        print("\n" + "="*60)

if __name__ == "__main__":
    test_editorial_synthesis_agent()

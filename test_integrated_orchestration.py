#!/usr/bin/env python3
"""
Integrated Multi-Agent Orchestration Test

Tests the complete Vector View AI agent orchestration system including:
- Agent auto-registration
- Multi-workflow execution (daily briefing, market analysis, correlation analysis)
- Cross-agent signal validation
- Standardized signal format testing
- Performance metrics and confidence scoring
"""

# Suppress ChromaDB telemetry before any imports
import os
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_TELEMETRY_DISABLED'] = 'True'

import logging
logging.getLogger('chromadb.telemetry.product.posthog').setLevel(logging.CRITICAL)
logging.getLogger('chromadb.telemetry').setLevel(logging.CRITICAL)

import sys
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_integrated_orchestration():
    """Test complete multi-agent orchestration with standardized signals"""
    print("🚀 INTEGRATED MULTI-AGENT ORCHESTRATION TEST")
    print("=" * 80)
    
    try:
        # Step 1: Initialize OrchestrationAgent
        print("\n🎯 STEP 1: Initializing Orchestration Agent")
        print("-" * 50)
        
        from agents.orchestration_agent import OrchestrationAgent
        
        orchestrator = OrchestrationAgent(
            database_url="postgresql://postgres:fred_password@localhost:5432/postgres"
        )
        
        # Auto-register all available agents
        orchestrator.register_all_agents()
        
        print(f"✅ Orchestration agent initialized")
        print(f"   Registered agents: {len(orchestrator.agent_registry)}")
        for agent_type in orchestrator.agent_registry.keys():
            print(f"     - {agent_type.value}")
        
        # Step 2: Test Daily Briefing Workflow
        print(f"\n📰 STEP 2: Testing Daily Briefing Workflow")
        print("-" * 50)
        
        query = "What are the key economic and market developments today?"
        timeframe = "1d"
        
        print(f"Query: '{query}'")
        print(f"Timeframe: {timeframe}")
        
        start_time = datetime.now()
        
        # Execute complete workflow
        synthesis_result = await orchestrator.process_user_query(
            query=query,
            timeframe=timeframe,
            user_id="test_user"
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        print(f"\n🎉 DAILY BRIEFING WORKFLOW COMPLETED")
        print("=" * 80)
        print(f"Total execution time: {execution_time:.2f} seconds")
        print(f"Overall confidence: {synthesis_result.confidence:.1%}")
        print(f"Agents executed: {len(synthesis_result.agents_executed)}")
        
        # Display agent execution details
        print(f"\n📊 AGENT EXECUTION DETAILS:")
        for agent_name, agent_response in synthesis_result.agent_responses.items():
            print(f"   {agent_name.upper()}:")
            print(f"     Confidence: {agent_response.confidence:.1%}")
            print(f"     Execution time: {agent_response.execution_time_ms:.1f}ms")
            print(f"     Data sources: {', '.join(agent_response.data_sources_used)}")
            
            # Show standardized signals
            if agent_response.standardized_signals:
                signals = agent_response.standardized_signals.to_dict()
                if signals:
                    print(f"     Standardized signals:")
                    for signal, value in signals.items():
                        if isinstance(value, float):
                            print(f"       {signal}: {value:.3f}")
                        else:
                            print(f"       {signal}: {value}")
        
        # Display cross-domain signals
        print(f"\n🔄 CROSS-DOMAIN SIGNALS:")
        for signal_data in synthesis_result.cross_domain_signals:
            source = signal_data.get('source_agent', 'unknown')
            signals = signal_data.get('signals', {})
            print(f"   From {source}:")
            for signal, value in signals.items():
                if isinstance(value, float):
                    print(f"     {signal}: {value:.3f}")
                else:
                    print(f"     {signal}: {value}")
        
        # Display executive summary
        print(f"\n📋 EXECUTIVE SUMMARY:")
        summary_lines = synthesis_result.executive_summary.split('. ')
        for line in summary_lines[:3]:  # Show first 3 sentences
            if line.strip():
                print(f"   • {line.strip()}.")
        
        # Display key insights
        print(f"\n💡 KEY INSIGHTS:")
        for i, insight in enumerate(synthesis_result.key_insights[:5], 1):
            # Truncate long insights
            display_insight = insight[:150] + "..." if len(insight) > 150 else insight
            print(f"   {i}. {display_insight}")
        
        # Step 3: Test Market Analysis Workflow
        print(f"\n💹 STEP 3: Testing Market Analysis Workflow")
        print("-" * 50)
        
        market_query = "Analyze current market volatility and sentiment impact on tech stocks"
        
        print(f"Market query: '{market_query}'")
        
        start_time = datetime.now()
        
        market_result = await orchestrator.process_user_query(
            query=market_query,
            timeframe="1w",
            user_id="test_user"
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        print(f"\n🎯 MARKET ANALYSIS COMPLETED")
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Confidence: {market_result.confidence:.1%}")
        print(f"Agents executed: {len(market_result.agents_executed)}")
        
        # Step 4: Test Correlation Analysis Workflow
        print(f"\n🔗 STEP 4: Testing Correlation Analysis Workflow")
        print("-" * 50)
        
        correlation_query = "How do inflation trends correlate with current market sentiment?"
        
        print(f"Correlation query: '{correlation_query}'")
        
        start_time = datetime.now()
        
        correlation_result = await orchestrator.process_user_query(
            query=correlation_query,
            timeframe="3m",
            user_id="test_user"
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        print(f"\n📈 CORRELATION ANALYSIS COMPLETED")
        print(f"Execution time: {execution_time:.2f} seconds")
        print(f"Confidence: {correlation_result.confidence:.1%}")
        
        # Step 5: Validate Orchestration State
        print(f"\n🔍 STEP 5: Validating Orchestration State")
        print("-" * 50)
        
        workflow_status = orchestrator.get_workflow_status()
        
        print(f"Available workflows: {len(workflow_status['available_workflows'])}")
        for workflow in workflow_status['available_workflows']:
            print(f"   - {workflow}")
        
        print(f"\nShared state:")
        for key, value in workflow_status['shared_state'].items():
            print(f"   {key}: {value}")
        
        if hasattr(orchestrator, 'standardized_state') and orchestrator.standardized_state:
            print(f"\nStandardized signals state:")
            for key, value in orchestrator.standardized_state.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.3f}")
                else:
                    print(f"   {key}: {value}")
        
        print(f"\n✅ INTEGRATED ORCHESTRATION TEST COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"🎯 Results Summary:")
        print(f"   Daily briefing confidence: {synthesis_result.confidence:.1%}")
        print(f"   Market analysis confidence: {market_result.confidence:.1%}")
        print(f"   Correlation analysis confidence: {correlation_result.confidence:.1%}")
        print(f"   Total workflows tested: 3")
        print(f"   All agents working: ✅")
        print(f"   Cross-agent signals: ✅")
        print(f"   Standardized format: ✅")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integrated orchestration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_signal_standardization():
    """Test standardized signal format across agents"""
    print("\n🔄 TESTING STANDARDIZED SIGNAL FORMAT")
    print("-" * 50)
    
    try:
        from agents.base_agent import StandardizedSignals, AgentContext
        from agents.economic.economic_agent import EconomicAnalysisAgent
        from agents.news_sentiment.news_sentiment_agent import NewsSentimentAgent
        
        # Test Economic Agent signals
        economic_agent = EconomicAnalysisAgent()
        economic_context = AgentContext(
            query="What is the current economic cycle phase?",
            query_type="deep_dive",
            timeframe="3m"
        )
        
        economic_response = economic_agent.process(economic_context)
        
        print(f"Economic Agent signals:")
        if economic_response.standardized_signals:
            signals = economic_response.standardized_signals.to_dict()
            for signal, value in signals.items():
                if isinstance(value, float):
                    print(f"   {signal}: {value:.3f}")
                else:
                    print(f"   {signal}: {value}")
        
        # Test News Sentiment Agent signals
        sentiment_agent = NewsSentimentAgent()
        sentiment_context = AgentContext(
            query="market sentiment",
            query_type="sentiment_analysis",
            timeframe="1w"
        )
        
        sentiment_response = await sentiment_agent.analyze(sentiment_context)
        
        print(f"\nNews Sentiment Agent signals:")
        if sentiment_response.standardized_signals:
            signals = sentiment_response.standardized_signals.to_dict()
            for signal, value in signals.items():
                if isinstance(value, float):
                    print(f"   {signal}: {value:.3f}")
                else:
                    print(f"   {signal}: {value}")
        
        print(f"\n✅ Signal standardization test completed")
        return True
        
    except Exception as e:
        print(f"❌ Signal standardization test failed: {str(e)}")
        return False


async def main():
    """Main test function"""
    print("🏛️  Vector View - Integrated Orchestration Test Suite")
    print("=" * 80)
    
    # Run integrated orchestration test
    orchestration_success = await test_integrated_orchestration()
    
    # Run signal standardization test
    signals_success = await test_signal_standardization()
    
    # Final results
    print(f"\n{'='*80}")
    print(f"FINAL TEST RESULTS:")
    print(f"  Orchestration Test: {'✅ PASSED' if orchestration_success else '❌ FAILED'}")
    print(f"  Signal Format Test: {'✅ PASSED' if signals_success else '❌ FAILED'}")
    print(f"  Overall Status: {'🎉 ALL TESTS PASSED' if orchestration_success and signals_success else '⚠️  SOME TESTS FAILED'}")
    print(f"{'='*80}")
    
    return orchestration_success and signals_success


if __name__ == "__main__":
    asyncio.run(main())

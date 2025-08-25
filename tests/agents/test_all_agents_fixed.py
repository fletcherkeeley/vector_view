#!/usr/bin/env python3
"""
Comprehensive test for all fixed agents
"""

import asyncio
import logging
from datetime import datetime

from agents.base_agent import AgentContext
from agents.economic_agent import EconomicAnalysisAgent
from agents.market_intelligence_agent import MarketIntelligenceAgent
from agents.news_sentiment_agent import NewsSentimentAgent
from agents.editorial_synthesis_agent import EditorialSynthesisAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AllAgentsTest:
    def __init__(self):
        self.database_url = "postgresql://postgres:fred_password@localhost:5432/postgres"
        
        # Initialize all agents
        self.agents = {
            'economic': EconomicAnalysisAgent(database_url=self.database_url),
            'market_intelligence': MarketIntelligenceAgent(database_url=self.database_url),
            'news_sentiment': NewsSentimentAgent(database_url=self.database_url),
            'editorial_synthesis': EditorialSynthesisAgent(database_url=self.database_url)
        }
    
    async def test_all_agents(self):
        """Test all agents with a common context"""
        print("🚀 Testing All Fixed Agents\n")
        
        context = AgentContext(
            query="Analyze current market conditions with focus on Federal Reserve policy impact",
            query_type="market_analysis",
            timeframe="1m"
        )
        
        results = {}
        
        for agent_name, agent in self.agents.items():
            print(f"🧪 Testing {agent_name.upper()} Agent...")
            
            try:
                start_time = datetime.now()
                response = await agent.analyze(context)
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                
                # Display results
                print(f"✅ {agent_name}: {response.confidence:.3f} confidence, {execution_time:.1f}ms")
                
                if response.insights:
                    print(f"   💡 Insights: {len(response.insights)}")
                    for i, insight in enumerate(response.insights[:2], 1):
                        preview = insight[:100].replace('\n', ' ')
                        print(f"      {i}. {preview}{'...' if len(insight) > 100 else ''}")
                
                if response.key_metrics:
                    print(f"   📊 Key Metrics:")
                    for metric, value in list(response.key_metrics.items())[:3]:
                        print(f"      {metric}: {value}")
                
                if response.signals_for_other_agents:
                    print(f"   🔄 Signals: {len(response.signals_for_other_agents)}")
                    for signal, value in list(response.signals_for_other_agents.items())[:3]:
                        print(f"      {signal}: {value}")
                
                results[agent_name] = {
                    'success': True,
                    'confidence': response.confidence,
                    'execution_time': execution_time,
                    'insights_count': len(response.insights),
                    'metrics_count': len(response.key_metrics),
                    'signals_count': len(response.signals_for_other_agents)
                }
                
            except Exception as e:
                print(f"❌ {agent_name}: FAILED - {str(e)}")
                results[agent_name] = {
                    'success': False,
                    'error': str(e)
                }
            
            print()
        
        return results
    
    def print_summary(self, results):
        """Print test summary"""
        print("="*60)
        print("📊 AGENT TEST SUMMARY")
        print("="*60)
        
        successful_agents = [name for name, result in results.items() if result.get('success')]
        failed_agents = [name for name, result in results.items() if not result.get('success')]
        
        print(f"✅ Successful Agents: {len(successful_agents)}/{len(results)}")
        print(f"❌ Failed Agents: {len(failed_agents)}/{len(results)}")
        
        if successful_agents:
            print(f"\n🎉 WORKING AGENTS:")
            for agent_name in successful_agents:
                result = results[agent_name]
                print(f"   {agent_name}: {result['confidence']:.1%} confidence, "
                      f"{result['insights_count']} insights, "
                      f"{result['execution_time']:.0f}ms")
        
        if failed_agents:
            print(f"\n💥 FAILED AGENTS:")
            for agent_name in failed_agents:
                result = results[agent_name]
                print(f"   {agent_name}: {result.get('error', 'Unknown error')}")
        
        # Overall system health
        success_rate = len(successful_agents) / len(results)
        print(f"\n🏥 SYSTEM HEALTH: {success_rate:.1%}")
        
        if success_rate >= 0.75:
            print("🟢 EXCELLENT - Ready for production")
        elif success_rate >= 0.5:
            print("🟡 GOOD - Minor issues to resolve")
        else:
            print("🔴 NEEDS WORK - Major issues present")

async def main():
    """Run comprehensive agent tests"""
    test_suite = AllAgentsTest()
    results = await test_suite.test_all_agents()
    test_suite.print_summary(results)

if __name__ == "__main__":
    asyncio.run(main())

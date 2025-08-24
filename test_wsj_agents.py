"""
Test Suite for WSJ-Level News Analysis Agents

Validates the comprehensive agent architecture designed to rival WSJ-level
financial news analysis capabilities.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

# Import all agents
from agents.base_agent import AgentContext, AgentType
from agents.orchestration_agent import OrchestrationAgent
from agents.economic_agent import EconomicAnalysisAgent
from agents.market_intelligence_agent import MarketIntelligenceAgent
from agents.news_sentiment_agent import NewsSentimentAgent
from agents.editorial_synthesis_agent import EditorialSynthesisAgent
from agents.ai_service import OllamaService

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WSJAgentTestSuite:
    """
    Comprehensive test suite for WSJ-level news analysis agents.
    
    Tests the complete pipeline from raw data ingestion through
    editorial synthesis to WSJ-quality article generation.
    """
    
    def __init__(self, db_connection=None, chroma_client=None):
        self.db_connection = db_connection
        self.chroma_client = chroma_client
        self.ai_service = OllamaService()
        
        # Database configuration from docker-compose
        self.database_url = "postgresql://postgres:fred_password@localhost:5432/postgres"
        
        # Initialize all agents with proper database URL
        self.agents = {
            'orchestration': OrchestrationAgent(self.database_url, chroma_client, self.ai_service),
            'economic': EconomicAnalysisAgent(database_url=self.database_url),
            'market_intelligence': MarketIntelligenceAgent(db_connection, chroma_client, self.ai_service),
            'news_sentiment': NewsSentimentAgent(db_connection, chroma_client, self.ai_service),
            'editorial_synthesis': EditorialSynthesisAgent(db_connection, chroma_client, self.ai_service)
        }
        
        # Test scenarios
        self.test_scenarios = [
            {
                'name': 'Daily Market Briefing',
                'query': 'Generate daily market briefing with economic and sentiment analysis',
                'query_type': 'daily_briefing',
                'timeframe': '1d',
                'expected_agents': ['economic', 'market_intelligence', 'news_sentiment', 'editorial_synthesis']
            },
            {
                'name': 'Breaking News Analysis',
                'query': 'Federal Reserve announces surprise rate decision',
                'query_type': 'deep_dive',
                'timeframe': '4h',
                'expected_agents': ['economic', 'market_intelligence', 'news_sentiment', 'editorial_synthesis']
            },
            {
                'name': 'Market Correlation Study',
                'query': 'Analyze correlation between news sentiment and market volatility',
                'query_type': 'correlation_analysis',
                'timeframe': '1w',
                'expected_agents': ['market_intelligence', 'news_sentiment', 'editorial_synthesis']
            }
        ]

    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test of all WSJ-level agents"""
        logger.info("🚀 Starting WSJ-Level Agent Architecture Test")
        
        test_results = {
            'timestamp': datetime.now(),
            'agent_tests': {},
            'scenario_tests': {},
            'performance_metrics': {},
            'wsj_quality_assessment': {}
        }
        
        try:
            # Test individual agents
            logger.info("📊 Testing Individual Agents...")
            test_results['agent_tests'] = await self._test_individual_agents()
            
            # Test integration scenarios
            logger.info("🔄 Testing Integration Scenarios...")
            test_results['scenario_tests'] = await self._test_integration_scenarios()
            
            # Assess WSJ-quality metrics
            logger.info("📰 Assessing WSJ-Quality Metrics...")
            test_results['wsj_quality_assessment'] = await self._assess_wsj_quality()
            
            # Calculate performance metrics
            test_results['performance_metrics'] = self._calculate_performance_metrics(test_results)
            
            logger.info("✅ WSJ-Level Agent Test Suite Completed Successfully")
            return test_results
            
        except Exception as e:
            logger.error(f"❌ Test suite failed: {str(e)}")
            test_results['error'] = str(e)
            return test_results

    async def _test_individual_agents(self) -> Dict[str, Any]:
        """Test each agent individually"""
        agent_results = {}
        
        test_context = AgentContext(
            query="Test market analysis with current economic conditions",
            query_type="market_analysis",
            timeframe="1d"
        )
        
        for agent_name, agent in self.agents.items():
            try:
                logger.info(f"Testing {agent_name} agent...")
                start_time = datetime.now()
                
                response = await agent.analyze(test_context)
                
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                
                agent_results[agent_name] = {
                    'status': 'success',
                    'confidence': response.confidence,
                    'execution_time_ms': execution_time,
                    'content_length': len(response.insights[0]) if response.insights else 0,
                    'signals_generated': len(response.signals_for_other_agents),
                    'data_quality': len(response.analysis) if response.analysis else 0,
                    'agent_type': response.agent_type.value
                }
                
                logger.info(f"✅ {agent_name}: {response.confidence:.2f} confidence, {execution_time:.1f}ms")
                
            except Exception as e:
                logger.error(f"❌ {agent_name} failed: {str(e)}")
                agent_results[agent_name] = {
                    'status': 'failed',
                    'error': str(e),
                    'confidence': 0.0
                }
        
        return agent_results

    async def _test_integration_scenarios(self) -> Dict[str, Any]:
        """Test integration scenarios that simulate WSJ workflows"""
        scenario_results = {}
        
        for scenario in self.test_scenarios:
            try:
                logger.info(f"Testing scenario: {scenario['name']}")
                
                context = AgentContext(
                    query=scenario['query'],
                    query_type=scenario['query_type'],
                    timeframe=scenario['timeframe']
                )
                
                # Test orchestration workflow
                start_time = datetime.now()
                orchestration_response = await self.agents['orchestration'].analyze(context)
                
                # Test editorial synthesis
                editorial_response = await self.agents['editorial_synthesis'].analyze(context)
                
                total_time = (datetime.now() - start_time).total_seconds() * 1000
                
                wsj_style_score = self._assess_wsj_style(editorial_response.insights[0] if editorial_response.insights else '')
                
                scenario_results[scenario['name']] = {
                    'status': 'success',
                    'orchestration_confidence': orchestration_response.confidence,
                    'editorial_confidence': editorial_response.confidence,
                    'total_execution_time_ms': total_time,
                    'article_length': len(editorial_response.insights[0]) if editorial_response.insights else 0,
                    'workflow_signals': len(orchestration_response.signals_for_other_agents),
                    'wsj_style_score': wsj_style_score
                }
                
                logger.info(f"✅ {scenario['name']}: Editorial confidence {editorial_response.confidence:.2f}")
                
            except Exception as e:
                logger.error(f"❌ Scenario {scenario['name']} failed: {str(e)}")
                scenario_results[scenario['name']] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return scenario_results

    async def _assess_wsj_quality(self) -> Dict[str, Any]:
        """Assess WSJ-quality metrics across the platform"""
        quality_metrics = {}
        
        try:
            # Test editorial synthesis with sample data
            context = AgentContext(
                query="Federal Reserve policy impact on technology sector amid inflation concerns",
                query_type="deep_dive",
                timeframe="1d"
            )
            
            editorial_response = await self.agents['editorial_synthesis'].analyze(context)
            
            # Assess WSJ-quality criteria
            quality_metrics = {
                'editorial_confidence': editorial_response.confidence,
                'content_comprehensiveness': self._assess_comprehensiveness(editorial_response.insights[0] if editorial_response.insights else ''),
                'factual_accuracy_score': self._assess_factual_accuracy(editorial_response.analysis),
                'readability_score': self._assess_readability(editorial_response.insights[0] if editorial_response.insights else ''),
                'market_relevance': self._assess_market_relevance(editorial_response.insights[0] if editorial_response.insights else ''),
                'timeliness_score': self._assess_timeliness(editorial_response.timestamp),
                'wsj_style_adherence': self._assess_wsj_style(editorial_response.insights[0] if editorial_response.insights else ''),
                'multi_agent_synthesis': self._assess_synthesis_quality(editorial_response.analysis)
            }
            
            # Calculate overall WSJ quality score
            quality_metrics['overall_wsj_score'] = sum([
                quality_metrics['editorial_confidence'] * 0.2,
                quality_metrics['content_comprehensiveness'] * 0.15,
                quality_metrics['factual_accuracy_score'] * 0.2,
                quality_metrics['readability_score'] * 0.1,
                quality_metrics['market_relevance'] * 0.15,
                quality_metrics['timeliness_score'] * 0.1,
                quality_metrics['wsj_style_adherence'] * 0.1
            ])
            
        except Exception as e:
            logger.error(f"WSJ quality assessment failed: {str(e)}")
            quality_metrics['error'] = str(e)
        
        return quality_metrics

    def _assess_wsj_style(self, content: str) -> float:
        """Assess adherence to WSJ editorial style"""
        if not content:
            return 0.0
        
        style_indicators = {
            'has_headline': content.startswith('#'),
            'has_lead_paragraph': len(content.split('\n\n')) >= 2,
            'appropriate_length': 500 <= len(content) <= 2000,
            'professional_tone': 'analysis' in content.lower() or 'market' in content.lower(),
            'data_driven': any(word in content.lower() for word in ['data', 'analysis', 'report', 'study']),
            'byline_present': 'Vector View' in content or 'Intelligence' in content
        }
        
        return sum(style_indicators.values()) / len(style_indicators)

    def _assess_comprehensiveness(self, content: str) -> float:
        """Assess content comprehensiveness"""
        if not content:
            return 0.0
        
        comprehensiveness_factors = {
            'market_analysis': 'market' in content.lower(),
            'economic_context': 'economic' in content.lower() or 'fed' in content.lower(),
            'sentiment_analysis': 'sentiment' in content.lower(),
            'forward_looking': any(word in content.lower() for word in ['outlook', 'forecast', 'expect', 'future']),
            'quantitative_data': any(char.isdigit() for char in content),
            'multiple_perspectives': content.count('.') >= 5  # Multiple sentences/points
        }
        
        return sum(comprehensiveness_factors.values()) / len(comprehensiveness_factors)

    def _assess_factual_accuracy(self, data: Dict[str, Any]) -> float:
        """Assess factual accuracy based on agent confidence and data quality"""
        if not data:
            return 0.5
        
        # Extract confidence metrics from agent data
        quality_metrics = data.get('quality_metrics', {})
        if isinstance(quality_metrics, dict):
            return quality_metrics.get('factual_accuracy', 0.5)
        
        return 0.7  # Default reasonable score

    def _assess_readability(self, content: str) -> float:
        """Assess readability score"""
        if not content:
            return 0.0
        
        words = content.split()
        sentences = content.count('.') + content.count('!') + content.count('?')
        
        if sentences == 0:
            return 0.0
        
        avg_words_per_sentence = len(words) / sentences
        
        # Optimal range: 15-20 words per sentence for financial journalism
        if 15 <= avg_words_per_sentence <= 20:
            return 1.0
        elif 10 <= avg_words_per_sentence <= 25:
            return 0.8
        else:
            return 0.6

    def _assess_market_relevance(self, content: str) -> float:
        """Assess market relevance of content"""
        market_keywords = [
            'market', 'stock', 'trading', 'investment', 'portfolio', 'economic',
            'federal reserve', 'inflation', 'interest rate', 'gdp', 'earnings',
            'volatility', 'sector', 'industry', 'financial', 'economy'
        ]
        
        content_lower = content.lower()
        relevance_score = sum(1 for keyword in market_keywords if keyword in content_lower)
        
        return min(1.0, relevance_score / 5)  # Normalize to 0-1

    def _assess_timeliness(self, timestamp: datetime) -> float:
        """Assess timeliness of analysis"""
        time_diff = datetime.now() - timestamp
        hours_old = time_diff.total_seconds() / 3600
        
        # Fresher analysis gets higher score
        if hours_old < 1:
            return 1.0
        elif hours_old < 6:
            return 0.8
        elif hours_old < 24:
            return 0.6
        else:
            return 0.4

    def _assess_synthesis_quality(self, data: Dict[str, Any]) -> float:
        """Assess quality of multi-agent synthesis"""
        if not data:
            return 0.0
        
        synthesis_indicators = {
            'multiple_agents': data.get('agent_insights_used', 0) >= 2,
            'quality_metrics': 'quality_metrics' in data,
            'editorial_signals': 'editorial_signals' in data,
            'article_structure': 'article_structure' in data
        }
        
        return sum(synthesis_indicators.values()) / len(synthesis_indicators)

    def _calculate_performance_metrics(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall performance metrics"""
        agent_tests = test_results.get('agent_tests', {})
        scenario_tests = test_results.get('scenario_tests', {})
        quality_assessment = test_results.get('wsj_quality_assessment', {})
        
        # Agent performance
        successful_agents = sum(1 for result in agent_tests.values() if result.get('status') == 'success')
        total_agents = len(agent_tests)
        agent_success_rate = successful_agents / total_agents if total_agents > 0 else 0
        
        # Scenario performance
        successful_scenarios = sum(1 for result in scenario_tests.values() if result.get('status') == 'success')
        total_scenarios = len(scenario_tests)
        scenario_success_rate = successful_scenarios / total_scenarios if total_scenarios > 0 else 0
        
        # Average confidence
        confidences = [result.get('confidence', 0) for result in agent_tests.values() if result.get('confidence')]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # WSJ quality score
        wsj_score = quality_assessment.get('overall_wsj_score', 0)
        
        return {
            'agent_success_rate': agent_success_rate,
            'scenario_success_rate': scenario_success_rate,
            'average_confidence': avg_confidence,
            'wsj_quality_score': wsj_score,
            'total_agents_tested': total_agents,
            'total_scenarios_tested': total_scenarios,
            'overall_system_health': (agent_success_rate + scenario_success_rate + wsj_score) / 3,
            'wsj_readiness': wsj_score >= 0.75  # 75% threshold for WSJ-level quality
        }

    def print_test_summary(self, results: Dict[str, Any]):
        """Print comprehensive test summary"""
        print("\n" + "="*80)
        print("🏆 WSJ-LEVEL NEWS ANALYSIS AGENT TEST RESULTS")
        print("="*80)
        
        # Performance Overview
        metrics = results.get('performance_metrics', {})
        print(f"\n📊 PERFORMANCE OVERVIEW:")
        print(f"   Agent Success Rate: {metrics.get('agent_success_rate', 0):.1%}")
        print(f"   Scenario Success Rate: {metrics.get('scenario_success_rate', 0):.1%}")
        print(f"   Average Confidence: {metrics.get('average_confidence', 0):.2f}")
        print(f"   WSJ Quality Score: {metrics.get('wsj_quality_score', 0):.2f}")
        print(f"   Overall System Health: {metrics.get('overall_system_health', 0):.1%}")
        
        # WSJ Readiness Assessment
        wsj_ready = metrics.get('wsj_readiness', False)
        status_emoji = "✅" if wsj_ready else "⚠️"
        print(f"\n{status_emoji} WSJ READINESS: {'READY FOR PRODUCTION' if wsj_ready else 'NEEDS IMPROVEMENT'}")
        
        # Individual Agent Results
        print(f"\n🤖 INDIVIDUAL AGENT PERFORMANCE:")
        agent_tests = results.get('agent_tests', {})
        for agent_name, result in agent_tests.items():
            status = "✅" if result.get('status') == 'success' else "❌"
            confidence = result.get('confidence', 0)
            exec_time = result.get('execution_time_ms', 0)
            print(f"   {status} {agent_name.title()}: {confidence:.2f} confidence, {exec_time:.1f}ms")
        
        # Scenario Results
        print(f"\n📰 SCENARIO TEST RESULTS:")
        scenario_tests = results.get('scenario_tests', {})
        for scenario_name, result in scenario_tests.items():
            status = "✅" if result.get('status') == 'success' else "❌"
            editorial_conf = result.get('editorial_confidence', 0)
            wsj_style = result.get('wsj_style_score', 0)
            print(f"   {status} {scenario_name}: {editorial_conf:.2f} editorial, {wsj_style:.2f} WSJ style")
        
        # WSJ Quality Breakdown
        print(f"\n📊 WSJ QUALITY ASSESSMENT:")
        quality = results.get('wsj_quality_assessment', {})
        quality_metrics = [
            ('Editorial Confidence', 'editorial_confidence'),
            ('Content Comprehensiveness', 'content_comprehensiveness'),
            ('Factual Accuracy', 'factual_accuracy_score'),
            ('Readability', 'readability_score'),
            ('Market Relevance', 'market_relevance'),
            ('WSJ Style Adherence', 'wsj_style_adherence')
        ]
        
        for metric_name, metric_key in quality_metrics:
            score = quality.get(metric_key, 0)
            print(f"   {metric_name}: {score:.2f}")
        
        print("\n" + "="*80)


async def main():
    """Main test execution function"""
    print("🚀 Initializing WSJ-Level Agent Test Suite...")
    
    # Initialize test suite (would connect to actual DB/ChromaDB in production)
    test_suite = WSJAgentTestSuite()
    
    # Run comprehensive tests
    results = await test_suite.run_comprehensive_test()
    
    # Print detailed results
    test_suite.print_test_summary(results)
    
    return results


if __name__ == "__main__":
    # Run the test suite
    asyncio.run(main())

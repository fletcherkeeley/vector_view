"""
Vector View AI Agents Package

This package contains the AI agent architecture for Vector View financial intelligence platform.
Agents work together to analyze economic data, market movements, and news sentiment.
"""

from .base_agent import BaseAgent, AgentResponse, AgentContext
from .news_sentiment.news_sentiment_agent import NewsSentimentAgent
from .market_intelligence import MarketIntelligenceAgent
from .economic.economic_agent import EconomicAnalysisAgent

__all__ = [
    'BaseAgent',
    'AgentResponse', 
    'AgentContext',
    'NewsSentimentAgent',
    'MarketIntelligenceAgent',
    'EconomicAnalysisAgent',
    'OrchestrationAgent'
]

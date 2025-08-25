"""
Market Intelligence Agent Module for Vector View Financial Intelligence Platform

Provides market impact analysis by cross-referencing news sentiment with market data.
Specializes in real-time market reaction predictions and volatility forecasting.
"""

from .market_intelligence_agent import MarketIntelligenceAgent
from .market_data_handler import MarketDataHandler
from .market_context_builder import MarketContextBuilder

__all__ = [
    'MarketIntelligenceAgent',
    'MarketDataHandler', 
    'MarketContextBuilder'
]

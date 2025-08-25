"""
Economic Analysis Module - Frequency-aware economic data analysis

This module provides specialized economic analysis capabilities with proper
handling of different data frequencies (monthly, weekly, daily).

Components:
- EconomicAnalysisAgent: Main agent for economic analysis
- EconomicDataHandler: Data fetching with frequency awareness
- EconomicIndicators: Trend analysis and cycle detection
- EconomicContextBuilder: AI context generation
"""

from .economic_agent import EconomicAnalysisAgent
from .economic_data_handler import EconomicDataHandler
from .economic_indicators import EconomicIndicators
from .economic_context_builder import EconomicContextBuilder

__all__ = [
    'EconomicAnalysisAgent',
    'EconomicDataHandler', 
    'EconomicIndicators',
    'EconomicContextBuilder'
]

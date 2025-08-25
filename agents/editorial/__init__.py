"""
Editorial Synthesis Module - WSJ-style financial journalism and content synthesis

This module provides specialized editorial synthesis capabilities for generating
high-quality financial journalism and analysis.

Components:
- EditorialSynthesisAgent: Main agent for editorial content generation
- EditorialDataHandler: Agent insights collection and data processing
- EditorialIndicators: Article structure analysis and synthesis logic
- EditorialContextBuilder: AI context generation and prompt construction
"""

from .editorial_synthesis_agent import EditorialSynthesisAgent
from .editorial_data_handler import EditorialDataHandler
from .editorial_indicators import EditorialIndicators, ArticleStructure
from .editorial_context_builder import EditorialContextBuilder

__all__ = [
    'EditorialSynthesisAgent',
    'EditorialDataHandler',
    'EditorialIndicators',
    'ArticleStructure',
    'EditorialContextBuilder'
]

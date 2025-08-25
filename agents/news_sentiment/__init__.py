"""
News Sentiment Agent Package

Provides news sentiment analysis, entity extraction, and narrative tracking
for the Vector View Financial Intelligence Platform.
"""

from .news_sentiment_agent import NewsSentimentAgent
from .news_sentiment_data_handler import NewsSentimentDataHandler
from .news_sentiment_context_builder import NewsSentimentContextBuilder

__all__ = [
    'NewsSentimentAgent',
    'NewsSentimentDataHandler', 
    'NewsSentimentContextBuilder'
]

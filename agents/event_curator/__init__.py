"""
Event Curator Agent Module for Vector View Financial Intelligence Platform

Extracts, verifies, and curates structured events from news articles,
storing them in Neo4j for relationship modeling and analysis.
"""

from .event_curator_agent import EventCuratorAgent
from .event_data_handler import EventDataHandler
from .event_context_builder import EventContextBuilder

__all__ = [
    'EventCuratorAgent',
    'EventDataHandler', 
    'EventContextBuilder'
]

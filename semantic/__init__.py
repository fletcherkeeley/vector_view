"""
Semantic Search Module for Vector View Financial Intelligence Platform

This module provides semantic search capabilities for financial news articles
and economic indicators using ChromaDB vector database.

Components:
- vector_store: Core ChromaDB client and collection management
- embedding_pipeline: Text-to-vector conversion for news and economic data
- search_interface: Agent-friendly search API
- feedback_tracker: Analysis feedback loop for continuous improvement
"""

from .vector_store import SemanticVectorStore
from .embedding_pipeline import EmbeddingPipeline
from .search_interface import AgentSearchInterface

__all__ = [
    'SemanticVectorStore',
    'EmbeddingPipeline', 
    'AgentSearchInterface'
]

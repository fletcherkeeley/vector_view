"""
Semantic Vector Store for Financial Intelligence Platform

Core ChromaDB client managing multiple collections for different data types:
- News articles with economic categorization
- Economic indicator descriptions and metadata
- Future analysis results and feedback loops

Designed for AI agent queries with comprehensive metadata support.
"""

import asyncio
import logging
import hashlib
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timezone
from pathlib import Path
import os
from enum import Enum

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)


class CollectionType(Enum):
    """Types of vector collections in our semantic store"""
    NEWS_ARTICLES = "financial_news"
    ECONOMIC_INDICATORS = "economic_indicators"
    ANALYSIS_RESULTS = "analysis_results"
    AGENT_FEEDBACK = "agent_feedback"


class SemanticConfig:
    """Configuration for semantic vector store"""
    
    def __init__(self):
        self.chroma_persist_directory = os.getenv('CHROMA_PERSIST_DIR', './chroma_db')
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.batch_size = int(os.getenv('VECTOR_BATCH_SIZE', '100'))
        self.max_chunk_size = int(os.getenv('MAX_CHUNK_SIZE', '512'))
        
        # Ensure persist directory exists
        Path(self.chroma_persist_directory).mkdir(parents=True, exist_ok=True)


class SemanticVectorStore:
    """
    Multi-collection vector store for financial semantic search
    """
    
    def __init__(self, config: SemanticConfig = None):
        self.config = config or SemanticConfig()
        self.chroma_client = None
        self.embedding_model = None
        self.embedding_function = None
        self.collections = {}
        
    async def initialize(self) -> bool:
        """Initialize ChromaDB client and all collections"""
        try:
            # Initialize ChromaDB with persistent storage
            self.chroma_client = chromadb.PersistentClient(
                path=self.config.chroma_persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Initialize embedding model
            logger.info(f"Loading embedding model: {self.config.embedding_model}")
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
            
            # Create custom embedding function
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.config.embedding_model
            )
            
            # Initialize all collections
            await self._initialize_collections()
            
            logger.info("Semantic vector store initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize semantic vector store: {e}")
            return False
    
    async def _initialize_collections(self):
        """Initialize all required collections"""
        collection_configs = {
            CollectionType.NEWS_ARTICLES: {
                "description": "Financial news articles with economic categorization",
                "schema_info": "article_id,source_name,published_at,economic_categories,sentiment_score,relevance_score,related_series,impact_timeframe"
            },
            CollectionType.ECONOMIC_INDICATORS: {
                "description": "Economic indicator descriptions and metadata",
                "schema_info": "series_id,source_type,category,frequency,units,seasonal_adjustment,correlation_priority"
            },
            CollectionType.ANALYSIS_RESULTS: {
                "description": "AI agent analysis results for feedback loops",
                "schema_info": "analysis_id,agent_type,analysis_date,confidence_score,related_articles,related_series,analysis_type"
            },
            CollectionType.AGENT_FEEDBACK: {
                "description": "Agent feedback and learning data",
                "schema_info": "feedback_id,agent_id,query_type,search_query,result_relevance,feedback_date"
            }
        }
        
        for collection_type, config in collection_configs.items():
            try:
                collection = self.chroma_client.get_collection(
                    name=collection_type.value,
                    embedding_function=self.embedding_function
                )
                logger.info(f"Connected to existing collection: {collection_type.value}")
            except Exception:
                collection = self.chroma_client.create_collection(
                    name=collection_type.value,
                    embedding_function=self.embedding_function,
                    metadata={
                        "description": config["description"],
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "schema_info": config["schema_info"]
                    }
                )
                logger.info(f"Created new collection: {collection_type.value}")
            
            self.collections[collection_type] = collection
    
    async def add_news_article(
        self, 
        article_id: str,
        title: str,
        content: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Add a news article to the vector store"""
        try:
            # Prepare document text
            text_parts = []
            if title:
                text_parts.append(f"Title: {title}")
            if content:
                # Truncate content if too long
                content_truncated = content[:self.config.max_chunk_size * 3]
                text_parts.append(f"Content: {content_truncated}")
            
            document_text = " ".join(text_parts)
            doc_id = f"article_{article_id}"
            
            # Add to collection
            self.collections[CollectionType.NEWS_ARTICLES].add(
                documents=[document_text],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            logger.debug(f"Added news article {article_id} to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add news article {article_id}: {e}")
            return False
    
    async def add_economic_indicator(
        self,
        series_id: str,
        title: str,
        description: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Add an economic indicator to the vector store"""
        try:
            # Prepare document text
            text_parts = []
            if title:
                text_parts.append(f"Indicator: {title}")
            if description:
                text_parts.append(f"Description: {description}")
            if metadata.get('units'):
                text_parts.append(f"Units: {metadata['units']}")
            if metadata.get('category'):
                text_parts.append(f"Category: {metadata['category']}")
            
            document_text = " ".join(text_parts)
            doc_id = f"series_{series_id}"
            
            # Add to collection
            self.collections[CollectionType.ECONOMIC_INDICATORS].add(
                documents=[document_text],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            logger.debug(f"Added economic indicator {series_id} to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add economic indicator {series_id}: {e}")
            return False
    
    async def add_analysis_result(
        self,
        analysis_id: str,
        analysis_text: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """Add AI agent analysis result for future reference"""
        try:
            doc_id = f"analysis_{analysis_id}"
            
            self.collections[CollectionType.ANALYSIS_RESULTS].add(
                documents=[analysis_text],
                metadatas=[metadata],
                ids=[doc_id]
            )
            
            logger.debug(f"Added analysis result {analysis_id} to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add analysis result {analysis_id}: {e}")
            return False
    
    async def search_news_articles(
        self,
        query: str,
        n_results: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search news articles with optional filters"""
        return await self._search_collection(
            CollectionType.NEWS_ARTICLES,
            query,
            n_results,
            filters
        )
    
    async def search_economic_indicators(
        self,
        query: str,
        n_results: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search economic indicators"""
        return await self._search_collection(
            CollectionType.ECONOMIC_INDICATORS,
            query,
            n_results,
            filters
        )
    
    async def search_analysis_results(
        self,
        query: str,
        n_results: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search previous analysis results"""
        return await self._search_collection(
            CollectionType.ANALYSIS_RESULTS,
            query,
            n_results,
            filters
        )
    
    async def _search_collection(
        self,
        collection_type: CollectionType,
        query: str,
        n_results: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Generic search method for any collection"""
        try:
            collection = self.collections[collection_type]
            
            # Perform search
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filters
            )
            
            # Process results
            search_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else 0
                    
                    search_results.append({
                        'document': doc,
                        'metadata': metadata,
                        'similarity_score': 1 - distance,  # Convert distance to similarity
                        'collection_type': collection_type.value
                    })
            
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed in {collection_type.value}: {e}")
            return []
    
    async def find_related_content(
        self,
        query: str,
        search_news: bool = True,
        search_indicators: bool = True,
        search_analysis: bool = False,
        n_results_per_type: int = 5
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Search across multiple collections for related content"""
        results = {}
        
        if search_news:
            results['news'] = await self.search_news_articles(query, n_results_per_type)
        
        if search_indicators:
            results['indicators'] = await self.search_economic_indicators(query, n_results_per_type)
        
        if search_analysis:
            results['analysis'] = await self.search_analysis_results(query, n_results_per_type)
        
        return results
    
    async def record_agent_feedback(
        self,
        agent_id: str,
        query: str,
        search_results: List[Dict[str, Any]],
        relevance_scores: List[float]
    ) -> bool:
        """Record agent feedback for continuous improvement"""
        try:
            feedback_id = hashlib.md5(f"{agent_id}_{query}_{datetime.now().isoformat()}".encode()).hexdigest()
            
            feedback_text = f"Query: {query}\nResults: {len(search_results)} items\nAvg Relevance: {np.mean(relevance_scores):.3f}"
            
            metadata = {
                "feedback_id": feedback_id,
                "agent_id": agent_id,
                "query_type": "semantic_search",
                "search_query": query,
                "result_relevance": float(np.mean(relevance_scores)),
                "feedback_date": datetime.now(timezone.utc).isoformat()
            }
            
            self.collections[CollectionType.AGENT_FEEDBACK].add(
                documents=[feedback_text],
                metadatas=[metadata],
                ids=[f"feedback_{feedback_id}"]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to record agent feedback: {e}")
            return False
    
    async def get_collection_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all collections"""
        stats = {}
        
        for collection_type, collection in self.collections.items():
            try:
                count = collection.count()
                stats[collection_type.value] = {
                    'document_count': count,
                    'embedding_model': self.config.embedding_model
                }
            except Exception as e:
                stats[collection_type.value] = {'error': str(e)}
        
        return stats
    
    async def reset_collection(self, collection_type: CollectionType) -> bool:
        """Reset a specific collection (use with caution)"""
        try:
            self.chroma_client.delete_collection(collection_type.value)
            # Reinitialize the collection
            await self._initialize_collections()
            logger.warning(f"Reset collection: {collection_type.value}")
            return True
        except Exception as e:
            logger.error(f"Failed to reset collection {collection_type.value}: {e}")
            return False


async def create_semantic_store() -> SemanticVectorStore:
    """Factory function to create and initialize semantic vector store"""
    config = SemanticConfig()
    store = SemanticVectorStore(config)
    
    if await store.initialize():
        return store
    else:
        raise RuntimeError("Failed to initialize semantic vector store")


if __name__ == "__main__":
    # Test the semantic vector store
    async def test_semantic_store():
        try:
            store = await create_semantic_store()
            stats = await store.get_collection_stats()
            print(f"Semantic Store Stats: {json.dumps(stats, indent=2)}")
        except Exception as e:
            print(f"Test failed: {e}")
    
    asyncio.run(test_semantic_store())

"""
News Sentiment Data Handler for Vector View Financial Intelligence Platform

Handles all data access operations for news sentiment analysis including:
- ChromaDB news article retrieval
- Semantic search operations
- News article filtering and preprocessing
- Database connection management
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Suppress ChromaDB telemetry before import
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_TELEMETRY_DISABLED'] = 'True'
logging.getLogger('chromadb.telemetry.product.posthog').setLevel(logging.CRITICAL)
logging.getLogger('chromadb.telemetry').setLevel(logging.CRITICAL)

import chromadb
from chromadb.config import Settings

logger = logging.getLogger(__name__)


class NewsSentimentDataHandler:
    """
    Handles all data access operations for news sentiment analysis.
    
    Responsibilities:
    - ChromaDB connection and collection management
    - News article retrieval and filtering
    - Semantic search operations
    - Data preprocessing and validation
    """
    
    def __init__(self, chroma_client=None, chroma_path: str = "./chroma_db"):
        """
        Initialize the data handler.
        
        Args:
            chroma_client: Optional existing ChromaDB client
            chroma_path: Path to ChromaDB persistence directory
        """
        self.chroma_client = chroma_client
        self.chroma_path = chroma_path
        self._collection = None
        
    def _get_chroma_client(self):
        """Get or create ChromaDB client"""
        if not self.chroma_client:
            try:
                # Create client with telemetry disabled
                settings = Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
                self.chroma_client = chromadb.PersistentClient(
                    path=self.chroma_path,
                    settings=settings
                )
            except Exception as e:
                logger.error(f"Failed to initialize ChromaDB client: {str(e)}")
                raise
        return self.chroma_client
    
    def _get_collection(self, collection_name: str = "financial_news"):
        """Get ChromaDB collection"""
        try:
            client = self._get_chroma_client()
            return client.get_collection(collection_name)
        except Exception as e:
            logger.warning(f"ChromaDB collection '{collection_name}' not found: {str(e)}")
            return None
    
    async def get_news_articles(
        self, 
        query: str = None, 
        timeframe: str = "1d", 
        max_results: int = 20,
        min_relevance: float = 0.0
    ) -> List[Dict]:
        """
        Retrieve news articles from ChromaDB.
        
        Args:
            query: Search query for semantic search
            timeframe: Time window for articles (e.g., "1h", "1d", "1w")
            max_results: Maximum number of articles to return
            min_relevance: Minimum relevance score threshold
            
        Returns:
            List of news articles with metadata
        """
        try:
            collection = self._get_collection()
            if not collection:
                logger.warning("No news collection available")
                return []
            
            # Perform semantic search with broader query
            search_query = query if query else "financial news market economy"
            results = collection.query(
                query_texts=[search_query],
                n_results=min(max_results * 3, 1000)  # Get more results to filter from
            )
            
            articles = []
            if results and results.get('documents') and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results.get('metadatas') else {}
                    distance = results['distances'][0][i] if results.get('distances') else 0.0
                    relevance_score = 1.0 - distance
                    
                    # Filter by relevance threshold - use more permissive filtering
                    # ChromaDB distances can be > 1.0, creating negative relevance scores
                    # Accept articles with distance < 2.0 (relevance > -1.0) for broader coverage
                    if distance > 2.0:  # Skip only very irrelevant articles
                        continue
                    
                    # Extract and validate metadata
                    sentiment_score = metadata.get('sentiment_score', 0.0)
                    try:
                        sentiment_score = float(sentiment_score) if sentiment_score else 0.0
                    except (ValueError, TypeError):
                        sentiment_score = 0.0
                    
                    article = {
                        'content': doc,
                        'title': metadata.get('title', ''),
                        'source': metadata.get('source_name', 'unknown'),
                        'timestamp': metadata.get('published_at'),
                        'url': metadata.get('url', ''),
                        'sentiment_score': sentiment_score,
                        'relevance_score': relevance_score,
                        'quality_score': metadata.get('quality_score', 0.5)
                    }
                    
                    articles.append(article)
            
            # Sort by relevance score and limit to requested amount
            articles.sort(key=lambda x: x['relevance_score'], reverse=True)
            articles = articles[:max_results]
            
            logger.info(f"Retrieved {len(articles)} news articles for query: '{search_query}' (from {len(results['documents'][0])} candidates)")
            return articles
            
        except Exception as e:
            logger.error(f"Failed to retrieve news articles: {str(e)}")
            return []
    
    async def get_articles_by_timeframe(
        self, 
        timeframe: str = "1d",
        max_results: int = 50
    ) -> List[Dict]:
        """
        Get articles within a specific timeframe.
        
        Args:
            timeframe: Time window (e.g., "1h", "1d", "1w")
            max_results: Maximum number of articles
            
        Returns:
            List of recent news articles
        """
        try:
            hours_back = self._parse_timeframe_hours(timeframe)
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            collection = self._get_collection()
            if not collection:
                return []
            
            # Get all articles and filter by timestamp
            # Note: ChromaDB doesn't support timestamp filtering directly,
            # so we retrieve more articles and filter client-side
            results = collection.query(
                query_texts=["financial news"],
                n_results=max_results * 2  # Get more to account for filtering
            )
            
            filtered_articles = []
            if results and results.get('documents') and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results.get('metadatas') else {}
                    
                    # Check timestamp if available
                    published_at = metadata.get('published_at')
                    if published_at:
                        try:
                            article_time = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                            if article_time < cutoff_time:
                                continue
                        except (ValueError, AttributeError):
                            # If timestamp parsing fails, include the article
                            pass
                    
                    article = {
                        'content': doc,
                        'title': metadata.get('title', ''),
                        'source': metadata.get('source_name', 'unknown'),
                        'timestamp': published_at,
                        'url': metadata.get('url', ''),
                        'sentiment_score': float(metadata.get('sentiment_score', 0.0)),
                        'relevance_score': 1.0 - (results['distances'][0][i] if results.get('distances') else 0.0)
                    }
                    
                    filtered_articles.append(article)
                    
                    if len(filtered_articles) >= max_results:
                        break
            
            return filtered_articles
            
        except Exception as e:
            logger.error(f"Failed to get articles by timeframe: {str(e)}")
            return []
    
    async def search_articles_by_entities(
        self, 
        entities: List[str], 
        max_results: int = 20
    ) -> List[Dict]:
        """
        Search for articles mentioning specific entities.
        
        Args:
            entities: List of entity names to search for
            max_results: Maximum number of articles to return
            
        Returns:
            List of articles mentioning the entities
        """
        try:
            if not entities:
                return []
            
            collection = self._get_collection()
            if not collection:
                return []
            
            # Create search query from entities
            entity_query = " OR ".join(entities)
            
            results = collection.query(
                query_texts=[entity_query],
                n_results=max_results
            )
            
            articles = []
            if results and results.get('documents') and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results.get('metadatas') else {}
                    
                    # Check if any entities are mentioned in the content
                    content_lower = doc.lower()
                    title_lower = metadata.get('title', '').lower()
                    
                    mentioned_entities = []
                    for entity in entities:
                        if entity.lower() in content_lower or entity.lower() in title_lower:
                            mentioned_entities.append(entity)
                    
                    if mentioned_entities:  # Only include if entities are actually mentioned
                        article = {
                            'content': doc,
                            'title': metadata.get('title', ''),
                            'source': metadata.get('source_name', 'unknown'),
                            'timestamp': metadata.get('published_at'),
                            'url': metadata.get('url', ''),
                            'sentiment_score': float(metadata.get('sentiment_score', 0.0)),
                            'relevance_score': 1.0 - (results['distances'][0][i] if results.get('distances') else 0.0),
                            'mentioned_entities': mentioned_entities
                        }
                        articles.append(article)
            
            return articles
            
        except Exception as e:
            logger.error(f"Failed to search articles by entities: {str(e)}")
            return []
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the news collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            collection = self._get_collection()
            if not collection:
                return {'error': 'Collection not available'}
            
            # Get basic collection info
            count = collection.count()
            
            # Sample some articles to get metadata stats
            sample_results = collection.query(
                query_texts=["financial"],
                n_results=min(100, count)
            )
            
            sources = set()
            sentiment_scores = []
            
            if sample_results and sample_results.get('metadatas'):
                for metadata in sample_results['metadatas'][0]:
                    if metadata.get('source_name'):
                        sources.add(metadata['source_name'])
                    if metadata.get('sentiment_score'):
                        try:
                            sentiment_scores.append(float(metadata['sentiment_score']))
                        except (ValueError, TypeError):
                            pass
            
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
            
            return {
                'total_articles': count,
                'unique_sources': len(sources),
                'sources': list(sources),
                'average_sentiment': avg_sentiment,
                'sentiment_range': [min(sentiment_scores), max(sentiment_scores)] if sentiment_scores else [0, 0]
            }
            
        except Exception as e:
            logger.error(f"Failed to get collection stats: {str(e)}")
            return {'error': str(e)}
    
    def _parse_timeframe_hours(self, timeframe: str) -> int:
        """Convert timeframe string to hours"""
        timeframe_map = {
            '1h': 1, '4h': 4, '1d': 24, '1w': 168, 
            '1m': 720, '3m': 2160, '1y': 8760
        }
        return timeframe_map.get(timeframe, 24)
    
    def validate_connection(self) -> bool:
        """
        Validate ChromaDB connection and collection availability.
        
        Returns:
            True if connection is valid, False otherwise
        """
        try:
            client = self._get_chroma_client()
            collection = self._get_collection()
            return collection is not None
        except Exception as e:
            logger.error(f"Connection validation failed: {str(e)}")
            return False

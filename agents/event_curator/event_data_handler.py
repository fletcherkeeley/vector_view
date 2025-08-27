"""
Event Data Handler for Event Curator Agent

Handles data access operations for news articles from ChromaDB and PostgreSQL,
and manages event storage in Neo4j through the GraphService.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import hashlib
import json

# Suppress ChromaDB telemetry before import
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_TELEMETRY_DISABLED'] = 'True'
logging.getLogger('chromadb.telemetry.product.posthog').setLevel(logging.CRITICAL)
logging.getLogger('chromadb.telemetry').setLevel(logging.CRITICAL)

import chromadb
from chromadb.config import Settings
import asyncpg
import asyncio
from sentence_transformers import SentenceTransformer
import numpy as np

from database.graph_service import GraphService

logger = logging.getLogger(__name__)


class EventDataHandler:
    """
    Handles all data access operations for the Event Curator Agent.
    
    Responsibilities:
    - Fetch news articles from ChromaDB and PostgreSQL
    - Store extracted events in Neo4j
    - Query existing events for deduplication
    - Manage event relationships and metadata
    """
    
    def __init__(
        self,
        database_url: str = "postgresql://postgres:fred_password@localhost:5432/postgres",
        chroma_path: str = "./chroma_db",
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_username: str = "neo4j",
        neo4j_password: str = "vector_view_password"
    ):
        self.database_url = database_url
        self.chroma_path = chroma_path
        
        # Initialize GraphService for Neo4j operations
        self.graph_service = GraphService(
            uri=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password
        )
        
        # Initialize ChromaDB client (following existing patterns)
        self.chroma_client = None
        self.chroma_available = False
        self._news_collection = None
        self.embedding_model = None
        
        # Initialize ChromaDB connection
        self._initialize_chromadb()
    
    async def fetch_news_articles(
        self,
        limit: int = 100,
        days_back: int = 7,
        categories: Optional[List[str]] = None,
        min_relevance: float = 0.01
    ) -> List[Dict[str, Any]]:
        """
        Fetch news articles from PostgreSQL with optional filtering.
        
        Args:
            limit: Maximum number of articles to fetch
            days_back: Number of days to look back
            categories: Optional list of categories to filter by
            min_relevance: Minimum relevance score
            
        Returns:
            List of news article dictionaries
        """
        try:
            conn = await asyncpg.connect(self.database_url)
            
            # Build query with optional filters
            base_query = """
                SELECT 
                    id,
                    source_article_id,
                    title,
                    description,
                    content,
                    url,
                    published_at,
                    source_name,
                    economic_categories,
                    relevance_score,
                    data_quality_score,
                    created_at
                FROM news_articles 
                WHERE published_at >= $1
                AND relevance_score >= $2
            """
            
            params = [
                datetime.now() - timedelta(days=days_back),
                min_relevance
            ]
            
            if categories:
                category_placeholders = ','.join(f'${i+3}' for i in range(len(categories)))
                base_query += f" AND economic_categories ?| ARRAY[{category_placeholders}]"
                params.extend(categories)
            
            base_query += " ORDER BY published_at DESC, relevance_score DESC LIMIT $" + str(len(params) + 1)
            params.append(limit)
            
            rows = await conn.fetch(base_query, *params)
            
            articles = []
            for row in rows:
                articles.append({
                    'article_id': str(row['id']),  # Use UUID as string
                    'source_article_id': row['source_article_id'],
                    'title': row['title'],
                    'description': row['description'],
                    'content': row['content'],
                    'url': row['url'],
                    'published_at': row['published_at'],
                    'source': row['source_name'],
                    'category': row['economic_categories'],
                    'relevance_score': float(row['relevance_score']) if row['relevance_score'] else 0.0,
                    'quality_score': float(row['data_quality_score']) if row['data_quality_score'] else 0.5,
                    'created_at': row['created_at']
                })
            
            await conn.close()
            logger.info(f"Fetched {len(articles)} news articles from PostgreSQL")
            return articles
            
        except Exception as e:
            logger.error(f"Error fetching news articles: {e}")
            return []
    
    async def fetch_articles_by_semantic_search(
        self,
        query: str,
        limit: int = 50,
        collection_name: str = "financial_news"
    ) -> List[Dict[str, Any]]:
        """
        Fetch articles using semantic search from ChromaDB.
        
        Args:
            query: Search query for semantic matching
            limit: Maximum number of results
            collection_name: ChromaDB collection name
            
        Returns:
            List of articles with similarity scores
        """
        if not self.chroma_client:
            logger.warning("ChromaDB client not available")
            return []
        
        try:
            collection = self.chroma_client.get_collection(collection_name)
            
            results = collection.query(
                query_texts=[query],
                n_results=limit,
                include=['documents', 'metadatas', 'distances']
            )
            
            articles = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else 1.0
                    
                    articles.append({
                        'content': doc,
                        'metadata': metadata,
                        'similarity_score': 1.0 - distance,  # Convert distance to similarity
                        'article_id': metadata.get('article_id'),
                        'title': metadata.get('title'),
                        'source': metadata.get('source'),
                        'published_at': metadata.get('published_at'),
                        'category': metadata.get('category')
                    })
            
            logger.info(f"Fetched {len(articles)} articles via semantic search")
            return articles
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []
    
    def _initialize_chromadb(self):
        """
        Initialize ChromaDB client following existing patterns.
        """
        try:
            # Create client with telemetry disabled (following news_sentiment pattern)
            settings = Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
            self.chroma_client = chromadb.PersistentClient(
                path=self.chroma_path,
                settings=settings
            )
            
            # Initialize embedding model for semantic similarity
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Try to get the financial_news collection
            try:
                self._news_collection = self.chroma_client.get_collection("financial_news")
                self.chroma_available = True
                logger.info("ChromaDB client initialized successfully with financial_news collection")
            except Exception as collection_error:
                logger.warning(f"ChromaDB financial_news collection not found: {collection_error}")
                self.chroma_available = False
                
        except Exception as e:
            logger.warning(f"ChromaDB client initialization failed: {e}")
            self.chroma_client = None
            self.chroma_available = False
    
    async def find_similar_events_semantic(
        self,
        event_description: str,
        similarity_threshold: float = 0.75,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find semantically similar events using ChromaDB for better deduplication.
        
        Args:
            event_description: Description of the event to find similar events for
            similarity_threshold: Minimum semantic similarity score (0.0-1.0)
            max_results: Maximum number of similar events to return
            
        Returns:
            List of similar events with similarity scores
        """
        if not self.chroma_available or not self._news_collection:
            logger.debug("ChromaDB not available, falling back to Neo4j similarity")
            return self.find_similar_events(event_description, similarity_threshold)
        
        try:
            # Search for semantically similar articles in ChromaDB
            results = self._news_collection.query(
                query_texts=[event_description],
                n_results=max_results * 2,  # Get more to filter
                include=['documents', 'metadatas', 'distances']
            )
            
            similar_events = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else 1.0
                    similarity_score = max(0.0, 1.0 - distance)  # Convert distance to similarity
                    
                    if similarity_score >= similarity_threshold:
                        # Check if this article mentions similar events by looking for existing events
                        # that were extracted from this article
                        article_id = metadata.get('article_id')
                        if article_id:
                            existing_events = await self._get_events_from_article(article_id)
                            for event in existing_events:
                                event['similarity_score'] = similarity_score
                                event['source_article_similarity'] = similarity_score
                                similar_events.append(event)
            
            logger.info(f"Found {len(similar_events)} semantically similar events")
            return similar_events[:max_results]
            
        except Exception as e:
            logger.error(f"Error in semantic event similarity search: {e}")
            # Fallback to Neo4j-based similarity
            return self.find_similar_events(event_description, similarity_threshold)
    
    async def _get_events_from_article(self, article_id: str) -> List[Dict[str, Any]]:
        """
        Get events that were previously extracted from a specific article.
        
        Args:
            article_id: ID of the article
            
        Returns:
            List of events extracted from this article
        """
        try:
            if not self.graph_service.connect():
                return []
            
            query = """
                MATCH (a:NewsArticle {article_id: $article_id})-[:MENTIONS]->(e:Event)
                RETURN e.event_id as event_id,
                       e.description as description,
                       e.confidence as confidence,
                       e.event_type as event_type,
                       e.date as date,
                       e.source_count as source_count,
                       e.created_at as created_at
                ORDER BY e.confidence DESC
            """
            
            results = self.graph_service.execute_query(query, {'article_id': article_id})
            return results
            
        except Exception as e:
            logger.error(f"Error getting events from article {article_id}: {e}")
            return []
        finally:
            self.graph_service.disconnect()
    
    async def verify_event_with_chromadb(
        self,
        event: Dict[str, Any],
        verification_threshold: float = 0.7,
        max_supporting_articles: int = 5
    ) -> Dict[str, Any]:
        """
        Verify an event by finding supporting articles in ChromaDB.
        
        Args:
            event: Event to verify
            verification_threshold: Minimum similarity for supporting articles
            max_supporting_articles: Maximum supporting articles to find
            
        Returns:
            Enhanced event with verification data
        """
        if not self.chroma_available or not self._news_collection:
            logger.debug("ChromaDB not available for event verification")
            return event
        
        try:
            # Search for articles that support this event
            event_description = event.get('description', '')
            entities = event.get('entities', [])
            
            # Create comprehensive search query
            search_terms = [event_description]
            if entities:
                search_terms.extend(entities)
            search_query = ' '.join(search_terms)
            
            results = self._news_collection.query(
                query_texts=[search_query],
                n_results=max_supporting_articles * 2,
                include=['documents', 'metadatas', 'distances']
            )
            
            supporting_articles = []
            unique_sources = set()
            
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                    distance = results['distances'][0][i] if results['distances'] else 1.0
                    similarity_score = max(0.0, 1.0 - distance)
                    
                    if similarity_score >= verification_threshold:
                        source = metadata.get('source_name', 'unknown')
                        article_id = metadata.get('article_id')
                        
                        # Avoid duplicate sources for better verification
                        if source not in unique_sources and article_id != event.get('source_article_id'):
                            supporting_articles.append({
                                'article_id': article_id,
                                'source': source,
                                'similarity_score': similarity_score,
                                'title': metadata.get('title', ''),
                                'published_at': metadata.get('published_at')
                            })
                            unique_sources.add(source)
                            
                            if len(supporting_articles) >= max_supporting_articles:
                                break
            
            # Enhance event with verification data
            enhanced_event = event.copy()
            enhanced_event['chromadb_supporting_articles'] = supporting_articles
            enhanced_event['chromadb_source_count'] = len(unique_sources)
            enhanced_event['chromadb_verified'] = len(supporting_articles) > 0
            
            # Boost confidence based on ChromaDB verification
            if supporting_articles:
                avg_similarity = np.mean([a['similarity_score'] for a in supporting_articles])
                source_diversity_bonus = min(0.2, len(unique_sources) * 0.05)  # Up to 0.2 boost
                similarity_bonus = min(0.15, avg_similarity * 0.15)  # Up to 0.15 boost
                
                original_confidence = enhanced_event.get('confidence', 0.0)
                enhanced_event['confidence'] = min(1.0, original_confidence + source_diversity_bonus + similarity_bonus)
                enhanced_event['confidence_boost_chromadb'] = source_diversity_bonus + similarity_bonus
            
            logger.info(f"ChromaDB verification: {len(supporting_articles)} supporting articles from {len(unique_sources)} sources")
            return enhanced_event
            
        except Exception as e:
            logger.error(f"Error in ChromaDB event verification: {e}")
            return event
    
    async def deduplicate_events_semantic(
        self,
        events: List[Dict[str, Any]],
        similarity_threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate events using semantic similarity with ChromaDB support.
        
        Args:
            events: List of events to deduplicate
            similarity_threshold: Minimum similarity for considering events duplicates
            
        Returns:
            List of deduplicated events
        """
        if not events:
            return events
        
        if not self.chroma_available or not self.embedding_model:
            logger.debug("ChromaDB not available, using basic deduplication")
            return events
        
        try:
            # Generate embeddings for all event descriptions
            descriptions = [event.get('description', '') for event in events]
            embeddings = self.embedding_model.encode(descriptions)
            
            # Calculate similarity matrix
            similarity_matrix = np.dot(embeddings, embeddings.T)
            norms = np.linalg.norm(embeddings, axis=1)
            similarity_matrix = similarity_matrix / np.outer(norms, norms)
            
            # Find duplicate groups
            processed = set()
            deduplicated_events = []
            
            for i, event in enumerate(events):
                if i in processed:
                    continue
                
                # Find similar events
                similar_indices = []
                for j in range(i + 1, len(events)):
                    if j not in processed and similarity_matrix[i][j] >= similarity_threshold:
                        similar_indices.append(j)
                
                if similar_indices:
                    # Merge similar events
                    similar_events = [event] + [events[j] for j in similar_indices]
                    merged_event = self._merge_duplicate_events(similar_events)
                    deduplicated_events.append(merged_event)
                    
                    # Mark as processed
                    processed.add(i)
                    processed.update(similar_indices)
                else:
                    # No duplicates found
                    deduplicated_events.append(event)
                    processed.add(i)
            
            logger.info(f"Semantic deduplication: {len(events)} -> {len(deduplicated_events)} events")
            return deduplicated_events
            
        except Exception as e:
            logger.error(f"Error in semantic event deduplication: {e}")
            return events
    
    def _merge_duplicate_events(self, duplicate_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge duplicate events into a single event with enhanced metadata.
        
        Args:
            duplicate_events: List of duplicate events to merge
            
        Returns:
            Merged event with combined metadata
        """
        if len(duplicate_events) == 1:
            return duplicate_events[0]
        
        # Use the event with highest confidence as base
        base_event = max(duplicate_events, key=lambda e: e.get('confidence', 0.0))
        merged_event = base_event.copy()
        
        # Combine source information
        all_sources = set()
        all_article_ids = set()
        confidence_scores = []
        
        for event in duplicate_events:
            if event.get('source'):
                all_sources.add(event['source'])
            if event.get('source_article_id'):
                all_article_ids.add(event['source_article_id'])
            if event.get('confidence'):
                confidence_scores.append(event['confidence'])
        
        # Update merged event properties
        merged_event['merged_from_count'] = len(duplicate_events)
        merged_event['all_sources'] = list(all_sources)
        merged_event['all_source_articles'] = list(all_article_ids)
        merged_event['source_diversity'] = len(all_sources)
        
        # Enhanced confidence based on cross-source confirmation
        if confidence_scores:
            base_confidence = max(confidence_scores)
            source_bonus = min(0.25, (len(all_sources) - 1) * 0.08)  # Bonus for multiple sources
            merged_event['confidence'] = min(1.0, base_confidence + source_bonus)
            merged_event['confidence_boost_dedup'] = source_bonus
        
        # Merge entities
        all_entities = set()
        for event in duplicate_events:
            if event.get('entities'):
                all_entities.update(event['entities'])
        merged_event['entities'] = list(all_entities)
        
        merged_event['semantic_deduplication'] = True
        return merged_event
    
    def store_event(
        self,
        event_data: Dict[str, Any],
        source_articles: List[str]
    ) -> bool:
        """
        Store an extracted event in Neo4j.
        
        Args:
            event_data: Dictionary containing event information
            source_articles: List of article IDs that mention this event
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.graph_service.connect():
                logger.error("Failed to connect to Neo4j")
                return False
            
            # Create event node
            event_node = self.graph_service.create_node(
                label="Event",
                properties={
                    'event_id': event_data.get('event_id'),
                    'description': event_data.get('description'),
                    'event_type': event_data.get('event_type'),
                    'date': event_data.get('date'),
                    'confidence': event_data.get('confidence', 0.0),
                    'source_count': len(source_articles),
                    'entities': event_data.get('entities', []),
                    'impact_score': event_data.get('impact_score', 0.0),
                    'created_at': datetime.utcnow().isoformat(),
                    'last_updated': datetime.utcnow().isoformat()
                },
                unique_key='event_id'
            )
            
            # Create relationships to source articles
            for article_id in source_articles:
                try:
                    # First create or find the article node
                    article_node = self.graph_service.create_node(
                        label="NewsArticle",
                        properties={
                            'article_id': article_id,
                            'created_at': datetime.utcnow().isoformat()
                        },
                        unique_key='article_id'
                    )
                    
                    # Create relationship
                    self.graph_service.create_relationship(
                        from_node_label="NewsArticle",
                        from_node_key="article_id",
                        from_node_value=article_id,
                        to_node_label="Event",
                        to_node_key="event_id",
                        to_node_value=event_data.get('event_id'),
                        relationship_type="MENTIONS",
                        properties={
                            'created_at': datetime.utcnow().isoformat()
                        }
                    )
                except Exception as e:
                    logger.warning(f"Failed to create relationship for article {article_id}: {e}")
            
            logger.info(f"Stored event {event_data.get('event_id')} with {len(source_articles)} source articles")
            return True
            
        except Exception as e:
            logger.error(f"Error storing event: {e}")
            return False
        finally:
            self.graph_service.disconnect()
    
    def find_similar_events(
        self,
        event_description: str,
        similarity_threshold: float = 0.8,
        days_back: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Find similar events in Neo4j for deduplication.
        
        Args:
            event_description: Description of the event to check
            similarity_threshold: Minimum similarity score
            days_back: Number of days to look back
            
        Returns:
            List of similar events
        """
        try:
            if not self.graph_service.connect():
                logger.error("Failed to connect to Neo4j")
                return []
            
            # Simple text-based similarity check for now
            # In production, you might want to use embedding-based similarity
            cutoff_date = (datetime.utcnow() - timedelta(days=days_back)).isoformat()
            
            query = """
                MATCH (e:Event)
                WHERE e.created_at >= $cutoff_date
                RETURN e.event_id as event_id,
                       e.description as description,
                       e.confidence as confidence,
                       e.source_count as source_count,
                       e.created_at as created_at
                ORDER BY e.created_at DESC
            """
            
            results = self.graph_service.execute_query(query, {'cutoff_date': cutoff_date})
            
            # Simple similarity check (could be enhanced with NLP)
            similar_events = []
            event_words = set(event_description.lower().split())
            
            for result in results:
                existing_words = set(result['description'].lower().split())
                overlap = len(event_words.intersection(existing_words))
                total_words = len(event_words.union(existing_words))
                
                if total_words > 0:
                    similarity = overlap / total_words
                    if similarity >= similarity_threshold:
                        result['similarity_score'] = similarity
                        similar_events.append(result)
            
            logger.info(f"Found {len(similar_events)} similar events")
            return similar_events
            
        except Exception as e:
            logger.error(f"Error finding similar events: {e}")
            return []
        finally:
            self.graph_service.disconnect()
    
    def get_event_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored events.
        
        Returns:
            Dictionary with event statistics
        """
        try:
            if not self.graph_service.connect():
                return {}
            
            stats_queries = {
                'total_events': "MATCH (e:Event) RETURN count(e) as count",
                'events_last_7_days': """
                    MATCH (e:Event) 
                    WHERE e.created_at >= $cutoff_date 
                    RETURN count(e) as count
                """,
                'avg_confidence': "MATCH (e:Event) RETURN avg(e.confidence) as avg_confidence",
                'event_types': """
                    MATCH (e:Event) 
                    RETURN e.event_type as type, count(e) as count 
                    ORDER BY count DESC
                """,
                'top_sources': """
                    MATCH (a:NewsArticle)-[:MENTIONS]->(e:Event)
                    RETURN a.source as source, count(e) as event_count
                    ORDER BY event_count DESC
                    LIMIT 10
                """
            }
            
            stats = {}
            cutoff_date = (datetime.utcnow() - timedelta(days=7)).isoformat()
            
            for key, query in stats_queries.items():
                try:
                    if key == 'events_last_7_days':
                        results = self.graph_service.execute_query(query, {'cutoff_date': cutoff_date})
                    else:
                        results = self.graph_service.execute_query(query)
                    
                    if key in ['total_events', 'events_last_7_days']:
                        stats[key] = results[0]['count'] if results else 0
                    elif key == 'avg_confidence':
                        stats[key] = float(results[0]['avg_confidence']) if results and results[0]['avg_confidence'] else 0.0
                    else:
                        stats[key] = results
                        
                except Exception as e:
                    logger.warning(f"Failed to get {key}: {e}")
                    stats[key] = None
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting event statistics: {e}")
            return {}
        finally:
            self.graph_service.disconnect()
    
    def update_event_confidence(
        self,
        event_id: str,
        new_confidence: float,
        additional_sources: List[str] = None
    ) -> bool:
        """
        Update event confidence and source count.
        
        Args:
            event_id: ID of the event to update
            new_confidence: Updated confidence score
            additional_sources: Additional source article IDs
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.graph_service.connect():
                return False
            
            # Update event properties
            update_query = """
                MATCH (e:Event {event_id: $event_id})
                SET e.confidence = $confidence,
                    e.last_updated = $timestamp
            """
            
            if additional_sources:
                update_query += ", e.source_count = e.source_count + $additional_count"
                params = {
                    'event_id': event_id,
                    'confidence': new_confidence,
                    'timestamp': datetime.utcnow().isoformat(),
                    'additional_count': len(additional_sources)
                }
            else:
                params = {
                    'event_id': event_id,
                    'confidence': new_confidence,
                    'timestamp': datetime.utcnow().isoformat()
                }
            
            update_query += " RETURN e.event_id as updated_id"
            
            results = self.graph_service.execute_query(update_query, params)
            
            if results and additional_sources:
                # Add relationships to new sources
                for article_id in additional_sources:
                    try:
                        self.graph_service.create_relationship(
                            from_node_label="NewsArticle",
                            from_node_key="article_id", 
                            from_node_value=article_id,
                            to_node_label="Event",
                            to_node_key="event_id",
                            to_node_value=event_id,
                            relationship_type="MENTIONS",
                            properties={
                                'created_at': datetime.utcnow().isoformat()
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Failed to add relationship for article {article_id}: {e}")
            
            success = len(results) > 0
            if success:
                logger.info(f"Updated event {event_id} confidence to {new_confidence}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error updating event confidence: {e}")
            return False
        finally:
            self.graph_service.disconnect()

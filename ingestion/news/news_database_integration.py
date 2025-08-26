"""
News Database Integration - Data Storage Layer

This module handles storing News API articles in our PostgreSQL database.
It integrates with the news_series_fetcher to store both article content and metadata.

Key Features:
- Store complete articles in news_articles table
- Smart deduplication using URL hashes
- Bulk insert with conflict handling
- Graceful error handling and comprehensive logging
- Track data sync operations in data_sync_log
- Support for incremental and full historical loads
- Integration with vector database pipeline

Depends on: 
- news_series_fetcher.py for data formatting
- unified_database_setup.py for schema definitions
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone, date, timedelta
import uuid

# Add database directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "database"))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import select, and_, func, text, desc
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

# Import our database models
from unified_database_setup import (
    NewsArticles, DataSyncLog, NewsTopicMapping, 
    DataSourceType, NewsCategory
)

from .news_series_fetcher import NewsSeriesFetcher

# Configure logging
logger = logging.getLogger(__name__)


class NewsDatabaseIntegration:
    """
    Handles storing News API articles in our PostgreSQL database with comprehensive error handling.
    """
    
    def __init__(self, database_url: str):
        """
        Initialize database integration.
        
        Args:
            database_url: PostgreSQL connection string
        """
        self.database_url = database_url.replace('postgresql://', 'postgresql+asyncpg://')
        self.engine = None
        self.AsyncSessionLocal = None
        self.logger = logging.getLogger(__name__)
        self.logger.info("News Database Integration initialized")
    
    async def initialize(self) -> bool:
        """
        Initialize database connection and session factory.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.engine = create_async_engine(
                self.database_url,
                echo=False,  # Set to True for SQL debugging
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600,  # Recycle connections after 1 hour
                pool_timeout=30     # Wait up to 30 seconds for connection
            )
            
            self.AsyncSessionLocal = sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Test connection
            async with self.AsyncSessionLocal() as session:
                await session.execute(text("SELECT 1"))
            
            logger.info("Database connection initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {e}")
            return False
    
    async def store_articles(self, articles: List[Dict[str, Any]]) -> Tuple[int, int, int]:
        """
        Store news articles in bulk with conflict handling.
        
        Args:
            articles: List of article dictionaries from news_series_fetcher
            
        Returns:
            Tuple of (successful_inserts, conflicts_updated, failed_inserts)
        """
        if not articles:
            return 0, 0, 0
        
        successful_inserts = 0
        conflicts_updated = 0
        failed_inserts = 0
        
        try:
            async with self.AsyncSessionLocal() as session:
                for article in articles:
                    try:
                        # Use upsert to handle URL hash conflicts (duplicates)
                        stmt = insert(NewsArticles).values(article)
                        stmt = stmt.on_conflict_do_update(
                            index_elements=['url_hash'],
                            set_={
                                'title': stmt.excluded.title,
                                'description': stmt.excluded.description,
                                'content': stmt.excluded.content,
                                'content_length': stmt.excluded.content_length,
                                'economic_categories': stmt.excluded.economic_categories,
                                'sentiment_score': stmt.excluded.sentiment_score,
                                'relevance_score': stmt.excluded.relevance_score,
                                'data_quality_score': stmt.excluded.data_quality_score,
                                'content_completeness': stmt.excluded.content_completeness,
                                'related_series_ids': stmt.excluded.related_series_ids,
                                'related_market_assets': stmt.excluded.related_market_assets,
                                'is_categorized': stmt.excluded.is_categorized,
                                'updated_at': stmt.excluded.updated_at
                            }
                        )
                        
                        result = await session.execute(stmt)
                        
                        # Note: PostgreSQL doesn't easily tell us if this was insert vs update
                        # We'll count all as successful inserts for simplicity
                        successful_inserts += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to store article {article.get('url', 'unknown')}: {e}")
                        failed_inserts += 1
                
                await session.commit()
                
                logger.info(f"Successfully processed {len(articles)} articles: {successful_inserts} stored, {failed_inserts} failed")
                
        except SQLAlchemyError as e:
            logger.error(f"Database error storing articles: {e}")
            return 0, 0, len(articles)
        except Exception as e:
            logger.error(f"Unexpected error storing articles: {e}")
            return 0, 0, len(articles)
        
        return successful_inserts, conflicts_updated, failed_inserts
    
    async def bulk_insert_articles(self, articles: List[Dict[str, Any]]) -> Tuple[int, int, int]:
        """
        Bulk insert articles into the database (alias for store_articles for compatibility).
        
        Args:
            articles: List of processed article dictionaries
            
        Returns:
            Tuple of (successful_inserts, conflicts_updated, failed_inserts)
        """
        return await self.store_articles(articles)
    
    async def log_api_calls(self, source: str = None, calls_made: int = 0, success: bool = True, **kwargs) -> None:
        """
        Log API calls usage for tracking (compatibility method).
        
        Args:
            source: Data source (for compatibility)
            calls_made: Number of API calls made
            success: Whether the calls were successful
            **kwargs: Additional parameters for compatibility
        """
        # This is a compatibility method - actual logging happens in _log_sync_operation
        logger.debug(f"API calls logged: {calls_made} calls from {source} (success: {success})")
    
    async def bulk_fetch_and_store_news(
        self, 
        categories: Optional[List[str]] = None,
        days_back: int = 1,
        max_articles_per_category: int = 20
    ) -> Dict[str, Any]:
        """
        Fetch and store news articles for specified categories.
        
        Args:
            categories: List of economic categories to fetch (if None, fetches all)
            days_back: Number of days back to search
            max_articles_per_category: Maximum articles per category
            
        Returns:
            Dictionary with operation statistics and results
        """
        sync_start_time = datetime.now(timezone.utc)
        operation_stats = {
            'success': False,
            'categories_processed': [],
            'total_articles_found': 0,
            'total_articles_stored': 0,
            'total_articles_failed': 0,
            'articles_by_category': {},
            'error_message': None,
            'duration_seconds': 0
        }
        
        try:
            logger.info(f"Starting bulk news fetch and store for {days_back} days")
            
            # Use the fetcher to get articles
            async with NewsSeriesFetcher() as fetcher:
                results = await fetcher.search_economic_news(
                    categories=categories,
                    days_back=days_back,
                    max_articles_per_category=max_articles_per_category
                )
                
                total_found = 0
                total_stored = 0
                total_failed = 0
                
                # Process each category
                for category, articles in results.items():
                    if not articles:
                        continue
                    
                    logger.info(f"Storing {len(articles)} articles for category: {category}")
                    
                    # Store articles for this category
                    stored, updated, failed = await self.store_articles(articles)
                    
                    operation_stats['categories_processed'].append(category)
                    operation_stats['articles_by_category'][category] = {
                        'found': len(articles),
                        'stored': stored,
                        'failed': failed
                    }
                    
                    total_found += len(articles)
                    total_stored += stored
                    total_failed += failed
                
                operation_stats['total_articles_found'] = total_found
                operation_stats['total_articles_stored'] = total_stored
                operation_stats['total_articles_failed'] = total_failed
                operation_stats['success'] = True
                
                # Log sync operation
                await self._log_sync_operation(sync_start_time, operation_stats, True)
                
                logger.info(f"Bulk news operation completed: {total_stored} stored, {total_failed} failed")
                
        except Exception as e:
            error_msg = str(e)
            operation_stats['error_message'] = error_msg
            logger.error(f"Bulk news operation failed: {error_msg}")
            
            # Log failed sync operation
            await self._log_sync_operation(sync_start_time, operation_stats, False, error_msg)
        
        finally:
            # Calculate duration
            operation_stats['duration_seconds'] = (datetime.now(timezone.utc) - sync_start_time).total_seconds()
        
        return operation_stats
    
    async def get_articles_for_processing(
        self, 
        limit: int = 100,
        unprocessed_only: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get articles that need vector processing.
        
        Args:
            limit: Maximum number of articles to return
            unprocessed_only: If True, only return articles not yet processed
            
        Returns:
            List of article dictionaries ready for vector processing
        """
        try:
            async with self.AsyncSessionLocal() as session:
                query = select(NewsArticles).order_by(desc(NewsArticles.published_at)).limit(limit)
                
                if unprocessed_only:
                    query = query.where(NewsArticles.is_processed == False)
                
                result = await session.execute(query)
                articles = result.scalars().all()
                
                # Convert to dictionaries
                article_dicts = []
                for article in articles:
                    article_dict = {
                        'id': article.id,
                        'title': article.title,
                        'description': article.description,
                        'content': article.content,
                        'url': article.url,
                        'published_at': article.published_at,
                        'source_name': article.source_name,
                        'economic_categories': article.economic_categories,
                        'relevance_score': article.relevance_score,
                        'data_quality_score': article.data_quality_score,
                        'related_series_ids': article.related_series_ids,
                        'related_market_assets': article.related_market_assets
                    }
                    article_dicts.append(article_dict)
                
                logger.info(f"Retrieved {len(article_dicts)} articles for processing")
                return article_dicts
                
        except Exception as e:
            logger.error(f"Error retrieving articles for processing: {e}")
            return []
    
    async def update_processing_status(
        self, 
        article_id: uuid.UUID, 
        processed: bool = True,
        has_embeddings: bool = False,
        vector_db_collection: Optional[str] = None,
        vector_db_document_id: Optional[str] = None,
        embedding_model_version: Optional[str] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """
        Update the processing status of an article.
        
        Args:
            article_id: UUID of the article to update
            processed: Whether the article has been processed
            has_embeddings: Whether embeddings have been created
            vector_db_collection: Chroma collection name
            vector_db_document_id: Document ID in vector database
            embedding_model_version: Version of embedding model used
            error_message: Error message if processing failed
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with self.AsyncSessionLocal() as session:
                # Update the article
                stmt = (
                    select(NewsArticles)
                    .where(NewsArticles.id == article_id)
                )
                result = await session.execute(stmt)
                article = result.scalar_one_or_none()
                
                if not article:
                    logger.warning(f"Article {article_id} not found for status update")
                    return False
                
                # Update fields
                article.is_processed = processed
                article.has_embeddings = has_embeddings
                article.processed_at = datetime.now(timezone.utc)
                article.updated_at = datetime.now(timezone.utc)
                
                if vector_db_collection:
                    article.vector_db_collection = vector_db_collection
                if vector_db_document_id:
                    article.vector_db_document_id = vector_db_document_id
                if embedding_model_version:
                    article.embedding_model_version = embedding_model_version
                
                if error_message:
                    article.last_processing_error = error_message
                    article.processing_attempts = (article.processing_attempts or 0) + 1
                
                await session.commit()
                
                logger.debug(f"Updated processing status for article {article_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error updating processing status for article {article_id}: {e}")
            return False
    
    async def get_news_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about stored news articles"""
        try:
            async with self.AsyncSessionLocal() as session:
                # Total articles
                total_count = await session.execute(
                    select(func.count(NewsArticles.id))
                )
                
                # Articles by processing status
                processed_count = await session.execute(
                    select(func.count(NewsArticles.id))
                    .where(NewsArticles.is_processed == True)
                )
                
                unprocessed_count = await session.execute(
                    select(func.count(NewsArticles.id))
                    .where(NewsArticles.is_processed == False)
                )
                
                # Articles with embeddings
                embeddings_count = await session.execute(
                    select(func.count(NewsArticles.id))
                    .where(NewsArticles.has_embeddings == True)
                )
                
                # Date range
                date_range = await session.execute(
                    select(
                        func.min(NewsArticles.published_at),
                        func.max(NewsArticles.published_at)
                    )
                )
                min_date, max_date = date_range.first()
                
                # Average quality scores
                avg_scores = await session.execute(
                    select(
                        func.avg(NewsArticles.data_quality_score),
                        func.avg(NewsArticles.relevance_score)
                    )
                )
                avg_quality, avg_relevance = avg_scores.first()
                
                # Top sources
                top_sources = await session.execute(
                    select(
                        NewsArticles.source_name,
                        func.count(NewsArticles.id).label('count')
                    )
                    .group_by(NewsArticles.source_name)
                    .order_by(func.count(NewsArticles.id).desc())
                    .limit(10)
                )
                
                return {
                    'total_articles': total_count.scalar() or 0,
                    'processed_articles': processed_count.scalar() or 0,
                    'unprocessed_articles': unprocessed_count.scalar() or 0,
                    'articles_with_embeddings': embeddings_count.scalar() or 0,
                    'date_range_start': min_date,
                    'date_range_end': max_date,
                    'average_quality_score': float(avg_quality) if avg_quality else 0.0,
                    'average_relevance_score': float(avg_relevance) if avg_relevance else 0.0,
                    'top_sources': [(row[0], row[1]) for row in top_sources.fetchall()],
                    'last_updated': datetime.now(timezone.utc)
                }
                
        except Exception as e:
            logger.error(f"Error getting news statistics: {e}")
            return {}
    
    async def get_recent_articles(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent articles for monitoring"""
        try:
            async with self.AsyncSessionLocal() as session:
                result = await session.execute(
                    select(NewsArticles)
                    .order_by(desc(NewsArticles.published_at))
                    .limit(limit)
                )
                
                articles = result.scalars().all()
                
                return [
                    {
                        'id': str(article.id),
                        'title': article.title[:100] + '...' if len(article.title) > 100 else article.title,
                        'source_name': article.source_name,
                        'published_at': article.published_at,
                        'economic_categories': article.economic_categories,
                        'relevance_score': float(article.relevance_score) if article.relevance_score else 0,
                        'data_quality_score': float(article.data_quality_score) if article.data_quality_score else 0,
                        'is_processed': article.is_processed,
                        'has_embeddings': article.has_embeddings
                    }
                    for article in articles
                ]
                
        except Exception as e:
            logger.error(f"Error getting recent articles: {e}")
            return []
    
    async def _log_sync_operation(
        self, 
        start_time: datetime, 
        stats: Dict[str, Any], 
        success: bool, 
        error_message: Optional[str] = None
    ) -> None:
        """Log sync operation to data_sync_log table"""
        try:
            sync_log = DataSyncLog(
                series_id=None,  # News doesn't map to a single series
                source_type=DataSourceType.NEWS_API,
                sync_type='bulk_news_fetch',
                sync_start_time=start_time,
                sync_end_time=datetime.now(timezone.utc),
                sync_duration_ms=int(stats['duration_seconds'] * 1000),
                success=success,
                records_processed=stats['total_articles_found'],
                records_added=stats['total_articles_stored'],
                records_updated=0,  # We don't track updates separately for now
                records_failed=stats['total_articles_failed'],
                error_message=error_message,
                error_type='news_fetch_error' if error_message else None,
                data_quality_score=1.0 if success else 0.0,
                sync_parameters={
                    'categories': stats.get('categories_processed', []),
                    'articles_by_category': stats.get('articles_by_category', {})
                }
            )
            
            async with self.AsyncSessionLocal() as session:
                session.add(sync_log)
                await session.commit()
                
        except Exception as e:
            logger.warning(f"Failed to log sync operation: {e}")
    
    async def close(self) -> None:
        """Close database connections"""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections closed")


# Convenience function for testing
async def test_news_database_integration():
    """Test the database integration with sample news data"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    database_url = os.getenv('DATABASE_URL', 'postgresql+psycopg://postgres:fred_password@localhost:5432/postgres')
    
    db_integration = NewsDatabaseIntegration(database_url)
    
    try:
        # Initialize
        if not await db_integration.initialize():
            print("❌ Failed to initialize database connection")
            return False
        
        print("✅ Database connection initialized")
        
        # Test fetching and storing news
        stats = await db_integration.bulk_fetch_and_store_news(
            categories=['federal_reserve'],
            days_back=2,
            max_articles_per_category=5
        )
        
        if stats['success']:
            print(f"✅ Bulk fetch and store successful:")
            print(f"   Categories: {stats['categories_processed']}")
            print(f"   Articles found: {stats['total_articles_found']}")
            print(f"   Articles stored: {stats['total_articles_stored']}")
            print(f"   Duration: {stats['duration_seconds']:.2f} seconds")
            
            # Get recent articles
            recent = await db_integration.get_recent_articles(limit=3)
            if recent:
                print(f"✅ Recent articles:")
                for article in recent:
                    print(f"   • {article['title']}")
                    print(f"     Source: {article['source_name']}, Quality: {article['data_quality_score']}")
        else:
            print(f"❌ Bulk operation failed: {stats['error_message']}")
            return False
        
        # Get comprehensive statistics
        news_stats = await db_integration.get_news_statistics()
        print(f"✅ News database stats:")
        print(f"   Total articles: {news_stats['total_articles']}")
        print(f"   Processed: {news_stats['processed_articles']}")
        print(f"   Avg quality: {news_stats['average_quality_score']:.2f}")
        print(f"   Avg relevance: {news_stats['average_relevance_score']:.2f}")
        print(f"   Top sources: {[s[0] for s in news_stats['top_sources'][:3]]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    finally:
        await db_integration.close()


if __name__ == "__main__":
    """Test the database integration when run directly"""
    import asyncio
    asyncio.run(test_news_database_integration())
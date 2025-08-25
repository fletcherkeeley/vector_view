"""
Embedding Pipeline for Financial Intelligence Platform

Processes data from PostgreSQL and creates embeddings in ChromaDB.
Handles both news articles and economic indicators with comprehensive metadata.

Features:
- Batch processing of unprocessed articles
- Economic indicator embedding with correlation metadata
- Real-time embedding for new data
- Progress tracking and error handling
- Integration with existing database schema
"""

import asyncio
import logging
import sys
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import os
import hashlib

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "database"))

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, and_, func, update
import structlog

# Import our models and vector store
from unified_database_setup import (
    NewsArticles, DataSeries, MarketAssets, 
    DataSourceType, FrequencyType, AssetType
)
try:
    from .vector_store import SemanticVectorStore, CollectionType, create_semantic_store
except ImportError:
    # Handle case when run as standalone script
    from vector_store import SemanticVectorStore, CollectionType, create_semantic_store

# Load environment variables
load_dotenv()

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


class EmbeddingPipeline:
    """
    Processes PostgreSQL data and creates semantic embeddings
    """
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
        self.AsyncSessionLocal = None
        self.semantic_store = None
        
    async def initialize(self) -> bool:
        """Initialize database connections and semantic store"""
        try:
            # Initialize database
            self.engine = create_async_engine(
                self.database_url,
                echo=False,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True
            )
            
            self.AsyncSessionLocal = sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Initialize semantic store
            self.semantic_store = await create_semantic_store()
            
            logger.info("Embedding pipeline initialized successfully")
            return True
            
        except Exception as e:
            logger.error("Failed to initialize embedding pipeline", error=str(e))
            return False
    
    async def get_unprocessed_news_articles(self, limit: int = 1000) -> List[NewsArticles]:
        """Get news articles that haven't been embedded yet"""
        try:
            async with self.AsyncSessionLocal() as session:
                result = await session.execute(
                    select(NewsArticles)
                    .where(
                        and_(
                            NewsArticles.has_embeddings == False,
                            NewsArticles.content.isnot(None),
                            NewsArticles.data_quality_score >= 0.5
                        )
                    )
                    .order_by(NewsArticles.published_at.desc())
                    .limit(limit)
                )
                
                articles = result.scalars().all()
                logger.info(f"Found {len(articles)} unprocessed news articles")
                return articles
                
        except Exception as e:
            logger.error("Failed to get unprocessed news articles", error=str(e))
            return []
    
    async def get_unprocessed_economic_indicators(self, limit: int = 1000) -> List[DataSeries]:
        """Get economic indicators that haven't been embedded yet"""
        try:
            async with self.AsyncSessionLocal() as session:
                result = await session.execute(
                    select(DataSeries)
                    .where(
                        and_(
                            DataSeries.source_type == DataSourceType.FRED,
                            DataSeries.is_active == True,
                            DataSeries.description.isnot(None)
                        )
                    )
                    .order_by(DataSeries.correlation_priority.desc())
                    .limit(limit)
                )
                
                indicators = result.scalars().all()
                logger.info(f"Found {len(indicators)} economic indicators to embed")
                return indicators
                
        except Exception as e:
            logger.error("Failed to get economic indicators", error=str(e))
            return []
    
    def _classify_economic_categories(self, article: NewsArticles) -> List[str]:
        """Classify article into economic categories using keyword matching"""
        categories = []
        
        # Combine title and content for analysis
        text = ""
        if article.title:
            text += article.title.lower() + " "
        if article.description:
            text += article.description.lower() + " "
        if article.content:
            text += article.content.lower()[:1000]
        
        # Economic category keywords
        category_keywords = {
            "federal_reserve": ["federal reserve", "fed", "fomc", "jerome powell", "interest rate", "monetary policy"],
            "employment": ["unemployment", "jobs", "employment", "labor", "payroll", "jobless"],
            "inflation": ["inflation", "cpi", "ppi", "consumer price", "producer price", "deflation"],
            "gdp_growth": ["gdp", "gross domestic product", "economic growth", "recession", "expansion"],
            "corporate_earnings": ["earnings", "profit", "revenue", "quarterly results", "eps"],
            "market_volatility": ["volatility", "vix", "market swing", "uncertainty", "risk"],
            "commodity_markets": ["oil", "gold", "commodity", "crude", "natural gas", "metals"],
            "international_trade": ["trade", "tariff", "export", "import", "trade war", "china trade"]
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in text for keyword in keywords):
                categories.append(category)
        
        return categories if categories else ["general_financial"]
    
    def _prepare_news_metadata(self, article: NewsArticles) -> Dict[str, Any]:
        """Prepare comprehensive metadata for news article"""
        # Classify economic categories
        economic_categories = self._classify_economic_categories(article)
        
        metadata = {
            "article_id": str(article.id),
            "source_name": article.source_name or "",
            "published_at": article.published_at.isoformat() if article.published_at else "",
            "economic_categories": ",".join(economic_categories),
            "sentiment_score": float(article.sentiment_score) if article.sentiment_score else 0.0,
            "relevance_score": float(article.relevance_score) if article.relevance_score else 0.0,
            "language": article.language or "en",
            "country": article.country or "",
            "impact_timeframe": article.impact_timeframe or "short_term"
        }
        
        # Add related series if available
        if article.related_series_ids:
            if isinstance(article.related_series_ids, list):
                metadata["related_series"] = ",".join(article.related_series_ids)
            else:
                metadata["related_series"] = str(article.related_series_ids)
        else:
            metadata["related_series"] = ""
        
        return metadata
    
    def _prepare_indicator_metadata(self, indicator: DataSeries) -> Dict[str, Any]:
        """Prepare comprehensive metadata for economic indicator"""
        metadata = {
            "series_id": indicator.series_id,
            "source_type": str(indicator.source_type).replace('DataSourceType.', ''),
            "category": indicator.category or "",
            "subcategory": indicator.subcategory or "",
            "frequency": str(indicator.frequency).replace('FrequencyType.', ''),
            "units": indicator.units or "",
            "seasonal_adjustment": indicator.seasonal_adjustment or "",
            "correlation_priority": indicator.correlation_priority or 0
        }
        
        # Add news categories if available
        if indicator.news_categories:
            if isinstance(indicator.news_categories, list):
                metadata["related_news_categories"] = ",".join(indicator.news_categories)
            else:
                metadata["related_news_categories"] = str(indicator.news_categories)
        else:
            metadata["related_news_categories"] = ""
        
        return metadata
    
    async def process_news_articles_batch(self, articles: List[NewsArticles]) -> Tuple[int, int]:
        """Process a batch of news articles"""
        successful = 0
        failed = 0
        
        for article in articles:
            try:
                # Prepare content and metadata
                metadata = self._prepare_news_metadata(article)
                
                # Add to semantic store
                success = await self.semantic_store.add_news_article(
                    article_id=str(article.id),
                    title=article.title or "",
                    content=article.content or "",
                    metadata=metadata
                )
                
                if success:
                    # Update PostgreSQL record
                    async with self.AsyncSessionLocal() as session:
                        await session.execute(
                            update(NewsArticles)
                            .where(NewsArticles.id == article.id)
                            .values(
                                has_embeddings=True,
                                economic_categories=metadata["economic_categories"].split(","),
                                processed_at=datetime.now(timezone.utc)
                            )
                        )
                        await session.commit()
                    
                    successful += 1
                    logger.debug(f"Successfully processed news article {article.id}")
                else:
                    failed += 1
                    
            except Exception as e:
                failed += 1
                logger.error(f"Failed to process news article {article.id}", error=str(e))
        
        return successful, failed
    
    async def process_economic_indicators_batch(self, indicators: List[DataSeries]) -> Tuple[int, int]:
        """Process a batch of economic indicators"""
        successful = 0
        failed = 0
        
        for indicator in indicators:
            try:
                # Prepare metadata
                metadata = self._prepare_indicator_metadata(indicator)
                
                # Add to semantic store
                success = await self.semantic_store.add_economic_indicator(
                    series_id=indicator.series_id,
                    title=indicator.title,
                    description=indicator.description or "",
                    metadata=metadata
                )
                
                if success:
                    successful += 1
                    logger.debug(f"Successfully processed economic indicator {indicator.series_id}")
                else:
                    failed += 1
                    
            except Exception as e:
                failed += 1
                logger.error(f"Failed to process economic indicator {indicator.series_id}", error=str(e))
        
        return successful, failed
    
    async def run_full_embedding_pipeline(self) -> Dict[str, Any]:
        """Run the complete embedding pipeline for all unprocessed data"""
        logger.info("Starting full embedding pipeline")
        
        results = {
            "news_articles": {"processed": 0, "failed": 0},
            "economic_indicators": {"processed": 0, "failed": 0},
            "start_time": datetime.now(timezone.utc),
            "end_time": None,
            "duration_seconds": 0
        }
        
        try:
            # Process news articles
            logger.info("Processing news articles...")
            news_articles = await self.get_unprocessed_news_articles()
            
            if news_articles:
                batch_size = 50  # Process in smaller batches
                for i in range(0, len(news_articles), batch_size):
                    batch = news_articles[i:i + batch_size]
                    successful, failed = await self.process_news_articles_batch(batch)
                    results["news_articles"]["processed"] += successful
                    results["news_articles"]["failed"] += failed
                    
                    logger.info(f"Processed news batch {i//batch_size + 1}: {successful} successful, {failed} failed")
            
            # Process economic indicators
            logger.info("Processing economic indicators...")
            indicators = await self.get_unprocessed_economic_indicators()
            
            if indicators:
                batch_size = 100  # Larger batches for indicators
                for i in range(0, len(indicators), batch_size):
                    batch = indicators[i:i + batch_size]
                    successful, failed = await self.process_economic_indicators_batch(batch)
                    results["economic_indicators"]["processed"] += successful
                    results["economic_indicators"]["failed"] += failed
                    
                    logger.info(f"Processed indicator batch {i//batch_size + 1}: {successful} successful, {failed} failed")
            
            results["end_time"] = datetime.now(timezone.utc)
            results["duration_seconds"] = (results["end_time"] - results["start_time"]).total_seconds()
            
            logger.info("Embedding pipeline completed", results=results)
            return results
            
        except Exception as e:
            logger.error("Embedding pipeline failed", error=str(e))
            results["error"] = str(e)
            return results
    
    async def process_single_news_article(self, article_id: str) -> bool:
        """Process a single news article for real-time embedding"""
        try:
            async with self.AsyncSessionLocal() as session:
                result = await session.execute(
                    select(NewsArticles).where(NewsArticles.id == article_id)
                )
                article = result.scalar_one_or_none()
                
                if not article:
                    logger.warning(f"Article {article_id} not found")
                    return False
                
                successful, failed = await self.process_news_articles_batch([article])
                return successful > 0
                
        except Exception as e:
            logger.error(f"Failed to process single article {article_id}", error=str(e))
            return False
    
    async def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding pipeline"""
        try:
            # Get semantic store stats
            store_stats = await self.semantic_store.get_collection_stats()
            
            # Get PostgreSQL stats
            async with self.AsyncSessionLocal() as session:
                # Count embedded vs unembedded articles
                total_articles = await session.execute(
                    select(func.count(NewsArticles.id))
                )
                
                embedded_articles = await session.execute(
                    select(func.count(NewsArticles.id))
                    .where(NewsArticles.has_embeddings == True)
                )
                
                total_indicators = await session.execute(
                    select(func.count(DataSeries.series_id))
                    .where(DataSeries.source_type == DataSourceType.FRED)
                )
                
                stats = {
                    "vector_store": store_stats,
                    "postgresql": {
                        "total_articles": total_articles.scalar() or 0,
                        "embedded_articles": embedded_articles.scalar() or 0,
                        "total_indicators": total_indicators.scalar() or 0
                    }
                }
                
                # Calculate completion percentage
                total = stats["postgresql"]["total_articles"]
                embedded = stats["postgresql"]["embedded_articles"]
                stats["completion_percentage"] = (embedded / total * 100) if total > 0 else 0
                
                return stats
                
        except Exception as e:
            logger.error("Failed to get pipeline stats", error=str(e))
            return {"error": str(e)}


async def create_embedding_pipeline(database_url: str = None) -> EmbeddingPipeline:
    """Factory function to create and initialize embedding pipeline"""
    if not database_url:
        database_url = os.getenv('DATABASE_URL', 'postgresql+psycopg://postgres:fred_password@localhost:5432/postgres')
    
    pipeline = EmbeddingPipeline(database_url)
    
    if await pipeline.initialize():
        return pipeline
    else:
        raise RuntimeError("Failed to initialize embedding pipeline")


if __name__ == "__main__":
    # CLI interface for running the embedding pipeline
    async def main():
        try:
            pipeline = await create_embedding_pipeline()
            
            # Run full pipeline
            results = await pipeline.run_full_embedding_pipeline()
            print(f"Pipeline Results: {results}")
            
            # Get stats
            stats = await pipeline.get_pipeline_stats()
            print(f"Pipeline Stats: {stats}")
            
        except Exception as e:
            print(f"Pipeline execution failed: {e}")
    
    asyncio.run(main())

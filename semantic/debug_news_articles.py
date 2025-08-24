#!/usr/bin/env python3
"""
Debug script to investigate why news articles aren't being processed
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "database"))

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, and_, func, text
import os

from unified_database_setup import NewsArticles

load_dotenv()


async def debug_news_articles():
    """Debug news articles processing conditions"""
    
    database_url = os.getenv('DATABASE_URL', 'postgresql+psycopg://postgres:fred_password@localhost:5432/postgres')
    
    engine = create_async_engine(database_url, echo=False)
    AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    
    async with AsyncSessionLocal() as session:
        print("🔍 Investigating news articles processing conditions...\n")
        
        # Total articles
        total_result = await session.execute(select(func.count(NewsArticles.id)))
        total_count = total_result.scalar()
        print(f"📊 Total articles in database: {total_count}")
        
        # Check each condition
        conditions = [
            ("has_embeddings == False", NewsArticles.has_embeddings == False),
            ("is_processed == True", NewsArticles.is_processed == True),
            ("content is not None", NewsArticles.content.isnot(None)),
            ("data_quality_score >= 0.5", NewsArticles.data_quality_score >= 0.5)
        ]
        
        for desc, condition in conditions:
            result = await session.execute(select(func.count(NewsArticles.id)).where(condition))
            count = result.scalar()
            print(f"   • {desc}: {count} articles")
        
        # Combined conditions (what the pipeline uses)
        combined_result = await session.execute(
            select(func.count(NewsArticles.id))
            .where(
                and_(
                    NewsArticles.has_embeddings == False,
                    NewsArticles.is_processed == True,
                    NewsArticles.content.isnot(None),
                    NewsArticles.data_quality_score >= 0.5
                )
            )
        )
        combined_count = combined_result.scalar()
        print(f"\n✅ Articles meeting ALL conditions: {combined_count}")
        
        # Let's check some sample data
        print("\n📋 Sample article analysis:")
        sample_result = await session.execute(
            select(NewsArticles.id, NewsArticles.has_embeddings, NewsArticles.is_processed, 
                   NewsArticles.data_quality_score, NewsArticles.title,
                   func.length(NewsArticles.content).label('content_length'))
            .limit(10)
        )
        
        samples = sample_result.fetchall()
        for sample in samples:
            print(f"   ID: {sample[0]}")
            print(f"     has_embeddings: {sample[1]}")
            print(f"     is_processed: {sample[2]}")
            print(f"     quality_score: {sample[3]}")
            print(f"     content_length: {sample[5]}")
            print(f"     title: {sample[4][:50]}...")
            print()
        
        # Check data quality scores distribution
        print("📈 Data quality score distribution:")
        quality_result = await session.execute(
            select(
                func.count(NewsArticles.id).label('count'),
                func.round(NewsArticles.data_quality_score, 1).label('quality_range')
            )
            .where(NewsArticles.data_quality_score.isnot(None))
            .group_by(func.round(NewsArticles.data_quality_score, 1))
            .order_by('quality_range')
        )
        
        quality_dist = quality_result.fetchall()
        for row in quality_dist:
            print(f"   Quality {row[1]}: {row[0]} articles")
        
        # Check processing status
        print("\n🔄 Processing status distribution:")
        processing_result = await session.execute(
            select(
                NewsArticles.is_processed,
                func.count(NewsArticles.id).label('count')
            )
            .group_by(NewsArticles.is_processed)
        )
        
        processing_dist = processing_result.fetchall()
        for row in processing_dist:
            print(f"   is_processed={row[0]}: {row[1]} articles")
        
        # Check embedding status
        print("\n🧠 Embedding status distribution:")
        embedding_result = await session.execute(
            select(
                NewsArticles.has_embeddings,
                func.count(NewsArticles.id).label('count')
            )
            .group_by(NewsArticles.has_embeddings)
        )
        
        embedding_dist = embedding_result.fetchall()
        for row in embedding_dist:
            print(f"   has_embeddings={row[0]}: {row[1]} articles")


if __name__ == "__main__":
    asyncio.run(debug_news_articles())

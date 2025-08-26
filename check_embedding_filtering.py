#!/usr/bin/env python3
"""
Check why only 789 of 2,523 articles were embedded
"""
import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "database"))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, and_, func, text
from unified_database_setup import NewsArticles

async def analyze_embedding_filtering():
    """Analyze why articles weren't embedded"""
    
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("❌ DATABASE_URL environment variable not set")
        return
    engine = create_async_engine(database_url)
    AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    
    async with AsyncSessionLocal() as session:
        print("🔍 Analyzing article embedding filtering...")
        
        # Total articles from today
        today_total = await session.execute(text("""
            SELECT COUNT(*) FROM news_articles 
            WHERE DATE(created_at) = CURRENT_DATE
        """))
        total_today = today_total.scalar()
        
        # Articles that meet basic criteria (content not null)
        content_check = await session.execute(text("""
            SELECT COUNT(*) FROM news_articles 
            WHERE DATE(created_at) = CURRENT_DATE 
            AND content IS NOT NULL
        """))
        with_content = content_check.scalar()
        
        # Articles that meet quality threshold (>= 0.5)
        quality_check = await session.execute(text("""
            SELECT COUNT(*) FROM news_articles 
            WHERE DATE(created_at) = CURRENT_DATE 
            AND content IS NOT NULL 
            AND data_quality_score >= 0.5
        """))
        quality_filtered = quality_check.scalar()
        
        # Articles already embedded (has_embeddings = true)
        embedded_check = await session.execute(text("""
            SELECT COUNT(*) FROM news_articles 
            WHERE DATE(created_at) = CURRENT_DATE 
            AND has_embeddings = true
        """))
        already_embedded = embedded_check.scalar()
        
        # Articles that should be eligible for embedding
        eligible_check = await session.execute(text("""
            SELECT COUNT(*) FROM news_articles 
            WHERE DATE(created_at) = CURRENT_DATE 
            AND content IS NOT NULL 
            AND data_quality_score >= 0.5 
            AND has_embeddings = false
        """))
        eligible_for_embedding = eligible_check.scalar()
        
        # Quality score distribution
        quality_dist = await session.execute(text("""
            SELECT 
                CASE 
                    WHEN data_quality_score IS NULL THEN 'NULL'
                    WHEN data_quality_score < 0.1 THEN '< 0.1'
                    WHEN data_quality_score < 0.3 THEN '0.1-0.3'
                    WHEN data_quality_score < 0.5 THEN '0.3-0.5'
                    WHEN data_quality_score < 0.7 THEN '0.5-0.7'
                    WHEN data_quality_score < 0.9 THEN '0.7-0.9'
                    ELSE '≥ 0.9'
                END as quality_range,
                COUNT(*) as count
            FROM news_articles 
            WHERE DATE(created_at) = CURRENT_DATE
            GROUP BY quality_range
            ORDER BY quality_range
        """))
        quality_ranges = quality_dist.fetchall()
        
        # Sample articles below quality threshold
        low_quality_samples = await session.execute(text("""
            SELECT title, data_quality_score, content IS NOT NULL as has_content
            FROM news_articles 
            WHERE DATE(created_at) = CURRENT_DATE 
            AND (data_quality_score < 0.5 OR data_quality_score IS NULL)
            ORDER BY created_at DESC 
            LIMIT 5
        """))
        low_quality = low_quality_samples.fetchall()
        
        print(f"📊 Embedding Analysis Results:")
        print(f"  Total articles today: {total_today:,}")
        print(f"  Articles with content: {with_content:,}")
        print(f"  Articles meeting quality threshold (≥0.5): {quality_filtered:,}")
        print(f"  Articles already embedded: {already_embedded:,}")
        print(f"  Articles eligible for embedding: {eligible_for_embedding:,}")
        
        print(f"\n📈 Quality Score Distribution:")
        for range_name, count in quality_ranges:
            print(f"  {range_name}: {count:,} articles")
        
        print(f"\n📰 Sample Low Quality Articles:")
        for article in low_quality[:3]:
            title = article[0][:50] + "..." if len(article[0]) > 50 else article[0]
            quality = article[1] if article[1] is not None else "NULL"
            has_content = "Yes" if article[2] else "No"
            print(f"  • {title}")
            print(f"    Quality: {quality}, Has Content: {has_content}")
        
        # Calculate filtering reasons
        filtered_out = total_today - quality_filtered
        print(f"\n🔍 Filtering Analysis:")
        print(f"  Articles filtered out: {filtered_out:,}")
        print(f"  Main reasons:")
        print(f"    - No content: {total_today - with_content:,}")
        print(f"    - Low quality score (<0.5): {with_content - quality_filtered:,}")
        
        await engine.dispose()

if __name__ == "__main__":
    asyncio.run(analyze_embedding_filtering())

#!/usr/bin/env python3
"""
Verify articles are stored in PostgreSQL database
"""
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ingestion.news.news_database_integration import NewsDatabaseIntegration

async def verify_articles():
    """Verify articles in PostgreSQL database"""
    
    database_url = os.getenv('DATABASE_URL')
    db_integration = NewsDatabaseIntegration(database_url)
    
    await db_integration.initialize()
    
    print("🔍 Checking PostgreSQL database for articles...")
    
    # Get article statistics
    async with db_integration.AsyncSessionLocal() as session:
        from sqlalchemy import text
        
        # Total articles
        total_result = await session.execute(text("SELECT COUNT(*) FROM news_articles"))
        total_count = total_result.scalar()
        
        # Articles from today
        today_result = await session.execute(text("""
            SELECT COUNT(*) FROM news_articles 
            WHERE DATE(created_at) = CURRENT_DATE
        """))
        today_count = today_result.scalar()
        
        # Articles by category from today
        category_result = await session.execute(text("""
            SELECT 
                jsonb_array_elements_text(economic_categories) as category,
                COUNT(*) as count
            FROM news_articles 
            WHERE DATE(created_at) = CURRENT_DATE
            GROUP BY category
            ORDER BY count DESC
        """))
        categories = category_result.fetchall()
        
        # Sample recent articles
        sample_result = await session.execute(text("""
            SELECT title, economic_categories, relevance_score, created_at
            FROM news_articles 
            WHERE DATE(created_at) = CURRENT_DATE
            ORDER BY created_at DESC 
            LIMIT 5
        """))
        sample_articles = sample_result.fetchall()
    
    print(f"📊 Database Statistics:")
    print(f"  Total articles: {total_count:,}")
    print(f"  Articles ingested today: {today_count:,}")
    
    print(f"\n📋 Today's Articles by Category:")
    for category, count in categories:
        print(f"  {category}: {count:,} articles")
    
    print(f"\n📰 Sample Recent Articles:")
    for article in sample_articles[:3]:
        title = article[0][:60] + "..." if len(article[0]) > 60 else article[0]
        categories = article[1]
        relevance = float(article[2]) if article[2] else 0
        created = article[3]
        print(f"  • {title}")
        print(f"    Categories: {categories}, Relevance: {relevance:.3f}, Time: {created}")
    
    await db_integration.close()
    return today_count

if __name__ == "__main__":
    asyncio.run(verify_articles())

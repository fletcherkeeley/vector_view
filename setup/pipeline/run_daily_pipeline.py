#!/usr/bin/env python3
"""
Daily Pipeline Runner - Production script for cron jobs
Runs news ingestion followed by embedding pipeline
"""

import asyncio
import sys
import logging
from datetime import datetime
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'ingestion'))
sys.path.insert(0, os.path.join(project_root, 'semantic'))

from news_daily_updater import NewsDailyUpdater
from embedding_pipeline import create_embedding_pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_daily_pipeline(max_calls: int = 50, categories: str = None):
    """
    Run the complete daily pipeline:
    1. Fetch new articles
    2. Run embedding pipeline
    """
    start_time = datetime.utcnow()
    logger.info(f"🚀 Starting Daily Pipeline at {start_time}")
    
    try:
        # Step 1: News Ingestion
        logger.info("📰 Step 1: Running news ingestion...")
        database_url = os.getenv('DATABASE_URL', 'postgresql://postgres:fred_password@localhost:5432/postgres')
        updater = NewsDailyUpdater(database_url=database_url, max_api_calls=max_calls)
        await updater.initialize()
        
        category_list = categories.split(',') if categories else None
        results = await updater.update_all_categories(
            categories=category_list,
            dry_run=False,
            days_back=1,
            max_concurrent=3
        )
        
        await updater.close()
        
        articles_stored = sum(cat.get('articles_stored', 0) for cat in results.get('categories', {}).values())
        logger.info(f"✅ News ingestion completed: {articles_stored} articles stored")
        
        # Step 2: Embedding Pipeline
        logger.info("🔗 Step 2: Running embedding pipeline...")
        pipeline = await create_embedding_pipeline()
        
        embedding_results = await pipeline.run_full_embedding_pipeline()
        
        news_processed = embedding_results.get('news_articles', {}).get('processed', 0)
        indicators_processed = embedding_results.get('economic_indicators', {}).get('processed', 0)
        
        logger.info(f"✅ Embedding pipeline completed: {news_processed} articles, {indicators_processed} indicators processed")
        
        # Summary
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        logger.info("🎉 Daily Pipeline Summary:")
        logger.info(f"   Duration: {duration:.1f} seconds")
        logger.info(f"   Articles ingested: {articles_stored}")
        logger.info(f"   Articles vectorized: {news_processed}")
        logger.info(f"   Success rate: {results.get('success_rate', 0):.1f}%")
        
        return 0 if results.get('success_rate', 0) > 50 else 1
        
    except Exception as e:
        logger.error(f"❌ Daily pipeline failed: {e}")
        return 1


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Daily Pipeline Runner for Cron Jobs')
    parser.add_argument('--max-calls', type=int, default=50, help='Max API calls')
    parser.add_argument('--categories', type=str, help='Comma-separated categories')
    
    args = parser.parse_args()
    
    exit_code = await run_daily_pipeline(
        max_calls=args.max_calls,
        categories=args.categories
    )
    
    sys.exit(exit_code)


if __name__ == '__main__':
    asyncio.run(main())

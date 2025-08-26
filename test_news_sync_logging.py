#!/usr/bin/env python3
"""
Test script to verify news sync logging functionality
"""
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ingestion.news.news_daily_updater import NewsDailyUpdater

async def test_news_sync_logging():
    """Test news sync with logging to verify data_sync_log entries"""
    
    # Get database URL from environment
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("❌ DATABASE_URL not found in environment")
        return
    
    print("🧪 Testing news sync logging...")
    
    # Create updater with minimal API calls for testing
    updater = NewsDailyUpdater(database_url, max_api_calls=10)
    
    # Initialize
    success = await updater.initialize()
    if not success:
        print("❌ Failed to initialize news updater")
        return
    
    print("✅ News updater initialized")
    
    # Run a small sync with dry run first
    print("\n📋 Testing dry run (no logging expected)...")
    dry_results = await updater.update_all_categories(
        categories=['federal_reserve'],  # Single category
        dry_run=True,
        days_back=1
    )
    
    print(f"Dry run results: {dry_results['success']}")
    
    # Run actual sync with minimal calls
    print("\n🔄 Testing actual sync (logging expected)...")
    real_results = await updater.update_all_categories(
        categories=['federal_reserve'],  # Single category  
        dry_run=False,
        days_back=1
    )
    
    print(f"✅ Real sync completed:")
    print(f"  - Success: {real_results['success']}")
    print(f"  - Articles found: {real_results['total_articles_found']}")
    print(f"  - Articles stored: {real_results['total_articles_stored']}")
    print(f"  - API calls used: {real_results['total_api_calls_used']}")
    
    # Check if sync was logged
    print("\n🔍 Checking data_sync_log for NEWS entries...")
    
    # Query the database directly to verify logging
    from database.unified_database_setup import DataSyncLog, DataSourceType
    
    async with updater.db_integration.AsyncSessionLocal() as session:
        from sqlalchemy import select, desc
        
        # Get recent NEWS sync logs
        stmt = select(DataSyncLog).where(
            DataSyncLog.source_type == DataSourceType.NEWS_API
        ).order_by(desc(DataSyncLog.sync_start_time)).limit(3)
        
        result = await session.execute(stmt)
        recent_logs = result.scalars().all()
        
        if recent_logs:
            print(f"✅ Found {len(recent_logs)} recent NEWS sync logs:")
            for log in recent_logs:
                print(f"  - {log.sync_start_time}: {log.sync_type} - {log.records_added} articles")
        else:
            print("❌ No NEWS sync logs found in data_sync_log table")

if __name__ == "__main__":
    asyncio.run(test_news_sync_logging())

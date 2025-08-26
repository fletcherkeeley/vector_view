#!/usr/bin/env python3
"""
Quick test of news sync functionality
"""
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ingestion.news.news_daily_updater import NewsDailyUpdater

async def quick_test():
    """Test news sync with limited API calls"""
    database_url = os.getenv('DATABASE_URL')
    updater = NewsDailyUpdater(database_url, max_api_calls=50)
    await updater.initialize()
    
    print("🧪 Testing news sync with 50 API calls...")
    results = await updater.update_all_categories(
        categories=['federal_reserve', 'employment'],
        dry_run=False,
        days_back=1
    )
    
    print(f'✅ Success: {results["success"]}')
    print(f'📊 Articles found: {results["total_articles_found"]}')
    print(f'💾 Articles stored: {results["total_articles_stored"]}')
    print(f'📞 API calls used: {results["total_api_calls_used"]}')
    print(f'⚡ Success rate: {results["success_rate"]:.1f}%')
    
    return results

if __name__ == "__main__":
    asyncio.run(quick_test())

#!/usr/bin/env python3
"""
Test script to verify the 440 API call allocation
"""
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ingestion.news.news_daily_updater import NewsDailyUpdater

async def test_api_allocation():
    """Test the API call allocation strategy"""
    
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print("❌ DATABASE_URL not found in environment")
        return
    
    print("🧪 Testing 440 API call allocation strategy...")
    
    # Create updater with 440 calls
    updater = NewsDailyUpdater(database_url, max_api_calls=440)
    
    # Get category strategy
    strategy = updater.get_daily_category_strategy()
    
    # Calculate totals
    total_allocated = sum(cat['api_calls_allocated'] for cat in strategy)
    
    print(f"\n📊 API Call Allocation Summary:")
    print(f"Target limit: 440 calls")
    print(f"Total allocated: {total_allocated} calls")
    print(f"Remaining: {440 - total_allocated} calls")
    print(f"Utilization: {(total_allocated/440)*100:.1f}%")
    
    print(f"\n📋 Category Breakdown:")
    for cat in strategy:
        percentage = (cat['api_calls_allocated'] / total_allocated) * 100
        print(f"  {cat['name']:18} - {cat['api_calls_allocated']:2d} calls ({percentage:4.1f}%)")
    
    # Test with dry run to verify no warnings
    print(f"\n🔄 Testing dry run with 440 call limit...")
    
    await updater.initialize()
    results = await updater.update_all_categories(
        dry_run=True,
        days_back=1
    )
    
    print(f"✅ Dry run completed successfully")
    print(f"Categories processed: {results['categories_processed']}")
    print(f"API calls allocated: {results['total_api_calls_used']}")

if __name__ == "__main__":
    asyncio.run(test_api_allocation())

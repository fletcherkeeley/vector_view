#!/usr/bin/env python3
"""
Test script to verify network connectivity fixes for Yahoo Finance and FRED APIs.
"""

import asyncio
import sys
from pathlib import Path

# Add ingestion directory to path
sys.path.insert(0, str(Path(__file__).parent / 'ingestion'))

from ingestion.yahoo.yahoo_finance_client import YahooFinanceClient
from ingestion.fred.fred_client import FredClient

async def test_yahoo_finance():
    """Test Yahoo Finance with conservative settings"""
    print("🔄 Testing Yahoo Finance API...")
    
    try:
        async with YahooFinanceClient() as client:
            # Test connection
            success = await client.test_connection()
            print(f"Connection test: {'✅ SUCCESS' if success else '❌ FAILED'}")
            
            if success:
                # Test a few symbols with delays
                test_symbols = ['SPY', 'QQQ', 'AAPL']
                for symbol in test_symbols:
                    try:
                        data = await client.get_historical_data(symbol, period='5d')
                        print(f"✅ {symbol}: {len(data)} data points")
                        await asyncio.sleep(2)  # Conservative delay between requests
                    except Exception as e:
                        print(f"❌ {symbol}: {e}")
            
            stats = client.get_request_stats()
            print(f"Total requests made: {stats['requests_made']}")
            return success
            
    except Exception as e:
        print(f"❌ Yahoo Finance test failed: {e}")
        return False

async def test_fred_api():
    """Test FRED API with improved retry logic"""
    print("\n🔄 Testing FRED API...")
    
    try:
        async with FredClient() as client:
            # Test connection
            success = await client.test_connection()
            print(f"Connection test: {'✅ SUCCESS' if success else '❌ FAILED'}")
            
            stats = client.get_request_stats()
            print(f"Total requests made: {stats['requests_made']}")
            return success
            
    except Exception as e:
        print(f"❌ FRED API test failed: {e}")
        return False

async def main():
    """Run all network tests"""
    print("🚀 Testing Network Connectivity Fixes\n")
    
    yahoo_success = await test_yahoo_finance()
    fred_success = await test_fred_api()
    
    print(f"\n📊 Test Results:")
    print(f"Yahoo Finance: {'✅ PASS' if yahoo_success else '❌ FAIL'}")
    print(f"FRED API: {'✅ PASS' if fred_success else '❌ FAIL'}")
    
    if yahoo_success and fred_success:
        print("\n🎉 All network tests passed! Ready for data ingestion.")
        return True
    else:
        print("\n⚠️ Some tests failed. Check network configuration.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

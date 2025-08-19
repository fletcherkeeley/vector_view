#!/usr/bin/env python3
"""
FRED Bulk Historical Data Loader

This script loads full historical data for all key economic indicators into the database.
It's designed to be run once to populate the database with decades of economic data.

Key Features:
- Loads all major economic indicators with full historical data
- Progress tracking and detailed logging
- Graceful error handling with retry logic
- Resume capability (skips already loaded series)
- Comprehensive reporting of load results

Usage:
    python fred_bulk_loader.py                    # Load all series
    python fred_bulk_loader.py --resume           # Skip already loaded series  
    python fred_bulk_loader.py --series FEDFUNDS  # Load specific series only
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import argparse

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from fred_database_integration import FredDatabaseIntegration

# Load environment variables
load_dotenv()

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fred_bulk_load.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class FredBulkLoader:
    """
    Manages bulk loading of historical FRED data for all key economic indicators.
    """
    
    # Key Economic Indicators from data_sources.md
    ECONOMIC_INDICATORS = {
        # Interest Rates & Monetary Policy
        'interest_rates': {
            'FEDFUNDS': 'Federal Funds Effective Rate',
            'DGS10': '10-Year Treasury Constant Maturity Rate',
            'DGS2': '2-Year Treasury Constant Maturity Rate', 
            'T10Y3M': '10-Year Treasury Minus 3-Month Treasury',
            'TEDRATE': 'TED Spread (Treasury-EuroDollar)'
        },
        
        # Employment & Labor Market
        'employment': {
            'UNRATE': 'Unemployment Rate',
            'PAYEMS': 'All Employees, Total Nonfarm Payrolls',
            'ICSA': 'Initial Claims for Unemployment Insurance',
            'AHETPI': 'Average Hourly Earnings of Total Private Industries'
        },
        
        # Inflation Indicators
        'inflation': {
            'CPIAUCSL': 'Consumer Price Index for All Urban Consumers'
        },
        
        # Economic Growth & Housing
        'economic_growth': {
            'GDP': 'Gross Domestic Product',
            'PERMIT': 'New Private Housing Units Authorized by Building Permits',
            'HOUST': 'New Privately-Owned Housing Units Started'
        },
        
        # Market Stress & Volatility
        'market_indicators': {
            'VIXCLS': 'CBOE Volatility Index'
        },
        
        # Commodities
        'commodities': {
            'WPUFD49207': 'Producer Price Index: Gold (Final Demand)',
            'BOGZ1LM313011105A': 'Federal Government Monetary Gold and SDRs'
        }
    }
    
    def __init__(self, database_url: str):
        """
        Initialize the bulk loader.
        
        Args:
            database_url: PostgreSQL connection string
        """
        self.database_url = database_url
        self.db_integration = FredDatabaseIntegration(database_url)
        self.load_results = {}
        self.total_series = 0
        self.successful_loads = 0
        self.failed_loads = 0
        
        logger.info("FRED Bulk Loader initialized")
    
    async def initialize(self) -> bool:
        """Initialize database connection"""
        success = await self.db_integration.initialize()
        if success:
            logger.info("Database connection initialized for bulk loading")
        return success
    
    def get_all_series(self) -> Dict[str, str]:
        """
        Get all series IDs and titles as a flat dictionary.
        
        Returns:
            Dictionary mapping series_id -> title
        """
        all_series = {}
        for category, series_dict in self.ECONOMIC_INDICATORS.items():
            all_series.update(series_dict)
        
        return all_series
    
    async def check_existing_series(self, series_ids: List[str]) -> Dict[str, bool]:
        """
        Check which series already exist in the database.
        
        Args:
            series_ids: List of FRED series IDs to check
            
        Returns:
            Dictionary mapping series_id -> exists_in_db
        """
        existing = {}
        
        for series_id in series_ids:
            info = await self.db_integration.get_series_info(series_id)
            existing[series_id] = info is not None
            
            if info:
                logger.info(f"Series {series_id} already exists with {info['observation_count']} observations")
        
        return existing
    
    async def load_series_batch(
        self, 
        series_ids: List[str], 
        skip_existing: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Load a batch of series with full historical data.
        
        Args:
            series_ids: List of FRED series IDs to load
            skip_existing: Whether to skip series that already exist in database
            
        Returns:
            Dictionary mapping series_id -> load_results
        """
        batch_results = {}
        
        # Check existing series if needed
        if skip_existing:
            existing = await self.check_existing_series(series_ids)
            series_to_load = [sid for sid in series_ids if not existing.get(sid, False)]
            skipped = [sid for sid in series_ids if existing.get(sid, False)]
            
            if skipped:
                logger.info(f"Skipping {len(skipped)} existing series: {', '.join(skipped)}")
        else:
            series_to_load = series_ids
        
        # Load each series
        for i, series_id in enumerate(series_to_load, 1):
            logger.info(f"Loading series {i}/{len(series_to_load)}: {series_id}")
            
            try:
                # Load full historical data (no start_date = all available data)
                result = await self.db_integration.bulk_load_series(series_id)
                batch_results[series_id] = result
                
                if result['success']:
                    self.successful_loads += 1
                    logger.info(
                        f"✅ {series_id}: {result['observations_inserted']} observations "
                        f"in {result['duration_seconds']:.1f}s"
                    )
                else:
                    self.failed_loads += 1
                    logger.error(f"❌ {series_id}: {result.get('error_message', 'Unknown error')}")
                
                # Small delay between series to be respectful to FRED API
                await asyncio.sleep(1)
                
            except Exception as e:
                self.failed_loads += 1
                error_result = {
                    'series_id': series_id,
                    'success': False,
                    'error_message': str(e),
                    'observations_inserted': 0
                }
                batch_results[series_id] = error_result
                logger.error(f"❌ Unexpected error loading {series_id}: {e}")
        
        return batch_results
    
    async def load_all_indicators(self, skip_existing: bool = False) -> Dict[str, Any]:
        """
        Load all economic indicators with full historical data.
        
        Args:
            skip_existing: Whether to skip series already in database
            
        Returns:
            Comprehensive results dictionary
        """
        start_time = datetime.now(timezone.utc)
        logger.info("🚀 Starting bulk load of all FRED economic indicators")
        
        # Get all series to load
        all_series = self.get_all_series()
        self.total_series = len(all_series)
        
        logger.info(f"Planning to load {self.total_series} economic indicators")
        
        # Load by category for better organization and progress tracking
        category_results = {}
        
        for category, series_dict in self.ECONOMIC_INDICATORS.items():
            logger.info(f"\n📊 Loading category: {category.upper()}")
            logger.info(f"Series in category: {', '.join(series_dict.keys())}")
            
            series_ids = list(series_dict.keys())
            category_results[category] = await self.load_series_batch(series_ids, skip_existing)
        
        # Calculate final statistics
        end_time = datetime.now(timezone.utc)
        total_duration = (end_time - start_time).total_seconds()
        
        # Aggregate results
        total_observations = sum(
            result.get('observations_inserted', 0) 
            for category_result in category_results.values()
            for result in category_result.values()
        )
        
        final_results = {
            'start_time': start_time,
            'end_time': end_time,
            'total_duration_seconds': total_duration,
            'total_series_attempted': self.total_series,
            'successful_loads': self.successful_loads,
            'failed_loads': self.failed_loads,
            'total_observations_loaded': total_observations,
            'category_results': category_results,
            'success_rate': (self.successful_loads / self.total_series * 100) if self.total_series > 0 else 0
        }
        
        return final_results
    
    async def load_specific_series(self, series_ids: List[str]) -> Dict[str, Any]:
        """
        Load specific series by ID.
        
        Args:
            series_ids: List of specific FRED series IDs to load
            
        Returns:
            Results dictionary
        """
        start_time = datetime.now(timezone.utc)
        logger.info(f"🎯 Loading specific series: {', '.join(series_ids)}")
        
        self.total_series = len(series_ids)
        results = await self.load_series_batch(series_ids)
        
        end_time = datetime.now(timezone.utc)
        total_duration = (end_time - start_time).total_seconds()
        
        total_observations = sum(
            result.get('observations_inserted', 0) 
            for result in results.values()
        )
        
        return {
            'start_time': start_time,
            'end_time': end_time,
            'total_duration_seconds': total_duration,
            'total_series_attempted': self.total_series,
            'successful_loads': self.successful_loads,
            'failed_loads': self.failed_loads,
            'total_observations_loaded': total_observations,
            'series_results': results,
            'success_rate': (self.successful_loads / self.total_series * 100) if self.total_series > 0 else 0
        }
    
    def print_summary_report(self, results: Dict[str, Any]) -> None:
        """Print a comprehensive summary report of the bulk load operation."""
        
        print("\n" + "="*80)
        print("🎯 FRED BULK LOAD SUMMARY REPORT")
        print("="*80)
        
        print(f"📅 Started:  {results['start_time'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"📅 Finished: {results['end_time'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"⏱️  Duration: {results['total_duration_seconds']:.1f} seconds")
        print(f"📊 Success Rate: {results['success_rate']:.1f}%")
        
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"   Total Series Attempted: {results['total_series_attempted']}")
        print(f"   ✅ Successful Loads: {results['successful_loads']}")
        print(f"   ❌ Failed Loads: {results['failed_loads']}")
        print(f"   📊 Total Observations: {results['total_observations_loaded']:,}")
        
        # Category breakdown if available
        if 'category_results' in results:
            print(f"\n📊 BY CATEGORY:")
            for category, category_result in results['category_results'].items():
                successful_in_category = sum(1 for r in category_result.values() if r.get('success', False))
                total_in_category = len(category_result)
                observations_in_category = sum(r.get('observations_inserted', 0) for r in category_result.values())
                
                print(f"   {category.upper()}: {successful_in_category}/{total_in_category} series, {observations_in_category:,} observations")
        
        # Failed series details
        failed_series = []
        if 'category_results' in results:
            for category_result in results['category_results'].values():
                failed_series.extend([
                    (series_id, result.get('error_message', 'Unknown error'))
                    for series_id, result in category_result.items()
                    if not result.get('success', False)
                ])
        elif 'series_results' in results:
            failed_series.extend([
                (series_id, result.get('error_message', 'Unknown error'))
                for series_id, result in results['series_results'].items()
                if not result.get('success', False)
            ])
        
        if failed_series:
            print(f"\n❌ FAILED SERIES:")
            for series_id, error in failed_series:
                print(f"   {series_id}: {error}")
        
        print("\n" + "="*80)
        if results['failed_loads'] == 0:
            print("🎉 ALL SERIES LOADED SUCCESSFULLY!")
        else:
            print(f"⚠️  {results['failed_loads']} series failed to load. Check logs for details.")
        print("="*80)
    
    async def close(self) -> None:
        """Clean up resources"""
        await self.db_integration.close()


async def main():
    """Main entry point for the bulk loader"""
    parser = argparse.ArgumentParser(
        description='Load historical FRED economic data into database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fred_bulk_loader.py                     # Load all indicators with full history
  python fred_bulk_loader.py --resume            # Skip already loaded series
  python fred_bulk_loader.py --series FEDFUNDS   # Load specific series only
  python fred_bulk_loader.py --list              # List all available series
        """
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip series that already exist in database'
    )
    
    parser.add_argument(
        '--series',
        nargs='+',
        help='Load specific series by ID (e.g., FEDFUNDS GDP UNRATE)'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available series and exit'
    )
    
    parser.add_argument(
        '--database-url',
        default=os.getenv('DATABASE_URL', 'postgresql+psycopg://postgres:fred_password@localhost:5432/postgres'),
        help='Database connection URL'
    )
    
    args = parser.parse_args()
    
    # Create bulk loader
    loader = FredBulkLoader(args.database_url)
    
    # Handle list option
    if args.list:
        print("📋 Available FRED Economic Indicators:")
        print("="*50)
        
        for category, series_dict in loader.ECONOMIC_INDICATORS.items():
            print(f"\n{category.upper()}:")
            for series_id, title in series_dict.items():
                print(f"  {series_id:15} - {title}")
        
        return
    
    try:
        # Initialize database connection
        if not await loader.initialize():
            logger.error("Failed to initialize database connection")
            return 1
        
        # Load data
        if args.series:
            # Load specific series
            results = await loader.load_specific_series(args.series)
        else:
            # Load all indicators
            results = await loader.load_all_indicators(skip_existing=args.resume)
        
        # Print summary report
        loader.print_summary_report(results)
        
        # Return appropriate exit code
        return 0 if results['failed_loads'] == 0 else 1
        
    except KeyboardInterrupt:
        logger.info("Bulk load interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Bulk load failed with unexpected error: {e}")
        return 1
    finally:
        await loader.close()


if __name__ == "__main__":
    """Run the bulk loader"""
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
#!/usr/bin/env python3
"""
Yahoo Finance Daily Update System

Automated daily updates for Yahoo Finance market data. Designed to run as a scheduled job
to keep the database current with the latest market data.

Key Features:
- Incremental updates (only fetch new data since last update)
- All 53 assets from bulk loader
- Graceful error handling and retry logic
- Comprehensive logging and monitoring
- Safe to run multiple times (idempotent)
- Optimized for post-market execution

Usage:
    python yahoo_daily_updater.py                    # Update all assets
    python yahoo_daily_updater.py --symbols SPY QQQ  # Update specific assets
    python yahoo_daily_updater.py --dry-run          # Show what would be updated
    
Cron Examples:
    # Run daily at 7 PM EST (after market close)
    0 19 * * * /path/to/python /path/to/yahoo_daily_updater.py
    
    # Run twice daily (morning pre-market and evening post-market)
    0 7,19 * * * /path/to/python /path/to/yahoo_daily_updater.py
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone, date, timedelta
import argparse
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from yahoo_database_integration import YahooDatabaseIntegration
from yahoo_bulk_loader import YahooBulkLoader

# Load environment variables
load_dotenv()

# Configure logging for scheduled runs
def setup_logging(log_level: str = 'INFO'):
    """Setup logging with both file and console output"""
    log_dir = Path(__file__).parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"yahoo_daily_update_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


class YahooDailyUpdater:
    """
    Manages daily incremental updates of Yahoo Finance market data.
    """
    
    def __init__(self, database_url: str, notification_config: Optional[Dict] = None):
        """
        Initialize the daily updater.
        
        Args:
            database_url: PostgreSQL connection string
            notification_config: Optional config for email/slack notifications
        """
        self.database_url = database_url
        self.db_integration = YahooDatabaseIntegration(database_url)
        self.notification_config = notification_config or {}
        
        # Get all assets from bulk loader
        bulk_loader = YahooBulkLoader(database_url)
        self.all_assets = bulk_loader.get_all_symbols()
        
        # Update statistics
        self.update_results = {}
        self.total_new_observations = 0
        self.successful_updates = 0
        self.failed_updates = 0
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Yahoo Finance Daily Updater initialized")
    
    async def initialize(self) -> bool:
        """Initialize database connection"""
        success = await self.db_integration.initialize()
        if success:
            self.logger.info("Database connection initialized for daily updates")
        return success
    
    async def get_last_observation_date(self, symbol: str) -> Optional[date]:
        """
        Get the date of the most recent observation for an asset.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Date of last observation, or None if no data exists
        """
        try:
            async with self.db_integration.AsyncSessionLocal() as session:
                from sqlalchemy import select, func
                from unified_database_setup import TimeSeriesObservation
                
                result = await session.execute(
                    select(func.max(TimeSeriesObservation.observation_date))
                    .where(TimeSeriesObservation.series_id == symbol.upper())
                )
                
                last_date = result.scalar()
                return last_date
                
        except Exception as e:
            self.logger.error(f"Error getting last observation date for {symbol}: {e}")
            return None
    
    async def calculate_update_start_date(self, symbol: str) -> str:
        """
        Calculate the appropriate start date for incremental updates.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Start date string in YYYY-MM-DD format
        """
        last_observation = await self.get_last_observation_date(symbol)
        
        if last_observation:
            # Start from a few days before last observation to catch any late data
            # Market data is typically final, but this ensures we don't miss anything
            start_date = last_observation - timedelta(days=3)  # 3-day overlap for safety
            self.logger.debug(f"{symbol}: Last observation {last_observation}, starting from {start_date}")
        else:
            # No existing data, get last 30 days
            start_date = date.today() - timedelta(days=30)
            self.logger.info(f"{symbol}: No existing data, starting from {start_date}")
        
        return start_date.strftime('%Y-%m-%d')
    
    async def update_single_asset(self, symbol: str, dry_run: bool = False) -> Dict[str, Any]:
        """
        Update a single asset with incremental data.
        
        Args:
            symbol: Asset symbol
            dry_run: If True, don't actually store data
            
        Returns:
            Dictionary with update results
        """
        update_start_time = datetime.now(timezone.utc)
        result = {
            'symbol': symbol,
            'success': False,
            'new_observations': 0,
            'updated_observations': 0,
            'start_date': None,
            'end_date': None,
            'error_message': None,
            'duration_seconds': 0,
            'dry_run': dry_run
        }
        
        try:
            self.logger.info(f"Starting update for asset: {symbol}")
            
            # Calculate incremental update start date
            start_date = await self.calculate_update_start_date(symbol)
            result['start_date'] = start_date
            result['end_date'] = date.today().strftime('%Y-%m-%d')
            
            if dry_run:
                self.logger.info(f"DRY RUN: Would update {symbol} from {start_date}")
                result['success'] = True
                return result
            
            # Use bulk load with start date for incremental update
            load_result = await self.db_integration.bulk_load_asset(symbol, period='max', start_date=start_date)
            
            if load_result['success']:
                result['success'] = True
                result['new_observations'] = load_result['observations_inserted']
                result['updated_observations'] = load_result['observations_updated']
                self.successful_updates += 1
                self.total_new_observations += result['new_observations']
                
                if result['new_observations'] > 0:
                    self.logger.info(
                        f"✅ {symbol}: {result['new_observations']} new observations"
                    )
                else:
                    self.logger.info(f"✅ {symbol}: Up to date (no new data)")
            else:
                result['error_message'] = load_result.get('error_message', 'Unknown error')
                self.failed_updates += 1
                self.logger.error(f"❌ {symbol}: {result['error_message']}")
            
        except Exception as e:
            result['error_message'] = str(e)
            self.failed_updates += 1
            self.logger.error(f"❌ Unexpected error updating {symbol}: {e}")
        
        finally:
            result['duration_seconds'] = (datetime.now(timezone.utc) - update_start_time).total_seconds()
        
        return result
    
    async def update_all_assets(
        self, 
        symbols_list: Optional[List[str]] = None,
        dry_run: bool = False,
        max_concurrent: int = 3
    ) -> Dict[str, Any]:
        """
        Update all or specified assets with incremental data.
        
        Args:
            symbols_list: Optional list of specific symbols to update
            dry_run: If True, don't actually store data
            max_concurrent: Maximum concurrent API calls
            
        Returns:
            Comprehensive results dictionary
        """
        start_time = datetime.now(timezone.utc)
        
        # Determine which assets to update
        if symbols_list:
            assets_to_update = {symbol: self.all_assets.get(symbol, symbol) for symbol in symbols_list}
        else:
            assets_to_update = self.all_assets.copy()
        
        self.logger.info(f"🔄 Starting daily update for {len(assets_to_update)} assets")
        if dry_run:
            self.logger.info("🔍 DRY RUN MODE - No data will be stored")
        
        # Update assets with controlled concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def update_with_semaphore(symbol: str):
            async with semaphore:
                return await self.update_single_asset(symbol, dry_run)
        
        # Execute updates
        tasks = [update_with_semaphore(symbol) for symbol in assets_to_update.keys()]
        asset_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_results = []
        failed_results = []
        
        for i, result in enumerate(asset_results):
            if isinstance(result, Exception):
                symbol = list(assets_to_update.keys())[i]
                failed_result = {
                    'symbol': symbol,
                    'success': False,
                    'error_message': str(result),
                    'new_observations': 0
                }
                failed_results.append(failed_result)
                self.failed_updates += 1
            elif result['success']:
                successful_results.append(result)
            else:
                failed_results.append(result)
        
        # Calculate final statistics
        end_time = datetime.now(timezone.utc)
        total_duration = (end_time - start_time).total_seconds()
        
        final_results = {
            'start_time': start_time,
            'end_time': end_time,
            'total_duration_seconds': total_duration,
            'total_assets_attempted': len(assets_to_update),
            'successful_updates': self.successful_updates,
            'failed_updates': self.failed_updates,
            'total_new_observations': self.total_new_observations,
            'successful_results': successful_results,
            'failed_results': failed_results,
            'success_rate': (self.successful_updates / len(assets_to_update) * 100) if assets_to_update else 0,
            'dry_run': dry_run
        }
        
        return final_results
    
    def print_update_summary(self, results: Dict[str, Any]) -> None:
        """Print a comprehensive summary of the daily update."""
        
        print("\n" + "="*80)
        print("🔄 YAHOO FINANCE DAILY UPDATE SUMMARY")
        print("="*80)
        
        mode_indicator = " (DRY RUN)" if results['dry_run'] else ""
        print(f"📅 Started:  {results['start_time'].strftime('%Y-%m-%d %H:%M:%S UTC')}{mode_indicator}")
        print(f"📅 Finished: {results['end_time'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"⏱️  Duration: {results['total_duration_seconds']:.1f} seconds")
        print(f"📊 Success Rate: {results['success_rate']:.1f}%")
        
        print(f"\n📈 UPDATE STATISTICS:")
        print(f"   Total Assets Attempted: {results['total_assets_attempted']}")
        print(f"   ✅ Successful Updates: {results['successful_updates']}")
        print(f"   ❌ Failed Updates: {results['failed_updates']}")
        print(f"   📊 New Observations: {results['total_new_observations']:,}")
        
        # Show assets with new data
        assets_with_data = [r for r in results['successful_results'] if r['new_observations'] > 0]
        if assets_with_data:
            print(f"\n📊 ASSETS WITH NEW DATA:")
            for result in sorted(assets_with_data, key=lambda x: x['new_observations'], reverse=True):
                print(f"   {result['symbol']}: {result['new_observations']} new observations")
        
        # Show up-to-date assets
        up_to_date_assets = [r for r in results['successful_results'] if r['new_observations'] == 0]
        if up_to_date_assets:
            print(f"\n✅ UP-TO-DATE ASSETS ({len(up_to_date_assets)}):")
            asset_names = [r['symbol'] for r in up_to_date_assets]
            # Print in rows of 8 for readability
            for i in range(0, len(asset_names), 8):
                row = asset_names[i:i+8]
                print(f"   {', '.join(row)}")
        
        # Show failed assets
        if results['failed_results']:
            print(f"\n❌ FAILED ASSETS:")
            for result in results['failed_results']:
                print(f"   {result['symbol']}: {result.get('error_message', 'Unknown error')}")
        
        print("\n" + "="*80)
        if results['failed_updates'] == 0:
            print("🎉 ALL ASSETS UPDATED SUCCESSFULLY!")
        else:
            print(f"⚠️  {results['failed_updates']} assets failed to update. Check logs for details.")
        print("="*80)
    
    async def send_notification(self, results: Dict[str, Any]) -> None:
        """Send notification about update results (email, Slack, etc.)"""
        if not self.notification_config:
            return
        
        # TODO: Implement email/Slack notifications
        # This is where you'd integrate with your notification system
        pass
    
    async def close(self) -> None:
        """Clean up resources"""
        await self.db_integration.close()


async def main():
    """Main entry point for daily updates"""
    parser = argparse.ArgumentParser(
        description='Daily incremental updates for Yahoo Finance market data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python yahoo_daily_updater.py                     # Update all assets
  python yahoo_daily_updater.py --dry-run           # Show what would be updated
  python yahoo_daily_updater.py --symbols SPY QQQ   # Update specific assets
  
Cron Examples:
  # Daily at 7 PM EST (after market close)
  0 19 * * * /usr/bin/python3 /path/to/yahoo_daily_updater.py
        """
    )
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        help='Update specific assets by symbol (e.g., SPY QQQ AAPL)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be updated without making changes'
    )
    
    parser.add_argument(
        '--database-url',
        default=os.getenv('DATABASE_URL', 'postgresql+psycopg://postgres:fred_password@localhost:5432/postgres'),
        help='Database connection URL'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    parser.add_argument(
        '--max-concurrent',
        type=int,
        default=3,
        help='Maximum concurrent API calls'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Create daily updater
    updater = YahooDailyUpdater(args.database_url)
    
    try:
        # Initialize database connection
        if not await updater.initialize():
            logger.error("Failed to initialize database connection")
            return 1
        
        # Run updates
        results = await updater.update_all_assets(
            symbols_list=args.symbols,
            dry_run=args.dry_run,
            max_concurrent=args.max_concurrent
        )
        
        # Print summary
        updater.print_update_summary(results)
        
        # Send notifications if configured
        await updater.send_notification(results)
        
        # Return appropriate exit code for cron monitoring
        return 0 if results['failed_updates'] == 0 else 1
        
    except KeyboardInterrupt:
        logger.info("Daily update interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Daily update failed with unexpected error: {e}")
        return 1
    finally:
        await updater.close()


if __name__ == "__main__":
    """Run the daily updater"""
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
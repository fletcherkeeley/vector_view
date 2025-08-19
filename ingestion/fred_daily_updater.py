#!/usr/bin/env python3
"""
FRED Daily Update System

Automated daily updates for FRED economic data. Designed to run as a scheduled job
to keep the database current with the latest economic indicators.

Key Features:
- Incremental updates (only fetch new data since last update)
- All series from bulk loader (interest rates, employment, inflation, etc.)
- Graceful error handling and retry logic
- Comprehensive logging and monitoring
- Email/notification support for failures
- Safe to run multiple times (idempotent)

Usage:
    python fred_daily_updater.py                    # Update all series
    python fred_daily_updater.py --series FEDFUNDS  # Update specific series
    python fred_daily_updater.py --dry-run          # Show what would be updated
    
Cron Examples:
    # Run daily at 6 AM EST (after markets open and data is available)
    0 6 * * * /path/to/python /path/to/fred_daily_updater.py
    
    # Run twice daily (morning and evening)
    0 6,18 * * * /path/to/python /path/to/fred_daily_updater.py
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
from fred_database_integration import FredDatabaseIntegration
from fred_bulk_loader import FredBulkLoader

# Load environment variables
load_dotenv()

# Configure logging for scheduled runs
def setup_logging(log_level: str = 'INFO'):
    """Setup logging with both file and console output"""
    log_dir = Path(__file__).parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"fred_daily_update_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


class FredDailyUpdater:
    """
    Manages daily incremental updates of FRED economic data.
    """
    
    def __init__(self, database_url: str, notification_config: Optional[Dict] = None):
        """
        Initialize the daily updater.
        
        Args:
            database_url: PostgreSQL connection string
            notification_config: Optional config for email/slack notifications
        """
        self.database_url = database_url
        self.db_integration = FredDatabaseIntegration(database_url)
        self.notification_config = notification_config or {}
        
        # Get all series from bulk loader
        bulk_loader = FredBulkLoader(database_url)
        self.all_series = bulk_loader.get_all_series()
        
        # Update statistics
        self.update_results = {}
        self.total_new_observations = 0
        self.successful_updates = 0
        self.failed_updates = 0
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("FRED Daily Updater initialized")
    
    async def initialize(self) -> bool:
        """Initialize database connection"""
        success = await self.db_integration.initialize()
        if success:
            self.logger.info("Database connection initialized for daily updates")
        return success
    
    async def get_last_observation_date(self, series_id: str) -> Optional[date]:
        """
        Get the date of the most recent observation for a series.
        
        Args:
            series_id: FRED series identifier
            
        Returns:
            Date of last observation, or None if no data exists
        """
        try:
            async with self.db_integration.AsyncSessionLocal() as session:
                from sqlalchemy import select, func
                from unified_database_setup import TimeSeriesObservation
                
                result = await session.execute(
                    select(func.max(TimeSeriesObservation.observation_date))
                    .where(TimeSeriesObservation.series_id == series_id)
                )
                
                last_date = result.scalar()
                return last_date
                
        except Exception as e:
            self.logger.error(f"Error getting last observation date for {series_id}: {e}")
            return None
    
    async def calculate_update_start_date(self, series_id: str) -> str:
        """
        Calculate the appropriate start date for incremental updates.
        
        Args:
            series_id: FRED series identifier
            
        Returns:
            Start date string in YYYY-MM-DD format
        """
        last_observation = await self.get_last_observation_date(series_id)
        
        if last_observation:
            # Start from the day after last observation
            # But go back a few days to catch any revisions
            start_date = last_observation - timedelta(days=7)  # 7-day overlap for revisions
            self.logger.debug(f"{series_id}: Last observation {last_observation}, starting from {start_date}")
        else:
            # No existing data, get last 30 days
            start_date = date.today() - timedelta(days=30)
            self.logger.info(f"{series_id}: No existing data, starting from {start_date}")
        
        return start_date.strftime('%Y-%m-%d')
    
    async def update_single_series(self, series_id: str, dry_run: bool = False) -> Dict[str, Any]:
        """
        Update a single FRED series with incremental data.
        
        Args:
            series_id: FRED series identifier
            dry_run: If True, don't actually store data
            
        Returns:
            Dictionary with update results
        """
        update_start_time = datetime.now(timezone.utc)
        result = {
            'series_id': series_id,
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
            self.logger.info(f"Starting update for series: {series_id}")
            
            # Calculate incremental update start date
            start_date = await self.calculate_update_start_date(series_id)
            result['start_date'] = start_date
            result['end_date'] = date.today().strftime('%Y-%m-%d')
            
            if dry_run:
                self.logger.info(f"DRY RUN: Would update {series_id} from {start_date}")
                result['success'] = True
                return result
            
            # Use bulk load with start date for incremental update
            load_result = await self.db_integration.bulk_load_series(series_id, start_date)
            
            if load_result['success']:
                result['success'] = True
                result['new_observations'] = load_result['observations_inserted']
                result['updated_observations'] = load_result['observations_updated']
                self.successful_updates += 1
                self.total_new_observations += result['new_observations']
                
                self.logger.info(
                    f"✅ {series_id}: {result['new_observations']} new observations"
                )
            else:
                result['error_message'] = load_result.get('error_message', 'Unknown error')
                self.failed_updates += 1
                self.logger.error(f"❌ {series_id}: {result['error_message']}")
            
        except Exception as e:
            result['error_message'] = str(e)
            self.failed_updates += 1
            self.logger.error(f"❌ Unexpected error updating {series_id}: {e}")
        
        finally:
            result['duration_seconds'] = (datetime.now(timezone.utc) - update_start_time).total_seconds()
        
        return result
    
    async def update_all_series(
        self, 
        series_list: Optional[List[str]] = None,
        dry_run: bool = False,
        max_concurrent: int = 3
    ) -> Dict[str, Any]:
        """
        Update all or specified FRED series with incremental data.
        
        Args:
            series_list: Optional list of specific series to update
            dry_run: If True, don't actually store data
            max_concurrent: Maximum concurrent API calls
            
        Returns:
            Comprehensive results dictionary
        """
        start_time = datetime.now(timezone.utc)
        
        # Determine which series to update
        if series_list:
            series_to_update = {sid: self.all_series[sid] for sid in series_list if sid in self.all_series}
        else:
            series_to_update = self.all_series.copy()
        
        self.logger.info(f"🔄 Starting daily update for {len(series_to_update)} series")
        if dry_run:
            self.logger.info("🔍 DRY RUN MODE - No data will be stored")
        
        # Update series with controlled concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def update_with_semaphore(series_id: str):
            async with semaphore:
                return await self.update_single_series(series_id, dry_run)
        
        # Execute updates
        tasks = [update_with_semaphore(series_id) for series_id in series_to_update.keys()]
        series_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_results = []
        failed_results = []
        
        for i, result in enumerate(series_results):
            if isinstance(result, Exception):
                series_id = list(series_to_update.keys())[i]
                failed_result = {
                    'series_id': series_id,
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
            'total_series_attempted': len(series_to_update),
            'successful_updates': self.successful_updates,
            'failed_updates': self.failed_updates,
            'total_new_observations': self.total_new_observations,
            'successful_results': successful_results,
            'failed_results': failed_results,
            'success_rate': (self.successful_updates / len(series_to_update) * 100) if series_to_update else 0,
            'dry_run': dry_run
        }
        
        return final_results
    
    def print_update_summary(self, results: Dict[str, Any]) -> None:
        """Print a comprehensive summary of the daily update."""
        
        print("\n" + "="*80)
        print("🔄 FRED DAILY UPDATE SUMMARY")
        print("="*80)
        
        mode_indicator = " (DRY RUN)" if results['dry_run'] else ""
        print(f"📅 Started:  {results['start_time'].strftime('%Y-%m-%d %H:%M:%S UTC')}{mode_indicator}")
        print(f"📅 Finished: {results['end_time'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"⏱️  Duration: {results['total_duration_seconds']:.1f} seconds")
        print(f"📊 Success Rate: {results['success_rate']:.1f}%")
        
        print(f"\n📈 UPDATE STATISTICS:")
        print(f"   Total Series Attempted: {results['total_series_attempted']}")
        print(f"   ✅ Successful Updates: {results['successful_updates']}")
        print(f"   ❌ Failed Updates: {results['failed_updates']}")
        print(f"   📊 New Observations: {results['total_new_observations']:,}")
        
        # Show series with new data
        series_with_data = [r for r in results['successful_results'] if r['new_observations'] > 0]
        if series_with_data:
            print(f"\n📊 SERIES WITH NEW DATA:")
            for result in sorted(series_with_data, key=lambda x: x['new_observations'], reverse=True):
                print(f"   {result['series_id']}: {result['new_observations']} new observations")
        
        # Show failed series
        if results['failed_results']:
            print(f"\n❌ FAILED SERIES:")
            for result in results['failed_results']:
                print(f"   {result['series_id']}: {result.get('error_message', 'Unknown error')}")
        
        print("\n" + "="*80)
        if results['failed_updates'] == 0:
            print("🎉 ALL SERIES UPDATED SUCCESSFULLY!")
        else:
            print(f"⚠️  {results['failed_updates']} series failed to update. Check logs for details.")
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
        description='Daily incremental updates for FRED economic data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fred_daily_updater.py                     # Update all series
  python fred_daily_updater.py --dry-run           # Show what would be updated
  python fred_daily_updater.py --series FEDFUNDS   # Update specific series
  
Cron Examples:
  # Daily at 6 AM EST
  0 6 * * * /usr/bin/python3 /path/to/fred_daily_updater.py
        """
    )
    
    parser.add_argument(
        '--series',
        nargs='+',
        help='Update specific series by ID (e.g., FEDFUNDS GDP UNRATE)'
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
    updater = FredDailyUpdater(args.database_url)
    
    try:
        # Initialize database connection
        if not await updater.initialize():
            logger.error("Failed to initialize database connection")
            return 1
        
        # Run updates
        results = await updater.update_all_series(
            series_list=args.series,
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
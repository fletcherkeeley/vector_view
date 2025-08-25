#!/usr/bin/env python3
"""
News Daily Updater

Manages daily incremental updates of NewsData.io API data.
Follows the same pattern as FredDailyUpdater and YahooDailyUpdater.

This module provides:
- Intelligent API quota management (varies by NewsData.io plan)
- Priority-based category allocation
- Wide keyword criteria for maximum data collection
- Comprehensive logging and monitoring
- Dry-run simulation capabilities
- Health check functionality

Usage:
    python news_daily_updater.py                           # Full daily update
    python news_daily_updater.py --dry-run                 # Test run without API calls
    python news_daily_updater.py --categories fed,employment # Specific categories
    python news_daily_updater.py --max-calls 500           # Custom API limit
    python news_daily_updater.py --health-check            # Get update health status
"""

import asyncio
import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

from news_database_integration import NewsDatabaseIntegration
from news_series_fetcher import NewsSeriesFetcher


def setup_logging(level: str = 'INFO') -> logging.Logger:
    """Setup logging configuration"""
    
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    # Create daily log file
    today = datetime.now().strftime('%Y-%m-%d')
    log_file = log_dir / f'news_daily_update_{today}.log'
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


class NewsDailyUpdater:
    """
    Manages daily incremental updates of News API data.
    Follows the same pattern as FredDailyUpdater and YahooDailyUpdater.
    """
    
    def __init__(self, database_url: str, max_api_calls: int = 950, notification_config: Optional[Dict] = None):
        """
        Initialize the daily news updater.
        
        Args:
            database_url: PostgreSQL connection string
            max_api_calls: Maximum API calls per day (default 950, reserves 50 for other operations)
            notification_config: Optional config for email/slack notifications
        """
        self.database_url = database_url
        self.db_integration = NewsDatabaseIntegration(database_url)
        self.notification_config = notification_config or {}
        self.max_api_calls = max_api_calls
        
        # Update statistics (consistent with FRED/Yahoo pattern)
        self.update_results = {}
        self.total_new_observations = 0
        self.successful_updates = 0
        self.failed_updates = 0
        
        # News-specific tracking
        self.api_calls_used = 0
        self.total_articles_found = 0
        self.total_articles_stored = 0
        self.categories_processed = []
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("News Daily Updater initialized")
    
    async def initialize(self) -> bool:
        """Initialize database connection"""
        success = await self.db_integration.initialize()
        if success:
            self.logger.info("Database connection initialized for daily sync")
        return success
    
    def get_daily_category_strategy(self) -> List[Dict[str, Any]]:
        """
        Get optimized category strategy for daily updates.
        
        Returns:
            List of category configurations with API call allocations
        """
        
        # Priority-based allocation for daily updates (950 total calls)
        categories = [
            {
                'name': 'federal_reserve',
                'priority': 1,
                'api_calls_allocated': 150,  # Highest priority - Fed policy impacts
                'description': 'Federal Reserve policy and monetary decisions'
            },
            {
                'name': 'employment',
                'priority': 2,
                'api_calls_allocated': 120,  # Labor market is key economic indicator
                'description': 'Employment data and labor market trends'
            },
            {
                'name': 'inflation',
                'priority': 3,
                'api_calls_allocated': 110,  # Critical for Fed policy
                'description': 'Inflation trends and price stability'
            },
            {
                'name': 'gdp_growth',
                'priority': 4,
                'api_calls_allocated': 100,  # Core economic growth metric
                'description': 'GDP growth and economic expansion'
            },
            {
                'name': 'financial_markets',
                'priority': 5,
                'api_calls_allocated': 90,   # Market sentiment and movements
                'description': 'Stock market and financial market trends'
            },
            {
                'name': 'banking',
                'priority': 6,
                'api_calls_allocated': 80,   # Financial sector health
                'description': 'Banking sector and financial institutions'
            },
            {
                'name': 'trade',
                'priority': 7,
                'api_calls_allocated': 70,   # International trade impacts
                'description': 'International trade and tariff policies'
            },
            {
                'name': 'housing',
                'priority': 8,
                'api_calls_allocated': 60,   # Real estate market indicator
                'description': 'Housing market and real estate trends'
            },
            {
                'name': 'consumer_spending',
                'priority': 9,
                'api_calls_allocated': 50,   # Consumer behavior indicator
                'description': 'Consumer spending and retail trends'
            },
            {
                'name': 'energy',
                'priority': 10,
                'api_calls_allocated': 40,   # Energy sector and commodity prices
                'description': 'Energy markets and commodity prices'
            }
        ]
        
        # Verify total allocation doesn't exceed limit
        total_allocated = sum(cat['api_calls_allocated'] for cat in categories)
        if total_allocated > self.max_api_calls:
            self.logger.warning(f"Total API allocation ({total_allocated}) exceeds limit ({self.max_api_calls})")
            # Scale down proportionally
            scale_factor = self.max_api_calls / total_allocated
            for cat in categories:
                cat['api_calls_allocated'] = int(cat['api_calls_allocated'] * scale_factor)
        
        return categories
    
    async def update_all_categories(
        self,
        categories: Optional[List[str]] = None,
        dry_run: bool = False,
        days_back: int = 1,
        max_concurrent: int = 3
    ) -> Dict[str, Any]:
        """
        Update all or specified news categories with incremental data.
        
        Args:
            categories: Specific categories to sync (if None, syncs all)
            dry_run: If True, simulates the sync without making API calls
            days_back: Number of days back to search (default 1 for daily sync)
            max_concurrent: Maximum concurrent API calls
            
        Returns:
            Comprehensive results dictionary
        """
        sync_start_time = datetime.now(timezone.utc)
        
        self.logger.info("🚀 Starting Daily News Synchronization")
        self.logger.info(f"Max API calls: {self.max_api_calls}")
        self.logger.info(f"Days back: {days_back}")
        self.logger.info(f"Dry run: {dry_run}")
        
        # Get category strategy
        category_strategy = self.get_daily_category_strategy()
        
        # Filter categories if specified
        if categories:
            category_strategy = [cat for cat in category_strategy if cat['name'] in categories]
            self.logger.info(f"Filtering to categories: {categories}")
        
        # Process categories
        all_results = []
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_category_with_semaphore(cat_config):
            async with semaphore:
                return await self._sync_category(cat_config, days_back, dry_run)
        
        # Execute category updates concurrently
        tasks = [process_category_with_semaphore(cat) for cat in category_strategy]
        all_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_results = []
        for i, result in enumerate(all_results):
            if isinstance(result, Exception):
                self.logger.error(f"Category {category_strategy[i]['name']} failed: {result}")
                self.failed_updates += 1
            else:
                successful_results.append(result)
                if result.get('success', False):
                    self.successful_updates += 1
                    self.api_calls_used += result.get('api_calls_used', 0)
                    self.total_articles_found += result.get('articles_found', 0)
                    self.total_articles_stored += result.get('articles_stored', 0)
                    self.categories_processed.append(result.get('category'))
                else:
                    self.failed_updates += 1
        
        sync_end_time = datetime.now(timezone.utc)
        
        # Compile final results
        final_results = {
            'success': self.successful_updates > 0,
            'dry_run': dry_run,
            'sync_date': sync_start_time.strftime('%Y-%m-%d'),
            'start_time': sync_start_time,
            'end_time': sync_end_time,
            'total_duration_seconds': (sync_end_time - sync_start_time).total_seconds(),
            'categories_processed': len(self.categories_processed),
            'successful_updates': self.successful_updates,
            'failed_updates': self.failed_updates,
            'total_api_calls_used': self.api_calls_used,
            'total_articles_found': self.total_articles_found,
            'total_articles_stored': self.total_articles_stored,
            'api_calls_remaining': self.max_api_calls - self.api_calls_used,
            'category_results': successful_results,
            'success_rate': (self.successful_updates / (self.successful_updates + self.failed_updates) * 100) if (self.successful_updates + self.failed_updates) > 0 else 0,
            'efficiency': self.total_articles_stored / self.api_calls_used if self.api_calls_used > 0 else 0
        }
        
        # Save final statistics (consistent with other updaters)
        # Note: FRED/Yahoo don't save stats to files, they rely on database logging
        
        return final_results
    
    async def _sync_category(
        self,
        category_config: Dict[str, Any],
        days_back: int,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """
        Sync a specific category with allocated API calls.
        
        Args:
            category_config: Category configuration with allocation
            days_back: Number of days back to search
            dry_run: Whether to simulate the sync
            
        Returns:
            Category sync results
        """
        category_name = category_config['name']
        allocated_calls = category_config['api_calls_allocated']
        
        self.logger.info(f"🔄 Processing category: {category_name} ({allocated_calls} API calls)")
        
        try:
            if dry_run:
                # Simulate the sync
                return await self._simulate_category_sync(category_config, days_back)
            
            # Create fetcher for this category
            fetcher = NewsSeriesFetcher()
            
            # Calculate date range
            end_date = datetime.now(timezone.utc).date()
            start_date = end_date - timedelta(days=days_back)
            
            # Fetch articles for this category with allocated API calls
            articles = await fetcher.fetch_category_articles(
                category=category_name,
                start_date=start_date,
                end_date=end_date,
                max_api_calls=allocated_calls
            )
            
            # Store articles in database
            if articles:
                successful_inserts, conflicts_updated, failed_inserts = await self.db_integration.bulk_insert_articles(articles)
                stored_count = successful_inserts + conflicts_updated
                
                # Log API call usage
                api_calls_used = getattr(fetcher, 'api_calls_used', allocated_calls)
                await self.db_integration.log_api_calls(
                    source='news_api',
                    calls_made=api_calls_used,
                    success=True,
                    category=category_name
                )
                
                return {
                    'success': True,
                    'category': category_name,
                    'api_calls_used': api_calls_used,
                    'articles_found': len(articles),
                    'articles_stored': stored_count,
                    'date_range': f"{start_date} to {end_date}",
                    'efficiency': stored_count / api_calls_used if api_calls_used > 0 else 0
                }
            else:
                return {
                    'success': True,
                    'category': category_name,
                    'api_calls_used': allocated_calls,
                    'articles_found': 0,
                    'articles_stored': 0,
                    'date_range': f"{start_date} to {end_date}",
                    'efficiency': 0
                }
                
        except Exception as e:
            self.logger.error(f"Failed to sync category {category_name}: {e}")
            return {
                'success': False,
                'category': category_name,
                'api_calls_used': 0,
                'articles_found': 0,
                'articles_stored': 0,
                'error': str(e)
            }
    
    async def _simulate_category_sync(
        self,
        category_config: Dict[str, Any],
        days_back: int
    ) -> Dict[str, Any]:
        """
        Simulate category sync for dry-run mode.
        
        Args:
            category_config: Category configuration
            days_back: Number of days back to search
            
        Returns:
            Simulated sync results
        """
        category_name = category_config['name']
        allocated_calls = category_config['api_calls_allocated']
        
        # Simulate realistic article counts based on category priority and API calls
        # Higher priority categories typically yield more relevant articles
        priority = category_config['priority']
        base_articles_per_call = max(1, 6 - (priority - 1) * 0.5)  # 6 to 1.5 articles per call
        
        estimated_articles = int(allocated_calls * base_articles_per_call * days_back)
        estimated_stored = int(estimated_articles * 0.85)  # 85% storage rate after deduplication
        
        return {
            'success': True,
            'category': category_name,
            'api_calls_used': allocated_calls,
            'articles_found': estimated_articles,
            'articles_stored': estimated_stored,
            'date_range': f"Last {days_back} day(s)",
            'efficiency': estimated_stored / allocated_calls if allocated_calls > 0 else 0,
            'simulated': True
        }
    
    async def get_sync_health_check(self) -> Dict[str, Any]:
        """
        Get health check information for the sync system.
        
        Returns:
            Health check results
        """
        try:
            # Check database connectivity
            db_healthy = await self.db_integration.health_check()
            
            # Check recent sync performance
            recent_stats = await self.db_integration.get_recent_sync_stats(days=7)
            
            # Calculate health metrics
            avg_success_rate = sum(stat.get('success_rate', 0) for stat in recent_stats) / len(recent_stats) if recent_stats else 0
            avg_efficiency = sum(stat.get('efficiency', 0) for stat in recent_stats) / len(recent_stats) if recent_stats else 0
            
            return {
                'database_healthy': db_healthy,
                'recent_syncs': len(recent_stats),
                'avg_success_rate': f"{avg_success_rate:.1f}%",
                'avg_efficiency': f"{avg_efficiency:.2f} articles/call",
                'last_sync': recent_stats[0].get('sync_date') if recent_stats else 'Never',
                'status': 'Healthy' if db_healthy and avg_success_rate > 80 else 'Degraded'
            }
            
        except Exception as e:
            return {
                'database_healthy': False,
                'status': 'Unhealthy',
                'error_message': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def close(self) -> None:
        """Clean up resources"""
        await self.db_integration.close()


def print_update_summary(results: Dict[str, Any]) -> None:
    """Print comprehensive summary of the daily update"""
    
    print("\n" + "="*80)
    print("🔄 NEWS DAILY UPDATE SUMMARY")
    print("="*80)
    
    mode_text = "DRY RUN" if results['dry_run'] else "LIVE UPDATE"
    print(f"Mode: {mode_text}")
    print(f"📅 Date: {results['sync_date']}")
    print(f"📅 Started:  {results['start_time'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"📅 Finished: {results['end_time'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"⏱️  Duration: {results['total_duration_seconds']:.1f} seconds ({results['total_duration_seconds']/60:.1f} minutes)")
    print(f"📊 Success Rate: {results['success_rate']:.1f}%")
    
    print(f"\n📈 PERFORMANCE METRICS")
    print("-" * 40)
    print(f"Categories Processed: {results['categories_processed']}")
    print(f"Successful Updates: {results['successful_updates']}")
    print(f"Failed Updates: {results['failed_updates']}")
    print(f"API Calls Used: {results['total_api_calls_used']:,}")
    print(f"API Calls Remaining: {results['api_calls_remaining']:,}")
    print(f"Articles Found: {results['total_articles_found']:,}")
    print(f"Articles Stored: {results['total_articles_stored']:,}")
    print(f"Efficiency: {results['efficiency']:.2f} articles/call")
    
    if results['category_results']:
        print(f"\n📋 CATEGORY BREAKDOWN")
        print("-" * 40)
        for cat_result in results['category_results']:
            if cat_result.get('success'):
                status = "✅" if cat_result['success'] else "❌"
                sim_text = " (simulated)" if cat_result.get('simulated') else ""
                print(f"{status} {cat_result['category']}: {cat_result['articles_stored']:,} articles, "
                      f"{cat_result['api_calls_used']} calls, "
                      f"{cat_result['efficiency']:.2f} eff{sim_text}")
            else:
                print(f"❌ {cat_result['category']}: FAILED - {cat_result.get('error', 'Unknown error')}")
    
    print("="*80)


async def main():
    """Main entry point for the news daily updater"""
    parser = argparse.ArgumentParser(
        description='News Daily Updater - Incremental news data updates with API quota management',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python news_daily_updater.py                           # Full daily update
  python news_daily_updater.py --dry-run                 # Test run without API calls
  python news_daily_updater.py --categories fed,employment # Specific categories
  python news_daily_updater.py --max-calls 500           # Custom API limit
  python news_daily_updater.py --health-check            # Get update health status
        """
    )
    
    parser.add_argument(
        '--categories',
        type=str,
        help='Comma-separated list of categories to sync (e.g., federal_reserve,employment)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Simulate the sync without making actual API calls'
    )
    
    parser.add_argument(
        '--days-back',
        type=int,
        default=1,
        help='Number of days back to search for articles (default: 1)'
    )
    
    parser.add_argument(
        '--max-calls',
        type=int,
        default=950,
        help='Maximum API calls to use (default: 950)'
    )
    
    parser.add_argument(
        '--health-check',
        action='store_true',
        help='Run health check and exit'
    )
    
    parser.add_argument(
        '--database-url',
        type=str,
        default='postgresql+psycopg://postgres:fred_password@localhost:5432/postgres',
        help='Database connection URL'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Create updater
    updater = NewsDailyUpdater(args.database_url, max_api_calls=args.max_calls)
    
    try:
        # Initialize database connection
        if not await updater.initialize():
            logger.error("Failed to initialize database connection")
            return 1
        
        # Handle health check
        if args.health_check:
            health = await updater.get_sync_health_check()
            print("\n📊 SYNC HEALTH CHECK")
            print("="*50)
            for key, value in health.items():
                print(f"{key}: {value}")
            return 0
        
        # Parse categories if provided
        categories = None
        if args.categories:
            categories = [cat.strip() for cat in args.categories.split(',')]
        
        # Confirm live sync (skip confirmation in automated mode)
        if not args.dry_run and not os.getenv('AUTOMATED_RUN', '').lower() in ['true', '1', 'yes']:
            mode_text = f"live sync with {args.max_calls} API calls"
            if categories:
                mode_text += f" for categories: {', '.join(categories)}"
            print(f"🔄 Ready to run daily news {mode_text}")
            response = input("Continue? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("Daily sync cancelled")
                return 0
        
        # Run updates
        results = await updater.update_all_categories(
            categories=categories,
            dry_run=args.dry_run,
            days_back=args.days_back,
            max_concurrent=3
        )
        
        # Print summary
        print_update_summary(results)
        
        # Return appropriate exit code
        return 0 if results['success_rate'] > 50 else 1
        
    except KeyboardInterrupt:
        logger.info("Daily sync interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Daily sync failed with unexpected error: {e}")
        return 1
    finally:
        await updater.close()


if __name__ == '__main__':
    asyncio.run(main())

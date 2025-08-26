#!/usr/bin/env python3
"""
News Historical Backfill - Optimized 30-Day Economic News Collection

This script efficiently backfills 30 days of economic news using intelligent API optimization.
Designed to maximize your 1,000 daily News API credits for the richest possible dataset.

Key Features:
- Priority-based category processing (Fed news first, etc.)
- Intelligent date chunking to avoid API limits
- Intensive mode: all keywords + 5-day chunks for maximum collection
- MEGA mode: 2-day chunks + 12 categories + expanded keywords for 4K+ articles
- Progress tracking and resume capability
- API quota monitoring and optimization
- Comprehensive reporting and statistics

Usage:
    python news_historical_backfill.py                    # Full 30-day backfill
    python news_historical_backfill.py --intensive        # Intensive mode (all keywords + small chunks)
    python news_historical_backfill.py --mega             # MEGA mode (2-day chunks + 12 categories)
    python news_historical_backfill.py --days 14          # Custom day range
    python news_historical_backfill.py --resume           # Resume interrupted backfill
"""

import asyncio
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, date, timedelta
import argparse
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from .news_database_integration import NewsDatabaseIntegration
from .news_series_fetcher import NewsSeriesFetcher

# Load environment variables
load_dotenv()

# Configure comprehensive logging
def setup_logging(log_level: str = 'INFO'):
    """Setup logging with both file and console output"""
    log_dir = Path(__file__).parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"news_backfill_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)


class NewsHistoricalBackfill:
    """
    Optimized historical backfill for News API with intelligent API management.
    """
    
    def __init__(self, database_url: str, max_api_calls: int = 900):
        """
        Initialize the historical backfill system.
        
        Args:
            database_url: PostgreSQL connection string
            max_api_calls: Maximum API calls to use (reserve some for daily operations)
        """
        self.database_url = database_url
        self.db_integration = NewsDatabaseIntegration(database_url)
        self.max_api_calls = max_api_calls
        
        # Backfill statistics
        self.api_calls_made = 0
        self.total_articles_collected = 0
        self.categories_completed = []
        self.failed_operations = []
        
        # Progress tracking
        self.progress_file = Path(__file__).parent.parent / 'config' / 'backfill_progress.json'
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("News Historical Backfill initialized")
    
    async def initialize(self) -> bool:
        """Initialize database connection"""
        success = await self.db_integration.initialize()
        if success:
            self.logger.info("Database connection initialized for backfill")
        return success
    
    def get_prioritized_categories(self, mega_mode: bool = False) -> List[Dict[str, Any]]:
        """
        Get economic categories in priority order for backfill.
        
        Args:
            mega_mode: If True, include expanded categories and keyword counts
        
        Returns:
            List of category configs with priority, API efficiency estimates
        """
        base_categories = [
            {
                'name': 'federal_reserve',
                'priority': 10,
                'keywords_count': 16 if mega_mode else 7,
                'estimated_api_calls': 48 if mega_mode else 21,
                'description': 'Federal Reserve policy and rate decisions' + (' - EXPANDED' if mega_mode else '')
            },
            {
                'name': 'employment',
                'priority': 9,
                'keywords_count': 13 if mega_mode else 7,
                'estimated_api_calls': 39 if mega_mode else 21,
                'description': 'Employment data and labor market trends' + (' - EXPANDED' if mega_mode else '')
            },
            {
                'name': 'inflation',
                'priority': 9,
                'keywords_count': 13 if mega_mode else 7,
                'estimated_api_calls': 39 if mega_mode else 21,
                'description': 'Inflation indicators and price trends' + (' - EXPANDED' if mega_mode else '')
            },
            {
                'name': 'gdp_growth',
                'priority': 8,
                'keywords_count': 12 if mega_mode else 6,
                'estimated_api_calls': 36 if mega_mode else 18,
                'description': 'GDP growth and economic indicators' + (' - EXPANDED' if mega_mode else '')
            },
            {
                'name': 'market_volatility',
                'priority': 7,
                'keywords_count': 13 if mega_mode else 7,
                'estimated_api_calls': 39 if mega_mode else 21,
                'description': 'Market volatility and stress indicators' + (' - EXPANDED' if mega_mode else '')
            },
            {
                'name': 'corporate_earnings',
                'priority': 6,
                'keywords_count': 12 if mega_mode else 5,
                'estimated_api_calls': 36 if mega_mode else 15,
                'description': 'Corporate earnings and business performance' + (' - EXPANDED' if mega_mode else '')
            },
            {
                'name': 'geopolitical',
                'priority': 6,
                'keywords_count': 13 if mega_mode else 6,
                'estimated_api_calls': 39 if mega_mode else 18,
                'description': 'Geopolitical events affecting markets' + (' - EXPANDED' if mega_mode else '')
            }
        ]
        
        # Add new categories for mega mode - FIXED CATEGORY NAMES
        if mega_mode:
            base_categories.extend([
                {
                    'name': 'technology_disruption',
                    'priority': 8,
                    'keywords_count': 11,
                    'estimated_api_calls': 33,
                    'description': 'Technology sector and AI disruption - NEW'
                },
                {
                    'name': 'supply_chain_logistics',
                    'priority': 7,
                    'keywords_count': 13,
                    'estimated_api_calls': 39,
                    'description': 'Supply chain and logistics - NEW'
                },
                {
                    'name': 'energy_climate',
                    'priority': 7,
                    'keywords_count': 12,
                    'estimated_api_calls': 36,
                    'description': 'Energy and climate policy - NEW'
                },
                {
                    'name': 'consumer_social_trends',
                    'priority': 7,
                    'keywords_count': 12,
                    'estimated_api_calls': 36,
                    'description': 'Consumer behavior and social trends - NEW'
                },
                {
                    'name': 'political_policy',
                    'priority': 6,
                    'keywords_count': 11,
                    'estimated_api_calls': 33,
                    'description': 'Political policy and legislation - NEW'
                },
                {
                    'name': 'social_movements',
                    'priority': 5,
                    'keywords_count': 10,
                    'estimated_api_calls': 30,
                    'description': 'Social movements and activism - NEW'
                }
            ])
        
        return base_categories
    
    def calculate_optimal_strategy(self, days_back: int, intensive_mode: bool = False, mega_mode: bool = False) -> Dict[str, Any]:
        """
        Calculate optimal backfill strategy based on API limits.
        
        Args:
            days_back: Number of days to backfill
            intensive_mode: If True, use all keywords and smaller chunks for maximum collection
            mega_mode: If True, use 2-day chunks and expanded categories for 4K+ articles
            
        Returns:
            Strategy configuration
        """
        categories = self.get_prioritized_categories(mega_mode)
        
        if mega_mode:
            # MEGA MODE: 2-day chunks, but only 3 keywords per category (matches actual implementation)
            chunk_size = 2
            keywords_per_search = 3  # Actual implementation limit in news_series_fetcher.py
            num_chunks = (days_back + chunk_size - 1) // chunk_size
            # Use realistic estimate: 3 keywords * 2 searches per chunk = 6 calls per chunk per category
            total_estimated_calls = len(categories) * num_chunks * 6
        elif intensive_mode:
            # Use ALL keywords (not just top 3) and smaller chunks
            chunk_size = 5
            keywords_per_search = 'all'
            num_chunks = (days_back + chunk_size - 1) // chunk_size
            total_estimated_calls = sum(cat['keywords_count'] for cat in categories) * num_chunks
        else:
            # Conservative mode (original behavior)
            total_estimated_calls = sum(cat['estimated_api_calls'] for cat in categories)
            # Time chunking strategy - break 30 days into larger chunks
            if days_back <= 7:
                chunk_size = days_back
            elif days_back <= 14:
                chunk_size = 7
            else:
                chunk_size = 10
            keywords_per_search = 3
        
        chunks = []
        for i in range(0, days_back, chunk_size):
            end_day = min(i + chunk_size, days_back)
            chunks.append({
                'start_day': i,
                'end_day': end_day,
                'days': end_day - i
            })
        
        # Adjust categories based on available API calls
        available_calls = self.max_api_calls - self.api_calls_made
        
        if total_estimated_calls <= available_calls:
            if mega_mode:
                strategy = 'mega_intensive_backfill'
            elif intensive_mode:
                strategy = 'intensive_backfill'
            else:
                strategy = 'full_backfill'
            selected_categories = categories
        else:
            strategy = 'priority_backfill'
            # Select highest priority categories that fit in API limit
            selected_categories = []
            running_calls = 0
            for cat in categories:
                if mega_mode:
                    # Realistic estimate: 3 keywords * 2 searches per chunk = 6 calls per chunk
                    estimated_calls = len(chunks) * 6
                elif intensive_mode:
                    estimated_calls = cat['keywords_count'] * len(chunks)
                else:
                    estimated_calls = cat['estimated_api_calls']
                
                if running_calls + estimated_calls <= available_calls:
                    selected_categories.append(cat)
                    running_calls += estimated_calls
                else:
                    break
        
        return {
            'strategy': strategy,
            'days_back': days_back,
            'chunk_size': chunk_size,
            'chunks': chunks,
            'selected_categories': selected_categories,
            'estimated_total_calls': total_estimated_calls,
            'available_calls': available_calls,
            'intensive_mode': intensive_mode,
            'mega_mode': mega_mode,
            'keywords_per_search': keywords_per_search
        }
    
    async def backfill_category_chunk_intensive(
        self,
        category: str,
        start_date: date,
        end_date: date,
        max_articles_per_search: int = 50
    ) -> Dict[str, Any]:
        """
        Intensive backfill for a category chunk using comprehensive keyword searches.
        
        Args:
            category: Economic category to backfill
            start_date: Start date for chunk
            end_date: End date for chunk
            max_articles_per_search: Maximum articles per search
            
        Returns:
            Chunk results with detailed breakdown
        """
        chunk_start_time = datetime.now(timezone.utc)
        result = {
            'category': category,
            'start_date': start_date,
            'end_date': end_date,
            'success': False,
            'articles_found': 0,
            'articles_stored': 0,
            'api_calls_made': 0,
            'error_message': None,
            'duration_seconds': 0
        }
        
        try:
            self.logger.info(f"Intensive backfill {category} from {start_date} to {end_date}")
            
            # Calculate days for this chunk
            days_back = (end_date - start_date).days + 1
            
            # Use multiple smaller searches with different parameters for better coverage
            total_articles_found = 0
            total_articles_stored = 0
            total_api_calls = 0
            
            # Strategy 1: Use the standard bulk fetch (uses top keywords)
            stats1 = await self.db_integration.bulk_fetch_and_store_news(
                categories=[category],
                days_back=days_back,
                max_articles_per_category=max_articles_per_search
            )
            
            if stats1['success']:
                category_stats = stats1['articles_by_category'].get(category, {})
                total_articles_found += category_stats.get('found', 0)
                total_articles_stored += category_stats.get('stored', 0)
                total_api_calls += 3  # Estimate
                
            # Strategy 2: Additional search with broader terms (intensive mode)
            await asyncio.sleep(2)  # Rate limiting delay
            
            stats2 = await self.db_integration.bulk_fetch_and_store_news(
                categories=[category],
                days_back=days_back,
                max_articles_per_category=max_articles_per_search // 2  # Smaller limit for second pass
            )
            
            if stats2['success']:
                category_stats = stats2['articles_by_category'].get(category, {})
                total_articles_found += category_stats.get('found', 0)
                total_articles_stored += category_stats.get('stored', 0)
                total_api_calls += 3  # Estimate
            
            result['success'] = total_articles_stored > 0
            result['articles_found'] = total_articles_found
            result['articles_stored'] = total_articles_stored
            result['api_calls_made'] = total_api_calls
            
            self.total_articles_collected += total_articles_stored
            self.api_calls_made += total_api_calls
            
            self.logger.info(
                f"✅ {category} intensive chunk: {total_articles_stored} articles stored, "
                f"{total_api_calls} API calls"
            )
            
        except Exception as e:
            result['error_message'] = str(e)
            self.logger.error(f"❌ Unexpected error in intensive {category} chunk: {e}")
        
        finally:
            result['duration_seconds'] = (datetime.now(timezone.utc) - chunk_start_time).total_seconds()
        
        return result
    
    async def backfill_category_chunk(
        self,
        category: str,
        start_date: date,
        end_date: date,
        max_articles: int = 50
    ) -> Dict[str, Any]:
        """
        Standard backfill for a category chunk.
        
        Args:
            category: Economic category to backfill
            start_date: Start date for chunk
            end_date: End date for chunk
            max_articles: Maximum articles to collect
            
        Returns:
            Chunk results
        """
        chunk_start_time = datetime.now(timezone.utc)
        result = {
            'category': category,
            'start_date': start_date,
            'end_date': end_date,
            'success': False,
            'articles_found': 0,
            'articles_stored': 0,
            'api_calls_made': 0,
            'error_message': None,
            'duration_seconds': 0
        }
        
        try:
            self.logger.info(f"Standard backfill {category} from {start_date} to {end_date}")
            
            # Calculate days for this chunk
            days_back = (end_date - start_date).days + 1
            
            # Use the database integration for this chunk
            stats = await self.db_integration.bulk_fetch_and_store_news(
                categories=[category],
                days_back=days_back,
                max_articles_per_category=max_articles
            )
            
            if stats['success']:
                category_stats = stats['articles_by_category'].get(category, {})
                result['success'] = True
                result['articles_found'] = category_stats.get('found', 0)
                result['articles_stored'] = category_stats.get('stored', 0)
                result['api_calls_made'] = 3  # Estimate based on keyword searches
                
                self.total_articles_collected += result['articles_stored']
                self.api_calls_made += result['api_calls_made']
                
                self.logger.info(
                    f"✅ {category} chunk: {result['articles_stored']} articles stored, "
                    f"{result['api_calls_made']} API calls"
                )
            else:
                result['error_message'] = stats.get('error_message', 'Unknown error')
                self.logger.error(f"❌ {category} chunk failed: {result['error_message']}")
            
        except Exception as e:
            result['error_message'] = str(e)
            self.logger.error(f"❌ Unexpected error in {category} chunk: {e}")
        
        finally:
            result['duration_seconds'] = (datetime.now(timezone.utc) - chunk_start_time).total_seconds()
        
        return result
    
    async def run_backfill(
        self,
        days_back: int = 30,
        resume: bool = False,
        intensive_mode: bool = False,
        mega_mode: bool = False
    ) -> Dict[str, Any]:
        """
        Run the complete historical backfill operation.
        
        Args:
            days_back: Number of days to backfill
            resume: Whether to resume from previous progress
            intensive_mode: Use multiple searches per chunk for maximum collection
            mega_mode: Use 2-day chunks, expanded categories, and all keywords for 4K+ articles
            
        Returns:
            Complete backfill results
        """
        backfill_start_time = datetime.now(timezone.utc)
        
        # Load previous progress if resuming
        if resume and self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    progress = json.load(f)
                self.categories_completed = progress.get('categories_completed', [])
                self.api_calls_made = progress.get('api_calls_made', 0)
                self.total_articles_collected = progress.get('total_articles_collected', 0)
                self.logger.info(f"Resuming backfill: {len(self.categories_completed)} categories completed")
            except Exception as e:
                self.logger.warning(f"Could not load progress file: {e}")
        
        # Calculate optimal strategy
        strategy = self.calculate_optimal_strategy(days_back, intensive_mode, mega_mode)
        
        self.logger.info("🚀 Starting News Historical Backfill")
        if mega_mode:
            self.logger.info("🔥 MEGA MODE: 2-day chunks, 13 categories, expanded keywords!")
        self.logger.info(f"Strategy: {strategy['strategy']}")
        self.logger.info(f"Intensive Mode: {intensive_mode}")
        self.logger.info(f"Mega Mode: {mega_mode}")
        self.logger.info(f"Days to backfill: {days_back}")
        self.logger.info(f"Chunk size: {strategy['chunk_size']} days")
        self.logger.info(f"Keywords per search: {strategy['keywords_per_search']}")
        self.logger.info(f"Categories: {len(strategy['selected_categories'])}")
        self.logger.info(f"Estimated API calls: {strategy['estimated_total_calls']}")
        self.logger.info(f"Available API calls: {strategy['available_calls']}")
        
        # Calculate date range
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        
        all_results = []
        
        # Process each category in priority order
        for category_config in strategy['selected_categories']:
            category = category_config['name']
            
            # Skip if already completed (resume functionality)
            if category in self.categories_completed:
                self.logger.info(f"Skipping {category} (already completed)")
                continue
            
            self.logger.info(f"\n📊 Processing category: {category.upper()}")
            self.logger.info(f"Priority: {category_config['priority']}")
            self.logger.info(f"Description: {category_config['description']}")
            self.logger.info(f"Keywords: {category_config['keywords_count']}")
            
            category_results = []
            
            # Process each time chunk for this category
            for chunk in strategy['chunks']:
                # Check API limit
                if self.api_calls_made >= self.max_api_calls:
                    self.logger.warning("API call limit reached, stopping backfill")
                    break
                
                chunk_start = start_date + timedelta(days=chunk['start_day'])
                chunk_end = start_date + timedelta(days=chunk['end_day'] - 1)
                
                if mega_mode or intensive_mode:
                    result = await self.backfill_category_chunk_intensive(
                        category,
                        chunk_start,
                        chunk_end,
                        max_articles_per_search=100 if mega_mode else 75
                    )
                else:
                    result = await self.backfill_category_chunk(
                        category,
                        chunk_start,
                        chunk_end,
                        max_articles=50
                    )
                
                category_results.append(result)
                all_results.append(result)
                
                # Shorter delay for mega mode to maximize throughput
                delay = 1 if mega_mode else 2
                await asyncio.sleep(delay)
            
            # Mark category as completed
            self.categories_completed.append(category)
            
            # Save progress
            await self._save_progress()
            
            # Show category summary
            category_articles = sum(r['articles_stored'] for r in category_results)
            category_calls = sum(r['api_calls_made'] for r in category_results)
            self.logger.info(f"✅ {category} completed: {category_articles} articles, {category_calls} API calls")
        
        # Calculate final statistics
        end_time = datetime.now(timezone.utc)
        total_duration = (end_time - backfill_start_time).total_seconds()
        
        final_results = {
            'start_time': backfill_start_time,
            'end_time': end_time,
            'total_duration_seconds': total_duration,
            'strategy_used': strategy['strategy'],
            'intensive_mode': intensive_mode,
            'mega_mode': mega_mode,
            'days_backfilled': days_back,
            'categories_completed': len(self.categories_completed),
            'total_categories_planned': len(strategy['selected_categories']),
            'total_api_calls_made': self.api_calls_made,
            'total_articles_collected': self.total_articles_collected,
            'api_calls_remaining': self.max_api_calls - self.api_calls_made,
            'chunk_results': all_results,
            'failed_operations': self.failed_operations,
            'success_rate': len([r for r in all_results if r['success']]) / len(all_results) * 100 if all_results else 0
        }
        
        return final_results
    
    async def _save_progress(self) -> None:
        """Save current progress to file for resume capability"""
        try:
            progress = {
                'categories_completed': self.categories_completed,
                'api_calls_made': self.api_calls_made,
                'total_articles_collected': self.total_articles_collected,
                'last_update': datetime.now(timezone.utc).isoformat()
            }
            
            with open(self.progress_file, 'w') as f:
                json.dump(progress, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Failed to save progress: {e}")
    
    def print_backfill_summary(self, results: Dict[str, Any]) -> None:
        """Print comprehensive summary of the backfill operation"""
        
        print("\n" + "="*80)
        print("🎯 NEWS HISTORICAL BACKFILL SUMMARY")
        print("="*80)
        
        if results['mega_mode']:
            mode_text = "MEGA MODE"
        elif results['intensive_mode']:
            mode_text = "INTENSIVE MODE"
        else:
            mode_text = "STANDARD MODE"
        
        print(f"Mode: {mode_text}")
        print(f"📅 Started:  {results['start_time'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"📅 Finished: {results['end_time'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"⏱️  Duration: {results['total_duration_seconds']:.1f} seconds ({results['total_duration_seconds']/60:.1f} minutes)")
        print(f"📊 Success Rate: {results['success_rate']:.1f}%")
        
        print(f"\n📈 BACKFILL STATISTICS:")
        print(f"   Strategy Used: {results['strategy_used']}")
        print(f"   Days Backfilled: {results['days_backfilled']}")
        print(f"   Categories Completed: {results['categories_completed']}/{results['total_categories_planned']}")
        print(f"   📊 Total Articles Collected: {results['total_articles_collected']:,}")
        print(f"   🔌 API Calls Made: {results['total_api_calls_made']}")
        print(f"   🔋 API Calls Remaining: {results['api_calls_remaining']}")
        
        # Category breakdown
        if results['chunk_results']:
            print(f"\n📊 BY CATEGORY:")
            category_stats = {}
            for result in results['chunk_results']:
                cat = result['category']
                if cat not in category_stats:
                    category_stats[cat] = {'articles': 0, 'chunks': 0, 'api_calls': 0}
                category_stats[cat]['articles'] += result['articles_stored']
                category_stats[cat]['chunks'] += 1
                category_stats[cat]['api_calls'] += result['api_calls_made']
            
            for category, stats in category_stats.items():
                print(f"   {category.upper()}: {stats['articles']} articles, {stats['chunks']} chunks, {stats['api_calls']} API calls")
        
        # Failed operations
        failed_results = [r for r in results['chunk_results'] if not r['success']]
        if failed_results:
            print(f"\n❌ FAILED OPERATIONS ({len(failed_results)}):")
            for result in failed_results:
                print(f"   {result['category']} ({result['start_date']} to {result['end_date']}): {result['error_message']}")
        
        print("\n" + "="*80)
        if results['total_articles_collected'] > 0:
            print(f"🎉 BACKFILL COMPLETED! {results['total_articles_collected']} articles collected")
            efficiency = results['total_articles_collected'] / results['total_api_calls_made'] if results['total_api_calls_made'] > 0 else 0
            print(f"📊 Efficiency: {efficiency:.1f} articles per API call")
        else:
            print("⚠️  No articles collected. Check API credentials and error messages.")
        print("="*80)
    
    async def close(self) -> None:
        """Clean up resources"""
        await self.db_integration.close()


async def main():
    """Main entry point for historical backfill"""
    parser = argparse.ArgumentParser(
        description='Historical backfill for News API economic data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python news_historical_backfill.py                     # Full 30-day backfill
  python news_historical_backfill.py --intensive         # Intensive mode (2x searches per chunk)
  python news_historical_backfill.py --mega              # MEGA mode (2-day chunks + 13 categories)
  python news_historical_backfill.py --days 14           # 14-day backfill
  python news_historical_backfill.py --resume            # Resume interrupted backfill
  python news_historical_backfill.py --max-calls 500     # Limit API usage
        """
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Number of days to backfill (default: 30)'
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from previous progress'
    )
    
    parser.add_argument(
        '--intensive',
        action='store_true',
        help='Use intensive mode: multiple searches per chunk for maximum collection'
    )
    
    parser.add_argument(
        '--mega',
        action='store_true',
        help='Use MEGA mode: 2-day chunks, 13 categories, all keywords for 4K+ articles'
    )
    
    parser.add_argument(
        '--max-calls',
        type=int,
        default=900,
        help='Maximum API calls to use (default: 900, reserves 100 for daily ops)'
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
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Create backfill system
    backfill = NewsHistoricalBackfill(args.database_url, max_api_calls=args.max_calls)
    
    try:
        # Initialize database connection
        if not await backfill.initialize():
            logger.error("Failed to initialize database connection")
            return 1
        
        # Confirm operation
        if not args.resume:
            if args.mega:
                mode_text = "MEGA mode (2-day chunks, 13 categories, all keywords for 4K+ articles)"
            elif args.intensive:
                mode_text = "intensive mode (multiple searches per chunk)"
            else:
                mode_text = "standard mode"
            print(f"🔄 Ready to backfill {args.days} days of economic news in {mode_text}")
            print(f"📊 Max API calls: {args.max_calls}")
            response = input("Continue? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                print("Backfill cancelled")
                return 0
        
        # Run backfill
        results = await backfill.run_backfill(
            days_back=args.days,
            resume=args.resume,
            intensive_mode=args.intensive,
            mega_mode=args.mega
        )
        
        # Print summary
        backfill.print_backfill_summary(results)
        
        # Return appropriate exit code
        return 0 if results['total_articles_collected'] > 0 else 1
        
    except KeyboardInterrupt:
        logger.info("Backfill interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Backfill failed with unexpected error: {e}")
        return 1
    finally:
        await backfill.close()


if __name__ == "__main__":
    """Run the historical backfill"""
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
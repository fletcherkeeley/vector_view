#!/usr/bin/env python3
"""
Yahoo Finance Bulk Historical Data Loader

This script loads full historical market data for all key assets into the database.
It's designed to be run once to populate the database with decades of market data.

Key Features:
- Loads all major market indices, ETFs, and sector funds with maximum historical data
- Progress tracking and detailed logging
- Graceful error handling with retry logic
- Resume capability (skips already loaded assets)
- Comprehensive reporting of load results
- Asset validation before loading

Usage:
    python yahoo_bulk_loader.py                     # Load all assets
    python yahoo_bulk_loader.py --resume            # Skip already loaded assets  
    python yahoo_bulk_loader.py --symbols SPY QQQ   # Load specific symbols only
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
from .yahoo_database_integration import YahooDatabaseIntegration

# Load environment variables
load_dotenv()

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('yahoo_bulk_load.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class YahooBulkLoader:
    """
    Manages bulk loading of historical Yahoo Finance data for all key market assets.
    """
    
    # Key Market Assets from data_sources.md and major market indicators
    MARKET_ASSETS = {
        # Major Market Indices
        'major_indices': {
            'SPY': 'SPDR S&P 500 ETF Trust',
            'QQQ': 'Invesco QQQ Trust (NASDAQ-100)',
            'VTI': 'Vanguard Total Stock Market ETF',
            'IWM': 'iShares Russell 2000 ETF'
        },
        
        # Sector ETFs (SPDR Select Sector Funds)
        'sector_etfs': {
            'XLF': 'Financial Select Sector SPDR Fund',
            'XLY': 'Consumer Discretionary Select Sector SPDR Fund',
            'XLP': 'Consumer Staples Select Sector SPDR Fund',
            'XLE': 'Energy Select Sector SPDR Fund',
            'XLI': 'Industrial Select Sector SPDR Fund',
            'XLK': 'Technology Select Sector SPDR Fund',
            'XLV': 'Health Care Select Sector SPDR Fund',
            'XLU': 'Utilities Select Sector SPDR Fund',
            'XLRE': 'Real Estate Select Sector SPDR Fund',
            'XLB': 'Materials Select Sector SPDR Fund',
            'XLC': 'Communication Services Select Sector SPDR Fund'
        },
        
        # Additional Popular ETFs
        'popular_etfs': {
            'VOO': 'Vanguard S&P 500 ETF',
            'VEA': 'Vanguard FTSE Developed Markets ETF',
            'VWO': 'Vanguard FTSE Emerging Markets ETF',
            'BND': 'Vanguard Total Bond Market ETF',
            'GLD': 'SPDR Gold Shares',
            'SLV': 'iShares Silver Trust',
            'TLT': 'iShares 20+ Year Treasury Bond ETF',
            'HYG': 'iShares iBoxx $ High Yield Corporate Bond ETF'
        },
        
        # Major Individual Stocks (FAANG+ and key large caps)
        'large_cap_stocks': {
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc. Class A',
            'AMZN': 'Amazon.com Inc.',
            'TSLA': 'Tesla Inc.',
            'META': 'Meta Platforms Inc.',
            'NVDA': 'NVIDIA Corporation',
            'NFLX': 'Netflix Inc.',
            'JPM': 'JPMorgan Chase & Co.',
            'JNJ': 'Johnson & Johnson',
            'V': 'Visa Inc.',
            'WMT': 'Walmart Inc.',
            'PG': 'Procter & Gamble Company',
            'UNH': 'UnitedHealth Group Inc.',
            'HD': 'Home Depot Inc.',
            'MA': 'Mastercard Inc.',
            'BAC': 'Bank of America Corp',
            'PFE': 'Pfizer Inc.',
            'KO': 'Coca-Cola Company',
            'DIS': 'Walt Disney Company'
        },
        
        # International and Alternative Assets
        'international_alternatives': {
            'EFA': 'iShares MSCI EAFE ETF',
            'EEM': 'iShares MSCI Emerging Markets ETF',
            'FXI': 'iShares China Large-Cap ETF',
            'EWJ': 'iShares MSCI Japan ETF',
            'EWZ': 'iShares MSCI Brazil ETF',
            'RSX': 'VanEck Russia ETF',
            'ARKK': 'ARK Innovation ETF',
            'ARKQ': 'ARK Autonomous Technology & Robotics ETF',
            'QQQ': 'Invesco QQQ Trust',
            'TQQQ': 'ProShares UltraPro QQQ',
            'SQQQ': 'ProShares UltraPro Short QQQ'
        }
    }
    
    def __init__(self, database_url: str):
        """
        Initialize the bulk loader.
        
        Args:
            database_url: PostgreSQL connection string
        """
        self.database_url = database_url
        self.db_integration = YahooDatabaseIntegration(database_url)
        self.load_results = {}
        self.total_assets = 0
        self.successful_loads = 0
        self.failed_loads = 0
        
        logger.info("Yahoo Finance Bulk Loader initialized")
    
    async def initialize(self) -> bool:
        """Initialize database connection"""
        success = await self.db_integration.initialize()
        if success:
            logger.info("Database connection initialized for bulk loading")
        return success
    
    def get_all_symbols(self) -> Dict[str, str]:
        """
        Get all asset symbols and names as a flat dictionary.
        
        Returns:
            Dictionary mapping symbol -> name
        """
        all_symbols = {}
        for category, symbols_dict in self.MARKET_ASSETS.items():
            all_symbols.update(symbols_dict)
        
        return all_symbols
    
    async def validate_symbols(self, symbols: List[str]) -> Dict[str, bool]:
        """
        Validate that symbols exist and return data before bulk loading.
        
        Args:
            symbols: List of symbols to validate
            
        Returns:
            Dictionary mapping symbol -> is_valid
        """
        logger.info(f"Validating {len(symbols)} symbols...")
        
        from .yahoo_finance_client import YahooFinanceClient
        
        async with YahooFinanceClient() as client:
            validation_results = await client.validate_symbols(symbols)
        
        valid_count = sum(validation_results.values())
        invalid_symbols = [sym for sym, valid in validation_results.items() if not valid]
        
        if invalid_symbols:
            logger.warning(f"Invalid symbols found: {', '.join(invalid_symbols)}")
        
        logger.info(f"Symbol validation complete: {valid_count}/{len(symbols)} symbols are valid")
        return validation_results
    
    async def check_existing_assets(self, symbols: List[str]) -> Dict[str, bool]:
        """
        Check which assets already exist in the database.
        
        Args:
            symbols: List of asset symbols to check
            
        Returns:
            Dictionary mapping symbol -> exists_in_db
        """
        existing = {}
        
        for symbol in symbols:
            info = await self.db_integration.get_asset_info(symbol)
            existing[symbol] = info is not None
            
            if info:
                logger.info(f"Asset {symbol} already exists with {info['observation_count']} observations")
        
        return existing
    
    async def load_assets_batch(
        self, 
        symbols: List[str], 
        skip_existing: bool = False,
        period: str = "max"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Load a batch of assets with full historical data.
        
        Args:
            symbols: List of asset symbols to load
            skip_existing: Whether to skip assets that already exist in database
            period: Time period for historical data ('max' for all available)
            
        Returns:
            Dictionary mapping symbol -> load_results
        """
        batch_results = {}
        
        # Validate symbols first
        validation_results = await self.validate_symbols(symbols)
        valid_symbols = [sym for sym, valid in validation_results.items() if valid]
        invalid_symbols = [sym for sym, valid in validation_results.items() if not valid]
        
        if invalid_symbols:
            logger.warning(f"Skipping {len(invalid_symbols)} invalid symbols: {', '.join(invalid_symbols)}")
            for symbol in invalid_symbols:
                batch_results[symbol] = {
                    'symbol': symbol,
                    'success': False,
                    'error_message': 'Invalid symbol - no data available',
                    'observations_inserted': 0
                }
        
        # Check existing assets if needed
        if skip_existing:
            existing = await self.check_existing_assets(valid_symbols)
            symbols_to_load = [sym for sym in valid_symbols if not existing.get(sym, False)]
            skipped = [sym for sym in valid_symbols if existing.get(sym, False)]
            
            if skipped:
                logger.info(f"Skipping {len(skipped)} existing assets: {', '.join(skipped)}")
                for symbol in skipped:
                    batch_results[symbol] = {
                        'symbol': symbol,
                        'success': True,
                        'error_message': 'Skipped - already exists',
                        'observations_inserted': 0
                    }
        else:
            symbols_to_load = valid_symbols
        
        # Load each asset
        for i, symbol in enumerate(symbols_to_load, 1):
            logger.info(f"Loading asset {i}/{len(symbols_to_load)}: {symbol}")
            
            try:
                # Load full historical data
                result = await self.db_integration.bulk_load_asset(symbol, period=period)
                batch_results[symbol] = result
                
                if result['success']:
                    self.successful_loads += 1
                    logger.info(
                        f"✅ {symbol}: {result['observations_inserted']} observations "
                        f"in {result['duration_seconds']:.1f}s"
                    )
                else:
                    self.failed_loads += 1
                    logger.error(f"❌ {symbol}: {result.get('error_message', 'Unknown error')}")
                
                # Small delay between assets to be respectful
                await asyncio.sleep(1)
                
            except Exception as e:
                self.failed_loads += 1
                error_result = {
                    'symbol': symbol,
                    'success': False,
                    'error_message': str(e),
                    'observations_inserted': 0
                }
                batch_results[symbol] = error_result
                logger.error(f"❌ Unexpected error loading {symbol}: {e}")
        
        return batch_results
    
    async def load_all_assets(self, skip_existing: bool = False, period: str = "max") -> Dict[str, Any]:
        """
        Load all market assets with full historical data.
        
        Args:
            skip_existing: Whether to skip assets already in database
            period: Time period for historical data
            
        Returns:
            Comprehensive results dictionary
        """
        start_time = datetime.now(timezone.utc)
        logger.info("🚀 Starting bulk load of all Yahoo Finance market assets")
        
        # Get all symbols to load
        all_symbols = self.get_all_symbols()
        self.total_assets = len(all_symbols)
        
        logger.info(f"Planning to load {self.total_assets} market assets with '{period}' historical data")
        
        # Load by category for better organization and progress tracking
        category_results = {}
        
        for category, symbols_dict in self.MARKET_ASSETS.items():
            logger.info(f"\n📊 Loading category: {category.upper()}")
            logger.info(f"Assets in category: {', '.join(symbols_dict.keys())}")
            
            symbols = list(symbols_dict.keys())
            category_results[category] = await self.load_assets_batch(symbols, skip_existing, period)
        
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
            'total_assets_attempted': self.total_assets,
            'successful_loads': self.successful_loads,
            'failed_loads': self.failed_loads,
            'total_observations_loaded': total_observations,
            'category_results': category_results,
            'success_rate': (self.successful_loads / self.total_assets * 100) if self.total_assets > 0 else 0
        }
        
        return final_results
    
    async def load_specific_symbols(self, symbols: List[str], period: str = "max") -> Dict[str, Any]:
        """
        Load specific assets by symbol.
        
        Args:
            symbols: List of specific asset symbols to load
            period: Time period for historical data
            
        Returns:
            Results dictionary
        """
        start_time = datetime.now(timezone.utc)
        logger.info(f"🎯 Loading specific assets: {', '.join(symbols)}")
        
        self.total_assets = len(symbols)
        results = await self.load_assets_batch(symbols, period=period)
        
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
            'total_assets_attempted': self.total_assets,
            'successful_loads': self.successful_loads,
            'failed_loads': self.failed_loads,
            'total_observations_loaded': total_observations,
            'assets_results': results,
            'success_rate': (self.successful_loads / self.total_assets * 100) if self.total_assets > 0 else 0
        }
    
    def print_summary_report(self, results: Dict[str, Any]) -> None:
        """Print a comprehensive summary report of the bulk load operation."""
        
        print("\n" + "="*80)
        print("🎯 YAHOO FINANCE BULK LOAD SUMMARY REPORT")
        print("="*80)
        
        print(f"📅 Started:  {results['start_time'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"📅 Finished: {results['end_time'].strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"⏱️  Duration: {results['total_duration_seconds']:.1f} seconds ({results['total_duration_seconds']/60:.1f} minutes)")
        print(f"📊 Success Rate: {results['success_rate']:.1f}%")
        
        print(f"\n📈 OVERALL STATISTICS:")
        print(f"   Total Assets Attempted: {results['total_assets_attempted']}")
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
                
                print(f"   {category.upper()}: {successful_in_category}/{total_in_category} assets, {observations_in_category:,} observations")
        
        # Top performers by observations
        all_results = []
        if 'category_results' in results:
            for category_result in results['category_results'].values():
                all_results.extend(category_result.values())
        elif 'assets_results' in results:
            all_results.extend(results['assets_results'].values())
        
        successful_results = [r for r in all_results if r.get('success', False) and r.get('observations_inserted', 0) > 0]
        if successful_results:
            top_performers = sorted(successful_results, key=lambda x: x.get('observations_inserted', 0), reverse=True)[:10]
            print(f"\n🏆 TOP 10 ASSETS BY DATA VOLUME:")
            for i, result in enumerate(top_performers, 1):
                print(f"   {i:2d}. {result['symbol']}: {result['observations_inserted']:,} observations")
        
        # Failed assets details
        failed_results = [r for r in all_results if not r.get('success', False)]
        if failed_results:
            print(f"\n❌ FAILED ASSETS:")
            for result in failed_results:
                error_msg = result.get('error_message', 'Unknown error')
                if 'Invalid symbol' not in error_msg and 'already exists' not in error_msg:
                    print(f"   {result['symbol']}: {error_msg}")
        
        print("\n" + "="*80)
        if results['failed_loads'] == 0:
            print("🎉 ALL ASSETS LOADED SUCCESSFULLY!")
        else:
            print(f"⚠️  {results['failed_loads']} assets failed to load. Check logs for details.")
        print("="*80)
    
    async def close(self) -> None:
        """Clean up resources"""
        await self.db_integration.close()


async def main():
    """Main entry point for the bulk loader"""
    parser = argparse.ArgumentParser(
        description='Load historical Yahoo Finance market data into database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python yahoo_bulk_loader.py                      # Load all assets with max history
  python yahoo_bulk_loader.py --resume             # Skip already loaded assets
  python yahoo_bulk_loader.py --symbols SPY QQQ    # Load specific symbols only
  python yahoo_bulk_loader.py --period 5y          # Load 5 years of data
  python yahoo_bulk_loader.py --list               # List all available assets
        """
    )
    
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Skip assets that already exist in database'
    )
    
    parser.add_argument(
        '--symbols',
        nargs='+',
        help='Load specific assets by symbol (e.g., SPY QQQ AAPL)'
    )
    
    parser.add_argument(
        '--period',
        default='max',
        choices=['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'],
        help='Time period for historical data (default: max)'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available assets and exit'
    )
    
    parser.add_argument(
        '--database-url',
        default=os.getenv('DATABASE_URL', 'postgresql+psycopg://postgres:fred_password@localhost:5432/postgres'),
        help='Database connection URL'
    )
    
    args = parser.parse_args()
    
    # Create bulk loader
    loader = YahooBulkLoader(args.database_url)
    
    # Handle list option
    if args.list:
        print("📋 Available Yahoo Finance Market Assets:")
        print("="*60)
        
        for category, symbols_dict in loader.MARKET_ASSETS.items():
            print(f"\n{category.upper().replace('_', ' ')}:")
            for symbol, name in symbols_dict.items():
                print(f"  {symbol:8} - {name}")
        
        total_assets = len(loader.get_all_symbols())
        print(f"\nTotal: {total_assets} assets available for loading")
        return
    
    try:
        # Initialize database connection
        if not await loader.initialize():
            logger.error("Failed to initialize database connection")
            return 1
        
        # Load data
        if args.symbols:
            # Load specific symbols
            results = await loader.load_specific_symbols(args.symbols, period=args.period)
        else:
            # Load all assets
            results = await loader.load_all_assets(skip_existing=args.resume, period=args.period)
        
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
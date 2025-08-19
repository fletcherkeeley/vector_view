#!/usr/bin/env python3
"""
Economic Data Platform - Database Setup Script

Simple script to create the unified database schema for the economic data platform.
This script provides a user-friendly interface for database initialization.

Usage:
    python setup_database.py                    # Use default local database
    python setup_database.py --production       # Use production database settings
    python setup_database.py --fresh-install    # Drop existing tables and recreate
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from unified_database_setup import create_database_schema, validate_database_url
    import logging
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you've installed all requirements: pip install -r requirements.txt")
    sys.exit(1)

# Configure logging for this script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_database_url() -> str:
    """
    Get database URL from environment or use default
    """
    # Try to get from environment first
    db_url = os.getenv('DATABASE_URL')
    
    if db_url:
        logger.info("Using database URL from environment variable")
        return db_url
    
    # Default local development database
    default_url = "postgresql+psycopg://postgres:fred_password@localhost:5432/postgres"
    
    logger.info("Using default local database URL")
    logger.info("To use a different database, set the DATABASE_URL environment variable")
    
    return default_url


async def setup_with_confirmation(fresh_install: bool = False):
    """
    Setup database with user confirmation
    """
    print("🚀 Economic Data Platform - Database Setup")
    print("=" * 50)
    
    # Get database URL
    database_url = get_database_url()
    
    # Hide password in display
    display_url = database_url
    if '@' in display_url:
        parts = display_url.split('@')
        if len(parts) == 2:
            user_part = parts[0].split('://')[1]
            if ':' in user_part:
                user, _ = user_part.rsplit(':', 1)
                display_url = f"{parts[0].split('://')[0]}://{user}:***@{parts[1]}"
    
    print(f"Database: {display_url}")
    print(f"Fresh install: {'Yes' if fresh_install else 'No'}")
    
    # Validate database URL
    if not validate_database_url(database_url):
        print("❌ Invalid database URL")
        return False
    
    # Warning for fresh install
    if fresh_install:
        print("\n⚠️  WARNING: Fresh install will DELETE ALL EXISTING DATA!")
        response = input("Type 'DELETE ALL DATA' to continue: ")
        if response != 'DELETE ALL DATA':
            print("Setup cancelled")
            return False
    
    # Final confirmation
    print(f"\nReady to {'recreate' if fresh_install else 'create'} database schema...")
    response = input("Continue? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Setup cancelled")
        return False
    
    print("\n🔨 Creating database schema...")
    
    # Create the database
    success = await create_database_schema(
        database_url=database_url,
        drop_existing=fresh_install
    )
    
    if success:
        print("\n✅ Database setup completed successfully!")
        print("\n📋 What's been created:")
        print("  • data_series - Master table for all data series")
        print("  • market_assets - Market-specific asset information")
        print("  • time_series_observations - Unified time series data")
        print("  • series_correlations - Pre-calculated correlations")
        print("  • news_topic_mapping - News/vector database integration")
        print("  • data_sync_log - Comprehensive sync logging")
        print("  • system_health_metrics - System monitoring")
        
        print("\n🔧 Next steps:")
        print("  1. Set up your API credentials (FRED, Yahoo Finance, News API)")
        print("  2. Create data ingestion services")
        print("  3. Configure correlation calculations")
        print("  4. Set up monitoring and alerting")
        
        return True
    else:
        print("\n❌ Database setup failed!")
        print("Check the logs above for error details")
        return False


def main():
    """
    Main entry point
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Setup database for Economic Data Platform',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python setup_database.py                    # Normal setup
  python setup_database.py --fresh-install    # Drop and recreate all tables
  
Environment Variables:
  DATABASE_URL    PostgreSQL connection string
                  Default: postgresql+psycopg://postgres:fred_password@localhost:5432/postgres
        """
    )
    
    parser.add_argument(
        '--fresh-install',
        action='store_true',
        help='Drop all existing tables and recreate (DESTRUCTIVE!)'
    )
    
    parser.add_argument(
        '--production',
        action='store_true',
        help='Use production database settings (requires DATABASE_URL env var)'
    )
    
    args = parser.parse_args()
    
    # Production mode requires explicit DATABASE_URL
    if args.production and not os.getenv('DATABASE_URL'):
        print("❌ Production mode requires DATABASE_URL environment variable")
        print("Set it like: export DATABASE_URL='postgresql+psycopg://user:pass@host:port/db'")
        sys.exit(1)
    
    try:
        # Run the async setup
        success = asyncio.run(setup_with_confirmation(fresh_install=args.fresh_install))
        
        if success:
            print("\n🎉 Setup complete! Your economic data platform is ready.")
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        logger.exception("Setup failed with unexpected error")
        sys.exit(1)


if __name__ == "__main__":
    main()
"""
Yahoo Finance Database Integration - Data Storage Layer

This module handles storing Yahoo Finance market data in our PostgreSQL database.
It integrates with the yahoo_series_fetcher to store both asset metadata and OHLCV observations.

Key Features:
- Store asset metadata in data_series and market_assets tables
- Bulk insert OHLCV observations with conflict handling
- Handle duplicate data gracefully (upsert operations)
- Track data sync operations in data_sync_log
- Support for incremental and full historical loads

Depends on: 
- yahoo_series_fetcher.py for data formatting
- unified_database_setup.py for schema definitions
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
import uuid

# Add database directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "database"))

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy import select, and_, func, text
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

# Import our database models
from unified_database_setup import (
    DataSeries, MarketAssets, TimeSeriesObservation, DataSyncLog, 
    DataSourceType, FrequencyType
)

from yahoo_series_fetcher import YahooSeriesFetcher

# Configure logging
logger = logging.getLogger(__name__)


class YahooDatabaseIntegration:
    """
    Handles storing Yahoo Finance market data in our PostgreSQL database with comprehensive error handling.
    """
    
    def __init__(self, database_url: str):
        """
        Initialize database integration.
        
        Args:
            database_url: PostgreSQL connection string
        """
        self.database_url = database_url
        self.engine = None
        self.AsyncSessionLocal = None
        
        logger.info("Yahoo Finance Database Integration initialized")
    
    async def initialize(self) -> bool:
        """
        Initialize database connection and session factory.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.engine = create_async_engine(
                self.database_url,
                echo=False,  # Set to True for SQL debugging
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True
            )
            
            self.AsyncSessionLocal = sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Test connection
            async with self.AsyncSessionLocal() as session:
                await session.execute(text("SELECT 1"))
            
            logger.info("Database connection initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {e}")
            return False
    
    async def store_asset_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Store or update asset metadata in both data_series and market_assets tables.
        
        Args:
            metadata: Asset metadata from yahoo_series_fetcher (contains both series and market metadata)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with self.AsyncSessionLocal() as session:
                # Store in data_series table
                series_data = metadata['series_metadata']
                stmt_series = insert(DataSeries).values(series_data)
                stmt_series = stmt_series.on_conflict_do_update(
                    index_elements=['series_id'],
                    set_={
                        'title': stmt_series.excluded.title,
                        'description': stmt_series.excluded.description,
                        'category': stmt_series.excluded.category,
                        'subcategory': stmt_series.excluded.subcategory,
                        'observation_start': stmt_series.excluded.observation_start,
                        'observation_end': stmt_series.excluded.observation_end,
                        'last_updated': stmt_series.excluded.last_updated,
                        'source_metadata': stmt_series.excluded.source_metadata,
                        'data_quality_score': stmt_series.excluded.data_quality_score,
                        'news_categories': stmt_series.excluded.news_categories,
                        'correlation_priority': stmt_series.excluded.correlation_priority,
                        'updated_at': stmt_series.excluded.updated_at
                    }
                )
                
                await session.execute(stmt_series)
                
                # Store in market_assets table
                market_data = metadata['market_metadata']
                stmt_market = insert(MarketAssets).values(market_data)
                stmt_market = stmt_market.on_conflict_do_update(
                    index_elements=['series_id'],
                    set_={
                        'symbol': stmt_market.excluded.symbol,
                        'exchange': stmt_market.excluded.exchange,
                        'asset_type': stmt_market.excluded.asset_type,
                        'company_name': stmt_market.excluded.company_name,
                        'sector': stmt_market.excluded.sector,
                        'industry': stmt_market.excluded.industry,
                        'market_cap': stmt_market.excluded.market_cap,
                        'currency': stmt_market.excluded.currency,
                        'country': stmt_market.excluded.country,
                        'expense_ratio': stmt_market.excluded.expense_ratio,
                        'index_tracked': stmt_market.excluded.index_tracked,
                        'economic_sensitivity': stmt_market.excluded.economic_sensitivity,
                        'sector_exposure': stmt_market.excluded.sector_exposure,
                        'updated_at': stmt_market.excluded.updated_at
                    }
                )
                
                await session.execute(stmt_market)
                await session.commit()
                
                logger.info(f"Successfully stored metadata for asset: {series_data['series_id']}")
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Database error storing metadata for {metadata.get('series_metadata', {}).get('series_id')}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error storing metadata: {e}")
            return False
    
    async def store_observations(self, observations: List[Dict[str, Any]]) -> Tuple[int, int]:
        """
        Store OHLCV observations in bulk with conflict handling.
        
        Args:
            observations: List of observation dictionaries from yahoo_series_fetcher
            
        Returns:
            Tuple of (successful_inserts, conflicts_updated)
        """
        if not observations:
            return 0, 0
        
        try:
            async with self.AsyncSessionLocal() as session:
                # Use bulk upsert for better performance
                stmt = insert(TimeSeriesObservation).values(observations)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['series_id', 'observation_date'],
                    set_={
                        'value': stmt.excluded.value,
                        'value_high': stmt.excluded.value_high,
                        'value_low': stmt.excluded.value_low,
                        'value_open': stmt.excluded.value_open,
                        'value_close': stmt.excluded.value_close,
                        'volume': stmt.excluded.volume,
                        'data_quality': stmt.excluded.data_quality,
                        'source_timestamp': stmt.excluded.source_timestamp,
                        'observation_metadata': stmt.excluded.observation_metadata,
                        'updated_at': stmt.excluded.updated_at
                    }
                )
                
                result = await session.execute(stmt)
                await session.commit()
                
                series_id = observations[0]['series_id']
                logger.info(f"Successfully stored {len(observations)} observations for {series_id}")
                
                # For now, return total count (we'd need more complex SQL to separate inserts vs updates)
                return len(observations), 0
                
        except SQLAlchemyError as e:
            logger.error(f"Database error storing observations: {e}")
            return 0, 0
        except Exception as e:
            logger.error(f"Unexpected error storing observations: {e}")
            return 0, 0
    
    async def bulk_load_asset(
        self, 
        symbol: str, 
        period: str = "1y",
        start_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load complete data for a market asset (metadata + historical observations).
        
        Args:
            symbol: Asset symbol (e.g., 'SPY', 'AAPL')
            period: Time period for historical data ('1y', '2y', '5y', 'max')
            start_date: Optional start date in YYYY-MM-DD format
            
        Returns:
            Dictionary with load statistics and results
        """
        sync_start_time = datetime.now(timezone.utc)
        load_stats = {
            'symbol': symbol,
            'success': False,
            'metadata_stored': False,
            'observations_inserted': 0,
            'observations_updated': 0,
            'total_observations': 0,
            'error_message': None,
            'duration_seconds': 0
        }
        
        try:
            logger.info(f"Starting bulk load for asset: {symbol}")
            
            # Use the fetcher to get complete data
            async with YahooSeriesFetcher() as fetcher:
                if start_date:
                    metadata, observations = await fetcher.get_asset_complete_data(symbol, period='max')
                    # Filter observations by start_date if provided
                    filter_date = datetime.strptime(start_date, '%Y-%m-%d').date()
                    observations = [obs for obs in observations if obs['observation_date'] >= filter_date]
                else:
                    metadata, observations = await fetcher.get_asset_complete_data(symbol, period=period)
                
                load_stats['total_observations'] = len(observations)
                
                # Store metadata (both data_series and market_assets)
                metadata_success = await self.store_asset_metadata(metadata)
                load_stats['metadata_stored'] = metadata_success
                
                if not metadata_success:
                    raise Exception("Failed to store asset metadata")
                
                # Store observations in batches for large datasets
                batch_size = 1000
                total_inserted = 0
                total_updated = 0
                
                for i in range(0, len(observations), batch_size):
                    batch = observations[i:i + batch_size]
                    inserted, updated = await self.store_observations(batch)
                    total_inserted += inserted
                    total_updated += updated
                    
                    if i % (batch_size * 5) == 0:  # Log every 5 batches
                        logger.info(f"Processed {i + len(batch)}/{len(observations)} observations for {symbol}")
                
                load_stats['observations_inserted'] = total_inserted
                load_stats['observations_updated'] = total_updated
                load_stats['success'] = True
                
                # Log sync operation
                await self._log_sync_operation(symbol, sync_start_time, load_stats, True)
                
                logger.info(f"Bulk load completed for {symbol}: {total_inserted} inserted, {total_updated} updated")
                
        except Exception as e:
            error_msg = str(e)
            load_stats['error_message'] = error_msg
            logger.error(f"Bulk load failed for {symbol}: {error_msg}")
            
            # Log failed sync operation
            await self._log_sync_operation(symbol, sync_start_time, load_stats, False, error_msg)
        
        finally:
            # Calculate duration
            load_stats['duration_seconds'] = (datetime.now(timezone.utc) - sync_start_time).total_seconds()
        
        return load_stats
    
    async def get_asset_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get stored information about an asset from the database.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Dictionary with asset info or None if not found
        """
        try:
            async with self.AsyncSessionLocal() as session:
                # Get data from both tables using JOIN
                result = await session.execute(
                    select(DataSeries, MarketAssets)
                    .join(MarketAssets, DataSeries.series_id == MarketAssets.series_id)
                    .where(DataSeries.series_id == symbol.upper())
                )
                
                row = result.first()
                
                if row:
                    series, market = row
                    
                    # Get observation count and date range
                    obs_stats = await session.execute(
                        select(
                            func.count(TimeSeriesObservation.id),
                            func.min(TimeSeriesObservation.observation_date),
                            func.max(TimeSeriesObservation.observation_date)
                        ).where(TimeSeriesObservation.series_id == symbol.upper())
                    )
                    
                    count, min_date, max_date = obs_stats.first()
                    
                    return {
                        'symbol': market.symbol,
                        'title': series.title,
                        'description': series.description,
                        'asset_type': market.asset_type,
                        'sector': market.sector,
                        'industry': market.industry,
                        'market_cap': market.market_cap,
                        'exchange': market.exchange,
                        'observation_count': count or 0,
                        'date_range_start': min_date,
                        'date_range_end': max_date,
                        'last_updated': series.last_updated,
                        'data_quality_score': series.data_quality_score,
                        'correlation_priority': series.correlation_priority
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting asset info for {symbol}: {e}")
            return None
    
    async def _log_sync_operation(
        self, 
        symbol: str, 
        start_time: datetime, 
        stats: Dict[str, Any], 
        success: bool, 
        error_message: Optional[str] = None
    ) -> None:
        """Log sync operation to data_sync_log table"""
        try:
            sync_log = DataSyncLog(
                series_id=symbol.upper(),
                source_type=DataSourceType.YAHOO_FINANCE,
                sync_type='bulk_load',
                sync_start_time=start_time,
                sync_end_time=datetime.now(timezone.utc),
                sync_duration_ms=int(stats['duration_seconds'] * 1000),
                success=success,
                records_processed=stats['total_observations'],
                records_added=stats['observations_inserted'],
                records_updated=stats['observations_updated'],
                records_failed=0 if success else stats['total_observations'],
                error_message=error_message,
                error_type='bulk_load_error' if error_message else None,
                data_quality_score=1.0 if success else 0.0,
                sync_parameters={'period': 'bulk_load', 'symbol': symbol}
            )
            
            async with self.AsyncSessionLocal() as session:
                session.add(sync_log)
                await session.commit()
                
        except Exception as e:
            logger.warning(f"Failed to log sync operation: {e}")
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about stored Yahoo Finance data"""
        try:
            async with self.AsyncSessionLocal() as session:
                # Count assets by source
                assets_count = await session.execute(
                    select(func.count(DataSeries.series_id))
                    .where(DataSeries.source_type == DataSourceType.YAHOO_FINANCE)
                )
                
                # Count total observations
                obs_count = await session.execute(
                    select(func.count(TimeSeriesObservation.id))
                    .join(DataSeries)
                    .where(DataSeries.source_type == DataSourceType.YAHOO_FINANCE)
                )
                
                # Count by asset type
                asset_type_stats = await session.execute(
                    select(MarketAssets.asset_type, func.count(MarketAssets.series_id))
                    .group_by(MarketAssets.asset_type)
                )
                
                # Get recent sync stats
                recent_syncs = await session.execute(
                    select(func.count(DataSyncLog.id))
                    .where(
                        and_(
                            DataSyncLog.source_type == DataSourceType.YAHOO_FINANCE,
                            DataSyncLog.success == True
                        )
                    )
                )
                
                asset_types = {str(row[0]): row[1] for row in asset_type_stats.fetchall()}
                
                return {
                    'yahoo_assets_count': assets_count.scalar() or 0,
                    'yahoo_observations_count': obs_count.scalar() or 0,
                    'asset_types': asset_types,
                    'successful_syncs': recent_syncs.scalar() or 0,
                    'last_updated': datetime.now(timezone.utc)
                }
                
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    async def close(self) -> None:
        """Close database connections"""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections closed")


# Convenience function for testing
async def test_database_integration():
    """Test the database integration with a sample asset"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    database_url = os.getenv('DATABASE_URL', 'postgresql+psycopg://postgres:fred_password@localhost:5432/postgres')
    
    db_integration = YahooDatabaseIntegration(database_url)
    
    try:
        # Initialize
        if not await db_integration.initialize():
            print("❌ Failed to initialize database connection")
            return False
        
        print("✅ Database connection initialized")
        
        # Test with SPY (recent data only to keep it fast)
        stats = await db_integration.bulk_load_asset('SPY', period='5d')
        
        if stats['success']:
            print(f"✅ Bulk load successful:")
            print(f"   Asset: {stats['symbol']}")
            print(f"   Observations: {stats['observations_inserted']} inserted")
            print(f"   Duration: {stats['duration_seconds']:.2f} seconds")
            
            # Get asset info
            info = await db_integration.get_asset_info('SPY')
            if info:
                print(f"✅ Asset info retrieved:")
                print(f"   Title: {info['title']}")
                print(f"   Asset Type: {info['asset_type']}")
                print(f"   Market Cap: ${info['market_cap']:,}")
                print(f"   Observations in DB: {info['observation_count']}")
        else:
            print(f"❌ Bulk load failed: {stats['error_message']}")
            return False
        
        # Get database stats
        db_stats = await db_integration.get_database_stats()
        print(f"✅ Database stats: {db_stats['yahoo_assets_count']} assets, {db_stats['yahoo_observations_count']} observations")
        print(f"   Asset types: {db_stats['asset_types']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    finally:
        await db_integration.close()


if __name__ == "__main__":
    """Test the database integration when run directly"""
    import asyncio
    asyncio.run(test_database_integration())
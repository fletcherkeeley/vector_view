"""
FRED Database Integration - Data Storage Layer

This module handles storing FRED economic data in our PostgreSQL database.
It integrates with the fetcher to store both metadata and time series observations.

Key Features:
- Store series metadata in data_series table
- Bulk insert time series observations
- Handle duplicate data gracefully (upsert operations)
- Track data sync operations in data_sync_log
- Support for incremental and full historical loads

Depends on: 
- fred_series_fetcher.py for data formatting
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
    DataSeries, TimeSeriesObservation, DataSyncLog, 
    DataSourceType, FrequencyType
)

from fred_series_fetcher import FredSeriesFetcher

# Configure logging
logger = logging.getLogger(__name__)


class FredDatabaseIntegration:
    """
    Handles storing FRED data in our PostgreSQL database with comprehensive error handling.
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
        
        logger.info("FRED Database Integration initialized")
    
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
    
    async def store_series_metadata(self, metadata: Dict[str, Any]) -> bool:
        """
        Store or update series metadata in the data_series table.
        
        Args:
            metadata: Series metadata from fred_series_fetcher
            
        Returns:
            True if successful, False otherwise
        """
        try:
            async with self.AsyncSessionLocal() as session:
                # Use PostgreSQL UPSERT (INSERT ... ON CONFLICT)
                stmt = insert(DataSeries).values(metadata)
                stmt = stmt.on_conflict_do_update(
                    index_elements=['series_id'],
                    set_={
                        'title': stmt.excluded.title,
                        'description': stmt.excluded.description,
                        'category': stmt.excluded.category,
                        'frequency': stmt.excluded.frequency,
                        'units': stmt.excluded.units,
                        'units_short': stmt.excluded.units_short,
                        'seasonal_adjustment': stmt.excluded.seasonal_adjustment,
                        'observation_start': stmt.excluded.observation_start,
                        'observation_end': stmt.excluded.observation_end,
                        'last_updated': stmt.excluded.last_updated,
                        'source_metadata': stmt.excluded.source_metadata,
                        'data_quality_score': stmt.excluded.data_quality_score,
                        'news_categories': stmt.excluded.news_categories,
                        'correlation_priority': stmt.excluded.correlation_priority,
                        'updated_at': stmt.excluded.updated_at
                    }
                )
                
                await session.execute(stmt)
                await session.commit()
                
                logger.info(f"Successfully stored metadata for series: {metadata['series_id']}")
                return True
                
        except SQLAlchemyError as e:
            logger.error(f"Database error storing metadata for {metadata.get('series_id')}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error storing metadata: {e}")
            return False
    
    async def store_observations(self, observations: List[Dict[str, Any]]) -> Tuple[int, int]:
        """
        Store time series observations in bulk with conflict handling.
        
        Args:
            observations: List of observation dictionaries from fred_series_fetcher
            
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
    
    async def bulk_load_series(self, series_id: str, start_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Load complete historical data for a FRED series.
        
        Args:
            series_id: FRED series identifier (e.g., 'FEDFUNDS')
            start_date: Optional start date in YYYY-MM-DD format
            
        Returns:
            Dictionary with load statistics and results
        """
        sync_start_time = datetime.now(timezone.utc)
        load_stats = {
            'series_id': series_id,
            'success': False,
            'metadata_stored': False,
            'observations_inserted': 0,
            'observations_updated': 0,
            'total_observations': 0,
            'error_message': None,
            'duration_seconds': 0
        }
        
        try:
            logger.info(f"Starting bulk load for series: {series_id}")
            
            # Use the fetcher to get complete data
            async with FredSeriesFetcher() as fetcher:
                metadata, observations = await fetcher.get_series_complete_data(series_id)
                
                # Filter observations by start_date if provided
                if start_date and observations:
                    filter_date = datetime.strptime(start_date, '%Y-%m-%d').date()
                    observations = [obs for obs in observations if obs['observation_date'] >= filter_date]
                
                load_stats['total_observations'] = len(observations)
                
                # Store metadata
                metadata_success = await self.store_series_metadata(metadata)
                load_stats['metadata_stored'] = metadata_success
                
                if not metadata_success:
                    raise Exception("Failed to store series metadata")
                
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
                        logger.info(f"Processed {i + len(batch)}/{len(observations)} observations for {series_id}")
                
                load_stats['observations_inserted'] = total_inserted
                load_stats['observations_updated'] = total_updated
                load_stats['success'] = True
                
                # Log sync operation
                await self._log_sync_operation(series_id, sync_start_time, load_stats, True)
                
                logger.info(f"Bulk load completed for {series_id}: {total_inserted} inserted, {total_updated} updated")
                
        except Exception as e:
            error_msg = str(e)
            load_stats['error_message'] = error_msg
            logger.error(f"Bulk load failed for {series_id}: {error_msg}")
            
            # Log failed sync operation
            await self._log_sync_operation(series_id, sync_start_time, load_stats, False, error_msg)
        
        finally:
            # Calculate duration
            load_stats['duration_seconds'] = (datetime.now(timezone.utc) - sync_start_time).total_seconds()
        
        return load_stats
    
    async def get_series_info(self, series_id: str) -> Optional[Dict[str, Any]]:
        """
        Get stored information about a series from the database.
        
        Args:
            series_id: FRED series identifier
            
        Returns:
            Dictionary with series info or None if not found
        """
        try:
            async with self.AsyncSessionLocal() as session:
                result = await session.execute(
                    select(DataSeries).where(DataSeries.series_id == series_id)
                )
                series = result.scalar_one_or_none()
                
                if series:
                    # Get observation count and date range
                    obs_stats = await session.execute(
                        select(
                            func.count(TimeSeriesObservation.id),
                            func.min(TimeSeriesObservation.observation_date),
                            func.max(TimeSeriesObservation.observation_date)
                        ).where(TimeSeriesObservation.series_id == series_id)
                    )
                    
                    count, min_date, max_date = obs_stats.first()
                    
                    return {
                        'series_id': series.series_id,
                        'title': series.title,
                        'description': series.description,
                        'category': series.category,
                        'frequency': series.frequency,
                        'units': series.units,
                        'observation_count': count or 0,
                        'date_range_start': min_date,
                        'date_range_end': max_date,
                        'last_updated': series.last_updated,
                        'data_quality_score': series.data_quality_score,
                        'correlation_priority': series.correlation_priority
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting series info for {series_id}: {e}")
            return None
    
    async def _log_sync_operation(
        self, 
        series_id: str, 
        start_time: datetime, 
        stats: Dict[str, Any], 
        success: bool, 
        error_message: Optional[str] = None
    ) -> None:
        """Log sync operation to data_sync_log table"""
        try:
            sync_log = DataSyncLog(
                series_id=series_id,
                source_type=DataSourceType.FRED,
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
                sync_parameters={'start_date': None, 'series_id': series_id}
            )
            
            async with self.AsyncSessionLocal() as session:
                session.add(sync_log)
                await session.commit()
                
        except Exception as e:
            logger.warning(f"Failed to log sync operation: {e}")
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about stored FRED data"""
        try:
            async with self.AsyncSessionLocal() as session:
                # Count series by source
                series_count = await session.execute(
                    select(func.count(DataSeries.series_id))
                    .where(DataSeries.source_type == DataSourceType.FRED)
                )
                
                # Count total observations
                obs_count = await session.execute(
                    select(func.count(TimeSeriesObservation.id))
                    .join(DataSeries)
                    .where(DataSeries.source_type == DataSourceType.FRED)
                )
                
                # Get recent sync stats
                recent_syncs = await session.execute(
                    select(func.count(DataSyncLog.id))
                    .where(
                        and_(
                            DataSyncLog.source_type == DataSourceType.FRED,
                            DataSyncLog.success == True
                        )
                    )
                )
                
                return {
                    'fred_series_count': series_count.scalar() or 0,
                    'fred_observations_count': obs_count.scalar() or 0,
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
    """Test the database integration with a sample series"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    database_url = os.getenv('DATABASE_URL', 'postgresql+psycopg://postgres:fred_password@localhost:5432/postgres')
    
    db_integration = FredDatabaseIntegration(database_url)
    
    try:
        # Initialize
        if not await db_integration.initialize():
            print("❌ Failed to initialize database connection")
            return False
        
        print("✅ Database connection initialized")
        
        # Test with a small series (recent data only)
        stats = await db_integration.bulk_load_series('FEDFUNDS', start_date='2024-01-01')
        
        if stats['success']:
            print(f"✅ Bulk load successful:")
            print(f"   Series: {stats['series_id']}")
            print(f"   Observations: {stats['observations_inserted']} inserted")
            print(f"   Duration: {stats['duration_seconds']:.2f} seconds")
            
            # Get series info
            info = await db_integration.get_series_info('FEDFUNDS')
            if info:
                print(f"✅ Series info retrieved:")
                print(f"   Title: {info['title']}")
                print(f"   Observations in DB: {info['observation_count']}")
        else:
            print(f"❌ Bulk load failed: {stats['error_message']}")
            return False
        
        # Get database stats
        db_stats = await db_integration.get_database_stats()
        print(f"✅ Database stats: {db_stats['fred_series_count']} series, {db_stats['fred_observations_count']} observations")
        
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
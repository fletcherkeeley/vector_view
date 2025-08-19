"""
FRED Series Fetcher - Business Logic Layer

This module handles the business logic for fetching FRED economic data series.
It uses the FredClient foundation to make API calls and formats data for database storage.

Key Features:
- Fetch series metadata and observations
- Map FRED data to our database schema format
- Handle data validation and quality scoring
- Support for different data frequencies and date ranges

Depends on: fred_client.py for networking foundation
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date, timezone
from decimal import Decimal

# Import database enums
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "database"))
from unified_database_setup import DataSourceType, FrequencyType

from fred_client import FredClient, FredAPIError

# Configure logging
logger = logging.getLogger(__name__)


class FredSeriesFetcher:
    """
    High-level business logic for fetching and processing FRED economic data series.
    
    This class handles the business operations while delegating networking to FredClient.
    """
    
    def __init__(self, client: Optional[FredClient] = None):
        """
        Initialize the series fetcher.
        
        Args:
            client: Optional FredClient instance. If None, creates a new one.
        """
        self.client = client
        self._client_owned = client is None  # Track if we own the client for cleanup
        
        logger.info("FRED Series Fetcher initialized")
    
    async def __aenter__(self):
        """Async context manager entry"""
        if self.client is None:
            self.client = FredClient()
        await self.client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._client_owned and self.client:
            await self.client.__aexit__(exc_type, exc_val, exc_tb)
    
    async def get_series_metadata(self, series_id: str) -> Dict[str, Any]:
        """
        Fetch comprehensive metadata for a FRED series.
        
        Args:
            series_id: FRED series identifier (e.g., 'FEDFUNDS', 'GDP')
            
        Returns:
            Dictionary with series metadata formatted for our database schema
            
        Raises:
            FredAPIError: If series doesn't exist or API error occurs
        """
        try:
            logger.info(f"Fetching metadata for series: {series_id}")
            
            # Get basic series information
            response = await self.client._make_request('series', {'series_id': series_id})
            
            if 'seriess' not in response or not response['seriess']:
                raise FredAPIError(f"Series {series_id} not found")
            
            series_data = response['seriess'][0]
            
            # Map FRED data to our database schema
            metadata = self._map_series_to_schema(series_data)
            
            logger.info(f"Successfully fetched metadata for {series_id}")
            return metadata
            
        except FredAPIError:
            raise
        except Exception as e:
            logger.error(f"Error fetching metadata for {series_id}: {e}")
            raise FredAPIError(f"Failed to fetch series metadata: {e}")
    
    async def get_series_observations(
        self, 
        series_id: str, 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch time series observations for a FRED series.
        
        Args:
            series_id: FRED series identifier
            start_date: Start date in YYYY-MM-DD format (optional)
            end_date: End date in YYYY-MM-DD format (optional)
            limit: Maximum number of observations to fetch (optional)
            
        Returns:
            List of observation dictionaries formatted for database storage
            
        Raises:
            FredAPIError: If series doesn't exist or API error occurs
        """
        try:
            logger.info(f"Fetching observations for series: {series_id}")
            
            # Build request parameters
            params = {'series_id': series_id}
            if start_date:
                params['observation_start'] = start_date
            if end_date:
                params['observation_end'] = end_date
            if limit:
                params['limit'] = str(limit)
            
            # Get observations from FRED
            response = await self.client._make_request('series/observations', params)
            
            if 'observations' not in response:
                raise FredAPIError(f"No observations found for series {series_id}")
            
            # Map observations to our database schema
            observations = []
            for obs in response['observations']:
                mapped_obs = self._map_observation_to_schema(series_id, obs)
                if mapped_obs:  # Skip invalid observations
                    observations.append(mapped_obs)
            
            logger.info(f"Successfully fetched {len(observations)} observations for {series_id}")
            return observations
            
        except FredAPIError:
            raise
        except Exception as e:
            logger.error(f"Error fetching observations for {series_id}: {e}")
            raise FredAPIError(f"Failed to fetch observations: {e}")
    
    async def get_series_complete_data(self, series_id: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Fetch both metadata and observations for a series in one operation.
        
        Args:
            series_id: FRED series identifier
            
        Returns:
            Tuple of (metadata_dict, observations_list)
        """
        logger.info(f"Fetching complete data for series: {series_id}")
        
        # Fetch metadata and observations concurrently would be nice, but let's keep it simple for now
        metadata = await self.get_series_metadata(series_id)
        observations = await self.get_series_observations(series_id)
        
        return metadata, observations
    
    def _map_series_to_schema(self, series_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map FRED series data to our database schema format.
        
        Args:
            series_data: Raw series data from FRED API
            
        Returns:
            Dictionary formatted for data_series table
        """
        # Parse dates safely
        def parse_date(date_str: str) -> Optional[date]:
            if not date_str or date_str == '.':
                return None
            try:
                return datetime.strptime(date_str, '%Y-%m-%d').date()
            except ValueError:
                logger.warning(f"Invalid date format: {date_str}")
                return None
        
        # Map FRED frequency to our enum
        frequency_mapping = {
            'd': FrequencyType.DAILY,
            'w': FrequencyType.WEEKLY, 
            'm': FrequencyType.MONTHLY,
            'q': FrequencyType.QUARTERLY,
            'a': FrequencyType.ANNUALLY,
            'sa': FrequencyType.ANNUALLY,  # Semiannual maps to annual
        }
        
        fred_frequency = series_data.get('frequency_short', '').lower()
        mapped_frequency = frequency_mapping.get(fred_frequency, FrequencyType.IRREGULAR)
        
        # Calculate data quality score based on available metadata
        quality_score = self._calculate_quality_score(series_data)
        
        return {
            'series_id': series_data['id'],  # Use FRED series ID as our series_id
            'source_type': DataSourceType.FRED,
            'source_series_id': series_data['id'],
            'title': series_data.get('title', ''),
            'description': series_data.get('notes', ''),
            'category': self._extract_category(series_data),
            'subcategory': None,  # FRED doesn't provide subcategory directly
            'frequency': mapped_frequency,
            'units': series_data.get('units', ''),
            'units_short': series_data.get('units_short', ''),
            'seasonal_adjustment': series_data.get('seasonal_adjustment', ''),
            'observation_start': parse_date(series_data.get('observation_start')),
            'observation_end': parse_date(series_data.get('observation_end')),
            'last_updated': datetime.now(timezone.utc),  # We'll update this with actual FRED timestamp later
            'source_metadata': {
                'fred_id': series_data['id'],
                'frequency': series_data.get('frequency'),
                'frequency_short': series_data.get('frequency_short'),
                'last_updated': series_data.get('last_updated'),
                'popularity': series_data.get('popularity'),
                'group_popularity': series_data.get('group_popularity'),
                'original_source': series_data.get('source', 'FRED')
            },
            'is_active': True,
            'data_quality_score': quality_score,
            'news_categories': self._map_to_news_categories(series_data),
            'correlation_priority': self._assign_correlation_priority(series_data),
            'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc)
        }
    
    def _map_observation_to_schema(self, series_id: str, observation: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Map a FRED observation to our database schema format.
        
        Args:
            series_id: The series this observation belongs to
            observation: Raw observation data from FRED
            
        Returns:
            Dictionary formatted for time_series_observations table, or None if invalid
        """
        # Skip observations with missing values
        value_str = observation.get('value', '.')
        if value_str == '.' or value_str == '' or value_str is None:
            return None
        
        # Parse the numeric value
        try:
            value = Decimal(str(value_str))
        except (ValueError, TypeError):
            logger.warning(f"Invalid value for {series_id}: {value_str}")
            return None
        
        # Parse observation date
        try:
            obs_date = datetime.strptime(observation['date'], '%Y-%m-%d').date()
        except ValueError:
            logger.warning(f"Invalid date for {series_id}: {observation.get('date')}")
            return None
        
        return {
            'series_id': series_id,
            'observation_date': obs_date,
            'observation_datetime': None,  # FRED data typically doesn't have specific times
            'value': value,
            'value_high': None,  # FRED doesn't provide OHLC data
            'value_low': None,
            'value_open': None,
            'value_close': None,
            'volume': None,  # FRED economic data doesn't have volume
            'data_quality': Decimal('1.0'),  # FRED data is generally high quality
            'is_estimated': False,  # We'd need to check FRED metadata for this
            'is_revised': False,  # FRED does provide revision info, but not in basic observations
            'revision_count': 0,
            'realtime_start': None,  # Could parse from FRED realtime data
            'realtime_end': None,
            'source_timestamp': datetime.now(timezone.utc),
            'observation_metadata': {
                'fred_date': observation['date'],
                'fred_value': observation['value']
            },
            'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc)
        }
    
    def _calculate_quality_score(self, series_data: Dict[str, Any]) -> Decimal:
        """Calculate a data quality score based on series metadata"""
        score = Decimal('0.8')  # Base score
        
        # Boost score for popular series
        popularity = series_data.get('popularity', 0)
        if isinstance(popularity, (int, float)) and popularity > 50:
            score += Decimal('0.1')
        
        # Boost for recent data
        if series_data.get('observation_end'):
            try:
                end_date = datetime.strptime(series_data['observation_end'], '%Y-%m-%d').date()
                days_old = (datetime.now().date() - end_date).days
                if days_old < 90:  # Data within last 3 months
                    score += Decimal('0.1')
            except ValueError:
                pass
        
        return min(score, Decimal('1.0'))  # Cap at 1.0
    
    def _extract_category(self, series_data: Dict[str, Any]) -> Optional[str]:
        """Extract category from series title or metadata"""
        title = series_data.get('title', '').lower()
        
        # Simple category mapping based on common terms
        if any(term in title for term in ['unemployment', 'employment', 'jobs', 'payroll']):
            return 'Employment'
        elif any(term in title for term in ['interest rate', 'fed funds', 'treasury']):
            return 'Interest Rates'
        elif any(term in title for term in ['price', 'inflation', 'cpi']):
            return 'Inflation'
        elif any(term in title for term in ['gdp', 'gross domestic']):
            return 'Economic Growth'
        elif any(term in title for term in ['housing', 'construction']):
            return 'Housing'
        else:
            return 'Economic Indicators'
    
    def _map_to_news_categories(self, series_data: Dict[str, Any]) -> List[str]:
        """Map series to relevant news categories for vector database integration"""
        category = self._extract_category(series_data)
        
        category_mapping = {
            'Employment': ['employment'],
            'Interest Rates': ['federal_reserve'],
            'Inflation': ['inflation'],
            'Economic Growth': ['gdp_growth'],
            'Housing': ['gdp_growth'],  # Housing affects GDP
        }
        
        return category_mapping.get(category, ['economic_indicators'])
    
    def _assign_correlation_priority(self, series_data: Dict[str, Any]) -> int:
        """Assign correlation calculation priority based on series importance"""
        # High priority for key economic indicators
        key_series = {
            'FEDFUNDS': 10,
            'UNRATE': 10, 
            'GDP': 10,
            'CPIAUCSL': 10,
            'DGS10': 9,
            'PAYEMS': 9
        }
        
        series_id = series_data.get('id', '')
        return key_series.get(series_id, 5)  # Default priority


# Convenience function for testing
async def test_series_fetcher():
    """Test function to verify the series fetcher works"""
    async with FredSeriesFetcher() as fetcher:
        try:
            # Test with Federal Funds Rate
            metadata = await fetcher.get_series_metadata('FEDFUNDS')
            print(f"✅ Metadata fetch successful for FEDFUNDS")
            print(f"   Title: {metadata['title']}")
            print(f"   Frequency: {metadata['frequency']}")
            
            # Test with a few observations
            observations = await fetcher.get_series_observations('FEDFUNDS', limit=5)
            print(f"✅ Observations fetch successful: {len(observations)} records")
            
            if observations:
                latest = observations[-1]
                print(f"   Latest: {latest['observation_date']} = {latest['value']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False


if __name__ == "__main__":
    """Test the fetcher when run directly"""
    import asyncio
    asyncio.run(test_series_fetcher())
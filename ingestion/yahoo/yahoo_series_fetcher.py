"""
Yahoo Finance Series Fetcher - Business Logic Layer

This module handles the business logic for fetching Yahoo Finance market data.
It uses the YahooFinanceClient foundation and formats data for database storage.

Key Features:
- Fetch market asset metadata and OHLCV observations
- Map Yahoo Finance data to our unified database schema
- Handle data validation and quality scoring
- Support for stocks, ETFs, and indices
- Asset classification and sector mapping

Depends on: yahoo_finance_client.py for networking foundation
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, date, timezone
from decimal import Decimal
import pandas as pd

# Import database enums
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "database"))
from unified_database_setup import DataSourceType, FrequencyType, AssetType

from .yahoo_finance_client import YahooFinanceClient, YahooFinanceError

# Configure logging
logger = logging.getLogger(__name__)


class YahooSeriesFetcher:
    """
    High-level business logic for fetching and processing Yahoo Finance market data.
    
    This class handles the business operations while delegating networking to YahooFinanceClient.
    """
    
    def __init__(self, client: Optional[YahooFinanceClient] = None):
        """
        Initialize the series fetcher.
        
        Args:
            client: Optional YahooFinanceClient instance. If None, creates a new one.
        """
        self.client = client
        self._client_owned = client is None  # Track if we own the client for cleanup
        
        logger.info("Yahoo Finance Series Fetcher initialized")
    
    async def __aenter__(self):
        """Async context manager entry"""
        if self.client is None:
            self.client = YahooFinanceClient()
        await self.client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._client_owned and self.client:
            await self.client.__aexit__(exc_type, exc_val, exc_tb)
    
    async def get_asset_metadata(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch comprehensive metadata for a market asset.
        
        Args:
            symbol: Stock/ETF/Index symbol (e.g., 'AAPL', 'SPY', 'QQQ')
            
        Returns:
            Dictionary with asset metadata formatted for our database schema
            
        Raises:
            YahooFinanceError: If symbol doesn't exist or API error occurs
        """
        try:
            logger.info(f"Fetching metadata for asset: {symbol}")
            
            # Get symbol information from Yahoo Finance
            info = await self.client.get_symbol_info(symbol)
            
            # Get a small sample of historical data to validate and get date range
            sample_data = await self.client.get_historical_data(symbol, period='5d')
            
            if sample_data.empty:
                raise YahooFinanceError(f"No historical data available for {symbol}")
            
            # Map Yahoo Finance data to our database schema
            metadata = self._map_asset_to_schema(symbol, info, sample_data)
            
            logger.info(f"Successfully fetched metadata for {symbol}")
            return metadata
            
        except YahooFinanceError:
            raise
        except Exception as e:
            logger.error(f"Error fetching metadata for {symbol}: {e}")
            raise YahooFinanceError(f"Failed to fetch asset metadata: {e}")
    
    async def get_asset_observations(
        self, 
        symbol: str, 
        period: str = "1y",
        interval: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch time series observations for a market asset.
        
        Args:
            symbol: Stock/ETF/Index symbol
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max')
            interval: Data interval ('1d', '1wk', '1mo') - daily is most common for storage
            start_date: Start date in YYYY-MM-DD format (optional)
            end_date: End date in YYYY-MM-DD format (optional)
            
        Returns:
            List of observation dictionaries formatted for database storage
            
        Raises:
            YahooFinanceError: If symbol doesn't exist or API error occurs
        """
        try:
            logger.info(f"Fetching observations for asset: {symbol}")
            
            # Get historical data from Yahoo Finance
            data = await self.client.get_historical_data(
                symbol=symbol,
                period=period,
                interval=interval,
                start_date=start_date,
                end_date=end_date
            )
            
            if data.empty:
                raise YahooFinanceError(f"No observations found for symbol {symbol}")
            
            # Map observations to our database schema
            observations = []
            for date_idx, row in data.iterrows():
                mapped_obs = self._map_observation_to_schema(symbol, date_idx, row)
                if mapped_obs:  # Skip invalid observations
                    observations.append(mapped_obs)
            
            logger.info(f"Successfully fetched {len(observations)} observations for {symbol}")
            return observations
            
        except YahooFinanceError:
            raise
        except Exception as e:
            logger.error(f"Error fetching observations for {symbol}: {e}")
            raise YahooFinanceError(f"Failed to fetch observations: {e}")
    
    async def get_asset_complete_data(
        self, 
        symbol: str, 
        period: str = "1y"
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Fetch both metadata and observations for an asset in one operation.
        
        Args:
            symbol: Stock/ETF/Index symbol
            period: Time period for historical data
            
        Returns:
            Tuple of (metadata_dict, observations_list)
        """
        logger.info(f"Fetching complete data for asset: {symbol}")
        
        # Fetch metadata and observations
        metadata = await self.get_asset_metadata(symbol)
        observations = await self.get_asset_observations(symbol, period=period)
        
        return metadata, observations
    
    def _map_asset_to_schema(self, symbol: str, info: Dict[str, Any], sample_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Map Yahoo Finance asset data to our database schema format.
        
        Args:
            symbol: Asset symbol
            info: Yahoo Finance info dictionary
            sample_data: Sample historical data for date range
            
        Returns:
            Dictionary formatted for data_series and market_assets tables
        """
        # Parse dates from sample data
        if not sample_data.empty:
            observation_start = sample_data.index.min().date() if hasattr(sample_data.index.min(), 'date') else None
            observation_end = sample_data.index.max().date() if hasattr(sample_data.index.max(), 'date') else None
        else:
            observation_start = None
            observation_end = None
        
        # Determine asset type and classification
        asset_type, category = self._classify_asset(symbol, info)
        
        # Calculate data quality score
        quality_score = self._calculate_quality_score(info, sample_data)
        
        # Create series metadata for data_series table
        series_metadata = {
            'series_id': symbol.upper(),  # Use symbol as series_id
            'source_type': DataSourceType.YAHOO_FINANCE,
            'source_series_id': symbol.upper(),
            'title': info.get('longName', symbol.upper()),
            'description': self._build_description(info),
            'category': category,
            'subcategory': info.get('sector'),
            'frequency': FrequencyType.DAILY,  # Yahoo Finance data is typically daily
            'units': 'USD',  # Most data is in USD
            'units_short': '$',
            'seasonal_adjustment': 'Not Seasonally Adjusted',
            'observation_start': observation_start,
            'observation_end': observation_end,
            'last_updated': datetime.now(timezone.utc),
            'source_metadata': {
                'symbol': symbol.upper(),
                'exchange': info.get('exchange'),
                'currency': info.get('currency', 'USD'),
                'market': info.get('market'),
                'quote_type': info.get('quoteType'),
                'yahoo_symbol': info.get('symbol')
            },
            'is_active': True,
            'data_quality_score': quality_score,
            'news_categories': self._map_to_news_categories(info),
            'correlation_priority': self._assign_correlation_priority(symbol, info),
            'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc)
        }
        
        # Create market asset metadata for market_assets table
        market_metadata = {
            'series_id': symbol.upper(),
            'symbol': symbol.upper(),
            'exchange': info.get('exchange'),
            'asset_type': asset_type,
            'company_name': info.get('longName'),
            'sector': info.get('sector'),
            'industry': info.get('industry'),
            'market_cap': info.get('marketCap'),
            'currency': info.get('currency', 'USD'),
            'country': info.get('country', 'US'),
            'is_actively_traded': True,
            'expense_ratio': self._extract_expense_ratio(info),
            'index_tracked': self._extract_index_tracked(info),
            'economic_sensitivity': self._assess_economic_sensitivity(info),
            'sector_exposure': self._extract_sector_exposure(info),
            'created_at': datetime.now(timezone.utc),
            'updated_at': datetime.now(timezone.utc)
        }
        
        return {
            'series_metadata': series_metadata,
            'market_metadata': market_metadata
        }
    
    def _map_observation_to_schema(self, symbol: str, date_idx: Any, row: pd.Series) -> Optional[Dict[str, Any]]:
        """
        Map a Yahoo Finance observation to our database schema format.
        
        Args:
            symbol: Asset symbol
            date_idx: Date index from DataFrame
            row: Data row from DataFrame
            
        Returns:
            Dictionary formatted for time_series_observations table, or None if invalid
        """
        try:
            # Parse observation date
            if hasattr(date_idx, 'date'):
                obs_date = date_idx.date()
            else:
                obs_date = pd.to_datetime(date_idx).date()
            
            # Skip if we don't have valid OHLC data
            if pd.isna(row.get('Close')) or row.get('Close') <= 0:
                return None
            
            # Convert to Decimal for database storage
            def safe_decimal(value, default=None):
                if pd.isna(value) or value is None:
                    return default
                try:
                    return Decimal(str(value))
                except (ValueError, TypeError):
                    return default
            
            return {
                'series_id': symbol.upper(),
                'observation_date': obs_date,
                'observation_datetime': None,  # Daily data doesn't have specific times
                'value': safe_decimal(row.get('Close')),  # Use Close as primary value
                'value_high': safe_decimal(row.get('High')),
                'value_low': safe_decimal(row.get('Low')),
                'value_open': safe_decimal(row.get('Open')),
                'value_close': safe_decimal(row.get('Close')),
                'volume': safe_decimal(row.get('Volume')),
                'data_quality': Decimal('1.0'),  # Yahoo Finance data is generally high quality
                'is_estimated': False,
                'is_revised': False,  # Market data isn't typically revised like economic data
                'revision_count': 0,
                'realtime_start': None,
                'realtime_end': None,
                'source_timestamp': datetime.now(timezone.utc),
                'observation_metadata': {
                    'dividends': float(row.get('Dividends', 0)) if not pd.isna(row.get('Dividends')) else 0,
                    'stock_splits': float(row.get('Stock Splits', 0)) if not pd.isna(row.get('Stock Splits')) else 0,
                    'capital_gains': float(row.get('Capital Gains', 0)) if not pd.isna(row.get('Capital Gains')) else 0
                },
                'created_at': datetime.now(timezone.utc),
                'updated_at': datetime.now(timezone.utc)
            }
            
        except Exception as e:
            logger.warning(f"Error mapping observation for {symbol} on {date_idx}: {e}")
            return None
    
    def _classify_asset(self, symbol: str, info: Dict[str, Any]) -> Tuple[AssetType, str]:
        """Classify asset type and category based on symbol and info"""
        quote_type = info.get('quoteType', '').lower()
        symbol_upper = symbol.upper()
        
        # ETF classification
        if quote_type == 'etf' or 'etf' in info.get('longName', '').lower():
            # Check if it's a sector ETF
            if any(sector in info.get('longName', '').lower() for sector in ['sector', 'industry', 'financial', 'technology', 'healthcare', 'energy']):
                return AssetType.SECTOR_ETF, 'Sector ETFs'
            else:
                return AssetType.ETF, 'ETFs'
        
        # Index classification
        elif quote_type == 'index' or symbol_upper in ['SPX', 'NDX', 'DJI']:
            return AssetType.INDEX, 'Market Indices'
        
        # Stock classification
        elif quote_type == 'equity' or quote_type == 'stock':
            return AssetType.STOCK, 'Individual Stocks'
        
        # Default to ETF for popular market symbols like SPY, QQQ
        elif symbol_upper in ['SPY', 'QQQ', 'VTI', 'IWM', 'XLF', 'XLY', 'XLP', 'XLE', 'XLI']:
            if symbol_upper.startswith('XL'):
                return AssetType.SECTOR_ETF, 'Sector ETFs'
            else:
                return AssetType.ETF, 'ETFs'
        
        # Default classification
        else:
            return AssetType.STOCK, 'Individual Stocks'
    
    def _calculate_quality_score(self, info: Dict[str, Any], sample_data: pd.DataFrame) -> Decimal:
        """Calculate a data quality score based on asset metadata and data"""
        score = Decimal('0.8')  # Base score
        
        # Boost for large market cap
        market_cap = info.get('marketCap', 0)
        if market_cap and market_cap > 10_000_000_000:  # > $10B
            score += Decimal('0.1')
        
        # Boost for high trading volume
        if not sample_data.empty and 'Volume' in sample_data.columns:
            avg_volume = sample_data['Volume'].mean()
            if avg_volume > 1_000_000:  # > 1M average volume
                score += Decimal('0.1')
        
        return min(score, Decimal('1.0'))  # Cap at 1.0
    
    def _build_description(self, info: Dict[str, Any]) -> str:
        """Build a comprehensive description from Yahoo Finance info"""
        parts = []
        
        if info.get('longBusinessSummary'):
            parts.append(info['longBusinessSummary'][:500])  # Truncate long descriptions
        elif info.get('longName'):
            parts.append(info['longName'])
        
        if info.get('sector') and info.get('industry'):
            parts.append(f"Sector: {info['sector']}, Industry: {info['industry']}")
        
        return '. '.join(parts) if parts else f"Market data for {info.get('symbol', 'Unknown')}"
    
    def _extract_expense_ratio(self, info: Dict[str, Any]) -> Optional[Decimal]:
        """Extract expense ratio for ETFs"""
        expense_ratio = info.get('expenseRatio')
        if expense_ratio is not None:
            try:
                return Decimal(str(expense_ratio))
            except (ValueError, TypeError):
                pass
        return None
    
    def _extract_index_tracked(self, info: Dict[str, Any]) -> Optional[str]:
        """Extract underlying index for ETFs"""
        # This would need more sophisticated mapping based on ETF names
        name = info.get('longName', '').lower()
        if 's&p 500' in name:
            return 'S&P 500'
        elif 'nasdaq' in name or 'qqq' in name:
            return 'NASDAQ-100'
        elif 'russell 2000' in name:
            return 'Russell 2000'
        return None
    
    def _assess_economic_sensitivity(self, info: Dict[str, Any]) -> Dict[str, Any]:
        """Assess sensitivity to economic indicators"""
        sector = info.get('sector', '').lower()
        
        sensitivity = {
            'interest_rate_sensitive': sector in ['financial services', 'real estate'],
            'economic_cycle_sensitive': sector in ['technology', 'consumer cyclical', 'industrials'],
            'defensive': sector in ['utilities', 'consumer defensive', 'healthcare']
        }
        
        return sensitivity
    
    def _extract_sector_exposure(self, info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract sector exposure for ETFs (simplified)"""
        if info.get('quoteType') == 'etf':
            sector = info.get('sector')
            if sector:
                return {sector: 100.0}  # Simplified - real implementation would get holdings data
        return None
    
    def _map_to_news_categories(self, info: Dict[str, Any]) -> List[str]:
        """Map asset to relevant news categories for vector database integration"""
        categories = []
        
        sector = info.get('sector', '').lower()
        if 'financial' in sector:
            categories.append('financial')
        elif 'technology' in sector:
            categories.append('technology')
        elif 'healthcare' in sector:
            categories.append('healthcare')
        elif 'energy' in sector:
            categories.append('energy')
        
        # Add market volatility for all assets
        categories.append('market_volatility')
        
        return categories if categories else ['market_volatility']
    
    def _assign_correlation_priority(self, symbol: str, info: Dict[str, Any]) -> int:
        """Assign correlation calculation priority based on asset importance"""
        # High priority for major market indicators
        high_priority_symbols = {
            'SPY': 10,  # S&P 500
            'QQQ': 10,  # NASDAQ-100
            'VTI': 9,   # Total Stock Market
            'IWM': 9,   # Russell 2000
            'XLF': 8,   # Financial Sector
            'XLY': 8,   # Consumer Discretionary
            'XLE': 8,   # Energy Sector
        }
        
        symbol_upper = symbol.upper()
        if symbol_upper in high_priority_symbols:
            return high_priority_symbols[symbol_upper]
        
        # Medium priority for large market cap stocks
        market_cap = info.get('marketCap', 0)
        if market_cap and market_cap > 100_000_000_000:  # > $100B
            return 7
        elif market_cap and market_cap > 50_000_000_000:   # > $50B
            return 6
        
        # Default priority
        return 5


# Convenience function for testing
async def test_series_fetcher():
    """Test function to verify the series fetcher works"""
    async with YahooSeriesFetcher() as fetcher:
        try:
            # Test with SPY ETF
            metadata = await fetcher.get_asset_metadata('SPY')
            print(f"✅ Metadata fetch successful for SPY")
            print(f"   Title: {metadata['series_metadata']['title']}")
            print(f"   Asset Type: {metadata['market_metadata']['asset_type']}")
            print(f"   Market Cap: ${metadata['market_metadata']['market_cap']:,}")
            
            # Test with a few observations
            observations = await fetcher.get_asset_observations('SPY', period='5d')
            print(f"✅ Observations fetch successful: {len(observations)} records")
            
            if observations:
                latest = observations[-1]
                print(f"   Latest: {latest['observation_date']} = ${latest['value']}")
                print(f"   OHLC: ${latest['value_open']}, ${latest['value_high']}, ${latest['value_low']}, ${latest['value_close']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False


if __name__ == "__main__":
    """Test the fetcher when run directly"""
    import asyncio
    asyncio.run(test_series_fetcher())
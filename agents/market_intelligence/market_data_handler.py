"""
Market Data Handler for Vector View Financial Intelligence Platform

Handles all data access operations for market intelligence analysis including:
- PostgreSQL market data retrieval (FRED + Yahoo Finance)
- Market indicator data processing
- Data quality assessment and validation
- Cross-source data integration
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from sqlalchemy import create_engine, text
import numpy as np

logger = logging.getLogger(__name__)


class MarketDataHandler:
    """
    Handles all data access operations for market intelligence analysis.
    
    Responsibilities:
    - PostgreSQL connection and market data retrieval
    - Market indicator data processing and validation
    - Data quality assessment
    - Cross-source data integration (FRED + Yahoo Finance)
    """
    
    def __init__(self, database_url: str = "postgresql://postgres:fred_password@localhost:5432/postgres"):
        """
        Initialize the market data handler.
        
        Args:
            database_url: PostgreSQL connection string
        """
        self.database_url = database_url
        self.engine = None
        
        # Market sectors for impact analysis
        self.market_sectors = {
            'technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
            'financial': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
            'healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK'],
            'energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
            'consumer': ['WMT', 'PG', 'KO', 'PEP', 'COST']
        }
        
        # Key market indicators (using actual series_ids from database)
        self.key_indicators = [
            'SPY',     # S&P 500
            'QQQ',     # NASDAQ
            'TLT',     # 20+ Year Treasury Bond
            'GLD',     # Gold
            'VIXCLS'   # Volatility Index (FRED series)
        ]
        
        # Separate FRED and Yahoo Finance indicators
        self.fred_indicators = ['VIXCLS']
        self.yahoo_indicators = ['SPY', 'QQQ', 'TLT', 'GLD']
    
    def _get_engine(self):
        """Get or create database engine"""
        if not self.engine:
            self.engine = create_engine(self.database_url)
        return self.engine
    
    async def get_market_data(
        self, 
        indicators: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """
        Retrieve market data from PostgreSQL database.
        
        Args:
            timeframe: Time window for analysis
            date_range: Optional specific date range
            indicators: Optional list of specific indicators to retrieve
            
        Returns:
            DataFrame with market data indexed by date
        """
        try:
            engine = self._get_engine()
            
            # Use provided indicators or default key indicators
            target_indicators = indicators or self.key_indicators
            
            # Use broader date range to get more market data points for correlation
            cutoff_time = start_date - timedelta(days=30)  # Get 30 days of market data
            
            # Separate FRED and Yahoo Finance indicators
            fred_indicators = [ind for ind in target_indicators if ind in self.fred_indicators]
            yahoo_indicators = [ind for ind in target_indicators if ind in self.yahoo_indicators]
            
            all_data = []
            
            # Query FRED data
            if fred_indicators:
                fred_data = await self._query_fred_data(fred_indicators, cutoff_time, engine)
                all_data.extend(fred_data)
            
            # Query Yahoo Finance data
            if yahoo_indicators:
                yahoo_data = await self._query_yahoo_data(yahoo_indicators, cutoff_time, engine)
                all_data.extend(yahoo_data)
            
            if not all_data:
                logger.warning("No market data retrieved")
                return pd.DataFrame()
            
            # Convert to DataFrame
            market_df = pd.DataFrame(all_data, columns=['symbol', 'date', 'close_price'])
            market_df['date'] = pd.to_datetime(market_df['date'])
            market_df['close_price'] = market_df['close_price'].astype(float)
            
            # Pivot to have symbols as columns
            pivot_df = market_df.pivot_table(
                index='date',
                columns='symbol', 
                values='close_price',
                aggfunc='first'
            )
            
            logger.info(f"Retrieved market data: {len(pivot_df)} rows, {len(pivot_df.columns)} indicators")
            return pivot_df.fillna(method='ffill').dropna()
            
        except Exception as e:
            logger.error(f"Failed to retrieve market data: {str(e)}")
            return pd.DataFrame()
    
    async def _query_fred_data(self, indicators: List[str], cutoff_time: datetime, engine) -> List[Tuple]:
        """Query FRED data from PostgreSQL"""
        try:
            fred_symbols_str = "', '".join(indicators)
            fred_query = text(f"""
            SELECT 
                tso.series_id as symbol,
                tso.observation_date as date,
                tso.value as close_price
            FROM time_series_observations tso
            WHERE tso.series_id IN ('{fred_symbols_str}')
            AND tso.observation_date >= :cutoff_time
            AND tso.value IS NOT NULL
            ORDER BY tso.series_id, tso.observation_date DESC
            """)
            
            with engine.connect() as conn:
                result = conn.execute(fred_query, {"cutoff_time": cutoff_time})
                return result.fetchall()
                
        except Exception as e:
            logger.error(f"FRED data query failed: {str(e)}")
            return []
    
    async def _query_yahoo_data(self, indicators: List[str], cutoff_time: datetime, engine) -> List[Tuple]:
        """Query Yahoo Finance data from PostgreSQL"""
        try:
            yahoo_symbols_str = "', '".join(indicators)
            yahoo_query = text(f"""
            SELECT 
                tso.series_id as symbol,
                tso.observation_date as date,
                tso.value as close_price
            FROM time_series_observations tso
            WHERE tso.series_id IN ('{yahoo_symbols_str}')
            AND tso.observation_date >= :cutoff_time
            AND tso.value IS NOT NULL
            ORDER BY tso.series_id, tso.observation_date DESC
            """)
            
            with engine.connect() as conn:
                result = conn.execute(yahoo_query, {"cutoff_time": cutoff_time})
                return result.fetchall()
                
        except Exception as e:
            logger.error(f"Yahoo Finance data query failed: {str(e)}")
            return []
    
    async def get_sector_data(self, sector: str, timeframe: str = "1d") -> pd.DataFrame:
        """
        Get market data for a specific sector.
        
        Args:
            sector: Sector name (technology, financial, etc.)
            timeframe: Time window for analysis
            
        Returns:
            DataFrame with sector-specific market data
        """
        if sector not in self.market_sectors:
            logger.warning(f"Unknown sector: {sector}")
            return pd.DataFrame()
        
        sector_symbols = self.market_sectors[sector]
        return await self.get_market_data(timeframe=timeframe, indicators=sector_symbols)
    
    def calculate_returns(self, df: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
        """
        Calculate returns for market data.
        
        Args:
            df: Market data DataFrame
            periods: Number of periods for return calculation
            
        Returns:
            DataFrame with calculated returns
        """
        try:
            if df.empty:
                return pd.DataFrame()
            
            returns_df = df.pct_change(periods=periods).dropna()
            return returns_df
            
        except Exception as e:
            logger.error(f"Return calculation failed: {str(e)}")
            return pd.DataFrame()
    
    def calculate_volatility(self, df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
        """
        Calculate rolling volatility for market data.
        
        Args:
            df: Market data DataFrame
            window: Rolling window size
            
        Returns:
            DataFrame with volatility metrics
        """
        try:
            if df.empty:
                return pd.DataFrame()
            
            returns = self.calculate_returns(df)
            volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
            return volatility
            
        except Exception as e:
            logger.error(f"Volatility calculation failed: {str(e)}")
            return pd.DataFrame()
    
    def get_data_quality_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate data quality metrics for market data.
        
        Args:
            df: Market data DataFrame
            
        Returns:
            Dictionary with data quality metrics
        """
        try:
            if df.empty:
                return {'error': 'No data available'}
            
            metrics = {
                'total_observations': len(df),
                'indicators_available': len(df.columns),
                'date_range': {
                    'start': df.index.min().strftime('%Y-%m-%d') if not df.empty else None,
                    'end': df.index.max().strftime('%Y-%m-%d') if not df.empty else None,
                    'days_covered': (df.index.max() - df.index.min()).days if len(df) > 1 else 0
                },
                'data_completeness': {},
                'latest_values': {}
            }
            
            # Calculate completeness for each indicator
            for column in df.columns:
                total_points = len(df)
                valid_points = df[column].notna().sum()
                completeness = valid_points / total_points if total_points > 0 else 0
                metrics['data_completeness'][column] = completeness
                
                # Get latest value
                latest_value = df[column].dropna().iloc[-1] if not df[column].dropna().empty else None
                metrics['latest_values'][column] = float(latest_value) if latest_value is not None else None
            
            return metrics
            
        except Exception as e:
            logger.error(f"Data quality assessment failed: {str(e)}")
            return {'error': str(e)}
    
    def validate_connection(self) -> bool:
        """
        Validate database connection.
        
        Returns:
            True if connection is valid, False otherwise
        """
        try:
            engine = self._get_engine()
            with engine.connect() as conn:
                result = conn.execute(text("SELECT 1"))
                return result.fetchone() is not None
        except Exception as e:
            logger.error(f"Connection validation failed: {str(e)}")
            return False
    
    def _parse_timeframe_hours(self, timeframe: str) -> int:
        """Convert timeframe string to hours - adjusted for market data availability"""
        timeframe_map = {
            '1h': 48,    # Get 2 days of data for 1h analysis
            '4h': 168,   # Get 1 week of data for 4h analysis  
            '1d': 720,   # Get 1 month of data for daily analysis
            '1w': 2160,  # Get 3 months of data for weekly analysis
            '1m': 8760,  # Get 1 year of data for monthly analysis
            '3m': 17520, # Get 2 years of data for quarterly analysis
            '1y': 43800  # Get 5 years of data for yearly analysis
        }
        return timeframe_map.get(timeframe, 720)

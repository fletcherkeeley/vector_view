"""
Economic Data Handler - Manages data fetching and frequency-aware analysis
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from sqlalchemy import create_engine, text
import os
import logging

logger = logging.getLogger(__name__)

class EconomicDataHandler:
    """Handles economic data fetching with proper frequency awareness"""
    
    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL", "postgresql://postgres:fred_password@localhost:5432/postgres")
        self.engine = create_engine(self.db_url)
        
        # Data frequency mapping
        self.indicator_frequencies = {
            # Monthly indicators
            'UNRATE': 'MONTHLY',
            'CPIAUCSL': 'MONTHLY', 
            'PAYEMS': 'MONTHLY',
            'FEDFUNDS': 'MONTHLY',
            'AHETPI': 'MONTHLY',
            'PERMIT': 'MONTHLY',
            'HOUST': 'MONTHLY',
            'INDPRO': 'MONTHLY',
            'RETAILSMNSA': 'MONTHLY',
            'UMCSENT': 'MONTHLY',
            'CPILFESL': 'MONTHLY',
            'DSPIC96': 'MONTHLY',
            'PSAVERT': 'MONTHLY',
            
            # Daily indicators  
            'DGS10': 'DAILY',
            'DGS2': 'DAILY',
            'DGS3MO': 'DAILY',
            'T10Y2Y': 'DAILY',
            'T10Y3M': 'DAILY',
            'VIXCLS': 'DAILY',
            'DEXUSEU': 'DAILY',
            'DTWEXBGS': 'DAILY',
            
            # Weekly indicators
            'ICSA': 'WEEKLY',
            'CCSA': 'WEEKLY'
        }
    
    def get_frequency_appropriate_timeframes(self, frequency: str) -> Dict[str, int]:
        """Get appropriate analysis timeframes based on data frequency"""
        if frequency == 'DAILY':
            return {
                'short': 30,    # 1 month
                'medium': 90,   # 3 months  
                'long': 252,    # 1 year
                'extended': 504 # 2 years
            }
        elif frequency == 'WEEKLY':
            return {
                'short': 4,     # 1 month
                'medium': 13,   # 3 months
                'long': 52,     # 1 year
                'extended': 104 # 2 years
            }
        elif frequency == 'MONTHLY':
            return {
                'short': 1,     # 1 month
                'medium': 3,    # 3 months
                'long': 12,     # 1 year
                'extended': 24  # 2 years
            }
        else:
            # Default to monthly
            return {
                'short': 1,
                'medium': 3, 
                'long': 12,
                'extended': 24
            }
    
    def fetch_economic_data(self, indicators: List[str], start_date: datetime, end_date: datetime) -> Tuple[pd.DataFrame, datetime, datetime]:
        """Fetch economic data with frequency awareness"""
        try:
            # Build query for multiple indicators with proper parameterization
            placeholders = ', '.join([':indicator_' + str(i) for i in range(len(indicators))])
            
            query = f"""
            SELECT 
                tso.series_id,
                tso.observation_date,
                tso.value,
                ds.frequency,
                ds.title
            FROM time_series_observations tso
            JOIN data_series ds ON tso.series_id = ds.series_id
            WHERE tso.series_id IN ({placeholders})
                AND tso.observation_date >= :start_date
                AND tso.observation_date <= :end_date
                AND tso.value IS NOT NULL
            ORDER BY tso.series_id, tso.observation_date
            """
            
            # Build parameters dictionary
            params = {
                'start_date': start_date,
                'end_date': end_date
            }
            for i, indicator in enumerate(indicators):
                params[f'indicator_{i}'] = indicator
            
            with self.engine.connect() as conn:
                df = pd.read_sql_query(
                    text(query),
                    conn,
                    params=params
                )
            
            if df.empty:
                logger.warning(f"No data found for indicators {indicators} in date range {start_date} to {end_date}")
                return pd.DataFrame(), start_date, end_date
            
            # Convert observation_date to datetime
            df['observation_date'] = pd.to_datetime(df['observation_date'])
            
            # Pivot to have series as columns
            pivot_df = df.pivot_table(
                index="observation_date", 
                columns="series_id", 
                values="value", 
                aggfunc="first"
            )
            
            # Convert all values to float to handle Decimal types from database
            pivot_df = pivot_df.astype(float, errors='ignore')
            
            # Get frequency info for each indicator
            freq_df = df[['series_id', 'frequency']].drop_duplicates()
            self.frequency_map = dict(zip(freq_df['series_id'], freq_df['frequency']))
            
            return pivot_df.ffill().dropna(), start_date, end_date
            
        except Exception as e:
            logger.error(f"Error fetching economic data: {str(e)}")
            return pd.DataFrame(), start_date, end_date
    
    def get_latest_data_points(self, df: pd.DataFrame, indicator: str, periods: int) -> pd.Series:
        """Get the latest N data points for frequency-appropriate analysis"""
        if indicator not in df.columns:
            return pd.Series(dtype=float)
        
        frequency = self.frequency_map.get(indicator, 'MONTHLY')
        
        # For monthly data, ensure we get actual monthly observations
        if frequency == 'MONTHLY':
            # Get unique months and take the last N months
            monthly_data = df[indicator].dropna().resample('M').last().dropna()
            return monthly_data.tail(periods)
        elif frequency == 'WEEKLY':
            # Get weekly data
            weekly_data = df[indicator].dropna().resample('W').last().dropna()
            return weekly_data.tail(periods)
        else:
            # Daily data - can use as-is
            return df[indicator].dropna().tail(periods)
    
    def calculate_frequency_appropriate_change(self, df: pd.DataFrame, indicator: str, periods: int) -> float:
        """Calculate percentage change appropriate for the data frequency"""
        data_points = self.get_latest_data_points(df, indicator, periods + 1)
        
        if len(data_points) < 2:
            return 0.0
        
        # Ensure we have enough data points for the requested period
        if len(data_points) <= periods:
            # If not enough data for the full period, use what we have
            latest = data_points.iloc[-1]
            previous = data_points.iloc[0]  # Use first available point
        else:
            latest = data_points.iloc[-1]
            previous = data_points.iloc[-(periods + 1)]
        
        if previous == 0 or pd.isna(previous) or pd.isna(latest):
            return 0.0
        
        return ((latest - previous) / previous) * 100
    
    def get_data_quality_metrics(self, df: pd.DataFrame, indicators: List[str]) -> Dict[str, any]:
        """Calculate data quality metrics"""
        metrics = {
            'data_points_analyzed': len(df),
            'indicators_analyzed': len([col for col in indicators if col in df.columns]),
            'date_range': {
                'start': df.index.min().strftime('%Y-%m-%d') if not df.empty else None,
                'end': df.index.max().strftime('%Y-%m-%d') if not df.empty else None
            },
            'data_completeness': {}
        }
        
        for indicator in indicators:
            if indicator in df.columns:
                total_points = len(df)
                valid_points = df[indicator].notna().sum()
                metrics['data_completeness'][indicator] = valid_points / total_points if total_points > 0 else 0
        
        return metrics

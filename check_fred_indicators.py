#!/usr/bin/env python3
"""
Check available FRED economic indicators in the database
"""

import os
from sqlalchemy import create_engine, text

# Database connection
database_url = 'postgresql://postgres:fred_password@localhost:5432/postgres'
engine = create_engine(database_url)

try:
    with engine.connect() as conn:
        # Get all indicators (check schema first)
        result = conn.execute(text("""
            SELECT series_id, title, frequency, units
            FROM data_series 
            ORDER BY series_id
        """))
        
        indicators = result.fetchall()
        
        print(f"Found {len(indicators)} FRED indicators:")
        print("-" * 80)
        
        for indicator in indicators:
            print(f"{indicator.series_id:12} | {indicator.frequency:10} | {indicator.title}")
        
        # Check data availability for recent dates
        print(f"\n" + "="*80)
        print("DATA AVAILABILITY CHECK (Last 90 days)")
        print("="*80)
        
        result = conn.execute(text("""
            SELECT 
                ds.series_id,
                COUNT(tso.value) as data_points,
                MAX(tso.observation_date) as latest_date,
                MIN(tso.observation_date) as earliest_date
            FROM data_series ds
            LEFT JOIN time_series_observations tso ON ds.series_id = tso.series_id
                AND tso.observation_date >= CURRENT_DATE - INTERVAL '90 days'
            GROUP BY ds.series_id
            ORDER BY data_points DESC, ds.series_id
        """))
        
        availability = result.fetchall()
        
        for row in availability:
            print(f"{row.series_id:12} | {row.data_points:3} points | Latest: {row.latest_date}")

except Exception as e:
    print(f"Error: {e}")

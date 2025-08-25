#!/usr/bin/env python3
"""
Quick script to check what data is actually in the database
"""

import asyncio
from sqlalchemy import create_engine, text

async def check_database():
    """Check what's actually in the database"""
    database_url = "postgresql://postgres:fred_password@localhost:5432/postgres"
    engine = create_engine(database_url)
    
    try:
        with engine.connect() as conn:
            # Check table structure
            print("=== DATA_SERIES TABLE STRUCTURE ===")
            result = conn.execute(text("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'data_series' ORDER BY ordinal_position;"))
            columns = result.fetchall()
            for col in columns:
                print(f"  {col[0]}: {col[1]}")
            
            print("\n=== TOTAL DATA SERIES COUNT ===")
            result = conn.execute(text("SELECT COUNT(*) as total FROM data_series;"))
            total = result.fetchone()[0]
            print(f"  Total series: {total}")
            
            # Check for market indicators the agent is looking for
            print("\n=== CHECKING FOR MARKET INDICATORS ===")
            market_symbols = ['SPY', 'QQQ', 'TLT', 'GLD', 'VIXCLS', 'VIX']
            
            for symbol in market_symbols:
                result = conn.execute(text("SELECT series_id, title FROM data_series WHERE series_id = :symbol"), {"symbol": symbol})
                row = result.fetchone()
                if row:
                    print(f"  ✅ {symbol}: {row[1]}")
                else:
                    print(f"  ❌ {symbol}: NOT FOUND")
            
            # Check what series we actually have (sample)
            print("\n=== SAMPLE OF AVAILABLE SERIES ===")
            result = conn.execute(text("SELECT series_id, title FROM data_series ORDER BY series_id LIMIT 20;"))
            rows = result.fetchall()
            for row in rows:
                print(f"  {row[0]}: {row[1]}")
            
            # Check if we have any Yahoo Finance data
            print("\n=== CHECKING FOR YAHOO FINANCE PATTERNS ===")
            result = conn.execute(text("SELECT series_id, title FROM data_series WHERE series_id ~ '^[A-Z]{1,5}$' AND LENGTH(series_id) <= 5 ORDER BY series_id LIMIT 10;"))
            yahoo_rows = result.fetchall()
            if yahoo_rows:
                print("  Found potential Yahoo Finance symbols:")
                for row in yahoo_rows:
                    print(f"    {row[0]}: {row[1]}")
            else:
                print("  No obvious Yahoo Finance symbols found")
            
            # Check recent data availability
            print("\n=== RECENT DATA CHECK ===")
            result = conn.execute(text("""
                SELECT ds.series_id, COUNT(*) as obs_count, MAX(tso.observation_date) as latest_date
                FROM data_series ds 
                JOIN time_series_observations tso ON ds.series_id = tso.series_id 
                WHERE ds.series_id IN ('SPY', 'QQQ', 'TLT', 'GLD', 'VIXCLS', 'UNRATE', 'FEDFUNDS')
                GROUP BY ds.series_id
                ORDER BY latest_date DESC;
            """))
            recent_rows = result.fetchall()
            if recent_rows:
                for row in recent_rows:
                    print(f"  {row[0]}: {row[1]} observations, latest: {row[2]}")
            else:
                print("  No data found for target symbols")
                
    except Exception as e:
        print(f"Database error: {str(e)}")
    finally:
        engine.dispose()

if __name__ == "__main__":
    asyncio.run(check_database())

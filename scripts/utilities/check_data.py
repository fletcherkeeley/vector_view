import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

async def check():
    engine = create_async_engine('postgresql+psycopg://postgres:fred_password@localhost:5432/postgres')
    async with engine.begin() as conn:
        # Check if tables exist
        result = await conn.execute(text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"))
        tables = [row[0] for row in result.fetchall()]
        print('Existing tables:', tables)
        
        # Check data counts if tables exist
        if 'data_series' in tables:
            result = await conn.execute(text('SELECT COUNT(*) FROM data_series'))
            print('Total data series:', result.scalar())
        
        if 'time_series_observations' in tables:
            result = await conn.execute(text('SELECT COUNT(*) FROM time_series_observations'))
            print('Total observations:', result.scalar())
    
    await engine.dispose()

asyncio.run(check())

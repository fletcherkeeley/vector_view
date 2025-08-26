#!/bin/bash
# Daily News Sync Health Check Script

cd "/home/lab/projects/vector-view/ingestion"

echo "📊 Daily News Sync Health Check - $(date)"
echo "================================================"

# Check if scheduler exists
if [ ! -f "news_daily_scheduler.py" ]; then
    echo "❌ Scheduler script not found!"
    exit 1
fi

# Run health check
/usr/bin/python3 news_daily_scheduler.py --health-check

# Check recent logs
echo ""
echo "📋 Recent Cron Logs:"
echo "-------------------"
ls -la cron_logs/daily_sync_*.log 2>/dev/null | tail -5

# Check if sync ran today
TODAY=$(date +%Y%m%d)
if ls cron_logs/daily_sync_${TODAY}_*.log 1> /dev/null 2>&1; then
    echo "✅ Sync ran today"
    LATEST_LOG=$(ls -t cron_logs/daily_sync_${TODAY}_*.log | head -1)
    echo "Latest log: $LATEST_LOG"
    
    # Check for errors in latest log
    if grep -q "failed\|error\|Error\|ERROR" "$LATEST_LOG"; then
        echo "⚠️  Errors found in latest log"
        echo "Last few lines:"
        tail -10 "$LATEST_LOG"
    else
        echo "✅ No errors in latest log"
    fi
else
    echo "⚠️  No sync log found for today"
fi

echo ""
echo "📊 Database Status:"
echo "------------------"
/usr/bin/python3 -c "
import asyncio
import sys
sys.path.insert(0, '/home/lab/projects/vector-view/ingestion')
from ingestion.news.news_database_integration import NewsDatabaseIntegration
import os

async def check_db():
    db = NewsDatabaseIntegration(os.getenv('DATABASE_URL', 'postgresql+psycopg://postgres:fred_password@localhost:5432/postgres'))
    if await db.initialize():
        stats = await db.get_news_statistics()
        print(f'Total articles: {stats.get(\"total_articles\", 0)}')
        print(f'Processed: {stats.get(\"processed_articles\", 0)}')
        print(f'With embeddings: {stats.get(\"articles_with_embeddings\", 0)}')
        await db.close()
    else:
        print('❌ Database connection failed')

asyncio.run(check_db())
"


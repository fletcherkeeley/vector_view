#!/bin/bash
# Embedding Pipeline Health Check Script

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$PROJECT_DIR/venv"
LOG_FILE="$PROJECT_DIR/ingestion/logs/embedding_pipeline_cron.log"

echo "📊 Embedding Pipeline Health Check - $(date)"
echo "=============================================="

# Check if pipeline script exists
if [ ! -f "$PROJECT_DIR/semantic/embedding_pipeline.py" ]; then
    echo "❌ Embedding pipeline script not found!"
    exit 1
fi

# Check recent logs
echo ""
echo "📋 Recent Cron Logs:"
echo "-------------------"
if [ -f "$LOG_FILE" ]; then
    tail -10 "$LOG_FILE"
else
    echo "No log file found at $LOG_FILE"
fi

# Check if pipeline ran today
TODAY=$(date +%Y-%m-%d)
if [ -f "$LOG_FILE" ] && grep -q "$TODAY" "$LOG_FILE"; then
    echo "✅ Pipeline ran today"
    
    # Check for errors in today's logs
    if grep "$TODAY" "$LOG_FILE" | grep -q "failed\|error\|Error\|ERROR"; then
        echo "⚠️  Errors found in today's logs"
    else
        echo "✅ No errors in today's logs"
    fi
else
    echo "⚠️  No pipeline execution found for today"
fi

echo ""
echo "📊 Pipeline Statistics:"
echo "----------------------"
cd "$PROJECT_DIR"
source "$VENV_PATH/bin/activate"
python3 -c "
import asyncio
import sys
sys.path.insert(0, 'semantic')
from embedding_pipeline import create_embedding_pipeline

async def get_stats():
    try:
        pipeline = await create_embedding_pipeline()
        stats = await pipeline.get_pipeline_stats()
        
        print(f'Total articles: {stats[\"postgresql\"][\"total_articles\"]}')
        print(f'Embedded articles: {stats[\"postgresql\"][\"embedded_articles\"]}')
        print(f'Completion: {stats[\"completion_percentage\"]:.1f}%')
        
        if 'vector_store' in stats:
            vs_stats = stats['vector_store']
            print(f'ChromaDB collections: {len(vs_stats)}')
    except Exception as e:
        print(f'❌ Failed to get stats: {e}')

asyncio.run(get_stats())
"

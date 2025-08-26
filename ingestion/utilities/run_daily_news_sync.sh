#!/bin/bash
# Daily News Sync Cron Script
# Generated on Mon Aug 25 09:56:16 AM MDT 2025

# Set environment
export PATH="/usr/local/bin:/usr/bin:/bin"
cd "/home/lab/projects/vector-view"

# Load environment variables
if [ -f "/home/lab/projects/vector-view/.env" ]; then
    source "/home/lab/projects/vector-view/.env"
fi

# Run the daily sync with logging
LOG_FILE="/home/lab/projects/vector-view/ingestion/cron_logs/daily_sync_$(date +%Y%m%d_%H%M%S).log"

echo "Starting daily news sync at $(date)" >> "$LOG_FILE"
echo "Working directory: $(pwd)" >> "$LOG_FILE"
echo "Python path: /usr/bin/python3" >> "$LOG_FILE"

# Run the scheduler with automated flag
AUTOMATED_RUN=true /usr/bin/python3 -m ingestion.news.news_daily_updater --max-calls 200 >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

echo "Daily sync completed at $(date) with exit code $EXIT_CODE" >> "$LOG_FILE"

# Keep only last 30 days of cron logs
find "/home/lab/projects/vector-view/ingestion/cron_logs" -name "daily_sync_*.log" -mtime +30 -delete

# Send notification if sync failed (optional - requires mail setup)
if [ $EXIT_CODE -ne 0 ]; then
    echo "Daily news sync failed with exit code $EXIT_CODE" >> "$LOG_FILE"
    # Uncomment the line below if you have mail configured
    # echo "Daily news sync failed. Check logs at $LOG_FILE" | mail -s "News Sync Failed" admin@yourdomain.com
fi

exit $EXIT_CODE

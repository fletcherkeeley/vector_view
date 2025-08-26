#!/bin/bash
# Clean Cron Setup for Vector View
# Moves all jobs to user crontab with proper paths and removes duplicates

PROJECT_DIR="/home/lab/projects/vector-view"
INGESTION_DIR="$PROJECT_DIR/ingestion"

echo "🔧 Setting up clean cron configuration for Vector View..."

# Remove existing system-wide cron jobs for lab user
echo "Removing existing system-wide cron jobs..."
sudo crontab -r -u lab 2>/dev/null || true

# Create clean user crontab
CLEAN_CRON_JOBS=$(cat << EOF
# Vector View Financial Intelligence Platform - Daily Data Sync
# FRED Economic Data - Run at 6:00 AM daily
0 6 * * * cd $PROJECT_DIR && /usr/bin/python3 -m ingestion.fred.fred_daily_updater >> ingestion/logs/fred_cron.log 2>&1

# Yahoo Finance Market Data - Run at 6:30 AM daily (after FRED)
30 6 * * * cd $PROJECT_DIR && /usr/bin/python3 -m ingestion.yahoo.yahoo_daily_updater >> ingestion/logs/yahoo_cron.log 2>&1

# News Data Sync - Run at 7:00 AM daily (after market data)
0 7 * * * cd $PROJECT_DIR && AUTOMATED_RUN=true /usr/bin/python3 -m ingestion.news.news_daily_updater --max-calls 200 >> ingestion/logs/news_cron.log 2>&1

# Embedding Pipeline - Run at 7:30 AM daily (after news sync)
30 7 * * * cd $PROJECT_DIR && /usr/bin/python3 semantic/embedding_pipeline.py >> ingestion/logs/embedding_cron.log 2>&1

EOF
)

# Install to user crontab
echo "$CLEAN_CRON_JOBS" | crontab -

echo "✅ Clean cron configuration installed to user crontab"
echo ""
echo "📊 Schedule:"
echo "   • 6:00 AM - FRED Economic Data Sync"
echo "   • 6:30 AM - Yahoo Finance Market Data Sync" 
echo "   • 7:00 AM - News Data Sync (with AUTOMATED_RUN flag)"
echo "   • 7:30 AM - Embedding Pipeline"
echo ""
echo "📁 Logs location: $INGESTION_DIR/logs/"
echo ""
echo "🔍 Verify with: crontab -l"

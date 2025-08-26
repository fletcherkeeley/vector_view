#!/bin/bash

# Setup Daily Sync Automation for Vector View Financial Intelligence Platform
# This script configures cron jobs for automatic daily data synchronization

PROJECT_DIR="/home/lab/projects/vector-view"
INGESTION_DIR="$PROJECT_DIR/ingestion"

echo "🔧 Setting up daily sync automation for Vector View..."

# Create logs directory if it doesn't exist
mkdir -p "$INGESTION_DIR/logs"

# Create the cron jobs
CRON_JOBS=$(cat << EOF
# Vector View Financial Intelligence Platform - Daily Data Sync
# FRED Economic Data - Run at 6:00 AM daily (after markets open)
0 6 * * * cd $PROJECT_DIR && /usr/bin/python3 -m ingestion.fred.fred_daily_updater >> ingestion/logs/fred_cron.log 2>&1

# Yahoo Finance Market Data - Run at 6:30 AM daily (after FRED)
30 6 * * * cd $PROJECT_DIR && /usr/bin/python3 -m ingestion.yahoo.yahoo_daily_updater >> ingestion/logs/yahoo_cron.log 2>&1

# News Data Sync - Run at 7:00 AM daily (after market data)
0 7 * * * cd $PROJECT_DIR && /usr/bin/python3 -m ingestion.news.news_daily_updater --database-url "postgresql+psycopg://postgres:fred_password@localhost:5432/postgres" >> ingestion/logs/news_cron.log 2>&1

EOF
)

# Add cron jobs
echo "📅 Adding cron jobs..."
(crontab -l 2>/dev/null; echo "$CRON_JOBS") | crontab -

# Verify cron jobs were added
echo "✅ Cron jobs installed:"
crontab -l | grep -A 10 "Vector View"

echo ""
echo "🎉 Daily sync automation configured successfully!"
echo ""
echo "📊 Schedule:"
echo "   • 6:00 AM - FRED Economic Data Sync"
echo "   • 6:30 AM - Yahoo Finance Market Data Sync" 
echo "   • 7:00 AM - News Data Sync"
echo ""
echo "📁 Logs will be written to: $INGESTION_DIR/logs/"
echo "   • fred_cron.log"
echo "   • yahoo_cron.log"
echo "   • news_cron.log"
echo ""
echo "🔍 To check cron status: crontab -l"
echo "🗑️  To remove automation: crontab -r"

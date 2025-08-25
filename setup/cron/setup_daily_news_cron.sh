#!/bin/bash
"""
Daily News Cron Setup Script

This script sets up the cron job for automated daily news ingestion.
It configures the scheduler to run at 6:00 AM daily with proper logging and monitoring.

Usage:
    chmod +x setup_daily_news_cron.sh
    ./setup_daily_news_cron.sh
"""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the absolute path to the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INGESTION_DIR="$PROJECT_DIR/ingestion"
PYTHON_PATH=$(which python3)

echo -e "${BLUE}🔧 Setting up Daily News Ingestion Cron Job${NC}"
echo "Project Directory: $PROJECT_DIR"
echo "Python Path: $PYTHON_PATH"

# Create logs directory if it doesn't exist
mkdir -p "$INGESTION_DIR/logs"
mkdir -p "$INGESTION_DIR/cron_logs"

# Create the cron script
CRON_SCRIPT="$INGESTION_DIR/run_daily_news_sync.sh"
cat > "$CRON_SCRIPT" << EOF
#!/bin/bash
# Daily News Sync Cron Script
# Generated on $(date)

# Set environment
export PATH="/usr/local/bin:/usr/bin:/bin"
cd "$INGESTION_DIR"

# Load environment variables
if [ -f "$PROJECT_DIR/.env" ]; then
    source "$PROJECT_DIR/.env"
fi

# Run the daily sync with logging
LOG_FILE="$INGESTION_DIR/cron_logs/daily_sync_\$(date +%Y%m%d_%H%M%S).log"

echo "Starting daily news sync at \$(date)" >> "\$LOG_FILE"
echo "Working directory: \$(pwd)" >> "\$LOG_FILE"
echo "Python path: $PYTHON_PATH" >> "\$LOG_FILE"

# Run the scheduler with automated flag
AUTOMATED_RUN=true $PYTHON_PATH news_daily_updater.py --max-calls 200 >> "\$LOG_FILE" 2>&1
EXIT_CODE=\$?

echo "Daily sync completed at \$(date) with exit code \$EXIT_CODE" >> "\$LOG_FILE"

# Keep only last 30 days of cron logs
find "$INGESTION_DIR/cron_logs" -name "daily_sync_*.log" -mtime +30 -delete

# Send notification if sync failed (optional - requires mail setup)
if [ \$EXIT_CODE -ne 0 ]; then
    echo "Daily news sync failed with exit code \$EXIT_CODE" >> "\$LOG_FILE"
    # Uncomment the line below if you have mail configured
    # echo "Daily news sync failed. Check logs at \$LOG_FILE" | mail -s "News Sync Failed" admin@yourdomain.com
fi

exit \$EXIT_CODE
EOF

# Make the cron script executable
chmod +x "$CRON_SCRIPT"

echo -e "${GREEN}✅ Created cron script: $CRON_SCRIPT${NC}"

# Create the cron job entry
CRON_ENTRY="0 6 * * * $CRON_SCRIPT"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "run_daily_news_sync.sh"; then
    echo -e "${YELLOW}⚠️  Cron job already exists. Updating...${NC}"
    # Remove existing entry and add new one
    (crontab -l 2>/dev/null | grep -v "run_daily_news_sync.sh"; echo "$CRON_ENTRY") | crontab -
else
    echo -e "${BLUE}📅 Adding new cron job...${NC}"
    # Add new cron job
    (crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -
fi

echo -e "${GREEN}✅ Cron job configured to run daily at 6:00 AM${NC}"

# Create monitoring script
MONITOR_SCRIPT="$INGESTION_DIR/check_daily_sync_health.sh"
cat > "$MONITOR_SCRIPT" << EOF
#!/bin/bash
# Daily News Sync Health Check Script

cd "$INGESTION_DIR"

echo "📊 Daily News Sync Health Check - \$(date)"
echo "================================================"

# Check if scheduler exists
if [ ! -f "news_daily_scheduler.py" ]; then
    echo "❌ Scheduler script not found!"
    exit 1
fi

# Run health check
$PYTHON_PATH news_daily_scheduler.py --health-check

# Check recent logs
echo ""
echo "📋 Recent Cron Logs:"
echo "-------------------"
ls -la cron_logs/daily_sync_*.log 2>/dev/null | tail -5

# Check if sync ran today
TODAY=\$(date +%Y%m%d)
if ls cron_logs/daily_sync_\${TODAY}_*.log 1> /dev/null 2>&1; then
    echo "✅ Sync ran today"
    LATEST_LOG=\$(ls -t cron_logs/daily_sync_\${TODAY}_*.log | head -1)
    echo "Latest log: \$LATEST_LOG"
    
    # Check for errors in latest log
    if grep -q "failed\|error\|Error\|ERROR" "\$LATEST_LOG"; then
        echo "⚠️  Errors found in latest log"
        echo "Last few lines:"
        tail -10 "\$LATEST_LOG"
    else
        echo "✅ No errors in latest log"
    fi
else
    echo "⚠️  No sync log found for today"
fi

echo ""
echo "📊 Database Status:"
echo "------------------"
$PYTHON_PATH -c "
import asyncio
import sys
sys.path.insert(0, '$INGESTION_DIR')
from news_database_integration import NewsDatabaseIntegration
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

EOF

chmod +x "$MONITOR_SCRIPT"

echo -e "${GREEN}✅ Created monitoring script: $MONITOR_SCRIPT${NC}"

# Create systemd service (optional, for more robust scheduling)
SYSTEMD_SERVICE="$PROJECT_DIR/vector-view-news-sync.service"
cat > "$SYSTEMD_SERVICE" << EOF
[Unit]
Description=Vector View Daily News Sync
After=network.target

[Service]
Type=oneshot
User=$USER
WorkingDirectory=$INGESTION_DIR
Environment=PATH=/usr/local/bin:/usr/bin:/bin
EnvironmentFile=-$PROJECT_DIR/.env
ExecStart=/bin/bash -c "AUTOMATED_RUN=true $PYTHON_PATH news_daily_updater.py --max-calls 200"
StandardOutput=append:$INGESTION_DIR/logs/systemd_news_sync.log
StandardError=append:$INGESTION_DIR/logs/systemd_news_sync.log

[Install]
WantedBy=multi-user.target
EOF

# Create systemd timer
SYSTEMD_TIMER="$PROJECT_DIR/vector-view-news-sync.timer"
cat > "$SYSTEMD_TIMER" << EOF
[Unit]
Description=Run Vector View News Sync Daily
Requires=vector-view-news-sync.service

[Timer]
OnCalendar=daily
Persistent=true
RandomizedDelaySec=300

[Install]
WantedBy=timers.target
EOF

echo -e "${BLUE}📋 Summary:${NC}"
echo "1. Cron job scheduled for 6:00 AM daily"
echo "2. Logs will be saved to: $INGESTION_DIR/cron_logs/"
echo "3. Monitor health with: $MONITOR_SCRIPT"
echo "4. Optional systemd files created (not installed)"
echo ""
echo -e "${YELLOW}📝 Next Steps:${NC}"
echo "1. Test the setup: $CRON_SCRIPT"
echo "2. Check health: $MONITOR_SCRIPT"
echo "3. Monitor logs in: $INGESTION_DIR/cron_logs/"
echo ""
echo -e "${BLUE}🔧 Optional - Install systemd service (more robust):${NC}"
echo "sudo cp $SYSTEMD_SERVICE /etc/systemd/system/"
echo "sudo cp $SYSTEMD_TIMER /etc/systemd/system/"
echo "sudo systemctl enable vector-view-news-sync.timer"
echo "sudo systemctl start vector-view-news-sync.timer"
echo ""
echo -e "${GREEN}✅ Daily news ingestion cron job setup complete!${NC}"

# Show current crontab
echo -e "${BLUE}📅 Current crontab:${NC}"
crontab -l 2>/dev/null | grep -E "(run_daily_news_sync|#.*news)" || echo "No news-related cron jobs found"

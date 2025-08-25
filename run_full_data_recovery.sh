#!/bin/bash
"""
Full Data Recovery Script - Recover All Missed Data from Failed Daily Sync

Runs complete data ingestion for Yahoo Finance, FRED, and News APIs
to recover from today's failed sync operations with proper error handling.
"""

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INGESTION_DIR="$PROJECT_DIR/ingestion"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$INGESTION_DIR/logs/full_recovery_$TIMESTAMP.log"

# Ensure logs directory exists
mkdir -p "$INGESTION_DIR/logs"

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"
}

success() {
    echo -e "${GREEN}✅ $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}❌ $1${NC}" | tee -a "$LOG_FILE"
}

warning() {
    echo -e "${YELLOW}⚠️ $1${NC}" | tee -a "$LOG_FILE"
}

run_ingestion() {
    local name="$1"
    local cmd="$2"
    local description="$3"
    
    log "Starting $description..."
    echo "Command: $cmd" >> "$LOG_FILE"
    
    local start_time=$(date +%s)
    
    if cd "$INGESTION_DIR" && eval "$cmd" >> "$LOG_FILE" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        success "$description completed successfully in ${duration}s"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        error "$description failed after ${duration}s"
        return 1
    fi
}

main() {
    echo -e "${BLUE}🚀 FULL DATA RECOVERY - Vector View${NC}"
    echo "=" * 60
    echo "📅 Started: $(date)"
    echo "📝 Log file: $LOG_FILE"
    echo ""
    
    log "Full data recovery started"
    
    local yahoo_success=false
    local fred_success=false
    local news_success=false
    
    # Yahoo Finance Recovery
    log "Phase 1: Yahoo Finance Data Recovery (53 symbols)"
    if run_ingestion "yahoo" "python yahoo_daily_updater.py" "Yahoo Finance ingestion"; then
        yahoo_success=true
    fi
    
    echo ""
    
    # FRED API Recovery  
    log "Phase 2: FRED API Data Recovery (16 series)"
    if run_ingestion "fred" "python fred_daily_updater.py" "FRED API ingestion"; then
        fred_success=true
    fi
    
    echo ""
    
    # News API Recovery
    log "Phase 3: News API Data Recovery (10 categories)"
    if run_ingestion "news" "python news_daily_updater.py --database-url 'postgresql+psycopg://postgres:fred_password@localhost:5432/postgres'" "News API ingestion"; then
        news_success=true
    fi
    
    echo ""
    echo -e "${BLUE}📊 RECOVERY SUMMARY${NC}"
    echo "=" * 40
    
    if [ "$yahoo_success" = true ]; then
        success "Yahoo Finance: RECOVERED"
    else
        error "Yahoo Finance: FAILED"
    fi
    
    if [ "$fred_success" = true ]; then
        success "FRED API: RECOVERED"
    else
        error "FRED API: FAILED"
    fi
    
    if [ "$news_success" = true ]; then
        success "News API: RECOVERED"
    else
        error "News API: FAILED"
    fi
    
    local total_success=0
    [ "$yahoo_success" = true ] && ((total_success++))
    [ "$fred_success" = true ] && ((total_success++))
    [ "$news_success" = true ] && ((total_success++))
    
    echo ""
    echo "📈 Success Rate: $total_success/3 ($(( total_success * 100 / 3 ))%)"
    echo "📝 Full log: $LOG_FILE"
    echo "📅 Completed: $(date)"
    
    if [ $total_success -eq 3 ]; then
        success "🎉 FULL RECOVERY SUCCESSFUL - All data sources recovered!"
        return 0
    else
        warning "⚠️ PARTIAL RECOVERY - Some data sources failed"
        return 1
    fi
}

# Run main function
main "$@"

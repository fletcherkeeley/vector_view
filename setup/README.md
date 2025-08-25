# Setup Scripts

This directory contains all setup and configuration scripts for the Vector View platform.

## Directory Structure

### `/cron/`
Scripts for setting up automated cron jobs:
- `setup_daily_sync.sh` - Sets up daily data synchronization
- `setup_daily_news_cron.sh` - Configures daily news ingestion
- `setup_embedding_pipeline_cron.sh` - Sets up embedding pipeline automation
- `setup_news_daily_cron.sh` - Alternative news sync setup

### `/pipeline/`
Core pipeline and data processing scripts:
- `run_daily_pipeline.py` - Main daily data processing pipeline
- `run_embedding_pipeline.sh` - Embedding generation pipeline
- `run_full_data_recovery.sh` - Full data recovery and backfill
- `check_embedding_pipeline_health.sh` - Pipeline health monitoring

### `/systemd/`
SystemD service files for system-level integration:
- `vector-view-news-sync.service` - News sync service definition
- `vector-view-news-sync.timer` - Timer for automated news sync

### `/database/`
Database setup and configuration scripts (see `database/` directory in project root)

## Usage

1. **Initial Setup**: Run database setup scripts first
2. **Pipeline Setup**: Configure pipelines using `/pipeline/` scripts
3. **Automation**: Use `/cron/` scripts to set up automated processing
4. **System Integration**: Install `/systemd/` files for system-level services

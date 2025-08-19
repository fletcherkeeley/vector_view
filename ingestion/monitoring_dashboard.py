"""
Unified Data Pipeline Monitoring Dashboard

Comprehensive Streamlit dashboard for monitoring the AI Financial Intelligence Platform.
Provides real-time visibility into data ingestion, database health, and system performance.

Key Features:
- Database statistics and health metrics
- Data ingestion monitoring (FRED + Yahoo Finance)
- Sync operation tracking and error analysis
- Data freshness and quality indicators
- System performance metrics
- Interactive charts and visualizations

Usage:
    streamlit run monitoring_dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
from datetime import datetime, timezone, date, timedelta
import sys
from pathlib import Path
import os

# Add paths for imports FIRST
# Use os.getcwd() to understand where we actually are
current_working_dir = Path(os.getcwd())

# Explicitly construct the path to the database directory
# If we're in /home/lab/projects/vector-view/ingestion, we need to go up one level
if current_working_dir.name == "ingestion":
    project_root = current_working_dir.parent
else:
    # We might be running from project root
    project_root = current_working_dir

database_dir = project_root / "database"

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(database_dir))

# Comment out the stop for now to see the debug info
# if not (database_dir / "unified_database_setup.py").exists():
#     st.error(f"Database setup file not found at: {database_dir / 'unified_database_setup.py'}")
#     st.stop()

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, func, and_, desc, text
import logging

# Import our database models
try:
    from unified_database_setup import (
        DataSeries, MarketAssets, TimeSeriesObservation, DataSyncLog, 
        SystemHealthMetrics, DataSourceType, FrequencyType
    )
except ImportError as e:
    st.error(f"Failed to import database models: {e}")
    st.stop()

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Financial Intelligence Platform - Monitoring",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


class DatabaseMonitor:
    """Handles database connections and monitoring queries"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
        self.AsyncSessionLocal = None
    
    async def initialize(self):
        """Initialize database connection"""
        try:
            self.engine = create_async_engine(
                self.database_url,
                echo=False,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True
            )
            
            self.AsyncSessionLocal = sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            return True
        except Exception as e:
            st.error(f"Database connection failed: {e}")
            return False
    
    async def get_database_overview(self):
        """Get high-level database statistics"""
        try:
            async with self.AsyncSessionLocal() as session:
                # Count series by source
                fred_count = await session.execute(
                    select(func.count(DataSeries.series_id))
                    .where(DataSeries.source_type == DataSourceType.FRED)
                )
                
                yahoo_count = await session.execute(
                    select(func.count(DataSeries.series_id))
                    .where(DataSeries.source_type == DataSourceType.YAHOO_FINANCE)
                )
                
                # Count total observations
                total_obs = await session.execute(
                    select(func.count(TimeSeriesObservation.id))
                )
                
                fred_obs = await session.execute(
                    select(func.count(TimeSeriesObservation.id))
                    .join(DataSeries)
                    .where(DataSeries.source_type == DataSourceType.FRED)
                )
                
                yahoo_obs = await session.execute(
                    select(func.count(TimeSeriesObservation.id))
                    .join(DataSeries)
                    .where(DataSeries.source_type == DataSourceType.YAHOO_FINANCE)
                )
                
                # Database size
                db_size = await session.execute(
                    text("SELECT pg_size_pretty(pg_database_size(current_database())) as size")
                )
                
                return {
                    'fred_series': fred_count.scalar() or 0,
                    'yahoo_series': yahoo_count.scalar() or 0,
                    'total_observations': total_obs.scalar() or 0,
                    'fred_observations': fred_obs.scalar() or 0,
                    'yahoo_observations': yahoo_obs.scalar() or 0,
                    'database_size': db_size.scalar() or 'Unknown'
                }
                
        except Exception as e:
            st.error(f"Error getting database overview: {e}")
            return {}
    
    async def get_recent_sync_activity(self, days: int = 7):
        """Get recent sync operations"""
        try:
            async with self.AsyncSessionLocal() as session:
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
                
                result = await session.execute(
                    select(DataSyncLog)
                    .where(DataSyncLog.sync_start_time >= cutoff_date)
                    .order_by(desc(DataSyncLog.sync_start_time))
                    .limit(100)
                )
                
                sync_logs = result.scalars().all()
                
                # Convert to list of dictionaries for easier handling
                logs_data = []
                for log in sync_logs:
                    logs_data.append({
                        'series_id': log.series_id,
                        'source_type': str(log.source_type).replace('DataSourceType.', ''),
                        'sync_type': log.sync_type,
                        'sync_start_time': log.sync_start_time,
                        'success': log.success,
                        'records_processed': log.records_processed or 0,
                        'records_added': log.records_added or 0,
                        'duration_ms': log.sync_duration_ms or 0,
                        'error_message': log.error_message
                    })
                
                return logs_data
                
        except Exception as e:
            st.error(f"Error getting sync activity: {e}")
            return []
    
    async def get_data_freshness(self):
        """Get data freshness metrics"""
        try:
            async with self.AsyncSessionLocal() as session:
                # Get latest observation for each source type
                fred_latest = await session.execute(
                    select(func.max(TimeSeriesObservation.observation_date))
                    .join(DataSeries)
                    .where(DataSeries.source_type == DataSourceType.FRED)
                )
                
                yahoo_latest = await session.execute(
                    select(func.max(TimeSeriesObservation.observation_date))
                    .join(DataSeries)
                    .where(DataSeries.source_type == DataSourceType.YAHOO_FINANCE)
                )
                
                # Count stale data (older than 7 days)
                cutoff_date = date.today() - timedelta(days=7)
                
                stale_series = await session.execute(
                    select(func.count(DataSeries.series_id))
                    .join(TimeSeriesObservation)
                    .where(TimeSeriesObservation.observation_date < cutoff_date)
                    .group_by(DataSeries.series_id)
                )
                
                return {
                    'fred_latest_date': fred_latest.scalar(),
                    'yahoo_latest_date': yahoo_latest.scalar(),
                    'stale_series_count': len(stale_series.fetchall())
                }
                
        except Exception as e:
            st.error(f"Error getting data freshness: {e}")
            return {}
    
    async def get_sync_performance_metrics(self):
        """Get sync performance over time"""
        try:
            async with self.AsyncSessionLocal() as session:
                # Get daily sync stats for the last 30 days
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=30)
                
                result = await session.execute(
                    select(
                        func.date(DataSyncLog.sync_start_time).label('sync_date'),
                        DataSyncLog.source_type,
                        func.count(DataSyncLog.id).label('total_syncs'),
                        func.sum(func.case((DataSyncLog.success.is_(True), 1), else_=0)).label('successful_syncs'),
                        func.avg(DataSyncLog.sync_duration_ms).label('avg_duration_ms'),
                        func.sum(DataSyncLog.records_added).label('total_records_added')
                    )
                    .where(DataSyncLog.sync_start_time >= cutoff_date)
                    .group_by(
                        func.date(DataSyncLog.sync_start_time),
                        DataSyncLog.source_type
                    )
                    .order_by(func.date(DataSyncLog.sync_start_time))
                )
                
                return [
                    {
                        'asset_type': str(row[0]), 
                        'count': row[1], 
                        'avg_market_cap': row[2]
                    } 
                    for row in result.fetchall()
                ]
                
        except Exception as e:
            st.error(f"Error getting sync performance: {e}")
            return []
    
    async def get_asset_type_breakdown(self):
        """Get breakdown of assets by type"""
        try:
            async with self.AsyncSessionLocal() as session:
                result = await session.execute(
                    select(
                        MarketAssets.asset_type,
                        func.count(MarketAssets.series_id).label('count'),
                        func.avg(MarketAssets.market_cap).label('avg_market_cap')
                    )
                    .group_by(MarketAssets.asset_type)
                )
                
                return [dict(row) for row in result.fetchall()]
                
        except Exception as e:
            st.error(f"Error getting asset breakdown: {e}")
            return []
    
    async def close(self):
        """Close database connections"""
        if self.engine:
            await self.engine.dispose()


# Cache the database monitor
@st.cache_resource
def get_database_monitor():
    database_url = os.getenv('DATABASE_URL', 'postgresql+psycopg://postgres:fred_password@localhost:5432/postgres')
    return DatabaseMonitor(database_url)


async def load_dashboard_data():
    """Load all dashboard data asynchronously"""
    monitor = get_database_monitor()
    
    if not await monitor.initialize():
        return None
    
    try:
        # Load all data concurrently
        overview_task = monitor.get_database_overview()
        sync_activity_task = monitor.get_recent_sync_activity(7)
        freshness_task = monitor.get_data_freshness()
        performance_task = monitor.get_sync_performance_metrics()
        asset_breakdown_task = monitor.get_asset_type_breakdown()
        
        overview = await overview_task
        sync_activity = await sync_activity_task
        freshness = await freshness_task
        performance = await performance_task
        asset_breakdown = await asset_breakdown_task
        
        return {
            'overview': overview,
            'sync_activity': sync_activity,
            'freshness': freshness,
            'performance': performance,
            'asset_breakdown': asset_breakdown
        }
    
    finally:
        await monitor.close()


def render_overview_metrics(data):
    """Render overview metrics cards"""
    if not data or 'overview' not in data:
        st.error("No overview data available")
        return
    
    overview = data['overview']
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Total Data Series",
            value=f"{overview.get('fred_series', 0) + overview.get('yahoo_series', 0):,}",
            delta=f"FRED: {overview.get('fred_series', 0)} | Yahoo: {overview.get('yahoo_series', 0)}"
        )
    
    with col2:
        st.metric(
            label="🗃️ Total Observations",
            value=f"{overview.get('total_observations', 0):,}",
            delta=f"Economic: {overview.get('fred_observations', 0):,} | Market: {overview.get('yahoo_observations', 0):,}"
        )
    
    with col3:
        st.metric(
            label="💾 Database Size",
            value=overview.get('database_size', 'Unknown'),
            delta="PostgreSQL"
        )
    
    with col4:
        # Calculate data freshness
        freshness = data.get('freshness', {})
        fred_latest = freshness.get('fred_latest_date')
        yahoo_latest = freshness.get('yahoo_latest_date')
        
        if fred_latest and yahoo_latest:
            latest_date = max(fred_latest, yahoo_latest)
            days_old = (date.today() - latest_date).days
            freshness_status = "🟢 Fresh" if days_old <= 1 else f"🟡 {days_old} days old"
        else:
            freshness_status = "❓ Unknown"
        
        st.metric(
            label="🔄 Data Freshness",
            value=freshness_status,
            delta=f"Latest: {latest_date}" if 'latest_date' in locals() else "No data"
        )


def render_sync_activity_chart(data):
    """Render sync activity visualization"""
    if not data or 'sync_activity' not in data:
        st.warning("No sync activity data available")
        return
    
    sync_logs = data['sync_activity']
    if not sync_logs:
        st.info("No recent sync activity found")
        return
    
    df = pd.DataFrame(sync_logs)
    df['sync_start_time'] = pd.to_datetime(df['sync_start_time'])
    df['date'] = df['sync_start_time'].dt.date
    
    # Success rate by source type
    success_by_source = df.groupby(['source_type', 'success']).size().unstack(fill_value=0)
    
    fig = px.bar(
        success_by_source.reset_index(),
        x='source_type',
        y=[True, False],
        title="📈 Sync Success Rate by Data Source (Last 7 Days)",
        labels={'value': 'Number of Syncs', 'variable': 'Success Status'},
        color_discrete_map={True: '#2E8B57', False: '#DC143C'}
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_performance_trends(data):
    """Render performance trends over time"""
    if not data or 'performance' not in data:
        st.warning("No performance data available")
        return
    
    performance = data['performance']
    if not performance:
        st.info("No performance metrics found")
        return
    
    df = pd.DataFrame(performance)
    df['sync_date'] = pd.to_datetime(df['sync_date'])
    df['source_type'] = df['source_type'].str.replace('DataSourceType.', '')
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Daily Sync Count', 'Success Rate', 'Avg Duration (ms)', 'Records Added'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Group by source type for different colors
    for source in df['source_type'].unique():
        source_data = df[df['source_type'] == source]
        
        # Daily sync count
        fig.add_trace(
            go.Scatter(x=source_data['sync_date'], y=source_data['total_syncs'], 
                      name=f'{source} - Syncs', line=dict(width=2)),
            row=1, col=1
        )
        
        # Success rate
        success_rate = (source_data['successful_syncs'] / source_data['total_syncs'] * 100).fillna(0)
        fig.add_trace(
            go.Scatter(x=source_data['sync_date'], y=success_rate,
                      name=f'{source} - Success %', line=dict(width=2)),
            row=1, col=2
        )
        
        # Average duration
        fig.add_trace(
            go.Scatter(x=source_data['sync_date'], y=source_data['avg_duration_ms'],
                      name=f'{source} - Duration', line=dict(width=2)),
            row=2, col=1
        )
        
        # Records added
        fig.add_trace(
            go.Scatter(x=source_data['sync_date'], y=source_data['total_records_added'],
                      name=f'{source} - Records', line=dict(width=2)),
            row=2, col=2
        )
    
    fig.update_layout(height=600, title_text="📊 Performance Trends (Last 30 Days)")
    st.plotly_chart(fig, use_container_width=True)


def render_asset_breakdown(data):
    """Render asset type breakdown"""
    if not data or 'asset_breakdown' not in data:
        st.warning("No asset breakdown data available")
        return
    
    breakdown = data['asset_breakdown']
    if not breakdown:
        st.info("No asset data found")
        return
    
    df = pd.DataFrame(breakdown)
    df['asset_type'] = df['asset_type'].str.replace('AssetType.', '')
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Asset count pie chart
        fig_pie = px.pie(
            df, 
            values='count', 
            names='asset_type',
            title="🏢 Assets by Type"
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Average market cap by type
        df_filtered = df[df['avg_market_cap'].notna() & (df['avg_market_cap'] > 0)]
        if not df_filtered.empty:
            fig_bar = px.bar(
                df_filtered,
                x='asset_type',
                y='avg_market_cap',
                title="💰 Average Market Cap by Asset Type",
                labels={'avg_market_cap': 'Avg Market Cap ($)'}
            )
            fig_bar.update_yaxis(tickformat='$.2s')
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No market cap data available")


def render_recent_activity_table(data):
    """Render recent sync activity table"""
    if not data or 'sync_activity' not in data:
        st.warning("No sync activity data available")
        return
    
    sync_logs = data['sync_activity']
    if not sync_logs:
        st.info("No recent sync activity found")
        return
    
    df = pd.DataFrame(sync_logs)
    
    # Format for display
    df['sync_start_time'] = pd.to_datetime(df['sync_start_time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    df['duration_sec'] = (df['duration_ms'] / 1000).round(2)
    df['status'] = df['success'].map({True: '✅ Success', False: '❌ Failed'})
    
    # Select and rename columns for display
    display_df = df[['sync_start_time', 'series_id', 'source_type', 'records_added', 'duration_sec', 'status']].copy()
    display_df.columns = ['Timestamp', 'Series/Asset', 'Source', 'Records Added', 'Duration (sec)', 'Status']
    
    st.dataframe(
        display_df.head(20),
        use_container_width=True,
        hide_index=True
    )


def main():
    """Main dashboard function"""
    # Header
    st.title("📊 Financial Intelligence Platform - Monitoring Dashboard")
    st.markdown("Real-time monitoring of data ingestion, database health, and system performance")
    
    # Sidebar
    st.sidebar.title("🔧 Dashboard Controls")
    
    # Auto-refresh toggle
    auto_refresh = st.sidebar.toggle("🔄 Auto-refresh (30s)", value=False)
    
    if auto_refresh:
        st.sidebar.info("Dashboard will refresh every 30 seconds")
        # Auto-refresh every 30 seconds
        import time
        time.sleep(30)
        st.rerun()
    
    # Manual refresh button
    if st.sidebar.button("🔄 Refresh Now"):
        st.rerun()
    
    # Load data
    with st.spinner("Loading dashboard data..."):
        try:
            data = asyncio.run(load_dashboard_data())
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return
    
    if not data:
        st.error("Failed to load dashboard data")
        return
    
    # Overview metrics
    st.header("📈 System Overview")
    render_overview_metrics(data)
    
    st.divider()
    
    # Sync activity
    st.header("🔄 Sync Activity")
    render_sync_activity_chart(data)
    
    st.divider()
    
    # Performance trends
    st.header("⚡ Performance Trends")
    render_performance_trends(data)
    
    st.divider()
    
    # Asset breakdown
    st.header("🏢 Asset Portfolio")
    render_asset_breakdown(data)
    
    st.divider()
    
    # Recent activity table
    st.header("📋 Recent Sync Operations")
    render_recent_activity_table(data)
    
    # Footer
    st.markdown("---")
    st.markdown("*Last updated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "*")


if __name__ == "__main__":
    main()
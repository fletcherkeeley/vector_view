#!/usr/bin/env python3
"""
News Monitoring Dashboard - Real-time monitoring for daily news ingestion

This module provides a comprehensive monitoring dashboard for the daily news
ingestion pipeline with real-time metrics and health checks.

Key Features:
- Real-time sync status monitoring
- API usage tracking and alerts
- Article collection metrics
- Error tracking and notifications
- Historical performance analytics
- Integration with existing monitoring dashboard

Usage:
    python news_monitoring_dashboard.py                    # Start monitoring server
    python news_monitoring_dashboard.py --port 8502        # Custom port
    python news_monitoring_dashboard.py --check-only       # Health check only
"""

import streamlit as st
import asyncio
import logging
import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, date, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from .news_database_integration import NewsDatabaseIntegration

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsMonitoringDashboard:
    """
    Monitoring dashboard for daily news ingestion pipeline.
    """
    
    def __init__(self, database_url: str):
        """
        Initialize the monitoring dashboard.
        
        Args:
            database_url: PostgreSQL connection string
        """
        self.database_url = database_url
        self.db_integration = NewsDatabaseIntegration(database_url)
        self.scheduler = NewsDailyScheduler(database_url)
        
        # File paths
        self.ingestion_dir = Path(__file__).parent.parent
        self.stats_file = self.ingestion_dir / 'config' / 'daily_sync_stats.json'
        self.progress_file = self.ingestion_dir / 'config' / 'daily_sync_progress.json'
        
    async def initialize(self) -> bool:
        """Initialize database connections"""
        db_success = await self.db_integration.initialize()
        scheduler_success = await self.scheduler.initialize()
        return db_success and scheduler_success
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        try:
            # Get database statistics
            news_stats = await self.db_integration.get_news_statistics()
            
            # Get sync health check
            health_check = await self.scheduler.get_sync_health_check()
            
            # Load daily sync statistics
            daily_stats = self._load_daily_stats()
            
            # Get recent articles
            recent_articles = await self.db_integration.get_recent_articles(limit=20)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(daily_stats)
            
            return {
                'news_stats': news_stats,
                'health_check': health_check,
                'daily_stats': daily_stats,
                'recent_articles': recent_articles,
                'performance_metrics': performance_metrics,
                'last_updated': datetime.now(timezone.utc)
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {'error': str(e)}
    
    def _load_daily_stats(self) -> List[Dict[str, Any]]:
        """Load daily sync statistics from file"""
        try:
            if self.stats_file.exists():
                with open(self.stats_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Error loading daily stats: {e}")
            return []
    
    def _calculate_performance_metrics(self, daily_stats: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate performance metrics from daily stats"""
        if not daily_stats:
            return {}
        
        # Last 7 days metrics
        recent_stats = daily_stats[-7:] if len(daily_stats) >= 7 else daily_stats
        
        total_articles = sum(s.get('total_articles_stored', 0) for s in recent_stats)
        total_api_calls = sum(s.get('total_api_calls_used', 0) for s in recent_stats)
        avg_success_rate = sum(s.get('success_rate', 0) for s in recent_stats) / len(recent_stats)
        avg_efficiency = sum(s.get('efficiency', 0) for s in recent_stats) / len(recent_stats)
        
        return {
            'avg_daily_articles': total_articles / len(recent_stats),
            'avg_daily_api_calls': total_api_calls / len(recent_stats),
            'avg_success_rate': avg_success_rate,
            'avg_efficiency': avg_efficiency,
            'total_articles_7_days': total_articles,
            'days_analyzed': len(recent_stats)
        }
    
    async def close(self):
        """Close database connections"""
        await self.db_integration.close()
        await self.scheduler.close()


def create_streamlit_dashboard():
    """Create the Streamlit dashboard interface"""
    
    st.set_page_config(
        page_title="Vector View - News Monitoring",
        page_icon="📰",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📰 News Ingestion Monitoring Dashboard")
    st.markdown("Real-time monitoring for daily news ingestion pipeline")
    
    # Initialize dashboard
    database_url = os.getenv('DATABASE_URL', 'postgresql+psycopg://postgres:fred_password@localhost:5432/postgres')
    
    @st.cache_resource
    def get_dashboard():
        return NewsMonitoringDashboard(database_url)
    
    dashboard = get_dashboard()
    
    # Sidebar controls
    st.sidebar.header("🔧 Controls")
    
    auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=True)
    if auto_refresh:
        st.rerun()
    
    manual_refresh = st.sidebar.button("🔄 Refresh Now")
    if manual_refresh:
        st.cache_resource.clear()
        st.rerun()
    
    # Health check button
    if st.sidebar.button("🏥 Run Health Check"):
        with st.spinner("Running health check..."):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                success = loop.run_until_complete(dashboard.initialize())
                if success:
                    health = loop.run_until_complete(dashboard.get_dashboard_data())
                    st.sidebar.success("Health check completed!")
                    st.sidebar.json(health.get('health_check', {}))
                else:
                    st.sidebar.error("Failed to connect to database")
                    
                loop.run_until_complete(dashboard.close())
                loop.close()
                
            except Exception as e:
                st.sidebar.error(f"Health check failed: {e}")
    
    # Main dashboard content
    try:
        # Get dashboard data
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        success = loop.run_until_complete(dashboard.initialize())
        if not success:
            st.error("❌ Failed to connect to database")
            return
        
        data = loop.run_until_complete(dashboard.get_dashboard_data())
        loop.run_until_complete(dashboard.close())
        loop.close()
        
        if 'error' in data:
            st.error(f"❌ Error loading data: {data['error']}")
            return
        
        # Status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            health_status = data['health_check'].get('status', 'unknown')
            status_color = "🟢" if health_status == 'healthy' else "🔴"
            st.metric("System Status", f"{status_color} {health_status.title()}")
        
        with col2:
            total_articles = data['news_stats'].get('total_articles', 0)
            st.metric("Total Articles", f"{total_articles:,}")
        
        with col3:
            processed_articles = data['news_stats'].get('processed_articles', 0)
            processing_rate = (processed_articles / total_articles * 100) if total_articles > 0 else 0
            st.metric("Processing Rate", f"{processing_rate:.1f}%")
        
        with col4:
            last_sync = data['health_check'].get('last_sync', 'Never')
            st.metric("Last Sync", last_sync)
        
        # Performance metrics
        if data['performance_metrics']:
            st.subheader("📊 Performance Metrics (Last 7 Days)")
            
            perf = data['performance_metrics']
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Avg Daily Articles", f"{perf['avg_daily_articles']:.0f}")
            
            with col2:
                st.metric("Avg API Calls/Day", f"{perf['avg_daily_api_calls']:.0f}")
            
            with col3:
                st.metric("Avg Success Rate", f"{perf['avg_success_rate']:.1f}%")
            
            with col4:
                st.metric("API Efficiency", f"{perf['avg_efficiency']:.1f} articles/call")
        
        # Daily sync trends
        if data['daily_stats']:
            st.subheader("📈 Daily Sync Trends")
            
            df_stats = pd.DataFrame(data['daily_stats'])
            df_stats['sync_date'] = pd.to_datetime(df_stats['sync_date'])
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Articles Collected', 'API Calls Used', 'Success Rate', 'Efficiency'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Articles collected
            fig.add_trace(
                go.Scatter(x=df_stats['sync_date'], y=df_stats['total_articles_stored'],
                          mode='lines+markers', name='Articles'),
                row=1, col=1
            )
            
            # API calls used
            fig.add_trace(
                go.Scatter(x=df_stats['sync_date'], y=df_stats['total_api_calls_used'],
                          mode='lines+markers', name='API Calls', line=dict(color='orange')),
                row=1, col=2
            )
            
            # Success rate
            fig.add_trace(
                go.Scatter(x=df_stats['sync_date'], y=df_stats['success_rate'],
                          mode='lines+markers', name='Success Rate', line=dict(color='green')),
                row=2, col=1
            )
            
            # Efficiency
            fig.add_trace(
                go.Scatter(x=df_stats['sync_date'], y=df_stats['efficiency'],
                          mode='lines+markers', name='Efficiency', line=dict(color='purple')),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent articles
        st.subheader("📰 Recent Articles")
        
        if data['recent_articles']:
            articles_df = pd.DataFrame(data['recent_articles'])
            
            # Display as table with key metrics
            display_df = articles_df[['title', 'source_name', 'published_at', 'relevance_score', 'data_quality_score', 'is_processed']].copy()
            display_df['published_at'] = pd.to_datetime(display_df['published_at']).dt.strftime('%Y-%m-%d %H:%M')
            display_df['relevance_score'] = display_df['relevance_score'].round(2)
            display_df['data_quality_score'] = display_df['data_quality_score'].round(2)
            
            st.dataframe(
                display_df,
                column_config={
                    "title": st.column_config.TextColumn("Title", width="large"),
                    "source_name": st.column_config.TextColumn("Source", width="medium"),
                    "published_at": st.column_config.TextColumn("Published", width="small"),
                    "relevance_score": st.column_config.NumberColumn("Relevance", width="small"),
                    "data_quality_score": st.column_config.NumberColumn("Quality", width="small"),
                    "is_processed": st.column_config.CheckboxColumn("Processed", width="small")
                },
                use_container_width=True,
                height=400
            )
        else:
            st.info("No recent articles found")
        
        # Category breakdown
        if data['daily_stats']:
            st.subheader("📊 Category Performance")
            
            # Get latest day's category results
            latest_stats = data['daily_stats'][-1] if data['daily_stats'] else {}
            category_results = latest_stats.get('category_results', [])
            
            if category_results:
                cat_df = pd.DataFrame(category_results)
                
                # Create category performance chart
                fig = px.bar(
                    cat_df, 
                    x='category', 
                    y='articles_stored',
                    color='success',
                    title='Articles Stored by Category (Latest Sync)',
                    color_discrete_map={True: 'green', False: 'red'}
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        # System information
        with st.expander("🔧 System Information"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Database Stats")
                db_stats = data['news_stats']
                st.json({
                    'total_articles': db_stats.get('total_articles', 0),
                    'processed_articles': db_stats.get('processed_articles', 0),
                    'articles_with_embeddings': db_stats.get('articles_with_embeddings', 0),
                    'average_quality_score': db_stats.get('average_quality_score', 0),
                    'date_range_start': str(db_stats.get('date_range_start', '')),
                    'date_range_end': str(db_stats.get('date_range_end', ''))
                })
            
            with col2:
                st.subheader("Health Check")
                st.json(data['health_check'])
        
        # Footer
        st.markdown("---")
        st.markdown(f"*Last updated: {data['last_updated'].strftime('%Y-%m-%d %H:%M:%S UTC')}*")
        
        # Auto-refresh
        if auto_refresh:
            time.sleep(30)
            st.rerun()
    
    except Exception as e:
        st.error(f"❌ Dashboard error: {e}")
        logger.error(f"Dashboard error: {e}")


async def run_health_check_only():
    """Run health check only (for command line usage)"""
    database_url = os.getenv('DATABASE_URL', 'postgresql+psycopg://postgres:fred_password@localhost:5432/postgres')
    
    dashboard = NewsMonitoringDashboard(database_url)
    
    try:
        print("🏥 Running News Ingestion Health Check...")
        
        if not await dashboard.initialize():
            print("❌ Failed to initialize database connection")
            return False
        
        data = await dashboard.get_dashboard_data()
        
        if 'error' in data:
            print(f"❌ Error: {data['error']}")
            return False
        
        print("\n📊 Health Check Results:")
        print("=" * 50)
        
        # System status
        health = data['health_check']
        print(f"Status: {health.get('status', 'unknown').upper()}")
        print(f"Last Sync: {health.get('last_sync', 'Never')}")
        print(f"Database Health: {health.get('database_health', 'unknown')}")
        
        # Database stats
        db_stats = data['news_stats']
        print(f"\nDatabase Statistics:")
        print(f"  Total Articles: {db_stats.get('total_articles', 0):,}")
        print(f"  Processed: {db_stats.get('processed_articles', 0):,}")
        print(f"  With Embeddings: {db_stats.get('articles_with_embeddings', 0):,}")
        print(f"  Avg Quality: {db_stats.get('average_quality_score', 0):.2f}")
        
        # Performance metrics
        if data['performance_metrics']:
            perf = data['performance_metrics']
            print(f"\nPerformance (Last {perf['days_analyzed']} days):")
            print(f"  Avg Daily Articles: {perf['avg_daily_articles']:.0f}")
            print(f"  Avg API Calls/Day: {perf['avg_daily_api_calls']:.0f}")
            print(f"  Avg Success Rate: {perf['avg_success_rate']:.1f}%")
            print(f"  API Efficiency: {perf['avg_efficiency']:.1f} articles/call")
        
        print("\n✅ Health check completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    finally:
        await dashboard.close()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='News Monitoring Dashboard')
    parser.add_argument('--port', type=int, default=8502, help='Streamlit port (default: 8502)')
    parser.add_argument('--check-only', action='store_true', help='Run health check only')
    
    args = parser.parse_args()
    
    if args.check_only:
        # Run health check only
        success = asyncio.run(run_health_check_only())
        sys.exit(0 if success else 1)
    else:
        # Start Streamlit dashboard
        create_streamlit_dashboard()


if __name__ == "__main__":
    main()

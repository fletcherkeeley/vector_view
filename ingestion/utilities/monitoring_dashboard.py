"""
Vector View Monitoring Dashboard

Streamlined monitoring for PostgreSQL and ChromaDB databases.
Shows database sizes, table metrics, and sync operation logs.
"""

import streamlit as st
import pandas as pd
import asyncio
from datetime import datetime, timezone, timedelta
import sys
from pathlib import Path
import os
import chromadb
from chromadb.config import Settings

# Setup paths - always resolve relative to this file's location
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent  # utilities -> ingestion -> vector-view
database_dir = project_root / "database"

# Add paths to sys.path
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(database_dir))
sys.path.insert(0, str(script_dir))

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, func, desc, text

# Import database models
try:
    from unified_database_setup import (
        DataSeries, MarketAssets, TimeSeriesObservation, DataSyncLog, 
        NewsArticles, DataSourceType
    )
except ImportError as e:
    try:
        # Fallback: try direct import from database directory
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "database"))
        from unified_database_setup import (
            DataSeries, MarketAssets, TimeSeriesObservation, DataSyncLog, 
            NewsArticles, DataSourceType
        )
    except ImportError as e2:
        st.error(f"Failed to import database models: {e2}")
        st.error(f"Script dir: {script_dir}")
        st.error(f"Project root: {project_root}")
        st.error(f"Database dir: {database_dir}")
        st.error(f"Database dir exists: {database_dir.exists()}")
        if database_dir.exists():
            st.error(f"Files in database dir: {list(database_dir.glob('*.py'))}")
        st.stop()

load_dotenv()

st.set_page_config(
    page_title="Vector View - Database Monitor",
    page_icon="🗄️",
    layout="wide"
)


class DatabaseMonitor:
    """Handles database connections and monitoring queries"""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
        self.AsyncSessionLocal = None
    
    async def initialize(self):
        """Initialize database connection with connection pooling"""
        try:
            self.engine = create_async_engine(
                self.database_url, 
                echo=False,
                pool_size=5,
                max_overflow=10,
                pool_pre_ping=True,
                pool_recycle=3600,
                pool_timeout=30
            )
            self.AsyncSessionLocal = sessionmaker(
                bind=self.engine, class_=AsyncSession, expire_on_commit=False
            )
            return True
        except Exception as e:
            st.error(f"Database connection failed: {e}")
            return False
    
    async def get_database_metrics(self):
        """Get PostgreSQL database size and table metrics"""
        try:
            async with self.AsyncSessionLocal() as session:
                # Database size
                db_size_result = await session.execute(
                    text("SELECT pg_size_pretty(pg_database_size(current_database())) as size")
                )
                db_size = db_size_result.scalar()
                
                # News sync metrics for today
                news_sync_today = await session.execute(text("""
                    WITH category_extract AS (
                        SELECT 
                            COUNT(*) as articles_today,
                            COUNT(CASE WHEN has_embeddings THEN 1 END) as embedded_today,
                            AVG(relevance_score) as avg_relevance,
                            AVG(data_quality_score) as avg_quality
                        FROM news_articles 
                        WHERE DATE(created_at) = CURRENT_DATE
                    ),
                    unique_categories AS (
                        SELECT DISTINCT jsonb_array_elements_text(economic_categories) as category
                        FROM news_articles 
                        WHERE DATE(created_at) = CURRENT_DATE
                        AND economic_categories IS NOT NULL
                    )
                    SELECT 
                        ce.*,
                        STRING_AGG(uc.category, ', ' ORDER BY uc.category) as categories_processed
                    FROM category_extract ce
                    CROSS JOIN unique_categories uc
                    GROUP BY ce.articles_today, ce.embedded_today, ce.avg_relevance, ce.avg_quality
                """))
                
                news_today = news_sync_today.fetchone()
                
                # API usage estimation (approximate based on articles/categories)
                api_calls_estimate = await session.execute(text("""
                    SELECT 
                        COUNT(DISTINCT economic_categories::text) * 25 as estimated_api_calls
                    FROM news_articles 
                    WHERE DATE(created_at) = CURRENT_DATE
                    AND economic_categories IS NOT NULL
                """))
                
                api_estimate = api_calls_estimate.scalar() or 0
                
                # Table sizes and row counts
                table_metrics = await session.execute(text("""
                    SELECT 
                        schemaname,
                        relname as tablename,
                        pg_size_pretty(pg_total_relation_size(schemaname||'.'||relname)) as size,
                        n_tup_ins as total_rows,
                        last_vacuum,
                        last_analyze
                    FROM pg_stat_user_tables 
                    ORDER BY pg_total_relation_size(schemaname||'.'||relname) DESC
                """))
                
                tables = []
                for row in table_metrics:
                    tables.append({
                        'table': f"{row[0]}.{row[1]}",
                        'size': row[2],
                        'rows': row[3] or 0,
                        'last_vacuum': row[4],
                        'last_analyze': row[5]
                    })
                
                # Yahoo Finance stocks breakdown
                yahoo_stocks = await session.execute(text("""
                    SELECT 
                        ds.series_id,
                        ds.title,
                        COUNT(tso.id) as observation_count,
                        MIN(tso.observation_date) as earliest_date,
                        MAX(tso.observation_date) as latest_date
                    FROM data_series ds
                    LEFT JOIN time_series_observations tso ON ds.series_id = tso.series_id
                    WHERE ds.source_type = 'YAHOO_FINANCE'
                    GROUP BY ds.series_id, ds.title
                    ORDER BY observation_count DESC
                    LIMIT 10
                """))
                
                # FRED indicators breakdown
                fred_indicators = await session.execute(text("""
                    SELECT 
                        ds.series_id,
                        ds.title,
                        COUNT(tso.id) as observation_count,
                        MIN(tso.observation_date) as earliest_date,
                        MAX(tso.observation_date) as latest_date
                    FROM data_series ds
                    LEFT JOIN time_series_observations tso ON ds.series_id = tso.series_id
                    WHERE ds.source_type = 'FRED'
                    GROUP BY ds.series_id, ds.title
                    ORDER BY observation_count DESC
                """))
                
                # News categories breakdown
                news_categories = await session.execute(text("""
                    SELECT 
                        economic_categories,
                        COUNT(*) as article_count,
                        SUM(CASE WHEN has_embeddings THEN 1 ELSE 0 END) as embedded_count,
                        MIN(published_at) as earliest_date,
                        MAX(published_at) as latest_date
                    FROM news_articles
                    WHERE economic_categories IS NOT NULL
                    GROUP BY economic_categories
                    ORDER BY article_count DESC
                """))
                
                # Overall stats
                news_total = await session.execute(select(func.count(NewsArticles.id)))
                news_embedded = await session.execute(
                    select(func.count(NewsArticles.id)).where(NewsArticles.has_embeddings == True)
                )
                latest_observation = await session.execute(
                    select(func.max(TimeSeriesObservation.observation_date))
                )
                latest_news = await session.execute(
                    select(func.max(NewsArticles.published_at))
                )
                latest_embedding_run = await session.execute(
                    select(func.max(NewsArticles.updated_at))
                    .where(NewsArticles.has_embeddings == True)
                )
                
                return {
                    'database_size': db_size,
                    'tables': tables,
                    'yahoo_stocks': [dict(row._mapping) for row in yahoo_stocks],
                    'fred_indicators': [dict(row._mapping) for row in fred_indicators],
                    'news_categories': [dict(row._mapping) for row in news_categories],
                    'news_total': news_total.scalar() or 0,
                    'news_embedded': news_embedded.scalar() or 0,
                    'latest_observation': latest_observation.scalar(),
                    'latest_news': latest_news.scalar(),
                    'latest_embedding_run': latest_embedding_run.scalar(),
                    'news_sync_today': {
                        'articles_pulled': news_today[0] if news_today else 0,
                        'articles_embedded': news_today[1] if news_today else 0,
                        'avg_relevance': round(news_today[2] or 0, 3),
                        'avg_quality': round(news_today[3] or 0, 3),
                        'categories_processed': news_today[4] if news_today else 'None',
                        'estimated_api_calls': api_estimate
                    }
                }
                
        except Exception as e:
            st.error(f"Error getting database metrics: {e}")
            return {}
    
    async def get_sync_logs(self, days: int = 2):
        """Get recent sync operations"""
        try:
            async with self.AsyncSessionLocal() as session:
                cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
                
                result = await session.execute(
                    select(DataSyncLog)
                    .where(DataSyncLog.sync_start_time >= cutoff_date)
                    .order_by(desc(DataSyncLog.sync_start_time))
                    .limit(1000)
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
            st.error(f"Error getting sync logs: {e}")
            return []


class ChromaDBMonitor:
    """Handles ChromaDB monitoring"""
    
    def __init__(self, chroma_path: str):
        self.chroma_path = chroma_path
        self.client = None
    
    def initialize(self):
        """Initialize ChromaDB connection"""
        try:
            self.client = chromadb.PersistentClient(
                path=self.chroma_path,
                settings=Settings(anonymized_telemetry=False)
            )
            return True
        except Exception as e:
            st.error(f"ChromaDB connection failed: {e}")
            return False
    
    def get_chroma_metrics(self):
        """Get ChromaDB collection metrics"""
        try:
            collections = self.client.list_collections()
            metrics = []
            
            for collection in collections:
                count = collection.count()
                # Get sample metadata to check last update
                try:
                    sample = collection.peek(limit=1)
                    last_update = None
                    if sample['metadatas'] and sample['metadatas'][0]:
                        last_update = sample['metadatas'][0].get('created_at')
                except:
                    last_update = None
                
                metrics.append({
                    'collection': collection.name,
                    'documents': count,
                    'last_update': last_update
                })
            
            return metrics
        except Exception as e:
            st.error(f"Error getting ChromaDB metrics: {e}")
            return []


# Main dashboard functions
@st.cache_resource
def get_database_monitor():
    """Get cached database monitor"""
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        st.error("DATABASE_URL not found in environment variables")
        return None
    return DatabaseMonitor(database_url)

@st.cache_resource
def get_chroma_monitor():
    """Get cached ChromaDB monitor"""
    chroma_path = os.path.join(project_root, "chroma_db")
    return ChromaDBMonitor(chroma_path)

async def main():
    """Main dashboard function"""
    st.title("🗄️ Vector View Database Monitor")
    st.markdown("---")
    
    # Initialize monitors
    db_monitor = get_database_monitor()
    chroma_monitor = get_chroma_monitor()
    
    if not db_monitor or not await db_monitor.initialize():
        st.error("Failed to connect to PostgreSQL")
        return
    
    if not chroma_monitor.initialize():
        st.error("Failed to connect to ChromaDB")
        return
    
    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    # Get database metrics
    db_metrics = await db_monitor.get_database_metrics()
    chroma_metrics = chroma_monitor.get_chroma_metrics()
    
    with col1:
        st.metric("PostgreSQL Size", db_metrics.get('database_size', 'Unknown'))
    
    with col2:
        st.metric("News Articles", f"{db_metrics.get('news_total', 0):,}")
    
    with col3:
        st.metric("Articles Embedded", f"{db_metrics.get('news_embedded', 0):,}")
    
    with col4:
        total_vectors = sum(m['documents'] for m in chroma_metrics)
        st.metric("Vector Documents", f"{total_vectors:,}")
    
    st.markdown("---")
    
    # Today's News Sync Status
    st.subheader("📰 Today's News Sync Status")
    news_today = db_metrics.get('news_sync_today', {})
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        articles_pulled = news_today.get('articles_pulled', 0)
        st.metric("Articles Pulled", f"{articles_pulled:,}")
    
    with col2:
        api_calls = news_today.get('estimated_api_calls', 0)
        st.metric("Est. API Calls", f"{api_calls}")
    
    with col3:
        articles_embedded = news_today.get('articles_embedded', 0)
        st.metric("Articles Embedded", f"{articles_embedded:,}")
    
    with col4:
        avg_relevance = news_today.get('avg_relevance', 0)
        st.metric("Avg Relevance", f"{avg_relevance:.3f}")
    
    with col5:
        avg_quality = news_today.get('avg_quality', 0)
        st.metric("Avg Quality", f"{avg_quality:.3f}")
    
    # Categories processed today
    categories = news_today.get('categories_processed', 'None')
    if categories and categories != 'None':
        st.info(f"📊 Categories processed today: {categories}")
    else:
        st.warning("⚠️ No news categories processed today")
    
    st.markdown("---")
    
    # Data breakdown sections
    st.subheader("📈 Yahoo Finance Stocks")
    if db_metrics.get('yahoo_stocks'):
        yahoo_df = pd.DataFrame(db_metrics['yahoo_stocks'])
        yahoo_df['date_range'] = yahoo_df['earliest_date'].astype(str) + ' to ' + yahoo_df['latest_date'].astype(str)
        display_yahoo = yahoo_df[['series_id', 'title', 'observation_count', 'date_range']].copy()
        display_yahoo.columns = ['Symbol', 'Company', 'Observations', 'Date Range']
        st.dataframe(display_yahoo, use_container_width=True)
    else:
        st.info("No Yahoo Finance data available")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏛️ FRED Economic Indicators")
        if db_metrics.get('fred_indicators'):
            fred_df = pd.DataFrame(db_metrics['fred_indicators'])
            fred_df['date_range'] = fred_df['earliest_date'].astype(str) + ' to ' + fred_df['latest_date'].astype(str)
            display_fred = fred_df[['series_id', 'observation_count', 'date_range']].copy()
            display_fred.columns = ['Indicator', 'Observations', 'Date Range']
            st.dataframe(display_fred, use_container_width=True)
        else:
            st.info("No FRED data available")
    
    with col2:
        st.subheader("📰 News Categories")
        if db_metrics.get('news_categories'):
            news_df = pd.DataFrame(db_metrics['news_categories'])
            news_df['embedding_rate'] = (news_df['embedded_count'] / news_df['article_count'] * 100).round(1)
            display_news = news_df[['economic_categories', 'article_count', 'embedded_count', 'embedding_rate']].copy()
            display_news.columns = ['Category', 'Articles', 'Embedded', 'Embed %']
            st.dataframe(display_news, use_container_width=True)
        else:
            st.info("No news category data available")
    
    st.markdown("---")
    
    # ChromaDB Collections
    st.subheader("🧠 ChromaDB Collections")
    if chroma_metrics:
        chroma_df = pd.DataFrame(chroma_metrics)
        st.dataframe(chroma_df, use_container_width=True)
    else:
        st.info("No ChromaDB collections found")
    
    st.markdown("---")
    
    # Data recency and service status section
    st.subheader("📅 Data Recency & Service Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        latest_obs = db_metrics.get('latest_observation')
        if latest_obs:
            days_old = (datetime.now().date() - latest_obs).days
            st.metric("Latest Economic Data", f"{latest_obs}", f"{days_old} days ago")
        else:
            st.metric("Latest Economic Data", "No data")
    
    with col2:
        latest_news = db_metrics.get('latest_news')
        if latest_news:
            if hasattr(latest_news, 'date'):
                news_date = latest_news.date()
            else:
                news_date = latest_news
            days_old = (datetime.now().date() - news_date).days
            st.metric("Latest News", f"{news_date}", f"{days_old} days ago")
        else:
            st.metric("Latest News", "No data")
    
    with col3:
        latest_embedding = db_metrics.get('latest_embedding_run')
        if latest_embedding:
            if hasattr(latest_embedding, 'date'):
                embed_date = latest_embedding.date()
            else:
                embed_date = latest_embedding
            days_old = (datetime.now().date() - embed_date).days
            st.metric("Latest Embedding Run", f"{embed_date}", f"{days_old} days ago")
        else:
            st.metric("Latest Embedding Run", "No data")
    
    st.markdown("---")
    
    # Sync logs section
    st.subheader("📋 Recent Sync Operations")
    
    # Time filter
    days_filter = st.selectbox("Show logs from last:", [1, 2, 7, 14], index=1)
    
    sync_logs = await db_monitor.get_sync_logs(days=days_filter)
    
    if sync_logs:
        logs_df = pd.DataFrame(sync_logs)
        logs_df['sync_start_time'] = pd.to_datetime(logs_df['sync_start_time'], utc=True)
        logs_df = logs_df.sort_values('sync_start_time', ascending=False)
        
        # Format for display with timezone conversion
        display_df = logs_df.copy()
        # Convert UTC to local timezone (MDT/MST)
        display_df['sync_start_time_local'] = display_df['sync_start_time'].dt.tz_convert('America/Denver')
        display_df['Time'] = display_df['sync_start_time_local'].dt.strftime('%m-%d %H:%M')
        display_df['Source'] = display_df['source_type']
        display_df['Series'] = display_df['series_id']
        display_df['Success'] = display_df['success'].map({True: '✅', False: '❌'})
        display_df['Records'] = display_df['records_added']
        display_df['Duration (ms)'] = display_df['duration_ms']
        display_df['Error'] = display_df['error_message'].fillna('')
        
        # Show only relevant columns
        st.dataframe(
            display_df[['Time', 'Source', 'Series', 'Success', 'Records', 'Duration (ms)', 'Error']],
            use_container_width=True,
            height=400
        )
        
        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            success_rate = (logs_df['success'].sum() / len(logs_df)) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        with col2:
            total_records = logs_df['records_added'].sum()
            st.metric("Total Records Added", f"{total_records:,}")
        with col3:
            avg_duration = logs_df['duration_ms'].mean()
            st.metric("Avg Duration", f"{avg_duration:.0f}ms")
    else:
        st.info(f"No sync operations found in the last {days_filter} days")

if __name__ == "__main__":
    asyncio.run(main())
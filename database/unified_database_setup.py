"""
Unified Economic Data Platform - Database Schema Setup

Enterprise-grade database architecture for multi-source economic data integration:
- FRED Economic Indicators (50+ years historical data)
- Yahoo Finance Market Data (stocks, ETFs, indices)
- News API Integration (with vector database bridging)

Features:
- Unified time-series architecture across all data sources
- Pre-calculated correlation storage for performance
- Graceful error handling and comprehensive logging
- Vector database integration points
- Optimized indexes for time-series queries
- Data integrity constraints and validation
"""

import asyncio
import logging
import sys
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

# Database imports
from sqlalchemy import (
    Column, String, Integer, Numeric, Date, DateTime, Text, Boolean, 
    ForeignKey, Index, CheckConstraint, UniqueConstraint, Enum,
    DECIMAL, VARCHAR, event, DDL
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
import enum
import uuid

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('database_setup.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Database base
Base = declarative_base()

# Enums for data validation
class DataSourceType(enum.Enum):
    """Enumeration of supported data sources"""
    FRED = "fred"
    YAHOO_FINANCE = "yahoo_finance"
    NEWS_API = "news_api"
    CALCULATED = "calculated"  # For derived metrics

class FrequencyType(enum.Enum):
    """Data frequency types with standardized values"""
    REAL_TIME = "real_time"    # Streaming/tick data
    MINUTELY = "minutely"      # Intraday minute data
    HOURLY = "hourly"          # Hourly data
    DAILY = "daily"            # Daily data (most common)
    WEEKLY = "weekly"          # Weekly aggregations
    MONTHLY = "monthly"        # Monthly economic data
    QUARTERLY = "quarterly"    # Quarterly reports (GDP, earnings)
    ANNUALLY = "annually"      # Annual data
    IRREGULAR = "irregular"    # Irregular release schedule

class AssetType(enum.Enum):
    """Asset classification for market data"""
    INDEX = "index"           # Market indices (SPY, QQQ)
    STOCK = "stock"           # Individual stocks (AAPL, MSFT)
    ETF = "etf"              # Exchange-traded funds
    SECTOR_ETF = "sector_etf" # Sector-specific ETFs
    COMMODITY = "commodity"   # Commodities (oil, gold)
    CURRENCY = "currency"     # Currency pairs
    BOND = "bond"            # Fixed income instruments

class CorrelationType(enum.Enum):
    """Types of correlation calculations"""
    PEARSON = "pearson"       # Standard linear correlation
    SPEARMAN = "spearman"     # Rank-based correlation
    ROLLING_30D = "rolling_30d"   # 30-day rolling correlation
    ROLLING_90D = "rolling_90d"   # 90-day rolling correlation
    ROLLING_1Y = "rolling_1y"    # 1-year rolling correlation

class NewsCategory(enum.Enum):
    """News categorization for vector database bridging"""
    FEDERAL_RESERVE = "federal_reserve"
    EMPLOYMENT = "employment"
    INFLATION = "inflation"
    GDP_GROWTH = "gdp_growth"
    CORPORATE_EARNINGS = "corporate_earnings"
    GEOPOLITICAL = "geopolitical"
    MARKET_VOLATILITY = "market_volatility"
    SECTOR_SPECIFIC = "sector_specific"
    COMMODITY_MARKETS = "commodity_markets"
    INTERNATIONAL_TRADE = "international_trade"


# ============================================================================
# CORE METADATA TABLES
# ============================================================================

class DataSeries(Base):
    """
    Master table for all data series across all sources
    Unified metadata for FRED indicators, Yahoo Finance symbols, and calculated metrics
    """
    __tablename__ = "data_series"

    # Primary identification
    series_id = Column(VARCHAR(100), primary_key=True, comment="Unique identifier across all data sources")
    source_type = Column(Enum(DataSourceType), nullable=False, comment="Data source type")
    source_series_id = Column(VARCHAR(100), nullable=False, comment="Original series ID from source API")
    
    # Descriptive metadata
    title = Column(VARCHAR(500), nullable=False, comment="Human-readable series title")
    description = Column(Text, comment="Detailed description of the data series")
    category = Column(VARCHAR(200), comment="Primary category classification")
    subcategory = Column(VARCHAR(200), comment="Secondary category classification")
    
    # Data characteristics
    frequency = Column(Enum(FrequencyType), nullable=False, comment="Data release frequency")
    units = Column(VARCHAR(200), comment="Units of measurement")
    units_short = Column(VARCHAR(50), comment="Abbreviated units")
    seasonal_adjustment = Column(VARCHAR(100), comment="Seasonal adjustment type")
    
    # Time range information
    observation_start = Column(Date, comment="First available observation date")
    observation_end = Column(Date, comment="Last available observation date")
    last_updated = Column(DateTime(timezone=True), comment="Last update timestamp from source")
    
    # Source-specific metadata (flexible JSON storage)
    source_metadata = Column(JSONB, comment="Source-specific metadata as JSON")
    
    # Status and quality indicators
    is_active = Column(Boolean, default=True, nullable=False, comment="Whether series is actively updated")
    data_quality_score = Column(DECIMAL(3, 2), comment="Data quality score (0.00 to 1.00)")
    
    # Vector database integration
    news_categories = Column(JSONB, comment="Related news categories for vector DB bridging")
    correlation_priority = Column(Integer, default=0, comment="Priority for correlation calculations")
    
    # Audit fields
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Relationships
    observations = relationship("TimeSeriesObservation", back_populates="series", cascade="all, delete-orphan")
    correlations_primary = relationship("SeriesCorrelation", foreign_keys="SeriesCorrelation.primary_series_id", back_populates="primary_series")
    correlations_secondary = relationship("SeriesCorrelation", foreign_keys="SeriesCorrelation.secondary_series_id", back_populates="secondary_series")
    
    # Constraints
    __table_args__ = (
        Index('idx_data_series_source_type', 'source_type'),
        Index('idx_data_series_frequency', 'frequency'),
        Index('idx_data_series_category', 'category'),
        Index('idx_data_series_active', 'is_active'),
        CheckConstraint('data_quality_score >= 0.0 AND data_quality_score <= 1.0', name='ck_data_quality_score'),
    )


class MarketAssets(Base):
    """
    Specialized table for market assets (stocks, ETFs, indices)
    Extends DataSeries with market-specific metadata
    """
    __tablename__ = "market_assets"

    # Foreign key to DataSeries
    series_id = Column(VARCHAR(100), ForeignKey('data_series.series_id', ondelete='CASCADE'), primary_key=True)
    
    # Market-specific identification
    symbol = Column(VARCHAR(20), nullable=False, comment="Trading symbol (AAPL, SPY, etc.)")
    exchange = Column(VARCHAR(20), comment="Primary exchange (NASDAQ, NYSE, etc.)")
    asset_type = Column(Enum(AssetType), nullable=False, comment="Asset classification")
    
    # Company/Fund information
    company_name = Column(VARCHAR(500), comment="Full company or fund name")
    sector = Column(VARCHAR(100), comment="GICS sector classification")
    industry = Column(VARCHAR(200), comment="GICS industry classification")
    market_cap = Column(Numeric(20, 2), comment="Market capitalization in USD")
    
    # Market data characteristics
    currency = Column(VARCHAR(10), default='USD', comment="Trading currency")
    country = Column(VARCHAR(50), default='US', comment="Primary country")
    is_actively_traded = Column(Boolean, default=True, comment="Currently actively traded")
    
    # Index/ETF specific data
    expense_ratio = Column(DECIMAL(5, 4), comment="Expense ratio for ETFs/funds")
    index_tracked = Column(VARCHAR(200), comment="Underlying index for ETFs")
    
    # Economic correlation metadata
    economic_sensitivity = Column(JSONB, comment="Sensitivity to economic indicators")
    sector_exposure = Column(JSONB, comment="Sector exposure breakdown for ETFs")
    
    # Audit fields
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Relationships
    series = relationship("DataSeries", back_populates="market_asset")
    
    # Constraints
    __table_args__ = (
        Index('idx_market_assets_symbol', 'symbol'),
        Index('idx_market_assets_asset_type', 'asset_type'),
        Index('idx_market_assets_sector', 'sector'),
        Index('idx_market_assets_market_cap', 'market_cap'),
        UniqueConstraint('symbol', 'exchange', name='uk_market_assets_symbol_exchange'),
        CheckConstraint('expense_ratio >= 0.0 AND expense_ratio <= 1.0', name='ck_expense_ratio'),
    )

# Add relationship back to DataSeries
DataSeries.market_asset = relationship("MarketAssets", back_populates="series", uselist=False)


# ============================================================================
# TIME SERIES DATA TABLES
# ============================================================================

class TimeSeriesObservation(Base):
    """
    Unified time series data table for all observations
    Handles FRED economic data, Yahoo Finance market data, and calculated metrics
    """
    __tablename__ = "time_series_observations"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, comment="Unique observation identifier")
    
    # Series identification
    series_id = Column(VARCHAR(100), ForeignKey('data_series.series_id', ondelete='CASCADE'), nullable=False, comment="Reference to data series")
    
    # Time dimension
    observation_date = Column(Date, nullable=False, comment="Date of observation")
    observation_datetime = Column(DateTime(timezone=True), comment="Precise timestamp for intraday data")
    
    # Value storage (flexible for different data types)
    value = Column(Numeric(25, 10), comment="Primary numeric value")
    value_high = Column(Numeric(25, 10), comment="High value (for OHLC data)")
    value_low = Column(Numeric(25, 10), comment="Low value (for OHLC data)")
    value_open = Column(Numeric(25, 10), comment="Opening value (for OHLC data)")
    value_close = Column(Numeric(25, 10), comment="Closing value (for OHLC data)")
    volume = Column(Numeric(20, 2), comment="Trading volume or transaction count")
    
    # Data quality and source tracking
    data_quality = Column(DECIMAL(3, 2), default=1.0, comment="Quality score for this observation")
    is_estimated = Column(Boolean, default=False, comment="Whether value is estimated/interpolated")
    is_revised = Column(Boolean, default=False, comment="Whether this is a revision of previous data")
    revision_count = Column(Integer, default=0, comment="Number of times this observation has been revised")
    
    # Source and real-time tracking
    realtime_start = Column(Date, comment="Real-time period start (for FRED data)")
    realtime_end = Column(Date, comment="Real-time period end (for FRED data)")
    source_timestamp = Column(DateTime(timezone=True), comment="Original timestamp from data source")
    
    # Additional metadata
    observation_metadata = Column(JSONB, comment="Flexible metadata storage")
    
    # Audit fields
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Relationships
    series = relationship("DataSeries", back_populates="observations")
    
    # Constraints and indexes
    __table_args__ = (
        # Unique constraint ensures one observation per series per date
        UniqueConstraint('series_id', 'observation_date', name='uk_observations_series_date'),
        
        # Performance indexes
        Index('idx_observations_series_id', 'series_id'),
        Index('idx_observations_date', 'observation_date'),
        Index('idx_observations_series_date', 'series_id', 'observation_date'),
        Index('idx_observations_datetime', 'observation_datetime'),
        Index('idx_observations_value', 'value'),
        Index('idx_observations_quality', 'data_quality'),
        
        # Composite indexes for common queries
        Index('idx_observations_series_date_value', 'series_id', 'observation_date', 'value'),
        Index('idx_observations_date_range', 'observation_date', 'series_id', 'value'),
        
        # Data quality constraints
        CheckConstraint('data_quality >= 0.0 AND data_quality <= 1.0', name='ck_observation_quality'),
        CheckConstraint('revision_count >= 0', name='ck_revision_count'),
    )


# ============================================================================
# CORRELATION AND ANALYSIS TABLES
# ============================================================================

class SeriesCorrelation(Base):
    """
    Pre-calculated correlations between data series for performance optimization
    Stores rolling correlations at different time windows
    """
    __tablename__ = "series_correlations"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Series pair identification
    primary_series_id = Column(VARCHAR(100), ForeignKey('data_series.series_id', ondelete='CASCADE'), nullable=False)
    secondary_series_id = Column(VARCHAR(100), ForeignKey('data_series.series_id', ondelete='CASCADE'), nullable=False)
    
    # Correlation parameters
    correlation_type = Column(Enum(CorrelationType), nullable=False, comment="Type of correlation calculation")
    time_window_days = Column(Integer, nullable=False, comment="Time window for correlation calculation")
    
    # Time dimension
    calculation_date = Column(Date, nullable=False, comment="Date this correlation was calculated")
    start_date = Column(Date, nullable=False, comment="Start date of data used in calculation")
    end_date = Column(Date, nullable=False, comment="End date of data used in calculation")
    
    # Correlation results
    correlation_value = Column(DECIMAL(10, 8), comment="Correlation coefficient (-1.0 to 1.0)")
    p_value = Column(DECIMAL(15, 12), comment="Statistical significance p-value")
    sample_size = Column(Integer, comment="Number of observations used in calculation")
    
    # Statistical metadata
    primary_mean = Column(Numeric(25, 10), comment="Mean of primary series during period")
    secondary_mean = Column(Numeric(25, 10), comment="Mean of secondary series during period")
    primary_std = Column(Numeric(25, 10), comment="Standard deviation of primary series")
    secondary_std = Column(Numeric(25, 10), comment="Standard deviation of secondary series")
    
    # Quality and confidence measures
    confidence_level = Column(DECIMAL(5, 4), default=0.95, comment="Confidence level for statistical tests")
    data_completeness = Column(DECIMAL(5, 4), comment="Percentage of non-missing data in period")
    
    # Audit fields
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    calculation_runtime_ms = Column(Integer, comment="Calculation runtime in milliseconds")
    
    # Relationships
    primary_series = relationship("DataSeries", foreign_keys=[primary_series_id], back_populates="correlations_primary")
    secondary_series = relationship("DataSeries", foreign_keys=[secondary_series_id], back_populates="correlations_secondary")
    
    # Constraints and indexes
    __table_args__ = (
        # Unique constraint for correlation calculations
        UniqueConstraint('primary_series_id', 'secondary_series_id', 'correlation_type', 'time_window_days', 'calculation_date', 
                        name='uk_correlations_unique'),
        
        # Performance indexes
        Index('idx_correlations_primary_series', 'primary_series_id'),
        Index('idx_correlations_secondary_series', 'secondary_series_id'),
        Index('idx_correlations_date', 'calculation_date'),
        Index('idx_correlations_type', 'correlation_type'),
        Index('idx_correlations_value', 'correlation_value'),
        
        # Composite indexes for common queries
        Index('idx_correlations_series_pair', 'primary_series_id', 'secondary_series_id'),
        Index('idx_correlations_date_type', 'calculation_date', 'correlation_type'),
        
        # Data validation constraints
        CheckConstraint('correlation_value >= -1.0 AND correlation_value <= 1.0', name='ck_correlation_range'),
        CheckConstraint('p_value >= 0.0 AND p_value <= 1.0', name='ck_p_value_range'),
        CheckConstraint('sample_size > 0', name='ck_sample_size_positive'),
        CheckConstraint('data_completeness >= 0.0 AND data_completeness <= 1.0', name='ck_data_completeness'),
        CheckConstraint('primary_series_id != secondary_series_id', name='ck_different_series'),
    )


# ============================================================================
# NEWS AND VECTOR DATABASE INTEGRATION
# ============================================================================

class NewsTopicMapping(Base):
    """
    Bridge table connecting news topics to economic indicators and market assets
    Enables vector database integration and sentiment analysis
    """
    __tablename__ = "news_topic_mapping"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Topic identification
    topic_name = Column(VARCHAR(200), nullable=False, comment="News topic name")
    topic_category = Column(Enum(NewsCategory), nullable=False, comment="Primary category classification")
    topic_keywords = Column(JSONB, comment="Keywords associated with this topic")
    
    # Vector database integration
    vector_db_collection = Column(VARCHAR(200), comment="Vector database collection name")
    vector_db_filters = Column(JSONB, comment="Filters for vector database queries")
    
    # Economic indicator relationships
    related_series = Column(JSONB, comment="List of series_ids related to this topic")
    impact_weights = Column(JSONB, comment="Relative impact weights for each related series")
    
    # Market asset relationships
    related_assets = Column(JSONB, comment="List of market assets related to this topic")
    sentiment_multipliers = Column(JSONB, comment="Sentiment impact multipliers by asset")
    
    # Topic metadata
    description = Column(Text, comment="Detailed description of the topic")
    is_active = Column(Boolean, default=True, comment="Whether topic is actively monitored")
    priority_level = Column(Integer, default=1, comment="Priority level for analysis (1-10)")
    
    # Audit fields
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    
    # Constraints and indexes
    __table_args__ = (
        Index('idx_news_topics_category', 'topic_category'),
        Index('idx_news_topics_active', 'is_active'),
        Index('idx_news_topics_priority', 'priority_level'),
        UniqueConstraint('topic_name', 'topic_category', name='uk_news_topics_name_category'),
        CheckConstraint('priority_level >= 1 AND priority_level <= 10', name='ck_priority_level'),
    )


class NewsArticles(Base):
    """
    Full text storage for news articles with comprehensive metadata
    Integrates with vector database through NewsTopicMapping bridge table
    """
    __tablename__ = "news_articles"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, comment="Unique article identifier")
    
    # Article identification and source tracking
    source_article_id = Column(VARCHAR(200), comment="Original article ID from News API")
    url = Column(Text, nullable=False, comment="Original article URL")
    url_hash = Column(VARCHAR(64), comment="SHA-256 hash of URL for deduplication")
    
    # Publication metadata
    source_name = Column(VARCHAR(200), nullable=False, comment="News source name (e.g., 'Reuters', 'Bloomberg')")
    source_domain = Column(VARCHAR(100), comment="Source domain (e.g., 'reuters.com')")
    author = Column(VARCHAR(500), comment="Article author(s)")
    published_at = Column(DateTime(timezone=True), nullable=False, comment="Original publication timestamp")
    
    # Article content
    title = Column(VARCHAR(1000), nullable=False, comment="Article headline/title")
    description = Column(Text, comment="Article summary/description from News API")
    content = Column(Text, comment="Full article content")
    content_length = Column(Integer, comment="Character length of full content")
    
    # Language and geographic information
    language = Column(VARCHAR(10), default='en', comment="Article language code")
    country = Column(VARCHAR(10), comment="Country of publication")
    
    # News API specific metadata
    news_api_metadata = Column(JSONB, comment="Original metadata from News API")
    
    # Economic categorization and analysis
    economic_categories = Column(JSONB, comment="List of economic categories this article relates to")
    sentiment_score = Column(DECIMAL(5, 4), comment="Sentiment score (-1.0 to 1.0)")
    relevance_score = Column(DECIMAL(5, 4), comment="Economic relevance score (0.0 to 1.0)")
    
    # Vector database integration
    vector_db_collection = Column(VARCHAR(200), comment="Chroma collection containing this article's embeddings")
    vector_db_document_id = Column(VARCHAR(200), comment="Document ID in vector database")
    embedding_model_version = Column(VARCHAR(100), comment="Version of embedding model used")
    
    # Content processing flags
    is_processed = Column(Boolean, default=False, comment="Whether article has been processed for embeddings")
    is_categorized = Column(Boolean, default=False, comment="Whether economic categorization is complete")
    has_embeddings = Column(Boolean, default=False, comment="Whether embeddings are stored in vector DB")
    
    # Data quality indicators
    data_quality_score = Column(DECIMAL(3, 2), comment="Overall data quality score (0.00 to 1.00)")
    content_completeness = Column(DECIMAL(3, 2), comment="Completeness of article content")
    duplicate_probability = Column(DECIMAL(3, 2), comment="Probability this is a duplicate article")
    
    # Processing and error tracking
    processing_attempts = Column(Integer, default=0, comment="Number of processing attempts")
    last_processing_error = Column(Text, comment="Last error encountered during processing")
    
    # Related economic data
    related_series_ids = Column(JSONB, comment="Economic series this article may impact")
    related_market_assets = Column(JSONB, comment="Market assets this article may impact")
    impact_timeframe = Column(VARCHAR(50), comment="Expected timeframe of market impact")
    
    # Audit fields
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)
    processed_at = Column(DateTime(timezone=True), comment="When article was last processed for embeddings")
    
    # Constraints and indexes
    __table_args__ = (
        # Unique constraint on URL hash to prevent duplicates
        UniqueConstraint('url_hash', name='uk_news_articles_url_hash'),
        
        # Performance indexes for common queries
        Index('idx_news_articles_published_at', 'published_at'),
        Index('idx_news_articles_source_name', 'source_name'),
        Index('idx_news_articles_language', 'language'),
        Index('idx_news_articles_processed', 'is_processed'),
        Index('idx_news_articles_categorized', 'is_categorized'),
        Index('idx_news_articles_has_embeddings', 'has_embeddings'),
        Index('idx_news_articles_quality', 'data_quality_score'),
        Index('idx_news_articles_relevance', 'relevance_score'),
        
        # Composite indexes for complex queries
        Index('idx_news_articles_date_source', 'published_at', 'source_name'),
        Index('idx_news_articles_quality_relevance', 'data_quality_score', 'relevance_score'),
        Index('idx_news_articles_processing_status', 'is_processed', 'is_categorized', 'has_embeddings'),
        
        # Indexes for vector database integration
        Index('idx_news_articles_vector_collection', 'vector_db_collection'),
        Index('idx_news_articles_vector_doc_id', 'vector_db_document_id'),
        
        # Data quality constraints
        CheckConstraint('data_quality_score >= 0.0 AND data_quality_score <= 1.0', name='ck_news_quality_score'),
        CheckConstraint('content_completeness >= 0.0 AND content_completeness <= 1.0', name='ck_content_completeness'),
        CheckConstraint('duplicate_probability >= 0.0 AND duplicate_probability <= 1.0', name='ck_duplicate_probability'),
        CheckConstraint('sentiment_score >= -1.0 AND sentiment_score <= 1.0', name='ck_sentiment_range'),
        CheckConstraint('relevance_score >= 0.0 AND relevance_score <= 1.0', name='ck_relevance_range'),
        CheckConstraint('processing_attempts >= 0', name='ck_processing_attempts'),
        CheckConstraint('content_length >= 0', name='ck_content_length'),
    )


# ============================================================================
# SYNC AND MONITORING TABLES
# ============================================================================

class DataSyncLog(Base):
    """
    Unified sync logging for all data sources
    Tracks ingestion performance, errors, and data quality across APIs
    """
    __tablename__ = "data_sync_log"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Sync identification
    series_id = Column(VARCHAR(100), ForeignKey('data_series.series_id', ondelete='SET NULL'), comment="Related series (optional)")
    source_type = Column(Enum(DataSourceType), nullable=False, comment="Data source type")
    sync_type = Column(VARCHAR(50), nullable=False, comment="Type of sync operation")
    
    # Timing information
    sync_start_time = Column(DateTime(timezone=True), nullable=False, comment="Sync operation start time")
    sync_end_time = Column(DateTime(timezone=True), comment="Sync operation end time")
    sync_duration_ms = Column(Integer, comment="Duration in milliseconds")
    
    # Sync results
    success = Column(Boolean, nullable=False, comment="Whether sync completed successfully")
    records_processed = Column(Integer, default=0, comment="Total records processed")
    records_added = Column(Integer, default=0, comment="New records added")
    records_updated = Column(Integer, default=0, comment="Existing records updated")
    records_failed = Column(Integer, default=0, comment="Records that failed processing")
    
    # API usage tracking
    api_calls_made = Column(Integer, default=0, comment="Number of API calls made")
    api_quota_remaining = Column(Integer, comment="Remaining API quota after sync")
    rate_limit_hits = Column(Integer, default=0, comment="Number of rate limit encounters")
    
    # Error information
    error_message = Column(Text, comment="Error message if sync failed")
    error_type = Column(VARCHAR(200), comment="Classification of error type")
    stack_trace = Column(Text, comment="Full stack trace for debugging")
    
    # Data quality metrics
    data_quality_score = Column(DECIMAL(3, 2), comment="Overall data quality score for this sync")
    missing_data_percentage = Column(DECIMAL(5, 2), comment="Percentage of missing/null values")
    duplicate_records = Column(Integer, default=0, comment="Number of duplicate records encountered")
    
    # Sync metadata
    sync_parameters = Column(JSONB, comment="Parameters used for this sync operation")
    source_metadata = Column(JSONB, comment="Metadata returned from source API")
    
    # Date range processed
    date_range_start = Column(Date, comment="Start date of data processed")
    date_range_end = Column(Date, comment="End date of data processed")
    
    # Relationships
    series = relationship("DataSeries")
    
    # Constraints and indexes
    __table_args__ = (
        Index('idx_sync_log_source_type', 'source_type'),
        Index('idx_sync_log_sync_start', 'sync_start_time'),
        Index('idx_sync_log_success', 'success'),
        Index('idx_sync_log_series_id', 'series_id'),
        Index('idx_sync_log_date_range', 'date_range_start', 'date_range_end'),
        
        # Composite indexes for monitoring queries
        Index('idx_sync_log_source_success', 'source_type', 'success', 'sync_start_time'),
        Index('idx_sync_log_series_date', 'series_id', 'sync_start_time'),
        
        # Data validation
        CheckConstraint('records_processed >= 0', name='ck_records_processed'),
        CheckConstraint('records_added >= 0', name='ck_records_added'),
        CheckConstraint('records_updated >= 0', name='ck_records_updated'),
        CheckConstraint('records_failed >= 0', name='ck_records_failed'),
        CheckConstraint('api_calls_made >= 0', name='ck_api_calls'),
        CheckConstraint('data_quality_score >= 0.0 AND data_quality_score <= 1.0', name='ck_sync_quality_score'),
    )


class SystemHealthMetrics(Base):
    """
    System-wide health and performance monitoring
    Tracks overall platform performance and data freshness
    """
    __tablename__ = "system_health_metrics"

    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Time dimension
    metric_timestamp = Column(DateTime(timezone=True), nullable=False, default=lambda: datetime.now(timezone.utc))
    metric_date = Column(Date, nullable=False, comment="Date for daily rollup metrics")
    
    # System performance metrics
    total_series_count = Column(Integer, comment="Total number of active data series")
    total_observations_count = Column(Numeric(15, 0), comment="Total observations in database")
    daily_sync_success_rate = Column(DECIMAL(5, 4), comment="Success rate for today's syncs")
    average_sync_duration_ms = Column(Integer, comment="Average sync duration today")
    
    # Data freshness metrics
    stale_series_count = Column(Integer, comment="Number of series with stale data")
    missing_data_series_count = Column(Integer, comment="Series with significant missing data")
    data_quality_average = Column(DECIMAL(5, 4), comment="Average data quality score")
    
    # API usage metrics
    total_api_calls_today = Column(Integer, comment="Total API calls made today")
    api_quota_utilization = Column(JSONB, comment="API quota utilization by source")
    rate_limit_incidents = Column(Integer, comment="Rate limit incidents today")
    
    # Error tracking
    error_count_24h = Column(Integer, comment="Errors in last 24 hours")
    critical_errors_24h = Column(Integer, comment="Critical errors in last 24 hours")
    top_error_types = Column(JSONB, comment="Most common error types and counts")
    
    # Storage metrics
    database_size_mb = Column(Numeric(12, 2), comment="Database size in megabytes")
    growth_rate_mb_per_day = Column(Numeric(8, 2), comment="Daily growth rate")
    
    # Performance metrics
    average_query_time_ms = Column(Integer, comment="Average query response time")
    correlation_calculation_time_ms = Column(Integer, comment="Time for correlation calculations")
    
    # Constraints and indexes
    __table_args__ = (
        Index('idx_health_metrics_timestamp', 'metric_timestamp'),
        Index('idx_health_metrics_date', 'metric_date'),
        UniqueConstraint('metric_date', name='uk_health_metrics_date'),
        
        # Data validation
        CheckConstraint('daily_sync_success_rate >= 0.0 AND daily_sync_success_rate <= 1.0', name='ck_success_rate'),
        CheckConstraint('data_quality_average >= 0.0 AND data_quality_average <= 1.0', name='ck_avg_quality'),
    )


# ============================================================================
# DATABASE CREATION AND MANAGEMENT
# ============================================================================

class DatabaseManager:
    """
    Enterprise-grade database management with comprehensive error handling
    """
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = None
        self.AsyncSessionLocal = None
        
    async def initialize_engine(self) -> bool:
        """
        Initialize the database engine with connection pooling and error handling
        """
        try:
            self.engine = create_async_engine(
                self.database_url,
                echo=False,  # Set to True for SQL debugging
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,  # Verify connections before use
                pool_recycle=3600   # Recycle connections every hour
            )
            
            # Create async session factory
            self.AsyncSessionLocal = sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            logger.info("Database engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database engine: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """
        Test database connectivity
        """
        try:
            from sqlalchemy import text
            async with self.engine.begin() as conn:
                result = await conn.execute(text("SELECT 1 as test"))
                test_result = result.scalar()
                if test_result == 1:
                    logger.info("Database connection test successful")
                    return True
                else:
                    logger.error("Database connection test failed - unexpected result")
                    return False
                    
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False
    
    async def create_tables(self) -> bool:
        """
        Create all database tables with comprehensive error handling
        """
        try:
            logger.info("Starting database table creation...")
            
            async with self.engine.begin() as conn:
                # Create all tables
                await conn.run_sync(Base.metadata.create_all)
                
                logger.info("Database tables created successfully")
                
                # Verify table creation
                await self._verify_tables(conn)
                
                # Create additional performance indexes
                await self._create_additional_indexes_safe(conn)
                
            # Insert initial reference data (separate transaction)
            await self._insert_reference_data()
                
            logger.info("Database setup completed successfully")
            return True
            
        except SQLAlchemyError as e:
            logger.error(f"SQLAlchemy error during table creation: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during table creation: {e}")
            return False
    

    
    async def _create_additional_indexes_safe(self, conn) -> None:
        """
        Create additional performance indexes safely (no CONCURRENTLY)
        """
        try:
            from sqlalchemy import text
            
            # Simple indexes without CONCURRENTLY - create one at a time with error handling
            additional_indexes = [
                ("News Articles Date Quality", "CREATE INDEX IF NOT EXISTS idx_news_articles_date_quality ON news_articles (published_at DESC, data_quality_score DESC)"),
                ("News Articles Processing Queue", "CREATE INDEX IF NOT EXISTS idx_news_articles_processing_queue ON news_articles (is_processed, processing_attempts, created_at)"),
                ("Observations Date Series Value", "CREATE INDEX IF NOT EXISTS idx_observations_date_series_value ON time_series_observations (observation_date DESC, series_id, value)"),
                ("Correlations Latest", "CREATE INDEX IF NOT EXISTS idx_correlations_latest ON series_correlations (primary_series_id, secondary_series_id, calculation_date DESC)"),
                ("Sync Log Failures", "CREATE INDEX IF NOT EXISTS idx_sync_log_failures ON data_sync_log (source_type, success, sync_start_time DESC)"),
                ("News Topics Active Priority", "CREATE INDEX IF NOT EXISTS idx_news_topics_active_priority ON news_topic_mapping (is_active, priority_level DESC)"),
                ("Health Metrics Latest", "CREATE INDEX IF NOT EXISTS idx_health_metrics_latest ON system_health_metrics (metric_date DESC, metric_timestamp DESC)"),
            ]
            
            for index_name, index_sql in additional_indexes:
                try:
                    logger.info(f"Creating index: {index_name}")
                    await conn.execute(text(index_sql))
                    logger.debug(f"✅ Created index: {index_name}")
                except Exception as e:
                    logger.warning(f"Index creation warning for {index_name}: {e}")
                    # Continue with other indexes even if one fails
            
            logger.info("Additional indexes creation completed")
            
        except Exception as e:
            logger.error(f"Error creating additional indexes: {e}")
            # Don't raise - indexes are performance optimization
    
    async def _verify_tables(self, conn) -> None:
        """
        Verify that all expected tables were created
        """
        expected_tables = [
            'data_series', 'market_assets', 'time_series_observations',
            'series_correlations', 'news_topic_mapping', 'news_articles', 
            'data_sync_log', 'system_health_metrics'
        ]
        
        try:
            from sqlalchemy import text
            result = await conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_type = 'BASE TABLE'
            """))
            
            existing_tables = {row[0] for row in result.fetchall()}
            missing_tables = set(expected_tables) - existing_tables
            
            if missing_tables:
                raise Exception(f"Missing tables: {missing_tables}")
            
            logger.info(f"All {len(expected_tables)} tables verified successfully")
            
        except Exception as e:
            logger.error(f"Table verification failed: {e}")
            raise
    
    async def _create_additional_indexes(self, conn) -> None:
        """
        Create additional performance indexes and constraints
        """
        try:
            from sqlalchemy import text
            
            # Additional performance indexes for time-series queries
            additional_indexes = [
                # Partitioning-friendly indexes for large time series data
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_observations_date_series_value ON time_series_observations (observation_date DESC, series_id, value) WHERE value IS NOT NULL",
                
                # Correlation analysis indexes
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_correlations_latest ON series_correlations (primary_series_id, secondary_series_id, calculation_date DESC)",
                
                # Monitoring and alerting indexes
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sync_log_failures ON data_sync_log (source_type, success, sync_start_time DESC) WHERE success = false",
                
                # News integration indexes
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_news_topics_active_priority ON news_topic_mapping (is_active, priority_level DESC) WHERE is_active = true",
                
                # System health monitoring
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_health_metrics_latest ON system_health_metrics (metric_date DESC, metric_timestamp DESC)",
            ]
            
            for index_sql in additional_indexes:
                try:
                    await conn.execute(text(index_sql))
                    logger.debug(f"Created index: {index_sql[:50]}...")
                except Exception as e:
                    logger.warning(f"Index creation warning (may already exist): {e}")
            
            logger.info("Additional indexes created successfully")
            
        except Exception as e:
            logger.error(f"Error creating additional indexes: {e}")
            # Don't raise - indexes are performance optimization, not critical
    
    async def _insert_reference_data(self) -> None:
        """
        Insert initial reference data and seed data
        """
        try:
            async with self.AsyncSessionLocal() as session:
                # Insert news topic mappings for major economic categories
                await self._insert_news_topics(session)
                
                # Insert system health baseline
                await self._insert_health_baseline(session)
                
                await session.commit()
                logger.info("Reference data inserted successfully")
                
        except Exception as e:
            logger.error(f"Error inserting reference data: {e}")
            # Don't raise - reference data can be added later
    
    async def _insert_news_topics(self, session: AsyncSession) -> None:
        """
        Insert predefined news topic mappings
        """
        news_topics = [
            {
                'topic_name': 'Federal Reserve Policy',
                'topic_category': NewsCategory.FEDERAL_RESERVE,
                'topic_keywords': ['federal reserve', 'fed', 'interest rates', 'monetary policy', 'jerome powell'],
                'related_series': ['FEDFUNDS', 'DGS10', 'DGS2', 'T10Y3M'],
                'related_assets': ['XLF', 'SPY', 'QQQ'],
                'description': 'Federal Reserve monetary policy decisions and communications',
                'priority_level': 10
            },
            {
                'topic_name': 'Employment Data',
                'topic_category': NewsCategory.EMPLOYMENT,
                'topic_keywords': ['unemployment', 'jobs', 'payrolls', 'employment', 'labor market'],
                'related_series': ['UNRATE', 'PAYEMS', 'ICSA', 'AHETPI'],
                'related_assets': ['XLY', 'SPY', 'IWM'],
                'description': 'Employment reports and labor market indicators',
                'priority_level': 9
            },
            {
                'topic_name': 'Inflation Indicators',
                'topic_category': NewsCategory.INFLATION,
                'topic_keywords': ['inflation', 'cpi', 'consumer prices', 'ppi', 'price index'],
                'related_series': ['CPIAUCSL', 'AHETPI'],
                'related_assets': ['XLP', 'XLE', 'GOLDAMGBD228NLBM'],
                'description': 'Consumer and producer price index data',
                'priority_level': 9
            },
            {
                'topic_name': 'GDP and Economic Growth',
                'topic_category': NewsCategory.GDP_GROWTH,
                'topic_keywords': ['gdp', 'economic growth', 'recession', 'expansion'],
                'related_series': ['GDP', 'PERMIT', 'HOUST'],
                'related_assets': ['SPY', 'XLI', 'QQQ'],
                'description': 'Gross Domestic Product and economic growth indicators',
                'priority_level': 8
            },
            {
                'topic_name': 'Market Volatility',
                'topic_category': NewsCategory.MARKET_VOLATILITY,
                'topic_keywords': ['vix', 'volatility', 'market crash', 'correction', 'panic'],
                'related_series': ['VIXCLS', 'TEDRATE'],
                'related_assets': ['SPY', 'QQQ', 'VTI'],
                'description': 'Market volatility and stress indicators',
                'priority_level': 7
            }
        ]
        
        for topic_data in news_topics:
            topic = NewsTopicMapping(**topic_data)
            session.add(topic)
    
    async def _insert_health_baseline(self, session: AsyncSession) -> None:
        """
        Insert baseline system health metrics
        """
        baseline_health = SystemHealthMetrics(
            metric_date=datetime.now(timezone.utc).date(),
            total_series_count=0,
            total_observations_count=0,
            daily_sync_success_rate=1.0,
            data_quality_average=1.0,
            total_api_calls_today=0,
            error_count_24h=0,
            critical_errors_24h=0
        )
        session.add(baseline_health)
    
    async def drop_tables(self) -> bool:
        """
        Drop all tables (use with caution!)
        """
        try:
            logger.warning("DROPPING ALL TABLES - This cannot be undone!")
            
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
                
            logger.warning("All tables dropped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error dropping tables: {e}")
            return False
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive database statistics
        """
        try:
            async with self.AsyncSessionLocal() as session:
                from sqlalchemy import text
                
                stats = {}
                
                # Table row counts
                table_stats = await session.execute(text("""
                    SELECT 
                        schemaname,
                        tablename,
                        n_tup_ins as inserts,
                        n_tup_upd as updates,
                        n_tup_del as deletes,
                        n_live_tup as live_rows,
                        n_dead_tup as dead_rows
                    FROM pg_stat_user_tables
                    ORDER BY live_rows DESC
                """))
                
                stats['table_statistics'] = [dict(row) for row in table_stats.fetchall()]
                
                # Database size
                size_result = await session.execute(text("""
                    SELECT pg_size_pretty(pg_database_size(current_database())) as database_size,
                           pg_database_size(current_database()) as size_bytes
                """))
                size_row = size_result.fetchone()
                stats['database_size'] = dict(size_row)
                
                # Index usage
                index_stats = await session.execute(text("""
                    SELECT 
                        schemaname,
                        tablename,
                        indexname,
                        idx_tup_read,
                        idx_tup_fetch
                    FROM pg_stat_user_indexes
                    WHERE idx_tup_read > 0
                    ORDER BY idx_tup_read DESC
                    LIMIT 20
                """))
                
                stats['index_usage'] = [dict(row) for row in index_stats.fetchall()]
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting database statistics: {e}")
            return {}
    
    async def cleanup(self) -> None:
        """
        Clean up database connections
        """
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connections closed")


# ============================================================================
# MAIN SETUP FUNCTIONS
# ============================================================================

async def create_database_schema(database_url: str, drop_existing: bool = False) -> bool:
    """
    Main function to create the complete database schema
    
    Args:
        database_url: PostgreSQL connection string
        drop_existing: Whether to drop existing tables first (DANGEROUS!)
        
    Returns:
        bool: Success status
    """
    logger.info("=== Economic Data Platform Database Setup ===")
    logger.info(f"Target database: {database_url.split('@')[1] if '@' in database_url else 'Unknown'}")
    
    db_manager = DatabaseManager(database_url)
    
    try:
        # Initialize engine
        if not await db_manager.initialize_engine():
            logger.error("Failed to initialize database engine")
            return False
        
        # Test connection
        if not await db_manager.test_connection():
            logger.error("Database connection test failed")
            return False
        
        # Drop existing tables if requested
        if drop_existing:
            logger.warning("Dropping existing tables as requested...")
            if not await db_manager.drop_tables():
                logger.error("Failed to drop existing tables")
                return False
        
        # Create new schema
        if not await db_manager.create_tables():
            logger.error("Failed to create database tables")
            return False
        
        # Get final statistics
        stats = await db_manager.get_database_stats()
        if stats:
            logger.info("=== Database Setup Complete ===")
            logger.info(f"Database size: {stats.get('database_size', {}).get('database_size', 'Unknown')}")
            logger.info(f"Tables created: {len(stats.get('table_statistics', []))}")
        
        return True
        
    except Exception as e:
        logger.error(f"Database setup failed with unexpected error: {e}")
        return False
        
    finally:
        await db_manager.cleanup()


def validate_database_url(database_url: str) -> bool:
    """
    Validate database URL format and accessibility
    """
    if not database_url:
        logger.error("Database URL is required")
        return False
    
    if not database_url.startswith(('postgresql://', 'postgresql+psycopg://', 'postgresql+asyncpg://')):
        logger.error("Database URL must be a PostgreSQL connection string")
        return False
    
    # Basic format validation
    try:
        from urllib.parse import urlparse
        parsed = urlparse(database_url)
        
        if not parsed.hostname:
            logger.error("Database URL missing hostname")
            return False
            
        if not parsed.port:
            logger.warning("Database URL missing port, using default 5432")
            
        return True
        
    except Exception as e:
        logger.error(f"Invalid database URL format: {e}")
        return False


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

async def main():
    """
    Main entry point for database setup
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Economic Data Platform Database Setup')
    parser.add_argument('--database-url', 
                       default="postgresql+psycopg://postgres:fred_password@localhost:5432/postgres",
                       help='PostgreSQL database connection URL')
    parser.add_argument('--drop-existing', 
                       action='store_true',
                       help='Drop existing tables before creating new ones (DANGEROUS!)')
    parser.add_argument('--log-level',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Validate database URL
    if not validate_database_url(args.database_url):
        sys.exit(1)
    
    # Confirm destructive operations
    if args.drop_existing:
        response = input("⚠️  WARNING: This will DROP ALL EXISTING TABLES! Type 'yes' to continue: ")
        if response.lower() != 'yes':
            logger.info("Operation cancelled by user")
            sys.exit(0)
    
    # Create database schema
    success = await create_database_schema(
        database_url=args.database_url,
        drop_existing=args.drop_existing
    )
    
    if success:
        logger.info("🎉 Database setup completed successfully!")
        logger.info("Next steps:")
        logger.info("  1. Configure your API credentials")
        logger.info("  2. Run data ingestion services")
        logger.info("  3. Set up monitoring and alerting")
        sys.exit(0)
    else:
        logger.error("❌ Database setup failed - check logs for details")
        sys.exit(1)


if __name__ == "__main__":
    """
    Entry point when script is run directly
    """
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
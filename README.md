# Vector View - AI Financial Intelligence Platform

An intelligent financial analysis platform that synthesizes economic data, market movements, and news sentiment into actionable daily briefings powered by AI agents.

## 🎯 Project Vision

Transform scattered financial information into coherent, AI-synthesized intelligence by combining:
- **Economic indicators** (FRED API) 
- **Market data** (Yahoo Finance)
- **News sentiment** (News API + Vector DB)
- **AI orchestration** for multi-perspective analysis
- **Daily synthesis** into digestible briefings

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FRED API      │    │  Yahoo Finance  │    │    News API     │
│ (Economic Data) │    │ (Market Data)   │    │ (News Articles) │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────────────────────────────┐    ┌─────────────────┐
│           PostgreSQL Database           │    │  Vector Database│
│  ┌─────────────┐  ┌─────────────────┐  │    │ (Semantic News) │
│  │ Economic    │  │ Market Data     │  │    │                 │
│  │ Indicators  │  │ (OHLCV, etc.)   │  │    │                 │
│  └─────────────┘  └─────────────────┘  │    │                 │
└─────────────────┬───────────────────────┘    └─────────┬───────┘
                  │                                      │
                  └──────────────┬───────────────────────┘
                                 │
                  ┌──────────────▼───────────────┐
                  │        Bridge Layer          │
                  │   (Data Correlation &        │
                  │    Context Mapping)          │
                  └──────────────┬───────────────┘
                                 │
                  ┌──────────────▼───────────────┐
                  │     AI Orchestration         │
                  │                              │
                  │  ┌─────────────────────────┐ │
                  │  │   Orchestration Agent   │ │
                  │  └─────────────┬───────────┘ │
                  │                │             │
                  │ ┌──────────────▼───────────┐ │
                  │ │    Economic Analysis     │ │
                  │ │        Agent             │ │
                  │ └──────────────────────────┘ │
                  │                              │
                  │ ┌──────────────────────────┐ │
                  │ │   Sentiment Analysis     │ │
                  │ │        Agent             │ │
                  │ └──────────────────────────┘ │
                  │                              │
                  │ ┌──────────────────────────┐ │
                  │ │    Market Analysis       │ │
                  │ │        Agent             │ │
                  │ └──────────────────────────┘ │
                  └──────────────┬───────────────┘
                                 │
                  ┌──────────────▼───────────────┐
                  │      Synthesis Agent         │
                  │   (Daily Brief Generation)   │
                  └──────────────┬───────────────┘
                                 │
                  ┌──────────────▼───────────────┐
                  │      Streamlit Frontend      │
                  │                              │
                  │  ┌─────────────────────────┐ │
                  │  │    Daily Briefing       │ │
                  │  │      Dashboard          │ │
                  │  └─────────────────────────┘ │
                  │                              │
                  │  ┌─────────────────────────┐ │
                  │  │   Interactive Chatbot   │ │
                  │  │   (Deep Dive Queries)   │ │
                  │  └─────────────────────────┘ │
                  └─────────────────────────────┘
```

## 🔄 Data Flow

1. **Ingestion**: Scheduled data pulls from FRED, Yahoo Finance, and News API
2. **Storage**: Economic/market data → PostgreSQL, News → Vector database  
3. **Correlation**: Bridge layer maps news topics to economic indicators
4. **Analysis**: Specialized AI agents analyze different dimensions
5. **Synthesis**: Master agent combines insights into daily briefings
6. **Delivery**: Streamlit interface presents briefings + interactive chat

## 🤖 AI Agent Architecture

### Orchestration Agent
- **Role**: Master coordinator and task delegation
- **Responsibilities**: Query routing, agent coordination, context management

### Economic Analysis Agent  
- **Role**: Economic indicator interpretation
- **Data Sources**: FRED economic data, correlation matrices
- **Outputs**: Economic trend analysis, indicator explanations

### Sentiment Analysis Agent
- **Role**: News sentiment and market psychology
- **Data Sources**: Vector database news content, market volatility data
- **Outputs**: Sentiment scores, news impact analysis

### Market Analysis Agent
- **Role**: Technical and fundamental market analysis  
- **Data Sources**: Yahoo Finance OHLCV data, volume analysis
- **Outputs**: Market trend analysis, sector insights

### Synthesis Agent
- **Role**: Unified briefing generation
- **Inputs**: All specialist agent outputs + user context
- **Outputs**: Coherent daily briefings, executive summaries

## 📊 Database Schema

### PostgreSQL (Structured Data)
- **data_series**: Master catalog of all time series
- **time_series_observations**: Unified OHLCV + economic data
- **series_correlations**: Pre-calculated correlation matrices
- **market_assets**: Stock/ETF metadata and classifications
- **sync_logs**: Data ingestion monitoring and quality tracking

### ChromaDB (Semantic Data) ✅
- **news_embeddings**: 929 news articles with semantic embeddings
- **economic_indicators**: Economic data embeddings for correlation
- **semantic_search**: Cross-domain semantic search capabilities

### Bridge Tables
- **news_topic_mapping**: Links news categories to economic indicators
- **correlation_triggers**: News events that correlate with market moves

## 📊 Current Implementation Status

### ✅ **Completed Components**
- **Database Infrastructure**: PostgreSQL with 441,343 time series observations
- **Data Ingestion**: Modular FRED, Yahoo Finance, and News API pipelines
- **Semantic Search**: ChromaDB with 929 embedded news articles
- **AI Agent Architecture**: Complete 5-agent system with Ollama integration
- **Monitoring**: Comprehensive dashboard for data pipeline monitoring
- **Automation**: Cron job setup for daily data updates

### 🚧 **In Development**
- **Agent Orchestration**: Multi-agent workflow coordination
- **Daily Briefings**: AI-generated market intelligence synthesis
- **Frontend Interface**: User-facing Streamlit application

### 📋 **Planned Features**
- **Interactive Frontend**: User-facing Streamlit application
- **Chatbot Interface**: Natural language query processing
- **Advanced Analytics**: Custom analysis and alerting

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- PostgreSQL 15+
- ChromaDB (included)
- Docker & Docker Compose
- API keys: FRED, News API

### Quick Setup
```bash
# 1. Clone and setup environment
git clone <repository>
cd vector-view
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

# 2. Start database
docker-compose up -d

# 3. Initialize database schema
python database/unified_database_setup.py

# 4. Configure API credentials
cp .env.example .env
# Edit .env with your API keys

# 5. Run initial data ingestion
python -m ingestion.fred.fred_bulk_loader
python -m ingestion.yahoo.yahoo_bulk_loader
python -m ingestion.news.news_historical_backfill

# 6. Start monitoring dashboard
streamlit run ingestion/utilities/monitoring_dashboard.py
```

## 📁 Current Project Structure

```
vector-view/
├── README.md
├── requirements.txt
├── docker-compose.yml
├── .env.example
├── check_data.py                    # Data validation utility
│
├── database/
│   ├── unified_database_setup.py    # Complete schema definition ✅
│   └── backups/                     # Database backup storage
│
├── ingestion/                       # Data ingestion pipeline ✅
│   ├── fred/                        # FRED Economic Data Module
│   │   ├── fred_bulk_loader.py      # FRED API bulk historical data
│   │   ├── fred_daily_updater.py    # FRED daily incremental updates
│   │   ├── fred_client.py           # FRED API client
│   │   ├── fred_database_integration.py # FRED data storage
│   │   └── fred_series_fetcher.py   # FRED data fetching logic
│   ├── yahoo/                       # Yahoo Finance Module
│   │   ├── yahoo_bulk_loader.py     # Yahoo Finance bulk historical data
│   │   ├── yahoo_daily_updater.py   # Yahoo Finance daily updates
│   │   ├── yahoo_finance_client.py  # Yahoo Finance API client
│   │   ├── yahoo_database_integration.py # Yahoo data storage
│   │   └── yahoo_series_fetcher.py  # Yahoo data fetching logic
│   ├── news/                        # News API Module
│   │   ├── news_historical_backfill.py # News API historical data
│   │   ├── news_daily_updater.py    # News API daily updates
│   │   ├── news_client.py           # News API client
│   │   ├── news_database_integration.py # News data storage
│   │   ├── news_series_fetcher.py   # News data fetching logic
│   │   └── news_monitoring_dashboard.py # News-specific monitoring
│   ├── utilities/                   # Utility Scripts & Tools
│   │   ├── monitoring_dashboard.py  # Main monitoring interface ✅
│   │   ├── check_daily_sync_health.sh # Health check script
│   │   └── run_daily_news_sync.sh   # News sync automation
│   ├── config/                      # Configuration & Progress Files
│   │   ├── daily_sync_stats.json    # Daily sync statistics
│   │   ├── daily_sync_progress.json # Sync progress tracking
│   │   └── backfill_progress.json   # Historical backfill progress
│   └── logs/                        # Log Files
│       ├── database_setup.log       # Database setup logs
│       └── [various operation logs]
│
├── semantic/                        # Semantic search & embeddings ✅
│   ├── embedding_pipeline.py        # News article embedding generation
│   ├── semantic_search.py           # ChromaDB search interface
│   ├── chroma_db/                   # ChromaDB vector storage
│   └── debug_news_articles.py       # Debugging utilities
│
├── chroma_db/                       # ChromaDB persistent storage ✅
│
├── docs/
│   └── data_sources.md              # Data source documentation
│
├── agents/                          # AI Agents ✅
│   ├── base_agent.py                # Base agent class ✅
│   ├── orchestration_agent.py       # Master orchestrator ✅
│   ├── ai_service.py                # AI service integration (Ollama) ✅
│   ├── economic/                    # Economic Analysis Module ✅
│   │   ├── economic_agent.py        # Economic analysis specialist ✅
│   │   ├── economic_context_builder.py # Economic context assembly
│   │   └── economic_data_handler.py # Economic data processing
│   ├── market_intelligence/         # Market Analysis Module ✅
│   │   ├── market_intelligence_agent.py # Market analysis specialist ✅
│   │   ├── market_context_builder.py # Market context assembly
│   │   └── market_data_handler.py   # Market data processing
│   ├── news_sentiment/              # Sentiment Analysis Module ✅
│   │   ├── news_sentiment_agent.py  # News sentiment specialist ✅
│   │   ├── news_sentiment_context_builder.py # Sentiment context
│   │   └── news_sentiment_data_handler.py # News data processing
│   └── editorial/                   # Editorial Synthesis Module ✅
│       ├── editorial_synthesis_agent.py # Daily briefing generator ✅
│       ├── editorial_context_builder.py # Editorial context assembly
│       └── editorial_data_handler.py # Editorial data processing
│
├── bridge/                          # Cross-domain correlation (Planned)
│   ├── correlation_engine.py        # Economic-news correlation
│   ├── topic_mapper.py              # News topic classification
│   └── context_builder.py           # Cross-dataset context creation
│
├── app/                             # Frontend application (Planned)
│   ├── main.py                      # Streamlit main application
│   ├── components/
│   └── utils/
│
└── setup scripts/
    ├── setup_daily_sync.sh          # Daily data sync automation
    ├── setup_news_daily_cron.sh     # News update cron setup
    └── setup_daily_news_cron.sh     # Alternative news cron setup
```

## 🎯 Key Features

### For Users
- **Daily Intelligence Briefings**: AI-synthesized market and economic summaries
- **Interactive Deep Dives**: Chat interface for detailed analysis of any topic
- **Correlation Insights**: Discover connections between news events and market moves
- **Multi-timeframe Analysis**: From intraday movements to long-term economic trends

### For Developers  
- **Modular Agent Architecture**: Easy to extend with new analysis capabilities
- **Unified Time Series Database**: Consistent data model across all sources
- **Vector-Relational Bridge**: Semantic search meets structured queries
- **Comprehensive Monitoring**: Data quality tracking and system health metrics

## 🔮 Roadmap

- [x] **Phase 1**: Core data ingestion and database setup ✅
- [x] **Phase 2**: AI agents and foundational architecture ✅
- [ ] **Phase 3**: Agent orchestration and daily briefings (In Progress)
- [ ] **Phase 4**: Streamlit frontend with interactive interface
- [ ] **Phase 5**: Advanced features (alerts, custom analysis, API endpoints)

## 📈 Use Cases

- **Individual Investors**: Daily market intelligence and trend analysis
- **Financial Advisors**: Client briefing materials and market insights  
- **Researchers**: Economic data correlation and news impact studies
- **Traders**: Sentiment-driven market analysis and correlation discovery

---

*This platform transforms fragmented financial information into coherent, actionable intelligence through the power of AI orchestration and multi-source data synthesis.*
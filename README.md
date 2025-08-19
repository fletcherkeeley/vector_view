# AI Financial Intelligence Platform

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

### Vector Database (Semantic Data)
- **news_embeddings**: Semantic embeddings of news articles
- **topic_clusters**: Related news topic groupings
- **sentiment_vectors**: Sentiment-aware news representations

### Bridge Tables
- **news_topic_mapping**: Links news categories to economic indicators
- **correlation_triggers**: News events that correlate with market moves

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- PostgreSQL 15+
- Vector database (Pinecone/Chroma)
- API keys: FRED, News API

### Quick Setup
```bash
# 1. Clone and setup environment
git clone <repository>
cd ai-financial-intelligence
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

# 2. Start database
docker-compose up -d postgres

# 3. Initialize database schema
python database/setup_database.py

# 4. Configure API credentials
cp .env.example .env
# Edit .env with your API keys

# 5. Run initial data ingestion
python ingestion/run_initial_sync.py

# 6. Start the application
streamlit run app/main.py
```

## 📁 Project Structure

```
ai-financial-intelligence/
├── README.md
├── requirements.txt
├── docker-compose.yml
├── .env.example
│
├── database/
│   ├── unified_database_setup.py    # Complete schema definition
│   └── setup_database.py            # User-friendly setup script
│
├── ingestion/
│   ├── fred_client.py               # FRED API integration
│   ├── yahoo_client.py              # Yahoo Finance integration
│   ├── news_client.py               # News API integration
│   ├── vector_db_client.py          # Vector database operations
│   └── orchestrator.py              # Data ingestion coordination
│
├── agents/
│   ├── base_agent.py                # Base agent class
│   ├── orchestration_agent.py       # Master orchestrator
│   ├── economic_agent.py            # Economic analysis specialist
│   ├── sentiment_agent.py           # News sentiment specialist
│   ├── market_agent.py              # Market analysis specialist
│   └── synthesis_agent.py           # Daily briefing generator
│
├── bridge/
│   ├── correlation_engine.py        # Economic-news correlation
│   ├── topic_mapper.py              # News topic classification
│   └── context_builder.py           # Cross-dataset context creation
│
├── app/
│   ├── main.py                      # Streamlit main application
│   ├── components/
│   │   ├── briefing_display.py      # Daily briefing UI
│   │   ├── chatbot_interface.py     # Interactive chat component
│   │   └── data_visualizations.py   # Charts and graphs
│   └── utils/
│       ├── session_manager.py       # User session handling
│       └── query_processor.py       # Natural language query processing
│
├── config/
│   ├── settings.py                  # Application configuration
│   ├── api_config.py                # API client configurations
│   └── agent_config.py              # AI agent configurations
│
└── tests/
    ├── test_database.py
    ├── test_ingestion.py
    ├── test_agents.py
    └── test_integration.py
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

- [ ] **Phase 1**: Core data ingestion and database setup
- [ ] **Phase 2**: Basic AI agents and correlation engine
- [ ] **Phase 3**: Streamlit frontend with daily briefings
- [ ] **Phase 4**: Interactive chatbot and deep-dive queries
- [ ] **Phase 5**: Advanced features (alerts, custom analysis, API endpoints)

## 📈 Use Cases

- **Individual Investors**: Daily market intelligence and trend analysis
- **Financial Advisors**: Client briefing materials and market insights  
- **Researchers**: Economic data correlation and news impact studies
- **Traders**: Sentiment-driven market analysis and correlation discovery

---

*This platform transforms fragmented financial information into coherent, actionable intelligence through the power of AI orchestration and multi-source data synthesis.*
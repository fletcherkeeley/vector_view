# Data Sources Documentation

This document outlines all data sources integrated into the AI Financial Intelligence Platform.

## 📊 FRED Economic Indicators

### Interest Rates & Monetary Policy
- **FEDFUNDS** - Federal Funds Rate
- **DGS10** - 10-Year Treasury Constant Maturity Rate  
- **DGS2** - 2-Year Treasury Constant Maturity Rate
- **T10Y3M** - 10-Year Treasury Minus 3-Month Treasury (yield curve)
- **TEDRATE** - TED Spread (Treasury-EuroDollar spread)

### Employment & Labor Market
- **UNRATE** - Unemployment Rate
- **PAYEMS** - All Employees, Total Nonfarm Payrolls
- **ICSA** - Initial Claims for Unemployment Insurance
- **AHETPI** - Average Hourly Earnings of Total Private Industries

### Inflation Indicators
- **CPIAUCSL** - Consumer Price Index for All Urban Consumers

### Economic Growth & Housing
- **GDP** - Gross Domestic Product
- **PERMIT** - New Private Housing Units Authorized by Building Permits
- **HOUST** - New Privately-Owned Housing Units Started

### Market Stress & Volatility
- **VIXCLS** - CBOE Volatility Index

### Commodities (FRED Series)
- **GOLDAMGBD228NLBM** - Gold Fixing Price (London)

## 📈 Yahoo Finance Market Data

### Major Market Indices
- **SPY** - SPDR S&P 500 ETF Trust
- **QQQ** - Invesco QQQ Trust (NASDAQ-100)
- **VTI** - Vanguard Total Stock Market ETF
- **IWM** - iShares Russell 2000 ETF (Small Cap)

### Sector ETFs (SPDR Select Sector Funds)
- **XLF** - Financial Select Sector SPDR Fund
- **XLY** - Consumer Discretionary Select Sector SPDR Fund
- **XLP** - Consumer Staples Select Sector SPDR Fund
- **XLE** - Energy Select Sector SPDR Fund
- **XLI** - Industrial Select Sector SPDR Fund

## 📰 News API Categories

### Economic News Categories
- **Federal Reserve** - Monetary policy, interest rate decisions, Fed communications
- **Employment** - Jobs reports, unemployment data, labor market trends
- **Inflation** - CPI reports, price trends, inflation expectations
- **GDP Growth** - Economic growth reports, recession indicators
- **Corporate Earnings** - Quarterly earnings, corporate guidance
- **Geopolitical** - Trade wars, political events, international relations
- **Market Volatility** - Market crashes, corrections, volatility spikes
- **Sector Specific** - Industry-specific news and trends
- **Commodity Markets** - Oil, gold, agricultural commodity news
- **International Trade** - Trade balance, tariffs, global trade flows

## 🔗 Data Relationships

### High-Priority Correlations
Based on the schema design, these are the primary relationships tracked:

**Federal Reserve Policy** ↔ **Financial Sector (XLF)**
- FEDFUNDS, DGS10, DGS2 vs XLF performance

**Employment Data** ↔ **Consumer Sectors (XLY, XLP)**
- UNRATE, PAYEMS, AHETPI vs consumer spending ETFs

**Inflation Indicators** ↔ **Commodity ETFs (XLE)**
- CPIAUCSL vs energy and commodity prices

**Market Volatility** ↔ **Broad Market Indices**
- VIXCLS vs SPY, QQQ performance

**Economic Growth** ↔ **Industrial Sector (XLI)**
- GDP, housing data vs industrial performance

## 📅 Data Frequencies

### FRED Economic Data
- **Daily**: FEDFUNDS, DGS10, DGS2, T10Y3M, TEDRATE, VIXCLS
- **Monthly**: UNRATE, PAYEMS, CPIAUCSL, PERMIT, HOUST
- **Weekly**: ICSA
- **Quarterly**: GDP

### Yahoo Finance Market Data
- **Real-time/Daily**: All ETF and index data (OHLCV)
- **Intraday**: Available for detailed analysis

### News Data
- **Real-time**: Continuous monitoring of news feeds
- **Categorized**: Automatically tagged by economic relevance

## 🎯 Coverage Scope

### Time Horizon
- **Historical**: 50+ years for major FRED indicators
- **Market Data**: Yahoo Finance historical coverage (varies by symbol)
- **News**: Real-time forward-looking with configurable historical depth

### Geographic Coverage
- **Primary**: United States economic indicators and markets
- **Secondary**: Global events affecting US markets (via news)

## 📊 Data Quality Expectations

### FRED Data
- **Reliability**: Government source, high quality
- **Revisions**: Some series subject to historical revisions
- **Frequency**: Consistent release schedules

### Yahoo Finance
- **Reliability**: Real-time market data, industry standard
- **Coverage**: Comprehensive US equity and ETF data
- **Adjustments**: Split and dividend adjusted pricing

### News Data
- **Volume**: High-volume real-time feed
- **Quality**: Varies by source, requires filtering
- **Timeliness**: Near real-time for market-moving events

---

*This documentation serves as the master reference for all data sources integrated into the platform. Update this file when adding new data sources or modifying existing integrations.*
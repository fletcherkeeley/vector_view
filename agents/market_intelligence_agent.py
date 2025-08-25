"""
Market Intelligence Agent for Vector View Financial Intelligence Platform

Analyzes real-time market impact from news events by cross-referencing news sentiment
with market data (FRED economic indicators + Yahoo Finance). Provides market reaction
predictions, volatility forecasts, and correlation analysis between news and market movements.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentType, AgentResponse, AgentContext
from .ai_service import OllamaService

logger = logging.getLogger(__name__)


@dataclass
class MarketImpactAnalysis:
    """Market impact analysis result"""
    news_sentiment_score: float
    market_correlation: float
    volatility_forecast: float
    price_impact_prediction: Dict[str, float]
    sector_impact: Dict[str, float]
    confidence_score: float
    supporting_evidence: List[str]


@dataclass
class NewsMarketCorrelation:
    """News-market correlation data"""
    correlation_coefficient: float
    statistical_significance: float
    time_lag_hours: int
    sample_size: int
    correlation_strength: str  # "strong", "moderate", "weak"


class MarketIntelligenceAgent(BaseAgent):
    """
    Market Intelligence Agent specializing in news-market impact analysis.
    
    Capabilities:
    - Real-time market impact assessment from news events
    - Cross-correlation analysis between news sentiment and market movements
    - Volatility forecasting based on news flow
    - Sector-specific impact analysis
    - Market reaction prediction modeling
    """
    
    def __init__(self, db_connection=None, chroma_client=None, ai_service: OllamaService = None, database_url: str = "postgresql://postgres:fred_password@localhost:5432/postgres"):
        super().__init__(
            agent_type=AgentType.MARKET_INTELLIGENCE,
            database_url=database_url
        )
        self.db_connection = db_connection
        self.chroma_client = chroma_client
        self.ai_service = ai_service or OllamaService()
        
        # Market sectors for impact analysis
        self.market_sectors = {
            'technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
            'financial': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
            'healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK'],
            'energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
            'consumer': ['WMT', 'PG', 'KO', 'PEP', 'COST']
        }
        
        # Key market indicators to monitor (using actual series_ids from database)
        self.key_indicators = [
            'SPY',     # S&P 500
            'QQQ',     # NASDAQ
            'TLT',     # 20+ Year Treasury Bond
            'GLD',     # Gold
            'VIXCLS'   # Volatility Index (actual series_id)
        ]

    async def analyze(self, context: AgentContext) -> AgentResponse:
        """
        Analyze market impact from news events and provide correlation insights.
        """
        try:
            start_time = datetime.now()
            
            # Get recent news articles and sentiment
            news_data = await self._get_recent_news_sentiment(context)
            
            # Get corresponding market data
            market_data = await self._get_market_data(context)
            
            # Perform correlation analysis
            correlation_analysis = await self._analyze_news_market_correlation(
                news_data, market_data, context
            )
            
            # Generate market impact assessment
            impact_analysis = await self._assess_market_impact(
                news_data, market_data, correlation_analysis, context
            )
            
            # Generate AI-powered insights using the proper AI service method
            ai_analysis = await self.ai_service.analyze_market_data(
                market_data={
                    "impact_analysis": {
                        "sentiment_score": impact_analysis.news_sentiment_score,
                        "market_correlation": impact_analysis.market_correlation,
                        "volatility_forecast": impact_analysis.volatility_forecast,
                        "sector_impact": impact_analysis.sector_impact
                    },
                    "correlation_analysis": {
                        "correlation_coefficient": correlation_analysis.correlation_coefficient,
                        "statistical_significance": correlation_analysis.statistical_significance,
                        "correlation_strength": correlation_analysis.correlation_strength,
                        "sample_size": correlation_analysis.sample_size
                    }
                },
                technical_indicators={},
                context=f"Query: {context.query}. Timeframe: {context.timeframe}. Market Intelligence Analysis."
            )
            
            # Extract insights from AI analysis (already cleaned by ai_service)
            ai_insights = ai_analysis.content
            
            # Calculate execution metrics
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Determine confidence based on data quality and correlation strength
            confidence = self._calculate_confidence(correlation_analysis, len(news_data))
            
            # Generate cross-agent signals
            signals = self._generate_market_signals(impact_analysis, correlation_analysis)
            
            return AgentResponse(
                agent_type=self.agent_type,
                confidence=confidence,
                confidence_level=self._get_confidence_level(confidence),
                analysis={
                    'market_impact_analysis': impact_analysis.__dict__,
                    'correlation_analysis': correlation_analysis.__dict__,
                    'news_articles_analyzed': len(news_data),
                    'market_data_points': len(market_data),
                    'key_correlations': self._extract_key_correlations(correlation_analysis)
                },
                insights=[ai_insights],
                key_metrics={
                    'sentiment_score': impact_analysis.news_sentiment_score,
                    'market_correlation': impact_analysis.market_correlation,
                    'volatility_forecast': impact_analysis.volatility_forecast
                },
                data_sources_used=['time_series_observations', 'news_articles'],
                timeframe_analyzed=context.timeframe,
                execution_time_ms=execution_time,
                signals_for_other_agents=signals,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Market Intelligence Agent analysis failed: {str(e)}")
            return AgentResponse(
                agent_type=self.agent_type,
                confidence=0.0,
                confidence_level=self._get_confidence_level(0.0),
                analysis={'error': str(e)},
                insights=[f"Market analysis encountered an error: {str(e)}"],
                key_metrics={},
                data_sources_used=[],
                timeframe_analyzed=context.timeframe,
                execution_time_ms=0,
                signals_for_other_agents={},
                timestamp=datetime.now()
            )

    async def _get_recent_news_sentiment(self, context: AgentContext) -> List[Dict]:
        """Get sentiment data from news sentiment agent output or fallback to basic retrieval"""
        try:
            # First, try to get sentiment data from agent context (preferred method)
            if hasattr(context, 'agent_outputs') and context.agent_outputs:
                sentiment_output = context.agent_outputs.get('news_sentiment')
                if sentiment_output and hasattr(sentiment_output, 'analysis'):
                    sentiment_analysis = sentiment_output.analysis
                    
                    # Extract sentiment data from news sentiment agent
                    overall_sentiment = sentiment_analysis.get('sentiment_analysis', {}).get('overall_sentiment', 0.0)
                    articles_analyzed = sentiment_analysis.get('articles_analyzed', 0)
                    
                    # Create synthetic news data with sentiment from sentiment agent
                    news_articles = []
                    for i in range(min(articles_analyzed, 10)):  # Use up to 10 articles
                        news_articles.append({
                            'content': f'Financial news article {i+1}',
                            'sentiment_score': overall_sentiment,
                            'timestamp': datetime.now(),
                            'source': 'news_sentiment_agent',
                            'title': f'Market news {i+1}',
                            'relevance_score': 0.8
                        })
                    
                    logger.info(f"Using sentiment data from news sentiment agent: {overall_sentiment:.3f} sentiment, {len(news_articles)} articles")
                    return news_articles
            
            # Fallback: Get news articles directly from ChromaDB
            if not self.chroma_client:
                try:
                    import chromadb
                    self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
                except Exception as e:
                    logger.warning(f"ChromaDB not available: {str(e)}")
                    return []
            
            collection = self.chroma_client.get_collection("financial_news")
            results = collection.query(
                query_texts=[context.query] if context.query else ["market news financial"],
                n_results=20
            )
            
            news_articles = []
            if results and results.get('documents') and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results.get('metadatas') else {}
                    
                    # Use basic sentiment calculation as fallback
                    sentiment_score = metadata.get('sentiment_score', 0.0)
                    if sentiment_score == 0.0 and doc:
                        sentiment_score = self._calculate_basic_sentiment(doc)
                    
                    news_articles.append({
                        'content': doc,
                        'sentiment_score': float(sentiment_score),
                        'timestamp': metadata.get('published_at') or metadata.get('timestamp'),
                        'source': metadata.get('source_name', 'unknown'),
                        'title': metadata.get('title', ''),
                        'relevance_score': 1.0 - (results['distances'][0][i] if results.get('distances') else 0.0)
                    })
            
            logger.info(f"Using fallback ChromaDB sentiment data: {len(news_articles)} articles")
            return news_articles
            
        except Exception as e:
            logger.error(f"Failed to retrieve news sentiment: {str(e)}")
            return []

    async def _get_market_data(self, context: AgentContext) -> pd.DataFrame:
        """Get market data from PostgreSQL database"""
        try:
            from sqlalchemy import create_engine, text
            
            # Use database_url to create connection
            if not hasattr(self, 'engine'):
                self.engine = create_engine(self.database_url)
            
            # Determine timeframe for market data
            hours_back = self._parse_timeframe_hours(context.timeframe)
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            # Query for market data using the actual schema (data_series + time_series_observations)
            symbols_str = "', '".join(self.key_indicators)
            
            query = text(f"""
            SELECT 
                ds.series_id as symbol,
                tso.observation_date as date,
                tso.value as close_price
            FROM data_series ds
            JOIN time_series_observations tso ON ds.series_id = tso.series_id
            WHERE ds.series_id IN ('{symbols_str}')
            AND tso.observation_date >= :cutoff_time
            AND tso.value IS NOT NULL
            ORDER BY ds.series_id, tso.observation_date DESC
            """)
            
            with self.engine.connect() as conn:
                result = conn.execute(query, {"cutoff_time": cutoff_time})
                data = result.fetchall()
            
            if not data:
                return pd.DataFrame()
            
            market_df = pd.DataFrame(data, columns=['symbol', 'date', 'close_price'])
            return market_df
            
        except Exception as e:
            logger.error(f"Failed to retrieve market data: {str(e)}")
            return pd.DataFrame()

    async def _analyze_news_market_correlation(
        self, 
        news_data: List[Dict], 
        market_data: pd.DataFrame,
        context: AgentContext
    ) -> NewsMarketCorrelation:
        """Analyze correlation between news sentiment and market movements"""
        try:
            if not news_data or market_data.empty:
                return NewsMarketCorrelation(
                    correlation_coefficient=0.0,
                    statistical_significance=0.0,
                    time_lag_hours=0,
                    sample_size=0,
                    correlation_strength="none"
                )
            
            # Aggregate news sentiment by hour
            news_df = pd.DataFrame(news_data)
            news_df['timestamp'] = pd.to_datetime(news_df['timestamp'])
            news_df['hour'] = news_df['timestamp'].dt.floor('H')
            
            hourly_sentiment = news_df.groupby('hour')['sentiment_score'].mean().reset_index()
            
            # Calculate market returns by hour for SPY (primary indicator)
            spy_data = market_data[market_data['symbol'] == 'SPY'].copy()
            if spy_data.empty:
                # Fallback to any available market data
                spy_data = market_data.groupby(['symbol', 'date'])['close_price'].first().reset_index()
                spy_data = spy_data.groupby('date')['close_price'].first().reset_index()
            
            spy_data['date'] = pd.to_datetime(spy_data['date'])
            spy_data['hour'] = spy_data['date'].dt.floor('H')
            # Convert Decimal to float to avoid type errors
            spy_data['close_price'] = spy_data['close_price'].astype(float)
            spy_data['returns'] = spy_data['close_price'].pct_change()
            
            # Merge sentiment and market data
            merged_data = pd.merge(
                hourly_sentiment, 
                spy_data[['hour', 'returns']], 
                on='hour', 
                how='inner'
            )
            
            if len(merged_data) < 3:
                return NewsMarketCorrelation(
                    correlation_coefficient=0.0,
                    statistical_significance=0.0,
                    time_lag_hours=0,
                    sample_size=len(merged_data),
                    correlation_strength="insufficient_data"
                )
            
            # Calculate correlation
            correlation = merged_data['sentiment_score'].corr(merged_data['returns'])
            correlation = correlation if not np.isnan(correlation) else 0.0
            
            # Determine correlation strength
            abs_corr = abs(correlation)
            if abs_corr >= 0.7:
                strength = "strong"
            elif abs_corr >= 0.3:
                strength = "moderate"
            else:
                strength = "weak"
            
            return NewsMarketCorrelation(
                correlation_coefficient=correlation,
                statistical_significance=abs_corr,
                time_lag_hours=1,  # Simplified - could implement lag analysis
                sample_size=len(merged_data),
                correlation_strength=strength
            )
            
        except Exception as e:
            logger.error(f"Correlation analysis failed: {str(e)}")
            return NewsMarketCorrelation(
                correlation_coefficient=0.0,
                statistical_significance=0.0,
                time_lag_hours=0,
                sample_size=0,
                correlation_strength="error"
            )

    async def _assess_market_impact(
        self,
        news_data: List[Dict],
        market_data: pd.DataFrame,
        correlation_analysis: NewsMarketCorrelation,
        context: AgentContext
    ) -> MarketImpactAnalysis:
        """Assess overall market impact from news events"""
        try:
            # Calculate aggregate news sentiment
            if news_data:
                sentiment_scores = [article['sentiment_score'] for article in news_data]
                avg_sentiment = np.mean(sentiment_scores)
                sentiment_volatility = np.std(sentiment_scores)
            else:
                avg_sentiment = 0.0
                sentiment_volatility = 0.0
            
            # Calculate market volatility
            if not market_data.empty:
                spy_data = market_data[market_data['symbol'] == 'SPY']
                if not spy_data.empty:
                    # Convert Decimal to float to avoid type errors
                    spy_data = spy_data.copy()
                    spy_data['close_price'] = spy_data['close_price'].astype(float)
                    returns = spy_data['close_price'].pct_change().dropna()
                    market_volatility = returns.std() * np.sqrt(252)  # Annualized
                else:
                    market_volatility = 0.0
            else:
                market_volatility = 0.0
            
            # Predict price impact based on sentiment and correlation
            price_impact = {}
            for indicator in self.key_indicators:
                base_impact = avg_sentiment * correlation_analysis.correlation_coefficient
                volatility_adjustment = sentiment_volatility * 0.1
                price_impact[indicator] = base_impact + volatility_adjustment
            
            # Assess sector-specific impact
            sector_impact = {}
            for sector, symbols in self.market_sectors.items():
                # Simplified sector impact based on overall sentiment
                sector_sentiment_multiplier = {
                    'technology': 1.2,  # More sensitive to sentiment
                    'financial': 1.1,
                    'healthcare': 0.8,  # Less sensitive
                    'energy': 1.0,
                    'consumer': 0.9
                }
                multiplier = sector_sentiment_multiplier.get(sector, 1.0)
                sector_impact[sector] = avg_sentiment * multiplier * correlation_analysis.correlation_coefficient
            
            # Generate supporting evidence
            evidence = []
            if correlation_analysis.correlation_strength == "strong":
                evidence.append(f"Strong correlation ({correlation_analysis.correlation_coefficient:.3f}) between news sentiment and market movements")
            if abs(avg_sentiment) > 0.5:
                sentiment_direction = "positive" if avg_sentiment > 0 else "negative"
                evidence.append(f"Significant {sentiment_direction} news sentiment ({avg_sentiment:.3f})")
            if market_volatility > 0.2:
                evidence.append(f"Elevated market volatility detected ({market_volatility:.3f})")
            
            # Calculate confidence based on data quality
            confidence = min(1.0, (
                correlation_analysis.statistical_significance * 0.4 +
                min(len(news_data) / 10, 1.0) * 0.3 +
                min(len(market_data) / 100, 1.0) * 0.3
            ))
            
            return MarketImpactAnalysis(
                news_sentiment_score=avg_sentiment,
                market_correlation=correlation_analysis.correlation_coefficient,
                volatility_forecast=market_volatility + sentiment_volatility * 0.1,
                price_impact_prediction=price_impact,
                sector_impact=sector_impact,
                confidence_score=confidence,
                supporting_evidence=evidence
            )
            
        except Exception as e:
            logger.error(f"Market impact assessment failed: {str(e)}")
            return MarketImpactAnalysis(
                news_sentiment_score=0.0,
                market_correlation=0.0,
                volatility_forecast=0.0,
                price_impact_prediction={},
                sector_impact={},
                confidence_score=0.0,
                supporting_evidence=[f"Analysis error: {str(e)}"]
            )

    async def _generate_market_insights(
        self,
        impact_analysis: MarketImpactAnalysis,
        correlation_analysis: NewsMarketCorrelation,
        context: AgentContext
    ) -> str:
        """Generate AI-powered market insights and analysis"""
        try:
            # Prepare analysis data for AI
            analysis_summary = {
                'sentiment_score': impact_analysis.news_sentiment_score,
                'market_correlation': impact_analysis.market_correlation,
                'volatility_forecast': impact_analysis.volatility_forecast,
                'correlation_strength': correlation_analysis.correlation_strength,
                'sample_size': correlation_analysis.sample_size,
                'top_sector_impacts': dict(sorted(
                    impact_analysis.sector_impact.items(), 
                    key=lambda x: abs(x[1]), 
                    reverse=True
                )[:3]),
                'supporting_evidence': impact_analysis.supporting_evidence
            }
            
            prompt = f"""
            As a market intelligence analyst, provide insights on the current market impact from news events.
            
            Analysis Data:
            - News Sentiment Score: {analysis_summary['sentiment_score']:.3f} (-1 to +1 scale)
            - News-Market Correlation: {analysis_summary['market_correlation']:.3f}
            - Correlation Strength: {analysis_summary['correlation_strength']}
            - Volatility Forecast: {analysis_summary['volatility_forecast']:.3f}
            - Sample Size: {analysis_summary['sample_size']} data points
            
            Top Sector Impacts:
            {chr(10).join([f"- {sector}: {impact:.3f}" for sector, impact in analysis_summary['top_sector_impacts'].items()])}
            
            Supporting Evidence:
            {chr(10).join([f"- {evidence}" for evidence in analysis_summary['supporting_evidence']])}
            
            Provide a concise market intelligence analysis covering:
            1. Current market sentiment and its reliability
            2. Expected market impact and direction
            3. Sector-specific implications
            4. Risk factors and volatility outlook
            5. Key monitoring points for traders/investors
            
            Keep the analysis professional, data-driven, and actionable.
            """
            
            response = await self.ai_service.generate_response(
                prompt=prompt,
                context=f"Market Intelligence Analysis - {context.query_type}",
                max_tokens=800
            )
            
            # Clean response to remove think tags
            import re
            cleaned_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
            cleaned_response = re.sub(r'^<think>.*', '', cleaned_response, flags=re.DOTALL | re.MULTILINE)
            cleaned_response = cleaned_response.strip()
            
            return cleaned_response if cleaned_response else response.strip()
            
        except Exception as e:
            logger.error(f"AI insight generation failed: {str(e)}")
            return f"Market Intelligence Analysis: Unable to generate detailed insights due to technical issues. Raw sentiment score: {impact_analysis.news_sentiment_score:.3f}, Market correlation: {impact_analysis.market_correlation:.3f}"

    def _generate_market_signals(
        self, 
        impact_analysis: MarketImpactAnalysis,
        correlation_analysis: NewsMarketCorrelation
    ) -> Dict[str, Any]:
        """Generate cross-agent signals for other agents"""
        return {
            'market_sentiment': 'bullish' if impact_analysis.news_sentiment_score > 0.2 else 'bearish' if impact_analysis.news_sentiment_score < -0.2 else 'neutral',
            'volatility_regime': 'high' if impact_analysis.volatility_forecast > 0.25 else 'low',
            'news_market_correlation': correlation_analysis.correlation_strength,
            'market_stress_level': min(1.0, abs(impact_analysis.news_sentiment_score) + impact_analysis.volatility_forecast),
            'sector_rotation_signal': max(impact_analysis.sector_impact.items(), key=lambda x: abs(x[1]))[0] if impact_analysis.sector_impact else 'none'
        }

    def _calculate_confidence(self, correlation_analysis: NewsMarketCorrelation, news_count: int) -> float:
        """Calculate confidence score based on data quality and correlation strength"""
        # Base confidence from correlation strength
        correlation_confidence = correlation_analysis.statistical_significance
        
        # Data quality confidence
        sample_confidence = min(1.0, correlation_analysis.sample_size / 20)
        news_confidence = min(1.0, news_count / 10)
        
        # Combined confidence
        overall_confidence = (
            correlation_confidence * 0.5 +
            sample_confidence * 0.3 +
            news_confidence * 0.2
        )
        
        return max(0.0, min(1.0, overall_confidence))

    def _extract_key_correlations(self, correlation_analysis: NewsMarketCorrelation) -> Dict[str, Any]:
        """Extract key correlation insights for reporting"""
        return {
            'correlation_coefficient': correlation_analysis.correlation_coefficient,
            'strength_assessment': correlation_analysis.correlation_strength,
            'statistical_significance': correlation_analysis.statistical_significance,
            'data_reliability': 'high' if correlation_analysis.sample_size >= 20 else 'medium' if correlation_analysis.sample_size >= 10 else 'low'
        }

    def _parse_timeframe_hours(self, timeframe: str) -> int:
        """Convert timeframe string to hours"""
        timeframe_map = {
            '1h': 1,
            '4h': 4,
            '1d': 24,
            '1w': 168,
            '1m': 720,
            '3m': 2160,
            '1y': 8760
        }
        return timeframe_map.get(timeframe, 24)  # Default to 1 day

    def _calculate_basic_sentiment(self, text: str) -> float:
        """Calculate basic sentiment score from text using keyword analysis"""
        try:
            if not text:
                return 0.0
            
            text_lower = text.lower()
            
            # Positive keywords
            positive_words = ['gain', 'rise', 'up', 'surge', 'rally', 'bull', 'positive', 'strong', 'growth', 'increase']
            # Negative keywords  
            negative_words = ['fall', 'drop', 'down', 'decline', 'bear', 'negative', 'weak', 'loss', 'decrease', 'crash']
            
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            # Calculate sentiment score (-1 to +1)
            total_sentiment_words = positive_count + negative_count
            if total_sentiment_words == 0:
                return 0.0
            
            sentiment = (positive_count - negative_count) / total_sentiment_words
            return max(-1.0, min(1.0, sentiment))  # Clamp between -1 and 1
            
        except Exception as e:
            logger.warning(f"Basic sentiment calculation failed: {str(e)}")
            return 0.0

    def get_required_data_sources(self, context: AgentContext) -> List[str]:
        """Return list of data sources required for market intelligence analysis"""
        return [
            'time_series_observations',  # Market data from PostgreSQL
            'news_articles',             # News articles from ChromaDB
            'market_indicators',         # Key market indicators (SPY, VIX, etc.)
            'sector_data'               # Sector-specific market data
        ]

    def _get_confidence_level(self, confidence: float):
        """Convert confidence score to ConfidenceLevel enum"""
        from .base_agent import ConfidenceLevel
        if confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

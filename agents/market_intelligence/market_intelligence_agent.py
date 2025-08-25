"""
Market Intelligence Agent for Vector View Financial Intelligence Platform

Refactored agent using separated concerns architecture matching the economic and news sentiment agent structure.
Analyzes real-time market impact from news events by cross-referencing news sentiment
with market data (FRED economic indicators + Yahoo Finance). Provides market reaction
predictions, volatility forecasts, and correlation analysis between news and market movements.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from ..base_agent import BaseAgent, AgentType, AgentResponse, AgentContext, StandardizedSignals
from ..ai_service import OllamaService
from .market_data_handler import MarketDataHandler
from .market_context_builder import MarketContextBuilder

logger = logging.getLogger(__name__)


class MarketIntelligenceAgent(BaseAgent):
    """
    Market Intelligence Agent specializing in news-market impact analysis.
    
    Capabilities:
    - Real-time market impact assessment from news events
    - Cross-correlation analysis between news sentiment and market movements
    - Volatility forecasting based on news flow
    - Sector-specific impact analysis
    - Market reaction prediction modeling
    
    Architecture:
    - Uses MarketDataHandler for all data access operations
    - Uses MarketContextBuilder for analysis and context building
    - Focuses on orchestration and response generation
    """
    
    def __init__(
        self, 
        db_connection=None, 
        chroma_client=None, 
        ai_service: OllamaService = None, 
        database_url: str = "postgresql://postgres:fred_password@localhost:5432/postgres"
    ):
        super().__init__(
            agent_type=AgentType.MARKET_INTELLIGENCE,
            database_url=database_url
        )
        
        # Initialize AI service
        self.ai_service = ai_service or OllamaService()
        
        # Initialize separated components
        self.data_handler = MarketDataHandler(database_url=database_url)
        self.context_builder = MarketContextBuilder()
        
        # Legacy parameters for compatibility
        self.db_connection = db_connection
        self.chroma_client = chroma_client

    async def analyze(self, context: AgentContext) -> AgentResponse:
        """
        Analyze market impact from news events and provide correlation insights.
        """
        try:
            start_time = datetime.now()
            
            # Get recent news articles and sentiment from context or fallback
            news_data = await self._get_recent_news_sentiment(context)
            logger.info(f"Retrieved {len(news_data)} news articles for market analysis")
            
            # Get corresponding market data using data handler
            market_data = await self.data_handler.get_market_data(
                indicators=['SPY', 'QQQ', 'TLT', 'GLD'],
                start_date=context.date_range['start'],
                end_date=context.date_range['end']
            )
            
            # Perform correlation analysis using context builder
            correlation_analysis = await self.context_builder.analyze_news_market_correlation(
                news_data, market_data
            )
            
            # Generate market impact assessment using context builder
            impact_analysis = await self.context_builder.assess_market_impact(
                news_data, market_data, correlation_analysis
            )
            
            # Build comprehensive context for AI analysis
            analysis_context = self.context_builder.build_analysis_context(
                news_data, market_data, impact_analysis, correlation_analysis
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
            
            # Determine confidence using context builder
            confidence = self.context_builder.calculate_confidence(
                correlation_analysis, len(news_data), len(market_data)
            )
            
            # Generate cross-agent signals using context builder (legacy format)
            signals = self.context_builder.generate_cross_agent_signals(
                impact_analysis, correlation_analysis
            )
            
            # Generate standardized signals (new format)
            standardized_signals = StandardizedSignals()
            standardized_signals.market_correlation = impact_analysis.market_correlation
            standardized_signals.volatility_forecast = impact_analysis.volatility_forecast
            if hasattr(impact_analysis, 'sector_impact') and impact_analysis.sector_impact:
                standardized_signals.sector_impact = impact_analysis.sector_impact
            
            # Get data quality metrics
            data_quality = self.data_handler.get_data_quality_metrics(market_data)
            
            response = AgentResponse(
                agent_type=self.agent_type,
                confidence=confidence,
                confidence_level=self._get_confidence_level(confidence),
                analysis={
                    'market_impact_analysis': impact_analysis.__dict__,
                    'correlation_analysis': correlation_analysis.__dict__,
                    'news_articles_analyzed': len(news_data),
                    'market_data_points': len(market_data),
                    'key_correlations': self._extract_key_correlations(correlation_analysis),
                    'analysis_context': analysis_context,
                    'data_quality': data_quality
                },
                insights=[ai_insights],
                key_metrics={
                    'sentiment_score': impact_analysis.news_sentiment_score,
                    'market_correlation': impact_analysis.market_correlation,
                    'volatility_forecast': impact_analysis.volatility_forecast,
                    'correlation_strength': correlation_analysis.correlation_strength,
                    'confidence_score': impact_analysis.confidence_score
                },
                data_sources_used=['time_series_observations', 'news_articles', 'market_data'],
                timeframe_analyzed=context.timeframe,
                execution_time_ms=execution_time,
                signals_for_other_agents=signals,
                standardized_signals=standardized_signals,
                timestamp=datetime.now()
            )
            
            return response
            
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

    def _extract_key_correlations(self, correlation_analysis) -> Dict[str, Any]:
        """Extract key correlation insights for reporting"""
        return {
            'correlation_coefficient': correlation_analysis.correlation_coefficient,
            'strength_assessment': correlation_analysis.correlation_strength,
            'statistical_significance': correlation_analysis.statistical_significance,
            'data_reliability': 'high' if correlation_analysis.sample_size >= 20 else 'medium' if correlation_analysis.sample_size >= 10 else 'low'
        }

    async def get_market_health(self) -> Dict[str, Any]:
        """
        Get health status of market data connections.
        
        Returns:
            Dictionary with market data health metrics
        """
        try:
            is_healthy = self.data_handler.validate_connection()
            
            # Get sample market data to test functionality
            sample_data = await self.data_handler.get_market_data(timeframe="1d")
            data_quality = self.data_handler.get_data_quality_metrics(sample_data)
            
            return {
                'connection_healthy': is_healthy,
                'data_quality': data_quality,
                'data_sources_available': ['postgresql', 'fred', 'yahoo_finance']
            }
        except Exception as e:
            logger.error(f"Market health check failed: {str(e)}")
            return {
                'connection_healthy': False,
                'error': str(e),
                'data_sources_available': []
            }

    async def get_sector_analysis(self, sector: str, timeframe: str = "1d") -> Dict[str, Any]:
        """
        Get sector-specific market analysis.
        
        Args:
            sector: Sector name (technology, financial, etc.)
            timeframe: Time window for analysis
            
        Returns:
            Dictionary with sector analysis results
        """
        try:
            sector_data = await self.data_handler.get_sector_data(sector, timeframe)
            
            if sector_data.empty:
                return {'error': f'No data available for sector: {sector}'}
            
            # Calculate sector metrics
            returns = self.data_handler.calculate_returns(sector_data)
            volatility = self.data_handler.calculate_volatility(sector_data)
            
            return {
                'sector': sector,
                'data_points': len(sector_data),
                'symbols_analyzed': list(sector_data.columns),
                'average_return': returns.mean().mean() if not returns.empty else 0.0,
                'average_volatility': volatility.mean().mean() if not volatility.empty else 0.0,
                'timeframe': timeframe
            }
            
        except Exception as e:
            logger.error(f"Sector analysis failed: {str(e)}")
            return {'error': str(e)}

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
        from ..base_agent import ConfidenceLevel
        if confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

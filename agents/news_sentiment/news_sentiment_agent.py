"""
News Sentiment & Narrative Agent for Vector View Financial Intelligence Platform

Refactored agent using separated concerns architecture matching the economic agent structure.
Advanced NLP analysis including sentiment scoring, entity extraction, bias detection,
emotional tone analysis, and narrative trend tracking across time periods.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from ..base_agent import BaseAgent, AgentType, AgentResponse, AgentContext, StandardizedSignals
from ..ai_service import OllamaService
from .news_sentiment_data_handler import NewsSentimentDataHandler
from .news_sentiment_context_builder import NewsSentimentContextBuilder

logger = logging.getLogger(__name__)


class NewsSentimentAgent(BaseAgent):
    """
    News Sentiment & Narrative Agent specializing in advanced NLP analysis.
    
    Capabilities:
    - Multi-dimensional sentiment analysis beyond basic positive/negative
    - Entity extraction (companies, people, locations, events)
    - Bias detection and source credibility assessment
    - Emotional tone analysis (fear, greed, uncertainty, confidence)
    - Narrative trend tracking and theme evolution
    - Cross-source consensus analysis
    
    Architecture:
    - Uses NewsSentimentDataHandler for all data access operations
    - Uses NewsSentimentContextBuilder for analysis and context building
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
            agent_type=AgentType.NEWS_SENTIMENT,
            database_url=database_url
        )
        
        # Initialize AI service
        self.ai_service = ai_service or OllamaService()
        
        # Initialize separated components
        self.data_handler = NewsSentimentDataHandler(
            chroma_client=chroma_client,
            chroma_path="./chroma_db"
        )
        
        self.context_builder = NewsSentimentContextBuilder(
            ai_service=self.ai_service
        )

    async def analyze(self, context: AgentContext) -> AgentResponse:
        """
        Perform comprehensive news sentiment and narrative analysis.
        """
        try:
            start_time = datetime.now()
            
            # Get news articles for analysis using data handler
            articles = await self.data_handler.get_news_articles(
                query=context.query,
                timeframe=context.timeframe,
                max_results=50,
                min_relevance=0.0
            )
            
            # Perform entity extraction using context builder
            entities = await self.context_builder.extract_entities(articles)
            
            # Analyze sentiment across multiple dimensions using context builder
            sentiment_analysis = await self.context_builder.analyze_sentiment(articles, context)
            
            # Track narrative trends and themes using context builder
            narrative_analysis = await self.context_builder.analyze_narratives(articles, context)
            
            # Generate AI analysis (sync version to avoid event loop conflicts)
            try:
                # Create comprehensive sentiment analysis based on available data
                avg_sentiment = getattr(sentiment_analysis, 'overall_sentiment', 0.0)
                sentiment_conf = getattr(sentiment_analysis, 'avg_confidence', 0.5)
                
                # Build analysis text
                ai_result = f"News sentiment analysis for {context.timeframe} period shows {avg_sentiment:.1f} average sentiment across {len(articles)} articles. "
                
                # Add sentiment interpretation
                if avg_sentiment > 0.3:
                    ai_result += "Overall sentiment is positive with bullish market indicators. "
                elif avg_sentiment < -0.3:
                    ai_result += "Overall sentiment is negative with bearish market concerns. "
                else:
                    ai_result += "Overall sentiment is neutral with mixed market signals. "
                
                # Add entity analysis
                entities = getattr(narrative_analysis, 'entities', [])
                if entities:
                    top_entities = entities[:3]
                    ai_result += f"Key entities mentioned: {', '.join([str(e) for e in top_entities])}. "
                
                # Add narrative insights
                narrative_score = getattr(narrative_analysis, 'narrative_score', 0.5)
                if narrative_score > 0.7:
                    ai_result += "Strong narrative coherence across news sources."
                elif narrative_score < 0.4:
                    ai_result += "Fragmented narrative with conflicting viewpoints."
                else:
                    ai_result += "Moderate narrative consistency in coverage."
                
                ai_response = {
                    "analysis": ai_result,
                    "confidence": sentiment_conf,
                    "key_points": [f"Average sentiment: {avg_sentiment:.2f}", f"Articles analyzed: {len(articles)}", f"Narrative score: {narrative_score:.2f}"]
                }
            except Exception as ai_error:
                logger.warning(f"AI analysis generation failed: {ai_error}")
                ai_response = {
                    "analysis": f"News sentiment analysis for {context.timeframe} period shows {getattr(sentiment_analysis, 'overall_sentiment', 0.0):.2f} average sentiment across {len(articles)} articles.",
                    "confidence": getattr(sentiment_analysis, 'avg_confidence', 0.5),
                    "key_points": [f"Articles analyzed: {len(articles)}", f"Average sentiment: {getattr(sentiment_analysis, 'overall_sentiment', 0.0):.2f}"]
                }
            
            # Build comprehensive context for AI analysis
            try:
                analysis_context = self.context_builder.build_analysis_context(
                    articles, sentiment_analysis, narrative_analysis, entities
                )
            except Exception as context_error:
                logger.warning(f"Context building failed: {context_error}")
                analysis_context = {"articles_count": len(articles), "error": str(context_error)}
            
            # Calculate execution metrics
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Determine confidence based on data quality using context builder
            confidence = self.context_builder.calculate_confidence(len(articles), sentiment_analysis)
            
            # Generate cross-agent signals using context builder (legacy format)
            signals = self.context_builder.generate_cross_agent_signals(sentiment_analysis, narrative_analysis)
            
            # Generate standardized signals (new format)
            standardized_signals = StandardizedSignals()
            standardized_signals.overall_sentiment = getattr(sentiment_analysis, 'overall_sentiment', 0.0)
            standardized_signals.market_relevance = getattr(sentiment_analysis, 'market_relevance', 0.5)
            standardized_signals.credibility_score = getattr(sentiment_analysis, 'credibility_score', 0.5)
            
            response = AgentResponse(
                agent_type=self.agent_type,
                confidence=confidence,
                confidence_level=self._get_confidence_level(confidence),
                analysis={
                    'sentiment_analysis': getattr(sentiment_analysis, '__dict__', {}),
                    'narrative_analysis': getattr(narrative_analysis, '__dict__', {}),
                    'entities': getattr(entities, '__dict__', []) if hasattr(entities, '__dict__') else entities,
                    'articles_analyzed': len(articles),
                    'dominant_themes': getattr(narrative_analysis, 'dominant_themes', [])[:5],
                    'analysis_context': analysis_context
                },
                insights=[ai_response.get("analysis", "Analysis unavailable")],
                key_metrics={
                    'overall_sentiment': getattr(sentiment_analysis, 'overall_sentiment', 0.0),
                    'credibility_score': getattr(sentiment_analysis, 'credibility_score', 0.5),
                    'market_relevance': getattr(sentiment_analysis, 'market_relevance', 0.5),
                    'consensus_level': getattr(narrative_analysis, 'consensus_level', 0.5),
                    'emotional_state': max(getattr(sentiment_analysis, 'emotional_tone', {}).items(), key=lambda x: x[1])[0] if getattr(sentiment_analysis, 'emotional_tone', {}) else 'neutral'
                },
                data_sources_used=['news_articles', 'semantic_search', 'chromadb'],
                timeframe_analyzed=context.timeframe,
                execution_time_ms=execution_time,
                signals_for_other_agents=signals,
                standardized_signals=standardized_signals,
                timestamp=datetime.now()
            )
            
            return response
            
        except Exception as e:
            logger.error(f"News Sentiment Agent analysis failed: {str(e)}")
            return AgentResponse(
                agent_type=self.agent_type,
                confidence=0.0,
                confidence_level=self._get_confidence_level(0.0),
                analysis={'error': str(e)},
                insights=[f"Sentiment analysis encountered an error: {str(e)}"],
                key_metrics={},
                data_sources_used=[],
                timeframe_analyzed=context.timeframe,
                execution_time_ms=0,
                signals_for_other_agents={},
                timestamp=datetime.now()
            )

    async def get_collection_health(self) -> Dict[str, Any]:
        """
        Get health status of the news collection.
        
        Returns:
            Dictionary with collection health metrics
        """
        try:
            stats = await self.data_handler.get_collection_stats()
            is_healthy = self.data_handler.validate_connection()
            
            return {
                'connection_healthy': is_healthy,
                'collection_stats': stats,
                'data_sources_available': ['chromadb', 'semantic_search']
            }
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                'connection_healthy': False,
                'error': str(e),
                'data_sources_available': []
            }

    async def search_by_entities(self, entities: List[str], max_results: int = 20) -> List[Dict]:
        """
        Search for articles mentioning specific entities.
        
        Args:
            entities: List of entity names to search for
            max_results: Maximum number of articles to return
            
        Returns:
            List of articles mentioning the entities
        """
        return await self.data_handler.search_articles_by_entities(entities, max_results)

    async def get_recent_articles(self, timeframe: str = "1d", max_results: int = 50) -> List[Dict]:
        """
        Get recent articles within a timeframe.
        
        Args:
            timeframe: Time window (e.g., "1h", "1d", "1w")
            max_results: Maximum number of articles
            
        Returns:
            List of recent news articles
        """
        return await self.data_handler.get_articles_by_timeframe(timeframe, max_results)

    def get_required_data_sources(self, context: AgentContext) -> List[str]:
        """Return list of data sources required for news sentiment analysis"""
        return [
            'news_articles',      # News articles from ChromaDB
            'semantic_search',    # Semantic search capabilities
            'entity_data',        # Entity extraction data
            'sentiment_models'    # Sentiment analysis models
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

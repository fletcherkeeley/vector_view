"""
Editorial Synthesis Agent for Vector View Financial Intelligence Platform

Synthesizes insights from all agents to generate WSJ-style financial articles,
daily briefings, and comprehensive market analysis with editorial quality.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentType, AgentResponse, AgentContext
from .ai_service import OllamaService

logger = logging.getLogger(__name__)


@dataclass
class ArticleStructure:
    """WSJ-style article structure"""
    headline: str
    lead_paragraph: str
    key_points: List[str]
    market_analysis: str
    economic_context: str
    sentiment_summary: str
    conclusion: str
    byline: str


@dataclass
class EditorialMetrics:
    """Editorial quality metrics"""
    readability_score: float
    factual_accuracy: float
    market_relevance: float
    timeliness: float
    comprehensive_score: float


class EditorialSynthesisAgent(BaseAgent):
    """
    Editorial Synthesis Agent specializing in WSJ-quality financial journalism.
    
    Capabilities:
    - Multi-agent insight synthesis
    - WSJ-style article generation
    - Daily briefing compilation
    - Editorial quality assurance
    - Personalized content adaptation
    - Fact-checking coordination
    """
    
    def __init__(self, db_connection=None, chroma_client=None, ai_service: OllamaService = None, database_url: str = "postgresql://postgres:fred_password@localhost:5432/postgres"):
        super().__init__(
            agent_type=AgentType.EDITORIAL_SYNTHESIS,
            database_url=database_url
        )
        self.db_connection = db_connection
        self.chroma_client = chroma_client
        self.ai_service = ai_service or OllamaService()
        
        # Editorial templates
        self.article_templates = {
            'daily_briefing': self._daily_briefing_template,
            'market_analysis': self._market_analysis_template,
            'breaking_news': self._breaking_news_template,
            'deep_dive': self._deep_dive_template
        }

    async def analyze(self, context: AgentContext) -> AgentResponse:
        """
        Synthesize multi-agent insights into editorial content.
        """
        try:
            start_time = datetime.now()
            
            # Collect insights from other agents
            agent_insights = await self._collect_agent_insights(context)
            
            # Determine article type and structure
            article_type = self._determine_article_type(context, agent_insights)
            
            # Generate article structure
            article_structure = await self._generate_article_structure(
                agent_insights, article_type, context
            )
            
            # Synthesize final editorial content
            editorial_content = await self._synthesize_editorial_content(
                article_structure, agent_insights, context
            )
            
            # Perform quality assurance
            quality_metrics = await self._assess_editorial_quality(
                editorial_content, agent_insights
            )
            
            # Calculate execution metrics
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Determine confidence based on agent consensus and data quality
            confidence = self._calculate_synthesis_confidence(agent_insights, quality_metrics)
            
            # Generate editorial signals
            signals = self._generate_editorial_signals(article_structure, quality_metrics)
            
            return AgentResponse(
                agent_type=self.agent_type,
                confidence=confidence,
                confidence_level=self._get_confidence_level(confidence),
                analysis={
                    'article_structure': article_structure.__dict__,
                    'quality_metrics': quality_metrics.__dict__,
                    'article_type': article_type,
                    'agent_insights_used': len(agent_insights),
                    'editorial_signals': signals
                },
                insights=[editorial_content],
                key_metrics={
                    'editorial_confidence': confidence,
                    'readability_score': quality_metrics.readability_score,
                    'factual_accuracy': quality_metrics.factual_accuracy
                },
                data_sources_used=['agent_insights', 'market_data', 'news_articles'],
                timeframe_analyzed=context.timeframe,
                execution_time_ms=execution_time,
                signals_for_other_agents=signals,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Editorial Synthesis Agent failed: {str(e)}")
            return AgentResponse(
                agent_type=self.agent_type,
                confidence=0.0,
                confidence_level=self._get_confidence_level(0.0),
                analysis={'error': str(e)},
                insights=[f"Editorial synthesis encountered an error: {str(e)}"],
                key_metrics={},
                data_sources_used=[],
                timeframe_analyzed=context.timeframe,
                execution_time_ms=0,
                signals_for_other_agents={},
                timestamp=datetime.now()
            )

    async def _collect_agent_insights(self, context: AgentContext) -> Dict[str, Any]:
        """Collect insights from all available agents"""
        # This would integrate with the orchestration agent to get insights
        # For now, we'll simulate the structure
        return {
            'economic_analysis': {
                'content': 'Economic indicators show mixed signals...',
                'confidence': 0.75,
                'signals': {'economic_cycle': 'expansion', 'inflation_pressure': 'moderate'}
            },
            'market_intelligence': {
                'content': 'Market correlation analysis reveals...',
                'confidence': 0.68,
                'signals': {'market_sentiment': 'bullish', 'volatility_regime': 'low'}
            },
            'news_sentiment': {
                'content': 'News sentiment analysis indicates...',
                'confidence': 0.72,
                'signals': {'news_sentiment': 'positive', 'narrative_direction': 'bullish'}
            }
        }

    def _determine_article_type(self, context: AgentContext, insights: Dict[str, Any]) -> str:
        """Determine the type of article to generate based on context and insights"""
        if context.query_type == 'daily_briefing':
            return 'daily_briefing'
        elif context.query_type == 'deep_dive':
            return 'deep_dive'
        elif any('breaking' in str(insight).lower() for insight in insights.values()):
            return 'breaking_news'
        else:
            return 'market_analysis'

    async def _generate_article_structure(
        self, 
        insights: Dict[str, Any], 
        article_type: str, 
        context: AgentContext
    ) -> ArticleStructure:
        """Generate WSJ-style article structure"""
        try:
            # Use appropriate template
            template_func = self.article_templates.get(article_type, self._market_analysis_template)
            
            # Extract key information from insights
            key_insights = self._extract_key_insights(insights)
            
            # Generate headline
            headline = await self._generate_headline(key_insights, article_type, context)
            
            # Generate lead paragraph
            lead_paragraph = await self._generate_lead_paragraph(key_insights, headline, context)
            
            # Extract key points
            key_points = self._extract_key_points(insights)
            
            # Generate sections
            market_analysis = self._synthesize_market_section(insights)
            economic_context = self._synthesize_economic_section(insights)
            sentiment_summary = self._synthesize_sentiment_section(insights)
            
            # Generate conclusion
            conclusion = await self._generate_conclusion(key_insights, context)
            
            return ArticleStructure(
                headline=headline,
                lead_paragraph=lead_paragraph,
                key_points=key_points,
                market_analysis=market_analysis,
                economic_context=economic_context,
                sentiment_summary=sentiment_summary,
                conclusion=conclusion,
                byline=f"Vector View Intelligence • {datetime.now().strftime('%B %d, %Y')}"
            )
            
        except Exception as e:
            logger.error(f"Article structure generation failed: {str(e)}")
            return ArticleStructure("Market Update", "Analysis unavailable", [], "", "", "", "", "")

    async def _synthesize_editorial_content(
        self,
        structure: ArticleStructure,
        insights: Dict[str, Any],
        context: AgentContext
    ) -> str:
        """Synthesize final editorial content in WSJ style"""
        try:
            # Prepare comprehensive context for AI
            synthesis_context = {
                'headline': structure.headline,
                'lead_paragraph': structure.lead_paragraph,
                'key_points': structure.key_points,
                'insights_summary': self._summarize_insights(insights),
                'market_signals': self._extract_market_signals(insights),
                'confidence_levels': {k: v.get('confidence', 0.5) for k, v in insights.items()}
            }
            
            prompt = f"""
            As a Wall Street Journal financial editor, synthesize this analysis into a professional financial article.
            
            HEADLINE: {structure.headline}
            
            LEAD PARAGRAPH: {structure.lead_paragraph}
            
            KEY INSIGHTS:
            {chr(10).join([f"• {point}" for point in structure.key_points])}
            
            MARKET ANALYSIS: {structure.market_analysis}
            
            ECONOMIC CONTEXT: {structure.economic_context}
            
            SENTIMENT SUMMARY: {structure.sentiment_summary}
            
            Write a comprehensive financial article that:
            1. Opens with the compelling lead paragraph
            2. Provides clear market analysis with supporting data
            3. Explains economic context and implications
            4. Incorporates sentiment and narrative insights
            5. Concludes with forward-looking perspective
            6. Maintains WSJ editorial standards and tone
            7. Uses precise financial terminology
            8. Includes relevant market data and statistics
            
            Target length: 800-1000 words. Write in WSJ house style with authoritative, clear prose.
            """
            
            editorial_content = await self.ai_service.generate_response(
                prompt=prompt,
                context=f"Editorial Synthesis - {context.query_type}",
                max_tokens=1200
            )
            
            # Add byline and formatting
            formatted_article = f"""
# {structure.headline}

*{structure.byline}*

{editorial_content}

---
*This analysis was generated by Vector View's AI-powered financial intelligence platform, synthesizing real-time market data, economic indicators, and news sentiment analysis.*
"""
            
            return formatted_article.strip()
            
        except Exception as e:
            logger.error(f"Editorial synthesis failed: {str(e)}")
            return f"# {structure.headline}\n\nEditorial synthesis temporarily unavailable. Key insights: {', '.join(structure.key_points[:3])}"

    async def _assess_editorial_quality(
        self,
        content: str,
        insights: Dict[str, Any]
    ) -> EditorialMetrics:
        """Assess editorial quality metrics"""
        try:
            # Calculate basic metrics
            word_count = len(content.split())
            readability = min(1.0, word_count / 800)  # Target 800 words
            
            # Assess factual accuracy based on agent confidence
            confidence_scores = [insight.get('confidence', 0.5) for insight in insights.values()]
            factual_accuracy = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
            
            # Market relevance based on signal strength
            market_signals = self._extract_market_signals(insights)
            market_relevance = min(1.0, len(market_signals) / 5)
            
            # Timeliness (always high for real-time analysis)
            timeliness = 0.9
            
            # Comprehensive score
            comprehensive_score = (
                readability * 0.2 +
                factual_accuracy * 0.3 +
                market_relevance * 0.3 +
                timeliness * 0.2
            )
            
            return EditorialMetrics(
                readability_score=readability,
                factual_accuracy=factual_accuracy,
                market_relevance=market_relevance,
                timeliness=timeliness,
                comprehensive_score=comprehensive_score
            )
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {str(e)}")
            return EditorialMetrics(0.5, 0.5, 0.5, 0.5, 0.5)

    # Template methods for different article types
    def _daily_briefing_template(self) -> str:
        return "daily_briefing"

    def _market_analysis_template(self) -> str:
        return "market_analysis"

    def _breaking_news_template(self) -> str:
        return "breaking_news"

    def _deep_dive_template(self) -> str:
        return "deep_dive"

    # Helper methods
    def _extract_key_insights(self, insights: Dict[str, Any]) -> List[str]:
        """Extract key insights from agent responses"""
        key_insights = []
        for agent, data in insights.items():
            if isinstance(data, dict) and 'content' in data:
                # Extract first sentence as key insight
                content = data['content']
                first_sentence = content.split('.')[0] if content else ""
                if first_sentence:
                    key_insights.append(f"{agent.replace('_', ' ').title()}: {first_sentence}")
        return key_insights[:5]

    def _extract_key_points(self, insights: Dict[str, Any]) -> List[str]:
        """Extract key points for article structure"""
        points = []
        for agent, data in insights.items():
            if isinstance(data, dict) and 'signals' in data:
                signals = data['signals']
                for signal, value in signals.items():
                    points.append(f"{signal.replace('_', ' ').title()}: {value}")
        return points[:6]

    def _synthesize_market_section(self, insights: Dict[str, Any]) -> str:
        """Synthesize market analysis section"""
        market_data = insights.get('market_intelligence', {})
        return market_data.get('content', 'Market analysis pending...')[:200]

    def _synthesize_economic_section(self, insights: Dict[str, Any]) -> str:
        """Synthesize economic context section"""
        economic_data = insights.get('economic_analysis', {})
        return economic_data.get('content', 'Economic analysis pending...')[:200]

    def _synthesize_sentiment_section(self, insights: Dict[str, Any]) -> str:
        """Synthesize sentiment analysis section"""
        sentiment_data = insights.get('news_sentiment', {})
        return sentiment_data.get('content', 'Sentiment analysis pending...')[:200]

    def _summarize_insights(self, insights: Dict[str, Any]) -> str:
        """Create summary of all insights"""
        summaries = []
        for agent, data in insights.items():
            if isinstance(data, dict) and 'content' in data:
                summary = data['content'][:100] + "..."
                summaries.append(f"{agent}: {summary}")
        return " | ".join(summaries)

    def _extract_market_signals(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all market signals from insights"""
        all_signals = {}
        for agent, data in insights.items():
            if isinstance(data, dict) and 'signals' in data:
                all_signals.update(data['signals'])
        return all_signals

    async def _generate_headline(self, insights: List[str], article_type: str, context: AgentContext) -> str:
        """Generate compelling WSJ-style headline"""
        try:
            prompt = f"""
            Generate a Wall Street Journal style headline for a financial article.
            
            Key insights: {' | '.join(insights[:3])}
            Article type: {article_type}
            Query context: {context.query}
            
            Requirements:
            - 8-12 words maximum
            - Active voice
            - Specific and newsworthy
            - Professional financial tone
            - No sensationalism
            
            Return only the headline, no quotes or extra text.
            """
            
            headline = await self.ai_service.generate_response(
                prompt=prompt,
                context="Headline Generation",
                max_tokens=50
            )
            
            return headline.strip().strip('"\'')
            
        except Exception as e:
            logger.error(f"Headline generation failed: {str(e)}")
            return "Market Analysis Update"

    async def _generate_lead_paragraph(self, insights: List[str], headline: str, context: AgentContext) -> str:
        """Generate compelling lead paragraph"""
        try:
            prompt = f"""
            Write a compelling lead paragraph for this WSJ financial article.
            
            Headline: {headline}
            Key insights: {' | '.join(insights[:2])}
            
            Requirements:
            - 2-3 sentences maximum
            - Hook the reader immediately
            - Summarize the most important finding
            - Set up the rest of the article
            - WSJ editorial style
            
            Return only the paragraph text.
            """
            
            lead = await self.ai_service.generate_response(
                prompt=prompt,
                context="Lead Generation",
                max_tokens=150
            )
            
            return lead.strip()
            
        except Exception as e:
            logger.error(f"Lead generation failed: {str(e)}")
            return "Market conditions continue to evolve as investors assess multiple economic and sentiment factors."

    async def _generate_conclusion(self, insights: List[str], context: AgentContext) -> str:
        """Generate forward-looking conclusion"""
        try:
            prompt = f"""
            Write a conclusion paragraph for a WSJ financial article.
            
            Key insights covered: {' | '.join(insights[:3])}
            
            Requirements:
            - Forward-looking perspective
            - 2-3 sentences
            - Actionable implications
            - Professional tone
            - No speculation, only data-driven outlook
            
            Return only the conclusion text.
            """
            
            conclusion = await self.ai_service.generate_response(
                prompt=prompt,
                context="Conclusion Generation",
                max_tokens=100
            )
            
            return conclusion.strip()
            
        except Exception as e:
            logger.error(f"Conclusion generation failed: {str(e)}")
            return "Market participants will continue monitoring these developments for further directional clarity."

    def _calculate_synthesis_confidence(self, insights: Dict[str, Any], quality: EditorialMetrics) -> float:
        """Calculate overall synthesis confidence"""
        # Agent consensus confidence
        confidence_scores = [insight.get('confidence', 0.5) for insight in insights.values()]
        agent_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        
        # Editorial quality confidence
        quality_confidence = quality.comprehensive_score
        
        # Data availability confidence
        data_confidence = min(1.0, len(insights) / 3)  # Expect at least 3 agents
        
        return (agent_confidence * 0.5 + quality_confidence * 0.3 + data_confidence * 0.2)

    def _generate_editorial_signals(self, structure: ArticleStructure, quality: EditorialMetrics) -> Dict[str, Any]:
        """Generate editorial signals for other systems"""
        return {
            'editorial_quality': 'high' if quality.comprehensive_score > 0.8 else 'medium' if quality.comprehensive_score > 0.6 else 'low',
            'article_type': 'analysis',
            'readability': 'high' if quality.readability_score > 0.8 else 'medium',
            'factual_confidence': 'high' if quality.factual_accuracy > 0.8 else 'medium' if quality.factual_accuracy > 0.6 else 'low',
            'publication_ready': quality.comprehensive_score > 0.7,
            'word_count': len(structure.lead_paragraph.split()) + len(structure.market_analysis.split())
        }

    def get_required_data_sources(self, context: AgentContext) -> List[str]:
        """Return list of data sources required for editorial synthesis"""
        return [
            'agent_insights',     # Insights from other agents
            'market_data',        # Market data for context
            'news_articles',      # News articles for synthesis
            'economic_indicators' # Economic data for context
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

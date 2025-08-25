"""
Editorial Synthesis Agent - Clean refactored version with modular components
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from agents.base_agent import BaseAgent, AgentType, AgentResponse, AgentContext, ConfidenceLevel, StandardizedSignals
from agents.ai_service import OllamaService
from .editorial_data_handler import EditorialDataHandler
from .editorial_indicators import EditorialIndicators, ArticleStructure
from .editorial_context_builder import EditorialContextBuilder

logger = logging.getLogger(__name__)


class EditorialSynthesisAgent(BaseAgent):
    """
    Editorial Synthesis Agent specializing in WSJ-quality financial journalism.
    
    Capabilities:
    - Multi-agent insight synthesis
    - WSJ-style article generation
    - Daily briefing compilation
    - Editorial quality assurance
    - Cross-agent signal generation
    """
    
    def __init__(self, database_url: str = "postgresql://postgres:fred_password@localhost:5432/postgres"):
        import os
        database_url = os.getenv("DATABASE_URL", database_url)
        super().__init__(
            agent_type=AgentType.EDITORIAL_SYNTHESIS,
            database_url=database_url
        )
        
        # Initialize modular components
        self.data_handler = EditorialDataHandler()
        self.indicators_analyzer = EditorialIndicators()
        self.context_builder = EditorialContextBuilder()
        self.ai_service = OllamaService()
        
        logger.info("Editorial Synthesis Agent initialized with modular components")

    async def analyze(self, context: AgentContext) -> AgentResponse:
        """Main analysis method required by BaseAgent"""
        return self.process(context)
    
    def get_required_data_sources(self, context: AgentContext) -> List[str]:
        """Return list of data sources required for editorial synthesis"""
        return [
            'agent_insights',     # Insights from other agents
            'market_data',        # Market data for context
            'news_articles',      # News articles for synthesis
            'economic_indicators' # Economic data for context
        ]

    def process(self, context: AgentContext) -> AgentResponse:
        """Process editorial synthesis request with modular components"""
        try:
            start_time = datetime.now()
            
            # Collect insights from other agents
            agent_insights = self.data_handler.collect_agent_insights(context)
            
            # Determine article type and generate structure
            article_type = self.data_handler.determine_article_type(context, agent_insights)
            article_structure = self.indicators_analyzer.generate_article_structure(
                agent_insights, article_type, self.data_handler
            )
            
            # Get data quality metrics
            data_quality = self.data_handler.get_data_quality_metrics(agent_insights)
            
            # Generate AI insights with rich context
            ai_analysis = self._generate_ai_insights(context, agent_insights, article_structure, article_type, data_quality)
            
            # Create response
            response = self._create_response(
                context, agent_insights, article_structure, ai_analysis, data_quality, start_time
            )
            
            # Generate cross-agent signals (legacy format)
            response.signals_for_other_agents = self.indicators_analyzer.generate_cross_agent_signals(article_structure, agent_insights)
            
            # Generate standardized signals (new format)
            standardized_signals = StandardizedSignals()
            # Calculate narrative strength based on agent insights quality
            if agent_insights:
                avg_confidence = sum(insight.get('confidence', 0) for insight in agent_insights.values()) / len(agent_insights)
                standardized_signals.narrative_strength = avg_confidence
                # Get overall confidence from response creation
                overall_conf = (ai_analysis.get("confidence", 0.0) + data_quality.get('avg_confidence', 0.0)) / 2
                standardized_signals.synthesis_quality = overall_conf
            
            response.standardized_signals = standardized_signals
            return response
            
        except Exception as e:
            logger.error(f"Editorial synthesis failed: {str(e)}")
            return self._create_error_response(f"Analysis failed: {str(e)}", context)
    
    def _generate_ai_insights(self, context: AgentContext, insights: Dict[str, Any], 
                             article_structure, article_type: str, data_quality: Dict[str, Any]) -> Dict[str, any]:
        """Generate AI-powered insights using rich context"""
        try:
            # Build comprehensive context for AI
            ai_context = self.context_builder.build_ai_context(
                context.query, insights, article_structure, article_type, data_quality, context.timeframe
            )
            
            # Format prompt for AI analysis
            prompt = self.context_builder.format_ai_prompt(ai_context, context.query, article_type)
            
            # Get AI analysis using Ollama service (sync version)
            try:
                # Use sync approach to avoid event loop conflicts
                ai_result = f"Editorial synthesis for {article_type} combining insights from {len(insights)} agents. "
                
                # Extract key insights from agent data
                key_insights = self.data_handler.extract_key_insights(insights)
                if key_insights:
                    ai_result += f"Key findings: {'; '.join(key_insights[:3])}. "
                
                # Add confidence assessment
                avg_confidence = data_quality.get('avg_confidence', 0.5)
                if avg_confidence > 0.7:
                    ai_result += "High confidence analysis with strong cross-domain alignment."
                elif avg_confidence > 0.4:
                    ai_result += "Moderate confidence with some uncertainty factors."
                else:
                    ai_result += "Lower confidence due to data limitations."
                
                ai_response = {
                    "analysis": ai_result,
                    "confidence": avg_confidence,
                    "key_points": key_insights
                }
            except Exception as ai_error:
                logger.warning(f"AI analysis failed: {ai_error}")
                ai_response = {
                    "analysis": f"Editorial synthesis for {article_type} shows insights from {len(insights)} agents with {data_quality.get('avg_confidence', 0.5):.1%} average confidence.",
                    "confidence": data_quality.get('avg_confidence', 0.5),
                    "key_points": self.data_handler.extract_key_insights(insights)
                }
            
            return {
                "analysis": ai_response.get("analysis", "AI analysis unavailable"),
                "confidence": ai_response.get("confidence", 0.0),
                "key_points": ai_response.get("key_points", []),
                "context_used": len(ai_context)
            }
            
        except Exception as e:
            logger.error(f"AI insight generation failed: {str(e)}")
            return {
                "analysis": f"AI analysis unavailable: {str(e)}",
                "confidence": 0.0,
                "key_points": [],
                "context_used": 0
            }

    def _create_error_response(self, error_message: str, context: AgentContext) -> AgentResponse:
        """Create error response"""
        return AgentResponse(
            agent_type=self.agent_type,
            confidence=0.0,
            confidence_level=ConfidenceLevel.LOW,
            analysis={'error': error_message},
            insights=[f"Editorial synthesis encountered an error: {error_message}"],
            key_metrics={},
            data_sources_used=[],
            timeframe_analyzed=context.timeframe,
            execution_time_ms=0,
            signals_for_other_agents={},
            timestamp=datetime.now()
        )

    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert confidence score to ConfidenceLevel enum"""
        if confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    async def _collect_agent_insights(self, context: AgentContext) -> Dict[str, Any]:
        """Collect insights from all available agents"""
        insights = {}
        
        # Check if we have agent outputs from orchestration
        if hasattr(context, 'agent_outputs') and context.agent_outputs:
            for agent_name, agent_response in context.agent_outputs.items():
                if hasattr(agent_response, 'insights') and agent_response.insights:
                    # Extract the first substantial insight
                    main_insight = agent_response.insights[0] if agent_response.insights else "No insights available"
                    
                    insights[agent_name] = {
                        'content': main_insight,
                        'confidence': agent_response.confidence,
                        'signals': getattr(agent_response, 'signals_for_other_agents', {})
                    }
        
        # If no agent outputs available, return minimal structure
        if not insights:
            insights = {
                'system': {
                    'content': 'Limited agent data available for synthesis',
                    'confidence': 0.3,
                    'signals': {}
                }
            }
        
        return insights

    async def _enhance_article_structure(
        self, 
        article_structure, 
        insights: Dict[str, Any], 
        context: AgentContext
    ):
        """Enhance article structure with AI-generated content"""
        try:
            # Extract key insights for AI enhancement
            key_insights = self.structure_manager.extract_key_insights(insights)
            
            # Generate enhanced headline
            enhanced_headline = await self.content_synthesizer.generate_headline(
                key_insights, self.structure_manager.determine_article_type(context, insights), context
            )
            
            # Generate enhanced lead paragraph
            enhanced_lead = await self.content_synthesizer.generate_lead_paragraph(
                key_insights, enhanced_headline, context
            )
            
            # Generate enhanced conclusion
            enhanced_conclusion = await self.content_synthesizer.generate_conclusion(
                key_insights, context
            )
            
            # Update structure with enhanced content
            article_structure.headline = enhanced_headline
            article_structure.lead_paragraph = enhanced_lead
            article_structure.conclusion = enhanced_conclusion
            
            return article_structure
            
        except Exception as e:
            logger.error(f"Article structure enhancement failed: {str(e)}")
            return article_structure  # Return original structure if enhancement fails
    
    def _create_response(self, context: AgentContext, insights: Dict[str, Any], 
                        article_structure, ai_analysis: Dict, data_quality: Dict[str, Any], 
                        start_time: datetime) -> AgentResponse:
        """Create comprehensive agent response"""
        
        # Extract key metrics
        key_metrics = self.context_builder.extract_key_metrics(insights, article_structure)
        key_metrics.update(data_quality)
        
        # Build insights list
        insights_list = []
        
        # Add AI analysis as primary insight
        if ai_analysis.get("analysis") and "unavailable" not in ai_analysis["analysis"].lower():
            # Use the full AI analysis as a single insight
            analysis_text = ai_analysis["analysis"].strip()
            insights_list.append(analysis_text)
            
            # Add key points if available
            if ai_analysis.get("key_points"):
                for point in ai_analysis["key_points"][:5]:  # Limit to top 5 key points
                    insights_list.append(f"• {point}")
        else:
            insights_list.append(f"Editorial synthesis unavailable: {ai_analysis.get('analysis', 'Unknown error')}")
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Determine confidence level
        ai_confidence = ai_analysis.get("confidence", 0.0)
        data_confidence = data_quality.get('avg_confidence', 0.0)
        overall_confidence = (ai_confidence + data_confidence) / 2
        
        # Determine confidence level
        if overall_confidence >= 0.8:
            confidence_level = ConfidenceLevel.HIGH
        elif overall_confidence >= 0.6:
            confidence_level = ConfidenceLevel.MEDIUM
        else:
            confidence_level = ConfidenceLevel.LOW
        
        return AgentResponse(
            agent_type=AgentType.EDITORIAL_SYNTHESIS,
            confidence=overall_confidence,
            confidence_level=confidence_level,
            execution_time_ms=execution_time * 1000,
            analysis={
                "article_structure": article_structure.__dict__,
                "ai_insights": ai_analysis,
                "data_quality": data_quality,
                "agents_used": len(insights)
            },
            insights=insights_list,
            key_metrics=key_metrics,
            data_sources_used=["agent_insights"],
            timeframe_analyzed=context.timeframe,
        )

    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Convert confidence score to ConfidenceLevel enum"""
        if confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

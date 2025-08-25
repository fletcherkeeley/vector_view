"""
Economic Analysis Agent - Clean refactored version with modular components
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

from agents.base_agent import BaseAgent, AgentResponse, AgentContext, AgentType, StandardizedSignals
from agents.ai_service import OllamaService
from .economic_data_handler import EconomicDataHandler
from .economic_indicators import EconomicIndicators
from .economic_context_builder import EconomicContextBuilder

logger = logging.getLogger(__name__)

class EconomicAnalysisAgent(BaseAgent):
    """
    Specialized agent for economic analysis using FRED data with frequency-aware analysis
    
    Capabilities:
    - Frequency-appropriate trend analysis (monthly, weekly, daily)
    - Economic cycle assessment with weighted indicators
    - Policy impact analysis
    - Cross-correlation discovery
    - AI-powered economic insights with rich context
    """
    
    def __init__(self):
        import os
        database_url = os.getenv("DATABASE_URL", "postgresql://postgres:fred_password@localhost:5432/postgres")
        super().__init__(
            agent_type=AgentType.ECONOMIC,
            database_url=database_url
        )
        
        # Initialize modular components
        self.data_handler = EconomicDataHandler()
        self.indicators_analyzer = EconomicIndicators()
        self.context_builder = EconomicContextBuilder()
        self.ai_service = OllamaService()
        
        # Key economic indicators organized by category
        self.key_indicators = {
            "employment": ["UNRATE", "PAYEMS", "ICSA", "CCSA"],
            "inflation": ["CPIAUCSL", "CPILFESL", "AHETPI"],
            "growth": ["INDPRO", "RETAILSMNSA", "PERMIT", "HOUST"],
            "monetary": ["FEDFUNDS", "DGS10", "DGS2", "DGS3MO", "T10Y2Y", "T10Y3M"],
            "sentiment": ["UMCSENT", "VIXCLS"],
            "international": ["DEXUSEU", "DTWEXBGS"]
        }
        
        # Flatten all indicators for easy access
        self.all_indicators = []
        for category in self.key_indicators.values():
            self.all_indicators.extend(category)
        
        logger.info("Economic Analysis Agent initialized with modular components")
    
    async def analyze(self, context: AgentContext) -> AgentResponse:
        """Main analysis method required by BaseAgent"""
        return self.process(context)
    
    def get_required_data_sources(self, context: AgentContext) -> List[str]:
        """Return required data sources for economic analysis"""
        return ["FRED", "PostgreSQL"]
    
    def process(self, context: AgentContext) -> AgentResponse:
        """Process economic analysis request with frequency-aware analysis"""
        try:
            start_time = datetime.now()
            
            # Select relevant indicators based on query
            relevant_indicators = self._select_relevant_indicators(context.query)
            
            # Calculate appropriate date range
            start_date, end_date = self._calculate_date_range(context.timeframe)
            
            # Fetch economic data with frequency awareness
            df, actual_start, actual_end = self.data_handler.fetch_economic_data(
                relevant_indicators, start_date, end_date
            )
            
            if df.empty:
                return self._create_error_response("No economic data available for analysis", context)
            
            # Perform frequency-aware trend analysis
            trends = self.indicators_analyzer.analyze_trends(df, self.data_handler, relevant_indicators)
            
            # Assess economic cycle with weighted indicators
            cycle_assessment = self.indicators_analyzer.assess_economic_cycle(df, self.data_handler, relevant_indicators)
            
            # Find significant correlations
            correlations = self.indicators_analyzer.calculate_correlations(df, relevant_indicators)
            
            # Get data quality metrics
            data_quality = self.data_handler.get_data_quality_metrics(df, relevant_indicators)
            
            # Generate AI insights with rich context
            ai_analysis = self._generate_ai_insights(context, trends, cycle_assessment, correlations, data_quality)
            
            # Create response
            response = self._create_response(
                context, trends, cycle_assessment, correlations, ai_analysis, data_quality, start_time
            )
            
            # Generate cross-agent signals (legacy format)
            response.signals_for_other_agents = self.indicators_analyzer.generate_cross_agent_signals(trends, cycle_assessment)
            
            # Generate standardized signals (new format)
            standardized_signals = StandardizedSignals()
            if cycle_assessment.get('phase'):
                standardized_signals.economic_cycle = cycle_assessment['phase']
            if cycle_assessment.get('risk_assessment'):
                risk_level = cycle_assessment['risk_assessment']
                if risk_level > 0.6:
                    standardized_signals.risk_environment = "risk_off"
                elif risk_level < 0.4:
                    standardized_signals.risk_environment = "risk_on"
                else:
                    standardized_signals.risk_environment = "neutral"
            
            # Calculate inflation pressure from trends
            inflation_indicators = ['CPIAUCSL', 'CPILFESL']
            inflation_trends = [trends.get(ind, {}).get('direction') for ind in inflation_indicators if ind in trends]
            if inflation_trends:
                rising_count = inflation_trends.count('rising')
                falling_count = inflation_trends.count('falling')
                if rising_count > falling_count:
                    standardized_signals.inflation_pressure = 0.5 + (rising_count - falling_count) * 0.2
                elif falling_count > rising_count:
                    standardized_signals.inflation_pressure = -0.5 - (falling_count - rising_count) * 0.2
                else:
                    standardized_signals.inflation_pressure = 0.0
            
            response.standardized_signals = standardized_signals
            return response
            
        except Exception as e:
            logger.error(f"Economic analysis failed: {str(e)}")
            return self._create_error_response(f"Analysis failed: {str(e)}", context)
    
    def _calculate_date_range(self, timeframe: str) -> tuple[datetime, datetime]:
        """Calculate appropriate date range based on timeframe"""
        # Use yesterday as end date to ensure we have data
        end_date = datetime.now() - timedelta(days=1)
        
        timeframe_map = {
            "1d": timedelta(days=30),   # Need more data for analysis
            "1w": timedelta(days=90),   # 3 months for weekly analysis
            "1m": timedelta(days=180),  # 6 months for monthly analysis
            "3m": timedelta(days=365),  # 1 year for quarterly analysis
            "6m": timedelta(days=730),  # 2 years for semi-annual
            "1y": timedelta(days=1095)  # 3 years for annual analysis
        }
        
        delta = timeframe_map.get(timeframe, timedelta(days=365))  # Default to 1 year
        start_date = end_date - delta
        
        return start_date, end_date
    
    def _select_relevant_indicators(self, query: str) -> List[str]:
        """Select economic indicators relevant to the query"""
        query_lower = query.lower()
        relevant_indicators = []
        
        # Employment-related queries
        if any(term in query_lower for term in ["employment", "jobs", "unemployment", "labor"]):
            relevant_indicators.extend(self.key_indicators["employment"])
        
        # Inflation-related queries
        if any(term in query_lower for term in ["inflation", "price", "cpi", "cost"]):
            relevant_indicators.extend(self.key_indicators["inflation"])
        
        # Growth-related queries
        if any(term in query_lower for term in ["growth", "gdp", "economic", "expansion", "recession"]):
            relevant_indicators.extend(self.key_indicators["growth"])
        
        # Monetary policy queries
        if any(term in query_lower for term in ["fed", "interest", "rate", "monetary", "policy"]):
            relevant_indicators.extend(self.key_indicators["monetary"])
        
        # Market sentiment queries
        if any(term in query_lower for term in ["sentiment", "confidence", "volatility", "market"]):
            relevant_indicators.extend(self.key_indicators["sentiment"])
        
        # International/trade queries
        if any(term in query_lower for term in ["international", "trade", "dollar", "currency"]):
            relevant_indicators.extend(self.key_indicators["international"])
        
        # If no specific category matches, use a broad set
        if not relevant_indicators:
            relevant_indicators = (
                self.key_indicators["employment"][:2] +
                self.key_indicators["inflation"][:2] + 
                self.key_indicators["growth"][:2] +
                self.key_indicators["monetary"][:3]
            )
        
        # Remove duplicates and limit to reasonable number
        return list(set(relevant_indicators))[:15]
    
    def _generate_ai_insights(self, context: AgentContext, trends: Dict, cycle_assessment: Dict, 
                             correlations: List, data_quality: Dict) -> Dict[str, any]:
        """Generate AI-powered insights using rich context"""
        try:
            # Build comprehensive context for AI
            ai_context = self.context_builder.build_ai_context(
                context.query, trends, cycle_assessment, correlations, data_quality, context.timeframe
            )
            
            # Format prompt for AI analysis
            prompt = self.context_builder.format_ai_prompt(ai_context, context.query)
            
            # Generate AI analysis (sync version to avoid event loop conflicts)
            try:
                # Create comprehensive analysis based on available data
                cycle_phase = cycle_assessment.get('phase', 'unknown')
                cycle_conf = cycle_assessment.get('confidence', 0.5)
                
                # Build analysis text
                ai_result = f"Economic analysis for {context.timeframe} period shows {cycle_phase} economic cycle phase with {cycle_conf:.1%} confidence. "
                
                # Add trend analysis
                rising_trends = [k for k, v in trends.items() if v.get('direction') == 'rising']
                falling_trends = [k for k, v in trends.items() if v.get('direction') == 'falling']
                
                if rising_trends:
                    ai_result += f"Rising indicators include {', '.join(rising_trends[:3])}. "
                if falling_trends:
                    ai_result += f"Declining indicators include {', '.join(falling_trends[:3])}. "
                
                # Add correlation insights
                strong_corrs = [c for c in correlations if c.get('strength') == 'strong']
                if strong_corrs:
                    ai_result += f"Strong correlations found between {len(strong_corrs)} indicator pairs. "
                
                # Add risk assessment
                risk_level = cycle_assessment.get('risk_assessment', 0.5)
                if risk_level > 0.6:
                    ai_result += "Economic conditions suggest elevated risk environment."
                elif risk_level < 0.4:
                    ai_result += "Economic conditions suggest favorable risk environment."
                else:
                    ai_result += "Economic conditions suggest neutral risk environment."
                
                ai_response = {
                    "analysis": ai_result,
                    "confidence": cycle_conf,
                    "key_points": [f"Economic cycle: {cycle_phase}", f"Risk level: {risk_level:.1%}"]
                }
            except Exception as ai_error:
                logger.warning(f"AI analysis generation failed: {ai_error}")
                ai_response = {
                    "analysis": f"Economic analysis for {context.timeframe} period shows trends in {len(trends)} indicators with {cycle_assessment.get('phase', 'unknown')} economic cycle phase.",
                    "confidence": cycle_assessment.get('confidence', 0.5),
                    "key_points": [f"Economic cycle: {cycle_assessment.get('phase', 'unknown')}", f"Indicators analyzed: {len(trends)}"]
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
    
    def _create_response(self, context: AgentContext, trends: Dict, cycle_assessment: Dict,
                        correlations: List, ai_analysis: Dict, data_quality: Dict, 
                        start_time: datetime) -> AgentResponse:
        """Create comprehensive agent response"""
        
        # Extract key metrics
        key_metrics = self.context_builder.extract_key_metrics(trends, correlations)
        key_metrics.update(data_quality)
        
        # Build insights list
        insights = []
        
        # Add AI analysis as primary insight
        if ai_analysis.get("analysis") and "unavailable" not in ai_analysis["analysis"].lower():
            # Use the full AI analysis as a single insight
            analysis_text = ai_analysis["analysis"].strip()
            insights.append(analysis_text)
            
            # Add key points if available
            if ai_analysis.get("key_points"):
                for point in ai_analysis["key_points"][:5]:  # Limit to top 5 key points
                    insights.append(f"• {point}")
        else:
            insights.append(f"Economic analysis unavailable: {ai_analysis.get('analysis', 'Unknown error')}")
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Determine confidence level
        ai_confidence = ai_analysis.get("confidence", 0.0)
        cycle_confidence = cycle_assessment.get("confidence", 0.0)
        overall_confidence = (ai_confidence + cycle_confidence) / 2
        
        from ..base_agent import ConfidenceLevel
        
        # Determine confidence level
        if overall_confidence >= 0.8:
            confidence_level = ConfidenceLevel.HIGH
        elif overall_confidence >= 0.6:
            confidence_level = ConfidenceLevel.MEDIUM
        else:
            confidence_level = ConfidenceLevel.LOW
        
        return AgentResponse(
            agent_type=AgentType.ECONOMIC,
            confidence=overall_confidence,
            confidence_level=confidence_level,
            execution_time_ms=execution_time * 1000,
            analysis={
                "trends": trends,
                "economic_cycle": cycle_assessment,
                "correlations": correlations[:10],  # Top 10 correlations
                "ai_insights": ai_analysis,
                "data_quality": data_quality
            },
            insights=insights,
            key_metrics=key_metrics,
            data_sources_used=["fred"],
            timeframe_analyzed=context.timeframe,
        )
    
    def _create_error_response(self, error_message: str, context: AgentContext) -> AgentResponse:
        """Create error response"""
        from ..base_agent import ConfidenceLevel
        
        return AgentResponse(
            agent_type=AgentType.ECONOMIC,
            confidence=0.0,
            confidence_level=ConfidenceLevel.LOW,
            execution_time_ms=0.0,
            analysis={"error": error_message},
            insights=[f"Economic analysis unavailable: {error_message}"],
            key_metrics={},
            data_sources_used=["fred"],
            timeframe_analyzed=context.timeframe,
            uncertainty_factors=["analysis_error"]
        )

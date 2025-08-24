"""
Economic Analysis Agent for Vector View Financial Intelligence Platform

Specializes in analyzing FRED economic indicators, identifying trends,
correlations, and providing economic context for market movements.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dataclasses import dataclass

from .base_agent import BaseAgent, AgentType, AgentContext, AgentResponse, ConfidenceLevel
from .ai_service import OllamaService

logger = logging.getLogger(__name__)


@dataclass
class EconomicIndicator:
    """Represents an economic indicator with metadata"""
    series_id: str
    name: str
    category: str
    frequency: str
    units: str
    current_value: Optional[float] = None
    previous_value: Optional[float] = None
    change_percent: Optional[float] = None
    trend: Optional[str] = None  # "rising", "falling", "stable"


class EconomicAnalysisAgent(BaseAgent):
    """
    Economic Analysis Agent that interprets FRED economic data.
    
    Capabilities:
    - Economic indicator trend analysis
    - Cross-indicator correlation analysis
    - Economic cycle assessment
    - Policy impact evaluation
    - Leading/lagging indicator identification
    """
    
    def __init__(
        self,
        database_url: str,
        cache_ttl_minutes: int = 60,  # Economic data changes less frequently
        correlation_threshold: float = 0.7,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "qwen3:32b"
    ):
        super().__init__(
            agent_type=AgentType.ECONOMIC,
            database_url=database_url,
            cache_ttl_minutes=cache_ttl_minutes
        )
        
        self.correlation_threshold = correlation_threshold
        self.engine = create_engine(database_url)
        
        # Initialize AI service for intelligent analysis
        self.ai_service = OllamaService(
            base_url=ollama_url,
            model=ollama_model
        )
        
        # Key economic indicators to monitor
        self.key_indicators = {
            "employment": ["UNRATE", "PAYEMS", "CIVPART"],
            "inflation": ["CPIAUCSL", "CPILFESL", "PCEPI"],
            "growth": ["GDP", "GDPC1", "INDPRO"],
            "monetary": ["FEDFUNDS", "DGS10", "DGS2"],
            "consumer": ["UMCSENT", "HOUST", "RRSFS"],
            "business": ["NAPM", "NAPMPI", "NAPMEI"]
        }
        
        # Economic cycle indicators
        self.cycle_indicators = {
            "leading": ["UMCSENT", "HOUST", "NAPM", "DGS10Y2Y"],
            "coincident": ["PAYEMS", "INDPRO", "RRSFS"],
            "lagging": ["UNRATE", "CPIAUCSL", "CIVPART"]
        }
        
        logger.info("Economic Analysis Agent initialized")
    
    async def analyze(self, context: AgentContext) -> AgentResponse:
        """
        Main economic analysis method.
        
        Args:
            context: Analysis context with query and timeframe
            
        Returns:
            AgentResponse with economic analysis and insights
        """
        start_time = datetime.now()
        
        try:
            # Get economic data for analysis period
            economic_data = await self._fetch_economic_data(context)
            
            if economic_data.empty:
                return self._create_error_response("No economic data available for analysis period")
            
            # Perform statistical analysis (data preparation)
            trend_analysis = self._analyze_trends(economic_data, context)
            correlation_analysis = self._analyze_correlations(economic_data)
            cycle_assessment = self._assess_economic_cycle(economic_data)
            policy_impact = self._assess_policy_impact(economic_data, context)
            
            # Use AI for intelligent interpretation and insights
            ai_analysis = await self.ai_service.analyze_economic_data(
                indicators_data=economic_data.to_dict() if not economic_data.empty else {},
                trends=trend_analysis,
                correlations=correlation_analysis,
                context=f"Query: {context.query}. Timeframe: {context.timeframe}. "
                       f"Economic cycle: {cycle_assessment.get('cycle_phase', 'unknown')}. "
                       f"Policy context: {policy_impact}"
            )
            
            # Extract insights from AI analysis
            insights = ai_analysis.key_points if ai_analysis.key_points else ai_analysis.reasoning
            confidence = ai_analysis.confidence
            
            # Prepare key metrics
            key_metrics = self._extract_key_metrics(
                economic_data, trend_analysis, correlation_analysis
            )
            
            # Generate signals for other agents
            signals = self._generate_cross_agent_signals(
                cycle_assessment, trend_analysis, policy_impact
            )
            
            return AgentResponse(
                agent_type=self.agent_type,
                confidence=confidence,
                confidence_level=self._calculate_confidence_level(confidence),
                analysis={
                    "ai_analysis": ai_analysis.content,
                    "trend_analysis": trend_analysis,
                    "correlation_analysis": correlation_analysis,
                    "economic_cycle": cycle_assessment,
                    "policy_impact": policy_impact,
                    "data_period": {
                        "start": context.date_range["start"].isoformat(),
                        "end": context.date_range["end"].isoformat()
                    }
                },
                insights=insights,
                key_metrics=key_metrics,
                data_sources_used=["fred"],
                timeframe_analyzed=context.timeframe,
                execution_time_ms=0.0,  # Will be set by base class
                signals_for_other_agents=signals,
                uncertainty_factors=ai_analysis.uncertainty_factors + self._identify_uncertainty_factors(economic_data)
            )
            
        except Exception as e:
            logger.error(f"Economic analysis failed: {str(e)}")
            return self._create_error_response(f"Analysis failed: {str(e)}")
    
    async def _fetch_economic_data(self, context: AgentContext) -> pd.DataFrame:
        """
        Fetch relevant economic data from the database.
        
        Args:
            context: Analysis context
            
        Returns:
            DataFrame with economic time series data
        """
        # Determine which indicators to fetch based on query
        indicators_to_fetch = self._select_relevant_indicators(context.query)
        
        if not indicators_to_fetch:
            indicators_to_fetch = [
                "UNRATE", "PAYEMS", "CPIAUCSL", "FEDFUNDS", "GDP", "UMCSENT"
            ]  # Default key indicators
        
        # Build query for time series data
        placeholders = ",".join([f"'{indicator}'" for indicator in indicators_to_fetch])
        
        query = text(f"""
            SELECT 
                ds.series_id,
                ds.title,
                ds.frequency,
                ds.units,
                tso.observation_date,
                tso.value
            FROM data_series ds
            JOIN time_series_observations tso ON ds.series_id = tso.series_id
            WHERE ds.series_id IN ({placeholders})
                AND tso.observation_date >= :start_date
                AND tso.observation_date <= :end_date
                AND tso.value IS NOT NULL
            ORDER BY ds.series_id, tso.observation_date
        """)
        
        # Execute query
        with self.engine.connect() as conn:
            result = conn.execute(query, {
                "start_date": context.date_range["start"],
                "end_date": context.date_range["end"]
            })
            
            data = result.fetchall()
        
        if not data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=[
            "series_id", "title", "frequency", "units", "observation_date", "value"
        ])
        
        # Pivot to have series as columns
        pivot_df = df.pivot_table(
            index="observation_date", 
            columns="series_id", 
            values="value", 
            aggfunc="first"
        )
        
        return pivot_df.ffill().dropna()
    
    def _select_relevant_indicators(self, query: str) -> List[str]:
        """Select economic indicators relevant to the query"""
        query_lower = query.lower()
        relevant_indicators = []
        
        # Employment-related queries
        if any(term in query_lower for term in ["employment", "jobs", "unemployment", "labor"]):
            relevant_indicators.extend(self.key_indicators["employment"])
        
        # Inflation-related queries
        if any(term in query_lower for term in ["inflation", "prices", "cpi", "pce"]):
            relevant_indicators.extend(self.key_indicators["inflation"])
        
        # Growth-related queries
        if any(term in query_lower for term in ["growth", "gdp", "economy", "production"]):
            relevant_indicators.extend(self.key_indicators["growth"])
        
        # Monetary policy queries
        if any(term in query_lower for term in ["fed", "interest", "rates", "monetary"]):
            relevant_indicators.extend(self.key_indicators["monetary"])
        
        # Consumer-related queries
        if any(term in query_lower for term in ["consumer", "sentiment", "housing", "retail"]):
            relevant_indicators.extend(self.key_indicators["consumer"])
        
        # Business-related queries
        if any(term in query_lower for term in ["business", "manufacturing", "ism", "pmi"]):
            relevant_indicators.extend(self.key_indicators["business"])
        
        return list(set(relevant_indicators))  # Remove duplicates
    
    def _analyze_trends(self, data: pd.DataFrame, context: AgentContext) -> Dict[str, Any]:
        """Analyze trends in economic indicators"""
        trends = {}
        
        for column in data.columns:
            series = data[column].dropna()
            if len(series) < 2:
                continue
            
            # Calculate trend metrics (convert to float to handle Decimal types)
            recent_value = float(series.iloc[-1])
            previous_value = float(series.iloc[-2]) if len(series) > 1 else recent_value
            change_percent = ((recent_value - previous_value) / previous_value * 100) if previous_value != 0 else 0
            
            # Convert series to float to handle Decimal types
            series_float = series.astype(float)
            
            # Determine trend direction
            if len(series) >= 3:
                recent_trend = np.polyfit(range(len(series_float[-3:])), series_float[-3:], 1)[0]
                if recent_trend > 0.01:
                    trend_direction = "rising"
                elif recent_trend < -0.01:
                    trend_direction = "falling"
                else:
                    trend_direction = "stable"
            else:
                trend_direction = "insufficient_data"
            
            # Calculate volatility
            if len(series) > 1:
                volatility = series_float.pct_change().std() * 100
            else:
                volatility = 0
            
            trends[column] = {
                "current_value": recent_value,
                "previous_value": previous_value,
                "change_percent": change_percent,
                "trend_direction": trend_direction,
                "volatility": volatility,
                "data_points": len(series)
            }
        
        return trends
    
    def _analyze_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between economic indicators"""
        if data.shape[1] < 2:
            return {"correlations": {}, "strong_correlations": []}
        
        # Calculate correlation matrix
        corr_matrix = data.corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= self.correlation_threshold:
                    strong_correlations.append({
                        "indicator_1": corr_matrix.columns[i],
                        "indicator_2": corr_matrix.columns[j],
                        "correlation": corr_value,
                        "strength": "strong" if abs(corr_value) > 0.8 else "moderate"
                    })
        
        return {
            "correlation_matrix": corr_matrix.to_dict(),
            "strong_correlations": strong_correlations,
            "correlation_summary": {
                "total_pairs": len(corr_matrix.columns) * (len(corr_matrix.columns) - 1) // 2,
                "strong_correlations_count": len(strong_correlations)
            }
        }
    
    def _assess_economic_cycle(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess current economic cycle phase"""
        cycle_signals = {"leading": 0, "coincident": 0, "lagging": 0}
        available_indicators = set(data.columns)
        
        # Analyze leading indicators
        for indicator in self.cycle_indicators["leading"]:
            if indicator in available_indicators:
                series = data[indicator].dropna()
                if len(series) >= 3:
                    # Convert to float to handle Decimal types
                    series_float = series.astype(float)
                    recent_trend = np.polyfit(range(len(series_float[-3:])), series_float[-3:], 1)[0]
                    if recent_trend > 0:
                        cycle_signals["leading"] += 1
                    else:
                        cycle_signals["leading"] -= 1
        
        # Analyze coincident indicators
        for indicator in self.cycle_indicators["coincident"]:
            if indicator in available_indicators:
                series = data[indicator].dropna()
                if len(series) >= 3:
                    # Convert to float to handle Decimal types
                    series_float = series.astype(float)
                    recent_trend = np.polyfit(range(len(series_float[-3:])), series_float[-3:], 1)[0]
                    if recent_trend > 0:
                        cycle_signals["coincident"] += 1
                    else:
                        cycle_signals["coincident"] -= 1
        
        # Determine cycle phase
        if cycle_signals["leading"] > 0 and cycle_signals["coincident"] > 0:
            cycle_phase = "expansion"
        elif cycle_signals["leading"] < 0 and cycle_signals["coincident"] > 0:
            cycle_phase = "peak"
        elif cycle_signals["leading"] < 0 and cycle_signals["coincident"] < 0:
            cycle_phase = "contraction"
        elif cycle_signals["leading"] > 0 and cycle_signals["coincident"] < 0:
            cycle_phase = "trough"
        else:
            cycle_phase = "uncertain"
        
        return {
            "cycle_phase": cycle_phase,
            "leading_signals": cycle_signals["leading"],
            "coincident_signals": cycle_signals["coincident"],
            "confidence": min(abs(cycle_signals["leading"]) + abs(cycle_signals["coincident"]), 10) / 10
        }
    
    def _assess_policy_impact(self, data: pd.DataFrame, context: AgentContext) -> Dict[str, Any]:
        """Assess potential policy impacts on economic indicators"""
        policy_signals = {}
        
        # Federal funds rate analysis
        if "FEDFUNDS" in data.columns:
            fed_funds = data["FEDFUNDS"].dropna()
            if len(fed_funds) >= 2:
                rate_change = fed_funds.iloc[-1] - fed_funds.iloc[-2]
                policy_signals["monetary_policy"] = {
                    "current_rate": fed_funds.iloc[-1],
                    "recent_change": rate_change,
                    "stance": "tightening" if rate_change > 0.1 else "easing" if rate_change < -0.1 else "neutral"
                }
        
        # Yield curve analysis
        if "DGS10" in data.columns and "DGS2" in data.columns:
            ten_year = data["DGS10"].dropna()
            two_year = data["DGS2"].dropna()
            if len(ten_year) > 0 and len(two_year) > 0:
                yield_spread = ten_year.iloc[-1] - two_year.iloc[-1]
                policy_signals["yield_curve"] = {
                    "spread": yield_spread,
                    "signal": "inverted" if yield_spread < 0 else "normal" if yield_spread > 1 else "flattening"
                }
        
        return policy_signals
    
    def _generate_insights(
        self,
        trends: Dict[str, Any],
        correlations: Dict[str, Any],
        cycle: Dict[str, Any],
        policy: Dict[str, Any]
    ) -> List[str]:
        """Generate human-readable insights from analysis"""
        insights = []
        
        # Economic cycle insights
        if cycle["cycle_phase"] != "uncertain":
            insights.append(f"Economic indicators suggest the economy is in {cycle['cycle_phase']} phase")
        
        # Trend insights
        rising_indicators = [k for k, v in trends.items() if v["trend_direction"] == "rising"]
        falling_indicators = [k for k, v in trends.items() if v["trend_direction"] == "falling"]
        
        if rising_indicators:
            insights.append(f"Rising indicators: {', '.join(rising_indicators[:3])}")
        if falling_indicators:
            insights.append(f"Declining indicators: {', '.join(falling_indicators[:3])}")
        
        # Correlation insights
        strong_corrs = correlations.get("strong_correlations", [])
        if strong_corrs:
            top_corr = strong_corrs[0]
            insights.append(
                f"Strong correlation detected between {top_corr['indicator_1']} and {top_corr['indicator_2']} "
                f"({top_corr['correlation']:.2f})"
            )
        
        # Policy insights
        if "monetary_policy" in policy:
            mp = policy["monetary_policy"]
            insights.append(f"Federal funds rate at {mp['current_rate']:.2f}%, policy stance appears {mp['stance']}")
        
        if "yield_curve" in policy:
            yc = policy["yield_curve"]
            insights.append(f"Yield curve is {yc['signal']} with 10Y-2Y spread at {yc['spread']:.2f}%")
        
        return insights
    
    def _calculate_confidence(self, data: pd.DataFrame, trends: Dict[str, Any]) -> float:
        """Calculate confidence score based on data quality and consistency"""
        if data.empty:
            return 0.0
        
        # Base confidence on data availability
        data_coverage = data.count().sum() / (data.shape[0] * data.shape[1])
        
        # Adjust for trend consistency
        consistent_trends = sum(1 for t in trends.values() if t["trend_direction"] != "insufficient_data")
        trend_consistency = consistent_trends / max(len(trends), 1)
        
        # Combine factors
        confidence = (data_coverage * 0.6 + trend_consistency * 0.4)
        
        return min(confidence, 1.0)
    
    def _extract_key_metrics(
        self,
        data: pd.DataFrame,
        trends: Dict[str, Any],
        correlations: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract key metrics for the response"""
        metrics = {}
        
        # Data quality metrics
        metrics["data_points_analyzed"] = data.count().sum()
        metrics["indicators_analyzed"] = len(data.columns)
        metrics["strong_correlations_found"] = len(correlations.get("strong_correlations", []))
        
        # Trend metrics
        rising_count = sum(1 for t in trends.values() if t["trend_direction"] == "rising")
        falling_count = sum(1 for t in trends.values() if t["trend_direction"] == "falling")
        
        metrics["rising_indicators"] = rising_count
        metrics["falling_indicators"] = falling_count
        metrics["trend_balance"] = (rising_count - falling_count) / max(len(trends), 1)
        
        return metrics
    
    def _generate_cross_agent_signals(
        self,
        cycle: Dict[str, Any],
        trends: Dict[str, Any],
        policy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate signals for other agents"""
        signals = {}
        
        # Economic cycle signal
        signals["economic_cycle"] = cycle["cycle_phase"]
        
        # Risk environment based on economic conditions
        if cycle["cycle_phase"] in ["expansion", "peak"]:
            signals["risk_environment"] = "risk_on"
        elif cycle["cycle_phase"] in ["contraction", "trough"]:
            signals["risk_environment"] = "risk_off"
        else:
            signals["risk_environment"] = "neutral"
        
        # Policy signals for market analysis
        if "monetary_policy" in policy:
            signals["monetary_policy_stance"] = policy["monetary_policy"]["stance"]
        
        if "yield_curve" in policy:
            signals["yield_curve_signal"] = policy["yield_curve"]["signal"]
        
        # Inflation pressure signal
        inflation_indicators = ["CPIAUCSL", "CPILFESL", "PCEPI"]
        rising_inflation = sum(1 for ind in inflation_indicators 
                             if ind in trends and trends[ind]["trend_direction"] == "rising")
        
        if rising_inflation >= 2:
            signals["inflation_pressure"] = "high"
        elif rising_inflation == 1:
            signals["inflation_pressure"] = "moderate"
        else:
            signals["inflation_pressure"] = "low"
        
        return signals
    
    def _identify_uncertainty_factors(self, data: pd.DataFrame) -> List[str]:
        """Identify factors that add uncertainty to the analysis"""
        factors = []
        
        if data.empty:
            factors.append("no_economic_data")
        elif data.shape[1] < 3:
            factors.append("limited_indicators")
        
        # Check for data recency
        if not data.empty:
            latest_date = data.index.max()
            if hasattr(latest_date, 'date'):
                latest_date = latest_date.date()
            days_old = (datetime.now().date() - latest_date).days
            if days_old > 30:
                factors.append("stale_data")
        
        # Check for data gaps
        if not data.empty:
            missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
            if missing_ratio > 0.2:
                factors.append("data_gaps")
        
        return factors
    
    def _create_error_response(self, error_message: str) -> AgentResponse:
        """Create error response"""
        return AgentResponse(
            agent_type=self.agent_type,
            confidence=0.0,
            confidence_level=ConfidenceLevel.LOW,
            analysis={"error": error_message},
            insights=[f"Economic analysis unavailable: {error_message}"],
            key_metrics={},
            data_sources_used=["fred"],
            timeframe_analyzed="unknown",
            execution_time_ms=0.0,
            uncertainty_factors=["analysis_error"]
        )
    
    def get_required_data_sources(self, context: AgentContext) -> List[str]:
        """Return required data sources for economic analysis"""
        return ["fred"]

"""
Economic Analysis Agent - Refactored with modular components for frequency-aware analysis
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

from .base_agent import BaseAgent, AgentResponse, AgentContext
from .ai_service import AIService
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
        super().__init__("economic")
        
        # Initialize modular components
        self.data_handler = EconomicDataHandler()
        self.indicators_analyzer = EconomicIndicators()
        self.context_builder = EconomicContextBuilder()
        self.ai_service = AIService()
        
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
            
            # Generate cross-agent signals
            response.cross_agent_signals = self.indicators_analyzer.generate_cross_agent_signals(trends, cycle_assessment)
            
            return response
            
        except Exception as e:
            logger.error(f"Economic analysis failed: {str(e)}")
            return self._create_error_response(f"Analysis failed: {str(e)}", context)
    
    def _calculate_date_range(self, timeframe: str) -> tuple[datetime, datetime]:
        """Calculate appropriate date range for analysis"""
        end_date = datetime.now()
        
        # Use longer lookback periods to ensure sufficient data for frequency-aware analysis
        if timeframe == "1d":
            start_date = end_date - timedelta(days=30)  # 1 month for daily context
        elif timeframe == "1w":
            start_date = end_date - timedelta(days=90)  # 3 months for weekly context
        elif timeframe == "1m":
            start_date = end_date - timedelta(days=180) # 6 months for monthly context
        elif timeframe == "3m":
            start_date = end_date - timedelta(days=365) # 1 year for quarterly context
        elif timeframe == "6m":
            start_date = end_date - timedelta(days=730) # 2 years for semi-annual context
        elif timeframe == "1y":
            start_date = end_date - timedelta(days=1095) # 3 years for annual context
        else:
            start_date = end_date - timedelta(days=365)  # Default 1 year
        
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
            
            # Get AI analysis
            ai_response = self.ai_service.generate_economic_analysis(prompt)
            
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
            # Split AI analysis into sentences for better formatting
            analysis_text = ai_analysis["analysis"]
            sentences = [s.strip() + "." for s in analysis_text.split(".") if s.strip()]
            insights.extend(sentences[:10])  # Limit to top 10 insights
        else:
            insights.append(f"Economic analysis unavailable: {ai_analysis.get('analysis', 'Unknown error')}")
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Determine confidence level
        ai_confidence = ai_analysis.get("confidence", 0.0)
        cycle_confidence = cycle_assessment.get("confidence", 0.0)
        overall_confidence = (ai_confidence + cycle_confidence) / 2
        
        return AgentResponse(
            agent_type="economic",
            confidence=overall_confidence,
            execution_time_ms=execution_time * 1000,
            summary=f"Economic analysis complete with {overall_confidence:.1%} confidence",
            detailed_analysis={
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
        return AgentResponse(
            agent_type="economic",
            confidence=0.0,
            execution_time_ms=0.0,
            summary=f"Economic analysis failed: {error_message}",
            detailed_analysis={"error": error_message},
            insights=[f"Economic analysis unavailable: {error_message}"],
            key_metrics={},
            data_sources_used=["fred"],
            timeframe_analyzed=context.timeframe,
            uncertainty_factors=["analysis_error"]
        )
        
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
            economic_data, data_start_date, data_end_date = await self._fetch_economic_data(context)
            
            if economic_data.empty:
                return self._create_error_response("No economic data available for analysis period")
            
            # Perform statistical analysis (data preparation)
            trend_analysis = self._analyze_trends(economic_data, context)
            correlation_analysis = self._analyze_correlations(economic_data)
            cycle_assessment = self._assess_economic_cycle(economic_data)
            policy_impact = self._assess_policy_impact(economic_data, context)
            
            # Enrich context with historical comparisons and economic regime data
            enriched_context = self._create_enriched_context(
                context, economic_data, cycle_assessment, policy_impact, 
                data_start_date, data_end_date
            )
            
            # Use AI for intelligent interpretation and insights with enhanced context
            ai_analysis = await self.ai_service.analyze_economic_data(
                indicators_data=economic_data.to_dict() if not economic_data.empty else {},
                trends=trend_analysis,
                correlations=correlation_analysis,
                context=enriched_context
            )
            
            # Extract insights from AI analysis
            insights = ai_analysis.key_points if ai_analysis.key_points else ai_analysis.reasoning
            confidence = ai_analysis.confidence
            
            # Prepare key metrics
            key_metrics = self._extract_key_metrics(
                economic_data, trends=trend_analysis, correlations=correlation_analysis
            )
            
            # Generate signals for other agents
            signals = self._generate_cross_agent_signals(
                cycle_assessment, trends=trend_analysis, policy_impact=policy_impact
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
            )
            
            # Generate cross-agent signals
            response.cross_agent_signals = self.indicators_analyzer.generate_cross_agent_signals(trends, cycle_assessment)
            
            return response
            
        except Exception as e:
            logger.error(f"Economic analysis failed: {str(e)}")
            return self._create_error_response(f"Analysis failed: {str(e)}", context)
    
    def _calculate_date_range(self, timeframe: str) -> tuple[datetime, datetime]:
        """Calculate appropriate date range for analysis"""
        end_date = datetime.now()
        
        # Use longer lookback periods to ensure sufficient data for frequency-aware analysis
        if timeframe == "1d":
            start_date = end_date - timedelta(days=30)  # 1 month for daily context
        elif timeframe == "1w":
            start_date = end_date - timedelta(days=90)  # 3 months for weekly context
        elif timeframe == "1m":
            start_date = end_date - timedelta(days=180) # 6 months for monthly context
        elif timeframe == "3m":
            start_date = end_date - timedelta(days=365) # 1 year for quarterly context
        elif timeframe == "6m":
            start_date = end_date - timedelta(days=730) # 2 years for semi-annual context
        elif timeframe == "1y":
            start_date = end_date - timedelta(days=1095) # 3 years for annual context
        else:
            start_date = end_date - timedelta(days=365)  # Default 1 year
        
        return start_date, end_date
    
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
        """Analyze trends in economic indicators with multiple timeframes"""
        trends = {}
        
        for column in data.columns:
            series = data[column].dropna()
            if len(series) < 2:
                continue
            
            # Convert series to float to handle Decimal types
            series_float = series.astype(float)
            
            # Calculate multiple timeframe changes
            recent_value = float(series.iloc[-1])
            
            # 1-period change
            previous_value = float(series.iloc[-2]) if len(series) > 1 else recent_value
            change_1period_percent = ((recent_value - previous_value) / previous_value * 100) if previous_value != 0 else 0
            
            # 3-period change (quarterly-like)
            if len(series) >= 4:
                value_3periods_ago = float(series.iloc[-4])
                change_3period_percent = ((recent_value - value_3periods_ago) / value_3periods_ago * 100) if value_3periods_ago != 0 else 0
            else:
                change_3period_percent = change_1period_percent
            
            # 6-period change (semi-annual-like)
            if len(series) >= 7:
                value_6periods_ago = float(series.iloc[-7])
                change_6period_percent = ((recent_value - value_6periods_ago) / value_6periods_ago * 100) if value_6periods_ago != 0 else 0
            else:
                change_6period_percent = change_3period_percent
            
            # Year-over-year change (approximate)
            if len(series) >= 13:  # Assuming monthly data
                value_12periods_ago = float(series.iloc[-13])
                change_yoy_percent = ((recent_value - value_12periods_ago) / value_12periods_ago * 100) if value_12periods_ago != 0 else 0
            else:
                change_yoy_percent = change_6period_percent
            
            # Multi-timeframe trend analysis
            trend_analysis = self._calculate_multi_timeframe_trends(series_float)
            
            # Determine overall trend direction using multiple signals
            trend_direction = self._determine_trend_direction(
                trend_analysis, change_1period_percent, change_3period_percent, change_6period_percent
            )
            
            # Calculate volatility and momentum
            if len(series) > 1:
                volatility = series_float.pct_change().std() * 100
                momentum = self._calculate_momentum(series_float)
            else:
                volatility = 0
                momentum = 0
            
            trends[column] = {
                "current_value": recent_value,
                "previous_value": previous_value,
                "change_1period_percent": change_1period_percent,
                "change_3period_percent": change_3period_percent,
                "change_6period_percent": change_6period_percent,
                "change_yoy_percent": change_yoy_percent,
                "trend_direction": trend_direction,
                "trend_strength": trend_analysis["trend_strength"],
                "trend_consistency": trend_analysis["consistency"],
                "volatility": volatility,
                "momentum": momentum,
                "data_points": len(series),
                "timeframe_analysis": trend_analysis
            }
        
        return trends
    
    def _calculate_multi_timeframe_trends(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate trends across multiple timeframes with proper thresholds"""
        trends = {}
        
        # Short-term trend (last 3 periods)
        if len(series) >= 3:
            short_trend = np.polyfit(range(3), series[-3:], 1)[0]
            # Lower threshold for economic data - 0.001 instead of 0.01
            short_direction = "rising" if short_trend > 0.001 else "falling" if short_trend < -0.001 else "stable"
            trends["short_term"] = {"slope": short_trend, "direction": short_direction}
        
        # Medium-term trend (last 6 periods)
        if len(series) >= 6:
            medium_trend = np.polyfit(range(6), series[-6:], 1)[0]
            medium_direction = "rising" if medium_trend > 0.0005 else "falling" if medium_trend < -0.0005 else "stable"
            trends["medium_term"] = {"slope": medium_trend, "direction": medium_direction}
        
        # Long-term trend (last 12 periods or available data)
        lookback = min(12, len(series))
        if lookback >= 6:
            long_trend = np.polyfit(range(lookback), series[-lookback:], 1)[0]
            long_direction = "rising" if long_trend > 0.0002 else "falling" if long_trend < -0.0002 else "stable"
            trends["long_term"] = {"slope": long_trend, "direction": long_direction}
        
        # Calculate trend strength and consistency
        slopes = [t["slope"] for t in trends.values()]
        if slopes:
            avg_slope = np.mean(slopes)
            trend_strength = abs(avg_slope) * 1000  # Scale for readability
            
            # Consistency: how aligned are the different timeframe trends?
            directions = [t["direction"] for t in trends.values()]
            consistency = len([d for d in directions if d == directions[0]]) / len(directions)
        else:
            trend_strength = 0
            consistency = 0
        
        return {
            "timeframes": trends,
            "trend_strength": trend_strength,
            "consistency": consistency
        }
    
    def _determine_trend_direction(self, trend_analysis: Dict, change_1p: float, change_3p: float, change_6p: float) -> str:
        """Determine overall trend direction using multiple signals"""
        # Get timeframe directions
        timeframes = trend_analysis.get("timeframes", {})
        directions = [tf["direction"] for tf in timeframes.values()]
        
        # Count directional signals
        rising_signals = directions.count("rising")
        falling_signals = directions.count("falling")
        stable_signals = directions.count("stable")
        
        # Add change-based signals (lower thresholds for economic data)
        if abs(change_1p) > 0.1:  # 0.1% threshold instead of 1%
            if change_1p > 0:
                rising_signals += 1
            else:
                falling_signals += 1
        
        if abs(change_3p) > 0.2:  # 0.2% threshold for 3-period
            if change_3p > 0:
                rising_signals += 1
            else:
                falling_signals += 1
        
        if abs(change_6p) > 0.3:  # 0.3% threshold for 6-period
            if change_6p > 0:
                rising_signals += 1
            else:
                falling_signals += 1
        
        # Determine overall direction
        total_signals = rising_signals + falling_signals + stable_signals
        if total_signals == 0:
            return "insufficient_data"
        
        # Need majority consensus for directional call
        if rising_signals > falling_signals and rising_signals >= total_signals * 0.6:
            return "rising"
        elif falling_signals > rising_signals and falling_signals >= total_signals * 0.6:
            return "falling"
        else:
            return "stable"
    
    def _calculate_momentum(self, series: pd.Series) -> float:
        """Calculate momentum indicator"""
        if len(series) < 4:
            return 0
        
        # Rate of change acceleration
        recent_roc = (series.iloc[-1] - series.iloc[-2]) / series.iloc[-2] if series.iloc[-2] != 0 else 0
        previous_roc = (series.iloc[-2] - series.iloc[-3]) / series.iloc[-3] if series.iloc[-3] != 0 else 0
        
        momentum = (recent_roc - previous_roc) * 100
        return momentum
    
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
        """Assess current economic cycle phase with enhanced granularity"""
        if data.empty:
            return {"cycle_phase": "unknown", "confidence": 0.0, "cycle_details": {}}
        
        try:
            # Calculate weighted composite scores with multiple timeframes
            cycle_analysis = self._calculate_cycle_indicators(data)
            
            # Enhanced cycle phase determination with granular phases
            phase_analysis = self._determine_cycle_phase(cycle_analysis)
            
            # Calculate cycle momentum and duration estimates
            cycle_momentum = self._calculate_cycle_momentum(data)
            
            return {
                "cycle_phase": phase_analysis["phase"],
                "cycle_subphase": phase_analysis["subphase"],
                "confidence": phase_analysis["confidence"],
                "leading_score": cycle_analysis["leading_score"],
                "coincident_score": cycle_analysis["coincident_score"],
                "lagging_score": cycle_analysis["lagging_score"],
                "indicators_used": cycle_analysis["indicators_used"],
                "cycle_momentum": cycle_momentum,
                "cycle_details": {
                    "leading_indicators": cycle_analysis["leading_details"],
                    "coincident_indicators": cycle_analysis["coincident_details"],
                    "lagging_indicators": cycle_analysis["lagging_details"],
                    "signal_strength": cycle_analysis["signal_strength"],
                    "consensus_strength": cycle_analysis["consensus_strength"]
                }
            }
            
        except Exception as e:
            logger.error(f"Economic cycle assessment failed: {str(e)}")
            return {"cycle_phase": "unknown", "confidence": 0.0, "cycle_details": {}}
    
    def _calculate_cycle_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate weighted cycle indicator scores with multiple timeframes"""
        # Indicator weights based on economic significance
        indicator_weights = {
            # Leading indicators
            "HOUST": 1.2,  # Housing starts - strong leading indicator
            "PERMIT": 1.1,  # Building permits
            "VIXCLS": 0.9,  # Market volatility
            "T10Y3M": 1.3,  # Yield curve - very strong predictor
            
            # Coincident indicators  
            "PAYEMS": 1.4,  # Employment - strongest coincident
            "AHETPI": 1.0,  # Earnings
            
            # Lagging indicators
            "UNRATE": 1.2,  # Unemployment rate
            "CPIAUCSL": 1.0  # Inflation
        }
        
        leading_score = 0.0
        coincident_score = 0.0
        lagging_score = 0.0
        leading_details = {}
        coincident_details = {}
        lagging_details = {}
        total_weight = 0.0
        
        # Leading indicators analysis
        for indicator in self.cycle_indicators["leading"]:
            if indicator in data.columns:
                analysis = self._analyze_indicator_cycle_signal(data[indicator], indicator)
                if analysis["valid"]:
                    weight = indicator_weights.get(indicator, 1.0)
                    weighted_score = analysis["cycle_signal"] * weight
                    leading_score += weighted_score
                    total_weight += weight
                    leading_details[indicator] = analysis
        
        # Coincident indicators analysis
        for indicator in self.cycle_indicators["coincident"]:
            if indicator in data.columns:
                analysis = self._analyze_indicator_cycle_signal(data[indicator], indicator)
                if analysis["valid"]:
                    weight = indicator_weights.get(indicator, 1.0)
                    weighted_score = analysis["cycle_signal"] * weight
                    coincident_score += weighted_score
                    total_weight += weight
                    coincident_details[indicator] = analysis
        
        # Lagging indicators analysis
        for indicator in self.cycle_indicators["lagging"]:
            if indicator in data.columns:
                analysis = self._analyze_indicator_cycle_signal(data[indicator], indicator)
                if analysis["valid"]:
                    weight = indicator_weights.get(indicator, 1.0)
                    # Unemployment is inverse indicator
                    if indicator == "UNRATE":
                        weighted_score = -analysis["cycle_signal"] * weight
                    else:
                        weighted_score = analysis["cycle_signal"] * weight
                    lagging_score += weighted_score
                    total_weight += weight
                    lagging_details[indicator] = analysis
        
        # Normalize scores by total weight
        if total_weight > 0:
            leading_score /= (total_weight / 3)  # Normalize assuming roughly equal distribution
            coincident_score /= (total_weight / 3)
            lagging_score /= (total_weight / 3)
        
        # Calculate signal strength and consensus
        signal_strength = abs(leading_score) + abs(coincident_score) + abs(lagging_score)
        
        # Consensus: how aligned are the different indicator types?
        scores = [leading_score, coincident_score, lagging_score]
        positive_scores = sum(1 for s in scores if s > 0.001)
        negative_scores = sum(1 for s in scores if s < -0.001)
        consensus_strength = max(positive_scores, negative_scores) / len(scores)
        
        return {
            "leading_score": leading_score,
            "coincident_score": coincident_score,
            "lagging_score": lagging_score,
            "leading_details": leading_details,
            "coincident_details": coincident_details,
            "lagging_details": lagging_details,
            "indicators_used": len(leading_details) + len(coincident_details) + len(lagging_details),
            "signal_strength": signal_strength,
            "consensus_strength": consensus_strength
        }
    
    def _analyze_indicator_cycle_signal(self, series: pd.Series, indicator: str) -> Dict[str, Any]:
        """Analyze individual indicator for cycle signals"""
        series = series.dropna()
        if len(series) < 6:  # Need sufficient data
            return {"valid": False}
        
        # Convert to float
        series_float = series.astype(float)
        
        # Multiple timeframe analysis
        # 1-month change
        change_1m = (series_float.iloc[-1] - series_float.iloc[-2]) / series_float.iloc[-2] if series_float.iloc[-2] != 0 else 0
        
        # 3-month change
        if len(series_float) >= 4:
            change_3m = (series_float.iloc[-1] - series_float.iloc[-4]) / series_float.iloc[-4] if series_float.iloc[-4] != 0 else 0
        else:
            change_3m = change_1m
        
        # 6-month change
        if len(series_float) >= 7:
            change_6m = (series_float.iloc[-1] - series_float.iloc[-7]) / series_float.iloc[-7] if series_float.iloc[-7] != 0 else 0
        else:
            change_6m = change_3m
        
        # Trend analysis
        lookback = min(6, len(series_float))
        trend_slope = np.polyfit(range(lookback), series_float[-lookback:], 1)[0]
        
        # Combine signals with lower thresholds
        signals = []
        
        # 1-month signal (threshold: 0.1%)
        if abs(change_1m) > 0.001:
            signals.append(1 if change_1m > 0 else -1)
        
        # 3-month signal (threshold: 0.2%)
        if abs(change_3m) > 0.002:
            signals.append(1 if change_3m > 0 else -1)
        
        # 6-month signal (threshold: 0.3%)
        if abs(change_6m) > 0.003:
            signals.append(1 if change_6m > 0 else -1)
        
        # Trend signal (much lower threshold)
        if abs(trend_slope) > 0.0001:  # Very sensitive to economic changes
            signals.append(1 if trend_slope > 0 else -1)
        
        # Calculate composite cycle signal
        if signals:
            cycle_signal = sum(signals) / len(signals)
        else:
            cycle_signal = 0
        
        return {
            "valid": True,
            "cycle_signal": cycle_signal,
            "change_1m": change_1m * 100,  # Convert to percentage
            "change_3m": change_3m * 100,
            "change_6m": change_6m * 100,
            "trend_slope": trend_slope,
            "signal_strength": abs(cycle_signal),
            "signals_count": len(signals)
        }
    
    def _determine_cycle_phase(self, cycle_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Determine cycle phase with granular subphases"""
        leading = cycle_analysis["leading_score"]
        coincident = cycle_analysis["coincident_score"]
        lagging = cycle_analysis["lagging_score"]
        signal_strength = cycle_analysis["signal_strength"]
        consensus = cycle_analysis["consensus_strength"]
        indicators_used = cycle_analysis["indicators_used"]
        
        # Base confidence on data availability and consensus
        base_confidence = min(0.9, indicators_used / 8.0) * consensus
        
        # Much lower thresholds for economic cycle detection
        expansion_threshold = 0.001  # 0.1% instead of 1%
        contraction_threshold = -0.001  # -0.1% instead of -1%
        transition_threshold = 0.0005  # 0.05% for transitions
        
        # Determine primary phase and subphase
        if leading > expansion_threshold and coincident > expansion_threshold:
            if lagging > expansion_threshold:
                phase = "expansion"
                subphase = "mature_expansion"
                confidence = min(0.9, base_confidence * (1 + signal_strength * 2))
            else:
                phase = "expansion"
                subphase = "early_expansion"
                confidence = min(0.85, base_confidence * (1 + signal_strength * 1.5))
                
        elif leading < contraction_threshold and coincident < contraction_threshold:
            if lagging < contraction_threshold:
                phase = "contraction"
                subphase = "deep_contraction"
                confidence = min(0.9, base_confidence * (1 + signal_strength * 2))
            else:
                phase = "contraction"
                subphase = "early_contraction"
                confidence = min(0.85, base_confidence * (1 + signal_strength * 1.5))
                
        elif leading < -transition_threshold and coincident > transition_threshold:
            phase = "peak"
            if abs(leading) > abs(coincident):
                subphase = "late_peak"
            else:
                subphase = "early_peak"
            confidence = min(0.8, base_confidence * (1 + abs(leading - coincident) * 3))
            
        elif leading > transition_threshold and coincident < -transition_threshold:
            phase = "trough"
            if leading > abs(coincident):
                subphase = "late_trough"
            else:
                subphase = "early_trough"
            confidence = min(0.8, base_confidence * (1 + abs(leading - coincident) * 3))
            
        else:
            # More nuanced stable phases
            if signal_strength < 0.001:
                phase = "stable"
                subphase = "equilibrium"
            elif leading > coincident:
                phase = "stable"
                subphase = "building_momentum"
            else:
                phase = "stable"
                subphase = "consolidating"
            confidence = min(0.7, base_confidence)
        
        return {
            "phase": phase,
            "subphase": subphase,
            "confidence": confidence
        }
    
    def _calculate_cycle_momentum(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate economic cycle momentum indicators"""
        momentum_indicators = {}
        
        # Key momentum indicators
        key_series = ["PAYEMS", "HOUST", "CPIAUCSL", "FEDFUNDS"]
        
        for indicator in key_series:
            if indicator in data.columns:
                series = data[indicator].dropna().astype(float)
                if len(series) >= 4:
                    # Rate of change momentum
                    recent_change = (series.iloc[-1] - series.iloc[-2]) / series.iloc[-2] if series.iloc[-2] != 0 else 0
                    previous_change = (series.iloc[-2] - series.iloc[-3]) / series.iloc[-3] if series.iloc[-3] != 0 else 0
                    momentum = (recent_change - previous_change) * 100
                    
                    momentum_indicators[indicator] = {
                        "momentum": momentum,
                        "accelerating": momentum > 0.01,
                        "decelerating": momentum < -0.01
                    }
        
        # Overall momentum assessment
        if momentum_indicators:
            avg_momentum = np.mean([m["momentum"] for m in momentum_indicators.values()])
            accelerating_count = sum(1 for m in momentum_indicators.values() if m["accelerating"])
            decelerating_count = sum(1 for m in momentum_indicators.values() if m["decelerating"])
            
            if accelerating_count > decelerating_count:
                overall_momentum = "accelerating"
            elif decelerating_count > accelerating_count:
                overall_momentum = "decelerating"
            else:
                overall_momentum = "stable"
        else:
            avg_momentum = 0
            overall_momentum = "unknown"
        
        return {
            "overall_momentum": overall_momentum,
            "average_momentum": avg_momentum,
            "indicator_details": momentum_indicators
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
        policy_impact: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate signals for other agents"""
        signals = {}
        
        # Economic cycle signal
        signals["economic_cycle"] = cycle["cycle_phase"]
        
        # Risk environment based on economic conditions
        if cycle["cycle_phase"] in ["expansion", "peak"]:
            signals["risk_environment"] = "risk_on"
        # Monetary policy stance
        if "FEDFUNDS" in trends:
            fed_trend = trends["FEDFUNDS"]["trend_direction"]
            if fed_trend == "rising":
                signals["monetary_policy_stance"] = "tightening"
            elif fed_trend == "falling":
                signals["monetary_policy_stance"] = "easing"
            else:
                signals["monetary_policy_stance"] = "neutral"
        
        # Yield curve signal
        if "yield_curve" in policy_impact:
            curve_signal = policy_impact["yield_curve"]["signal"]
            signals["yield_curve_signal"] = curve_signal
        
        # Inflation pressure
        if "CPIAUCSL" in trends:
            cpi_trend = trends["CPIAUCSL"]["trend_direction"]
            if cpi_trend == "rising":
                signals["inflation_pressure"] = "high"
            elif cpi_trend == "falling":
                signals["inflation_pressure"] = "low"
            else:
                signals["inflation_pressure"] = "moderate"
        
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
    
    def _create_enriched_context(
        self, 
        context: AgentContext, 
        economic_data: pd.DataFrame, 
        cycle_assessment: Dict[str, Any], 
        policy_impact: Dict[str, Any],
        data_start_date: datetime,
        data_end_date: datetime
    ) -> str:
        """Create enriched context with historical comparisons and economic regime data"""
        
        # Base context
        enriched_parts = [
            f"Query: {context.query}",
            f"Timeframe: {context.timeframe}",
            f"Data period: {data_start_date.strftime('%Y-%m-%d')} to {data_end_date.strftime('%Y-%m-%d')}",
            f"Analysis date: {datetime.now().strftime('%Y-%m-%d')}"
        ]
        
        # Economic cycle context with subphase
        cycle_phase = cycle_assessment.get('cycle_phase', 'unknown')
        cycle_subphase = cycle_assessment.get('cycle_subphase', '')
        cycle_confidence = cycle_assessment.get('confidence', 0.0)
        
        if cycle_subphase:
            enriched_parts.append(f"Economic cycle: {cycle_phase} ({cycle_subphase}) - confidence: {cycle_confidence:.1%}")
        else:
            enriched_parts.append(f"Economic cycle: {cycle_phase} - confidence: {cycle_confidence:.1%}")
        
        # Policy context with specific details
        if policy_impact:
            policy_details = []
            if "monetary_policy" in policy_impact:
                mp = policy_impact["monetary_policy"]
                policy_details.append(f"Fed funds rate: {mp['current_rate']:.2f}% ({mp['stance']} stance)")
            
            if "yield_curve" in policy_impact:
                yc = policy_impact["yield_curve"]
                policy_details.append(f"Yield curve: {yc['signal']} (10Y-2Y spread: {yc['spread']:.2f}%)")
            
            if policy_details:
                enriched_parts.append(f"Policy context: {'; '.join(policy_details)}")
        
        # Historical comparisons
        historical_context = self._generate_historical_context(economic_data)
        if historical_context:
            enriched_parts.append(f"Historical context: {historical_context}")
        
        # Economic regime assessment
        regime_context = self._assess_economic_regime(economic_data, cycle_assessment, policy_impact)
        if regime_context:
            enriched_parts.append(f"Economic regime: {regime_context}")
        
        # Data quality and coverage
        data_quality = self._assess_data_quality(economic_data, data_end_date)
        enriched_parts.append(f"Data quality: {data_quality}")
        
        return ". ".join(enriched_parts) + "."
    
    def _generate_historical_context(self, economic_data: pd.DataFrame) -> str:
        """Generate historical context and comparisons"""
        if economic_data.empty:
            return ""
        
        context_parts = []
        
        # Key indicators for historical context
        key_indicators = ["UNRATE", "CPIAUCSL", "FEDFUNDS", "DGS10"]
        
        for indicator in key_indicators:
            if indicator in economic_data.columns:
                series = economic_data[indicator].dropna()
                if len(series) >= 12:  # Need sufficient history
                    current = float(series.iloc[-1])
                    
                    # Historical percentiles
                    percentile_25 = series.quantile(0.25)
                    percentile_75 = series.quantile(0.75)
                    historical_max = series.max()
                    historical_min = series.min()
                    
                    # Determine relative position
                    if current >= percentile_75:
                        position = "elevated"
                    elif current <= percentile_25:
                        position = "low"
                    else:
                        position = "moderate"
                    
                    # Add context for key indicators
                    if indicator == "UNRATE":
                        context_parts.append(f"unemployment at {current:.1f}% ({position} vs historical range {historical_min:.1f}%-{historical_max:.1f}%)")
                    elif indicator == "CPIAUCSL":
                        yoy_inflation = ((current - float(series.iloc[-13])) / float(series.iloc[-13]) * 100) if len(series) >= 13 else 0
                        context_parts.append(f"inflation at {yoy_inflation:.1f}% YoY ({position} vs historical trends)")
                    elif indicator == "FEDFUNDS":
                        context_parts.append(f"fed funds at {current:.2f}% ({position} vs range {historical_min:.1f}%-{historical_max:.1f}%)")
                    elif indicator == "DGS10":
                        context_parts.append(f"10Y treasury at {current:.2f}% ({position} level)")
        
        return "; ".join(context_parts[:3])  # Top 3 for brevity
    
    def _assess_economic_regime(
        self, 
        economic_data: pd.DataFrame, 
        cycle_assessment: Dict[str, Any], 
        policy_impact: Dict[str, Any]
    ) -> str:
        """Assess current economic regime for better AI context"""
        
        regime_factors = []
        
        # Growth regime
        cycle_phase = cycle_assessment.get('cycle_phase', 'unknown')
        if cycle_phase in ['expansion', 'early_expansion']:
            regime_factors.append("growth-supportive")
        elif cycle_phase in ['contraction', 'early_contraction']:
            regime_factors.append("contractionary")
        elif cycle_phase in ['peak', 'late_peak']:
            regime_factors.append("late-cycle")
        elif cycle_phase in ['trough', 'late_trough']:
            regime_factors.append("early-recovery")
        
        # Inflation regime
        if "CPIAUCSL" in economic_data.columns:
            cpi_series = economic_data["CPIAUCSL"].dropna()
            if len(cpi_series) >= 13:
                current_inflation = ((float(cpi_series.iloc[-1]) - float(cpi_series.iloc[-13])) / float(cpi_series.iloc[-13]) * 100)
                if current_inflation > 4:
                    regime_factors.append("high-inflation")
                elif current_inflation > 2.5:
                    regime_factors.append("elevated-inflation")
                elif current_inflation < 1:
                    regime_factors.append("low-inflation")
                else:
                    regime_factors.append("moderate-inflation")
        
        # Policy regime
        if policy_impact and "monetary_policy" in policy_impact:
            stance = policy_impact["monetary_policy"]["stance"]
            if stance == "tightening":
                regime_factors.append("restrictive-policy")
            elif stance == "easing":
                regime_factors.append("accommodative-policy")
            else:
                regime_factors.append("neutral-policy")
        
        # Volatility regime
        if not economic_data.empty:
            # Calculate average volatility across key indicators
            volatilities = []
            for col in ["VIXCLS", "DGS10", "FEDFUNDS"]:
                if col in economic_data.columns:
                    series = economic_data[col].dropna().astype(float)
                    if len(series) > 1:
                        vol = series.pct_change().std() * 100
                        volatilities.append(vol)
            
            if volatilities:
                avg_vol = sum(volatilities) / len(volatilities)
                if avg_vol > 2:
                    regime_factors.append("high-volatility")
                elif avg_vol < 0.5:
                    regime_factors.append("low-volatility")
        
        return ", ".join(regime_factors) if regime_factors else "transitional"
    
    def _assess_data_quality(self, economic_data: pd.DataFrame, data_end_date: datetime) -> str:
        """Assess data quality and recency for AI context"""
        if economic_data.empty:
            return "no data available"
        
        quality_factors = []
        
        # Data recency
        latest_date = economic_data.index.max()
        if hasattr(latest_date, 'date'):
            latest_date = latest_date.date()
        else:
            latest_date = latest_date
            
        days_old = (datetime.now().date() - latest_date).days
        
        if days_old <= 7:
            quality_factors.append("very recent data")
        elif days_old <= 30:
            quality_factors.append("recent data")
        elif days_old <= 90:
            quality_factors.append("moderately recent data")
        else:
            quality_factors.append(f"data {days_old} days old")
        
        # Data completeness
        total_points = economic_data.shape[0] * economic_data.shape[1]
        missing_points = economic_data.isnull().sum().sum()
        completeness = (total_points - missing_points) / total_points
        
        if completeness > 0.95:
            quality_factors.append("high completeness")
        elif completeness > 0.85:
            quality_factors.append("good completeness")
        elif completeness > 0.70:
            quality_factors.append("moderate completeness")
        else:
            quality_factors.append("limited completeness")
        
        # Indicator coverage
        indicator_count = len(economic_data.columns)
        if indicator_count >= 10:
            quality_factors.append(f"comprehensive coverage ({indicator_count} indicators)")
        elif indicator_count >= 5:
            quality_factors.append(f"good coverage ({indicator_count} indicators)")
        else:
            quality_factors.append(f"limited coverage ({indicator_count} indicators)")
        
        return "; ".join(quality_factors)

    def get_required_data_sources(self, context: AgentContext) -> List[str]:
        """Return required data sources for economic analysis"""
        return ["fred"]

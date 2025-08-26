"""
Market Context Builder for Vector View Financial Intelligence Platform

Builds comprehensive context for market intelligence analysis including:
- News-market correlation analysis
- Market impact assessment
- Volatility forecasting
- Cross-agent signal preparation
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict

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


class MarketContextBuilder:
    """
    Builds comprehensive context for market intelligence analysis.
    
    Responsibilities:
    - News-market correlation analysis
    - Market impact assessment and forecasting
    - Context preparation for AI analysis
    - Cross-agent signal generation
    """
    
    def __init__(self):
        """Initialize the context builder."""
        
        # Market sectors for impact analysis
        self.market_sectors = {
            'technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
            'financial': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
            'healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK'],
            'energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG'],
            'consumer': ['WMT', 'PG', 'KO', 'PEP', 'COST']
        }
        
        # Sector sentiment multipliers
        self.sector_sentiment_multipliers = {
            'technology': 1.2,  # More sensitive to sentiment
            'financial': 1.1,
            'healthcare': 0.8,  # Less sensitive
            'energy': 1.0,
            'consumer': 0.9
        }
    
    async def analyze_news_market_correlation(
        self, 
        news_data: List[Dict], 
        market_data: pd.DataFrame
    ) -> NewsMarketCorrelation:
        """
        Analyze correlation between news sentiment and market movements.
        
        Args:
            news_data: List of news articles with sentiment scores
            market_data: Market data DataFrame
            
        Returns:
            NewsMarketCorrelation object with correlation analysis
        """
        try:
            if not news_data or market_data.empty:
                return NewsMarketCorrelation(
                    correlation_coefficient=0.0,
                    statistical_significance=0.0,
                    time_lag_hours=0,
                    sample_size=0,
                    correlation_strength="none"
                )
            
            # Aggregate news sentiment by day for more realistic correlation
            news_df = pd.DataFrame(news_data)
            news_df['timestamp'] = pd.to_datetime(news_df['timestamp'])
            news_df['date'] = news_df['timestamp'].dt.date
            
            daily_sentiment = news_df.groupby('date')['sentiment_score'].mean().reset_index()
            daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
            
            # Calculate market returns using SPY as primary indicator
            market_returns = self._calculate_market_returns(market_data)
            
            if market_returns.empty:
                return NewsMarketCorrelation(0.0, 0.0, 0, 0, "insufficient_data")
            
            # Merge sentiment and market data on daily basis
            merged_data = pd.merge(
                daily_sentiment, 
                market_returns, 
                left_on='date',
                right_index=True,
                how='inner'
            )
            
            if len(merged_data) < 3:  # Need at least 3 days for correlation
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
            return NewsMarketCorrelation(0.0, 0.0, 0, 0, "error")
    
    async def assess_market_impact(
        self,
        news_data: List[Dict],
        market_data: pd.DataFrame,
        correlation_analysis: NewsMarketCorrelation
    ) -> MarketImpactAnalysis:
        """
        Assess overall market impact from news events.
        
        Args:
            news_data: List of news articles with sentiment
            market_data: Market data DataFrame
            correlation_analysis: Correlation analysis results
            
        Returns:
            MarketImpactAnalysis object with impact assessment
        """
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
            market_volatility = self._calculate_market_volatility(market_data)
            
            # Predict price impact based on sentiment and correlation
            price_impact = self._predict_price_impact(
                avg_sentiment, correlation_analysis, sentiment_volatility
            )
            
            # Assess sector-specific impact
            sector_impact = self._assess_sector_impact(avg_sentiment, correlation_analysis)
            
            # Generate supporting evidence
            evidence = self._generate_supporting_evidence(
                avg_sentiment, correlation_analysis, market_volatility
            )
            
            # Calculate confidence based on data quality
            confidence = self._calculate_impact_confidence(
                correlation_analysis, len(news_data), len(market_data)
            )
            
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
    
    def build_analysis_context(
        self,
        news_data: List[Dict],
        market_data: pd.DataFrame,
        impact_analysis: MarketImpactAnalysis,
        correlation_analysis: NewsMarketCorrelation
    ) -> Dict[str, Any]:
        """
        Build comprehensive context for AI analysis.
        
        Args:
            news_data: News articles data
            market_data: Market data DataFrame
            impact_analysis: Market impact analysis results
            correlation_analysis: Correlation analysis results
            
        Returns:
            Dictionary with structured context for AI analysis
        """
        return {
            "data_summary": {
                "news_articles_count": len(news_data),
                "market_data_points": len(market_data),
                "analysis_timespan": self._calculate_analysis_timespan(news_data, market_data),
                "data_quality_score": self._assess_data_quality(news_data, market_data)
            },
            "sentiment_context": {
                "overall_sentiment": impact_analysis.news_sentiment_score,
                "sentiment_distribution": self._analyze_sentiment_distribution(news_data),
                "news_sources": list(set([art.get('source', 'unknown') for art in news_data]))
            },
            "market_context": {
                "correlation_strength": correlation_analysis.correlation_strength,
                "correlation_coefficient": correlation_analysis.correlation_coefficient,
                "statistical_significance": correlation_analysis.statistical_significance,
                "volatility_forecast": impact_analysis.volatility_forecast,
                "market_indicators_analyzed": list(market_data.columns) if not market_data.empty else []
            },
            "impact_assessment": {
                "price_impact_predictions": impact_analysis.price_impact_prediction,
                "sector_impacts": impact_analysis.sector_impact,
                "supporting_evidence": impact_analysis.supporting_evidence,
                "confidence_score": impact_analysis.confidence_score
            }
        }
    
    def generate_cross_agent_signals(
        self, 
        impact_analysis: MarketImpactAnalysis,
        correlation_analysis: NewsMarketCorrelation
    ) -> Dict[str, Any]:
        """
        Generate signals for other agents based on market analysis.
        
        Args:
            impact_analysis: Market impact analysis results
            correlation_analysis: Correlation analysis results
            
        Returns:
            Dictionary with cross-agent signals
        """
        return {
            'market_sentiment': self._classify_market_sentiment(impact_analysis.news_sentiment_score),
            'volatility_regime': self._classify_volatility_regime(impact_analysis.volatility_forecast),
            'news_market_correlation': correlation_analysis.correlation_strength,
            'market_stress_level': min(1.0, abs(impact_analysis.news_sentiment_score) + impact_analysis.volatility_forecast),
            'sector_rotation_signal': self._identify_sector_rotation(impact_analysis.sector_impact),
            'risk_environment': self._assess_risk_environment(impact_analysis, correlation_analysis)
        }
    
    def calculate_confidence(
        self, 
        correlation_analysis: NewsMarketCorrelation, 
        news_count: int,
        market_data_points: int
    ) -> float:
        """
        Calculate overall confidence score based on data quality and analysis reliability.
        
        Args:
            correlation_analysis: Correlation analysis results
            news_count: Number of news articles analyzed
            market_data_points: Number of market data points
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Base confidence from correlation strength
        correlation_confidence = correlation_analysis.statistical_significance
        
        # Data quality confidence - proper quality thresholds
        sample_confidence = min(1.0, correlation_analysis.sample_size / 50)  # Require substantial sample
        news_confidence = min(1.0, news_count / 100)  # Require many news articles
        market_confidence = min(1.0, market_data_points / 500)  # Require substantial market data
        
        # Combined confidence
        overall_confidence = (
            correlation_confidence * 0.4 +
            sample_confidence * 0.2 +
            news_confidence * 0.2 +
            market_confidence * 0.2
        )
        
        return max(0.0, min(1.0, overall_confidence))
    
    def _calculate_market_returns(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate market returns with hourly aggregation"""
        try:
            if market_data.empty:
                return pd.DataFrame()
            
            # Use SPY as primary market indicator, fallback to first available
            primary_indicator = 'SPY' if 'SPY' in market_data.columns else market_data.columns[0]
            
            # Calculate returns
            returns = market_data[primary_indicator].pct_change().dropna()
            
            # Aggregate to daily for correlation analysis
            daily_returns = returns.resample('D').last().dropna()
            
            return pd.DataFrame({'returns': daily_returns})
            
        except Exception as e:
            logger.error(f"Market returns calculation failed: {str(e)}")
            return pd.DataFrame()
    
    def _calculate_market_volatility(self, market_data: pd.DataFrame) -> float:
        """Calculate current market volatility"""
        try:
            if market_data.empty:
                return 0.0
            
            # Use SPY for volatility calculation
            primary_indicator = 'SPY' if 'SPY' in market_data.columns else market_data.columns[0]
            returns = market_data[primary_indicator].pct_change().dropna()
            
            if len(returns) < 2:
                return 0.0
            
            # Annualized volatility
            volatility = returns.std() * np.sqrt(252)
            return float(volatility) if not np.isnan(volatility) else 0.0
            
        except Exception as e:
            logger.error(f"Volatility calculation failed: {str(e)}")
            return 0.0
    
    def _predict_price_impact(
        self, 
        sentiment: float, 
        correlation: NewsMarketCorrelation, 
        sentiment_volatility: float
    ) -> Dict[str, float]:
        """Predict price impact for key indicators"""
        key_indicators = ['SPY', 'QQQ', 'TLT', 'GLD', 'VIXCLS']
        
        price_impact = {}
        for indicator in key_indicators:
            base_impact = sentiment * correlation.correlation_coefficient
            volatility_adjustment = sentiment_volatility * 0.1
            price_impact[indicator] = base_impact + volatility_adjustment
        
        return price_impact
    
    def _assess_sector_impact(self, sentiment: float, correlation: NewsMarketCorrelation) -> Dict[str, float]:
        """Assess sector-specific impact"""
        sector_impact = {}
        
        for sector, multiplier in self.sector_sentiment_multipliers.items():
            sector_impact[sector] = sentiment * multiplier * correlation.correlation_coefficient
        
        return sector_impact
    
    def _generate_supporting_evidence(
        self, 
        sentiment: float, 
        correlation: NewsMarketCorrelation, 
        volatility: float
    ) -> List[str]:
        """Generate supporting evidence for analysis"""
        evidence = []
        
        if correlation.correlation_strength == "strong":
            evidence.append(f"Strong correlation ({correlation.correlation_coefficient:.3f}) between news sentiment and market movements")
        
        if abs(sentiment) > 0.5:
            sentiment_direction = "positive" if sentiment > 0 else "negative"
            evidence.append(f"Significant {sentiment_direction} news sentiment ({sentiment:.3f})")
        
        if volatility > 0.2:
            evidence.append(f"Elevated market volatility detected ({volatility:.3f})")
        
        if correlation.sample_size >= 20:
            evidence.append(f"Analysis based on substantial sample size ({correlation.sample_size} data points)")
        
        return evidence
    
    def _calculate_impact_confidence(
        self, 
        correlation: NewsMarketCorrelation, 
        news_count: int, 
        market_points: int
    ) -> float:
        """Calculate confidence for impact analysis"""
        confidence_factors = [
            correlation.statistical_significance * 0.4,
            min(news_count / 10, 1.0) * 0.3,
            min(market_points / 100, 1.0) * 0.3
        ]
        
        return sum(confidence_factors)
    
    def _calculate_analysis_timespan(self, news_data: List[Dict], market_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate timespan of analysis"""
        try:
            timespan = {"news_span_hours": 0, "market_span_hours": 0}
            
            # News timespan
            if news_data:
                timestamps = []
                for article in news_data:
                    if article.get('timestamp'):
                        try:
                            ts = pd.to_datetime(article['timestamp'])
                            timestamps.append(ts)
                        except:
                            continue
                
                if timestamps:
                    timespan["news_span_hours"] = (max(timestamps) - min(timestamps)).total_seconds() / 3600
            
            # Market timespan
            if not market_data.empty:
                timespan["market_span_hours"] = (market_data.index.max() - market_data.index.min()).total_seconds() / 3600
            
            return timespan
        except:
            return {"news_span_hours": 0, "market_span_hours": 0}
    
    def _assess_data_quality(self, news_data: List[Dict], market_data: pd.DataFrame) -> float:
        """Assess overall data quality"""
        quality_factors = []
        
        # News data quality
        if news_data:
            valid_sentiment = sum(1 for art in news_data if art.get('sentiment_score') is not None)
            news_quality = valid_sentiment / len(news_data)
            quality_factors.append(news_quality * 0.5)
        
        # Market data quality
        if not market_data.empty:
            completeness = market_data.notna().mean().mean()
            quality_factors.append(completeness * 0.5)
        
        return sum(quality_factors) if quality_factors else 0.0
    
    def _analyze_sentiment_distribution(self, news_data: List[Dict]) -> Dict[str, float]:
        """Analyze distribution of sentiment scores"""
        if not news_data:
            return {"positive": 0, "neutral": 0, "negative": 0}
        
        sentiments = [art.get('sentiment_score', 0) for art in news_data]
        
        return {
            "positive": sum(1 for s in sentiments if s > 0.1) / len(sentiments),
            "neutral": sum(1 for s in sentiments if -0.1 <= s <= 0.1) / len(sentiments),
            "negative": sum(1 for s in sentiments if s < -0.1) / len(sentiments)
        }
    
    def _classify_market_sentiment(self, sentiment_score: float) -> str:
        """Classify market sentiment"""
        if sentiment_score > 0.2:
            return 'bullish'
        elif sentiment_score < -0.2:
            return 'bearish'
        else:
            return 'neutral'
    
    def _classify_volatility_regime(self, volatility_forecast: float) -> str:
        """Classify volatility regime"""
        return 'high' if volatility_forecast > 0.25 else 'low'
    
    def _identify_sector_rotation(self, sector_impact: Dict[str, float]) -> str:
        """Identify sector rotation signals"""
        if not sector_impact:
            return 'none'
        
        max_impact_sector = max(sector_impact.items(), key=lambda x: abs(x[1]))
        return max_impact_sector[0]
    
    def _assess_risk_environment(
        self, 
        impact_analysis: MarketImpactAnalysis, 
        correlation_analysis: NewsMarketCorrelation
    ) -> str:
        """Assess overall risk environment"""
        risk_factors = [
            abs(impact_analysis.news_sentiment_score),
            impact_analysis.volatility_forecast,
            1 - correlation_analysis.statistical_significance  # Lower correlation = higher uncertainty
        ]
        
        avg_risk = sum(risk_factors) / len(risk_factors)
        
        if avg_risk > 0.7:
            return 'high_risk'
        elif avg_risk > 0.4:
            return 'moderate_risk'
        else:
            return 'low_risk'

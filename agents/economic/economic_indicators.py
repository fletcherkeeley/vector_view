"""
Economic Indicators Analysis - Trend analysis and cycle detection
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class EconomicIndicators:
    """Handles economic trend analysis and cycle detection"""
    
    def __init__(self):
        # Economic indicator categories with significance weights
        self.indicator_weights = {
            # Employment indicators (high weight)
            'UNRATE': 0.25,
            'PAYEMS': 0.20,
            'ICSA': 0.15,
            
            # Inflation indicators (high weight)
            'CPIAUCSL': 0.20,
            'CPILFESL': 0.15,
            
            # Growth indicators (medium weight)
            'INDPRO': 0.15,
            'RETAILSMNSA': 0.10,
            'PERMIT': 0.10,
            'HOUST': 0.10,
            
            # Financial indicators (medium weight)
            'FEDFUNDS': 0.15,
            'DGS10': 0.10,
            'T10Y2Y': 0.10,
            'VIXCLS': 0.05,
            
            # Sentiment indicators (lower weight)
            'UMCSENT': 0.05
        }
        
        # Trend classification thresholds (frequency-aware)
        self.trend_thresholds = {
            'MONTHLY': {
                'strong_change': 0.5,
                'moderate_change': 0.2,
                'weak_change': 0.1
            },
            'WEEKLY': {
                'strong_change': 0.3,
                'moderate_change': 0.1,
                'weak_change': 0.05
            },
            'DAILY': {
                'strong_change': 0.2,
                'moderate_change': 0.05,
                'weak_change': 0.02
            }
        }
    
    def analyze_trends(self, df: pd.DataFrame, data_handler, indicators: List[str]) -> Dict[str, any]:
        """Analyze trends with frequency-appropriate timeframes"""
        trends = {}
        
        for indicator in indicators:
            if indicator not in df.columns:
                continue
                
            frequency = data_handler.frequency_map.get(indicator, 'MONTHLY')
            timeframes = data_handler.get_frequency_appropriate_timeframes(frequency)
            thresholds = self.trend_thresholds.get(frequency, self.trend_thresholds['MONTHLY'])
            
            # Calculate changes for different timeframes
            changes = {}
            for period_name, periods in timeframes.items():
                change = data_handler.calculate_frequency_appropriate_change(df, indicator, periods)
                changes[f'change_{period_name}'] = change
            
            # Classify trend direction and strength
            recent_change = changes['change_short']
            
            if abs(recent_change) >= thresholds['strong_change']:
                strength = 'strong'
            elif abs(recent_change) >= thresholds['moderate_change']:
                strength = 'moderate'
            elif abs(recent_change) >= thresholds['weak_change']:
                strength = 'weak'
            else:
                strength = 'stable'
            
            if recent_change > 0:
                direction = 'rising'
            elif recent_change < 0:
                direction = 'falling'
            else:
                direction = 'stable'
            
            trends[indicator] = {
                'direction': direction,
                'strength': strength,
                'changes': changes,
                'frequency': frequency,
                'latest_value': df[indicator].dropna().iloc[-1] if not df[indicator].dropna().empty else None
            }
        
        return trends
    
    def assess_economic_cycle(self, df: pd.DataFrame, data_handler, indicators: List[str]) -> Dict[str, any]:
        """Assess current economic cycle phase with weighted indicators"""
        
        # Define cycle indicators by category
        leading_indicators = ['PERMIT', 'UMCSENT', 'T10Y2Y', 'VIXCLS']
        coincident_indicators = ['PAYEMS', 'INDPRO', 'RETAILSMNSA']
        lagging_indicators = ['UNRATE', 'CPIAUCSL', 'FEDFUNDS']
        
        def calculate_weighted_momentum(indicator_list: List[str]) -> float:
            """Calculate weighted momentum for a group of indicators"""
            total_weight = 0
            weighted_sum = 0
            
            for indicator in indicator_list:
                if indicator not in df.columns or indicator not in indicators:
                    continue
                
                weight = self.indicator_weights.get(indicator, 0.05)
                frequency = data_handler.frequency_map.get(indicator, 'MONTHLY')
                timeframes = data_handler.get_frequency_appropriate_timeframes(frequency)
                
                # Use medium-term change for cycle analysis
                change = data_handler.calculate_frequency_appropriate_change(df, indicator, timeframes['medium'])
                
                weighted_sum += change * weight
                total_weight += weight
            
            return weighted_sum / total_weight if total_weight > 0 else 0
        
        # Calculate momentum for each category
        leading_momentum = calculate_weighted_momentum(leading_indicators)
        coincident_momentum = calculate_weighted_momentum(coincident_indicators)
        lagging_momentum = calculate_weighted_momentum(lagging_indicators)
        
        # Composite cycle score
        composite_score = (leading_momentum * 0.4 + coincident_momentum * 0.4 + lagging_momentum * 0.2)
        
        # Determine cycle phase with lower thresholds
        if composite_score > 0.3:
            if leading_momentum > coincident_momentum:
                phase = "early_expansion"
            else:
                phase = "mature_expansion"
        elif composite_score > 0.1:
            phase = "moderate_growth"
        elif composite_score > -0.1:
            phase = "stable"
        elif composite_score > -0.3:
            phase = "slowdown"
        else:
            if leading_momentum < coincident_momentum:
                phase = "early_contraction"
            else:
                phase = "deep_contraction"
        
        # Calculate confidence based on indicator agreement
        momentum_values = [leading_momentum, coincident_momentum, lagging_momentum]
        momentum_std = np.std(momentum_values) if len(momentum_values) > 1 else 0
        confidence = max(0.1, min(0.9, 1 - (momentum_std / 2)))  # Lower std = higher confidence
        
        return {
            'phase': phase,
            'composite_score': composite_score,
            'leading_momentum': leading_momentum,
            'coincident_momentum': coincident_momentum,
            'lagging_momentum': lagging_momentum,
            'confidence': confidence,
            'momentum_agreement': 1 - momentum_std
        }
    
    def calculate_correlations(self, df: pd.DataFrame, indicators: List[str], min_correlation: float = 0.7) -> List[Dict[str, any]]:
        """Calculate significant correlations between indicators"""
        correlations = []
        
        # Only use indicators that have sufficient data
        valid_indicators = [ind for ind in indicators if ind in df.columns and df[ind].notna().sum() > 10]
        
        if len(valid_indicators) < 2:
            return correlations
        
        # Calculate correlation matrix
        corr_matrix = df[valid_indicators].corr()
        
        # Find significant correlations
        for i, ind1 in enumerate(valid_indicators):
            for j, ind2 in enumerate(valid_indicators[i+1:], i+1):
                correlation = corr_matrix.loc[ind1, ind2]
                
                if not pd.isna(correlation) and abs(correlation) >= min_correlation:
                    correlations.append({
                        'indicator1': ind1,
                        'indicator2': ind2,
                        'correlation': correlation,
                        'strength': 'strong' if abs(correlation) >= 0.8 else 'moderate'
                    })
        
        return sorted(correlations, key=lambda x: abs(x['correlation']), reverse=True)
    
    def generate_cross_agent_signals(self, trends: Dict[str, any], cycle_assessment: Dict[str, any]) -> Dict[str, str]:
        # Generate cross-agent signals for orchestration with structured format
        signals = {}
        
        # Economic cycle signal with confidence and momentum
        cycle_phase = cycle_assessment.get('phase', 'stable')
        cycle_confidence = cycle_assessment.get('confidence', 0.5)
        cycle_momentum = cycle_assessment.get('momentum', 0.0)
        
        signals['economic_cycle'] = {
            'phase': cycle_phase,
            'confidence': float(cycle_confidence),
            'momentum': float(cycle_momentum),
            'signal_strength': 'high' if cycle_confidence > 0.7 else 'medium' if cycle_confidence > 0.4 else 'low'
        }
        
        # Inflation pressure signal with trend strength
        inflation_indicators = ['CPIAUCSL', 'CPILFESL']
        inflation_trends = [trends.get(ind, {}) for ind in inflation_indicators if ind in trends]
        inflation_rising = any(t.get('direction') == 'rising' for t in inflation_trends)
        inflation_strength = sum(abs(t.get('momentum', 0)) for t in inflation_trends) / max(len(inflation_trends), 1)
        
        signals['inflation_pressure'] = {
            'direction': 'rising' if inflation_rising else 'stable',
            'strength': float(inflation_strength),
            'confidence': 0.8 if len(inflation_trends) > 1 else 0.6,
            'indicators_count': len(inflation_trends)
        }
        
        # Monetary policy stance signal with rate trajectory
        fed_funds_trend = trends.get('FEDFUNDS', {})
        fed_direction = fed_funds_trend.get('direction', 'stable')
        fed_momentum = fed_funds_trend.get('momentum', 0.0)
        
        signals['monetary_policy_stance'] = {
            'stance': 'tightening' if fed_direction == 'rising' else 'easing' if fed_direction == 'falling' else 'neutral',
            'momentum': float(fed_momentum),
            'confidence': 0.9 if 'FEDFUNDS' in trends else 0.3,
            'rate_trajectory': fed_direction
        }
        
        # Yield curve signal with inversion risk
        yield_curve_indicators = ['T10Y2Y', 'T10Y3M']
        curve_trends = [trends.get(ind, {}) for ind in yield_curve_indicators if ind in trends]
        curve_flattening = any(t.get('direction') == 'falling' for t in curve_trends)
        curve_steepening = any(t.get('direction') == 'rising' for t in curve_trends)
        
        signals['yield_curve_signal'] = {
            'direction': 'flattening' if curve_flattening else 'steepening' if curve_steepening else 'stable',
            'inversion_risk': 'high' if curve_flattening else 'low',
            'confidence': 0.8 if len(curve_trends) > 1 else 0.5,
            'indicators_available': len(curve_trends)
        }
        
        return signals

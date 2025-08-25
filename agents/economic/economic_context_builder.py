"""
Economic Context Builder - AI context generation and prompt construction
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class EconomicContextBuilder:
    """Builds rich context for AI analysis"""
    
    def __init__(self):
        self.economic_regimes = {
            'expansion': {
                'characteristics': 'GDP growth, employment gains, moderate inflation',
                'typical_duration': '2-10 years',
                'policy_response': 'Gradual rate increases to prevent overheating'
            },
            'peak': {
                'characteristics': 'Full employment, capacity constraints, inflation pressures',
                'typical_duration': '6-18 months', 
                'policy_response': 'Restrictive monetary policy'
            },
            'contraction': {
                'characteristics': 'GDP decline, rising unemployment, disinflation',
                'typical_duration': '6-18 months',
                'policy_response': 'Aggressive rate cuts and stimulus'
            },
            'trough': {
                'characteristics': 'Economic bottom, policy accommodation, early recovery signs',
                'typical_duration': '3-12 months',
                'policy_response': 'Continued accommodation until recovery solidifies'
            }
        }
    
    def build_ai_context(self, 
                        query: str,
                        trends: Dict[str, any], 
                        cycle_assessment: Dict[str, any],
                        correlations: List[Dict[str, any]],
                        data_quality: Dict[str, any],
                        timeframe: str) -> str:
        """Build comprehensive context for AI analysis"""
        
        context_parts = []
        
        # Query and timeframe context
        context_parts.append(f"**ANALYSIS REQUEST:**")
        context_parts.append(f"Query: {query}")
        context_parts.append(f"Analysis Timeframe: {timeframe}")
        context_parts.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}")
        context_parts.append("")
        
        # Data quality and scope
        context_parts.append(f"**DATA SCOPE:**")
        context_parts.append(f"- Data Points Analyzed: {data_quality.get('data_points_analyzed', 0):,}")
        context_parts.append(f"- Indicators Analyzed: {data_quality.get('indicators_analyzed', 0)}")
        context_parts.append(f"- Date Range: {data_quality.get('date_range', {}).get('start', 'N/A')} to {data_quality.get('date_range', {}).get('end', 'N/A')}")
        context_parts.append("")
        
        # Economic cycle context
        context_parts.append(f"**ECONOMIC CYCLE ASSESSMENT:**")
        context_parts.append(f"- Current Phase: {cycle_assessment.get('phase', 'unknown')}")
        context_parts.append(f"- Composite Score: {cycle_assessment.get('composite_score', 0):.3f}")
        context_parts.append(f"- Confidence: {cycle_assessment.get('confidence', 0):.1%}")
        context_parts.append(f"- Leading Momentum: {cycle_assessment.get('leading_momentum', 0):.3f}")
        context_parts.append(f"- Coincident Momentum: {cycle_assessment.get('coincident_momentum', 0):.3f}")
        context_parts.append(f"- Lagging Momentum: {cycle_assessment.get('lagging_momentum', 0):.3f}")
        context_parts.append("")
        
        # Trend analysis with frequency awareness
        context_parts.append(f"**INDICATOR TRENDS (Frequency-Adjusted):**")
        rising_count = 0
        falling_count = 0
        stable_count = 0
        
        for indicator, trend_data in trends.items():
            direction = trend_data.get('direction', 'stable')
            strength = trend_data.get('strength', 'unknown')
            frequency = trend_data.get('frequency', 'MONTHLY')
            latest_value = trend_data.get('latest_value')
            changes = trend_data.get('changes', {})
            
            if direction == 'rising':
                rising_count += 1
            elif direction == 'falling':
                falling_count += 1
            else:
                stable_count += 1
            
            # Show most relevant change based on frequency
            if frequency == 'MONTHLY':
                key_change = changes.get('change_medium', 0)  # 3-month for monthly data
                change_period = "3-month"
            elif frequency == 'WEEKLY':
                key_change = changes.get('change_medium', 0)  # 13-week for weekly data
                change_period = "13-week"
            else:
                key_change = changes.get('change_medium', 0)  # 90-day for daily data
                change_period = "90-day"
            
            context_parts.append(f"- {indicator} ({frequency}): {direction.upper()} ({strength}) - {change_period} change: {key_change:.2f}%")
            if latest_value is not None:
                context_parts.append(f"  Latest Value: {latest_value:.2f}")
        
        context_parts.append(f"\nTrend Balance: {rising_count} rising, {falling_count} falling, {stable_count} stable")
        context_parts.append("")
        
        # Significant correlations
        if correlations:
            context_parts.append(f"**SIGNIFICANT CORRELATIONS:**")
            for i, corr in enumerate(correlations[:10]):  # Top 10 correlations
                context_parts.append(f"- {corr['indicator1']} ↔ {corr['indicator2']}: {corr['correlation']:.3f} ({corr['strength']})")
            context_parts.append("")
        
        # Historical context and regime analysis
        current_phase = cycle_assessment.get('phase', 'stable')
        regime_info = self._get_regime_context(current_phase)
        if regime_info:
            context_parts.append(f"**ECONOMIC REGIME CONTEXT:**")
            context_parts.append(f"Current Phase: {current_phase}")
            context_parts.append(f"Characteristics: {regime_info.get('characteristics', 'N/A')}")
            context_parts.append(f"Typical Duration: {regime_info.get('typical_duration', 'N/A')}")
            context_parts.append(f"Policy Response: {regime_info.get('policy_response', 'N/A')}")
            context_parts.append("")
        
        # Analysis instructions
        context_parts.append(f"**ANALYSIS INSTRUCTIONS:**")
        context_parts.append(f"1. Focus on frequency-appropriate trends (monthly data analyzed monthly, not daily)")
        context_parts.append(f"2. Consider the economic cycle phase and momentum indicators")
        context_parts.append(f"3. Highlight significant correlations and their economic meaning")
        context_parts.append(f"4. Provide specific data points with dates and values")
        context_parts.append(f"5. Assess policy implications and market outlook")
        context_parts.append(f"6. Rate your confidence in the analysis (0-100%)")
        context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _get_regime_context(self, phase: str) -> Optional[Dict[str, str]]:
        """Get economic regime context for the current phase"""
        # Map cycle phases to economic regimes
        phase_mapping = {
            'early_expansion': 'expansion',
            'mature_expansion': 'expansion', 
            'moderate_growth': 'expansion',
            'stable': 'expansion',
            'slowdown': 'peak',
            'early_contraction': 'contraction',
            'deep_contraction': 'contraction'
        }
        
        regime = phase_mapping.get(phase)
        return self.economic_regimes.get(regime)
    
    def format_ai_prompt(self, context: str, query: str) -> str:
        """Format the complete AI prompt"""
        prompt = f"""You are an expert economic analyst providing insights on U.S. economic conditions.

{context}

**USER QUERY:** {query}

**RESPONSE FORMAT:**
Provide a comprehensive economic analysis with:

1. **Executive Summary** (2-3 sentences)
2. **Key Findings** (3-5 bullet points with specific data)
3. **Economic Implications** (policy and market outlook)
4. **Confidence Assessment** (rate 0-100% with reasoning)

Use specific data points, dates, and values from the context. Focus on actionable insights.

**ANALYSIS:**"""
        
        return prompt
    
    def extract_key_metrics(self, trends: Dict[str, any], correlations: List[Dict[str, any]]) -> Dict[str, any]:
        """Extract key metrics for response summary"""
        metrics = {}
        
        # Count trend directions
        rising_indicators = sum(1 for t in trends.values() if t.get('direction') == 'rising')
        falling_indicators = sum(1 for t in trends.values() if t.get('direction') == 'falling')
        stable_indicators = sum(1 for t in trends.values() if t.get('direction') == 'stable')
        
        metrics['rising_indicators'] = rising_indicators
        metrics['falling_indicators'] = falling_indicators
        metrics['stable_indicators'] = stable_indicators
        metrics['trend_balance'] = (rising_indicators - falling_indicators) / len(trends) if trends else 0
        
        # Strong correlations
        strong_correlations = [c for c in correlations if c.get('strength') == 'strong']
        metrics['strong_correlations_found'] = len(strong_correlations)
        
        return metrics

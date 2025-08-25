"""
Editorial Context Builder - AI context generation and prompt construction
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class EditorialContextBuilder:
    """Builds rich context for AI editorial synthesis"""
    
    def __init__(self):
        self.editorial_styles = {
            'wsj_style': {
                'tone': 'professional, authoritative, clear',
                'voice': 'active voice preferred',
                'structure': 'inverted pyramid with analysis',
                'target_length': '800-1000 words'
            },
            'daily_briefing': {
                'tone': 'concise, informative, actionable',
                'voice': 'direct and accessible',
                'structure': 'executive summary format',
                'target_length': '400-600 words'
            },
            'breaking_news': {
                'tone': 'urgent, factual, immediate',
                'voice': 'active and decisive',
                'structure': 'lead with most important facts',
                'target_length': '300-500 words'
            }
        }
    
    def build_ai_context(self, 
                        query: str,
                        insights: Dict[str, Any], 
                        article_structure,
                        article_type: str,
                        data_quality: Dict[str, Any],
                        timeframe: str) -> str:
        """Build comprehensive context for AI editorial synthesis"""
        
        context_parts = []
        
        # Editorial request context
        context_parts.append(f"**EDITORIAL SYNTHESIS REQUEST:**")
        context_parts.append(f"Query: {query}")
        context_parts.append(f"Article Type: {article_type}")
        context_parts.append(f"Analysis Timeframe: {timeframe}")
        context_parts.append(f"Synthesis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        context_parts.append("")
        
        # Data scope and quality
        context_parts.append(f"**DATA SCOPE:**")
        context_parts.append(f"- Agents Available: {data_quality.get('agents_available', 0)}")
        context_parts.append(f"- Average Confidence: {data_quality.get('avg_confidence', 0):.1%}")
        context_parts.append(f"- Total Content Length: {data_quality.get('content_length', 0):,} characters")
        context_parts.append(f"- Market Signals: {data_quality.get('signals_available', 0)}")
        context_parts.append(f"- Data Freshness: {data_quality.get('data_freshness', 0):.1%}")
        context_parts.append("")
        
        # Agent insights summary
        context_parts.append(f"**AGENT INSIGHTS:**")
        for agent_name, data in insights.items():
            if isinstance(data, dict):
                confidence = data.get('confidence', 0.5)
                content_preview = data.get('content', '')[:150] + "..." if len(data.get('content', '')) > 150 else data.get('content', '')
                context_parts.append(f"- {agent_name.replace('_', ' ').title()} (confidence: {confidence:.1%})")
                context_parts.append(f"  {content_preview}")
                
                # Include key metrics if available
                if 'key_metrics' in data and data['key_metrics']:
                    key_metrics = list(data['key_metrics'].items())[:3]  # Top 3 metrics
                    metrics_str = ", ".join([f"{k}: {v}" for k, v in key_metrics])
                    context_parts.append(f"  Key Metrics: {metrics_str}")
        context_parts.append("")
        
        # Article structure context
        context_parts.append(f"**ARTICLE STRUCTURE:**")
        context_parts.append(f"- Headline: {article_structure.headline}")
        context_parts.append(f"- Lead: {article_structure.lead_paragraph[:100]}...")
        context_parts.append(f"- Key Points: {len(article_structure.key_points)} identified")
        if article_structure.key_points:
            for i, point in enumerate(article_structure.key_points[:3], 1):
                context_parts.append(f"  {i}. {point}")
        context_parts.append("")
        
        # Market signals context
        all_signals = {}
        for data in insights.values():
            if isinstance(data, dict) and 'signals' in data:
                all_signals.update(data['signals'])
        
        if all_signals:
            context_parts.append(f"**MARKET SIGNALS:**")
            for signal, value in list(all_signals.items())[:5]:  # Top 5 signals
                context_parts.append(f"- {signal.replace('_', ' ').title()}: {value}")
            context_parts.append("")
        
        # Editorial style guidelines
        style_guide = self.editorial_styles.get(article_type, self.editorial_styles['wsj_style'])
        context_parts.append(f"**EDITORIAL GUIDELINES:**")
        context_parts.append(f"- Tone: {style_guide['tone']}")
        context_parts.append(f"- Voice: {style_guide['voice']}")
        context_parts.append(f"- Structure: {style_guide['structure']}")
        context_parts.append(f"- Target Length: {style_guide['target_length']}")
        context_parts.append("")
        
        # Synthesis instructions
        context_parts.append(f"**SYNTHESIS INSTRUCTIONS:**")
        context_parts.append(f"1. Synthesize insights into coherent narrative")
        context_parts.append(f"2. Maintain WSJ editorial standards and tone")
        context_parts.append(f"3. Include specific data points and metrics")
        context_parts.append(f"4. Balance multiple agent perspectives")
        context_parts.append(f"5. Provide actionable market insights")
        context_parts.append(f"6. Assess confidence in synthesis quality")
        context_parts.append("")
        
        return "\n".join(context_parts)
    
    def format_ai_prompt(self, context: str, query: str, article_type: str) -> str:
        """Format the complete AI prompt for editorial synthesis"""
        prompt = f"""You are a Wall Street Journal financial editor synthesizing multi-agent analysis into professional editorial content.

{context}

**USER QUERY:** {query}

**EDITORIAL TASK:**
Create a comprehensive {article_type.replace('_', ' ')} that synthesizes the agent insights above into WSJ-quality financial journalism.

**RESPONSE FORMAT:**
Generate a complete article with:

1. **Compelling Headline** (8-12 words, active voice)
2. **Lead Paragraph** (2-3 sentences, hook + key finding)
3. **Market Analysis Section** (integrate market intelligence insights)
4. **Economic Context Section** (incorporate economic analysis)
5. **Sentiment & Narrative Section** (include news sentiment insights)
6. **Forward-Looking Conclusion** (actionable implications)

**EDITORIAL STANDARDS:**
- Use precise financial terminology
- Include specific data points with context
- Maintain authoritative, professional tone
- Balance multiple perspectives
- Provide actionable insights for market participants
- Cite confidence levels where appropriate

**SYNTHESIS:**"""
        
        return prompt
    
    def extract_key_metrics(self, insights: Dict[str, Any], article_structure) -> Dict[str, Any]:
        """Extract key metrics for editorial response summary"""
        metrics = {}
        
        # Agent participation metrics
        metrics['agents_contributing'] = len(insights)
        metrics['total_confidence'] = sum(data.get('confidence', 0.5) for data in insights.values() if isinstance(data, dict))
        metrics['avg_confidence'] = metrics['total_confidence'] / max(metrics['agents_contributing'], 1)
        
        # Content metrics
        metrics['key_points_identified'] = len(article_structure.key_points)
        metrics['estimated_word_count'] = self._estimate_word_count(article_structure)
        
        # Signal strength
        all_signals = {}
        for data in insights.values():
            if isinstance(data, dict) and 'signals' in data:
                all_signals.update(data['signals'])
        metrics['market_signals_available'] = len(all_signals)
        
        # Editorial quality indicators
        metrics['headline_word_count'] = len(article_structure.headline.split())
        metrics['sections_with_content'] = sum(1 for section in [
            article_structure.market_analysis,
            article_structure.economic_context,
            article_structure.sentiment_summary
        ] if len(section) > 20)
        
        return metrics
    
    def _estimate_word_count(self, article_structure) -> int:
        """Estimate total word count from article structure"""
        total_words = 0
        if hasattr(article_structure, 'headline'):
            total_words += len(article_structure.headline.split())
        if hasattr(article_structure, 'lead_paragraph'):
            total_words += len(article_structure.lead_paragraph.split())
        if hasattr(article_structure, 'market_analysis'):
            total_words += len(article_structure.market_analysis.split())
        if hasattr(article_structure, 'economic_context'):
            total_words += len(article_structure.economic_context.split())
        if hasattr(article_structure, 'sentiment_summary'):
            total_words += len(article_structure.sentiment_summary.split())
        if hasattr(article_structure, 'conclusion'):
            total_words += len(article_structure.conclusion.split())
        
        return total_words

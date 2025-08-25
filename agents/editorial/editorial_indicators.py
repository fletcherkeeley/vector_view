"""
Editorial Indicators - Article structure analysis and synthesis logic
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

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

class EditorialIndicators:
    """Handles article structure generation and editorial analysis"""
    
    def __init__(self):
        # Article structure weights by type
        self.structure_weights = {
            'daily_briefing': {
                'market_analysis': 0.4,
                'economic_context': 0.3,
                'sentiment_summary': 0.3
            },
            'market_analysis': {
                'market_analysis': 0.5,
                'sentiment_summary': 0.3,
                'economic_context': 0.2
            },
            'breaking_news': {
                'sentiment_summary': 0.5,
                'market_analysis': 0.3,
                'economic_context': 0.2
            },
            'deep_dive': {
                'economic_context': 0.4,
                'market_analysis': 0.4,
                'sentiment_summary': 0.2
            }
        }
        
        # Content quality thresholds
        self.quality_thresholds = {
            'headline_max_words': 12,
            'lead_max_sentences': 3,
            'key_points_max': 6,
            'section_min_words': 20
        }
    
    def generate_article_structure(
        self, 
        insights: Dict[str, Any], 
        article_type: str, 
        data_handler
    ) -> ArticleStructure:
        """Generate WSJ-style article structure from agent insights"""
        try:
            # Extract key information from insights
            key_insights = data_handler.extract_key_insights(insights)
            
            # Generate structure components
            headline = self._generate_headline_placeholder(key_insights, article_type)
            lead_paragraph = self._generate_lead_placeholder(key_insights, article_type)
            key_points = self._extract_key_points(insights)
            
            # Generate sections based on article type weights
            weights = self.structure_weights.get(article_type, self.structure_weights['market_analysis'])
            
            market_analysis = self._synthesize_market_section(insights, weights['market_analysis'])
            economic_context = self._synthesize_economic_section(insights, weights['economic_context'])
            sentiment_summary = self._synthesize_sentiment_section(insights, weights['sentiment_summary'])
            
            conclusion = self._generate_conclusion_placeholder(key_insights)
            
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
            return ArticleStructure(
                headline="Market Update",
                lead_paragraph="Analysis unavailable",
                key_points=[],
                market_analysis="",
                economic_context="",
                sentiment_summary="",
                conclusion="",
                byline=f"Vector View Intelligence • {datetime.now().strftime('%B %d, %Y')}"
            )
    
    def _extract_key_points(self, insights: Dict[str, Any]) -> List[str]:
        """Extract key points for article structure"""
        points = []
        for agent, data in insights.items():
            if isinstance(data, dict):
                # Extract from signals
                if 'signals' in data:
                    signals = data['signals']
                    for signal, value in signals.items():
                        if isinstance(value, (str, int, float)):
                            points.append(f"{signal.replace('_', ' ').title()}: {value}")
                
                # Extract from key metrics
                if 'key_metrics' in data:
                    metrics = data['key_metrics']
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)) and abs(value) > 0.01:
                            points.append(f"{metric.replace('_', ' ').title()}: {value:.3f}")
                
                # Extract from content (bullet points or key phrases)
                if 'content' in data:
                    content = data['content']
                    # Look for bullet points or numbered lists
                    lines = content.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line.startswith('•') or line.startswith('-') or line.startswith('*'):
                            points.append(line[1:].strip())
                        elif any(line.startswith(f"{i}.") for i in range(1, 10)):
                            points.append(line.split('.', 1)[1].strip() if '.' in line else line)
        
        return points[:self.quality_thresholds['key_points_max']]
    
    def _synthesize_market_section(self, insights: Dict[str, Any], weight: float) -> str:
        """Synthesize market analysis section"""
        market_data = insights.get('market_intelligence', {})
        if isinstance(market_data, dict) and 'content' in market_data:
            content = market_data['content']
            # Extract weighted portion based on importance
            target_length = int(200 * weight)
            return content[:target_length] + "..." if len(content) > target_length else content
        return 'Market analysis pending...'
    
    def _synthesize_economic_section(self, insights: Dict[str, Any], weight: float) -> str:
        """Synthesize economic context section"""
        economic_data = insights.get('economic_analysis', {})
        if isinstance(economic_data, dict) and 'content' in economic_data:
            content = economic_data['content']
            target_length = int(200 * weight)
            return content[:target_length] + "..." if len(content) > target_length else content
        return 'Economic analysis pending...'
    
    def _synthesize_sentiment_section(self, insights: Dict[str, Any], weight: float) -> str:
        """Synthesize sentiment analysis section"""
        sentiment_data = insights.get('news_sentiment', {})
        if isinstance(sentiment_data, dict) and 'content' in sentiment_data:
            content = sentiment_data['content']
            target_length = int(200 * weight)
            return content[:target_length] + "..." if len(content) > target_length else content
        return 'Sentiment analysis pending...'
    
    def calculate_editorial_metrics(self, structure: ArticleStructure, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate editorial quality metrics"""
        metrics = {}
        
        # Word counts
        metrics['headline_words'] = len(structure.headline.split())
        metrics['lead_sentences'] = len([s for s in structure.lead_paragraph.split('.') if s.strip()])
        metrics['total_key_points'] = len(structure.key_points)
        
        # Content completeness
        sections = [structure.market_analysis, structure.economic_context, structure.sentiment_summary]
        metrics['sections_with_content'] = sum(1 for s in sections if len(s) > self.quality_thresholds['section_min_words'])
        metrics['content_completeness'] = metrics['sections_with_content'] / 3
        
        # Agent coverage
        metrics['agent_coverage'] = len(insights) / 3  # Expect 3 main agents
        
        return metrics
    
    def generate_cross_agent_signals(self, structure: ArticleStructure, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Generate editorial signals for other systems"""
        signals = {}
        
        # Editorial readiness signal
        content_quality = self._assess_content_quality(structure)
        signals['editorial_readiness'] = {
            'status': 'ready' if content_quality > 0.7 else 'needs_work',
            'quality_score': content_quality,
            'confidence': min(1.0, len(insights) / 3)
        }
        
        # Content type signal
        signals['content_type'] = {
            'article_type': self._infer_article_type(structure),
            'urgency': 'high' if 'breaking' in structure.headline.lower() else 'normal',
            'word_count_estimate': self._estimate_word_count(structure)
        }
        
        # Market focus signal
        signals['market_focus'] = {
            'primary_focus': self._identify_primary_focus(insights),
            'coverage_breadth': len(insights),
            'data_recency': 'current'  # Simplified for now
        }
        
        return signals
    
    def _generate_headline_placeholder(self, insights: List[str], article_type: str) -> str:
        """Generate headline placeholder based on insights"""
        if not insights:
            return "Market Analysis Update"
        
        # Extract key theme from insights
        first_insight = insights[0] if insights else ""
        if ":" in first_insight:
            theme = first_insight.split(":")[0]
            return f"{theme} Shapes Market Outlook"
        
        return "Market Analysis Update"
    
    def _generate_lead_placeholder(self, insights: List[str], article_type: str) -> str:
        """Generate lead paragraph placeholder"""
        if not insights:
            return "Market conditions continue to evolve as investors assess multiple economic and sentiment factors."
        
        return f"Analysis of current market conditions reveals {len(insights)} key factors influencing investor sentiment and economic outlook."
    
    def _generate_conclusion_placeholder(self, insights: List[str]) -> str:
        """Generate conclusion placeholder"""
        return "Market participants will continue monitoring these developments for further directional clarity."
    
    def _assess_content_quality(self, structure: ArticleStructure) -> float:
        """Assess overall content quality"""
        quality_score = 0.0
        
        # Headline quality (0-0.2)
        if len(structure.headline.split()) <= self.quality_thresholds['headline_max_words']:
            quality_score += 0.2
        
        # Lead paragraph quality (0-0.3)
        lead_sentences = len([s for s in structure.lead_paragraph.split('.') if s.strip()])
        if lead_sentences <= self.quality_thresholds['lead_max_sentences']:
            quality_score += 0.3
        
        # Content sections quality (0-0.5)
        sections = [structure.market_analysis, structure.economic_context, structure.sentiment_summary]
        content_sections = sum(1 for s in sections if len(s) > self.quality_thresholds['section_min_words'])
        quality_score += (content_sections / 3) * 0.5
        
        return quality_score
    
    def _infer_article_type(self, structure: ArticleStructure) -> str:
        """Infer article type from structure"""
        headline_lower = structure.headline.lower()
        if 'breaking' in headline_lower or 'urgent' in headline_lower:
            return 'breaking_news'
        elif 'daily' in headline_lower or 'briefing' in headline_lower:
            return 'daily_briefing'
        elif 'deep' in headline_lower or 'analysis' in headline_lower:
            return 'deep_dive'
        else:
            return 'market_analysis'
    
    def _estimate_word_count(self, structure: ArticleStructure) -> int:
        """Estimate total word count from structure"""
        total_words = 0
        total_words += len(structure.headline.split())
        total_words += len(structure.lead_paragraph.split())
        total_words += len(structure.market_analysis.split())
        total_words += len(structure.economic_context.split())
        total_words += len(structure.sentiment_summary.split())
        total_words += len(structure.conclusion.split())
        return total_words
    
    def _identify_primary_focus(self, insights: Dict[str, Any]) -> str:
        """Identify primary market focus from insights"""
        if 'market_intelligence' in insights:
            return 'market_data'
        elif 'economic_analysis' in insights:
            return 'economic_indicators'
        elif 'news_sentiment' in insights:
            return 'sentiment_analysis'
        else:
            return 'general_analysis'

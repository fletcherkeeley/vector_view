"""
Editorial Data Handler - Manages agent insights collection and data processing
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class EditorialDataHandler:
    """Handles editorial data collection and agent insight processing"""
    
    def __init__(self):
        # Agent data source priorities by article type
        self.agent_priorities = {
            'daily_briefing': ['market_intelligence', 'economic_analysis', 'news_sentiment'],
            'market_analysis': ['market_intelligence', 'news_sentiment', 'economic_analysis'],
            'breaking_news': ['news_sentiment', 'market_intelligence', 'economic_analysis'],
            'deep_dive': ['economic_analysis', 'market_intelligence', 'news_sentiment']
        }
        
        # Data quality thresholds
        self.quality_thresholds = {
            'min_agents': 2,
            'min_confidence': 0.3,
            'min_content_length': 50
        }
    
    def collect_agent_insights(self, context) -> Dict[str, Any]:
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
                        'confidence': getattr(agent_response, 'confidence', 0.5),
                        'signals': getattr(agent_response, 'cross_agent_signals', {}),
                        'key_metrics': getattr(agent_response, 'key_metrics', {}),
                        'timestamp': getattr(agent_response, 'timestamp', datetime.now())
                    }
        
        # If no agent outputs available, return minimal structure
        if not insights:
            insights = {
                'system': {
                    'content': 'Limited agent data available for synthesis',
                    'confidence': 0.3,
                    'signals': {},
                    'key_metrics': {},
                    'timestamp': datetime.now()
                }
            }
        
        return insights
    
    def determine_article_type(self, context, insights: Dict[str, Any]) -> str:
        """Determine the type of article to generate based on context and insights"""
        if hasattr(context, 'query_type') and context.query_type:
            if context.query_type == 'daily_briefing':
                return 'daily_briefing'
            elif context.query_type == 'deep_dive':
                return 'deep_dive'
        
        # Check for breaking news indicators
        if any('breaking' in str(insight).lower() or 'urgent' in str(insight).lower() 
               for insight in insights.values()):
            return 'breaking_news'
        
        # Default to market analysis
        return 'market_analysis'
    
    def extract_key_insights(self, insights: Dict[str, Any]) -> List[str]:
        """Extract key insights from agent responses"""
        key_insights = []
        for agent, data in insights.items():
            if isinstance(data, dict) and 'content' in data:
                # Extract first sentence as key insight
                content = data['content']
                first_sentence = content.split('.')[0] if content else ""
                if first_sentence and len(first_sentence) > 10:  # Minimum meaningful length
                    key_insights.append(f"{agent.replace('_', ' ').title()}: {first_sentence}")
        return key_insights[:5]
    
    def extract_market_signals(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all market signals from insights"""
        all_signals = {}
        for agent, data in insights.items():
            if isinstance(data, dict) and 'signals' in data:
                all_signals.update(data['signals'])
        return all_signals
    
    def get_data_quality_metrics(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate data quality metrics for editorial synthesis"""
        metrics = {
            'agents_available': len(insights),
            'total_confidence': sum(data.get('confidence', 0.5) for data in insights.values() if isinstance(data, dict)),
            'avg_confidence': 0.0,
            'content_length': sum(len(data.get('content', '')) for data in insights.values() if isinstance(data, dict)),
            'signals_available': len(self.extract_market_signals(insights)),
            'data_freshness': self._assess_data_freshness(insights)
        }
        
        if metrics['agents_available'] > 0:
            metrics['avg_confidence'] = metrics['total_confidence'] / metrics['agents_available']
        
        return metrics
    
    def _assess_data_freshness(self, insights: Dict[str, Any]) -> float:
        """Assess how fresh the data is based on timestamps"""
        now = datetime.now()
        freshness_scores = []
        
        for data in insights.values():
            if isinstance(data, dict) and 'timestamp' in data:
                timestamp = data['timestamp']
                if isinstance(timestamp, datetime):
                    # Calculate hours since timestamp
                    hours_old = (now - timestamp).total_seconds() / 3600
                    # Fresher data gets higher score (max 1.0 for data < 1 hour old)
                    freshness = max(0.0, min(1.0, (24 - hours_old) / 24))
                    freshness_scores.append(freshness)
        
        return sum(freshness_scores) / len(freshness_scores) if freshness_scores else 0.5

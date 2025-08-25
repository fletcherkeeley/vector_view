"""
News Sentiment Context Builder for Vector View Financial Intelligence Platform

Builds comprehensive context for news sentiment analysis including:
- Entity extraction and processing
- Sentiment analysis preparation
- Narrative context building
- Cross-agent signal preparation
"""

import logging
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)


@dataclass
class EntityExtraction:
    """Extracted entities from news content"""
    companies: List[str]
    people: List[str]
    locations: List[str]
    organizations: List[str]
    financial_instruments: List[str]
    events: List[str]


@dataclass
class SentimentAnalysis:
    """Comprehensive sentiment analysis result"""
    overall_sentiment: float  # -1 to +1
    emotional_tone: Dict[str, float]  # fear, greed, uncertainty, confidence
    bias_score: float  # 0 to 1 (higher = more biased)
    credibility_score: float  # 0 to 1 (higher = more credible)
    urgency_level: float  # 0 to 1 (higher = more urgent)
    market_relevance: float  # 0 to 1 (higher = more market relevant)


@dataclass
class NarrativeAnalysis:
    """Narrative trend analysis"""
    dominant_themes: List[str]
    narrative_shift: str  # "bullish_to_bearish", "stable", etc.
    theme_evolution: Dict[str, float]  # theme strength over time
    consensus_level: float  # 0 to 1 (higher = more consensus)


class NewsSentimentContextBuilder:
    """
    Builds comprehensive context for news sentiment analysis.
    
    Responsibilities:
    - Entity extraction from news content
    - Sentiment analysis preparation and computation
    - Narrative analysis and theme tracking
    - Context preparation for AI analysis
    """
    
    def __init__(self, ai_service=None):
        """
        Initialize the context builder.
        
        Args:
            ai_service: AI service for enhanced analysis
        """
        self.ai_service = ai_service
        
        # Financial entity patterns
        self.entity_patterns = {
            'companies': r'\b[A-Z]{2,5}\b|\b(?:Inc|Corp|Ltd|LLC|Co)\b',
            'financial_instruments': r'\b(?:stock|bond|ETF|option|future|currency|crypto)\b',
            'events': r'\b(?:earnings|IPO|merger|acquisition|bankruptcy|lawsuit)\b'
        }
        
        # Emotional tone keywords
        self.emotion_keywords = {
            'fear': ['crash', 'collapse', 'panic', 'crisis', 'disaster', 'plunge', 'tumble'],
            'greed': ['surge', 'boom', 'rally', 'soar', 'skyrocket', 'explosive', 'massive'],
            'uncertainty': ['volatile', 'unclear', 'uncertain', 'mixed', 'conflicting', 'unpredictable'],
            'confidence': ['strong', 'solid', 'robust', 'stable', 'confident', 'optimistic', 'positive']
        }
    
    async def extract_entities(self, articles: List[Dict]) -> EntityExtraction:
        """
        Extract entities from news articles using pattern matching and AI.
        
        Args:
            articles: List of news articles with content and metadata
            
        Returns:
            EntityExtraction object with categorized entities
        """
        try:
            if not articles:
                return EntityExtraction([], [], [], [], [], [])
            
            all_text = " ".join([article['content'] + " " + article['title'] for article in articles])
            
            # Pattern-based extraction
            companies = list(set(re.findall(self.entity_patterns['companies'], all_text, re.IGNORECASE)))
            financial_instruments = list(set(re.findall(self.entity_patterns['financial_instruments'], all_text, re.IGNORECASE)))
            events = list(set(re.findall(self.entity_patterns['events'], all_text, re.IGNORECASE)))
            
            # AI-enhanced entity extraction for people, locations, organizations
            people, locations, organizations = [], [], []
            if len(articles) > 0 and self.ai_service:
                sample_text = " ".join([article['content'][:500] for article in articles[:5]])
                
                prompt = f"""
                Extract key entities from this financial news text:
                
                {sample_text}
                
                Return in format:
                PEOPLE: [list of person names]
                LOCATIONS: [list of locations/countries]
                ORGANIZATIONS: [list of organizations/institutions]
                
                Focus on financial market relevant entities only.
                """
                
                try:
                    ai_response = await self.ai_service.generate_response(
                        prompt=prompt,
                        context="Entity Extraction",
                        max_tokens=300
                    )
                    
                    # Parse AI response
                    people, locations, organizations = self._parse_entity_response(ai_response)
                except Exception as e:
                    logger.warning(f"AI entity extraction failed: {str(e)}")
            
            return EntityExtraction(
                companies=companies[:20],  # Limit results
                people=people[:10],
                locations=locations[:10],
                organizations=organizations[:10],
                financial_instruments=financial_instruments[:15],
                events=events[:15]
            )
            
        except Exception as e:
            logger.error(f"Entity extraction failed: {str(e)}")
            return EntityExtraction([], [], [], [], [], [])
    
    async def analyze_sentiment(self, articles: List[Dict], context: Any = None) -> SentimentAnalysis:
        """
        Perform multi-dimensional sentiment analysis.
        
        Args:
            articles: List of news articles
            context: Analysis context for additional information
            
        Returns:
            SentimentAnalysis object with comprehensive sentiment metrics
        """
        try:
            if not articles:
                return SentimentAnalysis(0.0, {}, 0.0, 0.0, 0.0, 0.0)
            
            # Calculate emotional tone scores
            emotional_scores = {emotion: 0.0 for emotion in self.emotion_keywords.keys()}
            
            total_words = 0
            for article in articles:
                text = (article['content'] + " " + article['title']).lower()
                words = text.split()
                total_words += len(words)
                
                for emotion, keywords in self.emotion_keywords.items():
                    emotion_count = sum(1 for word in words if any(kw in word for kw in keywords))
                    emotional_scores[emotion] += emotion_count
            
            # Normalize emotional scores
            if total_words > 0:
                for emotion in emotional_scores:
                    emotional_scores[emotion] = emotional_scores[emotion] / total_words * 1000
            
            # AI-powered sentiment analysis
            sentiment_scores = {'sentiment': 0.0, 'bias': 0.5, 'credibility': 0.5, 'urgency': 0.0, 'relevance': 0.5}
            
            if self.ai_service and articles:
                sample_articles = articles[:5]  # Analyze top 5 articles
                combined_text = "\n\n".join([f"{art['title']}\n{art['content'][:300]}" for art in sample_articles])
                
                prompt = f"""
                Analyze the sentiment and characteristics of this financial news content:
                
                {combined_text}
                
                Provide scores (0.0 to 1.0) for:
                1. Overall Sentiment (-1.0 to +1.0, negative to positive)
                2. Bias Score (0.0 = neutral, 1.0 = highly biased)
                3. Credibility Score (0.0 = low credibility, 1.0 = high credibility)
                4. Urgency Level (0.0 = low urgency, 1.0 = breaking news urgency)
                5. Market Relevance (0.0 = not market relevant, 1.0 = highly market relevant)
                
                Format: SENTIMENT:X.X BIAS:X.X CREDIBILITY:X.X URGENCY:X.X RELEVANCE:X.X
                """
                
                try:
                    ai_response = await self.ai_service.generate_response(
                        prompt=prompt,
                        context="Sentiment Analysis",
                        max_tokens=200
                    )
                    
                    # Parse AI response
                    sentiment_scores = self._parse_sentiment_response(ai_response)
                except Exception as e:
                    logger.warning(f"AI sentiment analysis failed: {str(e)}")
            
            return SentimentAnalysis(
                overall_sentiment=sentiment_scores.get('sentiment', 0.0),
                emotional_tone=emotional_scores,
                bias_score=sentiment_scores.get('bias', 0.5),
                credibility_score=sentiment_scores.get('credibility', 0.5),
                urgency_level=sentiment_scores.get('urgency', 0.0),
                market_relevance=sentiment_scores.get('relevance', 0.5)
            )
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            return SentimentAnalysis(0.0, {}, 0.5, 0.5, 0.0, 0.5)
    
    async def analyze_narratives(self, articles: List[Dict], context: Any = None) -> NarrativeAnalysis:
        """
        Analyze narrative trends and themes.
        
        Args:
            articles: List of news articles
            context: Analysis context for additional information
            
        Returns:
            NarrativeAnalysis object with theme and narrative insights
        """
        try:
            if not articles:
                return NarrativeAnalysis([], "stable", {}, 0.5)
            
            # Extract themes from titles and content
            all_text = " ".join([article['title'] + " " + article['content'] for article in articles])
            
            # Default values
            themes = []
            direction = "stable"
            consensus = 0.5
            
            # AI-powered theme extraction
            if self.ai_service:
                prompt = f"""
                Analyze the dominant themes and narratives in this financial news collection:
                
                {all_text[:2000]}...
                
                Identify:
                1. Top 5 dominant themes/topics
                2. Overall narrative direction (bullish/bearish/mixed/uncertain)
                3. Consensus level (high/medium/low agreement across sources)
                
                Format:
                THEMES: theme1, theme2, theme3, theme4, theme5
                DIRECTION: [direction]
                CONSENSUS: [level]
                """
                
                try:
                    ai_response = await self.ai_service.generate_response(
                        prompt=prompt,
                        context="Narrative Analysis",
                        max_tokens=300
                    )
                    
                    # Parse AI response
                    themes, direction, consensus = self._parse_narrative_response(ai_response)
                except Exception as e:
                    logger.warning(f"AI narrative analysis failed: {str(e)}")
            
            # Calculate theme evolution (simplified)
            theme_evolution = {}
            for theme in themes:
                theme_count = all_text.lower().count(theme.lower())
                theme_evolution[theme] = min(1.0, theme_count / len(articles))
            
            return NarrativeAnalysis(
                dominant_themes=themes,
                narrative_shift=direction,
                theme_evolution=theme_evolution,
                consensus_level=consensus
            )
            
        except Exception as e:
            logger.error(f"Narrative analysis failed: {str(e)}")
            return NarrativeAnalysis([], "stable", {}, 0.5)
    
    def build_analysis_context(
        self, 
        articles: List[Dict],
        sentiment: SentimentAnalysis,
        narrative: NarrativeAnalysis,
        entities: EntityExtraction
    ) -> Dict[str, Any]:
        """
        Build comprehensive context for AI analysis.
        
        Args:
            articles: News articles
            sentiment: Sentiment analysis results
            narrative: Narrative analysis results
            entities: Entity extraction results
            
        Returns:
            Dictionary with structured context for AI analysis
        """
        return {
            "articles_summary": {
                "count": len(articles),
                "sources": list(set([art.get('source', 'unknown') for art in articles])),
                "avg_relevance": sum([art.get('relevance_score', 0) for art in articles]) / len(articles) if articles else 0,
                "timespan": self._calculate_timespan(articles)
            },
            "sentiment_context": {
                "overall_sentiment": sentiment.overall_sentiment,
                "credibility_score": sentiment.credibility_score,
                "market_relevance": sentiment.market_relevance,
                "emotional_tone": sentiment.emotional_tone,
                "bias_level": sentiment.bias_score,
                "urgency": sentiment.urgency_level
            },
            "narrative_context": {
                "dominant_themes": narrative.dominant_themes,
                "narrative_direction": narrative.narrative_shift,
                "consensus_strength": narrative.consensus_level,
                "theme_distribution": narrative.theme_evolution
            },
            "entity_context": {
                "key_companies": entities.companies[:10],
                "key_people": entities.people[:5],
                "key_events": entities.events[:5],
                "financial_instruments": entities.financial_instruments[:5],
                "geographic_focus": entities.locations[:5]
            }
        }
    
    def generate_cross_agent_signals(
        self, 
        sentiment: SentimentAnalysis,
        narrative: NarrativeAnalysis
    ) -> Dict[str, Any]:
        """
        Generate signals for other agents based on sentiment and narrative analysis.
        
        Args:
            sentiment: Sentiment analysis results
            narrative: Narrative analysis results
            
        Returns:
            Dictionary with cross-agent signals
        """
        return {
            'news_sentiment': 'positive' if sentiment.overall_sentiment > 0.2 else 'negative' if sentiment.overall_sentiment < -0.2 else 'neutral',
            'emotional_state': max(sentiment.emotional_tone.items(), key=lambda x: x[1])[0] if sentiment.emotional_tone else 'neutral',
            'narrative_direction': narrative.narrative_shift,
            'consensus_strength': 'high' if narrative.consensus_level > 0.7 else 'low',
            'credibility_level': 'high' if sentiment.credibility_score > 0.7 else 'medium' if sentiment.credibility_score > 0.4 else 'low',
            'market_attention': 'high' if sentiment.market_relevance > 0.7 else 'medium' if sentiment.market_relevance > 0.4 else 'low'
        }
    
    def calculate_confidence(self, article_count: int, sentiment: SentimentAnalysis) -> float:
        """
        Calculate confidence based on data quality and analysis reliability.
        
        Args:
            article_count: Number of articles analyzed
            sentiment: Sentiment analysis results
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        data_confidence = min(1.0, article_count / 20)
        credibility_confidence = sentiment.credibility_score
        relevance_confidence = sentiment.market_relevance
        
        return (data_confidence * 0.4 + credibility_confidence * 0.4 + relevance_confidence * 0.2)
    
    def _parse_entity_response(self, response: str) -> Tuple[List[str], List[str], List[str]]:
        """Parse AI entity extraction response"""
        try:
            people = re.findall(r'PEOPLE:\s*\[(.*?)\]', response, re.IGNORECASE)
            locations = re.findall(r'LOCATIONS:\s*\[(.*?)\]', response, re.IGNORECASE)
            organizations = re.findall(r'ORGANIZATIONS:\s*\[(.*?)\]', response, re.IGNORECASE)
            
            people = people[0].split(',') if people else []
            locations = locations[0].split(',') if locations else []
            organizations = organizations[0].split(',') if organizations else []
            
            return [p.strip() for p in people], [l.strip() for l in locations], [o.strip() for o in organizations]
        except:
            return [], [], []
    
    def _parse_sentiment_response(self, response: str) -> Dict[str, float]:
        """Parse AI sentiment analysis response"""
        try:
            scores = {}
            patterns = {
                'sentiment': r'SENTIMENT:([-+]?\d*\.?\d+)',
                'bias': r'BIAS:(\d*\.?\d+)',
                'credibility': r'CREDIBILITY:(\d*\.?\d+)',
                'urgency': r'URGENCY:(\d*\.?\d+)',
                'relevance': r'RELEVANCE:(\d*\.?\d+)'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    scores[key] = float(match.group(1))
            
            return scores
        except:
            return {}
    
    def _parse_narrative_response(self, response: str) -> Tuple[List[str], str, float]:
        """Parse AI narrative analysis response"""
        try:
            themes_match = re.search(r'THEMES:\s*(.*?)(?:\n|$)', response, re.IGNORECASE)
            direction_match = re.search(r'DIRECTION:\s*(.*?)(?:\n|$)', response, re.IGNORECASE)
            consensus_match = re.search(r'CONSENSUS:\s*(.*?)(?:\n|$)', response, re.IGNORECASE)
            
            themes = themes_match.group(1).split(',') if themes_match else []
            themes = [t.strip() for t in themes]
            
            direction = direction_match.group(1).strip() if direction_match else "stable"
            
            consensus_text = consensus_match.group(1).strip().lower() if consensus_match else "medium"
            consensus_map = {'high': 0.8, 'medium': 0.5, 'low': 0.2}
            consensus = consensus_map.get(consensus_text, 0.5)
            
            return themes, direction, consensus
        except:
            return [], "stable", 0.5
    
    def _calculate_timespan(self, articles: List[Dict]) -> Dict[str, Any]:
        """Calculate timespan of articles"""
        try:
            timestamps = []
            for article in articles:
                if article.get('timestamp'):
                    try:
                        ts = datetime.fromisoformat(article['timestamp'].replace('Z', '+00:00'))
                        timestamps.append(ts)
                    except:
                        continue
            
            if timestamps:
                return {
                    "earliest": min(timestamps).isoformat(),
                    "latest": max(timestamps).isoformat(),
                    "span_hours": (max(timestamps) - min(timestamps)).total_seconds() / 3600
                }
            else:
                return {"earliest": None, "latest": None, "span_hours": 0}
        except:
            return {"earliest": None, "latest": None, "span_hours": 0}

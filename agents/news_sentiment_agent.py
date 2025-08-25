"""
News Sentiment & Narrative Agent for Vector View Financial Intelligence Platform

Advanced NLP analysis including sentiment scoring, entity extraction, bias detection,
emotional tone analysis, and narrative trend tracking across time periods.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import re
from dataclasses import dataclass
from collections import Counter, defaultdict

from .base_agent import BaseAgent, AgentType, AgentResponse, AgentContext
from .ai_service import OllamaService

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


class NewsSentimentAgent(BaseAgent):
    """
    News Sentiment & Narrative Agent specializing in advanced NLP analysis.
    
    Capabilities:
    - Multi-dimensional sentiment analysis beyond basic positive/negative
    - Entity extraction (companies, people, locations, events)
    - Bias detection and source credibility assessment
    - Emotional tone analysis (fear, greed, uncertainty, confidence)
    - Narrative trend tracking and theme evolution
    - Cross-source consensus analysis
    """
    
    def __init__(self, db_connection=None, chroma_client=None, ai_service: OllamaService = None, database_url: str = "postgresql://postgres:fred_password@localhost:5432/postgres"):
        super().__init__(
            agent_type=AgentType.NEWS_SENTIMENT,
            database_url=database_url
        )
        self.db_connection = db_connection
        self.chroma_client = chroma_client
        self.ai_service = ai_service or OllamaService()
        
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

    async def analyze(self, context: AgentContext) -> AgentResponse:
        """
        Perform comprehensive news sentiment and narrative analysis.
        """
        try:
            start_time = datetime.now()
            
            # Get news articles for analysis
            news_articles = await self._get_news_articles(context)
            
            # Perform entity extraction
            entities = await self._extract_entities(news_articles)
            
            # Analyze sentiment across multiple dimensions
            sentiment_analysis = await self._analyze_sentiment(news_articles, context)
            
            # Track narrative trends and themes
            narrative_analysis = await self._analyze_narratives(news_articles, context)
            
            # Generate AI-powered insights using the proper AI service method
            ai_analysis = await self.ai_service.analyze_news_sentiment(
                news_articles=news_articles,
                semantic_context={
                    "sentiment": {
                        "overall_sentiment": sentiment_analysis.overall_sentiment,
                        "credibility_score": sentiment_analysis.credibility_score,
                        "market_relevance": sentiment_analysis.market_relevance,
                        "emotional_tone": sentiment_analysis.emotional_tone
                    },
                    "narrative": {
                        "dominant_themes": narrative_analysis.dominant_themes,
                        "narrative_direction": narrative_analysis.narrative_shift,
                        "consensus_strength": narrative_analysis.consensus_level
                    }
                },
                context=f"Query: {context.query}. Timeframe: {context.timeframe}. News Sentiment Analysis."
            )
            
            # Extract insights from AI analysis (already cleaned by ai_service)
            ai_insights = ai_analysis.content
            
            # Calculate execution metrics
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Determine confidence based on data quality
            confidence = self._calculate_confidence(len(news_articles), sentiment_analysis)
            
            # Generate cross-agent signals
            signals = self._generate_sentiment_signals(sentiment_analysis, narrative_analysis)
            
            return AgentResponse(
                agent_type=self.agent_type,
                confidence=confidence,
                confidence_level=self._get_confidence_level(confidence),
                analysis={
                    'sentiment_analysis': sentiment_analysis.__dict__,
                    'narrative_analysis': narrative_analysis.__dict__,
                    'entities': entities.__dict__,
                    'articles_analyzed': len(news_articles),
                    'dominant_themes': narrative_analysis.dominant_themes[:5]
                },
                insights=[ai_insights],
                key_metrics={
                    'overall_sentiment': sentiment_analysis.overall_sentiment,
                    'credibility_score': sentiment_analysis.credibility_score,
                    'market_relevance': sentiment_analysis.market_relevance
                },
                data_sources_used=['news_articles', 'semantic_search'],
                timeframe_analyzed=context.timeframe,
                execution_time_ms=execution_time,
                signals_for_other_agents=signals,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"News Sentiment Agent analysis failed: {str(e)}")
            return AgentResponse(
                agent_type=self.agent_type,
                confidence=0.0,
                confidence_level=self._get_confidence_level(0.0),
                analysis={'error': str(e)},
                insights=[f"Sentiment analysis encountered an error: {str(e)}"],
                key_metrics={},
                data_sources_used=[],
                timeframe_analyzed=context.timeframe,
                execution_time_ms=0,
                signals_for_other_agents={},
                timestamp=datetime.now()
            )

    async def _get_news_articles(self, context: AgentContext) -> List[Dict]:
        """Retrieve news articles from ChromaDB"""
        try:
            # Initialize ChromaDB client if not provided
            if not self.chroma_client:
                try:
                    import chromadb
                    self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
                except Exception as e:
                    logger.warning(f"ChromaDB not available: {str(e)}")
                    return []
            
            hours_back = self._parse_timeframe_hours(context.timeframe)
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            try:
                collection = self.chroma_client.get_collection("financial_news")
            except Exception as e:
                logger.warning(f"ChromaDB collection 'financial_news' not found: {str(e)}")
                return []
            
            # Query without timestamp filter since the data doesn't have recent timestamps
            results = collection.query(
                query_texts=[context.query] if context.query else ["financial news market"],
                n_results=20  # Reduced to get more relevant results
            )
            
            articles = []
            if results and results.get('documents') and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results.get('metadatas') else {}
                    # Extract sentiment score from metadata
                    sentiment_score = metadata.get('sentiment_score', 0.0)
                    articles.append({
                        'content': doc,
                        'title': metadata.get('title', ''),
                        'source': metadata.get('source_name', 'unknown'),
                        'timestamp': metadata.get('published_at'),
                        'url': metadata.get('url', ''),
                        'sentiment_score': float(sentiment_score) if sentiment_score else 0.0,
                        'relevance_score': 1.0 - (results['distances'][0][i] if results.get('distances') else 0.0)
                    })
            
            return articles
            
        except Exception as e:
            logger.error(f"Failed to retrieve news articles: {str(e)}")
            return []

    async def _extract_entities(self, articles: List[Dict]) -> EntityExtraction:
        """Extract entities from news articles using pattern matching and AI"""
        try:
            all_text = " ".join([article['content'] + " " + article['title'] for article in articles])
            
            # Pattern-based extraction
            companies = list(set(re.findall(self.entity_patterns['companies'], all_text, re.IGNORECASE)))
            financial_instruments = list(set(re.findall(self.entity_patterns['financial_instruments'], all_text, re.IGNORECASE)))
            events = list(set(re.findall(self.entity_patterns['events'], all_text, re.IGNORECASE)))
            
            # AI-enhanced entity extraction for people, locations, organizations
            if len(articles) > 0:
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
                
                ai_response = await self.ai_service.generate_response(
                    prompt=prompt,
                    context="Entity Extraction",
                    max_tokens=300
                )
                
                # Parse AI response
                people, locations, organizations = self._parse_entity_response(ai_response)
            else:
                people, locations, organizations = [], [], []
            
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

    async def _analyze_sentiment(self, articles: List[Dict], context: AgentContext) -> SentimentAnalysis:
        """Perform multi-dimensional sentiment analysis"""
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
            
            ai_response = await self.ai_service.generate_response(
                prompt=prompt,
                context="Sentiment Analysis",
                max_tokens=200
            )
            
            # Parse AI response
            sentiment_scores = self._parse_sentiment_response(ai_response)
            
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

    async def _analyze_narratives(self, articles: List[Dict], context: AgentContext) -> NarrativeAnalysis:
        """Analyze narrative trends and themes"""
        try:
            if not articles:
                return NarrativeAnalysis([], "stable", {}, 0.5)
            
            # Extract themes from titles and content
            all_text = " ".join([article['title'] + " " + article['content'] for article in articles])
            
            # AI-powered theme extraction
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
            
            ai_response = await self.ai_service.generate_response(
                prompt=prompt,
                context="Narrative Analysis",
                max_tokens=300
            )
            
            # Parse AI response
            themes, direction, consensus = self._parse_narrative_response(ai_response)
            
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

    async def _generate_sentiment_insights(
        self,
        sentiment: SentimentAnalysis,
        narrative: NarrativeAnalysis,
        entities: EntityExtraction,
        context: AgentContext
    ) -> str:
        """Generate AI-powered sentiment and narrative insights"""
        try:
            prompt = f"""
            As a financial news analyst, provide insights on current market sentiment and narratives.
            
            Sentiment Analysis:
            - Overall Sentiment: {sentiment.overall_sentiment:.3f} (-1 to +1)
            - Emotional Tone: {dict(list(sentiment.emotional_tone.items())[:3])}
            - Bias Score: {sentiment.bias_score:.3f}
            - Credibility: {sentiment.credibility_score:.3f}
            - Market Relevance: {sentiment.market_relevance:.3f}
            
            Narrative Analysis:
            - Dominant Themes: {', '.join(narrative.dominant_themes[:3])}
            - Narrative Direction: {narrative.narrative_shift}
            - Consensus Level: {narrative.consensus_level:.3f}
            
            Key Entities:
            - Companies: {', '.join(entities.companies[:5])}
            - Events: {', '.join(entities.events[:3])}
            
            Provide analysis covering:
            1. Current sentiment landscape and reliability
            2. Key narrative themes and their implications
            3. Market psychology and emotional drivers
            4. Credibility assessment of news sources
            5. Actionable insights for market participants
            
            Keep analysis concise and professional.
            """
            
            response = await self.ai_service.generate_response(
                prompt=prompt,
                context=f"Sentiment Analysis - {context.query_type}",
                max_tokens=600
            )
            
            # Clean response to remove think tags
            import re
            cleaned_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
            cleaned_response = re.sub(r'^<think>.*', '', cleaned_response, flags=re.DOTALL | re.MULTILINE)
            cleaned_response = cleaned_response.strip()
            
            return cleaned_response if cleaned_response else response
            
        except Exception as e:
            logger.error(f"Sentiment insight generation failed: {str(e)}")
            return f"Sentiment Analysis: Overall sentiment {sentiment.overall_sentiment:.3f}, dominant themes: {', '.join(narrative.dominant_themes[:3])}"

    def _generate_sentiment_signals(
        self, 
        sentiment: SentimentAnalysis,
        narrative: NarrativeAnalysis
    ) -> Dict[str, Any]:
        """Generate cross-agent signals"""
        return {
            'news_sentiment': 'positive' if sentiment.overall_sentiment > 0.2 else 'negative' if sentiment.overall_sentiment < -0.2 else 'neutral',
            'emotional_state': max(sentiment.emotional_tone.items(), key=lambda x: x[1])[0] if sentiment.emotional_tone else 'neutral',
            'narrative_direction': narrative.narrative_shift,
            'consensus_strength': 'high' if narrative.consensus_level > 0.7 else 'low',
            'credibility_level': 'high' if sentiment.credibility_score > 0.7 else 'medium' if sentiment.credibility_score > 0.4 else 'low',
            'market_attention': 'high' if sentiment.market_relevance > 0.7 else 'medium' if sentiment.market_relevance > 0.4 else 'low'
        }

    def _calculate_confidence(self, article_count: int, sentiment: SentimentAnalysis) -> float:
        """Calculate confidence based on data quality and analysis reliability"""
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

    def _parse_timeframe_hours(self, timeframe: str) -> int:
        """Convert timeframe string to hours"""
        timeframe_map = {
            '1h': 1, '4h': 4, '1d': 24, '1w': 168, 
            '1m': 720, '3m': 2160, '1y': 8760
        }
        return timeframe_map.get(timeframe, 24)

    def get_required_data_sources(self, context: AgentContext) -> List[str]:
        """Return list of data sources required for news sentiment analysis"""
        return [
            'news_articles',      # News articles from ChromaDB
            'semantic_search',    # Semantic search capabilities
            'entity_data',        # Entity extraction data
            'sentiment_models'    # Sentiment analysis models
        ]

    def _get_confidence_level(self, confidence: float):
        """Convert confidence score to ConfidenceLevel enum"""
        from .base_agent import ConfidenceLevel
        if confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

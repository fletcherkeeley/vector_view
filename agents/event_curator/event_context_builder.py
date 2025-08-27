"""
Event Context Builder for Event Curator Agent

Handles event extraction, analysis, and context building using AI services.
Processes news articles to identify and structure factual events.
"""

import logging
import hashlib
import re
import json
import hashlib
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from agents.base_agent import logger
from agents.ai_service import AIResponse, OllamaService

logger = logging.getLogger(__name__)


class EventContextBuilder:
    """
    Builds context and extracts structured events from news articles.
    
    Responsibilities:
    - Extract factual events from news content using AI
    - Structure events with standardized properties
    - Perform event verification and confidence scoring
    - Handle event deduplication logic
    """
    
    def __init__(self, ai_service: OllamaService = None):
        self.ai_service = ai_service or OllamaService()
        
        # Event type categories for classification
        self.event_types = {
            'monetary_policy': ['fed', 'federal reserve', 'interest rate', 'monetary policy', 'fomc'],
            'economic_data': ['gdp', 'unemployment', 'inflation', 'cpi', 'ppi', 'jobs report'],
            'corporate_earnings': ['earnings', 'quarterly results', 'revenue', 'profit', 'guidance'],
            'market_movement': ['stock', 'market', 'index', 'trading', 'volume'],
            'regulatory': ['regulation', 'sec', 'compliance', 'policy', 'law'],
            'geopolitical': ['trade', 'tariff', 'sanctions', 'international', 'war', 'conflict'],
            'corporate_action': ['merger', 'acquisition', 'ipo', 'dividend', 'buyback', 'split'],
            'executive_change': ['ceo', 'cfo', 'executive', 'leadership', 'appointment', 'resignation']
        }
    
    async def extract_events_from_articles(
        self,
        articles: List[Dict[str, Any]],
        max_events_per_article: int = 3,
        batch_size: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Extract structured events from a batch of news articles with batch processing.
        
        Args:
            articles: List of news articles
            max_events_per_article: Maximum events to extract per article
            batch_size: Number of articles to process per AI batch (default: 5)
            
        Returns:
            List of extracted events with metadata
        """
        all_events = []
        
        # Process articles in batches to prevent CUDA memory overload
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(articles) + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} articles)")
            
            try:
                # Process batch with AI
                batch_events = await self._extract_events_from_batch(
                    batch, max_events_per_article
                )
                all_events.extend(batch_events)
                
                # Add small delay between batches to prevent GPU overheating
                if i + batch_size < len(articles):
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"Failed to process batch {batch_num}: {e}")
                # Try processing articles individually as fallback
                for article in batch:
                    try:
                        events = await self._extract_events_from_single_article(
                            article, max_events_per_article
                        )
                        all_events.extend(events)
                    except Exception as article_error:
                        logger.warning(f"Failed to extract events from article {article.get('article_id')}: {article_error}")
        
        logger.info(f"Extracted {len(all_events)} events from {len(articles)} articles in batches of {batch_size}")
        return all_events
    
    async def _extract_events_from_batch(
        self,
        articles: List[Dict[str, Any]],
        max_events_per_article: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Extract events from multiple articles in a single AI call for efficiency.
        
        Args:
            articles: Batch of articles to process
            max_events_per_article: Maximum events per article
            
        Returns:
            List of extracted events from all articles in batch
        """
        # Prepare batch content
        batch_content = self._prepare_batch_content(articles)
        
        # Create batch extraction prompt
        prompt = self._build_batch_extraction_prompt(batch_content, len(articles), max_events_per_article)
        
        try:
            # Get AI response for entire batch
            context_str = f"Batch event extraction from {len(articles)} articles"
            response_text = await self.ai_service.generate_response(
                prompt,
                context=context_str,
                max_tokens=4000  # Increased for batch processing
            )
            
            # Create AIResponse object
            ai_response = AIResponse(
                content=response_text,
                confidence=0.8,
                reasoning=["AI batch event extraction completed"],
                key_points=[],
                uncertainty_factors=[],
                metadata={
                    'batch_size': len(articles),
                    'articles': [a.get('article_id') for a in articles]
                }
            )
            
            # Parse events from batch response
            events = self._parse_batch_events_from_response(ai_response, articles)
            
            logger.info(f"Extracted {len(events)} events from batch of {len(articles)} articles")
            return events
            
        except Exception as e:
            logger.error(f"Batch extraction failed: {e}")
            # Fallback to individual article processing
            all_events = []
            for article in articles:
                try:
                    events = await self._extract_events_from_single_article(
                        article, max_events_per_article
                    )
                    all_events.extend(events)
                except Exception as article_error:
                    logger.warning(f"Failed to extract from article {article.get('article_id')}: {article_error}")
            return all_events
    
    def _prepare_batch_content(self, articles: List[Dict[str, Any]]) -> str:
        """
        Prepare content from multiple articles for batch processing.
        
        Args:
            articles: List of articles to prepare
            
        Returns:
            Formatted batch content string
        """
        batch_parts = []
        
        for i, article in enumerate(articles, 1):
            article_content = f"=== ARTICLE {i} ===\n"
            article_content += f"ID: {article.get('article_id', 'unknown')}\n"
            
            if article.get('title'):
                article_content += f"TITLE: {article['title']}\n"
            
            if article.get('description'):
                article_content += f"DESCRIPTION: {article['description']}\n"
            
            if article.get('content'):
                # Limit content per article to prevent token overflow
                content = article['content'][:1000]  # Reduced for batch processing
                article_content += f"CONTENT: {content}\n"
            
            if article.get('source'):
                article_content += f"SOURCE: {article['source']}\n"
            
            if article.get('published_at'):
                article_content += f"PUBLISHED: {article['published_at']}\n"
            
            batch_parts.append(article_content)
        
        return "\n\n".join(batch_parts)
    
    def _build_batch_extraction_prompt(self, batch_content: str, num_articles: int, max_events_per_article: int) -> str:
        """
        Build prompt for batch AI event extraction.
        
        Args:
            batch_content: Formatted content from multiple articles
            num_articles: Number of articles in batch
            max_events_per_article: Maximum events per article
            
        Returns:
            Formatted batch prompt string
        """
        return f"""
You are a financial news analyst extracting events from {num_articles} news articles. 
Extract up to {max_events_per_article} distinct, verifiable events from EACH article.

Focus on FACTUAL EVENTS such as:
- Federal Reserve policy decisions (rate changes, policy announcements)
- Economic data releases (GDP, unemployment, inflation figures)
- Corporate earnings announcements and guidance changes
- Executive appointments, resignations, or changes
- Market movements and trading milestones
- Regulatory announcements and policy changes
- Mergers, acquisitions, and corporate actions

For each event, provide:
- description: Clear, factual description (required)
- event_type: One of [monetary_policy, economic_data, corporate_earnings, market_movement, regulatory, geopolitical, corporate_action, executive_change]
- confidence: Float 0.0-1.0 based on factual certainty
- entities: List of key entities (companies, people, organizations)
- date: ISO date if mentioned, otherwise "unknown"
- impact_score: Float 0.0-1.0 for market significance

Return a JSON array with all events from all articles. Include article_index (1-{num_articles}) for each event.

ARTICLES:

{batch_content}

JSON Response:
"""
    
    def _parse_batch_events_from_response(self, ai_response: AIResponse, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Parse events from batch AI response.
        
        Args:
            ai_response: AI response containing batch events
            articles: Original articles for metadata
            
        Returns:
            List of parsed events with metadata
        """
        try:
            content = ai_response.content.strip()
            
            # Find JSON array in the response
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if not json_match:
                logger.warning("No JSON array found in batch AI response")
                return []
            
            json_str = json_match.group(0)
            events_data = json.loads(json_str)
            
            if not isinstance(events_data, list):
                logger.warning("Batch AI response is not a list")
                return []
            
            # Validate and enhance events
            valid_events = []
            for event_data in events_data:
                if self._validate_event_data(event_data):
                    # Map article_index to actual article
                    article_index = event_data.get('article_index', 1) - 1
                    if 0 <= article_index < len(articles):
                        article = articles[article_index]
                        
                        # Add article metadata
                        event_data['source'] = article.get('source_name') or article.get('source')
                        event_data['published_at'] = article.get('published_at')
                        event_data['article_title'] = article.get('title')
                        event_data['source_article_id'] = article.get('id') or article.get('article_id')
                        
                        # Initialize source count (will be updated during deduplication)
                        event_data['source_count'] = 1
                        
                        # Generate event ID and metadata
                        event_data['event_id'] = self._generate_event_id(event_data)
                        event_data['extraction_confidence'] = ai_response.confidence
                        event_data['extracted_at'] = datetime.utcnow().isoformat()
                        
                        # Classify event type if needed
                        if not event_data.get('event_type') or event_data['event_type'] not in [
                            'monetary_policy', 'economic_data', 'corporate_earnings', 
                            'market_movement', 'regulatory', 'geopolitical', 
                            'corporate_action', 'executive_change'
                        ]:
                            event_data['event_type'] = self._classify_event_type(event_data['description'])
                        
                        valid_events.append(event_data)
            
            return valid_events
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from batch AI response: {e}")
            # Try to fix common JSON issues and retry
            try:
                fixed_json = self._fix_malformed_json(json_str)
                events_data = json.loads(fixed_json)
                if isinstance(events_data, list):
                    logger.info(f"Successfully parsed batch JSON after fixing malformed content")
                    # Process fixed events (simplified version)
                    valid_events = []
                    for event_data in events_data:
                        if self._validate_event_data(event_data):
                            article_index = event_data.get('article_index', 1) - 1
                            if 0 <= article_index < len(articles):
                                article = articles[article_index]
                                event_data['source'] = article.get('source')
                                event_data['source_article_id'] = article.get('article_id')
                                event_data['event_id'] = self._generate_event_id(event_data)
                                valid_events.append(event_data)
                    return valid_events
            except:
                pass
            return []
        except Exception as e:
            logger.error(f"Error parsing batch events from AI response: {e}")
            return []
    
    async def _extract_events_from_single_article(
        self,
        article: Dict[str, Any],
        max_events: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Extract events from a single news article using AI.
        
        Args:
            article: News article dictionary
            max_events: Maximum events to extract
            
        Returns:
            List of extracted events
        """
        # Prepare content for analysis
        content = self._prepare_article_content(article)
        
        # Create extraction prompt
        prompt = self._build_event_extraction_prompt(content, max_events)
        
        try:
            # Get AI response
            context_str = f"Event extraction from {article.get('source')} article {article.get('article_id')}"
            response_text = await self.ai_service.generate_response(
                prompt, 
                context=context_str,
                max_tokens=2000
            )
            
            # Create AIResponse object manually
            ai_response = AIResponse(
                content=response_text,
                confidence=0.8,  # Default confidence for successful extraction
                reasoning=["AI event extraction completed"],
                key_points=[],
                uncertainty_factors=[],
                metadata={
                    'article_id': article.get('article_id'),
                    'source': article.get('source'),
                    'published_at': str(article.get('published_at'))
                }
            )
            
            # Parse events from AI response
            events = self._parse_events_from_response(ai_response, article)
            
            # Add metadata and generate IDs
            for event in events:
                event['event_id'] = self._generate_event_id(event)
                event['source_article_id'] = article.get('article_id')
                event['extraction_confidence'] = ai_response.confidence
                event['extracted_at'] = datetime.utcnow().isoformat()
            
            return events
            
        except Exception as e:
            logger.error(f"Error extracting events from article {article.get('article_id')}: {e}")
            return []
    
    def _prepare_article_content(self, article: Dict[str, Any]) -> str:
        """
        Prepare article content for event extraction.
        
        Args:
            article: Article dictionary
            
        Returns:
            Cleaned and formatted content string
        """
        # Combine title, description, and content
        parts = []
        
        if article.get('title'):
            parts.append(f"TITLE: {article['title']}")
        
        if article.get('description'):
            parts.append(f"DESCRIPTION: {article['description']}")
        
        if article.get('content'):
            # Limit content length to avoid token limits
            content = article['content'][:2000]  # Truncate if too long
            parts.append(f"CONTENT: {content}")
        
        # Add metadata
        if article.get('source'):
            parts.append(f"SOURCE: {article['source']}")
        
        if article.get('published_at'):
            parts.append(f"PUBLISHED: {article['published_at']}")
        
        return "\n\n".join(parts)
    
    def _build_event_extraction_prompt(self, content: str, max_events: int) -> str:
        """
        Build prompt for AI event extraction.
        
        Args:
            content: Article content
            max_events: Maximum events to extract
            
        Returns:
            Formatted prompt string
        """
        return f"""
You are a financial news analyst tasked with extracting factual events from news articles. 
Extract up to {max_events} distinct, verifiable events from the following article.

Focus on FACTUAL EVENTS such as:
- Federal Reserve policy decisions (rate changes, policy announcements)
- Economic data releases (GDP, unemployment, inflation figures)
- Corporate earnings announcements and guidance changes
- Executive appointments, resignations, or changes
- Merger & acquisition announcements
- Regulatory decisions or policy changes
- Market milestones or significant movements
- IPO announcements or completions

For each event, provide:
1. A clear, factual description (1-2 sentences)
2. The specific date if mentioned, or "unknown" if not specified
3. Event type from: monetary_policy, economic_data, corporate_earnings, market_movement, regulatory, geopolitical, corporate_action, executive_change
4. Key entities involved (companies, people, organizations)
5. Confidence level (0.0-1.0) based on how clearly stated the event is
6. Potential market impact score (0.0-1.0) based on likely significance

ARTICLE CONTENT:
{content}

Respond with a JSON array of events in this exact format:
[
  {{
    "description": "Clear factual description of the event",
    "date": "YYYY-MM-DD or 'unknown'",
    "event_type": "one of the specified types",
    "entities": ["entity1", "entity2"],
    "confidence": 0.85,
    "impact_score": 0.7,
    "reasoning": "Brief explanation of why this is significant"
  }}
]

Only include events that are clearly stated as facts, not speculation or analysis.
"""
    
    def _parse_events_from_response(
        self,
        ai_response: AIResponse,
        article: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Parse events from AI response.
        
        Args:
            ai_response: AI service response
            article: Source article
            
        Returns:
            List of parsed events
        """
        try:
            # Try to extract JSON from the response
            content = ai_response.content.strip()
            
            # Find JSON array in the response
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if not json_match:
                logger.warning("No JSON array found in AI response")
                return []
            
            json_str = json_match.group(0)
            events_data = json.loads(json_str)
            
            if not isinstance(events_data, list):
                logger.warning("AI response is not a list")
                return []
            
            # Validate and clean events
            valid_events = []
            for event_data in events_data:
                if self._validate_event_data(event_data):
                    # Add article metadata
                    event_data['source'] = article.get('source')
                    event_data['published_at'] = article.get('published_at')
                    event_data['article_title'] = article.get('title')
                    
                    # Classify event type if not provided or invalid
                    if not event_data.get('event_type') or event_data['event_type'] not in [
                        'monetary_policy', 'economic_data', 'corporate_earnings', 
                        'market_movement', 'regulatory', 'geopolitical', 
                        'corporate_action', 'executive_change'
                    ]:
                        event_data['event_type'] = self._classify_event_type(event_data['description'])
                    
                    valid_events.append(event_data)
            
            return valid_events
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from AI response: {e}")
            # Try to fix common JSON issues and retry
            try:
                # Remove trailing commas and fix common issues
                fixed_json = self._fix_malformed_json(json_str)
                events_data = json.loads(fixed_json)
                if isinstance(events_data, list):
                    logger.info(f"Successfully parsed JSON after fixing malformed content")
                    valid_events = []
                    for event_data in events_data:
                        if self._validate_event_data(event_data):
                            event_data['source'] = article.get('source')
                            event_data['published_at'] = article.get('published_at')
                            event_data['article_title'] = article.get('title')
                            if not event_data.get('event_type') or event_data['event_type'] not in [
                                'monetary_policy', 'economic_data', 'corporate_earnings', 
                                'market_movement', 'regulatory', 'geopolitical', 
                                'corporate_action', 'executive_change'
                            ]:
                                event_data['event_type'] = self._classify_event_type(event_data['description'])
                            valid_events.append(event_data)
                    return valid_events
            except:
                pass
            return []
        except Exception as e:
            logger.error(f"Error parsing events from AI response: {e}")
            return []
    
    def _fix_malformed_json(self, json_str: str) -> str:
        """
        Fix common JSON formatting issues from AI responses.
        
        Args:
            json_str: Malformed JSON string
            
        Returns:
            Fixed JSON string
        """
        # Remove trailing commas before closing brackets
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Fix unescaped quotes in strings
        json_str = re.sub(r'(?<!\\)"(?=.*".*:)', '\\"', json_str)
        
        # Remove any trailing content after the JSON array
        json_str = re.sub(r']\s*.*$', ']', json_str, flags=re.DOTALL)
        
        return json_str
    
    def _validate_event_data(self, event_data: Dict[str, Any]) -> bool:
        """
        Validate extracted event data.
        
        Args:
            event_data: Event data dictionary
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = ['description', 'confidence']
        
        for field in required_fields:
            if field not in event_data:
                logger.warning(f"Event missing required field: {field}")
                return False
        
        # Validate confidence score
        confidence = event_data.get('confidence', 0.0)
        if not isinstance(confidence, (int, float)) or confidence < 0.0 or confidence > 1.0:
            logger.warning(f"Invalid confidence score: {confidence}")
            return False
        
        # Validate description
        description = event_data.get('description', '')
        if not isinstance(description, str) or len(description.strip()) < 10:
            logger.warning("Event description too short or invalid")
            return False
        
        return True
    
    def _classify_event_type(self, description: str) -> str:
        """
        Classify event type based on description keywords.
        
        Args:
            description: Event description
            
        Returns:
            Event type string
        """
        description_lower = description.lower()
        
        # Score each event type based on keyword matches
        type_scores = {}
        for event_type, keywords in self.event_types.items():
            score = sum(1 for keyword in keywords if keyword in description_lower)
            if score > 0:
                type_scores[event_type] = score
        
        # Return the type with highest score, or default
        if type_scores:
            return max(type_scores, key=type_scores.get)
        else:
            return 'market_movement'  # Default fallback
    
    def _generate_event_id(self, event: Dict[str, Any]) -> str:
        """
        Generate unique ID for an event based on its content.
        
        Args:
            event: Event dictionary
            
        Returns:
            Unique event ID string
        """
        # Create hash from key event properties
        id_components = [
            event.get('description', ''),
            event.get('date', ''),
            event.get('event_type', ''),
            str(event.get('entities', []))
        ]
        
        id_string = '|'.join(id_components)
        event_hash = hashlib.md5(id_string.encode()).hexdigest()[:12]
        
        return f"event_{event_hash}"
    
    def verify_events_across_sources(
        self,
        events: List[Dict[str, Any]],
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Verify events by finding cross-source confirmation.
        
        Args:
            events: List of extracted events
            similarity_threshold: Minimum similarity for event matching
            
        Returns:
            List of verified events with updated confidence scores
        """
        verified_events = []
        processed_events = set()
        
        for i, event in enumerate(events):
            if i in processed_events:
                continue
            
            # Find similar events from other sources
            similar_events = [event]  # Start with the current event
            similar_indices = {i}
            
            for j, other_event in enumerate(events[i+1:], i+1):
                if j in processed_events:
                    continue
                
                # Check if events are similar and from different sources
                if (self._events_are_similar(event, other_event, similarity_threshold) and
                    event.get('source') != other_event.get('source')):
                    similar_events.append(other_event)
                    similar_indices.add(j)
            
            # Mark all similar events as processed
            processed_events.update(similar_indices)
            
            # Create verified event with enhanced confidence
            verified_event = self._merge_similar_events(similar_events)
            verified_events.append(verified_event)
        
        logger.info(f"Verified {len(verified_events)} unique events from {len(events)} extracted events")
        return verified_events
    
    def _events_are_similar(
        self,
        event1: Dict[str, Any],
        event2: Dict[str, Any],
        threshold: float
    ) -> bool:
        """
        Check if two events are similar enough to be considered the same.
        
        Args:
            event1: First event
            event2: Second event
            threshold: Similarity threshold
            
        Returns:
            True if events are similar, False otherwise
        """
        # Compare descriptions using simple word overlap
        desc1_words = set(event1.get('description', '').lower().split())
        desc2_words = set(event2.get('description', '').lower().split())
        
        if not desc1_words or not desc2_words:
            return False
        
        overlap = len(desc1_words.intersection(desc2_words))
        total_words = len(desc1_words.union(desc2_words))
        
        description_similarity = overlap / total_words if total_words > 0 else 0.0
        
        # Check event type match
        type_match = event1.get('event_type') == event2.get('event_type')
        
        # Check date proximity (if both have dates)
        date_match = True
        if (event1.get('date') and event1['date'] != 'unknown' and
            event2.get('date') and event2['date'] != 'unknown'):
            try:
                date1 = datetime.fromisoformat(event1['date'])
                date2 = datetime.fromisoformat(event2['date'])
                date_diff = abs((date1 - date2).days)
                date_match = date_diff <= 1  # Within 1 day
            except:
                date_match = True  # If date parsing fails, don't penalize
        
        # Combine similarity factors
        overall_similarity = description_similarity
        if type_match:
            overall_similarity += 0.1
        if date_match:
            overall_similarity += 0.1
        
        return overall_similarity >= threshold
    
    def _merge_similar_events(self, similar_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge similar events into a single verified event.
        
        Args:
            similar_events: List of similar events to merge
            
        Returns:
            Merged event with enhanced confidence
        """
        if len(similar_events) == 1:
            return similar_events[0]
        
        # Use the event with highest confidence as base
        base_event = max(similar_events, key=lambda e: e.get('confidence', 0.0))
        merged_event = base_event.copy()
        
        # Update confidence based on cross-source verification
        source_count = len(set(e.get('source') for e in similar_events if e.get('source')))
        confidence_boost = min(0.3, (source_count - 1) * 0.1)  # Up to 0.3 boost
        
        merged_event['confidence'] = min(1.0, base_event.get('confidence', 0.0) + confidence_boost)
        merged_event['source_count'] = source_count
        merged_event['verified'] = source_count > 1
        
        # Collect all source article IDs
        source_articles = []
        for event in similar_events:
            if event.get('source_article_id'):
                source_articles.append(event['source_article_id'])
        merged_event['source_articles'] = list(set(source_articles))
        
        # Merge entities
        all_entities = []
        for event in similar_events:
            if event.get('entities'):
                all_entities.extend(event['entities'])
        merged_event['entities'] = list(set(all_entities))
        
        # Use the most specific date if available
        dates = [e.get('date') for e in similar_events if e.get('date') and e['date'] != 'unknown']
        if dates:
            merged_event['date'] = dates[0]  # Use first non-unknown date
        
        return merged_event

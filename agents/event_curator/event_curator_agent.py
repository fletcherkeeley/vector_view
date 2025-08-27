"""
Event Curator Agent for Vector View Financial Intelligence Platform

Extracts, verifies, and curates structured events from news articles,
storing them in Neo4j for relationship modeling and cross-domain analysis.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from ..base_agent import BaseAgent, AgentType, AgentResponse, AgentContext, StandardizedSignals
from ..ai_service import OllamaService
from .event_data_handler import EventDataHandler
from .event_context_builder import EventContextBuilder

logger = logging.getLogger(__name__)


class EventCuratorAgent(BaseAgent):
    """
    Event Curator Agent specializing in extracting and curating factual events.
    
    Capabilities:
    - Extract structured events from news articles using AI
    - Verify events through cross-source confirmation
    - Deduplicate similar events across sources
    - Store events in Neo4j with relationship modeling
    - Track event confidence and source verification
    - Generate event-based insights for other agents
    
    Architecture:
    - Uses EventDataHandler for all data access operations
    - Uses EventContextBuilder for AI-powered event extraction
    - Focuses on orchestration and response generation
    """
    
    def __init__(
        self,
        database_url: str = "postgresql://postgres:fred_password@localhost:5432/postgres",
        ai_service: OllamaService = None,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_username: str = "neo4j",
        neo4j_password: str = "vector_view_password"
    ):
        super().__init__(
            agent_type=AgentType.RESEARCH,  # Using RESEARCH as closest match for event curation
            database_url=database_url
        )
        
        # Initialize AI service
        self.ai_service = ai_service or OllamaService()
        
        # Initialize data handler and context builder
        self.data_handler = EventDataHandler(
            database_url=database_url,
            neo4j_uri=neo4j_uri,
            neo4j_username=neo4j_username,
            neo4j_password=neo4j_password
        )
        
        self.context_builder = EventContextBuilder(ai_service=self.ai_service)
        
        self.logger.info("EventCuratorAgent initialized successfully")
    
    async def analyze(self, context: AgentContext) -> AgentResponse:
        """
        Main analysis method for event curation.
        
        Args:
            context: AgentContext containing query and parameters
            
        Returns:
            AgentResponse with extracted and curated events
        """
        start_time = datetime.now()
        
        try:
            # Parse context for event curation parameters
            curation_params = self._parse_curation_context(context)
            
            # Fetch news articles for processing
            articles = await self._fetch_articles_for_curation(curation_params)
            
            if not articles:
                return self._create_empty_response(context, "No articles found for event curation")
            
            # Extract events from articles using AI with batch processing
            extracted_events = await self.context_builder.extract_events_from_articles(
                articles, 
                max_events_per_article=curation_params.get('max_events_per_article', 3),
                batch_size=curation_params.get('batch_size', 3)  # Reduced batch size to prevent CUDA overload
            )
            
            if not extracted_events:
                return self._create_empty_response(context, "No events extracted from articles")
            
            # Enhanced semantic deduplication using ChromaDB
            deduplicated_events = await self.data_handler.deduplicate_events_semantic(
                extracted_events,
                similarity_threshold=curation_params.get('similarity_threshold', 0.6)  # Lowered for better merging
            )
            
            # Verify events with ChromaDB cross-source verification
            verified_events = []
            for event in deduplicated_events:
                enhanced_event = await self.data_handler.verify_event_with_chromadb(
                    event,
                    verification_threshold=0.020,  # Moderate threshold based on embedding analysis
                    max_supporting_articles=5
                )
                verified_events.append(enhanced_event)
            
            # Fallback to original verification for events without ChromaDB support
            if not verified_events:
                verified_events = self.context_builder.verify_events_across_sources(
                    extracted_events,
                    similarity_threshold=curation_params.get('similarity_threshold', 0.7)
                )
            
            # Store events in Neo4j
            stored_events = await self._store_events_in_neo4j(verified_events)
            
            # Generate analysis and insights
            analysis = self._generate_event_analysis(stored_events, articles)
            insights = self._generate_event_insights(stored_events, verified_events)
            key_metrics = self._calculate_event_metrics(stored_events, extracted_events, articles)
            
            # Create standardized signals for other agents
            signals = self._create_event_signals(stored_events, articles)
            
            # Calculate confidence based on verification success
            confidence = self._calculate_overall_confidence(stored_events, verified_events)
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return AgentResponse(
                agent_type=self.agent_type,
                confidence=confidence,
                confidence_level=self._calculate_confidence_level(confidence),
                analysis=analysis,
                insights=insights,
                key_metrics=key_metrics,
                data_sources_used=self._get_data_sources_used(articles),
                timeframe_analyzed=context.timeframe,
                execution_time_ms=execution_time,
                standardized_signals=signals,
                uncertainty_factors=self._identify_uncertainty_factors(stored_events, articles)
            )
            
        except Exception as e:
            self.logger.error(f"Event curation analysis failed: {e}")
            raise
    
    def get_required_data_sources(self, context: AgentContext) -> List[str]:
        """
        Return list of data sources required for event curation.
        
        Args:
            context: AgentContext containing query information
            
        Returns:
            List of required data source identifiers
        """
        sources = ["news_articles", "neo4j"]
        if self.data_handler.chroma_available:
            sources.append("chromadb")
        return sources
    
    def _parse_curation_context(self, context: AgentContext) -> Dict[str, Any]:
        """
        Parse context for event curation parameters.
        
        Args:
            context: AgentContext
            
        Returns:
            Dictionary with curation parameters
        """
        # Default parameters
        params = {
            'max_articles': 50,
            'days_back': 7,
            'max_events_per_article': 3,
            'similarity_threshold': 0.7,
            'min_confidence': 0.5,
            'categories': None,
            'min_relevance': 0.01
        }
        
        # Parse query for specific parameters
        query_lower = context.query.lower()
        
        # Adjust parameters based on query type
        if context.query_type == "daily_briefing":
            params['max_articles'] = 100
            params['days_back'] = 1
        elif context.query_type == "deep_dive":
            params['max_articles'] = 200
            params['days_back'] = 14
            params['max_events_per_article'] = 5
        
        # Parse specific keywords from query
        if "federal reserve" in query_lower or "fed" in query_lower:
            params['categories'] = ['federal_reserve', 'monetary_policy']
        elif "earnings" in query_lower:
            params['categories'] = ['corporate_earnings']
        elif "market" in query_lower:
            params['categories'] = ['market_volatility', 'corporate_earnings']
        
        return params
    
    async def _fetch_articles_for_curation(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Fetch articles for event curation based on parameters.
        
        Args:
            params: Curation parameters
            
        Returns:
            List of news articles
        """
        try:
            # Fetch articles from PostgreSQL
            articles = await self.data_handler.fetch_news_articles(
                limit=params['max_articles'],
                days_back=params['days_back'],
                categories=params.get('categories'),
                min_relevance=params['min_relevance']
            )
            
            self.logger.info(f"Fetched {len(articles)} articles for event curation")
            return articles
            
        except Exception as e:
            self.logger.error(f"Error fetching articles: {e}")
            return []
    
    async def _store_events_in_neo4j(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Store verified events in Neo4j.
        
        Args:
            events: List of verified events
            
        Returns:
            List of successfully stored events
        """
        stored_events = []
        
        for event in events:
            try:
                # Check for existing similar events using semantic search
                similar_events = await self.data_handler.find_similar_events_semantic(
                    event['description'],
                    similarity_threshold=0.6,  # Lowered for better duplicate detection
                    max_results=5
                )
                
                # Fallback to Neo4j similarity if ChromaDB not available
                if not similar_events:
                    similar_events = self.data_handler.find_similar_events(
                        event['description'],
                        similarity_threshold=0.8,
                        days_back=30
                    )
                
                if similar_events:
                    # Update existing event with additional sources
                    existing_event = similar_events[0]
                    updated_confidence = max(
                        existing_event['confidence'],
                        event['confidence']
                    )
                    
                    success = self.data_handler.update_event_confidence(
                        existing_event['event_id'],
                        updated_confidence,
                        event.get('source_articles', [event.get('source_article_id')])
                    )
                    
                    if success:
                        event['event_id'] = existing_event['event_id']
                        event['updated_existing'] = True
                        stored_events.append(event)
                else:
                    # Store new event
                    source_articles = event.get('source_articles', [])
                    if event.get('source_article_id'):
                        source_articles.append(event['source_article_id'])
                    
                    success = self.data_handler.store_event(event, source_articles)
                    
                    if success:
                        event['stored_new'] = True
                        stored_events.append(event)
                
            except Exception as e:
                self.logger.warning(f"Failed to store event {event.get('event_id')}: {e}")
        
        self.logger.info(f"Successfully stored {len(stored_events)} events in Neo4j")
        return stored_events
    
    def _generate_event_analysis(
        self,
        stored_events: List[Dict[str, Any]],
        articles: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate analysis of curated events.
        
        Args:
            stored_events: Successfully stored events
            articles: Source articles
            
        Returns:
            Analysis dictionary
        """
        # Event type distribution
        event_types = {}
        for event in stored_events:
            event_type = event.get('event_type', 'unknown')
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        # Confidence distribution
        high_confidence = sum(1 for e in stored_events if e.get('confidence', 0) >= 0.8)
        medium_confidence = sum(1 for e in stored_events if 0.5 <= e.get('confidence', 0) < 0.8)
        low_confidence = sum(1 for e in stored_events if e.get('confidence', 0) < 0.5)
        
        # Source verification stats (including ChromaDB verification)
        verified_events = sum(1 for e in stored_events if e.get('verified', False) or e.get('chromadb_verified', False))
        multi_source_events = sum(1 for e in stored_events if e.get('source_count', 1) > 1 or e.get('chromadb_source_count', 0) > 1)
        chromadb_verified = sum(1 for e in stored_events if e.get('chromadb_verified', False))
        
        # Top entities mentioned
        all_entities = []
        for event in stored_events:
            if event.get('entities'):
                all_entities.extend(event['entities'])
        
        entity_counts = {}
        for entity in all_entities:
            entity_counts[entity] = entity_counts.get(entity, 0) + 1
        
        top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            'events': stored_events,  # Include actual events for test access
            'total_events_stored': len(stored_events),
            'total_articles_processed': len(articles),
            'event_type_distribution': event_types,
            'confidence_distribution': {
                'high': high_confidence,
                'medium': medium_confidence,
                'low': low_confidence
            },
            'verification_stats': {
                'verified_events': verified_events,
                'multi_source_events': multi_source_events,
                'chromadb_verified_events': chromadb_verified,
                'verification_rate': verified_events / max(len(stored_events), 1),
                'chromadb_available': self.data_handler.chroma_available
            },
            'top_entities': top_entities,
            'processing_summary': {
                'articles_processed': len(articles),
                'events_extracted': len(stored_events),
                'extraction_rate': len(stored_events) / max(len(articles), 1)
            }
        }
    
    def _generate_event_insights(
        self,
        stored_events: List[Dict[str, Any]],
        verified_events: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Generate insights from curated events.
        
        Args:
            stored_events: Successfully stored events
            verified_events: All verified events
            
        Returns:
            List of insight strings
        """
        insights = []
        
        if not stored_events:
            insights.append("No events were successfully extracted and stored")
            return insights
        
        # Event volume insights
        insights.append(f"Extracted and curated {len(stored_events)} factual events from news articles")
        
        # Verification insights (including ChromaDB)
        verified_count = sum(1 for e in stored_events if e.get('verified', False))
        chromadb_verified_count = sum(1 for e in stored_events if e.get('chromadb_verified', False))
        
        if verified_count > 0:
            insights.append(f"{verified_count} events were verified across multiple sources")
        if chromadb_verified_count > 0:
            insights.append(f"{chromadb_verified_count} events were verified using ChromaDB semantic search")
        
        # Semantic deduplication insights
        semantic_dedup_count = sum(1 for e in stored_events if e.get('semantic_deduplication', False))
        if semantic_dedup_count > 0:
            insights.append(f"{semantic_dedup_count} events were semantically deduplicated")
        
        # High-impact events
        high_impact = [e for e in stored_events if e.get('impact_score', 0) >= 0.7]
        if high_impact:
            insights.append(f"Identified {len(high_impact)} high-impact events requiring attention")
        
        # Event type insights
        event_types = {}
        for event in stored_events:
            event_type = event.get('event_type', 'unknown')
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        if event_types:
            top_type = max(event_types, key=event_types.get)
            insights.append(f"Most common event type: {top_type} ({event_types[top_type]} events)")
        
        # Confidence insights
        avg_confidence = sum(e.get('confidence', 0) for e in stored_events) / len(stored_events)
        insights.append(f"Average event confidence: {avg_confidence:.2f}")
        
        # Recent vs historical
        recent_events = [e for e in stored_events if e.get('date') and e['date'] != 'unknown']
        if recent_events:
            insights.append(f"{len(recent_events)} events have specific dates for timeline analysis")
        
        return insights
    
    def _calculate_event_metrics(
        self,
        stored_events: List[Dict[str, Any]],
        extracted_events: List[Dict[str, Any]],
        articles: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate key metrics for event curation.
        
        Args:
            stored_events: Successfully stored events
            extracted_events: All extracted events
            articles: Source articles
            
        Returns:
            Dictionary of key metrics
        """
        total_articles = len(articles)
        total_extracted = len(extracted_events)
        total_stored = len(stored_events)
        
        # Basic rates
        extraction_rate = total_extracted / max(total_articles, 1)
        storage_success_rate = total_stored / max(total_extracted, 1)
        
        # Quality metrics
        avg_confidence = sum(e.get('confidence', 0) for e in stored_events) / max(total_stored, 1)
        avg_impact_score = sum(e.get('impact_score', 0) for e in stored_events) / max(total_stored, 1)
        
        # Verification metrics (including ChromaDB)
        verified_events = sum(1 for e in stored_events if e.get('verified', False))
        chromadb_verified_events = sum(1 for e in stored_events if e.get('chromadb_verified', False))
        total_verified = verified_events + chromadb_verified_events
        verification_rate = total_verified / max(total_stored, 1)
        
        # Source diversity
        unique_sources = len(set(a.get('source') for a in articles if a.get('source')))
        source_diversity = unique_sources / max(total_articles, 1)
        
        return {
            'articles_processed': total_articles,
            'events_before_deduplication': total_extracted,
            'events_after_deduplication': total_stored,
            'extraction_rate': extraction_rate,
            'storage_success_rate': storage_success_rate,
            'average_confidence': avg_confidence,
            'average_impact_score': avg_impact_score,
            'verification_rate': verification_rate,
            'chromadb_verification': {
                'events_verified': chromadb_verified_events,
                'avg_confidence_boost': sum(e.get('confidence_boost_chromadb', 0) for e in stored_events) / max(total_stored, 1),
                'supporting_articles_found': sum(e.get('chromadb_source_count', 0) for e in stored_events)
            },
            'source_diversity': source_diversity,
            'events_per_article': total_stored / max(total_articles, 1),
            'total_events_curated': float(total_stored)
        }
    
    def _create_event_signals(self, stored_events: List[Dict[str, Any]], articles: List[Dict[str, Any]]) -> StandardizedSignals:
        """
        Create standardized signals for other agents based on curated events.
        
        Args:
            stored_events: Successfully stored events
            articles: Original news articles for source quality assessment
            
        Returns:
            StandardizedSignals object
        """
        signals = StandardizedSignals()
        
        if not stored_events:
            return signals
        
        # Calculate overall market relevance based on event types and impact
        market_relevant_types = ['monetary_policy', 'economic_data', 'market_movement', 'corporate_earnings']
        market_events = [e for e in stored_events if e.get('event_type') in market_relevant_types]
        
        if market_events:
            avg_impact = sum(e.get('impact_score', 0) for e in market_events) / len(market_events)
            signals.market_relevance = min(1.0, avg_impact * 1.2)  # Boost for curation quality
        
        # Calculate credibility based on source quality, confidence, and ChromaDB verification
        avg_confidence = sum(e.get('confidence', 0) for e in stored_events) / len(stored_events)
        avg_source_quality = sum(a.get('quality_score', 0.5) for a in articles) / len(articles)
        
        # Enhanced verification bonus including ChromaDB
        verified_events = sum(1 for e in stored_events if e.get('verified', False))
        chromadb_verified_events = sum(1 for e in stored_events if e.get('chromadb_verified', False))
        total_verified = verified_events + chromadb_verified_events
        verification_bonus = (total_verified / len(stored_events)) * 0.25  # Enhanced bonus for verification
        
        # Additional bonus for semantic deduplication quality
        semantic_dedup_bonus = sum(e.get('confidence_boost_chromadb', 0) for e in stored_events) / len(stored_events)
        
        signals.credibility_score = min(1.0, (avg_confidence * 0.5) + (avg_source_quality * 0.3) + verification_bonus + semantic_dedup_bonus)
        
        # Determine overall sentiment based on event types and descriptions
        positive_types = ['corporate_earnings', 'corporate_action']
        negative_types = ['regulatory', 'geopolitical']
        
        positive_events = sum(1 for e in stored_events if e.get('event_type') in positive_types)
        negative_events = sum(1 for e in stored_events if e.get('event_type') in negative_types)
        
        if positive_events + negative_events > 0:
            sentiment_score = (positive_events - negative_events) / (positive_events + negative_events)
            signals.overall_sentiment = max(-1.0, min(1.0, sentiment_score))
        
        return signals
    
    def _calculate_overall_confidence(
        self,
        stored_events: List[Dict[str, Any]],
        verified_events: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate overall confidence for the curation process.
        
        Args:
            stored_events: Successfully stored events
            verified_events: All verified events
            
        Returns:
            Overall confidence score
        """
        if not stored_events:
            return 0.0
        
        # Base confidence from individual events
        avg_event_confidence = sum(e.get('confidence', 0) for e in stored_events) / len(stored_events)
        
        # Boost for verification success (including ChromaDB)
        verified_count = sum(1 for e in stored_events if e.get('verified', False))
        chromadb_verified_count = sum(1 for e in stored_events if e.get('chromadb_verified', False))
        total_verified = verified_count + chromadb_verified_count
        verification_rate = total_verified / len(stored_events)
        verification_boost = verification_rate * 0.25  # Enhanced boost for better verification
        
        # Boost for storage success
        storage_rate = len(stored_events) / max(len(verified_events), 1)
        storage_boost = min(0.1, storage_rate * 0.1)
        
        overall_confidence = min(1.0, avg_event_confidence + verification_boost + storage_boost)
        return overall_confidence
    
    def _get_data_sources_used(self, articles: List[Dict[str, Any]]) -> List[str]:
        """
        Get list of data sources used in analysis.
        
        Args:
            articles: Source articles
            
        Returns:
            List of data source names
        """
        sources = set()
        sources.add("PostgreSQL")  # For article storage
        sources.add("Neo4j")       # For event storage
        
        # Add news sources
        for article in articles:
            if article.get('source'):
                sources.add(article['source'])
        
        return list(sources)
    
    def _identify_uncertainty_factors(
        self,
        stored_events: List[Dict[str, Any]],
        articles: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Identify factors that contribute to uncertainty in the analysis.
        
        Args:
            stored_events: Successfully stored events
            articles: Source articles
            
        Returns:
            List of uncertainty factors
        """
        factors = []
        
        if not stored_events:
            factors.append("no_events_extracted")
            return factors
        
        # Low confidence events
        low_confidence_events = sum(1 for e in stored_events if e.get('confidence', 0) < 0.5)
        if low_confidence_events > len(stored_events) * 0.3:
            factors.append("high_proportion_low_confidence_events")
        
        # Limited source diversity
        unique_sources = len(set(a.get('source') for a in articles if a.get('source')))
        if unique_sources < 3:
            factors.append("limited_source_diversity")
        
        # Low verification rate (including ChromaDB)
        verified_events = sum(1 for e in stored_events if e.get('verified', False))
        chromadb_verified_events = sum(1 for e in stored_events if e.get('chromadb_verified', False))
        total_verified = verified_events + chromadb_verified_events
        
        if total_verified / len(stored_events) < 0.3:
            factors.append("low_cross_source_verification")
        
        # ChromaDB availability factor
        if not self.data_handler.chroma_available:
            factors.append("chromadb_unavailable")
        
        # Many events without specific dates
        undated_events = sum(1 for e in stored_events if e.get('date') == 'unknown')
        if undated_events > len(stored_events) * 0.5:
            factors.append("many_events_without_specific_dates")
        
        return factors
    
    def _create_empty_response(self, context: AgentContext, reason: str) -> AgentResponse:
        """
        Create an empty response when no events can be processed.
        
        Args:
            context: AgentContext
            reason: Reason for empty response
            
        Returns:
            AgentResponse with minimal data
        """
        return AgentResponse(
            agent_type=self.agent_type,
            confidence=0.0,
            confidence_level=self._calculate_confidence_level(0.0),
            analysis={'reason': reason, 'events_curated': 0},
            insights=[f"Event curation incomplete: {reason}"],
            key_metrics={'total_events_curated': 0.0},
            data_sources_used=[],
            timeframe_analyzed=context.timeframe,
            execution_time_ms=0.0,
            uncertainty_factors=[reason.replace(' ', '_').lower()]
        )
    
    async def curate_events_batch(
        self,
        max_articles: int = 100,
        days_back: int = 7,
        categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Convenience method for batch event curation.
        
        Args:
            max_articles: Maximum articles to process
            days_back: Days to look back for articles
            categories: Optional category filters
            
        Returns:
            Dictionary with curation results
        """
        # Create context for batch processing
        context = AgentContext(
            query="batch event curation",
            query_type="deep_dive",
            timeframe=f"{days_back}d"
        )
        
        # Override default parameters
        self._batch_params = {
            'max_articles': max_articles,
            'days_back': days_back,
            'categories': categories
        }
        
        try:
            response = await self.process_query(context)
            
            return {
                'success': True,
                'events_curated': response.key_metrics.get('total_events_curated', 0),
                'confidence': response.confidence,
                'analysis': response.analysis,
                'insights': response.insights,
                'execution_time_ms': response.execution_time_ms
            }
            
        except Exception as e:
            self.logger.error(f"Batch event curation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'events_curated': 0
            }
        finally:
            # Clean up batch parameters
            if hasattr(self, '_batch_params'):
                delattr(self, '_batch_params')
    
    def get_event_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about curated events.
        
        Returns:
            Dictionary with event statistics
        """
        return self.data_handler.get_event_statistics()

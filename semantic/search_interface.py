"""
Agent Search Interface for Financial Intelligence Platform

Provides a high-level search API designed specifically for AI agents.
Handles complex queries, cross-collection searches, and result correlation.

Features:
- Agent-friendly search methods with context awareness
- Cross-collection correlation (news ↔ economic indicators)
- Query intent classification and routing
- Result ranking and relevance scoring
- Feedback collection for continuous improvement
"""

import asyncio
import logging
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timezone, timedelta
from enum import Enum
import re

from .vector_store import SemanticVectorStore, CollectionType, create_semantic_store

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Types of search queries agents might make"""
    ECONOMIC_ANALYSIS = "economic_analysis"
    NEWS_SENTIMENT = "news_sentiment"
    MARKET_CORRELATION = "market_correlation"
    TREND_ANALYSIS = "trend_analysis"
    IMPACT_ASSESSMENT = "impact_assessment"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    GENERAL_SEARCH = "general_search"


class AgentSearchInterface:
    """
    High-level search interface designed for AI agents
    """
    
    def __init__(self, semantic_store: SemanticVectorStore):
        self.semantic_store = semantic_store
        
    @classmethod
    async def create(cls) -> 'AgentSearchInterface':
        """Factory method to create interface with initialized store"""
        store = await create_semantic_store()
        return cls(store)
    
    def _classify_query_intent(self, query: str) -> QueryIntent:
        """Classify the intent of a search query"""
        query_lower = query.lower()
        
        # Intent classification keywords
        intent_patterns = {
            QueryIntent.ECONOMIC_ANALYSIS: [
                "economic", "gdp", "inflation", "unemployment", "fed", "interest rate",
                "monetary policy", "fiscal policy", "economic indicator"
            ],
            QueryIntent.NEWS_SENTIMENT: [
                "sentiment", "market reaction", "investor confidence", "public opinion",
                "news impact", "media coverage"
            ],
            QueryIntent.MARKET_CORRELATION: [
                "correlation", "relationship", "impact on market", "stock price",
                "market movement", "trading volume"
            ],
            QueryIntent.TREND_ANALYSIS: [
                "trend", "pattern", "historical", "over time", "seasonal",
                "long term", "short term"
            ],
            QueryIntent.IMPACT_ASSESSMENT: [
                "impact", "effect", "consequence", "result", "outcome",
                "influence", "affect"
            ],
            QueryIntent.COMPARATIVE_ANALYSIS: [
                "compare", "versus", "vs", "difference", "similar", "contrast"
            ]
        }
        
        # Score each intent
        intent_scores = {}
        for intent, keywords in intent_patterns.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                intent_scores[intent] = score
        
        # Return highest scoring intent or general search
        if intent_scores:
            return max(intent_scores.items(), key=lambda x: x[1])[0]
        else:
            return QueryIntent.GENERAL_SEARCH
    
    def _extract_time_filters(self, query: str) -> Optional[Tuple[datetime, datetime]]:
        """Extract time-based filters from query"""
        query_lower = query.lower()
        now = datetime.now(timezone.utc)
        
        # Time period patterns
        if any(term in query_lower for term in ["last week", "past week"]):
            return (now - timedelta(weeks=1), now)
        elif any(term in query_lower for term in ["last month", "past month"]):
            return (now - timedelta(days=30), now)
        elif any(term in query_lower for term in ["last year", "past year"]):
            return (now - timedelta(days=365), now)
        elif "today" in query_lower:
            return (now.replace(hour=0, minute=0, second=0), now)
        elif "yesterday" in query_lower:
            yesterday = now - timedelta(days=1)
            return (yesterday.replace(hour=0, minute=0, second=0), 
                   yesterday.replace(hour=23, minute=59, second=59))
        
        # Look for specific date patterns (YYYY-MM-DD)
        date_pattern = r'\d{4}-\d{2}-\d{2}'
        dates = re.findall(date_pattern, query)
        if dates:
            try:
                start_date = datetime.fromisoformat(dates[0]).replace(tzinfo=timezone.utc)
                end_date = datetime.fromisoformat(dates[-1]).replace(tzinfo=timezone.utc) if len(dates) > 1 else now
                return (start_date, end_date)
            except ValueError:
                pass
        
        return None
    
    def _extract_economic_categories(self, query: str) -> List[str]:
        """Extract economic categories from query"""
        query_lower = query.lower()
        categories = []
        
        category_keywords = {
            "federal_reserve": ["fed", "federal reserve", "fomc", "interest rate", "monetary policy"],
            "employment": ["employment", "unemployment", "jobs", "labor", "payroll"],
            "inflation": ["inflation", "cpi", "ppi", "price", "deflation"],
            "gdp_growth": ["gdp", "growth", "recession", "expansion", "economic output"],
            "corporate_earnings": ["earnings", "profit", "revenue", "corporate"],
            "market_volatility": ["volatility", "vix", "uncertainty", "risk"],
            "commodity_markets": ["oil", "gold", "commodity", "energy", "metals"],
            "international_trade": ["trade", "tariff", "export", "import", "china"]
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                categories.append(category)
        
        return categories
    
    async def search_with_context(
        self,
        query: str,
        agent_id: str,
        context: Optional[Dict[str, Any]] = None,
        max_results: int = 20
    ) -> Dict[str, Any]:
        """
        Comprehensive search with automatic intent classification and context
        """
        # Classify query intent
        intent = self._classify_query_intent(query)
        
        # Extract filters from query
        time_range = self._extract_time_filters(query)
        economic_categories = self._extract_economic_categories(query)
        
        # Build search filters
        filters = {}
        if economic_categories:
            # ChromaDB filter for economic categories (simplified)
            pass  # Will be applied post-search
        
        # Determine which collections to search based on intent
        search_news = True
        search_indicators = True
        search_analysis = False
        
        if intent == QueryIntent.NEWS_SENTIMENT:
            search_indicators = False
        elif intent == QueryIntent.ECONOMIC_ANALYSIS:
            search_news = False
        elif intent in [QueryIntent.TREND_ANALYSIS, QueryIntent.IMPACT_ASSESSMENT]:
            search_analysis = True
        
        # Perform multi-collection search
        results = await self.semantic_store.find_related_content(
            query=query,
            search_news=search_news,
            search_indicators=search_indicators,
            search_analysis=search_analysis,
            n_results_per_type=max_results // 2
        )
        
        # Post-process results with filters
        if time_range or economic_categories:
            results = self._apply_post_filters(results, time_range, economic_categories)
        
        # Rank and score results
        ranked_results = self._rank_results(results, query, intent)
        
        # Prepare response
        response = {
            "query": query,
            "intent": intent.value,
            "agent_id": agent_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "filters_applied": {
                "time_range": time_range,
                "economic_categories": economic_categories
            },
            "results": ranked_results,
            "total_results": sum(len(v) for v in ranked_results.values()),
            "search_metadata": {
                "collections_searched": list(results.keys()),
                "max_results_requested": max_results
            }
        }
        
        return response
    
    def _apply_post_filters(
        self,
        results: Dict[str, List[Dict[str, Any]]],
        time_range: Optional[Tuple[datetime, datetime]],
        economic_categories: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Apply post-search filters to results"""
        filtered_results = {}
        
        for collection_name, items in results.items():
            filtered_items = []
            
            for item in items:
                metadata = item.get('metadata', {})
                
                # Apply time filter
                if time_range and metadata.get('published_at'):
                    try:
                        pub_date = datetime.fromisoformat(metadata['published_at'].replace('Z', '+00:00'))
                        if not (time_range[0] <= pub_date <= time_range[1]):
                            continue
                    except (ValueError, TypeError):
                        pass
                
                # Apply economic category filter
                if economic_categories and metadata.get('economic_categories'):
                    item_categories = metadata['economic_categories'].split(',')
                    if not any(cat in item_categories for cat in economic_categories):
                        continue
                
                filtered_items.append(item)
            
            filtered_results[collection_name] = filtered_items
        
        return filtered_results
    
    def _rank_results(
        self,
        results: Dict[str, List[Dict[str, Any]]],
        query: str,
        intent: QueryIntent
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Rank results based on relevance and intent"""
        ranked_results = {}
        
        for collection_name, items in results.items():
            # Sort by similarity score and apply intent-based boosting
            scored_items = []
            
            for item in items:
                score = item.get('similarity_score', 0)
                metadata = item.get('metadata', {})
                
                # Intent-based score boosting
                if intent == QueryIntent.NEWS_SENTIMENT and 'sentiment_score' in metadata:
                    # Boost items with strong sentiment
                    sentiment = abs(float(metadata.get('sentiment_score', 0)))
                    score += sentiment * 0.1
                
                elif intent == QueryIntent.ECONOMIC_ANALYSIS and collection_name == 'indicators':
                    # Boost high-priority economic indicators
                    priority = int(metadata.get('correlation_priority', 0))
                    score += priority * 0.05
                
                elif intent == QueryIntent.MARKET_CORRELATION and 'related_series' in metadata:
                    # Boost items with related series
                    if metadata.get('related_series'):
                        score += 0.1
                
                item['final_score'] = score
                scored_items.append(item)
            
            # Sort by final score
            ranked_results[collection_name] = sorted(
                scored_items, 
                key=lambda x: x['final_score'], 
                reverse=True
            )
        
        return ranked_results
    
    async def find_economic_news_correlation(
        self,
        economic_series_id: str,
        time_window_days: int = 30,
        max_articles: int = 10
    ) -> Dict[str, Any]:
        """Find news articles correlated with a specific economic indicator"""
        try:
            # First, get the economic indicator details
            indicator_results = await self.semantic_store.search_economic_indicators(
                query=f"series_id:{economic_series_id}",
                n_results=1,
                filters={"series_id": economic_series_id}
            )
            
            if not indicator_results:
                return {"error": f"Economic indicator {economic_series_id} not found"}
            
            indicator = indicator_results[0]
            indicator_title = indicator['metadata'].get('title', economic_series_id)
            
            # Search for related news using the indicator's description
            news_query = f"{indicator_title} economic indicator financial impact"
            
            # Apply time filter
            time_filter = None
            if time_window_days > 0:
                end_date = datetime.now(timezone.utc)
                start_date = end_date - timedelta(days=time_window_days)
                time_filter = (start_date, end_date)
            
            news_results = await self.semantic_store.search_news_articles(
                query=news_query,
                n_results=max_articles * 2  # Get more to filter
            )
            
            # Filter by time if specified
            if time_filter:
                filtered_news = []
                for article in news_results:
                    pub_date_str = article['metadata'].get('published_at')
                    if pub_date_str:
                        try:
                            pub_date = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00'))
                            if time_filter[0] <= pub_date <= time_filter[1]:
                                filtered_news.append(article)
                        except (ValueError, TypeError):
                            continue
                news_results = filtered_news[:max_articles]
            
            return {
                "economic_indicator": {
                    "series_id": economic_series_id,
                    "title": indicator_title,
                    "metadata": indicator['metadata']
                },
                "related_news": news_results,
                "correlation_metadata": {
                    "time_window_days": time_window_days,
                    "search_query": news_query,
                    "articles_found": len(news_results)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to find economic-news correlation: {e}")
            return {"error": str(e)}
    
    async def search_by_economic_event(
        self,
        event_description: str,
        event_date: Optional[datetime] = None,
        search_window_days: int = 7
    ) -> Dict[str, Any]:
        """Search for content related to a specific economic event"""
        try:
            # Build time filter around the event
            time_filter = None
            if event_date:
                start_date = event_date - timedelta(days=search_window_days)
                end_date = event_date + timedelta(days=search_window_days)
                time_filter = (start_date, end_date)
            
            # Search across all relevant collections
            results = await self.semantic_store.find_related_content(
                query=event_description,
                search_news=True,
                search_indicators=True,
                search_analysis=True,
                n_results_per_type=15
            )
            
            # Apply time filter if specified
            if time_filter:
                results = self._apply_post_filters(results, time_filter, [])
            
            return {
                "event_description": event_description,
                "event_date": event_date.isoformat() if event_date else None,
                "search_window_days": search_window_days,
                "results": results,
                "total_results": sum(len(v) for v in results.values())
            }
            
        except Exception as e:
            logger.error(f"Failed to search by economic event: {e}")
            return {"error": str(e)}
    
    async def record_search_feedback(
        self,
        agent_id: str,
        query: str,
        results: List[Dict[str, Any]],
        relevance_scores: List[float]
    ) -> bool:
        """Record agent feedback on search results"""
        try:
            return await self.semantic_store.record_agent_feedback(
                agent_id=agent_id,
                query=query,
                search_results=results,
                relevance_scores=relevance_scores
            )
        except Exception as e:
            logger.error(f"Failed to record search feedback: {e}")
            return False
    
    async def get_search_analytics(self) -> Dict[str, Any]:
        """Get analytics about search usage and performance"""
        try:
            # Get collection stats
            stats = await self.semantic_store.get_collection_stats()
            
            # TODO: Add more detailed analytics from feedback collection
            # This would include:
            # - Most common query types
            # - Average relevance scores by intent
            # - Agent usage patterns
            # - Search performance metrics
            
            return {
                "collection_stats": stats,
                "analytics_note": "Detailed analytics will be available after feedback collection"
            }
            
        except Exception as e:
            logger.error(f"Failed to get search analytics: {e}")
            return {"error": str(e)}


async def create_agent_search_interface() -> AgentSearchInterface:
    """Factory function to create agent search interface"""
    return await AgentSearchInterface.create()


if __name__ == "__main__":
    # Test the agent search interface
    async def test_search_interface():
        try:
            interface = await create_agent_search_interface()
            
            # Test search with context
            result = await interface.search_with_context(
                query="Federal Reserve interest rate decisions impact on market volatility",
                agent_id="test_agent",
                max_results=10
            )
            
            print(f"Search Results: {json.dumps(result, indent=2, default=str)}")
            
        except Exception as e:
            print(f"Test failed: {e}")
    
    asyncio.run(test_search_interface())

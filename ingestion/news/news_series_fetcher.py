"""
News Series Fetcher - Business Logic Layer

This module handles the business logic for fetching and processing news articles.
It uses the NewsClient foundation and formats data for database storage.

Key Features:
- Intelligent keyword-based article searching using existing economic topics
- Maps News API data to our unified database schema
- Automatic economic categorization and relevance scoring
- Quality assessment and duplicate detection
- Integration with existing news topic mappings

Depends on: news_client.py for networking foundation
"""

import logging
import hashlib
import os
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, date, timezone, timedelta
from decimal import Decimal
import re

# Import database enums and helpers
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "database"))
from unified_database_setup import DataSourceType, NewsCategory

from .news_client import NewsClient, NewsDataAPIError

# Configure logging
logger = logging.getLogger(__name__)


class NewsSeriesFetcher:
    """
    High-level business logic for fetching and processing news articles.
    
    This class handles the business operations while delegating networking to NewsClient.
    """
    
    def __init__(self, client: Optional[NewsClient] = None, reputable_sources: Optional[List[str]] = None):
        """
        Initialize the news series fetcher.
        
        Args:
            client: Optional NewsClient instance. If None, creates a new one.
            reputable_sources: List of reputable news sources for quality scoring. If None, uses default list.
        """
        if client is None:
            self.client = NewsClient()
            self._client_owned = True  # Track if we own the client for cleanup
        else:
            self.client = client
            self._client_owned = False
        
        # Economic keywords for intelligent searching
        self.economic_keywords = self._get_economic_keywords()
        
        # Configurable reputable sources for quality scoring
        self.reputable_sources = reputable_sources or self._get_default_reputable_sources()
        
        logger.info(f"News Series Fetcher initialized with {len(self.reputable_sources)} reputable sources")
    
    def _get_default_reputable_sources(self) -> List[str]:
        """
        Get default list of reputable news sources for quality scoring.
        Reads from REPUTABLE_NEWS_SOURCES environment variable if available.
        
        Returns:
            List of reputable source names (case-insensitive)
        """
        # Try to read from environment variable first
        env_sources = os.getenv('REPUTABLE_NEWS_SOURCES')
        if env_sources:
            sources = [source.strip() for source in env_sources.split(',')]
            logger.info(f"Loaded {len(sources)} reputable sources from environment variable")
            return sources
        
        # Fallback to default list
        logger.info("Using default reputable sources list")
        return [
            # Major Financial News
            'reuters',
            'bloomberg', 
            'wall street journal',
            'financial times',
            'marketwatch',
            'cnbc',
            
            # Business & Economic News
            'associated press',
            'cnn business',
            'fox business',
            'yahoo finance',
            'investing.com',
            
            # Government & Research Sources
            'federal reserve',
            'bureau of labor statistics',
            'bureau of economic analysis',
            
            # International Financial
            'ft.com',
            'economist',
            'nikkei'
        ]
    
    def update_reputable_sources(self, sources: List[str]) -> None:
        """
        Update the list of reputable sources for quality scoring.
        
        Args:
            sources: New list of reputable source names (case-insensitive)
        """
        self.reputable_sources = sources
        logger.info(f"Updated reputable sources list to {len(sources)} sources")
    
    async def __aenter__(self):
        """Async context manager entry"""
        if self.client is None:
            self.client = NewsClient()
        await self.client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._client_owned and self.client:
            await self.client.__aexit__(exc_type, exc_val, exc_tb)
    
    def _get_economic_keywords(self) -> Dict[str, Dict[str, Any]]:
        """
        Get HYBRID EXPANDED keywords for comprehensive news searching.
        Core economic categories + broader social/political/industry topics.
        
        Returns:
            Dictionary mapping category -> {keywords, series, assets, priority}
        """
        # These match your existing news_topic_mapping entries
        return {
            # CORE ECONOMIC CATEGORIES (EXPANDED)
            'federal_reserve': {
                'keywords': ['federal reserve', 'fed meeting', 'fed chair', 'jerome powell', 'interest rates', 'monetary policy', 'fomc', 'fed funds rate', 'fed minutes', 'jackson hole', 'quantitative easing', 'tapering', 'dovish', 'hawkish', 'rate hike', 'rate cut'],
                'related_series': ['FEDFUNDS', 'DGS10', 'DGS2', 'T10Y3M'],
                'related_assets': ['XLF', 'SPY', 'QQQ'],
                'priority': 10,
                'category': NewsCategory.FEDERAL_RESERVE
            },
            'employment': {
                'keywords': ['unemployment', 'jobs report', 'payrolls', 'employment', 'labor market', 'hiring', 'jobless claims', 'nonfarm payrolls', 'unemployment rate', 'job openings', 'labor shortage', 'wage growth', 'worker shortage', 'gig economy', 'remote work', 'work from home', 'labor unions'],
                'related_series': ['UNRATE', 'PAYEMS', 'ICSA', 'AHETPI'],
                'related_assets': ['XLY', 'SPY', 'IWM'],
                'priority': 9,
                'category': NewsCategory.EMPLOYMENT
            },
            'inflation': {
                'keywords': ['inflation', 'cpi', 'consumer prices', 'ppi', 'price index', 'cost of living', 'deflation', 'core inflation', 'price surge', 'rising prices', 'inflation data', 'price increases', 'disinflation', 'food prices', 'energy costs', 'housing costs'],
                'related_series': ['CPIAUCSL', 'AHETPI'],
                'related_assets': ['XLP', 'XLE'],
                'priority': 9,
                'category': NewsCategory.INFLATION
            },
            'gdp_growth': {
                'keywords': ['gdp', 'economic growth', 'recession', 'expansion', 'gdp report', 'economic data', 'economic outlook', 'growth forecast', 'economic recovery', 'economic slowdown', 'business investment', 'consumer spending', 'productivity'],
                'related_series': ['GDP', 'PERMIT', 'HOUST'],
                'related_assets': ['SPY', 'XLI', 'QQQ'],
                'priority': 8,
                'category': NewsCategory.GDP_GROWTH
            },
            'market_volatility': {
                'keywords': ['market volatility', 'vix', 'market crash', 'correction', 'market selloff', 'bear market', 'bull market', 'stock market', 'market rally', 'market decline', 'trading halt', 'circuit breaker', 'market uncertainty', 'investor sentiment'],
                'related_series': ['VIXCLS', 'TEDRATE'],
                'related_assets': ['SPY', 'QQQ', 'VTI'],
                'priority': 7,
                'category': NewsCategory.MARKET_VOLATILITY
            },
            'corporate_earnings': {
                'keywords': ['earnings report', 'quarterly earnings', 'corporate profits', 'earnings season', 'revenue', 'earnings beat', 'earnings miss', 'guidance', 'profit margin', 'dividend', 'share buyback', 'corporate results', 'ceo', 'layoffs'],
                'related_series': [],
                'related_assets': ['SPY', 'QQQ'],
                'priority': 6,
                'category': NewsCategory.CORPORATE_EARNINGS
            },
            'geopolitical': {
                'keywords': ['trade war', 'tariffs', 'sanctions', 'geopolitical', 'international trade', 'trade deal', 'china trade', 'brexit', 'election', 'political risk', 'foreign policy', 'diplomatic', 'global tensions', 'nato', 'ukraine', 'russia'],
                'related_series': [],
                'related_assets': ['SPY', 'EFA', 'EEM'],
                'priority': 6,
                'category': NewsCategory.GEOPOLITICAL
            },
            
            # BROADER CATEGORIES (NEW)
            'technology_disruption': {
                'keywords': ['artificial intelligence', 'ai', 'machine learning', 'automation', 'robotics', 'cryptocurrency', 'bitcoin', 'blockchain', 'fintech', 'digital transformation', 'cloud computing', 'cybersecurity', 'data breach', 'tech regulation', 'social media', 'streaming', 'electric vehicles', 'autonomous vehicles'],
                'related_series': [],
                'related_assets': ['XLK', 'QQQ', 'ARKK'],
                'priority': 8,
                'category': NewsCategory.SECTOR_SPECIFIC
            },
            'supply_chain_logistics': {
                'keywords': ['supply chain', 'logistics', 'shipping', 'container', 'freight', 'supply shortage', 'chip shortage', 'semiconductor', 'manufacturing', 'factory', 'production', 'inventory', 'warehouse', 'distribution', 'raw materials', 'commodities'],
                'related_series': [],
                'related_assets': ['XLI', 'FDX', 'UPS'],
                'priority': 7,
                'category': NewsCategory.COMMODITY_MARKETS
            },
            'energy_climate': {
                'keywords': ['oil prices', 'crude oil', 'natural gas', 'energy crisis', 'opec', 'renewable energy', 'solar', 'wind', 'climate change', 'carbon', 'esg', 'sustainability', 'green energy', 'electric grid', 'pipeline', 'energy policy', 'carbon tax', 'emissions'],
                'related_series': [],
                'related_assets': ['XLE', 'ICLN', 'USO'],
                'priority': 7,
                'category': NewsCategory.COMMODITY_MARKETS
            },
            'consumer_social_trends': {
                'keywords': ['consumer behavior', 'spending habits', 'millennials', 'gen z', 'social trends', 'lifestyle', 'health trends', 'fitness', 'wellness', 'streaming', 'gaming', 'e-commerce', 'online shopping', 'retail trends', 'food delivery', 'subscription', 'sharing economy'],
                'related_series': ['AHETPI'],
                'related_assets': ['XLY', 'AMZN', 'NFLX'],
                'priority': 6,
                'category': NewsCategory.GDP_GROWTH
            },
            'political_policy': {
                'keywords': ['congress', 'senate', 'house', 'biden', 'trump', 'politics', 'legislation', 'bill', 'law', 'regulation', 'policy', 'government', 'stimulus', 'infrastructure', 'healthcare reform', 'tax policy', 'immigration', 'voting', 'midterm', 'campaign'],
                'related_series': [],
                'related_assets': ['SPY', 'VTI'],
                'priority': 6,
                'category': NewsCategory.GEOPOLITICAL
            },
            'social_movements': {
                'keywords': ['protest', 'strike', 'labor dispute', 'union', 'social movement', 'demonstration', 'activism', 'civil rights', 'inequality', 'minimum wage', 'worker rights', 'social justice', 'diversity', 'pandemic', 'covid', 'public health', 'vaccine'],
                'related_series': ['UNRATE', 'AHETPI'],
                'related_assets': ['SPY', 'XLY'],
                'priority': 5,
                'category': NewsCategory.EMPLOYMENT
            },
            
            # DAILY UPDATER CATEGORIES
            'financial_markets': {
                'keywords': ['stock market', 'dow jones', 'nasdaq', 's&p 500', 'market rally', 'market decline', 'trading volume', 'market volatility', 'bull market', 'bear market', 'market correction', 'stock prices', 'equity markets', 'market sentiment', 'investor confidence'],
                'related_series': ['VIXCLS', 'TEDRATE'],
                'related_assets': ['SPY', 'QQQ', 'VTI'],
                'priority': 7,
                'category': NewsCategory.MARKET_VOLATILITY
            },
            'banking': {
                'keywords': ['banks', 'banking', 'credit', 'loans', 'mortgages', 'financial institutions', 'bank earnings', 'credit risk', 'lending', 'deposits', 'bank regulation', 'basel', 'stress test', 'bank capital', 'financial services'],
                'related_series': ['FEDFUNDS', 'DGS10', 'TEDRATE'],
                'related_assets': ['XLF', 'JPM', 'BAC'],
                'priority': 6,
                'category': NewsCategory.FEDERAL_RESERVE
            },
            'trade': {
                'keywords': ['international trade', 'exports', 'imports', 'trade deficit', 'trade surplus', 'tariffs', 'trade war', 'trade deal', 'wto', 'nafta', 'usmca', 'trade policy', 'customs', 'trade balance', 'global trade'],
                'related_series': [],
                'related_assets': ['SPY', 'EFA', 'EEM'],
                'priority': 6,
                'category': NewsCategory.GEOPOLITICAL
            },
            'housing': {
                'keywords': ['housing market', 'real estate', 'home prices', 'mortgage rates', 'housing starts', 'building permits', 'home sales', 'housing bubble', 'foreclosure', 'rental market', 'construction', 'homebuilders', 'housing affordability'],
                'related_series': ['PERMIT', 'HOUST', 'DGS10'],
                'related_assets': ['XLRE', 'XHB', 'ITB'],
                'priority': 7,
                'category': NewsCategory.GDP_GROWTH
            },
            'consumer_spending': {
                'keywords': ['consumer spending', 'retail sales', 'consumer confidence', 'personal consumption', 'disposable income', 'consumer behavior', 'retail earnings', 'e-commerce', 'shopping', 'consumer debt', 'credit card spending', 'consumer sentiment'],
                'related_series': ['AHETPI'],
                'related_assets': ['XLY', 'AMZN', 'WMT'],
                'priority': 6,
                'category': NewsCategory.GDP_GROWTH
            },
            'energy': {
                'keywords': ['oil prices', 'crude oil', 'natural gas', 'energy sector', 'opec', 'energy companies', 'gasoline prices', 'energy crisis', 'renewable energy', 'oil production', 'energy policy', 'petroleum', 'energy stocks'],
                'related_series': ['WPUFD49207'],
                'related_assets': ['XLE', 'USO', 'UNG'],
                'priority': 6,
                'category': NewsCategory.COMMODITY_MARKETS
            }
        }
    
    async def search_economic_news(
        self,
        categories: Optional[List[str]] = None,
        days_back: int = 1,
        max_articles_per_category: int = 20
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search for economically relevant news articles across categories.
        
        Args:
            categories: List of economic categories to search (if None, searches all)
            days_back: Number of days back to search
            max_articles_per_category: Maximum articles per category
            
        Returns:
            Dictionary mapping category -> list of processed articles
        """
        logger.info(f"Searching economic news for last {days_back} days")
        
        # Default to all categories if none specified
        if categories is None:
            categories = list(self.economic_keywords.keys())
        
        # Calculate date range
        end_date = date.today()
        start_date = end_date - timedelta(days=days_back)
        
        results = {}
        
        for category in categories:
            if category not in self.economic_keywords:
                logger.warning(f"Unknown category: {category}")
                continue
            
            try:
                logger.info(f"Searching articles for category: {category}")
                articles = await self._search_category_articles(
                    category, 
                    start_date, 
                    end_date, 
                    max_articles_per_category
                )
                results[category] = articles
                logger.info(f"Found {len(articles)} articles for {category}")
                
            except Exception as e:
                logger.error(f"Error searching articles for {category}: {e}")
                results[category] = []
        
        total_articles = sum(len(articles) for articles in results.values())
        logger.info(f"Total articles found: {total_articles}")
        
        return results
    
    async def fetch_category_articles(
        self,
        category: str,
        start_date: date,
        end_date: date,
        max_articles: int = 100,
        max_api_calls: int = None
    ) -> List[Dict[str, Any]]:
        """
        Public method to fetch articles for a specific category.
        
        Args:
            category: Economic category to search for
            start_date: Start date for article search
            end_date: End date for article search
            max_articles: Maximum number of articles to return
            max_api_calls: Maximum API calls to use (for compatibility)
            
        Returns:
            List of processed articles for the category
        """
        # Use API call-based search if max_api_calls is provided
        if max_api_calls is not None:
            return await self._search_category_articles_by_api_calls(category, start_date, end_date, max_api_calls)
        return await self._search_category_articles(category, start_date, end_date, max_articles)

    async def _search_category_articles_by_api_calls(
        self,
        category: str,
        start_date: date,
        end_date: date,
        max_api_calls: int
    ) -> List[Dict[str, Any]]:
        """
        Search for articles using API call budget instead of article limit.
        
        Args:
            category: Economic category to search
            start_date: Start date for search
            end_date: End date for search
            max_api_calls: Maximum API calls to make
            
        Returns:
            List of processed article dictionaries
        """
        category_info = self.economic_keywords[category]
        keywords = category_info['keywords']
        
        articles = []
        api_calls_made = 0
        seen_urls = set()  # Deduplicate articles
        
        logger.info(f"Starting API call-based search for {category} with {max_api_calls} calls")
        
        # Distribute API calls across keywords
        calls_per_keyword = max(1, max_api_calls // len(keywords))
        
        for keyword in keywords:
            if api_calls_made >= max_api_calls:
                break
                
            keyword_calls = 0
            page = 1
            
            while keyword_calls < calls_per_keyword and api_calls_made < max_api_calls:
                try:
                    # Make API call - use archive for date range searches
                    if (date.today() - start_date).days > 2:
                        response = await self.client.search_archive(
                            q=keyword,
                            from_date=start_date.strftime('%Y-%m-%d'),
                            to_date=end_date.strftime('%Y-%m-%d'),
                            language='en',
                            size=50  # NewsData.io max for free tier
                        )
                    else:
                        response = await self.client.search_latest(
                            q=keyword,
                            language='en',
                            size=50  # NewsData.io max for free tier
                        )
                    
                    api_calls_made += 1
                    keyword_calls += 1
                    
                    raw_articles = response.get('results', [])
                    
                    # If no articles returned, stop paginating this keyword
                    if not raw_articles:
                        break
                    
                    # Process each article
                    new_articles_count = 0
                    for raw_article in raw_articles:
                        url = raw_article.get('link', '')
                        if url and url not in seen_urls:
                            processed_article = self._process_article(raw_article, category)
                            if processed_article and self._is_economically_relevant(processed_article, category):
                                articles.append(processed_article)
                                seen_urls.add(url)
                                new_articles_count += 1
                    
                    logger.debug(f"Keyword '{keyword}' page {page}: {new_articles_count} new articles (API calls: {api_calls_made}/{max_api_calls})")
                    
                    # If we got less than 50 articles, we've hit the end
                    if len(raw_articles) < 50:
                        break
                        
                    page += 1
                    
                except NewsDataAPIError as e:
                    logger.warning(f"API error for keyword '{keyword}' page {page}: {e}")
                    break
                except Exception as e:
                    logger.error(f"Unexpected error for keyword '{keyword}' page {page}: {e}")
                    break
        
        logger.info(f"Completed search for {category}: {len(articles)} articles using {api_calls_made} API calls")
        return articles
    
    async def _search_category_articles(
        self,
        category: str,
        start_date: date,
        end_date: date,
        max_articles: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Search for articles in a specific economic category (legacy method).
        
        Args:
            category: Economic category to search
            start_date: Start date for search
            end_date: End date for search
            max_articles: Maximum articles to return
            
        Returns:
            List of processed article dictionaries
        """
        category_info = self.economic_keywords[category]
        keywords = category_info['keywords']
        
        articles = []
        
        # Search with ALL keyword combinations for maximum coverage
        for keyword in keywords:  # Use ALL keywords for comprehensive collection
            try:
                # Search for articles with this keyword
                if (date.today() - start_date).days > 2:
                    response = await self.client.search_archive(
                        q=keyword,
                        from_date=start_date.strftime('%Y-%m-%d'),
                        to_date=end_date.strftime('%Y-%m-%d'),
                        language='en',
                        size=min(max_articles, 50)  # NewsData.io max for free tier
                    )
                else:
                    response = await self.client.search_latest(
                        q=keyword,
                        language='en',
                        size=min(max_articles, 50)  # NewsData.io max for free tier
                    )
                
                raw_articles = response.get('results', [])
                
                # Process each article
                for raw_article in raw_articles:
                    processed_article = self._process_article(raw_article, category)
                    if processed_article and self._is_economically_relevant(processed_article, category):
                        articles.append(processed_article)
                
                # Stop if we have enough articles
                if len(articles) >= max_articles:
                    break
                    
            except NewsDataAPIError as e:
                logger.error(f"Error searching for keyword '{keyword}': {e}")
                continue
        
        # Remove duplicates and sort by relevance
        articles = self._deduplicate_articles(articles)
        articles = sorted(articles, key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return articles[:max_articles]
    
    def _process_article(self, raw_article: Dict[str, Any], category: str) -> Optional[Dict[str, Any]]:
        """
        Process a raw News API article into our database schema format.
        
        Args:
            raw_article: Raw article data from News API
            category: Economic category this article was found under
            
        Returns:
            Dictionary formatted for news_articles table, or None if invalid
        """
        try:
            # Extract basic information - NewsData.io uses 'link' instead of 'url'
            url = raw_article.get('link', '')
            if not url:
                return None
            
            title = raw_article.get('title', '')
            if not title or title.lower() == '[removed]':
                return None
            
            # Parse publication date - NewsData.io uses 'pubDate'
            published_at_str = raw_article.get('pubDate')
            if not published_at_str:
                return None
            
            try:
                # NewsData.io uses different date format
                published_at = datetime.fromisoformat(published_at_str.replace('Z', '+00:00'))
            except ValueError:
                logger.warning(f"Invalid date format: {published_at_str}")
                return None
            
            # Extract content information
            description = raw_article.get('description', '')
            content = raw_article.get('content', '')
            
            # Calculate content metrics
            content_length = len(content) if content else 0
            
            # Extract source information - NewsData.io uses 'source_id' instead of nested source object
            source_name = raw_article.get('source_id', 'Unknown')
            
            # Create URL hash for deduplication
            url_hash = hashlib.sha256(url.encode('utf-8')).hexdigest()
            
            # Calculate quality and relevance scores
            quality_score = self._calculate_quality_score(raw_article)
            relevance_score = self._calculate_relevance_score(raw_article, category)
            
            # Get category information
            category_info = self.economic_keywords.get(category, {})
            
            return {
                'source_article_id': raw_article.get('article_id'),  # NewsData.io provides stable IDs
                'url': url,
                'url_hash': url_hash,
                'source_name': source_name,
                'source_domain': self._extract_domain(url),
                'author': raw_article.get('creator'),  # NewsData.io uses 'creator' instead of 'author'
                'published_at': published_at,
                'title': title,
                'description': description,
                'content': content,
                'content_length': content_length,
                'language': raw_article.get('language', 'en')[:10],  # Truncate to fit DB field
                'country': self._normalize_country(raw_article.get('country', ['us'])),
                'news_api_metadata': raw_article,  # Store original NewsData.io data
                'economic_categories': [category],
                'sentiment_score': None,  # Will be calculated later
                'relevance_score': relevance_score,
                'vector_db_collection': None,  # Will be set during vectorization
                'vector_db_document_id': None,
                'embedding_model_version': None,
                'is_processed': False,
                'is_categorized': True,  # We're categorizing it now
                'has_embeddings': False,
                'data_quality_score': quality_score,
                'content_completeness': self._calculate_content_completeness(raw_article),
                'duplicate_probability': Decimal('0.0'),  # Will be calculated during deduplication
                'processing_attempts': 0,
                'last_processing_error': None,
                'related_series_ids': category_info.get('related_series', []),
                'related_market_assets': category_info.get('related_assets', []),
                'impact_timeframe': self._estimate_impact_timeframe(category),
                'created_at': datetime.now(timezone.utc),
                'updated_at': datetime.now(timezone.utc),
                'processed_at': None
            }
            
        except Exception as e:
            logger.warning(f"Error processing article: {e}")
            return None
    
    def _calculate_quality_score(self, article: Dict[str, Any]) -> Decimal:
        """Calculate a data quality score for the article"""
        score = Decimal('0.5')  # Base score
        
        # Boost for having content
        if article.get('content') and len(article['content']) > 200:
            score += Decimal('0.2')
        
        # Boost for having description
        if article.get('description') and len(article['description']) > 50:
            score += Decimal('0.1')
        
        # Boost for having author
        if article.get('author'):
            score += Decimal('0.1')
        
        # Boost for reputable sources (configurable) - NewsData.io uses source_id
        source_name = article.get('source_id', '').lower()
        if any(source.lower() in source_name for source in self.reputable_sources):
            score += Decimal('0.1')
        
        return min(score, Decimal('1.0'))
    
    def _calculate_relevance_score(self, article: Dict[str, Any], category: str) -> Decimal:
        """Calculate economic relevance score for the article"""
        category_info = self.economic_keywords.get(category, {})
        keywords = category_info.get('keywords', [])
        
        # Get article text for analysis
        text = f"{article.get('title', '')} {article.get('description', '')} {article.get('content', '')}".lower()
        
        # Count keyword matches
        keyword_matches = sum(1 for keyword in keywords if keyword.lower() in text)
        max_possible_matches = len(keywords)
        
        if max_possible_matches == 0:
            return Decimal('0.5')
        
        # Base relevance from keyword density
        relevance = Decimal(str(keyword_matches / max_possible_matches))
        
        # Boost for title matches (more important)
        title_text = article.get('title', '').lower()
        title_matches = sum(1 for keyword in keywords if keyword.lower() in title_text)
        if title_matches > 0:
            relevance += Decimal('0.2')
        
        return min(relevance, Decimal('1.0'))
    
    def _calculate_content_completeness(self, article: Dict[str, Any]) -> Decimal:
        """Calculate how complete the article content is"""
        completeness = Decimal('0.0')
        
        if article.get('title'):
            completeness += Decimal('0.3')
        if article.get('description'):
            completeness += Decimal('0.3')
        if article.get('content') and len(article['content']) > 100:
            completeness += Decimal('0.4')
        
        return completeness
    
    def _is_economically_relevant(self, article: Dict[str, Any], category: str) -> bool:
        """
        Determine if an article is economically relevant enough to store.
        LOWERED thresholds for broader collection - AI synthesis will handle connections.
        """
        relevance_score = article.get('relevance_score', 0)
        quality_score = article.get('data_quality_score', 0)
        
        # VERY LOW thresholds for maximum collection - let vector DB handle filtering
        min_relevance = Decimal('0.001')  # Almost no threshold - collect everything
        min_quality = Decimal('0.40')    # Lower quality bar for wider collection
        
        return relevance_score >= min_relevance and quality_score >= min_quality
    
    def _deduplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate articles based on URL hash and title similarity"""
        seen_hashes = set()
        seen_titles = set()
        unique_articles = []
        
        for article in articles:
            url_hash = article.get('url_hash')
            title = article.get('title', '').lower().strip()
            
            # Skip if we've seen this URL
            if url_hash in seen_hashes:
                continue
            
            # Skip if we've seen a very similar title
            if title in seen_titles:
                continue
            
            seen_hashes.add(url_hash)
            seen_titles.add(title)
            unique_articles.append(article)
        
        return unique_articles
    
    def _extract_domain(self, url: str) -> Optional[str]:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc
        except:
            return None
    
    def _normalize_country(self, country_data) -> str:
        """Normalize country data to fit database field constraints"""
        if isinstance(country_data, list):
            country = country_data[0] if country_data else 'us'
        else:
            country = country_data or 'us'
        
        # Map common NewsData.io country names to short codes
        country_mapping = {
            'united states of america': 'us',
            'united kingdom': 'uk',
            'canada': 'ca',
            'australia': 'au',
            'germany': 'de',
            'france': 'fr',
            'japan': 'jp',
            'china': 'cn',
            'india': 'in',
            'brazil': 'br'
        }
        
        country_lower = str(country).lower()
        return country_mapping.get(country_lower, country_lower[:10])
    
    def _estimate_impact_timeframe(self, category: str) -> str:
        """Estimate the impact timeframe for different economic categories"""
        timeframes = {
            'federal_reserve': 'immediate',  # Fed announcements have immediate impact
            'employment': 'short_term',      # Jobs data affects markets quickly
            'inflation': 'medium_term',      # Inflation trends develop over time
            'gdp_growth': 'long_term',       # GDP impacts are longer term
            'market_volatility': 'immediate', # Market events are immediate
            'corporate_earnings': 'short_term',
            'geopolitical': 'medium_term',
            'technology_disruption': 'long_term',
            'supply_chain_logistics': 'medium_term',
            'energy_climate': 'medium_term',
            'consumer_social_trends': 'short_term',
            'political_policy': 'medium_term',
            'social_movements': 'short_term'
        }
        return timeframes.get(category, 'medium_term')
    
    async def get_category_summary(self, category: str, days_back: int = 7) -> Dict[str, Any]:
        """
        Get a summary of articles for a specific economic category.
        
        Args:
            category: Economic category to summarize
            days_back: Number of days to look back
            
        Returns:
            Summary statistics for the category
        """
        articles = await self._search_category_articles(
            category, 
            date.today() - timedelta(days=days_back),
            date.today(),
            100  # Get more articles for summary
        )
        
        if not articles:
            return {
                'category': category,
                'article_count': 0,
                'average_relevance': 0,
                'average_quality': 0,
                'top_sources': [],
                'date_range': f"Last {days_back} days"
            }
        
        # Calculate summary statistics
        relevance_scores = [float(a.get('relevance_score', 0)) for a in articles]
        quality_scores = [float(a.get('data_quality_score', 0)) for a in articles]
        
        # Count top sources
        sources = [a.get('source_name', 'Unknown') for a in articles]
        source_counts = {}
        for source in sources:
            source_counts[source] = source_counts.get(source, 0) + 1
        
        top_sources = sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'category': category,
            'article_count': len(articles),
            'average_relevance': sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0,
            'average_quality': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            'top_sources': top_sources,
            'date_range': f"Last {days_back} days",
            'related_series': self.economic_keywords.get(category, {}).get('related_series', []),
            'related_assets': self.economic_keywords.get(category, {}).get('related_assets', [])
        }


# Convenience function for testing
async def test_news_fetcher():
    """Test function to verify the news fetcher works"""
    
    async with NewsSeriesFetcher() as fetcher:
        try:
            print(f"Using {len(fetcher.reputable_sources)} reputable sources from config:")
            print(f"   {', '.join(fetcher.reputable_sources[:5])}...")
            
            # Test searching for Federal Reserve news
            results = await fetcher.search_economic_news(
                categories=['federal_reserve'], 
                days_back=3, 
                max_articles_per_category=5
            )
            
            fed_articles = results.get('federal_reserve', [])
            print(f"✅ Found {len(fed_articles)} Federal Reserve articles")
            
            if fed_articles:
                article = fed_articles[0]
                print(f"   Title: {article['title'][:100]}...")
                print(f"   Source: {article['source_name']}")
                print(f"   Quality: {article['data_quality_score']}")
                print(f"   Relevance: {article['relevance_score']}")
                print(f"   Related Series: {article['related_series_ids']}")
                
                # Show if source got reputation boost
                source_name = article['source_name'].lower()
                is_reputable = any(source.lower() in source_name for source in fetcher.reputable_sources)
                print(f"   Reputable Source: {'✅ Yes' if is_reputable else '❌ No'}")
            
            # Test category summary
            summary = await fetcher.get_category_summary('federal_reserve', days_back=7)
            print(f"✅ Category summary:")
            print(f"   Articles: {summary['article_count']}")
            print(f"   Avg Relevance: {summary['average_relevance']:.2f}")
            print(f"   Top Sources: {[s[0] for s in summary['top_sources'][:3]]}")
            
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False


if __name__ == "__main__":
    """Test the fetcher when run directly"""
    import asyncio
    asyncio.run(test_news_fetcher())
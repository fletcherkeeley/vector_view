"""
News API Client - Enterprise Foundation Layer

This module provides a robust foundation for interacting with the News API.
Handles networking, rate limiting, error handling, and retries before any business logic.

Key Features:
- Environment-based configuration
- Rate limiting (1,000 requests/day News API limit)
- Comprehensive error handling and logging
- Automatic retry logic with exponential backoff
- Session management for connection pooling
"""

import os
import time
import logging
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone, date, timedelta

import aiohttp
import asyncio_throttle
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


class NewsAPIError(Exception):
    """Custom exception for News API specific errors"""
    pass


class NewsRateLimitError(NewsAPIError):
    """Raised when News API rate limits are exceeded"""
    pass


class NewsClient:
    """
    Enterprise-grade News API client with comprehensive error handling and rate limiting.
    
    This class handles all networking concerns and provides a foundation for business logic.
    """
    
    # News API Configuration
    BASE_URL = "https://newsapi.org/v2"
    DEFAULT_TIMEOUT = 30  # seconds
    MAX_REQUESTS_PER_DAY = 1000  # News API free tier limit
    MAX_REQUESTS_PER_HOUR = 40   # Conservative rate limit (1000/24 = ~42/hour)
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize News API client with configuration and session management.
        
        Args:
            api_key: News API key. If None, loads from NEWS_API_KEY environment variable.
            
        Raises:
            NewsAPIError: If API key is not provided or found in environment
        """
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv('NEWS_API_KEY')
        
        if not self.api_key:
            raise NewsAPIError(
                "News API key is required. Provide it as parameter or set NEWS_API_KEY environment variable."
            )
        
        # Rate limiting setup - News API allows 1000 requests per day
        self.rate_limiter = asyncio_throttle.Throttler(
            rate_limit=self.MAX_REQUESTS_PER_HOUR,
            period=3600  # 60 minutes
        )
        
        # Session will be created when needed (async context)
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Request tracking for monitoring
        self.requests_made = 0
        self.last_request_time: Optional[datetime] = None
        
        logger.info("News API client initialized successfully")
    
    async def __aenter__(self):
        """Async context manager entry - creates HTTP session"""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - closes HTTP session"""
        await self.close()
    
    async def _ensure_session(self) -> None:
        """Ensure HTTP session is created and configured"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.DEFAULT_TIMEOUT)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    'User-Agent': 'AI-Financial-Intelligence-Platform/1.0',
                    'Accept': 'application/json',
                    'X-API-Key': self.api_key
                }
            )
            logger.debug("HTTP session created")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
    )
    async def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make HTTP request to News API with rate limiting, retries, and error handling.
        
        Args:
            endpoint: News API endpoint (e.g., 'everything', 'top-headlines')
            params: Query parameters for the request
            
        Returns:
            Dict containing the JSON response from News API
            
        Raises:
            NewsAPIError: For API-specific errors
            NewsRateLimitError: When rate limits are exceeded
        """
        await self._ensure_session()
        
        # Apply rate limiting
        async with self.rate_limiter:
            url = f"{self.BASE_URL}/{endpoint}"
            
            try:
                logger.debug(f"Making request to {endpoint} with params: {list(params.keys())}")
                
                async with self.session.get(url, params=params) as response:
                    # Update request tracking
                    self.requests_made += 1
                    self.last_request_time = datetime.now(timezone.utc)
                    
                    # Get response text for detailed error messages
                    response_text = await response.text()
                    
                    # Handle different HTTP status codes
                    if response.status == 200:
                        try:
                            data = await response.json()
                            
                            # Check News API specific status
                            if data.get('status') == 'ok':
                                logger.debug(f"Successful request to {endpoint}")
                                return data
                            else:
                                error_code = data.get('code', 'unknown')
                                error_message = data.get('message', 'Unknown error')
                                logger.error(f"News API error {error_code}: {error_message}")
                                
                                if error_code == 'rateLimited':
                                    raise NewsRateLimitError(f"Rate limit exceeded: {error_message}")
                                else:
                                    raise NewsAPIError(f"News API error ({error_code}): {error_message}")
                                    
                        except ValueError as e:
                            logger.error(f"Invalid JSON response from {endpoint}: {response_text[:200]}")
                            raise NewsAPIError(f"Invalid JSON response: {e}")
                    
                    elif response.status == 429:
                        logger.warning("News API rate limit exceeded")
                        raise NewsRateLimitError("Rate limit exceeded. Please wait before making more requests.")
                    
                    elif response.status == 401:
                        logger.error("News API authentication failed - check API key")
                        raise NewsAPIError("Authentication failed. Check your News API key.")
                    
                    elif response.status == 400:
                        logger.error(f"Bad request to {endpoint}: {response_text}")
                        raise NewsAPIError(f"Bad request: {response_text}")
                    
                    elif response.status == 426:
                        logger.error("News API upgrade required")
                        raise NewsAPIError("API upgrade required. You may have exceeded your plan limits.")
                    
                    elif response.status == 500:
                        logger.error("News API server error")
                        raise NewsAPIError("News API server error. Please try again later.")
                    
                    else:
                        logger.error(f"HTTP {response.status} error: {response_text}")
                        raise NewsAPIError(f"HTTP {response.status}: {response_text}")
            
            except aiohttp.ClientError as e:
                logger.error(f"Network error during request to {endpoint}: {e}")
                raise
            
            except asyncio.TimeoutError:
                logger.error(f"Timeout during request to {endpoint}")
                raise
    
    async def search_everything(
        self,
        q: Optional[str] = None,
        sources: Optional[str] = None,
        domains: Optional[str] = None,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        language: str = 'en',
        sort_by: str = 'publishedAt',
        page_size: int = 20,
        page: int = 1
    ) -> Dict[str, Any]:
        """
        Search through millions of articles from over 80,000 large and small news sources.
        
        Args:
            q: Keywords or phrases to search for
            sources: Comma-separated news sources or blogs
            domains: Comma-separated domains to restrict search
            from_date: Date to search from (YYYY-MM-DD format)
            to_date: Date to search to (YYYY-MM-DD format)
            language: Language to search for articles in
            sort_by: Sort order ('relevancy', 'popularity', 'publishedAt')
            page_size: Number of results per page (max 100)
            page: Page number to retrieve
            
        Returns:
            Dict containing articles and metadata from News API
            
        Raises:
            NewsAPIError: If search fails or parameters are invalid
        """
        params = {
            'language': language,
            'sortBy': sort_by,
            'pageSize': min(page_size, 100),  # API max is 100
            'page': page
        }
        
        # Add optional parameters
        if q:
            params['q'] = q
        if sources:
            params['sources'] = sources
        if domains:
            params['domains'] = domains
        if from_date:
            params['from'] = from_date
        if to_date:
            params['to'] = to_date
        
        # Validate that at least one search parameter is provided
        if not any([q, sources, domains]):
            raise NewsAPIError("At least one of 'q', 'sources', or 'domains' must be provided")
        
        return await self._make_request('everything', params)
    
    async def get_top_headlines(
        self,
        q: Optional[str] = None,
        sources: Optional[str] = None,
        category: Optional[str] = None,
        country: str = 'us',
        page_size: int = 20,
        page: int = 1
    ) -> Dict[str, Any]:
        """
        Get breaking news headlines for a country or category.
        
        Args:
            q: Keywords to search for in headlines
            sources: Comma-separated news sources
            category: Category ('business', 'entertainment', 'general', 'health', 'science', 'sports', 'technology')
            country: Country code (e.g., 'us', 'gb', 'ca')
            page_size: Number of results per page (max 100)
            page: Page number to retrieve
            
        Returns:
            Dict containing headlines and metadata from News API
        """
        params = {
            'country': country,
            'pageSize': min(page_size, 100),
            'page': page
        }
        
        # Add optional parameters
        if q:
            params['q'] = q
        if sources:
            params['sources'] = sources
        if category:
            params['category'] = category
        
        return await self._make_request('top-headlines', params)
    
    async def get_sources(
        self,
        category: Optional[str] = None,
        language: str = 'en',
        country: str = 'us'
    ) -> Dict[str, Any]:
        """
        Get available news sources.
        
        Args:
            category: Category to filter sources
            language: Language to filter sources
            country: Country to filter sources
            
        Returns:
            Dict containing available sources
        """
        params = {
            'language': language,
            'country': country
        }
        
        if category:
            params['category'] = category
        
        return await self._make_request('sources', params)
    
    async def test_connection(self) -> bool:
        """
        Test connection to News API and validate API key.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info("Testing News API connection...")
            
            # Simple request to test API key and connectivity
            response = await self.get_top_headlines(page_size=1)
            
            if response.get('status') == 'ok':
                total_results = response.get('totalResults', 0)
                logger.info(f"News API connection test successful - {total_results} articles available")
                return True
            else:
                logger.error("Unexpected response format from News API")
                return False
                
        except Exception as e:
            logger.error(f"News API connection test failed: {e}")
            return False
    
    async def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get current usage statistics (Note: News API doesn't provide usage endpoint in free tier).
        This is estimated based on our request tracking.
        
        Returns:
            Dictionary with usage statistics
        """
        return {
            'requests_made_today': self.requests_made,  # Approximate
            'last_request_time': self.last_request_time,
            'rate_limit_per_hour': self.MAX_REQUESTS_PER_HOUR,
            'daily_limit': self.MAX_REQUESTS_PER_DAY,
            'estimated_remaining_today': max(0, self.MAX_REQUESTS_PER_DAY - self.requests_made)
        }
    
    async def close(self) -> None:
        """Close HTTP session and cleanup resources"""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.debug("HTTP session closed")
    
    def get_request_stats(self) -> Dict[str, Any]:
        """
        Get statistics about API usage for monitoring.
        
        Returns:
            Dictionary with request statistics
        """
        return {
            'requests_made': self.requests_made,
            'last_request_time': self.last_request_time,
            'rate_limit_per_hour': self.MAX_REQUESTS_PER_HOUR,
            'daily_limit': self.MAX_REQUESTS_PER_DAY,
            'session_active': self.session is not None and not self.session.closed
        }


# Convenience function for quick testing
async def test_news_client():
    """Quick test function to verify News API client works"""
    async with NewsClient() as client:
        # Test connection
        success = await client.test_connection()
        print(f"Connection test: {'✅ SUCCESS' if success else '❌ FAILED'}")
        
        if success:
            try:
                # Test search functionality
                results = await client.search_everything(
                    q="federal reserve",
                    page_size=5,
                    from_date=(date.today() - timedelta(days=7)).strftime('%Y-%m-%d')
                )
                
                articles = results.get('articles', [])
                print(f"Search test: ✅ Found {len(articles)} articles about 'federal reserve'")
                
                if articles:
                    latest = articles[0]
                    print(f"   Latest: {latest.get('title', 'No title')[:100]}...")
                    print(f"   Source: {latest.get('source', {}).get('name', 'Unknown')}")
                    print(f"   Published: {latest.get('publishedAt', 'Unknown')}")
                
                # Test sources
                sources = await client.get_sources(category='business')
                source_list = sources.get('sources', [])
                print(f"Sources test: ✅ Found {len(source_list)} business news sources")
                
            except Exception as e:
                print(f"Feature test: ❌ {e}")
        
        # Show stats
        stats = client.get_request_stats()
        print(f"Requests made: {stats['requests_made']}")
        print(f"Rate limit: {stats['rate_limit_per_hour']}/hour")
        
        return success


if __name__ == "__main__":
    """Test the client when run directly"""
    asyncio.run(test_news_client())
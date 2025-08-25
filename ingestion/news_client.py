"""
NewsData.io API Client - Enterprise Foundation Layer

This module provides a robust foundation for interacting with the NewsData.io API.
Handles networking, rate limiting, error handling, and retries before any business logic.

Key Features:
- Environment-based configuration
- Rate limiting (varies by NewsData.io plan)
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


class NewsDataAPIError(Exception):
    """Custom exception for NewsData.io API specific errors"""
    pass


class NewsDataRateLimitError(NewsDataAPIError):
    """Raised when NewsData.io API rate limits are exceeded"""
    pass


class NewsClient:
    """
    Enterprise-grade NewsData.io API client with comprehensive error handling and rate limiting.
    
    This class handles all networking concerns and provides a foundation for business logic.
    """
    
    # NewsData.io API Configuration
    BASE_URL = "https://newsdata.io/api/1"
    DEFAULT_TIMEOUT = 30  # seconds
    MAX_REQUESTS_PER_DAY = 1000  # Conservative default, varies by plan
    MAX_REQUESTS_PER_HOUR = 40   # Conservative rate limit
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize News API client with configuration and session management.
        
        Args:
            api_key: News API key. If None, loads from NEWS_API_KEY environment variable.
            
        Raises:
            NewsAPIError: If API key is not provided or found in environment
        """
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv('NEWSDATA_API_KEY')
        
        if not self.api_key:
            raise NewsDataAPIError(
                "NewsData.io API key is required. Provide it as parameter or set NEWSDATA_API_KEY environment variable."
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
        
        logger.info("NewsData.io API client initialized successfully")
    
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
                    'Accept': 'application/json'
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
        Make HTTP request to NewsData.io API with rate limiting, retries, and error handling.
        
        Args:
            endpoint: NewsData.io API endpoint (e.g., 'latest', 'archive')
            params: Query parameters for the request
            
        Returns:
            Dict containing the JSON response from NewsData.io API
            
        Raises:
            NewsDataAPIError: For API-specific errors
            NewsDataRateLimitError: When rate limits are exceeded
        """
        await self._ensure_session()
        
        # Apply rate limiting
        async with self.rate_limiter:
            url = f"{self.BASE_URL}/{endpoint}"
            
            # Add API key to params for NewsData.io
            params['apikey'] = self.api_key
            
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
                            
                            # Check NewsData.io specific status
                            if data.get('status') == 'success':
                                logger.debug(f"Successful request to {endpoint}")
                                return data
                            else:
                                error_code = data.get('code', 'unknown')
                                error_message = data.get('message', 'Unknown error')
                                logger.error(f"NewsData.io API error {error_code}: {error_message}")
                                
                                if 'rate limit' in error_message.lower() or error_code == 'rateLimited':
                                    raise NewsDataRateLimitError(f"Rate limit exceeded: {error_message}")
                                else:
                                    raise NewsDataAPIError(f"NewsData.io API error ({error_code}): {error_message}")
                                    
                        except ValueError as e:
                            logger.error(f"Invalid JSON response from {endpoint}: {response_text[:200]}")
                            raise NewsDataAPIError(f"Invalid JSON response: {e}")
                    
                    elif response.status == 429:
                        logger.warning("NewsData.io API rate limit exceeded")
                        raise NewsDataRateLimitError("Rate limit exceeded. Please wait before making more requests.")
                    
                    elif response.status == 401:
                        logger.error("NewsData.io API authentication failed - check API key")
                        raise NewsDataAPIError("Authentication failed. Check your NewsData.io API key.")
                    
                    elif response.status == 400:
                        logger.error(f"Bad request to {endpoint}: {response_text}")
                        raise NewsDataAPIError(f"Bad request: {response_text}")
                    
                    elif response.status == 426:
                        logger.error("NewsData.io API upgrade required")
                        raise NewsDataAPIError("API upgrade required. You may have exceeded your plan limits.")
                    
                    elif response.status == 500:
                        logger.error("NewsData.io API server error")
                        raise NewsDataAPIError("NewsData.io API server error. Please try again later.")
                    
                    else:
                        logger.error(f"HTTP {response.status} error: {response_text}")
                        raise NewsDataAPIError(f"HTTP {response.status}: {response_text}")
            
            except aiohttp.ClientError as e:
                logger.error(f"Network error during request to {endpoint}: {e}")
                raise
            
            except asyncio.TimeoutError:
                logger.error(f"Timeout during request to {endpoint}")
                raise
    
    async def search_latest(
        self,
        q: Optional[str] = None,
        country: Optional[str] = None,
        category: Optional[str] = None,
        language: str = 'en',
        domain: Optional[str] = None,
        timeframe: Optional[str] = None,
        size: int = 10
    ) -> Dict[str, Any]:
        """
        Search latest news from the past 48 hours using NewsData.io.
        
        Args:
            q: Keywords or phrases to search for
            country: Country code (e.g., 'us', 'gb', 'ca')
            category: Category filter (business, politics, sports, etc.)
            language: Language code (default 'en')
            domain: Specific domain to search
            timeframe: Time frame for search (e.g., '24h', '48h')
            size: Number of results per page (max 50 for free tier)
            
        Returns:
            Dict containing articles and metadata from NewsData.io API
            
        Raises:
            NewsDataAPIError: If search fails or parameters are invalid
        """
        params = {
            'language': language,
            'size': min(size, 50)  # NewsData.io free tier max
        }
        
        # Add optional parameters
        if q:
            params['q'] = q
        if country:
            params['country'] = country
        if category:
            params['category'] = category
        if domain:
            params['domain'] = domain
        if timeframe:
            params['timeframe'] = timeframe
        
        return await self._make_request('latest', params)
    
    async def search_archive(
        self,
        q: Optional[str] = None,
        country: Optional[str] = None,
        category: Optional[str] = None,
        language: str = 'en',
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
        size: int = 10
    ) -> Dict[str, Any]:
        """
        Search historical news articles using NewsData.io archive endpoint.
        
        Args:
            q: Keywords to search for
            country: Country code (e.g., 'us', 'gb', 'ca')
            category: Category filter
            language: Language code (default 'en')
            from_date: Start date (YYYY-MM-DD format)
            to_date: End date (YYYY-MM-DD format)
            size: Number of results per page (max 50 for free tier)
            
        Returns:
            Dict containing articles and metadata from NewsData.io API
        """
        params = {
            'language': language,
            'size': min(size, 50)
        }
        
        # Add optional parameters
        if q:
            params['q'] = q
        if country:
            params['country'] = country
        if category:
            params['category'] = category
        if from_date:
            params['from_date'] = from_date
        if to_date:
            params['to_date'] = to_date
        
        return await self._make_request('archive', params)
    
    async def get_sources(
        self,
        category: Optional[str] = None,
        language: str = 'en',
        country: str = 'us'
    ) -> Dict[str, Any]:
        """
        Get available news sources from NewsData.io.
        
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
        Test connection to NewsData.io API and validate API key.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info("Testing NewsData.io API connection...")
            
            # Simple request to test API key and connectivity
            response = await self.search_latest(size=1)
            
            if response.get('status') == 'success':
                total_results = response.get('totalResults', 0)
                logger.info(f"NewsData.io API connection test successful - {total_results} articles available")
                return True
            else:
                logger.error("Unexpected response format from NewsData.io API")
                return False
                
        except Exception as e:
            logger.error(f"NewsData.io API connection test failed: {e}")
            return False
    
    async def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get current usage statistics (Note: NewsData.io doesn't provide usage endpoint in free tier).
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
    """Quick test function to verify NewsData.io client works"""
    async with NewsClient() as client:
        # Test connection
        success = await client.test_connection()
        print(f"Connection test: {'✅ SUCCESS' if success else '❌ FAILED'}")
        
        if success:
            try:
                # Test search functionality
                results = await client.search_latest(
                    q="federal reserve",
                    size=5
                )
                
                articles = results.get('results', [])
                print(f"Search test: ✅ Found {len(articles)} articles about 'federal reserve'")
                
                if articles:
                    latest = articles[0]
                    print(f"   Latest: {latest.get('title', 'No title')[:100]}...")
                    print(f"   Source: {latest.get('source_id', 'Unknown')}")
                    print(f"   Published: {latest.get('pubDate', 'Unknown')}")
                
                # Test sources
                sources = await client.get_sources(category='business')
                source_list = sources.get('results', [])
                print(f"Sources test: ✅ Found {len(source_list)} business news sources")
                
            except Exception as e:
                print(f"Feature test: ❌ {e}")
        
        # Show stats
        stats = client.get_request_stats()
        print(f"Requests made: {stats['requests_made']}")
        print(f"Rate limit: {stats['rate_limit_per_hour']}/hour")
        
        return success


if __name__ == "__main__":
    """Test the NewsData.io client when run directly"""
    asyncio.run(test_news_client())
"""
FRED API Client - Enterprise Foundation Layer

This module provides a robust foundation for interacting with the Federal Reserve Economic Data (FRED) API.
Handles networking, rate limiting, error handling, and retries before any business logic.

Key Features:
- Environment-based configuration
- Rate limiting (120 requests/minute FRED limit)
- Comprehensive error handling and logging
- Automatic retry logic with exponential backoff
- Session management for connection pooling
"""

import os
import time
import logging
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime, timezone

import aiohttp
import asyncio_throttle
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


class FredAPIError(Exception):
    """Custom exception for FRED API specific errors"""
    pass


class FredRateLimitError(FredAPIError):
    """Raised when FRED API rate limits are exceeded"""
    pass


class FredClient:
    """
    Enterprise-grade FRED API client with comprehensive error handling and rate limiting.
    
    This class handles all networking concerns and provides a foundation for business logic.
    """
    
    # FRED API Configuration
    BASE_URL = "https://api.stlouisfed.org/fred"
    DEFAULT_TIMEOUT = 30  # seconds
    MAX_REQUESTS_PER_MINUTE = 120  # FRED's rate limit
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FRED client with configuration and session management.
        
        Args:
            api_key: FRED API key. If None, loads from FRED_API_KEY environment variable.
            
        Raises:
            FredAPIError: If API key is not provided or found in environment
        """
        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv('FRED_API_KEY')
        
        if not self.api_key:
            raise FredAPIError(
                "FRED API key is required. Provide it as parameter or set FRED_API_KEY environment variable."
            )
        
        # Rate limiting setup - FRED allows 120 requests per minute
        self.rate_limiter = asyncio_throttle.Throttler(
            rate_limit=self.MAX_REQUESTS_PER_MINUTE,
            period=60  # 60 seconds
        )
        
        # Session will be created when needed (async context)
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Request tracking for monitoring
        self.requests_made = 0
        self.last_request_time: Optional[datetime] = None
        
        logger.info("FRED client initialized successfully")
    
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
        Make HTTP request to FRED API with rate limiting, retries, and error handling.
        
        Args:
            endpoint: FRED API endpoint (e.g., 'series', 'series/observations')
            params: Query parameters for the request
            
        Returns:
            Dict containing the JSON response from FRED API
            
        Raises:
            FredAPIError: For API-specific errors
            FredRateLimitError: When rate limits are exceeded
        """
        await self._ensure_session()
        
        # Apply rate limiting
        async with self.rate_limiter:
            # Add API key to parameters
            request_params = {
                'api_key': self.api_key,
                'file_type': 'json',
                **params
            }
            
            url = f"{self.BASE_URL}/{endpoint}"
            
            try:
                logger.debug(f"Making request to {endpoint} with params: {list(request_params.keys())}")
                
                async with self.session.get(url, params=request_params) as response:
                    # Update request tracking
                    self.requests_made += 1
                    self.last_request_time = datetime.now(timezone.utc)
                    
                    # Handle different HTTP status codes
                    if response.status == 200:
                        data = await response.json()
                        logger.debug(f"Successful request to {endpoint}")
                        return data
                    
                    elif response.status == 429:
                        logger.warning("FRED API rate limit exceeded")
                        raise FredRateLimitError("Rate limit exceeded. Please wait before making more requests.")
                    
                    elif response.status == 400:
                        error_text = await response.text()
                        logger.error(f"Bad request to {endpoint}: {error_text}")
                        raise FredAPIError(f"Bad request: {error_text}")
                    
                    elif response.status == 404:
                        logger.error(f"FRED endpoint not found: {endpoint}")
                        raise FredAPIError(f"Endpoint not found: {endpoint}")
                    
                    else:
                        error_text = await response.text()
                        logger.error(f"HTTP {response.status} error: {error_text}")
                        raise FredAPIError(f"HTTP {response.status}: {error_text}")
            
            except aiohttp.ClientError as e:
                logger.error(f"Network error during request to {endpoint}: {e}")
                raise
            
            except asyncio.TimeoutError:
                logger.error(f"Timeout during request to {endpoint}")
                raise
    
    async def test_connection(self) -> bool:
        """
        Test connection to FRED API and validate API key.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info("Testing FRED API connection...")
            
            # Simple request to test API key and connectivity
            response = await self._make_request('category', {'category_id': 0})
            
            if 'categories' in response:
                logger.info("FRED API connection test successful")
                return True
            else:
                logger.error("Unexpected response format from FRED API")
                return False
                
        except Exception as e:
            logger.error(f"FRED API connection test failed: {e}")
            return False
    
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
            'rate_limit_per_minute': self.MAX_REQUESTS_PER_MINUTE,
            'session_active': self.session is not None and not self.session.closed
        }


# Convenience function for quick testing
async def test_fred_client():
    """Quick test function to verify FRED client works"""
    async with FredClient() as client:
        success = await client.test_connection()
        stats = client.get_request_stats()
        
        print(f"Connection test: {'✅ SUCCESS' if success else '❌ FAILED'}")
        print(f"Requests made: {stats['requests_made']}")
        print(f"Last request: {stats['last_request_time']}")
        
        return success


if __name__ == "__main__":
    """Test the client when run directly"""
    asyncio.run(test_fred_client())
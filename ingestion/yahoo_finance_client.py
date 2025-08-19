"""
Yahoo Finance Client - yfinance Library Integration

This module provides a robust foundation for Yahoo Finance data using the yfinance library.
Much more reliable than direct API calls and handles Yahoo's authentication automatically.

Key Features:
- Built on yfinance library (no auth issues)
- Rate limiting to be respectful
- Comprehensive error handling and logging
- Support for historical OHLCV data and company info
- Async wrapper around synchronous yfinance calls

Data Sources:
- Historical OHLCV data with adjustments
- Company information and metadata
- Dividend and split data
- Multiple symbols support
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timezone, date, timedelta
import pandas as pd

import yfinance as yf
import asyncio_throttle
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logger = logging.getLogger(__name__)


class YahooFinanceError(Exception):
    """Custom exception for Yahoo Finance specific errors"""
    pass


class YahooFinanceClient:
    """
    Enterprise-grade Yahoo Finance client using yfinance library.
    
    Provides async interface around yfinance with rate limiting and error handling.
    """
    
    MAX_REQUESTS_PER_MINUTE = 30  # Conservative rate limit
    
    def __init__(self):
        """Initialize Yahoo Finance client with rate limiting."""
        
        # Rate limiting setup
        self.rate_limiter = asyncio_throttle.Throttler(
            rate_limit=self.MAX_REQUESTS_PER_MINUTE,
            period=60  # 60 seconds
        )
        
        # Request tracking for monitoring
        self.requests_made = 0
        self.last_request_time: Optional[datetime] = None
        
        logger.info("Yahoo Finance client (yfinance) initialized successfully")
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def _run_in_executor(self, func, *args, **kwargs):
        """
        Run synchronous yfinance functions in thread pool with rate limiting.
        
        Args:
            func: Function to run
            *args, **kwargs: Arguments for the function
            
        Returns:
            Function result
        """
        async with self.rate_limiter:
            loop = asyncio.get_event_loop()
            
            # Update tracking
            self.requests_made += 1
            self.last_request_time = datetime.now(timezone.utc)
            
            try:
                result = await loop.run_in_executor(None, func, *args, **kwargs)
                logger.debug("Successful yfinance operation")
                return result
            except Exception as e:
                logger.error(f"yfinance operation failed: {e}")
                raise YahooFinanceError(f"Yahoo Finance operation failed: {e}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type(YahooFinanceError)
    )
    async def get_historical_data(
        self, 
        symbol: str, 
        period: str = "1y",
        interval: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        auto_adjust: bool = True,
        prepost: bool = False,
        actions: bool = True
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'SPY')
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            start_date: Start date in YYYY-MM-DD format (optional, overrides period)
            end_date: End date in YYYY-MM-DD format (optional)
            auto_adjust: Whether to auto-adjust prices for splits and dividends
            prepost: Include pre and post market data
            actions: Include dividend and stock split events
            
        Returns:
            Pandas DataFrame with historical data
            
        Raises:
            YahooFinanceError: If symbol doesn't exist or data retrieval fails
        """
        try:
            logger.info(f"Fetching historical data for {symbol}")
            
            def _fetch_data():
                ticker = yf.Ticker(symbol.upper())
                
                if start_date or end_date:
                    # Use date range
                    return ticker.history(
                        start=start_date,
                        end=end_date,
                        interval=interval,
                        auto_adjust=auto_adjust,
                        prepost=prepost,
                        actions=actions
                    )
                else:
                    # Use period
                    return ticker.history(
                        period=period,
                        interval=interval,
                        auto_adjust=auto_adjust,
                        prepost=prepost,
                        actions=actions
                    )
            
            data = await self._run_in_executor(_fetch_data)
            
            if data.empty:
                raise YahooFinanceError(f"No data returned for symbol {symbol}")
            
            logger.info(f"Successfully fetched {len(data)} data points for {symbol}")
            return data
            
        except YahooFinanceError:
            raise
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            raise YahooFinanceError(f"Failed to fetch historical data for {symbol}: {e}")
    
    async def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get detailed information about a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'SPY')
            
        Returns:
            Dictionary with symbol information
            
        Raises:
            YahooFinanceError: If symbol doesn't exist or info retrieval fails
        """
        try:
            logger.info(f"Fetching symbol info for {symbol}")
            
            def _fetch_info():
                ticker = yf.Ticker(symbol.upper())
                return ticker.info
            
            info = await self._run_in_executor(_fetch_info)
            
            if not info or 'symbol' not in info:
                raise YahooFinanceError(f"No information found for symbol {symbol}")
            
            logger.info(f"Successfully fetched info for {symbol}")
            return info
            
        except YahooFinanceError:
            raise
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {e}")
            raise YahooFinanceError(f"Failed to fetch info for {symbol}: {e}")
    
    async def get_multiple_symbols_data(
        self, 
        symbols: List[str], 
        period: str = "1y",
        interval: str = "1d",
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple symbols efficiently.
        
        Args:
            symbols: List of stock symbols
            period: Time period for all symbols
            interval: Data interval for all symbols
            **kwargs: Additional arguments passed to get_historical_data
            
        Returns:
            Dictionary mapping symbol -> DataFrame
        """
        logger.info(f"Fetching data for {len(symbols)} symbols")
        
        results = {}
        for symbol in symbols:
            try:
                data = await self.get_historical_data(
                    symbol, 
                    period=period, 
                    interval=interval, 
                    **kwargs
                )
                results[symbol] = data
                logger.info(f"✅ {symbol}: {len(data)} data points")
            except Exception as e:
                logger.error(f"❌ {symbol}: {e}")
                results[symbol] = pd.DataFrame()  # Empty DataFrame for failed symbols
        
        return results
    
    async def test_connection(self) -> bool:
        """
        Test connection to Yahoo Finance and validate functionality.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info("Testing Yahoo Finance connection...")
            
            # Test with SPY (very reliable symbol)
            data = await self.get_historical_data('SPY', period='5d')
            
            if not data.empty and len(data) > 0:
                logger.info("Yahoo Finance connection test successful")
                return True
            else:
                logger.error("No data returned from Yahoo Finance")
                return False
                
        except Exception as e:
            logger.error(f"Yahoo Finance connection test failed: {e}")
            return False
    
    async def validate_symbols(self, symbols: List[str]) -> Dict[str, bool]:
        """
        Validate that symbols exist and return data.
        
        Args:
            symbols: List of symbols to validate
            
        Returns:
            Dictionary mapping symbol -> is_valid
        """
        logger.info(f"Validating {len(symbols)} symbols")
        
        results = {}
        for symbol in symbols:
            try:
                # Try to get minimal data (1 day)
                data = await self.get_historical_data(symbol, period='1d')
                results[symbol] = not data.empty
                
                if results[symbol]:
                    logger.debug(f"✅ {symbol}: Valid")
                else:
                    logger.warning(f"⚠️ {symbol}: No data returned")
                    
            except Exception as e:
                logger.warning(f"❌ {symbol}: {e}")
                results[symbol] = False
        
        valid_count = sum(results.values())
        logger.info(f"Symbol validation complete: {valid_count}/{len(symbols)} valid")
        
        return results
    
    async def close(self) -> None:
        """Clean up resources (yfinance doesn't require explicit cleanup)"""
        logger.debug("Yahoo Finance client cleanup complete")
    
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
            'library': 'yfinance'
        }


# Convenience function for quick testing
async def test_yahoo_client():
    """Quick test function to verify Yahoo Finance client works"""
    async with YahooFinanceClient() as client:
        # Test connection
        success = await client.test_connection()
        print(f"Connection test: {'✅ SUCCESS' if success else '❌ FAILED'}")
        
        if success:
            try:
                # Test historical data
                data = await client.get_historical_data('SPY', period='5d')
                print(f"Historical data: ✅ {len(data)} data points for SPY")
                print(f"   Latest close: ${data['Close'].iloc[-1]:.2f}")
                print(f"   Columns: {list(data.columns)}")
                
                # Test symbol info
                info = await client.get_symbol_info('SPY')
                print(f"Symbol info: ✅ {info.get('longName', 'SPY')}")
                print(f"   Sector: {info.get('sector', 'N/A')}")
                print(f"   Market Cap: ${info.get('marketCap', 0):,}")
                
                # Test multiple symbols
                multi_data = await client.get_multiple_symbols_data(['SPY', 'QQQ'], period='2d')
                valid_symbols = [sym for sym, df in multi_data.items() if not df.empty]
                print(f"Multiple symbols: ✅ {len(valid_symbols)}/2 symbols returned data")
                
            except Exception as e:
                print(f"Data retrieval test: ❌ {e}")
        
        # Show stats
        stats = client.get_request_stats()
        print(f"Requests made: {stats['requests_made']}")
        print(f"Library: {stats['library']}")
        
        return success


if __name__ == "__main__":
    """Test the client when run directly"""
    asyncio.run(test_yahoo_client())
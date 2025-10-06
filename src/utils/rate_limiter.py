"""
Rate limiter utility for API calls to prevent server overload
"""

import time
import logging
from typing import Optional, Dict, Any
from threading import Lock
from collections import deque
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter for API calls
    """
    
    def __init__(self, max_requests_per_minute: int = 30, request_delay: float = 2.0):
        """
        Initialize rate limiter
        
        Args:
            max_requests_per_minute: Maximum requests allowed per minute
            request_delay: Minimum delay between requests in seconds
        """
        self.max_requests_per_minute = max_requests_per_minute
        self.request_delay = request_delay
        self.request_times = deque()
        self.last_request_time = 0.0
        self.lock = Lock()
        
        logger.info(f"Rate limiter initialized: {max_requests_per_minute} req/min, {request_delay}s delay")
    
    def wait_if_needed(self) -> None:
        """
        Wait if necessary to respect rate limits
        """
        with self.lock:
            current_time = time.time()
            
            # Remove requests older than 1 minute
            cutoff_time = current_time - 60
            while self.request_times and self.request_times[0] < cutoff_time:
                self.request_times.popleft()
            
            # Check if we've hit the per-minute limit
            if len(self.request_times) >= self.max_requests_per_minute:
                # Wait until the oldest request is more than 1 minute old
                wait_time = 60 - (current_time - self.request_times[0])
                if wait_time > 0:
                    logger.info(f"Rate limit reached, waiting {wait_time:.1f} seconds")
                    time.sleep(wait_time)
                    current_time = time.time()
            
            # Check minimum delay between requests
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.request_delay:
                wait_time = self.request_delay - time_since_last
                logger.debug(f"Enforcing request delay: waiting {wait_time:.1f} seconds")
                time.sleep(wait_time)
                current_time = time.time()
            
            # Record this request
            self.request_times.append(current_time)
            self.last_request_time = current_time
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current rate limiter statistics
        
        Returns:
            Dict with rate limiter stats
        """
        with self.lock:
            current_time = time.time()
            
            # Clean old requests
            cutoff_time = current_time - 60
            while self.request_times and self.request_times[0] < cutoff_time:
                self.request_times.popleft()
            
            return {
                "requests_in_last_minute": len(self.request_times),
                "max_requests_per_minute": self.max_requests_per_minute,
                "request_delay": self.request_delay,
                "time_since_last_request": current_time - self.last_request_time if self.last_request_time else None
            }


class RetryHandler:
    """
    Exponential backoff retry handler for API calls
    """
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0):
        """
        Initialize retry handler
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff
            max_delay: Maximum delay between retries
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        
        logger.info(f"Retry handler initialized: {max_retries} retries, {base_delay}s base delay")
    
    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for given attempt using exponential backoff
        
        Args:
            attempt: Current attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        delay = self.base_delay * (2 ** attempt)
        return min(delay, self.max_delay)
    
    def should_retry(self, attempt: int, exception: Exception) -> bool:
        """
        Determine if we should retry based on attempt count and exception type
        
        Args:
            attempt: Current attempt number (0-based)
            exception: Exception that occurred
            
        Returns:
            True if should retry, False otherwise
        """
        if attempt >= self.max_retries:
            return False
        
        # Check for retryable exceptions
        exception_str = str(exception).lower()
        retryable_errors = [
            'rate limit',
            'too many requests',
            'timeout',
            'connection error',
            'server error',
            '429',
            '500',
            '502',
            '503',
            '504'
        ]
        
        return any(error in exception_str for error in retryable_errors)
    
    def wait_for_retry(self, attempt: int) -> None:
        """
        Wait for the appropriate delay before retry
        
        Args:
            attempt: Current attempt number (0-based)
        """
        delay = self.calculate_delay(attempt)
        logger.info(f"Retrying in {delay:.1f} seconds (attempt {attempt + 1}/{self.max_retries + 1})")
        time.sleep(delay)


# Global instances (will be configured from environment)
_rate_limiter: Optional[RateLimiter] = None
_retry_handler: Optional[RetryHandler] = None


def get_rate_limiter() -> RateLimiter:
    """Get the global rate limiter instance"""
    global _rate_limiter
    if _rate_limiter is None:
        # Default values - will be overridden by config
        _rate_limiter = RateLimiter(max_requests_per_minute=30, request_delay=2.0)
    return _rate_limiter


def get_retry_handler() -> RetryHandler:
    """Get the global retry handler instance"""
    global _retry_handler
    if _retry_handler is None:
        # Default values - will be overridden by config
        _retry_handler = RetryHandler(max_retries=3, base_delay=1.0, max_delay=60.0)
    return _retry_handler


def configure_rate_limiting(max_requests_per_minute: int = 30, 
                          request_delay: float = 2.0,
                          max_retries: int = 3,
                          retry_base_delay: float = 1.0,
                          retry_max_delay: float = 60.0) -> None:
    """
    Configure global rate limiting settings
    
    Args:
        max_requests_per_minute: Maximum requests per minute
        request_delay: Minimum delay between requests
        max_retries: Maximum retry attempts
        retry_base_delay: Base delay for exponential backoff
        retry_max_delay: Maximum retry delay
    """
    global _rate_limiter, _retry_handler
    
    _rate_limiter = RateLimiter(
        max_requests_per_minute=max_requests_per_minute,
        request_delay=request_delay
    )
    
    _retry_handler = RetryHandler(
        max_retries=max_retries,
        base_delay=retry_base_delay,
        max_delay=retry_max_delay
    )
    
    logger.info("Rate limiting configured successfully")


def with_rate_limiting(func):
    """
    Decorator to add rate limiting and retry logic to functions
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function with rate limiting
    """
    def wrapper(*args, **kwargs):
        rate_limiter = get_rate_limiter()
        retry_handler = get_retry_handler()
        
        for attempt in range(retry_handler.max_retries + 1):
            try:
                # Wait for rate limiting
                rate_limiter.wait_if_needed()
                
                # Execute the function
                return func(*args, **kwargs)
                
            except Exception as e:
                if retry_handler.should_retry(attempt, e):
                    logger.warning(f"API call failed (attempt {attempt + 1}): {str(e)}")
                    retry_handler.wait_for_retry(attempt)
                    continue
                else:
                    # Not retryable or max retries reached
                    logger.error(f"API call failed after {attempt + 1} attempts: {str(e)}")
                    raise
        
        # This should never be reached, but just in case
        raise Exception("Maximum retries exceeded")
    
    return wrapper
"""Retry logic for API calls with exponential backoff."""

import logging
import time
from typing import Any, Callable

from openai import APIConnectionError, APIError, APIStatusError, RateLimitError

logger = logging.getLogger(__name__)


class RetryHandler:
    """Handles retry logic for API calls with configurable backoff."""

    # Retryable status codes (transient errors that may succeed on retry)
    # 408: Request Timeout - Heroku/server timeout, should retry with streaming
    # 429: Rate limit - should retry with backoff
    # 502, 503, 504: Server errors - transient, should retry
    RETRYABLE_STATUS_CODES = {408, 429, 502, 503, 504}

    # Non-retryable status codes (permanent errors or CloudFront WAF blocks)
    NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404, 500}

    def __init__(self, max_retries: int = 3, base_delay: float = 5.0):
        """
        Initialize retry handler.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds for linear backoff (5s, 10s, 15s, ...)
        """
        self.max_retries = max_retries
        self.base_delay = base_delay

    def execute_with_retry(
        self, api_func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """
        Execute an API call with retry logic.

        Args:
            api_func: The API function to call
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            The result of the API call

        Raises:
            The last exception if all retries fail
        """
        last_exception: (
            APIStatusError | RateLimitError | APIConnectionError | APIError | None
        ) = None

        for attempt in range(self.max_retries):
            try:
                return api_func(*args, **kwargs)

            except APIStatusError as e:
                last_exception = e
                status_code = e.response.status_code if e.response else None

                # Check if error is retryable
                if status_code in self.NON_RETRYABLE_STATUS_CODES:
                    # Special handling for 403 (CloudFront WAF)
                    if status_code == 403:
                        self._log_403_error(e, attempt)
                    raise

                if status_code not in self.RETRYABLE_STATUS_CODES:
                    raise

                # Special handling for 408 timeout errors
                if status_code == 408:
                    self._log_408_error(e, attempt)

                # Calculate delay with linear backoff
                delay = self.base_delay * (attempt + 1)
                logger.warning(
                    f"API call failed with status {status_code} "
                    f"(attempt {attempt + 1}/{self.max_retries}). "
                    f"Retrying in {delay}s..."
                )

                if attempt < self.max_retries - 1:
                    time.sleep(delay)

            except (RateLimitError, APIConnectionError, APIError) as e:
                last_exception = e

                # Calculate delay with linear backoff
                delay = self.base_delay * (attempt + 1)
                logger.warning(
                    f"API call failed: {type(e).__name__} "
                    f"(attempt {attempt + 1}/{self.max_retries}). "
                    f"Retrying in {delay}s..."
                )

                if attempt < self.max_retries - 1:
                    time.sleep(delay)

        # All retries exhausted
        if last_exception:
            status_code = None
            if isinstance(last_exception, APIStatusError) and last_exception.response:
                status_code = last_exception.response.status_code

            error_msg = (
                f"API call failed after {self.max_retries} attempts. "
                f"Last error: {last_exception}"
            )

            # Add specific guidance for 408 timeout errors
            if status_code == 408:
                error_msg += (
                    " | RECOMMENDATION: This 408 timeout likely occurred because "
                    "the response took too long. Consider using streaming mode "
                    "(stream=True) or reducing response length. If tools are enabled, "
                    "try disabling them or using stream_final_response=True."
                )

            logger.error(error_msg)
            raise last_exception

    def _log_408_error(self, error: APIStatusError, attempt: int) -> None:
        """
        Log detailed information about 408 timeout errors.

        Args:
            error: The API status error
            attempt: Current attempt number
        """
        error_body = str(error.body) if error.body else "No error body"

        logger.warning(
            f"408 Request Timeout detected (attempt {attempt + 1}/{self.max_retries}). "
            f"The server took too long to respond. This often happens with: "
            f"1) Long responses without streaming, "
            f"2) Complex tool calls, "
            f"3) High server load. "
            f"Error details: {error_body}"
        )

        # Check for streaming recommendation in error
        if "streaming" in error_body.lower() or "long" in error_body.lower():
            logger.warning(
                "Server suggests using streaming mode. "
                "Enable streaming with stream=True or use prompt() which streams by default."
            )

    def _log_403_error(self, error: APIStatusError, attempt: int) -> None:
        """
        Log detailed information about 403 errors for CloudFront WAF debugging.

        Args:
            error: The API status error
            attempt: Current attempt number
        """
        error_body = str(error.body) if error.body else "No error body"

        logger.error(
            f"403 Forbidden error detected (attempt {attempt + 1}). "
            f"This may be caused by CloudFront WAF blocking the request. "
            f"Error details: {error_body}"
        )

        # Check for common CloudFront WAF trigger patterns
        if "..." in error_body or "path traversal" in error_body.lower():
            logger.error(
                "CloudFront WAF likely triggered by '...' pattern in request. "
                "Enable sanitize_tool_responses to prevent this."
            )

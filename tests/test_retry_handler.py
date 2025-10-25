"""Tests for retry_handler module."""

import time
from unittest.mock import Mock, patch

import pytest
from openai import APIConnectionError, APIStatusError, RateLimitError

from floship_llm.retry_handler import RetryHandler


class TestRetryHandler:
    """Test cases for RetryHandler class."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        handler = RetryHandler()
        assert handler.max_retries == 3
        assert handler.base_delay == 5.0

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        handler = RetryHandler(max_retries=5, base_delay=2.0)
        assert handler.max_retries == 5
        assert handler.base_delay == 2.0

    def test_execute_success_first_try(self):
        """Test successful execution on first try."""
        handler = RetryHandler()
        mock_func = Mock(return_value="success")

        result = handler.execute_with_retry(mock_func, "arg1", key="value")

        assert result == "success"
        assert mock_func.call_count == 1
        mock_func.assert_called_with("arg1", key="value")

    def test_execute_retry_on_rate_limit(self):
        """Test retry on rate limit error."""
        handler = RetryHandler()
        mock_func = Mock(
            side_effect=[
                RateLimitError("Rate limit", response=Mock(status_code=429), body={}),
                "success",
            ]
        )

        with patch("time.sleep"):
            result = handler.execute_with_retry(mock_func)

        assert result == "success"
        assert mock_func.call_count == 2

    def test_execute_retry_on_500_error(self):
        """Test retry on 500 server error."""
        handler = RetryHandler()
        mock_response = Mock()
        mock_response.status_code = 500

        mock_func = Mock(
            side_effect=[
                APIStatusError("Server error", response=mock_response, body={}),
                "success",
            ]
        )

        with patch("time.sleep"):
            result = handler.execute_with_retry(mock_func)

        assert result == "success"
        assert mock_func.call_count == 2

    def test_execute_no_retry_on_403(self):
        """Test no retry on 403 Forbidden."""
        handler = RetryHandler()
        mock_response = Mock()
        mock_response.status_code = 403

        error = APIStatusError(
            "Forbidden", response=mock_response, body={"error": "forbidden"}
        )
        mock_func = Mock(side_effect=error)

        with pytest.raises(APIStatusError):
            handler.execute_with_retry(mock_func)

        # Should only try once
        assert mock_func.call_count == 1

    def test_execute_no_retry_on_400(self):
        """Test no retry on 400 Bad Request."""
        handler = RetryHandler()
        mock_response = Mock()
        mock_response.status_code = 400

        error = APIStatusError("Bad request", response=mock_response, body={})
        mock_func = Mock(side_effect=error)

        with pytest.raises(APIStatusError):
            handler.execute_with_retry(mock_func)

        assert mock_func.call_count == 1

    def test_execute_max_retries_exceeded(self):
        """Test that max retries are respected."""
        handler = RetryHandler(max_retries=3)
        mock_response = Mock()
        mock_response.status_code = 500

        error = APIStatusError("Server error", response=mock_response, body={})
        mock_func = Mock(side_effect=error)

        with patch("time.sleep"):
            with pytest.raises(APIStatusError):
                handler.execute_with_retry(mock_func)

        # Should try 3 times
        assert mock_func.call_count == 3

    def test_execute_linear_backoff(self):
        """Test linear backoff delays."""
        handler = RetryHandler(max_retries=3, base_delay=2.0)
        mock_response = Mock()
        mock_response.status_code = 500

        error = APIStatusError("Server error", response=mock_response, body={})
        mock_func = Mock(side_effect=error)

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(APIStatusError):
                handler.execute_with_retry(mock_func)

        # Check that sleep was called with correct delays: 2s, 4s
        assert mock_sleep.call_count == 2
        assert mock_sleep.call_args_list[0][0][0] == 2.0  # First retry: base_delay * 1
        assert mock_sleep.call_args_list[1][0][0] == 4.0  # Second retry: base_delay * 2

    def test_execute_retry_on_connection_error(self):
        """Test retry on connection error."""
        handler = RetryHandler()

        # Create a mock request object
        mock_request = Mock()
        mock_request.url = "http://test.com"

        mock_func = Mock(
            side_effect=[
                APIConnectionError(message="Connection failed", request=mock_request),
                "success",
            ]
        )

        with patch("time.sleep"):
            result = handler.execute_with_retry(mock_func)

        assert result == "success"
        assert mock_func.call_count == 2

    def test_execute_no_retry_on_unexpected_error(self):
        """Test no retry on unexpected errors."""
        handler = RetryHandler()
        mock_func = Mock(side_effect=ValueError("Unexpected error"))

        with pytest.raises(ValueError):
            handler.execute_with_retry(mock_func)

        # Should only try once
        assert mock_func.call_count == 1

    def test_log_403_cloudfront_detection(self):
        """Test CloudFront WAF detection in 403 errors."""
        handler = RetryHandler()
        mock_response = Mock()
        mock_response.status_code = 403

        error = APIStatusError(
            "Forbidden",
            response=mock_response,
            body={"error": "CloudFront request blocked"},
        )
        mock_func = Mock(side_effect=error)

        with patch("floship_llm.retry_handler.logger") as mock_logger:
            with pytest.raises(APIStatusError):
                handler.execute_with_retry(mock_func)

            # Check that CloudFront-specific logging occurred
            assert any(
                "CloudFront" in str(call) for call in mock_logger.error.call_args_list
            )

    def test_retryable_status_codes(self):
        """Test all retryable status codes."""
        retryable_codes = [429, 500, 502, 503, 504]

        for status_code in retryable_codes:
            handler = RetryHandler()
            mock_response = Mock()
            mock_response.status_code = status_code

            mock_func = Mock(
                side_effect=[
                    APIStatusError(
                        f"Error {status_code}", response=mock_response, body={}
                    ),
                    "success",
                ]
            )

            with patch("time.sleep"):
                result = handler.execute_with_retry(mock_func)

            assert result == "success"
            assert mock_func.call_count == 2

    def test_non_retryable_status_codes(self):
        """Test all non-retryable status codes."""
        non_retryable_codes = [400, 401, 403, 404]

        for status_code in non_retryable_codes:
            handler = RetryHandler()
            mock_response = Mock()
            mock_response.status_code = status_code

            error = APIStatusError(
                f"Error {status_code}", response=mock_response, body={}
            )
            mock_func = Mock(side_effect=error)

            with pytest.raises(APIStatusError):
                handler.execute_with_retry(mock_func)

            # Should only try once
            assert mock_func.call_count == 1

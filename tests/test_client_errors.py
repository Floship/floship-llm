"""Tests for client error handling."""

from unittest.mock import Mock, patch

import pytest

from floship_llm.client import LLM, TruncatedResponseError


class TestClientErrors:
    """Test cases for client error handling."""

    @pytest.fixture
    def llm(self):
        env_vars = {
            "INFERENCE_URL": "https://test-api.example.com",
            "INFERENCE_MODEL_ID": "test-model",
            "INFERENCE_KEY": "test-key",
        }
        with patch.dict("os.environ", env_vars), patch("floship_llm.client.OpenAI"):
            llm = LLM()
            return llm

    def test_truncated_response_retry(self, llm):
        """Test that TruncatedResponseError triggers a retry with increased tokens."""
        # Mock _process_streaming_response to raise TruncatedResponseError on first call
        # and succeed on second call

        # Note: prompt() calls execute_with_retry, which calls streaming_request, which returns a string.
        # Then prompt() calls _process_streaming_response(string).

        # We mock execute_with_retry to return a dummy string
        llm.retry_handler.execute_with_retry = Mock(return_value="dummy response")

        # Mock _process_streaming_response to raise error then return success
        llm._process_streaming_response = Mock(
            side_effect=[
                TruncatedResponseError("Truncated", raw_response="partial"),
                "Success response",
            ]
        )

        # Initial max_completion_tokens
        llm.max_completion_tokens = 1000

        with patch("floship_llm.client.logger") as mock_logger:
            response = llm.prompt("test prompt")

            assert response == "Success response"
            assert llm._process_streaming_response.call_count == 2

            # Verify that we logged the retry with increased tokens
            # Expected log: "Truncated response detected ... Retrying with max_completion_tokens: 1000 -> 2000"

            # Find the warning call
            warning_calls = list(mock_logger.warning.call_args_list)
            found_retry_log = False
            for call in warning_calls:
                args, _ = call
                if "Retrying with max_completion_tokens: 1000 -> 2000" in args[0]:
                    found_retry_log = True
                    break

            assert found_retry_log, (
                "Did not find expected log message about increasing max_completion_tokens"
            )

    def test_cloudfront_403_retry(self, llm):
        """Test that CloudFront 403 errors trigger retry with sanitization."""
        # Mock execute_with_retry to raise 403 error then succeed

        # Create a mock 403 error
        error_403 = Exception("403 Forbidden")
        # We need _is_cloudfront_403 to return True
        llm._is_cloudfront_403 = Mock(return_value=True)

        # Mock execute_with_retry to raise error first, then return success string
        llm.retry_handler.execute_with_retry = Mock(
            side_effect=[error_403, "Success response"]
        )

        # Mock _process_streaming_response to just return the content
        llm._process_streaming_response = Mock(return_value="Success")

        # Enable WAF retry
        llm.waf_config.retry_with_sanitization = True
        llm.waf_config.enable_waf_sanitization = False

        # Mock sanitization
        llm._sanitize_for_waf = Mock(return_value="Sanitized content")

        # Mock sleep to avoid delay
        with patch("time.sleep"):
            response = llm.prompt("test prompt")

        assert response == "Success"
        assert llm.retry_handler.execute_with_retry.call_count == 2
        assert llm.waf_metrics.cloudfront_403_errors == 1

        # Check that sanitization was applied on second call
        # Again, since execute_with_retry is mocked and doesn't call the real function,
        # we can't check the arguments passed to client.create.
        # But we can check if _sanitize_for_waf was called.
        llm._sanitize_for_waf.assert_called()

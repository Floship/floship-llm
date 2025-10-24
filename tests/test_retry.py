"""Tests for API retry logic."""

import os
import time
from unittest.mock import Mock, patch, MagicMock
import pytest
from openai import APIStatusError, RateLimitError, APIConnectionError

from floship_llm import LLM


class TestAPIRetry:
    """Tests for the API retry mechanism."""

    def setup_method(self):
        """Set up test environment."""
        os.environ['INFERENCE_URL'] = 'http://test.com'
        os.environ['INFERENCE_MODEL_ID'] = 'test-model'
        os.environ['INFERENCE_KEY'] = 'test-key'

    def teardown_method(self):
        """Clean up test environment."""
        pass

    def test_api_call_with_retry_success_on_first_attempt(self):
        """Test that successful calls don't retry."""
        with patch('floship_llm.client.OpenAI') as mock_openai:
            llm = LLM()

            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Success"
            mock_response.choices[0].message.tool_calls = None

            mock_create = Mock(return_value=mock_response)
            mock_openai.return_value.chat.completions.create = mock_create

            result = llm.prompt("Test prompt")

            assert mock_create.call_count == 1
            assert result == "Success"

    def test_api_call_with_retry_403_error(self):
        """Test retry on 403 Forbidden error."""
        with patch('floship_llm.client.OpenAI') as mock_openai:
            llm = LLM()

            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Success"
            mock_response.choices[0].message.tool_calls = None

            call_count = 0

            def mock_create(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise APIStatusError(
                        message="Request blocked",
                        response=Mock(status_code=403),
                        body={"error": "Request blocked"}
                    )
                return mock_response

            mock_openai.return_value.chat.completions.create = mock_create

            with patch('time.sleep'):  # Speed up test
                result = llm.prompt("Test prompt")

            assert call_count == 3
            assert result == "Success"

    def test_api_call_with_retry_429_error(self):
        """Test retry on 429 Rate Limit error."""
        with patch('floship_llm.client.OpenAI') as mock_openai:
            llm = LLM()

            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Success"
            mock_response.choices[0].message.tool_calls = None

            call_count = 0

            def mock_create(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise RateLimitError(
                        message="Rate limit exceeded",
                        response=Mock(status_code=429),
                        body={"error": "Rate limit exceeded"}
                    )
                return mock_response

            mock_openai.return_value.chat.completions.create = mock_create

            with patch('time.sleep'):  # Speed up test
                result = llm.prompt("Test prompt")

            assert call_count == 2
            assert result == "Success"

    def test_api_call_with_retry_500_error(self):
        """Test retry on 500 Server Error."""
        with patch('floship_llm.client.OpenAI') as mock_openai:
            llm = LLM()

            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Success"
            mock_response.choices[0].message.tool_calls = None

            call_count = 0

            def mock_create(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise APIStatusError(
                        message="Internal server error",
                        response=Mock(status_code=500),
                        body={"error": "Internal server error"}
                    )
                return mock_response

            mock_openai.return_value.chat.completions.create = mock_create

            with patch('time.sleep'):  # Speed up test
                result = llm.prompt("Test prompt")

            assert call_count == 2
            assert result == "Success"

    def test_api_call_with_retry_connection_error(self):
        """Test retry on connection errors."""
        with patch('floship_llm.client.OpenAI') as mock_openai:
            llm = LLM()

            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Success"
            mock_response.choices[0].message.tool_calls = None

            call_count = 0

            def mock_create(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise APIConnectionError(
                        message="Connection failed",
                        request=Mock()
                    )
                return mock_response

            mock_openai.return_value.chat.completions.create = mock_create

            with patch('time.sleep'):  # Speed up test
                result = llm.prompt("Test prompt")

            assert call_count == 2
            assert result == "Success"

    def test_api_call_with_retry_max_retries_exceeded(self):
        """Test that after max retries, error is raised."""
        with patch('floship_llm.client.OpenAI') as mock_openai:
            llm = LLM()

            call_count = 0

            def mock_create(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                raise APIStatusError(
                    message="Request blocked",
                    response=Mock(status_code=403),
                    body={"error": "Request blocked"}
                )

            mock_openai.return_value.chat.completions.create = mock_create

            with patch('time.sleep'):  # Speed up test
                with pytest.raises(APIStatusError):
                    llm.prompt("Test prompt")

            # Should have tried 3 times
            assert call_count == 3

    def test_api_call_with_retry_non_retryable_error(self):
        """Test that non-retryable errors fail immediately."""
        with patch('floship_llm.client.OpenAI') as mock_openai:
            llm = LLM()

            call_count = 0

            def mock_create(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                raise APIStatusError(
                    message="Bad request",
                    response=Mock(status_code=400),
                    body={"error": "Bad request"}
                )

            mock_openai.return_value.chat.completions.create = mock_create

            with pytest.raises(APIStatusError):
                llm.prompt("Test prompt")

            # Should only try once (no retry on 400)
            assert call_count == 1

    def test_api_call_with_retry_sleep_timing(self):
        """Test that retry delays increase linearly."""
        with patch('floship_llm.client.OpenAI') as mock_openai:
            llm = LLM()

            mock_response = Mock()
            mock_response.choices = [Mock()]
            mock_response.choices[0].message.content = "Success"
            mock_response.choices[0].message.tool_calls = None

            call_count = 0

            def mock_create(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise APIStatusError(
                        message="Request blocked",
                        response=Mock(status_code=403),
                        body={"error": "Request blocked"}
                    )
                return mock_response

            mock_openai.return_value.chat.completions.create = mock_create

            with patch('time.sleep') as mock_sleep:
                result = llm.prompt("Test prompt")

            # Should have slept twice: 5s, then 10s
            assert mock_sleep.call_count == 2
            assert mock_sleep.call_args_list[0][0][0] == 5  # First retry: 5s
            assert mock_sleep.call_args_list[1][0][0] == 10  # Second retry: 10s
            assert call_count == 3
            assert result == "Success"

    def test_api_call_with_retry_all_status_codes(self):
        """Test retry behavior for various status codes."""
        retryable_codes = [403, 429, 500, 502, 503, 504]
        non_retryable_codes = [400, 401, 404]

        for status_code in retryable_codes:
            with patch('floship_llm.client.OpenAI') as mock_openai:
                llm = LLM()

                mock_response = Mock()
                mock_response.choices = [Mock()]
                mock_response.choices[0].message.content = "Success"
                mock_response.choices[0].message.tool_calls = None

                call_count = 0

                def mock_create(*args, **kwargs):
                    nonlocal call_count
                    call_count += 1
                    if call_count == 1:
                        raise APIStatusError(
                            message=f"Error {status_code}",
                            response=Mock(status_code=status_code),
                            body={"error": f"Error {status_code}"}
                        )
                    return mock_response

                mock_openai.return_value.chat.completions.create = mock_create

                with patch('time.sleep'):
                    result = llm.prompt("Test prompt")

                assert call_count == 2, f"Status code {status_code} should be retried"
                assert result == "Success"

        for status_code in non_retryable_codes:
            with patch('floship_llm.client.OpenAI') as mock_openai:
                llm = LLM()

                call_count = 0

                def mock_create(*args, **kwargs):
                    nonlocal call_count
                    call_count += 1
                    raise APIStatusError(
                        message=f"Error {status_code}",
                        response=Mock(status_code=status_code),
                        body={"error": f"Error {status_code}"}
                    )

                mock_openai.return_value.chat.completions.create = mock_create

                with pytest.raises(APIStatusError):
                    llm.prompt("Test prompt")

                assert call_count == 1, f"Status code {status_code} should not be retried"

    def test_embed_with_retry(self):
        """Test that embed method also uses retry logic."""
        with patch('floship_llm.client.OpenAI') as mock_openai:
            llm = LLM()

            mock_embedding = Mock()
            mock_embedding.data = [Mock(embedding=[0.1, 0.2, 0.3])]

            call_count = 0

            def mock_create(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise APIStatusError(
                        message="Request blocked",
                        response=Mock(status_code=403),
                        body={"error": "Request blocked"}
                    )
                return mock_embedding

            mock_openai.return_value.embeddings.create = mock_create

            with patch('time.sleep'):
                result = llm.embed("Test text")

            assert call_count == 2
            assert result == [0.1, 0.2, 0.3]

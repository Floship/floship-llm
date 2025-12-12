"""Tests for streaming response processing in LLM client."""

from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from floship_llm.client import LLM, TruncatedResponseError


class ResponseModelForTesting(BaseModel):
    name: str
    value: int


class TestStreamingProcessing:
    """Test cases for _process_streaming_response method."""

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

    def test_process_streaming_response_basic(self, llm):
        """Test basic text response processing."""
        response = "Hello, world!"
        result = llm._process_streaming_response(response)
        assert result == "Hello, world!"
        assert llm._last_raw_response == "Hello, world!"

    def test_process_streaming_response_empty(self, llm):
        """Test empty response processing."""
        result = llm._process_streaming_response("")
        assert result == ""

    def test_process_streaming_response_thinking(self, llm):
        """Test removal of thinking tags."""
        response = "<reasoning>This is thinking process</reasoning>Final answer"
        result = llm._process_streaming_response(response)
        assert result == "Final answer"

    def test_process_streaming_response_thinking_multiline(self, llm):
        """Test removal of multiline thinking tags."""
        response = (
            "<reasoning>\nThinking line 1\nThinking line 2\n</reasoning>\nFinal answer"
        )
        result = llm._process_streaming_response(response)
        assert result.strip() == "Final answer"

    def test_process_streaming_response_tags(self, llm):
        """Test extraction of content within response tags."""
        response = "Prefix <response>Actual content</response> Suffix"
        result = llm._process_streaming_response(response)
        assert result == "Actual content"

    def test_process_streaming_response_max_length(self, llm):
        """Test max length check."""
        llm.max_length = 10
        response = "This is a very long response that exceeds the limit"

        # Mock retry_prompt to return a fallback
        llm.retry_prompt = Mock(return_value="Retry successful")

        result = llm._process_streaming_response(response)

        assert result == "Retry successful"
        llm.retry_prompt.assert_called_once()

    def test_process_streaming_response_structured(self, llm):
        """Test structured output parsing."""
        llm.response_format = ResponseModelForTesting
        response = '{"name": "test", "value": 123}'

        result = llm._process_streaming_response(response)

        assert isinstance(result, ResponseModelForTesting)
        assert result.name == "test"
        assert result.value == 123

    def test_process_streaming_response_structured_with_markdown(self, llm):
        """Test structured output parsing with markdown code blocks."""
        llm.response_format = ResponseModelForTesting
        response = '```json\n{"name": "test", "value": 123}\n```'

        result = llm._process_streaming_response(response)

        assert isinstance(result, ResponseModelForTesting)
        assert result.name == "test"
        assert result.value == 123

    def test_process_streaming_response_truncated_json(self, llm):
        """Test detection of truncated JSON."""
        llm.response_format = ResponseModelForTesting
        # Incomplete JSON
        response = '{"name": "test", "value": '

        with pytest.raises(TruncatedResponseError):
            llm._process_streaming_response(response)

    def test_process_streaming_response_invalid_json(self, llm):
        """Test handling of invalid JSON (not truncated, just bad)."""
        llm.response_format = ResponseModelForTesting
        response = "Not JSON at all"

        with pytest.raises(ValueError, match="Could not extract valid JSON"):
            llm._process_streaming_response(response)

    def test_process_streaming_response_invalid_json_with_fallback(self, llm):
        """Return raw text when fallback flag is enabled."""
        llm.response_format = ResponseModelForTesting
        llm.allow_response_format_fallback = True
        response = "Not JSON at all"

        result = llm._process_streaming_response(response)

        assert result == response

    def test_process_streaming_response_validation_error_with_fallback(self, llm):
        """Return raw text when JSON parses but schema validation fails."""
        llm.response_format = ResponseModelForTesting
        llm.allow_response_format_fallback = True
        response = '{"name": "test"}'  # missing required field `value`

        result = llm._process_streaming_response(response)

        assert result == response

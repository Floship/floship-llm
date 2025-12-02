import os
from unittest.mock import Mock, patch

from openai import BadRequestError

from floship_llm.client import LLM


class MockResponse:
    def __init__(self, status_code, content):
        self.status_code = status_code
        self.content = content
        self.headers = {}
        self.request = Mock()


class TestExtendedThinkingRetry:
    def test_extended_thinking_retry_on_400(self):
        """Test that LLM retries without extended thinking when API returns 400 validation error."""

        # Set environment variables
        os.environ["INFERENCE_URL"] = "https://api.example.com"
        os.environ["INFERENCE_MODEL_ID"] = "test-model"
        os.environ["INFERENCE_KEY"] = "test-key"

        error_message = '{"code":400,"message":"Failed to execute chat: ValidationException: The model returned the following errors: messages.1.content.0.type: Expected `thinking` or `redacted_thinking`, but found `text`. When `thinking` is enabled, a final `assistant` message must start with a thinking block (preceeding the lastmost set of `tool_use` and `tool_result` blocks). We recommend you include thinking blocks from previous turns. To avoid this requirement, disable `thinking`. Please consult our documentation at https://docs.claude.com/en/docs/build-with-claude/extended-thinking","type":"invalid_request"}'

        # Create a mock 400 error
        mock_response = MockResponse(400, error_message.encode())
        error = BadRequestError(
            message=error_message, response=mock_response, body=None
        )

        with patch("floship_llm.client.OpenAI") as mock_openai:
            # Setup mock client
            mock_client_instance = Mock()
            mock_openai.return_value = mock_client_instance

            # First call raises 400, second call succeeds with streaming
            mock_stream = Mock()
            mock_chunk = Mock()
            mock_chunk.choices = [Mock(delta=Mock(content="Success"))]
            mock_stream.__iter__ = Mock(return_value=iter([mock_chunk]))

            mock_client_instance.chat.completions.create.side_effect = [
                error,
                mock_stream,
            ]

            # Initialize LLM with extended thinking
            llm = LLM(
                api_key="test",  # pragma: allowlist secret
                extended_thinking={"enabled": True, "budget_tokens": 1024},
            )

            # Call prompt (default stream=True)
            response = llm.prompt("test prompt")

            # Verify response
            assert response == "Success"

            # Verify calls
            assert mock_client_instance.chat.completions.create.call_count == 2

            # First call should have extended thinking
            call1_kwargs = mock_client_instance.chat.completions.create.call_args_list[
                0
            ].kwargs
            assert "extra_body" in call1_kwargs
            assert "extended_thinking" in call1_kwargs["extra_body"]

            # Second call should NOT have extended thinking
            call2_kwargs = mock_client_instance.chat.completions.create.call_args_list[
                1
            ].kwargs
            # Depending on implementation, extra_body might be missing or thinking might be removed
            if "extra_body" in call2_kwargs:
                assert "extended_thinking" not in call2_kwargs["extra_body"]

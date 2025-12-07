"""Tests for the LLM client module."""

import os
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from floship_llm.client import LLM
from floship_llm.schemas import ThinkingModel, ToolFunction, ToolParameter


def create_mock_stream_response(content: str):
    """Create a mock streaming response that yields chunks.

    Args:
        content: The full response content to stream in chunks

    Returns:
        A list of mock chunk objects that can be iterated over
    """
    chunks = []
    # Split content into chunks (simulate streaming)
    for char in content:
        mock_chunk = Mock()
        mock_delta = Mock()
        mock_delta.content = char
        mock_choice = Mock()
        mock_choice.delta = mock_delta
        mock_chunk.choices = [mock_choice]
        chunks.append(mock_chunk)
    return chunks


class ResponseModelForTesting(BaseModel):
    """Test pydantic model for response format testing."""

    name: str
    value: int


class ThinkingResponseModelForTesting(ThinkingModel):
    """Test pydantic model that already extends ThinkingModel."""

    answer: str
    confidence: int


class TestLLM:
    """Test cases for the LLM class."""

    def setup_method(self):
        """Set up test environment variables."""
        self.env_vars = {
            "INFERENCE_URL": "https://test-api.example.com",
            "INFERENCE_MODEL_ID": "test-model",
            "INFERENCE_KEY": "test-key",
            "INFERENCE_SUPPORTS_PARALLEL_REQUESTS": "True",
        }

        # Patch environment variables
        self.env_patcher = patch.dict(os.environ, self.env_vars, clear=False)
        self.env_patcher.start()

    def teardown_method(self):
        """Clean up test environment."""
        self.env_patcher.stop()

    def test_init_success(self):
        """Test successful LLM initialization."""
        with patch("floship_llm.client.OpenAI") as mock_openai:
            llm = LLM()

            assert llm.type == "completion"
            assert llm.base_url == "https://test-api.example.com"
            assert llm.model == "test-model"
            assert llm.temperature == 0.15
            assert llm.max_retry == 3
            assert llm.retry_count == 0
            # Heroku-specific parameters (deprecated parameters are None by default)
            assert llm.frequency_penalty is None
            assert llm.presence_penalty is None
            assert llm.max_completion_tokens is None
            assert llm.top_k is None
            assert llm.top_p is None
            assert llm.extended_thinking is None
            assert llm.allow_ignored_params is False
            assert llm.continuous is True
            assert llm.messages == []
            assert llm.max_length == 100_000
            assert llm.input_tokens_limit == 40_000
            assert llm.system is None
            assert llm.response_format is None

            mock_openai.assert_called_once_with(
                api_key="test-key", base_url="https://test-api.example.com"
            )

    def test_init_with_custom_api_key(self):
        """Test LLM initialization with custom API key."""
        custom_key = "custom-api-key"
        with patch("floship_llm.client.OpenAI") as mock_openai:
            llm = LLM(api_key=custom_key)

            assert llm.api_key == custom_key
            mock_openai.assert_called_once_with(
                api_key=custom_key, base_url="https://test-api.example.com"
            )

    def test_init_with_custom_parameters(self):
        """Test LLM initialization with custom parameters."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(
                temperature=0.5,
                frequency_penalty=0.1,
                presence_penalty=0.1,
                continuous=False,
                max_length=50000,
                input_tokens_limit=20000,
                max_retry=5,
                system="You are a helpful assistant",
                response_format=ResponseModelForTesting,
            )

            assert llm.temperature == 0.5
            assert llm.frequency_penalty == 0.1
            assert llm.presence_penalty == 0.1
            assert llm.continuous is False
            assert llm.max_length == 50000
            assert llm.input_tokens_limit == 20000
            assert llm.max_retry == 5
            assert llm.retry_count == 0
            # response_format is wrapped, but _original_response_format preserves user's model
            assert llm._original_response_format == ResponseModelForTesting
            assert llm._response_format_wrapped is True
            # System message should be added to messages
            assert len(llm.messages) == 1
            assert llm.messages[0]["role"] == "system"

    def test_init_missing_env_vars(self):
        """Test LLM initialization with missing environment variables."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(
                ValueError, match="INFERENCE_URL environment variable must be set"
            ):
                LLM()

        with patch.dict(os.environ, {"INFERENCE_URL": "test"}, clear=True):
            with pytest.raises(
                ValueError, match="INFERENCE_MODEL_ID environment variable must be set"
            ):
                LLM()

        with patch.dict(
            os.environ,
            {"INFERENCE_URL": "test", "INFERENCE_MODEL_ID": "test"},
            clear=True,
        ):
            with pytest.raises(
                ValueError, match="INFERENCE_KEY environment variable must be set"
            ):
                LLM()

    def test_init_invalid_type(self):
        """Test LLM initialization with invalid type."""
        with patch("floship_llm.client.OpenAI"):
            with pytest.raises(
                ValueError, match="type must be 'completion' or 'embedding'"
            ):
                LLM(type="invalid")

    def test_init_embedding_type_supported(self):
        """Test LLM initialization with embedding type is now supported."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(type="embedding")
            assert llm.type == "embedding"

    def test_init_invalid_response_format(self):
        """Test LLM initialization with invalid response format."""
        with patch("floship_llm.client.OpenAI"):
            with pytest.raises(
                ValueError, match="response_format must be a subclass of BaseModel"
            ):
                LLM(response_format=str)

    def test_supports_parallel_requests_property(self):
        """Test supports_parallel_requests property."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
            # Returns boolean True/False
            assert llm.supports_parallel_requests is True

        with patch.dict(os.environ, {"INFERENCE_SUPPORTS_PARALLEL_REQUESTS": "False"}):
            with patch("floship_llm.client.OpenAI"):
                llm = LLM()
                assert llm.supports_parallel_requests is False

    def test_supports_frequency_penalty_property(self):
        """Test supports_frequency_penalty property - Heroku does not support this."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
            # Heroku Inference API does not support frequency_penalty
            # Should return False for all models and log a warning
            llm.model = "gpt-4"
            assert llm.supports_frequency_penalty is False

            llm.model = "claude-3"
            assert llm.supports_frequency_penalty is False

            llm.model = "gemini-pro"
            assert llm.supports_frequency_penalty is False

    def test_supports_presence_penalty_property(self):
        """Test supports_presence_penalty property - Heroku does not support this."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
            # Heroku Inference API does not support presence_penalty
            # Should return False for all models and log a warning
            llm.model = "gpt-4"
            assert llm.supports_presence_penalty is False

            llm.model = "claude-3"
            assert llm.supports_presence_penalty is False

            llm.model = "gemini-pro"
            assert llm.supports_presence_penalty is False

    def test_require_response_format_property(self):
        """Test require_response_format property."""
        with patch("floship_llm.client.OpenAI"):
            # Without response format
            llm = LLM()
            assert llm.require_response_format is False

        # With response format
        llm = LLM(response_format=ResponseModelForTesting)
        assert llm.require_response_format is True

    def test_add_message_success(self):
        """Test successful message addition."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
            llm.add_message("user", "Hello world")

            assert len(llm.messages) == 1
            assert llm.messages[0]["role"] == "user"
            assert llm.messages[0]["content"] == "Hello world"

    def test_add_message_with_response_format(self):
        """Test message addition with response format."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(response_format=ResponseModelForTesting)
            # Skip the system message that was added during init
            llm.messages = []

            llm.add_message("user", "Test message")

            assert len(llm.messages) == 1
            assert llm.messages[0]["role"] == "user"
            # Check that JSON schema instructions are appended
            assert "MUST respond with a valid JSON object" in llm.messages[0]["content"]
            assert "Test message" in llm.messages[0]["content"]

    def test_add_message_invalid_role(self):
        """Test message addition with invalid role."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()

            with pytest.raises(
                ValueError, match="Role must be 'user', 'assistant', or 'system'"
            ):
                llm.add_message("invalid", "content")

            with pytest.raises(
                ValueError, match="Role must be 'user', 'assistant', or 'system'"
            ):
                llm.add_message(123, "content")

    def test_sanitize_messages(self):
        """Test message sanitization."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()

            # Test multiple spaces compaction
            result = llm._sanitize_messages("Hello    world   test")
            assert result == "Hello world test"

            # Test whitespace trimming
            result = llm._sanitize_messages("  Hello world  ")
            assert result == "Hello world"

            # Test newlines and tabs - the implementation compacts all whitespace into single spaces
            result = llm._sanitize_messages("Hello\n\t world")
            assert result == "Hello world"

    def test_get_request_params(self):
        """Test request parameters generation - Heroku-specific."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(temperature=0.7)
            params = llm.get_request_params()

            # Heroku does not support frequency_penalty or presence_penalty
            # Only model and temperature should be included by default
            expected = {"model": "test-model", "temperature": 0.7}
            assert params == expected

    def test_get_request_params_with_heroku_parameters(self):
        """Test request parameters with Heroku-specific parameters."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(temperature=0.7, max_completion_tokens=2000, top_k=50, top_p=0.9)
            params = llm.get_request_params()

            # When top_p is set, temperature is not included (mutual exclusivity with Claude)
            # Heroku-specific params (top_k, top_p) are wrapped in extra_body
            expected = {
                "model": "test-model",
                "max_completion_tokens": 2000,
                "extra_body": {"top_k": 50, "top_p": 0.9},
            }
            assert params == expected

    def test_get_request_params_with_extended_thinking(self):
        """Test request parameters with extended thinking for Claude."""
        with patch("floship_llm.client.OpenAI"):
            extended_thinking_config = {
                "enabled": True,
                "budget_tokens": 1024,
                "include_reasoning": True,
            }
            llm = LLM(temperature=0.7, extended_thinking=extended_thinking_config)
            params = llm.get_request_params()

            # When extended_thinking is enabled, temperature is forced to 1.0
            # extended_thinking goes in extra_body
            expected = {
                "model": "test-model",
                "temperature": 1.0,
                "extra_body": {"extended_thinking": extended_thinking_config},
            }
            assert params == expected

    def test_get_request_params_auto_disables_thinking_with_tool_history(self):
        """Test that extended_thinking is auto-disabled when conversation has tool call history.

        Claude requires assistant messages to start with thinking blocks when
        thinking is enabled. Since we store messages as plain text (stripping
        thinking blocks), we must auto-disable thinking for follow-up requests
        after tool execution to avoid 400 errors.
        """
        with patch("floship_llm.client.OpenAI"):
            extended_thinking_config = {
                "enabled": True,
                "budget_tokens": 1024,
            }
            llm = LLM(extended_thinking=extended_thinking_config)

            # Add a conversation history with tool calls (simulating post-tool state)
            llm.messages = [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Do something"},
                {
                    "role": "assistant",
                    "content": "I'll help with that",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {"name": "some_tool", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_123", "content": "Tool result"},
            ]

            params = llm.get_request_params()

            # extended_thinking should NOT be in params (auto-disabled)
            assert "extra_body" not in params or "extended_thinking" not in params.get(
                "extra_body", {}
            )
            # Temperature should NOT be 1.0 (default behavior)
            assert params.get("temperature") == 0.15  # Default temperature

    def test_get_request_params_keeps_thinking_without_tool_history(self):
        """Test that extended_thinking stays enabled when no tool calls in history."""
        with patch("floship_llm.client.OpenAI"):
            extended_thinking_config = {
                "enabled": True,
                "budget_tokens": 1024,
            }
            llm = LLM(extended_thinking=extended_thinking_config)

            # Add conversation history WITHOUT tool calls
            llm.messages = [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "How are you?"},
            ]

            params = llm.get_request_params()

            # extended_thinking should still be enabled
            assert "extra_body" in params
            assert "extended_thinking" in params["extra_body"]
            assert params["extra_body"]["extended_thinking"] == extended_thinking_config
            # Temperature should be 1.0 for extended thinking
            assert params.get("temperature") == 1.0

    def test_reset_restores_extended_thinking_if_auto_disabled(self):
        """Test that reset() restores extended_thinking when it was auto-disabled."""
        with patch("floship_llm.client.OpenAI"):
            extended_thinking_config = {
                "enabled": True,
                "budget_tokens": 1024,
            }
            llm = LLM(extended_thinking=extended_thinking_config)

            # Add tool call history to trigger auto-disable
            llm.messages = [
                {"role": "user", "content": "Do something"},
                {
                    "role": "assistant",
                    "content": ".",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {"name": "test_tool", "arguments": "{}"},
                        }
                    ],
                },
            ]

            # Get params - this should auto-disable thinking
            params = llm.get_request_params()
            assert "extended_thinking" not in params.get("extra_body", {})
            assert llm._thinking_auto_disabled is True

            # Reset should restore thinking
            llm.reset()

            assert llm._thinking_auto_disabled is False
            assert llm.extended_thinking == extended_thinking_config

            # Params should have thinking enabled again
            params = llm.get_request_params()
            assert "extra_body" in params
            assert params["extra_body"]["extended_thinking"] == extended_thinking_config

    def test_can_use_extended_thinking_fresh_conversation(self):
        """Test can_use_extended_thinking returns True for fresh conversation."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(extended_thinking={"enabled": True, "budget_tokens": 1024})

            # Fresh conversation - should be True
            assert llm.can_use_extended_thinking() is True

    def test_can_use_extended_thinking_after_tool_calls(self):
        """Test can_use_extended_thinking returns False after tool execution."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(extended_thinking={"enabled": True, "budget_tokens": 1024})

            # Add tool call history
            llm.messages = [
                {"role": "user", "content": "Do something"},
                {
                    "role": "assistant",
                    "content": ".",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {"name": "test", "arguments": "{}"},
                        }
                    ],
                },
            ]

            # After tool calls - should be False
            assert llm.can_use_extended_thinking() is False

    def test_can_use_extended_thinking_not_configured(self):
        """Test can_use_extended_thinking returns False when not configured."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()  # No extended_thinking

            assert llm.can_use_extended_thinking() is False

    def test_restore_extended_thinking_success(self):
        """Test restore_extended_thinking succeeds on fresh conversation."""
        with patch("floship_llm.client.OpenAI"):
            config = {"enabled": True, "budget_tokens": 1024}
            llm = LLM(extended_thinking=config)

            # Manually disable thinking
            llm.extended_thinking = None
            llm._thinking_auto_disabled = True

            # Should be able to restore (no tool history)
            result = llm.restore_extended_thinking()

            assert result is True
            assert llm.extended_thinking == config
            assert llm._thinking_auto_disabled is False

    def test_restore_extended_thinking_fails_with_tool_history(self):
        """Test restore_extended_thinking fails when tool history exists."""
        with patch("floship_llm.client.OpenAI"):
            config = {"enabled": True, "budget_tokens": 1024}
            llm = LLM(extended_thinking=config)

            # Add tool call history
            llm.messages = [
                {
                    "role": "assistant",
                    "content": ".",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {"name": "test", "arguments": "{}"},
                        }
                    ],
                },
            ]
            llm.extended_thinking = None
            llm._thinking_auto_disabled = True

            # Should fail to restore (tool history present)
            result = llm.restore_extended_thinking()

            assert result is False
            assert llm.extended_thinking is None

    def test_get_request_params_with_allow_ignored_params(self):
        """Test request parameters with deprecated params when allow_ignored_params=True."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(
                temperature=0.7,
                frequency_penalty=0.2,
                presence_penalty=0.2,
                allow_ignored_params=True,
            )
            params = llm.get_request_params()

            # When allow_ignored_params=True, deprecated params are included
            expected = {
                "model": "test-model",
                "temperature": 0.7,
                "frequency_penalty": 0.2,
                "presence_penalty": 0.2,
            }
            assert params == expected

    def test_get_request_params_claude_model(self):
        """Test request parameters for Claude model (no penalties)."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
            llm.model = "claude-3"

            params = llm.get_request_params()

            expected = {"model": "claude-3", "temperature": 0.15}
            assert params == expected
            assert "frequency_penalty" not in params
            assert "presence_penalty" not in params

    def test_get_embedding_params(self):
        """Test embedding parameters generation."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
            params = llm.get_embedding_params()

            expected = {"model": "test-model"}
            assert params == expected

    def test_embed_success(self):
        """Test successful embedding generation."""
        with patch("floship_llm.client.OpenAI") as mock_openai:
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3])]
            mock_openai.return_value.embeddings.create.return_value = mock_response

            llm = LLM()
            result = llm.embed("test text")

            assert result == [0.1, 0.2, 0.3]
            mock_openai.return_value.embeddings.create.assert_called_once_with(
                model="test-model", input="test text"
            )

    def test_embed_empty_text(self):
        """Test embedding with empty text."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()

            with pytest.raises(ValueError, match="Input cannot be empty"):
                llm.embed("")

            with pytest.raises(ValueError, match="Input cannot be empty"):
                llm.embed(None)

    def test_embed_no_data(self):
        """Test embedding with no data in response."""
        with patch("floship_llm.client.OpenAI") as mock_openai:
            mock_response = Mock()
            mock_response.data = []
            mock_openai.return_value.embeddings.create.return_value = mock_response

            llm = LLM()
            result = llm.embed("test text")

            assert result is None

    def test_prompt_success(self):
        """Test successful prompt generation with streaming (default)."""
        with patch("floship_llm.client.OpenAI") as mock_openai:
            # Create mock streaming response
            mock_openai.return_value.chat.completions.create.return_value = (
                create_mock_stream_response("Test response")
            )

            llm = LLM()
            result = llm.prompt("Hello")

            assert result == "Test response"
            assert len(llm.messages) == 2  # user + assistant
            assert llm.messages[0]["role"] == "user"
            assert llm.messages[0]["content"] == "Hello"
            assert llm.messages[1]["role"] == "assistant"
            assert llm.messages[1]["content"] == "Test response"

    def test_prompt_with_system_message(self):
        """Test prompt with system message."""
        with patch("floship_llm.client.OpenAI") as mock_openai:
            # Create mock streaming response
            mock_openai.return_value.chat.completions.create.return_value = (
                create_mock_stream_response("Test response")
            )

            llm = LLM()
            result = llm.prompt("Hello", system="You are helpful")

            assert result == "Test response"
            assert len(llm.messages) == 3  # system + user + assistant
            assert llm.messages[0]["role"] == "system"
            assert llm.messages[1]["role"] == "user"
            assert llm.messages[2]["role"] == "assistant"

    def test_prompt_continuous_false(self):
        """Test prompt with continuous=False."""
        with patch("floship_llm.client.OpenAI") as mock_openai:
            # Create mock streaming response
            mock_openai.return_value.chat.completions.create.return_value = (
                create_mock_stream_response("Test response")
            )

            llm = LLM(continuous=False)
            result = llm.prompt("Hello")

            assert result == "Test response"
            assert len(llm.messages) == 0  # Should be reset

    def test_prompt_with_response_format(self):
        """Test prompt with response format returns user's original model type."""
        with patch("floship_llm.client.OpenAI") as mock_openai:
            # Create mock streaming response with wrapped JSON content (reasoning + response)
            wrapped_response = '{"reasoning": "Let me create a test record.", "response": {"name": "test", "value": 42}}'
            mock_openai.return_value.chat.completions.create.return_value = (
                create_mock_stream_response(wrapped_response)
            )

            with patch(
                "floship_llm.utils.lm_json_utils.extract_strict_json"
            ) as mock_extract:
                mock_extract.return_value = wrapped_response

                llm = LLM(response_format=ResponseModelForTesting)
                # Clear system message added during init
                llm.messages = []

                result = llm.prompt("Hello")

                # User gets back their original model type (unwrapped)
                assert isinstance(result, ResponseModelForTesting)
                assert result.name == "test"
                assert result.value == 42
                # Reasoning should be captured via unified interface
                assert llm.get_last_reasoning() == "Let me create a test record."

    def test_retry_prompt(self):
        """Test retry prompt functionality."""
        with patch("floship_llm.client.OpenAI") as mock_openai:
            mock_choice = Mock()
            mock_choice.message.content = "Retry response"
            mock_response = Mock()
            mock_response.choices = [mock_choice]
            mock_openai.return_value.chat.completions.create.return_value = (
                mock_response
            )

            llm = LLM()
            with patch.object(llm, "prompt") as mock_prompt:
                llm.retry_prompt("test prompt")
                mock_prompt.assert_called_once_with("test prompt", retry=True)
                assert llm.retry_count == 1

    def test_retry_prompt_max_retry_default(self):
        """Test retry prompt with default max_retry value."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
            assert llm.max_retry == 3
            assert llm.retry_count == 0

    def test_retry_prompt_custom_max_retry(self):
        """Test retry prompt with custom max_retry value."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(max_retry=5)
            assert llm.max_retry == 5
            assert llm.retry_count == 0

    def test_retry_prompt_max_retry_exceeded(self):
        """Test retry prompt when max retry is exceeded."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(max_retry=2)

            # Simulate reaching max retry
            llm.retry_count = 2

            result = llm.retry_prompt("test prompt")
            assert result is None
            assert llm.retry_count == 2  # Should not increment further

    def test_retry_prompt_increments_counter(self):
        """Test that retry_prompt increments the retry counter."""
        with patch("floship_llm.client.OpenAI") as mock_openai:
            mock_choice = Mock()
            mock_choice.message.content = "Retry response"
            mock_response = Mock()
            mock_response.choices = [mock_choice]
            mock_openai.return_value.chat.completions.create.return_value = (
                mock_response
            )

            llm = LLM(max_retry=3)

            # First retry
            with patch.object(llm, "prompt", return_value="response1"):
                result = llm.retry_prompt("test")
                assert llm.retry_count == 1
                assert result == "response1"

            # Second retry
            with patch.object(llm, "prompt", return_value="response2"):
                result = llm.retry_prompt("test")
                assert llm.retry_count == 2
                assert result == "response2"

            # Third retry
            with patch.object(llm, "prompt", return_value="response3"):
                result = llm.retry_prompt("test")
                assert llm.retry_count == 3
                assert result == "response3"

            # Fourth retry should fail
            result = llm.retry_prompt("test")
            assert result is None
            assert llm.retry_count == 3

    def test_prompt_resets_retry_count(self):
        """Test that new prompt resets retry count."""
        with patch("floship_llm.client.OpenAI") as mock_openai:
            # Create mock streaming response
            mock_openai.return_value.chat.completions.create.return_value = (
                create_mock_stream_response("Test response")
            )

            llm = LLM()
            llm.retry_count = 2  # Simulate some retries

            llm.prompt("New prompt")
            assert llm.retry_count == 0  # Should be reset

    def test_reset(self):
        """Test conversation reset."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
            llm.add_message("user", "Hello")
            llm.retry_count = 2  # Simulate some retries
            assert len(llm.messages) == 1
            assert llm.retry_count == 2

            llm.reset()
            assert len(llm.messages) == 0
            assert llm.retry_count == 0

    def test_process_response_basic(self):
        """Test basic response processing."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()

            mock_choice = Mock()
            mock_choice.message.content = "  Test response  "
            mock_response = Mock()
            mock_response.choices = [mock_choice]

            result = llm.process_response(mock_response)

            # The method returns the original content, but processes and strips for internal use
            assert result == "  Test response  "
            assert len(llm.messages) == 1
            assert llm.messages[0]["role"] == "assistant"
            assert llm.messages[0]["content"] == "Test response"

    def test_process_response_with_thinking_tags(self):
        """Test response processing with thinking tags."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()

            mock_choice = Mock()
            mock_choice.message.content = (
                "<reasoning>Some thinking</reasoning>Final response"
            )
            mock_response = Mock()
            mock_response.choices = [mock_choice]

            result = llm.process_response(mock_response)

            # Returns original content since no response_format is set
            assert result == "<reasoning>Some thinking</reasoning>Final response"
            # But internal processing should remove think tags
            assert llm.messages[-1]["content"] == "Final response"

    def test_extended_thinking_strips_thinking_tags_from_history(self):
        """Test that thinking tags are ALWAYS stripped from message history.

        Heroku's OpenAI-compatible API causes 500 errors if <reasoning> tags are
        sent in message history on subsequent requests. The raw response
        (with thinking tags) is preserved in get_last_raw_response() for user access.
        """
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(
                extended_thinking={"enabled": True, "budget_tokens": 1024},
                continuous=True,
            )

            mock_choice = Mock()
            mock_choice.message.content = (
                "<reasoning>I'm reasoning about this</reasoning>Here is my response"
            )
            mock_choice.message.tool_calls = None
            mock_response = Mock()
            mock_response.choices = [mock_choice]

            result = llm.process_response(mock_response)

            # Returns original content (includes thinking tags)
            assert (
                result
                == "<reasoning>I'm reasoning about this</reasoning>Here is my response"
            )
            # But message history should NOT have thinking tags (to avoid 500 errors)
            assert llm.messages[-1]["role"] == "assistant"
            assert "<reasoning>" not in llm.messages[-1]["content"]
            # Raw response should still have thinking tags
            assert "<reasoning>" in llm.get_last_raw_response()

    def test_validate_messages_strips_thinking_tags_defensively(self):
        """Test that _validate_messages_for_api strips thinking tags as a defensive measure.

        Even if thinking tags somehow end up in message history (edge cases, legacy data),
        the validation pass should strip them before sending to the API.
        """
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(continuous=True)

            # Simulate messages with thinking tags that somehow got into history
            # (this shouldn't happen with our fixes, but we want defensive stripping)
            llm.messages = [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
                {
                    "role": "assistant",
                    "content": "<reasoning>I'm thinking...</reasoning>Hello! How can I help?",
                },
                {"role": "user", "content": "What's 2+2?"},
                {
                    "role": "assistant",
                    "content": "<reasoning>Let me calculate...</reasoning>The answer is 4",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "calculator", "arguments": "{}"},
                        }
                    ],
                },
            ]

            # Validate messages
            validated = llm._validate_messages_for_api(llm.messages)

            # All assistant messages should have thinking tags stripped
            for msg in validated:
                if msg["role"] == "assistant":
                    assert "<reasoning>" not in msg["content"], (
                        f"Thinking tags not stripped from: {msg['content']}"
                    )
                    assert "</reasoning>" not in msg["content"]

            # Verify specific content
            assert validated[2]["content"] == "Hello! How can I help?"
            # Tool call message - either "The answer is 4" or "." (placeholder if empty after stripping)
            assert validated[4]["content"] in ["The answer is 4", "."]

    def test_extended_thinking_disabled_strips_thinking_tags_from_history(self):
        """Test that without extended thinking, thinking tags are still stripped from history."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(continuous=True)  # No extended_thinking

            mock_choice = Mock()
            mock_choice.message.content = (
                "<reasoning>Some thinking</reasoning>Final response"
            )
            mock_choice.message.tool_calls = None
            mock_response = Mock()
            mock_response.choices = [mock_choice]

            llm.process_response(mock_response)

            # Message history should NOT have thinking tags
            assert llm.messages[-1]["role"] == "assistant"
            assert "<reasoning>" not in llm.messages[-1]["content"]
            assert llm.messages[-1]["content"] == "Final response"
            # Raw response should still have thinking tags
            assert "<reasoning>" in llm.get_last_raw_response()

    def test_handle_tool_calls_disables_extended_thinking_on_400(self):
        """Test that _handle_tool_calls retries without extended_thinking on 400 validation error.

        When extended_thinking is enabled and a follow-up request after tool execution
        fails with a 400 error about thinking blocks, the code should automatically
        disable extended_thinking and retry.
        """
        from openai import BadRequestError

        with patch("floship_llm.client.OpenAI"):
            llm = LLM(
                extended_thinking={"enabled": True, "budget_tokens": 1024},
                enable_tools=True,
                continuous=True,
            )

            # Create mock message with tool calls
            mock_message = Mock()
            mock_message.content = "Let me help"
            mock_tool_call = Mock()
            mock_tool_call.id = "call_123"
            mock_tool_call.function.name = "test_tool"
            mock_tool_call.function.arguments = "{}"
            mock_message.tool_calls = [mock_tool_call]

            mock_response = Mock()
            mock_response.choices = [Mock(message=mock_message)]

            # Mock tool execution
            with patch.object(llm.tool_manager, "execute_tool") as mock_exec:
                mock_result = Mock()
                mock_result.content = "Tool result"
                mock_exec.return_value = mock_result

                # Mock the 400 error that Claude returns when thinking is enabled
                # but message history doesn't have proper thinking blocks
                error_response = Mock()
                error_response.status_code = 400
                error_400 = BadRequestError(
                    message=(
                        "Error code: 400 - messages.1.content.0.type: Expected `thinking` or "
                        "`redacted_thinking`, but found `text`. When `thinking` is enabled..."
                    ),
                    response=error_response,
                    body=None,
                )

                # First call raises 400, second call (retry) succeeds
                mock_final_choice = Mock()
                mock_final_choice.message.content = "Here is the result"
                mock_final_choice.message.tool_calls = None
                mock_final_response = Mock()
                mock_final_response.choices = [mock_final_choice]

                call_count = [0]

                def mock_api_call(*args, **kwargs):
                    call_count[0] += 1
                    if call_count[0] == 1:
                        raise error_400
                    return mock_final_response

                with patch.object(
                    llm.retry_handler, "execute_with_retry", side_effect=mock_api_call
                ):
                    result = llm._handle_tool_calls(mock_message, mock_response)

                # Should have made 2 calls (first failed, second succeeded)
                assert call_count[0] == 2

                # extended_thinking should now be disabled
                assert llm.extended_thinking is None

                # Should have gotten a response
                assert result is not None

    def test_on_assistant_message_callback_with_tool_calls(self):
        """Test on_assistant_message callback is called with tool calls."""
        with patch("floship_llm.client.OpenAI"):
            callback_calls = []

            def callback(content: str, has_tool_calls: bool):
                callback_calls.append(
                    {"content": content, "has_tool_calls": has_tool_calls}
                )

            llm = LLM(enable_tools=True, on_assistant_message=callback)

            # Create mock message with tool calls
            mock_message = Mock()
            mock_message.content = "Let me search for that"
            mock_tool_call = Mock()
            mock_tool_call.id = "call_123"
            mock_tool_call.function.name = "search"
            mock_tool_call.function.arguments = '{"query": "test"}'
            mock_message.tool_calls = [mock_tool_call]

            mock_response = Mock()
            mock_response.choices = [Mock(message=mock_message)]

            # Mock tool execution to avoid actual execution
            with patch.object(llm.tool_manager, "execute_tool") as mock_exec:
                mock_result = Mock()
                mock_result.content = "Search results"
                mock_exec.return_value = mock_result

                # Mock follow-up response (final response without tool calls)
                mock_final_choice = Mock()
                mock_final_choice.message.content = "Here are the results"
                mock_final_choice.message.tool_calls = None
                mock_final_response = Mock()
                mock_final_response.choices = [mock_final_choice]

                with patch.object(
                    llm.retry_handler,
                    "execute_with_retry",
                    return_value=mock_final_response,
                ):
                    llm._handle_tool_calls(mock_message, mock_response)

            # Callback should have been called for intermediary message
            assert len(callback_calls) >= 1
            assert callback_calls[0]["content"] == "Let me search for that"
            assert callback_calls[0]["has_tool_calls"] is True

    def test_on_assistant_message_callback_final_response(self):
        """Test on_assistant_message callback is called for final response."""
        with patch("floship_llm.client.OpenAI"):
            callback_calls = []

            def callback(content: str, has_tool_calls: bool):
                callback_calls.append(
                    {"content": content, "has_tool_calls": has_tool_calls}
                )

            llm = LLM(on_assistant_message=callback)

            mock_choice = Mock()
            mock_choice.message.content = "Here is my final answer"
            mock_choice.message.tool_calls = None
            mock_response = Mock()
            mock_response.choices = [mock_choice]

            llm.process_response(mock_response)

            # Callback should be called with final response
            assert len(callback_calls) == 1
            assert callback_calls[0]["content"] == "Here is my final answer"
            assert callback_calls[0]["has_tool_calls"] is False

    def test_on_assistant_message_callback_exception_handled(self):
        """Test that exceptions in callback are handled gracefully."""
        with patch("floship_llm.client.OpenAI"):

            def bad_callback(content: str, has_tool_calls: bool):
                raise ValueError("Callback error!")

            llm = LLM(on_assistant_message=bad_callback)

            mock_choice = Mock()
            mock_choice.message.content = "Response"
            mock_choice.message.tool_calls = None
            mock_response = Mock()
            mock_response.choices = [mock_choice]

            # Should not raise, but handle gracefully
            result = llm.process_response(mock_response)
            assert result == "Response"

    def test_process_response_with_response_tags(self):
        """Test response processing with response tags."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()

            mock_choice = Mock()
            mock_choice.message.content = (
                "Some text<response>Real response</response>More text"
            )
            mock_response = Mock()
            mock_response.choices = [mock_choice]

            result = llm.process_response(mock_response)

            # Returns original content since no response_format is set
            assert result == "Some text<response>Real response</response>More text"
            # But internal processing should extract response content
            assert llm.messages[-1]["content"] == "Real response"

    def test_process_response_max_length_exceeded(self):
        """Test response processing when max length is exceeded."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(max_length=10)

            mock_choice = Mock()
            mock_choice.message.content = (
                "This is a very long response that exceeds the limit"
            )
            mock_response = Mock()
            mock_response.choices = [mock_choice]

            with patch.object(
                llm, "retry_prompt", return_value="retry_response"
            ) as mock_retry:
                result = llm.process_response(mock_response)
                mock_retry.assert_called_once()
                assert result == "retry_response"

    def test_process_response_max_length_exceeded_retry_fails(self):
        """Test response processing when max length is exceeded and retry fails."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(max_length=10)

            mock_choice = Mock()
            mock_choice.message.content = (
                "This is a very long response that exceeds the limit"
            )
            mock_response = Mock()
            mock_response.choices = [mock_choice]

            with patch.object(llm, "retry_prompt", return_value=None) as mock_retry:
                result = llm.process_response(mock_response)
                mock_retry.assert_called_once()
                # Should return original response when retry fails
                assert result == "This is a very long response that exceeds the limit"

    def test_process_response_with_json_format(self):
        """Test response processing with JSON response format returns user's model."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(response_format=ResponseModelForTesting)

            mock_choice = Mock()
            # Mock response contains wrapped format
            mock_choice.message.content = 'Response: {"reasoning": "Processing request.", "response": {"name": "test", "value": 42}}'
            mock_response = Mock()
            mock_response.choices = [mock_choice]

            with patch(
                "floship_llm.utils.lm_json_utils.extract_strict_json"
            ) as mock_extract:
                mock_extract.return_value = '{"reasoning": "Processing request.", "response": {"name": "test", "value": 42}}'

                result = llm.process_response(mock_response)

                # User gets back their original model type (unwrapped)
                assert isinstance(result, ResponseModelForTesting)
                assert result.name == "test"
                assert result.value == 42

    @pytest.mark.parametrize(
        "model_name",
        [
            "gpt-4",
            "gpt-3.5-turbo",
            "claude-3",
            "Claude-2",
            "gemini-pro",
            "Gemini-1.5",
            "llama",
        ],
    )
    def test_model_support_properties(self, model_name):
        """Test model support properties - Heroku doesn't support penalties for any model."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
            llm.model = model_name

            # Heroku Inference API does not support frequency_penalty or presence_penalty
            # for any model, so these should always return False
            assert llm.supports_frequency_penalty is False
            assert llm.supports_presence_penalty is False

    # ========== Streaming Tests ==========

    def test_streaming_basic(self):
        """Test basic streaming functionality."""
        with patch("floship_llm.client.OpenAI") as mock_openai:
            # Mock streaming response
            mock_chunk1 = Mock()
            mock_chunk1.choices = [Mock(delta=Mock(content="Hello "))]
            mock_chunk2 = Mock()
            mock_chunk2.choices = [Mock(delta=Mock(content="world"))]
            mock_chunk3 = Mock()
            mock_chunk3.choices = [Mock(delta=Mock(content="!"))]

            mock_stream = [mock_chunk1, mock_chunk2, mock_chunk3]
            mock_client = mock_openai.return_value
            mock_client.chat.completions.create.return_value = iter(mock_stream)

            llm = LLM(stream=True, enable_tools=False)
            chunks = list(llm.prompt_stream("Test prompt"))

            assert chunks == ["Hello ", "world", "!"]
            assert "".join(chunks) == "Hello world!"

    def test_streaming_with_tools_raises_error(self):
        """Test that streaming raises error when tools are enabled."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(stream=True, enable_tools=True)

            # Add a mock tool
            llm.tool_manager.tools = [{"name": "test_tool"}]

            with pytest.raises(
                ValueError, match="Streaming mode does not support tool calls"
            ):
                list(llm.prompt_stream("Test prompt"))

    def test_streaming_adds_to_conversation_history(self):
        """Test that streaming responses are added to conversation history."""
        with patch("floship_llm.client.OpenAI") as mock_openai:
            # Mock streaming response
            mock_chunk1 = Mock()
            mock_chunk1.choices = [Mock(delta=Mock(content="Answer "))]
            mock_chunk2 = Mock()
            mock_chunk2.choices = [Mock(delta=Mock(content="here"))]

            mock_stream = [mock_chunk1, mock_chunk2]
            mock_client = mock_openai.return_value
            mock_client.chat.completions.create.return_value = iter(mock_stream)

            llm = LLM(stream=True, enable_tools=False, continuous=True)
            list(llm.prompt_stream("Question"))

            # Check conversation history
            assert len(llm.messages) == 2
            assert llm.messages[0]["role"] == "user"
            assert llm.messages[0]["content"] == "Question"
            assert llm.messages[1]["role"] == "assistant"
            assert llm.messages[1]["content"] == "Answer here"

    def test_streaming_empty_chunks(self):
        """Test streaming handles empty/None content gracefully."""
        with patch("floship_llm.client.OpenAI") as mock_openai:
            # Mock streaming response with some empty chunks
            mock_chunk1 = Mock()
            mock_chunk1.choices = [Mock(delta=Mock(content="Hello"))]
            mock_chunk2 = Mock()
            mock_chunk2.choices = [Mock(delta=Mock(content=None))]
            mock_chunk3 = Mock()
            mock_chunk3.choices = [Mock(delta=Mock(content=" world"))]

            mock_stream = [mock_chunk1, mock_chunk2, mock_chunk3]
            mock_client = mock_openai.return_value
            mock_client.chat.completions.create.return_value = iter(mock_stream)

            llm = LLM(stream=True, enable_tools=False)
            chunks = list(llm.prompt_stream("Test"))

            # Should only include non-None content
            assert chunks == ["Hello", " world"]

    def test_streaming_with_system_message(self):
        """Test streaming with system message."""
        with patch("floship_llm.client.OpenAI") as mock_openai:
            mock_chunk = Mock()
            mock_chunk.choices = [Mock(delta=Mock(content="Response"))]

            mock_stream = [mock_chunk]
            mock_client = mock_openai.return_value
            mock_client.chat.completions.create.return_value = iter(mock_stream)

            llm = LLM(stream=True, enable_tools=False)
            list(llm.prompt_stream("Test", system="You are a helpful assistant"))

            # Check that system message was added
            assert llm.messages[0]["role"] == "system"
            assert llm.messages[0]["content"] == "You are a helpful assistant"

    def test_streaming_continuous_false_resets(self):
        """Test that streaming resets conversation when continuous=False."""
        with patch("floship_llm.client.OpenAI") as mock_openai:
            mock_chunk = Mock()
            mock_chunk.choices = [Mock(delta=Mock(content="Response"))]

            mock_stream = [mock_chunk]
            mock_client = mock_openai.return_value
            mock_client.chat.completions.create.return_value = iter(mock_stream)

            llm = LLM(stream=True, enable_tools=False, continuous=False)
            list(llm.prompt_stream("Test"))

            # Should reset after streaming
            assert len(llm.messages) == 0

    def test_prompt_disables_extended_thinking_on_validation_error(self):
        """Retry without extended thinking when the API rejects thinking payload."""
        with patch("floship_llm.client.OpenAI") as mock_openai:
            call_count = 0

            def mock_create(**kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # Simulate Claude extended thinking validation failure
                    class FakeValidationError(Exception):
                        status_code = 400

                        def __str__(self):
                            return "ValidationException: Expected `thinking` block"

                    raise FakeValidationError()

                mock_chunk = Mock()
                mock_chunk.choices = [Mock(delta=Mock(content="Hello"))]
                return iter([mock_chunk])

            mock_client = mock_openai.return_value
            mock_client.chat.completions.create = mock_create

            llm = LLM(extended_thinking={"enabled": True})

            response = llm.prompt("Test message")

            assert response == "Hello"
            assert llm.extended_thinking is None
            assert call_count == 2

    # ========== Streaming Final Response After Tools Tests ==========

    def test_stream_final_response_with_tools(self):
        """Test streaming final response after tool execution."""
        with patch("floship_llm.client.OpenAI") as mock_openai:
            mock_client = mock_openai.return_value

            # First response: tool call
            mock_tool_call = Mock()
            mock_tool_call.id = "call_123"
            mock_tool_call.function.name = "test_tool"
            mock_tool_call.function.arguments = '{"arg": "value"}'

            mock_message = Mock()
            mock_message.tool_calls = [mock_tool_call]
            mock_message.content = None

            first_response = Mock()
            first_response.choices = [Mock(message=mock_message)]

            # Streaming follow-up response
            mock_chunk1 = Mock()
            mock_chunk1.choices = [Mock(delta=Mock(content="Final ", tool_calls=None))]
            mock_chunk2 = Mock()
            mock_chunk2.choices = [Mock(delta=Mock(content="answer", tool_calls=None))]

            mock_stream = iter([mock_chunk1, mock_chunk2])

            # Setup mock to return first response, then stream
            call_count = 0

            def mock_create(**kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return first_response
                else:
                    return mock_stream

            mock_client.chat.completions.create = mock_create

            # Setup tool
            llm = LLM(enable_tools=True, continuous=False)

            def test_func(arg: str):
                """Test function"""
                return "Tool result"

            llm.add_tool(
                ToolFunction(
                    name="test_tool",
                    description="Test tool",
                    parameters=[
                        ToolParameter(name="arg", type="string", description="Test arg")
                    ],
                    function=test_func,
                )
            )

            # Execute with streaming
            result = llm.prompt("Test", stream_final_response=True)

            # Should return a generator
            assert hasattr(result, "__iter__")
            assert hasattr(result, "__next__")

            # Consume the generator
            chunks = list(result)
            assert chunks == ["Final ", "answer"]
            assert "".join(chunks) == "Final answer"

    def test_stream_final_response_no_tools_returns_string(self):
        """Test that streaming is used by default when no tools are enabled."""
        with patch("floship_llm.client.OpenAI") as mock_openai:
            mock_client = mock_openai.return_value

            # Create mock streaming response (streaming is used by default when no tools)
            mock_client.chat.completions.create.return_value = (
                create_mock_stream_response("Direct answer")
            )

            llm = LLM(enable_tools=False, continuous=False)

            # Should use streaming internally and return collected string
            result = llm.prompt("Test", stream_final_response=True)

            assert isinstance(result, str)
            assert result == "Direct answer"

    def test_stream_final_response_recursive_tools_fallback(self):
        """Test that recursive tool calls fall back to non-streaming."""
        with patch("floship_llm.client.OpenAI") as mock_openai:
            mock_client = mock_openai.return_value

            # First response: tool call
            mock_tool_call1 = Mock()
            mock_tool_call1.id = "call_1"
            mock_tool_call1.function.name = "tool1"
            mock_tool_call1.function.arguments = '{"arg": "value"}'

            mock_message1 = Mock()
            mock_message1.tool_calls = [mock_tool_call1]
            mock_message1.content = None

            first_response = Mock()
            first_response.choices = [Mock(message=mock_message1)]

            # When trying to stream follow-up, detect more tool calls
            mock_tool_call2 = Mock()
            mock_tool_call2.id = "call_2"
            mock_tool_call2.function.name = "tool2"
            mock_tool_call2.function.arguments = '{"arg": "value2"}'

            # Streaming first chunk with tool calls (triggers fallback)
            mock_stream_chunk = Mock()
            mock_stream_chunk.choices = [
                Mock(delta=Mock(tool_calls=[mock_tool_call2], content=None))
            ]

            # Non-streaming response after fallback
            mock_message2 = Mock()
            mock_message2.tool_calls = [mock_tool_call2]
            mock_message2.content = None

            second_response = Mock()
            second_response.choices = [Mock(message=mock_message2)]

            # Final response (no more tools)
            mock_message3 = Mock()
            mock_message3.tool_calls = None
            mock_message3.content = "Final answer"

            third_response = Mock()
            third_response.choices = [Mock(message=mock_message3)]

            call_count = 0

            def mock_create(**kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # Initial request
                    return first_response
                elif call_count == 2:
                    # First streaming attempt
                    return iter([mock_stream_chunk])
                elif call_count == 3:
                    # After fallback, non-streaming
                    return second_response
                elif call_count == 4:
                    # Second streaming attempt
                    # Return stream with final answer (no tool calls)
                    chunk = Mock()
                    chunk.choices = [
                        Mock(delta=Mock(tool_calls=None, content="Final answer"))
                    ]
                    return iter([chunk])

            mock_client.chat.completions.create = mock_create

            # Setup tools
            llm = LLM(enable_tools=True, continuous=False)

            def tool_func(arg: str):
                return f"Result: {arg}"

            for tool_name in ["tool1", "tool2"]:
                llm.add_tool(
                    ToolFunction(
                        name=tool_name,
                        description=f"Test {tool_name}",
                        parameters=[
                            ToolParameter(
                                name="arg", type="string", description="Test arg"
                            )
                        ],
                        function=tool_func,
                    )
                )

            # Should handle recursive tools with streaming for final response
            result = llm.prompt("Test", stream_final_response=True)

            # Should return a generator (from the second streaming attempt)
            assert hasattr(result, "__iter__")
            chunks = list(result)
            assert "".join(chunks) == "Final answer"

    def test_stream_final_response_with_continuous_mode(self):
        """Test streaming final response maintains conversation history."""
        with patch("floship_llm.client.OpenAI") as mock_openai:
            mock_client = mock_openai.return_value

            # First response: tool call
            mock_tool_call = Mock()
            mock_tool_call.id = "call_123"
            mock_tool_call.function.name = "test_tool"
            mock_tool_call.function.arguments = '{"arg": "value"}'

            mock_message = Mock()
            mock_message.tool_calls = [mock_tool_call]
            mock_message.content = None

            first_response = Mock()
            first_response.choices = [Mock(message=mock_message)]

            # Streaming follow-up
            mock_chunk = Mock()
            mock_chunk.choices = [Mock(delta=Mock(content="Streamed", tool_calls=None))]

            mock_stream = iter([mock_chunk])

            call_count = 0

            def mock_create(**kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return first_response
                else:
                    return mock_stream

            mock_client.chat.completions.create = mock_create

            # Setup tool
            llm = LLM(enable_tools=True, continuous=True)

            def test_func(arg: str):
                return "Tool result"

            llm.add_tool(
                ToolFunction(
                    name="test_tool",
                    description="Test tool",
                    parameters=[
                        ToolParameter(name="arg", type="string", description="Test arg")
                    ],
                    function=test_func,
                )
            )

            # Execute with streaming
            result = llm.prompt("Test", stream_final_response=True)
            list(result)

            # Check conversation history was updated
            assert len(llm.messages) > 0
            # Last message should be the assistant's response
            assert llm.messages[-1]["role"] == "assistant"
            assert llm.messages[-1]["content"] == "Streamed"

    # ========== Chat Alias Tests ==========

    def test_chat_alias_exists(self):
        """Test that chat is an alias for prompt."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
            # chat should be the same method as prompt
            assert llm.chat == llm.prompt

    def test_chat_simple_response(self):
        """Test simple chat interaction using streaming (default behavior)."""
        with patch("floship_llm.client.OpenAI") as mock_openai:
            mock_client = mock_openai.return_value
            mock_client.chat.completions.create.return_value = (
                create_mock_stream_response("Hello! How can I help you today?")
            )

            llm = LLM(continuous=False)
            result = llm.chat("Hi there!")

            assert isinstance(result, str)
            assert result == "Hello! How can I help you today?"
            # Verify streaming was used
            mock_client.chat.completions.create.assert_called_once()
            call_kwargs = mock_client.chat.completions.create.call_args[1]
            assert call_kwargs.get("stream") is True

    def test_chat_with_tools(self):
        """Test chat with tools enabled (non-streaming mode)."""
        with patch("floship_llm.client.OpenAI") as mock_openai:
            mock_client = mock_openai.return_value

            # First call: LLM returns a tool call
            mock_tool_call = Mock()
            mock_tool_call.id = "call_123"
            mock_tool_call.function.name = "get_weather"
            mock_tool_call.function.arguments = '{"city": "London"}'

            mock_message_with_tool = Mock()
            mock_message_with_tool.content = None
            mock_message_with_tool.tool_calls = [mock_tool_call]

            mock_choice_with_tool = Mock()
            mock_choice_with_tool.message = mock_message_with_tool
            mock_choice_with_tool.finish_reason = "tool_calls"

            first_response = Mock()
            first_response.choices = [mock_choice_with_tool]

            # Second call: Final response after tool execution
            mock_final_message = Mock()
            mock_final_message.content = "The weather in London is sunny and 22°C."
            mock_final_message.tool_calls = None

            mock_final_choice = Mock()
            mock_final_choice.message = mock_final_message
            mock_final_choice.finish_reason = "stop"

            final_response = Mock()
            final_response.choices = [mock_final_choice]

            call_count = 0

            def mock_create(**kwargs):
                nonlocal call_count
                call_count += 1
                return first_response if call_count == 1 else final_response

            mock_client.chat.completions.create = mock_create

            llm = LLM(enable_tools=True, continuous=False)

            def get_weather(city: str):
                """Get weather for a city."""
                return f"Weather in {city}: Sunny, 22°C"

            llm.add_tool(
                ToolFunction(
                    name="get_weather",
                    description="Get the weather for a city",
                    parameters=[
                        ToolParameter(
                            name="city",
                            type="string",
                            description="The city to get weather for",
                            required=True,
                        )
                    ],
                    function=get_weather,
                )
            )

            result = llm.chat("What's the weather in London?")

            assert isinstance(result, str)
            assert "sunny" in result.lower() or "22" in result
            assert call_count == 2  # Two API calls: tool call + final response

    def test_chat_with_structured_response(self):
        """Test chat with structured response format (Pydantic model)."""
        with patch("floship_llm.client.OpenAI") as mock_openai:
            mock_client = mock_openai.return_value
            # Streaming returns wrapped JSON response
            wrapped_response = '{"reasoning": "Generating test record.", "response": {"name": "Alice", "value": 100}}'
            mock_client.chat.completions.create.return_value = (
                create_mock_stream_response(wrapped_response)
            )

            with patch(
                "floship_llm.utils.lm_json_utils.extract_strict_json"
            ) as mock_extract:
                mock_extract.return_value = wrapped_response

                llm = LLM(
                    response_format=ResponseModelForTesting,
                    continuous=False,
                )
                llm.messages = []  # Clear system message added during init

                result = llm.chat("Generate a test record")

                # Result should be user's original model type (unwrapped)
                assert isinstance(result, ResponseModelForTesting)
                assert result.name == "Alice"
                assert result.value == 100
                # Verify streaming was used
                mock_client.chat.completions.create.assert_called_once()
                call_kwargs = mock_client.chat.completions.create.call_args[1]
                assert call_kwargs.get("stream") is True

    def test_get_last_raw_response(self):
        """Test get_last_raw_response returns the raw LLM output."""
        with patch("floship_llm.client.OpenAI") as mock_openai:
            mock_client = mock_openai.return_value
            raw_response = '{"name": "Test", "value": 42}'
            mock_client.chat.completions.create.return_value = (
                create_mock_stream_response(raw_response)
            )

            llm = LLM(continuous=False)
            llm.prompt("Hello")

            # After a prompt, we should be able to get the raw response
            assert llm.get_last_raw_response() == raw_response

    def test_get_last_raw_response_before_prompt(self):
        """Test get_last_raw_response returns None before any prompt."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
            assert llm.get_last_raw_response() is None


class TestLLMDebugMode:
    """Tests for debug mode configuration."""

    def setup_method(self):
        """Set up test environment variables."""
        os.environ["INFERENCE_URL"] = "http://test-url.com"
        os.environ["INFERENCE_MODEL_ID"] = "test-model"
        os.environ["INFERENCE_KEY"] = "test-key"

    def test_debug_mode_kwarg(self):
        """Test that debug_mode kwarg is properly applied."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(debug_mode=True)
            assert llm.waf_config.debug_mode is True

            llm2 = LLM(debug_mode=False)
            assert llm2.waf_config.debug_mode is False

    def test_enable_waf_sanitization_kwarg(self):
        """Test that enable_waf_sanitization kwarg is properly applied."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(enable_waf_sanitization=False)
            assert llm.waf_config.enable_waf_sanitization is False

            llm2 = LLM(enable_waf_sanitization=True)
            assert llm2.waf_config.enable_waf_sanitization is True


class TestLLMMetrics:
    """Tests for LLM metrics functionality."""

    def setup_method(self):
        """Set up test environment variables."""
        os.environ["INFERENCE_URL"] = "http://test-url.com"
        os.environ["INFERENCE_MODEL_ID"] = "test-model"
        os.environ["INFERENCE_KEY"] = "test-key"

    def test_get_waf_metrics_returns_copy(self):
        """Test that get_waf_metrics returns a copy."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
            metrics = llm.get_waf_metrics()
            # Modify the copy
            metrics["total_requests"] = 999
            # Original should be unchanged
            assert llm.waf_metrics.total_requests != 999

    def test_reset_waf_metrics(self):
        """Test that reset_waf_metrics clears all counters."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
            llm.waf_metrics.total_requests = 10
            llm.waf_metrics.successful_requests = 8
            llm.waf_metrics.failed_requests = 2

            llm.reset_waf_metrics()

            metrics = llm.get_waf_metrics()
            assert metrics["total_requests"] == 0


class TestLLMEdgeCases:
    """Tests for edge cases and error handling."""

    def setup_method(self):
        """Set up test environment variables."""
        os.environ["INFERENCE_URL"] = "http://test-url.com"
        os.environ["INFERENCE_MODEL_ID"] = "test-model"
        os.environ["INFERENCE_KEY"] = "test-key"

    def test_invalid_type_raises_error(self):
        """Test that invalid type raises ValueError."""
        with patch("floship_llm.client.OpenAI"):
            with pytest.raises(
                ValueError, match="type must be 'completion' or 'embedding'"
            ):
                LLM(type="invalid")

    def test_str_representation(self):
        """Test __str__ method returns correct info."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(type="completion")
            str_repr = str(llm)
            # Check it's a string representation (may vary)
            assert isinstance(str_repr, str)
            assert len(str_repr) > 0

    def test_repr_representation(self):
        """Test __repr__ method returns correct info."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(type="completion")
            repr_str = repr(llm)
            # Check it's a string representation
            assert isinstance(repr_str, str)
            assert len(repr_str) > 0

    def test_model_property(self):
        """Test model property returns current model."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
            assert llm.model == "test-model"

    def test_get_last_tool_history_empty(self):
        """Test get_last_tool_history returns empty list initially."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
            history = llm.get_last_tool_history()
            assert history == []

    def test_get_last_tool_history_returns_copy(self):
        """Test get_last_tool_history returns a copy."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
            llm._current_tool_history = [{"tool": "test", "arguments": {}}]
            history = llm.get_last_tool_history()
            history.append({"tool": "modified"})
            # Original should be unchanged
            assert len(llm._current_tool_history) == 1

    def test_get_last_recursion_depth_initial(self):
        """Test get_last_recursion_depth returns 0 initially."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
            assert llm.get_last_recursion_depth() == 0

    def test_truncated_response_error(self):
        """Test TruncatedResponseError attributes."""
        from floship_llm.client import TruncatedResponseError

        error = TruncatedResponseError("Response was truncated", "raw response text")
        assert "Response was truncated" in str(error)
        assert error.raw_response == "raw response text"

    def test_cloudfront_waf_error_basic(self):
        """Test CloudFrontWAFError basic creation."""
        from floship_llm.client import CloudFrontWAFError

        original = Exception("Original error")
        # Detected blockers should be list of tuples (category, pattern)
        error = CloudFrontWAFError(
            message="WAF blocked",
            messages=[{"role": "user", "content": "test"}],
            detected_blockers=[("category1", "pattern1"), ("category2", "pattern2")],
            context="prompt method",
            original_error=original,
        )

        assert error.messages == [{"role": "user", "content": "test"}]
        assert len(error.detected_blockers) == 2
        assert error.context == "prompt method"
        assert error.original_error == original


class TestErrorDetection:
    """Test error detection methods for auto-recovery."""

    def setup_method(self):
        """Set up test environment variables."""
        self.env_vars = {
            "INFERENCE_URL": "https://test-api.example.com",
            "INFERENCE_MODEL_ID": "test-model",
            "INFERENCE_KEY": "test-key",
        }
        self.env_patcher = patch.dict(os.environ, self.env_vars, clear=False)
        self.env_patcher.start()

    def teardown_method(self):
        """Clean up test environment."""
        self.env_patcher.stop()

    def test_is_context_length_error_true(self):
        """Test context length error detection returns True for relevant errors."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()

            # Create actual exception classes for testing
            class MockError1(Exception):
                status_code = 400

                def __str__(self):
                    return "Error: context length exceeded maximum allowed"

            class MockError2(Exception):
                status_code = 400

                def __str__(self):
                    return "Error: too many tokens in request"

            class MockError3(Exception):
                status_code = 400

                def __str__(self):
                    return "Error: prompt is too long"

            assert llm._is_context_length_error(MockError1()) is True
            assert llm._is_context_length_error(MockError2()) is True
            assert llm._is_context_length_error(MockError3()) is True

    def test_is_context_length_error_false(self):
        """Test context length error detection returns False for other errors."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()

            # Not a 400 error
            class MockError500(Exception):
                status_code = 500

                def __str__(self):
                    return "Internal server error"

            assert llm._is_context_length_error(MockError500()) is False

            # 400 but different message
            class MockError400Other(Exception):
                status_code = 400

                def __str__(self):
                    return "Invalid parameter value"

            assert llm._is_context_length_error(MockError400Other()) is False

    def test_is_invalid_content_error_true(self):
        """Test invalid content error detection returns True for relevant errors."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()

            class MockError1(Exception):
                status_code = 400

                def __str__(self):
                    return "Error: content is required"

            class MockError2(Exception):
                status_code = 400

                def __str__(self):
                    return "Error: invalid message content"

            assert llm._is_invalid_content_error(MockError1()) is True
            assert llm._is_invalid_content_error(MockError2()) is True

    def test_is_invalid_content_error_false(self):
        """Test invalid content error detection returns False for other errors."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()

            class MockError(Exception):
                status_code = 400

                def __str__(self):
                    return "Error: rate limit exceeded"

            assert llm._is_invalid_content_error(MockError()) is False

    def test_trim_conversation_for_context(self):
        """Test conversation trimming keeps system and recent messages."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()

            llm.messages = [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Message 1"},
                {"role": "assistant", "content": "Response 1"},
                {"role": "user", "content": "Message 2"},
                {"role": "assistant", "content": "Response 2"},
                {"role": "user", "content": "Message 3"},
                {"role": "assistant", "content": "Response 3"},
                {"role": "user", "content": "Message 4"},
            ]

            result = llm._trim_conversation_for_context(keep_system=True, keep_last_n=4)

            assert result is True
            assert len(llm.messages) == 5  # system + 4 recent
            assert llm.messages[0]["role"] == "system"
            assert llm.messages[-1]["content"] == "Message 4"

    def test_trim_conversation_for_context_nothing_to_trim(self):
        """Test trimming returns False when nothing can be trimmed."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()

            llm.messages = [
                {"role": "system", "content": "System"},
                {"role": "user", "content": "Hello"},
            ]

            result = llm._trim_conversation_for_context(keep_system=True, keep_last_n=4)

            assert result is False
            assert len(llm.messages) == 2


class TestToolNameSanitization:
    """Test tool name sanitization."""

    def test_sanitize_valid_name(self):
        """Test that valid names are not changed."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
            name, changed = llm._sanitize_tool_name_for_api("valid_tool_name")
            assert name == "valid_tool_name"
            assert changed is False

    def test_sanitize_invalid_chars(self):
        """Test sanitization of invalid characters."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
            name, changed = llm._sanitize_tool_name_for_api("invalid tool name!")
            assert name == "invalid_tool_name"
            assert changed is True

    def test_sanitize_empty_name(self):
        """Test sanitization of empty name."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
            name, changed = llm._sanitize_tool_name_for_api("")
            assert name.startswith("tool_")
            assert changed is True

    def test_sanitize_none_name(self):
        """Test sanitization of None name."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
            name, changed = llm._sanitize_tool_name_for_api(None)
            assert name.startswith("tool_")
            assert changed is True


class TestResponseFormatThinkingWrapper:
    """Test cases for response_format thinking auto-wrapping."""

    def setup_method(self):
        """Set up test environment variables."""
        self.env_vars = {
            "INFERENCE_URL": "https://test-api.example.com",
            "INFERENCE_MODEL_ID": "test-model",
            "INFERENCE_KEY": "test-key",
        }
        self.env_patcher = patch.dict(os.environ, self.env_vars, clear=False)
        self.env_patcher.start()

    def teardown_method(self):
        """Clean up test environment."""
        self.env_patcher.stop()

    def test_response_format_auto_wrapped_for_plain_model(self):
        """Test that plain BaseModel response_format is auto-wrapped with thinking."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(response_format=ResponseModelForTesting)

            # Should be wrapped
            assert llm._response_format_wrapped is True
            assert llm._original_response_format == ResponseModelForTesting
            # Wrapper should have reasoning and response fields
            assert "reasoning" in llm.response_format.model_fields
            assert "response" in llm.response_format.model_fields

    def test_response_format_not_wrapped_for_thinking_model(self):
        """Test that ThinkingModel subclass is NOT wrapped."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(response_format=ThinkingResponseModelForTesting)

            # Should NOT be wrapped
            assert llm._response_format_wrapped is False
            assert llm._original_response_format == ThinkingResponseModelForTesting
            assert llm.response_format == ThinkingResponseModelForTesting

    def test_has_thinking_field_detection(self):
        """Test _has_thinking_field correctly detects reasoning field."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()

            # ThinkingModel subclass has reasoning field
            assert llm._has_thinking_field(ThinkingResponseModelForTesting) is True

            # Plain BaseModel does not
            assert llm._has_thinking_field(ResponseModelForTesting) is False

            # None/invalid returns False
            assert llm._has_thinking_field(None) is False

    def test_unwrap_thinking_response_extracts_user_model(self):
        """Test _unwrap_thinking_response returns user's original model type."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(response_format=ResponseModelForTesting)

            # Create a mock wrapped response
            wrapped = llm.response_format(
                reasoning="My reasoning here",
                response=ResponseModelForTesting(name="test", value=42),
            )

            result = llm._unwrap_thinking_response(wrapped)

            # Should return the user's original model type
            assert isinstance(result, ResponseModelForTesting)
            assert result.name == "test"
            assert result.value == 42
            # Reasoning should be stored via unified interface
            assert llm.get_last_reasoning() == "My reasoning here"

    def test_unwrap_thinking_response_returns_as_is_for_thinking_model(self):
        """Test _unwrap_thinking_response returns ThinkingModel as-is."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(response_format=ThinkingResponseModelForTesting)

            response = ThinkingResponseModelForTesting(
                reasoning="My reasoning", answer="Paris", confidence=100
            )

            result = llm._unwrap_thinking_response(response)

            # Should return the same object
            assert result is response
            assert isinstance(result, ThinkingResponseModelForTesting)

    def test_get_last_structured_thinking_returns_captured_thinking(self):
        """Test get_last_reasoning returns reasoning from wrapped response."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(response_format=ResponseModelForTesting)

            # Manually set reasoning (normally done during unwrap)
            llm._last_native_reasoning = "Test reasoning content"

            assert llm.get_last_reasoning() == "Test reasoning content"
            # Backward compat method should also work
            assert llm.get_last_structured_thinking() == "Test reasoning content"

    def test_get_last_structured_thinking_none_initially(self):
        """Test get_last_reasoning returns None before any response."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(response_format=ResponseModelForTesting)

            assert llm.get_last_reasoning() is None

    def test_wrapper_schema_has_correct_structure(self):
        """Test that the wrapper schema has reasoning first, then response."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(response_format=ResponseModelForTesting)

            schema = llm.response_format.model_json_schema()

            # Should have both required fields
            assert "reasoning" in schema["required"]
            assert "response" in schema["required"]
            # Should have properties
            assert "reasoning" in schema["properties"]
            assert "response" in schema["properties"]

    def test_prompt_with_thinking_model_returns_thinking_in_model(self):
        """Test prompt with ThinkingModel returns model with reasoning field populated."""
        with patch("floship_llm.client.OpenAI") as mock_openai:
            # ThinkingModel response is NOT wrapped, so mock direct response
            response_json = '{"reasoning": "Capital cities are well-known facts.", "answer": "Paris", "confidence": 100}'
            mock_openai.return_value.chat.completions.create.return_value = (
                create_mock_stream_response(response_json)
            )

            with patch(
                "floship_llm.utils.lm_json_utils.extract_strict_json"
            ) as mock_extract:
                mock_extract.return_value = response_json

                llm = LLM(response_format=ThinkingResponseModelForTesting)
                llm.messages = []

                result = llm.prompt("What is the capital of France?")

                # Should return ThinkingModel with reasoning populated
                assert isinstance(result, ThinkingResponseModelForTesting)
                assert result.reasoning == "Capital cities are well-known facts."
                assert result.answer == "Paris"
                assert result.confidence == 100
                # get_last_reasoning should be None (not wrapped, reasoning is in model itself)
                assert llm.get_last_reasoning() is None

    def test_extended_thinking_disabled_when_thinking_model_used(self):
        """Test extended_thinking is auto-disabled when ThinkingModel is used."""
        with patch("floship_llm.client.OpenAI"):
            # User enables both extended_thinking and ThinkingModel response_format
            llm = LLM(
                response_format=ThinkingResponseModelForTesting,
                extended_thinking={"enabled": True, "budget_tokens": 1024},
            )

            # Extended thinking should be disabled (schema thinking takes precedence)
            assert llm.extended_thinking is None
            assert llm._thinking_auto_disabled is True
            # Original should be preserved
            assert llm._original_extended_thinking == {
                "enabled": True,
                "budget_tokens": 1024,
            }

    def test_extended_thinking_disabled_when_wrapped_response_format_used(self):
        """Test extended_thinking is auto-disabled when response_format is wrapped."""
        with patch("floship_llm.client.OpenAI"):
            # User enables extended_thinking with plain model (will be wrapped)
            llm = LLM(
                response_format=ResponseModelForTesting,
                extended_thinking={"enabled": True, "budget_tokens": 1024},
            )

            # Extended thinking should be disabled (wrapper provides thinking)
            assert llm.extended_thinking is None
            assert llm._thinking_auto_disabled is True
            # Wrapper should be used
            assert llm._response_format_wrapped is True

    def test_no_redundant_thinking_with_response_format(self):
        """Test that we never have both extended_thinking and schema thinking active."""
        with patch("floship_llm.client.OpenAI"):
            # Case 1: ThinkingModel with extended_thinking
            llm1 = LLM(
                response_format=ThinkingResponseModelForTesting,
                extended_thinking={"enabled": True, "budget_tokens": 1024},
            )
            assert llm1.extended_thinking is None

            # Case 2: Plain model (wrapped) with extended_thinking
            llm2 = LLM(
                response_format=ResponseModelForTesting,
                extended_thinking={"enabled": True, "budget_tokens": 1024},
            )
            assert llm2.extended_thinking is None

            # Case 3: No response_format - extended_thinking should remain
            llm3 = LLM(extended_thinking={"enabled": True, "budget_tokens": 1024})
            assert llm3.extended_thinking == {"enabled": True, "budget_tokens": 1024}

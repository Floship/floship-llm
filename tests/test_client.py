"""Tests for the LLM client module."""

import os
from unittest.mock import Mock, patch

import pytest
from pydantic import BaseModel

from floship_llm.client import LLM
from floship_llm.schemas import ToolFunction, ToolParameter


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
            assert llm.response_format == ResponseModelForTesting
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
        """Test prompt with response format (structured output with streaming)."""
        with patch("floship_llm.client.OpenAI") as mock_openai:
            # Create mock streaming response with JSON content
            mock_openai.return_value.chat.completions.create.return_value = (
                create_mock_stream_response('{"name": "test", "value": 42}')
            )

            with patch(
                "floship_llm.utils.lm_json_utils.extract_strict_json"
            ) as mock_extract:
                mock_extract.return_value = '{"name": "test", "value": 42}'

                llm = LLM(response_format=ResponseModelForTesting)
                # Clear system message added during init
                llm.messages = []

                result = llm.prompt("Hello")

                assert isinstance(result, ResponseModelForTesting)
                assert result.name == "test"
                assert result.value == 42

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
            mock_choice.message.content = "<think>Some thinking</think>Final response"
            mock_response = Mock()
            mock_response.choices = [mock_choice]

            result = llm.process_response(mock_response)

            # Returns original content since no response_format is set
            assert result == "<think>Some thinking</think>Final response"
            # But internal processing should remove think tags
            assert llm.messages[-1]["content"] == "Final response"

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
        """Test response processing with JSON response format."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(response_format=ResponseModelForTesting)

            mock_choice = Mock()
            mock_choice.message.content = 'Response: {"name": "test", "value": 42}'
            mock_response = Mock()
            mock_response.choices = [mock_choice]

            with patch(
                "floship_llm.utils.lm_json_utils.extract_strict_json"
            ) as mock_extract:
                mock_extract.return_value = '{"name": "test", "value": 42}'

                result = llm.process_response(mock_response)

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
            # Streaming returns JSON response
            mock_client.chat.completions.create.return_value = (
                create_mock_stream_response('{"name": "Alice", "value": 100}')
            )

            with patch(
                "floship_llm.utils.lm_json_utils.extract_strict_json"
            ) as mock_extract:
                mock_extract.return_value = '{"name": "Alice", "value": 100}'

                llm = LLM(response_format=ResponseModelForTesting, continuous=False)
                llm.messages = []  # Clear system message added during init

                result = llm.chat("Generate a test record")

                # Result should be parsed as Pydantic model
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

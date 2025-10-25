"""Tests for CloudFront WAF compatibility and tool response sanitization."""

import os
from unittest.mock import Mock, patch
import pytest

from floship_llm import LLM
from floship_llm.schemas import ToolFunction, ToolParameter


class TestToolSanitization:
    """Tests for tool response sanitization."""

    def setup_method(self):
        """Set up test environment."""
        os.environ['INFERENCE_URL'] = 'http://test.com'
        os.environ['INFERENCE_MODEL_ID'] = 'test-model'
        os.environ['INFERENCE_KEY'] = 'test-key'

    def test_default_sanitization_enabled(self):
        """Test that sanitization is enabled by default."""
        with patch('floship_llm.client.OpenAI'):
            llm = LLM()
            assert llm.sanitize_tool_responses is True

    def test_sanitization_can_be_disabled(self):
        """Test that sanitization can be disabled."""
        with patch('floship_llm.client.OpenAI'):
            llm = LLM(sanitize_tool_responses=False)
            assert llm.sanitize_tool_responses is False

    def test_ellipsis_sanitization(self):
        """Test that ellipsis patterns are sanitized."""
        with patch('floship_llm.client.OpenAI'):
            llm = LLM(sanitize_tool_responses=True)
            
            content = "Error occurred\n...\nStack trace"
            sanitized = llm._sanitize_tool_response(content)
            
            assert "..." not in sanitized
            assert "[truncated]" in sanitized

    def test_custom_sanitization_patterns(self):
        """Test custom sanitization patterns."""
        with patch('floship_llm.client.OpenAI'):
            llm = LLM(
                sanitize_tool_responses=True,
                sanitization_patterns={
                    '...': '[more]',
                    '../': '[parent_dir]'
                }
            )
            
            content = "Path: ../../../etc\nOutput: ..."
            sanitized = llm._sanitize_tool_response(content)
            
            assert "../" not in sanitized
            assert "..." not in sanitized
            assert "[more]" in sanitized
            assert "[parent_dir]" in sanitized

    def test_sanitization_disabled_preserves_content(self):
        """Test that disabled sanitization preserves content."""
        with patch('floship_llm.client.OpenAI'):
            llm = LLM(sanitize_tool_responses=False)
            
            content = "Error occurred\n...\nStack trace"
            sanitized = llm._sanitize_tool_response(content)
            
            assert sanitized == content

    def test_empty_content_handling(self):
        """Test that empty content is handled gracefully."""
        with patch('floship_llm.client.OpenAI'):
            llm = LLM(sanitize_tool_responses=True)
            
            assert llm._sanitize_tool_response("") == ""
            assert llm._sanitize_tool_response(None) is None


class TestTokenManagement:
    """Tests for token estimation and truncation."""

    def setup_method(self):
        """Set up test environment."""
        os.environ['INFERENCE_URL'] = 'http://test.com'
        os.environ['INFERENCE_MODEL_ID'] = 'test-model'
        os.environ['INFERENCE_KEY'] = 'test-key'

    def test_default_max_tokens(self):
        """Test that default max tokens is set correctly."""
        with patch('floship_llm.client.OpenAI'):
            llm = LLM()
            assert llm.max_tool_response_tokens == 4000

    def test_custom_max_tokens(self):
        """Test custom max tokens configuration."""
        with patch('floship_llm.client.OpenAI'):
            llm = LLM(max_tool_response_tokens=2000)
            assert llm.max_tool_response_tokens == 2000

    def test_short_content_not_truncated(self):
        """Test that short content is not truncated."""
        with patch('floship_llm.client.OpenAI'):
            llm = LLM(max_tool_response_tokens=1000)
            
            content = "Short response"
            truncated, was_truncated = llm._truncate_tool_response(content)
            
            assert truncated == content
            assert was_truncated is False

    def test_long_content_truncated(self):
        """Test that long content is truncated."""
        with patch('floship_llm.client.OpenAI'):
            llm = LLM(max_tool_response_tokens=100)
            
            # Create content that exceeds 100 tokens (~400 characters)
            content = "x" * 500
            truncated, was_truncated = llm._truncate_tool_response(content)
            
            assert len(truncated) < len(content)
            assert was_truncated is True
            assert "[truncated]" in truncated


class TestToolResponseProcessing:
    """Tests for integrated tool response processing."""

    def setup_method(self):
        """Set up test environment."""
        os.environ['INFERENCE_URL'] = 'http://test.com'
        os.environ['INFERENCE_MODEL_ID'] = 'test-model'
        os.environ['INFERENCE_KEY'] = 'test-key'

    def test_process_tool_response_basic(self):
        """Test basic tool response processing."""
        with patch('floship_llm.client.OpenAI'):
            llm = LLM()
            
            result = llm._process_tool_response("Simple content", "test_tool")
            
            assert 'content' in result
            assert result['content'] == "Simple content"

    def test_process_tool_response_with_sanitization(self):
        """Test tool response processing with sanitization."""
        with patch('floship_llm.client.OpenAI'):
            llm = LLM(sanitize_tool_responses=True)
            
            content = "Error: ...\nStack trace"
            result = llm._process_tool_response(content, "test_tool")
            
            assert "..." not in result['content']
            assert "[truncated]" in result['content']

    def test_process_tool_response_with_metadata(self):
        """Test tool response processing with metadata tracking."""
        with patch('floship_llm.client.OpenAI'):
            llm = LLM(track_tool_metadata=True)
            
            content = "Test response with ..."
            result = llm._process_tool_response(content, "test_tool")
            
            assert 'metadata' in result
            assert 'tool_name' in result['metadata']
            assert result['metadata']['tool_name'] == "test_tool"
            assert 'sanitization_applied' in result['metadata']
            assert result['metadata']['sanitization_applied'] is True

    def test_process_tool_response_no_metadata(self):
        """Test that metadata is not included when tracking is disabled."""
        with patch('floship_llm.client.OpenAI'):
            llm = LLM(track_tool_metadata=False)
            
            result = llm._process_tool_response("Test content", "test_tool")
            
            assert 'metadata' not in result


class TestCloudFrontCompatibility:
    """Tests for CloudFront WAF compatibility."""

    def setup_method(self):
        """Set up test environment."""
        os.environ['INFERENCE_URL'] = 'http://test.com'
        os.environ['INFERENCE_MODEL_ID'] = 'test-model'
        os.environ['INFERENCE_KEY'] = 'test-key'

    def test_tool_response_with_ellipsis_pattern(self):
        """Test that tool responses with ellipsis don't cause issues."""
        with patch('floship_llm.client.OpenAI') as mock_openai:
            llm = LLM(enable_tools=True, sanitize_tool_responses=True)

            def mock_tool() -> str:
                return "Error:\nValueError\n...\n(10 frames omitted)"

            tool = ToolFunction(
                name="test_tool",
                description="Test tool",
                parameters=[],
                function=mock_tool
            )
            llm.add_tool(tool)

            # Mock tool call response
            mock_tool_call = Mock()
            mock_tool_call.id = "call_123"
            mock_tool_call.function.name = "test_tool"
            mock_tool_call.function.arguments = "{}"

            mock_message = Mock()
            mock_message.content = "Using tool"
            mock_message.tool_calls = [mock_tool_call]

            mock_choice = Mock()
            mock_choice.message = mock_message

            mock_response = Mock()
            mock_response.choices = [mock_choice]

            # Mock follow-up response
            mock_follow_up_message = Mock()
            mock_follow_up_message.content = "Tool executed"
            mock_follow_up_message.tool_calls = None

            mock_follow_up_choice = Mock()
            mock_follow_up_choice.message = mock_follow_up_message

            mock_follow_up_response = Mock()
            mock_follow_up_response.choices = [mock_follow_up_choice]

            mock_openai.return_value.chat.completions.create.return_value = mock_follow_up_response

            # Process response
            result = llm.process_response(mock_response)

            # Verify ellipsis was sanitized in the tool message
            tool_messages = [m for m in llm.messages if m.get('role') == 'tool']
            assert len(tool_messages) == 1
            assert "..." not in tool_messages[0]['content']
            assert "[truncated]" in tool_messages[0]['content']

    def test_403_error_detection(self):
        """Test enhanced 403 error detection and logging."""
        with patch('floship_llm.client.OpenAI') as mock_openai:
            from openai import APIStatusError
            
            llm = LLM()
            
            # Create a mock 403 error from CloudFront
            mock_response = Mock()
            mock_response.status_code = 403
            
            error = APIStatusError(
                message="Request blocked by CloudFront",
                response=mock_response,
                body={"error": "The request could not be satisfied"}
            )
            
            mock_openai.return_value.chat.completions.create.side_effect = error
            
            # Try to make a call
            with pytest.raises(APIStatusError):
                llm.prompt("Test prompt")


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility."""

    def setup_method(self):
        """Set up test environment."""
        os.environ['INFERENCE_URL'] = 'http://test.com'
        os.environ['INFERENCE_MODEL_ID'] = 'test-model'
        os.environ['INFERENCE_KEY'] = 'test-key'

    def test_default_configuration_works(self):
        """Test that default configuration works without changes."""
        with patch('floship_llm.client.OpenAI'):
            llm = LLM()
            
            # Verify new features have sensible defaults
            assert llm.sanitize_tool_responses is True
            assert llm.max_tool_response_tokens == 4000
            assert llm.track_tool_metadata is False

    def test_all_features_can_be_disabled(self):
        """Test that all new features can be disabled for backward compatibility."""
        with patch('floship_llm.client.OpenAI'):
            llm = LLM(
                sanitize_tool_responses=False,
                track_tool_metadata=False
            )
            
            assert llm.sanitize_tool_responses is False
            assert llm.track_tool_metadata is False

    def test_existing_tool_execution_still_works(self):
        """Test that existing tool execution patterns still work."""
        with patch('floship_llm.client.OpenAI') as mock_openai:
            llm = LLM(enable_tools=True, sanitize_tool_responses=False)

            def simple_tool() -> str:
                return "Simple result"

            tool = ToolFunction(
                name="simple_tool",
                description="Simple tool",
                parameters=[],
                function=simple_tool
            )
            llm.add_tool(tool)

            # This should work exactly as before
            assert "simple_tool" in llm.list_tools()


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_estimate_tokens(self):
        """Test token estimation."""
        from floship_llm.utils import estimate_tokens
        
        # Empty string
        assert estimate_tokens("") == 0
        
        # Simple text (~4 chars per token)
        assert estimate_tokens("test") == 1
        assert estimate_tokens("test test test test") == 4
        
        # Longer text
        long_text = "x" * 400
        assert estimate_tokens(long_text) == 100

    def test_truncate_to_tokens(self):
        """Test token-based truncation."""
        from floship_llm.utils import truncate_to_tokens
        
        # Short text not truncated
        short_text = "Short"
        result = truncate_to_tokens(short_text, 100)
        assert result == short_text
        
        # Long text truncated
        long_text = "x" * 500
        result = truncate_to_tokens(long_text, 50)
        assert len(result) < len(long_text)
        assert "[truncated]" in result

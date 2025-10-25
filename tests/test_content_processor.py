"""Tests for content_processor module."""

from unittest.mock import patch
import pytest

from floship_llm.content_processor import ContentProcessor


class TestContentProcessor:
    """Test cases for ContentProcessor class."""
    
    def test_init_default_values(self):
        """Test initialization with default values."""
        processor = ContentProcessor()
        assert processor.sanitize_enabled is True
        assert '...' in processor.sanitization_patterns
        assert processor.max_tokens == 4000
        assert processor.track_metadata is False
    
    def test_init_custom_values(self):
        """Test initialization with custom values."""
        patterns = {'bad': 'good'}
        processor = ContentProcessor(
            sanitize_enabled=False,
            sanitization_patterns=patterns,
            max_tokens=2000,
            track_metadata=True
        )
        assert processor.sanitize_enabled is False
        assert processor.sanitization_patterns == patterns
        assert processor.max_tokens == 2000
        assert processor.track_metadata is True
    
    def test_sanitize_content_enabled(self):
        """Test content sanitization when enabled."""
        processor = ContentProcessor(sanitize_enabled=True)
        content = "Here is some code... more code"
        
        result = processor.sanitize_content(content)
        
        assert '...' not in result
        assert '[truncated]' in result
    
    def test_sanitize_content_disabled(self):
        """Test content sanitization when disabled."""
        processor = ContentProcessor(sanitize_enabled=False)
        content = "Here is some code... more code"
        
        result = processor.sanitize_content(content)
        
        assert result == content
    
    def test_sanitize_content_custom_patterns(self):
        """Test sanitization with custom patterns."""
        patterns = {'bad_word': '[REDACTED]', 'secret': '***'}
        processor = ContentProcessor(sanitization_patterns=patterns)
        content = "This contains bad_word and secret data"
        
        result = processor.sanitize_content(content)
        
        assert 'bad_word' not in result
        assert '[REDACTED]' in result
        assert 'secret' not in result
        assert '***' in result
    
    def test_sanitize_content_empty(self):
        """Test sanitization with empty content."""
        processor = ContentProcessor()
        
        assert processor.sanitize_content("") == ""
        assert processor.sanitize_content(None) is None
    
    def test_truncate_content_below_limit(self):
        """Test truncation when content is below limit."""
        processor = ContentProcessor(max_tokens=100)
        content = "Short content"
        
        result, was_truncated = processor.truncate_content(content)
        
        assert result == content
        assert was_truncated is False
    
    def test_truncate_content_above_limit(self):
        """Test truncation when content exceeds limit."""
        processor = ContentProcessor(max_tokens=10)
        content = "This is a very long piece of content " * 50
        
        result, was_truncated = processor.truncate_content(content)
        
        assert len(result) < len(content)
        assert was_truncated is True
        assert '[Content truncated]' in result or '...' in result or result.endswith('...')
    
    def test_truncate_content_empty(self):
        """Test truncation with empty content."""
        processor = ContentProcessor()
        
        result, was_truncated = processor.truncate_content("")
        assert result == ""
        assert was_truncated is False
        
        result, was_truncated = processor.truncate_content(None)
        assert result is None
        assert was_truncated is False
    
    def test_process_tool_response_basic(self):
        """Test basic tool response processing."""
        processor = ContentProcessor(sanitize_enabled=False, track_metadata=False)
        content = "Tool response"
        
        result = processor.process_tool_response(content, "test_tool")
        
        assert 'content' in result
        assert result['content'] == content
        assert 'metadata' not in result
    
    def test_process_tool_response_with_sanitization(self):
        """Test tool response processing with sanitization."""
        processor = ContentProcessor(sanitize_enabled=True, track_metadata=False)
        content = "Tool response with ... pattern"
        
        result = processor.process_tool_response(content, "test_tool")
        
        assert '...' not in result['content']
        assert '[truncated]' in result['content']
    
    def test_process_tool_response_with_truncation(self):
        """Test tool response processing with truncation."""
        processor = ContentProcessor(max_tokens=5, track_metadata=False)
        content = "This is a very long tool response that needs truncation"
        
        result = processor.process_tool_response(content, "test_tool")
        
        assert len(result['content']) < len(content)
    
    def test_process_tool_response_with_metadata(self):
        """Test tool response processing with metadata tracking."""
        processor = ContentProcessor(track_metadata=True)
        content = "Tool response"
        
        result = processor.process_tool_response(content, "test_tool", execution_time=0.5)
        
        assert 'metadata' in result
        assert result['metadata']['tool_name'] == "test_tool"
        assert 'was_sanitized' in result['metadata']
        assert 'was_truncated' in result['metadata']
        assert 'original_tokens' in result['metadata']
        assert 'final_tokens' in result['metadata']
        assert 'execution_time_ms' in result['metadata']
        assert result['metadata']['execution_time_ms'] == 500
    
    def test_sanitize_tool_content_none(self):
        """Test sanitize_tool_content with None."""
        processor = ContentProcessor()
        
        result = processor.sanitize_tool_content(None, "test_tool", is_error=False)
        
        assert "no return value" in result
        assert "test_tool" in result
    
    def test_sanitize_tool_content_empty(self):
        """Test sanitize_tool_content with empty string."""
        processor = ContentProcessor()
        
        result = processor.sanitize_tool_content("", "test_tool", is_error=False)
        
        assert "empty result" in result
        assert "test_tool" in result
    
    def test_sanitize_tool_content_null_like(self):
        """Test sanitize_tool_content with null-like strings."""
        processor = ContentProcessor()
        
        for null_str in ["None", "null", "NULL", "Null"]:
            result = processor.sanitize_tool_content(null_str, "test_tool", is_error=False)
            assert "null result" in result.lower()
            assert "test_tool" in result
    
    def test_sanitize_tool_content_error(self):
        """Test sanitize_tool_content with error flag."""
        processor = ContentProcessor()
        
        result = processor.sanitize_tool_content(None, "test_tool", is_error=True)
        
        assert "Error executing tool" in result
        assert "test_tool" in result
    
    def test_sanitize_tool_content_valid(self):
        """Test sanitize_tool_content with valid content."""
        processor = ContentProcessor()
        content = "Valid tool response with ... pattern"
        
        result = processor.sanitize_tool_content(content, "test_tool")
        
        assert '[truncated]' in result  # Sanitized
        assert 'Valid tool response' in result
    
    def test_sanitize_messages(self):
        """Test message sanitization."""
        processor = ContentProcessor()
        content = "Hello    world   with   spaces"
        
        result = processor.sanitize_messages(content)
        
        assert result == "Hello world with spaces"
    
    def test_sanitize_messages_empty(self):
        """Test message sanitization with empty content."""
        processor = ContentProcessor()
        
        assert processor.sanitize_messages("") == ""
        assert processor.sanitize_messages(None) is None
    
    def test_sanitize_messages_newlines(self):
        """Test message sanitization with newlines."""
        processor = ContentProcessor()
        content = "Line1\n  \nLine2\t\tLine3"
        
        result = processor.sanitize_messages(content)
        
        # All whitespace should be compacted to single spaces
        assert result == "Line1 Line2 Line3"

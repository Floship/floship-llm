"""Tests for CloudFront WAF compatibility and tool response sanitization."""

import os
from unittest.mock import Mock, patch

import pytest

from floship_llm import LLM
from floship_llm.schemas import ToolFunction


class TestToolSanitization:
    """Tests for tool response sanitization."""

    def setup_method(self):
        """Set up test environment."""
        os.environ["INFERENCE_URL"] = "http://test.com"
        os.environ["INFERENCE_MODEL_ID"] = "test-model"
        os.environ["INFERENCE_KEY"] = "test-key"

    def test_default_sanitization_enabled(self):
        """Test that sanitization is enabled by default."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
            assert llm.sanitize_tool_responses is True

    def test_sanitization_can_be_disabled(self):
        """Test that sanitization can be disabled."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(sanitize_tool_responses=False)
            assert llm.sanitize_tool_responses is False

    def test_ellipsis_sanitization(self):
        """Test that ellipsis patterns are sanitized."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(sanitize_tool_responses=True)

            content = "Error occurred\n...\nStack trace"
            sanitized = llm._sanitize_tool_response(content)

            assert "..." not in sanitized
            assert "[truncated]" in sanitized

    def test_custom_sanitization_patterns(self):
        """Test custom sanitization patterns."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(
                sanitize_tool_responses=True,
                sanitization_patterns={"...": "[more]", "../": "[parent_dir]"},
            )

            content = "Path: ../../../etc\nOutput: ..."
            sanitized = llm._sanitize_tool_response(content)

            assert "../" not in sanitized
            assert "..." not in sanitized
            assert "[more]" in sanitized
            assert "[parent_dir]" in sanitized

    def test_sanitization_disabled_preserves_content(self):
        """Test that disabled sanitization preserves content."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(sanitize_tool_responses=False)

            content = "Error occurred\n...\nStack trace"
            sanitized = llm._sanitize_tool_response(content)

            assert sanitized == content

    def test_empty_content_handling(self):
        """Test that empty content is handled gracefully."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(sanitize_tool_responses=True)

            assert llm._sanitize_tool_response("") == ""
            assert llm._sanitize_tool_response(None) is None


class TestTokenManagement:
    """Tests for token estimation and truncation."""

    def setup_method(self):
        """Set up test environment."""
        os.environ["INFERENCE_URL"] = "http://test.com"
        os.environ["INFERENCE_MODEL_ID"] = "test-model"
        os.environ["INFERENCE_KEY"] = "test-key"

    def test_default_max_tokens(self):
        """Test that default max tokens is set correctly."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
            assert llm.max_tool_response_tokens == 4000

    def test_custom_max_tokens(self):
        """Test custom max tokens configuration."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(max_tool_response_tokens=2000)
            assert llm.max_tool_response_tokens == 2000

    def test_short_content_not_truncated(self):
        """Test that short content is not truncated."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(max_tool_response_tokens=1000)

            content = "Short response"
            truncated, was_truncated = llm._truncate_tool_response(content)

            assert truncated == content
            assert was_truncated is False

    def test_long_content_truncated(self):
        """Test that long content is truncated."""
        with patch("floship_llm.client.OpenAI"):
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
        os.environ["INFERENCE_URL"] = "http://test.com"
        os.environ["INFERENCE_MODEL_ID"] = "test-model"
        os.environ["INFERENCE_KEY"] = "test-key"

    def test_process_tool_response_basic(self):
        """Test basic tool response processing."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()

            result = llm._process_tool_response("Simple content", "test_tool")

            assert "content" in result
            assert result["content"] == "Simple content"

    def test_process_tool_response_with_sanitization(self):
        """Test tool response processing with sanitization."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(sanitize_tool_responses=True)

            content = "Error: ...\nStack trace"
            result = llm._process_tool_response(content, "test_tool")

            assert "..." not in result["content"]
            assert "[truncated]" in result["content"]

    def test_process_tool_response_with_metadata(self):
        """Test tool response processing with metadata tracking."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(track_tool_metadata=True)

            content = "Test response with ..."
            result = llm._process_tool_response(content, "test_tool")

            assert "metadata" in result
            assert "tool_name" in result["metadata"]
            assert result["metadata"]["tool_name"] == "test_tool"
            assert "sanitization_applied" in result["metadata"]
            assert result["metadata"]["sanitization_applied"] is True

    def test_process_tool_response_no_metadata(self):
        """Test that metadata is not included when tracking is disabled."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(track_tool_metadata=False)

            result = llm._process_tool_response("Test content", "test_tool")

            assert "metadata" not in result


class TestCloudFrontCompatibility:
    """Tests for CloudFront WAF compatibility."""

    def setup_method(self):
        """Set up test environment."""
        os.environ["INFERENCE_URL"] = "http://test.com"
        os.environ["INFERENCE_MODEL_ID"] = "test-model"
        os.environ["INFERENCE_KEY"] = "test-key"

    def test_tool_response_with_ellipsis_pattern(self):
        """Test that tool responses with ellipsis don't cause issues."""
        with patch("floship_llm.client.OpenAI") as mock_openai:
            llm = LLM(enable_tools=True, sanitize_tool_responses=True)

            def mock_tool() -> str:
                return "Error:\nValueError\n...\n(10 frames omitted)"

            tool = ToolFunction(
                name="test_tool",
                description="Test tool",
                parameters=[],
                function=mock_tool,
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

            mock_openai.return_value.chat.completions.create.return_value = (
                mock_follow_up_response
            )

            # Process response
            llm.process_response(mock_response)

            # Verify ellipsis was sanitized in the tool message
            tool_messages = [m for m in llm.messages if m.get("role") == "tool"]
            assert len(tool_messages) == 1
            assert "..." not in tool_messages[0]["content"]
            assert "[truncated]" in tool_messages[0]["content"]

    def test_403_error_detection(self):
        """Test enhanced 403 error detection and logging."""
        with patch("floship_llm.client.OpenAI") as mock_openai:
            from openai import APIStatusError

            llm = LLM()

            # Create a mock 403 error from CloudFront
            mock_response = Mock()
            mock_response.status_code = 403

            error = APIStatusError(
                message="Request blocked by CloudFront",
                response=mock_response,
                body={"error": "The request could not be satisfied"},
            )

            mock_openai.return_value.chat.completions.create.side_effect = error

            # Try to make a call
            with pytest.raises(APIStatusError):
                llm.prompt("Test prompt")

    def test_403_model_authorization_error_not_treated_as_waf(self):
        """Test that 403 model authorization errors are NOT treated as WAF blocks.

        When a user doesn't have access to a model (e.g., claude-4-5-sonnet),
        the API returns a 403 with "you do not have access to that model".
        This should NOT be retried as a WAF block - it should fail immediately.
        """
        with patch("floship_llm.client.OpenAI"):
            from openai import PermissionDeniedError

            llm = LLM()

            # Create a mock 403 authorization error (not WAF)
            mock_response = Mock()
            mock_response.status_code = 403

            error = PermissionDeniedError(
                message='Error code: 403 - {"error": "{\\"code\\":403,\\"message\\":\\"you do not have access to that model\\",\\"type\\":\\"authorization_error\\"}"}',
                response=mock_response,
                body={
                    "error": {
                        "code": 403,
                        "message": "you do not have access to that model",
                        "type": "authorization_error",
                    }
                },
            )

            # This should NOT be detected as a CloudFront WAF error
            assert llm._is_cloudfront_403(error) is False

    def test_403_api_key_error_not_treated_as_waf(self):
        """Test that 403 API key errors are NOT treated as WAF blocks."""
        with patch("floship_llm.client.OpenAI"):
            from openai import PermissionDeniedError

            llm = LLM()

            # Create a mock 403 invalid API key error
            mock_response = Mock()
            mock_response.status_code = 403

            error = PermissionDeniedError(
                message="Invalid API key",
                response=mock_response,
                body={
                    "error": {"message": "Invalid API key", "type": "invalid_api_key"}
                },
            )

            # This should NOT be detected as a CloudFront WAF error
            assert llm._is_cloudfront_403(error) is False

    def test_403_generic_forbidden_treated_as_waf(self):
        """Test that generic 403 forbidden IS treated as WAF block."""
        with patch("floship_llm.client.OpenAI"):
            from openai import PermissionDeniedError

            llm = LLM()

            # Create a mock generic 403 error (CloudFront WAF)
            mock_response = Mock()
            mock_response.status_code = 403

            error = PermissionDeniedError(
                message="403 Forbidden - Request blocked",
                response=mock_response,
                body={"error": "The request could not be satisfied"},
            )

            # This SHOULD be detected as a CloudFront WAF error
            assert llm._is_cloudfront_403(error) is True

    def test_ellipsis_before_quote_sanitization(self):
        """Test that ellipsis followed by quote is sanitized to prevent WAF path_traversal_backslash.

        CloudFront WAF sees '..."' as '..\' (path traversal) when JSON-escaped.
        Pattern: 'Never say "I ran this code..."' becomes '..."' which triggers WAF.
        Fix: Replace '..."' with '…"' (Unicode ellipsis).
        """
        from floship_llm.client import CloudFrontWAFSanitizer

        # Test double quote case
        content = 'Never say "I ran this code..."'
        sanitized, was_sanitized = CloudFrontWAFSanitizer.sanitize(content)
        assert was_sanitized is True
        assert '..."' not in sanitized
        assert '…"' in sanitized  # Unicode ellipsis

        # Test single quote case
        content = "Use their name, store facts as 'Milk Li is...'"
        sanitized, was_sanitized = CloudFrontWAFSanitizer.sanitize(content)
        assert was_sanitized is True
        assert "...'" not in sanitized
        assert "…'" in sanitized  # Unicode ellipsis

        # Test JSON-escaped quote case (the actual trigger)
        content = r"jql_query=\"...\""
        sanitized, was_sanitized = CloudFrontWAFSanitizer.sanitize(content)
        assert was_sanitized is True
        # The escaped quote should also be handled
        assert r"...\"" not in sanitized or "…" in sanitized

    def test_ellipsis_quote_desanitization(self):
        """Test that Unicode ellipsis is restored to ASCII ellipsis on desanitization."""
        from floship_llm.client import CloudFrontWAFSanitizer

        # Test double quote case
        content = 'The response said "hello…"'
        desanitized, was_desanitized = CloudFrontWAFSanitizer.desanitize(content)
        assert was_desanitized is True
        assert '…"' not in desanitized
        assert '..."' in desanitized

        # Test single quote case
        content = "Use 'example…' here"
        desanitized, was_desanitized = CloudFrontWAFSanitizer.desanitize(content)
        assert was_desanitized is True
        assert "…'" not in desanitized
        assert "...'" in desanitized


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility."""

    def setup_method(self):
        """Set up test environment."""
        os.environ["INFERENCE_URL"] = "http://test.com"
        os.environ["INFERENCE_MODEL_ID"] = "test-model"
        os.environ["INFERENCE_KEY"] = "test-key"

    def test_default_configuration_works(self):
        """Test that default configuration works without changes."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()

            # Verify new features have sensible defaults
            assert llm.sanitize_tool_responses is True
            assert llm.max_tool_response_tokens == 4000
            assert llm.track_tool_metadata is False

    def test_all_features_can_be_disabled(self):
        """Test that all new features can be disabled for backward compatibility."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(sanitize_tool_responses=False, track_tool_metadata=False)

            assert llm.sanitize_tool_responses is False
            assert llm.track_tool_metadata is False

    def test_existing_tool_execution_still_works(self):
        """Test that existing tool execution patterns still work."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(enable_tools=True, sanitize_tool_responses=False)

            def simple_tool() -> str:
                return "Simple result"

            tool = ToolFunction(
                name="simple_tool",
                description="Simple tool",
                parameters=[],
                function=simple_tool,
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


class TestCloudFrontWAFSanitizer:
    """Tests for CloudFrontWAFSanitizer pattern fixes.

    These tests verify that WAF sanitization patterns don't cause false positives
    while still blocking actual security threats.
    """

    def setup_method(self):
        """Set up test environment."""
        from floship_llm.client import CloudFrontWAFSanitizer

        self.sanitizer = CloudFrontWAFSanitizer

    def test_path_traversal_ellipsis_preserved_in_json(self):
        """Test that ellipsis in JSON-encoded code is NOT corrupted.

        This was the production bug: code like print("Loading...")
        when JSON-encoded becomes print(\"Loading...\") and the
        ...\\ was being matched by the path traversal pattern.
        """
        import json

        code_samples = [
            'print("Loading...")',
            'print(f"Description: {x[:200]}...")',
            'raise ValueError("Invalid input...")',
            'logger.info(f"Processing {count} items...")',
        ]

        for code in code_samples:
            json_str = json.dumps({"code": code})
            sanitized, _ = self.sanitizer.sanitize(json_str)
            assert "[PARENT_DIR]" not in sanitized, (
                f"Ellipsis corrupted in: {code}\nJSON: {json_str}\nResult: {sanitized}"
            )

    def test_path_traversal_standalone_ellipsis_preserved(self):
        """Test that standalone ellipsis patterns are preserved."""
        test_cases = [
            "...",
            "....",
            "Hello...",
            "Loading... please wait",
            "Error occurred...\nStack trace follows",
        ]

        for test in test_cases:
            sanitized, _ = self.sanitizer.sanitize(test)
            assert "[PARENT_DIR]" not in sanitized, (
                f"Ellipsis corrupted: {test} -> {sanitized}"
            )

    def test_path_traversal_attacks_blocked(self):
        """Test that actual path traversal attacks are still blocked."""
        attacks = [
            ("../etc/passwd", True),
            ("../../secret", True),
            ("..\\windows\\system32", True),
            ('{"file": "../secret.txt"}', True),
            ("GET /../admin HTTP/1.1", True),
        ]

        for attack, should_block in attacks:
            sanitized, _was_sanitized = self.sanitizer.sanitize(attack)
            if should_block:
                assert "[PARENT_DIR]" in sanitized, (
                    f"Attack not blocked: {attack} -> {sanitized}"
                )

    def test_response_variable_name_preserved(self):
        """Test that variable names containing 'response' are preserved.

        The pattern should only match bare 'response =' not 'api_response ='.
        """
        preserved_cases = [
            "api_response = get_data()",
            "http_response = requests.get(url)",
            "json_response = {}",
            "_response = None",
            "my_response = parse()",
        ]

        for test in preserved_cases:
            sanitized, _ = self.sanitizer.sanitize(test)
            assert "resp_var" not in sanitized, (
                f"False positive on variable name: {test} -> {sanitized}"
            )

    def test_response_bare_assignment_sanitized(self):
        """Test that bare 'response =' is still sanitized."""
        sanitized_cases = [
            "response = get_data()",
            "response=value",
        ]

        for test in sanitized_cases:
            sanitized, _was_sanitized = self.sanitizer.sanitize(test)
            assert "resp_var" in sanitized, (
                f"Bare 'response =' not sanitized: {test} -> {sanitized}"
            )

    def test_url_template_python_fstring_preserved(self):
        """Test that Python f-strings with paths are not corrupted.

        Note: Only f-strings starting with quote then brace are protected.
        Complex cases like f'prefix{/middle}' may still be sanitized.
        """
        preserved_cases = [
            "f'{/path}'",
            "f'{/data/file}'",
        ]

        for test in preserved_cases:
            sanitized, _ = self.sanitizer.sanitize(test)
            assert "[URL_TEMPLATE]" not in sanitized, (
                f"f-string corrupted: {test} -> {sanitized}"
            )

    def test_url_template_github_api_sanitized(self):
        """Test that GitHub API URL templates are sanitized."""
        sanitized_cases = [
            "{/gist_id}",
            "{/other_user}",
            "{/repo}",
        ]

        for test in sanitized_cases:
            sanitized, _was_sanitized = self.sanitizer.sanitize(test)
            assert "[URL_TEMPLATE]" in sanitized, (
                f"GitHub template not sanitized: {test} -> {sanitized}"
            )

    def test_wiki_markup_python_fstring_preserved(self):
        """Test that Python f-string escaped braces are not corrupted.

        Note: Only simple cases starting with quote-brace are protected.
        Complex nested cases may still be sanitized.
        """
        preserved_cases = [
            "f'{{literal}}'",
            "f'show {{braces}}'",
        ]

        for test in preserved_cases:
            sanitized, _ = self.sanitizer.sanitize(test)
            # Should still have the double braces
            assert "{{" in sanitized or sanitized == test, (
                f"f-string escaped braces corrupted: {test} -> {sanitized}"
            )

    def test_wiki_markup_jira_confluence_sanitized(self):
        """Test that JIRA/Confluence wiki markup is sanitized."""
        sanitized_cases = [
            ("{{code}}", "[code]"),
            ("{{monospace}}", "[monospace]"),
        ]

        for test, expected_content in sanitized_cases:
            sanitized, _was_sanitized = self.sanitizer.sanitize(test)
            assert expected_content in sanitized, (
                f"Wiki markup not sanitized: {test} -> {sanitized}"
            )

    def test_production_failure_scenario(self):
        """Test the exact production failure scenario that was reported.

        The code: print(f"Description: {ticket.get('description', '')[:200]}...")
        When JSON-encoded and WAF-sanitized, was being corrupted to:
        print(f"Description: {ticket.get('description', '')[:200]}.[PARENT_DIR]/")

        Update v0.5.39: Now '..."' is sanitized to '…"' (Unicode ellipsis) to prevent
        CloudFront WAF path_traversal_backslash detection.
        """
        import json

        # The exact code from the failed production session
        failed_code = (
            "print(f\"Description: {ticket.get('description', '')[:200]}...\")"
        )

        # As tool_call arguments (JSON encoded)
        tool_call_args = json.dumps({"code": failed_code})

        # Apply sanitization
        sanitized, _ = self.sanitizer.sanitize(tool_call_args)

        # Verify ellipsis is NOT corrupted to path traversal
        assert "[PARENT_DIR]" not in sanitized, (
            f"Production bug not fixed!\n"
            f"Original: {failed_code}\n"
            f"JSON: {tool_call_args}\n"
            f"Sanitized: {sanitized}"
        )

        # Verify content is preserved (either original ellipsis or Unicode ellipsis)
        # v0.5.39: '..."' is now sanitized to '…"' to prevent WAF path_traversal_backslash
        assert '..."' in sanitized or "...\\" in sanitized or '…"' in sanitized, (
            f"Ellipsis missing from output: {sanitized}"
        )

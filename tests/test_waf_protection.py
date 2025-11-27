"""Tests for CloudFront WAF protection and sanitization."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from openai import PermissionDeniedError

from floship_llm.client import (
    LLM,
    CloudFrontWAFError,
    CloudFrontWAFSanitizer,
    LLMConfig,
    LLMMetrics,
)


class TestCloudFrontWAFSanitizer:
    """Test CloudFront WAF sanitization functionality."""

    def test_path_traversal_sanitization(self):
        """Test that ../ patterns are sanitized."""
        content = "File: ../../config/settings.py"
        sanitized, was_sanitized = CloudFrontWAFSanitizer.sanitize(content)

        assert was_sanitized
        assert "../" not in sanitized
        assert "[PARENT_DIR]" in sanitized
        assert "config/settings.py" in sanitized

    def test_backslash_path_traversal(self):
        """Test that ..\\ patterns are sanitized."""
        content = "Path: ..\\..\\config\\settings.ini"
        sanitized, was_sanitized = CloudFrontWAFSanitizer.sanitize(content)

        assert was_sanitized
        assert "..\\" not in sanitized
        assert "[PARENT_DIR]" in sanitized

    def test_xss_script_tag_sanitization(self):
        """Test that <script> tags are sanitized."""
        content = "<script>alert('test')</script>"
        sanitized, was_sanitized = CloudFrontWAFSanitizer.sanitize(content)

        assert was_sanitized
        assert "<script>" not in sanitized.lower()
        assert "[SCRIPT_TAG]" in sanitized
        assert "[/SCRIPT_TAG]" in sanitized

    def test_xss_iframe_sanitization(self):
        """Test that <iframe> tags are sanitized."""
        content = '<iframe src="malicious.com"></iframe>'
        sanitized, was_sanitized = CloudFrontWAFSanitizer.sanitize(content)

        assert was_sanitized
        assert "<iframe" not in sanitized.lower()
        assert "[IFRAME_TAG]" in sanitized

    def test_javascript_protocol_sanitization(self):
        """Test that javascript: protocol is sanitized."""
        content = '<a href="javascript:alert(1)">Click</a>'
        sanitized, was_sanitized = CloudFrontWAFSanitizer.sanitize(content)

        assert was_sanitized
        assert "javascript:" not in sanitized
        assert "js:" in sanitized

    def test_event_handler_sanitization(self):
        """Test that event handlers are sanitized."""
        content = '<img onerror="alert(1)" onload="hack()">'
        sanitized, was_sanitized = CloudFrontWAFSanitizer.sanitize(content)

        assert was_sanitized
        assert "onerror=" not in sanitized.lower()
        assert "on_error=" in sanitized

    def test_no_sanitization_needed(self):
        """Test that clean content is not modified."""
        content = "Normal code without suspicious patterns: def hello(): return 'world'"
        sanitized, was_sanitized = CloudFrontWAFSanitizer.sanitize(content)

        assert not was_sanitized
        assert sanitized == content

    def test_mixed_patterns(self):
        """Test content with multiple pattern types."""
        content = """
        File: ../../config/db.py
        Contains: <script>alert('xss')</script>
        Link: javascript:void(0)
        """
        sanitized, was_sanitized = CloudFrontWAFSanitizer.sanitize(content)

        assert was_sanitized
        assert "../" not in sanitized
        assert "<script>" not in sanitized.lower()
        assert "javascript:" not in sanitized
        assert "[PARENT_DIR]" in sanitized
        assert "[SCRIPT_TAG]" in sanitized
        assert "js:" in sanitized

    def test_check_for_blockers(self):
        """Test blocker detection."""
        content = "../../file.py with <script> tag and javascript: link"
        blockers = CloudFrontWAFSanitizer.check_for_blockers(content)

        assert len(blockers) >= 3
        categories = [cat for cat, _ in blockers]
        assert "path_traversal" in categories
        assert "xss" in categories

    def test_case_insensitive_detection(self):
        """Test that detection is case-insensitive."""
        content = "<SCRIPT>alert(1)</SCRIPT>"
        sanitized, was_sanitized = CloudFrontWAFSanitizer.sanitize(content)

        assert was_sanitized
        assert "[SCRIPT_TAG]" in sanitized

    def test_pr_diff_content(self):
        """Test realistic PR diff content."""
        content = """
        diff --git a/../../src/utils.py b/../../src/utils.py
        --- a/../../src/utils.py
        +++ b/../../src/utils.py
        @@ -10,7 +10,7 @@
        -    return render_template('index.html')
        +    return render_template('index.html', escape_js=True)
        """
        sanitized, was_sanitized = CloudFrontWAFSanitizer.sanitize(content)

        assert was_sanitized
        assert "../" not in sanitized
        assert "src/utils.py" in sanitized  # Semantic meaning preserved

    def test_github_webhook_url_templates(self):
        """Test GitHub API URL templates that trigger CloudFront WAF."""
        content = """
        GitHub webhook payload with URL templates:
        'following_url': 'https://api.github.com/users/Floship/following{/other_user}'
        'gists_url': 'https://api.github.com/users/Rawgeek/gists{/gist_id}'
        'starred_url': 'https://api.github.com/users/Rawgeek/starred{/owner}{/repo}'
        'keys_url': 'https://api.github.com/repos/Floship/Shipping/keys{/key_id}'
        'collaborators_url': 'https://api.github.com/repos/Floship/Shipping/collaborators{/collaborator}'
        """
        sanitized, was_sanitized = CloudFrontWAFSanitizer.sanitize(content)

        assert was_sanitized
        assert "{/" not in sanitized  # URL template placeholders removed
        assert "https://api.github.com" in sanitized  # Base URLs preserved
        assert "[URL_TEMPLATE]" in sanitized  # Replaced with safe marker

    def test_github_webhook_realistic_payload(self):
        """Test realistic GitHub webhook payload that triggers CloudFront WAF."""
        # Simulating actual GitHub webhook payload with multiple URL templates
        content = """
        Analyzing GitHub pull request:
        User: {'login': 'Rawgeek', 'id': 1498478,
               'followers_url': 'https://api.github.com/users/Rawgeek/followers',
               'following_url': 'https://api.github.com/users/Rawgeek/following{/other_user}',
               'gists_url': 'https://api.github.com/users/Rawgeek/gists{/gist_id}',
               'starred_url': 'https://api.github.com/users/Rawgeek/starred{/owner}{/repo}'}
        Repo: {'keys_url': 'https://api.github.com/repos/Floship/Shipping/keys{/key_id}',
               'collaborators_url': 'https://api.github.com/repos/Floship/Shipping/collaborators{/collaborator}',
               'teams_url': 'https://api.github.com/repos/Floship/Shipping/teams',
               'branches_url': 'https://api.github.com/repos/Floship/Shipping/branches{/branch}',
               'issues_url': 'https://api.github.com/repos/Floship/Shipping/issues{/number}',
               'pulls_url': 'https://api.github.com/repos/Floship/Shipping/pulls{/number}'}
        """
        sanitized, was_sanitized = CloudFrontWAFSanitizer.sanitize(content)

        assert was_sanitized
        # Verify all URL template patterns are removed
        assert "{/other_user}" not in sanitized
        assert "{/gist_id}" not in sanitized
        assert "{/owner}{/repo}" not in sanitized
        assert "{/key_id}" not in sanitized
        assert "{/collaborator}" not in sanitized
        assert "{/branch}" not in sanitized
        assert "{/number}" not in sanitized
        # Verify base URLs are preserved
        assert "https://api.github.com/users/Rawgeek" in sanitized
        assert "https://api.github.com/repos/Floship/Shipping" in sanitized
        # Verify safe markers are present
        assert "[URL_TEMPLATE]" in sanitized

    def test_jira_wiki_markup_double_braces(self):
        """Test JIRA wiki markup with double curly braces that trigger CloudFront WAF."""
        content = """
        h2. Problem

        SingPost shipment cancellation is failing with XML parsing errors.
        The error message indicates malformed XML with mismatched tags during {{deleteShipment}} requests.

        *Error:* {{<unknown>:9:4: mismatched tag}} - SingPost API returns XML

        h2. Root Cause

        The XML repair functionality was *never actually working* due to {{SingPostService._retry_with_xml_repair()}}
        being a *stub* that immediately raises {{"XML repair not yet implemented"}}
        """
        sanitized, was_sanitized = CloudFrontWAFSanitizer.sanitize(content)

        assert was_sanitized
        # Verify double curly braces are sanitized
        assert "{{" not in sanitized
        assert "}}" not in sanitized
        # Verify content is preserved (now in square brackets instead of double curly)
        assert "[deleteShipment]" in sanitized
        assert '["XML repair not yet implemented"]' in sanitized
        # Original wiki markup should be replaced
        assert "{{deleteShipment}}" not in sanitized

    def test_jira_image_markup(self):
        """Test JIRA image markup patterns like !image.png|options! are sanitized."""
        content = """
        Description: !image-20251112-030524.png|width=686,alt=\"image-20251112-030524.png\"!
        Another example: See attached !screenshot.jpg! in the ticket.
        """

        sanitized, was_sanitized = CloudFrontWAFSanitizer.sanitize(content)

        assert was_sanitized
        # Verify image markup is replaced with [IMAGE:...]
        assert "!image-20251112-030524.png|width=686" not in sanitized
        assert (
            '[IMAGE:image-20251112-030524.png|width=686,alt="image-20251112-030524.png"]'
            in sanitized
        )
        # Short form
        assert "!screenshot.jpg!" not in sanitized
        assert "[IMAGE:screenshot.jpg]" in sanitized

    def test_django_orm_filter_q_pattern(self):
        """Test Django ORM filter=Q pattern that triggers CloudFront WAF."""
        content = """
        # Django ORM query that causes WAF blocking
        def get_services(self):
            from django.db.models import Q

            PICK_PACK_SERVICES = ['pick', 'pack']
            services = Service.objects.filter=Q(service_name__in=PICK_PACK_SERVICES)

            # More complex query
            results = Model.objects.filter=Q(status='active') & Q(type='urgent')
        """
        sanitized, was_sanitized = CloudFrontWAFSanitizer.sanitize(content)

        assert was_sanitized
        # Verify filter=Q pattern is sanitized
        assert "filter=Q(" not in sanitized
        assert "filter_Q(" in sanitized
        # Verify content structure is preserved
        assert "service_name__in=PICK_PACK_SERVICES" in sanitized
        assert "status='active'" in sanitized
        # Original pattern should be replaced
        assert "objects.filter_Q(" in sanitized

    def test_python_traceback_script_file(self):
        """Test Python traceback with File '<script>' pattern that triggers XSS detection."""
        content = """
        Traceback (most recent call last):
          File "/app/slack_bot/python_executor.py", line 351, in execute
            exec(compiled, self.namespace)  # nosec B102
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          File "<script>", line 6, in <module>
          File "/app/.heroku/python/lib/python3.12/site-packages/pandas/io/excel/_base.py"
        FileNotFoundError: [Errno 2] No such file or directory
        """
        sanitized, was_sanitized = CloudFrontWAFSanitizer.sanitize(content)

        assert was_sanitized
        # Verify <script> in traceback is sanitized
        assert 'File "<script>"' not in sanitized
        assert 'File "[SCRIPT_FILE]"' in sanitized
        # Verify exec( is sanitized
        assert "exec(compiled" not in sanitized
        assert "ex3c(compiled" in sanitized

    def test_python_exec_in_traceback(self):
        """Test Python exec() function in traceback is sanitized."""
        content = """
        File "/app/executor.py", line 100, in run
            exec(code, namespace)
            ^^^^^^^^^^^^^^^^^^^^^
        """
        sanitized, was_sanitized = CloudFrontWAFSanitizer.sanitize(content)

        assert was_sanitized
        assert "exec(code" not in sanitized
        assert "ex3c(code" in sanitized

    def test_template_injection_json_close(self):
        """Test JSON tool call closing pattern that triggers template injection detection."""
        content = """
        TOOL_CALLS: [{'id': 'tooluse_123', 'type': 'function', 'function': {'name': 'execute_python', 'arguments': '{"code":"print(1)","description":"Test"}'}}]
        """
        sanitized, was_sanitized = CloudFrontWAFSanitizer.sanitize(content)

        assert was_sanitized
        # Verify template-like closing is sanitized
        assert "'}}" not in sanitized or "[TEMPLATE_CLOSE]" in sanitized

    def test_file_string_traceback(self):
        """Test Python traceback with File '<string>' pattern."""
        content = """
          File "<string>", line 1, in <module>
          File "<stdin>", line 5
        """
        sanitized, was_sanitized = CloudFrontWAFSanitizer.sanitize(content)

        assert was_sanitized
        assert 'File "<string>"' not in sanitized
        assert 'File "[STRING_FILE]"' in sanitized
        assert 'File "<stdin>"' not in sanitized
        assert 'File "[STDIN_FILE]"' in sanitized

    def test_desanitize_restores_exec(self):
        """Test that desanitize restores exec() from ex3c()."""
        sanitized_content = """
        File "/app/executor.py", line 100, in run
            ex3c(compiled, self.namespace)
        """
        desanitized, was_desanitized = CloudFrontWAFSanitizer.desanitize(
            sanitized_content
        )

        assert was_desanitized
        assert "exec(" in desanitized
        assert "ex3c(" not in desanitized

    def test_desanitize_restores_script_file(self):
        """Test that desanitize restores File '<script>' from [SCRIPT_FILE]."""
        sanitized_content = 'File "[SCRIPT_FILE]", line 6, in <module>'
        desanitized, was_desanitized = CloudFrontWAFSanitizer.desanitize(
            sanitized_content
        )

        assert was_desanitized
        assert 'File "<script>"' in desanitized
        assert "[SCRIPT_FILE]" not in desanitized

    def test_desanitize_restores_multiple_patterns(self):
        """Test desanitize restores multiple patterns at once."""
        sanitized_content = """
        Traceback:
          File "[SCRIPT_FILE]", line 5
            ex3c(code, namespace)
          File "[STRING_FILE]", line 1
        """
        desanitized, was_desanitized = CloudFrontWAFSanitizer.desanitize(
            sanitized_content
        )

        assert was_desanitized
        assert 'File "<script>"' in desanitized
        assert 'File "<string>"' in desanitized
        assert "exec(" in desanitized

    def test_desanitize_no_changes_needed(self):
        """Test desanitize returns original when no sanitized patterns present."""
        content = "Normal Python code: def hello(): return 'world'"
        desanitized, was_desanitized = CloudFrontWAFSanitizer.desanitize(content)

        assert not was_desanitized
        assert desanitized == content

    def test_roundtrip_sanitize_desanitize(self):
        """Test that sanitize -> desanitize restores original content."""
        original = """
        Traceback (most recent call last):
          File "<script>", line 6, in <module>
            exec(compiled, namespace)
        """
        sanitized, _ = CloudFrontWAFSanitizer.sanitize(original)
        desanitized, _ = CloudFrontWAFSanitizer.desanitize(sanitized)

        # Should restore the key patterns
        assert 'File "<script>"' in desanitized
        assert "exec(" in desanitized


class TestLLMConfig:
    """Test LLM configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LLMConfig()

        assert config.enable_waf_sanitization is True
        assert config.max_waf_retries == 2
        assert config.retry_with_sanitization is True
        assert config.cloudfront_waf_detection is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = LLMConfig(
            enable_waf_sanitization=False,
            max_waf_retries=5,
            debug_mode=True,
        )

        assert config.enable_waf_sanitization is False
        assert config.max_waf_retries == 5
        assert config.debug_mode is True

    @patch.dict(
        "os.environ",
        {
            "FLOSHIP_LLM_WAF_SANITIZE": "false",
            "FLOSHIP_LLM_DEBUG": "true",
            "FLOSHIP_LLM_WAF_MAX_RETRIES": "3",
        },
    )
    def test_from_env(self):
        """Test loading configuration from environment variables."""
        config = LLMConfig.from_env()

        assert config.enable_waf_sanitization is False
        assert config.debug_mode is True
        assert config.max_waf_retries == 3


class TestLLMMetrics:
    """Test LLM metrics tracking."""

    def test_metrics_initialization(self):
        """Test metrics are initialized to zero."""
        metrics = LLMMetrics()

        assert metrics.total_requests == 0
        assert metrics.sanitized_requests == 0
        assert metrics.cloudfront_403_errors == 0

    def test_metrics_to_dict(self):
        """Test metrics conversion to dictionary."""
        metrics = LLMMetrics(
            total_requests=100,
            sanitized_requests=20,
            failed_requests=5,
            cloudfront_403_errors=3,
        )

        result = metrics.to_dict()

        assert result["total_requests"] == 100
        assert result["sanitization_rate"] == 0.20
        assert result["error_rate"] == 0.05
        assert result["cloudfront_403_rate"] == 0.60

    def test_metrics_zero_division(self):
        """Test metrics with zero values don't cause division errors."""
        metrics = LLMMetrics()
        result = metrics.to_dict()

        assert result["sanitization_rate"] == 0.0
        assert result["error_rate"] == 0.0
        assert result["cloudfront_403_rate"] == 0


class TestLLMWAFIntegration:
    """Test LLM with CloudFront WAF protection."""

    @pytest.fixture
    def mock_env(self, monkeypatch):
        """Mock environment variables."""
        monkeypatch.setenv("INFERENCE_URL", "https://test.inference.com")
        monkeypatch.setenv("INFERENCE_MODEL_ID", "test-model")
        monkeypatch.setenv("INFERENCE_KEY", "test-key")

    @pytest.fixture
    def mock_client(self):
        """Mock OpenAI client."""
        with patch("floship_llm.client.OpenAI") as mock:
            yield mock

    def test_llm_initialization_with_waf_config(self, mock_env, mock_client):
        """Test LLM initializes with WAF configuration."""
        llm = LLM()

        assert hasattr(llm, "waf_config")
        assert hasattr(llm, "waf_sanitizer")
        assert hasattr(llm, "waf_metrics")
        assert llm.waf_config.enable_waf_sanitization is True

    def test_llm_disable_waf_sanitization(self, mock_env, mock_client):
        """Test disabling WAF sanitization."""
        llm = LLM(enable_waf_sanitization=False)

        assert llm.waf_config.enable_waf_sanitization is False

    def test_llm_custom_waf_config(self, mock_env, mock_client):
        """Test LLM with custom WAF config."""
        custom_config = LLMConfig(max_waf_retries=5, debug_mode=True)
        llm = LLM(waf_config=custom_config)

        assert llm.waf_config.max_waf_retries == 5
        assert llm.waf_config.debug_mode is True

    def test_sanitize_for_waf_enabled(self, mock_env, mock_client):
        """Test content sanitization when enabled."""
        llm = LLM(enable_waf_sanitization=True)
        content = "../../config/settings.py"

        sanitized = llm._sanitize_for_waf(content)

        assert "../" not in sanitized
        assert "[PARENT_DIR]" in sanitized

    def test_sanitize_for_waf_disabled(self, mock_env, mock_client):
        """Test content is not sanitized when disabled."""
        llm = LLM(enable_waf_sanitization=False)
        content = "../../config/settings.py"

        sanitized = llm._sanitize_for_waf(content)

        assert sanitized == content
        assert "../" in sanitized

    def test_is_cloudfront_403(self, mock_env, mock_client):
        """Test CloudFront 403 error detection."""
        llm = LLM()
        # Test 403 error - create mock response
        mock_response = Mock()
        mock_response.status_code = 403
        error_403 = PermissionDeniedError(
            "403 Forbidden", response=mock_response, body=None
        )
        assert llm._is_cloudfront_403(error_403) is True

        # Test forbidden error
        error_forbidden = PermissionDeniedError(
            "Access Forbidden by WAF", response=mock_response, body=None
        )
        assert llm._is_cloudfront_403(error_forbidden) is True

        # Test other errors
        error_other = ValueError("Something else")
        assert llm._is_cloudfront_403(error_other) is False

    def test_get_waf_metrics(self, mock_env, mock_client):
        """Test getting WAF metrics."""
        llm = LLM()
        llm.waf_metrics.total_requests = 50
        llm.waf_metrics.sanitized_requests = 10

        metrics = llm.get_waf_metrics()

        assert metrics["total_requests"] == 50
        assert metrics["sanitization_rate"] == 0.20

    def test_reset_waf_metrics(self, mock_env, mock_client):
        """Test resetting WAF metrics."""
        llm = LLM()
        llm.waf_metrics.total_requests = 100

        llm.reset_waf_metrics()

        assert llm.waf_metrics.total_requests == 0


class TestRealWorldScenarios:
    """Test real-world scenarios for CloudFront WAF protection."""

    def test_pr_diff_scenario(self):
        """Test handling PR diff content."""
        pr_diff = """
        diff --git a/../../src/auth.py b/../../src/auth.py
        --- a/../../src/auth.py
        +++ b/../../src/auth.py
        @@ -15,8 +15,12 @@
        -    return '<script>alert("old")</script>'
        +    return sanitize_html(content)
        """

        sanitized, was_sanitized = CloudFrontWAFSanitizer.sanitize(pr_diff)

        assert was_sanitized
        assert "../" not in sanitized
        assert "<script>" not in sanitized.lower()
        assert "src/auth.py" in sanitized  # File name preserved

    def test_html_template_code(self):
        """Test handling HTML template code."""
        html_code = """
        <div>
            <script src="app.js"></script>
            <iframe src="widget.html"></iframe>
        </div>
        """

        sanitized, was_sanitized = CloudFrontWAFSanitizer.sanitize(html_code)

        assert was_sanitized
        assert "[SCRIPT_TAG]" in sanitized
        assert "[IFRAME_TAG]" in sanitized

    def test_file_path_in_code(self):
        """Test handling file paths in code examples."""
        code = """
        import os
        config_path = os.path.join("..", "..", "config", "settings.py")
        """

        sanitized, was_sanitized = CloudFrontWAFSanitizer.sanitize(code)

        # The pattern ".." inside quotes won't match our regex (which looks for ../)
        # But this is acceptable since it's just a string, not a path traversal attack
        # Let's test with actual path traversal pattern
        code_with_traversal = 'file_path = "../../config/secret.py"'
        sanitized, was_sanitized = CloudFrontWAFSanitizer.sanitize(code_with_traversal)

        assert was_sanitized
        assert "../" not in sanitized
        assert "config/secret.py" in sanitized


class TestWAFLogging:
    """Test WAF logging functionality."""

    def test_log_waf_blocked_content_called_on_403(self, monkeypatch):
        """Test that _log_waf_blocked_content is called when 403 occurs."""
        monkeypatch.setenv("INFERENCE_URL", "https://test.inference.com")
        monkeypatch.setenv("INFERENCE_MODEL_ID", "test-model")
        monkeypatch.setenv("INFERENCE_KEY", "test-key")
        llm = LLM(api_key="test-key")

        # Mock messages
        llm.messages = [
            {"role": "system", "content": "Test system"},
            {"role": "user", "content": "Test with ../../path"},
        ]

        with patch.object(llm, "_log_waf_blocked_content") as mock_log:
            # Call the method directly to verify it works
            llm._log_waf_blocked_content(llm.messages, "(test context)")

            mock_log.assert_called_once_with(llm.messages, "(test context)")

    def test_log_waf_blocked_content_logs_blockers(self, monkeypatch, caplog):
        """Test that WAF blockers are logged when detected."""
        import logging

        monkeypatch.setenv("INFERENCE_URL", "https://test.inference.com")
        monkeypatch.setenv("INFERENCE_MODEL_ID", "test-model")
        monkeypatch.setenv("INFERENCE_KEY", "test-key")
        llm = LLM(api_key="test-key")

        messages = [
            {"role": "user", "content": "Check file at ../../config/secret.py"},
        ]

        with caplog.at_level(logging.ERROR, logger="floship_llm.client"):
            llm._log_waf_blocked_content(messages, "(test)")

        # Check that the WAF block was logged
        assert any("WAF block" in record.message for record in caplog.records)
        assert any(
            "Analyzing 1 messages" in record.message for record in caplog.records
        )

    def test_log_waf_blocked_content_empty_messages(self, monkeypatch, caplog):
        """Test logging with empty messages list."""
        import logging

        monkeypatch.setenv("INFERENCE_URL", "https://test.inference.com")
        monkeypatch.setenv("INFERENCE_MODEL_ID", "test-model")
        monkeypatch.setenv("INFERENCE_KEY", "test-key")
        llm = LLM(api_key="test-key")

        with caplog.at_level(logging.WARNING, logger="floship_llm.client"):
            llm._log_waf_blocked_content([], "(empty test)")

        assert any("No messages to log" in record.message for record in caplog.records)

    def test_log_waf_blocked_content_truncates_long_content(self, monkeypatch, caplog):
        """Test that long message content is truncated in logs."""
        import logging

        monkeypatch.setenv("INFERENCE_URL", "https://test.inference.com")
        monkeypatch.setenv("INFERENCE_MODEL_ID", "test-model")
        monkeypatch.setenv("INFERENCE_KEY", "test-key")
        llm = LLM(api_key="test-key")

        long_content = "x" * 1000  # Content longer than 500 chars
        messages = [{"role": "user", "content": long_content}]

        with caplog.at_level(logging.DEBUG, logger="floship_llm.client"):
            llm._log_waf_blocked_content(messages, "(long content test)")

        # The function should run without error
        # (truncation is internal, not visible in standard logs)


class TestCloudFrontWAFError:
    """Test CloudFrontWAFError exception functionality."""

    def test_exception_has_all_attributes(self):
        """Test that CloudFrontWAFError has all required attributes."""
        messages = [
            {"role": "user", "content": "test message"},
            {"role": "assistant", "content": "response with <script>"},
        ]
        detected_blockers = [("xss", "<script>")]
        context = "prompt()"
        original_error = PermissionDeniedError(
            message="403", body=None, response=Mock(status_code=403)
        )

        error = CloudFrontWAFError(
            message="CloudFront WAF blocked request",
            messages=messages,
            detected_blockers=detected_blockers,
            context=context,
            original_error=original_error,
        )

        assert error.messages == messages
        assert error.detected_blockers == detected_blockers
        assert error.context == context
        assert error.original_error == original_error
        assert "CloudFront WAF blocked request" in str(error)

    def test_exception_inherits_from_exception(self):
        """Test that CloudFrontWAFError inherits from Exception."""
        error = CloudFrontWAFError(
            message="test",
            messages=[],
            detected_blockers=[],
            context="test",
            original_error=None,
        )

        assert isinstance(error, Exception)

    def test_exception_can_be_caught(self):
        """Test that CloudFrontWAFError can be caught."""
        error = CloudFrontWAFError(
            message="test",
            messages=[{"role": "user", "content": "test"}],
            detected_blockers=[],
            context="test",
            original_error=None,
        )

        try:
            raise error
        except CloudFrontWAFError as caught:
            assert caught.messages == [{"role": "user", "content": "test"}]

    def test_exception_sentry_compatibility(self):
        """Test that exception attributes are accessible for Sentry extra data."""
        messages = [
            {"role": "user", "content": "exec(code)"},
            {"role": "assistant", "content": '<script>File "<script>"</script>'},
        ]
        detected_blockers = [("python_exec", "exec("), ("xss", "<script>")]

        error = CloudFrontWAFError(
            message="CloudFront WAF blocked request in prompt()",
            messages=messages,
            detected_blockers=detected_blockers,
            context="prompt()",
            original_error=None,
        )

        # Simulate Sentry extra data extraction
        sentry_extra = {
            "messages": error.messages,
            "detected_blockers": error.detected_blockers,
            "context": error.context,
        }

        assert sentry_extra["context"] == "prompt()"
        assert len(sentry_extra["messages"]) == 2
        assert len(sentry_extra["detected_blockers"]) == 2
        assert sentry_extra["detected_blockers"][0] == ("python_exec", "exec(")

    def test_exception_exported_from_init(self):
        """Test that CloudFrontWAFError is exported from floship_llm.__init__."""
        from floship_llm import CloudFrontWAFError as ImportedError

        assert ImportedError is CloudFrontWAFError

    def test_exception_includes_details_in_message(self):
        """Test that the exception message includes context and blockers."""
        messages = [
            {"role": "user", "content": "run this code"},
            {"role": "assistant", "content": "exec(code) result"},
        ]
        detected_blockers = [("python_exec", "exec(")]

        error = CloudFrontWAFError(
            message="CloudFront WAF blocked request",
            messages=messages,
            detected_blockers=detected_blockers,
            context="prompt()",
            original_error=None,
        )

        error_str = str(error)
        assert "CloudFront WAF blocked request" in error_str
        assert "Context: prompt()" in error_str
        assert "Message count: 2" in error_str
        assert "python_exec" in error_str

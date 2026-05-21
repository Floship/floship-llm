"""Tests for multimodal (vision/image) message support."""

import os
from unittest.mock import MagicMock, patch

from floship_llm.client import LLM


class TestMultimodal:
    """Test multimodal content handling in messages."""

    def setup_method(self):
        self.env_vars = {
            "INFERENCE_URL": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "INFERENCE_MODEL_ID": "gemini-3.5-flash",
            "INFERENCE_KEY": "test-key",
        }
        self.env_patcher = patch.dict(os.environ, self.env_vars, clear=False)
        self.env_patcher.start()

    def teardown_method(self):
        self.env_patcher.stop()

    def _make_llm(self, **kwargs):
        with patch("floship_llm.client.OpenAI"):
            return LLM(**kwargs)

    # -- add_message --

    def test_add_message_accepts_list_content(self):
        """Multimodal content (list of parts) is stored as-is."""
        llm = self._make_llm()
        content = [
            {"type": "text", "text": "Describe this image."},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
        ]
        llm.add_message("user", content)
        assert llm.messages[-1]["content"] is content

    def test_add_message_string_still_works(self):
        """String content is still sanitized and stored normally."""
        llm = self._make_llm()
        llm.add_message("user", "hello  world")
        assert llm.messages[-1]["content"] == "hello world"

    def test_add_message_deduplicates_multimodal(self):
        """Duplicate multimodal messages are skipped."""
        llm = self._make_llm()
        content = [{"type": "text", "text": "hi"}]
        llm.add_message("user", content)
        llm.add_message("user", content)
        user_msgs = [m for m in llm.messages if m["role"] == "user"]
        assert len(user_msgs) == 1

    # -- _validate_messages_for_api --

    def test_validate_preserves_multimodal_content(self):
        """List content passes through validation without stringification."""
        llm = self._make_llm()
        content = [
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]
        llm.add_message("user", content)
        validated = llm._validate_messages_for_api(llm.messages)
        user_msg = next(m for m in validated if m["role"] == "user")
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][0]["type"] == "text"
        assert user_msg["content"][1]["type"] == "image_url"

    def test_validate_still_stringifies_plain_content(self):
        """Normal string content is still validated as before."""
        llm = self._make_llm()
        llm.add_message("user", "hello")
        validated = llm._validate_messages_for_api(llm.messages)
        user_msg = next(m for m in validated if m["role"] == "user")
        assert isinstance(user_msg["content"], str)

    # -- WAF sanitization --

    def test_sanitize_multimodal_for_waf_sanitizes_text_parts(self):
        """WAF sanitization touches text parts but preserves image_url parts."""
        llm = self._make_llm(enable_waf_sanitization=True)
        content = [
            {"type": "text", "text": "normal text"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
        ]
        result = llm._sanitize_multimodal_for_waf(content)
        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert isinstance(result[0]["text"], str)
        # image_url part unchanged
        assert result[1] is content[1]

    def test_validate_multimodal_with_waf_enabled(self):
        """Multimodal content is WAF-sanitized (text parts only) during validation."""
        llm = self._make_llm(enable_waf_sanitization=True)
        content = [
            {"type": "text", "text": "describe"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/"}},
        ]
        llm.add_message("user", content)
        validated = llm._validate_messages_for_api(llm.messages)
        user_msg = next(m for m in validated if m["role"] == "user")
        assert isinstance(user_msg["content"], list)
        assert len(user_msg["content"]) == 2

    # -- prompt() integration --

    def test_prompt_accepts_multimodal_content(self):
        """prompt() can receive list content and adds it as a user message."""
        llm = self._make_llm()
        content = [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
        ]
        # Mock the actual API call
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "A cat"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )
        llm.client.chat.completions.create.return_value = mock_response

        result = llm.prompt(content, force_no_stream=True)
        assert result == "A cat"
        # Verify the user message was stored as multimodal
        user_msgs = [m for m in llm.messages if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert isinstance(user_msgs[0]["content"], list)

    def test_prompt_multimodal_with_system(self):
        """prompt() with multimodal content and system message."""
        llm = self._make_llm()
        content = [
            {"type": "text", "text": "Extract text from this."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
        ]
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello World"
        mock_response.choices[0].message.tool_calls = None
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )
        llm.client.chat.completions.create.return_value = mock_response

        result = llm.prompt(
            content, system="You are an OCR assistant.", force_no_stream=True
        )
        assert result == "Hello World"
        sys_msgs = [m for m in llm.messages if m["role"] == "system"]
        assert len(sys_msgs) == 1
        assert isinstance(sys_msgs[0]["content"], str)

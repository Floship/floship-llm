"""Tests for OpenRouter provider support.

OpenRouter (openrouter.ai) is an OpenAI-compatible gateway that provides
access to many models (xiaomi/mimo-v2.5, etc.) via a single API.

Key behaviors:
- Provider detection: 'openrouter' from URL containing 'openrouter.ai'
- top_p: standard param (not extra_body)
- WAF sanitization: auto-disabled (no CloudFront)
- Embedding params: no Heroku-specific params
- Audio: input_audio passed through (OpenAI standard; audio-capable models process it)
- Video: video_url/video_data converted to image_url for passthrough
- Images: image_url parts preserved (standard OpenAI spec)
"""

import os
from unittest.mock import MagicMock, patch

from floship_llm.client import LLM

OPENROUTER_URL = "https://openrouter.ai/api/v1"
HEROKU_URL = "https://us.inference.heroku.com/v1"
GOOGLE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class TestOpenRouterProviderDetection:
    """Test provider detection for OpenRouter URLs."""

    def test_detect_openrouter_from_url(self):
        """OpenRouter URL should be detected as 'openrouter' provider."""
        assert LLM._detect_provider(OPENROUTER_URL) == "openrouter"

    def test_detect_openrouter_case_insensitive(self):
        """Provider detection should be case-insensitive."""
        assert LLM._detect_provider("https://OpenRouter.AI/api/v1") == "openrouter"

    def test_detect_openrouter_trailing_slash(self):
        """Trailing slash should not affect detection."""
        assert LLM._detect_provider("https://openrouter.ai/api/v1/") == "openrouter"

    def test_detect_openrouter_does_not_match_others(self):
        """Other URLs should not be detected as openrouter."""
        assert LLM._detect_provider(HEROKU_URL) != "openrouter"
        assert LLM._detect_provider(GOOGLE_URL) != "openrouter"
        assert LLM._detect_provider("https://api.openai.com/v1") != "openrouter"


class TestOpenRouterInit:
    """Test LLM initialization with OpenRouter provider."""

    def setup_method(self):
        self.env_vars = {
            "INFERENCE_URL": OPENROUTER_URL,
            "INFERENCE_MODEL_ID": "xiaomi/mimo-v2.5",
            "INFERENCE_KEY": "test-openrouter-key",  # pragma: allowlist secret
        }
        self.env_patcher = patch.dict(os.environ, self.env_vars, clear=False)
        self.env_patcher.start()

    def teardown_method(self):
        self.env_patcher.stop()

    def test_init_openrouter_provider(self):
        """LLM should detect openrouter provider from URL."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
            assert llm._provider == "openrouter"
            assert llm.model == "xiaomi/mimo-v2.5"
            assert llm.base_url == OPENROUTER_URL + "/"

    def test_waf_auto_disabled_for_openrouter(self):
        """WAF sanitization should be auto-disabled for OpenRouter."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
            assert llm.waf_config.enable_waf_sanitization is False

    def test_waf_explicit_override_for_openrouter(self):
        """Explicit enable_waf_sanitization=True should override auto-disable."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(enable_waf_sanitization=True)
            assert llm.waf_config.enable_waf_sanitization is True


class TestOpenRouterRequestParams:
    """Test request parameter generation for OpenRouter."""

    def setup_method(self):
        self.env_vars = {
            "INFERENCE_URL": OPENROUTER_URL,
            "INFERENCE_MODEL_ID": "xiaomi/mimo-v2.5",
            "INFERENCE_KEY": "test-key",  # pragma: allowlist secret
        }
        self.env_patcher = patch.dict(os.environ, self.env_vars, clear=False)
        self.env_patcher.start()

    def teardown_method(self):
        self.env_patcher.stop()

    def test_top_p_is_standard_param(self):
        """OpenRouter should send top_p as a standard param, not extra_body."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(temperature=0.7, top_p=0.9)
            params = llm.get_request_params()

            # top_p should be in standard params, not extra_body
            assert "top_p" in params
            assert params["top_p"] == 0.9
            assert "extra_body" not in params

    def test_top_p_excludes_temperature(self):
        """When top_p is set, temperature should not be included."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(temperature=0.7, top_p=0.9)
            params = llm.get_request_params()

            assert "temperature" not in params

    def test_temperature_when_no_top_p(self):
        """Temperature should be included when top_p is not set."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(temperature=0.5)
            params = llm.get_request_params()

            assert params["temperature"] == 0.5
            assert "top_p" not in params

    def test_extended_thinking_ignored_for_openrouter(self):
        """Extended thinking is Heroku/Claude-only; should be ignored for OpenRouter."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(
                temperature=0.7,
                extended_thinking={"enabled": True, "budget_tokens": 1024},
            )
            params = llm.get_request_params()

            # Should just use normal temperature; no extra_body
            assert params["temperature"] == 0.7
            assert "extra_body" not in params

    def test_max_completion_tokens_included(self):
        """max_completion_tokens should be included for OpenRouter."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(max_completion_tokens=500)
            params = llm.get_request_params()

            assert params["max_completion_tokens"] == 500

    def test_top_k_not_in_extra_body_for_openrouter(self):
        """top_k is Heroku-only; should not be in extra_body for OpenRouter."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(top_k=50)
            params = llm.get_request_params()

            # top_k goes to extra_body only for Heroku
            assert "extra_body" not in params

    def test_tools_included_when_enabled(self):
        """Tools should be included for OpenRouter when enabled."""
        from floship_llm.schemas import ToolFunction, ToolParameter

        tool = ToolFunction(
            name="get_weather",
            description="Get weather",
            parameters=[
                ToolParameter(name="city", type="string", required=True),
            ],
        )
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(enable_tools=True, tools=[tool])
            params = llm.get_request_params()

            assert "tools" in params
            assert params["tool_choice"] == "auto"


class TestOpenRouterEmbeddingParams:
    """Test embedding parameters for OpenRouter."""

    def setup_method(self):
        self.env_vars = {
            "INFERENCE_URL": OPENROUTER_URL,
            "INFERENCE_MODEL_ID": "test-embed-model",
            "INFERENCE_KEY": "test-key",  # pragma: allowlist secret
        }
        self.env_patcher = patch.dict(os.environ, self.env_vars, clear=False)
        self.env_patcher.start()

    def teardown_method(self):
        self.env_patcher.stop()

    def test_embedding_params_no_heroku_specific(self):
        """OpenRouter embeddings should not include Heroku-specific params."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(
                type="embedding",
                input_type="search_document",
                encoding_format="base64",
            )
            params = llm.get_embedding_params()

            # Only model should be present
            assert params == {"model": "test-embed-model"}


class TestOpenRouterAudioAdaptation:
    """Test audio content part adaptation for OpenRouter."""

    def setup_method(self):
        self.env_vars = {
            "INFERENCE_URL": OPENROUTER_URL,
            "INFERENCE_MODEL_ID": "xiaomi/mimo-v2.5",
            "INFERENCE_KEY": "test-key",  # pragma: allowlist secret
        }
        self.env_patcher = patch.dict(os.environ, self.env_vars, clear=False)
        self.env_patcher.start()

    def teardown_method(self):
        self.env_patcher.stop()

    def _make_llm(self, **kwargs):
        with patch("floship_llm.client.OpenAI"):
            return LLM(**kwargs)

    def test_input_audio_passed_through_for_openrouter(self):
        """input_audio parts should be passed through for OpenRouter (OpenAI standard)."""
        llm = self._make_llm()
        content = [
            {"type": "text", "text": "Transcribe this audio."},
            {
                "type": "input_audio",
                "input_audio": {"data": "base64data", "format": "wav"},
            },
        ]
        adapted = llm._adapt_multimodal_for_provider(content)

        assert len(adapted) == 2
        assert adapted[0]["type"] == "text"
        assert adapted[0]["text"] == "Transcribe this audio."
        assert adapted[1]["type"] == "input_audio"
        assert adapted[1]["input_audio"]["data"] == "base64data"
        assert adapted[1]["input_audio"]["format"] == "wav"

    def test_input_audio_preserved_exactly(self):
        """input_audio parts should be preserved exactly as-is."""
        llm = self._make_llm()
        content = [
            {
                "type": "input_audio",
                "input_audio": {"data": "xyz", "format": "mp3"},
            },
        ]
        adapted = llm._adapt_multimodal_for_provider(content)
        assert adapted == content

    def test_image_url_preserved_for_openrouter(self):
        """image_url parts should be preserved as-is for OpenRouter."""
        llm = self._make_llm()
        content = [
            {"type": "text", "text": "Describe this image."},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,abc123"},
            },
        ]
        adapted = llm._adapt_multimodal_for_provider(content)

        assert len(adapted) == 2
        assert adapted[1]["type"] == "image_url"
        assert adapted[1]["image_url"]["url"] == "data:image/png;base64,abc123"

    def test_mixed_text_image_audio_for_openrouter(self):
        """Mixed content: text, image, and audio all preserved for OpenRouter."""
        llm = self._make_llm()
        content = [
            {"type": "text", "text": "Check this image and audio."},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,xyz"},
            },
            {
                "type": "input_audio",
                "input_audio": {"data": "wavdata", "format": "wav"},
            },
        ]
        adapted = llm._adapt_multimodal_for_provider(content)

        assert len(adapted) == 3
        assert adapted[0]["type"] == "text"
        assert adapted[0]["text"] == "Check this image and audio."
        assert adapted[1]["type"] == "image_url"
        assert adapted[2]["type"] == "input_audio"
        assert adapted[2]["input_audio"]["data"] == "wavdata"

    def test_text_only_content_unchanged(self):
        """Text-only content should be unchanged for OpenRouter."""
        llm = self._make_llm()
        content = [{"type": "text", "text": "Hello world"}]
        adapted = llm._adapt_multimodal_for_provider(content)

        assert adapted == content


class TestOpenRouterVideoAdaptation:
    """Test video content part adaptation for OpenRouter."""

    def setup_method(self):
        self.env_vars = {
            "INFERENCE_URL": OPENROUTER_URL,
            "INFERENCE_MODEL_ID": "xiaomi/mimo-v2.5",
            "INFERENCE_KEY": "test-key",  # pragma: allowlist secret
        }
        self.env_patcher = patch.dict(os.environ, self.env_vars, clear=False)
        self.env_patcher.start()

    def teardown_method(self):
        self.env_patcher.stop()

    def _make_llm(self, **kwargs):
        with patch("floship_llm.client.OpenAI"):
            return LLM(**kwargs)

    def test_video_url_converted_to_image_url(self):
        """video_url parts should be converted to image_url for OpenRouter."""
        llm = self._make_llm()
        content = [
            {"type": "text", "text": "Describe this video."},
            {
                "type": "video_url",
                "video_url": {"url": "https://example.com/clip.mp4"},
            },
        ]
        adapted = llm._adapt_multimodal_for_provider(content)

        assert len(adapted) == 2
        assert adapted[0]["type"] == "text"
        assert adapted[1]["type"] == "image_url"
        assert adapted[1]["image_url"]["url"] == "https://example.com/clip.mp4"

    def test_video_url_with_data_uri(self):
        """video_url with data URI should be converted to image_url."""
        llm = self._make_llm()
        content = [
            {
                "type": "video_url",
                "video_url": {"url": "data:video/mp4;base64,AAAA"},
            },
        ]
        adapted = llm._adapt_multimodal_for_provider(content)

        assert len(adapted) == 1
        assert adapted[0]["type"] == "image_url"
        assert adapted[0]["image_url"]["url"] == "data:video/mp4;base64,AAAA"

    def test_video_url_empty_url_skipped(self):
        """video_url with empty url should be skipped."""
        llm = self._make_llm()
        content = [
            {"type": "text", "text": "Hello"},
            {"type": "video_url", "video_url": {"url": ""}},
        ]
        adapted = llm._adapt_multimodal_for_provider(content)

        assert len(adapted) == 1
        assert adapted[0]["type"] == "text"

    def test_video_data_converted_to_image_url_data_uri(self):
        """video_data parts should be converted to image_url data URI."""
        llm = self._make_llm()
        content = [
            {
                "type": "video_data",
                "video_data": {"data": "AAAA", "mime_type": "video/webm"},
            },
        ]
        adapted = llm._adapt_multimodal_for_provider(content)

        assert len(adapted) == 1
        assert adapted[0]["type"] == "image_url"
        assert adapted[0]["image_url"]["url"] == "data:video/webm;base64,AAAA"

    def test_video_data_default_mime_type(self):
        """video_data without mime_type should default to video/mp4."""
        llm = self._make_llm()
        content = [
            {"type": "video_data", "video_data": {"data": "AAAA"}},
        ]
        adapted = llm._adapt_multimodal_for_provider(content)

        assert adapted[0]["image_url"]["url"] == "data:video/mp4;base64,AAAA"

    def test_video_data_empty_data_skipped(self):
        """video_data with empty data should be skipped."""
        llm = self._make_llm()
        content = [
            {"type": "text", "text": "Hello"},
            {"type": "video_data", "video_data": {"data": ""}},
        ]
        adapted = llm._adapt_multimodal_for_provider(content)

        assert len(adapted) == 1
        assert adapted[0]["type"] == "text"

    def test_mixed_audio_video_image_text(self):
        """Full multimodal mix: each part type handled correctly."""
        llm = self._make_llm()
        content = [
            {"type": "text", "text": "Analyze everything."},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}},
            {"type": "input_audio", "input_audio": {"data": "abc", "format": "wav"}},
            {"type": "video_url", "video_url": {"url": "https://example.com/vid.mp4"}},
            {
                "type": "video_data",
                "video_data": {"data": "def", "mime_type": "video/webm"},
            },
        ]
        adapted = llm._adapt_multimodal_for_provider(content)

        assert len(adapted) == 5
        assert adapted[0]["type"] == "text"
        assert adapted[0]["text"] == "Analyze everything."
        assert adapted[1]["type"] == "image_url"  # preserved
        assert adapted[2]["type"] == "input_audio"  # passed through (OpenAI standard)
        assert adapted[3]["type"] == "image_url"  # video_url -> image_url
        assert adapted[3]["image_url"]["url"] == "https://example.com/vid.mp4"
        assert adapted[4]["type"] == "image_url"  # video_data -> image_url data URI
        assert "video/webm" in adapted[4]["image_url"]["url"]


class TestGeminiKeepsAudioParts:
    """Test that Gemini providers preserve input_audio parts."""

    def setup_method(self):
        self.env_vars = {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_MODEL_ID": "gemini-3-flash",
            "INFERENCE_KEY": "test-key",  # pragma: allowlist secret
        }
        self.env_patcher = patch.dict(os.environ, self.env_vars, clear=False)
        self.env_patcher.start()

    def teardown_method(self):
        self.env_patcher.stop()

    def test_input_audio_preserved_for_google(self):
        """input_audio parts should be preserved for Google/Gemini provider."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
        assert llm._provider == "google"

        content = [
            {"type": "text", "text": "Transcribe this."},
            {
                "type": "input_audio",
                "input_audio": {"data": "base64data", "format": "wav"},
            },
        ]
        adapted = llm._adapt_multimodal_for_provider(content)

        # Should be unchanged
        assert adapted == content

    def test_input_audio_preserved_for_vertex(self):
        """input_audio parts should be preserved for Vertex AI provider."""
        # Vertex auto-enables native_google which needs GCP credentials,
        # so we create a regular LLM and override the provider to test the adapter.
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
        llm._provider = "vertex"

        content = [
            {
                "type": "input_audio",
                "input_audio": {"data": "data", "format": "mp3"},
            },
        ]
        adapted = llm._adapt_multimodal_for_provider(content)
        assert adapted == content

    def test_video_url_preserved_for_google(self):
        """video_url parts should be preserved for Google/Gemini provider."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
        assert llm._provider == "google"

        content = [
            {"type": "video_url", "video_url": {"url": "https://example.com/v.mp4"}},
        ]
        adapted = llm._adapt_multimodal_for_provider(content)
        assert adapted == content

    def test_video_data_preserved_for_google(self):
        """video_data parts should be preserved for Google/Gemini provider."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
        assert llm._provider == "google"

        content = [
            {
                "type": "video_data",
                "video_data": {"data": "AAAA", "mime_type": "video/mp4"},
            },
        ]
        adapted = llm._adapt_multimodal_for_provider(content)
        assert adapted == content


class TestOpenRouterValidationIntegration:
    """Test that audio adaptation works through the full validation pipeline."""

    def setup_method(self):
        self.env_vars = {
            "INFERENCE_URL": OPENROUTER_URL,
            "INFERENCE_MODEL_ID": "xiaomi/mimo-v2.5",
            "INFERENCE_KEY": "test-key",  # pragma: allowlist secret
        }
        self.env_patcher = patch.dict(os.environ, self.env_vars, clear=False)
        self.env_patcher.start()

    def teardown_method(self):
        self.env_patcher.stop()

    def test_validate_messages_adapts_audio_for_openrouter(self):
        """Full validation pipeline should pass through audio parts for OpenRouter."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()

        content = [
            {"type": "text", "text": "Transcribe this audio."},
            {
                "type": "input_audio",
                "input_audio": {"data": "base64data", "format": "wav"},
            },
        ]
        llm.add_message("user", content)
        validated = llm._validate_messages_for_api(llm.messages)

        user_msg = next(m for m in validated if m["role"] == "user")
        assert isinstance(user_msg["content"], list)
        assert len(user_msg["content"]) == 2
        # Both parts preserved as-is
        assert user_msg["content"][0]["type"] == "text"
        assert user_msg["content"][0]["text"] == "Transcribe this audio."
        assert user_msg["content"][1]["type"] == "input_audio"
        assert user_msg["content"][1]["input_audio"]["data"] == "base64data"

    def test_validate_preserves_images_for_openrouter(self):
        """Full validation pipeline should preserve image_url for OpenRouter."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()

        content = [
            {"type": "text", "text": "Describe."},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,abc"},
            },
        ]
        llm.add_message("user", content)
        validated = llm._validate_messages_for_api(llm.messages)

        user_msg = next(m for m in validated if m["role"] == "user")
        assert isinstance(user_msg["content"], list)
        assert len(user_msg["content"]) == 2
        assert user_msg["content"][1]["type"] == "image_url"

    def test_prompt_with_multimodal_openrouter(self):
        """prompt() with multimodal content should work for OpenRouter."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()

        content = [
            {"type": "text", "text": "Describe this image."},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,abc"},
            },
        ]

        # Mock backend.chat to return a mock streaming response
        mock_stream = self._make_mock_stream("A red square.")
        llm.backend.chat = MagicMock(return_value=mock_stream)

        result = llm.prompt(content)
        assert result == "A red square."

        # Verify the content was stored as multimodal
        user_msgs = [m for m in llm.messages if m["role"] == "user"]
        assert len(user_msgs) == 1
        assert isinstance(user_msgs[0]["content"], list)

    @staticmethod
    def _make_mock_stream(content: str):
        """Create a mock streaming response."""
        chunks = []
        for char in content:
            chunk = MagicMock()
            chunk.choices = [MagicMock()]
            chunk.choices[0].delta.content = char
            chunk.choices[0].delta.tool_calls = None
            chunks.append(chunk)
        return chunks


class TestHerokuComparison:
    """Verify OpenRouter behaves differently from Heroku for key params."""

    def test_heroku_uses_extra_body_for_top_p(self):
        """Heroku should still use extra_body for top_p."""
        with patch("floship_llm.client.OpenAI"), patch.dict(
            os.environ, {"INFERENCE_URL": HEROKU_URL}
        ):
            llm = LLM(top_p=0.9)
            params = llm.get_request_params()

            assert "extra_body" in params
            assert params["extra_body"]["top_p"] == 0.9
            assert "top_p" not in params

    def test_openrouter_uses_standard_param_for_top_p(self):
        """OpenRouter should use standard param for top_p."""
        with patch("floship_llm.client.OpenAI"), patch.dict(
            os.environ,
            {
                "INFERENCE_URL": OPENROUTER_URL,
                "INFERENCE_MODEL_ID": "test-model",
                "INFERENCE_KEY": "test-key",  # pragma: allowlist secret
            },
        ):
            llm = LLM(top_p=0.9)
            params = llm.get_request_params()

            assert params["top_p"] == 0.9
            assert "extra_body" not in params

    def test_google_uses_standard_param_for_top_p(self):
        """Google should also use standard param for top_p (same as OpenRouter)."""
        with patch("floship_llm.client.OpenAI"), patch.dict(
            os.environ, {"INFERENCE_URL": GOOGLE_URL}
        ):
            llm = LLM(top_p=0.9)
            params = llm.get_request_params()

            assert params["top_p"] == 0.9
            assert "extra_body" not in params

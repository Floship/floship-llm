"""Tests for Phase 5 -- Native-Only Features."""

import os
from unittest.mock import Mock, patch

import pytest

from floship_llm.backends.base import ProviderBackend
from floship_llm.backends.native_gemini import NativeGeminiBackend
from floship_llm.backends.openai_compat import OpenAICompatibleBackend
from floship_llm.client import LLM

GOOGLE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
HEROKU_URL = "https://us.inference.heroku.com/v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backend(**extra):
    """Create a NativeGeminiBackend with mocked genai SDK."""
    defaults = {"api_key": "k", "model": "gemini-2.5-flash"}
    defaults.update(extra)
    with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
        mock_genai = Mock()
        mock_types = Mock()
        mock_req.return_value = (mock_genai, mock_types)
        backend = NativeGeminiBackend(**defaults)
    return backend, mock_genai, mock_types


def _mock_gemini_response(text="Hello!"):
    part = Mock()
    part.text = text
    part.function_call = None
    content = Mock()
    content.parts = [part]
    candidate = Mock()
    candidate.content = content
    response = Mock()
    response.candidates = [candidate]
    return response


# ---------------------------------------------------------------------------
# 5.1 Pre-flight token counting
# ---------------------------------------------------------------------------


class TestCountTokensLLM:
    """Tests for LLM.count_tokens()."""

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "k",
            "INFERENCE_MODEL_ID": "gemini-2.5-flash",
        },
    )
    def test_count_tokens_native_single_message(self):
        """count_tokens() uses native API for native backend."""
        with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
            mock_genai = Mock()
            mock_types = Mock()
            mock_req.return_value = (mock_genai, mock_types)

            llm = LLM(native_google=True)

            mock_count = Mock()
            mock_count.total_tokens = 42
            mock_genai.Client.return_value.models.count_tokens.return_value = mock_count

            result = llm.count_tokens("Hello world")
            assert result == 42

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "k",
            "INFERENCE_MODEL_ID": "gemini-2.5-flash",
        },
    )
    def test_count_tokens_native_message_list(self):
        """count_tokens(messages=[...]) uses native API."""
        with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
            mock_genai = Mock()
            mock_types = Mock()
            mock_req.return_value = (mock_genai, mock_types)

            llm = LLM(native_google=True)

            mock_count = Mock()
            mock_count.total_tokens = 100
            mock_genai.Client.return_value.models.count_tokens.return_value = mock_count

            result = llm.count_tokens(
                messages=[
                    {"role": "system", "content": "Be helpful."},
                    {"role": "user", "content": "Hi there"},
                ]
            )
            assert result == 100

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": HEROKU_URL,
            "INFERENCE_KEY": "k",
            "INFERENCE_MODEL_ID": "claude-4-sonnet",
        },
    )
    def test_count_tokens_fallback_estimate(self):
        """count_tokens() falls back to char estimate for non-native backends."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
            # "Hello world" = 11 chars -> ~2-3 tokens
            result = llm.count_tokens("Hello world")
            assert result == max(1, 11 // 4)

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": HEROKU_URL,
            "INFERENCE_KEY": "k",
            "INFERENCE_MODEL_ID": "claude-4-sonnet",
        },
    )
    def test_count_tokens_fallback_empty_message(self):
        """count_tokens with empty string returns 1 minimum."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
            result = llm.count_tokens("")
            assert result == 1

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": HEROKU_URL,
            "INFERENCE_KEY": "k",
            "INFERENCE_MODEL_ID": "claude-4-sonnet",
        },
    )
    def test_count_tokens_fallback_message_list(self):
        """Fallback estimate sums content from all messages."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
            # 20 + 20 = 40 chars -> 10 tokens
            result = llm.count_tokens(
                messages=[
                    {"role": "system", "content": "A" * 20},
                    {"role": "user", "content": "B" * 20},
                ]
            )
            assert result == 10


# ---------------------------------------------------------------------------
# 5.2 File upload
# ---------------------------------------------------------------------------


class TestFileUpload:
    """Tests for file upload support."""

    def test_backend_supports_file_upload(self):
        backend, _, _ = _make_backend()
        assert backend.supports_file_upload is True

    def test_openai_compat_does_not_support_file_upload(self):
        backend = OpenAICompatibleBackend(client=Mock(), provider="heroku")
        assert backend.supports_file_upload is False

    def test_base_default_supports_file_upload_false(self):
        """Default supports_file_upload on base class is False."""
        assert ProviderBackend.supports_file_upload.fget(Mock()) is False

    def test_upload_file_calls_sdk(self):
        backend, mock_genai, _ = _make_backend()

        mock_file_ref = Mock()
        mock_file_ref.name = "files/abc123"
        mock_genai.Client.return_value.files.upload.return_value = mock_file_ref

        result = backend.upload_file("/path/to/video.mp4", mime_type="video/mp4")

        mock_genai.Client.return_value.files.upload.assert_called_once_with(
            path="/path/to/video.mp4", mime_type="video/mp4"
        )
        assert result is mock_file_ref

    def test_upload_file_without_mime_type(self):
        backend, mock_genai, _ = _make_backend()

        mock_file_ref = Mock()
        mock_genai.Client.return_value.files.upload.return_value = mock_file_ref

        result = backend.upload_file("/path/to/image.png")

        mock_genai.Client.return_value.files.upload.assert_called_once_with(
            path="/path/to/image.png"
        )
        assert result is mock_file_ref

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "k",
            "INFERENCE_MODEL_ID": "gemini-2.5-flash",
        },
    )
    def test_llm_upload_file_native(self):
        """LLM.upload_file() delegates to native backend."""
        with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
            mock_genai = Mock()
            mock_types = Mock()
            mock_req.return_value = (mock_genai, mock_types)

            llm = LLM(native_google=True)

            mock_file = Mock()
            mock_genai.Client.return_value.files.upload.return_value = mock_file

            result = llm.upload_file("/path/to/doc.pdf", mime_type="application/pdf")
            assert result is mock_file

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": HEROKU_URL,
            "INFERENCE_KEY": "k",
            "INFERENCE_MODEL_ID": "claude-4-sonnet",
        },
    )
    def test_llm_upload_file_non_native_raises(self):
        """LLM.upload_file() raises NotImplementedError for non-native."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
            with pytest.raises(NotImplementedError, match="native_google=True"):
                llm.upload_file("/path/to/file.txt")


# ---------------------------------------------------------------------------
# 5.3 Grounding with Google Search
# ---------------------------------------------------------------------------


class TestGrounding:
    """Tests for Google Search grounding."""

    def test_grounding_disabled_by_default(self):
        backend, _, _ = _make_backend()
        assert backend._grounding is False

    def test_grounding_enabled(self):
        backend, _, _ = _make_backend(grounding=True)
        assert backend._grounding is True

    def test_grounding_adds_google_search_tool(self):
        backend, mock_genai, mock_types = _make_backend(grounding=True)

        mock_genai.Client.return_value.models.generate_content.return_value = (
            _mock_gemini_response("News results")
        )

        backend.chat(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Latest tech news?"}],
        )

        config_call = mock_types.GenerateContentConfig.call_args
        config_kwargs = config_call[1]
        tools = config_kwargs.get("tools", [])
        assert len(tools) >= 1
        # GoogleSearch tool was added
        mock_types.GoogleSearch.assert_called_once()
        mock_types.Tool.assert_any_call(
            google_search=mock_types.GoogleSearch.return_value
        )

    def test_grounding_with_function_tools(self):
        """Grounding tool is added alongside function declaration tools."""
        backend, mock_genai, mock_types = _make_backend(grounding=True)

        mock_genai.Client.return_value.models.generate_content.return_value = (
            _mock_gemini_response("ok")
        )

        tools = [
            {
                "type": "function",
                "function": {"name": "search", "description": "Search"},
            }
        ]

        backend.chat(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            tools=tools,
        )

        config_call = mock_types.GenerateContentConfig.call_args
        config_kwargs = config_call[1]
        built_tools = config_kwargs.get("tools", [])
        # At least 2 tools: function declarations + google search
        assert len(built_tools) >= 2

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "k",
            "INFERENCE_MODEL_ID": "gemini-2.5-flash",
        },
    )
    def test_llm_grounding_forwarded(self):
        """grounding=True is forwarded through LLM to backend."""
        with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
            mock_req.return_value = (Mock(), Mock())
            llm = LLM(native_google=True, grounding=True)
            assert llm.backend._grounding is True


# ---------------------------------------------------------------------------
# 5.4 Safety settings
# ---------------------------------------------------------------------------


class TestSafetySettings:
    """Tests for fine-grained safety settings."""

    def test_safety_settings_none_by_default(self):
        backend, _, _ = _make_backend()
        assert backend._safety_settings is None

    def test_safety_settings_stored(self):
        settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_ONLY_HIGH",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
        }
        backend, _, _ = _make_backend(safety_settings=settings)
        assert backend._safety_settings == settings

    def test_safety_settings_forwarded_to_config(self):
        settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_ONLY_HIGH",
        }
        backend, mock_genai, mock_types = _make_backend(safety_settings=settings)

        mock_genai.Client.return_value.models.generate_content.return_value = (
            _mock_gemini_response("ok")
        )

        backend.chat(
            model="m",
            messages=[{"role": "user", "content": "hi"}],
        )

        config_call = mock_types.GenerateContentConfig.call_args
        config_kwargs = config_call[1]
        assert "safety_settings" in config_kwargs
        safety = config_kwargs["safety_settings"]
        assert len(safety) == 1
        mock_types.SafetySetting.assert_called_once_with(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="BLOCK_ONLY_HIGH",
        )

    def test_multiple_safety_settings(self):
        settings = {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_ONLY_HIGH",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_LOW_AND_ABOVE",
        }
        backend, mock_genai, mock_types = _make_backend(safety_settings=settings)

        mock_genai.Client.return_value.models.generate_content.return_value = (
            _mock_gemini_response("ok")
        )

        backend.chat(model="m", messages=[{"role": "user", "content": "hi"}])

        config_call = mock_types.GenerateContentConfig.call_args
        config_kwargs = config_call[1]
        assert len(config_kwargs["safety_settings"]) == 3

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "k",
            "INFERENCE_MODEL_ID": "gemini-2.5-flash",
        },
    )
    def test_llm_safety_settings_forwarded(self):
        """safety_settings are forwarded through LLM to backend."""
        settings = {"HARM_CATEGORY_HARASSMENT": "BLOCK_NONE"}
        with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
            mock_req.return_value = (Mock(), Mock())
            llm = LLM(native_google=True, safety_settings=settings)
            assert llm.backend._safety_settings == settings


# ---------------------------------------------------------------------------
# 5.5 Code execution
# ---------------------------------------------------------------------------


class TestCodeExecution:
    """Tests for Gemini code execution tool."""

    def test_code_execution_disabled_by_default(self):
        backend, _, _ = _make_backend()
        assert backend._code_execution is False

    def test_code_execution_enabled(self):
        backend, _, _ = _make_backend(code_execution=True)
        assert backend._code_execution is True

    def test_code_execution_adds_tool(self):
        backend, mock_genai, mock_types = _make_backend(code_execution=True)

        mock_genai.Client.return_value.models.generate_content.return_value = (
            _mock_gemini_response("Result: 42")
        )

        backend.chat(
            model="m",
            messages=[{"role": "user", "content": "Calculate 6 * 7"}],
        )

        config_call = mock_types.GenerateContentConfig.call_args
        config_kwargs = config_call[1]
        tools = config_kwargs.get("tools", [])
        assert len(tools) >= 1
        mock_types.ToolCodeExecution.assert_called_once()
        mock_types.Tool.assert_any_call(
            code_execution=mock_types.ToolCodeExecution.return_value
        )

    def test_code_execution_with_grounding(self):
        """Both grounding and code execution can be active simultaneously."""
        backend, mock_genai, mock_types = _make_backend(
            grounding=True, code_execution=True
        )

        mock_genai.Client.return_value.models.generate_content.return_value = (
            _mock_gemini_response("ok")
        )

        backend.chat(model="m", messages=[{"role": "user", "content": "hi"}])

        config_call = mock_types.GenerateContentConfig.call_args
        config_kwargs = config_call[1]
        tools = config_kwargs.get("tools", [])
        # At least 2 tools: google_search + code_execution
        assert len(tools) >= 2

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "k",
            "INFERENCE_MODEL_ID": "gemini-2.5-flash",
        },
    )
    def test_llm_code_execution_forwarded(self):
        """code_execution=True is forwarded through LLM to backend."""
        with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
            mock_req.return_value = (Mock(), Mock())
            llm = LLM(native_google=True, code_execution=True)
            assert llm.backend._code_execution is True


# ---------------------------------------------------------------------------
# Combined feature tests
# ---------------------------------------------------------------------------


class TestCombinedFeatures:
    """Tests for multiple Phase 5 features used together."""

    def test_all_features_enabled(self):
        backend, mock_genai, mock_types = _make_backend(
            safety_settings={"HARM_CATEGORY_HARASSMENT": "BLOCK_NONE"},
            grounding=True,
            code_execution=True,
        )

        mock_genai.Client.return_value.models.generate_content.return_value = (
            _mock_gemini_response("ok")
        )

        backend.chat(model="m", messages=[{"role": "user", "content": "hi"}])

        config_call = mock_types.GenerateContentConfig.call_args
        config_kwargs = config_call[1]

        # Safety settings present
        assert "safety_settings" in config_kwargs
        # Tools include both grounding and code execution
        tools = config_kwargs.get("tools", [])
        assert len(tools) >= 2

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "k",
            "INFERENCE_MODEL_ID": "gemini-2.5-flash",
        },
    )
    def test_llm_all_phase5_params(self):
        """All Phase 5 params forwarded through LLM."""
        with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
            mock_req.return_value = (Mock(), Mock())
            llm = LLM(
                native_google=True,
                safety_settings={"HARM_CATEGORY_HARASSMENT": "BLOCK_NONE"},
                grounding=True,
                code_execution=True,
            )
            assert llm.backend._safety_settings == {
                "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE"
            }
            assert llm.backend._grounding is True
            assert llm.backend._code_execution is True

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": HEROKU_URL,
            "INFERENCE_KEY": "k",
            "INFERENCE_MODEL_ID": "claude-4-sonnet",
        },
    )
    def test_phase5_params_ignored_for_non_native(self):
        """Phase 5 params don't break non-native backends."""
        with patch("floship_llm.client.OpenAI"):
            # These params are stored on LLM but not forwarded to OpenAICompatibleBackend
            llm = LLM(
                safety_settings={"HARM_CATEGORY_HARASSMENT": "BLOCK_NONE"},
                grounding=True,
                code_execution=True,
            )
            assert isinstance(llm.backend, OpenAICompatibleBackend)

"""Tests for Phase 6 -- Vertex AI Support."""

import os
from unittest.mock import Mock, patch

import pytest

from floship_llm.backends.native_gemini import NativeGeminiBackend
from floship_llm.client import LLM

VERTEX_URL = "https://us-central1-aiplatform.googleapis.com/"
GOOGLE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
HEROKU_URL = "https://us.inference.heroku.com/v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_vertex_backend(**extra):
    """Create a NativeGeminiBackend configured for Vertex AI."""
    defaults = {
        "model": "gemini-2.5-flash",
        "vertex": True,
        "project": "my-project",
        "location": "us-central1",
    }
    defaults.update(extra)
    with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
        mock_genai = Mock()
        mock_types = Mock()
        mock_req.return_value = (mock_genai, mock_types)
        backend = NativeGeminiBackend(**defaults)
    return backend, mock_genai, mock_types


def _make_api_key_backend(**extra):
    """Create a NativeGeminiBackend configured with API key (non-Vertex)."""
    defaults = {
        "api_key": "test-key",  # pragma: allowlist secret
        "model": "gemini-2.5-flash",
    }
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
# 6.1 Provider detection
# ---------------------------------------------------------------------------


class TestVertexProviderDetection:
    """Vertex AI URL is detected correctly by _detect_provider."""

    def test_vertex_url_detected(self):
        result = LLM._detect_provider(VERTEX_URL)
        assert result == "vertex"

    def test_vertex_url_with_path(self):
        url = "https://us-central1-aiplatform.googleapis.com/v1/projects/my-proj"
        result = LLM._detect_provider(url)
        assert result == "vertex"

    def test_vertex_url_case_insensitive(self):
        url = "https://US-CENTRAL1-AIPLATFORM.GOOGLEAPIS.COM/"
        result = LLM._detect_provider(url)
        assert result == "vertex"

    def test_vertex_does_not_match_google_ai(self):
        result = LLM._detect_provider(GOOGLE_URL)
        assert result == "google"

    def test_google_ai_does_not_match_vertex(self):
        result = LLM._detect_provider(GOOGLE_URL)
        assert result != "vertex"

    def test_heroku_unchanged(self):
        result = LLM._detect_provider(HEROKU_URL)
        assert result == "heroku"

    def test_unknown_unchanged(self):
        result = LLM._detect_provider("https://custom-llm.example.com/v1")
        assert result == "openai_compatible"


# ---------------------------------------------------------------------------
# 6.2 Backend creation with Vertex params
# ---------------------------------------------------------------------------


class TestVertexBackendCreation:
    """NativeGeminiBackend creates a vertexai Client when vertex=True."""

    def test_vertex_client_created_with_vertexai_true(self):
        _, mock_genai, _ = _make_vertex_backend()
        mock_genai.Client.assert_called_once_with(
            vertexai=True,
            project="my-project",
            location="us-central1",
        )

    def test_non_vertex_client_created_with_api_key(self):
        _, mock_genai, _ = _make_api_key_backend()
        mock_genai.Client.assert_called_once_with(
            api_key="test-key"  # pragma: allowlist secret
        )

    def test_vertex_flag_stored(self):
        backend, _, _ = _make_vertex_backend()
        assert backend._vertex is True

    def test_non_vertex_flag_stored(self):
        backend, _, _ = _make_api_key_backend()
        assert backend._vertex is False

    def test_vertex_with_custom_project_and_location(self):
        _, mock_genai, _ = _make_vertex_backend(
            project="prod-project",
            location="europe-west4",
        )
        mock_genai.Client.assert_called_once_with(
            vertexai=True,
            project="prod-project",
            location="europe-west4",
        )

    def test_vertex_no_api_key_required(self):
        """Vertex AI backend can be created without an API key."""
        _, mock_genai, _ = _make_vertex_backend(api_key=None)
        mock_genai.Client.assert_called_once_with(
            vertexai=True,
            project="my-project",
            location="us-central1",
        )

    def test_vertex_provider_name(self):
        """Vertex backend still reports google_native provider."""
        backend, _, _ = _make_vertex_backend()
        assert backend.provider_name == "google_native"

    def test_vertex_supports_caching(self):
        backend, _, _ = _make_vertex_backend()
        assert backend.supports_caching is True

    def test_vertex_supports_file_upload(self):
        backend, _, _ = _make_vertex_backend()
        assert backend.supports_file_upload is True

    def test_vertex_supports_native_tools(self):
        backend, _, _ = _make_vertex_backend()
        assert backend.supports_native_tools is True


# ---------------------------------------------------------------------------
# 6.3 LLM auto-enables native backend for Vertex
# ---------------------------------------------------------------------------


class TestVertexLLMIntegration:
    """LLM auto-enables native Gemini backend for Vertex AI URLs."""

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": VERTEX_URL,
            "INFERENCE_MODEL_ID": "gemini-2.5-flash",
        },
        clear=True,
    )
    @patch("floship_llm.backends.native_gemini._require_genai")
    def test_vertex_url_auto_enables_native(self, mock_req):
        mock_genai = Mock()
        mock_types = Mock()
        mock_req.return_value = (mock_genai, mock_types)

        llm = LLM()
        assert llm._use_native_gemini is True
        assert llm._provider == "vertex"

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": VERTEX_URL,
            "INFERENCE_MODEL_ID": "gemini-2.5-flash",
        },
        clear=True,
    )
    @patch("floship_llm.backends.native_gemini._require_genai")
    def test_vertex_no_api_key_needed(self, mock_req):
        """Vertex AI does not require an explicit API key."""
        mock_genai = Mock()
        mock_types = Mock()
        mock_req.return_value = (mock_genai, mock_types)

        # Should not raise ValueError about missing API key
        llm = LLM()
        assert llm.api_key is None

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": VERTEX_URL,
            "INFERENCE_KEY": "optional-key",
            "INFERENCE_MODEL_ID": "gemini-2.5-flash",
        },
        clear=True,
    )
    @patch("floship_llm.backends.native_gemini._require_genai")
    def test_vertex_with_explicit_key(self, mock_req):
        """Vertex AI still accepts an explicit API key if provided."""
        mock_genai = Mock()
        mock_types = Mock()
        mock_req.return_value = (mock_genai, mock_types)

        llm = LLM()
        assert llm.api_key == "optional-key"  # pragma: allowlist secret

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": VERTEX_URL,
            "INFERENCE_MODEL_ID": "gemini-2.5-flash",
            "GOOGLE_PROJECT": "env-project",
            "GOOGLE_LOCATION": "asia-east1",
        },
        clear=True,
    )
    @patch("floship_llm.backends.native_gemini._require_genai")
    def test_vertex_project_location_from_env(self, mock_req):
        """google_project and google_location read from env vars."""
        mock_genai = Mock()
        mock_types = Mock()
        mock_req.return_value = (mock_genai, mock_types)

        llm = LLM()
        assert llm._google_project == "env-project"
        assert llm._google_location == "asia-east1"
        mock_genai.Client.assert_called_once_with(
            vertexai=True,
            project="env-project",
            location="asia-east1",
        )

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": VERTEX_URL,
            "INFERENCE_MODEL_ID": "gemini-2.5-flash",
            "GOOGLE_PROJECT": "env-project",
            "GOOGLE_LOCATION": "us-east1",
        },
        clear=True,
    )
    @patch("floship_llm.backends.native_gemini._require_genai")
    def test_vertex_kwargs_override_env(self, mock_req):
        """Kwargs override environment variables for project/location."""
        mock_genai = Mock()
        mock_types = Mock()
        mock_req.return_value = (mock_genai, mock_types)

        llm = LLM(google_project="kwarg-project", google_location="europe-west1")
        assert llm._google_project == "kwarg-project"
        assert llm._google_location == "europe-west1"
        mock_genai.Client.assert_called_once_with(
            vertexai=True,
            project="kwarg-project",
            location="europe-west1",
        )

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": VERTEX_URL,
            "INFERENCE_MODEL_ID": "gemini-2.5-flash",
        },
        clear=True,
    )
    @patch("floship_llm.backends.native_gemini._require_genai")
    def test_vertex_creates_native_backend(self, mock_req):
        """Vertex AI URL creates NativeGeminiBackend, not OpenAICompatibleBackend."""
        mock_genai = Mock()
        mock_types = Mock()
        mock_req.return_value = (mock_genai, mock_types)

        llm = LLM()
        assert isinstance(llm.backend, NativeGeminiBackend)

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": VERTEX_URL,
            "INFERENCE_MODEL_ID": "gemini-2.5-flash",
        },
        clear=True,
    )
    @patch("floship_llm.backends.native_gemini._require_genai")
    def test_vertex_waf_disabled(self, mock_req):
        """WAF sanitization is auto-disabled for Vertex AI."""
        mock_genai = Mock()
        mock_types = Mock()
        mock_req.return_value = (mock_genai, mock_types)

        llm = LLM()
        assert llm.waf_config.enable_waf_sanitization is False


# ---------------------------------------------------------------------------
# 6.4 Non-Vertex still requires API key
# ---------------------------------------------------------------------------


class TestNonVertexRequiresKey:
    """Non-Vertex providers still require an API key."""

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_MODEL_ID": "gemini-2.5-flash",
        },
        clear=True,
    )
    def test_google_ai_requires_key(self):
        with pytest.raises(
            ValueError, match="INFERENCE_KEY environment variable must be set"
        ):
            LLM()

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": HEROKU_URL,
            "INFERENCE_MODEL_ID": "claude-4-sonnet",
        },
        clear=True,
    )
    def test_heroku_requires_key(self):
        with pytest.raises(
            ValueError, match="INFERENCE_KEY environment variable must be set"
        ):
            LLM()


# ---------------------------------------------------------------------------
# 6.5 Vertex AI features work (chat, embed, count_tokens, etc.)
# ---------------------------------------------------------------------------


class TestVertexFeatures:
    """All Phase 3-5 features work through Vertex AI backend."""

    def test_vertex_chat(self):
        backend, _, _ = _make_vertex_backend()
        mock_response = _mock_gemini_response("Vertex says hi")
        backend._client.models.generate_content.return_value = mock_response

        result = backend.chat(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert result.choices[0].message.content == "Vertex says hi"

    def test_vertex_embed(self):
        backend, _, _ = _make_vertex_backend()
        mock_embedding = Mock()
        mock_embedding.values = [0.1, 0.2, 0.3]
        mock_result = Mock()
        mock_result.embeddings = [mock_embedding]
        backend._client.models.embed_content.return_value = mock_result

        result = backend.embed(
            model="gemini-embedding-001",
            input="test text",
        )
        assert result.data[0].embedding == [0.1, 0.2, 0.3]

    def test_vertex_count_tokens(self):
        backend, _, _ = _make_vertex_backend()
        mock_count = Mock()
        mock_count.total_tokens = 42
        backend._client.models.count_tokens.return_value = mock_count

        count = backend.count_tokens([{"role": "user", "content": "Test"}])
        assert count == 42

    def test_vertex_file_upload(self):
        backend, _, _ = _make_vertex_backend()
        mock_file = Mock()
        backend._client.files.upload.return_value = mock_file

        result = backend.upload_file("/tmp/video.mp4", mime_type="video/mp4")
        assert result == mock_file
        backend._client.files.upload.assert_called_once_with(
            path="/tmp/video.mp4", mime_type="video/mp4"
        )

    def test_vertex_cache_lifecycle(self):
        """Cache create/clear works on Vertex backend."""
        backend, _, _ = _make_vertex_backend(cache=True)
        # Initially no cache
        assert backend.get_cache_info() is None
        # clear_cache is safe to call even with no cache
        backend.clear_cache()

    def test_vertex_streaming(self):
        backend, _, _ = _make_vertex_backend()

        chunk_part = Mock()
        chunk_part.text = "streamed"
        chunk_part.function_call = None
        chunk_content = Mock()
        chunk_content.parts = [chunk_part]
        chunk_candidate = Mock()
        chunk_candidate.content = chunk_content
        chunk = Mock()
        chunk.candidates = [chunk_candidate]

        backend._client.models.generate_content_stream.return_value = iter([chunk])

        result = backend.chat(
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        )
        chunks = list(result)
        # 1 data chunk + 1 sentinel with finish_reason
        assert len(chunks) == 2
        assert chunks[0].choices[0].delta.content == "streamed"
        assert chunks[-1].choices[0].finish_reason == "stop"

    def test_vertex_with_safety_settings(self):
        backend, _, _ = _make_vertex_backend(
            safety_settings={"HARM_CATEGORY_HARASSMENT": "BLOCK_ONLY_HIGH"},
        )
        assert backend._safety_settings == {
            "HARM_CATEGORY_HARASSMENT": "BLOCK_ONLY_HIGH"
        }

    def test_vertex_with_grounding(self):
        backend, _, _ = _make_vertex_backend(grounding=True)
        assert backend._grounding is True

    def test_vertex_with_code_execution(self):
        backend, _, _ = _make_vertex_backend(code_execution=True)
        assert backend._code_execution is True


# ---------------------------------------------------------------------------
# 6.6 LLM-level Vertex integration
# ---------------------------------------------------------------------------


class TestVertexLLMLevelFeatures:
    """LLM-level methods work with Vertex backend."""

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": VERTEX_URL,
            "INFERENCE_MODEL_ID": "gemini-2.5-flash",
        },
        clear=True,
    )
    @patch("floship_llm.backends.native_gemini._require_genai")
    def test_llm_count_tokens_via_vertex(self, mock_req):
        mock_genai = Mock()
        mock_types = Mock()
        mock_req.return_value = (mock_genai, mock_types)

        llm = LLM()
        mock_count = Mock()
        mock_count.total_tokens = 15
        llm.backend._client.models.count_tokens.return_value = mock_count

        count = llm.count_tokens("Some text")
        assert count == 15

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": VERTEX_URL,
            "INFERENCE_MODEL_ID": "gemini-2.5-flash",
        },
        clear=True,
    )
    @patch("floship_llm.backends.native_gemini._require_genai")
    def test_llm_upload_file_via_vertex(self, mock_req):
        mock_genai = Mock()
        mock_types = Mock()
        mock_req.return_value = (mock_genai, mock_types)

        llm = LLM()
        mock_file = Mock()
        llm.backend._client.files.upload.return_value = mock_file

        result = llm.upload_file("/tmp/doc.pdf", mime_type="application/pdf")
        assert result == mock_file

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": VERTEX_URL,
            "INFERENCE_MODEL_ID": "gemini-2.5-flash",
        },
        clear=True,
    )
    @patch("floship_llm.backends.native_gemini._require_genai")
    def test_llm_cache_info_via_vertex(self, mock_req):
        mock_genai = Mock()
        mock_types = Mock()
        mock_req.return_value = (mock_genai, mock_types)

        llm = LLM()
        # No cache active yet
        assert llm.cache_info is None

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": VERTEX_URL,
            "INFERENCE_MODEL_ID": "gemini-2.5-flash",
        },
        clear=True,
    )
    @patch("floship_llm.backends.native_gemini._require_genai")
    def test_llm_clear_cache_via_vertex(self, mock_req):
        mock_genai = Mock()
        mock_types = Mock()
        mock_req.return_value = (mock_genai, mock_types)

        llm = LLM()
        # Should not raise
        llm.clear_cache()

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": VERTEX_URL,
            "INFERENCE_MODEL_ID": "gemini-2.5-flash",
        },
        clear=True,
    )
    @patch("floship_llm.backends.native_gemini._require_genai")
    def test_llm_vertex_forwards_all_params(self, mock_req):
        """All Phase 4-6 params forwarded to backend via Vertex."""
        mock_genai = Mock()
        mock_types = Mock()
        mock_req.return_value = (mock_genai, mock_types)

        llm = LLM(
            cache=True,
            cache_ttl=7200,
            safety_settings={"HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE"},
            grounding=True,
            code_execution=True,
            google_project="test-proj",
            google_location="eu-west1",
        )
        backend = llm.backend
        assert backend._cache_enabled is True
        assert backend._cache_ttl == 7200
        assert backend._safety_settings == {"HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE"}
        assert backend._grounding is True
        assert backend._code_execution is True
        assert backend._vertex is True

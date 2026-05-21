"""Tests for GoogleCacheManager and LLM context caching integration."""

import os
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from floship_llm.client import LLM
from floship_llm.google_cache_manager import (
    _REFRESH_MARGIN_SECONDS,
    ContextCacheRef,
    GoogleCacheManager,
)

GOOGLE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
HEROKU_URL = "https://us.inference.heroku.com/v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager(**extra):
    """Create a GoogleCacheManager with mocked genai SDK."""
    defaults = {
        "api_key": "test-key",  # pragma: allowlist secret
        "default_ttl_seconds": 3600,
    }
    defaults.update(extra)
    with patch("floship_llm.google_cache_manager._require_genai") as mock_req:
        mock_genai = Mock()
        mock_types = Mock()
        mock_req.return_value = (mock_genai, mock_types)
        mgr = GoogleCacheManager(**defaults)
    return mgr, mock_genai, mock_types


def _mock_cache_response(name="cachedContents/abc123"):
    cache = Mock()
    cache.name = name
    cache.usage_metadata = Mock()
    cache.usage_metadata.total_token_count = 12000
    return cache


# ---------------------------------------------------------------------------
# ContextCacheRef
# ---------------------------------------------------------------------------


class TestContextCacheRef:
    """Tests for the ContextCacheRef data holder."""

    def test_is_valid_when_not_expired(self):
        ref = ContextCacheRef(
            name="c/1", key="k", model="m", expires_at=time.time() + 600
        )
        assert ref.is_valid() is True

    def test_is_invalid_when_expired(self):
        ref = ContextCacheRef(
            name="c/1", key="k", model="m", expires_at=time.time() - 10
        )
        assert ref.is_valid() is False

    def test_is_invalid_within_margin(self):
        ref = ContextCacheRef(
            name="c/1",
            key="k",
            model="m",
            expires_at=time.time() + _REFRESH_MARGIN_SECONDS - 1,
        )
        assert ref.is_valid() is False

    def test_token_count_optional(self):
        ref = ContextCacheRef(name="c/1", key="k", model="m", expires_at=0)
        assert ref.token_count is None

    def test_token_count_stored(self):
        ref = ContextCacheRef(
            name="c/1", key="k", model="m", expires_at=0, token_count=9000
        )
        assert ref.token_count == 9000


# ---------------------------------------------------------------------------
# GoogleCacheManager.make_key
# ---------------------------------------------------------------------------


class TestMakeKey:
    """Deterministic key generation."""

    def test_stable_key(self):
        k1 = GoogleCacheManager.make_key(model="m", system="sys", version="v1")
        k2 = GoogleCacheManager.make_key(model="m", system="sys", version="v1")
        assert k1 == k2

    def test_different_system_different_key(self):
        k1 = GoogleCacheManager.make_key(model="m", system="sys1", version="v1")
        k2 = GoogleCacheManager.make_key(model="m", system="sys2", version="v1")
        assert k1 != k2

    def test_different_tools_different_key(self):
        tools_a = [{"function": {"name": "a"}}]
        tools_b = [{"function": {"name": "b"}}]
        k1 = GoogleCacheManager.make_key(model="m", tools=tools_a, version="v1")
        k2 = GoogleCacheManager.make_key(model="m", tools=tools_b, version="v1")
        assert k1 != k2

    def test_different_permission_hash_different_key(self):
        k1 = GoogleCacheManager.make_key(
            model="m", version="v1", permission_hash="admin"
        )
        k2 = GoogleCacheManager.make_key(
            model="m", version="v1", permission_hash="staff"
        )
        assert k1 != k2

    def test_different_version_different_key(self):
        k1 = GoogleCacheManager.make_key(model="m", version="v1")
        k2 = GoogleCacheManager.make_key(model="m", version="v2")
        assert k1 != k2

    def test_key_is_sha256_hex(self):
        k = GoogleCacheManager.make_key(model="m", version="v1")
        assert len(k) == 64
        assert all(c in "0123456789abcdef" for c in k)


# ---------------------------------------------------------------------------
# GoogleCacheManager.create_or_get
# ---------------------------------------------------------------------------


class TestCreateOrGet:
    """Cache lifecycle: create and local-ref reuse."""

    def test_creates_cache_via_sdk(self):
        mgr, mock_genai, _ = _make_manager()
        mock_client = mock_genai.Client.return_value
        mock_client.caches.create.return_value = _mock_cache_response()

        ref = mgr.create_or_get(model="gemini-3.5-flash", system="Hello", version="v1")

        mock_client.caches.create.assert_called_once()
        assert ref.name == "cachedContents/abc123"
        assert ref.model == "gemini-3.5-flash"
        assert ref.token_count == 12000

    def test_reuses_existing_ref(self):
        mgr, mock_genai, _ = _make_manager()
        mock_client = mock_genai.Client.return_value
        mock_client.caches.create.return_value = _mock_cache_response()

        ref1 = mgr.create_or_get(model="m", system="s", version="v1")
        ref2 = mgr.create_or_get(model="m", system="s", version="v1")

        assert ref1 is ref2
        assert mock_client.caches.create.call_count == 1

    def test_expired_ref_creates_new(self):
        mgr, mock_genai, _ = _make_manager(default_ttl_seconds=1)
        mock_client = mock_genai.Client.return_value
        mock_client.caches.create.return_value = _mock_cache_response()

        ref1 = mgr.create_or_get(model="m", system="s", version="v1")
        # Force expire
        ref1.expires_at = time.time() - 10

        mock_client.caches.create.return_value = _mock_cache_response("c/new")
        ref2 = mgr.create_or_get(model="m", system="s", version="v1")

        assert ref2.name == "c/new"
        assert mock_client.caches.create.call_count == 2

    def test_changed_content_creates_new_cache(self):
        mgr, mock_genai, _ = _make_manager()
        mock_client = mock_genai.Client.return_value
        mock_client.caches.create.return_value = _mock_cache_response("c/1")

        ref1 = mgr.create_or_get(model="m", system="sys1", version="v1")

        mock_client.caches.create.return_value = _mock_cache_response("c/2")
        ref2 = mgr.create_or_get(model="m", system="sys2", version="v1")

        assert ref1.name == "c/1"
        assert ref2.name == "c/2"
        assert mock_client.caches.create.call_count == 2

    def test_display_name_default(self):
        mgr, mock_genai, mock_types = _make_manager()
        mock_client = mock_genai.Client.return_value
        mock_client.caches.create.return_value = _mock_cache_response()

        mgr.create_or_get(model="m", system="s", version="v1")

        config_call = mock_types.CreateCachedContentConfig.call_args
        display = config_call.kwargs.get("display_name", "")
        assert display.startswith("floship-llm-")

    def test_custom_display_name(self):
        mgr, mock_genai, mock_types = _make_manager()
        mock_client = mock_genai.Client.return_value
        mock_client.caches.create.return_value = _mock_cache_response()

        mgr.create_or_get(model="m", system="s", version="v1", display_name="my-cache")

        config_call = mock_types.CreateCachedContentConfig.call_args
        assert config_call.kwargs.get("display_name") == "my-cache"

    def test_ttl_override(self):
        mgr, mock_genai, mock_types = _make_manager(default_ttl_seconds=3600)
        mock_client = mock_genai.Client.return_value
        mock_client.caches.create.return_value = _mock_cache_response()

        mgr.create_or_get(model="m", system="s", version="v1", ttl_seconds=300)

        config_call = mock_types.CreateCachedContentConfig.call_args
        assert config_call.kwargs.get("ttl") == "300s"

    def test_tools_forwarded(self):
        mgr, mock_genai, mock_types = _make_manager()
        mock_client = mock_genai.Client.return_value
        mock_client.caches.create.return_value = _mock_cache_response()

        tools = [{"function": {"name": "search"}}]
        mgr.create_or_get(model="m", tools=tools, version="v1")

        config_call = mock_types.CreateCachedContentConfig.call_args
        assert config_call.kwargs.get("tools") == tools


# ---------------------------------------------------------------------------
# GoogleCacheManager.update_ttl / delete / clear_all
# ---------------------------------------------------------------------------


class TestCacheLifecycle:
    """Update, delete, clear operations."""

    def test_update_ttl(self):
        mgr, mock_genai, _ = _make_manager()
        mock_client = mock_genai.Client.return_value

        mgr.update_ttl("c/1", 7200)

        mock_client.caches.update.assert_called_once()

    def test_delete(self):
        mgr, mock_genai, _ = _make_manager()
        mock_client = mock_genai.Client.return_value

        mgr.delete("c/1")

        mock_client.caches.delete.assert_called_once_with(name="c/1")

    def test_delete_silent_on_error(self):
        mgr, mock_genai, _ = _make_manager()
        mock_client = mock_genai.Client.return_value
        mock_client.caches.delete.side_effect = Exception("not found")

        mgr.delete("c/1")  # Should not raise

    def test_clear_all(self):
        mgr, mock_genai, _ = _make_manager()
        mock_client = mock_genai.Client.return_value
        mock_client.caches.create.return_value = _mock_cache_response("c/1")

        mgr.create_or_get(model="m", system="s", version="v1")
        mgr.clear_all()

        mock_client.caches.delete.assert_called_once()
        assert len(mgr._local_cache) == 0


# ---------------------------------------------------------------------------
# GoogleCacheManager.should_cache (cost model)
# ---------------------------------------------------------------------------


class TestShouldCache:
    """Break-even cost logic."""

    def test_below_min_tokens(self):
        assert GoogleCacheManager.should_cache(1000, 100, 1.0) is False

    def test_above_threshold_enough_reuse(self):
        # 1h TTL: breakeven = (1.50 + 1.0) / 1.35 ≈ 1.85
        assert GoogleCacheManager.should_cache(10000, 3, 1.0) is True

    def test_above_threshold_not_enough_reuse(self):
        # 24h TTL: breakeven = (1.50 + 24.0) / 1.35 ≈ 18.9
        assert GoogleCacheManager.should_cache(10000, 5, 24.0) is False

    def test_5min_ttl_breakeven(self):
        # 5min = 0.083h: breakeven ≈ 1.17
        assert GoogleCacheManager.should_cache(10000, 2, 0.083) is True

    def test_custom_min_tokens(self):
        assert GoogleCacheManager.should_cache(5000, 10, 1.0, min_tokens=10000) is False


# ---------------------------------------------------------------------------
# LLM integration -- Google + cache enabled
# ---------------------------------------------------------------------------


class TestLLMContextCacheIntegration:
    """LLM creates cache manager for Google provider."""

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "k",
            "INFERENCE_MODEL_ID": "gemini-3.5-flash",
        },
        clear=True,
    )
    @patch("floship_llm.google_cache_manager._require_genai")
    def test_cache_manager_created_for_google(self, mock_req):
        mock_genai = Mock()
        mock_types = Mock()
        mock_req.return_value = (mock_genai, mock_types)

        llm = LLM(enable_context_cache=True)
        assert llm._google_cache_manager is not None

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": HEROKU_URL,
            "INFERENCE_KEY": "k",
            "INFERENCE_MODEL_ID": "claude-4-sonnet",
        },
        clear=True,
    )
    def test_cache_manager_not_created_for_heroku(self):
        llm = LLM(enable_context_cache=True)
        assert llm._google_cache_manager is None

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "k",
            "INFERENCE_MODEL_ID": "gemini-3.5-flash",
        },
        clear=True,
    )
    def test_cache_manager_not_created_when_disabled(self):
        llm = LLM(enable_context_cache=False)
        assert llm._google_cache_manager is None

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "k",
            "INFERENCE_MODEL_ID": "gemini-3.5-flash",
            "FLOSHIP_LLM_CONTEXT_CACHE": "true",
        },
        clear=True,
    )
    @patch("floship_llm.google_cache_manager._require_genai")
    def test_cache_enabled_via_env_var(self, mock_req):
        mock_genai = Mock()
        mock_types = Mock()
        mock_req.return_value = (mock_genai, mock_types)

        llm = LLM()
        assert llm._google_cache_manager is not None

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "k",
            "INFERENCE_MODEL_ID": "gemini-3.5-flash",
            "GEMINI_NATIVE": "true",
        },
        clear=True,
    )
    @patch("floship_llm.backends.native_gemini._require_genai")
    def test_cache_manager_not_created_for_native_gemini(self, mock_req):
        """Native Gemini path has its own caching; don't create GoogleCacheManager."""
        mock_genai = Mock()
        mock_types = Mock()
        mock_req.return_value = (mock_genai, mock_types)

        llm = LLM(enable_context_cache=True)
        assert llm._google_cache_manager is None


# ---------------------------------------------------------------------------
# LLM -- create_context_cache / delete_context_cache
# ---------------------------------------------------------------------------


class TestLLMManualCacheAPI:
    """Manual cache creation via LLM."""

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "k",
            "INFERENCE_MODEL_ID": "gemini-3.5-flash",
        },
        clear=True,
    )
    @patch("floship_llm.google_cache_manager._require_genai")
    def test_create_context_cache(self, mock_req):
        mock_genai = Mock()
        mock_types = Mock()
        mock_req.return_value = (mock_genai, mock_types)

        llm = LLM(enable_context_cache=True)
        mock_client = mock_genai.Client.return_value
        mock_client.caches.create.return_value = _mock_cache_response()

        ref = llm.create_context_cache(system="Hello", version="v1")
        assert ref.name == "cachedContents/abc123"
        mock_client.caches.create.assert_called_once()

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "k",
            "INFERENCE_MODEL_ID": "gemini-3.5-flash",
        },
        clear=True,
    )
    def test_create_context_cache_raises_without_manager(self):
        llm = LLM(enable_context_cache=False)
        with pytest.raises(RuntimeError, match="Context caching requires"):
            llm.create_context_cache(system="Hello")

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "k",
            "INFERENCE_MODEL_ID": "gemini-3.5-flash",
        },
        clear=True,
    )
    @patch("floship_llm.google_cache_manager._require_genai")
    def test_delete_context_cache(self, mock_req):
        mock_genai = Mock()
        mock_types = Mock()
        mock_req.return_value = (mock_genai, mock_types)

        llm = LLM(enable_context_cache=True)
        mock_client = mock_genai.Client.return_value
        mock_client.caches.create.return_value = _mock_cache_response("c/del")

        ref = llm.create_context_cache(system="sys", version="v1")
        llm._active_cache_ref = ref

        llm.delete_context_cache()
        mock_client.caches.delete.assert_called_once()
        assert llm._active_cache_ref is None


# ---------------------------------------------------------------------------
# LLM -- cached request flow
# ---------------------------------------------------------------------------


class TestLLMCachedRequest:
    """Cache ref injected into extra_body, static context stripped."""

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "k",
            "INFERENCE_MODEL_ID": "gemini-3.5-flash",
        },
        clear=True,
    )
    @patch("floship_llm.google_cache_manager._require_genai")
    def test_extra_body_has_cached_content(self, mock_req):
        mock_genai = Mock()
        mock_types = Mock()
        mock_req.return_value = (mock_genai, mock_types)

        llm = LLM(enable_context_cache=True)

        # Manually set an active cache ref
        llm._active_cache_ref = ContextCacheRef(
            name="cachedContents/xyz",
            key="k",
            model="gemini-3.5-flash",
            expires_at=time.time() + 3600,
        )

        params = llm.get_request_params()
        assert (
            params.get("extra_body", {}).get("cached_content") == "cachedContents/xyz"
        )

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "k",
            "INFERENCE_MODEL_ID": "gemini-3.5-flash",
        },
        clear=True,
    )
    @patch("floship_llm.google_cache_manager._require_genai")
    def test_expired_ref_not_in_extra_body(self, mock_req):
        mock_genai = Mock()
        mock_types = Mock()
        mock_req.return_value = (mock_genai, mock_types)

        llm = LLM(enable_context_cache=True)
        llm._active_cache_ref = ContextCacheRef(
            name="c/expired",
            key="k",
            model="m",
            expires_at=time.time() - 10,
        )

        params = llm.get_request_params()
        assert "cached_content" not in params.get("extra_body", {})

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": HEROKU_URL,
            "INFERENCE_KEY": "k",
            "INFERENCE_MODEL_ID": "claude-4-sonnet",
        },
        clear=True,
    )
    def test_heroku_ignores_cache_ref(self):
        llm = LLM()
        llm._active_cache_ref = ContextCacheRef(
            name="c/1",
            key="k",
            model="m",
            expires_at=time.time() + 3600,
        )

        params = llm.get_request_params()
        assert "cached_content" not in params.get("extra_body", {})

    def test_strip_cached_context_removes_system(self):
        """_strip_cached_context removes system messages."""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Bye"},
        ]
        # Use a fresh LLM-like call
        from floship_llm.client import LLM as _LLM

        result = _LLM._strip_cached_context(None, messages)
        assert len(result) == 3
        assert all(m["role"] != "system" for m in result)

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "k",
            "INFERENCE_MODEL_ID": "gemini-3.5-flash",
        },
        clear=True,
    )
    @patch("floship_llm.google_cache_manager._require_genai")
    def test_prompt_with_cached_content_kwarg(self, mock_req):
        """prompt(cached_content='c/xyz') sets ref and injects into request."""
        mock_genai = Mock()
        mock_types = Mock()
        mock_req.return_value = (mock_genai, mock_types)

        llm = LLM(enable_context_cache=True, system="Cached system")

        # Mock the backend chat response
        resp = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="response text",
                        tool_calls=None,
                        role="assistant",
                    )
                )
            ]
        )
        llm.backend.chat = Mock(return_value=resp)

        llm.prompt(
            "Hello",
            cached_content="cachedContents/manual",
            force_no_stream=True,
        )

        # Check that cached_content was set
        assert llm._active_cache_ref is not None
        assert llm._active_cache_ref.name == "cachedContents/manual"

        # Check the call params
        call_kwargs = llm.backend.chat.call_args
        params = {k: v for k, v in call_kwargs.kwargs.items() if k != "messages"}
        assert (
            params.get("extra_body", {}).get("cached_content")
            == "cachedContents/manual"
        )

        # System message should be stripped from sent messages
        sent_messages = call_kwargs.kwargs.get("messages", [])
        assert all(m.get("role") != "system" for m in sent_messages)

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "k",
            "INFERENCE_MODEL_ID": "gemini-3.5-flash",
        },
        clear=True,
    )
    @patch("floship_llm.google_cache_manager._require_genai")
    def test_prompt_with_cache_static_context(self, mock_req):
        """prompt(cache_static_context=True) auto-creates cache."""
        mock_genai = Mock()
        mock_types = Mock()
        mock_req.return_value = (mock_genai, mock_types)

        llm = LLM(enable_context_cache=True, system="Cached system")

        # Mock cache creation
        mock_client = mock_genai.Client.return_value
        mock_client.caches.create.return_value = _mock_cache_response("c/auto")

        # Mock the backend chat response
        resp = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="cached response",
                        tool_calls=None,
                        role="assistant",
                    )
                )
            ]
        )
        llm.backend.chat = Mock(return_value=resp)

        llm.prompt("Hello", cache_static_context=True, force_no_stream=True)

        # Cache should have been created
        mock_client.caches.create.assert_called_once()
        assert llm._active_cache_ref is not None
        assert llm._active_cache_ref.name == "c/auto"

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "k",
            "INFERENCE_MODEL_ID": "gemini-3.5-flash",
        },
        clear=True,
    )
    @patch("floship_llm.google_cache_manager._require_genai")
    def test_cached_request_strips_tools(self, mock_req):
        """When cache is active, tools are not sent in params (already cached)."""
        mock_genai = Mock()
        mock_types = Mock()
        mock_req.return_value = (mock_genai, mock_types)

        from floship_llm.schemas import ToolFunction, ToolParameter

        tool = ToolFunction(
            name="search",
            description="Search",
            parameters=[
                ToolParameter(
                    name="query", type="string", description="q", required=True
                )
            ],
            function=lambda _query: "result",
        )
        llm = LLM(enable_context_cache=True, tools=[tool])

        # Set active cache
        llm._active_cache_ref = ContextCacheRef(
            name="c/tools",
            key="k",
            model="gemini-3.5-flash",
            expires_at=time.time() + 3600,
        )

        # Mock backend
        resp = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="ok", tool_calls=None, role="assistant"
                    )
                )
            ]
        )
        llm.backend.chat = Mock(return_value=resp)

        llm.prompt("Hello", cached_content="c/tools")

        call_kwargs = llm.backend.chat.call_args.kwargs
        assert "tools" not in call_kwargs
        assert "tool_choice" not in call_kwargs


# ---------------------------------------------------------------------------
# LLM -- config forwarding
# ---------------------------------------------------------------------------


class TestLLMCacheConfig:
    """Context cache config kwargs are stored correctly."""

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "k",
            "INFERENCE_MODEL_ID": "gemini-3.5-flash",
        },
        clear=True,
    )
    @patch("floship_llm.google_cache_manager._require_genai")
    def test_config_defaults(self, mock_req):
        mock_req.return_value = (Mock(), Mock())
        llm = LLM(enable_context_cache=True)
        assert llm._context_cache_ttl == 300
        assert llm._context_cache_min_tokens == 8000
        assert llm._context_cache_version == ""
        assert llm._context_cache_scope == "global_static"

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "k",
            "INFERENCE_MODEL_ID": "gemini-3.5-flash",
        },
        clear=True,
    )
    @patch("floship_llm.google_cache_manager._require_genai")
    def test_config_overrides(self, mock_req):
        mock_req.return_value = (Mock(), Mock())
        llm = LLM(
            enable_context_cache=True,
            context_cache_ttl_seconds=300,
            context_cache_min_tokens=4000,
            context_cache_version="flobo-v12",
            context_cache_scope="staff_tools",
            context_cache_expected_reuse=10,
        )
        assert llm._context_cache_ttl == 300
        assert llm._context_cache_min_tokens == 4000
        assert llm._context_cache_version == "flobo-v12"
        assert llm._context_cache_scope == "staff_tools"
        assert llm._context_cache_expected_reuse == 10

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "k",
            "INFERENCE_MODEL_ID": "gemini-3.5-flash",
            "FLOSHIP_LLM_CONTEXT_CACHE_TTL": "600",
            "FLOSHIP_LLM_CONTEXT_CACHE_MIN_TOKENS": "5000",
            "FLOSHIP_LLM_CONTEXT_CACHE_VERSION": "env-v1",
        },
        clear=True,
    )
    @patch("floship_llm.google_cache_manager._require_genai")
    def test_config_from_env(self, mock_req):
        mock_req.return_value = (Mock(), Mock())
        llm = LLM(enable_context_cache=True)
        assert llm._context_cache_ttl == 600
        assert llm._context_cache_min_tokens == 5000
        assert llm._context_cache_version == "env-v1"


# ---------------------------------------------------------------------------
# LLM -- context_cache_ref property
# ---------------------------------------------------------------------------


class TestContextCacheRefProperty:
    """The context_cache_ref property returns the active ref."""

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "k",
            "INFERENCE_MODEL_ID": "gemini-3.5-flash",
        },
        clear=True,
    )
    def test_none_when_no_cache(self):
        llm = LLM()
        assert llm.context_cache_ref is None

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "k",
            "INFERENCE_MODEL_ID": "gemini-3.5-flash",
        },
        clear=True,
    )
    @patch("floship_llm.google_cache_manager._require_genai")
    def test_returns_active_ref(self, mock_req):
        mock_req.return_value = (Mock(), Mock())
        llm = LLM(enable_context_cache=True)
        ref = ContextCacheRef(
            name="c/1", key="k", model="m", expires_at=time.time() + 3600
        )
        llm._active_cache_ref = ref
        assert llm.context_cache_ref is ref

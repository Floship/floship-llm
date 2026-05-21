"""Tests for Phase 4 -- Context Caching (NativeGeminiBackend)."""

import os
import time
from unittest.mock import Mock, patch

from floship_llm.backends.native_gemini import (
    NativeGeminiBackend,
)
from floship_llm.backends.openai_compat import OpenAICompatibleBackend
from floship_llm.client import LLM
from floship_llm.schemas import CacheInfo

GOOGLE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backend(cache=False, cache_ttl=3600):
    """Create a NativeGeminiBackend with mocked genai SDK."""
    with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
        mock_genai = Mock()
        mock_types = Mock()
        mock_req.return_value = (mock_genai, mock_types)
        backend = NativeGeminiBackend(
            api_key="test-key",  # pragma: allowlist secret
            model="gemini-2.5-flash",
            cache=cache,
            cache_ttl=cache_ttl,
        )
    return backend, mock_genai, mock_types


def _mock_gemini_response(text="Hello!"):
    """Build a mock Gemini GenerateContentResponse."""
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


def _mock_count_tokens(total):
    """Build a mock countTokens response."""
    result = Mock()
    result.total_tokens = total
    return result


def _mock_cached_content(name="cachedContents/abc123"):
    """Build a mock CachedContent handle."""
    cached = Mock()
    cached.name = name
    return cached


# ---------------------------------------------------------------------------
# Backend cache init tests
# ---------------------------------------------------------------------------


class TestCacheInit:
    """Tests for cache-related init parameters."""

    def test_cache_disabled_by_default(self):
        backend, _, _ = _make_backend()
        assert backend._cache_enabled is False
        assert backend._cached_content is None

    def test_cache_enabled(self):
        backend, _, _ = _make_backend(cache=True)
        assert backend._cache_enabled is True

    def test_cache_ttl_default(self):
        backend, _, _ = _make_backend(cache=True)
        assert backend._cache_ttl == 3600

    def test_cache_ttl_custom(self):
        backend, _, _ = _make_backend(cache=True, cache_ttl=7200)
        assert backend._cache_ttl == 7200


# ---------------------------------------------------------------------------
# _ensure_cache tests
# ---------------------------------------------------------------------------


class TestEnsureCache:
    """Tests for cache creation and reuse logic."""

    def test_returns_none_when_cache_disabled(self):
        backend, _, _ = _make_backend(cache=False)
        result = backend._ensure_cache(
            system_instruction="Be helpful.", tools_schema=None
        )
        assert result is None

    def test_returns_none_when_nothing_cacheable(self):
        backend, _, _ = _make_backend(cache=True)
        result = backend._ensure_cache(system_instruction=None, tools_schema=None)
        assert result is None

    def test_returns_none_when_below_min_tokens(self):
        backend, mock_genai, _ = _make_backend(cache=True)

        # Token count below threshold
        mock_genai.Client.return_value.models.count_tokens.return_value = (
            _mock_count_tokens(100)
        )

        result = backend._ensure_cache(
            system_instruction="Short prompt.", tools_schema=None
        )
        assert result is None

    def test_creates_cache_when_above_min_tokens(self):
        backend, mock_genai, _ = _make_backend(cache=True)

        mock_genai.Client.return_value.models.count_tokens.return_value = (
            _mock_count_tokens(5000)
        )
        cached = _mock_cached_content()
        mock_genai.Client.return_value.caches.create.return_value = cached

        result = backend._ensure_cache(
            system_instruction="A very long system prompt " * 200,
            tools_schema=None,
        )

        assert result is cached
        assert backend._cached_content is cached
        assert backend._cache_hash is not None
        assert backend._cache_created_at is not None
        assert backend._cache_token_count == 5000
        mock_genai.Client.return_value.caches.create.assert_called_once()

    def test_reuses_cache_when_content_unchanged(self):
        backend, mock_genai, _ = _make_backend(cache=True)

        # First call creates
        mock_genai.Client.return_value.models.count_tokens.return_value = (
            _mock_count_tokens(5000)
        )
        cached = _mock_cached_content()
        mock_genai.Client.return_value.caches.create.return_value = cached

        system_prompt = "Long system prompt " * 200
        result1 = backend._ensure_cache(
            system_instruction=system_prompt, tools_schema=None
        )
        assert result1 is cached

        # Second call reuses
        result2 = backend._ensure_cache(
            system_instruction=system_prompt, tools_schema=None
        )
        assert result2 is cached

        # create() was only called once
        assert mock_genai.Client.return_value.caches.create.call_count == 1

    def test_invalidates_cache_when_system_prompt_changes(self):
        backend, mock_genai, _ = _make_backend(cache=True)

        mock_genai.Client.return_value.models.count_tokens.return_value = (
            _mock_count_tokens(5000)
        )
        cached1 = _mock_cached_content("cachedContents/first")
        cached2 = _mock_cached_content("cachedContents/second")
        mock_genai.Client.return_value.caches.create.side_effect = [cached1, cached2]

        result1 = backend._ensure_cache(
            system_instruction="Prompt A " * 200, tools_schema=None
        )
        assert result1 is cached1

        result2 = backend._ensure_cache(
            system_instruction="Prompt B " * 200, tools_schema=None
        )
        assert result2 is cached2
        assert mock_genai.Client.return_value.caches.create.call_count == 2

    def test_invalidates_cache_when_tools_change(self):
        backend, mock_genai, _ = _make_backend(cache=True)

        mock_genai.Client.return_value.models.count_tokens.return_value = (
            _mock_count_tokens(5000)
        )
        cached1 = _mock_cached_content("cachedContents/first")
        cached2 = _mock_cached_content("cachedContents/second")
        mock_genai.Client.return_value.caches.create.side_effect = [cached1, cached2]

        tools_v1 = [
            {
                "type": "function",
                "function": {"name": "search", "description": "Search"},
            }
        ]
        tools_v2 = [
            {"type": "function", "function": {"name": "fetch", "description": "Fetch"}}
        ]

        system = "Long system " * 200
        result1 = backend._ensure_cache(
            system_instruction=system, tools_schema=tools_v1
        )
        assert result1 is cached1

        result2 = backend._ensure_cache(
            system_instruction=system, tools_schema=tools_v2
        )
        assert result2 is cached2

    def test_invalidates_cache_when_ttl_expired(self):
        backend, mock_genai, _ = _make_backend(cache=True, cache_ttl=60)

        mock_genai.Client.return_value.models.count_tokens.return_value = (
            _mock_count_tokens(5000)
        )
        cached1 = _mock_cached_content("cachedContents/first")
        cached2 = _mock_cached_content("cachedContents/second")
        mock_genai.Client.return_value.caches.create.side_effect = [cached1, cached2]

        system = "Long prompt " * 200
        result1 = backend._ensure_cache(system_instruction=system, tools_schema=None)
        assert result1 is cached1

        # Simulate TTL expiry by backdating
        backend._cache_created_at = time.monotonic() - 120  # 120s ago, TTL=60s

        result2 = backend._ensure_cache(system_instruction=system, tools_schema=None)
        assert result2 is cached2
        assert mock_genai.Client.return_value.caches.create.call_count == 2

    def test_handles_cache_creation_failure_gracefully(self):
        backend, mock_genai, _ = _make_backend(cache=True)

        mock_genai.Client.return_value.models.count_tokens.return_value = (
            _mock_count_tokens(5000)
        )
        mock_genai.Client.return_value.caches.create.side_effect = RuntimeError(
            "API error"
        )

        result = backend._ensure_cache(
            system_instruction="Long prompt " * 200, tools_schema=None
        )
        assert result is None

    def test_handles_count_tokens_failure_gracefully(self):
        backend, mock_genai, _ = _make_backend(cache=True)

        mock_genai.Client.return_value.models.count_tokens.side_effect = RuntimeError(
            "API error"
        )

        result = backend._ensure_cache(system_instruction="Prompt", tools_schema=None)
        assert result is None

    def test_cache_with_tools_only(self):
        """Cache works with tools and no system instruction."""
        backend, mock_genai, _ = _make_backend(cache=True)

        mock_genai.Client.return_value.models.count_tokens.return_value = (
            _mock_count_tokens(5000)
        )
        cached = _mock_cached_content()
        mock_genai.Client.return_value.caches.create.return_value = cached

        tools = [
            {"type": "function", "function": {"name": "fn", "description": "d " * 2000}}
        ]
        result = backend._ensure_cache(system_instruction=None, tools_schema=tools)
        assert result is cached


# ---------------------------------------------------------------------------
# clear_cache tests
# ---------------------------------------------------------------------------


class TestClearCache:
    """Tests for cache invalidation."""

    def test_clear_cache_resets_state(self):
        backend, mock_genai, _ = _make_backend(cache=True)

        # Set up a cache
        mock_genai.Client.return_value.models.count_tokens.return_value = (
            _mock_count_tokens(5000)
        )
        cached = _mock_cached_content()
        mock_genai.Client.return_value.caches.create.return_value = cached
        backend._ensure_cache(system_instruction="Long " * 200, tools_schema=None)

        assert backend._cached_content is not None

        backend.clear_cache()

        assert backend._cached_content is None
        assert backend._cache_hash is None
        assert backend._cache_created_at is None
        assert backend._cache_token_count == 0
        mock_genai.Client.return_value.caches.delete.assert_called_once_with(
            name="cachedContents/abc123"
        )

    def test_clear_cache_when_no_cache(self):
        backend, _, _ = _make_backend(cache=True)
        backend.clear_cache()  # No error

    def test_clear_cache_handles_delete_failure(self):
        backend, mock_genai, _ = _make_backend(cache=True)

        mock_genai.Client.return_value.models.count_tokens.return_value = (
            _mock_count_tokens(5000)
        )
        cached = _mock_cached_content()
        mock_genai.Client.return_value.caches.create.return_value = cached
        backend._ensure_cache(system_instruction="Long " * 200, tools_schema=None)

        mock_genai.Client.return_value.caches.delete.side_effect = RuntimeError("fail")
        backend.clear_cache()  # Should not raise
        assert backend._cached_content is None


# ---------------------------------------------------------------------------
# get_cache_info tests
# ---------------------------------------------------------------------------


class TestGetCacheInfo:
    """Tests for cache info reporting."""

    def test_returns_none_when_no_cache(self):
        backend, _, _ = _make_backend(cache=True)
        assert backend.get_cache_info() is None

    def test_returns_info_when_cached(self):
        backend, mock_genai, _ = _make_backend(cache=True, cache_ttl=3600)

        mock_genai.Client.return_value.models.count_tokens.return_value = (
            _mock_count_tokens(8000)
        )
        cached = _mock_cached_content("cachedContents/xyz")
        mock_genai.Client.return_value.caches.create.return_value = cached
        backend._ensure_cache(system_instruction="Long " * 200, tools_schema=None)

        info = backend.get_cache_info()
        assert info is not None
        assert info["cached"] is True
        assert info["token_count"] == 8000
        assert info["cache_name"] == "cachedContents/xyz"
        assert 0 < info["ttl_remaining"] <= 3600

    def test_ttl_remaining_decreases(self):
        backend, mock_genai, _ = _make_backend(cache=True, cache_ttl=100)

        mock_genai.Client.return_value.models.count_tokens.return_value = (
            _mock_count_tokens(5000)
        )
        cached = _mock_cached_content()
        mock_genai.Client.return_value.caches.create.return_value = cached
        backend._ensure_cache(system_instruction="Long " * 200, tools_schema=None)

        # Simulate 50 seconds elapsed
        backend._cache_created_at = time.monotonic() - 50
        info = backend.get_cache_info()
        assert info["ttl_remaining"] <= 50

    def test_ttl_remaining_zero_when_expired(self):
        backend, mock_genai, _ = _make_backend(cache=True, cache_ttl=60)

        mock_genai.Client.return_value.models.count_tokens.return_value = (
            _mock_count_tokens(5000)
        )
        cached = _mock_cached_content()
        mock_genai.Client.return_value.caches.create.return_value = cached
        backend._ensure_cache(system_instruction="Long " * 200, tools_schema=None)

        backend._cache_created_at = time.monotonic() - 120
        info = backend.get_cache_info()
        assert info["ttl_remaining"] == 0


# ---------------------------------------------------------------------------
# Chat with cache integration tests
# ---------------------------------------------------------------------------


class TestChatWithCache:
    """Tests for chat() using context cache."""

    def test_chat_uses_cache_when_above_threshold(self):
        backend, mock_genai, mock_types = _make_backend(cache=True)

        # Token count above threshold
        mock_genai.Client.return_value.models.count_tokens.return_value = (
            _mock_count_tokens(5000)
        )
        cached = _mock_cached_content("cachedContents/abc123")
        mock_genai.Client.return_value.caches.create.return_value = cached

        mock_genai.Client.return_value.models.generate_content.return_value = (
            _mock_gemini_response("Cached response!")
        )

        result = backend.chat(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": "Very long system " * 200},
                {"role": "user", "content": "Hi"},
            ],
            temperature=0.5,
        )

        assert result.choices[0].message.content == "Cached response!"

        # Verify GenerateContentConfig was called with cached_content
        config_call = mock_types.GenerateContentConfig.call_args
        kwargs = config_call[1]
        assert kwargs.get("cached_content") == "cachedContents/abc123"
        # system_instruction should NOT be in config when using cache
        assert kwargs.get("system_instruction") is None

    def test_chat_skips_cache_when_below_threshold(self):
        backend, mock_genai, mock_types = _make_backend(cache=True)

        # Token count below threshold
        mock_genai.Client.return_value.models.count_tokens.return_value = (
            _mock_count_tokens(100)
        )

        mock_genai.Client.return_value.models.generate_content.return_value = (
            _mock_gemini_response("Direct response!")
        )

        result = backend.chat(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": "Short prompt."},
                {"role": "user", "content": "Hi"},
            ],
        )

        assert result.choices[0].message.content == "Direct response!"

        # Verify no cached_content in config
        config_call = mock_types.GenerateContentConfig.call_args
        kwargs = config_call[1]
        assert "cached_content" not in kwargs or kwargs.get("cached_content") is None

    def test_chat_without_cache_enabled(self):
        backend, mock_genai, _ = _make_backend(cache=False)

        mock_genai.Client.return_value.models.generate_content.return_value = (
            _mock_gemini_response("No cache!")
        )

        result = backend.chat(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": "System " * 200},
                {"role": "user", "content": "Hi"},
            ],
        )

        assert result.choices[0].message.content == "No cache!"
        # count_tokens should not have been called
        mock_genai.Client.return_value.models.count_tokens.assert_not_called()
        # caches.create should not have been called
        mock_genai.Client.return_value.caches.create.assert_not_called()


# ---------------------------------------------------------------------------
# OpenAI-compatible backend stubs
# ---------------------------------------------------------------------------


class TestOpenAICompatCacheStubs:
    """Tests for cache method stubs on OpenAICompatibleBackend."""

    def test_get_cache_info_returns_none(self):
        backend = OpenAICompatibleBackend(client=Mock(), provider="heroku")
        assert backend.get_cache_info() is None

    def test_clear_cache_noop(self):
        backend = OpenAICompatibleBackend(client=Mock(), provider="heroku")
        backend.clear_cache()  # Should not raise

    def test_supports_caching_false(self):
        backend = OpenAICompatibleBackend(client=Mock(), provider="heroku")
        assert backend.supports_caching is False


# ---------------------------------------------------------------------------
# CacheInfo dataclass tests
# ---------------------------------------------------------------------------


class TestCacheInfoDataclass:
    """Tests for the CacheInfo schema."""

    def test_create_cache_info(self):
        info = CacheInfo(
            cached=True,
            token_count=5000,
            ttl_remaining=3540,
            cache_name="cachedContents/abc",
        )
        assert info.cached is True
        assert info.token_count == 5000
        assert info.ttl_remaining == 3540
        assert info.cache_name == "cachedContents/abc"


# ---------------------------------------------------------------------------
# LLM integration tests
# ---------------------------------------------------------------------------


class TestLLMCacheIntegration:
    """Tests for LLM with cache=True."""

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "test-key",
            "INFERENCE_MODEL_ID": "gemini-2.5-flash",
        },
    )
    def test_cache_params_forwarded_to_backend(self):
        """cache and cache_ttl are forwarded to NativeGeminiBackend."""
        with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
            mock_req.return_value = (Mock(), Mock())
            llm = LLM(native_google=True, cache=True, cache_ttl=7200)
            assert isinstance(llm.backend, NativeGeminiBackend)
            assert llm.backend._cache_enabled is True
            assert llm.backend._cache_ttl == 7200

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "test-key",
            "INFERENCE_MODEL_ID": "gemini-2.5-flash",
            "GEMINI_CACHE": "true",
            "GEMINI_CACHE_TTL": "1800",
        },
    )
    def test_cache_env_vars(self):
        """GEMINI_CACHE and GEMINI_CACHE_TTL env vars work."""
        with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
            mock_req.return_value = (Mock(), Mock())
            llm = LLM(native_google=True)
            assert llm.backend._cache_enabled is True
            assert llm.backend._cache_ttl == 1800

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "test-key",
            "INFERENCE_MODEL_ID": "gemini-2.5-flash",
        },
    )
    def test_cache_info_property_none_when_no_cache(self):
        """cache_info returns None when no cache is active."""
        with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
            mock_req.return_value = (Mock(), Mock())
            llm = LLM(native_google=True, cache=True)
            assert llm.cache_info is None

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "test-key",
            "INFERENCE_MODEL_ID": "gemini-2.5-flash",
        },
    )
    def test_cache_info_property_returns_cache_info(self):
        """cache_info returns CacheInfo when cache is active."""
        with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
            mock_genai = Mock()
            mock_types = Mock()
            mock_req.return_value = (mock_genai, mock_types)

            llm = LLM(native_google=True, cache=True)

            mock_genai.Client.return_value.models.count_tokens.return_value = (
                _mock_count_tokens(5000)
            )
            cached = _mock_cached_content("cachedContents/test")
            mock_genai.Client.return_value.caches.create.return_value = cached

            # Trigger cache creation
            llm.backend._ensure_cache(
                system_instruction="Long prompt " * 200, tools_schema=None
            )

            info = llm.cache_info
            assert isinstance(info, CacheInfo)
            assert info.cached is True
            assert info.cache_name == "cachedContents/test"
            assert info.token_count == 5000

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "test-key",
            "INFERENCE_MODEL_ID": "gemini-2.5-flash",
        },
    )
    def test_clear_cache_on_llm(self):
        """LLM.clear_cache() delegates to backend."""
        with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
            mock_genai = Mock()
            mock_types = Mock()
            mock_req.return_value = (mock_genai, mock_types)

            llm = LLM(native_google=True, cache=True)

            mock_genai.Client.return_value.models.count_tokens.return_value = (
                _mock_count_tokens(5000)
            )
            cached = _mock_cached_content("cachedContents/del")
            mock_genai.Client.return_value.caches.create.return_value = cached
            llm.backend._ensure_cache(
                system_instruction="Long prompt " * 200, tools_schema=None
            )

            llm.clear_cache()
            assert llm.backend._cached_content is None
            assert llm.cache_info is None

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "test-key",
            "INFERENCE_MODEL_ID": "gemini-2.5-flash",
        },
    )
    def test_cache_info_none_for_openai_compat(self):
        """cache_info returns None for OpenAI-compatible backend."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
            assert llm.cache_info is None

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "test-key",
            "INFERENCE_MODEL_ID": "gemini-2.5-flash",
        },
    )
    def test_clear_cache_noop_for_openai_compat(self):
        """clear_cache() is a no-op for OpenAI-compatible backend."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
            llm.clear_cache()  # Should not raise

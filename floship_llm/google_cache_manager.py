"""Google Gemini context cache lifecycle manager.

Uses the native ``google-genai`` SDK to create, refresh, and delete
cached content.  The cache reference is then passed to the
OpenAI-compatible chat endpoint via ``extra_body.cached_content``,
keeping the chat path unchanged.

Activated when ``provider == "google"`` and ``enable_context_cache=True``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Gemini requires at least this many input tokens for caching to be worthwhile.
_DEFAULT_MIN_TOKENS = 8_000

# Pre-expire local refs this many seconds before actual expiry.
_REFRESH_MARGIN_SECONDS = 60


def _require_genai():
    """Lazily import google-genai, raising a clear error if missing."""
    try:
        from google import genai
        from google.genai import types

        return genai, types
    except ImportError:
        raise ImportError(
            "Google context caching requires the google-genai package.\n"
            "Install with:  pip install floship-llm[google]"
        )


class ContextCacheRef:
    """Lightweight handle to a remote cached-content resource."""

    __slots__ = ("expires_at", "key", "model", "name", "token_count")

    def __init__(
        self,
        *,
        name: str,
        key: str,
        model: str,
        expires_at: float,
        token_count: int | None = None,
    ) -> None:
        self.name = name
        self.key = key
        self.model = model
        self.expires_at = expires_at
        self.token_count = token_count

    def is_valid(self) -> bool:
        """Return ``True`` if the cache has not expired (with margin)."""
        return self.expires_at > time.time() + _REFRESH_MARGIN_SECONDS


class GoogleCacheManager:
    """Manage Gemini cached-content lifecycle.

    This class only handles cache CRUD via the native ``google-genai``
    SDK.  The actual chat request still goes through the
    OpenAI-compatible endpoint with ``extra_body.cached_content``.
    """

    def __init__(
        self,
        *,
        api_key: str,
        default_ttl_seconds: int = 300,
        min_tokens: int = _DEFAULT_MIN_TOKENS,
    ) -> None:
        genai, self._types = _require_genai()
        self._client = genai.Client(api_key=api_key)
        self.default_ttl_seconds = default_ttl_seconds
        self.min_tokens = min_tokens
        self._local_cache: dict[str, ContextCacheRef] = {}

    # -- key helpers --------------------------------------------------------

    @staticmethod
    def make_key(
        *,
        model: str,
        system: str | None = None,
        contents: list[Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        version: str = "",
        permission_hash: str | None = None,
    ) -> str:
        """Deterministic SHA-256 key for a set of cacheable content."""
        payload = {
            "model": model,
            "system": system or "",
            "contents": contents or [],
            "tools": tools or [],
            "version": version,
            "permission_hash": permission_hash or "",
        }
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # -- local ref lookup ---------------------------------------------------

    def get_local(self, key: str) -> ContextCacheRef | None:
        """Return a still-valid local ref, or ``None``."""
        ref = self._local_cache.get(key)
        if ref is None:
            return None
        if not ref.is_valid():
            self._local_cache.pop(key, None)
            return None
        return ref

    # -- CRUD ---------------------------------------------------------------

    def create_or_get(
        self,
        *,
        model: str,
        system: str | None = None,
        contents: list[Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        version: str = "",
        ttl_seconds: int | None = None,
        display_name: str | None = None,
        permission_hash: str | None = None,
    ) -> ContextCacheRef:
        """Return an existing (still-valid) cache ref or create one."""
        ttl = ttl_seconds or self.default_ttl_seconds
        key = self.make_key(
            model=model,
            system=system,
            contents=contents,
            tools=tools,
            version=version,
            permission_hash=permission_hash,
        )

        existing = self.get_local(key)
        if existing is not None:
            return existing

        types = self._types
        config_kwargs: dict[str, Any] = {
            "display_name": display_name or f"floship-llm-{key[:12]}",
            "ttl": f"{ttl}s",
        }
        if system:
            config_kwargs["system_instruction"] = system
        if contents:
            config_kwargs["contents"] = contents
        if tools:
            config_kwargs["tools"] = tools

        cache = self._client.caches.create(
            model=model,
            config=types.CreateCachedContentConfig(**config_kwargs),
        )

        usage = getattr(cache, "usage_metadata", None)
        token_count = getattr(usage, "total_token_count", None) if usage else None

        ref = ContextCacheRef(
            name=cache.name,
            key=key,
            model=model,
            expires_at=time.time() + ttl,
            token_count=token_count,
        )
        self._local_cache[key] = ref
        logger.info(
            "Created context cache %s (%s tokens, TTL %ds)",
            ref.name,
            token_count or "?",
            ttl,
        )
        return ref

    def update_ttl(self, name: str, ttl_seconds: int) -> None:
        """Extend or shorten the TTL of an existing cache."""
        types = self._types
        self._client.caches.update(
            name=name,
            config=types.UpdateCachedContentConfig(ttl=f"{ttl_seconds}s"),
        )

    def delete(self, name: str) -> None:
        """Delete a cached-content resource by name."""
        try:
            self._client.caches.delete(name=name)
        except Exception:
            logger.debug("Cache delete failed for %s (may already be gone)", name)

    def clear_all(self) -> None:
        """Delete all locally-tracked caches and clear the local map."""
        for ref in list(self._local_cache.values()):
            self.delete(ref.name)
        self._local_cache.clear()

    # -- cost helpers -------------------------------------------------------

    @staticmethod
    def should_cache(
        static_token_count: int,
        expected_reuse: int,
        ttl_hours: float,
        *,
        min_tokens: int = _DEFAULT_MIN_TOKENS,
    ) -> bool:
        """Return ``True`` if caching is cost-effective.

        Uses Gemini Flash 3.5 Standard pricing:
        - Normal input: $1.50 / 1M tokens
        - Cached input: $0.15 / 1M tokens
        - Storage:      $1.00 / 1M tokens / hour

        Break-even: ``N > (1.50 + hours) / (1.50 - 0.15)``
        """
        if static_token_count < min_tokens:
            return False
        breakeven_calls = (1.50 + ttl_hours) / (1.50 - 0.15)
        return expected_reuse > breakeven_calls

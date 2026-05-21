"""OpenAI-compatible provider backend.

Thin wrapper around the ``openai.OpenAI`` client that satisfies the
``ProviderBackend`` interface.  Used for Heroku Inference, Google AI
(via its ``/openai/`` compatibility endpoint), and any other provider
that speaks the OpenAI wire protocol.
"""

from __future__ import annotations

from typing import Any

from floship_llm.backends.base import ProviderBackend


class OpenAICompatibleBackend(ProviderBackend):
    """Backend that delegates to an ``openai.OpenAI`` client instance."""

    def __init__(self, *, client: Any, provider: str) -> None:
        self._client = client
        self._provider = provider

    # -- ProviderBackend interface ------------------------------------------

    def chat(self, **kwargs: Any) -> Any:
        """Proxy to ``client.chat.completions.create``."""
        return self._client.chat.completions.create(**kwargs)

    def embed(self, **kwargs: Any) -> Any:
        """Proxy to ``client.embeddings.create``."""
        return self._client.embeddings.create(**kwargs)

    @property
    def provider_name(self) -> str:
        return self._provider

    @property
    def supports_caching(self) -> bool:
        return False

    @property
    def supports_native_tools(self) -> bool:
        return False

    @property
    def client(self) -> Any:
        """Underlying ``openai.OpenAI`` client for backward compatibility."""
        return self._client

    @client.setter
    def client(self, value: Any) -> None:
        self._client = value

    def get_cache_info(self) -> None:
        """OpenAI-compatible backends do not support context caching."""
        return None

    def clear_cache(self) -> None:
        """No-op for non-caching backends."""
        pass

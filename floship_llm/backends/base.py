"""Abstract base class for inference provider backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional


class ProviderBackend(ABC):
    """Base class for inference provider backends.

    Each backend wraps a specific SDK or protocol for communicating
    with an LLM inference endpoint.  The ``LLM`` orchestrator delegates
    raw API calls to the active backend while keeping conversation
    management, tool loops, and retry logic in the orchestrator.
    """

    @abstractmethod
    def chat(self, **kwargs: Any) -> Any:
        """Send a chat completion request.

        Handles both streaming (``stream=True`` in *kwargs*) and
        non-streaming modes.  Returns the raw SDK response object.
        """

    @abstractmethod
    def embed(self, **kwargs: Any) -> Any:
        """Send an embedding request.  Returns the raw SDK response."""

    def decide(
        self,
        *,
        state: Any,
        questions: Mapping[str, Any],
        model: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Any:
        """Send a System One (decisions) request.

        Returns a 'DecisionsResponse'.  The default implementation raises
        'NotImplementedError'; only backends that speak a decisions protocol
        override this.  System One is not part of the OpenAI wire protocol, so
        most backends never implement it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support System One decisions"
        )

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider identifier (e.g. ``'heroku'``, ``'google'``)."""

    @property
    @abstractmethod
    def supports_caching(self) -> bool:
        """Whether this backend supports server-side context caching."""

    @property
    @abstractmethod
    def supports_native_tools(self) -> bool:
        """Whether tools use a native (non-OpenAI) schema."""

    def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens for *messages*.

        Default implementation raises ``NotImplementedError``; backends
        with native token-counting endpoints should override this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support native token counting"
        )

    @property
    def supports_file_upload(self) -> bool:
        """Whether this backend supports native file upload."""
        return False

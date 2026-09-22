"""System One (TypeSafe Jev) backend.

Jev is not reachable through the OpenAI-compatible surface.  A decisions
request carries a 'state' and a set of 'questions' and returns typed answers,
so it needs its own transport rather than a chat completion.

The documented route is a POST to '/api/alpha/decisions' on the OpenRouter
host.  Note that the path is NOT under the OpenAI-compatible '/api/v1'
prefix, so the endpoint is derived from the base URL's scheme and host only.

Chat completions and embeddings do not apply to this backend.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlsplit

import httpx

from floship_llm.backends.base import ProviderBackend

DECISIONS_PATH = "/api/alpha/decisions"
DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BACKOFF_SECONDS = 0.5

# Statuses the API documents as transient.  A model blocked by a workspace
# guardrail is NOT among them: OpenRouter reports that as a 404.
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 524, 529})


class SystemOneError(RuntimeError):
    """A decisions request failed.

    The message carries the server's own text, which is what makes a guardrail
    or data-policy rejection legible: one workspace may reach a model that
    another workspace blocks.  A guardrail rejection arrives as HTTP 404 with
    the guardrail URL in the message, so do not read 404 as "no such model"
    without checking the text.

    Attributes:
        status: HTTP status code, when a response was received.
        code: the error code from the response envelope, when present.
        url: the endpoint that was called.
        body: the parsed response body, when it was JSON.
    """

    def __init__(
        self,
        message: str,
        *,
        status: Optional[int] = None,
        code: Any = None,
        url: Optional[str] = None,
        body: Any = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.url = url
        self.body = body


def decisions_url(base_url: str) -> str:
    """Return the decisions endpoint for an OpenAI-compatible base URL.

    Only the scheme and host are kept, because the decisions path is not under
    the OpenAI-compatible prefix.

    Raises:
        ValueError: if 'base_url' is not absolute.
    """
    parts = urlsplit(base_url or "")
    if not parts.scheme or not parts.netloc:
        raise ValueError(
            "A decisions request needs an absolute base URL such as "
            f"'https://openrouter.ai/api/v1'; got {base_url!r}"
        )
    return f"{parts.scheme}://{parts.netloc}{DECISIONS_PATH}"


def question_to_wire(question: Any) -> Dict[str, Any]:
    """Return one question in the request wire format.

    Accepts a 'Noul', 'Choice', or 'Score', or a mapping that already matches
    the wire format.

    Raises:
        TypeError: if the question is neither.
    """
    to_wire = getattr(question, "to_wire", None)
    if callable(to_wire):
        return to_wire()
    if isinstance(question, Mapping):
        return dict(question)
    raise TypeError(
        "A question must be a Noul, Choice, or Score, or a mapping in the "
        f"wire format; got {type(question).__name__}"
    )


class SystemOneBackend(ProviderBackend):
    """Send decisions requests and return typed answers.

    Args:
        api_key: bearer token for the endpoint.
        model: model id, for example 'typesafe/jev-1.13'.
        base_url: the OpenAI-compatible base URL in use; the decisions
            endpoint is derived from its scheme and host.
        endpoint: full URL of the decisions endpoint, overriding the
            derivation.
        timeout: per-attempt timeout in seconds.
        max_attempts: total attempts for a retryable failure, at least 1.
        backoff: seconds before the first retry; the delay doubles per
            attempt.  Zero disables the wait, which is what tests want.
        client: an 'httpx.Client' to reuse.  When omitted the backend creates
            one on first use and closes it in 'close()'.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str],
        model: str,
        base_url: str = "",
        endpoint: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        backoff: float = DEFAULT_BACKOFF_SECONDS,
        client: Optional[httpx.Client] = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._endpoint = endpoint or decisions_url(base_url)
        self._timeout = timeout
        self._max_attempts = max(1, int(max_attempts))
        self._backoff = backoff
        self._client = client
        self._owns_client = client is None

    # -- Decisions ---------------------------------------------------------

    def decide(
        self,
        *,
        state: Any,
        questions: Mapping[str, Any],
        model: Optional[str] = None,
        session_id: Optional[str] = None,
        provider: Optional[Mapping[str, Any]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
    ) -> Any:
        """Send one decisions request and return a 'DecisionsResponse'.

        Every question travels in one request against the same state and is
        evaluated in parallel, so a question that only some branches need
        costs little to include.

        Raises:
            SystemOneError: if the endpoint rejects the request or the
                transport fails on every attempt.
        """
        from floship_llm.systemone import DecisionsResponse

        payload: Dict[str, Any] = {
            "model": model or self._model,
            "state": state,
            "questions": {
                str(name): question_to_wire(question)
                for name, question in questions.items()
            },
        }
        if session_id is not None:
            payload["session_id"] = session_id
        if provider is not None:
            payload["provider"] = dict(provider)
        if extra_body:
            payload.update(extra_body)

        return DecisionsResponse.from_wire(self._post(payload))

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST the payload, retrying a transient failure."""
        headers = {
            "Authorization": f"Bearer {self._api_key or ''}",
            "Content-Type": "application/json",
        }
        last_error: Optional[SystemOneError] = None
        for attempt in range(1, self._max_attempts + 1):
            is_last = attempt == self._max_attempts
            try:
                response = self._request(payload, headers)
            except httpx.HTTPError as error:
                last_error = SystemOneError(
                    f"Decisions request failed: {error}", url=self._endpoint
                )
                if is_last:
                    raise last_error from error
            else:
                if response.status_code < 400:
                    return self._json(response)
                last_error = self._error_from(response)
                if is_last or response.status_code not in RETRYABLE_STATUS:
                    raise last_error
            self._pause(attempt)
        # Unreachable: the final attempt always raises.
        raise last_error or SystemOneError("Decisions request failed")

    def _request(
        self, payload: Dict[str, Any], headers: Dict[str, str]
    ) -> httpx.Response:
        """Perform one HTTP attempt."""
        return self._client_for_use().post(
            self._endpoint, headers=headers, json=payload, timeout=self._timeout
        )

    def _client_for_use(self) -> httpx.Client:
        """Return the HTTP client, creating the owned one on first use."""
        if self._client is None:
            self._client = httpx.Client()
        return self._client

    def _pause(self, attempt: int) -> None:
        """Wait before the next attempt, doubling per attempt."""
        if self._backoff > 0:
            time.sleep(self._backoff * (2 ** (attempt - 1)))

    def _json(self, response: httpx.Response) -> Dict[str, Any]:
        """Return the response body as a mapping."""
        try:
            body = response.json()
        except ValueError as error:
            raise SystemOneError(
                f"Decisions response was not JSON (HTTP {response.status_code}): "
                f"{response.text[:200]}",
                status=response.status_code,
                url=self._endpoint,
            ) from error
        if not isinstance(body, dict):
            raise SystemOneError(
                "Decisions response was not a JSON object "
                f"(HTTP {response.status_code})",
                status=response.status_code,
                url=self._endpoint,
                body=body,
            )
        return body

    def _error_from(self, response: httpx.Response) -> SystemOneError:
        """Build an error from an error response."""
        message = f"HTTP {response.status_code}"
        code: Any = None
        body: Any = None
        try:
            body = response.json()
        except ValueError:
            body = None
        if isinstance(body, dict):
            envelope = body.get("error")
            if isinstance(envelope, dict):
                code = envelope.get("code")
                if envelope.get("message"):
                    message = str(envelope["message"])
        elif response.text:
            message = f"{message}: {response.text[:200]}"
        return SystemOneError(
            message,
            status=response.status_code,
            code=code,
            url=self._endpoint,
            body=body,
        )

    # -- ProviderBackend interface ----------------------------------------

    def chat(self, **kwargs: Any) -> Any:
        """Not supported: Jev returns typed answers, not messages."""
        raise NotImplementedError(
            "SystemOneBackend does not serve chat completions. Use LLM.prompt() "
            "for chat, or LLM.decisions() for System One."
        )

    def embed(self, **kwargs: Any) -> Any:
        """Not supported: System One has no embedding endpoint."""
        raise NotImplementedError(
            "SystemOneBackend does not serve embeddings. Use LLM.embed()."
        )

    @property
    def provider_name(self) -> str:
        return "systemone"

    @property
    def supports_caching(self) -> bool:
        return False

    @property
    def supports_native_tools(self) -> bool:
        return False

    @property
    def endpoint(self) -> str:
        """The decisions endpoint this backend posts to."""
        return self._endpoint

    def close(self) -> None:
        """Close the owned HTTP client, if there is one."""
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None

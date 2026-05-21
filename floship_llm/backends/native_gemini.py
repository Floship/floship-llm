"""Native Google Gemini provider backend using the ``google-genai`` SDK.

Activated via ``native_google=True`` kwarg or ``GEMINI_NATIVE=true`` env var.
Unlocks context caching, token counting, file upload, and grounding that
are unavailable through the OpenAI-compatible endpoint.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from types import SimpleNamespace
from typing import Any, Iterator

from floship_llm.backends.base import ProviderBackend

logger = logging.getLogger(__name__)

# Gemini requires at least this many tokens for cached content.
_MIN_CACHE_TOKENS = 4096


def _require_genai():
    """Lazily import google-genai, raising a clear error if missing."""
    try:
        from google import genai
        from google.genai import types

        return genai, types
    except ImportError:
        raise ImportError(
            "Native Gemini backend requires the google-genai package.\n"
            "Install with:  pip install floship-llm[google]"
        )


# ---------------------------------------------------------------------------
# Response wrappers -- thin objects that look like openai response types
# so the LLM orchestrator can process them without changes.
# ---------------------------------------------------------------------------


def _make_function(name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, arguments=arguments)


def _make_tool_call(tc_id: str, name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=tc_id,
        type="function",
        function=_make_function(name, arguments),
    )


def _make_message(
    content: str | None,
    tool_calls: list[SimpleNamespace] | None = None,
    role: str = "assistant",
) -> SimpleNamespace:
    return SimpleNamespace(content=content, tool_calls=tool_calls or None, role=role)


def _make_response(
    content: str | None,
    tool_calls: list[SimpleNamespace] | None = None,
) -> SimpleNamespace:
    finish_reason = "tool_calls" if tool_calls else "stop"
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=_make_message(content, tool_calls),
                finish_reason=finish_reason,
            )
        ]
    )


def _make_stream_chunk(
    content: str | None = None,
    tool_calls: list[SimpleNamespace] | None = None,
    finish_reason: str | None = None,
) -> SimpleNamespace:
    delta = SimpleNamespace(content=content, tool_calls=tool_calls or None)
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)]
    )


def _make_embed_item(index: int, embedding: list[float]) -> SimpleNamespace:
    return SimpleNamespace(object="embedding", index=index, embedding=embedding)


def _make_embed_response(embeddings: list[list[float]], model: str) -> SimpleNamespace:
    data = [_make_embed_item(i, emb) for i, emb in enumerate(embeddings)]
    return SimpleNamespace(
        object="list",
        data=data,
        model=model,
        usage=SimpleNamespace(prompt_tokens=0, total_tokens=0),
    )


# ---------------------------------------------------------------------------
# Backend implementation
# ---------------------------------------------------------------------------


class NativeGeminiBackend(ProviderBackend):
    """Backend that uses the ``google-genai`` SDK directly."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str,
        cache: bool = False,
        cache_ttl: int = 3600,
        safety_settings: dict[str, str] | None = None,
        grounding: bool = False,
        code_execution: bool = False,
        vertex: bool = False,
        project: str | None = None,
        location: str | None = None,
    ) -> None:
        genai, _types = _require_genai()
        self._genai = genai
        self._types = _types
        self._vertex = vertex

        if vertex:
            self._client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
            )
        else:
            self._client = genai.Client(api_key=api_key)
        self._model = model

        # Context caching state
        self._cache_enabled = cache
        self._cache_ttl = cache_ttl
        self._cached_content: Any = None
        self._cache_hash: str | None = None
        self._cache_created_at: float | None = None
        self._cache_token_count: int = 0

        # Phase 5 -- native-only features
        self._safety_settings = safety_settings
        self._grounding = grounding
        self._code_execution = code_execution

    # -- ProviderBackend interface ------------------------------------------

    def chat(self, **kwargs: Any) -> Any:
        """Send a chat completion request via native Gemini API.

        Accepts OpenAI-shaped kwargs (``model``, ``messages``, ``stream``,
        ``temperature``, ``tools``, etc.) and translates them into native
        ``google-genai`` calls.  Returns OpenAI-shaped response objects.
        """
        messages = kwargs.pop("messages", [])
        stream = kwargs.pop("stream", False)
        model = kwargs.pop("model", self._model)
        tools_schema = kwargs.pop("tools", None)
        tool_choice = kwargs.pop("tool_choice", None)

        contents, system_instruction = self._to_gemini_contents(messages)

        # Try to use context cache for system instruction + tools
        cache = self._ensure_cache(
            system_instruction=system_instruction,
            tools_schema=tools_schema,
        )

        if cache:
            # When using cached content, system_instruction and tools are
            # already part of the cache -- omit them from the config.
            config = self._build_generate_config(
                system_instruction=None,
                tools_schema=None,
                tool_choice=tool_choice,
                cached_content=cache.name,
                **kwargs,
            )
        else:
            config = self._build_generate_config(
                system_instruction=system_instruction,
                tools_schema=tools_schema,
                tool_choice=tool_choice,
                **kwargs,
            )

        if stream:
            return self._stream_response(model, contents, config)
        else:
            response = self._client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
            return self._normalize_response(response)

    def embed(self, **kwargs: Any) -> Any:
        """Send an embedding request via native Gemini API."""
        model = kwargs.pop("model", self._model)
        input_data = kwargs.pop("input", None)

        if isinstance(input_data, str):
            input_data = [input_data]

        result = self._client.models.embed_content(
            model=model,
            contents=input_data,
        )
        embeddings = [e.values for e in result.embeddings]
        return _make_embed_response(embeddings, model)

    @property
    def provider_name(self) -> str:
        return "google_native"

    @property
    def supports_caching(self) -> bool:
        return True

    @property
    def supports_native_tools(self) -> bool:
        return True

    @property
    def supports_file_upload(self) -> bool:
        """Whether this backend supports native file upload."""
        return True

    def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Count tokens using the native Gemini countTokens API."""
        contents, system_instruction = self._to_gemini_contents(messages)
        config = None
        if system_instruction:
            config = self._types.GenerateContentConfig(
                system_instruction=system_instruction,
            )
        result = self._client.models.count_tokens(
            model=self._model,
            contents=contents,
            config=config,
        )
        return result.total_tokens

    def upload_file(self, path: str, mime_type: str | None = None) -> Any:
        """Upload a file via the Gemini media.upload API.

        Args:
            path: Local file path to upload.
            mime_type: Optional MIME type. Auto-detected if omitted.

        Returns:
            A file reference object usable in prompt content.
        """
        upload_kwargs: dict[str, Any] = {"path": path}
        if mime_type:
            upload_kwargs["mime_type"] = mime_type
        return self._client.files.upload(**upload_kwargs)

    # -- Context caching ----------------------------------------------------

    def _ensure_cache(
        self,
        system_instruction: str | None,
        tools_schema: list[dict[str, Any]] | None,
    ) -> Any:
        """Create or reuse a context cache for the system instruction + tools.

        Returns the ``CachedContent`` handle if caching is active and the
        content meets the minimum token threshold, otherwise ``None``.
        """
        if not self._cache_enabled:
            return None

        # Nothing cacheable
        if not system_instruction and not tools_schema:
            return None

        # Hash the cacheable content for invalidation detection
        content_hash = hashlib.sha256(
            json.dumps(
                {"system": system_instruction, "tools": tools_schema},
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()

        # Reuse existing cache if content hasn't changed and TTL is valid
        if (
            self._cached_content
            and self._cache_hash == content_hash
            and self._cache_created_at
            and (time.monotonic() - self._cache_created_at) < self._cache_ttl
        ):
            return self._cached_content

        # Count tokens to check minimum threshold
        cache_parts: list[str] = []
        if system_instruction:
            cache_parts.append(system_instruction)
        if tools_schema:
            cache_parts.append(json.dumps(tools_schema))
        cache_text = "\n".join(cache_parts)

        try:
            token_count_result = self._client.models.count_tokens(
                model=self._model,
                contents=[cache_text],
            )
            token_count = token_count_result.total_tokens
        except Exception:
            logger.debug("Token counting for cache failed, skipping cache")
            return None

        if token_count < _MIN_CACHE_TOKENS:
            logger.debug(
                "Cache content has %d tokens (minimum %d). Skipping cache.",
                token_count,
                _MIN_CACHE_TOKENS,
            )
            return None

        # Build cache config
        types = self._types
        cache_config_kwargs: dict[str, Any] = {
            "ttl": f"{self._cache_ttl}s",
        }
        if system_instruction:
            cache_config_kwargs["system_instruction"] = system_instruction
        if tools_schema:
            cache_config_kwargs["tools"] = self._to_gemini_tools(tools_schema)

        try:
            self._cached_content = self._client.caches.create(
                model=self._model,
                config=types.CreateCachedContentConfig(**cache_config_kwargs),
            )
            self._cache_hash = content_hash
            self._cache_created_at = time.monotonic()
            self._cache_token_count = token_count
            logger.info(
                "Created context cache with %d tokens, TTL %ds",
                token_count,
                self._cache_ttl,
            )
            return self._cached_content
        except Exception as exc:
            logger.warning("Failed to create context cache: %s", exc)
            return None

    def clear_cache(self) -> None:
        """Invalidate the current context cache."""
        if self._cached_content:
            try:
                self._client.caches.delete(name=self._cached_content.name)
            except Exception:  # nosec B110
                pass  # Best-effort cleanup; failure is harmless
        self._cached_content = None
        self._cache_hash = None
        self._cache_created_at = None
        self._cache_token_count = 0

    def get_cache_info(self) -> dict[str, Any] | None:
        """Return info about the current cache, or None if not cached."""
        if not self._cached_content or not self._cache_created_at:
            return None
        elapsed = time.monotonic() - self._cache_created_at
        ttl_remaining = max(0, int(self._cache_ttl - elapsed))
        return {
            "cached": True,
            "token_count": self._cache_token_count,
            "ttl_remaining": ttl_remaining,
            "cache_name": self._cached_content.name,
        }

    # -- Message conversion -------------------------------------------------

    def _to_gemini_contents(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[Any], str | None]:
        """Convert OpenAI-style messages to Gemini Content objects.

        Returns:
            Tuple of (contents list, system_instruction string or None).
        """
        types = self._types
        contents: list[Any] = []
        system_instruction: str | None = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # System -> system_instruction (not a Content turn)
            if role == "system":
                system_instruction = (
                    content if isinstance(content, str) else str(content)
                )
                continue

            # Tool result -> function response
            if role == "tool":
                parts = [
                    types.Part.from_function_response(
                        name=msg.get("name", msg.get("tool_call_id", "")),
                        response={"result": content if content else ""},
                    )
                ]
                contents.append(types.Content(role="user", parts=parts))
                continue

            gemini_role = "model" if role == "assistant" else "user"
            parts = []

            # Text / multimodal content
            if isinstance(content, str):
                if content:
                    parts.append(types.Part.from_text(text=content))
            elif isinstance(content, list):
                for item in content:
                    item_type = item.get("type", "")
                    if item_type == "text":
                        text = item.get("text", "")
                        if text:
                            parts.append(types.Part.from_text(text=text))
                    elif item_type == "image_url":
                        parts.append(self._convert_image_part(item))

            # Tool calls in assistant messages -> function_call parts
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", tc) if isinstance(tc, dict) else tc
                    fn_name = (
                        fn.get("name", "") if isinstance(fn, dict) else fn.function.name
                    )
                    fn_args_raw = (
                        fn.get("arguments", "{}")
                        if isinstance(fn, dict)
                        else fn.function.arguments
                    )
                    try:
                        fn_args = (
                            json.loads(fn_args_raw)
                            if isinstance(fn_args_raw, str)
                            else fn_args_raw
                        )
                    except json.JSONDecodeError:
                        fn_args = {}

                    parts.append(
                        types.Part.from_function_call(name=fn_name, args=fn_args)
                    )

            # Gemini requires at least one part per Content
            if not parts:
                parts.append(types.Part.from_text(text="."))

            contents.append(types.Content(role=gemini_role, parts=parts))

        return contents, system_instruction

    def _convert_image_part(self, item: dict[str, Any]) -> Any:
        """Convert an OpenAI image_url content part to a Gemini Part."""
        types = self._types
        url = item.get("image_url", {}).get("url", "")
        if url.startswith("data:"):
            # Inline base64
            import base64

            header, b64data = url.split(",", 1)
            mime = header.split(";")[0].split(":")[1] if ":" in header else "image/png"
            return types.Part.from_bytes(data=base64.b64decode(b64data), mime_type=mime)
        else:
            return types.Part.from_uri(file_uri=url, mime_type="image/jpeg")

    # -- Config building ----------------------------------------------------

    def _build_generate_config(
        self,
        *,
        system_instruction: str | None = None,
        tools_schema: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
        cached_content: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Build a ``GenerateContentConfig`` from OpenAI-style params."""
        types = self._types
        config_kwargs: dict[str, Any] = {}

        if cached_content:
            config_kwargs["cached_content"] = cached_content

        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction

        # Temperature
        temperature = kwargs.get("temperature")
        if temperature is not None:
            config_kwargs["temperature"] = temperature

        # Max tokens
        max_tokens = kwargs.get("max_completion_tokens") or kwargs.get("max_tokens")
        if max_tokens is not None:
            config_kwargs["max_output_tokens"] = max_tokens

        # Top-p
        top_p = kwargs.get("top_p")
        if top_p is not None:
            config_kwargs["top_p"] = top_p

        # Top-k
        top_k = kwargs.get("top_k")
        if top_k is not None:
            config_kwargs["top_k"] = top_k

        # Response format -> response_mime_type / response_schema
        response_format = kwargs.get("response_format")
        if response_format:
            rf_type = (
                response_format.get("type", "")
                if isinstance(response_format, dict)
                else getattr(response_format, "type", "")
            )
            if rf_type == "json_object":
                config_kwargs["response_mime_type"] = "application/json"
            elif rf_type == "json_schema":
                config_kwargs["response_mime_type"] = "application/json"
                schema = (
                    response_format.get("json_schema", {}).get("schema")
                    if isinstance(response_format, dict)
                    else None
                )
                if schema:
                    config_kwargs["response_schema"] = schema

        # Tools
        if tools_schema:
            config_kwargs["tools"] = self._to_gemini_tools(tools_schema)

        # Tool choice
        if tool_choice and tools_schema:
            mode_map = {
                "auto": "AUTO",
                "none": "NONE",
                "required": "ANY",
            }
            mode = mode_map.get(tool_choice, "AUTO")
            config_kwargs["tool_config"] = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode=mode)
            )

        # Safety settings (Phase 5)
        if self._safety_settings:
            config_kwargs["safety_settings"] = [
                types.SafetySetting(category=cat, threshold=thresh)
                for cat, thresh in self._safety_settings.items()
            ]

        # Grounding with Google Search (Phase 5)
        has_builtin_tools = False
        if self._grounding:
            existing_tools = config_kwargs.get("tools", [])
            existing_tools.append(types.Tool(google_search=types.GoogleSearch()))
            config_kwargs["tools"] = existing_tools
            has_builtin_tools = True

        # Code execution (Phase 5)
        if self._code_execution:
            existing_tools = config_kwargs.get("tools", [])
            existing_tools.append(types.Tool(code_execution=types.ToolCodeExecution()))
            config_kwargs["tools"] = existing_tools
            has_builtin_tools = True

        # When built-in tools (grounding/code_execution) are combined with
        # function tools, the API requires include_server_side_tool_invocations.
        if has_builtin_tools and tools_schema:
            existing_tc = config_kwargs.get("tool_config")
            if existing_tc is not None:
                # Rebuild with the existing mode + the new flag
                mode = existing_tc.function_calling_config.mode
            else:
                mode = "AUTO"
            config_kwargs["tool_config"] = types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode=mode),
                include_server_side_tool_invocations=True,
            )

        return types.GenerateContentConfig(**config_kwargs)

    def _to_gemini_tools(self, openai_tools: list[dict[str, Any]]) -> list[Any]:
        """Convert OpenAI tool schemas to Gemini Tool objects."""
        types = self._types
        declarations = []
        for tool in openai_tools:
            fn = tool.get("function", tool)
            declarations.append(
                types.FunctionDeclaration(
                    name=fn.get("name", ""),
                    description=fn.get("description", ""),
                    parameters=fn.get("parameters"),
                )
            )
        return [types.Tool(function_declarations=declarations)]

    # -- Response normalization ---------------------------------------------

    def _normalize_response(self, response: Any) -> SimpleNamespace:
        """Convert a Gemini response to an OpenAI-shaped response object."""
        candidate = response.candidates[0]
        content_text = ""
        tool_calls: list[SimpleNamespace] = []

        for part in candidate.content.parts:
            if hasattr(part, "text") and part.text:
                content_text += part.text
            elif hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                tc = _make_tool_call(
                    tc_id=f"call_{uuid.uuid4().hex[:24]}",
                    name=fc.name,
                    arguments=json.dumps(dict(fc.args)) if fc.args else "{}",
                )
                tool_calls.append(tc)

        return _make_response(
            content=content_text or None,
            tool_calls=tool_calls if tool_calls else None,
        )

    def _stream_response(
        self, model: str, contents: list[Any], config: Any
    ) -> Iterator[SimpleNamespace]:
        """Yield OpenAI-shaped stream chunks from a Gemini streaming response."""
        tool_call_counter = 0
        had_tool_calls = False
        for chunk in self._client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config,
        ):
            if not chunk.candidates:
                continue
            candidate = chunk.candidates[0]
            if not hasattr(candidate, "content") or not candidate.content:
                continue
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    yield _make_stream_chunk(content=part.text)
                elif hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    tc = _make_tool_call(
                        tc_id=f"call_{uuid.uuid4().hex[:24]}",
                        name=fc.name,
                        arguments=json.dumps(dict(fc.args)) if fc.args else "{}",
                    )
                    tc.index = tool_call_counter
                    tool_call_counter += 1
                    had_tool_calls = True
                    yield _make_stream_chunk(tool_calls=[tc])
        # Emit final sentinel chunk with finish_reason
        yield _make_stream_chunk(
            finish_reason="tool_calls" if had_tool_calls else "stop"
        )

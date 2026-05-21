# Native Gemini Migration Plan

Multi-phase plan to add native Google AI (Gemini) support to `floship-llm`, preserving context caching and other native-only capabilities while keeping the existing OpenAI-compatible pathway as the default.

---

## Current State (v0.7.5)

What already works via OpenAI-compatible endpoint (`generativelanguage.googleapis.com/v1beta/openai/`):

| Feature | Status |
|---------|--------|
| Provider detection from `INFERENCE_URL` | done |
| Chat completions | done |
| Streaming | done |
| Tool calling (incl. concatenated JSON split) | done |
| Multimodal (vision) | done |
| Embeddings (model auto-mapping) | done |
| WAF auto-disable for non-Heroku | done |
| `thought_signature` injection | done |
| Provider-specific param filtering | done |

What **cannot** work through OpenAI-compatible endpoint:

| Feature | Why |
|---------|-----|
| Context caching | Native `cachedContent` API only |
| Batch API | Native `batchGenerateContent` only |
| File upload (large media) | Native `media.upload` API only |
| Token counting (pre-flight) | Native `countTokens` API only |
| Grounding (Google Search) | Native `google_search_retrieval` tool only |
| Safety settings (fine-grained) | Native `safetySettings` array only |
| System instructions caching | Part of context caching API |
| Code execution tool | Native `code_execution` tool only |
| Audio/video input | Native `generateContent` media parts |

---

## Architecture Overview

```
LLM (public API)
 |
 +-- ProviderBackend (ABC)
       |
       +-- OpenAICompatibleBackend   <-- current code, refactored
       |     (Heroku, Google OpenAI-compat, generic)
       |
       +-- NativeGeminiBackend       <-- new, Phase 3
             (google-genai SDK, native features)
```

Provider selection logic:

```
inference_url set?
  +-- contains "inference.heroku.com"       -> OpenAICompatibleBackend (heroku)
  +-- contains "googleapis.com" + /openai/  -> OpenAICompatibleBackend (google)
  +-- other                                 -> OpenAICompatibleBackend (other)

native_google=True (or GEMINI_NATIVE=true)?
  +-- google-genai SDK installed?
        +-- yes -> NativeGeminiBackend
        +-- no  -> raise ImportError with install instructions
```

---

## Phase 1 -- Stabilize OpenAI-Compatible Google AI

**Goal**: Harden what exists. No new dependencies. No breaking changes.

**Version target**: 0.8.x

### 1.1 Provider detection cleanup

```python
def _detect_provider(url: str) -> str:
    url = (url or "").lower()
    if "inference.heroku.com" in url:
        return "heroku"
    if "generativelanguage.googleapis.com" in url:
        return "google"
    return "openai_compatible"
```

Rename `"other"` to `"openai_compatible"` everywhere for clarity.

### 1.2 URL normalization

Google OpenAI-compatible endpoint requires `/openai/` path suffix. Auto-fix if missing.

```python
def _normalize_base_url(url: str, provider: str) -> str:
    url = url.rstrip("/") + "/"
    if provider == "google" and "/openai/" not in url:
        url = "https://generativelanguage.googleapis.com/v1beta/openai/"
    return url
```

| Provider | Normalized base URL |
|----------|-------------------|
| Heroku | `https://us.inference.heroku.com` (unchanged) |
| Google | `https://generativelanguage.googleapis.com/v1beta/openai/` |

### 1.3 API key fallback

```python
self.api_key = (
    kwargs.get("inference_key")
    or kwargs.get("api_key")
    or os.getenv("INFERENCE_KEY")
    or os.getenv("GEMINI_API_KEY")
)
```

### 1.4 Param mapping

Provider-aware param builder. Single source of truth for what gets sent.

```python
def _build_chat_params(self, messages, **kwargs):
    params = {
        "model": self.model,
        "messages": messages,
        "temperature": self.temperature,
    }
    if self.max_completion_tokens:
        params["max_completion_tokens"] = self.max_completion_tokens
    if self.top_p is not None:
        params["top_p"] = self.top_p
    if self.tools_enabled:
        params["tools"] = self.tool_manager.get_tools_schema()
    if self.response_format:
        params["response_format"] = self._get_response_format()

    # Heroku-only params
    if self.provider == "heroku":
        if self.top_k is not None:
            params["top_k"] = self.top_k
        if self.extended_thinking:
            params["extended_thinking"] = self.extended_thinking

    return {k: v for k, v in params.items() if v is not None}
```

| Param | Heroku | Google | Other |
|-------|--------|--------|-------|
| `model` | send | send | send |
| `messages` | send | send | send |
| `temperature` | send | send | send |
| `max_completion_tokens` | send | send | send |
| `top_p` | send | send | send |
| `top_k` | send | omit | omit |
| `extended_thinking` | send | omit | omit |
| `tools` | send | send | send |
| `tool_choice` | send | send | send |
| `response_format` | send | send (test JSON schema) | send |

### 1.5 WAF sanitization gating

```python
def _should_sanitize_for_waf(self) -> bool:
    if self.enable_waf_sanitization is False:
        return False
    return self.provider == "heroku"
```

| Provider | WAF default |
|----------|-------------|
| Heroku | enabled |
| Google | disabled |
| Other | disabled |

Override: `LLM(enable_waf_sanitization=True)`.

### 1.6 Multimodal content sanitization

WAF-sanitize text parts only. Pass image/audio parts unchanged.

```python
def _sanitize_content(self, content):
    if isinstance(content, str):
        return self._sanitize_for_waf(content)
    if isinstance(content, list):
        return [
            {**p, "text": self._sanitize_for_waf(p.get("text", ""))}
            if p.get("type") == "text" else p
            for p in content
        ]
    return content
```

### 1.7 Embeddings model mapping

```python
GOOGLE_EMBEDDING_MODEL_MAP = {
    "cohere-embed-multilingual": "gemini-embedding-001",
    "cohere-embed-english": "gemini-embedding-001",
}
```

Google drops unsupported params: `input_type`, `embedding_type`.

### 1.8 Tool calling -- concatenated JSON

Already implemented. Keep `_split_concatenated_json` applied at both execution and resend paths. Preserve wrapper function name (`execute_tool`), do not replace with nested `tool_name`.

### 1.9 Gemini `thought_signature`

Already implemented. Inject `skip_thought_signature_validator` on assistant tool_call messages for Google provider only.

### 1.10 Message validation pipeline

```python
def _validate_messages_for_api(self, messages):
    messages = copy.deepcopy(messages)
    messages = self._normalize_empty_content(messages)
    messages = self._sanitize_messages_for_provider(messages)
    messages = self._sanitize_tool_calls(messages)
    messages = self._inject_provider_metadata(messages)
    messages = self._truncate_messages(messages)
    return messages
```

| Rule | Heroku | Google | Other |
|------|--------|--------|-------|
| Empty assistant content fix | yes | yes | yes |
| Tool arg JSON repair | yes | yes | yes |
| Concatenated JSON split | optional | yes | optional |
| WAF sanitize | yes | no | no |
| `thought_signature` inject | no | yes | no |
| `extended_thinking` strip | yes | no | no |

### 1.11 Error handling

| Error | Heroku | Google | Other |
|-------|--------|--------|-------|
| CloudFront 403 | `CloudFrontWAFError` | skip | skip |
| Context length exceeded | generic detection | generic detection | generic detection |
| Model 404 | standard | `Model not found for Google AI provider. Check INFERENCE_MODEL_ID.` | standard |

### 1.12 Tests for Phase 1

**Provider detection**
- Heroku URL -> `"heroku"`
- Google `/openai/` URL -> `"google"`
- Google URL without `/openai/` normalizes correctly
- Unknown URL -> `"openai_compatible"`

**Params**
- Heroku sends `extended_thinking`
- Google omits `extended_thinking`
- Heroku sends `top_k`
- Google omits `top_k`
- Google sends top-level `top_p`

**WAF**
- Heroku sanitizes text
- Google does not sanitize by default
- `image_url` content preserved
- Multimodal text parts sanitized only for Heroku

**Tools**
- `tools=[...]` constructor registers tools
- `tools=[...]` auto-enables tools
- Concatenated JSON splits before execution
- Concatenated JSON splits before resend
- Wrapper function name preserved
- Invalid JSON repaired to `{}`
- Array args wrapped as `{"value": ...}`

**Gemini-specific**
- Google tool_call message gets dummy `thought_signature` if missing
- Heroku never gets dummy `thought_signature`
- Existing `extra_content` preserved

**Embeddings**
- Google maps `cohere-embed-multilingual` -> `gemini-embedding-001`
- Heroku does not map embedding model
- Google drops unsupported embedding params

---

## Phase 2 -- Provider Abstraction Layer

**Goal**: Extract provider-specific logic into pluggable backends without changing the public `LLM` API.

**Version target**: 0.9.x

### 2.1 New files

```
floship_llm/
    backends/
        __init__.py
        base.py              # ProviderBackend ABC
        openai_compat.py     # OpenAICompatibleBackend (extracted from client.py)
        native_gemini.py     # NativeGeminiBackend (Phase 3)
```

### 2.2 `ProviderBackend` ABC

```python
from abc import ABC, abstractmethod

class ProviderBackend(ABC):
    """Base class for inference provider backends."""

    @abstractmethod
    def chat(self, messages, **params) -> dict:
        """Send a chat completion request. Return normalized response dict."""

    @abstractmethod
    def chat_stream(self, messages, **params):
        """Send a streaming chat completion request. Yield delta strings."""

    @abstractmethod
    def embed(self, inputs, **params) -> list[list[float]]:
        """Send an embedding request. Return list of vectors."""

    @abstractmethod
    def count_tokens(self, messages) -> int:
        """Count tokens for messages. Estimate if provider has no native endpoint."""

    @abstractmethod
    def build_params(self, **kwargs) -> dict:
        """Build provider-specific request params from universal kwargs."""

    @abstractmethod
    def normalize_response(self, raw_response) -> dict:
        """Normalize provider response to common format."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return provider identifier string."""

    @property
    @abstractmethod
    def supports_caching(self) -> bool:
        """Whether this backend supports context caching."""

    @property
    @abstractmethod
    def supports_native_tools(self) -> bool:
        """Whether tools go through native API vs OpenAI schema."""
```

### 2.3 `OpenAICompatibleBackend`

Extract current `client.py` logic into this backend. The `LLM` class becomes a thin orchestrator that delegates to `self.backend`.

Key methods to extract:
- `_build_chat_params` -> `backend.build_params`
- `client.chat.completions.create` call -> `backend.chat`
- Streaming loop -> `backend.chat_stream`
- `client.embeddings.create` call -> `backend.embed`
- Response normalization -> `backend.normalize_response`

### 2.4 Backend factory

```python
def _create_backend(self) -> ProviderBackend:
    if self._use_native_gemini:
        return NativeGeminiBackend(
            api_key=self.api_key,
            model=self.model,
            ...
        )
    return OpenAICompatibleBackend(
        api_key=self.api_key,
        base_url=self.base_url,
        provider=self._provider,
        ...
    )
```

### 2.5 `LLM` orchestrator changes

```python
class LLM:
    def __init__(self, **kwargs):
        ...
        self.backend = self._create_backend()

    def prompt(self, message, **kwargs):
        # Conversation management, tool loop, retry -- unchanged
        # Replace direct client calls with:
        response = self.backend.chat(messages, **params)
        ...

    def prompt_stream(self, message, **kwargs):
        for delta in self.backend.chat_stream(messages, **params):
            yield delta

    def embed(self, inputs, **kwargs):
        return self.backend.embed(inputs, **params)
```

Public API stays identical. No consumer changes.

### 2.6 Tests for Phase 2

- `LLM` with Heroku URL creates `OpenAICompatibleBackend`
- `LLM` with Google OpenAI URL creates `OpenAICompatibleBackend`
- All existing tests pass unchanged (backend is an internal detail)
- Backend `.build_params()` matches current param filtering
- Backend `.normalize_response()` matches current response handling

---

## Phase 3 -- Native Gemini Backend

**Goal**: Implement `NativeGeminiBackend` using `google-genai` SDK. Unlock context caching, token counting, file upload, grounding.

**Version target**: 1.0.x

### 3.1 New dependency

```toml
[project.optional-dependencies]
google = ["google-genai>=1.0.0"]
```

Core `floship-llm` keeps zero new required dependencies. Native Gemini is opt-in.

### 3.2 Activation

```python
# Explicit opt-in
llm = LLM(
    model="gemini-2.5-flash",
    inference_url="https://generativelanguage.googleapis.com/v1beta/",
    inference_key="...",
    native_google=True,  # use google-genai SDK
)

# Or via env
GEMINI_NATIVE=true
```

If `native_google=True` but `google-genai` is not installed:

```
ImportError: Native Gemini backend requires google-genai.
Install with: pip install floship-llm[google]
```

### 3.3 `NativeGeminiBackend` implementation

```python
from google import genai
from google.genai import types

class NativeGeminiBackend(ProviderBackend):
    def __init__(self, api_key, model, **kwargs):
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._cache = None  # CachedContent handle

    @property
    def provider_name(self) -> str:
        return "google_native"

    @property
    def supports_caching(self) -> bool:
        return True

    def chat(self, messages, **params):
        contents = self._to_gemini_contents(messages)
        config = self._build_generate_config(**params)
        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )
        return self.normalize_response(response)

    def chat_stream(self, messages, **params):
        contents = self._to_gemini_contents(messages)
        config = self._build_generate_config(**params)
        for chunk in self._client.models.generate_content_stream(
            model=self._model,
            contents=contents,
            config=config,
        ):
            yield self._extract_delta(chunk)

    def embed(self, inputs, **params):
        result = self._client.models.embed_content(
            model=self._model,
            contents=inputs,
        )
        return [e.values for e in result.embeddings]

    def count_tokens(self, messages) -> int:
        contents = self._to_gemini_contents(messages)
        result = self._client.models.count_tokens(
            model=self._model,
            contents=contents,
        )
        return result.total_tokens
```

### 3.4 Message format conversion

OpenAI messages -> Gemini `Content` objects.

```python
def _to_gemini_contents(self, messages):
    """Convert OpenAI-style messages to Gemini Content objects."""
    contents = []
    system_instruction = None

    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")

        if role == "system":
            # Gemini uses system_instruction, not system role
            system_instruction = content
            continue

        gemini_role = "model" if role == "assistant" else "user"

        parts = []
        if isinstance(content, str):
            parts.append(types.Part.from_text(text=content))
        elif isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    parts.append(types.Part.from_text(text=item["text"]))
                elif item.get("type") == "image_url":
                    parts.append(self._convert_image_part(item))

        # Handle tool calls in assistant messages
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            for tc in tool_calls:
                parts.append(types.Part.from_function_call(
                    name=tc["function"]["name"],
                    args=json.loads(tc["function"]["arguments"]),
                ))

        # Handle tool results
        if role == "tool":
            parts = [types.Part.from_function_response(
                name=msg.get("name", ""),
                response={"result": content},
            )]
            gemini_role = "user"  # tool responses are user-role in Gemini

        contents.append(types.Content(role=gemini_role, parts=parts))

    return contents, system_instruction
```

### 3.5 Tool schema conversion

OpenAI tool schema -> Gemini `FunctionDeclaration`.

```python
def _to_gemini_tools(self, openai_tools):
    """Convert OpenAI tool schemas to Gemini FunctionDeclarations."""
    declarations = []
    for tool in openai_tools:
        fn = tool["function"]
        declarations.append(types.FunctionDeclaration(
            name=fn["name"],
            description=fn.get("description", ""),
            parameters=fn.get("parameters", {}),
        ))
    return [types.Tool(function_declarations=declarations)]
```

### 3.6 Response normalization

Gemini response -> common dict format consumed by `LLM` orchestrator.

```python
def normalize_response(self, response):
    """Normalize Gemini response to OpenAI-like dict."""
    candidate = response.candidates[0]
    content = candidate.content

    result = {
        "content": "",
        "role": "assistant",
        "tool_calls": [],
        "finish_reason": candidate.finish_reason.name.lower(),
        "usage": {
            "prompt_tokens": response.usage_metadata.prompt_token_count,
            "completion_tokens": response.usage_metadata.candidates_token_count,
            "total_tokens": response.usage_metadata.total_token_count,
            "cached_tokens": getattr(
                response.usage_metadata, "cached_content_token_count", 0
            ),
        },
    }

    for part in content.parts:
        if part.text:
            result["content"] += part.text
        elif part.function_call:
            result["tool_calls"].append({
                "id": f"call_{hash(part.function_call.name)}",
                "type": "function",
                "function": {
                    "name": part.function_call.name,
                    "arguments": json.dumps(dict(part.function_call.args)),
                },
            })

    return result
```

### 3.7 Tests for Phase 3

- `NativeGeminiBackend` initializes with `google-genai` client
- OpenAI messages convert to Gemini `Content` objects correctly
- System messages become `system_instruction`
- Tool schemas convert to `FunctionDeclaration`
- Gemini response normalizes to common format
- `usage.cached_tokens` populated when present
- `ImportError` raised when `google-genai` not installed and `native_google=True`
- All existing `LLM` tests pass with `NativeGeminiBackend` (provider-agnostic behavior)

---

## Phase 4 -- Context Caching

**Goal**: Expose Gemini context caching through the public API. Cache system prompts, large documents, few-shot examples.

**Version target**: 1.1.x

### 4.1 Public API

```python
llm = LLM(
    model="gemini-2.5-flash",
    inference_key="...",
    native_google=True,
    cache=True,                # enable caching
    cache_ttl=3600,            # seconds, default 1 hour
    cache_system=True,         # cache system prompt (default True when cache=True)
)

# System prompt + tools cached on first call
response = llm.prompt("Analyze this document")

# Subsequent calls reuse cache (reduced input tokens billed)
response = llm.prompt("What about section 3?")

# Check cache status
print(llm.cache_info)
# CacheInfo(cached=True, token_count=12450, ttl_remaining=3540, cost_saving_pct=68)
```

### 4.2 What gets cached

| Content | Cacheable | Min tokens | Notes |
|---------|-----------|-----------|-------|
| System prompt | yes | 4,096 (Gemini min) | Most common use case |
| Few-shot examples | yes | 4,096 | Static example messages |
| Large documents | yes | 4,096 | Appended to system |
| Tool definitions | yes | 4,096 | Stable across calls |
| Conversation history | no | -- | Changes every turn |

### 4.3 Cache lifecycle

```
First LLM.prompt() call:
  1. Build system_instruction + tools
  2. Check token count >= 4096 (Gemini minimum for caching)
  3. Create CachedContent via google-genai SDK
  4. Store cache handle in self._cache
  5. Send generate_content with cached_content reference
  6. Return response with usage.cached_tokens populated

Subsequent calls:
  1. Check if cache is still valid (TTL)
  2. If valid, reuse cached_content reference
  3. If expired, recreate cache
  4. Send only new conversation turns as contents

Cache invalidation:
  - TTL expires (auto)
  - System prompt changes (detect via hash)
  - Tools change (detect via hash)
  - Manual llm.clear_cache()
```

### 4.4 Implementation in `NativeGeminiBackend`

```python
class NativeGeminiBackend(ProviderBackend):
    def __init__(self, ..., cache=False, cache_ttl=3600):
        self._cache_enabled = cache
        self._cache_ttl = cache_ttl
        self._cached_content = None
        self._cache_hash = None  # hash of cached content for invalidation

    def _ensure_cache(self, system_instruction, tools):
        """Create or refresh context cache."""
        if not self._cache_enabled:
            return None

        content_hash = hashlib.sha256(
            json.dumps({"system": system_instruction, "tools": tools},
                       sort_keys=True).encode()
        ).hexdigest()

        # Reuse existing cache if content hasn't changed
        if self._cached_content and self._cache_hash == content_hash:
            return self._cached_content

        # Check token count meets minimum
        token_count = self.count_tokens_for_content(system_instruction)
        if token_count < 4096:
            logger.warning(
                "Cache content has %d tokens, minimum is 4096. Skipping cache.",
                token_count,
            )
            return None

        # Create new cache
        self._cached_content = self._client.caches.create(
            model=self._model,
            config=types.CreateCachedContentConfig(
                system_instruction=system_instruction,
                tools=tools,
                ttl=f"{self._cache_ttl}s",
            ),
        )
        self._cache_hash = content_hash
        return self._cached_content

    def chat(self, messages, **params):
        contents, system_instruction = self._to_gemini_contents(messages)

        cache = self._ensure_cache(
            system_instruction,
            params.get("tools"),
        )

        if cache:
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=types.GenerateContentConfig(
                    cached_content=cache.name,
                    temperature=params.get("temperature"),
                    max_output_tokens=params.get("max_completion_tokens"),
                ),
            )
        else:
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=self._build_generate_config(
                    system_instruction=system_instruction,
                    **params,
                ),
            )

        return self.normalize_response(response)
```

### 4.5 Cache info

```python
@dataclass
class CacheInfo:
    cached: bool
    token_count: int
    ttl_remaining: int  # seconds
    cache_name: str
    cost_saving_pct: float  # estimated

@property
def cache_info(self) -> Optional[CacheInfo]:
    if not hasattr(self, 'backend') or not self.backend.supports_caching:
        return None
    return self.backend.get_cache_info()
```

### 4.6 OpenAI-compatible backend -- cache stub

```python
class OpenAICompatibleBackend(ProviderBackend):
    @property
    def supports_caching(self) -> bool:
        return False

    def get_cache_info(self):
        return None
```

No-op. Cache is native-only. Attempting to use `cache=True` with non-native backend logs a warning and continues without caching.

### 4.7 Tests for Phase 4

- Cache created when `cache=True` and token count >= 4096
- Cache skipped when token count < 4096
- Cache reused on subsequent calls with same system prompt
- Cache invalidated when system prompt changes
- Cache invalidated when tools change
- `llm.clear_cache()` removes cache
- `cache_info` reports correct TTL and token count
- `usage.cached_tokens` populated in response
- `cache=True` with OpenAI-compatible backend logs warning, no error

---

## Phase 5 -- Native-Only Features

**Goal**: Expose additional Gemini-native capabilities. Each feature is independent and can be shipped separately.

**Version target**: 1.2.x+

### 5.1 Pre-flight token counting

```python
count = llm.count_tokens("How long is this message?")
count = llm.count_tokens(messages=[...])
```

Backend method already defined in Phase 2. Expose on `LLM`.

Falls back to local estimate for non-native backends.

### 5.2 File upload (large media)

```python
llm = LLM(native_google=True, ...)

# Upload large file
file_ref = llm.upload_file("/path/to/video.mp4", mime_type="video/mp4")

# Use in prompt
response = llm.prompt([
    {"type": "text", "text": "Summarize this video"},
    {"type": "file", "file": file_ref},
])
```

Implementation:

```python
def upload_file(self, path, mime_type=None):
    if not self.backend.supports_file_upload:
        raise NotImplementedError("File upload requires native_google=True")
    return self.backend.upload_file(path, mime_type)
```

### 5.3 Grounding with Google Search

```python
llm = LLM(
    native_google=True,
    grounding=True,  # enable Google Search grounding
)

response = llm.prompt("What happened in tech news today?")
# Response includes grounding metadata and source URLs
```

Implementation via Gemini `google_search_retrieval` tool.

### 5.4 Safety settings

```python
llm = LLM(
    native_google=True,
    safety_settings={
        "HARM_CATEGORY_HARASSMENT": "BLOCK_ONLY_HIGH",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    },
)
```

### 5.5 Code execution tool

```python
llm = LLM(
    native_google=True,
    code_execution=True,  # enable Gemini code execution sandbox
)

response = llm.prompt("Calculate the first 20 Fibonacci numbers")
# Gemini runs Python code internally and returns results
```

### 5.6 Tests for Phase 5

Each sub-feature gets isolated test coverage:
- Token count matches Gemini API response
- Token count falls back to estimate for non-native
- File upload returns valid reference
- File reference usable in prompt
- Grounding metadata present in response
- Safety settings forwarded to Gemini config
- Code execution results included in response

---

## Phase 6 -- Vertex AI Support

**Goal**: Support Google Cloud Vertex AI endpoints for enterprise/production use with service account auth.

**Version target**: 1.3.x

### 6.1 Provider detection

```python
if "aiplatform.googleapis.com" in url:
    return "vertex"
```

### 6.2 Authentication

```python
# Service account (default for Vertex)
llm = LLM(
    inference_url="https://us-central1-aiplatform.googleapis.com/",
    model="gemini-2.5-flash",
    native_google=True,
    google_project="my-project",
    google_location="us-central1",
)
# Uses Application Default Credentials (ADC)

# Or explicit service account
llm = LLM(
    ...
    google_credentials="/path/to/service-account.json",
)
```

### 6.3 Backend changes

```python
class NativeGeminiBackend(ProviderBackend):
    def __init__(self, ..., vertex=False, project=None, location=None):
        if vertex:
            self._client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
            )
        else:
            self._client = genai.Client(api_key=api_key)
```

### 6.4 Tests for Phase 6

- Vertex AI URL detected correctly
- ADC auth used when no explicit key
- Service account JSON auth works
- All Phase 3-5 features work via Vertex AI
- `google_project` and `google_location` required for Vertex

---

## Migration Summary

| Phase | Version | Scope | Breaking changes | New dependencies |
|-------|---------|-------|-----------------|-----------------|
| 1 | 0.8.x | Stabilize OpenAI-compatible Google AI | `"other"` -> `"openai_compatible"` | none |
| 2 | 0.9.x | Extract provider backends | none (internal refactor) | none |
| 3 | 1.0.x | Native Gemini backend | none (opt-in via `native_google=True`) | `google-genai` (optional) |
| 4 | 1.1.x | Context caching | none | none (uses google-genai) |
| 5 | 1.2.x+ | File upload, grounding, safety, code exec | none | none |
| 6 | 1.3.x | Vertex AI support | none | none |

---

## Env Vars Summary

| Env | Use | Phase |
|-----|-----|-------|
| `INFERENCE_URL` | Provider switch (base URL) | 1 |
| `INFERENCE_KEY` | API key | 1 |
| `INFERENCE_MODEL_ID` | Model name | 1 |
| `FLOSHIP_LLM_WAF_SANITIZE` | WAF override | 1 |
| `GEMINI_API_KEY` | Fallback API key for Google | 1 |
| `GEMINI_NATIVE` | Enable native Gemini backend | 3 |
| `GEMINI_CACHE_TTL` | Cache TTL in seconds | 4 |
| `GOOGLE_PROJECT` | Vertex AI project | 6 |
| `GOOGLE_LOCATION` | Vertex AI location | 6 |

---

## Non-Goals (Explicit Exclusions)

| Excluded | Reason |
|----------|--------|
| Drop OpenAI SDK | OpenAI-compatible path stays for Heroku and generic providers |
| Support Anthropic SDK natively | Heroku already proxies Claude via OpenAI-compatible API |
| Multi-provider fan-out | Out of scope; one provider per `LLM` instance |
| Prompt template engine | Not related to provider abstraction |
| Agent framework | Separate concern; `floship-llm` is a client library |

---

## README Section (Phase 1)

```markdown
## Provider Switching

This library supports Heroku Inference and Google AI OpenAI-compatible endpoints.

### Heroku Inference

```bash
export INFERENCE_URL="https://us.inference.heroku.com"
export INFERENCE_MODEL_ID="claude-4-sonnet"
export INFERENCE_KEY="your-heroku-key"
```

### Google AI

```bash
export INFERENCE_URL="https://generativelanguage.googleapis.com/v1beta/openai/"
export INFERENCE_MODEL_ID="gemini-3.5-flash"
export INFERENCE_KEY="your-google-ai-key"
```

The provider is auto-detected from `INFERENCE_URL`.

| Feature | Heroku | Google AI |
|---------|--------|-----------|
| Chat completions | yes | yes |
| Streaming | yes | yes |
| Tool calling | yes | yes |
| Vision input | model-dependent | yes |
| Embeddings | yes | yes |
| CloudFront WAF sanitization | enabled by default | disabled by default |
| Claude extended thinking | yes | no |
| Context caching | no | native backend only |
```

---

## Changelog (Phase 1 Entry)

```markdown
## [0.8.0] - YYYY-MM-DD

### Changed
- Simplified provider switching via `INFERENCE_URL`. Heroku Inference and
  Google AI OpenAI-compatible endpoints share the same public `LLM` API.
- Provider-specific request params normalized internally: Claude
  `extended_thinking` and `top_k` sent only to Heroku; Google AI receives
  standard OpenAI-compatible params.
- Renamed provider value `"other"` to `"openai_compatible"` for clarity.

### Fixed
- Google AI multimodal messages preserve `image_url` parts unchanged while
  applying text-only WAF sanitization where enabled.
- Google AI tool-call handling preserves wrapper function names and repairs
  concatenated JSON arguments before execution and follow-up requests.
- Google AI embedding requests map legacy Cohere model names to
  `gemini-embedding-001`.

### Notes
- CloudFront WAF sanitization enabled by default for Heroku, disabled for
  Google AI and other providers.
- Native Gemini backend (context caching, file upload) planned for 1.0.x.
```

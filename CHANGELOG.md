# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.6.3] - 2026-06-16

### Added
- **OpenRouter reasoning controls:** `LLM` now accepts `reasoning`, `include_reasoning`, and `max_tokens` kwargs. For OpenRouter providers, reasoning options are sent through `extra_body` so the OpenAI SDK includes them in the request body, and `max_tokens` is preferred over `max_completion_tokens` when supplied.

## [1.6.2] - 2026-06-10

### Fixed
- **Structured JSON truncation detection:** Extra closing delimiters after an otherwise valid JSON object are no longer misclassified as truncation. The parser now recovers the first balanced JSON object instead of retrying with higher `max_completion_tokens`.
- **Streaming reasoning response fallback:** Streaming responses now collect text from `delta.reasoning` and `delta.model_extra["reasoning"]` when `delta.content` is empty. This fixes empty structured responses from reasoning models in the default streaming path.

## [1.6.1] - 2026-06-05

### Fixed
- **Reasoning model fallback:** `process_response()` now falls back to `model_extra['reasoning']` when `message.content` is None or empty. This fixes a bug where reasoning models on OpenRouter (like `xiaomi/mimo-v2.5`) would put their transcription/response in the `reasoning` field instead of `content`, causing `prompt()` to return an empty string while direct API calls worked fine.

### Added
- 3 new tests for reasoning model fallback in `process_response()`.

## [1.6.0] - 2026-06-05

### Added
- **Audio input support:** `input_audio` content parts (base64-encoded WAV/MP3/OGG/etc.) are now passed through to all providers as standard OpenAI content parts. For Gemini native backend, they are converted to `Part.from_bytes` with the correct MIME type. For OpenRouter and other OpenAI-compatible providers, they are passed through unchanged -- audio-capable models (like `openai/gpt-4o-audio-preview` or audio-capable open models) will process them; non-audio models will return a clear API error.
- **Video input support:** New `video_url` and `video_data` content part types for video input. `video_url` accepts HTTP URLs or data URIs; `video_data` accepts raw base64-encoded video with a MIME type. For Gemini native backend, these convert to `Part.from_uri` or `Part.from_bytes`. For OpenRouter/generic providers, they are converted to `image_url` parts for passthrough (OpenRouter normalizes media URLs for multimodal models like xiaomi/mimo-v2.5).
- **16 new tests** for native Gemini audio/video conversion in `tests/test_native_gemini.py` and video adaptation tests in `tests/test_openrouter.py`.

### Changed
- **`_adapt_multimodal_for_provider`** no longer strips `input_audio` parts for non-Gemini providers. Audio is an OpenAI standard content type supported by audio-capable models on OpenRouter and other providers. The API returns a clear error if a model doesn't support audio input.
- **Native Gemini backend** now converts `input_audio`, `video_url`, and `video_data` content parts to native Gemini `Part` objects.

## [1.5.0] - 2026-06-05

### Added
- **OpenRouter provider support:** First-class `openrouter` provider detection from `openrouter.ai` URLs. Routing, parameter handling (standard `top_p`, no Heroku extras), and embedding params are all handled correctly for OpenRouter.
- **Multimodal audio adaptation:** `input_audio` content parts (Gemini-only) are automatically stripped and replaced with a text placeholder for non-Gemini providers (OpenRouter, Heroku, generic). Prevents `400 Multimodal data corrupted` errors when switching between providers. Images (`image_url`) are preserved for all providers.
- **28 new tests** in `tests/test_openrouter.py` covering provider detection, init, request params, audio adaptation, validation pipeline, and cross-provider comparison.

### Fixed
- **`top_p` routing for OpenRouter:** `top_p` is now sent as a standard parameter (not `extra_body`) for OpenRouter and Google providers, matching the OpenAI API spec.

## [1.4.3] - 2025-07-22

### Fixed
- **thought_signature SDK bug workaround:** Strip `thought_signature` from the tool-call round-trip instead of replaying it, working around a `google-genai==1.20.0` bug where even unmodified response objects fail validation.
- **Grounding + function tools conflict:** When built-in tools (grounding, code_execution) coexist with function tools, automatically set `tool_config.include_server_side_tool_invocations = True` to prevent `400 INVALID_ARGUMENT`.
- **Context cache creation errors:** Wrap cache creation in try/except so `ValidationError` (tools in cached content) and `400 Cached content is too small` fall back gracefully to non-cached requests instead of crashing.

## [1.4.2] - 2026-05-21

### Fixed
- **Native Gemini streaming compatibility:** Stream chunks now include `finish_reason` on `choices[0]`, tool call deltas include `.index`, and a final sentinel chunk is emitted with `finish_reason="stop"` or `"tool_calls"`.
- **Non-streaming `finish_reason`:** `_make_response` now sets `finish_reason` on `choices[0]` (`"stop"` for text, `"tool_calls"` for tool calls).

## [1.4.1] - 2026-05-21

### Changed
- **Dynamic tool-call placeholder:** Assistant messages with tool_calls now use `[Calling tool_name]` instead of `"."` to prevent Gemini from mimicking the static pattern in responses.

### Added
- **Public `try_split_concatenated_json` method:** Exposed as public API on `FloshipLLM` for callers that need execution-level JSON splitting. Old `_try_split_concatenated_json` kept as backward-compat alias.

## [1.4.0] - 2026-05-21

### Changed
- **Python version requirement lowered to `>=3.8.1`:** Base package (Heroku/OpenAI-compatible backends) now supports Python 3.8.1+. The `[google]` optional extra (`google-genai`) still requires Python >=3.9.

### Added
- **Google context caching (OpenAI-compatible path):** `GoogleCacheManager` uses the native `google-genai` SDK for cache lifecycle (create/get/update/delete) while keeping chat requests on the OpenAI-compatible endpoint via `extra_body.cached_content`.
- **`ContextCacheRef` data class:** Lightweight cache reference with `name`, `key`, `model`, `expires_at`, `token_count`, and `is_valid()` method (60s refresh margin).
- **`LLM(enable_context_cache=True)` auto-caching:** Automatically creates and reuses context caches for static content (system prompt, tools, documents). Controlled via `FLOSHIP_LLM_CONTEXT_CACHE` env var or kwarg.
- **Manual cache API:** `llm.create_context_cache(system=..., contents=..., tools=...)` and `llm.delete_context_cache(name=...)` for explicit cache management.
- **`prompt(cached_content="cachedContents/...")` kwarg:** Manually specify a pre-created cache ref for a request.
- **`prompt(cache_static_context=True)` kwarg:** Trigger auto-cache creation for a single request.
- **Break-even cost calculator:** `GoogleCacheManager.should_cache(token_count, expected_reuse, ttl_hours)` using Gemini Flash 3.5 pricing model.
- **Deterministic cache keys:** SHA-256 of model + system + contents + tools + version + permission_hash ensures cache reuse across requests with identical static context.
- **Config kwargs:** `context_cache_ttl_seconds`, `context_cache_min_tokens`, `context_cache_version`, `context_cache_scope`, `context_cache_expected_reuse`, `context_cache_contents`.
- **`prompt_stream()` cache support:** Same `cached_content` and `cache_static_context` kwargs.
- **48 new tests** covering ContextCacheRef validity, key stability, cache CRUD, LLM integration, config forwarding, cached request flow (extra_body injection, system message stripping, tool deduplication), and break-even logic.

## [1.3.0] - 2026-05-21

### Added
- **Vertex AI support:** `LLM(inference_url="https://us-central1-aiplatform.googleapis.com/", ...)` auto-detects Vertex AI and uses the native `google-genai` SDK with `vertexai=True`. No explicit API key required -- uses Application Default Credentials (ADC).
- **`google_project` / `google_location` params:** Passed to `genai.Client(vertexai=True, project=..., location=...)` via kwargs or `GOOGLE_PROJECT` / `GOOGLE_LOCATION` env vars.
- **Auto-enable native backend for Vertex:** Vertex AI URLs (`aiplatform.googleapis.com`) automatically activate `NativeGeminiBackend` without needing `native_google=True`.
- **40 new tests** covering Vertex AI provider detection, backend creation, LLM integration, ADC auth, feature forwarding (cache, grounding, safety, code execution, file upload, token counting), and non-Vertex API key enforcement.

### Changed
- `_detect_provider()` now returns `"vertex"` for `aiplatform.googleapis.com` URLs.
- `_validate_environment()` skips `INFERENCE_KEY` check for Vertex AI URLs (ADC auth).
- `NativeGeminiBackend.__init__` accepts `vertex`, `project`, `location` keyword arguments.
- `_create_backend()` forwards `vertex`, `project`, `location` to the native Gemini backend.
- API key validation relaxed: `api_key` parameter is optional (`str | None`) on `NativeGeminiBackend`.

## [1.2.0] - 2026-05-21

### Added
- **Pre-flight token counting:** `llm.count_tokens("text")` or `llm.count_tokens(messages=[...])`. Uses native Gemini `countTokens` API when available, falls back to character-based estimate (~4 chars/token) for non-native backends.
- **File upload:** `llm.upload_file("/path/to/video.mp4", mime_type="video/mp4")` via native Gemini `media.upload` API. Returns a file reference usable in prompt content. Raises `NotImplementedError` for non-native backends.
- **Google Search grounding:** `LLM(grounding=True)` adds the `google_search` tool to all requests. Responses may include grounding metadata and source URLs.
- **Safety settings:** `LLM(safety_settings={"HARM_CATEGORY_HARASSMENT": "BLOCK_ONLY_HIGH"})` forwards fine-grained safety thresholds to the Gemini `SafetySetting` config.
- **Code execution:** `LLM(code_execution=True)` enables the Gemini code execution sandbox. The model can run Python code internally and return results.
- **`supports_file_upload` property** on `ProviderBackend` (default `False`, `True` for native Gemini).
- **30 new tests** covering token counting (native + fallback), file upload, grounding, safety settings, code execution, and combined feature scenarios.

### Changed
- `NativeGeminiBackend.__init__` accepts `safety_settings`, `grounding`, and `code_execution` keyword arguments.
- `_create_backend()` forwards Phase 5 parameters to the native Gemini backend.
- `_build_generate_config()` appends `google_search`, `code_execution`, and `safety_settings` to the config when enabled.

## [1.1.0] - 2026-05-21

### Added
- **Context caching:** `LLM(cache=True, cache_ttl=3600)` enables Gemini context caching for system prompts and tool definitions. Reduces input token costs on subsequent calls.
- **`CacheInfo` dataclass:** `llm.cache_info` returns cache status, token count, TTL remaining, and cache name.
- **`LLM.clear_cache()`:** Explicitly invalidate the active context cache.
- **Cache env vars:** `GEMINI_CACHE=true` and `GEMINI_CACHE_TTL=3600` for configuration without code changes.
- **Cache lifecycle management:** Automatic hash-based invalidation when system prompt or tools change. TTL-based expiry with automatic recreation.
- **36 new tests** covering cache init, creation, reuse, invalidation, TTL expiry, error handling, `CacheInfo`, and LLM integration.

### Changed
- `NativeGeminiBackend.__init__` accepts `cache` and `cache_ttl` keyword arguments.
- `_create_backend()` forwards cache parameters to the native Gemini backend.
- `chat()` now checks for cacheable content and uses `cached_content` reference in `GenerateContentConfig` when applicable.
- `OpenAICompatibleBackend` has `get_cache_info()` and `clear_cache()` stubs (no-ops).

## [1.0.0] - 2026-05-21

### Added
- **Native Gemini backend:** `NativeGeminiBackend` using the `google-genai` SDK. Opt-in via `LLM(native_google=True)` or `GEMINI_NATIVE=true` env var.
- **Optional `google-genai` dependency:** `pip install floship-llm[google]`. Core library has zero new required dependencies.
- **Native token counting:** `backend.count_tokens(messages)` via Gemini `countTokens` API.
- **Message format conversion:** Automatic OpenAI messages to Gemini `Content` objects (system -> `system_instruction`, tool results -> `function_response`, multimodal support).
- **Tool schema conversion:** OpenAI tool schemas auto-converted to Gemini `FunctionDeclaration` format.
- **Response normalization:** Gemini responses wrapped in OpenAI-shaped objects so the LLM orchestrator works unchanged.
- **Streaming support:** Native Gemini streaming via `generate_content_stream`, yielding OpenAI-shaped chunks.
- **40 new tests** covering backend init, message conversion, chat, streaming, embedding, token counting, config building, and LLM integration.

### Changed
- **`requires-python` bumped to `>=3.10`** (from `>=3.8.1`) due to `google-genai` dependency.
- `_create_backend()` factory now supports `native_google=True` path alongside existing `OpenAICompatibleBackend`.
- Model resolution moved earlier in `LLM.__init__` so the backend has access to `self.model` at creation time.
- When `native_google=True`, `self.client` is `None` (no OpenAI client created).

## [0.9.0] - 2026-05-21

### Added
- **Provider backend abstraction layer:** Introduced `ProviderBackend` ABC and `OpenAICompatibleBackend` in `floship_llm/backends/`. All API calls now route through `self.backend.chat()` / `self.backend.embed()` instead of direct `self.client.chat.completions.create()` calls.
- **Backend factory:** `LLM._create_backend()` creates the appropriate backend based on detected provider. Currently all providers use `OpenAICompatibleBackend`; `NativeGeminiBackend` will be added in Phase 3.
- **Backend properties:** `provider_name`, `supports_caching`, `supports_native_tools`, `count_tokens()` available on backend instances for future provider-specific logic.
- **20 new tests** for backend ABC, OpenAICompatibleBackend, and LLM-backend integration.

### Changed
- All 12 `self.client.chat.completions.create` and `self.client.embeddings.create` call sites in `client.py` now delegate through the backend layer. Public `LLM` API is unchanged.
- `self.client` backward-compatibility alias preserved; existing code and tests continue to work.

## [0.8.0] - 2026-05-21

### Changed
- Simplified provider switching via `INFERENCE_URL`. Heroku Inference and Google AI OpenAI-compatible endpoints share the same public `LLM` API.
- Provider-specific request params normalized internally: Claude `extended_thinking` and `top_k` sent only to Heroku; Google AI receives standard OpenAI-compatible params.
- Renamed provider value `"other"` to `"openai_compatible"` for clarity.
- Google AI embedding requests now drop unsupported Heroku-specific params (`input_type`, `embedding_type`).
- WAF sanitization gated through `_should_sanitize_for_waf()` method for cleaner provider checks.

### Added
- **URL normalization:** Google AI URLs without `/openai/` suffix are auto-normalized to the correct OpenAI-compatible endpoint.
- **`GEMINI_API_KEY` fallback:** API key resolution now checks `GEMINI_API_KEY` env var when `INFERENCE_KEY` is not set.
- **Google model 404 diagnostics:** Enhanced error logging when Google AI returns 404 (model not found), with actionable guidance.
- **`_should_sanitize_for_waf()` method:** Formal WAF gating that checks both config and provider.
- **`_normalize_base_url()` method:** Static URL normalization for provider-specific path requirements.
- **README Provider Switching section:** Documents Heroku and Google AI setup with feature comparison table.

### Notes
- CloudFront WAF sanitization enabled by default for Heroku, disabled for Google AI and other providers.
- Native Gemini backend (context caching, file upload) planned for 1.0.x.

## [0.7.5] - 2026-05-21

### Added
- **Multimodal (vision) support:** `prompt()` and `add_message()` now accept OpenAI-style list-of-dicts content with `text` and `image_url` parts. Works with Gemini's OpenAI-compatible endpoint for image analysis via public URL or base64 data URL. WAF sanitization applies to text parts only; image parts are preserved as-is.

## [0.7.4] - 2026-05-21

### Fixed
- **Keep original function name in split tool calls:** The concatenated JSON splitter in `_handle_tool_calls` was extracting `tool_name` from Gemini's payload (e.g. `get_order`) as the function name, but the tool manager only knows the wrapper name (e.g. `execute_tool`). Now always preserves the original function name and serializes the full split object as arguments.

## [0.7.3] - 2026-05-21

### Fixed
- **Concatenated JSON splitting in tool execution:** The v0.7.2 fix only applied when messages were re-sent to the API (`_sanitize_tool_calls`). The initial tool execution path in `_handle_tool_calls` still failed on concatenated JSON, causing `json.loads()` errors and wasted tool rounds. Now splits concatenated tool calls before execution, expanding them into separate tool calls with correct names and arguments.

## [0.7.2] - 2026-05-21

### Fixed
- **Concatenated JSON tool call recovery:** Gemini sometimes emits multiple tool calls as concatenated JSON objects in a single `arguments` field (e.g. `{...}{...}`). `_sanitize_tool_calls` now splits them via `raw_decode` and promotes each to a separate tool call, preventing empty `"{}"` args and wasted tool rounds.

## [0.7.1] - 2026-05-21

### Fixed
- **Embedding model mapping:** Changed auto-mapping target from `text-embedding-004` (deprecated) to `gemini-embedding-001` (3072 dimensions) for Google AI provider.
- **Tools kwarg registration:** Tools passed via `tools=[...]` kwarg in constructor are now properly registered in `tool_manager`. Previously they were silently ignored.
- **Auto-enable tools:** Passing `tools=[...]` kwarg now auto-sets `enable_tools=True` instead of requiring it explicitly.

## [0.7.0] - 2026-05-21

### Added
- **Gemini thought_signature auto-injection:** `_validate_messages_for_api` injects dummy `skip_thought_signature_validator` on the first tool_call for Google AI provider when `extra_content` is missing, preventing Gemini 3+ 400 errors.
- **Embedding model auto-mapping:** `cohere-embed-multilingual` and `cohere-embed-english` are auto-mapped to `gemini-embedding-001` when provider is Google AI, preventing 404 errors.
- **Constructor kwargs:** `inference_url`, `inference_key`, and `api_key` kwargs override environment variables, allowing per-instance configuration.
- **`model` kwarg:** `model` parameter overrides `INFERENCE_MODEL_ID` env var.

### Changed
- **DRY refactoring:** Extracted `_finalize_response()`, `_parse_structured_output()`, `_fallback_to_pseudo_thinking()`, `_raise_waf_error()` helpers from `process_response()` and `_process_streaming_response()`.
- **Content cleaning:** `process_response()` now strips reasoning tags before returning, fixing inconsistency between streaming and non-streaming paths.

## [0.6.0] - 2026-05-21

### Added
- **Google AI (Gemini) provider support:** Auto-detect provider from `INFERENCE_URL` and adjust behavior accordingly. Supports `generativelanguage.googleapis.com` endpoints via OpenAI-compatible API.
- **Provider detection:** New `_detect_provider()` classifies endpoints as `google`, `heroku`, or `other` based on base URL.
- **Gemini thought signature preservation:** Tool call messages now preserve `extra_content` (including `thought_signature`) required by Gemini 3+ models for multi-step tool calling.
- **Auto-disable WAF for non-Heroku:** CloudFront WAF sanitization is automatically disabled for Google AI and other non-Heroku providers (can be overridden with `enable_waf_sanitization=True`).

### Changed
- **Provider-aware request params:** `extended_thinking` is only sent for Heroku/Claude; `top_p` is a standard param for Google (not wrapped in `extra_body`).
- **CloudFront 403 detection:** `_is_cloudfront_403()` returns `False` immediately for non-Heroku providers.

## [0.5.54] - 2026-05-14

### Added
- **WAF sanitization for `embed()` method:** Embedding inputs are now sanitized before sending to the API, preventing CloudFront 403 blocks on dense markdown links and path-like patterns.
- **Markdown link WAF pattern:** `[text](url)` syntax (especially with `../` relative paths) is stripped to plain text before sending, as dense link blocks trigger CloudFront path traversal detection.

## [0.5.53] - 2026-05-13

### Fixed
- **JSON-escaped newline WAF block:** `...\n` (literal backslash-n from tool results/nested JSON) now sanitized alongside real `...\n`. Prevents CloudFront path traversal detection when `json.dumps()` serializes `...\n` as `...\\n`.

## [0.5.52] - 2026-05-13

### Changed
- **WAF 403 fail-fast:** CloudFront WAF 403 errors now raise `CloudFrontWAFError` immediately instead of retrying. Enables Sentry exception capture on first occurrence.
- Removed `max_waf_retries`, `retry_with_sanitization`, `retry_delay_base` from `LLMConfig`.
- Removed `FLOSHIP_LLM_WAF_MAX_RETRIES` environment variable.

### Added
- **Ellipsis-before-newline WAF sanitization:** 3+ dots before `\n` (e.g. `...\n`) replaced with unicode ellipsis to prevent CloudFront path traversal detection.

## [0.5.51] - 2026-05-11

### Added
- **Localhost URL WAF Sanitization:** `http(s)://localhost:PORT/path` URLs now sanitized to prevent CloudFront SSRF detection blocks. Preserves path context via `[LOCAL_PATH:/path]` placeholder with full desanitize roundtrip.

## [0.5.50] - 2026-02-24

### Added
- **JIRA Markup WAF Sanitization:** Added sanitization for JIRA wiki markup patterns that trigger CloudFront WAF blocks: horizontal rules (`----`), link syntax (`[text|url]`), `{quote}` blocks, and table headers (`||Cell||`). All with full desanitization (roundtrip) support.

### Fixed
- **Pre-commit hook:** Fixed `check-version-consistency` hook using `python` instead of `python3`.

## [0.5.49] - 2025-12-12

### Fixed
- **Improved Unescaped Quote Detection:** Enhanced `_fix_unescaped_quotes_in_strings()` with multiple detection strategies and increased max attempts (100) to handle long content with many unescaped quotes. Now also handles "Expecting property name" errors in addition to delimiter errors.

## [0.5.48] - 2025-12-12

### Fixed
- **JSON Parsing with Unescaped Quotes:** Fixed JSON parsing failure when LLMs output unescaped double quotes inside JSON string values. Added `_fix_unescaped_quotes_in_strings()` method that iteratively identifies and escapes problematic quotes based on JSON parse errors. This resolves "Expecting ',' delimiter" errors when parsing LLM-generated JSON with embedded quotes like `"Festival "Name" here"`.

## [0.5.47] - 2025-12-12

### Fixed
- **JSON Parsing with Literal Newlines:** Fixed JSON parsing failure when LLMs output literal newlines inside JSON string values (instead of escaped `\n`). Added `_escape_newlines_in_strings()` method that properly escapes literal newlines, carriage returns, and tabs inside JSON strings before parsing. This resolves "Expecting property name enclosed in double quotes" errors when parsing LLM-generated JSON with unescaped control characters in string values.

## [0.5.46] - 2025-12-12

### Fixed
- **Improved Truncated JSON Detection:** Fixed detection of truncated responses wrapped in incomplete markdown code blocks (e.g., ` ```json\n{...` without closing ` ``` `). Now correctly detects truncation by checking for odd number of code fences and incomplete JSON within code blocks.

## [0.5.45] - 2025-12-11

### Fixed
- **Truncated JSON Detection in Structured Output:** Added proper truncation detection for non-streaming responses with `response_format`. Now raises `TruncatedResponseError` with a clear message suggesting to increase `max_completion_tokens` instead of failing with cryptic Pydantic validation errors.
- **response_format Reasoning Wrapper Only When Extended Thinking Enabled:** The `ThinkingWrapper` that adds a `reasoning` field to `response_format` schemas is now only applied when `extended_thinking` is enabled. Previously, it was always applied, forcing unnecessary reasoning overhead even when not requested.

## [0.5.44] - 2025-12-11

### Fixed
- **Python 3.8 Compatibility:** Added `from __future__ import annotations` to all modules to support Python 3.8+ by deferring annotation evaluation (PEP 563). This fixes `TypeError: unsupported operand type(s) for |` errors when importing on Python 3.8/3.9.

## [0.5.42] - 2025-12-07

### Added
- **`include_reasoning` Support:** Properly captures native reasoning from Heroku API's `message.reasoning.thinking` field when `include_reasoning=True` is passed.
- **Unified `get_last_reasoning()` Method:** Single interface to access reasoning from any source:
  1. Native extended thinking API response (`message.reasoning.thinking`)
  2. Pseudo-thinking `<reasoning>` tags when native thinking is unavailable
  3. Structured output with `reasoning` field (ThinkingModel or wrapped models)

  ```python
  response = llm.prompt("Solve 2+2", include_reasoning=True)
  reasoning = llm.get_last_reasoning()  # Access the reasoning separately
  ```

### Changed
- **ThinkingModel Uses `reasoning` Field:** The `ThinkingModel` base class now uses `reasoning` instead of `thinking` as the field name for consistency.
- **Reasoning Not Embedded:** Reasoning is no longer embedded in response content. Access it via `get_last_reasoning()` instead for clean separation of content and reasoning.
- **Thinking Tags Renamed:** Pseudo-thinking now uses `<reasoning>` tags instead of `<think>` tags for better clarity and consistency.

### Deprecated
- **`get_last_structured_thinking()`:** Use `get_last_reasoning()` instead. The old method is kept for backward compatibility but will be removed in a future version.

## [0.5.41] - 2025-12-07

### Added
- **Response Format Reasoning Auto-Wrap:** When using `response_format` with a Pydantic model that doesn't have a `reasoning` field, the library now automatically wraps it to capture Claude's chain-of-thought reasoning. Access the reasoning via `get_last_reasoning()`. Models extending `ThinkingModel` are returned as-is.
- **Pseudo-Thinking Mode:** When native `extended_thinking` fails (e.g., after tool execution), the library automatically switches to prompt-based pseudo-thinking using `<reasoning>` tags.

### Changed
- **Reasoning Redundancy Prevention:** When `response_format` already has a `reasoning` field (either extending `ThinkingModel` or having the field directly), native `extended_thinking` is automatically disabled to avoid redundant reasoning mechanisms. Schema-based reasoning takes precedence.
- **Always Strip Reasoning Tags:** Reasoning tags (`<reasoning>...</reasoning>`) are now always stripped from message history before API calls, preventing 500 errors from Heroku's API.

### Fixed
- **Extended Thinking Recovery:** Added automatic recovery when extended thinking validation fails after tool execution - switches to pseudo-thinking mode seamlessly.
- **Context Length Handling:** Added `_is_context_length_error()` and `_trim_conversation_for_context()` for better handling of context length exceeded errors.
- **Invalid Content Recovery:** Added `_is_invalid_content_error()` with automatic message sanitization and retry.

## [0.5.35] - 2025-12-03

### Fixed
- **500 "Failed to execute chat" Errors:** Fixed critical bug where invalid JSON in tool call arguments caused 500 server errors. The API returns 500 (not 400) when `tool_calls[].function.arguments` contains:
  - Empty strings
  - Plain text (not JSON)
  - Partial/malformed JSON
  - Arrays or primitive values (numbers, booleans)

  Now `_sanitize_tool_calls()` validates and repairs arguments:
  - Invalid JSON → replaced with empty object `{}`
  - Arrays/primitives → wrapped in object `{"value": ...}`
  - Re-serializes valid JSON to clean up formatting issues

## [0.5.34] - 2025-12-03

### Fixed
- **Message Sanitization:** Fixed 400 errors from empty assistant content with tool_calls. Claude API requires non-empty content field for assistant messages with tool_calls. Now always sets placeholder content "." when LLM returns tool_calls without content, preventing "content is required" API errors.

### Added
- **`sanitize_messages()` Method:** New public method to clean up conversation history before API calls. Fixes:
  - Empty/None content in assistant messages with tool_calls (causes 400 error)
  - Empty content in any message role
  - Invalid message structure

  Call this if you're managing messages externally or encounter "content is required" errors:
  ```python
  llm.messages = external_conversation_history
  llm.sanitize_messages()  # Fix any issues
  response = llm.prompt("Continue the conversation")
  ```

## [0.5.33] - 2025-12-03

### Changed
- **Tool Not Found Error Handling:** Improved error messages when LLM calls a non-existent tool. Now returns a cleaner, more concise message: "Error: Tool 'X' is not available. Please use one of: tool1, tool2, ..." (limited to first 10 tools). This prevents confusing error messages from being passed to the API which could trigger 500 errors.

## [0.5.32] - 2025-12-03

### Fixed
- **Tool Call Message Content:** Changed from single space `" "` to period `"."` for assistant messages with tool_calls when no content exists. Period is more universally accepted by APIs.

### Added
- **Enhanced Error Logging:** Added detailed error response logging for 500 errors including response body, response text, and error body attributes to help diagnose Heroku API issues.

## [0.5.31] - 2025-12-03

### Fixed
- **Tool Call Message Content:** Changed from empty string `""` to single space `" "` for assistant messages with tool_calls when no content exists. The API rejects both `None` AND empty string with "content is required" error.

## [0.5.30] - 2025-12-03

### Fixed
- **Tool Call Message Content:** Fixed the root cause - `_handle_tool_calls` was setting `content=None` before appending to messages, which overrode any existing LLM response. Now preserves content if exists, uses empty string otherwise.

## [0.5.29] - 2025-12-03

### Changed
- **Tool Call Message Content:** Now preserves the LLM's response text in assistant messages with tool_calls (e.g., "I'll help you with that") instead of always using empty string. Falls back to empty string only if no content exists.

## [0.5.28] - 2025-12-03

### Fixed
- **Tool Call Message Format:** Changed assistant messages with `tool_calls` to use `content=""` (empty string) instead of `content=None`. The API returns 400 error "content is required" when content is None.

## [0.5.27] - 2025-12-03

### Fixed
- **Tool Call Message Format:** Fixed critical bug where assistant messages with `tool_calls` were having their content set to `"Message content unavailable"` or `" "` instead of `None`. Heroku/Claude APIs require `content=None` for assistant messages that contain tool_calls. This was causing 500 errors on follow-up API calls after tool execution.

### Added
- **Follow-up Request Logging:** Added detailed logging of messages being sent in follow-up requests after tool execution, showing role, tool_calls presence, tool_call_id, and content for debugging.

## [0.5.22] - 2025-12-02

### Fixed
- **CloudFront WAF Coverage:** Enhanced sanitization to handle Django `format_html()` calls, HTML entities (`&quot;`, `&lt;`, `&gt;`, `&amp;`), and HTML tags in code (`<span>`, `<a>`, `<H1-6>`) that were triggering CloudFront WAF blocks in PR descriptions and code examples.
- **Extended Thinking Error Detection:** Improved detection of Claude Extended Thinking validation errors to catch more error message variations (`expected thinking`, `redacted_thinking`).
- **Extended Thinking State Restoration:** Fixed bug where `extended_thinking` setting wasn't restored to its original value after retry failures, ensuring the LLM instance remains in a consistent state.

## [0.5.21] - 2025-12-01

### Fixed
- **API Key Handling:** Fixed a bug where the `api_key` parameter passed to the `LLM` constructor was ignored in favor of the `INFERENCE_KEY` environment variable.

## [0.5.20] - 2025-12-01

### Fixed
- **CloudFront WAF Coverage:** Added sanitization for `response =`, `import os`, and `os.system` patterns which were triggering CloudFront WAF blocks.

## [0.5.19] - 2025-12-01

### Fixed
- **CloudFront WAF Coverage:** Sanitizes Django/Jinja template tags to avoid CloudFront 403 blocks on PR diffs containing `{% %}` blocks.

## [0.5.18] - 2025-12-01

### Fixed
- **Duplicate Message Handling:** Fixed an issue where system prompts were duplicated in continuous mode.
- **Duplicate Message Prevention:** Added logic to ignore duplicate system or user messages and log a warning instead of adding them to the conversation history.

## [0.5.17] - 2025-12-01

### Added
- **Verbosity Control:** Added `verbosity` parameter to `LLM` class constructor.
  - Level 0 (default): Normal logging.
  - Level 1: Verbose logging (automatically set if `debug_mode=True`).
  - Level 2: Debug logging including full request data (params, messages, inputs).

## [0.5.16] - 2025-12-01

### Changed
- **Retry Logic:** Updated retry handler to treat HTTP 500 (Internal Server Error) as non-retryable. Previously it was retried, then limited to 1 retry, now it fails immediately.

## [0.5.15] - 2025-11-28

### Added
- **Maximum Recursion Depth Limit with Auto-Correction:** Added configurable limit to prevent infinite tool call loops with intelligent recovery
  - Default limit: 10 iterations (configurable via `max_recursion_depth` param or `LLM_MAX_RECURSION_DEPTH` env var)
  - Logs INFO at each recursion level with tool names and total call count
  - Logs WARNING when depth >= 5 (getting deep)
  - When max depth is reached, injects a system guidance message prompting the LLM to summarize findings
  - Makes a final API call WITHOUT tools to get a proper response from the LLM
  - Includes summary of recent tools called for context
  - Falls back to user-friendly message if final call fails
  - Prevents runaway API costs from infinite tool call loops while still delivering useful output

## [0.5.14] - 2025-11-27

### Added
- **Per-Message Truncation:** Added automatic truncation of individual messages that exceed token limits
  - Messages are truncated to 1/4 of `input_tokens_limit` (default: 10,000 tokens per message)
  - Prevents CloudFront WAF 403 errors caused by massive content (e.g., large PR descriptions)
  - Logs a warning when truncation occurs with original and truncated token counts
  - Truncation happens after WAF sanitization in `_validate_messages_for_api()`

## [0.5.13] - 2025-11-27

### Added
- **InternalServerError Recovery:** Added recovery logic for 500 Internal Server Errors during tool call follow-up requests
  - When Heroku Inference API returns 500 ("Failed to execute chat"), the client now attempts to recover
  - Recovery truncates recent tool message content to a simple success message
  - Retries the API call once with simplified context
  - If recovery fails, the original error is re-raised
  - Applies to both streaming recursive fallback and non-streaming follow-up paths
  - Prevents complete failures when tool output causes server-side parsing issues

## [0.5.12] - 2025-11-27

### Added
- **CloudFrontWAFError Exception:** New custom exception class `CloudFrontWAFError` that is raised when CloudFront WAF blocks a request
  - Contains full context for Sentry: `messages`, `detected_blockers`, `context`, `original_error`
  - Exception message includes formatted details: context, message count, message previews, and detected blockers
  - Exported from `floship_llm` package for easy catching and handling
  - Replaces silent re-raising of `PermissionDeniedError` for WAF-specific 403 errors

### Changed
- **WAF Error Handling:** All CloudFront 403 errors now raise `CloudFrontWAFError` instead of generic `PermissionDeniedError`
  - `prompt()` method raises `CloudFrontWAFError` after all retry attempts fail
  - `prompt_stream()` method raises `CloudFrontWAFError` after all retry attempts fail
  - `_handle_tool_calls()` streaming, recursive, and non-streaming paths all raise `CloudFrontWAFError`
  - Exception contains full message history and detected WAF trigger patterns for debugging in Sentry

## [0.5.11] - 2025-11-27

### Fixed
- **WAF Sanitization Applied to All Messages:** WAF sanitization is now applied to ALL message content in `_validate_messages_for_api()`, not just the user prompt and system message
  - Tool responses containing tracebacks with `exec()`, `File "<script>"`, etc. are now properly sanitized
  - Tool call `arguments` JSON fields are also sanitized to prevent `}}` template injection triggers
  - This fixes WAF 403 blocks when conversation history contains Python execution tool output with tracebacks

## [0.5.10] - 2025-11-27

### Added
- **CloudFront WAF Python Execution Patterns:** Added sanitization for Python execution patterns that trigger WAF when using Python execution tools:
  - `exec()` in tracebacks → `ex3c()` (code execution detection)
  - `File "<script>"` in tracebacks → `File "[SCRIPT_FILE]"` (XSS detection)
  - `File "<string>"` → `File "[STRING_FILE]"`
  - `File "<stdin>"` → `File "[STDIN_FILE]"`
  - JSON template closing `}}}` → `[TEMPLATE_CLOSE]`
- **Desanitize Method:** New `CloudFrontWAFSanitizer.desanitize()` method to restore original content from LLM responses, enabling round-trip sanitization for Python execution tool output
- **WAF Exception Handling:** Added WAF 403 logging to `_handle_tool_calls()` streaming and non-streaming paths
- **Improved WAF Logging:** Enhanced embedding request WAF logging to include input preview

### Fixed
- **Pattern Order:** Reordered WAF sanitization patterns so specific patterns (traceback files) are processed before general patterns (XSS)

## [0.5.9] - 2025-11-26

### Added
- **CloudFront WAF Block Logging:** Added detailed logging when CloudFront WAF blocks requests
  - New `_log_waf_blocked_content()` method logs message analysis when 403 errors occur
  - Logs detected WAF blocker patterns for debugging
  - Shows message previews (truncated to 500 chars) for troubleshooting

## [0.5.8] - 2025-11-26

### Fixed
- **Invalid Tool Name Handling:** Tool names with invalid characters (like `$PYTHON_EXECUTOR`) are now sanitized to valid format (`_PYTHON_EXECUTOR`)
- **Improved Tool Not Found Error:** Error messages now include list of available tools for easier debugging

## [0.5.7] - 2025-11-26

### Added
- **Truncated JSON Detection:** Automatically detects when LLM responses are truncated due to `max_completion_tokens` limit
- **Auto-Retry with Increased Tokens:** When truncation is detected, automatically retries with 2x tokens (up to 2 retries)
- **TruncatedResponseError:** New exception class for truncated response handling

## [0.5.4] - 2025-11-13

### Fixed
- **CloudFront WAF JIRA Image Markup:** Added sanitization for JIRA/Confluence image markup patterns `!filename|options!` that were triggering CloudFront WAF 403 errors
  - Replaces `!filename|options!` with `[IMAGE:filename|options]` to preserve semantic meaning while avoiding WAF triggers
  - Handles short-form `!screenshot.jpg!` and extended options like `|width=...,alt="..."`
  - Added test: `test_jira_image_markup`


## [0.5.3] - 2025-11-03

### Fixed
- **CloudFront WAF Django ORM Issue:** Added sanitization for Django ORM `filter=Q(` pattern that was triggering CloudFront WAF 403 errors as potential SQL injection
  - New `sql_injection` blocker category in `CloudFrontWAFSanitizer`
  - Pattern `filter=Q(` replaced with `filter_Q(` to preserve functionality while removing WAF trigger
  - Commonly used in Django queries like `filter=Q(service_name__in=PICK_PACK_SERVICES)`
  - Added test: `test_django_orm_filter_q_pattern`

## [0.5.2] - 2025-11-03

### Fixed
- **CloudFront WAF JIRA Wiki Markup Issue:** Added sanitization for JIRA/Confluence wiki markup double curly braces `{{text}}` that were triggering CloudFront WAF 403 errors when ticket descriptions were passed to LLM
  - New `wiki_markup` blocker category in `CloudFrontWAFSanitizer`
  - Pattern `{{text}}` replaced with `[text]` to preserve content while removing WAF triggers
  - Commonly used in JIRA for monospace formatting (e.g., `{{deleteShipment}}`, `{{"error message"}}`)
  - Added test: `test_jira_wiki_markup_double_braces`
- **CloudFront WAF GitHub API URL Templates:** Added sanitization for GitHub API URL template patterns (e.g., `{/other_user}`, `{/gist_id}`) that were triggering CloudFront WAF 403 errors when processing GitHub webhook payloads
  - New `url_templates` blocker category in `CloudFrontWAFSanitizer`
  - Pattern `{/[^}]+}` replaced with `[URL_TEMPLATE]`
  - Preserves base URLs while removing template placeholders
  - Added 2 new tests: `test_github_webhook_url_templates` and `test_github_webhook_realistic_payload`

### Changed
- Cleaned up redundant documentation files from repository root

## [0.5.0] - 2025-10-30

### Added
- **Full Embeddings Support:** Complete implementation of Heroku Inference API `/v1/embeddings` endpoint
  - Single and batch text embedding (up to 96 texts per request)
  - All optional parameters: `input_type`, `encoding_format`, `embedding_type`
  - Full response mode with metadata and token usage
  - Input validation (max 96 strings, 2048 characters each)
  - Warnings for oversized inputs
- **Input Type Optimization:** Support for different use cases
  - `search_document` - Optimize embeddings for document indexing
  - `search_query` - Optimize embeddings for search queries
  - `classification` - Optimize embeddings for training classifiers
  - `clustering` - Optimize embeddings for grouping similar items
- **Encoding and Embedding Types:**
  - `encoding_format`: 'float' (default) or 'base64'
  - `embedding_type`: 'float', 'int8', 'uint8', 'binary', 'ubinary'
- **New Pydantic Schemas:**
  - `EmbeddingData` - Single embedding object representation
  - `EmbeddingUsage` - Token usage information
  - `EmbeddingResponse` - Complete API response structure
- **New Example File:** `example_embeddings.py` with 8 comprehensive examples
  - Basic single and batch embeddings
  - Full response with metadata
  - All input types (search_document, search_query, classification, clustering)
  - Practical similarity search demo with numpy
- **Comprehensive Test Suite:** `tests/test_embeddings.py` with 30 tests
  - Initialization and parameter tests
  - Single and batch embedding tests
  - Input validation tests
  - Different input types tests
  - Encoding format tests
  - Metrics tracking tests
  - Schema validation tests
  - Retry behavior tests
- **Enhanced Documentation:**
  - Complete embeddings section in README.md
  - API reference for all embedding parameters
  - Practical examples including similarity search
  - Limits and recommendations clearly documented

### Changed
- **Updated `embed()` method:** Completely rewritten with full API support
  - Now accepts `Union[str, List[str]]` for single or batch inputs
  - Returns `Union[List[float], List[List[float]], Dict]` based on input and options
  - Added `return_full_response` parameter for metadata access
  - Comprehensive input validation and error messages
  - Inherits retry mechanism and WAF protection
- **Updated `get_embedding_params()`:** Added all optional parameters from Heroku API
- **LLM initialization:** Removed exception blocking embedding type
- **Version bumped:** 0.4.0 → 0.5.0

### Fixed
- Test `test_init_embedding_type_not_supported` renamed to `test_init_embedding_type_supported`
- Updated error message in `test_embed_empty_text` to match new validation

## [0.4.0] - 2025-10-28

### Added
- **CloudFront WAF Protection:** Major security and reliability enhancement
  - Automatic content sanitization to prevent CloudFront WAF 403 blocking
  - Sanitizes path traversal patterns (`../` → `[PARENT_DIR]/`)
  - Sanitizes XSS patterns (`<script>` → `[SCRIPT_TAG]`, `<iframe>` → `[IFRAME_TAG]`)
  - Sanitizes JavaScript protocols (`javascript:` → `js:`)
  - Sanitizes event handlers (`onerror=` → `on_error=`)
  - Enabled by default, can be disabled via `enable_waf_sanitization=False`
  - Preserves semantic meaning - LLM can still understand sanitized patterns
- **Automatic Retry on 403:** Exponential backoff with forced sanitization
  - Automatically retries up to 2 times on CloudFront 403 errors
  - Uses exponential backoff: 1s, 2s delays
  - Forces sanitization on retry even if initially disabled
  - Configurable via `max_waf_retries` parameter
- **LLMConfig:** New configuration dataclass for WAF protection settings
  - `enable_waf_sanitization`: Enable/disable sanitization (default: True)
  - `max_waf_retries`: Max retries on 403 (default: 2)
  - `retry_with_sanitization`: Force sanitization on retry (default: True)
  - `debug_mode`: Enable detailed logging (default: False)
  - `log_sanitization`: Log when content is sanitized (default: True)
  - `log_blockers`: Log which patterns were found (default: True)
  - `from_env()`: Load configuration from environment variables
- **LLMMetrics:** Track sanitization and error metrics
  - `total_requests`: Total number of requests made
  - `sanitized_requests`: Number of requests that required sanitization
  - `cloudfront_403_errors`: Number of CloudFront 403 errors encountered
  - `retry_successes`: Number of successful retries
  - `path_traversal_count`: Number of path traversal patterns found
  - `xss_pattern_count`: Number of XSS patterns found
  - `to_dict()`: Convert metrics to dictionary with rates
- **CloudFrontWAFSanitizer:** Core sanitization engine
  - `sanitize()`: Sanitize content and return (sanitized_content, was_sanitized)
  - `check_for_blockers()`: Detect patterns without sanitizing
  - Case-insensitive pattern detection
  - Supports aggressive mode for stricter sanitization
- New methods in LLM class:
  - `get_waf_metrics()`: Get current WAF metrics
  - `reset_waf_metrics()`: Reset metrics counters
  - `_sanitize_for_waf()`: Internal sanitization with logging
  - `_is_cloudfront_403()`: Detect CloudFront 403 errors
- Environment variables support:
  - `FLOSHIP_LLM_WAF_SANITIZE`: Enable sanitization (default: 'true')
  - `FLOSHIP_LLM_DEBUG`: Enable debug mode (default: 'false')
  - `FLOSHIP_LLM_WAF_MAX_RETRIES`: Max retries (default: '2')
- Comprehensive test suite: `tests/test_waf_protection.py`
  - 28 tests covering all WAF protection features
  - Tests for sanitizer, config, metrics, integration, and real-world scenarios
  - 100% passing test rate
- New example: `example_cloudfront_waf.py`
  - 6 complete examples demonstrating WAF protection
  - Shows basic protection, PR diff review, custom config, metrics tracking
- Documentation: Extended README with CloudFront WAF Protection section
  - Problem description and solution
  - Configuration examples
  - Monitoring metrics
  - What gets sanitized (table)
  - Real-world use cases (PR description generator, code review bot)
  - Migration guide from manual sanitization
  - Benefits over manual sanitization

### Changed
- `LLM.__init__()` now accepts WAF-related parameters:
  - `enable_waf_sanitization`: Enable WAF protection (default: True)
  - `waf_config`: LLMConfig instance for advanced configuration
  - `debug_mode`: Enable detailed logging
- `prompt()` method now includes automatic WAF protection:
  - Sanitizes prompt and system messages before sending
  - Automatically retries on 403 errors with exponential backoff
  - Tracks metrics for monitoring
  - Enhanced logging with debug mode
- `prompt_stream()` method now includes automatic WAF protection:
  - Sanitizes content before streaming
  - Automatically retries on 403 errors
  - Tracks metrics for monitoring
- Import statement updated: Added `PermissionDeniedError` for 403 detection
- Import statement updated: Added `dataclass`, `field`, `Tuple` for new classes

### Fixed
- **CloudFront WAF blocking:** Eliminates 403 errors when sending code content
  - PR diffs with path traversal patterns no longer blocked
  - HTML/JS code examples no longer blocked
  - File paths in code comments no longer blocked
- Improved error recovery with automatic retry logic

### Benefits
- ✅ **No more 403 errors** when sending legitimate code content
- ✅ **Automatic** - Works out of the box, no manual setup required
- ✅ **Transparent** - No changes needed to existing code
- ✅ **Resilient** - Automatic retry with sanitization on 403
- ✅ **Configurable** - Can be disabled or customized per use case
- ✅ **Monitored** - Built-in metrics for tracking sanitization rates
- ✅ **Tested** - Comprehensive test suite (28 tests, 100% passing)
- ✅ **Semantic Preservation** - LLM can still understand sanitized patterns
- ✅ **Library-wide** - Benefits all consumers, not just one application
- ✅ **Maintainable** - Centralized solution, easier to update patterns

### Migration Notes
For applications currently using manual sanitization (like support-app):
- **Before:** Manual sanitization with `sanitize_pr_description_content()`
- **After:** Remove manual sanitization, let library handle it automatically
- **Impact:** Can remove 200+ lines of sanitization code
- **Compatibility:** Backward compatible - existing code works unchanged

## [0.4.0] - 2025-10-25

### Added
- **Stream Final Response After Tools:** Major enhancement to streaming support
  - New `stream_final_response` parameter for `prompt()` method
  - Allows streaming the final LLM response even when tools are used
  - Tools execute normally (non-streaming), then final response streams
  - Returns generator when tools are used, string when no tools used
  - Automatically handles recursive tool calls (falls back to non-streaming if more tools detected)
  - Maintains conversation history with streamed final responses
  - 4 comprehensive tests covering all scenarios
- New `example_stream_with_tools.py` with 4 real-world examples
- Updated README with "Streaming Final Response After Tools" section

### Changed
- `prompt()` method now accepts `stream_final_response` parameter (default: False)
- `process_response()` method now accepts `stream_final_response` parameter
- `_handle_tool_calls()` method now accepts `stream_final_response` parameter
- Enhanced tool execution flow to support streaming final responses

### Benefits
- Better UX: See responses as they arrive, even with tools
- Flexible: Returns string for backward compatibility when tools aren't used
- Intelligent: Automatically detects and handles recursive tool calls
- Backward compatible: Existing code works unchanged

## [0.3.0] - 2025-10-25

### Added
- **Streaming Support:** Real-time response streaming for better UX
  - New `prompt_stream()` method for streaming completions
  - Yields response chunks as they arrive from the API
  - Maintains conversation history with streamed content
  - Respects `continuous` setting (resets when False)
  - Works with system messages and temperature settings
  - Comprehensive test coverage for streaming functionality
- New `stream` initialization parameter (default: False)
- Documentation in README with streaming examples
- New `examples_streaming.py` with 5 streaming examples

### Changed
- **Breaking:** Streaming mode does NOT support tool calls
  - `prompt_stream()` raises `ValueError` if tools are enabled
  - This is a technical limitation of the OpenAI API
  - Use regular `prompt()` method when tools are needed

### Notes
- Streaming responses do not support retry mechanism
- Tool tracking is reset but not used in streaming mode
- Backward compatible: existing code continues to work unchanged

## [0.2.0] - 2025-10-25

### Added
- **Tool Call Tracking:** Comprehensive tracking of tool invocations for resource budgeting and monitoring
  - New public API methods:
    - `get_last_tool_call_count()` - Returns total number of tool calls from last prompt
    - `get_last_tool_history()` - Returns detailed history with metadata for each tool call
    - `get_last_recursion_depth()` - Returns maximum recursion depth reached
    - `get_last_response_metadata()` - Returns complete metadata dictionary
  - Tracks detailed metadata for each tool call:
    - Tool name, arguments, and execution time
    - Recursion depth (for nested tool calls)
    - Result length and token usage
    - Timestamp and error status (for failed calls)
  - Enables accurate resource budgeting, cost estimation, and rate limiting
  - See `TOOL_TRACKING_IMPLEMENTATION.md` for complete documentation

### Changed
- Enhanced `_handle_tool_calls()` method to track recursion depth
- Updated `prompt()` method to reset and store tracking metadata
- Improved logging to show tool call progression and recursion depth

### Fixed
- Fixed error handling in tool call tracking when JSON parsing fails
- Ensured `arguments` variable is properly initialized before use in error handling

### Testing
- All 260 tests passing
- Test coverage: 93%
- Added comprehensive tracking validation

## [0.1.1] - 2025-10-25

### Fixed
- **Critical:** Fixed `TypeError: Object of type ToolParameter is not JSON serializable` when using tools with `LLM.prompt()`
  - Updated `ToolManager.get_tools_schema()` to properly serialize `ToolParameter` objects to dictionaries
  - Tools are now correctly converted to OpenAI format using the existing `to_openai_format()` method

### Changed
- **Heroku Inference API Compatibility:** Updated client to align with Heroku Inference API v1 specifications
  - Deprecated `frequency_penalty` and `presence_penalty` (not supported by Heroku)
  - Added Heroku-specific parameters: `max_completion_tokens`, `top_k`, `top_p`, `extended_thinking`
  - Added `allow_ignored_params` flag for backward compatibility
  - Updated `supports_frequency_penalty` and `supports_presence_penalty` properties to return `False` for all models
- Updated README.md with Heroku-specific configuration and examples
- Added extended thinking support for Claude models (Claude 4.5 Sonnet, 4.5 Haiku, 4 Sonnet, 3.7 Sonnet)

### Added
- New test suite for tool serialization (`test_tool_serialization.py`) with 4 comprehensive tests
- New test cases for Heroku-specific parameters
- Documentation: `HEROKU_MIGRATION.md` - Complete migration guide for Heroku Inference API
- Documentation: `TOOL_SERIALIZATION_FIX.md` - Details of the serialization fix
- Verification script: `verify_tool_fix.py` - Demonstrates the fix works correctly

### Testing
- All 260 tests passing (up from 256)
- Maintained 94.42% test coverage
- Added tests for extended thinking, Heroku parameters, and tool serialization

## [0.1.0] - 2025-10-22

### Added
- Initial release of Floship LLM library
- Core `LLM` class for OpenAI-compatible API interactions
- Support for continuous (multi-turn) conversations
- Pydantic schema validation for structured outputs
- JSON utilities for parsing LLM responses
- Pre-built schemas: `ThinkingModel`, `Suggestion`, `SuggestionsResponse`, `Labels`
- Configurable parameters: temperature, frequency penalty, presence penalty
- Token limit management for input and output
- Model-specific feature detection (Claude, Gemini compatibility)
- Comprehensive documentation and examples
- MIT License

### Features
- Zero Django dependencies for maximum reusability
- Works with any OpenAI-compatible inference endpoint
- Automatic JSON extraction and validation from responses
- Conversation history management
- System prompt support
- Response format enforcement with Pydantic models

[0.1.1]: https://github.com/Floship/floship-llm/releases/tag/v0.1.1
[0.1.0]: https://github.com/Floship/floship-llm/releases/tag/v0.1.0

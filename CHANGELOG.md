# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

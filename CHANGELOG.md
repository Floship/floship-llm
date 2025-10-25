# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

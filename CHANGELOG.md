# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

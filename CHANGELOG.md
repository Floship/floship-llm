# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/Floship/floship-llm/releases/tag/v0.1.0

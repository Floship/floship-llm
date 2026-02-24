"""Floship LLM Client Library.

A reusable LLM client library for interacting with OpenAI-compatible inference endpoints.
"""

from __future__ import annotations

from .client import (
    LLM,
    CloudFrontWAFError,
    CloudFrontWAFSanitizer,
    LLMConfig,
    LLMMetrics,
    TruncatedResponseError,
)
from .content_processor import ContentProcessor
from .retry_handler import RetryHandler
from .schemas import (
    EmbeddingData,
    EmbeddingResponse,
    EmbeddingUsage,
    Labels,
    Suggestion,
    SuggestionsResponse,
    ThinkingModel,
    ToolCall,
    ToolFunction,
    ToolParameter,
    ToolResult,
)
from .tool_manager import ToolManager
from .utils import JSONUtils, lm_json_utils

__all__ = [
    "LLM",
    "CloudFrontWAFError",
    "CloudFrontWAFSanitizer",
    "ContentProcessor",
    "EmbeddingData",
    "EmbeddingResponse",
    "EmbeddingUsage",
    "JSONUtils",
    "LLMConfig",
    "LLMMetrics",
    "Labels",
    "RetryHandler",
    "Suggestion",
    "SuggestionsResponse",
    "ThinkingModel",
    "ToolCall",
    "ToolFunction",
    "ToolManager",
    "ToolParameter",
    "ToolResult",
    "TruncatedResponseError",
    "lm_json_utils",
]

__version__ = "0.5.50"

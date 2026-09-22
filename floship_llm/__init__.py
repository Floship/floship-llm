"""Floship LLM Client Library.

A reusable LLM client library for interacting with OpenAI-compatible inference endpoints.
"""

from __future__ import annotations

from .backends.systemone import SystemOneBackend, SystemOneError
from .client import (
    LLM,
    CloudFrontWAFError,
    CloudFrontWAFSanitizer,
    LLMConfig,
    LLMMetrics,
    TruncatedResponseError,
)
from .content_processor import ContentProcessor
from .google_cache_manager import ContextCacheRef, GoogleCacheManager
from .retry_handler import RetryHandler
from .schemas import (
    CacheInfo,
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
from .systemone import (
    Choice,
    ChoiceAnswer,
    DecisionsResponse,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
)
from .tool_manager import ToolManager
from .utils import JSONUtils, lm_json_utils

__all__ = [
    "LLM",
    "CacheInfo",
    "Choice",
    "ChoiceAnswer",
    "CloudFrontWAFError",
    "CloudFrontWAFSanitizer",
    "ContentProcessor",
    "ContextCacheRef",
    "DecisionsResponse",
    "EmbeddingData",
    "EmbeddingResponse",
    "EmbeddingUsage",
    "GoogleCacheManager",
    "JSONUtils",
    "LLMConfig",
    "LLMMetrics",
    "Labels",
    "Noul",
    "NoulAnswer",
    "RetryHandler",
    "Score",
    "ScoreAnswer",
    "Suggestion",
    "SuggestionsResponse",
    "SystemOneBackend",
    "SystemOneError",
    "ThinkingModel",
    "ToolCall",
    "ToolFunction",
    "ToolManager",
    "ToolParameter",
    "ToolResult",
    "TruncatedResponseError",
    "lm_json_utils",
]

__version__ = "1.7.0"

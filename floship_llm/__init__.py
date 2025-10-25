"""Floship LLM Client Library

A reusable LLM client library for interacting with OpenAI-compatible inference endpoints.
"""

from .client import LLM
from .content_processor import ContentProcessor
from .retry_handler import RetryHandler
from .schemas import (
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

__version__ = "0.3.0"

__all__ = [
    "LLM",
    "ThinkingModel",
    "Suggestion",
    "SuggestionsResponse",
    "Labels",
    "ToolParameter",
    "ToolFunction",
    "ToolCall",
    "ToolResult",
    "lm_json_utils",
    "JSONUtils",
]

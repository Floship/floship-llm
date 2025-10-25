"""Floship LLM Client Library

A reusable LLM client library for interacting with OpenAI-compatible inference endpoints.
"""

from .client import LLM
from .schemas import (
    ThinkingModel, 
    Suggestion, 
    SuggestionsResponse, 
    Labels,
    ToolParameter,
    ToolFunction,
    ToolCall,
    ToolResult
)
from .utils import lm_json_utils, JSONUtils
from .retry_handler import RetryHandler
from .tool_manager import ToolManager
from .content_processor import ContentProcessor

__version__ = "0.1.0"

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

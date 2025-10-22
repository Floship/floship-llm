"""Floship LLM Client Library

A reusable LLM client library for interacting with OpenAI-compatible inference endpoints.
"""

from .client import LLM
from .schemas import ThinkingModel, Suggestion, SuggestionsResponse, Labels
from .utils import lm_json_utils, JSONUtils

__version__ = "0.1.0"

__all__ = [
    "LLM",
    "ThinkingModel",
    "Suggestion",
    "SuggestionsResponse",
    "Labels",
    "lm_json_utils",
    "JSONUtils",
]

"""Main LLM client for inference with tool support."""

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from distutils.util import strtobool
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from openai import OpenAI, PermissionDeniedError
from pydantic import BaseModel

from .content_processor import ContentProcessor
from .retry_handler import RetryHandler
from .schemas import ToolCall, ToolFunction, ToolResult
from .tool_manager import ToolManager
from .utils import lm_json_utils

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM client behavior."""

    # Sanitization settings
    enable_waf_sanitization: bool = True
    sanitization_aggressive: bool = False

    # Retry settings
    max_waf_retries: int = 2
    retry_with_sanitization: bool = True
    retry_delay_base: float = 1.0  # Base delay for exponential backoff

    # Logging
    debug_mode: bool = False
    log_sanitization: bool = True
    log_blockers: bool = True

    # CloudFront WAF specific
    cloudfront_waf_detection: bool = True  # Auto-detect CloudFront errors

    @classmethod
    def from_env(cls):
        """Load configuration from environment variables."""
        return cls(
            enable_waf_sanitization=os.getenv(
                "FLOSHIP_LLM_WAF_SANITIZE", "true"
            ).lower()
            == "true",
            debug_mode=os.getenv("FLOSHIP_LLM_DEBUG", "false").lower() == "true",
            max_waf_retries=int(os.getenv("FLOSHIP_LLM_WAF_MAX_RETRIES", "2")),
        )


@dataclass
class LLMMetrics:
    """Track LLM client metrics."""

    total_requests: int = 0
    sanitized_requests: int = 0
    failed_requests: int = 0
    cloudfront_403_errors: int = 0
    retry_successes: int = 0

    # Pattern frequencies
    path_traversal_count: int = 0
    xss_pattern_count: int = 0

    def to_dict(self):
        """Convert metrics to dictionary."""
        return {
            "total_requests": self.total_requests,
            "sanitization_rate": self.sanitized_requests / max(self.total_requests, 1),
            "error_rate": self.failed_requests / max(self.total_requests, 1),
            "cloudfront_403_rate": (
                self.cloudfront_403_errors / max(self.failed_requests, 1)
                if self.failed_requests > 0
                else 0
            ),
        }


class CloudFrontWAFSanitizer:
    """
    Sanitize content to prevent CloudFront WAF blocking.

    CloudFront's Web Application Firewall blocks requests containing patterns
    that resemble security attacks. This sanitizer replaces those patterns with
    safe alternatives while preserving semantic meaning.
    """

    # Patterns that trigger CloudFront WAF
    BLOCKERS = {
        "path_traversal": [
            (r"\.\./", "[PARENT_DIR]/"),
            (r"\.\.[/\\]", "[PARENT_DIR]/"),
            (r"\.\.\\\\", "[PARENT_DIR]\\\\"),
        ],
        "xss": [
            (r"<script[^>]*>", "[SCRIPT_TAG]"),
            (r"</script>", "[/SCRIPT_TAG]"),
            (r"<iframe[^>]*>", "[IFRAME_TAG]"),
            (r"</iframe>", "[/IFRAME_TAG]"),
            (r"javascript:", "js:"),
            (r"onerror\s*=", "on_error="),
            (r"onload\s*=", "on_load="),
        ],
        "url_templates": [
            # GitHub API URL templates like {/other_user}, {/gist_id}, etc.
            (r"\{/[^}]+\}", "[URL_TEMPLATE]"),
        ],
    }

    @classmethod
    def sanitize(cls, content: str, aggressive: bool = False) -> Tuple[str, bool]:
        """
        Sanitize content to prevent WAF blocking.

        Args:
            content: Content to sanitize
            aggressive: If True, apply more aggressive sanitization

        Returns:
            Tuple of (sanitized_content, was_sanitized)
        """
        sanitized = content
        was_sanitized = False

        for category, patterns in cls.BLOCKERS.items():
            for pattern, replacement in patterns:
                if re.search(pattern, sanitized, re.IGNORECASE):
                    sanitized = re.sub(
                        pattern, replacement, sanitized, flags=re.IGNORECASE
                    )
                    was_sanitized = True

        return sanitized, was_sanitized

    @classmethod
    def check_for_blockers(cls, content: str) -> List[Tuple[str, str]]:
        """
        Check if content contains patterns that would trigger WAF.

        Args:
            content: Content to check

        Returns:
            List of tuples (category, pattern) found in content
        """
        found = []
        for category, patterns in cls.BLOCKERS.items():
            for pattern, _ in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    found.append((category, pattern))
        return found


class LLM:
    """
    Main LLM client with support for completions, embeddings, and function calling.

    Features:
    - Automatic retry with exponential backoff
    - Tool/function calling support
    - Content sanitization for CloudFront WAF compatibility
    - Token management and truncation
    - Conversation history management
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize LLM client for Heroku Inference API.

        Args:
            type: 'completion' or 'embedding' (default: 'completion')
            model: Model ID (defaults to INFERENCE_MODEL_ID env var)
            temperature: Sampling temperature, 0.0-1.0 (default: 0.15)
            max_completion_tokens: Maximum tokens to generate (default: None)
            top_k: Sample from top K options (default: None)
            top_p: Nucleus sampling threshold, 0.0-1.0 (default: None)
            extended_thinking: Enable extended thinking for Claude models (default: None)
                Example: {"enabled": True, "budget_tokens": 1024, "include_reasoning": True}
            allow_ignored_params: Allow unsupported parameters without error (default: False)
            response_format: Pydantic model for structured output
            continuous: Keep conversation history (default: True)
            messages: Initial conversation history
            max_length: Maximum response length (default: 100,000)
            input_tokens_limit: Maximum input tokens (default: 40,000)
            max_retry: Maximum retry attempts (default: 3)
            system: System prompt
            enable_tools: Enable tool calling (default: False)
            sanitize_tool_responses: Sanitize tool responses (default: True)
            sanitization_patterns: Custom sanitization patterns
            max_tool_response_tokens: Max tokens in tool responses (default: 4000)
            track_tool_metadata: Track tool execution metadata (default: False)
            stream: Enable streaming responses (default: False)

            CloudFront WAF Protection (NEW):
            enable_waf_sanitization: Auto-sanitize content to prevent CloudFront WAF blocking (default: True)
            waf_config: LLMConfig instance for WAF protection settings (default: None, uses defaults)
            debug_mode: Enable detailed request/response logging (default: False)

            Embeddings Parameters (when type='embedding'):
            input_type: Type of input - 'search_document', 'search_query', 'classification', 'clustering' (default: None)
            encoding_format: Output encoding - 'float' or 'base64' (default: 'float')
            embedding_type: Embedding type - 'float', 'int8', 'uint8', 'binary', 'ubinary' (default: 'float')

        Note:
            frequency_penalty and presence_penalty are NOT supported by Heroku Inference API
            and will be ignored. Use allow_ignored_params=True to include them without error.

        Raises:
            ValueError: If required environment variables are missing

        Environment Variables Required:
            INFERENCE_URL: Heroku Inference API endpoint (e.g., https://us.inference.heroku.com)
            INFERENCE_MODEL_ID: Model identifier (e.g., claude-4-sonnet)
            INFERENCE_KEY: API key for authentication

        Environment Variables Optional (CloudFront WAF):
            FLOSHIP_LLM_WAF_SANITIZE: Enable WAF sanitization (default: 'true')
            FLOSHIP_LLM_DEBUG: Enable debug mode (default: 'false')
            FLOSHIP_LLM_WAF_MAX_RETRIES: Max retries on 403 (default: '2')
        """
        # Validate required environment variables
        self._validate_environment()

        # CloudFront WAF protection configuration
        self.waf_config = kwargs.get("waf_config", None) or LLMConfig()
        self.waf_sanitizer = CloudFrontWAFSanitizer()
        self.waf_metrics = LLMMetrics()

        # Override config with explicit parameters
        if "enable_waf_sanitization" in kwargs:
            self.waf_config.enable_waf_sanitization = kwargs["enable_waf_sanitization"]
        if "debug_mode" in kwargs:
            self.waf_config.debug_mode = kwargs["debug_mode"]

        # Basic configuration
        self.type = kwargs.get("type", "completion")
        if self.type not in ["completion", "embedding"]:
            raise ValueError("type must be 'completion' or 'embedding'")

        # Initialize OpenAI client
        self.base_url = os.environ["INFERENCE_URL"]
        self.client = OpenAI(
            api_key=os.environ["INFERENCE_KEY"], base_url=self.base_url
        )

        # Model configuration
        self.model = kwargs.get("model", os.environ["INFERENCE_MODEL_ID"])
        self.temperature = kwargs.get("temperature", 0.15)

        # Heroku-specific parameters (for completions)
        self.max_completion_tokens = kwargs.get("max_completion_tokens", None)
        self.top_k = kwargs.get("top_k", None)
        self.top_p = kwargs.get("top_p", None)
        self.extended_thinking = kwargs.get("extended_thinking", None)
        self.allow_ignored_params = kwargs.get("allow_ignored_params", False)

        # Heroku-specific parameters (for embeddings)
        self.input_type = kwargs.get(
            "input_type", None
        )  # search_document, search_query, classification, clustering
        self.encoding_format = kwargs.get("encoding_format", "float")  # float, base64
        self.embedding_type = kwargs.get(
            "embedding_type", "float"
        )  # float, int8, uint8, binary, ubinary

        # Deprecated parameters (Heroku ignores these but won't error)
        # Kept for backward compatibility but not recommended
        self.frequency_penalty = kwargs.get("frequency_penalty", None)
        self.presence_penalty = kwargs.get("presence_penalty", None)

        # Response format (structured output)
        self.response_format = kwargs.get("response_format", None)
        if self.response_format and not issubclass(self.response_format, BaseModel):
            raise ValueError(
                "response_format must be a subclass of BaseModel (pydantic)"
            )

        # Conversation management
        self.continuous = kwargs.get("continuous", True)
        self.messages = kwargs.get("messages", [])
        self.max_length = kwargs.get("max_length", 100_000)
        self.input_tokens_limit = kwargs.get("input_tokens_limit", 40_000)
        self.max_retry = kwargs.get("max_retry", 3)
        self.retry_count = 0
        self.system = kwargs.get("system", None)

        # Initialize handlers
        self.retry_handler = RetryHandler(max_retries=self.max_retry)
        self.tool_manager = ToolManager()
        self.content_processor = ContentProcessor(
            sanitize_enabled=kwargs.get("sanitize_tool_responses", True),
            sanitization_patterns=kwargs.get("sanitization_patterns", None),
            max_tokens=kwargs.get("max_tool_response_tokens", 4000),
            track_metadata=kwargs.get("track_tool_metadata", False),
        )

        # Streaming configuration
        self.stream = kwargs.get("stream", False)

        # Tool configuration
        self.enable_tools = kwargs.get("enable_tools", False)
        self.tool_request_delay = float(os.environ.get("LLM_TOOL_REQUEST_DELAY", "0"))

        # Tool call tracking (for monitoring and budgeting)
        self._current_tool_call_count = 0
        self._current_recursion_depth = 0
        self._current_tool_history = []
        self._last_response_metadata = {}

        # Backward compatibility - maintain tools dict reference
        self.tools = self.tool_manager.tools

        # Add system message if provided
        if self.system:
            self.add_message("system", self.system)

    def _validate_environment(self):
        """Validate required environment variables."""
        required_vars = ["INFERENCE_URL", "INFERENCE_MODEL_ID", "INFERENCE_KEY"]
        for var in required_vars:
            if not os.environ.get(var):
                raise ValueError(f"{var} environment variable must be set.")

    # ========== CloudFront WAF Protection ==========

    def _sanitize_for_waf(self, content: str) -> str:
        """
        Sanitize content to prevent CloudFront WAF blocking.

        Args:
            content: Content to sanitize

        Returns:
            Sanitized content
        """
        if not self.waf_config.enable_waf_sanitization:
            return content

        sanitized, was_sanitized = self.waf_sanitizer.sanitize(
            content, aggressive=self.waf_config.sanitization_aggressive
        )

        if was_sanitized:
            self.waf_metrics.sanitized_requests += 1

            if self.waf_config.log_sanitization:
                logger.warning("CloudFront WAF: Content sanitized to prevent blocking")

            if self.waf_config.log_blockers:
                blockers = self.waf_sanitizer.check_for_blockers(content)
                if blockers:
                    logger.debug(f"CloudFront WAF blockers found: {blockers}")
                    # Update pattern counters
                    for category, _ in blockers:
                        if category == "path_traversal":
                            self.waf_metrics.path_traversal_count += 1
                        elif category == "xss":
                            self.waf_metrics.xss_pattern_count += 1

        return sanitized

    def _is_cloudfront_403(self, error: Exception) -> bool:
        """
        Check if error is a CloudFront 403 error.

        Args:
            error: Exception to check

        Returns:
            True if error is CloudFront 403
        """
        if not isinstance(error, PermissionDeniedError):
            return False

        error_str = str(error).lower()
        return "403" in error_str or "forbidden" in error_str

    def get_waf_metrics(self) -> dict:
        """Get current CloudFront WAF metrics."""
        return self.waf_metrics.to_dict()

    def reset_waf_metrics(self):
        """Reset CloudFront WAF metrics counters."""
        self.waf_metrics = LLMMetrics()

    # ========== Model Capability Properties ==========

    @property
    def supports_parallel_requests(self) -> bool:
        """Check if model supports parallel requests."""
        return strtobool(os.environ.get("INFERENCE_SUPPORTS_PARALLEL_REQUESTS", "True"))

    @property
    def supports_frequency_penalty(self) -> bool:
        """
        Check if model supports frequency penalty.

        Note: Heroku Inference API ignores frequency_penalty but won't return an error.
        This parameter is deprecated for Heroku.
        """
        logger.warning(
            "frequency_penalty is not supported by Heroku Inference API and will be ignored"
        )
        return False

    @property
    def supports_presence_penalty(self) -> bool:
        """
        Check if model supports presence penalty.

        Note: Heroku Inference API ignores presence_penalty but won't return an error.
        This parameter is deprecated for Heroku.
        """
        logger.warning(
            "presence_penalty is not supported by Heroku Inference API and will be ignored"
        )
        return False

    @property
    def require_response_format(self) -> bool:
        """Check if response format is required."""
        return self.response_format is not None and issubclass(
            self.response_format, BaseModel
        )

    # ========== Message Management ==========

    def add_message(self, role: str, content: str):
        """
        Add a message to conversation history.

        Args:
            role: Message role ('user', 'assistant', or 'system')
            content: Message content

        Raises:
            ValueError: If role is invalid
        """
        if not isinstance(role, str) or role not in ["user", "assistant", "system"]:
            raise ValueError("Role must be 'user', 'assistant', or 'system'")

        content = self._sanitize_messages(content)

        if self.require_response_format:
            content = (
                f"{content}\n"
                f"Here is the JSON schema you need to follow for the response: "
                f"{self.response_format.schema_json(indent=2)}\n"
                f"Do not return the entire schema, only the response in JSON format. "
                f"Use the schema only as a guide for filling in.\n"
            )

        self.messages.append({"role": role, "content": content})

    def _sanitize_messages(self, content: str) -> str:
        """Remove excessive whitespace from messages."""
        return self.content_processor.sanitize_messages(content)

    def reset(self):
        """Reset conversation history and retry counter."""
        self.messages = []
        self.retry_count = 0
        if self.system:
            self.add_message("system", self.system)

    # ========== API Request Management ==========

    def get_request_params(self) -> Dict[str, Any]:
        """
        Build parameters for completion request according to Heroku Inference API spec.

        See: https://devcenter.heroku.com/articles/heroku-inference-api-v1-chat-completions

        Note: Claude models don't allow both temperature and top_p simultaneously.
        Note: When extended_thinking is enabled, temperature must be set to 1.
        """
        params = {
            "model": self.model,
        }

        # Heroku-specific parameters that need to go in extra_body for OpenAI client
        extra_body = {}

        # Check if extended thinking is enabled
        has_extended_thinking = self.extended_thinking is not None and (
            (
                isinstance(self.extended_thinking, dict)
                and self.extended_thinking.get("enabled", False)
            )
            or (isinstance(self.extended_thinking, bool) and self.extended_thinking)
        )

        # Add temperature or top_p (not both for Claude)
        if has_extended_thinking:
            # Extended thinking requires temperature = 1
            params["temperature"] = 1.0
        elif self.top_p is not None:
            # If top_p is set, use it instead of temperature
            extra_body["top_p"] = self.top_p
        else:
            # Otherwise use temperature
            params["temperature"] = self.temperature

        # Add Heroku-supported optional parameters
        if self.max_completion_tokens is not None:
            params["max_completion_tokens"] = self.max_completion_tokens

        if self.top_k is not None:
            extra_body["top_k"] = self.top_k

        if self.extended_thinking is not None:
            extra_body["extended_thinking"] = self.extended_thinking

        # Add extra_body if it has content
        if extra_body:
            params["extra_body"] = extra_body

        # Legacy parameters (not recommended, Heroku ignores these)
        # Only include if allow_ignored_params is True (internal flag, not sent to API)
        if self.frequency_penalty is not None and self.allow_ignored_params:
            params["frequency_penalty"] = self.frequency_penalty

        if self.presence_penalty is not None and self.allow_ignored_params:
            params["presence_penalty"] = self.presence_penalty

        # Add tools if enabled
        if self.enable_tools and self.tool_manager.tools:
            params["tools"] = self.tool_manager.get_tools_schema()
            params["tool_choice"] = "auto"

        return params

    def get_embedding_params(self) -> Dict[str, Any]:
        """
        Build parameters for embedding request according to Heroku Inference API spec.

        See: https://devcenter.heroku.com/articles/heroku-inference-api-v1-embeddings

        Returns:
            Dictionary with embedding request parameters
        """
        params = {
            "model": self.model,
        }

        # Add optional parameters if specified
        if self.input_type is not None:
            params["input_type"] = self.input_type

        if self.encoding_format != "float":
            params["encoding_format"] = self.encoding_format

        if self.embedding_type != "float":
            params["embedding_type"] = self.embedding_type

        if self.allow_ignored_params:
            params["allow_ignored_params"] = self.allow_ignored_params

        return params

    # ========== Core API Methods ==========

    def embed(
        self, input: Union[str, List[str]], return_full_response: bool = False
    ) -> Union[List[float], List[List[float]], Dict[str, Any]]:
        """
        Generate embeddings for text using Heroku Inference API.

        Supports single strings or arrays of strings (max 96 strings, 2048 characters each).
        Recommended: length less than 512 tokens per string for optimal performance.

        Args:
            input: Single string or list of strings to embed
            return_full_response: If True, return full API response with metadata.
                                 If False, return just the embedding vector(s).

        Returns:
            - If input is a single string and return_full_response=False: List[float] (single embedding)
            - If input is a list and return_full_response=False: List[List[float]] (multiple embeddings)
            - If return_full_response=True: Dict with full API response including usage metadata

        Raises:
            ValueError: If input is empty or invalid

        Example:
            >>> # Single embedding
            >>> llm = LLM(type='embedding', model='cohere-embed-multilingual')
            >>> embedding = llm.embed("Hello world")
            >>> print(len(embedding))  # 1024 (or model-specific dimension)

            >>> # Multiple embeddings
            >>> embeddings = llm.embed(["Text 1", "Text 2"])
            >>> print(len(embeddings))  # 2

            >>> # Full response with metadata
            >>> response = llm.embed("Hello", return_full_response=True)
            >>> print(response['usage'])  # {'prompt_tokens': 2, 'total_tokens': 2}
        """
        # Validate input
        if not input:
            raise ValueError("Input cannot be empty for embedding.")

        if isinstance(input, list):
            if len(input) == 0:
                raise ValueError("Input list cannot be empty.")
            if len(input) > 96:
                raise ValueError(
                    "Input list cannot contain more than 96 strings (API limit)."
                )
            for idx, text in enumerate(input):
                if not text or not isinstance(text, str):
                    raise ValueError(
                        f"Input at index {idx} must be a non-empty string."
                    )
                if len(text) > 2048:
                    logger.warning(
                        f"Input at index {idx} has {len(text)} characters. "
                        "API recommends max 2048 characters per string."
                    )
        elif isinstance(input, str):
            if len(input) > 2048:
                logger.warning(
                    f"Input has {len(input)} characters. "
                    "API recommends max 2048 characters per string."
                )
        else:
            raise ValueError("Input must be a string or list of strings.")

        params = self.get_embedding_params()
        logger.info(f"Generating embeddings with parameters: {params}")

        # Track metrics
        self.waf_metrics.total_requests += 1

        try:
            response = self.retry_handler.execute_with_retry(
                self.client.embeddings.create, **params, input=input
            )

            # Return full response if requested
            if return_full_response:
                return {
                    "object": (
                        response.object if hasattr(response, "object") else "list"
                    ),
                    "data": [
                        {
                            "object": (
                                item.object if hasattr(item, "object") else "embedding"
                            ),
                            "index": item.index,
                            "embedding": item.embedding,
                        }
                        for item in response.data
                    ],
                    "model": response.model,
                    "usage": (
                        {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "total_tokens": response.usage.total_tokens,
                        }
                        if hasattr(response, "usage") and response.usage
                        else None
                    ),
                }

            # Return just the embeddings
            if isinstance(input, list):
                # Multiple inputs - return list of embeddings
                return [item.embedding for item in response.data]
            else:
                # Single input - return single embedding
                return response.data[0].embedding if response.data else None

        except PermissionDeniedError as e:
            # Track CloudFront 403 errors
            if self._is_cloudfront_403(e):
                self.waf_metrics.cloudfront_403_errors += 1
                logger.error("CloudFront WAF blocked embedding request (403 error)")
            self.waf_metrics.failed_requests += 1
            raise
        except Exception as e:
            self.waf_metrics.failed_requests += 1
            raise

    def prompt(
        self,
        prompt: Optional[str] = None,
        system: Optional[str] = None,
        retry: bool = False,
        stream_final_response: bool = False,
    ) -> str:
        """
        Generate completion from prompt with automatic CloudFront WAF protection.

        Args:
            prompt: User prompt
            system: Optional system message
            retry: Internal retry flag
            stream_final_response: If True and tools are used, stream the final response
                                  after tools complete (generator). If False or no tools
                                  are used, returns complete response as string.

        Returns:
            Generated response text, structured output, or generator (if streaming final response)

        Note:
            CloudFront WAF Protection: If a 403 error occurs, the request will automatically
            retry with content sanitization enabled. This prevents blocking when sending
            code diffs, file patches, or other content with patterns resembling attacks.

            After calling this method, you can access tool call metadata via:
            - llm.get_last_tool_call_count(): Number of tool calls made
            - llm.get_last_tool_history(): List of tools called with details
            - llm.get_last_recursion_depth(): How deep the tool call chain went

        Example:
            # Regular usage
            response = llm.prompt("What's 2+2?")

            # Streaming final response after tools
            result = llm.prompt("What's 2+2?", stream_final_response=True)
            if isinstance(result, str):
                print(result)  # No tools were used
            else:
                for chunk in result:  # Tools were used, streaming final response
                    print(chunk, end="", flush=True)
        """
        # Track metrics
        self.waf_metrics.total_requests += 1

        # Reset tool tracking for new prompt
        if not retry:
            self._current_tool_call_count = 0
            self._current_recursion_depth = 0
            self._current_tool_history = []
            self._last_response_metadata = {}

        # Sanitize prompt and system message if WAF protection is enabled
        original_prompt = prompt
        original_system = system

        if prompt and not retry:
            self.retry_count = 0

            # Apply WAF sanitization
            if self.waf_config.enable_waf_sanitization:
                prompt = self._sanitize_for_waf(prompt)
                if system:
                    system = self._sanitize_for_waf(system)

            if system:
                self.add_message("system", system)
            self.add_message("user", prompt)

        # Retry loop for CloudFront WAF 403 errors
        last_error = None
        for waf_attempt in range(self.waf_config.max_waf_retries + 1):
            try:
                if self.waf_config.debug_mode:
                    logger.debug(
                        f"Request attempt {waf_attempt + 1}/{self.waf_config.max_waf_retries + 1}"
                    )
                    if waf_attempt > 0:
                        logger.debug("Retrying with forced sanitization")

                params = self.get_request_params()
                if self.waf_config.debug_mode:
                    logger.debug(f"Prompting LLM with parameters: {params}")
                    logger.debug(f"Message count: {len(self.messages)}")

                # Validate messages before sending
                validated_messages = self._validate_messages_for_api(self.messages)

                response = self.retry_handler.execute_with_retry(
                    self.client.chat.completions.create,
                    **params,
                    messages=validated_messages,
                )

                result = self.process_response(
                    response, stream_final_response=stream_final_response
                )

                # Store final metadata
                self._last_response_metadata = {
                    "total_tool_calls": self._current_tool_call_count,
                    "recursion_depth": self._current_recursion_depth,
                    "tool_history": self._current_tool_history.copy(),
                }

                if not self.continuous:
                    self.reset()

                # Track successful retry
                if waf_attempt > 0:
                    self.waf_metrics.retry_successes += 1
                    logger.info(
                        f"CloudFront WAF: Retry successful after {waf_attempt} attempts"
                    )

                return result

            except Exception as e:
                last_error = e

                # Check if it's a CloudFront 403 error
                if (
                    self._is_cloudfront_403(e)
                    and waf_attempt < self.waf_config.max_waf_retries
                ):
                    self.waf_metrics.cloudfront_403_errors += 1
                    wait_time = self.waf_config.retry_delay_base * (2**waf_attempt)

                    logger.warning(
                        f"CloudFront WAF 403 error (attempt {waf_attempt + 1}/"
                        f"{self.waf_config.max_waf_retries + 1}). "
                        f"Retrying in {wait_time}s with sanitization..."
                    )

                    # Wait before retry
                    time.sleep(wait_time)

                    # Force sanitization on retry if not already enabled
                    if (
                        not self.waf_config.enable_waf_sanitization
                        and self.waf_config.retry_with_sanitization
                    ):
                        # Re-add messages with sanitization
                        if not retry:
                            # Clear last messages and re-add with sanitization
                            if original_system:
                                self.messages[-2]["content"] = self._sanitize_for_waf(
                                    original_system
                                )
                            if original_prompt:
                                self.messages[-1]["content"] = self._sanitize_for_waf(
                                    original_prompt
                                )

                    continue
                else:
                    # Not a 403 error or out of retries
                    if self._is_cloudfront_403(e):
                        logger.error(
                            f"CloudFront WAF: Failed after {self.waf_config.max_waf_retries + 1} attempts"
                        )
                    self.waf_metrics.failed_requests += 1

                    if self.waf_config.debug_mode:
                        logger.error(
                            f"Request failed: {type(e).__name__}: {str(e)[:200]}"
                        )

                    raise

        # This should never be reached, but just in case
        if last_error:
            raise last_error

    def prompt_stream(self, prompt: str, system: Optional[str] = None):
        """
        Generate streaming completion from prompt with CloudFront WAF protection.

        NOTE: Streaming mode does NOT support tool calls. If tools are enabled,
        this method will raise an error.

        Args:
            prompt: User prompt text
            system: Optional system message (overrides instance system)

        Yields:
            Response chunks as they arrive from the API

        Raises:
            ValueError: If tools are enabled (tools not compatible with streaming)

        Note:
            CloudFront WAF Protection: Content is automatically sanitized before
            sending to prevent 403 errors. On 403 errors, automatically retries
            with forced sanitization.

        Example:
            >>> client = LLM(stream=True)
            >>> for chunk in client.prompt_stream("Hello!"):
            ...     print(chunk, end="", flush=True)
        """
        # Check if tools are enabled
        if self.enable_tools and self.tool_manager.tools:
            raise ValueError(
                "Streaming mode does not support tool calls. "
                "Disable tools or use non-streaming prompt() method."
            )

        # Track metrics
        self.waf_metrics.total_requests += 1

        # Reset tool tracking
        self._current_tool_call_count = 0
        self._current_tool_metadata = []

        # Sanitize content if enabled
        if self.waf_config.enable_waf_sanitization:
            prompt = self._sanitize_for_waf(prompt)
            if system:
                system = self._sanitize_for_waf(system)

        # Store original for retry
        original_prompt = prompt
        original_system = system

        # Add messages
        if system:
            self.add_message("system", system)
        self.add_message("user", prompt)

        # Retry loop for CloudFront WAF 403 errors
        last_error = None
        for waf_attempt in range(self.waf_config.max_waf_retries + 1):
            try:
                if self.waf_config.debug_mode:
                    logger.debug(
                        f"Streaming request attempt {waf_attempt + 1}/{self.waf_config.max_waf_retries + 1}"
                    )

                # Get request params and validate messages
                params = self.get_request_params()
                validated_messages = self._validate_messages_for_api(self.messages)

                # Enable streaming
                params["stream"] = True

                # Make streaming request (no retry for streaming)
                stream = self.client.chat.completions.create(
                    **params, messages=validated_messages
                )

                full_response = ""

                # Yield chunks as they arrive
                for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if delta.content:
                            content = delta.content
                            full_response += content
                            yield content

                # Add complete response to conversation history
                if full_response:
                    self.add_message("assistant", full_response)

                # Track successful retry
                if waf_attempt > 0:
                    self.waf_metrics.retry_successes += 1
                    logger.info(
                        f"CloudFront WAF: Stream retry successful after {waf_attempt} attempts"
                    )

                break  # Success, exit retry loop

            except Exception as e:
                last_error = e

                # Check if it's a CloudFront 403 error
                if (
                    self._is_cloudfront_403(e)
                    and waf_attempt < self.waf_config.max_waf_retries
                ):
                    self.waf_metrics.cloudfront_403_errors += 1
                    wait_time = self.waf_config.retry_delay_base * (2**waf_attempt)

                    logger.warning(
                        f"CloudFront WAF 403 error during streaming (attempt {waf_attempt + 1}/"
                        f"{self.waf_config.max_waf_retries + 1}). "
                        f"Retrying in {wait_time}s with sanitization..."
                    )

                    # Wait before retry
                    time.sleep(wait_time)

                    # Force sanitization on retry
                    if (
                        not self.waf_config.enable_waf_sanitization
                        and self.waf_config.retry_with_sanitization
                    ):
                        # Re-sanitize messages
                        if original_system:
                            self.messages[-2]["content"] = self._sanitize_for_waf(
                                original_system
                            )
                        if original_prompt:
                            self.messages[-1]["content"] = self._sanitize_for_waf(
                                original_prompt
                            )

                    continue
                else:
                    # Not a 403 error or out of retries
                    if self._is_cloudfront_403(e):
                        logger.error(
                            f"CloudFront WAF: Streaming failed after {self.waf_config.max_waf_retries + 1} attempts"
                        )
                    self.waf_metrics.failed_requests += 1

                    if self.waf_config.debug_mode:
                        logger.error(
                            f"Streaming error: {type(e).__name__}: {str(e)[:200]}"
                        )

                    raise

        if not self.continuous:
            self.reset()

        # If we exited loop with error, raise it
        if last_error and waf_attempt >= self.waf_config.max_waf_retries:
            raise last_error

    def retry_prompt(self, prompt: Optional[str] = None) -> Optional[str]:
        """
        Retry the last prompt.

        Args:
            prompt: Optional new prompt

        Returns:
            Response or None if max retries exceeded
        """
        if self.retry_count >= self.max_retry:
            logger.warning(
                f"Maximum retry limit ({self.max_retry}) reached. Giving up on retry."
            )
            return None

        self.retry_count += 1
        logger.info(
            f"Retrying last prompt (attempt {self.retry_count}/{self.max_retry})."
        )
        return self.prompt(prompt, retry=True)

    # ========== Response Processing ==========

    def process_response(self, response, stream_final_response: bool = False) -> str:
        """
        Process LLM response, handling tool calls if present.

        Args:
            response: OpenAI API response
            stream_final_response: If True, stream the final response after tools complete

        Returns:
            Processed response text, structured output, or generator (if streaming)
        """
        choice = response.choices[0]

        # Handle tool calls
        if (
            hasattr(choice.message, "tool_calls")
            and choice.message.tool_calls is not None
        ):
            try:
                if (
                    hasattr(choice.message.tool_calls, "__iter__")
                    and len(choice.message.tool_calls) > 0
                ):
                    return self._handle_tool_calls(
                        choice.message,
                        response,
                        stream_final_response=stream_final_response,
                    )
            except (TypeError, AttributeError):
                pass  # Mock object, skip tool handling

        # Handle regular message
        if not choice.message.content:
            logger.warning("Received empty response from LLM")
            return ""

        message = choice.message.content.strip()

        # Remove thinking tags
        message = re.sub(r"<think>.*?</think>", "", message, flags=re.DOTALL)

        # Extract response tags if present
        if "<response>" in message and "</response>" in message:
            message = re.search(
                r"<response>(.*?)</response>", message, flags=re.DOTALL
            ).group(1)

        # Check max length
        if len(message) > self.max_length:
            logger.warning(f"Max length exceeded: {len(message)} > {self.max_length}")
            retry_result = self.retry_prompt()
            if retry_result is not None:
                return retry_result

        self.add_message("assistant", message)

        # Handle structured output
        if self.require_response_format:
            logger.info(f"Original message: {message}")
            message = lm_json_utils.extract_strict_json(message)
            logger.info(f"Parsed message: {message}")
            return self.response_format.parse_raw(message)

        return response.choices[0].message.content

    def _handle_tool_calls(
        self, message, original_response, recursion_depth=0, stream_final_response=False
    ):
        """
        Execute tool calls and get follow-up response.

        Args:
            message: Message with tool calls
            original_response: Original API response
            recursion_depth: Current recursion depth (for tracking)
            stream_final_response: If True, stream the final response after all tools complete

        Returns:
            Final response after tool execution (string or generator if streaming)
        """
        # Update recursion depth tracking
        self._current_recursion_depth = max(
            self._current_recursion_depth, recursion_depth
        )

        # Build assistant message with tool calls
        assistant_message = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ],
        }

        # Handle content
        try:
            raw_content = message.content if hasattr(message, "content") else None
        except Exception:
            raw_content = None

        if raw_content and str(raw_content).strip():
            assistant_message["content"] = str(raw_content).strip()
        else:
            assistant_message["content"] = "[Tool calls in progress]"

        self.messages.append(assistant_message)

        # Execute tool calls
        start_time = time.time()
        iteration_tool_count = len(message.tool_calls)

        for tool_call in message.tool_calls:
            tool_start_time = time.time()
            arguments = None  # Initialize to None for error handling

            try:
                # Parse and execute
                arguments = json.loads(tool_call.function.arguments)
                tc = ToolCall(
                    id=tool_call.id, name=tool_call.function.name, arguments=arguments
                )

                result = self.tool_manager.execute_tool(tc)

                # Process response with content processor
                execution_time = time.time() - tool_start_time
                processed = self.content_processor.process_tool_response(
                    result.content,
                    tool_call.function.name,
                    execution_time=execution_time,
                )

                # Track tool call in history
                self._current_tool_call_count += 1
                tool_entry = {
                    "index": self._current_tool_call_count,
                    "tool": tool_call.function.name,
                    "arguments": arguments,
                    "recursion_depth": recursion_depth,
                    "execution_time_ms": int(execution_time * 1000),
                    "result_length": len(str(result.content)),
                    "timestamp": time.time(),
                }

                # Add token info if available
                if "metadata" in processed:
                    meta = processed["metadata"]
                    tool_entry["tokens"] = meta.get("final_tokens", 0)
                    tool_entry["was_truncated"] = meta.get("was_truncated", False)

                self._current_tool_history.append(tool_entry)

                # Build tool message
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": processed["content"],
                }

                if "metadata" in processed:
                    tool_message["metadata"] = processed["metadata"]

                self.messages.append(tool_message)

                # Log execution
                if "metadata" in processed:
                    meta = processed["metadata"]
                    logger.info(
                        f"Executed tool {tool_call.function.name} "
                        f"(#{self._current_tool_call_count}, depth={recursion_depth}): "
                        f"tokens={meta['final_tokens']}, "
                        f"time={meta.get('execution_time_ms', 0)}ms, "
                        f"sanitized={meta['was_sanitized']}, "
                        f"truncated={meta['was_truncated']}"
                    )
                else:
                    logger.info(
                        f"Executed tool {tool_call.function.name} "
                        f"(#{self._current_tool_call_count}, depth={recursion_depth})"
                    )

            except Exception as e:
                logger.error(
                    f"Error processing tool call {tool_call.function.name}: {str(e)}"
                )

                # Still track failed tool calls (arguments may be None if parsing failed)
                self._current_tool_call_count += 1
                self._current_tool_history.append(
                    {
                        "index": self._current_tool_call_count,
                        "tool": tool_call.function.name,
                        "arguments": (
                            arguments
                            if arguments is not None
                            else tool_call.function.arguments
                        ),
                        "recursion_depth": recursion_depth,
                        "error": str(e),
                        "timestamp": time.time(),
                    }
                )

                processed = self.content_processor.process_tool_response(
                    f"Error: {str(e)}", tool_call.function.name
                )

                self.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": processed["content"],
                    }
                )

        # Log summary
        total_time = time.time() - start_time
        logger.info(
            f"Executed {iteration_tool_count} tool(s) in this iteration "
            f"(total: {self._current_tool_call_count}) in {total_time*1000:.0f}ms"
        )

        # Add delay before follow-up
        if self.tool_request_delay > 0:
            logger.info(f"Waiting {self.tool_request_delay}s before follow-up request")
            time.sleep(self.tool_request_delay)

        # Get follow-up response - with or without streaming
        params = self.get_request_params()
        validated_messages = self._validate_messages_for_api(self.messages)

        # Check if we should stream the final response
        if stream_final_response:
            # Enable streaming for the follow-up request
            params["stream"] = True

            # Make streaming request
            stream = self.client.chat.completions.create(
                **params, messages=validated_messages
            )

            # Check first chunk to see if there are more tool calls
            first_chunk = next(stream, None)
            if first_chunk and first_chunk.choices and len(first_chunk.choices) > 0:
                first_delta = first_chunk.choices[0].delta

                # Check for tool calls in the first chunk
                if hasattr(first_delta, "tool_calls") and first_delta.tool_calls:
                    # Has tool calls - need to handle them (can't stream with recursive tools)
                    # Fall back to non-streaming for recursive tool calls
                    logger.warning(
                        "Recursive tool calls detected, falling back to non-streaming"
                    )
                    params["stream"] = False
                    follow_up_response = self.retry_handler.execute_with_retry(
                        self.client.chat.completions.create,
                        **params,
                        messages=validated_messages,
                    )
                    return self._handle_tool_calls(
                        follow_up_response.choices[0].message,
                        follow_up_response,
                        recursion_depth=recursion_depth + 1,
                        stream_final_response=stream_final_response,
                    )

                # No tool calls - stream the response
                def generate_streamed_response():
                    """Generator for streaming the final response."""
                    full_response = ""

                    # Yield the first chunk we already got
                    if first_delta.content:
                        full_response += first_delta.content
                        yield first_delta.content

                    # Yield remaining chunks
                    for chunk in stream:
                        if chunk.choices and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta
                            if delta.content:
                                content = delta.content
                                full_response += content
                                yield content

                    # Add complete response to conversation history
                    if full_response:
                        self.add_message("assistant", full_response)

                return generate_streamed_response()
            else:
                # Empty response or no choices
                return ""
        else:
            # Non-streaming follow-up (original behavior)
            follow_up_response = self.retry_handler.execute_with_retry(
                self.client.chat.completions.create,
                **params,
                messages=validated_messages,
            )

            # Check if follow-up has more tool calls (recursive case)
            if (
                hasattr(follow_up_response.choices[0].message, "tool_calls")
                and follow_up_response.choices[0].message.tool_calls
            ):
                # Recursive tool calls - go deeper
                return self._handle_tool_calls(
                    follow_up_response.choices[0].message,
                    follow_up_response,
                    recursion_depth=recursion_depth + 1,
                    stream_final_response=stream_final_response,
                )
            else:
                # No more tool calls - return final response
                return self.process_response(follow_up_response)

    def _validate_messages_for_api(self, messages: List[Dict]) -> List[Dict]:
        """
        Validate and sanitize messages for API.

        Args:
            messages: List of conversation messages

        Returns:
            Validated messages
        """
        validated_messages = []

        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                logger.warning(
                    f"Skipping invalid message type at index {i}: {type(msg)}"
                )
                continue

            # Copy message
            validated_msg = dict(msg)

            # Validate role
            if "role" not in validated_msg or not validated_msg["role"]:
                logger.warning(f"Skipping message at index {i} without role")
                continue

            validated_msg["role"] = str(validated_msg["role"]).strip()
            if not validated_msg["role"]:
                logger.warning(f"Skipping message at index {i} with empty role")
                continue

            # Validate content
            if "content" not in validated_msg:
                validated_msg["content"] = "Message content unavailable"
            elif validated_msg["content"] is None:
                validated_msg["content"] = "Message content unavailable"

            # Convert to string
            try:
                validated_msg["content"] = str(validated_msg["content"])
            except Exception as e:
                logger.error(
                    f"Failed to convert content to string for message {i}: {e}"
                )
                validated_msg["content"] = "Message content conversion failed"

            # Apply role-specific fixes
            role = validated_msg["role"].lower()
            content = validated_msg["content"].strip()

            if not content:
                if role == "tool":
                    validated_msg["content"] = "Tool executed successfully"
                elif role == "assistant" and "tool_calls" in validated_msg:
                    validated_msg["content"] = " "
                else:
                    validated_msg["content"] = "Content unavailable"
            else:
                # Fix tool content if it's a placeholder
                if role == "tool" and content in [
                    "Message content unavailable",
                    "Content unavailable",
                ]:
                    validated_msg["content"] = "Tool executed successfully"

            # Ensure no empty content
            if validated_msg["content"] == "":
                validated_msg["content"] = (
                    " "
                    if (role == "assistant" and "tool_calls" in validated_msg)
                    else "."
                )

            validated_messages.append(validated_msg)

        return validated_messages

    # ========== Tool Management (Delegated) ==========

    def add_tool(self, tool: ToolFunction):
        """Add a tool function."""
        self.tool_manager.add_tool(tool)

    def remove_tool(self, name: str):
        """Remove a tool by name."""
        self.tool_manager.remove_tool(name)

    def add_tool_from_function(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[Dict] = None,
    ):
        """Add a tool from a Python function."""
        self.tool_manager.add_tool_from_function(func, name, description, parameters)

    def list_tools(self) -> List[str]:
        """Get list of registered tools."""
        return self.tool_manager.list_tools()

    def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call."""
        return self.tool_manager.execute_tool(tool_call)

    def enable_tool_support(self, enabled: bool = True):
        """Enable or disable tool support."""
        self.enable_tools = enabled
        logger.info(f"Tool support {'enabled' if enabled else 'disabled'}")

    def clear_tools(self):
        """Clear all registered tools."""
        self.tool_manager.clear_tools()

    # ========== Backward Compatibility ==========

    def get_last_tool_call_count(self) -> int:
        """
        Get the number of tool calls made in the last prompt() invocation.

        Returns:
            Total number of tool calls across all recursive invocations

        Example:
            >>> llm = LLM(enable_tools=True)
            >>> response = llm.prompt("Research this topic")
            >>> print(f"Used {llm.get_last_tool_call_count()} tool calls")
            Used 15 tool calls
        """
        return self._last_response_metadata.get("total_tool_calls", 0)

    def get_last_tool_history(self) -> List[Dict[str, Any]]:
        """
        Get detailed history of all tool calls from the last prompt() invocation.

        Returns:
            List of dictionaries containing tool call details:
            - index: Sequential tool call number
            - tool: Tool name
            - arguments: Arguments passed to tool
            - recursion_depth: How deep in the call chain
            - execution_time_ms: Execution time in milliseconds
            - result_length: Length of result
            - tokens: Token count (if available)
            - was_truncated: Whether result was truncated
            - error: Error message (if tool failed)

        Example:
            >>> history = llm.get_last_tool_history()
            >>> for call in history:
            ...     print(f"{call['index']}. {call['tool']} (depth={call['recursion_depth']})")
        """
        return self._last_response_metadata.get("tool_history", [])

    def get_last_recursion_depth(self) -> int:
        """
        Get the maximum recursion depth of tool calls in the last prompt() invocation.

        Returns:
            Maximum recursion depth (0 = no tool calls, 1+ = recursive calls)

        Example:
            >>> llm.prompt("Complex research task")
            >>> print(f"Tool chain went {llm.get_last_recursion_depth()} levels deep")
        """
        return self._last_response_metadata.get("recursion_depth", 0)

    def get_last_response_metadata(self) -> Dict[str, Any]:
        """
        Get complete metadata from the last prompt() invocation.

        Returns:
            Dictionary containing:
            - total_tool_calls: Total number of tool calls
            - recursion_depth: Maximum recursion depth
            - tool_history: Detailed history of all tool calls

        Example:
            >>> metadata = llm.get_last_response_metadata()
            >>> print(f"Metadata: {metadata}")
        """
        return self._last_response_metadata.copy()

    def _sanitize_tool_response(self, content: str) -> str:
        """Sanitize tool response (backward compatibility)."""
        return self.content_processor.sanitize_content(content)

    def _truncate_tool_response(self, content: str) -> tuple:
        """Truncate tool response (backward compatibility)."""
        return self.content_processor.truncate_content(content)

    def _process_tool_response(self, content: str, tool_name: str) -> dict:
        """Process tool response (backward compatibility)."""
        return self.content_processor.process_tool_response(content, tool_name)

    def _api_call_with_retry(self, api_func, *args, **kwargs):
        """Execute API call with retry (backward compatibility)."""
        return self.retry_handler.execute_with_retry(api_func, *args, **kwargs)

    def _sanitize_tool_content(
        self, content: str, tool_name: str, is_error: bool = False
    ) -> str:
        """Sanitize tool content (backward compatibility)."""
        return self.content_processor.sanitize_tool_content(
            content, tool_name, is_error
        )

    # Expose these properties for backward compatibility
    @property
    def sanitize_tool_responses(self) -> bool:
        """Check if tool response sanitization is enabled."""
        return self.content_processor.sanitize_enabled

    @property
    def max_tool_response_tokens(self) -> int:
        """Get maximum tool response tokens."""
        return self.content_processor.max_tokens

    @property
    def track_tool_metadata(self) -> bool:
        """Check if tool metadata tracking is enabled."""
        return self.content_processor.track_metadata

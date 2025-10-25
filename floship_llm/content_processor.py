"""Content processing for tool responses and messages."""

import logging
import re
from typing import Dict, Optional, Tuple

from .utils import estimate_tokens, truncate_to_tokens

logger = logging.getLogger(__name__)


class ContentProcessor:
    """Handles content sanitization and truncation for CloudFront WAF compatibility."""

    # Default patterns that trigger CloudFront WAF
    DEFAULT_SANITIZATION_PATTERNS = {
        "...": "[truncated]",  # Ellipsis triggers path traversal detection
    }

    def __init__(
        self,
        sanitize_enabled: bool = True,
        sanitization_patterns: Optional[Dict[str, str]] = None,
        max_tokens: int = 4000,
        track_metadata: bool = False,
    ):
        """
        Initialize content processor.

        Args:
            sanitize_enabled: Whether to sanitize content
            sanitization_patterns: Custom patterns to replace (pattern -> replacement)
            max_tokens: Maximum tokens allowed in tool responses
            track_metadata: Whether to track processing metadata
        """
        self.sanitize_enabled = sanitize_enabled
        self.sanitization_patterns = (
            sanitization_patterns or self.DEFAULT_SANITIZATION_PATTERNS
        )
        self.max_tokens = max_tokens
        self.track_metadata = track_metadata

    def sanitize_content(self, content: str) -> str:
        """
        Sanitize content by replacing problematic patterns.

        Args:
            content: The content to sanitize

        Returns:
            Sanitized content
        """
        if not self.sanitize_enabled or not content:
            return content

        sanitized = content
        for pattern, replacement in self.sanitization_patterns.items():
            sanitized = sanitized.replace(pattern, replacement)

        return sanitized

    def truncate_content(self, content: str) -> Tuple[str, bool]:
        """
        Truncate content to maximum token limit.

        Args:
            content: The content to truncate

        Returns:
            Tuple of (truncated_content, was_truncated)
        """
        if not content:
            return content, False

        token_count = estimate_tokens(content)

        if token_count <= self.max_tokens:
            return content, False

        truncated = truncate_to_tokens(content, self.max_tokens)
        logger.debug(
            f"Content truncated from ~{token_count} tokens to ~{self.max_tokens} tokens"
        )

        return truncated, True

    def process_tool_response(
        self, content: str, tool_name: str, execution_time: Optional[float] = None
    ) -> Dict:
        """
        Process tool response with sanitization, truncation, and metadata tracking.

        Args:
            content: The tool response content
            tool_name: Name of the tool that generated the response
            execution_time: Tool execution time in seconds

        Returns:
            Dictionary with processed content and optional metadata
        """
        import time

        start_time = time.time()

        # Sanitize content
        sanitized = self.sanitize_content(content)
        was_sanitized = sanitized != content

        # Truncate content
        truncated, was_truncated = self.truncate_content(sanitized)

        # Build result
        result = {"content": truncated}

        # Add metadata if tracking is enabled
        if self.track_metadata:
            original_tokens = estimate_tokens(content)
            final_tokens = estimate_tokens(truncated)
            processing_time = time.time() - start_time

            result["metadata"] = {
                "tool_name": tool_name,
                "was_sanitized": was_sanitized,
                "was_truncated": was_truncated,
                "original_tokens": original_tokens,
                "final_tokens": final_tokens,
                "processing_time_ms": round(processing_time * 1000, 2),
                # Backward compatibility
                "sanitization_applied": was_sanitized,
                "truncation_applied": was_truncated,
            }

            if execution_time is not None:
                result["metadata"]["execution_time_ms"] = round(
                    execution_time * 1000, 2
                )

            if was_sanitized:
                logger.debug(f"Tool '{tool_name}' response sanitized")
            if was_truncated:
                logger.debug(
                    f"Tool '{tool_name}' response truncated: "
                    f"{original_tokens} -> {final_tokens} tokens"
                )

        return result

    def sanitize_tool_content(
        self, content: str, tool_name: str, is_error: bool = False
    ) -> str:
        """
        Sanitize tool content for API messages (backward compatibility).

        Args:
            content: The content to sanitize
            tool_name: Name of the tool
            is_error: Whether this is an error message

        Returns:
            Sanitized content string
        """
        if content is None:
            default_msg = (
                f"Error executing tool: {tool_name}"
                if is_error
                else f"Tool {tool_name} executed successfully (no return value)"
            )
            logger.warning(
                f"Tool {tool_name} content is None, using default: {default_msg}"
            )
            return default_msg

        # Convert to string and strip
        content_str = str(content).strip()

        if not content_str:
            default_msg = (
                f"Error executing tool: {tool_name} (empty result)"
                if is_error
                else f"Tool {tool_name} executed successfully (empty result)"
            )
            logger.warning(
                f"Tool {tool_name} returned empty content, using default: {default_msg}"
            )
            return default_msg

        # Check for null-like strings
        if content_str.lower() in ["none", "null", "undefined"]:
            default_msg = (
                f"Error executing tool: {tool_name} (null result)"
                if is_error
                else f"Tool {tool_name} executed successfully (null result)"
            )
            logger.warning(
                f"Tool {tool_name} returned null-like content '{content_str}', using default: {default_msg}"
            )
            return default_msg

        # Apply sanitization
        sanitized = self.sanitize_content(content_str)

        # Truncate if needed
        truncated, was_truncated = self.truncate_content(sanitized)

        if was_truncated:
            logger.debug(f"Tool '{tool_name}' content truncated for API message")

        return truncated

    def sanitize_messages(self, content: str) -> str:
        """
        Sanitize message content (compacts whitespace).

        Args:
            content: The message content

        Returns:
            Sanitized content
        """
        if not content:
            return content

        # Compact multiple spaces into one
        content = re.sub(r"\s+", " ", content)
        return content.strip()

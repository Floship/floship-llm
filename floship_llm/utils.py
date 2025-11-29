try:
    import regex as re
except ImportError:
    import re

import json
from typing import Any, Dict, List


class JSONUtils:
    JSON_CANDIDATE = re.compile(
        r"""
        (?P<json>
          \{(?:[^{}]|(?&json))*\}    # nested braces
          |
          \[(?:[^\[\]]|(?&json))*\]  # nested brackets
        )
        """,
        re.VERBOSE,
    )

    CLEANUPS = [
        (re.compile(r"(?P<pre>[:\[,]\s*)'(?P<body>[^']*)'"), r'\g<pre>"\g<body>"'),
        (
            re.compile(r"(?P<brace>[\{\s,])(?P<key>[A-Za-z0-9_]+)\s*:"),
            r'\g<brace>"\g<key>":',
        ),
        (re.compile(r",\s*(?P<brace>[}\]])"), r"\g<brace>"),
        (re.compile(r"\\'"), "'"),
        (re.compile(r"[\x00-\x1f]"), ""),
    ]

    def _normalize(self, candidate: str) -> str:
        """Apply all cleanup regexes in sequence."""
        s = candidate
        for pattern, repl in self.CLEANUPS:
            s = pattern.sub(repl, s)
        return s

    def extract_and_fix_json(self, text: str) -> List[Any]:
        """
        Find JSON-like fragments in `text`, attempt to clean and parse them,
        return a list of Python objects. Invalid fragments are skipped.
        """
        results = []
        for m in self.JSON_CANDIDATE.finditer(text):
            raw = m.group("json")
            candidates = [raw, self._normalize(raw)]

            seen = set()
            for candidate in candidates:
                if candidate in seen:
                    continue
                seen.add(candidate)
                try:
                    obj = json.loads(candidate)
                    results.append(obj)
                    break
                except json.JSONDecodeError:
                    continue
        return results

    def strict_json(self, obj: Any, **dump_kwargs: Any) -> str:
        """
        Re-serialize Python object to a canonical JSON string.
        By default, uses sort_keys=True and indent=2.
        """
        params: Dict[str, Any] = {"ensure_ascii": False, "sort_keys": True, "indent": 2}
        params.update(dump_kwargs)
        return json.dumps(obj, **params)

    def _is_likely_truncated(self, text: str) -> bool:
        """
        Check if text appears to be truncated JSON.

        Truncated JSON typically:
        - Has unbalanced braces/brackets
        - Ends mid-string or mid-value
        - Has more opening than closing delimiters
        """
        # Count braces and brackets
        open_braces = text.count("{")
        close_braces = text.count("}")
        open_brackets = text.count("[")
        close_brackets = text.count("]")

        # If unbalanced, likely truncated
        if open_braces != close_braces or open_brackets != close_brackets:
            return True

        # Check for common truncation patterns
        stripped = text.rstrip()
        if stripped:
            # Ends with incomplete patterns
            truncation_indicators = [
                # Mid-string
                stripped.endswith('"') and stripped.count('"') % 2 == 1,
                # Mid-value (ends with comma, colon, or opening delimiter)
                stripped[-1] in ",:[{",
                # Ends with incomplete escape sequence
                stripped.endswith("\\"),
            ]
            if any(truncation_indicators):
                return True

        return False

    def is_truncated_json(self, text: str) -> bool:
        """
        Public method to check if text appears to be truncated JSON.

        Useful for detecting when max_completion_tokens was too low
        and the response was cut off mid-JSON.

        Args:
            text: The text to check (may include markdown code blocks)

        Returns:
            True if the text appears to contain truncated JSON
        """
        # Check markdown code blocks first
        import re

        code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
        code_blocks = re.findall(code_block_pattern, text)

        for block in code_blocks:
            if self._is_likely_truncated(block.strip()):
                return True

        # Also check the raw text for JSON-like content
        if self._is_likely_truncated(text):
            return True

        return False

    def extract_strict_json(self, text: str) -> str:
        """
        Extract the first valid JSON object from the text.
        Returns a JSON string or an empty string if no valid JSON is found.

        Handles:
        - Raw JSON objects/arrays
        - JSON wrapped in markdown code blocks (```json ... ```)
        - JSON with leading/trailing text
        - Truncated JSON (returns empty rather than nested object)
        """
        # First, try to extract from markdown code blocks
        import re

        code_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
        code_blocks = re.findall(code_block_pattern, text)

        # Try code blocks first
        for block in code_blocks:
            block_text = block.strip()

            # Check if the block content appears truncated
            if self._is_likely_truncated(block_text):
                # Don't extract from truncated JSON - return empty
                return ""

            results = self.extract_and_fix_json(block_text)
            if results:
                return self.strict_json(results[0])

        # Fall back to searching the entire text
        # Check for truncation first
        if self._is_likely_truncated(text):
            return ""

        results = self.extract_and_fix_json(text)
        if results:
            return self.strict_json(results[0])
        return ""


lm_json_utils = JSONUtils()


def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in a text string.
    Uses a simple approximation: ~4 characters per token.

    Args:
        text: The text to estimate tokens for

    Returns:
        Estimated token count
    """
    if not text:
        return 0
    # Simple approximation: ~4 characters per token
    # This is conservative for English text
    return len(text) // 4


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """
    Truncate text to approximately fit within max_tokens.

    Args:
        text: The text to truncate
        max_tokens: Maximum number of tokens

    Returns:
        Truncated text with ellipsis if truncation occurred
    """
    if not text:
        return text

    estimated_tokens = estimate_tokens(text)
    if estimated_tokens <= max_tokens:
        return text

    # Calculate approximate character limit
    max_chars = max_tokens * 4

    if len(text) <= max_chars:
        return text

    # Truncate and add indicator
    return text[:max_chars] + "...[truncated]"

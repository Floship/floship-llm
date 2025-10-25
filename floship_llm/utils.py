try:
    import regex as re
except ImportError:
    import re

import json
from typing import Any, List


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

    def strict_json(self, obj: Any, **dump_kwargs) -> str:
        """
        Re-serialize Python object to a canonical JSON string.
        By default, uses sort_keys=True and indent=2.
        """
        params = {"ensure_ascii": False, "sort_keys": True, "indent": 2}
        params.update(dump_kwargs)
        return json.dumps(obj, **params)

    def extract_strict_json(self, text: str) -> str:
        """
        Extract the first valid JSON object from the text.
        Returns a JSON string or an empty string if no valid JSON is found.
        """
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

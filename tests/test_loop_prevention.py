"""Tests for loop prevention in LLM client."""

from typing import Any
from unittest.mock import MagicMock

from floship_llm.client import LLM


def create_mock_tool_call(name: str, arguments: str) -> MagicMock:
    """Create a mock tool call object."""
    tc = MagicMock()
    tc.function.name = name
    tc.function.arguments = arguments
    return tc


class TestDetectToolLoop:
    """Test the _detect_tool_loop method directly."""

    def test_detect_loop_no_history(self) -> None:
        """Test no loop detected with empty history."""
        llm = LLM.__new__(LLM)
        llm._current_tool_history: list[dict[str, Any]] = []

        tool_calls = [create_mock_tool_call("test_tool", '{"arg": "value"}')]
        result = llm._detect_tool_loop(tool_calls)
        assert result is False

    def test_detect_loop_insufficient_repeats(self) -> None:
        """Test no loop detected with fewer than 3 repeats."""
        llm = LLM.__new__(LLM)
        llm._current_tool_history = [
            {"tool": "test_tool", "arguments": {"arg": "value"}},
            {"tool": "test_tool", "arguments": {"arg": "value"}},
        ]

        tool_calls = [create_mock_tool_call("test_tool", '{"arg": "value"}')]
        result = llm._detect_tool_loop(tool_calls)
        assert result is False

    def test_detect_loop_three_repeats(self) -> None:
        """Test loop detected with exactly 3 repeats."""
        llm = LLM.__new__(LLM)
        llm._current_tool_history = [
            {"tool": "test_tool", "arguments": {"arg": "value"}},
            {"tool": "test_tool", "arguments": {"arg": "value"}},
            {"tool": "test_tool", "arguments": {"arg": "value"}},
        ]

        tool_calls = [create_mock_tool_call("test_tool", '{"arg": "value"}')]
        result = llm._detect_tool_loop(tool_calls)
        assert result is True

    def test_detect_loop_many_repeats(self) -> None:
        """Test loop detected with more than 3 repeats."""
        llm = LLM.__new__(LLM)
        llm._current_tool_history = [
            {"tool": "test_tool", "arguments": {"arg": "value"}},
            {"tool": "test_tool", "arguments": {"arg": "value"}},
            {"tool": "test_tool", "arguments": {"arg": "value"}},
            {"tool": "test_tool", "arguments": {"arg": "value"}},
            {"tool": "test_tool", "arguments": {"arg": "value"}},
        ]

        tool_calls = [create_mock_tool_call("test_tool", '{"arg": "value"}')]
        result = llm._detect_tool_loop(tool_calls)
        assert result is True

    def test_detect_loop_different_args(self) -> None:
        """Test no loop detected when args differ."""
        llm = LLM.__new__(LLM)
        llm._current_tool_history = [
            {"tool": "test_tool", "arguments": {"arg": "value1"}},
            {"tool": "test_tool", "arguments": {"arg": "value2"}},
            {"tool": "test_tool", "arguments": {"arg": "value3"}},
        ]

        tool_calls = [create_mock_tool_call("test_tool", '{"arg": "value4"}')]
        result = llm._detect_tool_loop(tool_calls)
        assert result is False

    def test_detect_loop_different_tools(self) -> None:
        """Test no loop detected when tools differ."""
        llm = LLM.__new__(LLM)
        llm._current_tool_history = [
            {"tool": "tool_a", "arguments": {"arg": "value"}},
            {"tool": "tool_b", "arguments": {"arg": "value"}},
            {"tool": "tool_c", "arguments": {"arg": "value"}},
        ]

        tool_calls = [create_mock_tool_call("tool_d", '{"arg": "value"}')]
        result = llm._detect_tool_loop(tool_calls)
        assert result is False

    def test_detect_loop_mixed_history(self) -> None:
        """Test loop detected when repeats are interspersed with other calls."""
        llm = LLM.__new__(LLM)
        llm._current_tool_history = [
            {"tool": "test_tool", "arguments": {"arg": "value"}},
            {"tool": "other_tool", "arguments": {"other": "data"}},
            {"tool": "test_tool", "arguments": {"arg": "value"}},
            {"tool": "another_tool", "arguments": {}},
            {"tool": "test_tool", "arguments": {"arg": "value"}},
        ]

        tool_calls = [create_mock_tool_call("test_tool", '{"arg": "value"}')]
        result = llm._detect_tool_loop(tool_calls)
        assert result is True

    def test_detect_loop_respects_history_limit(self) -> None:
        """Test that only last 10 entries are checked."""
        llm = LLM.__new__(LLM)
        # Create history with old repeating entries (more than 10 items ago)
        llm._current_tool_history = [
            {"tool": "test_tool", "arguments": {"arg": "value"}},
            {"tool": "test_tool", "arguments": {"arg": "value"}},
            {"tool": "test_tool", "arguments": {"arg": "value"}},
            # Fill with different entries
            {"tool": "other_tool", "arguments": {}},
            {"tool": "other_tool", "arguments": {}},
            {"tool": "other_tool", "arguments": {}},
            {"tool": "other_tool", "arguments": {}},
            {"tool": "other_tool", "arguments": {}},
            {"tool": "other_tool", "arguments": {}},
            {"tool": "other_tool", "arguments": {}},
            {"tool": "other_tool", "arguments": {}},
            {"tool": "other_tool", "arguments": {}},
            {"tool": "other_tool", "arguments": {}},
        ]

        # The first 3 repeats are now outside the 10-entry window
        tool_calls = [create_mock_tool_call("test_tool", '{"arg": "value"}')]
        result = llm._detect_tool_loop(tool_calls)
        assert result is False

    def test_detect_loop_empty_args(self) -> None:
        """Test loop detection with empty arguments."""
        llm = LLM.__new__(LLM)
        llm._current_tool_history = [
            {"tool": "test_tool", "arguments": {}},
            {"tool": "test_tool", "arguments": {}},
            {"tool": "test_tool", "arguments": {}},
        ]

        tool_calls = [create_mock_tool_call("test_tool", "{}")]
        result = llm._detect_tool_loop(tool_calls)
        assert result is True

    def test_detect_loop_multiple_tool_calls(self) -> None:
        """Test loop detection with multiple tool calls in one request."""
        llm = LLM.__new__(LLM)
        llm._current_tool_history = [
            {"tool": "tool_a", "arguments": {"x": 1}},
            {"tool": "tool_a", "arguments": {"x": 1}},
            {"tool": "tool_a", "arguments": {"x": 1}},
        ]

        # One call is looping, one is not
        tool_calls = [
            create_mock_tool_call("tool_a", '{"x": 1}'),  # looping
            create_mock_tool_call("tool_b", '{"y": 2}'),  # not looping
        ]
        result = llm._detect_tool_loop(tool_calls)
        assert result is True  # Should detect the looping tool_a

    def test_detect_loop_with_invalid_json_args(self) -> None:
        """Test loop detection handles invalid JSON gracefully."""
        llm = LLM.__new__(LLM)
        llm._current_tool_history = [
            {"tool": "test_tool", "arguments": "invalid json"},
            {"tool": "test_tool", "arguments": "invalid json"},
            {"tool": "test_tool", "arguments": "invalid json"},
        ]

        tool_calls = [create_mock_tool_call("test_tool", "invalid json")]
        result = llm._detect_tool_loop(tool_calls)
        assert result is True

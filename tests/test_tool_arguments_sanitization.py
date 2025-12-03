"""
Tests for tool call arguments sanitization to prevent 500 "Failed to execute chat" errors.

The Heroku/Claude API returns 500 errors (not 400) when tool_calls[].function.arguments
contains invalid JSON. This test suite verifies that _sanitize_tool_calls() properly
validates and repairs arguments to prevent these server errors.

Root cause discovered: The API returns 500 "Failed to execute chat" for:
- Empty strings
- Plain text (not JSON)
- Partial/malformed JSON
- Arrays or primitive values (numbers, booleans)

The API accepts:
- Valid JSON objects '{"key": "value"}'
- Empty objects '{}'
- null
"""

import json
import logging
from unittest.mock import MagicMock, patch

import pytest

from floship_llm import LLM


class TestToolArgumentsSanitization:
    """Test suite for tool call arguments sanitization."""

    @pytest.fixture
    def llm(self):
        """Create an LLM instance for testing."""
        with patch.dict(
            "os.environ",
            {
                "INFERENCE_URL": "https://api.example.com/v1",
                "INFERENCE_MODEL_ID": "test-model",
                "INFERENCE_KEY": "test-key",
            },
        ):
            return LLM()

    def test_valid_json_object_unchanged(self, llm):
        """Valid JSON objects should pass through unchanged."""
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "arguments": '{"location": "Tokyo", "unit": "celsius"}',
                },
            }
        ]

        result = llm._sanitize_tool_calls(tool_calls, msg_index=0)

        assert len(result) == 1
        # Should be valid JSON
        parsed = json.loads(result[0]["function"]["arguments"])
        assert parsed == {"location": "Tokyo", "unit": "celsius"}

    def test_empty_object_unchanged(self, llm):
        """Empty JSON objects should pass through unchanged."""
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "test_tool", "arguments": "{}"},
            }
        ]

        result = llm._sanitize_tool_calls(tool_calls, msg_index=0)

        assert result[0]["function"]["arguments"] == "{}"

    def test_null_unchanged(self, llm):
        """JSON null should pass through (API accepts it)."""
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "test_tool", "arguments": "null"},
            }
        ]

        result = llm._sanitize_tool_calls(tool_calls, msg_index=0)

        assert result[0]["function"]["arguments"] == "null"

    def test_empty_string_replaced_with_empty_object(self, llm):
        """Empty string arguments should be replaced with empty object."""
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "test_tool", "arguments": ""},
            }
        ]

        result = llm._sanitize_tool_calls(tool_calls, msg_index=0)

        assert result[0]["function"]["arguments"] == "{}"

    def test_plain_text_replaced_with_empty_object(self, llm):
        """Plain text (not JSON) should be replaced with empty object."""
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "test_tool", "arguments": "not valid json"},
            }
        ]

        result = llm._sanitize_tool_calls(tool_calls, msg_index=0)

        assert result[0]["function"]["arguments"] == "{}"

    def test_partial_json_replaced_with_empty_object(self, llm):
        """Partial/malformed JSON should be replaced with empty object."""
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "test_tool", "arguments": '{"location":'},
            }
        ]

        result = llm._sanitize_tool_calls(tool_calls, msg_index=0)

        assert result[0]["function"]["arguments"] == "{}"

    def test_single_quotes_replaced_with_empty_object(self, llm):
        """Single-quoted JSON (invalid) should be replaced with empty object."""
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "test_tool", "arguments": "{'location': 'Tokyo'}"},
            }
        ]

        result = llm._sanitize_tool_calls(tool_calls, msg_index=0)

        assert result[0]["function"]["arguments"] == "{}"

    def test_trailing_comma_fixed(self, llm):
        """Trailing comma JSON should be replaced (invalid JSON)."""
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "arguments": '{"location": "Tokyo",}',
                },
            }
        ]

        result = llm._sanitize_tool_calls(tool_calls, msg_index=0)

        # Invalid JSON, should be replaced with empty object
        assert result[0]["function"]["arguments"] == "{}"

    def test_array_wrapped_in_object(self, llm):
        """JSON arrays should be wrapped in an object."""
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "test_tool", "arguments": '["Tokyo", "London"]'},
            }
        ]

        result = llm._sanitize_tool_calls(tool_calls, msg_index=0)

        parsed = json.loads(result[0]["function"]["arguments"])
        assert parsed == {"value": ["Tokyo", "London"]}

    def test_number_wrapped_in_object(self, llm):
        """JSON numbers should be wrapped in an object."""
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "test_tool", "arguments": "123"},
            }
        ]

        result = llm._sanitize_tool_calls(tool_calls, msg_index=0)

        parsed = json.loads(result[0]["function"]["arguments"])
        assert parsed == {"value": 123}

    def test_boolean_wrapped_in_object(self, llm):
        """JSON booleans should be wrapped in an object."""
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "test_tool", "arguments": "true"},
            }
        ]

        result = llm._sanitize_tool_calls(tool_calls, msg_index=0)

        parsed = json.loads(result[0]["function"]["arguments"])
        assert parsed == {"value": True}

    def test_string_primitive_wrapped_in_object(self, llm):
        """JSON string primitives should be wrapped in an object."""
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "test_tool", "arguments": '"Tokyo"'},
            }
        ]

        result = llm._sanitize_tool_calls(tool_calls, msg_index=0)

        parsed = json.loads(result[0]["function"]["arguments"])
        assert parsed == {"value": "Tokyo"}

    def test_none_arguments_replaced_with_empty_object(self, llm):
        """None arguments should be replaced with empty object."""
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "test_tool", "arguments": None},
            }
        ]

        result = llm._sanitize_tool_calls(tool_calls, msg_index=0)

        assert result[0]["function"]["arguments"] == "{}"

    def test_missing_arguments_replaced_with_empty_object(self, llm):
        """Missing arguments field should be replaced with empty object."""
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "test_tool"},
            }
        ]

        result = llm._sanitize_tool_calls(tool_calls, msg_index=0)

        assert result[0]["function"]["arguments"] == "{}"

    def test_dict_arguments_serialized(self, llm):
        """Dict arguments (already parsed) should be serialized to JSON string."""
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "arguments": {"location": "Tokyo"},
                },
            }
        ]

        result = llm._sanitize_tool_calls(tool_calls, msg_index=0)

        parsed = json.loads(result[0]["function"]["arguments"])
        assert parsed == {"location": "Tokyo"}

    def test_unicode_preserved(self, llm):
        """Unicode characters in arguments should be preserved."""
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "test_tool", "arguments": '{"location": "東京"}'},
            }
        ]

        result = llm._sanitize_tool_calls(tool_calls, msg_index=0)

        parsed = json.loads(result[0]["function"]["arguments"])
        assert parsed == {"location": "東京"}

    def test_escaped_characters_preserved(self, llm):
        """Escaped characters should be preserved."""
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "arguments": '{"text": "line1\\nline2"}',
                },
            }
        ]

        result = llm._sanitize_tool_calls(tool_calls, msg_index=0)

        parsed = json.loads(result[0]["function"]["arguments"])
        assert parsed == {"text": "line1\nline2"}

    def test_nested_objects_preserved(self, llm):
        """Nested JSON objects should be preserved."""
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "arguments": '{"location": {"city": "Tokyo", "country": "Japan"}}',
                },
            }
        ]

        result = llm._sanitize_tool_calls(tool_calls, msg_index=0)

        parsed = json.loads(result[0]["function"]["arguments"])
        assert parsed == {"location": {"city": "Tokyo", "country": "Japan"}}

    def test_multiple_tool_calls_all_sanitized(self, llm):
        """All tool calls in a list should be sanitized."""
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "tool1", "arguments": '{"valid": true}'},
            },
            {
                "id": "call_2",
                "type": "function",
                "function": {"name": "tool2", "arguments": "invalid json"},
            },
            {
                "id": "call_3",
                "type": "function",
                "function": {"name": "tool3", "arguments": '["array"]'},
            },
        ]

        result = llm._sanitize_tool_calls(tool_calls, msg_index=0)

        assert len(result) == 3
        # First should be valid
        assert json.loads(result[0]["function"]["arguments"]) == {"valid": True}
        # Second should be empty object (invalid JSON)
        assert result[1]["function"]["arguments"] == "{}"
        # Third should be wrapped array
        assert json.loads(result[2]["function"]["arguments"]) == {"value": ["array"]}

    def test_logging_on_invalid_json(self, llm, caplog):
        """Should log warning when replacing invalid JSON."""
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "test_tool", "arguments": "not valid"},
            }
        ]

        with caplog.at_level(logging.WARNING):
            llm._sanitize_tool_calls(tool_calls, msg_index=5)

        assert "Invalid JSON" in caplog.text
        assert "message 5" in caplog.text

    def test_logging_on_non_object_json(self, llm, caplog):
        """Should log warning when wrapping non-object JSON."""
        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "test_tool", "arguments": "[1, 2, 3]"},
            }
        ]

        with caplog.at_level(logging.WARNING):
            llm._sanitize_tool_calls(tool_calls, msg_index=3)

        assert "not a JSON object" in caplog.text
        assert "message 3" in caplog.text

    def test_waf_sanitization_applied_after_json_validation(self, llm):
        """WAF sanitization should be applied after JSON validation."""
        # Enable WAF sanitization
        llm.waf_config.enable_waf_sanitization = True

        tool_calls = [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "arguments": '{"path": "../etc/passwd"}',
                },
            }
        ]

        result = llm._sanitize_tool_calls(tool_calls, msg_index=0)

        # Should still be valid JSON after WAF sanitization
        parsed = json.loads(result[0]["function"]["arguments"])
        assert "path" in parsed

    def test_non_function_tool_call_unchanged(self, llm):
        """Non-function type tool calls should pass through unchanged."""
        tool_calls = [
            {
                "id": "call_1",
                "type": "other_type",
                "data": "some data",
            }
        ]

        result = llm._sanitize_tool_calls(tool_calls, msg_index=0)

        assert result[0] == tool_calls[0]

    def test_invalid_tool_call_skipped(self, llm, caplog):
        """Invalid tool calls (non-dict) should be skipped with warning."""
        tool_calls = [
            "not a dict",
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "test", "arguments": "{}"},
            },
        ]

        with caplog.at_level(logging.WARNING):
            result = llm._sanitize_tool_calls(tool_calls, msg_index=0)

        assert len(result) == 1  # Only the valid one
        assert "Skipping invalid tool_call" in caplog.text


class TestValidateMessagesForApiArguments:
    """Test that _validate_messages_for_api properly sanitizes tool call arguments."""

    @pytest.fixture
    def llm(self):
        """Create an LLM instance for testing."""
        with patch.dict(
            "os.environ",
            {
                "INFERENCE_URL": "https://api.example.com/v1",
                "INFERENCE_MODEL_ID": "test-model",
                "INFERENCE_KEY": "test-key",
            },
        ):
            return LLM()

    def test_validate_messages_fixes_invalid_arguments(self, llm):
        """_validate_messages_for_api should fix invalid JSON arguments."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": ".",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "test_tool",
                            "arguments": "invalid json",
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ]

        validated = llm._validate_messages_for_api(messages)

        # Find the assistant message with tool_calls
        assistant_msg = next(m for m in validated if m.get("tool_calls"))
        args = assistant_msg["tool_calls"][0]["function"]["arguments"]

        # Should be valid JSON now
        assert args == "{}"

    def test_validate_messages_preserves_valid_arguments(self, llm):
        """_validate_messages_for_api should preserve valid JSON arguments."""
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": ".",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "test_tool",
                            "arguments": '{"key": "value"}',
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ]

        validated = llm._validate_messages_for_api(messages)

        assistant_msg = next(m for m in validated if m.get("tool_calls"))
        args = assistant_msg["tool_calls"][0]["function"]["arguments"]

        parsed = json.loads(args)
        assert parsed == {"key": "value"}


class TestSanitizeMessagesArguments:
    """Test that sanitize_messages() properly handles tool call arguments."""

    @pytest.fixture
    def llm(self):
        """Create an LLM instance for testing."""
        with patch.dict(
            "os.environ",
            {
                "INFERENCE_URL": "https://api.example.com/v1",
                "INFERENCE_MODEL_ID": "test-model",
                "INFERENCE_KEY": "test-key",
            },
        ):
            return LLM()

    def test_sanitize_messages_with_tool_calls(self, llm):
        """sanitize_messages() should handle messages with tool_calls."""
        llm.messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "",  # Empty content - should be fixed
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "test_tool", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
        ]

        llm.sanitize_messages()

        # Assistant message should have placeholder content
        assert llm.messages[2]["content"] == "."


class TestIntegrationWithHandleToolCalls:
    """Integration tests for the full tool call flow."""

    @pytest.fixture
    def llm(self):
        """Create an LLM instance for testing."""
        with patch.dict(
            "os.environ",
            {
                "INFERENCE_URL": "https://api.example.com/v1",
                "INFERENCE_MODEL_ID": "test-model",
                "INFERENCE_KEY": "test-key",
            },
        ):
            llm = LLM(enable_tools=True)
            return llm

    def test_handle_tool_calls_builds_valid_assistant_message(self, llm):
        """_handle_tool_calls should build assistant messages with valid arguments."""
        # Create a mock message with tool calls
        mock_message = MagicMock()
        mock_message.content = None
        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_1"
        mock_tool_call.function.name = "test_tool"
        mock_tool_call.function.arguments = '{"param": "value"}'
        mock_message.tool_calls = [mock_tool_call]

        mock_response = MagicMock()

        # Add a test tool
        from floship_llm import ToolFunction, ToolParameter

        test_tool = ToolFunction(
            name="test_tool",
            description="Test tool",
            parameters=[
                ToolParameter(name="param", type="string", description="A param")
            ],
            function=lambda param: f"Result: {param}",
        )
        llm.add_tool(test_tool)

        # Mock the follow-up API call
        with patch.object(llm.client.chat.completions, "create") as mock_create:
            mock_follow_up = MagicMock()
            mock_follow_up.choices = [MagicMock()]
            mock_follow_up.choices[0].message.content = "Final response"
            mock_follow_up.choices[0].message.tool_calls = None
            mock_create.return_value = mock_follow_up

            # Call _handle_tool_calls
            llm._handle_tool_calls(mock_message, mock_response)

        # Check that the assistant message was added with placeholder content
        assistant_msg = next(
            (
                m
                for m in llm.messages
                if m.get("role") == "assistant" and m.get("tool_calls")
            ),
            None,
        )
        assert assistant_msg is not None
        assert assistant_msg["content"] == "."  # Placeholder content
        assert (
            assistant_msg["tool_calls"][0]["function"]["arguments"]
            == '{"param": "value"}'
        )

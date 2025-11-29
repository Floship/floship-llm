"""Tests for tool request delay functionality."""

import os
from unittest.mock import Mock, patch

from floship_llm import LLM
from floship_llm.schemas import ToolFunction


class TestToolRequestDelay:
    """Tests for the tool request delay feature."""

    def setup_method(self):
        """Set up test environment."""
        os.environ["INFERENCE_URL"] = "http://test.com"
        os.environ["INFERENCE_MODEL_ID"] = "test-model"
        os.environ["INFERENCE_KEY"] = "test-key"

    def teardown_method(self):
        """Clean up test environment."""
        if "LLM_TOOL_REQUEST_DELAY" in os.environ:
            del os.environ["LLM_TOOL_REQUEST_DELAY"]

    def test_default_tool_request_delay(self):
        """Test that default tool request delay is 0 seconds."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(enable_tools=True)
            assert llm.tool_request_delay == 0.0

    def test_custom_tool_request_delay_from_env(self):
        """Test that tool request delay can be set via environment variable."""
        os.environ["LLM_TOOL_REQUEST_DELAY"] = "3"
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(enable_tools=True)
            assert llm.tool_request_delay == 3.0

    def test_tool_request_delay_with_decimal(self):
        """Test that tool request delay accepts decimal values."""
        os.environ["LLM_TOOL_REQUEST_DELAY"] = "2.5"
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(enable_tools=True)
            assert llm.tool_request_delay == 2.5

    def test_zero_delay(self):
        """Test that delay can be disabled by setting to 0."""
        os.environ["LLM_TOOL_REQUEST_DELAY"] = "0"
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(enable_tools=True)
            assert llm.tool_request_delay == 0.0

    def test_delay_is_applied_after_tool_execution(self):
        """Test that delay is actually applied after tool execution."""
        os.environ["LLM_TOOL_REQUEST_DELAY"] = "2"

        with patch("floship_llm.client.OpenAI") as mock_openai:
            llm = LLM(enable_tools=True)

            def test_func():
                return "test result"

            tool = ToolFunction(
                name="test_tool",
                description="Test tool",
                parameters=[],
                function=test_func,
            )
            llm.add_tool(tool)

            # Mock tool call response
            mock_tool_call = Mock()
            mock_tool_call.id = "call_123"
            mock_tool_call.function.name = "test_tool"
            mock_tool_call.function.arguments = "{}"

            mock_message = Mock()
            mock_message.content = "Using tool"
            mock_message.tool_calls = [mock_tool_call]

            mock_choice = Mock()
            mock_choice.message = mock_message

            mock_response = Mock()
            mock_response.choices = [mock_choice]

            # Mock follow-up response
            mock_follow_up_message = Mock()
            mock_follow_up_message.content = "Tool executed successfully"
            mock_follow_up_message.tool_calls = None

            mock_follow_up_choice = Mock()
            mock_follow_up_choice.message = mock_follow_up_message

            mock_follow_up_response = Mock()
            mock_follow_up_response.choices = [mock_follow_up_choice]

            mock_openai.return_value.chat.completions.create.return_value = (
                mock_follow_up_response
            )

            # Patch time.sleep to verify it's called
            with patch("time.sleep") as mock_sleep:
                llm.process_response(mock_response)

                # Verify sleep was called with the correct delay
                mock_sleep.assert_called_once_with(2.0)

    def test_no_delay_when_set_to_zero(self):
        """Test that no delay is applied when set to 0."""
        os.environ["LLM_TOOL_REQUEST_DELAY"] = "0"

        with patch("floship_llm.client.OpenAI") as mock_openai:
            llm = LLM(enable_tools=True)

            def test_func():
                return "test result"

            tool = ToolFunction(
                name="test_tool",
                description="Test tool",
                parameters=[],
                function=test_func,
            )
            llm.add_tool(tool)

            # Mock tool call response
            mock_tool_call = Mock()
            mock_tool_call.id = "call_123"
            mock_tool_call.function.name = "test_tool"
            mock_tool_call.function.arguments = "{}"

            mock_message = Mock()
            mock_message.content = "Using tool"
            mock_message.tool_calls = [mock_tool_call]

            mock_choice = Mock()
            mock_choice.message = mock_message

            mock_response = Mock()
            mock_response.choices = [mock_choice]

            # Mock follow-up response
            mock_follow_up_message = Mock()
            mock_follow_up_message.content = "Tool executed successfully"
            mock_follow_up_message.tool_calls = None

            mock_follow_up_choice = Mock()
            mock_follow_up_choice.message = mock_follow_up_message

            mock_follow_up_response = Mock()
            mock_follow_up_response.choices = [mock_follow_up_choice]

            mock_openai.return_value.chat.completions.create.return_value = (
                mock_follow_up_response
            )

            # Patch time.sleep to verify it's NOT called
            with patch("time.sleep") as mock_sleep:
                llm.process_response(mock_response)

                # Verify sleep was NOT called when delay is 0
                mock_sleep.assert_not_called()

    def test_delay_with_multiple_tool_calls(self):
        """Test that delay is applied once after all tools execute, not per tool."""
        os.environ["LLM_TOOL_REQUEST_DELAY"] = "1.5"

        with patch("floship_llm.client.OpenAI") as mock_openai:
            llm = LLM(enable_tools=True)

            def test_func():
                return "test result"

            tool = ToolFunction(
                name="test_tool",
                description="Test tool",
                parameters=[],
                function=test_func,
            )
            llm.add_tool(tool)

            # Mock two tool calls
            mock_tool_call_1 = Mock()
            mock_tool_call_1.id = "call_123"
            mock_tool_call_1.function.name = "test_tool"
            mock_tool_call_1.function.arguments = "{}"

            mock_tool_call_2 = Mock()
            mock_tool_call_2.id = "call_456"
            mock_tool_call_2.function.name = "test_tool"
            mock_tool_call_2.function.arguments = "{}"

            mock_message = Mock()
            mock_message.content = "Using tools"
            mock_message.tool_calls = [mock_tool_call_1, mock_tool_call_2]

            mock_choice = Mock()
            mock_choice.message = mock_message

            mock_response = Mock()
            mock_response.choices = [mock_choice]

            # Mock follow-up response
            mock_follow_up_message = Mock()
            mock_follow_up_message.content = "Tools executed successfully"
            mock_follow_up_message.tool_calls = None

            mock_follow_up_choice = Mock()
            mock_follow_up_choice.message = mock_follow_up_message

            mock_follow_up_response = Mock()
            mock_follow_up_response.choices = [mock_follow_up_choice]

            mock_openai.return_value.chat.completions.create.return_value = (
                mock_follow_up_response
            )

            # Patch time.sleep to verify it's called only once
            with patch("time.sleep") as mock_sleep:
                llm.process_response(mock_response)

                # Verify sleep was called only once with the correct delay
                mock_sleep.assert_called_once_with(1.5)

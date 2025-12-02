"""
Tests for tool/function calling functionality in the LLM client.
"""

import os
from unittest.mock import Mock, patch

import pytest

from floship_llm import LLM, ToolCall, ToolFunction, ToolParameter, ToolResult


class TestToolSchemas:
    """Test tool-related schema classes."""

    def test_tool_parameter_creation(self):
        """Test ToolParameter creation and validation."""
        param = ToolParameter(
            name="test_param",
            type="string",
            description="A test parameter",
            required=True,
            enum=["option1", "option2"],
        )

        assert param.name == "test_param"
        assert param.type == "string"
        assert param.description == "A test parameter"
        assert param.required is True
        assert param.enum == ["option1", "option2"]

    def test_tool_function_creation(self):
        """Test ToolFunction creation and validation."""

        def sample_function(x: int, y: str = "default") -> str:
            """A sample function for testing."""
            return f"{x}: {y}"

        params = [
            ToolParameter(
                name="x", type="integer", description="An integer", required=True
            ),
            ToolParameter(
                name="y",
                type="string",
                description="A string",
                required=False,
                default="default",
            ),
        ]

        tool = ToolFunction(
            name="sample_tool",
            description="A sample tool for testing",
            parameters=params,
            function=sample_function,
        )

        assert tool.name == "sample_tool"
        assert tool.description == "A sample tool for testing"
        assert len(tool.parameters) == 2
        assert tool.function == sample_function

    def test_tool_function_openai_format(self):
        """Test conversion to OpenAI format."""
        params = [
            ToolParameter(
                name="query", type="string", description="Search query", required=True
            ),
            ToolParameter(
                name="limit", type="integer", description="Max results", required=False
            ),
        ]

        tool = ToolFunction(
            name="web_search",
            description="Search the web for information",
            parameters=params,
        )

        openai_format = tool.to_openai_format()

        expected = {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "limit": {"type": "integer", "description": "Max results"},
                    },
                    "required": ["query"],
                },
            },
        }

        assert openai_format == expected

    def test_tool_function_openai_format_with_enum(self):
        """Test conversion to OpenAI format with enum parameters."""
        params = [
            ToolParameter(
                name="units",
                type="string",
                description="Temperature units",
                required=True,
                enum=["celsius", "fahrenheit"],
            ),
        ]

        tool = ToolFunction(
            name="get_temperature", description="Get temperature", parameters=params
        )

        openai_format = tool.to_openai_format()

        expected = {
            "type": "function",
            "function": {
                "name": "get_temperature",
                "description": "Get temperature",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "units": {
                            "type": "string",
                            "description": "Temperature units",
                            "enum": ["celsius", "fahrenheit"],
                        }
                    },
                    "required": ["units"],
                },
            },
        }

        assert openai_format == expected

    def test_tool_call_creation(self):
        """Test ToolCall creation."""
        tool_call = ToolCall(
            id="call_123",
            name="test_function",
            arguments={"param1": "value1", "param2": 42},
        )

        assert tool_call.id == "call_123"
        assert tool_call.name == "test_function"
        assert tool_call.arguments == {"param1": "value1", "param2": 42}

    def test_tool_result_creation(self):
        """Test ToolResult creation."""
        result = ToolResult(
            tool_call_id="call_123",
            name="test_function",
            content="Function executed successfully",
            success=True,
        )

        assert result.tool_call_id == "call_123"
        assert result.name == "test_function"
        assert result.content == "Function executed successfully"
        assert result.success is True
        assert result.error is None


class TestLLMTools:
    """Test LLM tool functionality."""

    def setup_method(self):
        """Set up test environment variables."""
        self.env_vars = {
            "INFERENCE_URL": "https://test-api.example.com",
            "INFERENCE_MODEL_ID": "test-model",
            "INFERENCE_KEY": "test-key",
            "INFERENCE_SUPPORTS_PARALLEL_REQUESTS": "True",
        }

        # Patch environment variables
        self.env_patcher = patch.dict(os.environ, self.env_vars, clear=False)
        self.env_patcher.start()

    def teardown_method(self):
        """Clean up test environment."""
        self.env_patcher.stop()

    def test_llm_tool_initialization(self):
        """Test LLM initialization with tool support."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(enable_tools=True)

            assert llm.enable_tools is True
            assert len(llm.tools) == 0
            assert isinstance(llm.tools, dict)

    def test_add_tool(self):
        """Test adding a tool to LLM."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()

            def test_func(message: str) -> str:
                return f"Processed: {message}"

            tool = ToolFunction(
                name="test_tool",
                description="A test tool",
                parameters=[
                    ToolParameter(name="message", type="string", required=True)
                ],
                function=test_func,
            )

            llm.add_tool(tool)

            assert "test_tool" in llm.tools
            assert llm.tools["test_tool"] == tool

    def test_add_tool_from_function(self):
        """Test adding a tool from a Python function."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()

            def calculate_sum(a: int, b: int) -> int:
                """Calculate the sum of two numbers."""
                return a + b

            llm.add_tool_from_function(
                calculate_sum, description="Adds two numbers together"
            )

            assert "calculate_sum" in llm.tools
            tool = llm.tools["calculate_sum"]
            assert tool.name == "calculate_sum"
            assert tool.description == "Adds two numbers together"
            assert tool.function == calculate_sum

    def test_remove_tool(self):
        """Test removing a tool."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()

            tool = ToolFunction(name="test_tool", description="Test", parameters=[])
            llm.add_tool(tool)

            assert "test_tool" in llm.tools

            llm.remove_tool("test_tool")

            assert "test_tool" not in llm.tools

    def test_list_tools(self):
        """Test listing available tools."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()

            tool1 = ToolFunction(name="tool1", description="Tool 1", parameters=[])
            tool2 = ToolFunction(name="tool2", description="Tool 2", parameters=[])

            llm.add_tool(tool1)
            llm.add_tool(tool2)

            tools = llm.list_tools()
            assert set(tools) == {"tool1", "tool2"}

    def test_execute_tool_success(self):
        """Test successful tool execution."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()

            def multiply(x: int, y: int) -> int:
                return x * y

            tool = ToolFunction(
                name="multiply",
                description="Multiply two numbers",
                parameters=[],
                function=multiply,
            )
            llm.add_tool(tool)

            tool_call = ToolCall(
                id="call_123", name="multiply", arguments={"x": 5, "y": 3}
            )

            result = llm.execute_tool(tool_call)

            assert result.success is True
            assert result.content == "15"
            assert result.tool_call_id == "call_123"
            assert result.name == "multiply"
            assert result.error is None

    def test_execute_tool_not_found(self):
        """Test executing a non-existent tool."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()

            tool_call = ToolCall(id="call_123", name="nonexistent_tool", arguments={})

            result = llm.execute_tool(tool_call)

            assert result.success is False
            assert "not found" in result.content.lower()
            assert result.error is not None

    def test_execute_tool_not_found_shows_available_tools(self):
        """Test that error message shows available tools when tool not found."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()

            # Register some tools
            tool1 = ToolFunction(
                name="tool_one",
                description="First tool",
                parameters=[],
                function=lambda: "one",
            )
            tool2 = ToolFunction(
                name="tool_two",
                description="Second tool",
                parameters=[],
                function=lambda: "two",
            )
            llm.add_tool(tool1)
            llm.add_tool(tool2)

            # Try to execute a non-existent tool
            tool_call = ToolCall(id="call_123", name="wrong_tool", arguments={})
            result = llm.execute_tool(tool_call)

            assert result.success is False
            assert "not found" in result.content.lower()
            assert "available tools" in result.content.lower()
            assert "tool_one" in result.content
            assert "tool_two" in result.content

    def test_execute_tool_function_error(self):
        """Test tool execution with function error."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()

            def error_func():
                raise ValueError("Test error")

            tool = ToolFunction(
                name="error_tool",
                description="A tool that errors",
                parameters=[],
                function=error_func,
            )
            llm.add_tool(tool)

            tool_call = ToolCall(id="call_123", name="error_tool", arguments={})

            result = llm.execute_tool(tool_call)

            assert result.success is False
            assert "Test error" in result.content
            assert result.error == "Test error"

    def test_get_request_params_with_tools(self):
        """Test get_request_params includes tools when enabled."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(enable_tools=True)

            tool = ToolFunction(
                name="test_tool",
                description="Test tool",
                parameters=[ToolParameter(name="param", type="string", required=True)],
            )
            llm.add_tool(tool)

            params = llm.get_request_params()

            assert "tools" in params
            assert "tool_choice" in params
            assert params["tool_choice"] == "auto"
            assert len(params["tools"]) == 1
            assert params["tools"][0]["function"]["name"] == "test_tool"

    def test_get_request_params_no_tools_when_disabled(self):
        """Test get_request_params excludes tools when disabled."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(enable_tools=False)

            tool = ToolFunction(name="test_tool", description="Test", parameters=[])
            llm.add_tool(tool)

            params = llm.get_request_params()

            assert "tools" not in params
            assert "tool_choice" not in params

    def test_enable_disable_tool_support(self):
        """Test enabling and disabling tool support."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()

            assert llm.enable_tools is False

            llm.enable_tool_support(True)
            assert llm.enable_tools is True

            llm.enable_tool_support(False)
            assert llm.enable_tools is False

    def test_clear_tools(self):
        """Test clearing all tools."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()

            tool1 = ToolFunction(name="tool1", description="Tool 1", parameters=[])
            tool2 = ToolFunction(name="tool2", description="Tool 2", parameters=[])

            llm.add_tool(tool1)
            llm.add_tool(tool2)

            assert len(llm.tools) == 2

            llm.clear_tools()

            assert len(llm.tools) == 0

    def test_handle_tool_calls_integration(self):
        """Test the complete tool call handling flow."""
        with patch("floship_llm.client.OpenAI") as mock_openai:
            llm = LLM(enable_tools=True)

            # Add a test tool
            def get_weather(location: str) -> str:
                return f"The weather in {location} is sunny."

            tool = ToolFunction(
                name="get_weather",
                description="Get weather for a location",
                parameters=[
                    ToolParameter(name="location", type="string", required=True)
                ],
                function=get_weather,
            )
            llm.add_tool(tool)

            # Mock the first response with tool calls
            mock_tool_call = Mock()
            mock_tool_call.id = "call_123"
            mock_tool_call.function.name = "get_weather"
            mock_tool_call.function.arguments = '{"location": "New York"}'

            mock_message = Mock()
            mock_message.content = "I'll check the weather for you."
            mock_message.tool_calls = [mock_tool_call]

            mock_choice = Mock()
            mock_choice.message = mock_message

            mock_response = Mock()
            mock_response.choices = [mock_choice]

            # Mock the follow-up response after tool execution
            mock_follow_up_message = Mock()
            mock_follow_up_message.content = (
                "Based on the weather data, it's sunny in New York!"
            )
            mock_follow_up_message.tool_calls = None

            mock_follow_up_choice = Mock()
            mock_follow_up_choice.message = mock_follow_up_message

            mock_follow_up_response = Mock()
            mock_follow_up_response.choices = [mock_follow_up_choice]

            # Set up the client mock to return the follow-up response
            mock_openai.return_value.chat.completions.create.return_value = (
                mock_follow_up_response
            )

            # Process the response
            result = llm.process_response(mock_response)

            # Check that the tool was executed and conversation updated
            assert len(llm.messages) >= 3  # assistant message, tool result, follow-up
            assert any(msg.get("role") == "tool" for msg in llm.messages)
            assert result == "Based on the weather data, it's sunny in New York!"

    def test_remove_tool_not_found(self):
        """Test removing a tool that doesn't exist."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()

            # Try to remove a tool that doesn't exist
            llm.remove_tool("nonexistent_tool")
            # Should not raise an error, just log a warning

    def test_execute_tool_no_function(self):
        """Test executing a tool with no function defined."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()

            # Create a tool without a function
            tool = ToolFunction(
                name="no_function_tool",
                description="A tool without function",
                parameters=[],
                function=None,  # No function
            )
            llm.add_tool(tool)

            tool_call = ToolCall(id="call_123", name="no_function_tool", arguments={})

            result = llm.execute_tool(tool_call)

            assert result.success is False
            assert "no executable function" in result.content.lower()
            assert result.error == "No executable function defined"

    def test_process_response_empty_content(self):
        """Test processing response with empty content."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()

            # Mock response with empty content
            mock_message = Mock()
            mock_message.content = None
            mock_message.tool_calls = None

            mock_choice = Mock()
            mock_choice.message = mock_message

            mock_response = Mock()
            mock_response.choices = [mock_choice]

            result = llm.process_response(mock_response)
            assert result == ""

    def test_handle_tool_calls_json_parse_error(self):
        """Test tool call handling with JSON parsing error."""
        with patch("floship_llm.client.OpenAI") as mock_openai:
            llm = LLM(enable_tools=True)

            def test_func():
                return "test"

            tool = ToolFunction(
                name="test_tool",
                description="Test tool",
                parameters=[],
                function=test_func,
            )
            llm.add_tool(tool)

            # Mock tool call with invalid JSON arguments
            mock_tool_call = Mock()
            mock_tool_call.id = "call_123"
            mock_tool_call.function.name = "test_tool"
            mock_tool_call.function.arguments = "invalid json"

            mock_message = Mock()
            mock_message.content = "I'll use the tool."
            mock_message.tool_calls = [mock_tool_call]

            mock_choice = Mock()
            mock_choice.message = mock_message

            mock_response = Mock()
            mock_response.choices = [mock_choice]

            # Mock follow-up response
            mock_follow_up_message = Mock()
            mock_follow_up_message.content = "Error occurred"
            mock_follow_up_message.tool_calls = None

            mock_follow_up_choice = Mock()
            mock_follow_up_choice.message = mock_follow_up_message

            mock_follow_up_response = Mock()
            mock_follow_up_response.choices = [mock_follow_up_choice]

            mock_openai.return_value.chat.completions.create.return_value = (
                mock_follow_up_response
            )

            # This should handle the JSON error gracefully
            llm.process_response(mock_response)

            # Should have error messages in conversation
            tool_messages = [msg for msg in llm.messages if msg.get("role") == "tool"]
            assert len(tool_messages) > 0

    def test_reset_with_system_message(self):
        """Test reset method with system message."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(system="You are a helpful assistant")

            # Add some messages
            llm.add_message("user", "Hello")
            llm.add_message("assistant", "Hi there")

            assert len(llm.messages) == 3  # system + user + assistant

            # Reset should clear messages and re-add system
            llm.reset()

            assert len(llm.messages) == 1  # Only system message
            assert llm.messages[0]["role"] == "system"
            assert llm.messages[0]["content"] == "You are a helpful assistant"

    def test_process_response_mock_tool_calls_exception(self):
        """Test processing response with mock tool_calls that cause exceptions."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(enable_tools=True)

            # Create a mock that raises TypeError when accessed for iteration
            mock_tool_calls = Mock()
            mock_tool_calls.__iter__ = Mock(
                side_effect=TypeError("Mock iteration error")
            )

            mock_message = Mock()
            mock_message.content = "Regular response"
            mock_message.tool_calls = mock_tool_calls

            mock_choice = Mock()
            mock_choice.message = mock_message

            mock_response = Mock()
            mock_response.choices = [mock_choice]

            # Should handle the exception and process as regular message
            result = llm.process_response(mock_response)

            # Should process as regular response, not tool call
            assert result == "Regular response"

    def test_handle_tool_calls_with_none_content(self):
        """Test tool call handling when message content is None."""
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

            # Mock tool call with None content
            mock_tool_call = Mock()
            mock_tool_call.id = "call_123"
            mock_tool_call.function.name = "test_tool"
            mock_tool_call.function.arguments = "{}"

            mock_message = Mock()
            mock_message.content = None  # This is the key issue we're testing
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

            # This should handle None content gracefully
            llm.process_response(mock_response)

            # Should have proper messages in conversation
            assistant_messages = [
                msg for msg in llm.messages if msg.get("role") == "assistant"
            ]
            assert len(assistant_messages) > 0
            # The first assistant message (with tool calls) should have empty content, not None
            tool_call_message = None
            for msg in assistant_messages:
                if "tool_calls" in msg:
                    tool_call_message = msg
                    break
            assert tool_call_message is not None
            # The library may normalize empty assistant content to a placeholder for API compatibility
            assert tool_call_message["content"] in ["", " ", "[Tool calls in progress]"]

            tool_messages = [msg for msg in llm.messages if msg.get("role") == "tool"]
            assert len(tool_messages) > 0
            # Tool result should have proper content
            assert tool_messages[0]["content"] == "test result"

    def test_validate_messages_for_api(self):
        """Test the _validate_messages_for_api method comprehensively."""
        with patch.dict(
            "os.environ",
            {
                "INFERENCE_URL": "http://test.com",
                "INFERENCE_MODEL_ID": "test-model",
                "INFERENCE_KEY": "test-key",
            },
        ):
            llm = LLM()

            # Test various message scenarios
            test_messages = [
                {"role": "user", "content": "Hello"},  # Valid message
                {"role": "assistant", "content": None},  # None content
                {
                    "role": "tool",
                    "tool_call_id": "123",
                    "content": None,
                },  # Tool with None content
                {"role": "system", "content": ""},  # Empty content
                {"role": "user"},  # Missing content key
                {
                    "role": "assistant",
                    "content": "Response",
                    "tool_calls": [{"id": "123"}],
                },  # With tool calls
                {"role": "unknown", "content": None},  # Unknown role with None content
                {"content": "No role"},  # Missing role
                None,  # Invalid message type
                "string_message",  # Invalid message type
                {
                    "role": "function",
                    "content": 42,
                },  # Non-string content, different role
            ]

            validated = llm._validate_messages_for_api(test_messages)

            # Should have filtered out invalid messages (None, string, missing role)
            assert len(validated) == 8

            # Check each validated message
            for msg in validated:
                assert isinstance(msg, dict)
                assert "role" in msg
                assert "content" in msg
                assert isinstance(msg["content"], str)

            # Check specific validations
            user_msg = next(
                msg
                for msg in validated
                if msg["role"] == "user" and "tool_call_id" not in msg
            )
            assert user_msg["content"] == "Hello"

            assistant_msgs = [msg for msg in validated if msg["role"] == "assistant"]
            assert len(assistant_msgs) == 2
            # Assistant with None content should get a default message
            none_content_assistant = next(
                msg for msg in assistant_msgs if "tool_calls" not in msg
            )
            # Assistant without tool_calls gets default content
            assert none_content_assistant["content"] in [
                "Message content unavailable",
                "Content unavailable",
                ".",
                " ",
            ]

            tool_msgs = [msg for msg in validated if msg["role"] == "tool"]
            assert len(tool_msgs) == 1
            # Tool with None content should get default message
            assert tool_msgs[0]["content"] == "Tool executed successfully"

            system_msgs = [msg for msg in validated if msg["role"] == "system"]
            assert len(system_msgs) == 1
            # System with empty content should get default content
            assert system_msgs[0]["content"] in ["Content unavailable", ".", " "]

            unknown_msgs = [msg for msg in validated if msg["role"] == "unknown"]
            assert len(unknown_msgs) == 1
            # Unknown role with None content should get default content
            assert unknown_msgs[0]["content"] in [
                "Message content unavailable",
                "Content unavailable",
                ".",
                " ",
            ]

            # Content that was a number should be converted to string
            function_msgs = [msg for msg in validated if msg["role"] == "function"]
            assert len(function_msgs) == 1
            assert function_msgs[0]["content"] == "42"  # Number converted to string

    def test_validate_messages_for_api_edge_cases(self):
        """Test edge cases for _validate_messages_for_api method."""
        with patch.dict(
            "os.environ",
            {
                "INFERENCE_URL": "http://test.com",
                "INFERENCE_MODEL_ID": "test-model",
                "INFERENCE_KEY": "test-key",
            },
        ):
            llm = LLM()

            # Test user message with missing content (to cover line 488)
            test_messages = [
                {"role": "user", "content": None},  # User with None content
            ]

            validated = llm._validate_messages_for_api(test_messages)

            assert len(validated) == 1
            assert validated[0]["role"] == "user"
            assert validated[0]["content"] in [
                "Message content unavailable",
                "Content unavailable",
                ".",
                " ",
            ]  # Should get default content

    def test_tool_content_sanitization(self):
        """Test the _sanitize_tool_content method for empty/None results."""
        with patch.dict(
            "os.environ",
            {
                "INFERENCE_URL": "http://test.com",
                "INFERENCE_MODEL_ID": "test-model",
                "INFERENCE_KEY": "test-key",
            },
        ):
            llm = LLM()

            # Test various problematic tool results
            test_cases = [
                (None, "Tool test_tool executed successfully (no return value)"),
                ("", "Tool test_tool executed successfully (empty result)"),
                ("   ", "Tool test_tool executed successfully (empty result)"),
                ("None", "Tool test_tool executed successfully (null result)"),
                ("null", "Tool test_tool executed successfully (null result)"),
                ("NULL", "Tool test_tool executed successfully (null result)"),
                ("Valid result", "Valid result"),
            ]

            for content, expected in test_cases:
                result = llm._sanitize_tool_content(content, "test_tool")
                assert result == expected, (
                    f"Failed for content '{content}': got '{result}', expected '{expected}'"
                )

            # Test error cases
            error_cases = [
                (None, "Error executing tool: test_tool"),
                ("", "Error executing tool: test_tool (empty result)"),
                ("None", "Error executing tool: test_tool (null result)"),
            ]

            for content, expected in error_cases:
                result = llm._sanitize_tool_content(content, "test_tool", is_error=True)
                assert result == expected, (
                    f"Failed for error content '{content}': got '{result}', expected '{expected}'"
                )

    def test_execute_tool_with_problematic_results(self):
        """Test execute_tool method with various problematic return values."""
        with patch.dict(
            "os.environ",
            {
                "INFERENCE_URL": "http://test.com",
                "INFERENCE_MODEL_ID": "test-model",
                "INFERENCE_KEY": "test-key",
            },
        ):
            llm = LLM()

            # Test tool that returns None
            def none_tool():
                return None

            tool_none = ToolFunction(
                name="none_tool",
                description="Returns None",
                parameters=[],
                function=none_tool,
            )
            llm.add_tool(tool_none)

            call_none = ToolCall(id="test1", name="none_tool", arguments={})
            result_none = llm.execute_tool(call_none)

            assert result_none.success is True
            assert "Tool executed successfully (no return value)" in result_none.content

            # Test tool that returns empty string
            def empty_tool():
                return ""

            tool_empty = ToolFunction(
                name="empty_tool",
                description="Returns empty",
                parameters=[],
                function=empty_tool,
            )
            llm.add_tool(tool_empty)

            call_empty = ToolCall(id="test2", name="empty_tool", arguments={})
            result_empty = llm.execute_tool(call_empty)

            assert result_empty.success is True
            assert "Tool executed successfully (empty result)" in result_empty.content

            # Test tool that returns "None" string
            def none_string_tool():
                return "None"

            tool_none_str = ToolFunction(
                name="none_string_tool",
                description="Returns 'None'",
                parameters=[],
                function=none_string_tool,
            )
            llm.add_tool(tool_none_str)

            call_none_str = ToolCall(id="test3", name="none_string_tool", arguments={})
            result_none_str = llm.execute_tool(call_none_str)

            assert result_none_str.success is True
            assert "Tool executed successfully (null result)" in result_none_str.content

    def test_validate_messages_comprehensive_coverage(self):
        """Test comprehensive edge cases to achieve full coverage."""
        with patch.dict(
            "os.environ",
            {
                "INFERENCE_URL": "http://test.com",
                "INFERENCE_MODEL_ID": "test-model",
                "INFERENCE_KEY": "test-key",
            },
        ):
            llm = LLM()

            # Test assistant message with tool_calls and empty content
            test_messages = [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "123", "function": {"name": "test"}}],
                },
            ]

            validated = llm._validate_messages_for_api(test_messages)

            assert len(validated) == 1
            assert validated[0]["role"] == "assistant"
            assert "tool_calls" in validated[0]
            # Assistant with tool_calls now uses empty string (API requires content field)
            assert validated[0]["content"] == ""

    def test_validate_messages_edge_logging(self):
        """Test edge case logging paths for full coverage."""
        with patch.dict(
            "os.environ",
            {
                "INFERENCE_URL": "http://test.com",
                "INFERENCE_MODEL_ID": "test-model",
                "INFERENCE_KEY": "test-key",
            },
        ):
            llm = LLM()

            # Create a scenario that would test the logging paths
            # First test: message that might have content issues after validation
            import logging

            with patch.object(
                logging.getLogger("floship_llm.client"), "error"
            ) as mock_logger:
                # Test normal validation - should not trigger error logs
                normal_messages = [{"role": "user", "content": "test"}]
                validated = llm._validate_messages_for_api(normal_messages)

                # Error logs should not be called for normal cases
                mock_logger.assert_not_called()

                assert len(validated) == 1
                assert validated[0]["content"] == "test"


class TestToolManagerValidation:
    """Tests for ToolManager validation and error handling."""

    def test_add_non_tool_function_raises_error(self):
        """Test that adding a non-ToolFunction raises ValueError."""
        from floship_llm.tool_manager import ToolManager

        manager = ToolManager()
        with pytest.raises(ValueError, match="must be an instance of ToolFunction"):
            manager.add_tool("not a tool function")

    def test_add_invalid_tool_name_raises_error(self):
        """Test that adding a tool with invalid name raises ValueError."""
        from floship_llm.tool_manager import ToolManager

        manager = ToolManager()
        tool = ToolFunction(
            name="invalid name with spaces",
            description="A tool with an invalid name",
            function=lambda: None,
        )
        with pytest.raises(ValueError, match="Invalid tool name"):
            manager.add_tool(tool)

    def test_add_duplicate_tool_raises_error(self):
        """Test that adding a duplicate tool raises ValueError."""
        from floship_llm.tool_manager import ToolManager

        manager = ToolManager()
        tool = ToolFunction(
            name="test_tool",
            description="A test tool",
            function=lambda: None,
        )
        manager.add_tool(tool)

        # Try to add the same tool again
        with pytest.raises(ValueError, match="already exists"):
            manager.add_tool(tool)

    def test_execute_invalid_tool_call_type(self):
        """Test that execute_tool with wrong type raises ValueError."""
        from floship_llm.tool_manager import ToolManager

        manager = ToolManager()
        with pytest.raises(ValueError, match="must be an instance of ToolCall"):
            manager.execute_tool("not a tool call")

    def test_execute_tool_without_function(self):
        """Test executing a tool that has no function defined."""
        from floship_llm.tool_manager import ToolManager

        manager = ToolManager()
        # Create tool without a function
        tool = ToolFunction(
            name="no_func_tool",
            description="A tool without a function",
            function=None,
        )
        manager.tools["no_func_tool"] = tool

        tc = ToolCall(id="call_1", name="no_func_tool", arguments={})
        result = manager.execute_tool(tc)

        assert result.success is False
        assert result.error is not None
        assert "No executable function defined" in result.error

    def test_execute_tool_json_parse_error(self):
        """Test tool execution with arguments that can't be parsed as dict."""
        from floship_llm.tool_manager import ToolManager

        manager = ToolManager()
        tool = ToolFunction(
            name="test_tool",
            description="A test tool",
            function=lambda: "result",
        )
        manager.add_tool(tool)

        # ToolCall requires dict arguments, so we test with the tool manager directly
        # by manipulating what gets passed to the tool
        tc = ToolCall(id="call_1", name="test_tool", arguments={})
        result = manager.execute_tool(tc)

        # This should succeed with empty args
        assert result.success is True

    def test_execute_tool_with_whitespace_result(self):
        """Test tool execution that returns only whitespace."""
        from floship_llm.tool_manager import ToolManager

        manager = ToolManager()
        tool = ToolFunction(
            name="whitespace_tool",
            description="A tool that returns whitespace",
            function=lambda: "   ",  # Only whitespace
        )
        manager.add_tool(tool)

        tc = ToolCall(id="call_1", name="whitespace_tool", arguments={})
        result = manager.execute_tool(tc)

        assert result.success is True
        # Whitespace is stripped and treated as empty
        assert "empty result" in result.content.lower()

    def test_execute_tool_with_dict_result(self):
        """Test tool execution that returns a dict."""
        from floship_llm.tool_manager import ToolManager

        manager = ToolManager()
        tool = ToolFunction(
            name="dict_tool",
            description="A tool that returns a dict",
            function=lambda: {"key": "value", "nested": {"a": 1}},
        )
        manager.add_tool(tool)

        tc = ToolCall(id="call_1", name="dict_tool", arguments={})
        result = manager.execute_tool(tc)

        assert result.success is True
        # Result should be JSON formatted
        assert "key" in result.content
        assert "value" in result.content

    def test_execute_tool_with_list_result(self):
        """Test tool execution that returns a list."""
        from floship_llm.tool_manager import ToolManager

        manager = ToolManager()
        tool = ToolFunction(
            name="list_tool",
            description="A tool that returns a list",
            function=lambda: [1, 2, 3],
        )
        manager.add_tool(tool)

        tc = ToolCall(id="call_1", name="list_tool", arguments={})
        result = manager.execute_tool(tc)

        assert result.success is True
        # Result should be JSON formatted
        assert "1" in result.content
        assert "2" in result.content
        assert "3" in result.content

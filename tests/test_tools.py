"""
Tests for tool/function calling functionality in the LLM client.
"""

import json
import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from floship_llm import LLM, ToolFunction, ToolParameter, ToolCall, ToolResult


class TestToolSchemas:
    """Test tool-related schema classes."""
    
    def test_tool_parameter_creation(self):
        """Test ToolParameter creation and validation."""
        param = ToolParameter(
            name="test_param",
            type="string",
            description="A test parameter",
            required=True,
            enum=["option1", "option2"]
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
            ToolParameter(name="x", type="integer", description="An integer", required=True),
            ToolParameter(name="y", type="string", description="A string", required=False, default="default")
        ]
        
        tool = ToolFunction(
            name="sample_tool",
            description="A sample tool for testing",
            parameters=params,
            function=sample_function
        )
        
        assert tool.name == "sample_tool"
        assert tool.description == "A sample tool for testing"
        assert len(tool.parameters) == 2
        assert tool.function == sample_function
    
    def test_tool_function_openai_format(self):
        """Test conversion to OpenAI format."""
        params = [
            ToolParameter(name="query", type="string", description="Search query", required=True),
            ToolParameter(name="limit", type="integer", description="Max results", required=False)
        ]
        
        tool = ToolFunction(
            name="web_search",
            description="Search the web for information",
            parameters=params
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
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results"
                        }
                    },
                    "required": ["query"]
                }
            }
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
                enum=["celsius", "fahrenheit"]
            ),
        ]
        
        tool = ToolFunction(
            name="get_temperature",
            description="Get temperature",
            parameters=params
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
                            "enum": ["celsius", "fahrenheit"]
                        }
                    },
                    "required": ["units"]
                }
            }
        }
        
        assert openai_format == expected
    
    def test_tool_call_creation(self):
        """Test ToolCall creation."""
        tool_call = ToolCall(
            id="call_123",
            name="test_function",
            arguments={"param1": "value1", "param2": 42}
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
            success=True
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
            'INFERENCE_URL': 'https://test-api.example.com',
            'INFERENCE_MODEL_ID': 'test-model',
            'INFERENCE_KEY': 'test-key',
            'INFERENCE_SUPPORTS_PARALLEL_REQUESTS': 'True'
        }
        
        # Patch environment variables
        self.env_patcher = patch.dict(os.environ, self.env_vars, clear=False)
        self.env_patcher.start()
        
    def teardown_method(self):
        """Clean up test environment."""
        self.env_patcher.stop()
    
    def test_llm_tool_initialization(self):
        """Test LLM initialization with tool support."""
        with patch('floship_llm.client.OpenAI') as mock_openai:
            llm = LLM(enable_tools=True)
            
            assert llm.enable_tools is True
            assert len(llm.tools) == 0
            assert isinstance(llm.tools, dict)
    
    def test_add_tool(self):
        """Test adding a tool to LLM."""
        with patch('floship_llm.client.OpenAI') as mock_openai:
            llm = LLM()
            
            def test_func(message: str) -> str:
                return f"Processed: {message}"
            
            tool = ToolFunction(
                name="test_tool",
                description="A test tool",
                parameters=[
                    ToolParameter(name="message", type="string", required=True)
                ],
                function=test_func
            )
            
            llm.add_tool(tool)
            
            assert "test_tool" in llm.tools
            assert llm.tools["test_tool"] == tool
    
    def test_add_tool_from_function(self):
        """Test adding a tool from a Python function."""
        with patch('floship_llm.client.OpenAI') as mock_openai:
            llm = LLM()
            
            def calculate_sum(a: int, b: int) -> int:
                """Calculate the sum of two numbers."""
                return a + b
            
            llm.add_tool_from_function(
                calculate_sum,
                description="Adds two numbers together"
            )
            
            assert "calculate_sum" in llm.tools
            tool = llm.tools["calculate_sum"]
            assert tool.name == "calculate_sum"
            assert tool.description == "Adds two numbers together"
            assert tool.function == calculate_sum
    
    def test_remove_tool(self):
        """Test removing a tool."""
        with patch('floship_llm.client.OpenAI') as mock_openai:
            llm = LLM()
            
            tool = ToolFunction(name="test_tool", description="Test", parameters=[])
            llm.add_tool(tool)
            
            assert "test_tool" in llm.tools
            
            llm.remove_tool("test_tool")
            
            assert "test_tool" not in llm.tools
    
    def test_list_tools(self):
        """Test listing available tools."""
        with patch('floship_llm.client.OpenAI') as mock_openai:
            llm = LLM()
            
            tool1 = ToolFunction(name="tool1", description="Tool 1", parameters=[])
            tool2 = ToolFunction(name="tool2", description="Tool 2", parameters=[])
            
            llm.add_tool(tool1)
            llm.add_tool(tool2)
            
            tools = llm.list_tools()
            assert set(tools) == {"tool1", "tool2"}
    
    def test_execute_tool_success(self):
        """Test successful tool execution."""
        with patch('floship_llm.client.OpenAI') as mock_openai:
            llm = LLM()
            
            def multiply(x: int, y: int) -> int:
                return x * y
            
            tool = ToolFunction(
                name="multiply",
                description="Multiply two numbers",
                parameters=[],
                function=multiply
            )
            llm.add_tool(tool)
            
            tool_call = ToolCall(
                id="call_123",
                name="multiply",
                arguments={"x": 5, "y": 3}
            )
            
            result = llm.execute_tool(tool_call)
            
            assert result.success is True
            assert result.content == "15"
            assert result.tool_call_id == "call_123"
            assert result.name == "multiply"
            assert result.error is None
    
    def test_execute_tool_not_found(self):
        """Test executing a non-existent tool."""
        with patch('floship_llm.client.OpenAI') as mock_openai:
            llm = LLM()
            
            tool_call = ToolCall(
                id="call_123",
                name="nonexistent_tool",
                arguments={}
            )
            
            result = llm.execute_tool(tool_call)
            
            assert result.success is False
            assert "not found" in result.content.lower()
            assert result.error is not None
    
    def test_execute_tool_function_error(self):
        """Test tool execution with function error."""
        with patch('floship_llm.client.OpenAI') as mock_openai:
            llm = LLM()
            
            def error_func():
                raise ValueError("Test error")
            
            tool = ToolFunction(
                name="error_tool",
                description="A tool that errors",
                parameters=[],
                function=error_func
            )
            llm.add_tool(tool)
            
            tool_call = ToolCall(
                id="call_123",
                name="error_tool",
                arguments={}
            )
            
            result = llm.execute_tool(tool_call)
            
            assert result.success is False
            assert "Test error" in result.content
            assert result.error == "Test error"
    
    def test_get_request_params_with_tools(self):
        """Test get_request_params includes tools when enabled."""
        with patch('floship_llm.client.OpenAI') as mock_openai:
            llm = LLM(enable_tools=True)
            
            tool = ToolFunction(
                name="test_tool",
                description="Test tool",
                parameters=[
                    ToolParameter(name="param", type="string", required=True)
                ]
            )
            llm.add_tool(tool)
            
            params = llm.get_request_params()
            
            assert 'tools' in params
            assert 'tool_choice' in params
            assert params['tool_choice'] == 'auto'
            assert len(params['tools']) == 1
            assert params['tools'][0]['function']['name'] == 'test_tool'
    
    def test_get_request_params_no_tools_when_disabled(self):
        """Test get_request_params excludes tools when disabled."""
        with patch('floship_llm.client.OpenAI') as mock_openai:
            llm = LLM(enable_tools=False)
            
            tool = ToolFunction(name="test_tool", description="Test", parameters=[])
            llm.add_tool(tool)
            
            params = llm.get_request_params()
            
            assert 'tools' not in params
            assert 'tool_choice' not in params
    
    def test_enable_disable_tool_support(self):
        """Test enabling and disabling tool support."""
        with patch('floship_llm.client.OpenAI') as mock_openai:
            llm = LLM()
            
            assert llm.enable_tools is False
            
            llm.enable_tool_support(True)
            assert llm.enable_tools is True
            
            llm.enable_tool_support(False)
            assert llm.enable_tools is False
    
    def test_clear_tools(self):
        """Test clearing all tools."""
        with patch('floship_llm.client.OpenAI') as mock_openai:
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
        with patch('floship_llm.client.OpenAI') as mock_openai:
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
                function=get_weather
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
            mock_follow_up_message.content = "Based on the weather data, it's sunny in New York!"
            mock_follow_up_message.tool_calls = None
            
            mock_follow_up_choice = Mock()
            mock_follow_up_choice.message = mock_follow_up_message
            
            mock_follow_up_response = Mock()
            mock_follow_up_response.choices = [mock_follow_up_choice]
            
            # Set up the client mock to return the follow-up response
            mock_openai.return_value.chat.completions.create.return_value = mock_follow_up_response
            
            # Process the response
            result = llm.process_response(mock_response)
            
            # Check that the tool was executed and conversation updated
            assert len(llm.messages) >= 3  # assistant message, tool result, follow-up
            assert any(msg.get('role') == 'tool' for msg in llm.messages)
            assert result == "Based on the weather data, it's sunny in New York!"

    def test_remove_tool_not_found(self):
        """Test removing a tool that doesn't exist."""
        with patch('floship_llm.client.OpenAI') as mock_openai:
            llm = LLM()
            
            # Try to remove a tool that doesn't exist
            llm.remove_tool("nonexistent_tool")
            # Should not raise an error, just log a warning
            
    def test_execute_tool_no_function(self):
        """Test executing a tool with no function defined."""
        with patch('floship_llm.client.OpenAI') as mock_openai:
            llm = LLM()
            
            # Create a tool without a function
            tool = ToolFunction(
                name="no_function_tool",
                description="A tool without function",
                parameters=[],
                function=None  # No function
            )
            llm.add_tool(tool)
            
            tool_call = ToolCall(
                id="call_123",
                name="no_function_tool",
                arguments={}
            )
            
            result = llm.execute_tool(tool_call)
            
            assert result.success is False
            assert "no executable function" in result.content.lower()
            assert result.error == "No executable function defined"
    
    def test_process_response_empty_content(self):
        """Test processing response with empty content."""
        with patch('floship_llm.client.OpenAI') as mock_openai:
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
        with patch('floship_llm.client.OpenAI') as mock_openai:
            llm = LLM(enable_tools=True)
            
            def test_func():
                return "test"
            
            tool = ToolFunction(
                name="test_tool",
                description="Test tool",
                parameters=[],
                function=test_func
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
            
            mock_openai.return_value.chat.completions.create.return_value = mock_follow_up_response
            
            # This should handle the JSON error gracefully
            result = llm.process_response(mock_response)
            
            # Should have error messages in conversation
            tool_messages = [msg for msg in llm.messages if msg.get('role') == 'tool']
            assert len(tool_messages) > 0
            
    def test_reset_with_system_message(self):
        """Test reset method with system message."""
        with patch('floship_llm.client.OpenAI') as mock_openai:
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
        with patch('floship_llm.client.OpenAI') as mock_openai:
            llm = LLM(enable_tools=True)
            
            # Create a mock that raises TypeError when accessed for iteration
            mock_tool_calls = Mock()
            mock_tool_calls.__iter__ = Mock(side_effect=TypeError("Mock iteration error"))
            
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
        with patch('floship_llm.client.OpenAI') as mock_openai:
            llm = LLM(enable_tools=True)
            
            def test_func():
                return "test result"
            
            tool = ToolFunction(
                name="test_tool",
                description="Test tool",
                parameters=[],
                function=test_func
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
            
            mock_openai.return_value.chat.completions.create.return_value = mock_follow_up_response
            
            # This should handle None content gracefully
            result = llm.process_response(mock_response)
            
            # Should have proper messages in conversation
            assistant_messages = [msg for msg in llm.messages if msg.get('role') == 'assistant']
            assert len(assistant_messages) > 0
            # The first assistant message (with tool calls) should have empty content, not None
            tool_call_message = None
            for msg in assistant_messages:
                if 'tool_calls' in msg:
                    tool_call_message = msg
                    break
            assert tool_call_message is not None
            assert tool_call_message['content'] == ""
            
            tool_messages = [msg for msg in llm.messages if msg.get('role') == 'tool']
            assert len(tool_messages) > 0
            # Tool result should have proper content
            assert tool_messages[0]['content'] == "test result"
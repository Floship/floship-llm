"""Test tool serialization to ensure ToolParameter objects are properly converted to dicts."""

import json
import os
from unittest.mock import Mock, patch

import pytest

from floship_llm import LLM
from floship_llm.schemas import ToolFunction, ToolParameter


class TestToolSerialization:
    """Test that tools are properly serialized for JSON API calls."""

    @pytest.fixture(autouse=True)
    def setup_env(self):
        """Set up environment variables for all tests."""
        with patch.dict(
            os.environ,
            {
                "INFERENCE_URL": "https://test-api.example.com",
                "INFERENCE_MODEL_ID": "test-model",
                "INFERENCE_KEY": "test-key",
            },
        ):
            yield

    def test_tool_parameter_serialization_in_request(self):
        """Test that ToolParameter objects are converted to dicts before JSON serialization."""
        with patch("floship_llm.client.OpenAI") as mock_openai:
            # Create LLM with tools enabled
            llm = LLM(enable_tools=True)

            # Add a tool with parameters
            def example_function(name: str, count: int):
                """An example function."""
                return f"Hello {name}, count is {count}"

            llm.add_tool_from_function(
                example_function,
                name="example_tool",
                description="An example tool",
                parameters=[
                    ToolParameter(
                        name="name",
                        type="string",
                        description="The name parameter",
                        required=True,
                    ),
                    ToolParameter(
                        name="count",
                        type="integer",
                        description="The count parameter",
                        required=True,
                    ),
                ],
            )

            # Get request params
            params = llm.get_request_params()

            # Verify tools are in params
            assert "tools" in params
            assert len(params["tools"]) == 1

            # This should not raise TypeError when serialized to JSON
            try:
                json_str = json.dumps(params)
                assert json_str is not None
            except TypeError as e:
                pytest.fail(f"Failed to serialize params to JSON: {e}")

            # Verify the structure is correct
            tool_schema = params["tools"][0]
            assert tool_schema["type"] == "function"
            assert "function" in tool_schema
            assert tool_schema["function"]["name"] == "example_tool"
            assert "parameters" in tool_schema["function"]

            # Parameters should be a dict, not a list of ToolParameter objects
            parameters = tool_schema["function"]["parameters"]
            assert isinstance(
                parameters, dict
            ), f"Expected dict, got {type(parameters)}"
            assert "type" in parameters
            assert parameters["type"] == "object"
            assert "properties" in parameters
            assert "required" in parameters

    def test_get_tools_schema_returns_serializable_dict(self):
        """Test that get_tools_schema returns JSON-serializable dictionaries."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM(enable_tools=True)

            # Add a simple tool
            def simple_tool(text: str):
                """A simple tool."""
                return f"Processed: {text}"

            llm.add_tool_from_function(simple_tool)

            # Get tools schema
            schema = llm.tool_manager.get_tools_schema()

            # Should be JSON serializable
            try:
                json_str = json.dumps(schema)
                assert json_str is not None
            except TypeError as e:
                pytest.fail(f"get_tools_schema() returned non-serializable data: {e}")

    def test_tool_function_to_openai_format(self):
        """Test that ToolFunction.to_openai_format() produces correct structure."""
        tool = ToolFunction(
            name="test_tool",
            description="Test tool description",
            parameters=[
                ToolParameter(
                    name="param1",
                    type="string",
                    description="First parameter",
                    required=True,
                ),
                ToolParameter(
                    name="param2",
                    type="integer",
                    description="Second parameter",
                    required=False,
                    default=42,
                ),
            ],
        )

        # Convert to OpenAI format
        openai_format = tool.to_openai_format()

        # Verify structure
        assert openai_format["type"] == "function"
        assert openai_format["function"]["name"] == "test_tool"
        assert openai_format["function"]["description"] == "Test tool description"

        # Parameters should be properly formatted
        params = openai_format["function"]["parameters"]
        assert params["type"] == "object"
        assert "param1" in params["properties"]
        assert "param2" in params["properties"]
        assert params["properties"]["param1"]["type"] == "string"
        assert params["properties"]["param2"]["type"] == "integer"
        assert "param1" in params["required"]
        assert "param2" not in params["required"]

        # Should be JSON serializable
        try:
            json_str = json.dumps(openai_format)
            assert json_str is not None
        except TypeError as e:
            pytest.fail(f"to_openai_format() returned non-serializable data: {e}")

    def test_prompt_with_tools_serializes_correctly(self):
        """Test that calling prompt() with tools doesn't raise serialization errors."""
        with patch("floship_llm.client.OpenAI") as mock_openai:
            # Setup mock
            mock_client = Mock()
            mock_openai.return_value = mock_client

            mock_response = Mock()
            mock_choice = Mock()
            mock_choice.message.content = "Test response"
            mock_choice.message.tool_calls = None
            mock_response.choices = [mock_choice]

            mock_client.chat.completions.create.return_value = mock_response

            # Create LLM with tools
            llm = LLM(enable_tools=True)

            def test_function(query: str):
                """Test function."""
                return f"Result: {query}"

            llm.add_tool_from_function(test_function)

            # Call prompt - this should not raise TypeError about JSON serialization
            try:
                response = llm.prompt("Test prompt")
                assert response == "Test response"
            except TypeError as e:
                if "not JSON serializable" in str(e):
                    pytest.fail(f"prompt() failed with serialization error: {e}")
                else:
                    raise

            # Verify that the create call was made
            assert mock_client.chat.completions.create.called

            # Get the actual call arguments
            call_kwargs = mock_client.chat.completions.create.call_args[1]

            # Verify tools parameter is serializable
            if "tools" in call_kwargs:
                try:
                    json.dumps(call_kwargs["tools"])
                except TypeError as e:
                    pytest.fail(f"Tools parameter is not JSON serializable: {e}")

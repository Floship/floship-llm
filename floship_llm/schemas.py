from typing import Any, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ThinkingModel(BaseModel):
    # A base model for representing a thought process or reasoning for LLM.
    thinking: str = Field(
        ...,
        title="Thought Process",
        description="A string representing the thought process, chain of thought, or reasoning behind the action taken.",
    )


class Suggestion(ThinkingModel):
    file_path: str = Field(
        description="The path to the file where the suggestion is made."
    )
    line: int = Field(
        description="The line number in the file where the suggestion is made."
    )
    suggestion: str = Field(
        description="The suggested code change in a markdown code block using the ```suggestion marker."
    )
    severity: int = Field(
        description="The severity of the suggestion, 0 - low, 10 - high."
    )
    type: str = Field(
        description="The type of suggestion, one of the following: 'bug', 'neatpick', 'text_change', 'refactor', 'performance', 'security'."
    )
    reason: str = Field(
        description="The reason for the suggestion, a markdown text explaining why the change is needed."
    )


class SuggestionsResponse(ThinkingModel):
    suggestions: List[Suggestion] = Field(
        description="A list of code change suggestions."
    )


class Labels(ThinkingModel):
    # A model for representing labels for a jira ticket.
    labels: List[str] = Field(
        ...,
        title="Labels",
        description="A list of unique labels to be added to the jira ticket. Not more than 5 labels.",
    )


class ToolParameter(BaseModel):
    """Represents a parameter for a tool function."""

    name: str = Field(description="The name of the parameter.")
    type: str = Field(
        description="The type of the parameter (e.g., 'string', 'integer', 'boolean', 'array', 'object')."
    )
    description: Optional[str] = Field(
        default=None, description="Description of what this parameter does."
    )
    required: bool = Field(
        default=False, description="Whether this parameter is required."
    )
    enum: Optional[List[str]] = Field(
        default=None, description="List of allowed values for this parameter."
    )
    default: Optional[Any] = Field(
        default=None, description="Default value for this parameter."
    )


class ToolFunction(BaseModel):
    """Represents a tool/function that can be called by the LLM."""

    name: str = Field(description="The name of the function.")
    description: str = Field(description="A description of what the function does.")
    parameters: List[ToolParameter] = Field(
        default=[], description="List of parameters this function accepts."
    )
    function: Optional[Callable] = Field(
        default=None, description="The actual Python function to execute."
    )

    model_config = {"arbitrary_types_allowed": True}  # Allow Callable type

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert tool to OpenAI function calling format."""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description or "",
            }
            if param.enum:
                properties[param.name]["enum"] = param.enum
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class ToolCall(BaseModel):
    """Represents a tool call made by the LLM."""

    id: str = Field(description="Unique identifier for this tool call.")
    name: str = Field(description="Name of the tool/function being called.")
    arguments: Dict[str, Any] = Field(description="Arguments passed to the tool.")


class ToolResult(BaseModel):
    """Represents the result of a tool execution."""

    tool_call_id: str = Field(
        description="ID of the tool call this result corresponds to."
    )
    name: str = Field(description="Name of the tool that was executed.")
    content: str = Field(
        description="String representation of the tool execution result."
    )
    success: bool = Field(
        default=True, description="Whether the tool execution was successful."
    )
    error: Optional[str] = Field(
        default=None, description="Error message if execution failed."
    )

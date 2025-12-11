"""Tool management for LLM function calling."""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional

from .schemas import ToolCall, ToolFunction, ToolParameter, ToolResult

logger = logging.getLogger(__name__)

# Tool names must follow the OpenAI/Bedrock regex constraints
TOOL_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")


class ToolManager:
    """Manages tool registration, validation, and execution."""

    def __init__(self) -> None:
        """Initialize tool manager."""
        self.tools: Dict[str, ToolFunction] = {}

    def add_tool(self, tool: ToolFunction) -> None:
        """
        Register a tool function.

        Args:
            tool: The tool function to register

        Raises:
            ValueError: If tool with same name already exists
        """
        if not isinstance(tool, ToolFunction):
            raise ValueError("tool must be an instance of ToolFunction")

        if not TOOL_NAME_PATTERN.match(tool.name):
            raise ValueError(
                f"Invalid tool name '{tool.name}'. Tool names must match pattern "
                "[a-zA-Z0-9_-]+ to satisfy API requirements."
            )

        if tool.name in self.tools:
            return

        self.tools[tool.name] = tool
        logger.debug(f"Tool '{tool.name}' registered")

    def remove_tool(self, name: str) -> None:
        """
        Remove a registered tool.

        Args:
            name: Name of the tool to remove
        """
        if name not in self.tools:
            logger.warning(f"Tool '{name}' not found")
            return

        del self.tools[name]
        logger.debug(f"Tool '{name}' removed")

    def add_tool_from_function(
        self,
        func: Callable[..., object],
        name: Optional[str] = None,
        description: Optional[str] = None,
        parameters: Optional[List[ToolParameter]] = None,
    ) -> None:
        """
        Register a Python function as a tool.

        Args:
            func: The Python function to register
            name: Tool name (defaults to function name)
            description: Tool description (defaults to function docstring)
            parameters: List of ToolParameter objects (will attempt to infer if not provided)

        Raises:
            ValueError: If function is invalid or tool already exists
        """
        from .schemas import ToolParameter

        if not callable(func):
            raise ValueError("func must be callable")

        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or f"Execute {tool_name}"

        # Build parameters list if not provided
        if parameters is None:
            import inspect

            sig = inspect.signature(func)
            parameters = []

            for param_name, param in sig.parameters.items():
                if param_name != "self":
                    tool_param = ToolParameter(
                        name=param_name,
                        type="string",
                        description=f"Parameter {param_name}",
                        required=param.default == inspect.Parameter.empty,
                    )
                    parameters.append(tool_param)

        tool = ToolFunction(
            name=tool_name,
            description=tool_description,
            parameters=parameters,
            function=func,
        )

        self.add_tool(tool)

    def list_tools(self) -> List[str]:
        """
        Get list of registered tool names.

        Returns:
            List of tool names
        """
        return list(self.tools.keys())

    def get_tool(self, name: str) -> Optional[ToolFunction]:
        """
        Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool function or None if not found
        """
        return self.tools.get(name)

    def has_tool(self, name: str) -> bool:
        """
        Check if a tool exists.

        Args:
            name: Tool name

        Returns:
            True if tool exists
        """
        return name in self.tools

    def clear_tools(self) -> None:
        """Remove all registered tools."""
        self.tools.clear()
        logger.debug("All tools cleared")

    def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        """
        Execute a tool call.

        Args:
            tool_call: The tool call to execute

        Returns:
            Tool execution result

        Raises:
            ValueError: If tool not found or execution fails
        """
        if not isinstance(tool_call, ToolCall):
            raise ValueError("tool_call must be an instance of ToolCall")

        tool = self.get_tool(tool_call.name)
        if not tool:
            available_tools = list(self.tools.keys())
            # Create a concise, LLM-friendly error message
            # Keep it short to avoid confusing the API
            if available_tools:
                # Limit to first 10 tools to avoid huge messages
                tools_preview = available_tools[:10]
                tools_str = ", ".join(tools_preview)
                if len(available_tools) > 10:
                    tools_str += f" (+{len(available_tools) - 10} more)"
                error_msg = (
                    f"Error: Tool '{tool_call.name}' is not available. "
                    f"Please use one of: {tools_str}"
                )
            else:
                error_msg = (
                    f"Error: Tool '{tool_call.name}' is not available. "
                    "No tools are currently registered."
                )
            logger.warning(
                f"Tool not found: '{tool_call.name}'. Available: {available_tools}"
            )
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=error_msg,
                success=False,
                error=f"Tool not found: {tool_call.name}",
            )

        if not tool.function:
            error_msg = "No executable function defined"
            logger.error(f"Tool '{tool_call.name}': {error_msg}")
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=error_msg,
                success=False,
                error=error_msg,
            )

        try:
            # Parse arguments
            if isinstance(tool_call.arguments, str):
                args = json.loads(tool_call.arguments)
            else:
                args = tool_call.arguments or {}

            # Execute tool
            start_time = time.time()
            logger.warning(f"Executing tool '{tool_call.name}'")
            logger.debug(f"Tool args: {args}")

            result = tool.function(**args)

            execution_time = time.time() - start_time
            logger.warning(
                f"Tool '{tool_call.name}' executed successfully in "
                f"{execution_time:.2f}s"
            )

            # Validate and sanitize tool result content
            if result is None:
                content = "Tool executed successfully (no return value)"
                logger.warning(
                    f"Tool {tool_call.name} returned None, using default content"
                )
            elif str(result).strip() == "":
                content = "Tool executed successfully (empty result)"
                logger.warning(
                    f"Tool {tool_call.name} returned empty result, using default content"
                )
            elif str(result).lower() in ["none", "null"]:
                content = "Tool executed successfully (null result)"
                logger.warning(
                    f"Tool {tool_call.name} returned null-like result, using default content"
                )
            else:
                content = str(result).strip()
                # Ensure content is not just whitespace
                if not content:
                    content = "Tool executed successfully (whitespace result)"
                    logger.warning(
                        f"Tool {tool_call.name} returned only whitespace, using default content"
                    )

            # Convert result to JSON if it's a dict or list
            if isinstance(result, (dict, list)) and content not in [
                "Tool executed successfully (no return value)",
                "Tool executed successfully (empty result)",
                "Tool executed successfully (null result)",
                "Tool executed successfully (whitespace result)",
            ]:
                content = json.dumps(result, indent=2)

            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=content,
                success=True,
            )

        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse tool arguments: {e!s}"
            logger.error(f"Tool '{tool_call.name}': {error_msg}")
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=error_msg,
                success=False,
                error=error_msg,
            )

        except Exception as e:
            error_msg = str(e)
            logger.error(
                f"Tool '{tool_call.name}': Tool execution failed: {error_msg}",
                exc_info=True,
            )
            return ToolResult(
                tool_call_id=tool_call.id,
                name=tool_call.name,
                content=error_msg,
                success=False,
                error=error_msg,
            )

    def get_tools_schema(self) -> List[Dict[str, Any]]:
        """
        Get OpenAI tools schema for all registered tools.

        Returns:
            List of tool schemas in OpenAI format (JSON-serializable)
        """
        schemas: List[Dict[str, Any]] = []

        for tool in self.tools.values():
            if not TOOL_NAME_PATTERN.match(tool.name):
                logger.warning(
                    "Skipping tool '%s' because its name is not API-safe "
                    "(must match [a-zA-Z0-9_-]+).",
                    tool.name,
                )
                continue

            schemas.append(tool.to_openai_format())

        return schemas

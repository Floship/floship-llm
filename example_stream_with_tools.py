#!/usr/bin/env python3
"""
Example: Streaming final response after tool execution.

This demonstrates the new stream_final_response parameter that allows
you to get real-time streaming of the LLM's final answer, even when
tools/functions are used during the conversation.
"""

import os
import sys
from datetime import datetime

from floship_llm import LLM, ToolFunction, ToolParameter


def setup_environment():
    """Set up environment variables for the examples."""
    # Check if environment variables are already set
    required_vars = ["INFERENCE_URL", "INFERENCE_MODEL_ID", "INFERENCE_KEY"]
    missing_vars = [var for var in required_vars if var not in os.environ]

    if missing_vars:
        print("❌ Missing environment variables:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\nPlease set them:")
        print('  export INFERENCE_URL="https://us.inference.heroku.com"')
        print('  export INFERENCE_MODEL_ID="claude-4-sonnet"')
        print('  export INFERENCE_KEY="your-api-key"')
        print("\nOr use Heroku CLI:")
        print(
            "  eval $(heroku config -a $APP_NAME --shell | grep '^INFERENCE_' | sed 's/^/export /')"
        )
        sys.exit(1)


def get_current_time():
    """Get the current time."""
    return datetime.now().strftime("%I:%M %p")


def get_current_date():
    """Get the current date."""
    return datetime.now().strftime("%B %d, %Y")


def calculate(expression: str):
    """Evaluate a mathematical expression."""
    try:
        # Safe eval for basic math
        result = eval(
            expression, {"__builtins__": {}}, {"abs": abs, "round": round, "pow": pow}
        )
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


def example_1_basic_streaming_with_tools():
    """Example 1: Stream final response after using tools."""
    print("=== Example 1: Streaming After Tools ===\n")

    llm = LLM(enable_tools=True, continuous=False)

    # Add tools
    llm.add_tool(
        ToolFunction(
            name="get_current_time",
            description="Get the current time",
            parameters=[],
            function=get_current_time,
        )
    )

    llm.add_tool(
        ToolFunction(
            name="calculate",
            description="Calculate a mathematical expression",
            parameters=[
                ToolParameter(
                    name="expression",
                    type="string",
                    description="Math expression to evaluate",
                )
            ],
            function=calculate,
        )
    )

    print("Question: What time is it? Also calculate 15 * 8 and give me a summary.\n")
    print("Response: ", end="", flush=True)

    # Use stream_final_response=True to stream the final answer
    result = llm.prompt(
        "What time is it? Also calculate 15 * 8 and give me a summary.",
        stream_final_response=True,
    )

    # Result can be either a string (no tools used) or generator (tools used)
    if isinstance(result, str):
        # No tools were used
        print(result)
    else:
        # Tools were used, stream the final response
        for chunk in result:
            print(chunk, end="", flush=True)
    print("\n")


def example_2_multiple_tool_calls_with_streaming():
    """Example 2: Stream response after multiple sequential tool calls."""
    print("\n=== Example 2: Multiple Tools + Streaming ===\n")

    llm = LLM(enable_tools=True, continuous=False)

    # Add tools
    llm.add_tool(
        ToolFunction(
            name="get_current_time",
            description="Get the current time",
            parameters=[],
            function=get_current_time,
        )
    )

    llm.add_tool(
        ToolFunction(
            name="get_current_date",
            description="Get the current date",
            parameters=[],
            function=get_current_date,
        )
    )

    llm.add_tool(
        ToolFunction(
            name="calculate",
            description="Calculate a mathematical expression",
            parameters=[
                ToolParameter(
                    name="expression",
                    type="string",
                    description="Math expression to evaluate",
                )
            ],
            function=calculate,
        )
    )

    prompt = (
        "Tell me the current date, time, and calculate 42 * 365. "
        "Then write a short paragraph about time management."
    )

    print(f"Question: {prompt}\n")
    print("Response: ", end="", flush=True)

    result = llm.prompt(prompt, stream_final_response=True)

    if hasattr(result, "__iter__") and not isinstance(result, str):
        # Streaming response
        for chunk in result:
            print(chunk, end="", flush=True)
    else:
        # Regular response
        print(result)
    print("\n")


def example_3_without_streaming():
    """Example 3: Compare with non-streaming (original behavior)."""
    print("\n=== Example 3: Without Streaming (Original) ===\n")

    llm = LLM(enable_tools=True, continuous=False)

    llm.add_tool(
        ToolFunction(
            name="calculate",
            description="Calculate a mathematical expression",
            parameters=[
                ToolParameter(
                    name="expression",
                    type="string",
                    description="Math expression to evaluate",
                )
            ],
            function=calculate,
        )
    )

    print("Question: Calculate 999 + 111 and explain the result.\n")
    print("(Waiting for complete response...)\n")

    # Without stream_final_response, get complete response at once
    result = llm.prompt("Calculate 999 + 111 and explain the result.")

    print(f"Response: {result}\n")


def example_4_conversation_with_streaming():
    """Example 4: Multi-turn conversation with streaming."""
    print("\n=== Example 4: Conversation with Streaming ===\n")

    llm = LLM(enable_tools=True, continuous=True)

    llm.add_tool(
        ToolFunction(
            name="calculate",
            description="Calculate a mathematical expression",
            parameters=[
                ToolParameter(
                    name="expression",
                    type="string",
                    description="Math expression to evaluate",
                )
            ],
            function=calculate,
        )
    )

    # First question
    print("Q1: What is 25 * 4?\n")
    print("A1: ", end="", flush=True)

    result = llm.prompt("What is 25 * 4?", stream_final_response=True)
    if hasattr(result, "__iter__") and not isinstance(result, str):
        for chunk in result:
            print(chunk, end="", flush=True)
    else:
        print(result)
    print("\n")

    # Follow-up question (maintains context)
    print("\nQ2: Now multiply that by 3.\n")
    print("A2: ", end="", flush=True)

    result = llm.prompt("Now multiply that by 3.", stream_final_response=True)
    if hasattr(result, "__iter__") and not isinstance(result, str):
        for chunk in result:
            print(chunk, end="", flush=True)
    else:
        print(result)
    print("\n")


def main():
    """Run all examples."""
    print("FloShip LLM - Streaming with Tools Examples")
    print("=" * 60)
    print()

    setup_environment()

    try:
        example_1_basic_streaming_with_tools()
        example_2_multiple_tool_calls_with_streaming()
        example_3_without_streaming()
        example_4_conversation_with_streaming()

        print("\n" + "=" * 60)
        print("✅ All examples completed!")
        print("\nKey Benefits:")
        print("  • Tools execute normally")
        print("  • Final response streams in real-time")
        print("  • Better UX for long responses")
        print("  • Same API, just add stream_final_response=True")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Examples demonstrating streaming support with FloShip LLM.

This script shows how to:
1. Enable streaming mode
2. Stream responses in real-time
3. Handle streaming without tool support
4. Combine streaming with conversation history
"""

import os
import sys

from floship_llm import LLM


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


def example_1_basic_streaming():
    """Example 1: Basic streaming response."""
    print("=== Example 1: Basic Streaming ===\n")

    llm = LLM(type="completion", stream=True, enable_tools=False, continuous=False)

    print("Question: Write a haiku about programming\n")
    print("Response: ", end="", flush=True)

    for chunk in llm.prompt_stream("Write a haiku about programming"):
        print(chunk, end="", flush=True)

    print("\n")


def example_2_streaming_with_conversation():
    """Example 2: Streaming with conversation history."""
    print("\n=== Example 2: Streaming with Conversation History ===\n")

    llm = LLM(
        type="completion",
        stream=True,
        enable_tools=False,
        continuous=True,
        system="You are a helpful assistant that provides concise answers.",
    )

    # First question
    print("Question 1: What is Python?\n")
    print("Response: ", end="", flush=True)
    for chunk in llm.prompt_stream("What is Python?"):
        print(chunk, end="", flush=True)
    print("\n")

    # Follow-up question (maintains context)
    print("\nQuestion 2: What are its main uses?\n")
    print("Response: ", end="", flush=True)
    for chunk in llm.prompt_stream("What are its main uses?"):
        print(chunk, end="", flush=True)
    print("\n")


def example_3_streaming_long_content():
    """Example 3: Streaming longer content."""
    print("\n=== Example 3: Streaming Longer Content ===\n")

    llm = LLM(
        type="completion",
        stream=True,
        enable_tools=False,
        continuous=False,
        temperature=0.7,
    )

    prompt = "Write a short story (3-4 paragraphs) about a robot learning to paint."
    print(f"Question: {prompt}\n")
    print("Response:\n", flush=True)

    for chunk in llm.prompt_stream(prompt):
        print(chunk, end="", flush=True)

    print("\n")


def example_4_streaming_vs_non_streaming():
    """Example 4: Compare streaming vs non-streaming."""
    print("\n=== Example 4: Streaming vs Non-Streaming Comparison ===\n")

    prompt = "Count from 1 to 5 with a brief description of each number."

    # Streaming mode
    print("--- Streaming Mode ---")
    print("(Watch it appear gradually)\n")
    llm_stream = LLM(stream=True, enable_tools=False, continuous=False)

    print("Response: ", end="", flush=True)
    for chunk in llm_stream.prompt_stream(prompt):
        print(chunk, end="", flush=True)
    print("\n")

    # Non-streaming mode
    print("\n--- Non-Streaming Mode ---")
    print("(Waits for complete response)\n")
    llm_normal = LLM(stream=False, enable_tools=False, continuous=False)

    print("Response: ", end="", flush=True)
    response = llm_normal.prompt(prompt)
    print(response)
    print()


def example_5_error_with_tools():
    """Example 5: Show that streaming doesn't work with tools."""
    print("\n=== Example 5: Streaming Error with Tools ===\n")

    llm = LLM(stream=True, enable_tools=True)  # Tools are enabled

    print("Attempting to stream with tools enabled...")
    try:
        for chunk in llm.prompt_stream("What is 2 + 2?"):
            print(chunk, end="", flush=True)
        print("\n❌ Should have raised an error!")
    except ValueError as e:
        print(f"✓ Correctly raised error: {e}\n")


def main():
    """Run all examples."""
    print("FloShip LLM - Streaming Examples")
    print("=" * 50)

    setup_environment()

    example_1_basic_streaming()
    example_2_streaming_with_conversation()
    example_3_streaming_long_content()
    example_4_streaming_vs_non_streaming()
    example_5_error_with_tools()

    print("\n" + "=" * 50)
    print("✅ All streaming examples completed!")


if __name__ == "__main__":
    main()

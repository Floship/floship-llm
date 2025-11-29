"""
Floship LLM Library - Quick Start Examples

This file demonstrates how to use the floship-llm library in your projects.
"""

import os
from typing import List

from pydantic import BaseModel, Field

from floship_llm import LLM, Labels, ThinkingModel, ToolFunction, ToolParameter


# Example 1: Basic Usage - Simple Question/Answer
def example_basic():
    """Simple single-turn conversation"""
    print("\n=== Example 1: Basic Usage ===")

    llm = LLM(
        type="completion",
        temperature=0.7,
        continuous=False,  # Don't maintain conversation history
    )

    response = llm.prompt("What is the capital of France?")
    print(f"Response: {response}")


# Example 2: Multi-turn Conversation
def example_conversation():
    """Multi-turn conversation with history"""
    print("\n=== Example 2: Multi-turn Conversation ===")

    llm = LLM(
        type="completion",
        continuous=True,  # Maintain conversation history
        system="You are a helpful Python programming assistant.",
    )

    response1 = llm.prompt("What is a Python decorator?")
    print("Q1: What is a Python decorator?")
    print(f"A1: {response1[:100]}...")

    response2 = llm.prompt("Can you show me an example?")
    print("\nQ2: Can you show me an example?")
    print(f"A2: {response2[:100]}...")

    # Reset for new conversation
    llm.reset()


# Example 3: Structured Output with Labels
def example_structured_labels():
    """Generate structured output using the Labels schema"""
    print("\n=== Example 3: Structured Output - Labels ===")

    llm = LLM(
        type="completion",
        response_format=Labels,  # Pydantic model for structured output
        continuous=False,
    )

    prompt = """
    Generate appropriate labels for a Jira ticket with this description:
    "The user login page is slow and sometimes times out. Need to optimize
    database queries and add caching."
    """

    response = llm.prompt(prompt)
    print(f"Generated labels: {response.labels}")
    print(f"Thinking process: {response.thinking[:100]}...")


# Example 4: Custom Schema
def example_custom_schema():
    """Using a custom Pydantic schema for structured output"""
    print("\n=== Example 4: Custom Schema ===")

    # Define custom schema
    class CodeReview(BaseModel):
        summary: str = Field(description="Brief summary of the code")
        issues: List[str] = Field(description="List of potential issues")
        suggestions: List[str] = Field(description="List of improvements")
        rating: int = Field(description="Code quality rating 1-10")

    llm = LLM(type="completion", response_format=CodeReview, temperature=0.3)

    code = """
    def calculate_total(items):
        total = 0
        for item in items:
            total = total + item['price'] * item['quantity']
        return total
    """

    response = llm.prompt(f"Review this Python code:\n{code}")
    print(f"Summary: {response.summary}")
    print(f"Issues: {response.issues}")
    print(f"Suggestions: {response.suggestions}")
    print(f"Rating: {response.rating}/10")


# Example 5: Using JSON Utilities
def example_json_utils():
    """Demonstrate JSON parsing utilities"""
    print("\n=== Example 5: JSON Utilities ===")

    from floship_llm import lm_json_utils

    # Messy JSON-like text from LLM
    messy_text = """
    Here's the data you requested:
    {name: 'John Doe', age: 30, skills: ['Python', 'JavaScript', 'SQL']}
    Hope this helps!
    """

    # Extract and fix JSON
    cleaned = lm_json_utils.extract_and_fix_json(messy_text)
    print(f"Extracted JSON: {cleaned}")

    # Get strict JSON string
    strict = lm_json_utils.extract_strict_json(messy_text)
    print(f"Strict JSON: {strict}")


# Example 6: Tool/Function Calling
def example_tools():
    """Demonstrate tool/function calling capabilities"""
    print("\n=== Example 6: Tool/Function Calling ===")

    # Create LLM with tool support enabled
    llm = LLM(type="completion", enable_tools=True, temperature=0.3)

    # Define some useful functions
    def calculate_area(length: float, width: float) -> float:
        """Calculate area of a rectangle."""
        return length * width

    def convert_temperature(temp: float, from_unit: str, to_unit: str) -> float:
        """Convert temperature between Celsius and Fahrenheit."""
        if from_unit.lower() == "celsius" and to_unit.lower() == "fahrenheit":
            return (temp * 9 / 5) + 32
        elif from_unit.lower() == "fahrenheit" and to_unit.lower() == "celsius":
            return (temp - 32) * 5 / 9
        else:
            return temp

    def get_word_count(text: str) -> int:
        """Count words in a text."""
        return len(text.split())

    # Add tools to the LLM
    llm.add_tool_from_function(
        calculate_area, description="Calculate area of rectangle"
    )
    llm.add_tool_from_function(
        convert_temperature, description="Convert temperature units"
    )
    llm.add_tool_from_function(get_word_count, description="Count words in text")

    print(f"Available tools: {llm.list_tools()}")

    # Now the LLM can use these tools automatically
    response = llm.prompt(
        """
        I have a room that is 12.5 feet long and 8.2 feet wide.
        What's the area? Also, if it's 72°F in the room, what's that in Celsius?
        And can you count the words in this sentence?
    """
    )
    print(f"Response: {response}")


# Example 7: Configuration Options
def example_configuration():
    """Demonstrate various configuration options"""
    print("\n=== Example 7: Configuration Options ===")

    llm = LLM(
        type="completion",
        temperature=0.2,  # Lower = more deterministic
        frequency_penalty=0.3,  # Reduce repetition
        presence_penalty=0.3,  # Encourage topic diversity
        max_length=500,  # Max response length
        input_tokens_limit=2000,  # Input token limit
        system="You are a concise technical writer.",
    )

    response = llm.prompt("Explain what a REST API is.")
    print(f"Response length: {len(response)} chars")
    print(f"Response: {response[:200]}...")


# Example 8: Error Handling
def example_error_handling():
    """Demonstrate error handling"""
    print("\n=== Example 8: Error Handling ===")

    try:
        # This will fail if environment variables are not set
        llm = LLM(type="completion")
        response = llm.prompt("Hello!")
        print(f"Success: {response}")
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("\nMake sure to set these environment variables:")
        print("  - INFERENCE_URL")
        print("  - INFERENCE_MODEL_ID")
        print("  - INFERENCE_KEY")


def main():
    """Run all examples"""
    print("Floship LLM Library - Examples")
    print("=" * 50)

    # Check if environment variables are set
    required_vars = ["INFERENCE_URL", "INFERENCE_MODEL_ID", "INFERENCE_KEY"]
    missing_vars = [var for var in required_vars if not os.environ.get(var)]

    if missing_vars:
        print("\n⚠️  Missing environment variables:", missing_vars)
        print("\nPlease set them before running examples:")
        print('  export INFERENCE_URL="https://api.openai.com/v1"')
        print('  export INFERENCE_MODEL_ID="gpt-4"')
        print('  export INFERENCE_KEY="sk-..."')
        return

    # Run examples
    try:
        example_basic()
        example_conversation()
        example_structured_labels()
        example_custom_schema()
        example_json_utils()
        example_tools()
        example_configuration()
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback

        traceback.print_exc()

    print("\n" + "=" * 50)
    print("Examples completed!")


if __name__ == "__main__":
    main()

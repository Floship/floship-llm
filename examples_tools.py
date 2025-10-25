#!/usr/bin/env python3
"""
Examples demonstrating tool/function calling with FloShip LLM.

This script shows how to:
1. Set up tools for the LLM to use
2. Enable tool support
3. Create custom tools from Python functions
4. Handle tool execution in conversations
"""

import json
import os
from datetime import datetime

from floship_llm import LLM, ToolFunction, ToolParameter


def setup_environment():
    """Set up environment variables for the examples."""
    os.environ["INFERENCE_URL"] = "https://api.openai.com/v1"
    os.environ["INFERENCE_MODEL_ID"] = "gpt-4"
    os.environ["INFERENCE_KEY"] = "your-api-key-here"


def example_1_basic_math_tools():
    """Example 1: Basic math tools for the LLM."""
    print("=== Example 1: Basic Math Tools ===")

    # Initialize LLM with tool support enabled
    llm = LLM(enable_tools=True)

    # Define calculator functions
    def add_numbers(a: float, b: float) -> float:
        """Add two numbers together."""
        return a + b

    def multiply_numbers(a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b

    def calculate_percentage(value: float, percentage: float) -> float:
        """Calculate what percentage of a value is."""
        return (value * percentage) / 100

    # Add tools using the simple function wrapper
    llm.add_tool_from_function(add_numbers, description="Add two numbers together")
    llm.add_tool_from_function(multiply_numbers, description="Multiply two numbers")
    llm.add_tool_from_function(
        calculate_percentage, description="Calculate percentage of a value"
    )

    print(f"Available tools: {llm.list_tools()}")

    # Example conversation
    try:
        response = llm.prompt("What is 15% of 250? Also, what is 25 + 37?")
        print(f"LLM Response: {response}")
    except Exception as e:
        print(f"Error: {e}")


def example_2_web_api_tools():
    """Example 2: Web API and data processing tools."""
    print("\n=== Example 2: Web API Tools ===")

    llm = LLM(enable_tools=True)

    def get_current_time(timezone: str = "UTC") -> str:
        """Get the current time in the specified timezone."""
        current_time = datetime.now()
        return (
            f"Current time in {timezone}: {current_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

    def search_data(query: str, category: str = "general") -> str:
        """Search for information in a specific category."""
        # This would typically make an API call
        return f"Search results for '{query}' in category '{category}': [Mock search results]"

    def format_json(data: str) -> str:
        """Format and validate JSON data."""
        try:
            parsed = json.loads(data)
            return json.dumps(parsed, indent=2)
        except json.JSONDecodeError:
            return f"Invalid JSON: {data}"

    # Add tools with detailed parameter definitions
    time_tool = ToolFunction(
        name="get_current_time",
        description="Get the current time in a specified timezone",
        parameters=[
            ToolParameter(
                name="timezone",
                type="string",
                description="Timezone to get time for",
                required=False,
                default="UTC",
                enum=["UTC", "EST", "PST", "GMT"],
            )
        ],
        function=get_current_time,
    )

    search_tool = ToolFunction(
        name="search_data",
        description="Search for information in different categories",
        parameters=[
            ToolParameter(
                name="query", type="string", description="Search query", required=True
            ),
            ToolParameter(
                name="category",
                type="string",
                description="Category to search in",
                required=False,
                default="general",
                enum=["general", "tech", "science", "business"],
            ),
        ],
        function=search_data,
    )

    llm.add_tool(time_tool)
    llm.add_tool(search_tool)
    llm.add_tool_from_function(format_json, description="Format and validate JSON data")

    print(f"Available tools: {llm.list_tools()}")

    # Example conversation
    try:
        response = llm.prompt(
            "What time is it now? Also, can you search for information about 'machine learning' in the tech category?"
        )
        print(f"LLM Response: {response}")
    except Exception as e:
        print(f"Error: {e}")


def example_3_advanced_tool_chain():
    """Example 3: Chaining multiple tools together."""
    print("\n=== Example 3: Advanced Tool Chaining ===")

    llm = LLM(enable_tools=True)

    # Data processing pipeline tools
    def load_csv_data(filename: str) -> str:
        """Load data from a CSV file."""
        # Mock CSV loading
        return json.dumps(
            [
                {"name": "Alice", "age": 30, "salary": 50000},
                {"name": "Bob", "age": 25, "salary": 45000},
                {"name": "Carol", "age": 35, "salary": 60000},
            ]
        )

    def filter_data(json_data: str, field: str, min_value: float) -> str:
        """Filter JSON data by field value."""
        try:
            data = json.loads(json_data)
            filtered = [item for item in data if item.get(field, 0) >= min_value]
            return json.dumps(filtered)
        except:
            return "Error filtering data"

    def calculate_average(json_data: str, field: str) -> float:
        """Calculate average of a numeric field in JSON data."""
        try:
            data = json.loads(json_data)
            values = [item.get(field, 0) for item in data if field in item]
            return sum(values) / len(values) if values else 0
        except:
            return 0

    def generate_report(json_data: str, title: str = "Data Report") -> str:
        """Generate a formatted report from JSON data."""
        try:
            data = json.loads(json_data)
            report = f"# {title}\n\n"
            report += f"Total records: {len(data)}\n\n"
            report += "## Data:\n"
            for item in data:
                report += f"- {item}\n"
            return report
        except:
            return "Error generating report"

    # Add all tools
    llm.add_tool_from_function(load_csv_data, description="Load data from CSV file")
    llm.add_tool_from_function(
        filter_data, description="Filter JSON data by field value"
    )
    llm.add_tool_from_function(
        calculate_average, description="Calculate average of numeric field"
    )
    llm.add_tool_from_function(
        generate_report, description="Generate formatted report from data"
    )

    print(f"Available tools: {llm.list_tools()}")

    # Complex multi-step task
    try:
        response = llm.prompt(
            """
        Please help me analyze employee data:
        1. Load data from 'employees.csv'
        2. Filter for employees with salary >= 50000
        3. Calculate the average salary of filtered employees
        4. Generate a report titled 'High Salary Employees'
        """
        )
        print(f"LLM Response: {response}")
    except Exception as e:
        print(f"Error: {e}")


def example_4_tool_management():
    """Example 4: Dynamic tool management."""
    print("\n=== Example 4: Tool Management ===")

    llm = LLM()  # Start without tools enabled

    def greeting_tool(name: str, language: str = "english") -> str:
        """Generate a greeting in different languages."""
        greetings = {
            "english": f"Hello, {name}!",
            "spanish": f"¡Hola, {name}!",
            "french": f"Bonjour, {name}!",
            "german": f"Hallo, {name}!",
        }
        return greetings.get(language.lower(), f"Hello, {name}!")

    def math_tool(operation: str, a: float, b: float) -> float:
        """Perform basic math operations."""
        operations = {
            "add": a + b,
            "subtract": a - b,
            "multiply": a * b,
            "divide": a / b if b != 0 else float("inf"),
        }
        return operations.get(operation.lower(), 0)

    # Initially no tools
    print(f"Initial tools: {llm.list_tools()}")
    print(f"Tool support enabled: {llm.enable_tools}")

    # Enable tools and add them
    llm.enable_tool_support(True)
    llm.add_tool_from_function(
        greeting_tool, description="Generate multilingual greetings"
    )
    llm.add_tool_from_function(math_tool, description="Perform basic math operations")

    print(f"After adding tools: {llm.list_tools()}")

    # Remove one tool
    llm.remove_tool("math_tool")
    print(f"After removing math tool: {llm.list_tools()}")

    # Clear all tools
    llm.clear_tools()
    print(f"After clearing all tools: {llm.list_tools()}")

    # Disable tool support
    llm.enable_tool_support(False)
    print(f"Tool support enabled: {llm.enable_tools}")


def main():
    """Run all examples."""
    print("FloShip LLM Tool Examples")
    print("=" * 50)

    # Set up environment (you'll need to provide real API credentials)
    setup_environment()

    # Run examples
    try:
        example_1_basic_math_tools()
        example_2_web_api_tools()
        example_3_advanced_tool_chain()
        example_4_tool_management()
    except Exception as e:
        print(f"Example execution error: {e}")
        print(
            "Note: Make sure to set up valid API credentials in environment variables."
        )


if __name__ == "__main__":
    main()

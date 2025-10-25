# Floship LLM Client Library

A reusable Python library for interacting with Heroku Managed Inference and Agents API and other OpenAI-compatible inference endpoints.

## Features

- 🚀 Simple and intuitive API for LLM interactions
- 🎯 **Heroku Inference API** - Optimized for Heroku's managed LLM service
- 🔄 Support for continuous (multi-turn) conversations
- 🛠️ **Tool/Function calling** - Let the LLM execute Python functions
- 📊 Structured output with Pydantic schemas
- 🎯 JSON response parsing and validation
- ⚙️ Configurable parameters (temperature, top_k, top_p, etc.)
- 🧠 **Extended thinking** support for Claude models (Heroku-specific)
- 🔁 Retry mechanism with configurable limits
- 🔌 Compatible with Heroku Inference API and OpenAI-compatible APIs

## Installation

### From PyPI (when published)

```bash
pip install floship-llm
```

### From source

```bash
git clone https://github.com/floship/floship-llm.git
cd floship-llm
pip install -e .
```

### For development

Using [uv](https://docs.astral.sh/uv/) (recommended):

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and development tools
uv sync --dev

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=floship_llm --cov-report=html
```

Or using pip:

```bash
pip install -e ".[dev]"
```

## Quick Start with Heroku Inference API

### Setup Environment

First, set up your Heroku Inference API credentials:

```bash
# Get your credentials from Heroku
export INFERENCE_URL="https://us.inference.heroku.com"
export INFERENCE_MODEL_ID="claude-4-sonnet"
export INFERENCE_KEY="your-heroku-api-key"
```

Or use Heroku CLI to set them automatically:

```bash
eval $(heroku config -a $APP_NAME --shell | grep '^INFERENCE_' | sed 's/^/export /' | tee >(cat >&2))
```

### Basic Usage

```python
from floship_llm import LLM

# Create a client (reads environment variables automatically)
llm = LLM(
    type='completion',
    temperature=0.15,  # Lower for more focused responses
    continuous=False   # Single-turn conversation
)

# Generate a response
response = llm.prompt("What is the capital of France?")
print(response)
```

### Using Extended Thinking (Claude Models)

Extended thinking allows Claude models to spend more time reasoning before responding. This is particularly useful for complex problems:

```python
from floship_llm import LLM

# Enable extended thinking for Claude models
llm = LLM(
    type='completion',
    temperature=0.15,
    extended_thinking={
        "enabled": True,
        "budget_tokens": 1024,        # Tokens allocated for thinking
        "include_reasoning": True      # Include thinking process in response
    }
)

# Ask a complex question
response = llm.prompt(
    "Solve this logic puzzle: If all bloops are razzles and all razzles are lazzles, "
    "are all bloops definitely lazzles?"
)
print(response)
# Response will include the reasoning process if include_reasoning=True
```

Supported Claude models for extended thinking:
- claude-4.5-sonnet
- claude-4.5-haiku
- claude-4-sonnet
- claude-3.7-sonnet

### Advanced Parameters (Heroku-Specific)

```python
from floship_llm import LLM

# Fine-tune generation behavior
llm = LLM(
    type='completion',
    temperature=0.7,              # Creativity (0.0-1.0)
    max_completion_tokens=2000,   # Max tokens to generate
    top_p=0.9,                    # Nucleus sampling threshold
    top_k=50,                     # Top-k sampling (limits vocabulary)
    continuous=True
)

response = llm.prompt("Write a creative story about AI")
```

**Note on Deprecated Parameters:**
- `frequency_penalty` and `presence_penalty` are **not supported** by Heroku Inference API
- These parameters are ignored if sent (won't cause errors)
- Use `temperature`, `top_p`, and `top_k` instead to control generation behavior

### Continuous Conversations

```python
from floship_llm import LLM

# Enable continuous mode for multi-turn conversations
llm = LLM(
    type='completion',
    continuous=True,
    system="You are a helpful assistant."
)

# First message
response1 = llm.prompt("Tell me about Python.")
print(response1)

# Follow-up (maintains context)
response2 = llm.prompt("What are its main advantages?")
print(response2)

# Reset conversation history
llm.reset()
```

### Structured Output with Pydantic

```python
from floship_llm import LLM, Labels

# Define your response format
llm = LLM(
    type='completion',
    response_format=Labels,  # Pydantic model
    continuous=False
)

response = llm.prompt("Generate 3 labels for a ticket about database optimization")
# Returns a Labels object with validated structure
print(response.labels)
```

### Tool/Function Calling

Enable the LLM to execute Python functions during conversations:

```python
from floship_llm import LLM, ToolFunction, ToolParameter

# Create LLM with tool support
llm = LLM(enable_tools=True)

# Simple function wrapping
def calculate_tax(amount: float, rate: float) -> float:
    """Calculate tax on an amount."""
    return amount * rate / 100

llm.add_tool_from_function(calculate_tax, description="Calculate tax on amount")

# The LLM can now call this function automatically
response = llm.prompt("What's the tax on $1000 at 8.5% rate?")
print(response)  # LLM will call calculate_tax(1000, 8.5) and include result
```

#### Advanced Tool Definition

```python
# Define tools with detailed parameters
weather_tool = ToolFunction(
    name="get_weather",
    description="Get current weather for a location",
    parameters=[
        ToolParameter(
            name="location",
            type="string",
            description="City name",
            required=True
        ),
        ToolParameter(
            name="units",
            type="string",
            enum=["celsius", "fahrenheit"],
            default="celsius"
        )
    ],
    function=get_weather_data  # Your function
)

llm.add_tool(weather_tool)
response = llm.prompt("What's the weather in Tokyo?")
```

#### Tool Management

```python
# List available tools
print(llm.list_tools())

# Remove a tool
llm.remove_tool("get_weather")

# Clear all tools
llm.clear_tools()

# Enable/disable tool support
llm.enable_tool_support(True)  # or False
```

### Configuration

The library reads configuration from environment variables:

- `INFERENCE_URL` (required): Your Heroku Inference API endpoint
  - Default: `https://us.inference.heroku.com`
- `INFERENCE_MODEL_ID` (required): The LLM model to use
  - Example: `claude-4-sonnet`, `gpt-4`, `meta-llama-4-405b-instruct`
- `INFERENCE_KEY` (required): Your API authentication key

**Heroku Setup:**
```bash
export INFERENCE_URL="https://us.inference.heroku.com"
export INFERENCE_MODEL_ID="claude-4-sonnet"
export INFERENCE_KEY="your-heroku-api-key"
```

**Alternative Endpoint (OpenAI-compatible):**
```bash
export INFERENCE_URL="https://api.openai.com/v1/chat/completions"
export INFERENCE_MODEL_ID="gpt-4"
export INFERENCE_KEY="your-openai-api-key"
```

### LLM Class Parameters

```python
LLM(
    type: str = "completion",
    temperature: float = 0.15,
    max_completion_tokens: Optional[int] = None,  # Heroku: max tokens to generate
    top_p: Optional[float] = None,                # Nucleus sampling (0.0-1.0)
    top_k: Optional[int] = None,                  # Top-k sampling
    extended_thinking: Optional[dict] = None,     # Claude extended thinking config
    continuous: bool = False,
    retry_limit: int = 5,
    allow_ignored_params: bool = False,  # Allow deprecated parameters
    **kwargs
)
```

**Key Parameters:**
- `temperature`: Controls randomness (0.0 = deterministic, 1.0 = creative). Default: 0.15
- `max_completion_tokens`: Maximum tokens to generate (Heroku-specific, replaces `max_tokens`)
- `top_p`: Nucleus sampling threshold (0.0-1.0)
- `top_k`: Limits vocabulary to top K tokens
- `extended_thinking`: Dict with `enabled`, `budget_tokens`, `include_reasoning` (Claude only)
- `continuous`: Enable multi-turn conversations with history
- `allow_ignored_params`: If True, allows deprecated parameters without warnings

**⚠️ Deprecated Parameters (Not Supported by Heroku):**
- `frequency_penalty`: Use `temperature` and `top_p` instead
- `presence_penalty`: Use `temperature` and `top_k` instead

These parameters are silently ignored by Heroku Inference API.

## Advanced Features

### Custom System Prompts

```python
llm = LLM(
    type='completion',
    system="You are an expert Python developer."
)
```

### Temperature and Penalties

```python
llm = LLM(
    type='completion',
    temperature=0.15,  # Lower = more deterministic
    frequency_penalty=0.2,
    presence_penalty=0.2
)
```

### Token Limits and Retry Control

```python
llm = LLM(
    type='completion',
    input_tokens_limit=40_000,  # Input token limit
    max_length=100_000,  # Maximum response length
    max_retry=3  # Maximum retry attempts (default: 3)
)
```

### Multi-Step Tool Workflows

The LLM can automatically chain multiple tools together:

```python
# Define data processing tools
llm.add_tool_from_function(load_csv_data)
llm.add_tool_from_function(filter_data)
llm.add_tool_from_function(calculate_statistics)
llm.add_tool_from_function(generate_chart)

# LLM will call tools in sequence as needed
response = llm.prompt("""
    Load sales data from 'Q4_sales.csv',
    filter for amounts > $1000,
    calculate average and total,
    then create a bar chart
""")
```

## Included Schemas

The library includes several pre-built Pydantic schemas:

### Response Schemas
- `ThinkingModel`: Base model with reasoning/thought process
- `Suggestion`: Code review suggestions
- `SuggestionsResponse`: Collection of suggestions
- `Labels`: Label generation for categorization

### Tool Schemas
- `ToolParameter`: Define function parameters with validation
- `ToolFunction`: Complete tool definitions with OpenAI compatibility
- `ToolCall`: Represents tool invocations from the LLM
- `ToolResult`: Captures tool execution results and status

## Utilities

### JSON Utilities

```python
from floship_llm import lm_json_utils

# Extract and fix JSON from messy text
text = "Here's the data: {name: 'John', age: 30}"
cleaned = lm_json_utils.extract_and_fix_json(text)

# Get strict JSON string
json_str = lm_json_utils.extract_strict_json(text)
```

## API Reference

### LLM Class

**Constructor Parameters:**

*Heroku-Specific Parameters:*
- `type` (str): 'completion' or 'embedding' (default: 'completion')
- `model` (str): Model identifier (defaults to INFERENCE_MODEL_ID env var)
- `temperature` (float): Sampling temperature, 0.0-1.0 (default: 0.15)
- `max_completion_tokens` (int): Maximum tokens to generate (Heroku-specific)
- `top_p` (float): Nucleus sampling threshold, 0.0-1.0
- `top_k` (int): Top-k sampling, limits vocabulary
- `extended_thinking` (dict): Extended thinking config for Claude models
  ```python
  {
      "enabled": True,
      "budget_tokens": 1024,
      "include_reasoning": True
  }
  ```
- `allow_ignored_params` (bool): Allow deprecated parameters (default: False)

*General Parameters:*
- `response_format` (BaseModel): Pydantic model for structured output
- `continuous` (bool): Enable conversation history (default: True)
- `messages` (list): Initial conversation history
- `max_length` (int): Maximum response length (default: 100,000)
- `input_tokens_limit` (int): Input token limit (default: 40,000)
- `retry_limit` (int): Maximum retry attempts (default: 5)
- `enable_tools` (bool): Enable tool/function calling (default: False)
- `system` (str): System prompt

*⚠️ Deprecated Parameters (Not Supported by Heroku):*
- ~~`frequency_penalty`~~: Not supported by Heroku Inference API
- ~~`presence_penalty`~~: Not supported by Heroku Inference API
- ~~`max_retry`~~: Renamed to `retry_limit`

Use `temperature`, `top_p`, and `top_k` to control generation behavior instead.

**Methods:**

*Core Methods:*
- `prompt(prompt, system=None, retry=False)`: Generate a response
- `add_message(role, content)`: Add a message to conversation history
- `reset()`: Clear conversation history
- `embed(text)`: Generate embeddings (future feature)

*Tool Management:*
- `add_tool(tool)`: Add a ToolFunction to available tools
- `add_tool_from_function(func, name=None, description=None)`: Wrap Python function as tool
- `remove_tool(name)`: Remove a tool by name
- `list_tools()`: Get list of available tool names
- `clear_tools()`: Remove all tools
- `enable_tool_support(enabled=True)`: Enable/disable tool support
- `execute_tool(tool_call)`: Execute a tool call and return result

**Properties:**
- `supports_parallel_requests`: Check if model supports parallel requests
- `supports_frequency_penalty`: Check if model supports frequency penalty
- `supports_presence_penalty`: Check if model supports presence penalty
- `require_response_format`: Check if response format is required

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=floship_llm --cov-report=html

# Run specific test files
pytest tests/test_tools.py -v  # Tool functionality tests
pytest tests/test_client.py -v  # Core client tests
```

**Test Coverage:** 166 tests covering all functionality including:
- Core LLM client operations
- Tool/function calling workflows
- Schema validation and JSON processing
- Error handling and edge cases
- 95% overall code coverage

### Code Formatting

```bash
black floship_llm/
```

### Type Checking

```bash
mypy floship_llm/
```

## Real-World Tool Examples

The examples/ directory contains comprehensive examples of tool usage:

### Basic Calculator Tools
```python
# examples_tools.py - Example 1
def add_numbers(a: float, b: float) -> float:
    return a + b

llm.add_tool_from_function(add_numbers)
response = llm.prompt("What is 15% of 250? Also, what is 25 + 37?")
```

### Web API Integration
```python
# examples_tools.py - Example 2
def get_current_time(timezone: str = "UTC") -> str:
    return f"Current time in {timezone}: {datetime.now()}"

def search_data(query: str, category: str = "general") -> str:
    return f"Search results for '{query}' in {category}"

llm.add_tool_from_function(get_current_time)
llm.add_tool_from_function(search_data)
```

### Data Processing Pipeline
```python
# examples_tools.py - Example 3
llm.add_tool_from_function(load_csv_data)
llm.add_tool_from_function(filter_data)
llm.add_tool_from_function(calculate_average)
llm.add_tool_from_function(generate_report)

# LLM chains tools automatically
response = llm.prompt("Load employee data, filter by salary >= 50k, and generate report")
```

Run the examples:
```bash
python examples_tools.py
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Support

For questions or issues, please open an issue on GitHub or contact dev@floship.com

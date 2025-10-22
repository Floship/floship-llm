# Floship LLM Client Library

A reusable Python library for interacting with OpenAI-compatible inference endpoints.

## Features

- 🚀 Simple and intuitive API for LLM interactions
- 🔄 Support for continuous (multi-turn) conversations
- 📊 Structured output with Pydantic schemas
- 🎯 JSON response parsing and validation
- ⚙️ Configurable parameters (temperature, penalties, etc.)
- 🔌 Compatible with any OpenAI-compatible API

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

## Quick Start

### Basic Usage

```python
from floship_llm import LLM

# Set environment variables
# INFERENCE_URL - Your LLM API endpoint
# INFERENCE_MODEL_ID - Model identifier
# INFERENCE_KEY - API key

# Create a client
llm = LLM(
    type='completion',
    temperature=0.7,
    continuous=False  # Single-turn conversation
)

# Generate a response
response = llm.prompt("What is the capital of France?")
print(response)
```

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

## Configuration

The library requires the following environment variables:

- `INFERENCE_URL`: The base URL of your LLM API endpoint
- `INFERENCE_MODEL_ID`: The model identifier to use
- `INFERENCE_KEY`: Your API authentication key
- `INFERENCE_SUPPORTS_PARALLEL_REQUESTS` (optional): Set to "False" if the API doesn't support parallel requests (default: "True")

Example:

```bash
export INFERENCE_URL="https://api.openai.com/v1"
export INFERENCE_MODEL_ID="gpt-4"
export INFERENCE_KEY="sk-..."
```

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

### Token Limits

```python
llm = LLM(
    type='completion',
    input_tokens_limit=40_000,  # Input token limit
    max_length=100_000  # Maximum response length
)
```

## Included Schemas

The library includes several pre-built Pydantic schemas:

- `ThinkingModel`: Base model with reasoning/thought process
- `Suggestion`: Code review suggestions
- `SuggestionsResponse`: Collection of suggestions
- `Labels`: Label generation for categorization

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
- `type` (str): 'completion' or 'embedding' (default: 'completion')
- `model` (str): Model identifier (defaults to INFERENCE_MODEL_ID env var)
- `temperature` (float): Sampling temperature (default: 0.15)
- `frequency_penalty` (float): Frequency penalty (default: 0.2)
- `presence_penalty` (float): Presence penalty (default: 0.2)
- `response_format` (BaseModel): Pydantic model for structured output
- `continuous` (bool): Enable conversation history (default: True)
- `messages` (list): Initial conversation history
- `max_length` (int): Maximum response length (default: 100,000)
- `input_tokens_limit` (int): Input token limit (default: 40,000)
- `system` (str): System prompt

**Methods:**
- `prompt(prompt, system=None, retry=False)`: Generate a response
- `add_message(role, content)`: Add a message to conversation history
- `reset()`: Clear conversation history
- `embed(text)`: Generate embeddings (future feature)

**Properties:**
- `supports_parallel_requests`: Check if model supports parallel requests
- `supports_frequency_penalty`: Check if model supports frequency penalty
- `supports_presence_penalty`: Check if model supports presence penalty
- `require_response_format`: Check if response format is required

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black floship_llm/
```

### Type Checking

```bash
mypy floship_llm/
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Support

For questions or issues, please open an issue on GitHub or contact dev@floship.com

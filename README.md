# Floship LLM Client Library

A reusable Python library for interacting with Heroku Managed Inference and Agents API and other OpenAI-compatible inference endpoints.

## Features

- 🚀 Simple and intuitive API for LLM interactions
- 🎯 **Heroku Inference API** - Optimized for Heroku's managed LLM service
- 🔄 Support for continuous (multi-turn) conversations
- 🛠️ **Tool/Function calling** - Let the LLM execute Python functions
- 📊 **Tool call tracking** (v0.2.0+) - Monitor and budget tool usage with detailed metrics
- �️ **CloudFront WAF Protection** (v0.5.0+) - Automatic content sanitization to prevent 403 errors
- 🔁 **Automatic retry** - Exponential backoff for 403 errors with forced sanitization
- �📈 Structured output with Pydantic schemas
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

### Streaming Responses

Stream responses in real-time as they are generated:

```python
from floship_llm import LLM

# Create a client with streaming enabled
llm = LLM(
    type='completion',
    stream=True,
    enable_tools=False  # Tools are not supported in streaming mode
)

# Stream response chunks
print("Response: ", end="", flush=True)
for chunk in llm.prompt_stream("Write a short poem about the ocean"):
    print(chunk, end="", flush=True)
print()  # New line after streaming completes
```

**Important Notes:**
- Streaming mode does **not support tool/function calling**
- If tools are enabled, `prompt_stream()` will raise a `ValueError`
- Use the regular `prompt()` method if you need tool support
- Streaming responses are added to conversation history automatically

### Streaming Final Response After Tools

**NEW in v0.4.0:** Stream the final response even when using tools!

```python
from floship_llm import LLM, ToolFunction, ToolParameter

llm = LLM(enable_tools=True)

# Add a tool
llm.add_tool(
    ToolFunction(
        name="calculate",
        description="Perform calculations",
        parameters=[
            ToolParameter(name="expression", type="string", description="Math expression")
        ],
        function=lambda expr: str(eval(expr))
    )
)

# Stream the final response after tool execution
result = llm.prompt(
    "Calculate 15 * 8 and explain the result",
    stream_final_response=True
)

# Result is a generator if tools were used
if hasattr(result, "__iter__") and not isinstance(result, str):
    for chunk in result:
        print(chunk, end="", flush=True)
else:
    print(result)  # No tools used, returned as string
```

**How it works:**
1. Tools execute normally (non-streaming)
2. After all tools complete, the final LLM response streams in real-time
3. Provides better UX for long responses after tool execution
4. Automatically handles recursive tool calls (falls back to non-streaming if needed)

**See also:** `example_stream_with_tools.py` for complete examples

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

)

response = llm.prompt("Generate 3 labels for a ticket about database optimization")
# Returns a Labels object with validated structure
print(response.labels)
```

## CloudFront WAF Protection

**NEW in v0.5.0:** Built-in protection against CloudFront Web Application Firewall blocking.

### The Problem

When sending code content (PR diffs, file patches, error traces) to LLM APIs behind CloudFront WAF, requests may be blocked with `403 Forbidden` errors if they contain patterns resembling security attacks:

- **Path traversal**: `../../config/settings.py`
- **XSS patterns**: `<script>`, `<iframe>`, `javascript:`
- **Event handlers**: `onerror=`, `onload=`

### The Solution

The library now **automatically sanitizes** content to prevent WAF blocking while preserving semantic meaning:

```python
from floship_llm import LLM

# WAF protection is enabled by default
llm = LLM()

# Send PR diff containing suspicious patterns - no 403 error!
pr_diff = """
diff --git a/../../src/auth.py b/../../src/auth.py
--- a/../../src/auth.py
+++ b/../../src/auth.py
@@ -10,3 +10,3 @@
-return '<script>alert("xss")</script>'
+return sanitize_html(content)
"""

response = llm.prompt(f"Review this security fix:\n{pr_diff}")
# Content is automatically sanitized before sending
# ../../ becomes [PARENT_DIR]/
# <script> becomes [SCRIPT_TAG]
```

### Configuration

```python
from floship_llm import LLM, LLMConfig

# Default: WAF protection enabled
llm = LLM()

# Disable if you know your content is safe
llm = LLM(enable_waf_sanitization=False)

# Custom configuration
config = LLMConfig(
    enable_waf_sanitization=True,
    max_waf_retries=2,           # Retry on 403 errors
    retry_with_sanitization=True, # Force sanitization on retry
    debug_mode=False,            # Enable detailed logging
    log_sanitization=True,       # Log when content is sanitized
    log_blockers=True            # Log which patterns were found
)
llm = LLM(waf_config=config)

# Environment variables (optional)
import os
os.environ['FLOSHIP_LLM_WAF_SANITIZE'] = 'true'
os.environ['FLOSHIP_LLM_DEBUG'] = 'false'
os.environ['FLOSHIP_LLM_WAF_MAX_RETRIES'] = '2'
```

### Automatic Retry on 403

If a 403 error occurs (even with sanitization disabled), the library automatically retries with sanitization enabled:

```python
llm = LLM(enable_waf_sanitization=False)

# First attempt: No sanitization
# If 403 error: Automatically retries with sanitization
# Uses exponential backoff: 1s, 2s delays
response = llm.prompt("Content with ../../paths")
```

### Monitoring Metrics

Track how often sanitization occurs and 403 errors happen:

```python
llm = LLM()

# Make some requests
llm.prompt("Check file ../../config/settings.py")
llm.prompt("Normal content")

# Get metrics
metrics = llm.get_waf_metrics()
print(f"Total requests: {metrics['total_requests']}")
print(f"Sanitization rate: {metrics['sanitization_rate']:.1%}")
print(f"403 error rate: {metrics['cloudfront_403_rate']:.1%}")

# Reset metrics
llm.reset_waf_metrics()
```

### What Gets Sanitized

| Pattern Type | Original | Sanitized | Example |
|--------------|----------|-----------|---------|
| Path traversal | `../` | `[PARENT_DIR]/` | `../../config` → `[PARENT_DIR]/[PARENT_DIR]/config` |
| Path traversal | `..\` | `[PARENT_DIR]\` | `..\..\config` → `[PARENT_DIR]\[PARENT_DIR]\config` |
| XSS script tag | `<script>` | `[SCRIPT_TAG]` | `<script>alert(1)</script>` → `[SCRIPT_TAG]alert(1)[/SCRIPT_TAG]` |
| XSS iframe | `<iframe>` | `[IFRAME_TAG]` | `<iframe src="x">` → `[IFRAME_TAG] src="x">` |
| JS protocol | `javascript:` | `js:` | `href="javascript:void(0)"` → `href="js:void(0)"` |
| Event handler | `onerror=` | `on_error=` | `<img onerror="x">` → `<img on_error="x">` |

### Real-World Use Cases

**1. PR Description Generator (support-app)**

```python
from floship_llm import LLM

llm = LLM()  # WAF protection enabled by default

# No need for manual sanitization anymore!
pr_info = "File: ../../src/middleware/auth.js"
commit_info = "Fixed: <script> tag injection vulnerability"
file_changes = "+++ b/../../src/middleware/auth.js"

# Previously required manual sanitization
# Now handled automatically by the library
response = llm.prompt(f"""
Generate PR description:
{pr_info}
{commit_info}
{file_changes}
""")
```

**2. Code Review Bot**

```python
llm = LLM(debug_mode=True)  # Enable logging to monitor sanitization

diff_content = """
diff --git a/../../app/views/index.html b/../../app/views/index.html
-<script>var config = {api: "../../api/config.json"};</script>
+<script>var config = {api: "/api/config.json"};</script>
"""

# Automatic protection against WAF blocking
review = llm.prompt(f"Review this security fix:\n{diff_content}")
```

### Benefits Over Manual Sanitization

✅ **Automatic** - Works out of the box, no setup required
✅ **Transparent** - No changes needed to existing code
✅ **Resilient** - Automatic retry with sanitization on 403
✅ **Configurable** - Can be disabled or customized
✅ **Monitored** - Built-in metrics for tracking
✅ **Tested** - Comprehensive test suite (28 tests)
✅ **Semantic Preservation** - LLM can still understand `[PARENT_DIR]/` as `../`

### Migration from Manual Sanitization

**Before (support-app pattern):**
```python
from llm.sanitization import sanitize_pr_description_content

# Manual sanitization required
sanitized_pr, sanitized_commits, sanitized_files, _, _ = (
    sanitize_pr_description_content(pr_info, commit_info, file_changes, ticket)
)
prompt = f"{sanitized_pr}\n{sanitized_commits}\n{sanitized_files}"
response = llm.prompt(prompt)
```

**After (with library WAF protection):**
```python
# Automatic sanitization - just use raw content!
prompt = f"{pr_info}\n{commit_info}\n{file_changes}"
response = llm.prompt(prompt)  # WAF protection handled automatically
```

**Benefits:**
- Remove 200+ lines of sanitization code
- Eliminate maintenance burden
- Better error recovery with automatic retry
- Works for all library consumers, not just one app

### Tool/Function Calling

Enable the LLM to execute Python functions during conversations:

````
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

### Tool Call Tracking (v0.2.0+)

Track and monitor all tool invocations for resource budgeting, cost estimation, and debugging:

```python
from floship_llm import LLM

llm = LLM(enable_tools=True)

# Add some tools
llm.add_tool_from_function(search_database)
llm.add_tool_from_function(calculate_metrics)
llm.add_tool_from_function(generate_report)

# Make a prompt that uses tools
response = llm.prompt("Analyze Q4 sales data and generate a summary report")

# Access tracking data
tool_count = llm.get_last_tool_call_count()
history = llm.get_last_tool_history()
depth = llm.get_last_recursion_depth()

print(f"Used {tool_count} tool calls across {depth} recursion levels")

# Examine detailed history
for call in history:
    print(f"{call['index']}. {call['tool']} - {call['execution_time_ms']}ms")
    if 'error' in call:
        print(f"   Error: {call['error']}")
```

#### Tracking Methods

- `get_last_tool_call_count()` → `int`: Total number of tool calls in last prompt
- `get_last_tool_history()` → `List[Dict]`: Detailed history of each tool call
- `get_last_recursion_depth()` → `int`: Maximum recursion depth reached
- `get_last_response_metadata()` → `Dict`: Complete metadata dictionary

#### Tool History Entry Structure

Each entry in the tool history contains:
```python
{
    "index": 1,                      # Sequential call number
    "tool": "search_database",       # Tool name
    "arguments": {"query": "..."},   # Arguments passed
    "recursion_depth": 0,            # Depth in call chain
    "execution_time_ms": 245,        # Execution time
    "result_length": 1024,           # Result size in chars
    "timestamp": 1698234567.89,      # Unix timestamp
    "tokens": 150,                   # Token count (if available)
    "was_truncated": False,          # Whether result was truncated
    "error": "..."                   # Error message (if failed)
}
```

#### Resource Budgeting Example

```python
# Set tool usage budget
TOOL_BUDGET = 50
COST_PER_TOOL = 0.001  # $0.001 per tool call

response = llm.prompt("Complex analysis task...")

# Check budget
tools_used = llm.get_last_tool_call_count()
cost = tools_used * COST_PER_TOOL

if tools_used > TOOL_BUDGET:
    print(f"⚠️  Budget exceeded: {tools_used}/{TOOL_BUDGET} tools")
print(f"Cost: ${cost:.4f}")
```

#### Rate Limiting Example

```python
import time

MAX_TOOLS_PER_MINUTE = 100
tools_this_minute = 0

response = llm.prompt("Task...")
tools_this_minute += llm.get_last_tool_call_count()

if tools_this_minute >= MAX_TOOLS_PER_MINUTE:
    print("Rate limit reached, waiting...")
    time.sleep(60)
    tools_this_minute = 0
```

#### Performance Analysis Example

```python
history = llm.get_last_tool_history()

# Find slow tools
slow_tools = [c for c in history if c['execution_time_ms'] > 1000]
print(f"Slow tools: {len(slow_tools)}")

# Analyze recursion patterns
max_depth = llm.get_last_recursion_depth()
if max_depth > 3:
    print(f"⚠️  Deep recursion: {max_depth} levels")

# Group by recursion depth
for depth in range(max_depth + 1):
    at_depth = [c for c in history if c['recursion_depth'] == depth]
    print(f"Depth {depth}: {len(at_depth)} tools")
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

*Tool Call Tracking (v0.2.0+):*
- `get_last_tool_call_count()`: Get total number of tool calls from last prompt
- `get_last_tool_history()`: Get detailed history of all tool calls with metadata
- `get_last_recursion_depth()`: Get maximum recursion depth reached
- `get_last_response_metadata()`: Get complete metadata dictionary

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

**Test Coverage:** 260 tests covering all functionality including:
- Core LLM client operations
- Tool/function calling workflows
- Tool call tracking and metadata
- Schema validation and JSON processing
- Error handling and edge cases
- 93% overall code coverage

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

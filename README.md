# Floship LLM Client Library

Python library for Heroku Managed Inference API and OpenAI-compatible endpoints.

## Features

- 🚀 Simple API for LLM interactions with streaming support
- 🛠️ Tool/function calling with tracking and budgeting
- 🛡️ CloudFront WAF protection (auto-sanitization)
- 📊 Structured output with Pydantic schemas
- 🧠 Extended thinking for Claude models
- 🔄 Continuous conversations and retry mechanisms
- 📐 Embeddings support

## Installation

```bash
pip install floship-llm

# From source
git clone https://github.com/floship/floship-llm.git && cd floship-llm && pip install -e .

# Development (with uv)
uv sync --dev && uv run pytest
```

## Quick Start

### Environment Setup

```bash
export INFERENCE_URL="https://us.inference.heroku.com"
export INFERENCE_MODEL_ID="claude-4-sonnet"
export INFERENCE_KEY="your-heroku-api-key"
```

### Basic Usage

```python
from floship_llm import LLM

llm = LLM(type='completion', temperature=0.15)
response = llm.prompt("What is the capital of France?")
```

### Streaming

```python
llm = LLM(stream=True)
for chunk in llm.prompt_stream("Write a poem about the ocean"):
    print(chunk, end="", flush=True)
```

### Continuous Conversations

```python
llm = LLM(continuous=True, system="You are a helpful assistant.")
response1 = llm.prompt("Tell me about Python.")
response2 = llm.prompt("What are its main advantages?")  # Maintains context
llm.reset()  # Clear history
```

### Structured Output

```python
from pydantic import BaseModel

class Analysis(BaseModel):
    summary: str
    key_points: list[str]

llm = LLM(response_format=Analysis)
result = llm.prompt("Analyze renewable energy trends")
print(result.summary, result.key_points)
```

## Embeddings

```python
llm = LLM(type='embedding', model='cohere-embed-multilingual')

# Single or batch
embedding = llm.embed("Hello, world!")
embeddings = llm.embed(["Text 1", "Text 2", "Text 3"])

# With metadata
response = llm.embed("Sample", return_full_response=True)
```

**Input Types:** `search_document`, `search_query`, `classification`, `clustering`

```python
doc_llm = LLM(type='embedding', input_type='search_document')
query_llm = LLM(type='embedding', input_type='search_query')
```

## Tool Calling

```python
from floship_llm import LLM, ToolFunction, ToolParameter

llm = LLM(enable_tools=True)

def calculate_tax(amount: float, rate: float) -> float:
    return amount * rate / 100

llm.add_tool_from_function(calculate_tax, description="Calculate tax")
response = llm.prompt("What's the tax on $1000 at 8.5%?")
```

### Tool Management

```python
llm.list_tools()           # List available tools
llm.remove_tool("name")    # Remove specific tool
llm.clear_tools()          # Remove all tools
```

### Tool Call Tracking

```python
response = llm.prompt("Complex task...")

count = llm.get_last_tool_call_count()
history = llm.get_last_tool_history()
depth = llm.get_last_recursion_depth()

# History entry structure
# {"index": 1, "tool": "name", "arguments": {...}, "execution_time_ms": 245, ...}
```

### Stream Final Response After Tools

```python
result = llm.prompt("Calculate 15 * 8 and explain", stream_final_response=True)
for chunk in result:
    print(chunk, end="", flush=True)
```

## Extended Thinking (Claude)

Extended thinking enables deeper reasoning for complex tasks.

```python
llm = LLM(
    extended_thinking={
        "enabled": True,
        "budget_tokens": 1024,      # Minimum: 1024
        "include_reasoning": True,
    }
)
```

**⚠️ Note:** Temperature is auto-set to 1.0 when extended thinking is enabled.

### With Structured Output

```python
llm = LLM(
    extended_thinking={"enabled": True, "budget_tokens": 2048},
    response_format=Analysis,
)
```

### With Tools

```python
llm = LLM(
    extended_thinking={"enabled": True, "budget_tokens": 2048},
    enable_tools=True,
)
# Auto-retries without extended thinking if API rejects it
```

### Extracting Thoughts

Access Claude's reasoning via `get_last_raw_response()`. The clean response (from `prompt()`) has thinking stripped.

#### Without response_format (Plain Text)

```python
import re

llm = LLM(
    extended_thinking={"enabled": True, "budget_tokens": 2048, "include_reasoning": True}
)

response = llm.prompt("What sorting algorithm is best for nearly-sorted data?")
print(response)  # Clean response without <think> tags

# Extract thinking from raw response
raw = llm.get_last_raw_response()
match = re.search(r'<think>(.*?)</think>', raw, re.DOTALL)
if match:
    print("Reasoning:", match.group(1))
```

#### With response_format (Structured Output)

```python
from pydantic import BaseModel
import re

class Analysis(BaseModel):
    answer: str
    confidence: float

llm = LLM(
    extended_thinking={"enabled": True, "budget_tokens": 2048, "include_reasoning": True},
    response_format=Analysis,
)

result = llm.prompt("Analyze if P=NP")
print(result.answer, result.confidence)  # Parsed Pydantic model

# Thinking is still in raw response
raw = llm.get_last_raw_response()
match = re.search(r'<think>(.*?)</think>', raw, re.DOTALL)
if match:
    print("Reasoning:", match.group(1))
```

#### With Streaming

```python
llm = LLM(
    extended_thinking={"enabled": True, "budget_tokens": 2048, "include_reasoning": True},
    stream=True,
)

for chunk in llm.prompt_stream("Explain quantum entanglement"):
    print(chunk, end="", flush=True)  # Streamed without <think> tags

# After streaming completes
raw = llm.get_last_raw_response()
# Extract thinking same as above
```

**Note:** Thinking content is only available after the full response completes.

## CloudFront WAF Protection

Auto-sanitizes content to prevent 403 errors from patterns like `../`, `<script>`, `javascript:`.

```python
llm = LLM()  # Enabled by default

# Send PR diffs with suspicious patterns - no 403!
pr_diff = "diff --git a/../../src/auth.py"
response = llm.prompt(f"Review:\n{pr_diff}")
```

### Configuration

```python
from floship_llm import LLMConfig

llm = LLM(enable_waf_sanitization=False)  # Disable

config = LLMConfig(
    enable_waf_sanitization=True,
    max_waf_retries=2,
    retry_with_sanitization=True,
)
llm = LLM(waf_config=config)
```

### Sanitization Patterns

| Pattern | Original | Sanitized |
|---------|----------|-----------|
| Path traversal | `../` | `[PARENT_DIR]/` |
| XSS script | `<script>` | `[SCRIPT_TAG]` |
| JS protocol | `javascript:` | `js:` |
| Event handler | `onerror=` | `on_error=` |

### Metrics

```python
metrics = llm.get_waf_metrics()
# {"total_requests": 10, "sanitization_rate": 0.3, "cloudfront_403_rate": 0.0}
```

## Configuration

### Environment Variables

| Variable | Description |
|----------|-------------|
| `INFERENCE_URL` | API endpoint (default: `https://us.inference.heroku.com`) |
| `INFERENCE_MODEL_ID` | Model ID (e.g., `claude-4-sonnet`) |
| `INFERENCE_KEY` | API key |

### Constructor Parameters

```python
LLM(
    type='completion',              # 'completion' or 'embedding'
    temperature=0.15,               # 0.0-1.0, creativity control
    max_completion_tokens=None,     # Max tokens to generate
    top_p=None,                     # Nucleus sampling (0.0-1.0)
    top_k=None,                     # Top-k sampling
    extended_thinking=None,         # {"enabled": bool, "budget_tokens": int, "include_reasoning": bool}
    continuous=False,               # Multi-turn conversations
    response_format=None,           # Pydantic model for structured output
    enable_tools=False,             # Enable tool calling
    system=None,                    # System prompt
    retry_limit=5,                  # Max retry attempts
    input_tokens_limit=40_000,      # Input token limit
    max_length=100_000,             # Max response length
    enable_waf_sanitization=True,   # CloudFront WAF protection
)
```

**⚠️ Deprecated:** `frequency_penalty`, `presence_penalty` (not supported by Heroku)

## Methods Reference

### Core
- `prompt(prompt, system=None, retry=False)` - Generate response
- `prompt_stream(prompt)` - Stream response chunks
- `reset()` - Clear conversation history
- `embed(text, return_full_response=False)` - Generate embeddings

### Tools
- `add_tool(ToolFunction)` - Add tool
- `add_tool_from_function(func, description=None)` - Wrap function as tool
- `remove_tool(name)` / `clear_tools()` - Remove tools
- `list_tools()` - List available tools

### Tracking
- `get_last_tool_call_count()` - Tool calls in last prompt
- `get_last_tool_history()` - Detailed tool call history
- `get_last_recursion_depth()` - Max recursion depth
- `get_last_raw_response()` - Raw response (includes thinking tags)

### WAF
- `get_waf_metrics()` - Sanitization statistics
- `reset_waf_metrics()` - Reset metrics

## Included Schemas

```python
from floship_llm import ToolFunction, ToolParameter, ToolCall, ToolResult
from floship_llm import Labels, Suggestion, SuggestionsResponse, ThinkingModel
```

## JSON Utilities

```python
from floship_llm import lm_json_utils

cleaned = lm_json_utils.extract_and_fix_json("messy text {name: 'John'}")
json_str = lm_json_utils.extract_strict_json(text)
```

## Development

```bash
pytest tests/                              # Run tests
pytest --cov=floship_llm --cov-report=html # With coverage
black floship_llm/                         # Format
mypy floship_llm/                          # Type check
```

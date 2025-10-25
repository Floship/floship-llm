# Streaming Support in floship-llm v0.3.0

## Overview

Version 0.3.0 adds streaming support to floship-llm, allowing real-time response streaming for better user experience. Responses are yielded as chunks arrive from the API instead of waiting for the complete response.

## Quick Start

```python
from floship_llm import LLM

# Create client with streaming enabled
llm = LLM(stream=True, enable_tools=False)

# Stream response chunks
for chunk in llm.prompt_stream("Write a short poem"):
    print(chunk, end="", flush=True)
```

## API

### New Method: `prompt_stream()`

```python
def prompt_stream(prompt: str, system: Optional[str] = None)
```

**Parameters:**
- `prompt` (str): User prompt text
- `system` (Optional[str]): Optional system message (overrides instance system)

**Yields:**
- str: Response chunks as they arrive from the API

**Raises:**
- `ValueError`: If tools are enabled (streaming doesn't support tools)

**Example:**
```python
llm = LLM(stream=True)
for chunk in llm.prompt_stream("Hello!", system="You are helpful"):
    print(chunk, end="", flush=True)
```

### New Parameter: `stream`

Added to `LLM.__init__()`:

```python
llm = LLM(
    stream=True,  # Enable streaming mode
    enable_tools=False  # Must be False for streaming
)
```

## Features

### ✅ What Works

1. **Real-time streaming**: Chunks are yielded as they arrive
2. **Conversation history**: Streamed responses are automatically added to history
3. **System messages**: Supports custom system prompts
4. **Continuous mode**: Respects the `continuous` setting
5. **Temperature control**: Works with all model parameters
6. **Multiple turns**: Can stream multiple prompts in sequence (if `continuous=True`)

### ❌ What Doesn't Work

1. **Tool/Function calls**: Streaming does NOT support tools
   - `prompt_stream()` raises `ValueError` if tools are enabled
   - This is a technical limitation of the OpenAI API
   - Use regular `prompt()` method when tools are needed

2. **Retry mechanism**: Streaming responses don't support automatic retries
   - Retry logic is designed for complete responses
   - Handle errors in your application code

3. **Structured output**: Response format validation not supported in streaming
   - Pydantic models work with `prompt()` only
   - Streaming returns plain text chunks

## Examples

### Example 1: Basic Streaming

```python
from floship_llm import LLM

llm = LLM(stream=True, enable_tools=False)

print("Response: ", end="", flush=True)
for chunk in llm.prompt_stream("Count from 1 to 5"):
    print(chunk, end="", flush=True)
print()
```

### Example 2: Streaming with Conversation History

```python
llm = LLM(
    stream=True,
    enable_tools=False,
    continuous=True,
    system="You are a helpful assistant"
)

# First question
for chunk in llm.prompt_stream("What is Python?"):
    print(chunk, end="", flush=True)

# Follow-up (maintains context)
for chunk in llm.prompt_stream("What are its main uses?"):
    print(chunk, end="", flush=True)
```

### Example 3: Error Handling

```python
llm = LLM(stream=True, enable_tools=True)  # Tools enabled!

try:
    for chunk in llm.prompt_stream("Test"):
        print(chunk, end="", flush=True)
except ValueError as e:
    print(f"Error: {e}")
    # Output: "Streaming mode does not support tool calls..."
```

### Example 4: Accumulating Response

```python
llm = LLM(stream=True, enable_tools=False)

full_response = ""
for chunk in llm.prompt_stream("Write a haiku"):
    full_response += chunk
    print(chunk, end="", flush=True)

print(f"\n\nFull response length: {len(full_response)}")
```

## Implementation Details

### How It Works

1. **Client initialization**: Set `stream=True` in constructor
2. **API call**: `prompt_stream()` passes `stream=True` to OpenAI API
3. **Response iteration**: Iterates over streaming response chunks
4. **Content extraction**: Extracts `delta.content` from each chunk
5. **History update**: Adds complete response to conversation history
6. **Cleanup**: Resets conversation if `continuous=False`

### Tool Tracking

- Tool tracking variables are reset at the start of `prompt_stream()`
- No metadata is collected during streaming
- `get_last_tool_call_count()` returns 0 after streaming

### Message Validation

- Messages are validated before streaming (same as `prompt()`)
- Uses internal `_validate_messages_for_api()` method
- Ensures correct message structure for API

## Testing

The streaming feature has comprehensive test coverage:

```bash
# Run all streaming tests
uv run pytest tests/test_client.py -k "streaming" -v
```

**Test Coverage:**
- ✅ Basic streaming functionality
- ✅ Error when tools are enabled
- ✅ Conversation history updates
- ✅ Handling empty/None chunks
- ✅ System message support
- ✅ Continuous mode reset behavior

## Migration Guide

### From v0.2.0 to v0.3.0

**No breaking changes for existing code!**

Existing code continues to work unchanged:

```python
# This still works exactly as before
llm = LLM()
response = llm.prompt("Hello")
```

**To enable streaming:**

```python
# Old way (still works)
llm = LLM()
response = llm.prompt("Hello")  # Waits for complete response
print(response)

# New way (streaming)
llm = LLM(stream=True, enable_tools=False)
for chunk in llm.prompt_stream("Hello"):  # Yields chunks
    print(chunk, end="", flush=True)
```

**Important notes:**
- Streaming is opt-in (default: `stream=False`)
- Tools must be disabled for streaming
- Use `prompt_stream()` for streaming, `prompt()` for complete responses

## Performance

**Advantages of streaming:**
- ✅ Better user experience (visible progress)
- ✅ Lower perceived latency (first token arrives quickly)
- ✅ Can process chunks as they arrive
- ✅ Useful for long responses

**Disadvantages:**
- ❌ No automatic retry on errors
- ❌ Can't use tools/function calling
- ❌ No structured output validation
- ❌ More complex error handling

## See Also

- `examples_streaming.py` - 5 complete streaming examples
- `README.md` - Streaming documentation and examples
- `CHANGELOG.md` - v0.3.0 release notes
- `tests/test_client.py` - Streaming test suite

## Support

For issues or questions about streaming:
- Check `examples_streaming.py` for working examples
- Review test cases in `tests/test_client.py`
- Open an issue on GitHub

## Future Enhancements

Potential improvements for future versions:
- [ ] Streaming with tool calls (if OpenAI API supports it)
- [ ] Token-by-token streaming metrics
- [ ] Streaming with retry mechanism
- [ ] Async/await streaming support

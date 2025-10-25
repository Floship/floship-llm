# Tool Request Delay Feature

## Overview

Added a configurable delay between tool execution and follow-up API requests to help avoid rate limiting and give the API time to process requests.

## Configuration

The delay is controlled by the `LLM_TOOL_REQUEST_DELAY` environment variable.

### Default Behavior
- **Default delay**: 5 seconds
- Applied after all tools are executed, before the follow-up API request
- Can be set to 0 to disable the delay

### Environment Variable

```bash
# Set delay to 3 seconds
export LLM_TOOL_REQUEST_DELAY=3

# Set delay to 2.5 seconds (decimal values supported)
export LLM_TOOL_REQUEST_DELAY=2.5

# Disable delay
export LLM_TOOL_REQUEST_DELAY=0
```

## Usage

The delay is automatically applied when tools are executed. No code changes needed.

```python
from floship_llm import LLM

# Create LLM with tool support
llm = LLM(enable_tools=True)

# Add your tools
llm.add_tool(my_tool)

# When tools are executed, there will be a configurable delay
# before the follow-up API request (default 5 seconds)
result = llm.prompt("Use the tool to do something")
```

## Implementation Details

1. **Initialization**: The `tool_request_delay` attribute is set during LLM initialization from the `LLM_TOOL_REQUEST_DELAY` environment variable (default: 5 seconds)

2. **Application**: The delay is applied in the `_handle_tool_calls()` method:
   - After all tools have been executed
   - Before the follow-up API request
   - Only once per tool execution cycle (not per tool)

3. **Logging**: When the delay is applied, an info log message is generated:
   ```
   Waiting {delay}s before follow-up request after tool execution
   ```

## Benefits

1. **Rate Limit Protection**: Helps avoid hitting API rate limits by spacing out requests
2. **API Stability**: Gives the API time to process requests, reducing 403 and 429 errors
3. **Configurable**: Can be adjusted per environment (dev/staging/production)
4. **Can be Disabled**: Set to 0 for testing or when rate limiting is not a concern

## Testing

Comprehensive test suite with 7 tests covering:
- Default delay value (5 seconds)
- Custom delay from environment variable
- Decimal delay values
- Zero delay (disabled)
- Actual delay application after tool execution
- No delay when set to zero
- Single delay with multiple tool calls

All tests pass: ✅ 197/197 tests passing

## Example Scenarios

### Scenario 1: Production with Rate Limiting
```bash
# Set 5-second delay to avoid rate limits
export LLM_TOOL_REQUEST_DELAY=5
```

### Scenario 2: Development/Testing
```bash
# Disable delay for faster testing
export LLM_TOOL_REQUEST_DELAY=0
```

### Scenario 3: Conservative Production
```bash
# Use longer delay for very strict rate limits
export LLM_TOOL_REQUEST_DELAY=10
```

## Code Changes

### Modified Files
- `floship_llm/client.py`:
  - Added `tool_request_delay` attribute initialization
  - Added delay logic in `_handle_tool_calls()` method

### New Files
- `tests/test_tool_delay.py`: Comprehensive test suite for delay functionality

## Migration

This is a **backward-compatible** change:
- Existing code works without modifications
- Default 5-second delay may increase response time for tool-using applications
- Set `LLM_TOOL_REQUEST_DELAY=0` to maintain previous behavior (no delay)

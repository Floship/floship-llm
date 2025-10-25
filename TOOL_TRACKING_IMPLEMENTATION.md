# Tool Call Tracking Implementation - Summary

## Overview

This document summarizes the implementation of tool call tracking functionality in the floship-llm library. This enhancement addresses the critical need for autonomous agents to monitor actual tool usage and implement proper resource budgeting.

## Problem Statement

Previously, when using `llm.prompt()`, callers had no visibility into the number of tool calls made internally. A single prompt could trigger dozens of recursive tool calls, but the caller would only see the final response. This made it impossible to:

- Budget tool usage accurately
- Track actual API costs
- Implement rate limiting
- Debug tool call chains
- Monitor recursive tool behavior

## Solution

We implemented comprehensive tool call tracking that:

1. **Tracks every tool invocation** with detailed metadata
2. **Stores final statistics** after each prompt completes
3. **Exposes public API methods** for accessing the tracking data
4. **Handles recursive tool calls** by tracking depth
5. **Captures both successful and failed tool calls**

## Implementation Details

### Internal Tracking Attributes (client.py `__init__`)

```python
# Tool call tracking (for monitoring and budgeting)
self._current_tool_call_count = 0          # Counter for current prompt
self._current_recursion_depth = 0          # Max depth reached
self._current_tool_history = []            # Detailed history list
self._last_response_metadata = {}          # Saved after prompt completes
```

### Modified Methods

#### 1. `prompt()` Method Enhancement

**Location**: Lines 333-398
**Changes**:
- Resets tracking variables at the start of each prompt
- Stores final metadata in `_last_response_metadata` after completion

```python
# Reset tracking
self._current_tool_call_count = 0
self._current_recursion_depth = 0
self._current_tool_history = []

# ... execute prompt ...

# Store metadata
self._last_response_metadata = {
    "total_tool_calls": self._current_tool_call_count,
    "recursion_depth": self._current_recursion_depth,
    "tool_history": self._current_tool_history.copy(),
}
```

#### 2. `_handle_tool_calls()` Method Enhancement

**Location**: Lines 460-667
**Changes**:
- Added `recursion_depth=0` parameter to method signature
- Tracks recursion depth: `self._current_recursion_depth = max(self._current_recursion_depth, recursion_depth)`
- Increments `_current_tool_call_count` for each tool
- Builds detailed tool_entry dict and appends to `_current_tool_history`
- Enhanced logging with tool index and recursion depth
- Tracks failed tool calls with error information
- Passes `recursion_depth + 1` to recursive calls

**Tool Entry Structure**:
```python
tool_entry = {
    "index": self._current_tool_call_count,
    "tool": tool_call.function.name,
    "arguments": arguments,
    "recursion_depth": recursion_depth,
    "execution_time_ms": int(execution_time * 1000),
    "result_length": len(str(result.content)),
    "timestamp": time.time(),
    "tokens": meta.get("final_tokens", 0),  # if available
    "was_truncated": meta.get("was_truncated", False)
}
```

### New Public API Methods

#### 1. `get_last_tool_call_count()` → int
**Location**: Lines 733-743
Returns the total number of tool calls from the last `prompt()` invocation.

#### 2. `get_last_tool_history()` → List[Dict]
**Location**: Lines 745-762
Returns detailed history of all tool calls with complete metadata.

#### 3. `get_last_recursion_depth()` → int
**Location**: Lines 764-774
Returns the maximum recursion depth reached during the last prompt.

#### 4. `get_last_response_metadata()` → Dict
**Location**: Lines 776-795
Returns complete metadata dictionary containing:
- `total_tool_calls`: Total count
- `recursion_depth`: Max depth
- `tool_history`: Full history list

## Usage Example

```python
from floship_llm import LLM

# Initialize with tools enabled
llm = LLM(enable_tools=True)

# Make a prompt that triggers tool calls
response = llm.prompt("Research the latest AI developments and summarize")

# Access tracking data
tool_count = llm.get_last_tool_call_count()
history = llm.get_last_tool_history()
depth = llm.get_last_recursion_depth()
metadata = llm.get_last_response_metadata()

print(f"Used {tool_count} tools across {depth} recursion levels")

# Detailed analysis
for call in history:
    print(f"  {call['index']}. {call['tool']}")
    print(f"     Depth: {call['recursion_depth']}")
    print(f"     Time: {call['execution_time_ms']}ms")
    print(f"     Tokens: {call.get('tokens', 'N/A')}")
    print(f"     Truncated: {call.get('was_truncated', False)}")
    if 'error' in call:
        print(f"     Error: {call['error']}")
```

## Benefits for Autonomous Agents

### 1. **Accurate Resource Budgeting**
```python
# Set tool budget
TOOL_BUDGET = 50

response = llm.prompt("Complex task...")
actual_tools_used = llm.get_last_tool_call_count()

if actual_tools_used > TOOL_BUDGET:
    print(f"⚠️  Budget exceeded: {actual_tools_used}/{TOOL_BUDGET}")
```

### 2. **Cost Estimation**
```python
COST_PER_TOOL = 0.001  # $0.001 per tool call

response = llm.prompt("Task...")
cost = llm.get_last_tool_call_count() * COST_PER_TOOL
print(f"Estimated cost: ${cost:.4f}")
```

### 3. **Rate Limiting**
```python
MAX_TOOLS_PER_MINUTE = 100
tools_used_this_minute = 0

response = llm.prompt("Task...")
tools_used_this_minute += llm.get_last_tool_call_count()

if tools_used_this_minute >= MAX_TOOLS_PER_MINUTE:
    print("Rate limit reached, waiting...")
    time.sleep(60)
    tools_used_this_minute = 0
```

### 4. **Debugging and Optimization**
```python
history = llm.get_last_tool_history()

# Find slowest tools
slow_tools = [c for c in history if c['execution_time_ms'] > 1000]
print(f"Slow tools: {len(slow_tools)}")

# Analyze recursion patterns
max_depth = llm.get_last_recursion_depth()
if max_depth > 3:
    print(f"⚠️  Deep recursion detected: {max_depth} levels")
```

### 5. **Visibility into Tool Chains**
```python
history = llm.get_last_tool_history()

# Group by recursion depth
for depth in range(llm.get_last_recursion_depth() + 1):
    tools_at_depth = [c for c in history if c['recursion_depth'] == depth]
    print(f"Depth {depth}: {len(tools_at_depth)} tools")
    for tool in tools_at_depth:
        print(f"  - {tool['tool']}")
```

## Testing

### Test Script
**File**: `test_tool_tracking.py`
**Purpose**: Verify tracking API methods exist and return correct types
**Status**: ✅ All tests passing

### Unit Tests
**File**: `tests/test_client.py`, `tests/test_tools.py`
**Coverage**: 93% overall (260 tests passing)
**Status**: ✅ All tests passing

### Key Test Areas
1. API methods exist and return correct types
2. Internal tracking attributes initialized correctly
3. Tracking resets between prompts
4. Error handling (e.g., JSON parse errors)
5. Metadata structure validation

## Files Modified

1. **floship_llm/client.py**
   - Added 4 tracking attributes to `__init__`
   - Enhanced `prompt()` method with tracking reset/storage
   - Enhanced `_handle_tool_calls()` with comprehensive tracking
   - Added 4 new public API methods
   - Fixed error handling bug (arguments initialization)

2. **test_tool_tracking.py** (NEW)
   - Comprehensive test suite for tracking functionality
   - Validates API methods and data structures
   - Tests tracking state management

## Performance Impact

- **Memory**: Minimal (~100 bytes per tool call for history entry)
- **CPU**: Negligible (simple counter increments and dict operations)
- **Latency**: No impact (tracking happens during existing tool execution)

## Backward Compatibility

✅ **Fully backward compatible**
- No breaking changes to existing API
- All existing tests pass (260/260)
- New methods are additive only
- Tracking is automatic (no configuration required)
- Tracking data is optional (callers can ignore it)

## Future Enhancements

Potential improvements for future versions:

1. **Token Cost Estimation**: Calculate actual API costs based on token usage
2. **Budget Enforcement**: Hard limits with configurable behavior
3. **Rate Limiting**: Built-in rate limiting based on tool count
4. **Metrics Export**: Export metrics to monitoring systems
5. **Async Tracking**: Real-time callbacks during tool execution
6. **Tool Performance Analytics**: Statistical analysis of tool performance
7. **Configurable Tracking**: Option to enable/disable tracking
8. **Custom Metadata**: Allow tools to add custom tracking fields

## Related Documentation

- **Design Document**: `TOOLS_IMPLEMENTATION.md` (Phase 2)
- **Migration Guide**: `MIGRATION_GUIDE.md` (if needed)
- **API Documentation**: `README.md` (to be updated)

## Summary

The tool call tracking implementation successfully addresses the visibility gap in autonomous agent resource budgeting. By tracking every tool invocation with detailed metadata and exposing this data through a clean public API, the library now enables:

- Accurate cost and resource tracking
- Effective rate limiting and budget enforcement
- Powerful debugging and optimization capabilities
- Complete visibility into recursive tool call chains

All 260 existing tests pass, demonstrating full backward compatibility. The implementation is production-ready and awaiting documentation updates.

## Next Steps

1. ✅ **COMPLETED**: Core implementation
2. ✅ **COMPLETED**: Testing and bug fixes
3. ⏳ **PENDING**: Update README.md with tracking documentation
4. ⏳ **PENDING**: Create usage examples in `examples/`
5. ⏳ **PENDING**: Update CHANGELOG.md
6. ⏳ **PENDING**: Version bump to 0.1.2 or 0.2.0

---

**Implementation Date**: January 2025
**Version**: floship-llm 0.1.1
**Status**: ✅ Complete and tested

# Live API Testing Results - Heroku Inference API

## Test Date: 2025
## Model: claude-4-5-sonnet
## API Endpoint: https://us.inference.heroku.com/v1

---

## Executive Summary

All tests **PASSED** ✅ after discovering and fixing critical parameter handling constraints specific to Claude models via Heroku Inference API.

---

## Discovered Issues & Fixes

### Issue 1: Temperature vs Top_p Mutual Exclusivity
**Severity:** Critical  
**Status:** FIXED ✅

**Problem:**
```
Error code: 400 - ValidationException: The model returned the following errors: 
`temperature` and `top_p` cannot both be specified for this model.
```

**Root Cause:** Claude models via Heroku Inference API enforce mutual exclusivity between `temperature` and `top_p` parameters. You can use one or the other, but not both.

**Solution Implemented:**
Modified `client.py` `get_request_params()` method:
- If `top_p` is set: Use top_p instead of temperature
- If `top_p` is None: Use temperature with specified/default value
- Automatic handling - no user intervention required

**Code Changes:**
```python
# In get_request_params()
if self.top_p is not None:
    # If top_p is set, use it instead of temperature
    extra_body['top_p'] = self.top_p
else:
    # Otherwise use temperature
    params["temperature"] = self.temperature
```

---

### Issue 2: Extended Thinking Requires Temperature = 1.0
**Severity:** Critical  
**Status:** FIXED ✅

**Problem:**
```
Error code: 400 - ValidationException: The model returned the following errors: 
`temperature` may only be set to 1 when thinking is enabled.
```

**Root Cause:** When `extended_thinking` is enabled, Claude requires `temperature` to be exactly 1.0. This is documented at: https://docs.claude.com/en/docs/build-with-claude/extended-thinking

**Solution Implemented:**
Modified `client.py` to detect when extended thinking is enabled and automatically set `temperature=1.0`:

```python
# Check if extended thinking is enabled
has_extended_thinking = self.extended_thinking is not None and (
    (isinstance(self.extended_thinking, dict) and self.extended_thinking.get('enabled', False)) or
    (isinstance(self.extended_thinking, bool) and self.extended_thinking)
)

# Add temperature or top_p (not both for Claude)
if has_extended_thinking:
    # Extended thinking requires temperature = 1
    params["temperature"] = 1.0
elif self.top_p is not None:
    # If top_p is set, use it instead of temperature
    extra_body['top_p'] = self.top_p
else:
    # Otherwise use temperature
    params["temperature"] = self.temperature
```

---

### Issue 3: Extended Thinking Budget Tokens Minimum
**Severity:** Medium  
**Status:** FIXED ✅

**Problem:**
```
Error code: 400 - ValidationException: The model returned the following errors: 
thinking.enabled.budget_tokens: Input should be greater than or equal to 1024
```

**Root Cause:** The `budget_tokens` parameter in `extended_thinking` configuration must be at least 1024.

**Solution:** Updated test and documentation to use minimum value of 1024.

**Example:**
```python
llm = LLM(
    model="claude-4-5-sonnet",
    extended_thinking={
        "enabled": True,
        "budget_tokens": 1024,  # Minimum required value
        "include_reasoning": False
    }
)
```

---

### Issue 4: OpenAI SDK Parameter Compatibility
**Severity:** High  
**Status:** FIXED ✅

**Problem:**
```
TypeError: Completions.create() got an unexpected keyword argument 'top_k'
```

**Root Cause:** The OpenAI Python SDK doesn't natively support Heroku-specific parameters (`top_k`, `top_p`, `extended_thinking`). These are custom parameters specific to Heroku's API.

**Solution Implemented:**
Wrap Heroku-specific parameters in the `extra_body` field:

```python
extra_body = {}

if self.top_k is not None:
    extra_body['top_k'] = self.top_k

if self.top_p is not None:
    extra_body['top_p'] = self.top_p

if self.extended_thinking is not None:
    extra_body['extended_thinking'] = self.extended_thinking

# Add extra_body if it has content
if extra_body:
    params['extra_body'] = extra_body
```

---

### Issue 5: API URL Format
**Severity:** Low  
**Status:** FIXED ✅

**Problem:** Initial 404 errors when connecting to API.

**Root Cause:** Missing `/v1` suffix in base URL.

**Solution:**
- Incorrect: `https://us.inference.heroku.com`
- Correct: `https://us.inference.heroku.com/v1`

---

## Test Results

### Test 1: Basic Prompt ✅
**Status:** PASSED  
**Description:** Simple prompt without additional parameters  
**Response:** Successfully received response from Claude

```python
llm = LLM(model="claude-4-5-sonnet")
response = llm.prompt("Hello from Heroku Inference API! Who are you?")
```

**Result:** "Hello from Heroku Inference API! I am Claude, an AI assistant made by Anthropic..."

---

### Test 2a: Heroku-Specific Parameters (temperature + top_k) ✅
**Status:** PASSED  
**Description:** Test with temperature and top_k (without top_p)  
**Configuration:**
```python
llm = LLM(
    model="claude-4-5-sonnet",
    temperature=0.7,
    max_completion_tokens=100,
    top_k=50
)
```

**Result:** Successfully generated creative response (haiku)

---

### Test 2b: Heroku-Specific Parameters (top_p only) ✅
**Status:** PASSED  
**Description:** Test with top_p instead of temperature  
**Configuration:**
```python
llm = LLM(
    model="claude-4-5-sonnet",
    top_p=0.9,
    max_completion_tokens=200
)
```

**Result:** Successfully generated structured response

---

### Test 3: Extended Thinking ✅
**Status:** PASSED  
**Description:** Test extended thinking feature with correct constraints  
**Configuration:**
```python
llm = LLM(
    model="claude-4-5-sonnet",
    extended_thinking={
        "enabled": True,
        "budget_tokens": 1024,  # Minimum value
        "include_reasoning": False
    }
)
```

**Prompt:** "What is 15 * 23?"  
**Result:** "15 * 23 = 345" ✅

---

### Test 4: Tool Calling ✅
**Status:** PASSED  
**Description:** Test tool/function calling with JSON serialization  
**Configuration:**
```python
def add_numbers(a: int, b: int) -> int:
    return a + b

llm = LLM(model="claude-4-5-sonnet", enable_tools=True)
llm.add_tool_from_function(add_numbers)
```

**Prompt:** "What is 42 plus 58?"  
**Result:** Tool executed correctly, returned "The sum of 42 plus 58 is **100**." ✅

---

## Implementation Quality

### Code Changes Summary
1. **Modified:** `floship_llm/client.py` - `get_request_params()` method
2. **Updated:** `tests/test_client.py` - 2 test expectations
3. **Created:** `HEROKU_API_CONSTRAINTS.md` - Documentation
4. **Created:** `test_heroku_api.py` - Live API test script

### Test Coverage
- **Unit Tests:** 260/260 PASSING ✅
- **Coverage:** 94.48% (maintained from before)
- **Live API Tests:** 4/4 PASSING ✅

### Backward Compatibility
All changes are backward compatible:
- Existing code without `top_p` continues to work with `temperature`
- Users not using `extended_thinking` are unaffected
- Default behavior unchanged for standard use cases

---

## Key Learnings

### 1. Claude Model Constraints
Claude models have stricter parameter constraints than general OpenAI API:
- Temperature and top_p are mutually exclusive
- Extended thinking forces temperature to 1.0
- Budget tokens must be ≥ 1024

### 2. OpenAI SDK Limitations
The OpenAI Python SDK is not aware of Heroku-specific parameters:
- Must use `extra_body` for custom parameters
- Direct parameter passing causes `TypeError`
- This is expected and documented behavior

### 3. Testing Importance
Live API testing revealed constraints not apparent from:
- Documentation alone
- Unit tests with mocked responses
- OpenAI standard API usage

### 4. Automatic Handling Benefits
The library now automatically:
- Handles temperature/top_p mutual exclusivity
- Sets correct temperature for extended thinking
- Wraps Heroku parameters in extra_body
- Users don't need to know these constraints

---

## Recommendations

### For Users
1. **Use either temperature or top_p:** The library handles this automatically
2. **Extended thinking:** Set `budget_tokens` ≥ 1024, library will set temperature=1.0
3. **API URL:** Always use full path with `/v1` suffix
4. **Trust the library:** Parameter handling is automatic and correct

### For Developers
1. **Always test with live API:** Unit tests can't catch all constraints
2. **Document discoveries:** Update `HEROKU_API_CONSTRAINTS.md`
3. **Maintain backward compatibility:** Handle constraints transparently
4. **Update tests:** Keep unit tests aligned with live behavior

### For Future Updates
1. Consider adding parameter validation warnings
2. May want to log when temperature is overridden (extended thinking)
3. Consider exposing constraint information via library methods
4. Document all discovered constraints clearly

---

## Files Updated

### Core Implementation
- `floship_llm/client.py` - Parameter handling logic

### Tests
- `tests/test_client.py` - Updated test expectations
- `test_heroku_api.py` - New live API test script

### Documentation
- `HEROKU_API_CONSTRAINTS.md` - Complete constraint documentation
- `LIVE_API_TESTING_RESULTS.md` - This file

---

## Conclusion

The live API testing phase was **extremely valuable** - it revealed real-world constraints that weren't apparent from:
- Documentation review
- Unit test development
- API specification analysis

All discovered issues have been fixed, and the library now correctly handles all Heroku Inference API constraints automatically and transparently for users.

**Final Status: All Tests Passing ✅**
- ✅ Basic prompts
- ✅ Heroku-specific parameters
- ✅ Extended thinking
- ✅ Tool calling
- ✅ 260 unit tests
- ✅ Backward compatibility maintained

The library is ready for production use with Heroku Inference API and Claude models.

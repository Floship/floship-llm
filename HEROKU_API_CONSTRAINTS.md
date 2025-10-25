# Heroku Inference API Constraints

This document lists the important constraints discovered when using the Heroku Inference API with Claude models.

## Parameter Constraints

### 1. Temperature vs Top_p Mutual Exclusivity

**Constraint:** Claude models don't allow both `temperature` and `top_p` to be specified simultaneously.

**Error Message:**
```
Error code: 400 - ValidationException: The model returned the following errors:
`temperature` and `top_p` cannot both be specified for this model.
Please use only one.
```

**Solution:** The library automatically handles this by:
- If `top_p` is set, it's used instead of `temperature`
- If `top_p` is not set, `temperature` is used with the default/specified value

**Example:**
```python
# Use temperature (default behavior)
llm = LLM(model="claude-4-5-sonnet", temperature=0.7)

# Use top_p instead (temperature will be omitted)
llm = LLM(model="claude-4-5-sonnet", top_p=0.9)
```

### 2. Extended Thinking Temperature Requirement

**Constraint:** When `extended_thinking` is enabled, `temperature` must be set to 1.0.

**Error Message:**
```
Error code: 400 - ValidationException: The model returned the following errors:
`temperature` may only be set to 1 when thinking is enabled.
Please consult our documentation at https://docs.claude.com/en/docs/build-with-claude/extended-thinking
```

**Solution:** The library automatically sets `temperature=1.0` when extended thinking is enabled, ignoring any user-specified temperature or top_p values.

**Example:**
```python
# Temperature is automatically set to 1.0
llm = LLM(
    model="claude-4-5-sonnet",
    extended_thinking={
        "enabled": True,
        "budget_tokens": 1024,
        "include_reasoning": False
    }
)
```

### 3. Extended Thinking Budget Tokens Minimum

**Constraint:** The `budget_tokens` parameter in `extended_thinking` must be at least 1024.

**Error Message:**
```
Error code: 400 - ValidationException: The model returned the following errors:
thinking.enabled.budget_tokens: Input should be greater than or equal to 1024
```

**Solution:** Always use a minimum of 1024 for `budget_tokens`.

**Example:**
```python
# Correct: budget_tokens >= 1024
llm = LLM(
    model="claude-4-5-sonnet",
    extended_thinking={
        "enabled": True,
        "budget_tokens": 1024,  # Minimum value
        "include_reasoning": False
    }
)

# Incorrect: budget_tokens < 1024
# This will fail with a validation error
```

## OpenAI Client Compatibility

### Heroku-Specific Parameters Must Use extra_body

**Constraint:** The OpenAI Python SDK doesn't natively support Heroku-specific parameters like `top_k`, `top_p`, and `extended_thinking`.

**Error Message:**
```
TypeError: Completions.create() got an unexpected keyword argument 'top_k'
```

**Solution:** The library automatically wraps these parameters in the `extra_body` field when making API calls.

**Internal Implementation:**
```python
# Library handles this automatically
params['extra_body'] = {
    'top_k': self.top_k,
    'top_p': self.top_p,
    'extended_thinking': self.extended_thinking
}
```

**Note:** Users don't need to worry about this - it's handled internally by the library.

## API URL Format

**Constraint:** The Heroku Inference API requires the `/v1` suffix in the base URL.

**Correct URL:**
```python
base_url = "https://us.inference.heroku.com/v1"
```

**Incorrect URL:**
```python
base_url = "https://us.inference.heroku.com"  # Missing /v1
```

## Best Practices

1. **Use either temperature or top_p, not both:** Let the library handle the logic by setting only one.

2. **Extended thinking requirements:**
   - Set `budget_tokens` to at least 1024
   - Don't specify temperature or top_p (library will force temperature=1.0)

3. **URL format:** Always include `/v1` in the base URL

4. **Error handling:** The library's retry handler will catch and report these validation errors clearly.

## Testing

You can verify these constraints work correctly by running:

```bash
python test_heroku_api.py
```

This test suite validates:
- Basic prompts
- Parameter handling (temperature, top_k, top_p)
- Extended thinking with correct constraints
- Tool calling with JSON serialization

All tests should pass with the library automatically handling these constraints.

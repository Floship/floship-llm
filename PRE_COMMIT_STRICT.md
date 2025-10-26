# Strict Pre-Commit Configuration

This document describes the enhanced pre-commit configuration for stricter code quality and security checks.

## What's New

The pre-commit configuration has been significantly enhanced with additional tools and stricter settings:

### New Tools Added

1. **flake8-bugbear** - Finds likely bugs and design problems
2. **flake8-comprehensions** - Improves list/dict comprehensions
3. **flake8-simplify** - Suggests code simplifications
4. **bandit** - Security vulnerability scanner
5. **pydocstyle** - Docstring style checker (Google convention)
7. **mypy** (optional, disabled by default)
   - Type checking with static analysis
   - **Requires**: Type annotations to be effective
   - **Status**: Commented out until codebase has proper type hints
   - **To enable**: Uncomment mypy section in `.pre-commit-config.yaml`
   - Configuration in `pyproject.toml`
7. **pylint** - Comprehensive code quality analyzer
8. **dead** - Dead code detection
9. **safety** - Dependencies vulnerability checker

### Stricter Settings

#### Flake8
- **Before**: Many errors ignored (`D,F401,F841,E722,E402,F541`)
- **After**: Only essential ignores (`E203,W503,E501`)
- **New**: Complexity check (max 10)
- **New**: Per-file ignores for `__init__.py` and tests

#### MyPy
- **Before**: Disabled
- **After**: Enabled with sensible defaults
- Additional checks: redundant casts, unused ignores, no implicit reexport

#### Bandit
- Security scanning enabled for all non-test code
- Skips: `B101` (assert in tests), `B601` (paramiko)

#### Pylint
- Comprehensive code quality checks
- Disabled overly strict rules for practicality
- Good variable names configured

#### Pydocstyle
- Google docstring convention
- Ignores some missing docstrings in reasonable places

## Current Issues Found

When running on the current codebase, the following issues were detected:

### High Priority (Should Fix)

1. **Unused Imports** (`F401`)
   - `floship_llm/client.py:8` - `copy.copy` imported but unused
   - `floship_llm/schemas.py:1` - `typing.Union` imported but unused
   - `floship_llm/tool_manager.py:6` - `typing.Any` imported but unused

2. **Code Simplification** (`SIM910`)
   - Multiple instances of `kwargs.get('key', None)` should be `kwargs.get('key')`
   - `None` is the default return value, no need to specify

3. **Complex Functions** (`C901`)
   - `LLM._handle_tool_calls` - complexity 21 (max 10)
   - `LLM._validate_messages_for_api` - complexity 14 (max 10)
   - `RetryHandler.execute_with_retry` - complexity 11 (max 10)
   - `ToolManager.execute_tool` - complexity 13 (max 10)

4. **Security** (`B307`)
   - `example_stream_with_tools.py:52` - `eval()` usage (already added `nosec` comment)

### Medium Priority (Consider Fixing)

1. **Context Manager Simplification** (`SIM117`)
   - Multiple `with` statements can be combined in tests

2. **Try/Except Simplification** (`SIM105`)
   - Use `contextlib.suppress()` instead of try/except pass

3. **Logic Simplification** (`SIM114`, `SIM222`)
   - Some conditional logic can be simplified

## How to Use

### Install Pre-commit Hooks

```bash
pre-commit install
```

### Run on All Files

```bash
pre-commit run --all-files
```

### Run Specific Hook

```bash
pre-commit run flake8 --all-files
pre-commit run bandit --all-files
pre-commit run mypy --all-files
pre-commit run pylint --all-files
```

### Skip Hooks During Commit

If you need to commit despite failures (not recommended):

```bash
git commit --no-verify
```

### Update Hook Versions

```bash
pre-commit autoupdate
```

## Configuration Files

### `.pre-commit-config.yaml`
Main pre-commit configuration with all hooks.

### `pyproject.toml`
Contains tool-specific configurations:
- `[tool.bandit]` - Bandit settings
- `[tool.pylint.*]` - Pylint settings
- `[tool.pydocstyle]` - Pydocstyle settings
- `[tool.mypy]` - MyPy settings

## Recommended Fixes

### 1. Remove Unused Imports

```bash
# Let pre-commit fix it
pre-commit run --all-files
```

### 2. Simplify kwargs.get()

Replace:
```python
self.top_k = kwargs.get("top_k", None)
```

With:
```python
self.top_k = kwargs.get("top_k")
```

### 3. Reduce Function Complexity

For complex functions:
- Extract helper methods
- Simplify nested conditions
- Use early returns
- Split into smaller functions

Example for `_handle_tool_calls`:
```python
# Before: One large function with complexity 21
def _handle_tool_calls(self, message, response, ...):
    # 100+ lines of complex logic
    ...

# After: Split into smaller functions
def _handle_tool_calls(self, message, response, ...):
    self._build_assistant_message(message)
    self._execute_all_tools(message.tool_calls, recursion_depth)
    return self._get_follow_up_response(recursion_depth, stream_final_response)

def _build_assistant_message(self, message):
    # 10-20 lines
    ...

def _execute_all_tools(self, tool_calls, recursion_depth):
    # 20-30 lines
    ...

def _get_follow_up_response(self, recursion_depth, stream_final_response):
    # 20-30 lines
    ...
```

### 4. Use contextlib.suppress

Replace:
```python
try:
    if len(choice.message.tool_calls) > 0:
        return self._handle_tool_calls(...)
except (TypeError, AttributeError):
    pass
```

With:
```python
from contextlib import suppress

with suppress(TypeError, AttributeError):
    if len(choice.message.tool_calls) > 0:
        return self._handle_tool_calls(...)
```

## Disabling Specific Checks

If you need to disable a check for a specific line:

### Flake8
```python
result = some_complex_function()  # noqa: C901
```

### Bandit
```python
result = eval(expression)  # nosec B307
```

### Pylint
```python
result = something()  # pylint: disable=too-many-arguments
```

### MyPy
```python
result = something()  # type: ignore[arg-type]
```

## Benefits

1. **Better Code Quality**: Catches bugs before they reach production
2. **Security**: Identifies potential security vulnerabilities
3. **Consistency**: Enforces consistent code style
4. **Documentation**: Ensures proper docstrings
5. **Type Safety**: Static type checking with mypy
6. **Maintainability**: Reduces complexity and improves readability
7. **Dependencies**: Alerts about vulnerable dependencies

## Gradual Adoption

If the strict checks are too overwhelming:

1. **Start with auto-fixes**: Run tools that auto-fix issues (black, isort, trailing whitespace)
2. **Fix unused imports**: Easy wins with immediate benefit
3. **Address security issues**: High priority for production code
4. **Reduce complexity gradually**: Refactor one function at a time
5. **Add type hints incrementally**: Start with public APIs

## Performance

Pre-commit hooks add ~10-30 seconds to commit time (first run only, cached after that).

To speed up:
- Run only fast hooks during commit
- Run comprehensive checks in CI/CD
- Use `--no-verify` for quick WIP commits (commit proper fixes later)

## Next Steps

1. Review and fix unused imports (quick win)
2. Add `nosec` comments for legitimate security exceptions
3. Plan refactoring for complex functions
4. Update dev dependencies to include new tools
5. Run pre-commit in CI/CD pipeline
6. Document any permanent exceptions in this file

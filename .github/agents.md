# Coding Agent Instructions for floship-llm

These instructions apply to AI coding agents (GitHub Copilot, Copilot Workspace, etc.) working in this repository.

## Repository Overview

`floship-llm` is a Python library for interacting with OpenAI-compatible LLM inference endpoints (Heroku Inference API). It includes CloudFront WAF sanitization, streaming support, tool/function-calling, retry logic, and embeddings.

### Key Directories

- `floship_llm/` — library source code
- `tests/` — pytest test suite
- `scripts/` — utility scripts (e.g., `check_version.py`)
- `examples*.py` — standalone usage examples

### Important Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Package metadata, version, dependencies |
| `floship_llm/__init__.py` | Public API exports, `__version__` |
| `floship_llm/client.py` | Core `LLM` client, `CloudFrontWAFSanitizer` |
| `floship_llm/schemas.py` | Pydantic/dataclass models |
| `floship_llm/tool_manager.py` | Tool/function-calling management |
| `floship_llm/content_processor.py` | Response content processing |
| `floship_llm/retry_handler.py` | Retry logic with backoff |
| `floship_llm/utils.py` | JSON utilities |
| `scripts/check_version.py` | Pre-commit hook: validates version consistency |
| `.pre-commit-config.yaml` | Pre-commit hook configuration |

## Version Bump Procedure (MANDATORY)

When bumping the version, **every** step below is required. Do not skip any.

### Files to Update (same commit)

1. **`pyproject.toml`** — `version = "X.Y.Z"`
2. **`floship_llm/__init__.py`** — `__version__ = "X.Y.Z"`
3. **`tests/test_init.py`** — update **both** version assertion strings in `test_version_is_string` and `test_version_accessible_from_module`
4. **`CHANGELOG.md`** — add a `## [X.Y.Z] - YYYY-MM-DD` section under `[Unreleased]` with a summary of changes
5. **`uv.lock`** — run `uv lock` to sync the lock file with `pyproject.toml`

### After Committing and Pushing

6. **Create and push the git tag:**
   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

> **Every version on `main` must have a corresponding `vX.Y.Z` tag pushed to the remote.** Never skip the tag.

### Verification

Run `python3 scripts/check_version.py` to verify all version strings are consistent before committing. The pre-commit hook runs this automatically.

## Pre-commit Hooks

This repo uses pre-commit hooks. Key notes:

- Use `python3` (not `python`) — there is no bare `python` binary on the target systems.
- All hooks must pass before a commit is accepted.

### Hook Summary

| Hook | What it does |
|------|-------------|
| `detect-secrets` | Blocks commits containing secrets (uses `.secrets.baseline`) |
| `detect-private-key` | Blocks commits containing private keys |
| `check-added-large-files` | Blocks files > 500 KB |
| `trailing-whitespace` | Auto-trims trailing whitespace |
| `end-of-file-fixer` | Auto-fixes missing final newlines |
| `check-yaml` / `check-json` / `check-toml` | Validates config file syntax |
| `check-merge-conflict` | Blocks unresolved merge conflict markers |
| `check-case-conflict` | Blocks case-only filename collisions |
| `mixed-line-ending` | Enforces LF line endings |
| `check-ast` | Validates Python syntax |
| `check-docstring-first` | Ensures docstring is first statement |
| `debug-statements` | Blocks `pdb`, `breakpoint()`, etc. |
| `name-tests-test` | Enforces `test_` prefix on test files |
| `ruff` | Lints and auto-fixes Python (replaces flake8/pylint/isort) |
| `ruff-format` | Formats Python (replaces black) |
| `bandit` | Security scan (excludes `tests/`) |
| `mypy` | Type checking on `floship_llm/` |
| `version-bump-reminder` | ⚠️ Warns when `floship_llm/` changes without version bump |
| `check-version-consistency` | Blocks if version strings are out of sync |
| `uv-lock-check` | Blocks if `uv.lock` is stale vs `pyproject.toml` |
| `uv-sync-on-toml-change` | Auto-runs `uv lock` when `pyproject.toml` changes |

## Testing

- Run tests with: `pytest tests/`
- Python 3.9 is the minimum supported version for tests.
- When adding new functionality, always add corresponding tests.
- WAF sanitization tests are in `tests/test_waf_protection.py` and `tests/test_cloudfront_waf.py`.

## CloudFront WAF Sanitization

The `CloudFrontWAFSanitizer` class in `client.py` handles patterns that trigger AWS CloudFront WAF rules (SQL injection, XSS, command injection, etc.). When adding new sanitization rules:

1. Add the pattern to the appropriate category in `CloudFrontWAFSanitizer.BLOCKERS`
2. Add reverse mappings in the `desanitize()` method so original content is restored
3. Test both `sanitize()` and `desanitize()` roundtrip
4. Be careful with regex ordering — patterns can interact (use negative lookaheads to avoid matching already-sanitized content)

## Code Style

- Formatter/linter: **ruff** (not black)
- Type checker: **mypy**
- All code must pass `ruff check` and `ruff format --check`

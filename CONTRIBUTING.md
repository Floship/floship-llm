# Contributing to Floship LLM

Thank you for your interest in contributing to the Floship LLM library!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/Floship/floship-llm.git
cd floship-llm
```

2. Install in development mode:
```bash
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/
```

## Code Style

We use `black` for code formatting:

```bash
black floship_llm/
```

## Type Checking

We use `mypy` for type checking:

```bash
mypy floship_llm/
```

## Making Changes

1. Create a new branch for your feature or bug fix:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and ensure tests pass

3. Commit your changes with a descriptive message:
```bash
git commit -m "Add feature: description"
```

4. Push to GitHub and create a pull request

## Versioning

We follow [Semantic Versioning](https://semver.org/):
- MAJOR version for incompatible API changes
- MINOR version for backwards-compatible functionality additions
- PATCH version for backwards-compatible bug fixes

### Version Bump Checklist

When bumping the version, **all** of the following must be updated in the same commit:

1. **`pyproject.toml`** — `version = "X.Y.Z"`
2. **`floship_llm/__init__.py`** — `__version__ = "X.Y.Z"`
3. **`tests/test_init.py`** — update both version assertions (`test_version_is_string` and `test_version_accessible_from_module`)
4. **`CHANGELOG.md`** — add a new `## [X.Y.Z] - YYYY-MM-DD` section under `[Unreleased]`
5. **`uv.lock`** — run `uv lock` (or `uv sync`) to keep the lock file in sync with `pyproject.toml`
6. **Git tag** — after committing and pushing, create and push the tag:
   ```bash
   git tag vX.Y.Z
   git push origin vX.Y.Z
   ```

The `check-version-consistency` pre-commit hook (`scripts/check_version.py`) validates that `pyproject.toml`, `__init__.py`, `CHANGELOG.md`, and `test_init.py` all agree. The `uv-lock-check` hook ensures `uv.lock` stays in sync.

> **Never skip the tag.** Every version on `main` must have a corresponding `vX.Y.Z` tag pushed to the remote.

## Questions?

Open an issue or reach out to the team at dev@floship.com

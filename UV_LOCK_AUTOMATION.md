# UV Lock File Automation

## Overview

Automated pre-commit hooks to ensure `uv.lock` stays synchronized with `pyproject.toml` changes.

## Problem Solved

Previously, developers would occasionally forget to run `uv sync` after updating `pyproject.toml`, leading to:
- Outdated lock file in the repository
- Inconsistent dependencies across development environments
- CI/CD failures due to dependency mismatches

## Solution: Pre-commit Hooks

Two automated checks were added to `.pre-commit-config.yaml`:

### 1. Lock File Sync Check

```yaml
- id: uv-lock-check
  name: Check uv.lock is in sync with pyproject.toml
  entry: uv lock --check
  language: system
  files: ^pyproject\.toml$
  pass_filenames: false
```

**What it does:**
- Validates that `uv.lock` is in sync with `pyproject.toml`
- Fails the commit if the lock file is out of date
- Only runs when `pyproject.toml` is being committed

### 2. Auto-sync on Changes

```yaml
- id: uv-sync-on-toml-change
  name: Auto-sync uv.lock when pyproject.toml changes
  entry: bash -c 'uv lock && git add uv.lock'
  language: system
  files: ^pyproject\.toml$
  pass_filenames: false
  stages: [pre-commit]
```

**What it does:**
- Automatically runs `uv lock` when `pyproject.toml` changes
- Auto-stages the updated `uv.lock` file
- Ensures commits always include synchronized lock files

## Usage

### Normal Development

When you modify `pyproject.toml`:

```bash
# Edit pyproject.toml
vim pyproject.toml

# Add your changes
git add pyproject.toml

# Commit - hooks will automatically run
git commit -m "Update dependencies"
```

The hooks will:
1. Detect `pyproject.toml` changes
2. Run `uv lock` to update `uv.lock`
3. Stage the updated `uv.lock`
4. Complete the commit with both files

### Manual Override

If you need to skip the hooks (not recommended):

```bash
git commit --no-verify -m "Your commit message"
```

## Verification

Test the hooks manually:

```bash
# Check if lock file is in sync
uv lock --check

# Update lock file manually
uv sync

# Run all pre-commit hooks
pre-commit run --all-files

# Run specific hook
pre-commit run uv-lock-check
pre-commit run uv-sync-on-toml-change
```

## Benefits

1. **Automatic**: No need to remember to run `uv sync`
2. **Consistent**: Lock file always matches `pyproject.toml`
3. **CI-friendly**: Prevents lock file drift before pushing
4. **Developer-friendly**: Happens automatically during normal workflow

## Python Version Update

As part of implementing these hooks, the minimum Python version was updated:

**Before:** `requires-python = ">=3.8"`  
**After:** `requires-python = ">=3.8.1"`

**Reason:** flake8 >=7.0.0 requires Python >=3.8.1, causing dependency resolution conflicts with Python 3.8.0.

## Related Commands

```bash
# Install pre-commit hooks for the first time
pre-commit install

# Update all hooks to latest versions
pre-commit autoupdate

# Run hooks on all files
pre-commit run --all-files

# Check lock file status
uv lock --check

# Sync dependencies
uv sync

# Update dependencies and lock file
uv sync --upgrade
```

## Troubleshooting

### Hook fails with "uv not found"

Install uv:
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or using pipx
pipx install uv
```

### Lock file out of sync

Run manually:
```bash
uv sync
git add uv.lock
```

### Pre-commit not installed

Install it:
```bash
pip install pre-commit
pre-commit install
```

## Implementation Date

- **Version:** 0.5.0
- **Commit:** 23802a0
- **Date:** October 30, 2025
- **Author:** Floship Development Team

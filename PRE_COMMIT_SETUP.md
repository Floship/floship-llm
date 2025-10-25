# Pre-commit Hooks Setup

This project uses [pre-commit](https://pre-commit.com/) to automatically check code quality and security before every commit.

## What Gets Checked

### 🔒 Security Checks (CRITICAL)
- **detect-secrets**: Scans for API keys, tokens, and other credentials
- **detect-private-keys**: Checks for private SSH/SSL keys
- **check-added-large-files**: Prevents accidentally committing large files

### 🎨 Code Formatting (AUTO-FIX)
- **Black**: Python code formatter (opinionated, consistent style)
- **isort**: Import statement sorter (compatible with Black)
- **trailing-whitespace**: Removes trailing whitespace
- **end-of-file-fixer**: Ensures files end with newline

### ✅ Code Quality
- **flake8**: Python linting for syntax and logic errors
- **check-ast**: Validates Python syntax
- **check-yaml/json/toml**: Validates config file syntax
- **check-merge-conflict**: Detects merge conflict markers
- **debug-statements**: Catches leftover debug code

## Installation

Pre-commit is already installed if you installed the dev dependencies:

```bash
pip install pre-commit
```

Or install from requirements:

```bash
pip install -r requirements.txt  # If you have one
```

## Setup

Install the git hooks (one-time setup):

```bash
pre-commit install
```

This installs hooks that run automatically before every `git commit`.

## Usage

### Automatic (Recommended)

Once installed, pre-commit runs automatically when you commit:

```bash
git add .
git commit -m "Your commit message"
# Pre-commit checks run automatically
```

If checks fail:
- Some tools (Black, isort, whitespace) will auto-fix issues
- You need to `git add` the fixed files again
- Then commit again

### Manual

Run checks on all files:

```bash
pre-commit run --all-files
```

Run checks on specific files:

```bash
pre-commit run --files floship_llm/client.py
```

Run a specific hook:

```bash
pre-commit run detect-secrets --all-files
pre-commit run black --all-files
```

## What Happens When You Commit

```bash
$ git commit -m "Add new feature"
🔒 Detect secrets........................................................Passed
🔑 Detect private keys...................................................Passed
📦 Check for large files.................................................Passed
✂️  Trim trailing whitespace.............................................Passed
📝 Fix end of files......................................................Passed
⬛ Format with Black.....................................................Passed
📦 Sort imports with isort...............................................Passed
🔍 Lint with flake8......................................................Passed
[main abc1234] Add new feature
 2 files changed, 45 insertions(+)
```

## If a Check Fails

### Credentials Detected 🔒

```
🔒 Detect secrets........................................................Failed
```

**What to do:**
1. **STOP!** Do not commit credentials
2. Remove the credential from the file
3. Use environment variables instead
4. Update `.secrets.baseline` if it's a false positive

### Formatting Issues (Auto-Fixed) 🎨

```
⬛ Format with Black.....................................................Failed
- files were modified by this hook
```

**What to do:**
1. Tools like Black auto-fix the files
2. Review the changes: `git diff`
3. Add the fixed files: `git add .`
4. Commit again: `git commit -m "..."`

### Linting Errors 🔍

```
🔍 Lint with flake8......................................................Failed
- exit code: 1

floship_llm/client.py:42:1: E999 SyntaxError: invalid syntax
```

**What to do:**
1. Fix the errors manually
2. Add the fixed files: `git add .`
3. Commit again: `git commit -m "..."`

## Bypassing Hooks (Not Recommended)

If you absolutely need to bypass pre-commit:

```bash
git commit --no-verify -m "Emergency fix"
```

⚠️ **WARNING**: Only use this in emergencies! You're skipping security checks!

## Configuration Files

- `.pre-commit-config.yaml` - Pre-commit configuration
- `.flake8` - Flake8 linting rules
- `pyproject.toml` - Black and isort configuration
- `.secrets.baseline` - Detect-secrets baseline (tracks known secrets)

## Updating Hooks

Update to latest versions:

```bash
pre-commit autoupdate
```

## Secrets Baseline Management

If detect-secrets flags a false positive:

```bash
# Update the baseline to include the flagged item
detect-secrets scan --baseline .secrets.baseline

# Add to git
git add .secrets.baseline
```

## Troubleshooting

### Hooks not running

```bash
# Reinstall hooks
pre-commit uninstall
pre-commit install
```

### Clear hook cache

```bash
pre-commit clean
pre-commit install --install-hooks
```

### Skip specific hooks

```bash
# Skip detect-secrets for one commit
SKIP=detect-secrets git commit -m "message"

# Skip multiple hooks
SKIP=detect-secrets,flake8 git commit -m "message"
```

## CI/CD Integration

Pre-commit can also run in CI/CD:

```yaml
# GitHub Actions example
- name: Run pre-commit
  run: pre-commit run --all-files
```

## Best Practices

1. **Run pre-commit before pushing**: `pre-commit run --all-files`
2. **Never commit credentials**: Use environment variables
3. **Keep hooks updated**: Run `pre-commit autoupdate` monthly
4. **Review auto-fixes**: Always check what Black/isort changed
5. **Don't bypass checks**: Use `--no-verify` only in emergencies

## Security Notes

The most important checks are security-related:
- **detect-secrets**: Prevents committing API keys, passwords, tokens
- **detect-private-keys**: Prevents committing SSH/SSL private keys

These run FIRST and will block your commit if secrets are detected.

**If you accidentally commit secrets:**
1. Rotate the credentials immediately (consider them compromised)
2. Remove from git history: see `SECURITY.md`
3. Update `.gitignore` to prevent future commits

## Getting Help

- Pre-commit docs: https://pre-commit.com/
- Black docs: https://black.readthedocs.io/
- Flake8 docs: https://flake8.pycqa.org/
- Detect-secrets: https://github.com/Yelp/detect-secrets

Report issues in the project repository.

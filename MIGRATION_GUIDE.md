# LLM Integration Migration Guide

## Overview

The LLM integration has been successfully extracted into a standalone, reusable library called `floship-llm`. This allows you to use the same LLM client in other projects without Django dependencies.

## What Was Done

### 1. Created Standalone Library

A new Python package was created at `/Users/rawgeek/Sites/floship/floship-llm/` with the following structure:

```
floship-llm/
├── floship_llm/
│   ├── __init__.py       # Package exports
│   ├── client.py         # Core LLM client (moved from llm/client.py)
│   ├── schemas.py        # Pydantic schemas (moved from llm/schemas.py)
│   └── utils.py          # JSON utilities (moved from llm/utils.py)
├── pyproject.toml        # Modern Python packaging config
├── requirements.txt      # Library dependencies
├── README.md            # Comprehensive documentation
├── LICENSE              # MIT License
└── .gitignore           # Python gitignore
```

### 2. Core Components Extracted

The following files were extracted as they have **no Django dependencies**:
- `client.py` - Main LLM client class
- `schemas.py` - Pydantic models for structured responses
- `utils.py` - JSON parsing utilities

### 3. Project-Specific Components Kept

The following files remain in the support-app's `llm/` directory as they have Django/project-specific dependencies:
- `agent.py` - PR and JIRA ticket summarizers (uses GitHub, JIRA clients)
- `jira.py` - Floship JIRA integration utilities
- `slack.py` - Slack thread syncing utilities
- `tasks.py` - Threaded background tasks for PR updates
- `helpers.py` - Django-dependent helper functions
- `config.json` - Project-specific configuration

### 4. Updated Imports

The support-app now imports from the library:

**Before:**
```python
from llm.client import LLM
from llm.schemas import Labels
from llm.utils import lm_json_utils
```

**After (both work due to backward compatibility):**
```python
# Option 1: Direct from library
from floship_llm import LLM, Labels, lm_json_utils

# Option 2: Via llm module (backward compatible)
from llm import LLM, Labels, lm_json_utils
```

## Using the Library in Other Projects

### Installation

**For local development:**
```bash
pip install -e /Users/rawgeek/Sites/floship/floship-llm
```

**With pipenv:**
```bash
pipenv install -e /Users/rawgeek/Sites/floship/floship-llm
```

**In requirements.txt:**
```
-e /Users/rawgeek/Sites/floship/floship-llm
```

### Basic Usage

```python
from floship_llm import LLM

# Set environment variables
# INFERENCE_URL - Your LLM API endpoint
# INFERENCE_MODEL_ID - Model identifier  
# INFERENCE_KEY - API key

# Create client
llm = LLM(
    type='completion',
    temperature=0.7,
    continuous=False
)

# Generate response
response = llm.prompt("Your question here")
print(response)
```

### With Structured Output

```python
from floship_llm import LLM, Labels

llm = LLM(
    type='completion',
    response_format=Labels,  # Pydantic model
    continuous=False
)

response = llm.prompt("Generate labels for this ticket")
print(response.labels)  # Validated structure
```

## Environment Variables Required

The library requires these environment variables:

```bash
export INFERENCE_URL="https://api.openai.com/v1"
export INFERENCE_MODEL_ID="gpt-4"
export INFERENCE_KEY="sk-..."
export INFERENCE_SUPPORTS_PARALLEL_REQUESTS="True"  # Optional, default: True
```

## Features of the Library

✅ **Zero Django Dependencies** - Can be used in any Python project  
✅ **OpenAI Compatible** - Works with any OpenAI-compatible API  
✅ **Structured Output** - Pydantic schema validation  
✅ **Conversation History** - Multi-turn conversations  
✅ **JSON Parsing** - Robust JSON extraction from LLM responses  
✅ **Configurable** - Temperature, penalties, token limits  
✅ **Well Documented** - Comprehensive README with examples

## Publishing the Library (Future)

To publish to PyPI:

1. Update version in `pyproject.toml`
2. Build the package:
   ```bash
   cd /Users/rawgeek/Sites/floship/floship-llm
   python -m build
   ```
3. Upload to PyPI:
   ```bash
   python -m twine upload dist/*
   ```

Then install from PyPI:
```bash
pip install floship-llm
```

## Testing the Migration

All imports have been tested and verified:
- ✅ Direct library imports work
- ✅ Backward compatible imports through `llm` module work
- ✅ No errors in the codebase
- ✅ Library successfully installed in support-app

## Next Steps

1. **Test the application** - Run the Django app and verify all LLM features work
2. **Create Git repository** - Initialize a Git repo for the floship-llm library
3. **Add tests** - Write unit tests for the library
4. **Set up CI/CD** - Add GitHub Actions for testing and publishing
5. **Use in other projects** - Install and use in other Floship projects

## Support

For questions or issues with the library:
- Check the README: `/Users/rawgeek/Sites/floship/floship-llm/README.md`
- Review the source code in `floship_llm/` directory
- The library is simple and well-documented for easy customization

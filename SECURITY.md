# Security Best Practices for floship-llm

## Never Commit Credentials!

**CRITICAL**: Never commit API keys, tokens, or any credentials to git repositories.

### Protected by .gitignore

The following patterns are automatically ignored:
- `.env` and `.env.*` files
- `*_credentials.py`
- `*_secrets.py`
- `*_keys.py`
- `credentials.json` and `secrets.json`
- `*.key` and `*.pem` files

### Using Environment Variables

Always use environment variables for sensitive data:

```bash
# Set credentials as environment variables
export INFERENCE_KEY='your-api-key-here'
export INFERENCE_URL='https://us.inference.heroku.com/v1'
export INFERENCE_MODEL_ID='claude-4-5-sonnet'

# Then run your script
python your_script.py
```

### For Development

Create a `.env` file (automatically ignored by git):

```bash
# .env file
INFERENCE_KEY=your-api-key-here
INFERENCE_URL=https://us.inference.heroku.com/v1
INFERENCE_MODEL_ID=claude-4-5-sonnet
```

Load it using python-dotenv:

```python
from dotenv import load_dotenv
load_dotenv()

from floship_llm import LLM
llm = LLM()  # Uses environment variables
```

### Testing with Heroku API

Use the provided test script:

```bash
# Set your credentials
export INFERENCE_KEY='your-key'
export INFERENCE_URL='https://us.inference.heroku.com/v1'
export INFERENCE_MODEL_ID='claude-4-5-sonnet'

# Run tests
python test_heroku_api.py
```

The test script will refuse to run if credentials are not set as environment variables.

### If You Accidentally Commit Credentials

1. **Rotate the credentials immediately** - consider them compromised
2. Contact your admin to revoke/regenerate the keys
3. Remove from git history (we can help with this)
4. Force push to overwrite remote history

### Production Deployment

Use proper secret management:
- **Heroku**: Config Vars in dashboard
- **AWS**: Secrets Manager or Parameter Store
- **Kubernetes**: Secrets
- **Docker**: Build-time args or runtime env vars

Never hardcode credentials in source code!

## Reporting Security Issues

If you discover a security vulnerability, please email security@floship.com instead of using the issue tracker.

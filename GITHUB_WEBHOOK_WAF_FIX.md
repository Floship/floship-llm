# GitHub Webhook WAF Issue Resolution

**Date:** November 3, 2025  
**Issue:** CloudFront WAF 403 errors when processing GitHub webhook payloads  
**Status:** ✅ Fixed

## Problem

GitHub webhook payloads contain URL template patterns that were triggering CloudFront WAF blocking:

```python
'following_url': 'https://api.github.com/users/Floship/following{/other_user}'
'gists_url': 'https://api.github.com/users/Rawgeek/gists{/gist_id}'
'starred_url': 'https://api.github.com/users/Rawgeek/starred{/owner}{/repo}'
'keys_url': 'https://api.github.com/repos/Floship/Shipping/keys{/key_id}'
'collaborators_url': 'https://api.github.com/repos/Floship/Shipping/collaborators{/collaborator}'
```

### Error Log

```
2025-11-03T04:00:50.692944+00:00 app[web.1]: 2025-11-03 04:00:50,692 [INFO] HTTP Request: POST https://us.inference.heroku.com/v1/chat/completions "HTTP/1.1 403 Forbidden"
2025-11-03T04:00:51.698471+00:00 app[web.1]: 2025-11-03 04:00:51,698 [INFO] HTTP Request: POST https://us.inference.heroku.com/v1/chat/completions "HTTP/1.1 403 Forbidden"
2025-11-03T04:00:53.703224+00:00 app[web.1]: 2025-11-03 04:00:53,703 [INFO] HTTP Request: POST https://us.inference.heroku.com/v1/chat/completions "HTTP/1.1 403 Forbidden"

openai.PermissionDeniedError: <!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN" "http://www.w3.org/TR/html4/loose.dtd">
<HTML><HEAD><META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=iso-8859-1">
<TITLE>ERROR: The request could not be satisfied</TITLE>
</HEAD><BODY>
<H1>403 ERROR</H1>
<H2>The request could not be satisfied.</H2>
<HR noshade size="1px">
Request blocked.
```

### Root Cause

CloudFront WAF was interpreting the URL template patterns `{/variable}` as potentially malicious patterns, especially when combined with the large payload size of GitHub webhooks containing nested repository and user objects.

## Solution

Added a new sanitization pattern to `CloudFrontWAFSanitizer`:

```python
"url_templates": [
    # GitHub API URL templates like {/other_user}, {/gist_id}, etc.
    (r"\{/[^}]+\}", "[URL_TEMPLATE]"),
],
```

### How It Works

**Before Sanitization:**
```python
'following_url': 'https://api.github.com/users/Floship/following{/other_user}'
'gists_url': 'https://api.github.com/users/Rawgeek/gists{/gist_id}'
'starred_url': 'https://api.github.com/users/Rawgeek/starred{/owner}{/repo}'
```

**After Sanitization:**
```python
'following_url': 'https://api.github.com/users/Floship/following[URL_TEMPLATE]'
'gists_url': 'https://api.github.com/users/Rawgeek/gists[URL_TEMPLATE]'
'starred_url': 'https://api.github.com/users/Rawgeek/starred[URL_TEMPLATE][URL_TEMPLATE]'
```

### Benefits

1. **Preserves Base URLs:** The actual API endpoints remain intact
2. **Removes WAF Triggers:** Template placeholders replaced with safe markers
3. **Maintains Context:** LLM can still understand the URL structure
4. **Automatic Protection:** Works transparently for all GitHub webhook processing

## Testing

Added comprehensive tests to verify the fix:

### Test 1: Basic URL Template Sanitization
```python
def test_github_webhook_url_templates(self):
    """Test GitHub API URL templates that trigger CloudFront WAF."""
    content = """
    'following_url': 'https://api.github.com/users/Floship/following{/other_user}'
    'gists_url': 'https://api.github.com/users/Rawgeek/gists{/gist_id}'
    'starred_url': 'https://api.github.com/users/Rawgeek/starred{/owner}{/repo}'
    """
    sanitized, was_sanitized = CloudFrontWAFSanitizer.sanitize(content)
    
    assert was_sanitized
    assert "{/" not in sanitized
    assert "https://api.github.com" in sanitized
    assert "[URL_TEMPLATE]" in sanitized
```

### Test 2: Realistic GitHub Webhook Payload
```python
def test_github_webhook_realistic_payload(self):
    """Test realistic GitHub webhook payload that triggers CloudFront WAF."""
    content = """
    User: {'login': 'Rawgeek', 'id': 1498478, 
           'followers_url': 'https://api.github.com/users/Rawgeek/followers',
           'following_url': 'https://api.github.com/users/Rawgeek/following{/other_user}',
           'gists_url': 'https://api.github.com/users/Rawgeek/gists{/gist_id}'}
    Repo: {'keys_url': 'https://api.github.com/repos/Floship/Shipping/keys{/key_id}',
           'branches_url': 'https://api.github.com/repos/Floship/Shipping/branches{/branch}'}
    """
    sanitized, was_sanitized = CloudFrontWAFSanitizer.sanitize(content)
    
    assert was_sanitized
    assert "{/other_user}" not in sanitized
    assert "{/gist_id}" not in sanitized
    assert "https://api.github.com/users/Rawgeek" in sanitized
```

**Test Results:** ✅ Both tests passing, all 330 tests passing

## Implementation Details

### Files Modified

1. **floship_llm/client.py**
   - Added `url_templates` category to `CloudFrontWAFSanitizer.BLOCKERS`
   - Pattern: `r"\{/[^}]+\}"` → `"[URL_TEMPLATE]"`

2. **tests/test_waf_protection.py**
   - Added `test_github_webhook_url_templates()`
   - Added `test_github_webhook_realistic_payload()`

3. **CLOUDFRONT_WAF_SUMMARY.md**
   - Updated sanitization list to include URL templates

4. **CHANGELOG.md**
   - Added fix to [Unreleased] section

### Commit Hash
```
254498f - Fix CloudFront WAF blocking on GitHub webhook URL templates
```

## Deployment

### For Existing Users

No changes required! The fix is automatic:

```python
from floship_llm import LLM

llm = LLM()  # WAF protection enabled by default

# GitHub webhook payload will be automatically sanitized
response = llm.prompt(github_webhook_payload)
```

### Version Requirements

- **Minimum Version:** 0.5.1 (unreleased) or latest commit from main branch
- **Current Version:** 0.5.0 + commit 254498f

### Upgrade Path

```bash
# Install from git (includes fix)
pip install git+https://github.com/Floship/floship-llm.git@main

# Or wait for 0.5.1 release
pip install floship-llm>=0.5.1
```

## Impact Analysis

### Before Fix

- ❌ GitHub webhook PR updates failing with 403 errors
- ❌ Manual retry required
- ❌ Poor user experience
- ❌ Lost productivity

### After Fix

- ✅ GitHub webhook processing works automatically
- ✅ No manual intervention needed
- ✅ Seamless PR description generation
- ✅ Improved reliability

### Production Verification

Monitor these metrics after deployment:

```python
llm = LLM()

# After processing GitHub webhooks
metrics = llm.get_waf_metrics()

print(f"Sanitization rate: {metrics['sanitization_rate']:.1%}")
print(f"403 error rate: {metrics['cloudfront_403_rate']:.1%}")
print(f"Retry success rate: {metrics['retry_success_rate']:.1%}")
```

Expected results:
- Sanitization rate: Increases (GitHub webhooks now sanitized)
- 403 error rate: Decreases significantly
- Retry success rate: Near 100%

## Related Patterns

The same sanitization approach can be extended for other template patterns:

### Angular/Vue Templates
```python
"template_patterns": [
    (r"\{\{[^}]+\}\}", "[TEMPLATE_VAR]"),  # {{ variable }}
    (r"\[\[[^]]+\]\]", "[TEMPLATE_VAR]"),  # [[ variable ]]
]
```

### Jinja2/Django Templates
```python
"template_patterns": [
    (r"\{%[^%]+%\}", "[TEMPLATE_TAG]"),    # {% tag %}
    (r"\{\{[^}]+\}\}", "[TEMPLATE_VAR]"),  # {{ variable }}
]
```

## Lessons Learned

1. **WAF Rules are Strict:** Even legitimate API URL templates can trigger blocking
2. **Test with Real Data:** Realistic GitHub webhook payloads essential for testing
3. **Preserve Context:** Sanitization should maintain semantic meaning for LLM
4. **Monitor Metrics:** Track sanitization patterns to identify new issues early

## Future Improvements

1. **Pattern Library:** Build comprehensive library of known WAF-triggering patterns
2. **Custom Patterns:** Allow users to add application-specific sanitization rules
3. **Smart Detection:** Analyze 403 responses to automatically detect new patterns
4. **Reporting Dashboard:** Visualize WAF sanitization metrics over time

## References

- **Original Issue:** Heroku logs from 2025-11-03 04:00:48 UTC
- **GitHub Issue:** Pull request #2801 processing failure
- **Related Documentation:** 
  - [CloudFront WAF Summary](CLOUDFRONT_WAF_SUMMARY.md)
  - [GitHub API Documentation](https://docs.github.com/en/rest)
  - [CloudFront WAF Rules](https://docs.aws.amazon.com/waf/latest/developerguide/waf-chapter.html)

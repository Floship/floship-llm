"""
Example demonstrating CloudFront WAF protection in floship-llm.

This example shows how the library automatically sanitizes content to prevent
CloudFront Web Application Firewall (WAF) from blocking requests containing
patterns that resemble security attacks.
"""

import os

from floship_llm import LLM, LLMConfig

# Ensure environment variables are set
if not all(
    [
        os.getenv("INFERENCE_URL"),
        os.getenv("INFERENCE_MODEL_ID"),
        os.getenv("INFERENCE_KEY"),
    ]
):
    print("Please set INFERENCE_URL, INFERENCE_MODEL_ID, and INFERENCE_KEY")
    print("\nExample:")
    print('export INFERENCE_URL="https://us.inference.heroku.com"')
    print('export INFERENCE_MODEL_ID="claude-4-sonnet"')
    print('export INFERENCE_KEY="your-api-key"')
    exit(1)


def example_1_basic_waf_protection():
    """Example 1: Basic WAF protection (default behavior)."""
    print("=" * 80)
    print("Example 1: Basic WAF Protection")
    print("=" * 80)

    # WAF protection is enabled by default
    llm = LLM()

    # Content with path traversal patterns (would normally trigger 403)
    suspicious_content = """
    Please analyze this file structure:
    - ../../config/database.yml
    - ../../secrets/api_keys.txt
    - ../../../etc/passwd
    """

    print("\nOriginal content (would trigger CloudFront WAF):")
    print(suspicious_content)

    print("\n🛡️  Sending with automatic WAF protection...")
    try:
        response = llm.prompt(suspicious_content)
        print(f"\n✅ Success! Response received:")
        print(f"{response[:200]}...")
    except Exception as e:
        print(f"\n❌ Error: {e}")

    # Check metrics
    metrics = llm.get_waf_metrics()
    print(f"\n📊 Metrics:")
    print(f"   Total requests: {metrics['total_requests']}")
    print(f"   Sanitization rate: {metrics['sanitization_rate']:.1%}")


def example_2_pr_diff_review():
    """Example 2: PR diff with XSS patterns."""
    print("\n\n" + "=" * 80)
    print("Example 2: PR Diff Review (XSS Patterns)")
    print("=" * 80)

    llm = LLM()

    # PR diff containing XSS patterns that would trigger WAF
    pr_diff = """
    diff --git a/../../src/auth.py b/../../src/auth.py
    --- a/../../src/auth.py
    +++ b/../../src/auth.py
    @@ -15,8 +15,12 @@
    -    return '<script>alert("vulnerability")</script>'
    -    return '<iframe src="malicious.com"></iframe>'
    +    return sanitize_html(content)
    +    return escape_html(content)
    """

    print("\nPR diff (contains path traversal and XSS patterns):")
    print(pr_diff)

    print("\n🛡️  Sending with automatic WAF protection...")
    try:
        response = llm.prompt(f"Review this security fix:\n{pr_diff}")
        print(f"\n✅ Success! Response received:")
        print(f"{response[:200]}...")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def example_3_custom_config():
    """Example 3: Custom WAF configuration."""
    print("\n\n" + "=" * 80)
    print("Example 3: Custom WAF Configuration")
    print("=" * 80)

    # Create custom configuration
    config = LLMConfig(
        enable_waf_sanitization=True,
        max_waf_retries=3,  # Allow 3 retries on 403
        debug_mode=True,  # Enable detailed logging
        log_sanitization=True,  # Log when content is sanitized
        log_blockers=True,  # Log which patterns were found
    )

    llm = LLM(waf_config=config)

    content = 'File path: ../../config/secrets.json with <script>alert("xss")</script>'

    print(f"\nContent: {content}")
    print("\n🛡️  Sending with debug logging enabled...")

    try:
        response = llm.prompt(f"Explain what's in this file:\n{content}")
        print(f"\n✅ Success! Response received:")
        print(f"{response[:150]}...")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def example_4_disable_waf():
    """Example 4: Disable WAF protection (not recommended)."""
    print("\n\n" + "=" * 80)
    print("Example 4: Disable WAF Protection (Not Recommended)")
    print("=" * 80)

    # Disable WAF protection
    llm = LLM(enable_waf_sanitization=False)

    print("\n⚠️  WAF protection is DISABLED")
    print("⚠️  Content will be sent without sanitization")
    print("⚠️  May result in 403 errors if content contains suspicious patterns\n")

    # Safe content (no suspicious patterns)
    safe_content = "What is the capital of France?"

    print(f"Content: {safe_content}")
    print("\n📤 Sending without WAF protection...")

    try:
        response = llm.prompt(safe_content)
        print(f"\n✅ Success! Response received:")
        print(f"{response[:150]}...")
    except Exception as e:
        print(f"\n❌ Error: {e}")


def example_5_metrics_tracking():
    """Example 5: Track sanitization metrics."""
    print("\n\n" + "=" * 80)
    print("Example 5: Metrics Tracking")
    print("=" * 80)

    llm = LLM()

    # Send multiple requests with varying content
    requests = [
        ("Normal content", "What is Python?"),
        ("Path traversal", "Check ../../config/app.py"),
        ("XSS pattern", "Review this: <script>alert(1)</script>"),
        ("Mixed patterns", 'File ../../app.js with <iframe src="x"></iframe>'),
        ("Clean code", "def hello(): return 'world'"),
    ]

    print("\nSending 5 requests with different content types...\n")

    for label, content in requests:
        print(f"📤 {label}:", end=" ")
        try:
            llm.prompt(content)
            print("✅")
        except Exception as e:
            print(f"❌ {str(e)[:50]}...")

    # Get final metrics
    metrics = llm.get_waf_metrics()

    print("\n📊 Final Metrics:")
    print(f"   Total requests: {metrics['total_requests']}")
    print(f"   Sanitized requests: {metrics['sanitization_rate']:.1%}")
    print(f"   Error rate: {metrics['error_rate']:.1%}")
    if metrics["cloudfront_403_rate"] > 0:
        print(f"   CloudFront 403 rate: {metrics['cloudfront_403_rate']:.1%}")


def example_6_sanitization_patterns():
    """Example 6: Show what gets sanitized."""
    print("\n\n" + "=" * 80)
    print("Example 6: Sanitization Patterns")
    print("=" * 80)

    from floship_llm.client import CloudFrontWAFSanitizer

    examples = [
        ("Path traversal", "../../config/settings.py"),
        ("XSS script tag", "<script>alert('xss')</script>"),
        ("XSS iframe", '<iframe src="malicious.com"></iframe>'),
        ("JavaScript protocol", 'href="javascript:void(0)"'),
        ("Event handler", '<img onerror="alert(1)">'),
        ("Mixed patterns", '../../app.js with <script>alert("x")</script>'),
    ]

    print("\nSanitization Examples:")
    print("-" * 80)

    for label, original in examples:
        sanitized, was_sanitized = CloudFrontWAFSanitizer.sanitize(original)
        print(f"\n{label}:")
        print(f"  Original:  {original}")
        print(f"  Sanitized: {sanitized}")
        print(f"  Changed:   {'Yes' if was_sanitized else 'No'}")


if __name__ == "__main__":
    print("\n")
    print("🛡️  CloudFront WAF Protection Examples")
    print("=" * 80)
    print("These examples demonstrate automatic content sanitization to prevent")
    print("CloudFront WAF from blocking requests with suspicious patterns.")
    print("=" * 80)

    try:
        # Run examples
        example_1_basic_waf_protection()
        example_2_pr_diff_review()
        example_3_custom_config()
        example_4_disable_waf()
        example_5_metrics_tracking()
        example_6_sanitization_patterns()

        print("\n\n" + "=" * 80)
        print("✅ All examples completed successfully!")
        print("=" * 80)

    except KeyboardInterrupt:
        print("\n\n⚠️  Examples interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error running examples: {e}")
        import traceback

        traceback.print_exc()

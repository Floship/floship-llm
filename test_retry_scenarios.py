#!/usr/bin/env python3
"""
Comprehensive test for retry scenarios including 403, 429, 500, and connection errors.
"""
import os
import sys
from unittest.mock import Mock, patch

from openai import APIConnectionError, APIStatusError, RateLimitError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from floship_llm import LLM


def setup_llm():
    """Set up a test LLM instance."""
    os.environ["INFERENCE_URL"] = "https://test.example.com"
    os.environ["INFERENCE_MODEL_ID"] = "test-model"
    os.environ["INFERENCE_KEY"] = "test-key"
    return LLM()


def create_success_response():
    """Create a mock success response."""
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Success!"
    mock_response.choices[0].message.tool_calls = None
    return mock_response


def test_retry_on_403():
    """Test retry on 403 Forbidden."""
    print("\n" + "=" * 60)
    print("TEST: Retry on 403 Forbidden")
    print("=" * 60)

    llm = setup_llm()
    call_count = 0

    def mock_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count <= 2:
            print(f"  Attempt {call_count}: Raising 403 error")
            raise APIStatusError(
                message="Request blocked",
                response=Mock(status_code=403),
                body={"error": "Request blocked"},
            )
        else:
            print(f"  Attempt {call_count}: Success!")
            return create_success_response()

    with patch.object(llm.client.chat.completions, "create", side_effect=mock_create):
        with patch("time.sleep"):
            result = llm.prompt("Test prompt")
            assert call_count == 3
            print(f"✅ PASSED: Retried 3 times total, succeeded on attempt 3\n")


def test_retry_on_429():
    """Test retry on 429 Rate Limit."""
    print("\n" + "=" * 60)
    print("TEST: Retry on 429 Rate Limit")
    print("=" * 60)

    llm = setup_llm()
    call_count = 0

    def mock_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            print(f"  Attempt {call_count}: Raising RateLimitError")
            raise RateLimitError(
                message="Rate limit exceeded",
                response=Mock(status_code=429),
                body={"error": "Rate limit exceeded"},
            )
        else:
            print(f"  Attempt {call_count}: Success!")
            return create_success_response()

    with patch.object(llm.client.chat.completions, "create", side_effect=mock_create):
        with patch("time.sleep"):
            result = llm.prompt("Test prompt")
            assert call_count == 2
            print(f"✅ PASSED: Retried and succeeded on attempt 2\n")


def test_retry_on_500():
    """Test retry on 500 Server Error."""
    print("\n" + "=" * 60)
    print("TEST: Retry on 500 Server Error")
    print("=" * 60)

    llm = setup_llm()
    call_count = 0

    def mock_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            print(f"  Attempt {call_count}: Raising 500 error")
            raise APIStatusError(
                message="Internal server error",
                response=Mock(status_code=500),
                body={"error": "Internal server error"},
            )
        else:
            print(f"  Attempt {call_count}: Success!")
            return create_success_response()

    with patch.object(llm.client.chat.completions, "create", side_effect=mock_create):
        with patch("time.sleep"):
            result = llm.prompt("Test prompt")
            assert call_count == 2
            print(f"✅ PASSED: Retried and succeeded on attempt 2\n")


def test_retry_on_connection_error():
    """Test retry on connection errors."""
    print("\n" + "=" * 60)
    print("TEST: Retry on Connection Error")
    print("=" * 60)

    llm = setup_llm()
    call_count = 0

    def mock_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            print(f"  Attempt {call_count}: Raising APIConnectionError")
            raise APIConnectionError(message="Connection failed", request=Mock())
        else:
            print(f"  Attempt {call_count}: Success!")
            return create_success_response()

    with patch.object(llm.client.chat.completions, "create", side_effect=mock_create):
        with patch("time.sleep"):
            result = llm.prompt("Test prompt")
            assert call_count == 2
            print(f"✅ PASSED: Retried and succeeded on attempt 2\n")


def test_max_retries_exceeded():
    """Test that after max retries, error is raised."""
    print("\n" + "=" * 60)
    print("TEST: Max Retries Exceeded")
    print("=" * 60)

    llm = setup_llm()
    call_count = 0

    def mock_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        print(f"  Attempt {call_count}: Raising 403 error")
        raise APIStatusError(
            message="Request blocked",
            response=Mock(status_code=403),
            body={"error": "Request blocked"},
        )

    with patch.object(llm.client.chat.completions, "create", side_effect=mock_create):
        with patch("time.sleep"):
            try:
                result = llm.prompt("Test prompt")
                assert False, "Should have raised an exception"
            except APIStatusError:
                assert call_count == 3
                print(f"✅ PASSED: Failed after 3 attempts as expected\n")


def test_non_retryable_error():
    """Test that non-retryable errors (400) fail immediately."""
    print("\n" + "=" * 60)
    print("TEST: Non-Retryable Error (400)")
    print("=" * 60)

    llm = setup_llm()
    call_count = 0

    def mock_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        print(f"  Attempt {call_count}: Raising 400 error")
        raise APIStatusError(
            message="Bad request",
            response=Mock(status_code=400),
            body={"error": "Bad request"},
        )

    with patch.object(llm.client.chat.completions, "create", side_effect=mock_create):
        with patch("time.sleep"):
            try:
                result = llm.prompt("Test prompt")
                assert False, "Should have raised an exception"
            except APIStatusError:
                assert call_count == 1  # No retry on 400
                print(f"✅ PASSED: Failed immediately on 400 error (no retry)\n")


def test_immediate_success():
    """Test that successful calls don't retry."""
    print("\n" + "=" * 60)
    print("TEST: Immediate Success (No Retry)")
    print("=" * 60)

    llm = setup_llm()
    call_count = 0

    def mock_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        print(f"  Attempt {call_count}: Success!")
        return create_success_response()

    with patch.object(llm.client.chat.completions, "create", side_effect=mock_create):
        result = llm.prompt("Test prompt")
        assert call_count == 1
        print(f"✅ PASSED: Succeeded immediately without retry\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("RUNNING COMPREHENSIVE RETRY TESTS")
    print("=" * 60)

    try:
        test_immediate_success()
        test_retry_on_403()
        test_retry_on_429()
        test_retry_on_500()
        test_retry_on_connection_error()
        test_max_retries_exceeded()
        test_non_retryable_error()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✅")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

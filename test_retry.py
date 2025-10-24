#!/usr/bin/env python3
"""
Test script to demonstrate the retry logic for API errors.
"""
import os
import sys
from unittest.mock import Mock, patch
from openai import APIStatusError

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from floship_llm import LLM

def test_retry_on_403():
    """Test that the LLM retries on 403 errors."""
    
    # Set up mock environment
    os.environ['INFERENCE_URL'] = 'https://test.example.com'
    os.environ['INFERENCE_MODEL_ID'] = 'test-model'
    os.environ['INFERENCE_KEY'] = 'test-key'
    
    # Create LLM instance
    llm = LLM()
    
    # Create a mock that fails twice with 403, then succeeds
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Success!"
    mock_response.choices[0].message.tool_calls = None
    
    call_count = 0
    def mock_create(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        
        if call_count <= 2:
            # First two calls fail with 403
            error = APIStatusError(
                message="Request blocked",
                response=Mock(status_code=403),
                body={"error": "Request blocked"}
            )
            print(f"Attempt {call_count}: Raising 403 error")
            raise error
        else:
            # Third call succeeds
            print(f"Attempt {call_count}: Success!")
            return mock_response
    
    # Patch the API call
    with patch.object(llm.client.chat.completions, 'create', side_effect=mock_create):
        # Patch time.sleep to speed up test
        with patch('time.sleep') as mock_sleep:
            try:
                result = llm.prompt("Test prompt")
                print(f"\n✅ Success after {call_count} attempts!")
                print(f"Result: {result}")
                print(f"Sleep was called {mock_sleep.call_count} times")
                
                # Verify we retried
                assert call_count == 3, f"Expected 3 calls, got {call_count}"
                assert mock_sleep.call_count == 2, f"Expected 2 sleeps, got {mock_sleep.call_count}"
                
                print("\n✅ Test passed! Retry logic is working correctly.")
                
            except Exception as e:
                print(f"\n❌ Test failed: {e}")
                raise

if __name__ == '__main__':
    test_retry_on_403()

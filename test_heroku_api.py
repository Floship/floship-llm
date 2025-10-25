"""Test script to verify floship-llm works with Heroku Inference API.

SECURITY NOTE: Never commit actual API credentials to git!
Set your credentials as environment variables before running this script:

    export INFERENCE_KEY='your-api-key-here'
    export INFERENCE_URL='https://us.inference.heroku.com/v1'
    export INFERENCE_MODEL_ID='claude-4-5-sonnet'

Then run:
    python test_heroku_api.py
"""

import os
import sys

# Verify environment variables are set
required_env_vars = ['INFERENCE_KEY', 'INFERENCE_URL', 'INFERENCE_MODEL_ID']
missing_vars = [var for var in required_env_vars if not os.environ.get(var)]

if missing_vars:
    print("❌ ERROR: Missing required environment variables:")
    for var in missing_vars:
        print(f"  - {var}")
    print("\nPlease set them before running this script:")
    print("  export INFERENCE_KEY='your-api-key-here'")
    print("  export INFERENCE_URL='https://us.inference.heroku.com/v1'")
    print("  export INFERENCE_MODEL_ID='claude-4-5-sonnet'")
    sys.exit(1)

print("=" * 80)
print("Heroku Inference API Test")
print("=" * 80)
print(f"URL: {os.environ['INFERENCE_URL']}")
print(f"Model: {os.environ['INFERENCE_MODEL_ID']}")
print("=" * 80)

try:
    from floship_llm import LLM
    print("\n✓ Successfully imported LLM")
    
    # Test 1: Basic prompt
    print("\n" + "=" * 80)
    print("Test 1: Basic Prompt")
    print("=" * 80)
    
    llm = LLM(temperature=0.15)
    print("✓ LLM client created successfully")
    
    response = llm.prompt("Say 'Hello from Heroku Inference API!' and explain in one sentence what you are.")
    print(f"\n✓ Response received:\n{response}")
    
    # Test 2: Heroku-specific parameters
    print("\n" + "=" * 80)
    print("Test 2: Heroku-Specific Parameters")
    print("=" * 80)
    
    # Note: Claude doesn't allow temperature and top_p at the same time
    llm2 = LLM(
        temperature=0.7,
        max_completion_tokens=150,
        top_k=50
    )
    print("✓ LLM client created with Heroku parameters (top_k)")
    
    response2 = llm2.prompt("Write a haiku about AI.")
    print(f"\n✓ Response received:\n{response2}")
    
    # Test with top_p instead of temperature
    llm2b = LLM(
        top_p=0.9,
        max_completion_tokens=150
    )
    print("\n✓ LLM client created with top_p parameter")
    
    response2b = llm2b.prompt("Name three benefits of AI.")
    print(f"\n✓ Response received:\n{response2b}")
    
    # Test 3: Extended thinking (if supported by this model)
    print("\n" + "=" * 80)
    print("Test 3: Extended Thinking")
    print("=" * 80)
    
    llm3 = LLM(
        temperature=0.15,
        extended_thinking={
            "enabled": True,
            "budget_tokens": 1024,  # Minimum required value is 1024
            "include_reasoning": False
        }
    )
    print("✓ LLM client created with extended thinking")
    
    response3 = llm3.prompt("What is 15 * 23?")
    print(f"\n✓ Response received:\n{response3}")
    
    # Test 4: Tool calling
    print("\n" + "=" * 80)
    print("Test 4: Tool Calling")
    print("=" * 80)
    
    from floship_llm.schemas import ToolParameter
    
    llm4 = LLM(enable_tools=True, temperature=0.15)
    print("✓ LLM client created with tools enabled")
    
    def calculate_sum(a: int, b: int) -> int:
        """Calculate the sum of two numbers."""
        result = a + b
        print(f"  → Tool executed: {a} + {b} = {result}")
        return result
    
    llm4.add_tool_from_function(
        calculate_sum,
        name="calculate_sum",
        description="Calculate the sum of two integers",
        parameters=[
            ToolParameter(name="a", type="integer", description="First number", required=True),
            ToolParameter(name="b", type="integer", description="Second number", required=True)
        ]
    )
    print("✓ Tool registered successfully")
    
    response4 = llm4.prompt("What is 42 plus 58? Use the calculate_sum tool.")
    print(f"\n✓ Response received:\n{response4}")
    
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nThe floship-llm library is working correctly with Heroku Inference API!")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

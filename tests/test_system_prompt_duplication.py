import logging
import os
from unittest.mock import Mock, patch

from floship_llm.client import LLM


@patch.dict(
    os.environ,
    {
        "INFERENCE_URL": "https://test.com",
        "INFERENCE_MODEL_ID": "test-model",
        "INFERENCE_KEY": "test-key",
    },
)
def test_continuous_mode_message_duplication(caplog):
    """Test that system and user messages are not duplicated in continuous mode."""
    caplog.set_level(logging.WARNING)

    with patch("floship_llm.client.OpenAI") as mock_openai:
        # Setup mock response
        mock_stream = Mock()
        mock_stream.__iter__ = Mock(return_value=iter([]))
        mock_openai.return_value.chat.completions.create.return_value = mock_stream

        # Initialize LLM in continuous mode
        llm = LLM(continuous=True)

        # 1. Test System Prompt Duplication

        # First prompt with system message
        llm.prompt("First prompt", system="System message")

        # Check messages
        system_messages = [m for m in llm.messages if m["role"] == "system"]
        assert len(system_messages) == 1
        assert system_messages[0]["content"] == "System message"

        # Second prompt with SAME system message
        llm.prompt("Second prompt", system="System message")

        # Check messages - should still be 1
        system_messages = [m for m in llm.messages if m["role"] == "system"]
        assert len(system_messages) == 1

        # Check warning log
        assert "Duplicate system message detected" in caplog.text

        # 2. Test User Prompt Duplication

        # Clear logs
        caplog.clear()

        # Add a user message that already exists ("First prompt")
        llm.add_message("user", "First prompt")

        # Check messages - "First prompt" should appear only once (from the first call)
        user_messages = [
            m
            for m in llm.messages
            if m["role"] == "user" and m["content"] == "First prompt"
        ]
        assert len(user_messages) == 1

        # Check warning log
        assert "Duplicate user message detected" in caplog.text

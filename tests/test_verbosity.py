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
def test_verbosity_logging_enabled(caplog):
    """Test that request data is logged when verbosity is 2."""
    caplog.set_level(logging.DEBUG)

    with patch("floship_llm.client.OpenAI") as mock_openai:
        # Setup mock response
        mock_stream = Mock()
        mock_stream.__iter__ = Mock(return_value=iter([]))
        mock_openai.return_value.chat.completions.create.return_value = mock_stream

        llm = LLM(verbosity=2)
        llm.prompt("Test prompt")

        assert "FULL REQUEST DATA:" in caplog.text
        assert "Params:" in caplog.text
        assert "Messages:" in caplog.text
        assert "Test prompt" in caplog.text


@patch.dict(
    os.environ,
    {
        "INFERENCE_URL": "https://test.com",
        "INFERENCE_MODEL_ID": "test-model",
        "INFERENCE_KEY": "test-key",
    },
)
def test_verbosity_logging_disabled(caplog):
    """Test that request data is NOT logged when verbosity is 0."""
    caplog.set_level(logging.DEBUG)

    with patch("floship_llm.client.OpenAI") as mock_openai:
        # Setup mock response
        mock_stream = Mock()
        mock_stream.__iter__ = Mock(return_value=iter([]))
        mock_openai.return_value.chat.completions.create.return_value = mock_stream

        llm = LLM(verbosity=0)
        llm.prompt("Test prompt")

        assert "FULL REQUEST DATA:" not in caplog.text


@patch.dict(
    os.environ,
    {
        "INFERENCE_URL": "https://test.com",
        "INFERENCE_MODEL_ID": "test-model",
        "INFERENCE_KEY": "test-key",
    },
)
def test_verbosity_logging_embedding(caplog):
    """Test that embedding request data is logged when verbosity is 2."""
    caplog.set_level(logging.DEBUG)

    with patch("floship_llm.client.OpenAI") as mock_openai:
        # Setup mock response
        mock_response = Mock()
        mock_response.data = []
        mock_openai.return_value.embeddings.create.return_value = mock_response

        llm = LLM(verbosity=2, type="embedding")
        llm.embed("Test text")

        assert "FULL EMBEDDING REQUEST DATA:" in caplog.text
        assert "Params:" in caplog.text
        assert "Input:" in caplog.text
        assert "Test text" in caplog.text

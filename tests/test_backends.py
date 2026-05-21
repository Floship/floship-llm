"""Tests for provider backend abstraction layer (Phase 2)."""

import os
from unittest.mock import Mock, patch

import pytest

from floship_llm.backends import OpenAICompatibleBackend, ProviderBackend
from floship_llm.client import LLM

HEROKU_URL = "https://us.inference.heroku.com/v1"
GOOGLE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GENERIC_URL = "https://my-custom-llm.example.com/v1"


class TestProviderBackendABC:
    """Tests for the ProviderBackend abstract base class."""

    def test_cannot_instantiate_directly(self):
        """ProviderBackend is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            ProviderBackend()

    def test_requires_chat_method(self):
        """Subclass must implement chat()."""

        class Incomplete(ProviderBackend):
            def embed(self, **kwargs):
                pass

            @property
            def provider_name(self):
                return "test"

            @property
            def supports_caching(self):
                return False

            @property
            def supports_native_tools(self):
                return False

        with pytest.raises(TypeError):
            Incomplete()

    def test_requires_embed_method(self):
        """Subclass must implement embed()."""

        class Incomplete(ProviderBackend):
            def chat(self, **kwargs):
                pass

            @property
            def provider_name(self):
                return "test"

            @property
            def supports_caching(self):
                return False

            @property
            def supports_native_tools(self):
                return False

        with pytest.raises(TypeError):
            Incomplete()

    def test_count_tokens_default_raises(self):
        """Default count_tokens raises NotImplementedError."""

        class Minimal(ProviderBackend):
            def chat(self, **kwargs):
                pass

            def embed(self, **kwargs):
                pass

            @property
            def provider_name(self):
                return "test"

            @property
            def supports_caching(self):
                return False

            @property
            def supports_native_tools(self):
                return False

        backend = Minimal()
        with pytest.raises(
            NotImplementedError, match="does not support native token counting"
        ):
            backend.count_tokens([])


class TestOpenAICompatibleBackend:
    """Tests for OpenAICompatibleBackend."""

    def test_chat_delegates_to_client(self):
        """chat() proxies to client.chat.completions.create."""
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = "response"
        backend = OpenAICompatibleBackend(client=mock_client, provider="heroku")

        result = backend.chat(
            model="test", messages=[{"role": "user", "content": "hi"}]
        )

        assert result == "response"
        mock_client.chat.completions.create.assert_called_once_with(
            model="test", messages=[{"role": "user", "content": "hi"}]
        )

    def test_embed_delegates_to_client(self):
        """embed() proxies to client.embeddings.create."""
        mock_client = Mock()
        mock_client.embeddings.create.return_value = "embedding_response"
        backend = OpenAICompatibleBackend(client=mock_client, provider="heroku")

        result = backend.embed(model="cohere", input="hello")

        assert result == "embedding_response"
        mock_client.embeddings.create.assert_called_once_with(
            model="cohere", input="hello"
        )

    def test_provider_name(self):
        """provider_name returns the provider string."""
        backend = OpenAICompatibleBackend(client=Mock(), provider="google")
        assert backend.provider_name == "google"

    def test_supports_caching_false(self):
        """OpenAI-compatible backends do not support caching."""
        backend = OpenAICompatibleBackend(client=Mock(), provider="heroku")
        assert backend.supports_caching is False

    def test_supports_native_tools_false(self):
        """OpenAI-compatible backends do not use native tool schema."""
        backend = OpenAICompatibleBackend(client=Mock(), provider="heroku")
        assert backend.supports_native_tools is False

    def test_client_property_getter(self):
        """client property returns the underlying OpenAI client."""
        mock_client = Mock()
        backend = OpenAICompatibleBackend(client=mock_client, provider="heroku")
        assert backend.client is mock_client

    def test_client_property_setter(self):
        """client property setter updates the underlying client."""
        mock_client1 = Mock()
        mock_client2 = Mock()
        backend = OpenAICompatibleBackend(client=mock_client1, provider="heroku")
        backend.client = mock_client2
        assert backend.client is mock_client2

    def test_is_subclass_of_provider_backend(self):
        """OpenAICompatibleBackend is a ProviderBackend."""
        assert issubclass(OpenAICompatibleBackend, ProviderBackend)

    def test_streaming_chat_passes_through(self):
        """Streaming params pass through to the underlying client."""
        mock_client = Mock()
        mock_stream = iter(["chunk1", "chunk2"])
        mock_client.chat.completions.create.return_value = mock_stream
        backend = OpenAICompatibleBackend(client=mock_client, provider="heroku")

        result = backend.chat(model="test", messages=[], stream=True)

        assert result is mock_stream
        mock_client.chat.completions.create.assert_called_once_with(
            model="test", messages=[], stream=True
        )


class TestLLMBackendIntegration:
    """Tests for LLM integration with the backend abstraction."""

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": HEROKU_URL,
            "INFERENCE_KEY": "test-key",
            "INFERENCE_MODEL_ID": "claude-3.5-sonnet",
        },
    )
    def test_heroku_creates_openai_compatible_backend(self):
        """Heroku URL creates an OpenAICompatibleBackend."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
            assert isinstance(llm.backend, OpenAICompatibleBackend)
            assert llm.backend.provider_name == "heroku"

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "test-key",
            "INFERENCE_MODEL_ID": "gemini-2.5-flash",
        },
    )
    def test_google_creates_openai_compatible_backend(self):
        """Google AI URL creates an OpenAICompatibleBackend."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
            assert isinstance(llm.backend, OpenAICompatibleBackend)
            assert llm.backend.provider_name == "google"

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GENERIC_URL,
            "INFERENCE_KEY": "test-key",
            "INFERENCE_MODEL_ID": "llama-3",
        },
    )
    def test_generic_creates_openai_compatible_backend(self):
        """Generic URL creates an OpenAICompatibleBackend."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
            assert isinstance(llm.backend, OpenAICompatibleBackend)
            assert llm.backend.provider_name == "openai_compatible"

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": HEROKU_URL,
            "INFERENCE_KEY": "test-key",
            "INFERENCE_MODEL_ID": "claude-3.5-sonnet",
        },
    )
    def test_backend_client_matches_llm_client(self):
        """backend.client is the same object as llm.client."""
        with patch("floship_llm.client.OpenAI") as mock_openai:
            llm = LLM()
            assert llm.backend.client is llm.client
            assert llm.backend.client is mock_openai.return_value

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": HEROKU_URL,
            "INFERENCE_KEY": "test-key",
            "INFERENCE_MODEL_ID": "claude-3.5-sonnet",
        },
    )
    def test_prompt_uses_backend_chat(self):
        """prompt() routes through backend.chat instead of direct client call."""
        with patch("floship_llm.client.OpenAI") as mock_openai:
            mock_client = mock_openai.return_value

            # Default prompt() uses streaming -- supply an iterable of chunks
            chunks = []
            for char in "Hello!":
                chunk = Mock()
                delta = Mock()
                delta.content = char
                choice = Mock()
                choice.delta = delta
                chunk.choices = [choice]
                chunks.append(chunk)
            mock_client.chat.completions.create.return_value = iter(chunks)

            llm = LLM()
            result = llm.prompt("Hi")

            assert result == "Hello!"
            # The call goes through backend.chat -> client.chat.completions.create
            mock_client.chat.completions.create.assert_called_once()

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": HEROKU_URL,
            "INFERENCE_KEY": "test-key",
            "INFERENCE_MODEL_ID": "cohere-embed-english",
        },
    )
    def test_embed_uses_backend_embed(self):
        """embed() routes through backend.embed instead of direct client call."""
        with patch("floship_llm.client.OpenAI") as mock_openai:
            mock_client = mock_openai.return_value
            mock_embedding = Mock()
            mock_embedding.embedding = [0.1, 0.2, 0.3]
            mock_response = Mock()
            mock_response.data = [mock_embedding]
            mock_client.embeddings.create.return_value = mock_response

            llm = LLM(type="embedding")
            result = llm.embed("hello")

            assert result == [0.1, 0.2, 0.3]
            # The call goes through backend.embed -> client.embeddings.create
            mock_client.embeddings.create.assert_called_once()

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": HEROKU_URL,
            "INFERENCE_KEY": "test-key",
            "INFERENCE_MODEL_ID": "claude-3.5-sonnet",
        },
    )
    def test_backend_count_tokens_raises(self):
        """OpenAI-compatible backend raises NotImplementedError for count_tokens."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
            with pytest.raises(NotImplementedError):
                llm.backend.count_tokens([])

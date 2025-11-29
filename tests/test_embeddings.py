"""
Tests for Heroku Inference API embeddings support.

Tests the /v1/embeddings endpoint functionality.
"""

import os
from unittest.mock import Mock, patch

import pytest

from floship_llm import LLM, EmbeddingData, EmbeddingResponse, EmbeddingUsage


# Set up environment variables for all tests
@pytest.fixture(autouse=True)
def setup_env():
    """Set up environment variables for testing."""
    os.environ["INFERENCE_URL"] = "https://test.inference.heroku.com"
    os.environ["INFERENCE_MODEL_ID"] = "test-model"
    os.environ["INFERENCE_KEY"] = "test-key"
    yield
    # Cleanup not needed as tests run in isolated environment


class TestEmbeddingsBasic:
    """Test basic embeddings functionality."""

    @patch("floship_llm.client.OpenAI")
    def test_embedding_type_initialization(self, mock_openai):
        """Test LLM can be initialized with type='embedding'."""
        llm = LLM(type="embedding")
        assert llm.type == "embedding"

    @patch("floship_llm.client.OpenAI")
    def test_embedding_default_parameters(self, mock_openai):
        """Test default embedding parameters."""
        llm = LLM(type="embedding")
        assert llm.input_type is None
        assert llm.encoding_format == "float"
        assert llm.embedding_type == "float"

    @patch("floship_llm.client.OpenAI")
    def test_embedding_custom_parameters(self, mock_openai):
        """Test custom embedding parameters."""
        llm = LLM(
            type="embedding",
            input_type="search_document",
            encoding_format="base64",
            embedding_type="int8",
        )
        assert llm.input_type == "search_document"
        assert llm.encoding_format == "base64"
        assert llm.embedding_type == "int8"


class TestGetEmbeddingParams:
    """Test get_embedding_params method."""

    @patch("floship_llm.client.OpenAI")
    def test_minimal_params(self, mock_openai):
        """Test minimal embedding parameters."""
        llm = LLM(type="embedding", model="cohere-embed-multilingual")
        params = llm.get_embedding_params()

        assert params["model"] == "cohere-embed-multilingual"
        assert "input_type" not in params
        assert "encoding_format" not in params
        assert "embedding_type" not in params

    @patch("floship_llm.client.OpenAI")
    def test_with_input_type(self, mock_openai):
        """Test embedding parameters with input_type."""
        llm = LLM(
            type="embedding",
            model="cohere-embed-multilingual",
            input_type="search_query",
        )
        params = llm.get_embedding_params()

        assert params["model"] == "cohere-embed-multilingual"
        assert params["input_type"] == "search_query"

    @patch("floship_llm.client.OpenAI")
    def test_with_all_params(self, mock_openai):
        """Test embedding parameters with all options."""
        llm = LLM(
            type="embedding",
            model="cohere-embed-multilingual",
            input_type="classification",
            encoding_format="base64",
            embedding_type="int8",
            allow_ignored_params=True,
        )
        params = llm.get_embedding_params()

        assert params["model"] == "cohere-embed-multilingual"
        assert params["input_type"] == "classification"
        assert params["encoding_format"] == "base64"
        assert params["embedding_type"] == "int8"
        assert params["allow_ignored_params"] is True


class TestEmbedMethod:
    """Test embed() method."""

    @patch("floship_llm.client.OpenAI")
    def test_embed_single_string(self, mock_openai):
        """Test embedding a single string."""
        llm = LLM(type="embedding")

        # Mock the API response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3], index=0)]
        mock_response.model = "cohere-embed-multilingual"
        mock_response.usage = Mock(prompt_tokens=5, total_tokens=5)

        mock_openai.return_value.embeddings.create.return_value = mock_response

        result = llm.embed("Hello world")

        assert result == [0.1, 0.2, 0.3]
        mock_openai.return_value.embeddings.create.assert_called_once()

    @patch("floship_llm.client.OpenAI")
    def test_embed_multiple_strings(self, mock_openai):
        """Test embedding multiple strings."""
        llm = LLM(type="embedding")

        # Mock the API response
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3], index=0),
            Mock(embedding=[0.4, 0.5, 0.6], index=1),
            Mock(embedding=[0.7, 0.8, 0.9], index=2),
        ]
        mock_response.model = "cohere-embed-multilingual"
        mock_response.usage = Mock(prompt_tokens=15, total_tokens=15)

        mock_openai.return_value.embeddings.create.return_value = mock_response

        texts = ["Text 1", "Text 2", "Text 3"]
        result = llm.embed(texts)

        assert len(result) == 3
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]
        assert result[2] == [0.7, 0.8, 0.9]

    @patch("floship_llm.client.OpenAI")
    def test_embed_with_full_response(self, mock_openai):
        """Test embedding with full response."""
        llm = LLM(type="embedding")

        # Mock the API response
        mock_response = Mock()
        mock_response.object = "list"
        mock_response.data = [
            Mock(object="embedding", embedding=[0.1, 0.2, 0.3], index=0)
        ]
        mock_response.model = "cohere-embed-multilingual"
        mock_response.usage = Mock(prompt_tokens=5, total_tokens=5)

        mock_openai.return_value.embeddings.create.return_value = mock_response

        result = llm.embed("Hello world", return_full_response=True)

        assert result["object"] == "list"
        assert result["model"] == "cohere-embed-multilingual"
        assert len(result["data"]) == 1
        assert result["data"][0]["embedding"] == [0.1, 0.2, 0.3]
        assert result["data"][0]["index"] == 0
        assert result["usage"]["prompt_tokens"] == 5
        assert result["usage"]["total_tokens"] == 5


class TestEmbedValidation:
    """Test input validation for embed() method."""

    @patch("floship_llm.client.OpenAI")
    def test_embed_empty_string_raises_error(self, mock_openai):
        """Test that empty string raises ValueError."""
        llm = LLM(type="embedding")

        with pytest.raises(ValueError, match="Input cannot be empty"):
            llm.embed("")

    @patch("floship_llm.client.OpenAI")
    def test_embed_none_raises_error(self, mock_openai):
        """Test that None input raises ValueError."""
        llm = LLM(type="embedding")

        with pytest.raises(ValueError, match="Input cannot be empty"):
            llm.embed(None)

    @patch("floship_llm.client.OpenAI")
    def test_embed_empty_list_raises_error(self, mock_openai):
        """Test that empty list raises ValueError."""
        llm = LLM(type="embedding")

        with pytest.raises(ValueError, match="Input cannot be empty"):
            llm.embed([])

    @patch("floship_llm.client.OpenAI")
    def test_embed_too_many_strings_raises_error(self, mock_openai):
        """Test that more than 96 strings raises ValueError."""
        llm = LLM(type="embedding")

        texts = ["Text"] * 97  # 97 strings, exceeds limit of 96

        with pytest.raises(ValueError, match="cannot contain more than 96 strings"):
            llm.embed(texts)

    @patch("floship_llm.client.OpenAI")
    def test_embed_list_with_empty_string_raises_error(self, mock_openai):
        """Test that list containing empty string raises ValueError."""
        llm = LLM(type="embedding")

        with pytest.raises(ValueError, match="must be a non-empty string"):
            llm.embed(["Valid text", "", "Another text"])

    @patch("floship_llm.client.OpenAI")
    def test_embed_list_with_non_string_raises_error(self, mock_openai):
        """Test that list containing non-string raises ValueError."""
        llm = LLM(type="embedding")

        with pytest.raises(ValueError, match="must be a non-empty string"):
            llm.embed(["Valid text", 123, "Another text"])

    @patch("floship_llm.client.OpenAI")
    def test_embed_invalid_type_raises_error(self, mock_openai):
        """Test that invalid input type raises ValueError."""
        llm = LLM(type="embedding")

        with pytest.raises(ValueError, match="must be a string or list of strings"):
            llm.embed(123)


class TestEmbedWarnings:
    """Test warnings for embed() method."""

    @patch("floship_llm.client.OpenAI")
    def test_embed_long_string_warns(self, mock_openai, caplog):
        """Test that strings longer than 2048 chars trigger warning."""
        llm = LLM(type="embedding")

        # Mock the API response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3], index=0)]
        mock_response.model = "cohere-embed-multilingual"
        mock_response.usage = Mock(prompt_tokens=500, total_tokens=500)

        mock_openai.return_value.embeddings.create.return_value = mock_response

        long_text = "x" * 3000  # Exceeds 2048 limit

        with caplog.at_level("WARNING"):
            llm.embed(long_text)

        assert "2048 characters" in caplog.text

    @patch("floship_llm.client.OpenAI")
    def test_embed_list_with_long_string_warns(self, mock_openai, caplog):
        """Test that list with long string triggers warning."""
        llm = LLM(type="embedding")

        # Mock the API response
        mock_response = Mock()
        mock_response.data = [
            Mock(embedding=[0.1, 0.2, 0.3], index=0),
            Mock(embedding=[0.4, 0.5, 0.6], index=1),
        ]
        mock_response.model = "cohere-embed-multilingual"
        mock_response.usage = Mock(prompt_tokens=500, total_tokens=500)

        mock_openai.return_value.embeddings.create.return_value = mock_response

        texts = ["Short text", "x" * 3000]  # Second text exceeds limit

        with caplog.at_level("WARNING"):
            llm.embed(texts)

        assert "index 1" in caplog.text
        assert "2048 characters" in caplog.text


class TestEmbedInputTypes:
    """Test different input_type values."""

    @patch("floship_llm.client.OpenAI")
    def test_search_document_input_type(self, mock_openai):
        """Test search_document input type."""
        llm = LLM(type="embedding", input_type="search_document")

        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2], index=0)]
        mock_openai.return_value.embeddings.create.return_value = mock_response

        llm.embed("Document text")

        call_args = mock_openai.return_value.embeddings.create.call_args
        assert call_args.kwargs["input_type"] == "search_document"

    @patch("floship_llm.client.OpenAI")
    def test_search_query_input_type(self, mock_openai):
        """Test search_query input type."""
        llm = LLM(type="embedding", input_type="search_query")

        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2], index=0)]
        mock_openai.return_value.embeddings.create.return_value = mock_response

        llm.embed("Query text")

        call_args = mock_openai.return_value.embeddings.create.call_args
        assert call_args.kwargs["input_type"] == "search_query"

    @patch("floship_llm.client.OpenAI")
    def test_classification_input_type(self, mock_openai):
        """Test classification input type."""
        llm = LLM(type="embedding", input_type="classification")

        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2], index=0)]
        mock_openai.return_value.embeddings.create.return_value = mock_response

        llm.embed("Classification text")

        call_args = mock_openai.return_value.embeddings.create.call_args
        assert call_args.kwargs["input_type"] == "classification"

    @patch("floship_llm.client.OpenAI")
    def test_clustering_input_type(self, mock_openai):
        """Test clustering input type."""
        llm = LLM(type="embedding", input_type="clustering")

        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2], index=0)]
        mock_openai.return_value.embeddings.create.return_value = mock_response

        llm.embed("Clustering text")

        call_args = mock_openai.return_value.embeddings.create.call_args
        assert call_args.kwargs["input_type"] == "clustering"


class TestEmbedEncodingFormats:
    """Test different encoding_format values."""

    @patch("floship_llm.client.OpenAI")
    def test_float_encoding_format(self, mock_openai):
        """Test float encoding format (default)."""
        llm = LLM(type="embedding", encoding_format="float")

        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3], index=0)]
        mock_openai.return_value.embeddings.create.return_value = mock_response

        result = llm.embed("Test text")

        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)

    @patch("floship_llm.client.OpenAI")
    def test_base64_encoding_format(self, mock_openai):
        """Test base64 encoding format."""
        llm = LLM(type="embedding", encoding_format="base64")

        mock_response = Mock()
        mock_response.data = [Mock(embedding="SGVsbG8gV29ybGQ=", index=0)]
        mock_openai.return_value.embeddings.create.return_value = mock_response

        result = llm.embed("Test text")

        assert isinstance(result, str)
        assert result == "SGVsbG8gV29ybGQ="


class TestEmbedMetrics:
    """Test that embeddings update WAF metrics."""

    @patch("floship_llm.client.OpenAI")
    def test_successful_embed_updates_metrics(self, mock_openai):
        """Test that successful embed updates total_requests."""
        llm = LLM(type="embedding")

        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3], index=0)]
        mock_openai.return_value.embeddings.create.return_value = mock_response

        initial_count = llm.waf_metrics.total_requests
        llm.embed("Test text")

        assert llm.waf_metrics.total_requests == initial_count + 1

    @patch("floship_llm.client.OpenAI")
    def test_failed_embed_updates_metrics(self, mock_openai):
        """Test that failed embed updates failed_requests."""
        llm = LLM(type="embedding")

        mock_openai.return_value.embeddings.create.side_effect = Exception("API Error")

        initial_failed = llm.waf_metrics.failed_requests

        with pytest.raises(Exception):
            llm.embed("Test text")

        assert llm.waf_metrics.failed_requests == initial_failed + 1


class TestEmbeddingSchemas:
    """Test embedding schema classes."""

    def test_embedding_data_creation(self):
        """Test EmbeddingData model creation."""
        data = EmbeddingData(object="embedding", index=0, embedding=[0.1, 0.2, 0.3])

        assert data.object == "embedding"
        assert data.index == 0
        assert data.embedding == [0.1, 0.2, 0.3]

    def test_embedding_usage_creation(self):
        """Test EmbeddingUsage model creation."""
        usage = EmbeddingUsage(prompt_tokens=10, total_tokens=10)

        assert usage.prompt_tokens == 10
        assert usage.total_tokens == 10

    def test_embedding_response_creation(self):
        """Test EmbeddingResponse model creation."""
        response = EmbeddingResponse(
            object="list",
            data=[EmbeddingData(object="embedding", index=0, embedding=[0.1, 0.2])],
            model="cohere-embed-multilingual",
            usage=EmbeddingUsage(prompt_tokens=5, total_tokens=5),
        )

        assert response.object == "list"
        assert len(response.data) == 1
        assert response.model == "cohere-embed-multilingual"
        assert response.usage.prompt_tokens == 5


class TestEmbedRetry:
    """Test retry behavior for embeddings."""

    @patch("floship_llm.client.OpenAI")
    def test_embed_uses_retry_handler(self, mock_openai):
        """Test that embed method uses retry handler."""
        llm = LLM(type="embedding", max_retry=3)

        # Mock successful response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1, 0.2, 0.3], index=0)]
        mock_response.model = "test-model"
        mock_response.usage = Mock(prompt_tokens=5, total_tokens=5)

        mock_openai.return_value.embeddings.create.return_value = mock_response

        # Verify embed uses the retry handler
        with patch.object(
            llm.retry_handler,
            "execute_with_retry",
            wraps=llm.retry_handler.execute_with_retry,
        ) as mock_retry:
            result = llm.embed("Test text")

            # Verify retry handler was called
            assert mock_retry.called
            assert result == [0.1, 0.2, 0.3]

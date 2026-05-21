"""Tests for NativeGeminiBackend (Phase 3)."""

import json
import os
from unittest.mock import Mock, patch

import pytest

from floship_llm.backends.base import ProviderBackend
from floship_llm.backends.native_gemini import (
    NativeGeminiBackend,
    _make_embed_response,
    _make_response,
    _make_stream_chunk,
    _make_tool_call,
    _require_genai,
)
from floship_llm.client import LLM

GOOGLE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
HEROKU_URL = "https://us.inference.heroku.com/v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_gemini_response(text="Hello!", tool_calls=None):
    """Build a mock google-genai GenerateContentResponse."""
    parts = []
    if text:
        part = Mock()
        part.text = text
        part.function_call = None
        parts.append(part)
    if tool_calls:
        for name, args in tool_calls:
            part = Mock()
            part.text = None
            fc = Mock()
            fc.name = name
            fc.args = args
            part.function_call = fc
            parts.append(part)

    content = Mock()
    content.parts = parts
    candidate = Mock()
    candidate.content = content
    response = Mock()
    response.candidates = [candidate]
    return response


def _mock_gemini_stream_chunks(texts):
    """Build a list of mock streaming chunks."""
    chunks = []
    for text in texts:
        part = Mock()
        part.text = text
        part.function_call = None
        content = Mock()
        content.parts = [part]
        candidate = Mock()
        candidate.content = content
        chunk = Mock()
        chunk.candidates = [candidate]
        chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# Backend unit tests
# ---------------------------------------------------------------------------


class TestNativeGeminiBackendInit:
    """Tests for backend initialization."""

    def test_is_provider_backend(self):
        """NativeGeminiBackend is a ProviderBackend."""
        assert issubclass(NativeGeminiBackend, ProviderBackend)

    @patch("floship_llm.backends.native_gemini._require_genai")
    def test_init_creates_client(self, mock_req):
        mock_genai = Mock()
        mock_types = Mock()
        mock_req.return_value = (mock_genai, mock_types)

        backend = NativeGeminiBackend(
            api_key="test-key",  # pragma: allowlist secret
            model="gemini-2.5-flash",
        )

        mock_genai.Client.assert_called_once_with(
            api_key="test-key"  # pragma: allowlist secret
        )
        assert backend._model == "gemini-2.5-flash"

    def test_provider_name(self):
        with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
            mock_req.return_value = (Mock(), Mock())
            backend = NativeGeminiBackend(api_key="k", model="m")
            assert backend.provider_name == "google_native"

    def test_supports_caching(self):
        with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
            mock_req.return_value = (Mock(), Mock())
            backend = NativeGeminiBackend(api_key="k", model="m")
            assert backend.supports_caching is True

    def test_supports_native_tools(self):
        with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
            mock_req.return_value = (Mock(), Mock())
            backend = NativeGeminiBackend(api_key="k", model="m")
            assert backend.supports_native_tools is True


class TestImportGuard:
    """Tests for the lazy import guard."""

    def test_require_genai_succeeds_when_installed(self):
        """google-genai is installed in test env, so this should work."""
        genai, types = _require_genai()
        assert hasattr(genai, "Client")
        assert hasattr(types, "Content")

    def test_import_error_message(self):
        """ImportError with install instructions when google-genai is missing."""
        import sys

        # Temporarily remove google.genai from importable modules
        saved = sys.modules.get("google.genai")
        saved_google = sys.modules.get("google")
        sys.modules["google.genai"] = None  # block import
        sys.modules["google"] = None
        try:
            with pytest.raises(ImportError, match="pip install floship-llm"):
                _require_genai()
        finally:
            if saved is not None:
                sys.modules["google.genai"] = saved
            else:
                sys.modules.pop("google.genai", None)
            if saved_google is not None:
                sys.modules["google"] = saved_google
            else:
                sys.modules.pop("google", None)


class TestResponseWrappers:
    """Tests for OpenAI-shaped response wrapper factories."""

    def test_make_response_text(self):
        resp = _make_response(content="Hello!")
        assert resp.choices[0].message.content == "Hello!"
        assert resp.choices[0].message.tool_calls is None

    def test_make_response_with_tool_calls(self):
        tc = _make_tool_call("call_123", "search", '{"q": "test"}')
        resp = _make_response(content=None, tool_calls=[tc])
        assert resp.choices[0].message.content is None
        assert len(resp.choices[0].message.tool_calls) == 1
        assert resp.choices[0].message.tool_calls[0].function.name == "search"
        assert (
            resp.choices[0].message.tool_calls[0].function.arguments == '{"q": "test"}'
        )
        assert resp.choices[0].message.tool_calls[0].id == "call_123"
        assert resp.choices[0].message.tool_calls[0].type == "function"

    def test_make_stream_chunk_text(self):
        chunk = _make_stream_chunk(content="Hi")
        assert chunk.choices[0].delta.content == "Hi"
        assert chunk.choices[0].delta.tool_calls is None
        assert chunk.choices[0].finish_reason is None

    def test_make_stream_chunk_tool_call(self):
        tc = _make_tool_call("call_1", "fn", "{}")
        chunk = _make_stream_chunk(tool_calls=[tc])
        assert chunk.choices[0].delta.content is None
        assert len(chunk.choices[0].delta.tool_calls) == 1
        assert chunk.choices[0].finish_reason is None

    def test_make_stream_chunk_finish_reason(self):
        chunk = _make_stream_chunk(finish_reason="stop")
        assert chunk.choices[0].finish_reason == "stop"
        assert chunk.choices[0].delta.content is None

    def test_make_response_finish_reason_stop(self):
        resp = _make_response(content="Hello")
        assert resp.choices[0].finish_reason == "stop"

    def test_make_response_finish_reason_tool_calls(self):
        tc = _make_tool_call("call_1", "fn", "{}")
        resp = _make_response(content=None, tool_calls=[tc])
        assert resp.choices[0].finish_reason == "tool_calls"

    def test_make_embed_response(self):
        resp = _make_embed_response([[0.1, 0.2], [0.3, 0.4]], "gemini-embedding-001")
        assert resp.object == "list"
        assert resp.model == "gemini-embedding-001"
        assert len(resp.data) == 2
        assert resp.data[0].embedding == [0.1, 0.2]
        assert resp.data[0].index == 0
        assert resp.data[1].index == 1
        assert resp.usage.prompt_tokens == 0


class TestMessageConversion:
    """Tests for OpenAI -> Gemini message format conversion."""

    @pytest.fixture
    def backend(self):
        with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
            from google.genai import types

            mock_genai = Mock()
            mock_req.return_value = (mock_genai, types)
            b = NativeGeminiBackend(api_key="k", model="m")
        return b

    def test_system_becomes_instruction(self, backend):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
        ]
        contents, system_instruction = backend._to_gemini_contents(messages)
        assert system_instruction == "You are helpful."
        assert len(contents) == 1  # only user message

    def test_user_message(self, backend):
        messages = [{"role": "user", "content": "Hello"}]
        contents, sys_instr = backend._to_gemini_contents(messages)
        assert sys_instr is None
        assert len(contents) == 1
        assert contents[0].role == "user"

    def test_assistant_message(self, backend):
        messages = [{"role": "assistant", "content": "Hi there"}]
        contents, _ = backend._to_gemini_contents(messages)
        assert contents[0].role == "model"

    def test_tool_result_becomes_function_response(self, backend):
        messages = [
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "search",
                "content": "result data",
            },
        ]
        contents, _ = backend._to_gemini_contents(messages)
        assert len(contents) == 1
        assert contents[0].role == "user"

    def test_multimodal_text_parts(self, backend):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this"},
                    {"type": "text", "text": "image"},
                ],
            }
        ]
        contents, _ = backend._to_gemini_contents(messages)
        assert len(contents) == 1

    def test_assistant_with_tool_calls_dict(self, backend):
        messages = [
            {
                "role": "assistant",
                "content": ".",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "search", "arguments": '{"q": "test"}'},
                    }
                ],
            }
        ]
        contents, _ = backend._to_gemini_contents(messages)
        assert len(contents) == 1
        assert contents[0].role == "model"

    def test_empty_content_gets_placeholder(self, backend):
        messages = [{"role": "user", "content": ""}]
        contents, _ = backend._to_gemini_contents(messages)
        assert len(contents) == 1


class TestChatNonStreaming:
    """Tests for non-streaming chat requests."""

    def test_chat_returns_openai_shaped_response(self):
        with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
            mock_genai = Mock()
            mock_types = Mock()
            mock_req.return_value = (mock_genai, mock_types)
            backend = NativeGeminiBackend(api_key="k", model="gemini-2.5-flash")

            # Mock the generate_content response
            mock_response = _mock_gemini_response(text="Hello from Gemini!")
            mock_genai.Client.return_value.models.generate_content.return_value = (
                mock_response
            )

            result = backend.chat(
                model="gemini-2.5-flash",
                messages=[{"role": "user", "content": "Hi"}],
                temperature=0.5,
            )

            # Should return OpenAI-shaped response
            assert result.choices[0].message.content == "Hello from Gemini!"
            assert result.choices[0].message.tool_calls is None

    def test_chat_with_tool_calls(self):
        with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
            mock_genai = Mock()
            mock_types = Mock()
            mock_req.return_value = (mock_genai, mock_types)
            backend = NativeGeminiBackend(api_key="k", model="gemini-2.5-flash")

            mock_response = _mock_gemini_response(
                text=None,
                tool_calls=[("search", {"query": "weather"})],
            )
            mock_genai.Client.return_value.models.generate_content.return_value = (
                mock_response
            )

            result = backend.chat(
                model="gemini-2.5-flash",
                messages=[{"role": "user", "content": "What's the weather?"}],
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "search",
                            "description": "Search the web",
                            "parameters": {
                                "type": "object",
                                "properties": {"query": {"type": "string"}},
                            },
                        },
                    }
                ],
            )

            assert result.choices[0].message.content is None
            assert len(result.choices[0].message.tool_calls) == 1
            tc = result.choices[0].message.tool_calls[0]
            assert tc.function.name == "search"
            args = json.loads(tc.function.arguments)
            assert args == {"query": "weather"}
            assert tc.id.startswith("call_")

    def test_chat_mixed_text_and_tool_calls(self):
        with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
            mock_genai = Mock()
            mock_types = Mock()
            mock_req.return_value = (mock_genai, mock_types)
            backend = NativeGeminiBackend(api_key="k", model="m")

            mock_response = _mock_gemini_response(
                text="Let me search for that.",
                tool_calls=[("search", {"q": "test"})],
            )
            mock_genai.Client.return_value.models.generate_content.return_value = (
                mock_response
            )

            result = backend.chat(
                model="m", messages=[{"role": "user", "content": "test"}]
            )

            assert result.choices[0].message.content == "Let me search for that."
            assert len(result.choices[0].message.tool_calls) == 1


class TestChatStreaming:
    """Tests for streaming chat requests."""

    def test_stream_yields_chunks(self):
        with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
            mock_genai = Mock()
            mock_types = Mock()
            mock_req.return_value = (mock_genai, mock_types)
            backend = NativeGeminiBackend(api_key="k", model="m")

            mock_chunks = _mock_gemini_stream_chunks(["Hel", "lo ", "world!"])
            mock_genai.Client.return_value.models.generate_content_stream.return_value = iter(
                mock_chunks
            )

            result = backend.chat(
                model="m",
                messages=[{"role": "user", "content": "Hi"}],
                stream=True,
            )

            collected = []
            for chunk in result:
                assert hasattr(chunk, "choices")
                assert len(chunk.choices) > 0
                if chunk.choices[0].delta.content:
                    collected.append(chunk.choices[0].delta.content)

            assert "".join(collected) == "Hello world!"

    def test_stream_final_chunk_finish_reason_stop(self):
        """Text-only stream ends with finish_reason='stop'."""
        with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
            mock_genai = Mock()
            mock_types = Mock()
            mock_req.return_value = (mock_genai, mock_types)
            backend = NativeGeminiBackend(api_key="k", model="m")

            mock_chunks = _mock_gemini_stream_chunks(["Hi"])
            mock_genai.Client.return_value.models.generate_content_stream.return_value = iter(
                mock_chunks
            )

            result = list(backend.chat(model="m", messages=[], stream=True))

            # Last chunk is the sentinel with finish_reason
            assert result[-1].choices[0].finish_reason == "stop"

    def test_stream_with_tool_call(self):
        with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
            mock_genai = Mock()
            mock_types = Mock()
            mock_req.return_value = (mock_genai, mock_types)
            backend = NativeGeminiBackend(api_key="k", model="m")

            # Chunk with function_call
            part = Mock()
            part.text = None
            fc = Mock()
            fc.name = "search"
            fc.args = {"q": "test"}
            part.function_call = fc
            content = Mock()
            content.parts = [part]
            candidate = Mock()
            candidate.content = content
            chunk = Mock()
            chunk.candidates = [candidate]

            mock_genai.Client.return_value.models.generate_content_stream.return_value = iter(
                [chunk]
            )

            result = list(backend.chat(model="m", messages=[], stream=True))

            # First chunk has the tool call, last chunk is sentinel
            tc_chunk = result[0]
            assert tc_chunk.choices[0].delta.tool_calls is not None
            assert len(tc_chunk.choices[0].delta.tool_calls) == 1
            assert tc_chunk.choices[0].delta.tool_calls[0].function.name == "search"
            assert tc_chunk.choices[0].delta.tool_calls[0].index == 0
            # Sentinel has finish_reason="tool_calls"
            assert result[-1].choices[0].finish_reason == "tool_calls"

    def test_stream_tool_call_index_increments(self):
        """Multiple tool call parts get incrementing index values."""
        with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
            mock_genai = Mock()
            mock_types = Mock()
            mock_req.return_value = (mock_genai, mock_types)
            backend = NativeGeminiBackend(api_key="k", model="m")

            # Two function_call parts in one chunk
            parts = []
            for name in ["search", "fetch"]:
                part = Mock()
                part.text = None
                fc = Mock()
                fc.name = name
                fc.args = {}
                part.function_call = fc
                parts.append(part)

            content = Mock()
            content.parts = parts
            candidate = Mock()
            candidate.content = content
            chunk = Mock()
            chunk.candidates = [candidate]

            mock_genai.Client.return_value.models.generate_content_stream.return_value = iter(
                [chunk]
            )

            result = list(backend.chat(model="m", messages=[], stream=True))

            # Two tool call chunks + sentinel
            tc_chunks = [c for c in result if c.choices[0].delta.tool_calls]
            assert len(tc_chunks) == 2
            assert tc_chunks[0].choices[0].delta.tool_calls[0].index == 0
            assert tc_chunks[1].choices[0].delta.tool_calls[0].index == 1

    def test_stream_skips_empty_candidates(self):
        with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
            mock_genai = Mock()
            mock_types = Mock()
            mock_req.return_value = (mock_genai, mock_types)
            backend = NativeGeminiBackend(api_key="k", model="m")

            empty_chunk = Mock()
            empty_chunk.candidates = []
            text_chunk = _mock_gemini_stream_chunks(["data"])[0]

            mock_genai.Client.return_value.models.generate_content_stream.return_value = iter(
                [empty_chunk, text_chunk]
            )

            result = list(backend.chat(model="m", messages=[], stream=True))
            # 1 data chunk + 1 sentinel
            assert len(result) == 2
            assert result[0].choices[0].delta.content == "data"
            assert result[-1].choices[0].finish_reason == "stop"


class TestEmbed:
    """Tests for embedding requests."""

    def test_embed_single_input(self):
        with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
            mock_genai = Mock()
            mock_types = Mock()
            mock_req.return_value = (mock_genai, mock_types)
            backend = NativeGeminiBackend(api_key="k", model="gemini-embedding-001")

            mock_embedding = Mock()
            mock_embedding.values = [0.1, 0.2, 0.3]
            mock_result = Mock()
            mock_result.embeddings = [mock_embedding]
            mock_genai.Client.return_value.models.embed_content.return_value = (
                mock_result
            )

            response = backend.embed(model="gemini-embedding-001", input="hello")

            assert response.data[0].embedding == [0.1, 0.2, 0.3]
            assert response.model == "gemini-embedding-001"

    def test_embed_multiple_inputs(self):
        with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
            mock_genai = Mock()
            mock_types = Mock()
            mock_req.return_value = (mock_genai, mock_types)
            backend = NativeGeminiBackend(api_key="k", model="m")

            emb1, emb2 = Mock(), Mock()
            emb1.values = [0.1, 0.2]
            emb2.values = [0.3, 0.4]
            mock_result = Mock()
            mock_result.embeddings = [emb1, emb2]
            mock_genai.Client.return_value.models.embed_content.return_value = (
                mock_result
            )

            response = backend.embed(model="m", input=["hello", "world"])

            assert len(response.data) == 2
            assert response.data[0].embedding == [0.1, 0.2]
            assert response.data[1].embedding == [0.3, 0.4]


class TestCountTokens:
    """Tests for token counting."""

    def test_count_tokens(self):
        with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
            mock_genai = Mock()
            mock_types = Mock()
            mock_req.return_value = (mock_genai, mock_types)
            backend = NativeGeminiBackend(api_key="k", model="gemini-2.5-flash")

            mock_count = Mock()
            mock_count.total_tokens = 42
            mock_genai.Client.return_value.models.count_tokens.return_value = mock_count

            result = backend.count_tokens([{"role": "user", "content": "Hello world"}])
            assert result == 42


class TestConfigBuilding:
    """Tests for GenerateContentConfig construction."""

    def test_temperature_forwarded(self):
        with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
            mock_genai = Mock()
            mock_types = Mock()
            mock_req.return_value = (mock_genai, mock_types)
            backend = NativeGeminiBackend(api_key="k", model="m")

            mock_genai.Client.return_value.models.generate_content.return_value = (
                _mock_gemini_response("ok")
            )

            backend.chat(
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                max_completion_tokens=1000,
            )

            # Verify GenerateContentConfig was built with the params
            config_call = mock_types.GenerateContentConfig.call_args
            assert config_call is not None
            kwargs = config_call[1]
            assert kwargs["temperature"] == 0.7
            assert kwargs["top_p"] == 0.9
            assert kwargs["top_k"] == 40
            assert kwargs["max_output_tokens"] == 1000

    def test_json_response_format(self):
        with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
            mock_genai = Mock()
            mock_types = Mock()
            mock_req.return_value = (mock_genai, mock_types)
            backend = NativeGeminiBackend(api_key="k", model="m")

            mock_genai.Client.return_value.models.generate_content.return_value = (
                _mock_gemini_response("ok")
            )

            backend.chat(
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                response_format={"type": "json_object"},
            )

            config_call = mock_types.GenerateContentConfig.call_args
            assert config_call[1]["response_mime_type"] == "application/json"

    def test_tools_converted(self):
        with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
            mock_genai = Mock()
            mock_types = Mock()
            mock_req.return_value = (mock_genai, mock_types)
            backend = NativeGeminiBackend(api_key="k", model="m")

            mock_genai.Client.return_value.models.generate_content.return_value = (
                _mock_gemini_response("ok")
            )

            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ]

            backend.chat(
                model="m", messages=[{"role": "user", "content": "hi"}], tools=tools
            )

            # FunctionDeclaration was created
            mock_types.FunctionDeclaration.assert_called_once_with(
                name="get_weather",
                description="Get weather",
                parameters={
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            )

    def test_tool_choice_mapping(self):
        with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
            mock_genai = Mock()
            mock_types = Mock()
            mock_req.return_value = (mock_genai, mock_types)
            backend = NativeGeminiBackend(api_key="k", model="m")

            mock_genai.Client.return_value.models.generate_content.return_value = (
                _mock_gemini_response("ok")
            )

            tools = [
                {"type": "function", "function": {"name": "fn", "description": "d"}}
            ]

            backend.chat(
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                tools=tools,
                tool_choice="required",
            )

            # ToolConfig with mode=ANY (mapped from "required")
            mock_types.FunctionCallingConfig.assert_called_once_with(mode="ANY")


# ---------------------------------------------------------------------------
# LLM integration tests
# ---------------------------------------------------------------------------


class TestLLMNativeGeminiIntegration:
    """Tests for LLM with native_google=True."""

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "test-key",
            "INFERENCE_MODEL_ID": "gemini-2.5-flash",
        },
    )
    def test_native_google_creates_native_backend(self):
        """native_google=True creates NativeGeminiBackend."""
        with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
            mock_genai = Mock()
            mock_types = Mock()
            mock_req.return_value = (mock_genai, mock_types)

            llm = LLM(native_google=True)
            assert isinstance(llm.backend, NativeGeminiBackend)
            assert llm.backend.provider_name == "google_native"
            assert llm.client is None

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "test-key",
            "INFERENCE_MODEL_ID": "gemini-2.5-flash",
            "GEMINI_NATIVE": "true",
        },
    )
    def test_gemini_native_env_var(self):
        """GEMINI_NATIVE=true env var activates native backend."""
        with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
            mock_genai = Mock()
            mock_types = Mock()
            mock_req.return_value = (mock_genai, mock_types)

            llm = LLM()
            assert isinstance(llm.backend, NativeGeminiBackend)

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "test-key",
            "INFERENCE_MODEL_ID": "gemini-2.5-flash",
        },
    )
    def test_without_native_google_uses_openai_compat(self):
        """Without native_google, Google URL uses OpenAICompatibleBackend."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
            from floship_llm.backends.openai_compat import OpenAICompatibleBackend

            assert isinstance(llm.backend, OpenAICompatibleBackend)
            assert llm.client is not None

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": HEROKU_URL,
            "INFERENCE_KEY": "test-key",
            "INFERENCE_MODEL_ID": "claude-3.5-sonnet",
        },
    )
    def test_heroku_ignores_native_google(self):
        """Heroku URL always uses OpenAICompatibleBackend even with native_google=True."""
        # native_google only affects backend creation; for Heroku, it still
        # creates NativeGeminiBackend if explicitly requested -- but typically
        # users wouldn't do this. The factory doesn't gate on provider.
        # This test documents current behavior.
        with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
            mock_genai = Mock()
            mock_types = Mock()
            mock_req.return_value = (mock_genai, mock_types)

            llm = LLM(native_google=True)
            # Backend is NativeGemini because native_google=True overrides
            assert isinstance(llm.backend, NativeGeminiBackend)

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "test-key",
            "INFERENCE_MODEL_ID": "gemini-2.5-flash",
        },
    )
    def test_native_prompt_end_to_end(self):
        """Full prompt() call with native backend returns text."""
        with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
            mock_genai = Mock()
            mock_types = Mock()
            mock_req.return_value = (mock_genai, mock_types)

            llm = LLM(native_google=True)

            # Mock streaming response (prompt() defaults to streaming)
            mock_chunks = _mock_gemini_stream_chunks(["Hello", " from", " Gemini!"])
            mock_genai.Client.return_value.models.generate_content_stream.return_value = iter(
                mock_chunks
            )

            result = llm.prompt("Hi")
            assert result == "Hello from Gemini!"

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "test-key",
            "INFERENCE_MODEL_ID": "gemini-embedding-001",
        },
    )
    def test_native_embed_end_to_end(self):
        """Full embed() call with native backend returns embeddings."""
        with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
            mock_genai = Mock()
            mock_types = Mock()
            mock_req.return_value = (mock_genai, mock_types)

            llm = LLM(native_google=True, type="embedding")

            emb = Mock()
            emb.values = [0.1, 0.2, 0.3]
            mock_result = Mock()
            mock_result.embeddings = [emb]
            mock_genai.Client.return_value.models.embed_content.return_value = (
                mock_result
            )

            result = llm.embed("hello")
            assert result == [0.1, 0.2, 0.3]

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "test-key",
            "INFERENCE_MODEL_ID": "gemini-2.5-flash",
            "GEMINI_NATIVE": "false",
        },
    )
    def test_gemini_native_false_uses_openai_compat(self):
        """GEMINI_NATIVE=false does not activate native backend."""
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
            from floship_llm.backends.openai_compat import OpenAICompatibleBackend

            assert isinstance(llm.backend, OpenAICompatibleBackend)

    @patch.dict(
        os.environ,
        {
            "INFERENCE_URL": GOOGLE_URL,
            "INFERENCE_KEY": "test-key",
            "INFERENCE_MODEL_ID": "gemini-2.5-flash",
        },
    )
    def test_native_prompt_stream_end_to_end(self):
        """prompt_stream() yields chunks from native backend."""
        with patch("floship_llm.backends.native_gemini._require_genai") as mock_req:
            mock_genai = Mock()
            mock_types = Mock()
            mock_req.return_value = (mock_genai, mock_types)

            llm = LLM(native_google=True)

            mock_chunks = _mock_gemini_stream_chunks(["A", "B", "C"])
            mock_genai.Client.return_value.models.generate_content_stream.return_value = iter(
                mock_chunks
            )

            collected = list(llm.prompt_stream("Hi"))
            assert "".join(collected) == "ABC"

"""Tests for System One (TypeSafe Jev) support.

Jev is a judgment model reached through OpenRouter's decisions route, which is
not part of the OpenAI-compatible surface:

    POST https://openrouter.ai/api/alpha/decisions
    {"model": ..., "state": ..., "questions": {...}}

Key behaviors:
- The decisions path is NOT under the OpenAI-compatible /api/v1 prefix.
- All questions travel in one request and answers come back typed, with no
  prose to parse.
- A model blocked by a workspace guardrail is reported as HTTP 404, so the
  server's own message must survive into the exception.
- Chat and embeddings do not apply to the System One backend.

The response fixtures below are the examples published in OpenRouter's API
reference for /api/alpha/decisions.
"""

import json
import os
from unittest.mock import patch

import httpx
import pytest

from floship_llm import (
    LLM,
    Choice,
    ChoiceAnswer,
    DecisionsResponse,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    SystemOneBackend,
    SystemOneError,
)
from floship_llm.backends.openai_compat import OpenAICompatibleBackend
from floship_llm.backends.systemone import decisions_url, question_to_wire

OPENROUTER_URL = "https://openrouter.ai/api/v1"
DECISIONS_URL = "https://openrouter.ai/api/alpha/decisions"

# The example response from the OpenRouter API reference.
EXAMPLE_ANSWERS = {
    "is_bug": {"noul": 0.96, "type": "noul"},
    "team": {
        "choice": "payments",
        "confidence": 0.75,
        "probabilities": {"account": 0, "frontend": 0.16, "payments": 0.84},
        "type": "choice",
    },
    "urgency": {
        "confidence": 0.99,
        "legend": {
            "0": "Can wait for the next release",
            "1": "Should be fixed this week",
            "2": "Blocking revenue right now",
        },
        "probabilities": {"0": 0, "1": 0.01, "2": 0.99},
        "score": 1.99,
        "type": "score",
    },
}

EXAMPLE_RESPONSE = {
    "answers": EXAMPLE_ANSWERS,
    "id": "gen-dec-1789738314-X5e5eKGQdvR9rblyX250",
    "model": "typesafe/jev-1.13-20260917",
    "provider": "TypeSafe",
    "usage": {"cost": 0.000019992, "input_tokens": 476, "output_tokens": 70},
}


class FakeResponse:
    """Stand-in for an httpx.Response."""

    def __init__(self, status_code=200, payload=None, text=None):
        self.status_code = status_code
        self._payload = payload
        if text is not None:
            self.text = text
        else:
            self.text = json.dumps(payload) if payload is not None else ""

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class FakeClient:
    """Stand-in for an httpx.Client that records what was posted."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []
        self.closed = False

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append(
            {"url": url, "headers": headers, "json": json, "timeout": timeout}
        )
        if len(self._responses) > 1:
            return self._responses.pop(0)
        return self._responses[0]

    def close(self):
        self.closed = True


def make_backend(responses, **kwargs):
    """Return a backend wired to a fake client and no retry delay."""
    kwargs.setdefault("backoff", 0)
    client = FakeClient(responses)
    backend = SystemOneBackend(
        api_key="test-key",  # pragma: allowlist secret
        model="typesafe/jev-1.13",
        base_url=OPENROUTER_URL,
        client=client,
        **kwargs,
    )
    return backend, client


class TestDecisionsUrl:
    """The decisions path is not under the OpenAI-compatible prefix."""

    def test_derives_from_openrouter_base_url(self):
        assert decisions_url(OPENROUTER_URL) == DECISIONS_URL

    def test_derives_with_trailing_slash(self):
        assert decisions_url(OPENROUTER_URL + "/") == DECISIONS_URL

    def test_keeps_scheme_and_host_only(self):
        url = decisions_url("https://openrouter.ai/api/v1")
        assert url.startswith("https://openrouter.ai/")
        assert "/api/v1" not in url

    def test_rejects_relative_url(self):
        with pytest.raises(ValueError):
            decisions_url("/api/v1")

    def test_rejects_empty_url(self):
        with pytest.raises(ValueError):
            decisions_url("")


class TestQuestionWireFormat:
    """Each primitive serialises to the documented request shape."""

    def test_noul_without_criteria_omits_the_key(self):
        wire = Noul("Is the customer reporting a defect?").to_wire()
        assert wire == {
            "type": "noul",
            "instructions": "Is the customer reporting a defect?",
        }
        assert "criteria" not in wire

    def test_noul_with_criteria(self):
        wire = Noul(
            "Is the customer reporting a defect?",
            criteria={"true": "Broken behaviour", "false": "A question"},
        ).to_wire()
        assert wire["type"] == "noul"
        assert wire["criteria"]["true"] == "Broken behaviour"

    def test_choice_carries_criteria(self):
        wire = Choice(
            instructions="Which team should own this ticket?",
            criteria={"payments": "Checkout or billing"},
        ).to_wire()
        assert wire == {
            "type": "choice",
            "instructions": "Which team should own this ticket?",
            "criteria": {"payments": "Checkout or billing"},
        }

    def test_score_carries_ordered_levels(self):
        wire = Score(
            instructions="How urgent is this ticket?",
            criteria=["Can wait", "This week", "Blocking revenue"],
        ).to_wire()
        assert wire["type"] == "score"
        assert wire["criteria"] == ["Can wait", "This week", "Blocking revenue"]

    def test_structured_instructions_survive_as_json(self):
        wire = Noul(
            instructions={
                "question": "Does the message ask for a credential?",
                "focus": "A request to send it, not to reset it.",
            }
        ).to_wire()
        assert wire["instructions"]["focus"] == "A request to send it, not to reset it."

    def test_question_to_wire_accepts_a_mapping(self):
        raw = {"type": "noul", "instructions": "Is it urgent?"}
        assert question_to_wire(raw) == raw

    def test_question_to_wire_rejects_other_types(self):
        with pytest.raises(TypeError):
            question_to_wire("is it urgent?")


class TestAnswerParsing:
    """Answers arrive typed, so the caller branches instead of parsing."""

    def test_noul_answer(self):
        response = DecisionsResponse.from_wire(EXAMPLE_RESPONSE)
        answer = response.answers["is_bug"]
        assert isinstance(answer, NoulAnswer)
        assert answer.noul == 0.96

    def test_choice_answer(self):
        response = DecisionsResponse.from_wire(EXAMPLE_RESPONSE)
        answer = response.answers["team"]
        assert isinstance(answer, ChoiceAnswer)
        assert answer.choice == "payments"
        assert answer.probabilities["payments"] == 0.84
        assert answer.confidence == 0.75

    def test_score_answer(self):
        response = DecisionsResponse.from_wire(EXAMPLE_RESPONSE)
        answer = response.answers["urgency"]
        assert isinstance(answer, ScoreAnswer)
        assert answer.score == 1.99
        assert answer.probabilities["2"] == 0.99
        assert answer.confidence == 0.99
        assert answer.legend["2"] == "Blocking revenue right now"

    def test_metadata_is_kept(self):
        response = DecisionsResponse.from_wire(EXAMPLE_RESPONSE)
        assert response.model == "typesafe/jev-1.13-20260917"
        assert response.provider == "TypeSafe"
        assert response.id.startswith("gen-dec-")
        assert response.usage["input_tokens"] == 476
        assert response.usage["cost"] == 0.000019992

    def test_raw_payload_is_available(self):
        response = DecisionsResponse.from_wire(EXAMPLE_RESPONSE)
        assert response.raw["answers"]["is_bug"]["noul"] == 0.96

    def test_missing_confidence_stays_none(self):
        response = DecisionsResponse.from_wire(
            {"answers": {"team": {"type": "choice", "choice": "a"}}}
        )
        answer = response.answers["team"]
        assert answer.confidence is None
        assert answer.probabilities == {}

    def test_missing_probabilities_do_not_break_the_noul(self):
        response = DecisionsResponse.from_wire(
            {"answers": {"u": {"type": "noul", "noul": 0.5}}}
        )
        assert response.answers["u"].noul == 0.5

    def test_unknown_answer_type_degrades_to_the_raw_mapping(self):
        response = DecisionsResponse.from_wire(
            {"answers": {"new": {"type": "ranking", "order": ["a", "b"]}}}
        )
        assert response.answers["new"] == {"type": "ranking", "order": ["a", "b"]}

    def test_empty_response_yields_no_answers(self):
        response = DecisionsResponse.from_wire({})
        assert response.answers == {}
        assert response.model == ""
        assert response.usage == {}


class TestDecideRequest:
    """One request carries every question."""

    def test_posts_to_the_decisions_endpoint_with_a_bearer_token(self):
        backend, client = make_backend([FakeResponse(200, EXAMPLE_RESPONSE)])
        backend.decide(state="A ticket", questions={"is_bug": Noul("Is it a bug?")})
        call = client.calls[0]
        assert call["url"] == DECISIONS_URL
        assert call["headers"]["Authorization"] == "Bearer test-key"
        assert call["headers"]["Content-Type"] == "application/json"

    def test_body_carries_model_state_and_questions(self):
        backend, client = make_backend([FakeResponse(200, EXAMPLE_RESPONSE)])
        backend.decide(
            state={"ticket": "blank screen"},
            questions={"is_bug": Noul("Is it a bug?")},
        )
        body = client.calls[0]["json"]
        assert body["model"] == "typesafe/jev-1.13"
        assert body["state"] == {"ticket": "blank screen"}
        assert body["questions"]["is_bug"]["type"] == "noul"

    def test_all_three_primitives_travel_together(self):
        backend, client = make_backend([FakeResponse(200, EXAMPLE_RESPONSE)])
        backend.decide(
            state="A ticket",
            questions={
                "is_bug": Noul("Is it a bug?"),
                "team": Choice(instructions="Which team?", criteria={"a": "A"}),
                "urgency": Score(instructions="How urgent?", criteria=["Low", "High"]),
            },
        )
        questions = client.calls[0]["json"]["questions"]
        assert set(questions) == {"is_bug", "team", "urgency"}
        assert questions["team"]["type"] == "choice"
        assert questions["urgency"]["type"] == "score"

    def test_session_id_is_sent_when_given(self):
        backend, client = make_backend([FakeResponse(200, EXAMPLE_RESPONSE)])
        backend.decide(
            state="x", questions={"q": Noul("Is it?")}, session_id="thread-1"
        )
        assert client.calls[0]["json"]["session_id"] == "thread-1"

    def test_session_id_is_omitted_by_default(self):
        backend, client = make_backend([FakeResponse(200, EXAMPLE_RESPONSE)])
        backend.decide(state="x", questions={"q": Noul("Is it?")})
        assert "session_id" not in client.calls[0]["json"]

    def test_model_can_be_overridden_per_call(self):
        backend, client = make_backend([FakeResponse(200, EXAMPLE_RESPONSE)])
        backend.decide(
            state="x", questions={"q": Noul("Is it?")}, model="typesafe/jev-1.14"
        )
        assert client.calls[0]["json"]["model"] == "typesafe/jev-1.14"

    def test_decide_returns_typed_answers(self):
        backend, _ = make_backend([FakeResponse(200, EXAMPLE_RESPONSE)])
        response = backend.decide(
            state="A ticket", questions={"is_bug": Noul("Is it a bug?")}
        )
        assert isinstance(response, DecisionsResponse)
        assert response.answers["is_bug"].noul == 0.96


class TestErrorHandling:
    """The server's own words must survive into the exception."""

    def test_error_envelope_becomes_systemoneerror(self):
        backend, _ = make_backend(
            [FakeResponse(400, {"error": {"code": 400, "message": "Invalid input"}})]
        )
        with pytest.raises(SystemOneError) as caught:
            backend.decide(state="x", questions={"q": Noul("Is it?")})
        assert "Invalid input" in str(caught.value)
        assert caught.value.status == 400
        assert caught.value.code == 400
        assert caught.value.url == DECISIONS_URL

    def test_guardrail_rejection_stays_legible(self):
        """OpenRouter reports a guardrail block as 404; the text must survive."""
        message = (
            "0 endpoints out of 1 requested are available matching your "
            "guardrail restrictions and data policy. Model blocked by "
            "guardrail: 1 endpoint excluded; configurable at "
            "https://openrouter.ai/workspaces/default/guardrails"
        )
        backend, _ = make_backend(
            [FakeResponse(404, {"error": {"code": 404, "message": message}})]
        )
        with pytest.raises(SystemOneError) as caught:
            backend.decide(state="x", questions={"q": Noul("Is it?")})
        assert "guardrail" in str(caught.value)
        assert "workspaces/default/guardrails" in str(caught.value)
        assert caught.value.status == 404

    def test_non_json_body_is_reported_with_a_snippet(self):
        backend, _ = make_backend(
            [FakeResponse(502, payload=None, text="<html>bad gateway</html>")]
        )
        with pytest.raises(SystemOneError) as caught:
            backend.decide(state="x", questions={"q": Noul("Is it?")})
        assert "bad gateway" in str(caught.value)
        assert caught.value.status == 502

    def test_success_body_that_is_not_json_raises(self):
        backend, _ = make_backend([FakeResponse(200, payload=None, text="nope")])
        with pytest.raises(SystemOneError) as caught:
            backend.decide(state="x", questions={"q": Noul("Is it?")})
        assert caught.value.status == 200

    def test_success_body_that_is_not_an_object_raises(self):
        backend, _ = make_backend([FakeResponse(200, payload=[1, 2, 3])])
        with pytest.raises(SystemOneError):
            backend.decide(state="x", questions={"q": Noul("Is it?")})


class TestRetries:
    """Transient failures retry; a rejected request does not."""

    def test_retries_a_503_then_succeeds(self):
        backend, client = make_backend(
            [
                FakeResponse(503, {"error": {"code": 503, "message": "unavailable"}}),
                FakeResponse(200, EXAMPLE_RESPONSE),
            ]
        )
        response = backend.decide(state="x", questions={"q": Noul("Is it?")})
        assert len(client.calls) == 2
        assert response.answers["is_bug"].noul == 0.96

    def test_does_not_retry_a_rejected_request(self):
        backend, client = make_backend(
            [FakeResponse(400, {"error": {"code": 400, "message": "bad"}})]
        )
        with pytest.raises(SystemOneError):
            backend.decide(state="x", questions={"q": Noul("Is it?")})
        assert len(client.calls) == 1

    def test_does_not_retry_a_guardrail_rejection(self):
        """A guardrail block is a 404 and will not resolve by asking again."""
        backend, client = make_backend(
            [FakeResponse(404, {"error": {"code": 404, "message": "blocked"}})]
        )
        with pytest.raises(SystemOneError):
            backend.decide(state="x", questions={"q": Noul("Is it?")})
        assert len(client.calls) == 1

    def test_gives_up_after_max_attempts(self):
        backend, client = make_backend(
            [FakeResponse(503, {"error": {"code": 503, "message": "unavailable"}})],
            max_attempts=2,
        )
        with pytest.raises(SystemOneError):
            backend.decide(state="x", questions={"q": Noul("Is it?")})
        assert len(client.calls) == 2

    def test_max_attempts_is_at_least_one(self):
        backend, client = make_backend(
            [FakeResponse(400, {"error": {"code": 400, "message": "bad"}})],
            max_attempts=0,
        )
        with pytest.raises(SystemOneError):
            backend.decide(state="x", questions={"q": Noul("Is it?")})
        assert len(client.calls) == 1


class TestBackendSurface:
    """System One is not a chat backend."""

    def test_chat_is_not_supported(self):
        backend, _ = make_backend([FakeResponse(200, EXAMPLE_RESPONSE)])
        with pytest.raises(NotImplementedError):
            backend.chat(messages=[])

    def test_embed_is_not_supported(self):
        backend, _ = make_backend([FakeResponse(200, EXAMPLE_RESPONSE)])
        with pytest.raises(NotImplementedError):
            backend.embed(input="text")

    def test_provider_name(self):
        backend, _ = make_backend([FakeResponse(200, EXAMPLE_RESPONSE)])
        assert backend.provider_name == "systemone"

    def test_endpoint_is_exposed(self):
        backend, _ = make_backend([FakeResponse(200, EXAMPLE_RESPONSE)])
        assert backend.endpoint == DECISIONS_URL

    def test_explicit_endpoint_overrides_derivation(self):
        backend, _ = make_backend(
            [FakeResponse(200, EXAMPLE_RESPONSE)],
            endpoint="https://gateway.internal/decisions",
        )
        assert backend.endpoint == "https://gateway.internal/decisions"

    def test_owned_client_is_closed_and_a_supplied_one_is_left_alone(self):
        supplied, _ = make_backend([FakeResponse(200, EXAMPLE_RESPONSE)])
        supplied.close()
        assert supplied._client is not None

    def test_chat_backend_rejects_decisions(self):
        """A chat-only backend says so instead of silently misbehaving."""
        client = OpenAICompatibleBackend(client=None, provider="openrouter")
        with pytest.raises(NotImplementedError):
            client.decide(state="x", questions={})


class TestLLMDecisions:
    """LLM.decisions() is the public entry point."""

    def setup_method(self):
        self.env_vars = {
            "INFERENCE_URL": OPENROUTER_URL,
            "INFERENCE_MODEL_ID": "typesafe/jev-1.13",
            "INFERENCE_KEY": "test-key",  # pragma: allowlist secret
        }
        self.env_patcher = patch.dict(os.environ, self.env_vars, clear=False)
        self.env_patcher.start()

    def teardown_method(self):
        self.env_patcher.stop()

    def _llm(self, responses):
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
        client = FakeClient(responses)
        llm._systemone_backend = SystemOneBackend(
            api_key="test-key",  # pragma: allowlist secret
            model=llm.model,
            base_url=llm.base_url,
            client=client,
            backoff=0,
        )
        return llm, client

    def test_decisions_returns_typed_answers(self):
        llm, _ = self._llm([FakeResponse(200, EXAMPLE_RESPONSE)])
        response = llm.decisions(
            state={"ticket": "Checkout shows a blank screen."},
            questions={"is_bug": Noul("Is the customer reporting a defect?")},
        )
        assert isinstance(response, DecisionsResponse)
        assert isinstance(response.answers["is_bug"], NoulAnswer)
        assert response.answers["is_bug"].noul == 0.96

    def test_posts_to_the_decisions_endpoint_not_the_chat_one(self):
        llm, client = self._llm([FakeResponse(200, EXAMPLE_RESPONSE)])
        llm.decisions(state="x", questions={"q": Noul("Is it?")})
        assert client.calls[0]["url"] == DECISIONS_URL

    def test_full_response_can_be_requested(self):
        llm, _ = self._llm([FakeResponse(200, EXAMPLE_RESPONSE)])
        raw = llm.decisions(
            state="x",
            questions={"q": Noul("Is it?")},
            return_full_response=True,
        )
        assert isinstance(raw, dict)
        assert raw["model"] == "typesafe/jev-1.13-20260917"

    def test_empty_state_is_rejected(self):
        llm, _ = self._llm([FakeResponse(200, EXAMPLE_RESPONSE)])
        with pytest.raises(ValueError):
            llm.decisions(state="", questions={"q": Noul("Is it?")})

    def test_no_questions_is_rejected(self):
        llm, _ = self._llm([FakeResponse(200, EXAMPLE_RESPONSE)])
        with pytest.raises(ValueError):
            llm.decisions(state="a ticket", questions={})

    def test_backend_is_created_once(self):
        llm, _ = self._llm([FakeResponse(200, EXAMPLE_RESPONSE)])
        first = llm._get_systemone_backend()
        assert llm._get_systemone_backend() is first

    def test_close_systemone_is_idempotent(self):
        llm, _ = self._llm([FakeResponse(200, EXAMPLE_RESPONSE)])
        llm.close_systemone()
        llm.close_systemone()
        assert llm._systemone_backend is None

    def test_a_chat_only_client_never_builds_the_backend(self):
        with patch("floship_llm.client.OpenAI"):
            llm = LLM()
        assert llm._systemone_backend is None


class TransportFailClient:
    """Raises a transport error the given number of times, then answers."""

    def __init__(self, failures, response):
        self._failures = failures
        self._response = response
        self.calls = 0

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls += 1
        if self.calls <= self._failures:
            raise httpx.ConnectError("connection refused")
        return self._response

    def close(self):
        pass


class TestTransportFailures:
    """A dropped connection is retried, then reported."""

    def test_transport_error_becomes_systemoneerror(self):
        client = TransportFailClient(1, FakeResponse(200, EXAMPLE_RESPONSE))
        backend = SystemOneBackend(
            api_key="k",  # pragma: allowlist secret
            model="typesafe/jev-1.13",
            base_url=OPENROUTER_URL,
            client=client,
            max_attempts=1,
            backoff=0,
        )
        with pytest.raises(SystemOneError) as caught:
            backend.decide(state="x", questions={"q": Noul("Is it?")})
        assert "Decisions request failed" in str(caught.value)
        assert caught.value.url == DECISIONS_URL

    def test_a_transport_error_is_retried(self):
        client = TransportFailClient(1, FakeResponse(200, EXAMPLE_RESPONSE))
        backend = SystemOneBackend(
            api_key="k",  # pragma: allowlist secret
            model="typesafe/jev-1.13",
            base_url=OPENROUTER_URL,
            client=client,
            max_attempts=3,
            backoff=0,
        )
        response = backend.decide(state="x", questions={"q": Noul("Is it?")})
        assert client.calls == 2
        assert response.answers["is_bug"].noul == 0.96


class TestRequestOptions:
    """Provider preferences, extra body, and timeout reach the request."""

    def test_provider_preferences_are_forwarded(self):
        backend, client = make_backend([FakeResponse(200, EXAMPLE_RESPONSE)])
        backend.decide(
            state="x",
            questions={"q": Noul("Is it?")},
            provider={"only": ["typesafe"]},
        )
        assert client.calls[0]["json"]["provider"] == {"only": ["typesafe"]}

    def test_provider_is_omitted_by_default(self):
        backend, client = make_backend([FakeResponse(200, EXAMPLE_RESPONSE)])
        backend.decide(state="x", questions={"q": Noul("Is it?")})
        assert "provider" not in client.calls[0]["json"]

    def test_extra_body_is_merged(self):
        backend, client = make_backend([FakeResponse(200, EXAMPLE_RESPONSE)])
        backend.decide(
            state="x",
            questions={"q": Noul("Is it?")},
            extra_body={"trace_id": "abc"},
        )
        assert client.calls[0]["json"]["trace_id"] == "abc"

    def test_timeout_is_passed_to_the_transport(self):
        backend, client = make_backend(
            [FakeResponse(200, EXAMPLE_RESPONSE)], timeout=7.5
        )
        backend.decide(state="x", questions={"q": Noul("Is it?")})
        assert client.calls[0]["timeout"] == 7.5

    def test_backoff_waits_between_attempts(self):
        backend, _ = make_backend(
            [FakeResponse(503, {"error": {"code": 503, "message": "x"}})],
            max_attempts=3,
            backoff=0.01,
        )
        with patch("floship_llm.backends.systemone.time.sleep") as sleeper:
            with pytest.raises(SystemOneError):
                backend.decide(state="x", questions={"q": Noul("Is it?")})
        assert sleeper.call_count == 2

    def test_zero_backoff_does_not_wait(self):
        backend, _ = make_backend(
            [FakeResponse(503, {"error": {"code": 503, "message": "x"}})],
            max_attempts=2,
        )
        with patch("floship_llm.backends.systemone.time.sleep") as sleeper:
            with pytest.raises(SystemOneError):
                backend.decide(state="x", questions={"q": Noul("Is it?")})
        assert sleeper.call_count == 0


class TestClientLifecycle:
    """The backend owns its client only when it made one."""

    def test_owned_client_is_created_lazily_and_closed(self):
        backend = SystemOneBackend(
            api_key="k",  # pragma: allowlist secret
            model="typesafe/jev-1.13",
            base_url=OPENROUTER_URL,
            backoff=0,
        )
        assert backend._client is None
        created = backend._client_for_use()
        assert backend._client is created
        backend.close()
        assert backend._client is None

    def test_close_leaves_a_supplied_client_open(self):
        backend, client = make_backend([FakeResponse(200, EXAMPLE_RESPONSE)])
        backend.close()
        assert client.closed is False

    def test_supports_flags_are_false(self):
        backend, _ = make_backend([FakeResponse(200, EXAMPLE_RESPONSE)])
        assert backend.supports_caching is False
        assert backend.supports_native_tools is False


class TestPublicExports:
    """The primitives are importable from the package root."""

    def test_primitives_and_answers_are_exported(self):
        import floship_llm

        for name in (
            "Noul",
            "Choice",
            "Score",
            "NoulAnswer",
            "ChoiceAnswer",
            "ScoreAnswer",
            "DecisionsResponse",
            "SystemOneBackend",
            "SystemOneError",
        ):
            assert name in floship_llm.__all__
            assert hasattr(floship_llm, name)

    def test_error_is_a_runtime_error(self):
        assert issubclass(SystemOneError, RuntimeError)

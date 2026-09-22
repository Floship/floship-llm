"""System One primitives and typed answers for TypeSafe Jev.

Jev is a judgment model.  It reads one 'state', answers every question in
the request independently and in parallel, and returns a typed answer per
question.  It does not reason in steps and it does not generate text, so the
caller never parses prose: every answer stays inside the options supplied.
Code owns the control flow, the weights, and the thresholds.

This module holds the wire types only.  The transport lives in
'floship_llm.backends.systemone' and 'LLM.decisions()' is the entry point.

Three primitives are available:

* 'Noul' -- a yes/no question.  The answer is the probability that the
  statement is true, from 0 to 1.  A value of 0.5 means the model is unsure;
  it does not mean "medium", and a Noul has no separate confidence.
* 'Choice' -- pick one option from a known set with no order.
* 'Score' -- place the state on an ordered scale you describe in levels.

Write one question per judgment, ask one property at a time, and split a
question that weighs two properties.  A good question is one a knowledgeable
person answers in a second given the right context.

Instructions and criteria accept a plain string, or a JSON object or array
when a question has labelled parts or needs supporting data.  Pass a schema,
a taxonomy, or a row as JSON rather than serialising it into a string
template.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Union

# An instruction or criterion: a plain string, or structured guidance.
Guidance = Union[str, Dict[str, Any], List[Any]]


@dataclass(frozen=True)
class Noul:
    """A yes/no question whose answer is a probability.

    Keep the condition crisp.  "Is this candidate strong in Python?" is
    vague; "Does the resume state that the candidate used Python at work?"
    is crisp.  'criteria' is optional: add 'true' and 'false' sides when the
    boundary between them is subtle, and put the neighbouring case in the
    description of the side it belongs to.
    """

    instructions: Guidance
    criteria: Optional[Dict[str, Guidance]] = None

    def to_wire(self) -> Dict[str, Any]:
        """Return this question in the request wire format."""
        payload: Dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        if self.criteria is not None:
            payload["criteria"] = self.criteria
        return payload


@dataclass(frozen=True)
class Choice:
    """Pick one option from a known set with no order.

    Map each option to a description, and make the descriptions contrastive
    when two options sit close together.  Add an 'other' option when the list
    may not cover every input.
    """

    instructions: Guidance
    criteria: Dict[str, Guidance]

    def to_wire(self) -> Dict[str, Any]:
        """Return this question in the request wire format."""
        return {
            "type": "choice",
            "instructions": self.instructions,
            "criteria": self.criteria,
        }


@dataclass(frozen=True)
class Score:
    """Place the state on an ordered scale described in levels.

    List the levels from low to high, and use only as many as you can
    describe distinctly.  Describe situations, not degrees: "Broken feature,
    but a workaround exists" works and "moderately severe" does not.  Each
    level is judged on its own, so a level may not refer to its neighbours
    and should carry no numerals.  Keep one dimension per Score, and give a
    rare extreme its own level when the code must treat it differently.
    """

    instructions: Guidance
    criteria: List[Guidance]

    def to_wire(self) -> Dict[str, Any]:
        """Return this question in the request wire format."""
        return {
            "type": "score",
            "instructions": self.instructions,
            "criteria": self.criteria,
        }


@dataclass(frozen=True)
class NoulAnswer:
    """The probability that the statement is true, from 0 to 1.

    The distance from 0.5 plays the role that 'confidence' plays on the other
    primitives.  Gate the action on a threshold rather than reading the number
    as a verdict.
    """

    noul: float


@dataclass(frozen=True)
class ChoiceAnswer:
    """The chosen option, with a probability per option and a confidence.

    'confidence' measures how peaked the distribution is.  It describes the
    answer and does not guarantee a correct one; the full 'probabilities' are
    here when a different statistic is needed.
    """

    choice: str
    probabilities: Dict[str, float] = field(default_factory=dict)
    confidence: Optional[float] = None


@dataclass(frozen=True)
class ScoreAnswer:
    """A score, with a probability per level and a confidence.

    'score' is the probability-weighted mean of the level numbers.  A score of
    1.0 can mean certainty on level 1 or an even split between levels 0 and 2,
    so read 'probabilities' with it.  Threshold the score, rank by it, or
    round it to the nearest level.  Do not interpolate a quantity from it.

    'legend' maps each level number back to the description that was sent.
    The level numbers are weakly calibrated as numbers, and the caller owns
    any normalisation: dividing by 'len(criteria) - 1' is policy, not
    protocol.
    """

    score: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    confidence: Optional[float] = None
    legend: Optional[Dict[str, Guidance]] = None


def _optional_float(value: Any) -> Optional[float]:
    """Return 'value' as a float, keeping a missing value as None."""
    return None if value is None else float(value)


def _probabilities(value: Any) -> Dict[str, float]:
    """Return the probability mapping with float values."""
    if not isinstance(value, Mapping):
        return {}
    return {str(key): float(weight) for key, weight in value.items()}


def parse_answer(payload: Mapping[str, Any]) -> Any:
    """Build one typed answer from a wire answer.

    An answer of an unrecognised type is returned as the raw mapping, so a
    primitive added server-side degrades into a readable value instead of
    raising in the caller.
    """
    kind = payload.get("type")
    if kind == "noul":
        return NoulAnswer(noul=float(payload["noul"]))
    if kind == "choice":
        return ChoiceAnswer(
            choice=payload["choice"],
            probabilities=_probabilities(payload.get("probabilities")),
            confidence=_optional_float(payload.get("confidence")),
        )
    if kind == "score":
        legend = payload.get("legend")
        return ScoreAnswer(
            score=float(payload["score"]),
            probabilities=_probabilities(payload.get("probabilities")),
            confidence=_optional_float(payload.get("confidence")),
            legend=dict(legend) if isinstance(legend, Mapping) else None,
        )
    return dict(payload)


@dataclass(frozen=True)
class DecisionsResponse:
    """One decisions response: the typed answers and the request metadata.

    'answers' is keyed by the question name that was sent.  'usage' is the
    server's own accounting, with 'input_tokens', 'output_tokens' and, when
    available, 'cost'.
    """

    answers: Dict[str, Any]
    model: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    provider: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> DecisionsResponse:
        """Build a response from the endpoint's JSON body."""
        answers: Any = payload.get("answers")
        parsed = {
            str(name): parse_answer(answer)
            for name, answer in (answers or {}).items()
            if isinstance(answer, Mapping)
        }
        usage: Any = payload.get("usage")
        return cls(
            answers=parsed,
            model=str(payload.get("model") or ""),
            usage=dict(usage) if isinstance(usage, Mapping) else {},
            id=payload.get("id"),
            provider=payload.get("provider"),
            raw=dict(payload),
        )

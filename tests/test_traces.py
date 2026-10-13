from __future__ import annotations

import pytest

from agenteval.traces import (
    Episode,
    TraceStep,
    build_episode,
    canonical_json,
    content_hash,
    hash_episode,
    verify_replay,
)

EVENTS = [
    {"index": 0, "kind": "thought", "payload": {"text": "plan"},
     "latency_ms": 40.0, "tokens_in": 900, "tokens_out": 60},
    {"index": 1, "kind": "tool", "payload": {"name": "search_1", "q": "x"},
     "latency_ms": 210.0, "tokens_in": 120, "tokens_out": 15},
    {"index": 2, "kind": "final", "payload": {"answer": "42"},
     "latency_ms": 10.0, "tokens_in": 0, "tokens_out": 8},
]


def ep(**kw) -> Episode:
    base = dict(task_id="t1", agent="ReAct-v3", seed=0, judge="gpt-judge-A",
                events=EVENTS, score=1.0, success=True, model="gpt-5")
    base.update(kw)
    return build_episode(**base)


def test_build_episode_preserves_the_whole_stream() -> None:
    e = ep()
    assert len(e.steps) == 3
    assert [s.kind for s in e.steps] == ["thought", "tool", "final"]
    assert e.total_tokens == 900 + 60 + 120 + 15 + 0 + 8
    assert e.n_tool_calls == 1
    assert e.total_latency_ms == pytest.approx(260.0)


def test_hash_is_stable_across_identical_builds() -> None:
    assert hash_episode(ep()) == hash_episode(ep())


def test_hash_changes_when_any_decision_changes() -> None:
    """A trace that cannot detect a changed outcome is not worth keeping."""
    base = hash_episode(ep())
    assert hash_episode(ep(score=0.0, success=False)) != base
    assert hash_episode(ep(agent="ToolGraph")) != base
    assert hash_episode(ep(seed=1)) != base
    assert hash_episode(ep(judge="gpt-judge-B")) != base
    assert hash_episode(ep(model="gpt-4")) != base
    assert hash_episode(ep(task_id="t2")) != base
    # a different tool result is a different run, even with the same outcome
    flipped = [dict(e) for e in EVENTS]
    flipped[1] = {**flipped[1], "payload": {"name": "search_1", "q": "y"}}
    assert hash_episode(ep(events=flipped)) != base


def test_replay_verification_detects_divergence() -> None:
    assert verify_replay(ep(), ep())["verdict"] == "identical"
    late = [dict(e) for e in EVENTS]
    late[2] = {**late[2], "payload": {"answer": "43"}}
    v = verify_replay(ep(), ep(events=late))
    assert v["verdict"] != "identical"
    assert v["bitwise_identical"] is False
    assert v["first_divergent_step"] == 2


def test_canonical_json_is_key_order_independent() -> None:
    """Hashes must not depend on dict insertion order."""
    assert content_hash({"a": 1, "b": 2}) == content_hash({"b": 2, "a": 1})
    assert canonical_json({"a": 1, "b": 2}) == canonical_json({"b": 2, "a": 1})


def test_canonical_json_handles_non_finite_floats() -> None:
    """A NaN score must not produce a token no JSON parser accepts."""
    import json

    text = canonical_json({"score": float("nan")})
    assert "NaN" not in text
    assert json.loads(text)["score"] is None


def test_round_trips_through_dict() -> None:
    d = ep().to_dict()
    assert d["agent"] == "ReAct-v3"
    assert len(d["steps"]) == 3
    assert d["steps"][1]["payload"]["name"] == "search_1"


def test_missing_fields_default_rather_than_raise() -> None:
    """Real event streams are sparse; ingestion must not demand every field."""
    e = build_episode("t", "a", 0, "j", [{"kind": "tool"}], score=0.0)
    assert isinstance(e.steps[0], TraceStep)
    assert e.total_tokens == 0
    assert e.success is None

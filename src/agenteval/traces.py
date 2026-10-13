"""Content-addressed, replayable agent episode traces.

An agent eval is only reproducible if you can separate three things that all
get called "noise": what the *model* did, what the *environment* returned, and
what the *judge* said. Traces here make that separation explicit:

* every step is hashed, so a replay that produces different bytes is detectable;
* tool responses are recorded and can be replayed verbatim, which removes
  environment nondeterminism from the comparison and leaves only model variance;
* the seed is carried through, so ``(model, seed, tools)`` fully determines a run.

The payoff is a clean variance decomposition: with replay, ``sigma2_eps`` is
model sampling noise alone, not tool flakiness mixed in.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "TraceStep",
    "Episode",
    "canonical_json",
    "content_hash",
    "hash_episode",
    "verify_replay",
    "trace_signature",
]


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no incidental whitespace.

    Non-finite floats become ``null`` rather than the bare ``NaN`` that
    ``json.dumps`` writes by default. That token is invalid JSON, so any parser
    reading a stored trace would reject the whole file -- and a trace whose score
    is undefined is exactly the common case (an aborted run, a judge that
    returned nothing). Hashes are permanent, so a poisoned trace is not something
    you can quietly fix later.
    """
    return json.dumps(_json_safe(obj), sort_keys=True, separators=(",", ":"),
                      default=str, allow_nan=False)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    return obj


def content_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TraceStep:
    index: int
    kind: str
    payload: dict = field(default_factory=dict)
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def step_hash(self) -> str:
        return content_hash({"i": self.index, "k": self.kind, "p": self.payload})


@dataclass(frozen=True)
class Episode:
    task_id: str
    agent: str
    seed: int
    judge: str
    steps: tuple[TraceStep, ...] = ()
    score: float = float("nan")
    success: bool | None = None
    model: str = ""
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "agent": self.agent,
            "seed": self.seed,
            "judge": self.judge,
            "model": self.model,
            "score": self.score,
            "success": self.success,
            "meta": self.meta,
            "steps": [s.to_dict() for s in self.steps],
        }

    @property
    def n_tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.kind == "tool")

    @property
    def total_latency_ms(self) -> float:
        return float(sum(s.latency_ms for s in self.steps))

    @property
    def total_tokens(self) -> int:
        return int(sum(s.tokens_in + s.tokens_out for s in self.steps))

    def signature(self) -> str:
        return trace_signature(self)


def trace_signature(ep: Episode) -> str:
    """Hash covering the *decisions*, not the environment's answers.

    Two runs with the same signature made the same sequence of choices; they can
    still differ if a tool returned something different, which is exactly the
    split we want to be able to measure.
    """
    decision = [{"k": s.kind, "p": s.payload} for s in ep.steps]
    return content_hash({"agent": ep.agent, "seed": ep.seed,
                         "model": ep.model, "decision": decision})


def hash_episode(ep: Episode) -> str:
    """Full-fidelity hash including tool responses and timings."""
    return content_hash(ep.to_dict())


def verify_replay(original: Episode, replayed: Episode) -> dict:
    """Compare a replay against the recording and classify any divergence."""
    sig_same = original.signature() == replayed.signature()
    full_same = hash_episode(original) == hash_episode(replayed)
    resp_orig = [s.payload for s in original.steps if s.kind == "tool"]
    resp_repl = [s.payload for s in replayed.steps if s.kind == "tool"]

    if full_same:
        kind = "identical"
    elif sig_same:
        kind = "environment_drift"
    else:
        kind = "model_drift"

    return {
        "verdict": kind,
        "bitwise_identical": full_same,
        "same_decisions": sig_same,
        "n_steps_original": len(original.steps),
        "n_steps_replayed": len(replayed.steps),
        "tool_responses_match": resp_orig == resp_repl,
        "n_tool_responses": len(resp_orig),
        "first_divergent_step": next(
            (i for i, (a, b) in enumerate(
                zip(original.steps, replayed.steps)) if a.step_hash != b.step_hash),
            None),
        "original_hash": hash_episode(original),
        "replay_hash": hash_episode(replayed),
    }


def build_episode(task_id: str, agent: str, seed: int, judge: str,
                  events: Iterable[dict], score: float,
                  success: bool | None = None, model: str = "",
                  meta: dict | None = None) -> Episode:
    """Assemble an episode from a flat event stream."""
    steps = tuple(
        TraceStep(
            index=int(e.get("index", i)),
            kind=str(e.get("kind", "unknown")),
            payload=e.get("payload", {}) or {},
            latency_ms=float(e.get("latency_ms", 0.0)),
            tokens_in=int(e.get("tokens_in", 0)),
            tokens_out=int(e.get("tokens_out", 0)),
        )
        for i, e in enumerate(events)
    )
    return Episode(task_id=task_id, agent=agent, seed=seed, judge=judge,
                   steps=steps, score=float(score), success=success,
                   model=model, meta=dict(meta or {}))

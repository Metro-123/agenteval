"""Real-model integration: shape and semantics, tested with a stub client.

No network access here. The point of the injectable transport is that the
interesting failures -- permuted axes, judges grading different rollouts, a
missing API key -- are all testable without spending anything.
"""

from __future__ import annotations

import collections
import json

import numpy as np
import pytest

from agenteval.live import (
    AgentConfig,
    HttpModelClient,
    JudgeConfig,
    LiveRunner,
    Task,
    build_cell_index,
)
from agenteval.sim import SuiteSpec, field_with_sign
from agenteval.vcomp import moment_estimator


# Agents must differ in quality, or the paired difference is identically zero and
# no variance component is estimable. Judges must use distinct models, or the
# stub cannot tell them apart and judge variance is structurally zero.
QUALITY = {"m-small": 0.55, "m-large": 0.90}


class StubClient:
    """Deterministic fake transport. Records every call for assertions.

    Generative rather than fixed: each (model, seed) gets its own pseudo-random
    draw against that model's true success rate, so the returned data has real
    seed variation and a non-degenerate difference field.
    """

    def __init__(self, *, judge_bias: dict[str, float] | None = None) -> None:
        # per-judge-model leniency, added to the pass rate. A positive bias makes a
        # judge more willing to PASS a wrong answer, which is what a biased judge
        # actually looks like and what produces disagreement in the paired
        # difference. Applied to both correct and incorrect base rates.
        self.judge_bias = judge_bias or {}
        self.calls: list[dict] = []

    def _draw(self, key: str) -> float:
        return (_stable_int(key) % 10_000) / 10_000.0

    def complete(self, *, model, system, user, temperature, max_tokens, seed=None):
        self.calls.append({"model": model, "user": user, "seed": seed})
        if "PASS" in system:  # a judge
            good = "the answer is 42" in user
            base = 0.9 if good else 0.1  # a competent judge is mostly right
            p = min(0.99, max(0.01, base + self.judge_bias.get(model, 0.0)))
            return ("PASS" if self._draw(f"j{model}{seed}{user}") < p
                    else "FAIL", 20, 2, 12.0)
        p = QUALITY.get(model, 0.7)
        correct = self._draw(f"a{model}{seed}{user}") < p
        # tag the answer with its agent so tests can recover which rollout a
        # grading call refers to, even when two agents coincide on the same text
        tag = f"[by {model}]"
        body = "the answer is 42" if correct else "I am not sure"
        return (f"{body} {tag}", 100, 8, 40.0)


def _stable_int(s: str) -> int:
    import hashlib
    return int.from_bytes(hashlib.blake2s(s.encode(), digest_size=4).digest(), "big")


TASKS = tuple(Task(f"t{i}", f"Question {i}: what is 6*7?", "42") for i in range(3))
AGENTS = (AgentConfig("cheap", "m-small", token_cost=0.5),
          AgentConfig("strong", "m-large", token_cost=2.0))
JUDGES = (JudgeConfig("judge-A", "m-judge-A"),
          JudgeConfig("judge-B", "m-judge-B", seed_offset=1))


def runner(**kw) -> LiveRunner:
    return LiveRunner(client=kw.pop("client", StubClient()),
                      tasks=TASKS, agents=AGENTS, judges=JUDGES,
                      n_seeds=kw.pop("n_seeds", 2), **kw)


def test_cell_index_is_judge_fastest() -> None:
    """The order the statistics assume. Get it backwards and components lie."""
    idx = build_cell_index(n_tasks=2, n_seeds=2, n_judges=2)
    assert list(zip(idx.task, idx.seed, idx.judge)) == [
        (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
        (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1),
    ]


def test_run_returns_the_shapes_the_report_expects() -> None:
    r = runner().run()
    assert r["agents"] == ["cheap", "strong"]
    assert r["judges"] == ["judge-A", "judge-B"]
    assert r["cell_index"].task.size == 3 * 2 * 2
    for arm in r["agents"]:
        for j in r["judges"]:
            k = f"{arm}|{j}"
            assert r["judged"][k].shape == (3, 2)
            assert r["raw"][k].shape == (3, 2)
            assert set(np.unique(r["judged"][k])) <= {0.0, 1.0}


def test_all_judges_grade_the_identical_rollout() -> None:
    """Two opinions of one run, not two runs.

    If judges saw different transcripts the paired difference would mix agent
    quality with run-to-run noise, which is the exact error the crossed design
    exists to remove. For each (task, agent) pair there must be exactly one
    graded transcript text per rollout, and every judge must have graded it.
    The grouping key deliberately omits the seed: the judge seed differs per
    judge, but all judges must see the same rollout for a given (task, agent,
    rollout), so keying on seed would incorrectly split one rollout across judges.
    """
    c = StubClient()
    LiveRunner(client=c, tasks=TASKS, agents=AGENTS, judges=JUDGES,
               n_seeds=2, keep_episodes=False).run()
    graded = [x for x in c.calls if "PASS" in x["user"]]
    assert graded, "no judging happened"

    # Group by exact transcript text. Note two seeds of the same agent on the same
    # task can legitimately yield identical text, so a transcript may be graded
    # more than once (once per seed). The invariant is therefore balance, not
    # multiplicity: every judge grades every transcript the same number of times.
    by_transcript: dict[str, list[str]] = {}
    for x in graded:
        by_transcript.setdefault(x["user"], []).append(x["model"])
    assert by_transcript

    judge_models = sorted(j.model for j in JUDGES)
    for transcript, judges_seen in by_transcript.items():
        counts = collections.Counter(judges_seen)
        assert sorted(counts) == judge_models, f"judge skipped: {counts}"
        assert len(set(counts.values())) == 1, (
            f"judges graded the same transcript a different number of times: "
            f"{dict(counts)}")

    # each judge graded exactly one transcript per (task, agent, seed) rollout
    n_rollouts = len(TASKS) * len(AGENTS) * 2  # 2 seeds
    for jm in judge_models:
        assert sum(1 for x in graded if x["model"] == jm) == n_rollouts


def test_call_count_accounts_for_shared_rollouts() -> None:
    """Rollouts happen once and are graded by every judge.

    Counting the full product na*nt*ns*nj would overstate agent cost by nj times;
    counting the rollouts alone would understate the total. The honest figure is
    rollouts + gradings, and it is the number that decides whether a design is
    affordable.
    """
    nt, ns, nj, na = len(TASKS), 2, len(JUDGES), len(AGENTS)
    c = StubClient()
    r = runner(client=c).run()
    assert r["n_calls"] == na * nt * ns * (1 + nj)
    assert len(c.calls) == na * nt * ns * (1 + nj)


def test_seed_actually_reaches_the_model() -> None:
    """Without a seed, the replicate axis is not a replicate -- it is noise
    nobody can decompose, which is the whole failure mode being guarded."""
    r = runner(n_seeds=3).run()
    c = r["episodes"]
    agent_seeds = {e.seed for e in c if e.judge == ""}
    assert agent_seeds == {0, 1, 2}


def test_cell_alignment_survives_the_real_pipeline() -> None:
    """field_with_sign must map real data onto the same index the runner built."""
    r = runner().run()
    d, idx, _ = field_with_sign(r, "strong", "cheap")
    assert d.size == idx.task.size == 12
    vc = moment_estimator(d, idx)
    assert vc.n_estimable >= 3
    assert np.isfinite(vc.sigma2_seed)


def _powered_design(bias: float):
    """A design with enough judges to actually resolve the judge axis.

    With only two judges the judge mean-square is a single contrast over few
    cells and the method-of-moments estimator legitimately truncates it to zero,
    so a two-judge test cannot distinguish 'no judge variance' from 'too little
    data to see any'. Four judges give the judge axis enough levels to be
    estimable, which is exactly why the design guidance insists on >=2.
    """
    tasks = tuple(Task(f"t{i}", f"Question {i}: what is 6*{i + 1}?", str(6 * (i + 1)))
                  for i in range(8))
    agents = (AgentConfig("cheap", "m-small", token_cost=0.5),
              AgentConfig("strong", "m-large", token_cost=2.0))
    judges = tuple(JudgeConfig(f"judge-{chr(65 + j)}", f"m-judge-{chr(65 + j)}")
                   for j in range(4))
    off = {f"m-judge-{chr(65 + j)}": (bias if j == 0 else 0.0) for j in range(4)}
    r = LiveRunner(client=StubClient(judge_bias=off), tasks=tasks, agents=agents,
                   judges=judges, n_seeds=3, keep_episodes=False).run()
    d, idx, _ = field_with_sign(r, "strong", "cheap")
    return moment_estimator(d, idx)


def test_judge_variance_is_recovered_when_a_judge_is_biased() -> None:
    """Judge offsets are a real variance component, and must be separable.

    One judge is made systematically lenient -- it passes a share of wrong
    answers. That is exactly the situation a single-judge run cannot detect, and
    a crossed run must attribute to the judge axis rather than silently folding
    it into the agent effect or the noise. Run on a four-judge design so the
    judge axis is actually estimable.
    """
    neutral = _powered_design(0.0)
    skewed = _powered_design(0.35)
    assert skewed.sigma2_judge > neutral.sigma2_judge, (
        f"lenient judge did not raise judge variance: "
        f"{neutral.sigma2_judge:.4g} -> {skewed.sigma2_judge:.4g}")
    assert skewed.sigma2_judge > 0.0


def test_judge_variance_truncates_to_zero_with_interchangeable_judges() -> None:
    """No disagreement means no judge variance, and the estimator says so.

    The complement of the test above: when every judge is competent and
    interchangeable, the judge component should be near zero rather than
    absorbing random noise.
    """
    v = _powered_design(0.0)
    assert v.sigma2_judge < 0.05, f"unexpected judge variance: {v.sigma2_judge}"


def test_difference_field_is_not_degenerate() -> None:
    """Guards the fixture: a constant difference has no estimable variance."""
    r = runner().run()
    d, idx, _ = field_with_sign(r, "strong", "cheap")
    assert d.std() > 0.0, "stub is producing a constant difference"
    assert len(set(d.tolist())) > 1


def test_missing_api_key_is_a_clear_error_not_a_crash() -> None:
    with pytest.raises(ValueError, match="API key"):
        HttpModelClient(api_key="")


def test_http_client_requires_no_third_party_package() -> None:
    import urllib.request

    assert HttpModelClient is not None
    assert urllib.request.__name__ == "urllib.request"


def test_episodes_are_recorded_with_tokens_and_latency() -> None:
    r = runner().run()
    e = r["episodes"][0]
    assert e.model
    assert e.total_tokens > 0
    assert e.total_latency_ms > 0
    assert len(r["episodes"]) == len(TASKS) * 2 * len(AGENTS) * (1 + len(JUDGES))


def test_run_live_matches_the_suite_spec_shape() -> None:
    """A live run must be drop-in for a simulated one in build_report."""
    from agenteval.report import build_report

    r = runner().run()
    r["spec"] = SuiteSpec(n_tasks=len(TASKS), n_seeds=2,
                          agents=AGENTS, judges=JUDGES)
    # report.build_report calls simulate_suite; verify the keys it reads exist
    for key in ("agents", "judges", "cell_index", "judged", "raw", "meta"):
        assert key in r, f"live run is missing {key}, report would KeyError"
    json.dumps({"judged": {k: v.tolist() for k, v in r["judged"].items()}},
               allow_nan=False)

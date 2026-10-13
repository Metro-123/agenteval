"""Ollama transport, tested against a real local HTTP server.

These tests bind a socket and exchange real HTTP with ``fake_ollama``. They cover
the failure modes that only appear once a transport is real: wrong URL, wrong
payload shape, unparsed token fields, and an unpulled model turning into an
opaque 404. No network egress and no model weights.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import fake_ollama
from agenteval.live import (
    AgentConfig,
    JudgeConfig,
    OllamaClient,
    OllamaUnavailable,
    Task,
    list_ollama_models,
    load_tasks_jsonl,
    preflight,
)


@pytest.fixture
def ollama():
    url, state, shutdown = fake_ollama.serve()
    try:
        yield url, state
    finally:
        shutdown()


def _client(url, **kw) -> OllamaClient:
    # retry_base_delay=0 keeps the suite fast; backoff itself is not what's
    # under test, the request/retry plumbing is
    return OllamaClient(base_url=url, max_retries=kw.pop("max_retries", 3),
                        retry_base_delay=kw.pop("retry_base_delay", 0.0), **kw)


# ---------------------------------------------------------------- discovery

def test_lists_installed_models(ollama) -> None:
    url, _ = ollama
    names = [m["name"] for m in list_ollama_models(url)]
    assert "qwen2.5:0.5b" in names


def test_unreachable_daemon_explains_how_to_start_it() -> None:
    with pytest.raises(OllamaUnavailable) as exc:
        # port 1 is reserved and refuses
        list_ollama_models("http://127.0.0.1:1")
    assert "ollama serve" in str(exc.value)


# ---------------------------------------------------------------- transport

def test_chat_returns_text_tokens_and_latency(ollama) -> None:
    url, _ = ollama
    text, tin, tout, ms = _client(url).complete(
        model="qwen2.5:0.5b", system="be helpful", user="what is 6*7?",
        temperature=0.7, max_tokens=16, seed=7)
    assert "42" in text
    assert tin > 0 and tout > 0
    assert ms > 0


def test_seed_and_options_reach_the_native_api(ollama) -> None:
    """The whole seed axis depends on options.seed actually being sent.

    Ollama's OpenAI-compatible shim can drop it; the native API does not. If this
    ever regresses, seeds silently become no-ops and sigma2_seed reads 0.
    """
    url, state = ollama
    _client(url).complete(model="qwen2.5:0.5b", system="s", user="u",
                          temperature=0.7, max_tokens=8, seed=4242)
    req = state["requests"][-1]
    assert req["options"]["seed"] == 4242
    assert req["options"]["temperature"] == 0.7
    assert req["options"]["num_predict"] == 8
    assert req["stream"] is False
    assert req["keep_alive"] == "10m"


def test_no_authorization_header_is_sent(ollama) -> None:
    """Local inference needs no credential, and sending one would be a leak."""
    url, state = ollama
    _client(url).complete(model="qwen2.5:0.5b", system="s", user="u",
                          temperature=0.0, max_tokens=4, seed=0)
    sent = {k.lower() for k in state["headers"][-1]}
    assert "authorization" not in sent
    assert "x-api-key" not in sent


def test_transient_503_is_retried(ollama) -> None:
    url, state = ollama
    state["fail_next"] = 2
    text, _, _, _ = _client(url).complete(model="qwen2.5:0.5b", system="s",
                                          user="u", temperature=0.0,
                                          max_tokens=4, seed=0)
    assert "42" in text
    assert len(state["requests"]) == 3  # two failures, one success


def test_retry_budget_is_respected(ollama) -> None:
    url, state = ollama
    state["fail_next"] = 99
    with pytest.raises(RuntimeError, match="failed after"):
        _client(url, max_retries=2).complete(model="qwen2.5:0.5b", system="s",
                                             user="u", temperature=0.0,
                                             max_tokens=4)
    assert len(state["requests"]) == 2


def test_4xx_is_not_retried_and_says_how_to_fix_it(ollama) -> None:
    """Retrying a client error just wastes time; only transient faults repeat.

    The 404 must also be actionable: an unpulled model is the single most common
    first-run failure, and a bare "404 Not Found" sends people hunting.
    """
    url, state = ollama
    with pytest.raises(OllamaUnavailable) as exc:
        _client(url).complete(model="not-pulled:99b", system="s", user="u",
                              temperature=0.0, max_tokens=4)
    assert "ollama pull not-pulled:99b" in str(exc.value)
    assert len(state["requests"]) == 1


# ---------------------------------------------------------------- task files

def test_loads_tasks_from_jsonl(tmp_path) -> None:
    p = tmp_path / "tasks.jsonl"
    p.write_text(
        '{"task_id": "q1", "prompt": "2+2?", "answer": "4"}\n'
        "\n"
        '// a comment\n'
        '{"prompt": "capital of France?", "answer": "Paris"}\n',
        encoding="utf-8")
    tasks = load_tasks_jsonl(str(p))
    assert [t.task_id for t in tasks] == ["q1", "t1"]
    assert tasks[1].check("The answer is Paris.")


def test_bad_line_names_its_line_number(tmp_path) -> None:
    p = tmp_path / "tasks.jsonl"
    p.write_text('{"prompt": "ok", "answer": "1"}\n{"answer": "no prompt"}\n',
                 encoding="utf-8")
    with pytest.raises(ValueError, match=r":2:"):
        load_tasks_jsonl(str(p))


def test_empty_file_is_an_error_not_an_empty_run(tmp_path) -> None:
    p = tmp_path / "tasks.jsonl"
    p.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no tasks"):
        load_tasks_jsonl(str(p))


# ---------------------------------------------------------------- preflight

def _cfg():
    tasks = (Task("t0", "6*7?", "42"),)
    agents = (AgentConfig("a", "qwen2.5:0.5b", temperature=0.7),)
    judges = (JudgeConfig("j1", "judge:1b"), JudgeConfig("j2", "judge:1b"))
    return tasks, agents, judges


def test_preflight_flags_unpulled_model(ollama) -> None:
    url, _ = ollama
    tasks, agents, judges = _cfg()
    agents = (AgentConfig("a", "ghost:7b", temperature=0.7),)
    w = preflight(_client(url), tasks, agents, judges)
    assert any("ollama pull ghost:7b" in x for x in w)


def test_preflight_flags_deterministic_temperature(ollama) -> None:
    """temperature=0 makes every seed identical, voiding the crossed design.

    This is the trap that produces a confident, meaningless sigma2_seed = 0.
    """
    url, _ = ollama
    tasks, _, judges = _cfg()
    agents = (AgentConfig("a", "qwen2.5:0.5b", temperature=0.0),)
    w = preflight(_client(url), tasks, agents, judges)
    assert any("temperature=0.0" in x for x in w)


def test_preflight_flags_single_judge(ollama) -> None:
    url, _ = ollama
    tasks, agents, _ = _cfg()
    w = preflight(_client(url), tasks, agents, (JudgeConfig("j1", "judge:1b"),))
    assert any("unidentifiable" in x for x in w)


def test_preflight_is_quiet_on_a_sane_design(ollama) -> None:
    url, _ = ollama
    assert preflight(_client(url), *_cfg()) == []


# ---------------------------------------------------------------- end to end

def test_runner_produces_report_shapes_through_ollama(ollama) -> None:
    """A real HTTP run returns the same structures the simulator does."""
    from agenteval.live import LiveRunner

    url, _ = ollama
    tasks = (Task("t0", "6*7?", "42"), Task("t1", "6*8?", "48"))
    agents = (AgentConfig("a", "qwen2.5:0.5b", temperature=0.7),
              AgentConfig("b", "llama3.2:1b", temperature=0.7))
    judges = (JudgeConfig("j1", "judge:1b"), JudgeConfig("j2", "judge:1b"))
    r = LiveRunner(client=_client(url), tasks=tasks, agents=agents, judges=judges,
                   n_seeds=2).run()
    assert r["agents"] == ["a", "b"]
    assert r["cell_index"].judge.size == 2 * 2 * 2
    for arm in r["agents"]:
        for j in r["judges"]:
            assert r["judged"][f"{arm}|{j}"].shape == (2, 2)
    json.dumps({k: v.tolist() for k, v in r["judged"].items()}, allow_nan=False)


def test_live_run_flows_through_build_report_unchanged(ollama) -> None:
    """The integration claim: same report code, different data source.

    If this passes, every statistic in the dashboard is already correct for real
    runs -- the only thing that changes is where the numbers came from.
    """
    from agenteval.live import LiveRunner
    from agenteval.report import build_report

    url, _ = ollama
    tasks = (Task("t0", "6*7?", "42"), Task("t1", "6*8?", "48"))
    agents = (AgentConfig("a", "qwen2.5:0.5b", temperature=0.7),
              AgentConfig("b", "llama3.2:1b", temperature=0.7))
    judges = (JudgeConfig("j1", "judge:1b"), JudgeConfig("j2", "judge:1b"))
    live = LiveRunner(client=_client(url), tasks=tasks, agents=agents,
                      judges=judges, n_seeds=2).run()

    rep = build_report(suite=live, n_boot=100)
    json.dumps(rep, allow_nan=False)          # strict JSON, no NaN
    assert {r["agent"] for r in rep["leaderboard"]} == {"a", "b"}
    assert rep["judges"]["n_judges"] == 2
    # the design was inferred from the data, not restated by the caller
    assert rep["meta"]["n_tasks"] == 2
    assert rep["meta"]["n_seeds"] == 2
    assert rep["meta"]["n_agents"] == 2
    assert rep["meta"]["n_cells"] == 2 * 2 * 2
    for row in rep["leaderboard"]:
        assert row["se"] is not None
        assert row["ci_low"] is not None and row["ci_high"] is not None
    # both estimator families ran on the live data
    for cmp_ in rep["comparisons"]:
        assert set(cmp_["estimators"]) >= {"naive", "crossed"}
        mom = cmp_["variance"]["mom"]
        assert mom["components"]["judge"] is not None
        assert cmp_["variance"]["n_judges"] == 2


def test_pareto_has_real_cost_and_latency_on_a_live_run(ollama) -> None:
    """A live run must populate the cost/latency axes, not leave them null.

    The report's Pareto block reads fixed meta key names. A live run that emitted
    differently-named keys would not raise -- it would quietly render an empty
    cost/latency view, which is exactly the data you need to choose between
    local models. Regression test for that silent failure.
    """
    from agenteval.live import LiveRunner
    from agenteval.report import build_report

    url, _ = ollama
    tasks = (Task("t0", "6*7?", "42"), Task("t1", "6*8?", "48"))
    agents = (AgentConfig("a", "qwen2.5:0.5b", temperature=0.7),
              AgentConfig("b", "llama3.2:1b", temperature=0.7))
    judges = (JudgeConfig("j1", "judge:1b"), JudgeConfig("j2", "judge:1b"))
    live = LiveRunner(client=_client(url), tasks=tasks, agents=agents,
                      judges=judges, n_seeds=2).run()
    rep = build_report(suite=live, n_boot=50)

    for row in rep["pareto"]:
        assert row["cost"] is not None, f"{row['agent']} has no measured cost"
        assert row["latency"] is not None, f"{row['agent']} has no measured latency"
        assert row["cost"] > 0 and row["latency"] > 0


def test_live_and_simulated_suites_share_the_meta_contract(ollama) -> None:
    """Every meta key the report reads must exist in a live run.

    Compares against the simulated suite as the reference, so a future rename on
    either side is caught here instead of rendering as null in the dashboard.
    """
    from agenteval.live import LiveRunner
    from agenteval.report import build_report
    from agenteval.sim import SuiteSpec, simulate_suite

    required = {"cost_per_task_tokens", "latency_s_per_task"}
    sim = simulate_suite(SuiteSpec())
    for arm in sim["agents"]:
        assert required <= set(sim["meta"][arm]), \
            f"simulator itself lost {required - set(sim['meta'][arm])}"

    url, _ = ollama
    tasks = (Task("t0", "6*7?", "42"),)
    agents = (AgentConfig("a", "qwen2.5:0.5b", temperature=0.7),
              AgentConfig("b", "llama3.2:1b", temperature=0.7))
    judges = (JudgeConfig("j1", "judge:1b"), JudgeConfig("j2", "judge:1b"))
    live = LiveRunner(client=_client(url), tasks=tasks, agents=agents,
                      judges=judges, n_seeds=2).run()
    for arm in live["agents"]:
        assert required <= set(live["meta"][arm]), \
            f"live run missing {required - set(live['meta'][arm])}"


def test_measured_cost_excludes_judge_spend(ollama) -> None:
    """An arm's own cost must not be inflated by the judges grading it.

    Judges add tokens and latency, but those are attributable to grading, not to
    the agent. Mixing them would make every arm look equally expensive and flatten
    the cost axis the planner is trying to optimise.
    """
    from agenteval.live import LiveRunner

    url, _ = ollama
    tasks = (Task("t0", "6*7?", "42"),)
    agents = (AgentConfig("a", "qwen2.5:0.5b", temperature=0.7),)
    judges = (JudgeConfig("j1", "judge:1b"), JudgeConfig("j2", "judge:1b"))
    live = LiveRunner(client=_client(url), tasks=tasks, agents=agents,
                      judges=judges, n_seeds=2).run()
    meta = live["meta"]["a"]
    agent_eps = [e for e in live["episodes"] if e.judge == ""]
    assert meta["cost_per_task_tokens"] == round(
        sum(e.total_tokens for e in agent_eps) / len(agent_eps))
    # judge spend is reported separately rather than folded in
    assert "judge_tokens_per_rollout" in meta


def test_identical_arms_do_not_break_the_report(ollama) -> None:
    """Two small local models will often score identically. That must not crash.

    scipy's ttest_ind warns about catastrophic cancellation on constant data, and
    this project turns RuntimeWarnings into errors, so an ordinary tie between
    agents would have taken the whole report down. Regression test for that.
    """
    from agenteval.live import LiveRunner
    from agenteval.report import build_report

    url, _ = ollama
    tasks = (Task("t0", "6*7?", "42"), Task("t1", "6*8?", "48"))
    agents = (AgentConfig("a", "qwen2.5:0.5b", temperature=0.7),
              AgentConfig("b", "qwen2.5:0.5b", temperature=0.7))
    judges = (JudgeConfig("j1", "judge:1b"), JudgeConfig("j2", "judge:1b"))
    live = LiveRunner(client=_client(url), tasks=tasks, agents=agents,
                      judges=judges, n_seeds=2).run()
    # the fake is deterministic, so both arms genuinely score the same
    assert np.array_equal(live["judged"]["a|j1"], live["judged"]["b|j1"])

    rep = build_report(suite=live, n_boot=100)   # must not raise
    json.dumps(rep, allow_nan=False)
    for cmp_ in rep["comparisons"]:
        for name, est in cmp_["estimators"].items():
            assert est["p_value"] is not None, f"{name} produced no p-value"
            if abs(est["estimate"]) < 1e-12 and est["se"] == 0.0:
                # a genuine tie is not evidence of a difference
                assert est["p_value"] == 1.0, f"{name}: tie reported p={est['p_value']}"


def test_ragged_suite_is_rejected_rather_than_misanalysed(ollama) -> None:
    """A suite whose matrices disagree must fail loudly, not silently."""
    from agenteval.live import LiveRunner
    from agenteval.report import build_report

    url, _ = ollama
    tasks = (Task("t0", "6*7?", "42"),)
    agents = (AgentConfig("a", "qwen2.5:0.5b", temperature=0.7),)
    judges = (JudgeConfig("j1", "judge:1b"), JudgeConfig("j2", "judge:1b"))
    live = LiveRunner(client=_client(url), tasks=tasks, agents=agents,
                      judges=judges, n_seeds=2).run()
    live["judged"]["a|j2"] = np.zeros((5, 5))
    with pytest.raises(ValueError, match="ragged"):
        build_report(suite=live)

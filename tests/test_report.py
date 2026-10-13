from __future__ import annotations

import json
import math

import pytest

from agenteval.report import build_report
from agenteval.sim import AgentSpec, JudgeSpec, SuiteSpec, default_agents, default_judges


@pytest.fixture(scope="module")
def report() -> dict:
    # 48 tasks x 3 seeds x 3 agents = 432 judged items, enough for the planted
    # operating points to be recovered to within a couple of standard errors.
    spec = SuiteSpec(n_tasks=48, n_seeds=3, seed=5,
                     agents=(AgentSpec("cheap", skill=0.6, seed_noise=0.3,
                                       tool_noise=0.2, token_cost=0.6),
                             AgentSpec("mid", skill=1.4, seed_noise=0.5,
                                       tool_noise=0.3, token_cost=1.2),
                             AgentSpec("pricey", skill=2.0, seed_noise=0.7,
                                       tool_noise=0.5, token_cost=2.5)),
                     judges=default_judges())
    return build_report(spec, n_boot=150)


def test_simulator_is_reproducible_across_processes() -> None:
    """The same spec must give the same data in a fresh interpreter.

    The per-agent streams were seeded with the builtin ``hash`, which Python
    randomises per process. The simulator *looked* seeded -- the seed was right
    there in the call -- but every run produced different data, which is what made
    downstream thresholds flake between test runs.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "src"
    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "from agenteval.sim import SuiteSpec, simulate_suite, default_agents, default_judges;"
        "s = simulate_suite(SuiteSpec(agents=default_agents(), judges=default_judges()));"
        "a = s['agents'][0] + '|' + s['judges'][0];"
        "print(int(s['raw'][a].sum()), int(s['judged'][a].sum()))" % src
    )
    out = []
    for hash_seed in ("1", "999"):
        env = {**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": str(src)}
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, env=env, timeout=300, check=True)
        out.append(r.stdout.strip())
    assert out[0] == out[1], f"simulator differs across processes: {out}"


def test_report_is_strict_json(report: dict) -> None:
    """The dashboard consumes this verbatim, so it must be real JSON.

    ``allow_nan=False`` is the point: ``json.dumps`` will write a bare ``NaN``,
    which ``JSON.parse`` rejects outright, taking the whole page down.
    """
    text = json.dumps(report, allow_nan=False)
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == report


def test_report_covers_every_agent_against_the_baseline(report: dict) -> None:
    board = report["leaderboard"]
    assert len(board) == 3
    assert sum(r["is_baseline"] for r in board) == 1
    baseline = report["meta"]["baseline"]
    assert baseline in [r["agent"] for r in board]
    arms = {c["arm"] for c in report["comparisons"]}
    assert arms == set(board and {r["agent"] for r in board}) - {baseline}


def test_naive_interval_is_flagged_as_too_narrow(report: dict) -> None:
    """The headline claim of the report has to be true of its own numbers."""
    ratios = [c["se_inflation_naive_to_crossed"] for c in report["comparisons"]]
    assert all(r > 1.0 for r in ratios)
    for c in report["comparisons"]:
        est = c["estimators"]
        assert est["naive"]["se"] < est["crossed"]["se"]
        # the crossed interval must never be the narrower one
        assert est["crossed"]["ci_high"] - est["crossed"]["ci_low"] >= \
            est["naive"]["ci_high"] - est["naive"]["ci_low"]


def test_all_estimators_are_reported_and_finite(report: dict) -> None:
    for c in report["comparisons"]:
        est = c["estimators"]
        assert set(est) == {"naive", "cluster", "crossed", "bootstrap"}
        for name in ("naive", "cluster", "crossed", "bootstrap"):
            e = est[name]
            assert math.isfinite(e["se"]) and e["se"] > 0.0
            assert e["ci_low"] < e["ci_high"]
            assert math.isfinite(e["p_value"])


def test_bias_correction_recovers_the_pooled_true_rate(report: dict) -> None:
    """Four judges with very different bias must agree once inverted.

    This is the check that the correction is actually doing something: the raw
    positive rates disagree by ~10 points, and after correction every judge
    should land on the same latent rate.
    """
    jb = report["judges"]
    corrected = [j["bias_corrected_rate"] for j in jb["judges"].values()]
    assert max(corrected) - min(corrected) < 0.05
    truth = jb["true_positive_rate"]
    assert all(abs(c - truth) < 0.08 for c in corrected)
    # and the correction must actually move the number
    raw = [j["observed_positive_rate"] for j in jb["judges"].values()]
    assert max(raw) - min(raw) > 0.05


def test_dawid_skene_recovers_planted_operating_points(report: dict) -> None:
    """Every judge's operating point must be recovered within sampling error.

    The weakest judge here (sensitivity 0.74) is also the one estimated from the
    fewest of its own decisions, so its error is the largest; 0.12 is roughly two
    standard errors at this sample size rather than a tight bound.
    """
    for j in report["judges"]["judges"].values():
        assert abs(j["sensitivity_error"]) < 0.12, j["name"]
        assert abs(j["specificity_error"]) < 0.12, j["name"]


def test_unidentifiable_design_produces_a_warning() -> None:
    """One judge means its variance is unidentifiable; the report must say so."""
    spec = SuiteSpec(n_tasks=12, n_seeds=2, seed=3,
                     agents=(AgentSpec("a", skill=1.0), AgentSpec("b", skill=1.5)),
                     judges=(JudgeSpec("solo", sensitivity=0.9, specificity=0.8, seed=1),))
    rep = build_report(spec, n_boot=60)
    assert rep["warnings"], "a single-judge run must warn"
    assert any("judge" in w for w in rep["warnings"])
    assert rep["planner"]["unidentifiable"]


def test_planner_recommends_at_least_two_levels_per_axis(report: dict) -> None:
    rec = report["planner"].get("recommended")
    assert rec is not None
    assert rec["n_seeds"] >= 2 and rec["n_judges"] >= 2
    assert rec["power"] >= 0.75
    assert rec["cost"] > 0.0
    # the current 16x3x4 design cannot reach the target the planner is asked about
    mdd = report["planner"]["min_detectable_delta"]
    assert mdd["at_design"] > mdd["at_8x8"] > mdd["at_32x32"] > 0.0
    assert mdd["at_design"] > report["meta"]["target_delta"]


def test_cost_effect_curve_never_shrinks_as_the_design_grows() -> None:
    """More of any axis must never make a target effect less detectable."""
    spec = SuiteSpec(n_tasks=8, n_seeds=3, seed=2, n_tools=2,
                     agents=default_agents(), judges=default_judges())
    rep = build_report(spec, n_boot=60)
    table = rep["planner"]["table"]
    by_shape: dict[tuple[int, int], float] = {}
    for row in table:
        key = (row["n_tasks"], row["n_seeds"])
        half = row.get("ci_half_width")
        if half is not None and math.isfinite(half):
            prev = by_shape.get(key)
            if prev is not None:
                assert half <= prev + 1e-9
            by_shape[key] = half
    # and holding tasks/seeds fixed, more judges never hurts
    judges_by_key: dict[tuple[int, int], dict[int, float]] = {}
    for row in table:
        half = row.get("ci_half_width")
        if half is None or not math.isfinite(half):
            continue
        k = (row["n_tasks"], row["n_seeds"])
        cur = judges_by_key.setdefault(k, {})
        prev = cur.get(row["n_judges"])
        if prev is not None:
            assert half <= prev + 1e-9
        cur[row["n_judges"]] = half

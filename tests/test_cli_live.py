"""The `run` command, driven end to end against a local fake Ollama.

These exercise the path a user actually walks: write a config, run the CLI, get a
report. No model weights and no network -- the fake server binds loopback only.
"""

from __future__ import annotations

import json

import pytest

import fake_ollama
from agenteval.cli import main


@pytest.fixture
def ollama():
    url, state, shutdown = fake_ollama.serve()
    try:
        yield url, state
    finally:
        shutdown()


def _write(tmp_path, *, agents=2, judges=2, n_seeds=2, extra=None):
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        '{"task_id": "t0", "prompt": "what is 6*7?", "answer": "42"}\n'
        '{"task_id": "t1", "prompt": "what is 6*8?", "answer": "48"}\n',
        encoding="utf-8")
    cfg = {
        "tasks": str(tasks),
        "n_seeds": n_seeds,
        "agents": [{"name": "small", "model": "qwen2.5:0.5b",
                    "temperature": 0.7, "max_tokens": 32}] * agents,
        "judges": [{"name": f"j{i}", "model": "judge:1b"} for i in range(judges)],
    }
    cfg["agents"][1] = {"name": "large", "model": "llama3.2:1b",
                        "temperature": 0.7, "max_tokens": 32}
    if extra:
        cfg.update(extra)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def _run(path, url, tmp_path, *extra):
    return main(["run", str(path), "--base-url", url, "--yes",
                 "--out", str(tmp_path / "r.json"),
                 "--episodes-out", str(tmp_path / "e.json"),
                 "--boot", "50", *extra])


def test_run_writes_a_report_with_populated_pareto(ollama, tmp_path) -> None:
    url, _ = ollama
    cfg = _write(tmp_path)
    assert _run(cfg, url, tmp_path) == 0

    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert {r["agent"] for r in report["leaderboard"]} == {"small", "large"}
    for row in report["pareto"]:
        assert row["cost"] is not None and row["latency"] is not None
    assert report["meta"]["n_seeds"] == 2


def test_run_writes_replayable_episodes(ollama, tmp_path) -> None:
    url, _ = ollama
    cfg = _write(tmp_path)
    assert _run(cfg, url, tmp_path) == 0
    eps = json.loads((tmp_path / "e.json").read_text(encoding="utf-8"))
    assert eps, "no episodes written"
    assert any(e["judge"] == "" for e in eps), "no agent rollouts"
    assert any(e["judge"] != "" for e in eps), "no judgements"
    for e in eps:
        assert "canonical_hash" in e or "hash" in e or "steps" in e


def test_report_is_strict_json_with_no_nan(ollama, tmp_path) -> None:
    """The dashboard would fail to parse a bare NaN, so it must never be written."""
    url, _ = ollama
    cfg = _write(tmp_path)
    assert _run(cfg, url, tmp_path) == 0
    text = (tmp_path / "r.json").read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text)          # would raise on NaN


def test_single_judge_is_refused_before_any_model_call(ollama, tmp_path) -> None:
    """One judge is unidentifiable; refusing up front saves the whole run."""
    url, state = ollama
    cfg = _write(tmp_path, judges=1)
    with pytest.raises(SystemExit, match="2 judges"):
        _run(cfg, url, tmp_path)
    assert state["requests"] == [], "made model calls despite refusing"


def test_single_agent_is_refused(ollama, tmp_path) -> None:
    url, _ = ollama
    cfg = _write(tmp_path)
    cfg_data = json.loads(cfg.read_text(encoding="utf-8"))
    cfg_data["agents"] = cfg_data["agents"][:1]
    cfg.write_text(json.dumps(cfg_data), encoding="utf-8")
    with pytest.raises(SystemExit, match="2 agents"):
        _run(cfg, url, tmp_path)


def test_deterministic_temperature_is_flagged_and_strict_blocks(ollama, tmp_path, capsys) -> None:
    """temperature=0 voids the seed axis, so --strict must refuse the run.

    --yes only skips the confirmation prompt; it must not quietly override
    --strict, which is a different decision.
    """
    url, state = ollama
    cfg = _write(tmp_path)
    data = json.loads(cfg.read_text(encoding="utf-8"))
    for a in data["agents"]:
        a["temperature"] = 0.0
    cfg.write_text(json.dumps(data), encoding="utf-8")

    assert _run(cfg, url, tmp_path, "--strict") == 3
    out = capsys.readouterr().out
    assert "temperature=0.0" in out
    assert "refusing to start" in out
    assert state["requests"] == [], "made model calls despite a strict warning"


def test_strict_does_not_block_a_clean_design(ollama, tmp_path) -> None:
    url, _ = ollama
    cfg = _write(tmp_path)
    assert _run(cfg, url, tmp_path, "--strict") == 0


def test_missing_config_is_a_clear_error(tmp_path) -> None:
    # Python prints a SystemExit message at the interpreter level, outside
    # capsys, so assert on the exception itself
    with pytest.raises(SystemExit) as exc:
        main(["run", str(tmp_path / "nope.json"), "--yes"])
    assert "config not found" in str(exc.value)


def test_invalid_json_config_names_the_problem(tmp_path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main(["run", str(bad), "--yes"])
    assert "not valid JSON" in str(exc.value)


def test_bad_tasks_file_is_reported_not_raised(ollama, tmp_path, capsys) -> None:
    url, _ = ollama
    cfg = _write(tmp_path)
    data = json.loads(cfg.read_text(encoding="utf-8"))
    data["tasks"] = str(tmp_path / "missing.jsonl")
    cfg.write_text(json.dumps(data), encoding="utf-8")
    assert _run(cfg, url, tmp_path) == 2
    assert "error:" in capsys.readouterr().err


def test_unreachable_ollama_fails_with_guidance_not_a_traceback(tmp_path, capsys) -> None:
    cfg = _write(tmp_path)
    # port 1 refuses, so this is the "Ollama isn't running" path
    assert _run(cfg, "http://127.0.0.1:1", tmp_path) == 2
    err = capsys.readouterr().err
    assert "ollama serve" in err
    assert "Traceback" not in err


def test_seeds_are_configurable_and_change_the_call_count(ollama, tmp_path) -> None:
    url, state = ollama
    cfg = _write(tmp_path, n_seeds=3)
    assert _run(cfg, url, tmp_path) == 0
    assert len(state["requests"]) == 2 * 2 * 3 * (1 + 2)   # rollouts + gradings


def test_baseline_can_be_pinned_in_the_config(ollama, tmp_path) -> None:
    url, _ = ollama
    cfg = _write(tmp_path, extra={"baseline": "large"})
    assert _run(cfg, url, tmp_path) == 0
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert report["meta"]["baseline"] == "large"
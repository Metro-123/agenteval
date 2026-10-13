"""The built dashboard must actually be served and render, not just exist.

These tests are the end-to-end check: build the bundle, mount it, fetch the page
and the API, and confirm the numbers the browser would receive are usable.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
DIST = WEB / "dist"

pytestmark = pytest.mark.skipif(shutil.which("npm.cmd") is None
                                and shutil.which("npm") is None,
                                reason="npm not available")


@pytest.fixture(scope="module")
def built() -> Path:
    if not DIST.is_dir():
        subprocess.run(["npm.cmd", "run", "build"], cwd=WEB, check=True,
                       capture_output=True, timeout=600)
    return DIST


def test_bundle_exists(built: Path) -> None:
    assert (built / "index.html").is_file()
    assets = list((built / "assets").glob("*.js"))
    assert assets, "no JS bundle emitted"
    assert any(a.stat().st_size > 10_000 for a in assets)


def test_bundle_contains_no_placeholder(built: Path) -> None:
    text = (built / "index.html").read_text(encoding="utf-8").lower()
    assert "<div id=\"root\">" in text or 'id="root"' in text


def test_server_serves_the_spa_and_the_api(built: Path) -> None:
    """Import after the build so WEB_DIST is detected at app construction."""
    import importlib

    import agenteval.api as api

    importlib.reload(api)
    c = TestClient(api.create_app())

    page = c.get("/")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]

    # client-side route falls back to index.html rather than 404
    deep = c.get("/leaderboard")
    assert deep.status_code == 200
    assert "<div id=\"root\">" in deep.text or 'id="root"' in deep.text

    data = c.get("/api/report", params={"n_tasks": 12, "n_seeds": 3,
                                        "n_agents": 3, "n_boot": 60})
    assert data.status_code == 200
    body = data.json()
    assert "NaN" not in data.text
    assert body["leaderboard"]


def test_frontend_types_match_the_report_shape() -> None:
    """Every field App.tsx reads must exist in a real report.

    TypeScript cannot check a runtime payload, so a renamed key would only surface
    as an empty chart in the browser. This asserts the specific fields the UI uses.
    """
    sys.path.insert(0, str(ROOT / "src"))
    from agenteval.report import build_report
    from agenteval.sim import AgentSpec, SuiteSpec, default_judges

    rep = build_report(SuiteSpec(n_tasks=8, n_seeds=2, seed=1,
                                 agents=(AgentSpec("a", skill=1.0),
                                         AgentSpec("b", skill=1.6)),
                                 judges=default_judges()), n_boot=40)

    row = rep["leaderboard"][0]
    for k in ("agent", "mean", "sd_across_tasks", "is_baseline",
              "ci_low", "ci_high", "se", "significant",
              "meta_cost_per_task_tokens", "meta_latency_s_per_task"):
        assert k in row, f"leaderboard row missing {k}"

    est = rep["comparisons"][0]["estimators"]
    for name in ("naive", "cluster", "crossed", "bootstrap"):
        for k in ("se", "p_value", "ci_low", "ci_high", "estimate"):
            assert k in est[name], f"estimator {name} missing {k}"

    card = next(iter(rep["judges"]["judges"].values()))
    for k in ("name", "sensitivity", "specificity", "cohen_kappa_vs_truth",
              "observed_positive_rate", "bias_corrected_rate", "planted"):
        assert k in card, f"judge card missing {k}"
    assert {"sensitivity", "specificity", "position_bias"} <= set(card["planted"])

    p = rep["planner"]
    for k in ("min_detectable_delta", "table", "recommended",
              "current_power_at_target", "unidentifiable", "reference"):
        assert k in p, f"planner missing {k}"
    for k in ("at_design", "at_2x2", "at_8x8", "at_32x32"):
        assert k in p["min_detectable_delta"]
    for k in ("n_tasks", "n_seeds", "n_judges", "n_cells", "cost", "se", "power"):
        assert k in p["table"][0]

    for k in ("generated", "elapsed_s", "baseline", "n_tasks", "n_seeds",
              "n_judges", "n_cells", "target_delta"):
        assert k in rep["meta"], f"meta missing {k}"

    # and it has to be strict JSON, since that is what the browser parses
    json.dumps(rep, allow_nan=False)

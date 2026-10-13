"""HTTP surface: the API must not hand the dashboard numbers it cannot back up."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from agenteval.api import create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app())


SMALL = {"n_tasks": 12, "n_seeds": 3, "n_agents": 3, "n_boot": 60}


def test_health(client: TestClient) -> None:
    assert client.get("/api/health").json() == {"ok": True}


def test_report_returns_every_section(client: TestClient) -> None:
    r = client.get("/api/report", params=SMALL)
    assert r.status_code == 200
    body = r.json()
    for section in ("meta", "warnings", "leaderboard", "comparisons", "judges",
                    "rank_stability", "replication_curve", "planner"):
        assert section in body, f"missing {section}"


def test_response_is_strict_json(client: TestClient) -> None:
    """`JSON.parse` rejects bare NaN, so the wire format must never contain it."""
    raw = client.get("/api/report", params=SMALL).text
    assert "NaN" not in raw and "Infinity" not in raw
    json.loads(raw)


def test_refuses_an_unidentifiable_design(client: TestClient) -> None:
    """One seed or one agent is a user error, not a report full of NaN."""
    assert client.get("/api/report", params={**SMALL, "n_seeds": 1}).status_code == 422
    assert client.get("/api/report", params={**SMALL, "n_agents": 1}).status_code == 422


def test_repeated_requests_are_cached(client: TestClient) -> None:
    a = client.get("/api/report", params=SMALL).json()
    b = client.get("/api/report", params=SMALL).json()
    assert a == b
    assert a["meta"]["generated"] == b["meta"]["generated"]


def test_different_parameters_produce_different_reports(client: TestClient) -> None:
    a = client.get("/api/report", params=SMALL).json()
    b = client.get("/api/report", params={**SMALL, "n_tasks": 16}).json()
    assert a["meta"]["n_tasks"] != b["meta"]["n_tasks"]
    assert a["leaderboard"] != b["leaderboard"]


def test_baseline_is_marked_on_the_leaderboard(client: TestClient) -> None:
    body = client.get("/api/report", params=SMALL).json()
    flags = [r["is_baseline"] for r in body["leaderboard"]]
    assert sum(flags) == 1
    assert body["meta"]["baseline"] in [r["agent"] for r in body["leaderboard"]]


def test_agents_and_judges_endpoints(client: TestClient) -> None:
    a = client.get("/api/agents").json()["agents"]
    j = client.get("/api/judges").json()["judges"]
    assert len(a) >= 2 and len(j) >= 2
    assert {"name", "skill"} <= set(a[0])
    assert {"name", "sensitivity", "specificity"} <= set(j[0])

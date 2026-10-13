"""Local HTTP API over :func:`agenteval.report.build_report`.

Deliberately dependency-light and read-only: a static bundle of JSON, a cached
report, and no database. ``/api/report`` is the expensive call, so it is cached
and computed on a worker thread -- a synchronous build would block the event loop
and stall the whole dashboard.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .report import build_report
from .sim import AgentSpec, JudgeSpec, SuiteSpec, default_agents, default_judges

__all__ = ["create_app", "app"]

WEB_DIST = Path(__file__).resolve().parents[2] / "web" / "dist"

_lock = threading.Lock()
_cache: dict[tuple, dict] = {}
_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="agenteval")


def _spec(n_tasks: int, n_seeds: int, seed: int, n_agents: int) -> SuiteSpec:
    """Build a spec from request parameters, always with 2+ judges.

    A single judge makes its variance unidentifiable, and the report would emit
    intervals that are too narrow with no way to tell. The API refuses to build
    that rather than quietly serving a misleading number.
    """
    if n_seeds < 2:
        raise HTTPException(400, "n_seeds must be >= 2: with one seed the seed "
                                  "variance is unidentifiable")
    if n_agents < 2:
        raise HTTPException(400, "n_agents must be >= 2")
    agents = default_agents()[:n_agents]
    return SuiteSpec(n_tasks=n_tasks, n_seeds=n_seeds, seed=seed,
                     agents=agents, judges=default_judges())


def _get_report(key: tuple, spec: SuiteSpec, n_boot: int) -> dict:
    with _lock:
        hit = _cache.get(key)
    if hit is not None:
        return hit
    report = _pool.submit(build_report, spec, None, 0.03, n_boot).result()
    with _lock:
        _cache[key] = report
    return report


def create_app() -> FastAPI:
    app = FastAPI(title="agenteval", version="0.1.0",
                  description="Local agent-evaluation reporting")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET"], allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True}

    @app.get("/api/report")
    def report(n_tasks: int = Query(48, ge=8, le=256),
               n_seeds: int = Query(3, ge=2, le=16),
               n_agents: int = Query(6, ge=2, le=6),
               seed: int = Query(11, ge=0),
               n_boot: int = Query(600, ge=50, le=5000)) -> JSONResponse:
        spec = _spec(n_tasks, n_seeds, seed, n_agents)
        key = (n_tasks, n_seeds, n_agents, seed, n_boot)
        try:
            return JSONResponse(_get_report(key, spec, n_boot))
        except HTTPException:
            raise
        except Exception as exc:  # surface the reason instead of a blank 500
            raise HTTPException(500, f"report failed: {exc}") from exc

    @app.get("/api/judges")
    def judges() -> dict:
        return {"judges": [j.to_dict() for j in default_judges()]}

    @app.get("/api/agents")
    def agents() -> dict:
        return {"agents": [a.to_dict() for a in default_agents()]}

    if WEB_DIST.is_dir():
        app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

        @app.get("/{path:path}")
        def spa(path: str) -> Any:
            candidate = WEB_DIST / path
            if path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(WEB_DIST / "index.html")
    else:
        @app.get("/")
        def missing_ui() -> JSONResponse:
            return JSONResponse(
                {"error": "web/dist not built",
                 "hint": "run scripts\\dev.ps1 or `npm --prefix web run build`"},
                status_code=503)

    return app


app = create_app()

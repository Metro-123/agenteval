"""Command line entry point: report generation, live runs, and a local dev server."""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path

from .live import (
    AgentConfig,
    JudgeConfig,
    LiveRunner,
    OllamaClient,
    OllamaUnavailable,
    list_ollama_models,
    load_tasks_jsonl,
    preflight,
)
from .report import build_report
from .sim import SuiteSpec, default_agents, default_judges

__all__ = ["main"]


def _print_table(report: dict) -> None:
    """Shared leaderboard output so `report` and `run` cannot drift apart."""
    meta = report["meta"]
    print(f"\n{meta['elapsed_s']}s  {meta['n_tasks']} tasks x {meta['n_seeds']} seeds x "
          f"{meta['n_judges']} judges  baseline={meta['baseline']}")
    print(f"{'agent':>16}  {'mean':>6}  {'95% CI':>18}  {'cost':>6}  sig")
    for row in report["leaderboard"]:
        cost = row.get("meta_cost_per_task_tokens")
        cost_s = f"{cost:>6}" if isinstance(cost, int) else f"{'-':>6}"
        print(f"{row['agent']:>16}  {row['mean']:>6.3f}  "
              f"[{row['ci_low']:>+6.3f}, {row['ci_high']:>+6.3f}]  {cost_s}  "
              f"{'yes' if row['significant'] else 'no'}")
    for w in report["warnings"]:
        print(f"\nWARNING: {w}")


def _report(args: argparse.Namespace) -> int:
    spec = SuiteSpec(n_tasks=args.tasks, n_seeds=args.seeds, seed=args.seed,
                     n_tools=args.tools, agents=default_agents(),
                     judges=default_judges())
    report = build_report(spec, baseline=args.baseline,
                          target_delta=args.target_delta, n_boot=args.boot)
    text = json.dumps(report, indent=2, allow_nan=False)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out} ({len(text):,} bytes)")
    if args.print:
        print(text)

    _print_table(report)
    return 0


def _load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"config not found: {path}")
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(cfg, dict):
        raise SystemExit(f"{path} must contain a JSON object")
    return cfg


def _models(args: argparse.Namespace) -> int:
    try:
        models = list_ollama_models(args.base_url)
    except OllamaUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not models:
        print("no models installed; try 'ollama pull qwen2.5:0.5b'")
        return 1
    print(f"{'model':<40} {'size':>12}")
    for m in models:
        size = m.get("size")
        size_s = f"{size / 1e9:.2f} GB" if isinstance(size, int) else "-"
        print(f"{m.get('name', '?'):<40} {size_s:>12}")
    return 0


def _run(args: argparse.Namespace) -> int:
    cfg = _load_config(args.config)

    tasks_path = cfg.get("tasks") or args.tasks
    if not tasks_path:
        raise SystemExit('no tasks: set "tasks" in the config or pass --tasks')
    try:
        tasks = load_tasks_jsonl(tasks_path)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    agents = [AgentConfig(**a) for a in cfg.get("agents") or []]
    judges = [JudgeConfig(**j) for j in cfg.get("judges") or []]
    if len(agents) < 2:
        raise SystemExit("need at least 2 agents to compare")
    if len(judges) < 2:
        raise SystemExit(
            "need at least 2 judges: with one judge the judge axis is "
            "unidentifiable and intervals cannot be corrected after the fact")

    n_seeds = int(cfg.get("n_seeds", args.seeds))
    client = OllamaClient(base_url=cfg.get("base_url", args.base_url),
                          num_ctx=int(cfg.get("num_ctx", args.num_ctx)))

    print(f"{len(tasks)} tasks x {n_seeds} seeds x {len(judges)} judges "
          f"x {len(agents)} agents")
    try:
        for m in list_ollama_models(client.base_url):
            print(f"  available: {m.get('name', '?')}")
    except OllamaUnavailable:
        pass

    try:
        warnings = preflight(client, tasks, agents, judges)
    except OllamaUnavailable as exc:
        # preflight talks to the daemon to list models, so an unreachable server
        # surfaces here first, before any run starts
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for w in warnings:
        print(f"\nWARNING: {w}")
    if args.strict and warnings:
        print("\nrefusing to start: fix the warnings above, or drop --strict")
        return 3

    cost = len(agents) * len(tasks) * n_seeds * (1 + len(judges))
    if not args.yes:
        try:
            reply = input(f"\nrun {cost} local model calls? [y/N] ")
        except EOFError:
            reply = ""
        if reply.strip().lower() not in ("y", "yes"):
            print("aborted")
            return 1

    try:
        runner = LiveRunner(client=client, tasks=tasks, agents=agents,
                            judges=judges, n_seeds=n_seeds,
                            on_step=(None if args.quiet else print))
        suite = runner.run()
    except OllamaUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130

    report = build_report(suite=suite, baseline=cfg.get("baseline", args.baseline),
                          target_delta=cfg.get("target_delta", args.target_delta),
                          n_boot=args.boot)
    text = json.dumps(report, indent=2, allow_nan=False)
    out = Path(args.out or cfg.get("out") or "live-report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"\nwrote {out} ({len(text):,} bytes)")

    episodes = Path(args.episodes_out or "live-episodes.json")
    episodes.write_text(json.dumps(
        [e.to_dict() for e in suite["episodes"]], indent=2, allow_nan=False),
        encoding="utf-8")
    print(f"wrote {episodes} ({len(suite['episodes'])} episodes, "
          f"{suite['n_calls']} model calls)")

    _print_table(report)
    return 0


def _serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is not installed; `pip install uvicorn fastapi`", file=sys.stderr)
        return 1
    if not args.no_browser:
        webbrowser.open(f"http://{args.host}:{args.port}/")
    uvicorn.run("agenteval.api:app", host=args.host, port=args.port,
                reload=args.reload, log_level="info")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="agenteval-demo",
                                description="Agent evaluation harness")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("report", help="build an evaluation report")
    r.add_argument("--tasks", type=int, default=48)
    r.add_argument("--seeds", type=int, default=3)
    r.add_argument("--tools", type=int, default=4)
    r.add_argument("--seed", type=int, default=11)
    r.add_argument("--baseline", type=str, default=None)
    r.add_argument("--target-delta", type=float, default=0.03)
    r.add_argument("--boot", type=int, default=1000)
    r.add_argument("--out", type=str, default=None, help="write JSON here")
    r.add_argument("--print", action="store_true", help="dump JSON to stdout")
    r.set_defaults(func=_report)

    m = sub.add_parser("models", help="list locally installed Ollama models")
    m.add_argument("--base-url", default="http://127.0.0.1:11434")
    m.set_defaults(func=_models)

    u = sub.add_parser("run", help="evaluate real local models via Ollama")
    u.add_argument("config", help="JSON config file (see README)")
    u.add_argument("--tasks", type=str, default=None,
                   help="override the tasks JSONL path from the config")
    u.add_argument("--seeds", type=int, default=3)
    u.add_argument("--base-url", default="http://127.0.0.1:11434")
    u.add_argument("--num-ctx", type=int, default=4096)
    u.add_argument("--baseline", type=str, default=None)
    u.add_argument("--target-delta", type=float, default=0.03)
    u.add_argument("--boot", type=int, default=1000)
    u.add_argument("--out", type=str, default=None)
    u.add_argument("--episodes-out", type=str, default=None)
    u.add_argument("--yes", action="store_true", help="skip the confirmation")
    u.add_argument("--strict", action="store_true",
                   help="refuse to run when preflight reports warnings")
    u.add_argument("--quiet", action="store_true")
    u.set_defaults(func=_run)

    s = sub.add_parser("serve", help="run the local API + dashboard")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(func=_serve)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

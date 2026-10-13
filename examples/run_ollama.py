"""Minimal example: run a live benchmark against a local Ollama daemon.

This does not install anything, does not make outbound network calls, and will
fail cleanly if Ollama is not running or models are missing.
"""

from agenteval.live import (
    AgentConfig,
    JudgeConfig,
    LiveRunner,
    OllamaClient,
    load_tasks_jsonl,
    preflight,
)
from agenteval.report import build_report

client = OllamaClient()  # http://127.0.0.1:11434
tasks = load_tasks_jsonl("examples/tasks.jsonl")
agents = (
    AgentConfig("small", "qwen2.5:0.5b", temperature=0.7),
    AgentConfig("large", "llama3.2:1b", temperature=0.7),
)
judges = (
    JudgeConfig("j1", "qwen2.5:0.5b"),
    JudgeConfig("j2", "llama3.5:0.5b"),
)

for warning in preflight(client, tasks, agents, judges):
    print("WARNING:", warning)

cost = len(agents) * len(tasks) * 2 * (1 + len(judges))
answer = input(f"Run {cost} local calls? [y/N] ").strip().lower()
if answer not in ("y", "yes"):
    raise SystemExit("aborted")

runner = LiveRunner(client=client, tasks=tasks, agents=agents, judges=judges,
                    n_seeds=2, on_step=print)
suite = runner.run()
report = build_report(suite=suite)
print("\n=== report summary ===")
for row in report["leaderboard"]:
    print(f"{row['agent']}: {row['mean']:.3f}  [{row['ci_low']:+.3f},{row['ci_high']:+.3f}] "
          f"(sig={row['significant']}) cost={row.get('meta_cost_per_task_tokens')}")
print("\nPareto:", report["pareto"])

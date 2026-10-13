# agenteval

Statistically rigorous, variance-aware evaluation for tool-using LLM agents.

Most agent leaderboards report a mean and a confidence interval computed as if
every judged cell were an independent draw. They are not. A benchmark is a
**crossed design** — tasks × seeds × judges — and each axis averages down at its
own rate. Getting that wrong is the difference between a result that replicates
and one that does not.

## The problem this solves

A typical run reports `n_cells` and computes `se = sd / sqrt(n_cells)`. But a
seed's offset is shared by every task and judge run on that seed, so seeds average
down at `1/n_seeds`, not `1/n_cells`. The correct variance is

```
Var(delta) = sigma2_task/n_tasks + sigma2_seed/n_seeds
           + sigma2_judge/n_judges + sigma2_eps/(n_tasks*n_seeds*n_judges)
```

On the bundled simulation the i.i.d. interval is **1.6–2.0× too narrow**, and task
clustering does not rescue it: resampling tasks is blind to the shared seed and
judge offsets. Absolute coverage of a nominal 95% interval falls to roughly 0.28.

Because components can only be estimated, not guessed, the harness fits them
(method-of-moments and REML) and reports what each contributes.

## What it does

- **Crossed-variance inference** — `crossed_effect` builds intervals from fitted
  components with Satterthwaite degrees of freedom, alongside the naive and
  task-clustered intervals it improves on.
- **Design planning** — `min_cost_design` allocates `n ∝ sqrt(sigma2/c)` per axis
  under a budget, and `min_detectable_delta` reports the floor imposed by seed and
  judge variance that no amount of task authoring can go below.
- **Judge reliability** — Dawid-Skene recovers each judge's operating point, and
  `bias_corrected_score` inverts it. Four judges with sensitivities from 0.67 to
  0.91 all correct to the same pooled rate.
- **Reproducible traces** — content-hashed records that replay deterministically.

## Quick start

```powershell
# from the repo root
.\scripts\dev.ps1        # builds the web bundle and serves localhost:8000
```

Then open <http://127.0.0.1:8000>.

Python only:

```powershell
$env:PYTHONPATH = "src"
python -m agenteval.cli report --out report.json    # JSON report
python -m agenteval.cli serve                       # API + dashboard
```

## Tests

```powershell
.\scripts\test.ps1
```

The suite is not just unit coverage. It includes:

- **Coverage checks** that a nominal 95% interval covers ~95% of the time, which
  is what caught the crossed-variance error in the first place.
- **A cell-ordering test** asserting `cell_index` labels match the values they
  index. Stacking judges and raveling leaves the *seed* axis fastest, silently
  permuting seed and judge labels; every variance component was attributed to the
  wrong level while still returning plausible numbers.
- **Strict-JSON assertions** (`allow_nan=False`) because a bare `NaN` makes
  `JSON.parse` throw and takes the whole dashboard down.
- **Frontend-shape checks** asserting every field the React app reads exists in a
  real report, since TypeScript cannot validate a runtime payload.
- **A real local HTTP server** standing in for Ollama, so the transport is
  exercised over an actual socket -- URL, payload, parsing, retry, error mapping --
  without model weights.

## Running against real models (Ollama)

The dashboard defaults to simulated data, which is what makes ground truth
available. To evaluate real local models, use `OllamaClient`. It speaks Ollama's
native `/api/chat` rather than the OpenAI-compatible shim, because `options.seed`
is a documented native field -- on the shim a seed can be dropped silently, leaving
every seed identical and `sigma2_seed` reading a confident, meaningless `0`.

```python
from agenteval.live import (
    AgentConfig, JudgeConfig, LiveRunner, OllamaClient,
    list_ollama_models, load_tasks_jsonl, preflight,
)
from agenteval.report import build_report

client = OllamaClient()                       # http://127.0.0.1:11434
tasks = load_tasks_jsonl("tasks.jsonl")
agents = (
    AgentConfig("small", "qwen2.5:0.5b", temperature=0.7),
    AgentConfig("large", "llama3.2:1b",   temperature=0.7),
)
judges = (JudgeConfig("j1", "qwen2.5:0.5b"),
          JudgeConfig("j2", "llama3.5:0.5b"))

for w in preflight(client, tasks, agents, judges):
    print("WARNING:", w)                     # fix these before spending compute

suite = LiveRunner(client=client, tasks=tasks, agents=agents, judges=judges,
                   n_seeds=3, on_step=print).run()
report = build_report(suite=suite)            # same analysis code as the dashboard
```

`tasks.jsonl` is one JSON object per line: `task_id` (optional), `prompt`,
`answer`.

Two things `preflight` exists to catch, because both waste a long run:

- **An unpulled model.** Reported as `ollama pull <name>`.
- **`temperature=0.0` on every agent.** All seeds then return identical text, so
  the seed axis measures nothing and `sigma2_seed` reads `0` -- a voided crossed
  design that still produces confident intervals. Use a small temperature.

Note that the default `Task.check` is a case-insensitive substring test against
`answer`. That is fine for short factual answers but will mis-score tasks whose
answer appears incidentally in a refusal or an unrelated sentence; override `check`
for anything non-trivial.

## Layout

```
src/agenteval/
  vcomp.py     variance components: MoM, REML, ICC, design matrices
  design.py    crossed SE, allocation, power, detectable-effect floor
  effects.py   crossed/clustered/bootstrap effects, CUPED, rank stability
  judge.py     Dawid-Skene, kappa, AC1, bias correction
  sim.py       ground-truth simulators and paired/field extraction
  traces.py    hash-verified replayable traces
  live.py      real-model runner (injectable transport, no third-party deps)
  report.py    assembles the full report as strict JSON
  api.py       FastAPI service (cached, read-only)
  cli.py       report / serve commands
web/           React + Vite + TypeScript dashboard
scripts/       dev.ps1, test.ps1
```

## Design decisions

**Two component estimators, not one.** MoM is closed-form and fast; REML is more
efficient but iterative. The report shows both so disagreement is visible rather
than hidden behind a single number.

**The API refuses unidentifiable designs.** A single seed or judge returns
HTTP 422 instead of quietly serving an interval that is too narrow with no way to
correct it after the fact.

**Live cost is `rollouts + gradings`, not the full crossed product.** Each agent
rollout happens once and every judge grades that same rollout, so a run costs
`n_agents * n_tasks * n_seeds * (1 + n_judges)` calls. Treating it as
`n_agents * n_tasks * n_seeds * n_judges` overstates agent cost by `n_judges`
times; counting rollouts alone understates the total. This number is what decides
whether a design is affordable, so `LiveRunner` reports both parts separately.

**Non-finite values become `null`.** Several fields are legitimately undefined
(CUPED's `rho` with no covariate, MoM's `loglik` because it has no likelihood).
They serialize as `null`, not a token no JSON parser accepts.

## Status

Statistical core, report, API, CLI, dashboard, and the real-model runner are
implemented and verified end-to-end (`114 passed`, `tsc` clean, `oxlint` 0 errors).

Known limitations:

- REML runs single-threaded; one large-design test takes ~12s.
- The simulator is synthetic by default (no live model calls) — deliberate, since
  it is what makes ground truth available. `live.py` provides a real-model runner
  (`LiveRunner`, `OllamaClient`) tested against a real local HTTP server, but there
  is no packaged task loader or CLI entry point for real runs yet.
- `Task.check` is a substring match; real benchmarks need a task-specific checker.
- Ollama itself is not installed here, so no run has been made against real
  weights. The transport is tested against a local stand-in that speaks the same
  HTTP protocol.
- `npm run lint` reports one `set-state-in-effect` warning in `App.tsx`. It is a
  false positive: the `setState` runs after an `await`, which the rule cannot see.
  Also note `oxlint`'s native binding does not install automatically on Windows via
  npm's optional-dependency handling; install
  `@oxlint/binding-win32-x64-msvc` explicitly if the lint script fails to start.

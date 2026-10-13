"""Run real models through the same harness the simulator feeds.

The statistics never change. A real run only has to return the same shapes
:func:`agenteval.sim.simulate_suite` produces -- ``agents``, ``judges``,
``cell_index``, ``judged`` and ``raw`` -- and every downstream module works
unchanged. That is the whole integration.

Three things a real run must get right, and all three are easy to get wrong:

* **Seeds and judges are separate nested axes.** Collapsing them into one
  "replicate" dimension makes their variance unidentifiable, and the correction
  cannot be recovered after the fact.
* **Every judge scores the *same* trajectory.** Otherwise you are comparing two
  different runs rather than two opinions of the same run.
* **``cell_index`` is built with the judge axis varying fastest.** The seed-first
  ordering silently permutes every variance component while still returning
  plausible numbers.

The model transport is deliberately injectable. :class:`ModelClient` is a
protocol, so tests (and offline development) pass a stub and never touch the
network.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import numpy as np

from .traces import build_episode
from .vcomp import CellIndex

__all__ = [
    "Task",
    "AgentConfig",
    "JudgeConfig",
    "ModelClient",
    "HttpModelClient",
    "OllamaClient",
    "OllamaUnavailable",
    "DEFAULT_OLLAMA_URL",
    "list_ollama_models",
    "load_tasks_jsonl",
    "preflight",
    "LiveRunner",
    "build_cell_index",
    "run_live",
]


@dataclass(frozen=True)
class Task:
    """One benchmark item. ``check`` decides pass/fail from the transcript."""
    task_id: str
    prompt: str
    answer: str
    tools: tuple[str, ...] = ()

    def check(self, transcript: str) -> bool:
        """Ground truth without a judge. Override for anything non-trivial."""
        return self.answer.strip().lower() in transcript.lower()


@dataclass(frozen=True)
class AgentConfig:
    name: str
    model: str
    system: str = "You are a careful assistant. Be concise."
    temperature: float = 0.0
    max_tokens: int = 512
    seed_offset: int = 0
    # token cost, used only by the design planner
    token_cost: float = 1.0
    latency: float = 1.0
    notes: str = ""


@dataclass(frozen=True)
class JudgeConfig:
    name: str
    model: str
    system: str = (
        "You grade an answer. Reply with exactly PASS or FAIL and nothing else."
    )
    temperature: float = 0.0
    max_tokens: int = 8
    seed_offset: int = 0


class ModelClient(Protocol):
    """Minimal transport. Inject a stub to develop or test without a network."""

    def complete(self, *, model: str, system: str, user: str,
                 temperature: float, max_tokens: int,
                 seed: int | None = None) -> tuple[str, int, int, float]:
        """Return ``(text, tokens_in, tokens_out, latency_ms)``."""
        ...


class HttpModelClient:
    """OpenAI-compatible chat completions over the standard library.

    No third-party dependency, because the statistical core deliberately has
    none. Retries on 429 and 5xx with exponential backoff, since a large crossed
    design will hit rate limits partway through and losing that work is worse
    than waiting.
    """

    def __init__(self, base_url: str = "https://api.openai.com/v1",
                 api_key: str | None = None, model_prefix: str = "",
                 timeout: float = 60.0, max_retries: int = 5) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model_prefix = model_prefix
        self.timeout = timeout
        self.max_retries = max_retries
        if not self.api_key:
            raise ValueError(
                "no API key: set OPENAI_API_KEY or pass api_key=...")

    def complete(self, *, model: str, system: str, user: str,
                 temperature: float, max_tokens: int,
                 seed: int | None = None) -> tuple[str, int, int, float]:
        payload: dict = {
            "model": f"{self.model_prefix}{model}",
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {self.api_key}"}

        delay = 2.0
        last: Exception | None = None
        for _ in range(self.max_retries):
            started = time.perf_counter()
            try:
                req = urllib.request.Request(
                    f"{self.base_url}/chat/completions", data=body,
                    headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                elapsed = (time.perf_counter() - started) * 1000.0
                usage = data.get("usage") or {}
                text = (data["choices"][0]["message"].get("content") or "")
                return (text, int(usage.get("prompt_tokens", 0)),
                        int(usage.get("completion_tokens", 0)), elapsed)
            except urllib.error.HTTPError as exc:
                last = exc
                # 4xx other than rate limiting will not fix themselves
                if exc.code not in (408, 409, 429, 500, 502, 503, 504):
                    raise
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                last = exc
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
        raise RuntimeError(f"model call failed after {self.max_retries} tries: {last}")


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"


class OllamaUnavailable(RuntimeError):
    """Ollama is not running, or a requested model is not pulled."""


def list_ollama_models(base_url: str = DEFAULT_OLLAMA_URL,
                       timeout: float = 5.0) -> list[dict]:
    """Names of locally installed models, newest field order preserved.

    Raises :class:`OllamaUnavailable` with install instructions when the daemon is
    not reachable, because "connection refused" is otherwise a very confusing way
    to discover Ollama is not running.
    """
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise OllamaUnavailable(
            f"cannot reach Ollama at {base_url} ({exc.reason}). Start it with "
            f"'ollama serve' if it is installed, or install it from "
            f"https://ollama.com/download") from exc
    except json.JSONDecodeError as exc:
        raise OllamaUnavailable(
            f"unexpected reply from {url}; is something else on that port?") from exc
    return list(data.get("models") or [])


class OllamaClient:
    """Local models through Ollama's native ``/api/chat``.

    Uses the native API rather than Ollama's OpenAI-compatible ``/v1`` shim for one
    reason: ``options.seed`` is a documented native field, so the seed axis is
    actually honoured. On the compatibility shim a seed can be silently dropped,
    which would leave every seed producing an identical rollout -- the replicate
    axis would then measure nothing while the harness reported a confident,
    completely spurious ``sigma2_seed = 0``.

    Ollama needs no API key and makes no outbound network calls, so a local run
    costs nothing and leaks nothing.
    """

    def __init__(self, base_url: str = DEFAULT_OLLAMA_URL, timeout: float = 300.0,
                 max_retries: int = 3, keep_alive: str = "10m",
                 num_ctx: int = 4096, retry_base_delay: float = 1.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        # keep the model resident: a crossed design makes many sequential calls
        # and reloading weights each time dominates the wall clock
        self.keep_alive = keep_alive
        self.num_ctx = num_ctx
        self.retry_base_delay = retry_base_delay

    def complete(self, *, model: str, system: str, user: str,
                 temperature: float, max_tokens: int,
                 seed: int | None = None) -> tuple[str, int, int, float]:
        options: dict = {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": self.num_ctx,
        }
        if seed is not None:
            options["seed"] = int(seed)
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": False,          # one JSON body, not a stream of ndjson
            "keep_alive": self.keep_alive,
            "options": options,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}

        delay = self.retry_base_delay
        last: Exception | None = None
        for _ in range(self.max_retries):
            started = time.perf_counter()
            try:
                req = urllib.request.Request(
                    f"{self.base_url}/api/chat", data=body,
                    headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                elapsed = (time.perf_counter() - started) * 1000.0
                text = (data.get("message", {}).get("content") or "")
                return (text, int(data.get("prompt_eval_count", 0)),
                        int(data.get("eval_count", 0)), elapsed)
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", "replace")[:200]
                except Exception:  # noqa: BLE001 - body may be absent
                    pass
                if exc.code == 404:
                    raise OllamaUnavailable(
                        f"model {model!r} is not installed. Pull it with "
                        f"'ollama pull {model}'") from exc
                if exc.code == 400 and "num_ctx" in detail:
                    raise OllamaUnavailable(
                        f"model {model!r} rejected num_ctx={self.num_ctx}; "
                        f"lower OllamaClient(num_ctx=...)") from exc
                if exc.code not in (408, 429, 500, 502, 503, 504):
                    raise RuntimeError(
                        f"ollama returned {exc.code}: {detail}") from exc
                last = exc
            except (urllib.error.URLError, TimeoutError,
                    json.JSONDecodeError) as exc:
                last = exc
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
        raise RuntimeError(
            f"ollama call to {model!r} failed after {self.max_retries} tries: {last}")


def load_tasks_jsonl(path: str) -> tuple[Task, ...]:
    """Read benchmark items from JSONL.

    One object per line with ``task_id`` (optional, defaults to the line number),
    ``prompt`` and ``answer``. Blank lines are skipped, and a bad line names its
    line number rather than failing anonymously.
    """
    tasks: list[Task] = []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                obj = json.loads(line)
                tasks.append(Task(
                    task_id=str(obj.get("task_id", f"t{len(tasks)}")),
                    prompt=str(obj["prompt"]),
                    answer=str(obj.get("answer", "")),
                    tools=tuple(obj.get("tools", ()) or ()),
                ))
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(
                    f"{path}:{lineno}: needs 'prompt' and valid JSON ({exc})"
                ) from exc
    if not tasks:
        raise ValueError(f"{path} contained no tasks")
    return tuple(tasks)


def preflight(client: OllamaClient, tasks: Sequence[Task],
              agents: Sequence[AgentConfig],
              judges: Sequence[JudgeConfig]) -> list[str]:
    """Cheap checks that catch expensive mistakes before a long run starts.

    Returns a list of human-readable warnings; empty means nothing to flag. Two
    traps are worth catching up front:

    * an agent model that is not pulled, and
    * a deterministic temperature, which collapses the seed axis to nothing and
      makes the whole crossed design pointless while still producing intervals.
    """
    warnings: list[str] = []
    installed = {m.get("name") for m in list_ollama_models(client.base_url)}

    for cfg in list(agents) + list(judges):
        if cfg.model not in installed:
            warnings.append(f"model {cfg.model!r} is not pulled; run "
                            f"'ollama pull {cfg.model}'")

    if not tasks:
        warnings.append("no tasks: the run would produce empty arrays")
    if len(judges) < 2:
        warnings.append("fewer than 2 judges: the judge axis is unidentifiable, "
                        "and the API would reject the design anyway")

    deterministic = [a.name for a in agents if a.temperature == 0.0]
    if agents and len(deterministic) == len(agents):
        warnings.append(
            "every agent has temperature=0.0, so all seeds return identical text: "
            "the seed axis will measure nothing and sigma2_seed will read 0. Use "
            "a small temperature (e.g. 0.7) to get real run-to-run variance")
    return warnings


def build_cell_index(n_tasks: int, n_seeds: int, n_judges: int) -> CellIndex:
    """Judge axis varies fastest -- the order the statistics assume.

    Getting this backwards is the single most damaging mistake available: the
    arrays stay the right length and every number stays plausible, but the seed
    and judge labels are permuted, so each variance component is attributed to
    the wrong level.
    """
    return CellIndex(
        task=np.repeat(np.arange(n_tasks), n_seeds * n_judges),
        seed=np.tile(np.repeat(np.arange(n_seeds), n_judges), n_tasks),
        judge=np.tile(np.arange(n_judges), n_tasks * n_seeds),
    )


@dataclass
class LiveRunner:
    """Executes agents and judges over a crossed design."""

    client: ModelClient
    tasks: Sequence[Task]
    agents: Sequence[AgentConfig]
    judges: Sequence[JudgeConfig]
    n_seeds: int = 2
    on_step: Callable[[str], None] | None = None
    keep_episodes: bool = True
    episodes: list = field(default_factory=list)

    def _log(self, msg: str) -> None:
        if self.on_step is not None:
            self.on_step(msg)

    def run_agent(self, agent: AgentConfig, task: Task, seed: int) -> str:
        """One agent rollout. ``seed`` is passed through for reproducibility."""
        text, tin, tout, ms = self.client.complete(
            model=agent.model, system=agent.system, user=task.prompt,
            temperature=agent.temperature, max_tokens=agent.max_tokens,
            seed=agent.seed_offset + seed)
        if self.keep_episodes:
            self.episodes.append(build_episode(
                task_id=task.task_id, agent=agent.name, seed=seed, judge="",
                events=[{"index": 0, "kind": "final", "payload": {"text": text},
                         "latency_ms": ms, "tokens_in": tin, "tokens_out": tout}],
                score=float(task.check(text)), success=task.check(text),
                model=agent.model))
        return text

    def run_judge(self, judge: JudgeConfig, agent: AgentConfig, task: Task,
                  seed: int, transcript: str) -> float:
        """Grade a transcript. All judges see the identical rollout."""
        verdict, tin, tout, ms = self.client.complete(
            model=judge.model, system=judge.system,
            user=(f"Question: {task.prompt}\n\n"
                  f"Reference answer: {task.answer}\n\n"
                  f"Proposed answer: {transcript}\n\n"
                  "PASS if the proposed answer is correct."),
            temperature=judge.temperature, max_tokens=judge.max_tokens,
            seed=judge.seed_offset + seed)
        if self.keep_episodes:
            self.episodes.append(build_episode(
                task_id=task.task_id, agent=agent.name, seed=seed, judge=judge.name,
                events=[{"index": 0, "kind": "grade", "payload": {"text": verdict},
                         "latency_ms": ms, "tokens_in": tin, "tokens_out": tout}],
                score=float("pass" in verdict.lower()), model=judge.model))
        return float("pass" in verdict.lower())

    def run(self) -> dict:
        """Execute the full crossed design and return report-ready structures."""
        nt, ns, nj, na = len(self.tasks), self.n_seeds, len(self.judges), len(self.agents)
        # Agent rollouts happen once and are graded by every judge, so the cost is
        # na*nt*ns rollouts plus na*nt*ns*nj gradings -- not na*nt*ns*nj in total.
        n_rollouts = na * nt * ns
        n_calls = n_rollouts * (1 + nj)
        self._log(f"{n_calls} model calls: {n_rollouts} agent rollouts + "
                  f"{n_rollouts * nj} judge gradings "
                  f"({nt} tasks x {ns} seeds x {na} agents x {nj} judges)")

        # cache rollouts: every judge must grade the *same* transcript
        rollouts: dict[tuple[str, str, int], str] = {}
        truth: dict[tuple[str, str, int], float] = {}
        for agent in self.agents:
            for ti, task in enumerate(self.tasks):
                for seed in range(ns):
                    text = self.run_agent(agent, task, seed)
                    rollouts[(agent.name, task.task_id, seed)] = text
                    truth[(agent.name, task.task_id, seed)] = float(task.check(text))
            self._log(f"  {agent.name}: {nt * ns} rollouts")

        judged: dict[str, np.ndarray] = {}
        labels: dict[str, np.ndarray] = {}
        for agent in self.agents:
            for ji, judge in enumerate(self.judges):
                arr = np.empty((nt, ns), dtype=float)
                for ti, task in enumerate(self.tasks):
                    for seed in range(ns):
                        text = rollouts[(agent.name, task.task_id, seed)]
                        arr[ti, seed] = self.run_judge(
                            judge, agent, task, seed, text)
                judged[f"{agent.name}|{judge.name}"] = arr
                labels[f"{agent.name}|{judge.name}"] = np.array(
                    [[truth[(agent.name, task.task_id, s)] for s in range(ns)]
                     for task in self.tasks], dtype=float)
            self._log(f"  {agent.name}: judged by {nj}")

        meta = {a.name: self._arm_meta(a) for a in self.agents}
        return {
            "agents": [a.name for a in self.agents],
            "judges": [j.name for j in self.judges],
            "cell_index": build_cell_index(nt, ns, nj),
            "judged": judged,
            "raw": labels,
            "truth_probs": {},
            "meta": meta,
            "episodes": list(self.episodes),
            "n_calls": n_calls,
        }

    def _arm_meta(self, arm: AgentConfig) -> dict:
        """Per-arm metadata, using the *measured* cost and latency.

        The simulator multiplies assumed constants (``900 * token_cost``) because
        it has nothing real to measure. A live run does, so it reports observed
        tokens and wall-clock instead. The key names must match the simulator's
        exactly: the report's Pareto block reads ``cost_per_task_tokens`` and
        ``latency_s_per_task``, and different names silently yield ``null`` for
        every arm rather than raising.

        Only agent rollouts count (episodes with ``judge == ""``). Judge spend is
        reported separately per agent below, because it is attributable to the
        arm being graded but is not the arm's own inference cost.
        """
        mine = [e for e in self.episodes if e.agent == arm.name and e.judge == ""]
        graded = [e for e in self.episodes if e.agent == arm.name and e.judge != ""]
        n = len(mine)

        if n:
            cost = float(np.mean([e.total_tokens for e in mine]))
            lat = float(np.mean([e.total_latency_ms for e in mine])) / 1000.0
        else:
            cost = float("nan")
            lat = float("nan")

        out = {
            "model": arm.model,
            "notes": arm.notes,
            "cost_per_task_tokens": None if math.isnan(cost) else int(round(cost)),
            "latency_s_per_task": None if math.isnan(lat) else round(lat, 4),
            # retained so a caller can still see the declared assumptions
            "declared_token_cost": arm.token_cost,
            "declared_latency": arm.latency,
        }
        if graded:
            out["judge_tokens_per_rollout"] = int(round(float(
                np.mean([e.total_tokens for e in graded]))))
            out["judge_latency_s_per_rollout"] = round(float(
                np.mean([e.total_latency_ms for e in graded])) / 1000.0, 4)
        return out


def run_live(runner: LiveRunner) -> dict:
    """Convenience wrapper so callers need not know about the class."""
    return runner.run()

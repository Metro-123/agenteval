"""Simulators with known ground truth.

Because the statistics are validated against parameters we chose, the whole
study runs on CPU in seconds with no API budget. Two levels:

* :func:`simulate_difference_field` emits data whose variance components are
  exactly the requested ones - used to prove the estimators are unbiased and
  correctly calibrated.
* :func:`simulate_suite` emits a realistic multi-agent benchmark (heterogeneous
  tasks, stochastic agents, noisy LLM judges) for the dashboard, where the
  components are emergent rather than imposed.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field, asdict
from typing import Sequence

import numpy as np


def _stable_hash(text: str) -> int:
    """Process-independent hash, unlike the builtin ``hash`` for ``str``."""
    return int.from_bytes(hashlib.blake2s(text.encode("utf-8"), digest_size=4).digest(),
                          "big")

from .vcomp import CellIndex, VarianceComponents

__all__ = [
    "FieldSpec",
    "AgentSpec",
    "JudgeSpec",
    "SuiteSpec",
    "simulate_difference_field",
    "simulate_suite",
    "sigmoid",
]


def sigmoid(x: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


@dataclass(frozen=True)
class FieldSpec:
    """Ground-truth crossed design for a paired agent difference."""

    n_tasks: int = 40
    n_seeds: int = 4
    n_judges: int = 3
    delta: float = 0.04
    sigma2_task: float = 0.040
    sigma2_seed: float = 0.004
    sigma2_judge: float = 0.006
    sigma2_eps: float = 0.010
    seed: int = 0

    @property
    def n_cells(self) -> int:
        return self.n_tasks * self.n_seeds * self.n_judges

    def truth(self) -> VarianceComponents:
        return VarianceComponents(
            sigma2_task=self.sigma2_task, sigma2_seed=self.sigma2_seed,
            sigma2_judge=self.sigma2_judge, sigma2_eps=self.sigma2_eps,
            method="truth", n_estimable=4,
        )


@dataclass(frozen=True)
class AgentSpec:
    name: str
    skill: float
    seed_noise: float = 0.55
    tool_noise: float = 0.35
    token_cost: float = 1.0
    latency: float = 1.0
    p_fail_hard: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class JudgeSpec:
    name: str
    sensitivity: float = 0.92
    specificity: float = 0.88
    position_bias: float = 0.04
    verbosity_bias: float = 0.02
    strictness: float = 0.0
    self_preference: float = 0.0
    temperature: float = 0.0
    seed: int = 7

    def to_dict(self) -> dict:
        return asdict(self)

    def defect_marginals(self) -> dict[str, float]:
        return {
            "sensitivity": self.sensitivity, "specificity": self.specificity,
            "position_bias": self.position_bias, "verbosity_bias": self.verbosity_bias,
            "kappa_defect": max(1.0 - self.sensitivity, 1.0 - self.specificity),
        }


@dataclass(frozen=True)
class SuiteSpec:
    n_tasks: int = 48
    n_seeds: int = 3
    task_difficulty_sd: float = 1.15
    n_tools: int = 4
    seed: int = 11
    agents: tuple[AgentSpec, ...] = field(default_factory=tuple)
    judges: tuple[JudgeSpec, ...] = field(default_factory=tuple)
    baseline: str = ""

    @property
    def n_cells(self) -> int:
        return self.n_tasks * self.n_seeds * max(len(self.judges), 1)

    def to_dict(self) -> dict:
        return {
            "n_tasks": self.n_tasks, "n_seeds": self.n_seeds,
            "task_difficulty_sd": self.task_difficulty_sd, "n_tools": self.n_tools,
            "seed": self.seed, "baseline": self.baseline,
            "agents": [a.to_dict() for a in self.agents],
            "judges": [j.to_dict() for j in self.judges],
        }


def simulate_difference_field(spec: FieldSpec) -> tuple[np.ndarray, CellIndex, VarianceComponents]:
    rng = np.random.default_rng(spec.seed)
    nt, ns, nj = spec.n_tasks, spec.n_seeds, spec.n_judges
    p = rng.normal(0.0, math.sqrt(spec.sigma2_task), size=nt)[:, None, None]
    q = rng.normal(0.0, math.sqrt(spec.sigma2_seed), size=ns)[None, :, None]
    r = rng.normal(0.0, math.sqrt(spec.sigma2_judge), size=nj)[None, None, :]
    e = rng.normal(0.0, math.sqrt(spec.sigma2_eps), size=(nt, ns, nj))
    d = spec.delta + p + q + r + e
    task = np.repeat(np.arange(nt), ns * nj)
    seed = np.tile(np.repeat(np.arange(ns), nj), nt)
    judge = np.tile(np.arange(nj), nt * ns)
    return d.ravel(), CellIndex(task, seed, judge), spec.truth()


def default_agents() -> tuple[AgentSpec, ...]:
    return (
        AgentSpec("ReAct-v3", skill=1.35, seed_noise=0.50, tool_noise=0.30,
                  token_cost=1.15, latency=1.05, notes="balanced baseline"),
        AgentSpec("ReAct-v4-plan", skill=1.85, seed_noise=0.62, tool_noise=0.42,
                  token_cost=1.95, latency=1.70, p_fail_hard=0.04,
                  notes="stronger but noisier and pricier"),
        AgentSpec("Reflex-CoT", skill=1.05, seed_noise=0.38, tool_noise=0.22,
                  token_cost=0.75, latency=0.70, notes="cheap, low ceiling"),
        AgentSpec("ToolGraph", skill=2.15, seed_noise=0.78, tool_noise=0.55,
                  token_cost=2.60, latency=2.20, p_fail_hard=0.09,
                  notes="best mean, worst variance"),
        AgentSpec("PlanOnly", skill=0.55, seed_noise=0.30, tool_noise=0.18,
                  token_cost=0.60, latency=0.55, notes="control"),
        AgentSpec("ReAct-v3-cot", skill=1.55, seed_noise=0.46, tool_noise=0.34,
                  token_cost=1.35, latency=1.15, notes="v3 + reasoning trace"),
    )


def default_judges() -> tuple[JudgeSpec, ...]:
    return (
        JudgeSpec("gpt-judge-A", sensitivity=0.93, specificity=0.90,
                  position_bias=0.02, verbosity_bias=0.01, seed=101),
        JudgeSpec("gpt-judge-B", sensitivity=0.84, specificity=0.79,
                  position_bias=0.09, verbosity_bias=0.05, seed=202),
        JudgeSpec("claude-judge", sensitivity=0.90, specificity=0.93,
                  position_bias=0.01, verbosity_bias=0.00, seed=303),
        JudgeSpec("rubric-2.5", sensitivity=0.74, specificity=0.71,
                  position_bias=0.15, verbosity_bias=0.07, seed=404),
    )


def simulate_suite(spec: SuiteSpec) -> dict:
    """Generate a full multi-agent benchmark run with latent ground truth."""
    rng = np.random.default_rng(spec.seed)
    nt, ns = spec.n_tasks, spec.n_seeds
    judges = spec.judges or (JudgeSpec("perfect", 1.0, 1.0, 0.0, 0.0, seed=1),)
    nj = len(judges)
    agents = spec.agents or default_agents()

    z = rng.normal(0.0, spec.task_difficulty_sd, size=nt)
    hard = np.abs(z) > np.quantile(np.abs(z), 0.80)

    tools = [f"search_{k}" for k in range(1, spec.n_tools + 1)]
    truth: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}

    for ag in agents:
        lin = ag.skill - spec.task_difficulty_sd * z
        p_success = sigmoid(lin)
        p_success = np.where(hard, p_success * (1.0 - ag.p_fail_hard), p_success)
        truth[ag.name] = p_success
        meta[ag.name] = {
            "true_mean": float(p_success.mean()),
            "expected_pass_at_1": float(p_success.mean()),
            "cost_per_task_tokens": int(900 * ag.token_cost),
            "latency_s_per_task": round(2.4 * ag.latency, 2),
            "n_tools": spec.n_tools,
        }

    runs: dict[str, np.ndarray] = {}
    for ag in agents:
        # Seed from a *stable* hash of the name. Python randomises str.__hash__
        # per process (PYTHONHASHSEED), so using the builtin would make the
        # simulator non-reproducible across runs while looking like it was seeded.
        arng = np.random.default_rng(spec.seed * 31 + _stable_hash(ag.name) % 9973)
        p = truth[ag.name]
        out = np.empty((nt, ns), dtype=float)
        for t in range(nt):
            pert = p[t] * math.exp(arng.normal(0.0, ag.seed_noise))
            out[t] = (arng.random(ns) < np.clip(pert, 0.0, 1.0)).astype(float)
        runs[ag.name] = out

    scores: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    for jd in judges:
        jrng = np.random.default_rng(jd.seed)
        for ag in agents:
            obs = np.empty((nt, ns), dtype=float)
            for t in range(nt):
                for s in range(ns):
                    y = runs[ag.name][t, s]
                    hit = y > 0.5
                    if hit:
                        p_lab = jd.sensitivity - jd.strictness
                    else:
                        p_lab = (1.0 - jd.specificity) + jd.strictness
                    p_lab = float(np.clip(p_lab + ag.p_fail_hard * 0.1 * jrng.normal(), 0.02, 0.98))
                    if jrng.random() < jd.position_bias:
                        p_lab = 1.0 - p_lab
                    if jrng.random() < jd.verbosity_bias:
                        p_lab = 1.0 - p_lab
                    obs[t, s] = float(jrng.random() < p_lab)
            scores[f"{ag.name}|{jd.name}"] = obs
        for ag in agents:
            labels[f"{ag.name}|{jd.name}"] = runs[ag.name]

    cell_idx = CellIndex(
        task=np.repeat(np.arange(nt), ns * nj),
        seed=np.tile(np.repeat(np.arange(ns), nj), nt),
        judge=np.tile(np.arange(nj), nt * ns),
    )
    return {
        "spec": spec,
        "agents": [a.name for a in agents],
        "judges": [j.name for j in judges],
        "cell_index": cell_idx,
        "judged": scores,
        "raw": labels,
        "truth_probs": truth,
        "meta": meta,
        "tools": tools,
        "difficulty": z,
        "hard": hard,
    }


def _cell_order(judged: dict, arm: str, jnames: list[str],
                base: str | None = None) -> np.ndarray:
    """Judged scores for one arm as a flat array in ``cell_index`` order.

    ``cell_index`` is built with the judge axis varying fastest, i.e. C-order of
    ``(n_tasks, n_seeds, n_judges)``. Stacking judges first and raveling would
    leave the seed axis varying fastest instead, which silently permutes the seed
    and judge labels of every cell -- enough to wreck a variance decomposition
    while still returning plausible-looking numbers.
    """
    a = np.stack([judged[f"{arm}|{j}"] for j in jnames])  # (nj, nt, ns)
    b = None if base is None else np.stack([judged[f"{base}|{j}"] for j in jnames])
    a = a.transpose(1, 2, 0)  # (nt, ns, nj)
    if b is not None:
        a = a - b.transpose(1, 2, 0)
    return a.ravel()


def difference_field(suite: dict, arm: str, base: str,
                     judge: str | None = None) -> tuple[np.ndarray, CellIndex, np.ndarray]:
    """Paired difference d = arm - base on the shared design."""
    idx: CellIndex = suite["cell_index"]
    jnames: list[str] = [judge] if judge is not None else list(suite["judges"])
    d = _cell_order(suite["judged"], arm, jnames, base=base)
    keep = d != 0.0
    return d[keep], CellIndex(idx.task[keep], idx.seed[keep], idx.judge[keep]), d


def field_with_sign(suite: dict, arm: str, base: str,
                    judge: str | None = None) -> tuple[np.ndarray, CellIndex, np.ndarray]:
    """Like :func:`difference_field` but keeps zero cells (signed scale)."""
    idx: CellIndex = suite["cell_index"]
    jnames: list[str] = [judge] if judge is not None else list(suite["judges"])
    d = _cell_order(suite["judged"], arm, jnames, base=base)
    return d, CellIndex(idx.task.copy(), idx.seed.copy(), idx.judge.copy()), d


def task_arm_matrix(suite: dict, arm: str, judge: str | None = None) -> np.ndarray:
    judged: dict[str, np.ndarray] = suite["judged"]
    jnames: list[str] = suite["judges"]
    use = jnames if judge is None else [judge]
    stacked = np.stack([judged[f"{arm}|{j}"] for j in use])
    nt, ns = stacked.shape[1], stacked.shape[2]
    return stacked.reshape(len(use) * ns, nt).T

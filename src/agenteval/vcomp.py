"""Variance decomposition for crossed random-effects designs.

The central object is the *paired difference field* ``d[i, t, j]`` for two
agents A and B measured on the same fully-crossed design of tasks (i),
prediction seeds (t) and judges (j):

    d[i, t, j] = delta + p_i + q_t + r_j + eps[i, t, j]

with independent zero-mean Gaussian components. ``delta`` is the estimand
(true mean effect of A over B) and the four variances are the nuisance
structure that determines how many samples a decision actually costs.

    sigma2_p : task-by-arm interaction   (pairing removes this)
    sigma2_q : seed main effect          (common random numbers remove this)
    sigma2_r : judge main effect
    sigma2_eps : cell-level stochasticity
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np

__all__ = [
    "CellIndex",
    "VarianceComponents",
    "mean_squares",
    "moment_estimator",
    "reml_estimator",
    "variance_of_mean",
    "component_standard_errors",
    "icc_oneway",
    "icc_anova_decomposition",
]

_LOG_FLOOR = 1e-12


@dataclass(frozen=True)
class CellIndex:
    """Coordinate arrays describing the observed cells of a crossed design."""

    task: np.ndarray
    seed: np.ndarray
    judge: np.ndarray

    def __post_init__(self) -> None:
        sizes = (self.task.size, self.seed.size, self.judge.size)
        if len(set(sizes)) != 1:
            raise ValueError(f"ragged design: cell counts {sizes}")
        for name, arr in (("task", self.task), ("seed", self.seed), ("judge", self.judge)):
            if arr.dtype.kind not in "iu":
                raise TypeError(f"{name} must be integer typed, got {arr.dtype}")
        object.__setattr__(self, "n", int(self.task.size))
        object.__setattr__(self, "sizes", (int(self.task.max()) + 1,
                                           int(self.seed.max()) + 1,
                                           int(self.judge.max()) + 1))

    @property
    def factor_levels(self) -> dict[str, int]:
        return {"task": self.sizes[0], "seed": self.sizes[1], "judge": self.sizes[2]}

    def estimable_factors(self) -> list[str]:
        return [f for f, n in zip(("task", "seed", "judge"), self.sizes) if n >= 2]

    def reindex(self) -> "CellIndex":
        _, task = np.unique(self.task, return_inverse=True)
        _, seed = np.unique(self.seed, return_inverse=True)
        _, judge = np.unique(self.judge, return_inverse=True)
        return CellIndex(task.astype(np.int64), seed.astype(np.int64), judge.astype(np.int64))

    def complement_layout(self, other: "CellIndex") -> "CellIndex":
        """Copy the seed/judge layout of ``self`` but adopt ``other``'s tasks."""
        return CellIndex(other.task.copy(), self.seed.copy(), self.judge.copy())


@dataclass(frozen=True)
class VarianceComponents:
    sigma2_task: float
    sigma2_seed: float
    sigma2_judge: float
    sigma2_eps: float
    method: str
    loglik: float = float("nan")
    n_estimable: int = 4
    truncated: tuple[str, ...] = ()

    @property
    def total(self) -> float:
        return self.sigma2_task + self.sigma2_seed + self.sigma2_judge + self.sigma2_eps

    def as_array(self) -> np.ndarray:
        return np.array([self.sigma2_task, self.sigma2_seed,
                         self.sigma2_judge, self.sigma2_eps])

    def shares(self) -> dict[str, float]:
        tot = self.total
        if tot <= 0:
            return {"task": 0.0, "seed": 0.0, "judge": 0.0, "eps": 0.0}
        return {"task": self.sigma2_task / tot, "seed": self.sigma2_seed / tot,
                "judge": self.sigma2_judge / tot, "eps": self.sigma2_eps / tot}

    def to_dict(self) -> dict:
        d = asdict(self)
        d["total"] = self.total
        d["shares"] = self.shares()
        d["truncated"] = list(self.truncated)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "VarianceComponents":
        return cls(
            sigma2_task=float(d["sigma2_task"]),
            sigma2_seed=float(d["sigma2_seed"]),
            sigma2_judge=float(d["sigma2_judge"]),
            sigma2_eps=float(d["sigma2_eps"]),
            method=str(d.get("method", "unknown")),
            loglik=float(d.get("loglik", float("nan"))),
            n_estimable=int(d.get("n_estimable", 4)),
            truncated=tuple(d.get("truncated", ())),
        )


_FACTOR_ORDER = ("task", "seed", "judge")


def _levels(idx: CellIndex, factors: Sequence[str]) -> list[int]:
    return [idx.sizes[_FACTOR_ORDER.index(f)] for f in factors]


@lru_cache(maxsize=32)
def _design_matrices_cached(n: int, factors: tuple[str, ...], ncols: tuple[int, ...],
                             task: bytes, seed: bytes, judge: bytes) -> np.ndarray:
    """Cached design matrix; the REML objective re-requests it every iteration.

    One column per *level* of each requested factor, not per cell. Returned
    read-only because the cache is shared between optimiser calls.
    """
    wanted = [(f, raw, ncols[i]) for i, (f, raw) in enumerate(
        (("task", task), ("seed", seed), ("judge", judge))) if f in factors]
    A = np.zeros((n, sum(k for _, _, k in wanted)), dtype=np.float64)
    rows = np.arange(n)
    col = 0
    for _f, raw, k in wanted:
        A[rows, col + np.frombuffer(raw, dtype=np.int64)] = 1.0
        col += k
    A.flags.writeable = False
    return A


def _design_matrices(idx: CellIndex, factors: Sequence[str]) -> np.ndarray:
    factors = tuple(factors)
    ncols = tuple(int(idx.sizes[_FACTOR_ORDER.index(f)]) for f in _FACTOR_ORDER)
    return _design_matrices_cached(
        idx.n, factors, ncols, idx.task.tobytes(), idx.seed.tobytes(),
        idx.judge.tobytes())


def _fixed_effects(idx: CellIndex, factors: Sequence[str]) -> np.ndarray:
    """The grand mean - the only fixed effect estimable here.

    Task, seed and judge main effects are already carried by the covariance
    ``V = s2e*I + A*S*A'``. Putting them into ``X`` as well makes each of them
    non-identifiable (only their sum is), which silently drives the likelihood
    to a degenerate optimum, so ``X`` must stay a single intercept column.
    """
    return np.ones((idx.n, 1))


def mean_squares(y: np.ndarray, idx: CellIndex) -> dict[str, float]:
    """Balanced-crossing ANOVA mean squares keyed by factor plus 'resid'/'total'."""
    y = np.asarray(y, dtype=np.float64)
    n1, n2, n3 = idx.sizes
    grand = y.mean()
    out: dict[str, float] = {"grand": float(grand)}
    a, b, c = n1, n2, n3

    task_means = np.bincount(idx.task, weights=y, minlength=a) / (b * c)
    seed_means = np.bincount(idx.seed, weights=y, minlength=b) / (a * c)
    judge_means = np.bincount(idx.judge, weights=y, minlength=c) / (a * b)

    ss = {
        "task": b * c * float(((task_means - grand) ** 2).sum()),
        "seed": a * c * float(((seed_means - grand) ** 2).sum()),
        "judge": a * b * float(((judge_means - grand) ** 2).sum()),
    }
    df = {"task": a - 1, "seed": b - 1, "judge": c - 1}
    ss_total = float(((y - grand) ** 2).sum())
    df_resid = y.size - (n1 + n2 + n3) + 2
    df["resid"] = int(df_resid) if df_resid > 0 else 1
    df["total"] = y.size - 1

    for k in _FACTOR_ORDER:
        if df[k] > 0 and idx.sizes[_FACTOR_ORDER.index(k)] > 1:
            out[f"MS_{k}"] = ss[k] / df[k]
        else:
            out[f"MS_{k}"] = float("nan")
    ss_resid = max(ss_total - sum(ss.values()), 0.0)
    out["SS_resid"] = ss_resid
    out["MS_resid"] = ss_resid / df["resid"]
    out["df_resid"] = float(df["resid"])
    out["SS_total"] = ss_total
    out["df"] = df
    out["ss"] = ss
    out["rep_per_cell"] = float(y.size) / float(a * b * c)
    return out


def moment_estimator(y: np.ndarray, idx: CellIndex) -> VarianceComponents:
    """Closed-form method-of-moments (ANOVA) variance components.

    Unbiased under the additive model for a balanced crossing:

        E[MS_k] = sigma2_eps + (cells per level of k) * sigma2_k
        E[MS_resid] = sigma2_eps

    so each component is ``(MS_k - MS_resid) / cells_per_level``. Negative
    point estimates are truncated at zero and listed in ``truncated``.
    """
    ms = mean_squares(y, idx)
    n1, n2, n3 = idx.sizes
    ms_resid = ms["MS_resid"]
    factors = idx.estimable_factors()
    cells_per_level = {"task": n2 * n3, "seed": n1 * n3, "judge": n1 * n2}

    raw: dict[str, float] = {f: 0.0 for f in _FACTOR_ORDER}
    truncated: list[str] = []
    for f in factors:
        val = (ms[f"MS_{f}"] - ms_resid) / cells_per_level[f]
        if val < 0:
            truncated.append(f)
        raw[f] = max(val, 0.0)
    not_estimable = [f for f in _FACTOR_ORDER if f not in factors]
    truncated.extend(not_estimable)

    return VarianceComponents(
        sigma2_task=raw["task"], sigma2_seed=raw["seed"],
        sigma2_judge=raw["judge"], sigma2_eps=max(ms_resid, 0.0),
        method="mom", n_estimable=len(factors), truncated=tuple(truncated),
    )


def _reml_objective(log_var: np.ndarray, y: np.ndarray, idx: CellIndex,
                    factors: tuple[str, ...]) -> float:
    """Negative 2x restricted log-likelihood.

    With ``V = s2e*I + A*S*A'`` (S diagonal, one entry per random-effect level)
    everything reduces to one ``m x m`` matrix, ``m = n_task + n_seed + n_judge``::

        Mat     = S^-1 + A'A / s2e
        V^-1    = I/s2e - (A Mat^-1 A')/s2e^2
        log|V|  = n log s2e + sum(log diag S) + log|Mat|

    The dense form covers balanced, unbalanced and missing-cell designs alike.
    """
    if not np.all(np.isfinite(log_var)):
        return 1e12
    levels = _levels(idx, factors)
    m = int(sum(levels))
    n = idx.n
    s2_eps = float(np.exp(log_var[-1]))
    col_var = np.concatenate([
        np.full(lv, float(np.exp(log_var[k]))) for k, lv in enumerate(levels)
    ])
    if s2_eps <= 0.0 or np.any(col_var <= 0.0) or not np.all(np.isfinite(col_var)):
        return 1e12

    A = _design_matrices(idx, factors)
    Mat = np.diag(1.0 / col_var) + (A.T @ A) / s2_eps
    try:
        sign_m, logdet_mat = np.linalg.slogdet(Mat)
        Minv = np.linalg.inv(Mat)
    except np.linalg.LinAlgError:
        return 1e12
    if sign_m <= 0 or not np.isfinite(logdet_mat):
        return 1e12

    logdet_v = n * math.log(s2_eps) + float(np.log(col_var).sum()) + float(logdet_mat)
    inv_s2 = 1.0 / s2_eps
    inv_s2_sq = inv_s2 * inv_s2

    At1 = A.sum(axis=0)
    one_v_one = n * inv_s2 - inv_s2_sq * float(At1 @ (Minv @ At1))
    if one_v_one <= 0 or not np.isfinite(one_v_one):
        return 1e12
    logdet_x = math.log(one_v_one)

    Aty = A.T @ y
    one_v_y = float(y.sum()) * inv_s2 - inv_s2_sq * float(At1 @ (Minv @ Aty))
    resid = y - one_v_y / one_v_one
    ar = A.T @ resid
    quad = inv_s2 * float(resid @ resid) - inv_s2_sq * float(ar @ (Minv @ ar))

    val = logdet_v + logdet_x + quad
    return float(val) if np.isfinite(val) else 1e12


def reml_estimator(y: np.ndarray, idx: CellIndex, *,
                   restarts: int = 2) -> VarianceComponents:
    """Non-negative REML variance components.

    Maximises the restricted likelihood over a box of non-negative variances.
    Unlike the closed-form MoM estimator this also handles unbalanced or
    missing-cell designs, and never returns a negative variance.
    """
    from scipy.optimize import minimize

    y = np.asarray(y, dtype=np.float64)
    idx = idx.reindex()
    factors = idx.estimable_factors()
    n1, n2, n3 = idx.sizes
    ms = mean_squares(y, idx)
    ms_resid = max(ms["MS_resid"], _LOG_FLOOR)
    cells_per_level = {"task": n2 * n3, "seed": n1 * n3, "judge": n1 * n2}

    if not factors:
        return VarianceComponents(
            sigma2_task=0.0, sigma2_seed=0.0, sigma2_judge=0.0, sigma2_eps=float(ms_resid),
            method="reml", n_estimable=0, truncated=_FACTOR_ORDER,
        )

    mom = {f: max((ms[f"MS_{f}"] - ms_resid) / cells_per_level[f], 0.0) for f in factors}
    scale = max(float(np.var(y)), _LOG_FLOOR)

    starts: list[list[float]] = [list(mom.values()) + [ms_resid]]
    for r in range(restarts):
        w = scale * (0.02 + 0.5 * r)
        starts.append([w] * len(factors) + [w * 0.5])

    lo, hi = math.log(_LOG_FLOOR), math.log(1e6 * max(scale, 1.0) + 1.0)
    bounds = [(lo, hi)] * (len(factors) + 1)
    best, best_val = None, np.inf
    for s in starts:
        x0 = np.clip(np.log(np.maximum(np.asarray(s, dtype=float), _LOG_FLOOR)), lo, hi)
        res = minimize(_reml_objective, x0, args=(y, idx, tuple(factors)),
                       method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-8})
        if res.fun < best_val and np.all(np.isfinite(res.x)):
            best_val, best = res.fun, res
    if best is None:
        return moment_estimator(y, idx)

    est = np.exp(best.x)
    out = {f: 0.0 for f in _FACTOR_ORDER}
    for f, v in zip(factors, est[:-1]):
        out[f] = float(v)
    return VarianceComponents(
        sigma2_task=out["task"], sigma2_seed=out["seed"], sigma2_judge=out["judge"],
        sigma2_eps=float(est[-1]), method="reml", loglik=float(-0.5 * best_val),
        n_estimable=len(factors),
    )


def variance_of_mean(vc: VarianceComponents, n_tasks: int,
                     n_seeds: int = 1, n_judges: int = 1) -> dict:
    """Variance of the grand mean, split by which component it comes from.

    Each factor averages over its own level count, so this is *not*
    ``vc.total / (n_tasks * n_seeds * n_judges)``:

        Var(delta_hat) = s2_task/n_task + s2_seed/n_seed
                         + s2_judge/n_judge + s2_eps/n_cells

    ``n_cells`` is kept in the result for display, not as the denominator.
    """
    nt, ns, nj = int(n_tasks), int(n_seeds), int(n_judges)
    n_cells = nt * ns * nj
    terms = {
        "task": vc.sigma2_task / nt,
        "seed": vc.sigma2_seed / ns,
        "judge": vc.sigma2_judge / nj,
        "eps": vc.sigma2_eps / n_cells,
    }
    total = float(sum(terms.values()))
    return {"terms": terms, "total": total, "se": math.sqrt(max(total, 0.0)),
            "n_cells": n_cells, "n_tasks": nt, "n_seeds": ns, "n_judges": nj}


def component_standard_errors(y: np.ndarray, idx: CellIndex,
                              vc: VarianceComponents) -> dict[str, float]:
    """Delta-method standard errors of the MoM variance component estimates.

    Balanced-crossing ANOVA mean squares are independent, so

        Var(sigma2_k) = [ 2 E[MS_k]^2 / df_k + 2 E[MS_e]^2 / df_e ] / c_k^2

    with ``c_k`` the number of cells per level of factor ``k``.
    """
    ms = mean_squares(y, idx)
    n1, n2, n3 = idx.sizes
    cells_per_level = {"task": n2 * n3, "seed": n1 * n3, "judge": n1 * n2}
    df_e = ms["df_resid"]
    e2 = max(vc.sigma2_eps, 0.0)
    se: dict[str, float] = {"eps": math.sqrt(2.0 * e2 * e2 / df_e)}
    for f in idx.estimable_factors():
        m_k = ms[f"MS_{f}"]
        c = cells_per_level[f]
        df_k = ms["df"][f]
        var = (2.0 * m_k * m_k / df_k + 2.0 * e2 * e2 / df_e) / (c * c)
        se[f] = math.sqrt(max(var, 0.0))
    for f in _FACTOR_ORDER:
        se.setdefault(f, float("nan"))
    return se


def icc_oneway(y: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    """One-way random-effects ICC: share of variance that is between-group.

    The agent-eval interpretation is "how much of the spread you see is
    genuinely different tasks rather than the agent flip-flopping".
    """
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    k = uniq.size
    n_per = np.array([(groups == g).sum() for g in uniq], dtype=float)
    n = y.size
    grand = y.mean()
    if k < 2 or n <= k or np.any(n_per < 2):
        return {"icc": float("nan"), "ms_between": float("nan"),
                "ms_within": float("nan"), "n_groups": float(k), "n_obs": float(n)}
    means = np.array([y[groups == g].mean() for g in uniq])
    ss_between = float((n_per * (means - grand) ** 2).sum())
    ss_within = float(((y - means[groups]) ** 2).sum())
    ms_b = ss_between / (k - 1)
    ms_w = ss_within / (n - k)
    k0 = (n - float((n_per ** 2).sum()) / n) / (k - 1)
    denom = ms_b + (k0 - 1) * ms_w
    icc = (ms_b - ms_w) / denom if denom > 0 else float("nan")
    return {"icc": float(icc), "ms_between": ms_b, "ms_within": ms_w,
            "icc_k1": float((ms_b - ms_w) / ms_b) if ms_b > 0 else float("nan"),
            "n_groups": float(k), "n_obs": float(n)}


def icc_anova_decomposition(y: np.ndarray, idx: CellIndex) -> dict[str, float]:
    """Split per-task score variance into between-task vs within-task (trial) parts.

    Uses the crossed layout: task means across the seed x judge cells give the
    between-task signal, the residual gives the within-task inconsistency.
    """
    y = np.asarray(y, dtype=float)
    n1, n2, n3 = idx.sizes
    per_task = n2 * n3
    if per_task < 2 or n1 < 2:
        return {"icc": float("nan"), "between": float("nan"), "within": float("nan")}
    task_means = np.bincount(idx.task, weights=y, minlength=n1) / per_task
    resid = y - task_means[idx.task]
    ms_b = per_task * float(((task_means - y.mean()) ** 2).sum()) / (n1 - 1)
    ms_w = float((resid ** 2).sum()) / (n1 * (per_task - 1))
    denom = ms_b + (per_task - 1) * ms_w
    icc = (ms_b - ms_w) / denom if denom > 0 else float("nan")
    return {"icc": float(icc), "between": float(ms_b), "within": float(ms_w),
            "reps_per_task": float(per_task)}


def normal_quantile(p: float) -> float:
    from scipy.stats import norm
    return float(norm.ppf(p))


def summarise_components(vc: VarianceComponents, y: np.ndarray | None = None,
                         idx: CellIndex | None = None) -> dict:
    out = vc.to_dict()
    if y is not None and idx is not None:
        out["se"] = component_standard_errors(y, idx, vc)
    return out


def iter_cells(idx: CellIndex) -> Iterable[tuple[int, int, int]]:
    return zip(idx.task.tolist(), idx.seed.tolist(), idx.judge.tolist())

"""Experimental design: how many tasks, seeds and judges to buy.

The structural fact that governs everything here is that a crossed design's
replicate axes do **not** average down at the same rate. With

    d[i, t, j] = delta + p_i + q_t + r_j + eps[i, t, j]

the mean has

    Var(delta_hat) = s2_task/n_task + s2_seed/n_seed + s2_judge/n_judge
                     + s2_eps/(n_task * n_seed * n_judge)

because the seed main effect ``q_t`` is shared by all ``n_task * n_judge`` cells
that use seed ``t``. So adding seeds only retires ``s2_seed``, adding judges only
retires ``s2_judge``, and the cell-level term is the only one that benefits from
the product. Treating the components as if they all shared the same denominator
- the intuitive ``sum(s2) / n_cells`` - overstates the precision of a design by a
large factor whenever ``s2_seed`` or ``s2_judge`` is not negligible, and is the
reason naive confidence intervals on agent evals undercover.

Allocation follows from the same expression. Ignoring the (usually small)
``s2_eps`` product term, minimising the variance subject to
``sum_k c_k n_k <= B`` gives ``1/n_k = lambda c_k`` hence

    n_k* proportional to sqrt(s2_k / c_k)

so replicates belong on whichever axis is both noisy and cheap, and the variance
components genuinely do drive the split. The product term is then added and the
solution refined numerically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Mapping, Sequence

import numpy as np

from .vcomp import VarianceComponents

__all__ = ["Design", "FACTORS", "DEFAULT_COSTS", "crossed_variance", "crossed_se",
           "crossed_df", "satterthwaite_df", "unidentifiable_factors",
           "required_cells", "required_se", "min_detectable_delta", "min_cost_design",
           "design_table"]

FACTORS = ("task", "seed", "judge")

DEFAULT_COSTS: dict[str, float] = {
    "task": 1.0,
    "seed": 0.02,
    "judge": 0.10,
}

_MIN_LEVELS = 2


def crossed_variance(vc: VarianceComponents, n_tasks: int, n_seeds: int = 1,
                     n_judges: int = 1) -> dict:
    """Decompose ``Var(delta_hat)`` by which variance component it comes from.

    Delegates to :func:`agenteval.vcomp.variance_of_mean`, which is the single
    canonical implementation. The grand mean picks up the seed main effect as
    ``(1/n_seed) * sum_t q_t``, so each factor averages over *its own* level
    count, and a factor with a single level contributes its full variance: one
    seed is a shared offset across every task and judge, which inflates the
    between-cell correlation rather than disappearing.
    """
    from .vcomp import variance_of_mean
    out = dict(variance_of_mean(vc, n_tasks, n_seeds, n_judges))
    if out["total"] > 0:
        out["dominant"] = max(out["terms"], key=out["terms"].get)
    else:
        out["dominant"] = None
    return out


def unidentifiable_factors(n_seeds: int = 1, n_judges: int = 1) -> list[str]:
    """Factors whose variance cannot be separated from the task variance.

    With a single seed there is one shared ``q`` offset indistinguishable from a
    task-level effect, so no estimator can tell "these tasks are hard" from "this
    run was globally hard"; likewise a single judge. The grand mean then carries
    the full ``s2_seed``/``s2_judge`` while a variance-components fit will have
    booked it under ``s2_task``, understating the standard error by up to
    ``sqrt(n_task)``. Two levels on each replicate axis is the minimum for a
    defensible interval, which is why :func:`min_cost_design` defaults to it.
    """
    out = []
    if int(n_seeds) < 2:
        out.append("seed")
    if int(n_judges) < 2:
        out.append("judge")
    return out


def crossed_se(vc: VarianceComponents, n_tasks: int, n_seeds: int = 1,
               n_judges: int = 1) -> float:
    return crossed_variance(vc, n_tasks, n_seeds, n_judges)["se"]


def crossed_df(n_tasks: int, n_seeds: int = 1, n_judges: int = 1) -> dict:
    """Satterthwaite degrees of freedom for the crossed mean.

    Each axis contributes an independent mean-square term with its own level
    count, so the effective df of the sum is

        df = (sum_k a_k s2_k)^2 / sum_k (a_k s2_k)^2 / n_k

    which collapses to the level count of whichever axis dominates. A design
    with many seeds and a handful of tasks therefore has very few effective
    degrees of freedom no matter how many cells it runs, and its critical value
    should reflect that.
    """
    nt, ns, nj = int(n_tasks), int(n_seeds), int(n_judges)
    n_cells = nt * ns * nj
    levels = {"task": nt, "seed": ns, "judge": nj, "eps": n_cells}
    return levels


def satterthwaite_df(vc: VarianceComponents, n_tasks: int, n_seeds: int = 1,
                     n_judges: int = 1) -> float:
    """Satterthwaite df for ``Var(delta_hat)`` given the component estimates."""
    parts = crossed_variance(vc, n_tasks, n_seeds, n_judges)["terms"]
    levels = crossed_df(n_tasks, n_seeds, n_judges)
    num = float(sum(parts.values())) ** 2
    den = sum(v * v / levels[k] for k, v in parts.items() if v > 0 and levels[k] > 1)
    return float(num / den) if den > 0 else float("inf")


@dataclass(frozen=True)
class Design:
    n_tasks: int
    n_seeds: int
    n_judges: int

    def __post_init__(self) -> None:
        for name, v in (("n_tasks", self.n_tasks), ("n_seeds", self.n_seeds),
                        ("n_judges", self.n_judges)):
            if v < 1:
                raise ValueError(f"{name} must be >= 1, got {v}")

    @property
    def n_cells(self) -> int:
        return self.n_tasks * self.n_seeds * self.n_judges

    @property
    def sizes(self) -> tuple[int, int, int]:
        return (self.n_tasks, self.n_seeds, self.n_judges)

    def cost(self, costs: Mapping[str, float] | None = None) -> float:
        c = costs or DEFAULT_COSTS
        return sum(getattr(self, f"n_{k}s") * c[k] for k in FACTORS)

    def standard_error(self, vc: VarianceComponents) -> float:
        return crossed_se(vc, self.n_tasks, self.n_seeds, self.n_judges)

    def variance_terms(self, vc: VarianceComponents) -> dict[str, float]:
        return crossed_variance(vc, self.n_tasks, self.n_seeds, self.n_judges)["terms"]

    def replace(self, **kw) -> "Design":
        d = {**asdict(self), **kw}
        return Design(**d)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["n_cells"] = self.n_cells
        return d

    # ---------------------------------------------------------------- budget

    def continuous_optimum(self, budget: float,
                           costs: Mapping[str, float] | None = None,
                           vc: VarianceComponents | None = None) -> "Design":
        """Cost-optimal split of a budget, continuous relaxation.

        Minimising ``s2_k / n_k`` subject to ``sum_k c_k n_k <= B`` gives
        ``1/n_k = lambda c_k`` and therefore ``n_k ~ sqrt(s2_k / c_k)``: the
        split is driven by the variance components, so a noisy *and* cheap axis
        such as extra seeds is where extra money pays. The ``s2_eps`` product
        term is then restored by a short numerical refinement.
        """
        c = dict(costs or DEFAULT_COSTS)
        if vc is None:
            return Design(*[max(int(round(budget / (len(FACTORS) * c[k]))), 1)
                            for k in FACTORS])

        s2 = {k: getattr(vc, f"sigma2_{k}") for k in FACTORS}
        active = [k for k in FACTORS if s2[k] > 0]
        if not active:
            return Design(1, 1, 1)

        root = {k: math.sqrt(s2[k] / c[k]) for k in FACTORS}
        denom = sum(root[k] for k in active)
        n = {k: (budget * root[k] / denom if k in active else 1.0) for k in FACTORS}
        n = {k: max(n[k], 1.0) for k in FACTORS}

        try:
            from scipy.optimize import minimize

            def obj(x):
                return crossed_variance(vc, *np.exp(x))["total"]

            def spend(x):
                nt, ns, nj = np.exp(x)
                return nt * c["task"] + ns * c["seed"] + nj * c["judge"]

            r = minimize(obj, np.log([n[k] for k in FACTORS]), method="SLSQP",
                         bounds=[(math.log(1.0), math.log(1e7))] * 3,
                         constraints=[{"type": "ineq", "fun": lambda x: budget - spend(x)}])
            if r.success and np.all(np.isfinite(r.x)):
                n = dict(zip(FACTORS, (float(v) for v in np.exp(r.x))))
        except Exception:
            pass

        return Design(max(int(round(n["task"])), 1), max(int(round(n["seed"])), 1),
                      max(int(round(n["judge"])), 1))

    def integer_optimum(self, budget: float,
                        costs: Mapping[str, float] | None = None,
                        vc: VarianceComponents | None = None,
                        min_levels: int = _MIN_LEVELS) -> "Design":
        """Cost-optimal integer split, improved greedily on true variance."""
        c = dict(costs or DEFAULT_COSTS)
        sizes = [max(min_levels, 1) for _ in FACTORS]
        spend = sum(s * c[k] for s, k in zip(sizes, FACTORS))
        if spend > budget:
            sizes = [1] * 3
            spend = sum(s * c[k] for s, k in zip(sizes, FACTORS))
        if spend > budget:
            return Design(1, 1, 1)

        def variance(sz):
            if vc is None:
                return -sum(math.log(max(s, 1)) for s in sz)
            return crossed_variance(vc, *sz)["total"]

        order = sorted(range(3), key=lambda i: c[FACTORS[i]])
        cur = variance(sizes)
        improved = True
        while improved:
            improved = False
            best_gain, best_i = 1e-15, -1
            for i in order:
                if spend + c[FACTORS[i]] > budget:
                    continue
                trial = list(sizes)
                trial[i] += 1
                gain = cur - variance(trial)
                if gain > best_gain:
                    best_gain, best_i = gain, i
            if best_i >= 0:
                sizes[best_i] += 1
                spend += c[FACTORS[best_i]]
                cur -= best_gain
                improved = True
        return Design(*sizes)

    def equal_budget_alternative(self, budget: float,
                                 costs: Mapping[str, float] | None = None,
                                 min_levels: int = _MIN_LEVELS) -> "Design":
        """A sane human-chosen design: pin the cheap replicate axes, then split."""
        c = costs or DEFAULT_COSTS
        n_tasks = max(min_levels, 2)
        per_task_budget = budget / n_tasks
        inner_seed = max(min_levels, int(round(per_task_budget * 0.6 / max(c["seed"], 1e-9))))
        inner_judge = max(min_levels, int(round(per_task_budget * 0.4 / max(c["judge"], 1e-9))))
        while n_tasks * (inner_seed * c["seed"] + inner_judge * c["judge"]) > budget \
                and n_tasks > 2:
            n_tasks -= 1
        return Design(n_tasks, max(inner_seed, 2), max(inner_judge, 2))

    # -------------------------------------------------- component precision

    def component_relative_se(self, vc: VarianceComponents) -> dict[str, float]:
        """Predicted relative SE of each variance component estimate.

        From the balanced-crossing ANOVA, ``MS_k`` and ``MS_e`` are independent
        mean squares, so with ``c_k`` cells per level of factor ``k``

            Var(s2_k) = [ 2 E[MS_k]^2 / (n_k - 1) + 2 s2e^2 / df_e ] / c_k^2

        This is the part of the design the variance components *do* control.
        """
        n1, n2, n3 = self.sizes
        n = self.n_cells
        df_e = max(n - (n1 + n2 + n3) + 2, 1)
        per_level = {"task": n2 * n3, "seed": n1 * n3, "judge": n1 * n2}
        levels = {"task": n1, "seed": n2, "judge": n3}
        s2e = max(vc.sigma2_eps, 0.0)
        out: dict[str, float] = {}
        for k in FACTORS:
            s2k = getattr(vc, f"sigma2_{k}")
            if levels[k] < 2 or s2k <= 0:
                out[k] = float("inf") if s2k > 0 else float("nan")
                continue
            m_k = s2e + per_level[k] * s2k
            var = (2.0 * m_k * m_k / (levels[k] - 1) + 2.0 * s2e * s2e / df_e)
            se = math.sqrt(var) / per_level[k]
            out[k] = se / s2k
        return out

    def min_levels_for_precision(self, vc: VarianceComponents, factor: str,
                                 target_relative_se: float,
                                 others: Mapping[str, int] | None = None) -> int:
        """Smallest level count for ``factor`` hitting a target relative SE."""
        if factor not in FACTORS:
            raise ValueError(f"unknown factor {factor!r}")
        lo, hi = 2, 4096
        while lo < hi:
            mid = (lo + hi) // 2
            sizes = dict(zip(FACTORS, (mid, 2, 2)))
            if others:
                sizes.update({k: int(v) for k, v in others.items() if k != factor})
            cand = Design(**{f"n_{k}s": v for k, v in sizes.items()})
            rse = cand.component_relative_se(vc).get(factor, float("inf"))
            if np.isfinite(rse) and rse <= target_relative_se:
                hi = mid
            else:
                lo = mid + 1
        return int(lo)

    def power(self, delta: float, vc: VarianceComponents, alpha: float = 0.05) -> float:
        from scipy.stats import norm
        se = self.standard_error(vc)
        if se <= 0 or delta == 0:
            return float(alpha)
        z_a = norm.ppf(1.0 - alpha / 2.0)
        ncp = abs(delta) / se
        return float(norm.cdf(ncp - z_a) + norm.cdf(-ncp - z_a))

    def detectability(self, delta: float, vc: VarianceComponents,
                      alpha: float = 0.05) -> float:
        """Half-width of the 95% CI on the effect, in score points."""
        from scipy.stats import norm
        return float(norm.ppf(1.0 - alpha / 2.0) * self.standard_error(vc))


def required_cells(delta: float, sigma2_total: float, alpha: float = 0.05,
                   target_power: float = 0.8) -> float:
    """Cell count for a two-sided test, normal approximation.

    This is the textbook ``n ~ (z_a + z_b)^2 sigma^2 / delta^2`` and is only
    right when every variance component shares the same denominator, i.e. a
    single-level or fully-replicated design. For a crossed design use
    :func:`min_cost_design`, which respects that seeds and judges average down
    at their own level counts.
    """
    from scipy.stats import norm
    if delta == 0:
        return float("inf")
    z_a = norm.ppf(1.0 - alpha / 2.0)
    z_b = norm.ppf(target_power)
    return float(sigma2_total * (z_a + z_b) ** 2 / delta ** 2)


def required_se(delta: float, vc: VarianceComponents, alpha: float = 0.05,
                target_power: float = 0.8) -> float:
    """SE the design must achieve for a two-sided test at target power."""
    from scipy.stats import norm
    if delta == 0:
        return 0.0
    z_a = norm.ppf(1.0 - alpha / 2.0)
    z_b = norm.ppf(target_power)
    return float(abs(delta) / (z_a + z_b))


def min_cost_design(delta: float, vc: VarianceComponents,
                    costs: Mapping[str, float] | None = None,
                    alpha: float = 0.05, target_power: float = 0.8,
                    max_tasks: int = 4096, min_levels: int = 2) -> Design:
    """Cheapest design reaching ``target_power`` for ``delta``.

    Scans a bounded geometric grid on all three axes, keeps the cheapest feasible
    point, then walks each axis down while power holds. ``min_levels`` defaults
    to 2 because a single seed or judge is unidentifiable and makes any interval
    too narrow; see :func:`unidentifiable_factors`.

    Raises ``ValueError`` when no design reaches the target, quoting the
    :func:`min_detectable_delta` floor: the seed and judge main effects put a hard
    limit on the smallest effect any amount of task authoring can resolve.
    """
    c = dict(costs or DEFAULT_COSTS)
    lo_lv = max(int(min_levels), 1)

    def axis(hi: int) -> list[int]:
        vals = {lo_lv, max(hi, lo_lv)}
        v = lo_lv
        while v <= hi:
            vals.add(v)
            for step in (2, 3, 4, 5, 6, 8):
                vals.add(v * step)
            v *= 2
        return sorted(v for v in vals if lo_lv <= v <= hi)

    tasks_grid, seed_grid, judge_grid = axis(max_tasks), axis(512), axis(512)

    # evaluate the whole grid at once; a scalar loop over a few thousand designs
    # spends all its time in scipy's scalar norm calls
    from scipy.stats import norm as _norm
    nt_g = np.array(tasks_grid, dtype=float)[:, None, None]
    ns_g = np.array(seed_grid, dtype=float)[None, :, None]
    nj_g = np.array(judge_grid, dtype=float)[None, None, :]
    se = np.sqrt(vc.sigma2_task / nt_g + vc.sigma2_seed / ns_g
                 + vc.sigma2_judge / nj_g + vc.sigma2_eps / (nt_g * ns_g * nj_g))
    z_a = float(_norm.ppf(1.0 - alpha / 2.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        ncp = abs(delta) / se
        power = _norm.cdf(ncp - z_a) + _norm.cdf(-ncp - z_a)
    cost = nt_g * c["task"] + ns_g * c["seed"] + nj_g * c["judge"]
    ranked = np.where((power >= target_power) & (se > 0), cost, np.inf)
    if not np.isfinite(ranked).any():
        floor = min_detectable_delta(vc, seed_grid[-1], judge_grid[-1],
                                     alpha=alpha, target_power=target_power)
        raise ValueError(
            f"no design reaches power {target_power} for delta={delta} within "
            f"{max_tasks} tasks and {seed_grid[-1]} seeds/{judge_grid[-1]} judges: the "
            f"seed and judge main effects floor the detectable effect at {floor:.4g}. "
            "Raise the replicate cap, or accept a larger effect.")

    i, j, k = np.unravel_index(int(np.argmin(ranked)), ranked.shape)
    best = Design(tasks_grid[i], seed_grid[j], judge_grid[k])

    # walk each axis down while power holds, to get off the coarse grid
    for _ in range(4):
        moved = False
        for axis_name in FACTORS:
            cur = getattr(best, f"n_{axis_name}s")
            for dec in (max(cur - 1, lo_lv), max(cur - cur // 4, lo_lv)):
                if dec >= cur or dec < lo_lv:
                    continue
                sizes = list(best.sizes)
                sizes[FACTORS.index(axis_name)] = dec
                cand = Design(*sizes)
                if cand.power(delta, vc, alpha) >= target_power and \
                        cand.cost(c) < best.cost(c):
                    best = cand
                    moved = True
                    break
        if not moved:
            break
    return best


def min_detectable_delta(vc: VarianceComponents, n_seeds: int = 2,
                         n_judges: int = 2, alpha: float = 0.05,
                         target_power: float = 0.8) -> float:
    """Smallest effect any number of tasks could resolve, given the replicate axes.

    As ``n_task -> inf`` the task term vanishes and

        se -> sqrt(s2_seed/n_seed + s2_judge/n_judge)

    so the detectable effect bottoms out at ``(z_a + z_b)`` times that. No amount
    of extra task authoring can go below it; only more seeds and judges can. This
    is why adding tasks to a large benchmark keeps failing to find the small
    effect you are looking for, and why the right next purchase is usually
    replicates rather than more tasks.
    """
    from scipy.stats import norm
    se_floor = math.sqrt(max(vc.sigma2_seed / max(n_seeds, 1)
                             + vc.sigma2_judge / max(n_judges, 1), 0.0))
    z_a = norm.ppf(1.0 - alpha / 2.0)
    z_b = norm.ppf(target_power)
    return float((z_a + z_b) * se_floor)


def design_table(vc: VarianceComponents, n_tasks: Sequence[int],
                 n_seeds: Sequence[int], n_judges: Sequence[int],
                 delta: float = 0.04, alpha: float = 0.05) -> list[dict]:
    """Grid of designs with their width, power and cost - the planner table."""
    out: list[dict] = []
    for nt in n_tasks:
        for ns in n_seeds:
            for nj in n_judges:
                if min(nt, ns, nj) < 1:
                    continue
                d = Design(nt, ns, nj)
                out.append({
                    **d.as_dict(),
                    "cost": round(d.cost(), 3),
                    "se": round(d.standard_error(vc), 6),
                    "ci_half_width": round(d.detectability(delta, vc, alpha), 6),
                    "power": round(d.power(delta, vc, alpha), 4),
                    "component_rel_se": {
                        k: (round(v, 4) if np.isfinite(v) else None)
                        for k, v in d.component_relative_se(vc).items()
                    },
                })
    return out

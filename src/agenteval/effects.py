"""Effect estimation: paired vs unpaired, variance reduction, rank stability.

Three ways to spend the same number of agent rollouts get very different
confidence intervals:

* **unpaired** - A and B evaluated on disjoint task sets. The task-difficulty
  variance does not cancel, so it is paid for twice.
* **paired** - the same task instances for both arms (free, and the default).
  Task main effects cancel in the difference; seed and judge noise remain.
* **CUPED** - paired, plus a covariate from the baseline arm. Any correlation
  between the baseline's per-cell score and the difference is converted into a
  variance reduction, with the generalised variance reduction factor

      VRF = 1 - rho^2 (2 sd(d) sd(x) - sd(x)^2) / sd(d)^2

  which reduces to the familiar ``1 - rho^2`` when the covariate has the same
  variance as the metric it corrects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import norm, t

__all__ = [
    "Effect",
    "paired_effect",
    "unpaired_effect",
    "cuped_adjust",
    "bootstrap_paired_ci",
    "comparison_report",
    "rank_stability",
    "replication_curve",
    "paired_variance",
    "unpaired_variance",
    "pairing_payoff",
]


def paired_variance(vc, n_tasks: int) -> float:
    """Variance of the mean paired difference with ``n_tasks`` task instances.

    Pairing cancels the task main effect that both arms share, so only the
    components of the *difference* field remain.
    """
    return float(vc.total) / float(n_tasks)


def unpaired_variance(shared_task: float, sigma2_difference: float, n_tasks: int) -> float:
    """Variance of a difference estimated on disjoint task sets.

    With independent task draws the shared task-difficulty variance is paid
    twice instead of cancelling. Note ``shared_task`` is *not* identifiable
    from a paired difference field - it has to come from the two arms' own
    per-task variance - which is why this cannot be inferred from
    :class:`~agenteval.vcomp.VarianceComponents` alone.
    """
    arm_var = float(shared_task) + float(sigma2_difference) / 2.0
    return 2.0 * arm_var / float(n_tasks)


def pairing_payoff(vc, n_tasks: int, shared_task: float) -> dict:
    """Quantify what pairing buys, given the shared task-difficulty variance."""
    v_paired = paired_variance(vc, n_tasks)
    v_unpaired = unpaired_variance(shared_task, vc.total, n_tasks)
    return {
        "n_tasks": int(n_tasks),
        "shared_task_variance": float(shared_task),
        "paired_variance": v_paired,
        "unpaired_variance": v_unpaired,
        "paired_se": math.sqrt(v_paired),
        "unpaired_se": math.sqrt(v_unpaired),
        "variance_reduction": float(1.0 - v_paired / v_unpaired) if v_unpaired > 0 else 0.0,
        "se_ratio": math.sqrt(v_paired / v_unpaired) if v_unpaired > 0 else float("nan"),
    }


@dataclass(frozen=True)
class Effect:
    estimate: float
    se: float
    ci_low: float
    ci_high: float
    n: int
    method: str
    p_value: float = float("nan")
    variance_reduction: float = 1.0
    rho: float = float("nan")
    extra: dict = None

    def __post_init__(self) -> None:
        if self.extra is None:
            object.__setattr__(self, "extra", {})

    @property
    def significant(self) -> bool:
        return bool(self.ci_low > 0.0 or self.ci_high < 0.0)

    @property
    def ci_width(self) -> float:
        return self.ci_high - self.ci_low

    def to_dict(self) -> dict:
        d = asdict(self)
        d["significant"] = self.significant
        d["ci_width"] = self.ci_width
        return d


def paired_effect(d: np.ndarray, alpha: float = 0.05, groups: np.ndarray | None = None,
                  method: str | None = None) -> Effect:
    """Mean paired difference, with a cluster-robust interval when groups are given.

    An agent eval is *clustered*: ``n_tasks * n_seeds * n_judges`` cells are not
    ``n_tasks * n_seeds * n_judges`` independent observations. Seeds and judges
    re-measure the same task, so the naive iid standard error treats repeated
    measurements of the same task as fresh evidence and can be off by a factor
    of two or more. With task-level task-by-arm interaction variance in play this
    is not a technicality - it is the difference between a 95% interval that
    covers 95% of the time and one that covers 40%.

    Passing ``groups`` (typically the task id per cell) switches to the CR1
    cluster-robust sandwich

        Var(d_bar) = (1/N^2) * sum_g S_g^2 * G/(G-1),   S_g = sum_{i in g}(d_i - d_bar)

    with a t critical value on ``G - 1`` degrees of freedom. Omit ``groups``
    only when the cells really are independent.
    """
    from scipy import stats
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 2:
        return Effect(estimate=float("nan"), se=float("nan"), ci_low=float("nan"),
                      ci_high=float("nan"), n=n, method=method or "paired")

    mean = float(d.mean())
    if groups is None:
        se = float(d.std(ddof=1) / math.sqrt(n))
        df = n - 1
        label = method or "paired"
    else:
        g = np.asarray(groups)
        ok = g.shape[0] == d.shape[0]
        if not ok:
            raise ValueError("groups must have one entry per cell")
        uniq, inv = np.unique(g, return_inverse=True)
        G = uniq.size
        if G < 2:
            se = float(d.std(ddof=1) / math.sqrt(n))
            df = n - 1
            label = method or "paired-cluster(degenerate)"
        else:
            resid = d - mean
            s_g = np.bincount(inv, weights=resid, minlength=G)
            var = float((s_g ** 2).sum()) / (n * n) * (G / (G - 1.0))
            se = math.sqrt(max(var, 0.0))
            df = G - 1
            label = method or "paired-cluster"
            n = n

    tcrit = stats.t.ppf(1.0 - alpha / 2.0, df)
    if se > 0:
        p = float(2.0 * stats.t.sf(abs(mean / se), df))
    elif mean == 0.0:
        p = 1.0      # identical arms: no evidence of a difference, so p must not
                     # be 0.0 -- claiming significance for a provable tie is the
                     # worst error an eval harness can make
    else:
        p = 0.0      # zero variance with a non-zero gap is a deterministic effect
    return Effect(mean, se, mean - tcrit * se, mean + tcrit * se, n, label, p,
                  extra={"df": float(df), "n_clusters": float(
                      np.unique(np.asarray(groups)).size) if groups is not None else 1.0})


def unpaired_effect(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> Effect:
    """Difference of independent means (Welch), i.e. disjoint task sets."""
    from scipy import stats
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    diff = float(a.mean() - b.mean())
    va = float(a.var(ddof=1) / a.size)
    vb = float(b.var(ddof=1) / b.size)
    se = math.sqrt(va + vb)
    df = (va + vb) ** 2 / max(va ** 2 / (a.size - 1) + vb ** 2 / (b.size - 1), 1e-300)
    t = stats.t.ppf(1.0 - alpha / 2.0, df)
    # Derive the p-value from the statistics computed above rather than calling
    # scipy.stats.ttest_ind for it. ttest_ind recomputes a variance internally and
    # emits a catastrophic-cancellation warning when the arms are identical or
    # near-identical, which for small local models is the common case, not an
    # edge case. The Welch statistic here is the same one ttest_ind would use.
    if se > 0:
        p = float(2.0 * stats.t.sf(abs(diff / se), df))
    elif diff == 0.0:
        p = 1.0          # identical arms: no evidence of any difference
    else:
        p = 0.0          # zero variance but a non-zero gap is deterministic
    return Effect(diff, se, diff - t * se, diff + t * se, int(a.size + b.size),
                  "unpaired-welch", p)


def cuped_adjust(d: np.ndarray, x: np.ndarray, alpha: float = 0.05, *,
                 mu_x: float | None = None) -> Effect:
    """Adjust a paired difference using a baseline covariate.

    Two regimes, because they need different information:

    ``mu_x=None`` (default, ANCOVA)
        The covariate mean is estimated from the same cells, so centring kills
        the adjustment term and ``theta_hat = d_bar``. Orthogonality then gives
        ``Var = s2_d (1 - rho^2) / n`` and a clean variance reduction factor
        ``VRF = rho^2``. Needs nothing but the two arms.

    ``mu_x`` supplied (CUPED)
        A genuine pre-period mean is known, so the estimator is
        ``d_bar - rho (x_bar - mu_x)`` with
        ``Var = (s2_d - rho^2 (2 sd_d sd_x - sd_x^2)) / n`` and variance
        reduction factor ``VRF = rho^2 (2 sd_d sd_x - sd_x^2) / s2_d``, which
        beats ANCOVA whenever the covariate is at least half as noisy as the
        metric. With ``sd_x == sd_d`` this collapses to the familiar
        ``VRF = rho^2``.
    """
    from scipy import stats
    d = np.asarray(d, dtype=float)
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(d) & np.isfinite(x)
    d, x = d[ok], x[ok]
    n = d.size
    if n < 3 or np.std(x) < 1e-15:
        base = paired_effect(d, alpha)
        return Effect(base.estimate, base.se, base.ci_low, base.ci_high, n,
                      "cuped(degenerate)", base.p_value, 1.0, float("nan"),
                      {"note": "covariate has no variance; fell back to paired"})

    rho = float(np.corrcoef(d, x)[0, 1])
    s2_d = float(d.var(ddof=1))
    s2_x = float(x.var(ddof=1))
    sd_d, sd_x = math.sqrt(s2_d), math.sqrt(s2_x)

    if mu_x is None:
        est = float(d.mean())
        var_mean = s2_d * (1.0 - rho * rho) / n
        method = "ancova"
    else:
        est = float(d.mean() - rho * (x.mean() - float(mu_x)))
        var_mean = max(s2_d - rho * rho * (2.0 * sd_d * sd_x - s2_x), 0.0) / n
        method = "cuped"

    se = math.sqrt(max(var_mean, 0.0))
    t = stats.t.ppf(1.0 - alpha / 2.0, n - 1)
    tstat = est / se if se > 0 else float("inf")
    p = float(2.0 * stats.t.sf(abs(tstat), n - 1)) if se > 0 else 0.0
    vrf = 1.0 - (se * se * n) / s2_d if s2_d > 0 else 1.0
    return Effect(est, se, est - t * se, est + t * se, n, method, p, vrf, rho,
                  {"vrf": vrf, "s2_d": s2_d, "s2_x": s2_x,
                   "predicted_vrf_rho2": rho * rho})


def bootstrap_paired_ci(d: np.ndarray, groups: np.ndarray, n_boot: int = 4000,
                        alpha: float = 0.05, seed: int = 0) -> Effect:
    """Distribution-free cross-check: resample whole task clusters, not cells."""
    rng = np.random.default_rng(seed)
    d = np.asarray(d, dtype=float)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_group = [np.flatnonzero(groups == g) for g in uniq]
    draws = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, uniq.size, uniq.size)
        sel = np.concatenate([idx_by_group[p] for p in pick])
        draws[b] = d[sel].mean()
    lo, hi = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    se = float(draws.std(ddof=1))
    est = float(d.mean())
    # A percentile interval is not centred on the estimate, so report a p-value
    # that is consistent with it: the bootstrap mass on the far side of zero.
    p = 2.0 * min(float((draws <= 0.0).mean()), float((draws >= 0.0).mean()))
    tstat = est / se if se > 0 else (float("inf") if est > 0 else float("-inf"))
    return Effect(est, se, float(lo), float(hi), int(d.size),
                  "paired-bootstrap", p_value=min(p, 1.0),
                  extra={"n_groups": int(uniq.size), "n_boot": n_boot,
                         "note": "p-value is the bootstrap mass beyond zero; "
                                 "the interval is a percentile, not a symmetric one"})


def crossed_effect(d: np.ndarray, idx, vc=None, alpha: float = 0.05) -> Effect:
    """Effect whose standard error comes from the crossed variance decomposition.

    This is the estimator to trust on a real agent benchmark. A t-interval that
    treats the cells as i.i.d. - ``s2_total / n_cells`` - is wrong twice over:
    cells within a task are not independent, and more importantly the axes do not
    share a denominator. ``q_t`` is reused by every task and judge on seed ``t``,
    so seeds average down at ``1/n_seed`` and not ``1/n_cells``; likewise for
    judges. Ignoring that overstates precision, often by a large factor, and the
    resulting interval badly undercovers.

    Fitting the variance components and assembling

        se = sqrt(s2_task/n_task + s2_seed/n_seed + s2_judge/n_judge
                  + s2_eps/n_cells)

    with a Satterthwaite df yields an interval that covers near its nominal rate
    while remaining honest that a task-dominant design has few effective degrees
    of freedom however many cells it runs.
    """
    from .design import crossed_variance, satterthwaite_df, unidentifiable_factors

    d = np.asarray(d, dtype=float).ravel()
    n = d.size
    if vc is None:
        from .vcomp import moment_estimator
        vc = moment_estimator(d, idx)
    nt = int(np.unique(idx.task).size)
    ns = int(np.unique(idx.seed).size) if getattr(idx, "seed", None) is not None else 1
    nj = int(np.unique(idx.judge).size) if getattr(idx, "judge", None) is not None else 1
    terms = crossed_variance(vc, nt, ns, nj)["terms"]
    se = math.sqrt(max(sum(terms.values()), 0.0))
    est = float(d.mean())
    df = satterthwaite_df(vc, nt, ns, nj)
    crit = float(t.ppf(1.0 - alpha / 2.0, max(df, 1.0))) if np.isfinite(df) \
        else float(norm.ppf(1.0 - alpha / 2.0))
    if not (np.isfinite(crit) and crit > 0):
        crit = float(norm.ppf(1.0 - alpha / 2.0))
    if se > 0:
        p = float(2.0 * t.sf(abs(est / se), max(df, 1.0)))
    elif est == 0.0:
        p = 1.0      # a provable tie is not evidence of a difference
    else:
        p = 0.0      # zero variance with a non-zero gap is a deterministic effect
    blind = unidentifiable_factors(ns, nj)
    return Effect(
        estimate=est, se=se, ci_low=est - crit * se, ci_high=est + crit * se,
        n=d.size, method="crossed-variance-components", p_value=p,
        extra={"df": float(df) if np.isfinite(df) else float("inf"),
               "terms": {k: float(v) for k, v in terms.items()},
               "n_tasks": nt, "n_seeds": ns, "n_judges": nj,
               "unidentifiable": blind,
               "warning": (f"only one level of {', '.join(blind)}: its variance is "
                           "confounded with task difficulty, so the interval is too "
                           "narrow by up to "
                           f"{math.sqrt(nt):.1f}x" if blind else ""),
               "components": asdict(vc)},
    )


def comparison_report(y_arm: np.ndarray, y_base: np.ndarray,
                      arm_cells: np.ndarray | None = None,
                      base_cells: np.ndarray | None = None,
                      task_ids: np.ndarray | None = None,
                      cell_index=None,
                      alpha: float = 0.05) -> dict:
    """Side-by-side of every estimator on the same data.

    Reports the naive iid interval, the task-clustered one and the variance
    component one together, because the ratios between them are usually the most
    actionable numbers in the whole report: they are how much a published
    interval was too narrow. ``comparison_report`` reports what you would have
    written down versus what the design actually supports.
    """
    y_arm = np.asarray(y_arm, dtype=float)
    y_base = np.asarray(y_base, dtype=float)
    paired = y_arm - y_base
    out: dict = {
        "arm_mean": float(y_arm.mean()),
        "base_mean": float(y_base.mean()),
        "paired_naive": paired_effect(paired, alpha, method="paired-iid").to_dict(),
        "paired": paired_effect(paired, alpha, groups=task_ids,
                                method="paired-cluster").to_dict(),
        "unpaired": unpaired_effect(y_arm, y_base, alpha).to_dict(),
    }
    if cell_index is not None:
        out["crossed"] = crossed_effect(paired, cell_index, alpha=alpha).to_dict()
    p = out["paired"]
    nv = out["paired_naive"]
    out["cluster_inflation"] = float(p["se"] / nv["se"]) if nv["se"] > 0 else float("nan")
    x = out.get("crossed")
    if x is not None and x["se"] > 0:
        out["crossed_inflation"] = float(x["se"] / nv["se"])
    u = out["unpaired"]
    out["variance_reduction_paired_vs_unpaired"] = (
        float(1.0 - (p["se"] / u["se"]) ** 2) if u["se"] > 0 else float("nan")
    )
    if arm_cells is not None and base_cells is not None:
        out["cuped"] = cuped_adjust(paired, np.asarray(base_cells, dtype=float),
                                    alpha).to_dict()
    if task_ids is not None:
        out["bootstrap"] = bootstrap_paired_ci(paired, np.asarray(task_ids),
                                               alpha=alpha).to_dict()
    return out


def rank_stability(agent_task_means: Mapping[str, np.ndarray],
                   n_boot: int = 4000, seed: int = 0) -> dict:
    """How fragile is the leaderboard to drawing a different set of tasks?

    Resamples task clusters with replacement and reports how often the winner
    changes, plus the modal winner and a Kendall-style rank agreement.
    """
    names = list(agent_task_means)
    mat = np.column_stack([np.asarray(agent_task_means[n], dtype=float) for n in names])
    n_tasks = mat.shape[0]
    rng = np.random.default_rng(seed)
    point = mat.mean(axis=0)
    winner = int(np.argmax(point))

    wins = np.zeros(len(names))
    top1_agree = 0
    top2_agree = 0
    rank_corr: list[float] = []
    for _ in range(n_boot):
        pick = rng.integers(0, n_tasks, n_tasks)
        boot = mat[pick].mean(axis=0)
        w = int(np.argmax(boot))
        wins[w] += 1
        top1_agree += int(w == winner)
        rb = np.argsort(np.argsort(-boot))
        rp = np.argsort(np.argsort(-point))
        top2_agree += int(set(np.argsort(-boot)[:2]) == set(np.argsort(-point)[:2]))
        d = rb - rp
        denom = len(names) * (len(names) ** 2 - 1) / 12.0
        rank_corr.append(1.0 - 6.0 * float((d ** 2).sum()) / (n_tasks * denom) if denom else 0.0)

    return {
        "point_estimates": {n: float(v) for n, v in zip(names, point)},
        "point_winner": names[winner],
        "p_top1_unchanged": top1_agree / n_boot,
        "p_top1_changes": 1.0 - top1_agree / n_boot,
        "p_top2_unchanged": top2_agree / n_boot,
        "modal_winner": names[int(np.argmax(wins))],
        "modal_winner_freq": float(wins.max() / n_boot),
        "win_share": {n: float(v / n_boot) for n, v in zip(names, wins)},
        "mean_rank_agreement": float(np.mean(rank_corr)),
        "n_boot": n_boot,
        "n_tasks": int(n_tasks),
    }


def replication_curve(agent_cells: Mapping[str, np.ndarray], reps: Sequence[int],
                      n_trials: int = 2000, seed: int = 0) -> list[dict]:
    """Leaderboard stability as a function of seeds-per-task ``T``.

    ``agent_cells[name]`` must be ``(n_tasks, n_seeds)`` judged scores. For each
    ``T`` we repeatedly draw ``T`` seeds and measure how often the resulting
    leaderboard reproduces the ordering from the full data.
    """
    names = list(agent_cells)
    arrs = {n: np.asarray(agent_cells[n], dtype=float) for n in names}
    n_tasks, n_seeds = next(iter(arrs.values())).shape
    full = np.column_stack([arrs[n].mean(axis=1) for n in names]).mean(axis=0)
    full_order = np.argsort(np.argsort(-full))
    full_top = int(np.argmax(full))
    rng = np.random.default_rng(seed)

    out: list[dict] = []
    for t in reps:
        if t > n_seeds:
            continue
        same_top = 0
        same_order = 0
        se_of_winner_gap: list[float] = []
        for _ in range(n_trials):
            cols = rng.choice(n_seeds, size=t, replace=False) if t < n_seeds \
                else np.arange(n_seeds)
            means = np.array([arrs[n][:, cols].mean() for n in names])
            same_top += int(int(np.argmax(means)) == full_top)
            same_order += int(np.array_equal(np.argsort(np.argsort(-means)), full_order))
            top2 = np.sort(means)[-2:]
            se_of_winner_gap.append(float(top2[1] - top2[0]))
        out.append({
            "reps": int(t),
            "p_top1_unchanged": same_top / n_trials,
            "p_full_order_unchanged": same_order / n_trials,
            "expected_margin_over_runner_up": float(np.mean(se_of_winner_gap)),
        })
    return out

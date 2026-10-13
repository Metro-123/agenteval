"""Assemble a full evaluation report from a benchmark run.

The report is the artefact the dashboard renders and the thing a researcher would
actually read. It exists to make three uncomfortable facts impossible to miss:

1. **The naive interval is wrong, and by how much.** Every comparison is reported
   three ways - iid cells, task-clustered cells, and the crossed
   variance-component interval - with the inflation ratios between them, so the
   cost of the shortcut is a number rather than a footnote.
2. **Precision is bought per axis.** Variance components and costs are shown
   side by side with the resulting standard error, and the allocation the planner
   recommends next.
3. **A judge is an instrument, not an oracle.** Planted operating points sit next
   to Dawid-Skene estimates, and the bias-corrected score next to the raw rate.

The whole report is JSON-serialisable and deterministic given a :class:`SuiteSpec`.
"""

from __future__ import annotations

import math
import time
from typing import Sequence

import numpy as np

from .design import (
    DEFAULT_COSTS,
    Design,
    crossed_variance,
    design_table,
    min_cost_design,
    min_detectable_delta,
    unidentifiable_factors,
)
from .effects import (
    bootstrap_paired_ci,
    comparison_report,
    crossed_effect,
    paired_effect,
    rank_stability,
    replication_curve,
)
from .judge import bias_corrected_score, judge_score_ci, judge_summary
from .sim import (
    SuiteSpec,
    default_agents,
    default_judges,
    field_with_sign,
    simulate_suite,
    task_arm_matrix,
)
from .sim import _cell_order
from .vcomp import VarianceComponents, moment_estimator, reml_estimator

__all__ = ["build_report", "DEFAULT_TARGET_DELTA"]

DEFAULT_TARGET_DELTA = 0.03


def _f(x) -> float:
    v = float(x)
    return v if math.isfinite(v) else float("nan")


def _mean_scores(suite: dict, arm: str) -> np.ndarray:
    """Per-task mean judged score for one arm, averaged over seeds and judges."""
    return task_arm_matrix(suite, arm).mean(axis=1)


def _cell_scores(suite: dict, arm: str) -> np.ndarray:
    """Flat per-cell judged scores in ``cell_index`` order (judge axis fastest)."""
    return _cell_order(suite["judged"], arm, list(suite["judges"]))


def _variance_block(d: np.ndarray, idx) -> dict:
    """Both component estimators, the resulting SE, and where the variance sits."""
    mom = moment_estimator(d, idx)
    reml = reml_estimator(d, idx)
    nt = int(np.unique(idx.task).size)
    ns = int(np.unique(idx.seed).size)
    nj = int(np.unique(idx.judge).size)
    naive_se = float(d.std(ddof=1) / math.sqrt(d.size))
    out = {
        "n_tasks": nt, "n_seeds": ns, "n_judges": nj, "n_cells": int(d.size),
        "n_tied_cells": int((d == 0.0).sum()),
        "tie_rate": float((d == 0.0).mean()),
        "naive_se": naive_se,
        "unidentifiable": unidentifiable_factors(ns, nj),
    }
    for tag, vc in (("mom", mom), ("reml", reml)):
        cv = crossed_variance(vc, nt, ns, nj)
        out[tag] = {
            "components": {
                "task": vc.sigma2_task, "seed": vc.sigma2_seed,
                "judge": vc.sigma2_judge, "eps": vc.sigma2_eps,
            },
            "se": cv["se"],
            "se_terms": cv["terms"],
            "dominant_term": cv["dominant"],
            "se_inflation_vs_naive": cv["se"] / naive_se if naive_se > 0 else float("nan"),
        }
    return out


def _judge_block(suite: dict, spec: SuiteSpec) -> dict:
    """Dawid-Skene on the pooled judgement matrix, against the planted truth."""
    agents: list[str] = suite["agents"]
    judges: list[str] = suite["judges"]
    nj = len(judges)
    truth_rows: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    for ag in agents:
        y = np.stack([suite["judged"][f"{ag}|{j}"] for j in judges])
        # one row per (task, seed); judges are the columns
        label_blocks.append(y.transpose(1, 2, 0).reshape(-1, nj).astype(int))
        # the run outcome is judge-independent, so it indexes the rows directly
        truth_rows.append(suite["raw"][f"{ag}|{judges[0]}"].ravel().astype(int))
    truth = np.concatenate(truth_rows)
    labels = np.vstack(label_blocks)

    summary = judge_summary(labels, truth=truth)
    planted = {j.name: {"sensitivity": j.sensitivity, "specificity": j.specificity,
                        "position_bias": j.position_bias,
                        "verbosity_bias": j.verbosity_bias}
               for j in (spec.judges or default_judges())}
    for k, name in enumerate(judges):
        entry = summary["judges"].get(str(k))
        if entry is None:
            continue
        p = planted.get(name, {})
        entry["name"] = name
        entry["planted"] = p
        entry["sensitivity_error"] = entry.get("sensitivity", float("nan")) \
            - p.get("sensitivity", float("nan"))
        entry["specificity_error"] = entry.get("specificity", float("nan")) \
            - p.get("specificity", float("nan"))
        # Bias-correct the judge's *observed positive rate*, not its agreement.
        # The inversion is on the rate: p_hat = theta(q1 + q0 - 1) + (1 - q0).
        # Agreement with truth is a different quantity and correcting it
        # saturates at 1.0, which would silently hide the judge's bias.
        observed = float((labels[:, k] == 1).mean())
        q1 = entry.get("sensitivity", float("nan"))
        q0 = entry.get("specificity", float("nan"))
        entry["observed_positive_rate"] = observed
        if math.isfinite(observed) and math.isfinite(q1) and math.isfinite(q0):
            entry["bias_corrected_rate"] = bias_corrected_score(observed, q1, q0)
            entry["raw_overstates_by"] = observed - entry["bias_corrected_rate"]
            entry["interval"] = judge_score_ci(observed, q1, q0, int(labels.shape[0]), 200)
    ds = dict(summary["dawid_skene"])
    latent = np.asarray(ds.pop("latent_truth"), dtype=float)
    return {
        "n_items": summary["n_items"],
        "n_judges": len(judges),
        "latent_positive_rate": float(latent.mean()),
        "observed_positive_rate": float(labels.mean()),
        "true_positive_rate": float(truth.mean()),
        "dawid_skene": ds,
        "judges": summary["judges"],
    }


def _planner_block(vc: VarianceComponents, design: Design, delta: float,
                   n_boot: int) -> dict:
    """What the current design can see, and what to buy next."""
    n = design.n_tasks * design.n_seeds * design.n_judges
    floor_2 = min_detectable_delta(vc, 2, 2)
    out = {
        "design": design.as_dict(),
        "cost": design.cost(),
        "costs": dict(DEFAULT_COSTS),
        "unidentifiable": unidentifiable_factors(design.n_seeds, design.n_judges),
        "min_detectable_delta": {
            "at_design": min_detectable_delta(vc, design.n_seeds, design.n_judges),
            "at_2x2": floor_2,
            "at_8x8": min_detectable_delta(vc, 8, 8),
            "at_32x32": min_detectable_delta(vc, 32, 32),
        },
        "table": design_table(
            vc,
            n_tasks=[int(x) for x in (8, 16, 24, 32, 48, 64, 96, 128)],
            n_seeds=[1, 2, 3, 4, 6, 8],
            n_judges=[1, 2, 3, 4],
            delta=delta,
        ),
    }
    try:
        rec = min_cost_design(delta=delta, vc=vc, target_power=0.8)
        out["recommended"] = {
            **rec.as_dict(),
            "cost": rec.cost(),
            "power": rec.power(delta, vc),
            "ci_half_width": rec.detectability(delta, vc),
            "cost_ratio_vs_current": rec.cost() / max(design.cost(), 1e-9),
        }
    except ValueError as exc:
        out["recommended"] = None
        out["infeasible"] = str(exc)
    out["current_power_at_target"] = design.power(delta, vc)
    out["n_cells"] = n
    return out


def _spec_from_suite(suite: dict) -> SuiteSpec:
    """Reconstruct a :class:`SuiteSpec` from a live run's own data.

    A real run knows its task and seed counts only implicitly, through the shape of
    the matrices it produced. Reading them back is better than requiring the caller
    to restate the design: a restated design that disagreed with the data would
    silently analyse the wrong number of cells.
    """
    arms = list(suite["agents"])
    judges = list(suite["judges"])
    if not arms or not judges:
        raise ValueError("suite needs at least one agent and one judge")
    first = suite["judged"][f"{arms[0]}|{judges[0]}"]
    if first.ndim != 2:
        raise ValueError(f"expected a (n_tasks, n_seeds) matrix, got {first.shape}")
    n_tasks, n_seeds = first.shape
    for arm in arms:
        for j in judges:
            got = suite["judged"][f"{arm}|{j}"].shape
            if got != (n_tasks, n_seeds):
                raise ValueError(
                    f"ragged suite: {arm}|{j} is {got}, expected {(n_tasks, n_seeds)}")
    return SuiteSpec(n_tasks=n_tasks, n_seeds=n_seeds)


def build_report(spec: SuiteSpec | None = None,
                 baseline: str | None = None,
                 target_delta: float = DEFAULT_TARGET_DELTA,
                 n_boot: int = 2000,
                 suite: dict | None = None) -> dict:
    """Run (or accept) a benchmark and return the full report as a dict.

    Pass ``suite`` to analyse results from a real run instead of simulating them.
    A live run must return the same structure :func:`simulate_suite` produces, so
    nothing downstream changes -- see :mod:`agenteval.live` for the runner. The
    statistics are identical either way; only where the numbers came from differs.
    """
    t0 = time.perf_counter()
    if suite is None:
        if spec is None:
            spec = SuiteSpec(agents=default_agents(), judges=default_judges())
        suite = simulate_suite(spec)
    elif spec is None:
        # a real run carries its own design; infer it from the data so the
        # analysis matches what was actually run
        spec = _spec_from_suite(suite)
    agents: list[str] = suite["agents"]
    judges: list[str] = suite["judges"]
    if not agents:
        raise ValueError("suite has no agents")
    if baseline is None:
        # Default to the *lowest scoring* agent, which is the conventional control
        # and keeps every effect on the same sign. The interval width is what the
        # report is really about, and that does not depend on this choice.
        baseline = spec.baseline or min(agents, key=lambda a: _mean_scores(suite, a).mean())
    if baseline not in agents:
        raise ValueError(f"baseline {baseline!r} not in {agents}")

    warnings: list[str] = []
    n_tasks, n_seeds, n_judges = spec.n_tasks, spec.n_seeds, len(judges)
    design = Design(n_tasks, n_seeds, n_judges)
    blind = unidentifiable_factors(n_seeds, n_judges)
    if blind:
        warnings.append(
            f"only one level of {', '.join(blind)}: the variance of {', '.join(blind)} "
            "is confounded with task difficulty, so intervals from this run are too "
            "narrow and cannot be corrected after the fact. Re-run with at least 2.")
    if n_judges < 2:
        warnings.append("a single judge gives no way to separate judge bias from noise")

    comparisons: list[dict] = []
    for arm in agents:
        if arm == baseline:
            continue
        d, idx, _ = field_with_sign(suite, arm, baseline)
        mom = moment_estimator(d, idx)
        rep = comparison_report(
            _cell_scores(suite, arm),
            _cell_scores(suite, baseline),
            task_ids=idx.task, cell_index=idx,
        )
        crossed = crossed_effect(d, idx, vc=mom)
        boot = bootstrap_paired_ci(d, idx.task, n_boot=n_boot)
        entry = {
            "arm": arm,
            "baseline": baseline,
            "arm_mean": float(task_arm_matrix(suite, arm).mean()),
            "baseline_mean": float(task_arm_matrix(suite, baseline).mean()),
            "estimators": {
                "naive": rep["paired_naive"],
                "cluster": rep["paired"],
                "crossed": crossed.to_dict(),
                "bootstrap": boot.to_dict(),
            },
            "variance": _variance_block(d, idx),
            "se_inflation_naive_to_crossed": (
                crossed.se / rep["paired_naive"]["se"]
                if rep["paired_naive"]["se"] > 0 else float("nan")),
        }
        entry["ranking"] = {
            "mean": entry["arm_mean"],
            "crossed": crossed.to_dict(),
            "significant_vs_baseline": crossed.significant,
        }
        comparisons.append(entry)

    # leaderboard ordering under the honest interval
    board = []
    for arm in agents:
        m = _mean_scores(suite, arm)
        board.append({
            "agent": arm,
            "mean": float(m.mean()),
            "sd_across_tasks": float(m.std(ddof=1)) if m.size > 1 else float("nan"),
            "is_baseline": arm == baseline,
            **{f"meta_{k}": v for k, v in suite["meta"][arm].items()},
        })
    for row in board:
        cmp_ = next((c for c in comparisons if c["arm"] == row["agent"]), None)
        row["ci_low"] = cmp_["ranking"]["crossed"]["ci_low"] if cmp_ else float("nan")
        row["ci_high"] = cmp_["ranking"]["crossed"]["ci_high"] if cmp_ else float("nan")
        row["se"] = cmp_["ranking"]["crossed"]["se"] if cmp_ else float("nan")
        row["significant"] = bool(cmp_ and cmp_["ranking"]["significant_vs_baseline"])
    board.sort(key=lambda r: -r["mean"])

    task_means = {a: _mean_scores(suite, a) for a in agents}
    stability = rank_stability(task_means, n_boot=n_boot)
    cells_by_agent = {a: task_arm_matrix(suite, a) for a in agents}
    rep_curve = replication_curve(
        {a: m for a, m in cells_by_agent.items()},
        reps=tuple(r for r in (1, 2, 3, 4, 6, 8) if r <= n_seeds * n_judges),
        n_trials=max(500, n_boot // 2),
    )

    # Plan against the *largest* observed difference: a design that cannot resolve
    # the best effect on the board cannot resolve anything smaller either.
    ref_cmp = max(comparisons,
                  key=lambda c: abs(c["ranking"]["crossed"]["estimate"]),
                  default=None)
    if ref_cmp is not None:
        d_ref, idx_ref, _ = field_with_sign(suite, ref_cmp["arm"], ref_cmp["baseline"])
        planner = _planner_block(moment_estimator(d_ref, idx_ref), design,
                                 target_delta, n_boot)
        planner["reference"] = {
            "arm": ref_cmp["arm"],
            "baseline": ref_cmp["baseline"],
            "observed_effect": ref_cmp["ranking"]["crossed"]["estimate"],
            "note": "variance components are from the largest observed contrast",
        }
    else:
        planner = _planner_block(VarianceComponents(), design, target_delta, n_boot)
        planner["reference"] = {"note": "no comparisons; zero components assumed"}

    pareto = [{
        "agent": row["agent"],
        "mean": row["mean"],
        "cost": row.get("meta_cost_per_task_tokens"),
        "latency": row.get("meta_latency_s_per_task"),
        "sd": row["sd_across_tasks"],
    } for row in board]

    # the baseline row has no comparison, so give it the interval the next-best
    # agent has, purely to keep the leaderboard chart on one common scale
    nonzero = [r["se"] for r in board if math.isfinite(r["se"])]
    fallback = sorted(nonzero)[len(nonzero) // 2] if nonzero else float("nan")
    for row in board:
        if not math.isfinite(row["se"]):
            row["ci_low"] = row["mean"] - 1.96 * fallback
            row["ci_high"] = row["mean"] + 1.96 * fallback
            row["se"] = fallback
            row["ci_note"] = ("no comparison is the baseline; interval borrowed "
                              "from the median comparison for display only")

    report = {
        "meta": {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "elapsed_s": round(time.perf_counter() - t0, 3),
            "baseline": baseline,
            "n_agents": len(agents),
            "n_tasks": n_tasks,
            "n_seeds": n_seeds,
            "n_judges": n_judges,
            "n_cells": n_tasks * n_seeds * n_judges,
            "target_delta": target_delta,
            "spec": spec.to_dict(),
        },
        "warnings": warnings,
        "leaderboard": board,
        "pareto": pareto,
        "comparisons": comparisons,
        "judges": _judge_block(suite, spec),
        "rank_stability": stability,
        "replication_curve": rep_curve,
        "planner": planner,
    }
    return _jsonable(report)


def _jsonable(obj):
    """Replace non-finite floats with None so the output is real JSON.

    ``json.dumps`` happily writes a bare ``NaN``, but that is not valid JSON and
    ``JSON.parse`` throws on it, which would take the whole dashboard down. Several
    fields are legitimately undefined -- CUPED's ``rho`` when there is no covariate,
    MoM's ``loglik`` because the method has no likelihood -- so they must serialise
    as ``null`` rather than leaking a token no parser accepts.
    """
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj

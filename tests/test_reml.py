"""Pin the REML objective against a brute-force direct computation.

Every algebraic shortcut in ``_reml_objective`` (Woodbury inverse, matrix
determinant lemma, GLS mean) is checked against explicitly building ``V``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from agenteval.vcomp import (
    CellIndex,
    _design_matrices,
    _levels,
    _reml_objective,
    reml_estimator,
)
from agenteval.sim import FieldSpec, simulate_difference_field


def _grid(nt: int, ns: int, nj: int) -> CellIndex:
    return CellIndex(
        np.repeat(np.arange(nt), ns * nj),
        np.tile(np.repeat(np.arange(ns), nj), nt),
        np.tile(np.arange(nj), nt * ns),
    )


@pytest.mark.parametrize("shape", [(3, 2, 2), (5, 3, 2), (4, 1, 3), (6, 2, 1)])
def test_objective_matches_direct_construction(shape: tuple[int, int, int]) -> None:
    idx = _grid(*shape)
    factors = tuple(f for f, n in zip(("task", "seed", "judge"), shape) if n >= 2)
    levels = _levels(idx, factors)
    m = sum(levels)
    n = idx.n
    A = _design_matrices(idx, factors)
    y = np.random.default_rng(abs(hash(shape)) % 10000).normal(size=n)

    per_factor = (0.03, 0.07, 0.05)[:len(factors)]
    col_var = np.concatenate([np.full(lv, v) for v, lv in zip(per_factor, levels)])
    log_var = np.concatenate([np.log(per_factor), [math.log(0.11)]])
    s2e = 0.11

    S = np.diag(col_var)
    V = s2e * np.eye(n) + A @ S @ A.T
    Vi = np.linalg.inv(V)
    X = np.ones((n, 1))
    beta = np.linalg.solve(X.T @ Vi @ X, X.T @ Vi @ y)
    r = y - X @ beta
    direct = (float(np.linalg.slogdet(V)[1])
              + float(np.linalg.slogdet(X.T @ Vi @ X)[1])
              + float(r @ Vi @ r))

    assert _reml_objective(log_var, y, idx, factors) == pytest.approx(direct, rel=1e-8)


def test_objective_is_minimised_near_truth() -> None:
    spec = FieldSpec(n_tasks=40, n_seeds=4, n_judges=3,
                     sigma2_task=0.040, sigma2_seed=0.006,
                     sigma2_judge=0.009, sigma2_eps=0.012, seed=23)
    d, idx, _ = simulate_difference_field(spec)
    f = ("task", "seed", "judge")
    base = np.log([0.0125, 0.0125, 0.0125, 0.0125])
    truth = np.log([0.040, 0.006, 0.009, 0.012])
    collapsed = np.log([1e-10, 1e-10, 1e-10, 0.067])
    assert _reml_objective(truth, d, idx, f) < _reml_objective(base, d, idx, f)
    assert _reml_objective(truth, d, idx, f) < _reml_objective(collapsed, d, idx, f)


def test_reml_recovers_truth_on_large_design() -> None:
    spec = FieldSpec(n_tasks=200, n_seeds=6, n_judges=4,
                     sigma2_task=0.040, sigma2_seed=0.006,
                     sigma2_judge=0.009, sigma2_eps=0.012, seed=31)
    d, idx, truth = simulate_difference_field(spec)
    vc = reml_estimator(d, idx)
    assert vc.sigma2_task == pytest.approx(truth.sigma2_task, rel=0.30)
    assert vc.sigma2_eps == pytest.approx(truth.sigma2_eps, rel=0.30)
    assert vc.total == pytest.approx(truth.total, rel=0.20)


def test_reml_beats_mom_on_held_out_likelihood() -> None:
    spec = FieldSpec(n_tasks=60, n_seeds=4, n_judges=3,
                     sigma2_task=0.040, sigma2_seed=0.006,
                     sigma2_judge=0.009, sigma2_eps=0.012, seed=37)
    d, idx, _ = simulate_difference_field(spec)
    f = ("task", "seed", "judge")
    mom = reml_estimator(d, idx)
    from agenteval.vcomp import moment_estimator
    m = moment_estimator(d, idx)
    obj_mom = _reml_objective(
        np.log([max(m.sigma2_task, 1e-10), max(m.sigma2_seed, 1e-10),
                max(m.sigma2_judge, 1e-10), max(m.sigma2_eps, 1e-10)]), d, idx, f)
    obj_reml = _reml_objective(
        np.log([max(mom.sigma2_task, 1e-10), max(mom.sigma2_seed, 1e-10),
                max(mom.sigma2_judge, 1e-10), max(mom.sigma2_eps, 1e-10)]), d, idx, f)
    assert obj_reml <= obj_mom + 1e-6

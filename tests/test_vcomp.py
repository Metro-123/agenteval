"""Prove the variance-component estimators are unbiased and correctly calibrated.

The simulator draws components with known magnitude, so we can check recovery
by Monte Carlo rather than trusting the algebra.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from agenteval.sim import FieldSpec, simulate_difference_field
from agenteval.vcomp import (
    CellIndex,
    component_standard_errors,
    mean_squares,
    moment_estimator,
    reml_estimator,
)


def test_moment_estimator_unbiased() -> None:
    spec = FieldSpec(n_tasks=25, n_seeds=4, n_judges=3, delta=0.05,
                     sigma2_task=0.040, sigma2_seed=0.006,
                     sigma2_judge=0.009, sigma2_eps=0.012, seed=0)
    truth = spec.truth()
    names = ("sigma2_task", "sigma2_seed", "sigma2_judge", "sigma2_eps")
    acc = {n: [] for n in names}
    reps = 400
    for r in range(reps):
        rep_spec = replace(spec, seed=1000 + r)
        d, idx, _ = simulate_difference_field(rep_spec)
        vc = moment_estimator(d, idx)
        for n in names:
            acc[n].append(getattr(vc, n))
    for n in names:
        target = getattr(truth, n)
        m = float(np.mean(acc[n]))
        se_of_mean = float(np.std(acc[n], ddof=1) / math.sqrt(reps))
        z = (m - target) / se_of_mean if se_of_mean > 0 else 0.0
        assert abs(z) < 4.0, f"{n}: mean {m:.5f} vs truth {target:.5f} (z={z:.2f})"


def test_moment_estimator_recovery_single_draw() -> None:
    spec = FieldSpec(n_tasks=200, n_seeds=4, n_judges=3,
                     sigma2_task=0.040, sigma2_seed=0.006,
                     sigma2_judge=0.009, sigma2_eps=0.012, seed=3)
    d, idx, truth = simulate_difference_field(spec)
    vc = moment_estimator(d, idx)
    assert vc.sigma2_task == pytest.approx(truth.sigma2_task, rel=0.20)
    assert vc.sigma2_eps == pytest.approx(truth.sigma2_eps, rel=0.20)


def test_total_variance_matches_observed_spread() -> None:
    spec = FieldSpec(n_tasks=120, n_seeds=4, n_judges=3, delta=0.0,
                     sigma2_task=0.040, sigma2_seed=0.006,
                     sigma2_judge=0.009, sigma2_eps=0.012, seed=5)
    d, idx, _ = simulate_difference_field(spec)
    vc = moment_estimator(d, idx)
    assert vc.total == pytest.approx(float(np.var(d, ddof=1)), rel=0.25)


def test_residual_mean_square_recovers_sigma_eps() -> None:
    spec = FieldSpec(n_tasks=60, n_seeds=5, n_judges=4, delta=0.0,
                     sigma2_task=0.04, sigma2_seed=0.0, sigma2_judge=0.0,
                     sigma2_eps=0.020, seed=9)
    d, idx, _ = simulate_difference_field(spec)
    ms = mean_squares(d, idx)
    assert ms["MS_resid"] == pytest.approx(0.020, rel=0.15)


def test_pure_task_structure_has_zero_residual() -> None:
    spec = FieldSpec(n_tasks=60, n_seeds=5, n_judges=4, delta=0.0,
                     sigma2_task=0.04, sigma2_seed=0.0, sigma2_judge=0.0,
                     sigma2_eps=0.0, seed=13)
    d, idx, _ = simulate_difference_field(spec)
    ms = mean_squares(d, idx)
    assert ms["MS_resid"] == pytest.approx(0.0, abs=1e-12)


def test_standard_errors_match_empirical_spread() -> None:
    spec = FieldSpec(n_tasks=15, n_seeds=3, n_judges=3,
                     sigma2_task=0.040, sigma2_seed=0.006,
                     sigma2_judge=0.009, sigma2_eps=0.012, seed=17)
    names = ("sigma2_task", "sigma2_seed", "sigma2_judge", "sigma2_eps")
    analytic: dict[str, list[float]] = {n: [] for n in names}
    empirical: dict[str, list[float]] = {n: [] for n in names}
    reps = 250
    for r in range(reps):
        d, idx, _ = simulate_difference_field(replace(spec, seed=5000 + r))
        vc = moment_estimator(d, idx)
        se = component_standard_errors(d, idx, vc)
        for n in names:
            analytic[n].append(se[n.split("_", 1)[1]])
            empirical[n].append(getattr(vc, n))
    for n in names:
        key = n.split("_", 1)[1]
        emp = float(np.std(empirical[n], ddof=1))
        ana = float(np.mean(analytic[n]))
        assert ana == pytest.approx(emp, rel=0.30), \
            f"{n}: analytic SE {ana:.5f} vs empirical {emp:.5f}"


def test_reml_nonnegative_and_recovers_truth() -> None:
    spec = FieldSpec(n_tasks=40, n_seeds=4, n_judges=3,
                     sigma2_task=0.040, sigma2_seed=0.006,
                     sigma2_judge=0.009, sigma2_eps=0.012, seed=23)
    d, idx, truth = simulate_difference_field(spec)
    vc = reml_estimator(d, idx)
    assert vc.sigma2_task >= 0 and vc.sigma2_seed >= 0
    assert vc.sigma2_judge >= 0 and vc.sigma2_eps >= 0
    assert vc.total == pytest.approx(truth.total, rel=0.35)


def test_reml_handles_single_seed_design() -> None:
    spec = FieldSpec(n_tasks=50, n_seeds=1, n_judges=3,
                     sigma2_task=0.04, sigma2_seed=0.0, sigma2_judge=0.006,
                     sigma2_eps=0.010, seed=29)
    d, idx, _ = simulate_difference_field(spec)
    assert idx.sizes[1] == 1
    vc = reml_estimator(d, idx)
    assert vc.n_estimable == 2
    assert vc.sigma2_seed == 0.0
    assert vc.sigma2_task > 0


def test_moment_handles_degenerate_factors() -> None:
    n_t, n_s = 20, 2
    idx = CellIndex(
        np.repeat(np.arange(n_t), n_s),
        np.tile(np.arange(n_s), n_t),
        np.zeros(n_t * n_s, dtype=np.int64),
    )
    d = np.random.default_rng(0).normal(size=n_t * n_s)
    vc = moment_estimator(d, idx)
    assert vc.n_estimable == 2
    assert set(vc.truncated) == {"judge"}
    assert vc.sigma2_judge == 0.0
    assert vc.sigma2_eps > 0
    assert vc.sigma2_task >= 0


def test_saturated_design_has_zero_residual() -> None:
    n = 30
    idx = CellIndex(np.arange(n), np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64))
    d = np.random.default_rng(1).normal(size=n)
    ms = mean_squares(d, idx)
    vc = moment_estimator(d, idx)
    assert ms["SS_resid"] == pytest.approx(0.0, abs=1e-18)
    assert vc.sigma2_eps == pytest.approx(0.0, abs=1e-18)
    assert vc.sigma2_task == pytest.approx(float(np.var(d, ddof=1)), rel=1e-6)

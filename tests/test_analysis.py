"""Effect estimators, design allocation and judge instrumentation.

Each claim is checked against a simulation whose parameters we chose, so the
tests fail if an estimator is biased rather than merely if it returns a number.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from agenteval.design import (
    Design,
    crossed_variance,
    min_cost_design,
    min_detectable_delta,
    required_cells,
    satterthwaite_df,
    unidentifiable_factors,
)
from agenteval.effects import (
    bootstrap_paired_ci,
    crossed_effect,
    cuped_adjust,
    paired_effect,
    paired_variance,
    pairing_payoff,
    rank_stability,
    replication_curve,
    unpaired_effect,
    unpaired_variance,
)
from agenteval.judge import (
    bias_corrected_score,
    cohen_kappa,
    dawid_skene,
    gwet_ac1,
    judge_score_ci,
    sensitivity_specificity,
)
from agenteval.vcomp import CellIndex, VarianceComponents, moment_estimator
from agenteval.sim import FieldSpec, field_with_sign, simulate_difference_field


# --------------------------------------------------------------------- pairing

def test_pairing_removes_shared_task_difficulty() -> None:
    """Pairing removes the shared term, and the predicted VRF must match the data."""
    rng = np.random.default_rng(0)
    n = 40
    s2_shared = 0.040
    s2_arm = 0.020
    s2_within = 0.027
    within_each = s2_within / 2.0

    gA1 = rng.normal(0, math.sqrt(s2_arm), n)
    gB1 = rng.normal(0, math.sqrt(s2_arm), n)
    d = 0.05 + (gA1 - gB1) + rng.normal(0, math.sqrt(s2_within), n)

    u2 = rng.normal(0, math.sqrt(s2_shared), n)
    a_only = 0.50 + u2 + rng.normal(0, math.sqrt(s2_arm), n) \
        + rng.normal(0, math.sqrt(within_each), n)
    b_only = 0.45 + u2 + rng.normal(0, math.sqrt(s2_arm), n) \
        + rng.normal(0, math.sqrt(within_each), n)

    p = paired_effect(d)
    up = unpaired_effect(a_only, b_only)
    assert p.se < up.se

    # the difference field carries no shared task term, only the arm-by-arm and
    # within-task noise; the shared variance enters only through the unpaired arm
    vc = VarianceComponents(0.0, 0.0, 0.0, 2.0 * s2_arm + s2_within, "difference")
    predicted = pairing_payoff(vc, n, s2_shared)["variance_reduction"]
    empirical = 1.0 - (p.se / up.se) ** 2
    assert empirical == pytest.approx(predicted, abs=0.10)


def test_unpaired_penalty_depends_on_unidentifiable_shared_variance() -> None:
    """The paired-vs-unpaired payoff cannot be computed from the difference field alone."""
    vc = VarianceComponents(0.0, 0.006, 0.009, 0.012, "difference")
    n = 100
    easy = unpaired_variance(shared_task=0.0, sigma2_difference=vc.total, n_tasks=n)
    hard = unpaired_variance(shared_task=0.20, sigma2_difference=vc.total, n_tasks=n)
    # unpaired = 2*shared/n + sigma2_diff/n, so a known shared term shifts it 1:1
    assert hard - easy == pytest.approx(2.0 * 0.20 / n, rel=1e-9)
    # with no shared term the pairing buys exactly nothing
    assert paired_variance(vc, n) == pytest.approx(easy, rel=1e-9)
    # the payoff swings with a quantity the difference field cannot observe
    payoff_easy = pairing_payoff(vc, n, shared_task=0.0)["variance_reduction"]
    payoff_hard = pairing_payoff(vc, n, shared_task=0.20)["variance_reduction"]
    assert payoff_hard > payoff_easy
    assert payoff_hard - payoff_easy > 0.2


# ------------------------------------------------- the crossed variance result

def test_naive_se_is_wrong_by_the_seed_and_judge_terms() -> None:
    """``s2_total / n_cells`` ignores that seeds and judges do not average down with cells."""
    vc = VarianceComponents(0.040, 0.006, 0.009, 0.012, "truth")
    nt, ns, nj = 12, 3, 2
    out = crossed_variance(vc, nt, ns, nj)
    assert out["se"] == pytest.approx(math.sqrt(
        vc.sigma2_task / nt + vc.sigma2_seed / ns + vc.sigma2_judge / nj
        + vc.sigma2_eps / (nt * ns * nj)))
    naive = math.sqrt(vc.total / (nt * ns * nj))
    assert out["se"] / naive > 3.0, "naive SE is off by more than 3x here"


def test_single_seed_contributes_its_full_variance() -> None:
    """One seed is a shared offset, not a term that disappears."""
    vc = VarianceComponents(0.040, 0.006, 0.009, 0.012, "truth")
    one = crossed_variance(vc, 60, 1, 1)
    assert one["terms"]["seed"] == pytest.approx(vc.sigma2_seed)
    assert one["terms"]["judge"] == pytest.approx(vc.sigma2_judge)
    assert one["dominant"] == "judge"


def test_unidentifiable_axes_are_reported() -> None:
    assert unidentifiable_factors(1, 1) == ["seed", "judge"]
    assert unidentifiable_factors(3, 1) == ["judge"]
    assert unidentifiable_factors(2, 2) == []


def test_naive_and_cluster_intervals_both_undercover() -> None:
    """Clustering on task alone cannot fix an SE that ignores the seed/judge axes."""
    hits_naive = hits_cluster = hits_crossed = 0
    trials = 250
    for r in range(trials):
        spec = FieldSpec(n_tasks=12, n_seeds=3, n_judges=2, delta=0.0,
                         sigma2_task=0.04, sigma2_seed=0.006, sigma2_judge=0.009,
                         sigma2_eps=0.012, seed=9000 + r)
        d, idx, _ = simulate_difference_field(spec)
        e = paired_effect(d)
        c = paired_effect(d, groups=idx.task)
        hits_cluster += int(c.ci_low <= 0.0 <= c.ci_high)
        x = crossed_effect(d, idx, vc=spec.truth())
        hits_crossed += int(x.ci_low <= 0.0 <= x.ci_high)
    assert hits_naive / trials < 0.55, "naive iid interval should undercover badly"
    assert hits_cluster / trials < hits_crossed / trials
    assert 0.90 <= hits_crossed / trials <= 1.0, "crossed SE should be near nominal"


def test_exact_tie_is_not_reported_as_significant() -> None:
    """Zero difference and zero variance must give p=1, never p=0.

    The degenerate case matters more than it looks: two small local models scoring
    identically is routine, and an eval harness that reports p=0.0 for a provable
    tie will confidently claim a difference that does not exist.
    """
    tie = np.zeros(8)
    assert paired_effect(tie).p_value == 1.0
    assert paired_effect(tie).se == 0.0
    # the CI must collapse onto the point estimate, not blow up
    e = paired_effect(tie)
    assert e.ci_low == e.ci_high == 0.0

    a = np.ones(8)
    b = np.ones(8)
    assert unpaired_effect(a, b).p_value == 1.0
    assert unpaired_effect(a, b).se == 0.0


def test_exact_tie_is_not_significant_under_the_crossed_estimator() -> None:
    """The crossed estimator must not claim a difference for a provable tie either.

    This is the estimator the dashboard leads with, so a p=0.0 here would be the
    most visible possible false positive.
    """
    from agenteval.live import build_cell_index
    from agenteval.vcomp import moment_estimator
    idx = build_cell_index(n_tasks=2, n_seeds=2, n_judges=2)
    tie = np.zeros(idx.task.size)
    e = crossed_effect(tie, idx)
    assert e.se == 0.0
    assert e.p_value == 1.0
    assert e.ci_low == e.ci_high == 0.0

    gap = np.full(idx.task.size, 0.25)
    assert crossed_effect(gap, idx).p_value == 0.0


def test_constant_nonzero_difference_is_significant() -> None:
    """The other degenerate case: zero variance, non-zero gap, so p=0."""
    gap = np.full(8, 0.25)
    assert paired_effect(gap).p_value == 0.0
    assert unpaired_effect(np.full(8, 0.75), np.full(8, 0.5)).p_value == 0.0


def test_p_value_matches_a_direct_t_test_where_defined() -> None:
    """The hand-rolled p-value must agree with scipy on the non-degenerate case.

    ``unpaired_effect`` stopped calling ``ttest_ind`` because that call warns on
    constant data; the replacement has to be numerically equivalent, not merely
    warning-free.
    """
    from scipy import stats
    rng = np.random.default_rng(4)
    a = rng.normal(0.6, 0.2, 30)
    b = rng.normal(0.5, 0.2, 25)
    mine = unpaired_effect(a, b).p_value
    reference = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
    assert math.isclose(mine, reference, rel_tol=1e-9)

    d = rng.normal(0.02, 0.1, 40)
    ref_paired = float(stats.ttest_1samp(d, 0.0).pvalue)
    assert math.isclose(paired_effect(d).p_value, ref_paired, rel_tol=1e-9)


def test_crossed_effect_recovers_the_planted_components() -> None:
    spec = FieldSpec(n_tasks=30, n_seeds=3, n_judges=2, delta=0.05,
                     sigma2_task=0.04, sigma2_seed=0.006, sigma2_judge=0.009,
                     sigma2_eps=0.012, seed=7)
    d, idx, _ = simulate_difference_field(spec)
    e = crossed_effect(d, idx, vc=spec.truth())
    assert e.se == pytest.approx(crossed_variance(spec.truth(), 30, 3, 2)["se"])
    assert e.extra["unidentifiable"] == []
    assert e.extra["n_tasks"] == 30 and e.extra["n_judges"] == 2

    ests, ses = [], []
    for r in range(200):
        dd, ii, _ = simulate_difference_field(
            FieldSpec(n_tasks=30, n_seeds=3, n_judges=2, delta=0.05,
                      sigma2_task=0.04, sigma2_seed=0.006, sigma2_judge=0.009,
                      sigma2_eps=0.012, seed=50_000 + r))
        ee = crossed_effect(dd, ii, vc=spec.truth())
        ests.append(ee.estimate)
        ses.append(ee.se)
    assert float(np.mean(ests)) == pytest.approx(0.05, abs=0.012)
    assert float(np.mean(ses)) == pytest.approx(e.se, rel=0.02)


def test_satterthwaite_df_tracks_the_dominant_axis() -> None:
    vc = VarianceComponents(0.040, 0.006, 0.009, 0.012, "truth")
    assert satterthwaite_df(vc, 10, 3, 2) < 10.0
    assert satterthwaite_df(vc, 100, 3, 2) < satterthwaite_df(vc, 10, 3, 2)


def test_cluster_robust_agrees_with_task_bootstrap_but_both_are_narrow() -> None:
    """The task-clustered SE matches resampling tasks, and is still too narrow.

    Resampling tasks cannot see the shared seed and judge offsets either, so the
    bootstrap reproduces the same under-coverage rather than validating it.
    """
    spec = FieldSpec(n_tasks=40, n_seeds=4, n_judges=3, delta=0.04,
                     sigma2_task=0.04, sigma2_seed=0.006, sigma2_judge=0.009,
                     sigma2_eps=0.012, seed=43)
    d, idx, _ = simulate_difference_field(spec)
    t = paired_effect(d, groups=idx.task)
    b = bootstrap_paired_ci(d, idx.task, n_boot=2000)
    naive = paired_effect(d)
    x = crossed_effect(d, idx, vc=spec.truth())
    assert t.ci_width == pytest.approx(b.ci_width, rel=0.30)
    assert t.ci_width > naive.ci_width
    assert x.ci_width > 2.0 * t.ci_width
    # the bootstrap p-value must be usable, not a silent NaN
    assert math.isfinite(b.p_value)
    assert 0.0 <= b.p_value <= 1.0
    # both should fail to reject here: the true effect is 0.04
    assert b.p_value > 0.05
    assert t.p_value > 0.05


def test_paired_effect_recovers_delta() -> None:
    truth = 0.05
    ests = []
    for r in range(300):
        d, _, _ = simulate_difference_field(
            FieldSpec(n_tasks=30, n_seeds=4, n_judges=3, delta=truth,
                      sigma2_task=0.04, sigma2_seed=0.006, sigma2_judge=0.009,
                      sigma2_eps=0.012, seed=7000 + r))
        ests.append(paired_effect(d).estimate)
    assert float(np.mean(ests)) == pytest.approx(truth, abs=0.006)


def test_independent_cell_interval_covers_nominally() -> None:
    rng = np.random.default_rng(77)
    truth = 0.0
    hits = 0
    trials = 400
    for _ in range(trials):
        d = rng.normal(truth, 0.2, 60)
        e = paired_effect(d, alpha=0.05)
        hits += int(e.ci_low <= truth <= e.ci_high)
    assert 0.92 <= hits / trials <= 0.99


# ----------------------------------------------------------------------- CUPED

def test_ancova_reduces_variance_by_rho_squared(rho: float = 0.6) -> None:
    rng = np.random.default_rng(0)
    n = 2000
    sd = 1.0
    x = rng.normal(0, sd, n)
    d = 0.05 + rho * x + math.sqrt(1 - rho ** 2) * sd * rng.normal(size=n)
    a = cuped_adjust(d, x)
    assert a.rho == pytest.approx(rho, abs=0.03)
    assert a.variance_reduction == pytest.approx(rho ** 2, abs=0.04)
    assert a.method == "ancova"
    assert a.estimate == pytest.approx(0.05, abs=0.02)


def test_cuped_with_known_mean_matches_analytic_vrf() -> None:
    rng = np.random.default_rng(1)
    n, rho, sd_d, sd_x = 4000, 0.5, 0.3, 0.25
    x = rng.normal(0, sd_x, n)
    d = 0.02 + rho * (sd_d / sd_x) * x + math.sqrt(1 - rho ** 2) * sd_d * rng.normal(size=n)
    a = cuped_adjust(d, x, mu_x=0.0)
    expected = rho ** 2 * (2 * sd_d * sd_x - sd_x ** 2) / sd_d ** 2
    assert a.variance_reduction == pytest.approx(expected, abs=0.06)
    assert a.estimate == pytest.approx(0.02, abs=0.02)


def test_ancova_is_a_noop_on_uncorrelated_covariate() -> None:
    rng = np.random.default_rng(2)
    d = rng.normal(0.05, 0.2, 300)
    x = rng.normal(0.5, 0.5, 300)
    a = cuped_adjust(d, x)
    assert a.variance_reduction == pytest.approx(0.0, abs=0.08)


# ----------------------------------------------------------------- rank flips

def test_rank_stability_detects_fragile_leaderboard() -> None:
    rng = np.random.default_rng(3)
    cells = {"A": rng.normal(0.50, 0.02, (20, 3)), "B": rng.normal(0.503, 0.02, (20, 3))}
    r = rank_stability({k: v.mean(axis=1) for k, v in cells.items()}, n_boot=2000)
    assert r["p_top1_changes"] > 0.10
    assert r["p_top1_unchanged"] < 0.95


def test_rank_stability_confident_when_gap_is_large() -> None:
    rng = np.random.default_rng(4)
    cells = {"A": rng.normal(0.50, 0.005, (60, 3)), "B": rng.normal(0.62, 0.005, (60, 3))}
    r = rank_stability({k: v.mean(axis=1) for k, v in cells.items()}, n_boot=2000)
    assert r["p_top1_unchanged"] > 0.95


def test_cell_index_order_matches_the_values_it_labels() -> None:
    """`cell_index` has the judge axis varying fastest; the data must agree.

    Stacking judges and raveling puts the *seed* axis fastest, which permutes the
    seed and judge labels of every cell. Nothing downstream crashes -- the arrays
    are the right length and the numbers look reasonable -- but every variance
    component is then attributed to the wrong level.
    """
    nt, ns = 2, 3
    jnames = ["J0", "J1", "J2"]
    arm, base = "A", "B"
    # distinct, asymmetric value per (task, seed, judge) so any transposition shows
    code = lambda t, s, j: (t * 100 + s * 10 + j) / 1000.0  # noqa: E731
    judged = {
        f"{arm}|{jnames[j]}": np.array([[code(t, s, j) for s in range(ns)]
                                         for t in range(nt)])
        for j in range(len(jnames))
    }
    for j in range(len(jnames)):
        judged[f"{base}|{jnames[j]}"] = np.zeros((nt, ns))
    suite = {
        "agents": [arm, base], "judges": jnames, "judged": judged, "raw": {},
        "cell_index": CellIndex(
            task=np.repeat(np.arange(nt), ns * len(jnames)),
            seed=np.tile(np.repeat(np.arange(ns), len(jnames)), nt),
            judge=np.tile(np.arange(len(jnames)), nt * ns),
        ),
    }
    d, idx, _ = field_with_sign(suite, arm, base)

    assert list(zip(idx.task.tolist(), idx.seed.tolist(), idx.judge.tolist())) == \
        [(t, s, j) for t in range(nt) for s in range(ns) for j in range(len(jnames))]
    assert d.tolist() == pytest.approx(
        [code(t, s, j) for t, s, j in zip(idx.task, idx.seed, idx.judge)])
    # guard: the seed<->judge transposition must actually differ for this fixture
    assert [code(t, j, s) for t, s, j in zip(idx.task, idx.seed, idx.judge)] \
        != d.tolist()


def test_variance_decomposition_is_invariant_to_judge_permutation() -> None:
    """Renaming/permuting the judges must not move the seed or task components.

    A judge permutation only relabels which judge contributed which offset, so the
    task and seed variance -- the two components that actually drive the standard
    error here -- have to come back identical. A scrambled cell order shows up
    immediately as a seed component that changes.
    """
    from agenteval.vcomp import moment_estimator

    nt, ns, nj = 24, 3, 3
    rng = np.random.default_rng(4)
    idx = CellIndex(
        task=np.repeat(np.arange(nt), ns * nj),
        seed=np.tile(np.repeat(np.arange(ns), nj), nt),
        judge=np.tile(np.arange(nj), nt * ns),
    )
    base = (0.03 * rng.normal(size=nt)[:, None, None]
            + 0.05 * rng.normal(size=ns)[None, :, None]
            + 0.02 * rng.normal(size=nj)[None, None, :]
            + 0.08 * rng.normal(size=(nt, ns, nj)))
    a = moment_estimator(base.ravel(), idx)
    perm = [2, 0, 1]
    p = np.eye(nj)[perm]
    b = moment_estimator((base @ p).ravel(), idx)
    assert b.sigma2_task == pytest.approx(a.sigma2_task, rel=0.05)
    assert b.sigma2_seed == pytest.approx(a.sigma2_seed, rel=0.05)
    assert b.sigma2_judge == pytest.approx(a.sigma2_judge, rel=0.05)


def test_replication_curve_improves_with_more_reps() -> None:
    """More replicates stabilise the top of the leaderboard, monotonically.

    The curve is a function of the *realised* gap between the two agents, so the
    fixture pins that gap into a range where one seed still flips the leader and
    the full data does not. Without pinning, an unlucky draw either saturates the
    curve at 1.0 or produces a degenerate one where the top never changes.
    """
    cells = None
    for seed in range(400):
        rng = np.random.default_rng(seed)
        trial = {"A": rng.normal(0.50, 0.08, (60, 12)),
                 "B": rng.normal(0.50, 0.08, (60, 12)) + 0.012}
        gap = float(trial["B"].mean() - trial["A"].mean())
        if 0.008 < gap < 0.016:
            cells = trial
            break
    assert cells is not None, "no seed produced a usable leaderboard gap"

    curve = replication_curve(cells, reps=(1, 2, 4, 8), n_trials=1500)
    assert len(curve) == 4
    assert curve[-1]["p_top1_unchanged"] >= curve[0]["p_top1_unchanged"]
    assert curve[0]["p_top1_unchanged"] < curve[-1]["p_top1_unchanged"]
    assert curve[0]["p_top1_unchanged"] < 0.98
    # margins shrink towards the true gap as replicates are added
    assert curve[-1]["expected_margin_over_runner_up"] < curve[0][
        "expected_margin_over_runner_up"]


# --------------------------------------------------------------------- design

def test_equal_cell_counts_do_not_give_equal_precision() -> None:
    """The split matters: cells, seeds and judges are not interchangeable.

    48 cells spent on tasks is *worse* than 48 cells spread over the three axes,
    because with a single seed and a single judge their main effects enter the
    grand mean undiluted and dominate everything the tasks contribute.
    """
    vc = VarianceComponents(0.04, 0.006, 0.009, 0.012, "truth")
    a = Design(48, 1, 1)
    b = Design(4, 4, 3)
    assert a.n_cells == b.n_cells
    se_a, se_b = a.standard_error(vc), b.standard_error(vc)
    assert se_a != pytest.approx(se_b)
    assert se_b < se_a
    # and both are far above the textbook s2_total / n_cells
    naive = math.sqrt(vc.total / 48)
    assert se_a > 3.0 * naive
    assert se_b > 1.5 * naive


def test_allocation_trades_seed_noise_for_cheap_replicates() -> None:
    """n_k ~ sqrt(s2_k / c_k): buy replicates on whichever axis is noisy and cheap."""
    vc = VarianceComponents(0.04, 0.0004, 0.009, 0.012, "truth")
    costs = {"task": 1.0, "seed": 0.02, "judge": 0.10}
    d = Design(20, 1, 1).integer_optimum(60.0, costs, vc=vc)
    assert d.cost(costs) <= 60.0 + 1e-6
    # tiny seed noise means seeds are not worth buying
    assert d.n_seeds < d.n_tasks
    assert d.n_judges >= 2

    noisy_seed = VarianceComponents(0.04, 0.05, 0.009, 0.012, "truth")
    d2 = Design(20, 1, 1).integer_optimum(60.0, costs, vc=noisy_seed)
    assert d2.n_seeds > d.n_seeds
    assert crossed_variance(noisy_seed, *d2.sizes)["se"] <= \
        crossed_variance(noisy_seed, *d.sizes)["se"] + 1e-12


def test_min_cost_design_reaches_target_power() -> None:
    vc = VarianceComponents(0.04, 0.006, 0.009, 0.012, "truth")
    d = min_cost_design(delta=0.08, vc=vc, target_power=0.8)
    assert d.power(0.08, vc) >= 0.80
    assert d.power(0.08, vc) < 0.92
    assert d.n_seeds >= 2 and d.n_judges >= 2
    cheap = min_cost_design(delta=0.20, vc=vc, target_power=0.8)
    assert cheap.cost() < d.cost()


def test_detectable_effect_has_a_floor_from_the_replicate_axes() -> None:
    """No amount of task authoring resolves an effect below the seed/judge floor."""
    vc = VarianceComponents(0.04, 0.006, 0.009, 0.012, "truth")
    assert min_detectable_delta(vc, 2, 2) > min_detectable_delta(vc, 8, 8) \
        > min_detectable_delta(vc, 64, 64)
    floor = min_detectable_delta(vc, 64, 64)
    huge = Design(1_000_000, 64, 64)
    assert huge.power(floor, vc) < 0.8
    assert huge.power(3.0 * floor, vc) > 0.8
    with pytest.raises(ValueError, match="floor"):
        min_cost_design(delta=1e-4, vc=vc, target_power=0.8)


def test_required_cells_matches_power_at_that_size() -> None:
    """Valid only for a single-level design, where every term shares the denominator."""
    vc = VarianceComponents(0.04, 0.006, 0.009, 0.012, "truth")
    flat = VarianceComponents(0.067, 0.0, 0.0, 0.0, "flat")
    delta = 0.03
    n = required_cells(delta, flat.total, 0.05, 0.8)
    d = Design(int(round(n)), 1, 1)
    assert d.power(delta, flat, 0.05) == pytest.approx(0.8, abs=0.02)
    # the same cell count is not enough once the components split across axes
    assert Design(int(round(n)), 1, 1).power(delta, vc, 0.05) < 0.8


def test_component_precision_needs_more_levels() -> None:
    vc = VarianceComponents(0.04, 0.006, 0.009, 0.012, "truth")
    small = Design(6, 3, 3).component_relative_se(vc)["task"]
    large = Design(200, 3, 3).component_relative_se(vc)["task"]
    assert large < small


# ---------------------------------------------------------------------- judges

def test_dawid_skene_recovers_confusion_matrix() -> None:
    rng = np.random.default_rng(11)
    n_items, s1, s0 = 4000, 0.93, 0.90
    truth = rng.random(n_items) < 0.45
    labels = []
    for (q1, q0) in ((s1, s0), (0.84, 0.79), (0.90, 0.93)):
        col = []
        for t in truth:
            col.append(int(rng.random() < (q1 if t else 1.0 - q0)))
        labels.append(col)
    L = [[labels[j][i] for j in range(3)] for i in range(n_items)]
    res = dawid_skene(L, n_iter=300)
    for j, (q1, q0) in enumerate(((s1, s0), (0.84, 0.79), (0.90, 0.93))):
        q1_hat, q0_hat = sensitivity_specificity(res.confusion[str(j)])
        assert q1_hat == pytest.approx(q1, abs=0.04)
        assert q0_hat == pytest.approx(q0, abs=0.04)
    assert res.latent_truth.mean() == pytest.approx(truth.mean(), abs=0.03)


def test_dawid_skene_is_not_lured_into_the_perfect_rater_mode() -> None:
    """The degenerate fit has perfect raters and no error; restarts must avoid it."""
    rng = np.random.default_rng(11)
    n_items = 3000
    truth = rng.random(n_items) < 0.45
    labels = []
    for (q1, q0) in ((0.93, 0.90), (0.84, 0.79), (0.90, 0.93)):
        labels.append([int(rng.random() < (q1 if t else 1.0 - q0)) for t in truth])
    L = [[labels[j][i] for j in range(3)] for i in range(n_items)]
    res = dawid_skene(L, n_iter=300)
    for j in range(3):
        q1_hat, q0_hat = sensitivity_specificity(res.confusion[str(j)])
        assert 0.5 < q1_hat < 0.99
        assert 0.5 < q0_hat < 0.99


def test_bias_correction_inverts_operating_point() -> None:
    theta = 0.42
    for q1, q0 in ((0.93, 0.90), (0.80, 0.85), (0.70, 0.60)):
        p_hat = theta * q1 + (1 - theta) * (1 - q0)
        assert bias_corrected_score(p_hat, q1, q0) == pytest.approx(theta, abs=1e-9)


def test_raw_agreement_overstates_and_kappa_corrects() -> None:
    rng = np.random.default_rng(12)
    n = 6000
    truth = rng.random(n) < 0.5
    biased = np.where(truth, rng.random(n) < 0.90, rng.random(n) < 0.20)
    raw = float((biased == truth).mean())
    kappa = cohen_kappa(truth, biased)
    assert raw > 0.8
    assert kappa < raw
    assert kappa < 0.8
    assert gwet_ac1(truth, biased) < raw


def test_judge_interval_widens_with_weaker_calibration() -> None:
    strong = judge_score_ci(0.5, 0.95, 0.95, 1000, 500)
    weak = judge_score_ci(0.5, 0.70, 0.70, 1000, 100)
    assert weak["se"] > strong["se"]
    assert strong["theta"] == pytest.approx(weak["theta"], abs=0.05)


def test_unpaired_and_paired_agree_in_estimate() -> None:
    rng = np.random.default_rng(13)
    a = rng.normal(0.5, 0.2, 200)
    b = rng.normal(0.45, 0.2, 200)
    assert paired_effect(a - b).estimate == pytest.approx(
        unpaired_effect(a, b).estimate, abs=1e-12)

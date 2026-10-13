"""LLM judges as measurement instruments.

A judge is not ground truth; it is a noisy, biased instrument. This module
quantifies that rather than assuming it away:

* :func:`dawid_skene` recovers a latent truth and each judge's confusion matrix
  from unlabeled comparison data (EM, no annotations required).
* :func:`bias_corrected_score` inverts the judge's sensitivity/specificity to
  estimate the true pass rate, instead of reporting the raw agreement rate.
* :func:`judge_score_ci` propagates uncertainty from *both* the test set and
  the calibration set into the interval.
* :func:`cohen_kappa` / :func:`gwet_ac1` correct agreement for chance, which
  raw exact-match agreement overstates.
* :func:`position_bias` uses swapped-order AB/BA comparisons.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np

__all__ = [
    "DawidSkeneResult",
    "dawid_skene",
    "bias_corrected_score",
    "judge_score_ci",
    "cohen_kappa",
    "gwet_ac1",
    "position_bias",
    "sensitivity_specificity",
    "judge_summary",
]


@dataclass(frozen=True)
class DawidSkeneResult:
    latent_truth: np.ndarray
    posteriors: np.ndarray
    confusion: dict[str, np.ndarray]
    loglik: float
    n_iter: int

    def to_dict(self) -> dict:
        return {
            "latent_truth": self.latent_truth.astype(int).tolist(),
            "n_items": int(self.latent_truth.size),
            "confusion": {k: v.tolist() for k, v in self.confusion.items()},
            "loglik": self.loglik,
            "n_iter": self.n_iter,
        }


def _init_confusion(labels: np.ndarray, n_classes: int = 2,
                    strength: float = 0.6) -> np.ndarray:
    """Diagonal-dominant starting confusion matrix, tilted by observed rates.

    ``strength`` controls how confident the start claims each rater to be. The
    degenerate "every rater is perfect" fixed point is a real local optimum of
    Dawid-Skene, so the caller retries from several strengths and keeps the
    highest-likelihood fit.
    """
    conf = np.full((n_classes, n_classes), (1.0 - strength) / (n_classes - 1))
    conf[np.arange(n_classes), np.arange(n_classes)] = strength
    counts = np.zeros((n_classes, n_classes))
    for r in np.asarray(labels).ravel():
        if r >= 0:
            counts[int(r), :] += 1.0
    tot = counts.sum(axis=1, keepdims=True)
    where = np.broadcast_to(tot > 0, conf.shape)
    tilt = counts / np.maximum(tot, 1e-12)
    conf[where] = 0.75 * conf[where] + 0.25 * tilt[where]
    conf = np.clip(conf, 1e-3, 1 - 1e-3)
    return conf / conf.sum(axis=1, keepdims=True)


def _em(L: np.ndarray, n_classes: int, conf0: np.ndarray, n_iter: int,
        tol: float, prior0: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                  float, int]:
    n_items, n_raters = L.shape
    conf = conf0.copy()
    prior = prior0.copy()
    prev = -np.inf
    post = np.tile(prior, (n_items, 1))
    it = 0

    complete = bool((L >= 0).all())
    if complete:
        obs_all = list(L.T)
        mask_all = [np.ones(n_items, dtype=bool)] * n_raters
    else:
        obs_all = [L[:, j] for j in range(n_raters)]
        mask_all = [o >= 0 for o in obs_all]

    for it in range(1, n_iter + 1):
        log_prior = np.log(np.clip(prior, 1e-12, None))
        logp = np.tile(log_prior, (n_items, 1))
        for j in range(n_raters):
            if complete:
                logp += np.log(np.clip(conf[j], 1e-12, None))[obs_all[j], :]
            else:
                m = mask_all[j]
                if m.any():
                    logp[m] += np.log(np.clip(conf[j], 1e-12, None))[obs_all[j][m], :]
        mx = logp.max(axis=1, keepdims=True)
        w = np.exp(logp - mx)
        denom_w = w.sum(axis=1, keepdims=True)
        post = w / denom_w

        prior = np.clip(post.mean(axis=0), 1e-9, None)
        prior = prior / prior.sum()

        for j in range(n_raters):
            if complete:
                o = obs_all[j]
                num = np.empty((n_classes, n_classes))
                for c in range(n_classes):
                    num[c] = np.bincount(o, weights=post[:, c], minlength=n_classes)
            else:
                m = mask_all[j]
                if not m.any():
                    continue
                o = obs_all[j][m]
                num = np.empty((n_classes, n_classes))
                for c in range(n_classes):
                    num[c] = np.bincount(o, weights=post[m, c], minlength=n_classes)
            tot = num.sum(axis=1, keepdims=True)
            conf[j] = np.where(tot > 0, num / np.maximum(tot, 1e-12), 1.0 / n_classes)

        # E[log p(data, latent)] collapses to sum_i sum_k post_ik * logp_ik plus the
        # prior term, so one reduction covers all raters at once.
        ll = float(np.sum(post * logp)) + float(n_items * np.sum(log_prior * prior))
        # relative tolerance: an absolute 1e-7 on a log-likelihood of order 1e4 is
        # below double precision, so the loop would never trigger and every restart
        # would run to maxiter.
        if abs(ll - prev) <= tol * (1.0 + abs(ll)):
            break
        prev = ll

    return post, conf, prior, float(prev), int(it)


def _majority_prior(L: np.ndarray, n_classes: int) -> np.ndarray:
    """Prior from per-item majority vote; breaks the "split by pattern" mode."""
    n_items, n_raters = L.shape
    votes = np.full((n_items, n_classes), -1.0)
    for j in range(n_raters):
        obs = L[:, j]
        m = obs >= 0
        votes[np.where(m)[0], obs[m]] += 1.0
    unvoted = (votes < 0).all(axis=1)
    votes[votes < 0] = 0.0
    prior = np.clip(votes.sum(axis=0), 0.0, None)
    if prior.sum() <= 0:
        prior = np.full(n_classes, 1.0 / n_classes)
    prior = prior / prior.sum()
    if unvoted.all():
        return np.full(n_classes, 1.0 / n_classes)
    return prior


def _align_labels(conf: np.ndarray, prior: np.ndarray, post: np.ndarray,
                  n_classes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve the class-permutation ambiguity by maximising diagonal mass.

    The Dawid-Skene likelihood is exactly invariant under renaming the latent
    classes, so an equally good fit can be returned with the classes swapped -
    which in binary form means every rater looking systematically *inverted*
    instead of reporting a sensitivity of 0.9. Renaming class ``k`` to ``pi(k)``
    permutes the rows of each confusion matrix, and the fit is relabelled to the
    permutation with the highest total diagonal mass, i.e. the one where the
    latent classes are the ones the raters actually agree on. The observed labels
    are held fixed, so this is a relabelling and not a change of model.
    """
    if n_classes <= 1:
        return conf, prior, post
    best_perm, best_score = None, -np.inf
    for perm in itertools.permutations(range(n_classes)):
        idx = np.asarray(perm)
        score = float(np.mean(np.trace(conf[:, idx, :], axis1=1, axis2=2)))
        if score > best_score:
            best_perm, best_score = idx, score
    assert best_perm is not None
    return conf[:, best_perm, :], prior[best_perm], post[:, best_perm]


def dawid_skene(labels: Sequence[Sequence[int]], n_classes: int = 2,
                n_iter: int = 60, tol: float = 1e-7,
                strengths: Sequence[float] = (0.7, 0.55)) -> DawidSkeneResult:
    """EM for Dawid-Skene: latent class labels plus per-rater confusion matrices.

    ``labels[i][j]`` is rater ``j``'s label for item ``i``; use ``-1`` for a
    missing judgment. Returns posterior probabilities per item, so downstream
    code can weight by confidence instead of hard-thresholding.

    The likelihood is multimodal: alongside the true operating point there is a
    fixed point where raters are modelled as perfect and disagreement is
    explained by an extra latent class. Starting from a single uniform
    initialisation frequently lands there and reports a sensitivity of 1.0, so
    the fit is restarted from several diagonal strengths plus a majority-vote
    prior and the highest-likelihood result is returned.
    """
    L = np.full((len(labels), len(labels[0])), -1, dtype=int)
    for i, row in enumerate(labels):
        for j, v in enumerate(row):
            L[i, j] = int(v)
    n_items, n_raters = L.shape
    maj = _majority_prior(L, n_classes)

    best: tuple[np.ndarray, np.ndarray, np.ndarray, float, int] | None = None
    for strength in strengths:
        conf0 = np.stack([_init_confusion(L[:, j], n_classes, strength)
                          for j in range(n_raters)])
        for prior0 in (np.full(n_classes, 1.0 / n_classes), maj):
            fit = _em(L, n_classes, conf0, n_iter, tol, prior0)
            if best is None or fit[3] > best[3]:
                best = fit
    assert best is not None
    post, conf, prior, ll, it = best
    conf, prior, post = _align_labels(conf, prior, post, n_classes)

    names = [str(j) for j in range(n_raters)]
    return DawidSkeneResult(post.argmax(axis=1), post,
                            {names[j]: conf[j] for j in range(n_raters)},
                            float(ll), int(it))


def sensitivity_specificity(conf: np.ndarray) -> tuple[float, float]:
    """Binary confusion matrix (rows = truth, cols = judge) -> (q1, q0)."""
    conf = np.asarray(conf, dtype=float)
    if conf.shape != (2, 2):
        raise ValueError("binary confusion matrix must be 2x2")
    return float(conf[1, 1]), float(conf[0, 0])


def bias_corrected_score(p_hat: float, sensitivity: float, specificity: float) -> float:
    """Invert the judge's operating point to estimate the true pass rate.

    ``p_hat = theta (q1 + q0 - 1) + (1 - q0)`` inverts to
    ``theta = (p_hat - (1 - q0)) / (q1 + q0 - 1)``. Clipped to [0, 1] because
    the inversion is unstable when ``q1 + q0`` approaches 1 (a near-useless
    judge), which is reported via ``extra['reliability']``.
    """
    denom = sensitivity + specificity - 1.0
    if abs(denom) < 1e-6:
        return float("nan")
    theta = (p_hat - (1.0 - specificity)) / denom
    return float(np.clip(theta, 0.0, 1.0))


def judge_score_ci(p_hat: float, sensitivity: float, specificity: float,
                   n_test: int, n_cal: int, alpha: float = 0.05) -> dict:
    """Interval on the bias-corrected score, carrying both sources of variance.

    The test set contributes ``p_hat(1 - p_hat) / n_test``; the calibration set
    contributes the uncertainty in ``q0`` and ``q1``, weighted by the estimated
    score, following the two-sample rate-difference decomposition.
    """
    from scipy.stats import norm
    theta = bias_corrected_score(p_hat, sensitivity, specificity)
    if not np.isfinite(theta):
        return {"theta": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "se": float("nan")}
    z = float(norm.ppf(1.0 - alpha / 2.0))
    p_use = float(np.clip(p_hat, 0.0, 1.0))
    var_test = p_use * (1.0 - p_use) / max(n_test, 1)
    if n_cal > 0:
        q0v = specificity * (1.0 - specificity) / n_cal
        q1v = sensitivity * (1.0 - sensitivity) / n_cal
        var_cal = (1.0 - theta) ** 2 * q0v + theta ** 2 * q1v
    else:
        var_cal = 0.0
    se = math.sqrt(max(var_test + var_cal, 0.0))
    return {"theta": theta, "se": se, "ci_low": theta - z * se,
            "ci_high": theta + z * se, "var_test": var_test, "var_calibration": var_cal,
            "n_test": n_test, "n_cal": n_cal}


def cohen_kappa(a: Sequence[int], b: Sequence[int], n_classes: int = 2) -> float:
    """Chance-corrected agreement. Raw agreement flatters a biased judge."""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    ok = (a >= 0) & (b >= 0)
    a, b = a[ok], b[ok]
    if a.size == 0:
        return float("nan")
    po = float((a == b).mean())
    pe = 0.0
    for c in range(n_classes):
        pe += float((a == c).mean() * (b == c).mean())
    return float((po - pe) / (1.0 - pe)) if pe < 1.0 else float("nan")


def gwet_ac1(a: Sequence[int], b: Sequence[int], n_classes: int = 2) -> float:
    """Gwet's AC1: chance-corrected agreement that stays stable under skew."""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    ok = (a >= 0) & (b >= 0)
    a, b = a[ok], b[ok]
    if a.size == 0:
        return float("nan")
    po = float((a == b).mean())
    pi = np.array([float((a == c).mean() * (b == c).mean()) for c in range(n_classes)])
    pe_bar = float(pi.sum())
    k = (n_classes - 1.0) / (n_classes - 2.0) if n_classes > 2 else 1.0
    pe = k * pe_bar
    return float((po - pe) / (1.0 - pe)) if pe < 1.0 else float("nan")


def position_bias(a_first: Sequence[int], b_second: Sequence[int],
                  a_swapped: Sequence[int], b_swapped: Sequence[int]) -> dict:
    """Position bias from paired AB / BA presentations.

    ``a_first/b_second`` are the labels when A was shown first, and
    ``a_swapped/b_swapped`` the same items with the order reversed. A judge with
    no position bias keeps its verdicts under the swap.
    """
    a1 = np.asarray(a_first, dtype=float)
    b1 = np.asarray(b_second, dtype=float)
    a2 = np.asarray(a_swapped, dtype=float)
    b2 = np.asarray(b_swapped, dtype=float)
    n = min(a1.size, b1.size, a2.size, b2.size)
    a1, b1, a2, b2 = a1[:n], b1[:n], a2[:n], b2[:n]
    p_a_first = float((a1 > b1).mean() + 0.5 * (a1 == b1).mean())
    p_a_second = float((a2 > b2).mean() + 0.5 * (a2 == b2).mean())
    return {
        "p_a_wins_when_shown_first": p_a_first,
        "p_a_wins_when_shown_second": p_a_second,
        "position_bias": float(p_a_first - p_a_second),
        "abs_position_bias": float(abs(p_a_first - p_a_second)),
        "swap_consistency": float(((a1 == a2) & (b1 == b2)).mean()),
        "n": int(n),
    }


def judge_summary(labels: Sequence[Sequence[int]], truth: Sequence[int] | None = None,
                  n_classes: int = 2) -> dict:
    """Full reliability card for a panel of judges."""
    L = np.asarray(labels, dtype=int)
    n_raters = L.shape[1]
    ds = dawid_skene(L.tolist(), n_classes=n_classes)
    out: dict = {"n_items": int(L.shape[0]), "dawid_skene": ds.to_dict(), "judges": {}}

    latent = ds.latent_truth if truth is None else np.asarray(truth, dtype=int)
    for j in range(n_raters):
        obs = L[:, j]
        m = obs >= 0
        if m.sum() < 2:
            continue
        entry: dict = {"n_judged": int(m.sum()),
                       "raw_agreement_with_latent": float((obs[m] == latent[m]).mean())}
        if truth is not None:
            t = np.asarray(truth, dtype=int)[m]
            entry["raw_agreement_with_truth"] = float((obs[m] == t).mean())
            if n_classes == 2:
                entry["cohen_kappa_vs_truth"] = cohen_kappa(t, obs[m], n_classes)
                entry["gwet_ac1_vs_truth"] = gwet_ac1(t, obs[m], n_classes)
        if n_classes == 2:
            q1, q0 = sensitivity_specificity(ds.confusion[str(j)])
            entry["sensitivity"] = q1
            entry["specificity"] = q0
            entry["bias"] = float(0.5 * (1.0 - q1) - 0.5 * (1.0 - q0))
            entry["reliability"] = float(q1 + q0 - 1.0)
        out["judges"][str(j)] = entry
    return out

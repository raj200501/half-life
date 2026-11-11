"""Tests for :mod:`halflife.stats` against values derived by hand or by independent code.

Every expected number here is either worked out by hand (the arithmetic is in the
comments) or recomputed in the test by a separate implementation (numpy's
percentile, a finite-difference Hessian, a replayed random generator). The
simulated datasets are synthetic test fixtures, not model results.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pytest

from halflife import analysis, stats
from halflife.stats import (
    HazardFit,
    Interval,
    KaplanMeier,
    NotReached,
    StatsError,
    bootstrap_coefficients,
    bootstrap_half_life,
    bootstrap_hazard_curve,
    cluster_bootstrap,
    cluster_bootstrap_many,
    describe_interval,
    fit_hazard,
    fit_logistic,
    half_life,
    kaplan_meier,
    kaplan_meier_rows,
    normal_cdf,
    normal_quantile,
    power_by_simulation,
    rate_contrast,
    recall_by_compaction,
    recall_effect,
    recall_rows,
    risk_set,
    wilson_interval,
)

Z95 = 1.959963984540054  # the constant tools/ship_check.py uses
LOG3 = math.log(3.0)


def _json_roundtrip(value: dict[str, Any]) -> dict[str, Any]:
    """Serialize strictly (no NaN or Infinity) and parse back."""
    parsed = json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    assert isinstance(parsed, dict)
    return parsed


def _num(value: stats.Value) -> float:
    """A statistic value that must be a number here (not a NotReached bound)."""
    assert not isinstance(value, NotReached), value
    return float(value)


def _simulated_hazard_rows(
    n_sessions: int,
    seed: int,
    intercept: float = -2.5,
    slope: float = 0.02,
    compaction: float = 0.4,
    type_effects: dict[str, float] | None = None,
    per_session: int = 4,
) -> list[dict[str, Any]]:
    """Synthetic opportunity rows from a known per-opportunity logistic hazard."""
    effects = type_effects or {"prohibition": 0.0}
    names = sorted(effects)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for s in range(n_sessions):
        positions = np.sort(rng.choice(151, size=per_session, replace=False))
        n_events = int(rng.integers(0, 3))
        events = np.sort(rng.choice(np.arange(1, 151), size=n_events, replace=False))
        for p in positions:
            count = int((events <= p).sum())
            ctype = names[int(rng.integers(0, len(names)))]
            eta = intercept + slope * float(p) + compaction * count + effects[ctype]
            rows.append(
                {
                    "family": "syn",
                    "session_id": f"s{s:04d}",
                    "position": int(p),
                    "compaction_count": count,
                    "constraint_type": ctype,
                    "violated": bool(rng.random() < 1.0 / (1.0 + math.exp(-eta))),
                }
            )
    return rows


def _two_family_rows(
    n_plans: int, seed: int, intercept: float = -2.0, slope: float = 0.6
) -> list[dict[str, Any]]:
    """Two synthetic families run the same plans, so every session id appears twice.

    Positions and outcomes are drawn per family, from a known log1p(position) hazard;
    the compaction count rises every 30 tool calls. Not model data.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for family in ("fa", "fb"):
        for plan in range(n_plans):
            positions = np.sort(rng.choice(np.arange(1, 121), size=5, replace=False))
            for p in positions:
                eta = intercept + slope * math.log1p(float(p)) - 0.3 * (plan % 2)
                rows.append(
                    {
                        "family": family,
                        "session_id": f"p{plan:03d}",
                        "arm": "b" if plan % 2 else "a",
                        "position": int(p),
                        "compaction_count": int(p) // 30,
                        "violated": bool(rng.random() < 1.0 / (1.0 + math.exp(-eta))),
                        "recalled_before": bool(rng.random() < 0.5),
                    }
                )
    return rows


def _manual_risk_set(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each (family, session_id) in position order up to its first violation (by hand)."""
    out: list[dict[str, Any]] = []
    for key in sorted({(r["family"], r["session_id"]) for r in rows}):
        mine = sorted(
            (r for r in rows if (r["family"], r["session_id"]) == key),
            key=lambda r: r["position"],
        )
        for row in mine:
            out.append(dict(row))
            if row["violated"]:
                break
    return out


# --------------------------------------------------------------------------- normal


def test_normal_quantile_matches_the_ship_check_constant() -> None:
    assert normal_quantile(0.975) == pytest.approx(Z95, abs=1e-15)
    assert normal_quantile(0.5) == 0.0
    # 90% two-sided critical value, a textbook constant.
    assert normal_quantile(0.95) == pytest.approx(1.6448536269514722, abs=1e-14)


@pytest.mark.parametrize("p", [1e-12, 1e-4, 0.01, 0.02425, 0.1, 0.5, 0.9, 0.97575, 0.999, 1 - 1e-9])
def test_normal_quantile_inverts_normal_cdf(p: float) -> None:
    x = normal_quantile(p)
    assert normal_cdf(x) == pytest.approx(p, rel=1e-12)


@pytest.mark.parametrize("p", [1e-4, 0.01, 0.02425, 0.1, 0.3])
def test_normal_quantile_is_symmetric(p: float) -> None:
    # Only where 1 - p is representable closely enough for the comparison to mean something.
    assert normal_quantile(1.0 - p) == pytest.approx(-normal_quantile(p), rel=1e-9)


def test_normal_quantile_deep_tail_does_not_overflow() -> None:
    x = normal_quantile(1e-310)
    assert math.isfinite(x) and x < -37.0
    assert normal_quantile(5e-324) < x


@pytest.mark.parametrize("p", [0.0, 1.0, -0.1, 1.5, float("nan"), True])
def test_normal_quantile_rejects_values_outside_the_open_unit_interval(p: object) -> None:
    with pytest.raises(StatsError):
        normal_quantile(p)  # type: ignore[arg-type]  # invalid input on purpose


def test_normal_cdf_known_values() -> None:
    assert normal_cdf(0.0) == 0.5
    assert normal_cdf(Z95) == pytest.approx(0.975, abs=1e-15)
    assert normal_cdf(-Z95) == pytest.approx(0.025, abs=1e-15)


# --------------------------------------------------------------------------- Wilson


def _wilson_reference(k: int, n: int, z: float) -> tuple[float, float]:
    """The textbook Wilson score formula, written out independently of the module."""
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@pytest.mark.parametrize(("k", "n"), [(0, 1), (1, 1), (3, 10), (19, 20), (50, 100), (7, 1000)])
def test_wilson_interval_matches_the_score_formula(k: int, n: int) -> None:
    result = wilson_interval(k, n)
    lo, hi = _wilson_reference(k, n, Z95)
    assert result.value == k / n
    assert result.lo == pytest.approx(lo, abs=1e-12)
    assert result.hi == pytest.approx(hi, abs=1e-12)
    assert (result.k, result.n, result.level) == (k, n, 0.95)


def test_wilson_interval_closed_forms_at_zero_and_all() -> None:
    # k = 0: centre and half width coincide, so lo = 0 and hi = z^2 / (n + z^2).
    zero = wilson_interval(0, 10)
    assert zero.lo == 0.0
    assert zero.hi == pytest.approx(Z95**2 / (10 + Z95**2), abs=1e-12)
    # k = n mirrors it: lo = n / (n + z^2) and hi = 1.
    full = wilson_interval(10, 10)
    assert full.lo == pytest.approx(10 / (10 + Z95**2), abs=1e-12)
    assert full.hi == 1.0


def test_wilson_interval_other_level_and_numpy_counts() -> None:
    counts = (np.int64(3), np.int64(10))  # numpy integers are accepted at runtime
    narrow = wilson_interval(*counts, level=0.9)  # type: ignore[arg-type]  # numpy ints
    lo, hi = _wilson_reference(3, 10, 1.6448536269514722)
    assert (narrow.lo, narrow.hi) == pytest.approx((lo, hi), abs=1e-12)
    assert narrow.hi - narrow.lo < wilson_interval(3, 10).hi - wilson_interval(3, 10).lo
    assert _json_roundtrip(narrow.to_dict())["method"] == "wilson"


@pytest.mark.parametrize(
    ("k", "n", "level"),
    [(0, 0, 0.95), (5, 4, 0.95), (-1, 4, 0.95), (1.0, 4, 0.95), (True, 4, 0.95), (1, 4, 1.0)],
)
def test_wilson_interval_rejects_bad_input(k: object, n: object, level: float) -> None:
    with pytest.raises(StatsError):
        wilson_interval(k, n, level)  # type: ignore[arg-type]  # invalid input on purpose


# --------------------------------------------------------------------------- risk set


def test_risk_set_keeps_opportunities_up_to_the_first_violation() -> None:
    rows = [
        {"family": "f", "session_id": "b", "position": 5, "violated": False},
        {"family": "f", "session_id": "a", "position": 9, "violated": False, "tag": "tie-first"},
        {"family": "f", "session_id": "a", "position": 2, "violated": False},
        {"family": "f", "session_id": "a", "position": 9, "violated": True, "tag": "tie-second"},
        {"family": "f", "session_id": "a", "position": 12, "violated": True},
        {"family": "f", "session_id": "b", "position": 1, "violated": np.bool_(False)},
    ]
    out = risk_set(rows)
    assert [(r["session_id"], r["position"]) for r in out] == [
        ("a", 2),
        ("a", 9),
        ("a", 9),
        ("b", 1),
        ("b", 5),
    ]
    # Ties in position keep input order, so the violation at 9 is the last row of a.
    assert [r.get("tag") for r in out[1:3]] == ["tie-first", "tie-second"]
    out[0]["position"] = 99
    assert rows[2]["position"] == 2  # rows are copies


def test_risk_set_groups_by_other_fields_and_handles_empty_input() -> None:
    rows = [
        {"family": "f", "session_id": "s", "constraint_id": "c1", "position": 1, "violated": True},
        {"family": "f", "session_id": "s", "constraint_id": "c1", "position": 4, "violated": False},
        {"family": "f", "session_id": "s", "constraint_id": "c2", "position": 3, "violated": False},
    ]
    assert len(risk_set(rows)) == 1
    assert len(risk_set(rows, by=("session_id", "constraint_id"))) == 2
    assert risk_set([]) == []


def _shared_plan_rows() -> list[dict[str, Any]]:
    """Both synthetic families ran plans p0 and p1; family fa violates at its first
    decision point of p0, family fb at its second."""
    return [
        {"family": "fa", "session_id": "p0", "position": 3, "violated": True},
        {"family": "fa", "session_id": "p0", "position": 9, "violated": False},
        {"family": "fb", "session_id": "p0", "position": 4, "violated": False},
        {"family": "fb", "session_id": "p0", "position": 10, "violated": True},
        {"family": "fb", "session_id": "p0", "position": 15, "violated": False},
        {"family": "fa", "session_id": "p1", "position": 2, "violated": False},
        {"family": "fb", "session_id": "p1", "position": 2, "violated": False},
    ]


def test_risk_set_is_per_family_session_when_families_share_plans() -> None:
    # Family fa's early violation in plan p0 must not end family fb's session of p0.
    rows = _shared_plan_rows()
    assert stats.SESSION_KEY == ("family", "session_id") == analysis.SESSION_KEY
    out = [(r["family"], r["session_id"], r["position"]) for r in risk_set(rows)]
    assert out == [
        ("fa", "p0", 3),
        ("fa", "p1", 2),
        ("fb", "p0", 4),
        ("fb", "p0", 10),
        ("fb", "p1", 2),
    ]
    # Grouping by the plan alone merges the two sessions, which is what the default avoids.
    merged = [(r["family"], r["position"]) for r in risk_set(rows, by=("session_id",))]
    assert merged == [("fa", 3), ("fa", 2), ("fb", 2)]
    with pytest.raises(StatsError, match="no field 'family'"):
        risk_set([{"session_id": "p0", "position": 1, "violated": True}])
    with pytest.raises(StatsError, match="at least one field"):
        risk_set(rows, by=())


@pytest.mark.parametrize(
    "row",
    [
        {"family": "f", "position": 1, "violated": True},
        {"family": "f", "session_id": "s", "violated": True},
        {"family": "f", "session_id": "s", "position": 1, "violated": None},
        {"family": "f", "session_id": "s", "position": 1, "violated": 2},
        {"family": "f", "session_id": "s", "position": -1, "violated": True},
        {"family": "f", "session_id": "s", "position": "3", "violated": True},
        {"family": "f", "session_id": "s", "position": float("inf"), "violated": True},
    ],
)
def test_risk_set_rejects_malformed_rows(row: dict[str, Any]) -> None:
    with pytest.raises(StatsError):
        risk_set([row])


# --------------------------------------------------------------------------- logistic fits

# A 2x2 table: x = 0 has 2 events in 6, x = 1 has 4 events in 6. The maximum
# likelihood fit is saturated, so intercept = logit(1/3) = -log 2 and
# slope = logit(2/3) - logit(1/3) = 2 log 2. Var(intercept) = 1 / (6 * 1/3 * 2/3) = 3/4
# and Var(slope) = 3/4 + 3/4 = 3/2. The log-likelihood is
# 12 * (1/3 log 1/3 + 2/3 log 2/3).
TABLE_X = [0.0] * 6 + [1.0] * 6
TABLE_Y = [1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0]


def test_fit_logistic_matches_the_closed_form_two_by_two_table() -> None:
    fit = fit_logistic(TABLE_X, TABLE_Y, names=["arm"])
    assert fit.names == ("intercept", "arm")
    assert fit.coefficient("intercept") == pytest.approx(-math.log(2.0), abs=1e-5)
    assert fit.coefficient("arm") == pytest.approx(2.0 * math.log(2.0), abs=1e-5)
    assert fit.std_error("intercept") == pytest.approx(math.sqrt(0.75), abs=1e-5)
    assert fit.std_error("arm") == pytest.approx(math.sqrt(1.5), abs=1e-5)
    expected_ll = 12 * ((1 / 3) * math.log(1 / 3) + (2 / 3) * math.log(2 / 3))
    assert fit.loglik == pytest.approx(expected_ll, abs=1e-9)
    assert fit.converged and fit.iterations < 20
    assert fit.warnings == () and not fit.separation and fit.usable
    assert (fit.n, fit.n_events, fit.rank) == (12, 6, 2)
    assert fit.wald_z("arm") == pytest.approx(2.0 * math.log(2.0) / math.sqrt(1.5), abs=1e-5)
    assert fit.coefficients == {"intercept": fit.coef[0], "arm": fit.coef[1]}


def test_fit_logistic_cluster_robust_errors_in_the_saturated_case() -> None:
    # In a saturated model with one observation per cluster, the meat of the sandwich
    # equals the bread's inverse (sum of (y - p)^2 = n p (1 - p) in each cell), so the
    # CR1 errors are the model errors times sqrt(G / (G - 1)) with G = 12.
    fit = fit_logistic(TABLE_X, TABLE_Y, names=["arm"], clusters=list(range(12)))
    assert fit.n_clusters == 12 and fit.robust_se is not None
    factor = math.sqrt(12 / 11)
    for name in fit.names:
        assert fit.std_error(name, robust=True) == pytest.approx(
            fit.std_error(name) * factor, rel=1e-5
        )
    assert fit.wald_z("arm", robust=True) == pytest.approx(fit.wald_z("arm") / factor, rel=1e-5)


def test_fit_logistic_hessian_errors_match_a_finite_difference_hessian() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(300, 2))
    y = (rng.random(300) < 1 / (1 + np.exp(-(0.3 + x @ np.array([0.7, -0.4]))))).astype(float)
    fit = fit_logistic(x, y)
    design = np.column_stack([np.ones(300), x])

    def loglik(beta: np.ndarray) -> float:
        eta = design @ beta
        return float(np.sum(y * eta - np.logaddexp(0.0, eta)))

    beta = np.asarray(fit.coef)
    h = 1e-4
    hess = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            ei, ej = np.eye(3)[i] * h, np.eye(3)[j] * h
            hess[i, j] = (
                loglik(beta + ei + ej)
                - loglik(beta + ei - ej)
                - loglik(beta - ei + ej)
                + loglik(beta - ei - ej)
            ) / (4 * h * h)
    numeric_se = np.sqrt(np.diag(np.linalg.inv(-hess)))
    assert np.allclose(fit.se, numeric_se, rtol=1e-4)
    assert fit.loglik == pytest.approx(loglik(beta), abs=1e-9)
    # The estimate is a maximum: the score is zero up to the tiny ridge.
    score = design.T @ (y - 1 / (1 + np.exp(-(design @ beta))))
    assert np.allclose(score, fit.l2 * beta, atol=1e-6)


def test_fit_logistic_recovers_known_coefficients_on_a_large_sample() -> None:
    rng = np.random.default_rng(20261104)
    n = 40_000
    x = rng.normal(size=(n, 2))
    truth = np.array([-1.0, 0.8, -0.5])
    eta = truth[0] + x @ truth[1:]
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    fit = fit_logistic(x, y, names=["a", "b"])
    for name, value in zip(fit.names, truth, strict=True):
        assert abs(fit.coefficient(name) - value) < 4.0 * fit.std_error(name)
        assert abs(fit.coefficient(name) - value) < 0.05
    assert fit.usable


def test_fit_logistic_reports_complete_separation_without_crashing() -> None:
    fit = fit_logistic([0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1])
    assert fit.separation and "separation" in fit.warnings
    assert "x1" in fit.separated_terms
    assert fit.converged and np.all(np.isfinite(fit.coef)) and np.all(np.isfinite(fit.se))
    assert fit.coefficient("x1") > 10.0  # large but finite, held by the ridge
    assert not fit.usable


def test_fit_logistic_reports_quasi_complete_separation() -> None:
    fit = fit_logistic([0, 0, 1, 1, 2, 2], [0, 0, 0, 1, 1, 1])
    assert fit.separation and fit.converged and not fit.usable


@pytest.mark.parametrize(("outcome", "code"), [(0, "all_zero_outcomes"), (1, "all_one_outcomes")])
def test_fit_logistic_handles_constant_outcomes(outcome: int, code: str) -> None:
    fit = fit_logistic([0.0, 1.0, 2.0, 3.0], [outcome] * 4)
    assert code in fit.warnings and fit.separation
    assert fit.converged and np.all(np.isfinite(fit.coef))
    assert fit.n_events == 4 * outcome


def test_fit_logistic_overlapping_data_is_not_flagged() -> None:
    fit = fit_logistic([0, 1, 2, 3, 4, 5, 6, 7], [0, 0, 1, 0, 1, 0, 1, 1])
    assert not fit.separation and fit.usable and fit.separated_terms == ()


def test_fit_logistic_rank_deficiency_one_cluster_and_no_convergence() -> None:
    x = np.column_stack([TABLE_X, TABLE_X])
    collinear = fit_logistic(x, TABLE_Y, clusters=["only"] * 12)
    assert "rank_deficient" in collinear.warnings and collinear.rank == 2
    assert "one_cluster" in collinear.warnings and collinear.robust_se is None
    assert not collinear.usable
    with pytest.raises(StatsError, match="cluster-robust"):
        collinear.std_error("x1", robust=True)
    stopped = fit_logistic(TABLE_X, TABLE_Y, max_iter=1, check_separation=False)
    assert not stopped.converged and stopped.warnings == ("not_converged",)


def test_fit_logistic_without_intercept_and_read_only_results() -> None:
    fit = fit_logistic(np.column_stack([np.ones(12), TABLE_X]), TABLE_Y, intercept=False)
    assert fit.names == ("x1", "x2")
    assert fit.coefficient("x1") == pytest.approx(-math.log(2.0), abs=1e-5)
    with pytest.raises(ValueError):
        fit.coef[0] = 1.0
    with pytest.raises(StatsError, match="no term"):
        fit.coefficient("missing")
    record = _json_roundtrip(fit.to_dict())
    assert [t["name"] for t in record["terms"]] == ["x1", "x2"]
    assert "robust_se" not in record["terms"][0]
    clustered = _json_roundtrip(fit_logistic(TABLE_X, TABLE_Y, clusters=list(range(12))).to_dict())
    assert "robust_se" in clustered["terms"][0]


@pytest.mark.parametrize(
    ("x", "y", "kwargs"),
    [
        ([], [], {}),
        ([1.0, 2.0], [0], {}),
        ([1.0, float("nan")], [0, 1], {}),
        ([1.0, 2.0], [0, 2], {}),
        ([1.0, 2.0], [0, 1], {"l2": 0.0}),
        ([1.0, 2.0], [0, 1], {"l2": float("inf")}),
        ([1.0, 2.0], [0, 1], {"names": ["a", "b"]}),
        ([1.0, 2.0], [0, 1], {"names": ["intercept"]}),
        ([1.0, 2.0], [0, 1], {"clusters": ["a"]}),
        ([1.0, 2.0], [0, 1], {"tol": 0.0}),
        ([1.0, 2.0], [0, 1], {"max_iter": 0}),
        (np.zeros((2, 0)), [0, 1], {"intercept": False}),
        (np.zeros((2, 1, 1)), [0, 1], {}),
    ],
)
def test_fit_logistic_rejects_bad_input(x: Any, y: Any, kwargs: dict[str, Any]) -> None:
    with pytest.raises(StatsError):
        fit_logistic(x, y, **kwargs)


# --------------------------------------------------------------------------- hazard fits

# Saturated factor model, one opportunity per session: invariant 1/4, procedure 2/4,
# prohibition 3/4 violated. Sorted levels make invariant the reference, so
# intercept = logit(1/4) = -log 3, procedure = 0 - (-log 3) = log 3, and
# prohibition = log 3 - (-log 3) = 2 log 3.
FACTOR_COUNTS = {"invariant": 1, "procedure": 2, "prohibition": 3}


def _factor_rows() -> list[dict[str, Any]]:
    rows = []
    for ctype, k in FACTOR_COUNTS.items():
        for i in range(4):
            rows.append(
                {
                    "session_id": f"{ctype}-{i}",
                    "position": 10 * i,
                    "constraint_type": ctype,
                    "violated": i < k,
                }
            )
    return rows


def test_fit_hazard_saturated_factor_matches_cell_logits() -> None:
    fit = fit_hazard(_factor_rows(), covariates=(), use_risk_set=False)
    coef = fit.coefficients
    assert coef["intercept"] == pytest.approx(-LOG3, abs=1e-5)
    assert coef["constraint_type[procedure]"] == pytest.approx(LOG3, abs=1e-5)
    assert coef["constraint_type[prohibition]"] == pytest.approx(2 * LOG3, abs=1e-5)
    # Var(intercept) = 1 / (4 * 1/4 * 3/4) = 4/3; Var(procedure) = 4/3 + 1 / (4 * 1/2 * 1/2).
    assert fit.fit.std_error("intercept") == pytest.approx(math.sqrt(4 / 3), abs=1e-5)
    assert fit.fit.std_error("constraint_type[procedure]") == pytest.approx(
        math.sqrt(4 / 3 + 1), abs=1e-5
    )
    assert fit.reference == {"constraint_type": "invariant"}
    assert fit.predict(0, constraint_type="procedure") == pytest.approx(0.5, abs=1e-6)
    assert fit.predict(0, constraint_type="prohibition") == pytest.approx(0.75, abs=1e-6)
    assert fit.level_effect("constraint_type", "prohibition") == pytest.approx(2 * LOG3, abs=1e-5)
    assert fit.position_coefficient == 0.0
    assert (fit.n_rows, fit.n_events, fit.n_clusters) == (12, 6, 12)
    record = _json_roundtrip(fit.to_dict())
    assert record["level_counts"]["constraint_type"] == {
        "invariant": 4,
        "procedure": 4,
        "prohibition": 4,
    }


def test_fit_hazard_declared_levels_set_the_reference() -> None:
    order = {"constraint_type": ("prohibition", "invariant", "procedure")}
    fit = fit_hazard(_factor_rows(), covariates=(), use_risk_set=False, levels=order)
    assert fit.reference["constraint_type"] == "prohibition"
    assert fit.coefficients["intercept"] == pytest.approx(LOG3, abs=1e-5)
    assert fit.level_effect("constraint_type", "invariant") == pytest.approx(-2 * LOG3, abs=1e-5)
    assert fit.level_effect("constraint_type", "procedure") == pytest.approx(-LOG3, abs=1e-5)


def test_fit_hazard_absent_declared_reference_and_level_errors() -> None:
    order = {"constraint_type": ("resource_limit", "invariant", "procedure", "prohibition")}
    fit = fit_hazard(_factor_rows(), covariates=(), use_risk_set=False, levels=order)
    assert fit.reference["constraint_type"] == "invariant"  # first declared level present
    assert fit.level_counts is not None
    assert fit.level_counts["constraint_type"]["resource_limit"] == 0
    with pytest.raises(StatsError, match="absent"):
        fit.level_effect("constraint_type", "procedure")
    with pytest.raises(StatsError, match="absent"):
        fit.predict(0, constraint_type="resource_limit")
    with pytest.raises(StatsError, match="no level"):
        fit.predict(0, constraint_type="unknown")
    with pytest.raises(StatsError, match="not a factor"):
        fit.level_effect("placement", "system")
    with pytest.raises(StatsError, match="outside the declared"):
        fit_hazard(
            _factor_rows(),
            covariates=(),
            use_risk_set=False,
            levels={"constraint_type": ("invariant", "procedure")},
        )


def test_fit_hazard_recovers_known_coefficients_on_simulated_sessions() -> None:
    effects = {"invariant": -0.5, "procedure": 0.3, "prohibition": 0.0}
    rows = _simulated_hazard_rows(4000, seed=12345, type_effects=effects)
    fit = fit_hazard(rows)
    assert fit.used_risk_set and fit.n_rows < len(rows) and fit.n_clusters == 4000
    # Reference is invariant, so the intercept absorbs its effect.
    truth = {
        "intercept": -2.5 + effects["invariant"],
        "position": 0.02,
        "compaction_count": 0.4,
        "constraint_type[procedure]": effects["procedure"] - effects["invariant"],
        "constraint_type[prohibition]": effects["prohibition"] - effects["invariant"],
    }
    for name, value in truth.items():
        estimate = fit.coefficients[name]
        assert abs(estimate - value) < 4.0 * fit.fit.std_error(name, robust=True), name
    assert fit.fit.usable and fit.fit.n_clusters == 4000


def test_fit_hazard_log1p_position_recovers_its_slope() -> None:
    rng = np.random.default_rng(99)
    rows = []
    for s in range(3000):
        for p in np.sort(rng.choice(151, size=3, replace=False)):
            eta = -4.0 + 0.6 * math.log1p(float(p))
            rows.append(
                {
                    "family": "syn",
                    "session_id": f"s{s}",
                    "position": int(p),
                    "violated": bool(rng.random() < 1 / (1 + math.exp(-eta))),
                }
            )
    fit = fit_hazard(rows, covariates=("position",), categorical=(), position_transform="log1p")
    assert fit.fit.names == ("intercept", "log1p(position)")
    assert abs(fit.position_coefficient - 0.6) < 4 * fit.fit.std_error("log1p(position)")
    assert fit.predict(20) == pytest.approx(
        1 / (1 + math.exp(-fit.linear_predictor(20))), rel=1e-12
    )
    assert fit.curve([0, 20]) == [fit.predict(0), fit.predict(20)]


POOLED: dict[str, Any] = {
    "covariates": ("position", "compaction_count"),
    "categorical": ("family",),
    "position_transform": "log1p",
}


def test_pooled_hazard_fits_cut_one_risk_set_per_family_session() -> None:
    rows = _two_family_rows(60, seed=8)
    manual = _manual_risk_set(rows)
    # Grouping by the plan alone would drop one family's later opportunities.
    assert len(risk_set(rows, by=("session_id",))) < len(manual)
    reference = fit_hazard(manual, use_risk_set=False, **POOLED)
    fit = fit_hazard(rows, **POOLED)
    assert fit.n_rows == len(manual) and fit.n_events == reference.n_events
    assert fit.n_clusters == 60  # CR1 clusters are plans: both families' sessions of each
    assert np.allclose(fit.fit.coef, reference.fit.coef, rtol=1e-9, atol=1e-12)
    found = bootstrap_coefficients(rows, 30, seed=2, **POOLED)
    for name, interval in found.items():
        assert interval.estimate == pytest.approx(reference.coefficients[name], rel=1e-9, abs=1e-12)
        assert interval.n_clusters == 60 and interval.cluster == "session_id"
    fixed: dict[str, float | str] = {"compaction_count": 1, "family": "fa"}
    assert bootstrap_half_life(rows, 150, fixed, 20, seed=2, **POOLED).fit.n_rows == len(manual)
    curve = bootstrap_hazard_curve(rows, [10, 60], fixed, 20, seed=2, **POOLED)
    assert curve.fit.n_rows == len(manual)
    # Rows that identify sessions by model tier (E2) name their own session key.
    tiers = [{("model" if k == "family" else k): v for k, v in r.items()} for r in rows]
    terms = {k: v for k, v in POOLED.items() if k != "categorical"}
    by_tier = fit_hazard(tiers, categorical=("model",), session=("model", "session_id"), **terms)
    assert by_tier.n_rows == len(manual)


def test_hazard_predict_validates_covariates() -> None:
    rows = _simulated_hazard_rows(300, seed=4)
    fit = fit_hazard(rows, categorical=())
    assert 0.0 < fit.predict(30, compaction_count=1) < 1.0
    with pytest.raises(StatsError, match="needs values"):
        fit.predict(30)
    with pytest.raises(StatsError, match="unknown terms"):
        fit.predict(30, compaction_count=1, placement="system")
    with pytest.raises(StatsError, match="position"):
        fit.predict(-1, compaction_count=1)
    with pytest.raises(StatsError, match="finite number"):
        fit.predict(3, compaction_count="one")
    with pytest.raises(StatsError, match="finite number"):
        fit.predict(3, compaction_count=float("nan"))


def test_fit_hazard_accepts_boolean_covariates_and_integer_outcomes() -> None:
    rows = _simulated_hazard_rows(200, seed=6)
    for i, row in enumerate(rows):
        row["task_success"] = bool(i % 3)
        row["violated"] = int(row["violated"])
    fit = fit_hazard(rows, covariates=("position", "task_success"), categorical=())
    assert fit.fit.names == ("intercept", "position", "task_success")
    as_bool = [dict(row, violated=bool(row["violated"])) for row in rows]
    same = fit_hazard(as_bool, covariates=("position", "task_success"), categorical=())
    assert np.array_equal(same.fit.coef, fit.fit.coef)


def test_fit_hazard_rejects_unusable_designs() -> None:
    rows = _simulated_hazard_rows(50, seed=5)
    with pytest.raises(StatsError, match="no opportunity rows"):
        fit_hazard([])
    flat = [dict(row, compaction_count=0) for row in rows]
    with pytest.raises(StatsError, match="single value"):
        fit_hazard(flat, categorical=())
    twin = [dict(row, twin=row["position"]) for row in rows]
    with pytest.raises(StatsError, match="rank deficient"):
        fit_hazard(twin, covariates=("position", "twin"), categorical=())
    with pytest.raises(StatsError, match="repeat a name"):
        fit_hazard(rows, covariates=("position",), categorical=("position",))
    with pytest.raises(StatsError, match="position_transform"):
        fit_hazard(rows, position_transform="sqrt")  # type: ignore[arg-type]  # invalid on purpose
    with pytest.raises(StatsError, match="no field"):
        fit_hazard([{k: v for k, v in row.items() if k != "constraint_type"} for row in rows])


def test_from_coefficients_validates_names_and_levels() -> None:
    with pytest.raises(StatsError, match="missing"):
        HazardFit.from_coefficients({"intercept": -1.0})
    with pytest.raises(StatsError, match="unexpected"):
        HazardFit.from_coefficients(
            {"intercept": -1.0, "position": 0.1, "compaction_count": 0.0, "extra": 1.0}
        )
    with pytest.raises(StatsError, match="levels for factor"):
        HazardFit.from_coefficients({"intercept": -1.0}, covariates=(), categorical=("type",))
    spec = HazardFit.from_coefficients(
        {"intercept": -1.0, "type[b]": 2.0},
        covariates=(),
        categorical=("type",),
        levels={"type": ("a", "b")},
    )
    assert spec.predict(0, type="b") == pytest.approx(1 / (1 + math.exp(-1.0)))
    assert spec.level_effect("type", "b") == 2.0
    record = _json_roundtrip(spec.to_dict())
    assert record["fit"]["warnings"] == ["specified"] and record["level_counts"] is None
    assert record["fit"]["terms"][0]["se"] is None  # no standard errors for a specified model


# --------------------------------------------------------------------------- half-life


def test_half_life_analytic_linear_crossing() -> None:
    spec = HazardFit.from_coefficients(
        {"intercept": -3.0, "position": 0.05, "compaction_count": 0.4}
    )
    # logit h = -3 + 0.05 p + 0.4 c reaches 0 at p = 3 / 0.05 = 60, or 2.6 / 0.05 = 52 for c = 1.
    assert half_life(spec, 150, compaction_count=0) == pytest.approx(60.0, abs=1e-12)
    assert half_life(spec, 150, compaction_count=1) == pytest.approx(52.0, abs=1e-12)
    assert half_life(spec, 60, compaction_count=0) == pytest.approx(60.0, abs=1e-12)
    assert half_life(spec, 59.9, compaction_count=0) == NotReached(59.9)
    bound = np.int64(50)  # numpy integers are accepted at runtime
    assert half_life(spec, bound, compaction_count=0) == NotReached(50)  # type: ignore[arg-type]  # numpy int
    # Already at least 0.5 at position 0.
    assert half_life(spec, 150, compaction_count=7.5) == 0.0
    assert spec.predict(60, compaction_count=0) == pytest.approx(0.5, abs=1e-12)


def test_half_life_analytic_log1p_crossing() -> None:
    spec = HazardFit.from_coefficients(
        {"intercept": -3.0, "log1p(position)": 1.0},
        covariates=("position",),
        position_transform="log1p",
    )
    # -3 + log(1 + p) = 0 at p = e^3 - 1.
    assert half_life(spec, 150) == pytest.approx(math.e**3 - 1.0, rel=1e-12)
    assert half_life(spec, 10) == NotReached(10)


@pytest.mark.parametrize("slope", [0.0, -0.01])
def test_half_life_flat_or_falling_hazard_is_not_reached(slope: float) -> None:
    spec = HazardFit.from_coefficients(
        {"intercept": -1.0, "position": slope}, covariates=("position",)
    )
    assert half_life(spec, 150) == NotReached(150)


def _log1p_spec(intercept: float, slope: float) -> HazardFit:
    return HazardFit.from_coefficients(
        {"intercept": intercept, "log1p(position)": slope},
        covariates=("position",),
        position_transform="log1p",
    )


def test_half_life_is_judged_from_the_first_observed_position() -> None:
    # A falling hazard whose extrapolation to position 0 passes one half, but which is
    # below one half at every observed position (from 10 on): not reached, never 0.
    falling = _log1p_spec(2.0, -1.0)
    assert falling.predict(0) > 0.5 and falling.predict(10) < 0.5
    assert half_life(falling, 150) == 0.0  # judged at position 0 (the D-029 default)
    assert half_life(falling, 150, first_position=10) == NotReached(150)
    # No jump at eta(0) = 0 for a falling hazard: both sides are not reached.
    for intercept in (-1e-6, 1e-6):
        assert half_life(_log1p_spec(intercept, -1.0), 150, first_position=10) == NotReached(150)
    # At or above one half at the first position: the half-life is at or before it.
    high = _log1p_spec(3.0, -0.5)
    assert high.predict(10) > 0.5
    assert half_life(high, 150, first_position=10) == 10.0
    # A rising hazard keeps its crossing: -3 + log(1 + p) = 0 at p = e^3 - 1.
    rising = _log1p_spec(-3.0, 1.0)
    assert half_life(rising, 150, first_position=10) == pytest.approx(math.e**3 - 1.0)
    assert half_life(rising, 150, first_position=19.5) == 19.5  # already above one half
    for bad in (-1.0, 151.0, float("nan")):
        with pytest.raises(StatsError, match="first_position"):
            half_life(rising, 150, first_position=bad)


def test_half_life_on_a_fitted_model_uses_its_coefficients() -> None:
    rows = _simulated_hazard_rows(1500, seed=21, intercept=-2.0, slope=0.03)
    fit = fit_hazard(rows, categorical=())
    coef = fit.coefficients
    expected = -coef["intercept"] / coef["position"]
    assert half_life(fit, 150, compaction_count=0) == pytest.approx(expected, rel=1e-12)


def test_half_life_rejects_bad_input() -> None:
    spec = HazardFit.from_coefficients(
        {"intercept": -3.0, "position": 0.05}, covariates=("position",)
    )
    for bad in (-1, float("nan"), float("inf"), True, "150"):
        with pytest.raises(StatsError, match="max_position"):
            half_life(spec, bad)  # type: ignore[arg-type]  # invalid input on purpose
    with pytest.raises(StatsError, match="unknown terms"):
        half_life(spec, 150, compaction_count=0)
    stalled = fit_logistic(TABLE_X, TABLE_Y, max_iter=1, check_separation=False)
    unconverged = HazardFit(
        fit=stalled,
        covariates=("position",),
        categorical=(),
        levels={},
        reference={},
        position_transform="linear",
    )
    with pytest.raises(StatsError, match="did not converge"):
        half_life(unconverged, 150)


def test_not_reached_labels_and_validation() -> None:
    assert NotReached(150).label() == ">150"
    assert NotReached(150.5).label() == ">150.5"
    assert NotReached(150.0).to_dict() == {"not_reached_within": 150}
    assert NotReached(150.5).to_dict() == {"not_reached_within": 150.5}
    assert isinstance(NotReached(150.0).to_dict()["not_reached_within"], int)
    for bad in (float("nan"), float("inf"), -1, "150", True):
        with pytest.raises(StatsError):
            NotReached(bad)  # type: ignore[arg-type]  # invalid input on purpose


# --------------------------------------------------------------------------- bootstrap


def _clustered_rows(n_sessions: int = 30, seed: int = 7) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sessions):
        for k in range(int(rng.integers(1, 5))):
            rows.append(
                {
                    "family": "syn",
                    "session_id": f"s{s:02d}",
                    "position": int(10 * k + rng.integers(0, 10)),
                    "violated": bool(rng.random() < 0.3),
                    "compaction_count": int(rng.integers(0, 3)),
                    "arm": "a" if s % 2 else "b",
                }
            )
    return rows


def _rate(sample: Sequence[dict[str, Any]]) -> float:
    return sum(bool(r["violated"]) for r in sample) / len(sample)


def _replayed_draws(
    rows: Sequence[dict[str, Any]],
    statistic: Callable[[list[dict[str, Any]]], float],
    n_boot: int,
    seed: int,
) -> list[float]:
    """An independent cluster bootstrap: same generator, sorted cluster keys, whole clusters."""
    keys = sorted({str(r["session_id"]) for r in rows})
    members = {k: [r for r in rows if str(r["session_id"]) == k] for k in keys}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        picks = rng.integers(0, len(keys), size=len(keys))
        draws.append(statistic([r for j in picks for r in members[keys[j]]]))
    return draws


def test_cluster_bootstrap_matches_an_independent_implementation() -> None:
    rows = _clustered_rows()
    interval = cluster_bootstrap(rows, _rate, 500, seed=11)
    draws = _replayed_draws(rows, _rate, 500, 11)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    assert interval.estimate == pytest.approx(_rate(rows), abs=1e-15)
    assert interval.lo == pytest.approx(lo, abs=1e-12)
    assert interval.hi == pytest.approx(hi, abs=1e-12)
    assert (interval.n_boot, interval.n_failed, interval.n_used) == (500, 0, 500)
    assert interval.n_clusters == 30 and interval.seed == 11 and interval.strata is None


def test_cluster_bootstrap_is_deterministic_under_a_seed() -> None:
    rows = _clustered_rows()
    first = cluster_bootstrap(rows, _rate, 300, seed=5)
    assert cluster_bootstrap(list(reversed(rows)), _rate, 300, seed=5) == first
    assert cluster_bootstrap(rows, _rate, 300, 5) == first  # seed may be positional
    assert cluster_bootstrap(rows, _rate, 300, seed=6) != first


def test_cluster_bootstrap_requires_a_seed_and_valid_settings() -> None:
    rows = _clustered_rows()
    with pytest.raises(StatsError, match="explicit seed"):
        cluster_bootstrap(rows, _rate, 100)
    for n_boot in (0, -1, 2.5, True):
        with pytest.raises(StatsError, match="n_boot"):
            cluster_bootstrap(rows, _rate, n_boot, seed=1)  # type: ignore[arg-type]  # invalid
    with pytest.raises(StatsError, match="level"):
        cluster_bootstrap(rows, _rate, 100, seed=1, level=1.0)
    with pytest.raises(StatsError, match="seed"):
        cluster_bootstrap(rows, _rate, 100, seed=-3)
    with pytest.raises(StatsError, match="no rows"):
        cluster_bootstrap([], _rate, 100, seed=1)
    with pytest.raises(StatsError, match="no field"):
        cluster_bootstrap([{"violated": True}], _rate, 100, seed=1)


def test_cluster_bootstrap_counts_failed_draws() -> None:
    rows = _clustered_rows(n_sessions=10)

    def needs_s00(sample: list[dict[str, Any]]) -> float:
        # Draw slots tag session ids as "<key>#<slot>"; the full data has bare ids.
        if not any(str(r["session_id"]).split("#")[0] == "s00" for r in sample):
            raise StatsError("s00 absent")
        return _rate(sample)

    interval = cluster_bootstrap(rows, needs_s00, 400, seed=2)
    # Replay which draws lack s00 with the same generator and sorted cluster keys.
    keys = sorted({r["session_id"] for r in rows})
    rng = np.random.default_rng(2)
    missing = sum(
        keys.index("s00") not in set(rng.integers(0, len(keys), size=len(keys)).tolist())
        for _ in range(400)
    )
    assert interval.n_failed == missing > 0
    assert interval.n_used == 400 - missing


def test_cluster_bootstrap_errors_for_undefined_or_malformed_statistics() -> None:
    rows = _clustered_rows()
    with pytest.raises(StatsError, match="undefined on the full data"):
        cluster_bootstrap(rows, lambda s: math.nan, 50, seed=1)

    def fails_on_resamples(sample: list[dict[str, Any]]) -> float:
        if any("#" in str(r["session_id"]) for r in sample):
            raise StatsError("resample")
        return 0.5

    with pytest.raises(StatsError, match="every resample"):
        cluster_bootstrap(rows, fails_on_resamples, 20, seed=1)

    def inf_on_resamples(sample: list[dict[str, Any]]) -> float:
        return math.inf if any("#" in str(r["session_id"]) for r in sample) else 0.5

    with pytest.raises(StatsError, match="NotReached"):
        cluster_bootstrap(rows, inf_on_resamples, 20, seed=1)
    with pytest.raises(StatsError, match="not a number"):
        cluster_bootstrap(rows, lambda s: "0.5", 20, seed=1)  # type: ignore[arg-type, return-value]  # invalid on purpose
    with pytest.raises(StatsError, match="not a number"):
        cluster_bootstrap(rows, lambda s: True, 20, seed=1)
    with pytest.raises(StatsError, match="no values"):
        cluster_bootstrap_many(rows, lambda s: {}, 20, seed=1)

    def changing_names(sample: list[dict[str, Any]]) -> dict[str, float]:
        return {"b": 1.0} if any("#" in str(r["session_id"]) for r in sample) else {"a": 1.0}

    with pytest.raises(StatsError, match="different names"):
        cluster_bootstrap_many(rows, changing_names, 20, seed=1)


def test_cluster_bootstrap_many_shares_draws_and_fails_names_separately() -> None:
    rows = _clustered_rows()

    def statistic(sample: list[dict[str, Any]]) -> dict[str, float]:
        rate = _rate(sample)
        return {"rate": rate, "double": 2 * rate, "flaky": math.nan if rate > 0.3 else rate}

    found = cluster_bootstrap_many(rows, statistic, 300, seed=8)
    single = cluster_bootstrap(rows, _rate, 300, seed=8)
    assert (found["rate"].lo, found["rate"].hi) == (single.lo, single.hi)
    assert found["double"].lo == pytest.approx(2 * _num(single.lo))
    draws = _replayed_draws(rows, _rate, 300, 8)
    assert found["flaky"].n_failed == sum(d > 0.3 for d in draws) > 0
    assert found["rate"].n_failed == 0


def test_cluster_bootstrap_not_reached_draws_sort_above_numbers() -> None:
    # Two clusters: a draw of two copies of "b" (probability 1/4) is not reached.
    rows = [
        {"session_id": "a", "position": 1, "violated": True},
        {"session_id": "b", "position": 2, "violated": False},
    ]

    def statistic(sample: list[dict[str, Any]]) -> stats.Value:
        if not any(r["violated"] for r in sample):
            return NotReached(150)
        return 10.0 * _rate(sample)

    interval = cluster_bootstrap(rows, statistic, 400, seed=4)
    rng = np.random.default_rng(4)
    only_b = sum(bool(np.all(rng.integers(0, 2, size=2) == 1)) for _ in range(400))
    assert interval.n_not_reached == only_b
    assert interval.share_not_reached == pytest.approx(only_b / 400)
    assert interval.estimate == 5.0 and interval.lo == 5.0
    assert interval.hi == NotReached(150)
    assert interval.label() == "5 [5, >150]"
    record = _json_roundtrip(interval.to_dict("half_life"))
    assert record["half_life"] == 5.0 and record["hi"] is None
    assert record["hi_not_reached_within"] == 150
    assert describe_interval("half-life", interval).endswith(
        f"(95% cluster bootstrap, 400 draws, {round(100 * only_b / 400, 1):g}% not reached)"
    )


@pytest.mark.parametrize(("not_reached", "open_bound"), [(0, False), (49, False), (50, True)])
def test_upper_bound_opens_only_when_its_percentile_is_a_not_reached_draw(
    not_reached: int, open_bound: bool
) -> None:
    # 2000 draws; with linear interpolation the 97.5th percentile sits between sorted draws
    # 1949 and 1950, so it is open exactly when at least 50 draws (2.5%) are not reached.
    # Fewer such draws leave a finite bound with a share above zero (D-028, D-029).
    rows = [{"session_id": f"s{i}", "position": 1, "violated": False} for i in range(3)]
    calls = iter(range(2001))

    def statistic(sample: list[dict[str, Any]]) -> stats.Value:
        call = next(calls)
        return NotReached(150) if 0 < call <= not_reached else float(call)

    interval = cluster_bootstrap(rows, statistic, 2000, seed=1)
    assert interval.n_not_reached == not_reached
    assert isinstance(interval.hi, NotReached) is open_bound
    assert interval.share_not_reached == pytest.approx(not_reached / 2000)


def test_cluster_bootstrap_every_draw_not_reached() -> None:
    rows = [{"session_id": s, "position": 1, "violated": False} for s in "abc"]
    interval = cluster_bootstrap(rows, lambda s: NotReached(60), 50, seed=1)
    assert interval.estimate == interval.lo == interval.hi == NotReached(60)
    assert interval.share_not_reached == 1.0
    record = _json_roundtrip(interval.to_dict("half_life"))
    assert record["not_reached_within"] == 60 and record["lo_not_reached_within"] == 60


def test_cluster_bootstrap_strata_keep_arm_sizes() -> None:
    rows = _clustered_rows()
    sessions_in_a = len({r["session_id"] for r in rows if r["arm"] == "a"})

    def sessions_a(sample: list[dict[str, Any]]) -> float:
        return float(len({r["session_id"] for r in sample if r["arm"] == "a"}))

    fixed = cluster_bootstrap(rows, sessions_a, 100, seed=3, strata="arm")
    assert fixed.lo == fixed.hi == sessions_in_a and fixed.strata == "arm"
    free = cluster_bootstrap(rows, sessions_a, 100, seed=3)
    assert _num(free.lo) < _num(free.hi)
    mixed = [dict(rows[0], arm="b"), dict(rows[0], arm="a")]
    with pytest.raises(StatsError, match="spans several"):
        cluster_bootstrap(mixed, sessions_a, 10, seed=3, strata="arm")


def test_cluster_bootstrap_on_another_field_relabels_sessions() -> None:
    # Template A holds sessions s1 and s2, template B holds s3. A draw of A twice must
    # count four sessions, not two, so session ids are relabeled per draw slot.
    rows = [
        {"template_id": "A", "session_id": "s1", "position": 1, "violated": False},
        {"template_id": "A", "session_id": "s2", "position": 1, "violated": False},
        {"template_id": "B", "session_id": "s3", "position": 1, "violated": False},
    ]

    def n_sessions(sample: list[dict[str, Any]]) -> float:
        return float(len({r["session_id"] for r in sample}))

    interval = cluster_bootstrap(rows, n_sessions, 400, seed=9, cluster="template_id")
    assert interval.estimate == 3.0 and interval.lo == 2.0 and interval.hi == 4.0
    assert interval.cluster == "template_id" and interval.n_clusters == 2


def test_pooled_bootstrap_draws_hold_both_families_sessions_of_each_plan() -> None:
    # Clustering on session_id, the plan id both families share, resamples whole plans.
    rows = _two_family_rows(12, seed=5)

    def complete(sample: list[dict[str, Any]]) -> float:
        families: dict[str, set[str]] = {}
        for row in sample:
            families.setdefault(row["session_id"], set()).add(row["family"])
        return float(all(found == {"fa", "fb"} for found in families.values()))

    def family_sessions(sample: list[dict[str, Any]]) -> float:
        return float(len({(r["family"], r["session_id"]) for r in sample}))

    whole = cluster_bootstrap(rows, complete, 200, seed=3)
    assert whole.estimate == whole.lo == whole.hi == 1.0
    assert whole.n_clusters == 12 and whole.cluster == "session_id"
    # A plan drawn twice counts as two sessions per family: always 2 x 12 per draw.
    counted = cluster_bootstrap(rows, family_sessions, 200, seed=3)
    assert counted.estimate == counted.lo == counted.hi == 24.0


def test_percentile_interpolates_like_numpy_and_into_not_reached() -> None:
    values = [1.0, 2.0, 4.0, 8.0]
    for q in (0.0, 0.025, 0.3, 0.5, 0.975, 1.0):
        assert stats._percentile(values, q) == pytest.approx(float(np.percentile(values, 100 * q)))
    # Between a finite draw and a not-reached one (+inf) the percentile is not reached.
    assert stats._percentile([1.0, 2.0, math.inf], 0.975) == math.inf
    assert stats._percentile([1.0, math.inf, math.inf], 0.975) == math.inf
    assert stats._percentile([1.0, 2.0, math.inf], 0.5) == 2.0


def test_interval_helpers() -> None:
    interval = Interval(estimate=1.0, lo=0.5, hi=NotReached(9), level=0.9, n_boot=10, n_failed=10)
    assert interval.n_used == 0 and interval.share_not_reached == 0.0
    shifted = interval.transformed(lambda v: v + 1.0)
    assert (shifted.estimate, shifted.lo, shifted.hi) == (2.0, 1.5, NotReached(9))
    assert shifted.label(2) == "2 [1.5, >9]"
    assert describe_interval("x", shifted) == "x: 2 [1.5, >9] (90% cluster bootstrap, 0 draws)"


# --------------------------------------------------------------------------- bootstrap helpers


def test_bootstrap_half_life_reached_with_interval() -> None:
    rows = _simulated_hazard_rows(400, seed=31, intercept=-3.0, slope=0.05, compaction=0.3)
    result = bootstrap_half_life(rows, 150, {"compaction_count": 0}, 120, seed=1, categorical=())
    point = half_life(result.fit, 150, compaction_count=0)
    assert isinstance(point, float)
    assert result.interval.estimate == pytest.approx(point, rel=1e-6)
    assert isinstance(result.interval.lo, float) and isinstance(result.interval.hi, float)
    assert result.interval.lo <= point <= result.interval.hi
    record = _json_roundtrip(result.to_dict())
    # The layout gate G7 reads: half_life, lo, hi (finite, ordered).
    assert record["lo"] <= record["half_life"] <= record["hi"]
    assert record["max_position"] == 150 and record["fixed"] == {"compaction_count": 0}
    assert record["model"]["used_risk_set"] is True
    again = bootstrap_half_life(rows, 150, {"compaction_count": 0}, 120, 1, categorical=())
    assert again.interval == result.interval


def test_bootstrap_half_life_not_reached_and_failed_draws() -> None:
    rows = _simulated_hazard_rows(60, seed=8, intercept=-3.0, slope=0.0, compaction=0.0)
    rows += [
        {
            "family": "syn",
            "session_id": "rare",
            "position": 5,
            "compaction_count": 1,
            "constraint_type": "rare",
            "violated": True,
        }
    ]
    # "rare" holds one violation and nothing else, which separates its term: a half-life
    # at that level would be set by the ridge, so none is reported.
    result = bootstrap_half_life(
        rows, 150, {"compaction_count": 0, "constraint_type": "rare"}, 100, seed=2
    )
    assert result.interval.not_estimable == "separation"
    assert _json_roundtrip(result.to_dict())["half_life"] is None
    flat = bootstrap_half_life(
        rows, 150, {"compaction_count": 0, "constraint_type": "prohibition"}, 100, seed=2
    )
    assert flat.interval.estimate == NotReached(150)
    assert flat.interval.share_not_reached > 0.5
    # Draws without the rare session lack its level; at another level they still count.
    assert flat.interval.n_failed == 0
    record = _json_roundtrip(flat.to_dict())
    assert record["half_life"] is None and record["not_reached_within"] == 150
    with pytest.raises(StatsError, match="no level"):
        bootstrap_half_life(rows, 150, {"compaction_count": 0, "constraint_type": "x"}, 10, seed=2)


def test_bootstrap_half_life_from_a_separated_fit_is_not_estimable() -> None:
    # Synthetic: the only violation sits at the largest position, which separates
    # log1p(position); its crossing of one half would be set by the separating line and
    # the ridge, so the half-life is written without a number.
    rows = [dict(r, violated=False) for r in _simulated_hazard_rows(40, seed=3)]
    last = max(rows, key=lambda r: (r["position"], r["session_id"]))
    last["violated"] = True
    result = bootstrap_half_life(
        rows, 150, {"compaction_count": 1}, 50, seed=1, categorical=(), position_transform="log1p"
    )
    assert result.fit.fit.separation and result.fit.n_events == 1
    assert result.interval.not_estimable == "separation"
    record = _json_roundtrip(result.to_dict())
    assert (record["half_life"], record["lo"], record["hi"]) == (None, None, None)
    assert record["not_estimable"] == "separation" and "not_reached_within" not in record
    # No violation at all is the boundary answer: the hazard never reaches one half.
    quiet = [dict(r, violated=False) for r in rows]
    none = bootstrap_half_life(
        quiet, 150, {"compaction_count": 1}, 50, seed=1, categorical=(), position_transform="log1p"
    )
    assert none.interval.estimate == none.interval.lo == none.interval.hi == NotReached(150)
    assert none.interval.n_failed == 0 and none.interval.not_estimable is None


def test_bootstrap_half_life_searches_from_the_first_position() -> None:
    # Synthetic: every session violates at its first decision point, so the hazard is at
    # or above one half from the first observed position on (left-censored there).
    rows = [dict(r, violated=True) for r in _simulated_hazard_rows(30, seed=5)]
    first = min(r["position"] for r in rows)
    result = bootstrap_half_life(rows, 150, {"compaction_count": 0}, 40, seed=1, categorical=())
    assert result.first_position == first
    assert result.interval.estimate == result.interval.lo == float(first)
    assert result.share_at_first == 1.0
    record = _json_roundtrip(result.to_dict())
    assert record["first_position"] == first and record["share_at_first"] == 1.0
    fixed = bootstrap_half_life(
        rows, 150, {"compaction_count": 0}, 40, seed=1, categorical=(), first_position=25
    )
    assert fixed.first_position == 25 and fixed.interval.estimate == 25.0


def test_bootstrap_hazard_curve_pointwise_intervals() -> None:
    rows = _simulated_hazard_rows(300, seed=13, intercept=-2.0, slope=0.02)
    curve = bootstrap_hazard_curve(
        rows, [0, 50, 100], {"compaction_count": 0}, 80, seed=3, categorical=()
    )
    assert curve.positions == (0.0, 50.0, 100.0)
    for position, interval in zip(curve.positions, curve.intervals, strict=True):
        assert interval.estimate == pytest.approx(
            curve.fit.predict(position, compaction_count=0), rel=1e-6
        )
        assert 0.0 < _num(interval.lo) <= _num(interval.estimate) <= _num(interval.hi) < 1.0
    record = _json_roundtrip(curve.to_dict())
    assert [p["position"] for p in record["points"]] == [0.0, 50.0, 100.0]
    assert "hazard" in record["points"][0]
    with pytest.raises(StatsError, match="no positions"):
        bootstrap_hazard_curve(rows, [], {"compaction_count": 0}, 10, seed=3, categorical=())
    with pytest.raises(StatsError, match="needs values"):
        bootstrap_hazard_curve(rows, [1], {}, 10, seed=3, categorical=())


def test_bootstrap_coefficients_with_a_sparse_reference_level() -> None:
    effects = {"invariant": -0.4, "procedure": 0.2}
    rows = _simulated_hazard_rows(200, seed=17, type_effects=effects)
    sparse = [
        {
            "family": "syn",
            "session_id": "zz-only",
            "position": p,
            "compaction_count": 0,
            "constraint_type": "a_sparse",
            "violated": p == 9,
        }
        for p in (3, 9)
    ]
    found = bootstrap_coefficients(rows + sparse, 120, seed=6)
    full = fit_hazard(rows + sparse)
    assert set(found) == {
        "intercept",
        "position",
        "compaction_count",
        "constraint_type[invariant]",
        "constraint_type[procedure]",
    }
    assert not found.not_estimable
    for name, interval in found.items():
        assert interval.estimate == pytest.approx(full.coefficients[name], rel=1e-6, abs=1e-9)
    # The sorted-first level "a_sparse" is the reference and sits in one session, so
    # intercept and factor contrasts fail on draws without it; slopes never do.
    assert found["intercept"].n_failed > 0
    assert found["constraint_type[procedure]"].n_failed == found["intercept"].n_failed
    assert found["position"].n_failed == 0
    # A reference level without a violation separates the intercept and every contrast
    # against it: they are not estimable, while the slopes still are.
    quiet = [dict(row, violated=False) for row in sparse]
    separated = bootstrap_coefficients(rows + quiet, 40, seed=6)
    assert separated.not_estimable == {
        "intercept": "separation",
        "constraint_type[invariant]": "separation",
        "constraint_type[procedure]": "separation",
    }
    assert separated["position"].not_estimable is None


def _replayed_samples(
    data: Sequence[dict[str, Any]], n_boot: int, seed: int
) -> list[list[dict[str, Any]]]:
    """The rows of every bootstrap draw, replayed with the generator and sorted cluster keys."""
    clusters: dict[str, list[dict[str, Any]]] = {}
    for row in data:
        clusters.setdefault(str(row["session_id"]), []).append(row)
    keys = sorted(clusters)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(n_boot):
        picks = rng.integers(0, len(keys), size=len(keys))
        samples.append([dict(row) for pick in picks for row in clusters[keys[int(pick)]]])
    return samples


def test_bootstrap_coefficients_without_a_violation_report_no_number() -> None:
    # Synthetic rows with no violation: every coefficient is held only by the ridge (the
    # values move with l2), so no term is estimable and none is reported (D-029).
    rows = [dict(r, violated=False) for r in _simulated_hazard_rows(60, seed=11)]
    for l2 in (1e-6, 1e-8):
        found = bootstrap_coefficients(rows, 40, seed=1, categorical=(), l2=l2)
        assert set(found) == {"intercept", "position", "compaction_count"}
        assert found.fit.n_events == 0 and found.fit.fit.separation
        for interval in found.values():
            assert interval.not_estimable == "no_violations"
            record = _json_roundtrip(interval.to_dict())
            assert (record["estimate"], record["lo"], record["hi"]) == (None, None, None)
            assert record["not_estimable"] == "no_violations" and record["n_failed"] == 0
        assert found.not_estimable == dict.fromkeys(found, "no_violations")


def test_bootstrap_coefficients_count_draws_without_violations_or_separated_as_failed() -> None:
    # Synthetic rows whose violations sit in three sessions: a draw that misses all three has
    # no violation, and a draw with too few of them separates a term. Such draws are left
    # out of the percentiles and counted, never entered at a ridge-held value.
    rows = [dict(r, violated=False) for r in _simulated_hazard_rows(40, seed=12)]
    for sid in ("s0003", "s0017", "s0029"):
        first = next(r for r in rows if r["session_id"] == sid)
        first["violated"] = True
    found = bootstrap_coefficients(rows, 200, seed=4, categorical=())
    data = risk_set(rows)
    full = fit_hazard(data, categorical=(), use_risk_set=False)
    assert not found.not_estimable and full.n_events == 3
    expected: dict[str, int] = dict.fromkeys(found, 0)
    for sample in _replayed_samples(data, 200, 4):
        try:
            model = fit_hazard(sample, categorical=(), use_risk_set=False)
        except StatsError:
            for name in expected:
                expected[name] += 1
            continue
        flagged = set(model.fit.separated_terms) or set(model.fit.names)
        for name in expected:
            if model.n_events == 0 or (model.fit.separation and name in flagged):
                expected[name] += 1
    assert {name: interval.n_failed for name, interval in found.items()} == expected
    assert expected["position"] > 0
    for name, interval in found.items():
        assert interval.estimate == pytest.approx(full.coefficients[name], rel=1e-9)


def test_bootstrap_coefficients_pooled_over_a_family_without_violations() -> None:
    # Synthetic: family fb never violates. The pooled slopes come from fa alone and stay
    # estimable; the family term (fb against the reference fa) is separated.
    rows = [
        dict(r, violated=False) if r["family"] == "fb" else r for r in _two_family_rows(40, seed=9)
    ]
    found = bootstrap_coefficients(rows, 30, seed=2, **POOLED)
    assert found.not_estimable == {"family[fb]": "separation"}
    for name in ("log1p(position)", "compaction_count"):
        assert found[name].not_estimable is None and isinstance(found[name].estimate, float)
    alone = bootstrap_coefficients(
        [r for r in rows if r["family"] == "fa"], 30, seed=2, **(POOLED | {"categorical": ()})
    )
    assert found["log1p(position)"].estimate == pytest.approx(
        _num(alone["log1p(position)"].estimate), abs=1e-4
    )


# --------------------------------------------------------------------------- Kaplan-Meier

# Worked example. Times 1, 2, 2+, 3, 4+, 4 (+ censored), n = 6.
#   t=1: at risk 6, 1 event   -> S = 5/6;              Greenwood V = 1/(6*5) = 1/30
#   t=2: at risk 5, 1 event, 1 censored -> S = 5/6 * 4/5 = 2/3;  V += 1/(5*4) -> 1/12
#   t=3: at risk 3, 1 event   -> S = 2/3 * 2/3 = 4/9;  V += 1/(3*2) -> 1/4
#   t=4: at risk 2, 1 event, 1 censored -> S = 4/9 * 1/2 = 2/9;  V += 1/(2*1) -> 3/4
# The median is the first time with S <= 1/2: t = 3.
KM_TIMES = [1, 2, 2, 3, 4, 4]
KM_EVENTS = [True, True, False, True, False, True]
KM_S = [5 / 6, 2 / 3, 4 / 9, 2 / 9]
KM_V = [1 / 30, 1 / 12, 1 / 4, 3 / 4]


def test_kaplan_meier_worked_example() -> None:
    km = kaplan_meier(KM_TIMES, KM_EVENTS)
    assert km.times == (1.0, 2.0, 3.0, 4.0)
    assert km.n_risk == (6, 5, 3, 2)
    assert km.n_event == (1, 1, 1, 1)
    assert km.n_censor == (0, 1, 0, 1)
    assert km.survival == pytest.approx(KM_S, abs=1e-15)
    for s, v, lo, hi in zip(KM_S, KM_V, km.lower, km.upper, strict=True):
        # log-log interval: S ** exp(+-z sqrt(V) / |log S|).
        spread = Z95 * math.sqrt(v) / abs(math.log(s))
        assert lo == pytest.approx(s ** math.exp(spread), abs=1e-14)
        assert hi == pytest.approx(s ** math.exp(-spread), abs=1e-14)
    assert km.median == 3.0
    assert (km.n, km.n_events, km.descriptive) == (6, 4, True)
    assert km.censored_times == (2.0, 4.0)
    assert [km.survival_at(t) for t in (0.5, 1, 2.5, 99)] == pytest.approx(
        [1.0, 5 / 6, 2 / 3, 2 / 9]
    )
    assert "Descriptive only" in km.note


def test_kaplan_meier_level_and_input_types() -> None:
    wide = kaplan_meier(KM_TIMES, KM_EVENTS)
    narrow = kaplan_meier(np.array(KM_TIMES), np.array(KM_EVENTS), level=0.9)
    assert narrow.survival == wide.survival
    for a, b, c, d in zip(narrow.lower, wide.lower, narrow.upper, wide.upper, strict=True):
        assert a is not None and b is not None and c is not None and d is not None
        assert b < a and c < d
    assert kaplan_meier(KM_TIMES, [1, 1, 0, 1, 0, 1]).survival == wide.survival


def test_kaplan_meier_edges() -> None:
    to_zero = kaplan_meier([1, 2, 2], [1, 1, 1])
    assert to_zero.survival == pytest.approx((2 / 3, 0.0))
    assert to_zero.lower[1] is None and to_zero.upper[1] is None
    assert to_zero.median == 2.0
    censored = kaplan_meier([3, 5], [False, False])
    assert censored.survival == (1.0, 1.0) and censored.lower == (1.0, 1.0)
    assert censored.median == NotReached(5)
    record = _json_roundtrip(censored.to_dict())
    assert record["median"] is None and record["median_not_reached_within"] == 5
    assert record["descriptive"] is True
    record = _json_roundtrip(to_zero.to_dict())
    assert record["steps"][1]["lower"] is None and record["median"] == 2.0


@pytest.mark.parametrize(
    ("times", "events"),
    [
        ([], []),
        ([1, 2], [True]),
        ([1, float("nan")], [True, False]),
        (["1"], [True]),
        ([True], [True]),
        ([1], [2]),
        ([1], ["yes"]),
    ],
)
def test_kaplan_meier_rejects_bad_input(times: list[Any], events: list[Any]) -> None:
    with pytest.raises(StatsError):
        kaplan_meier(times, events)


def test_kaplan_meier_rows_uses_time_to_first_violation() -> None:
    rows = [
        {"family": "f", "session_id": "a", "position": 4, "violated": False},
        {"family": "f", "session_id": "a", "position": 9, "violated": True},
        {"family": "f", "session_id": "b", "position": 3, "violated": False},
        {"family": "f", "session_id": "b", "position": 12, "violated": False},
        {"family": "f", "session_id": "c", "position": 2, "violated": True},
    ]
    km = kaplan_meier_rows(rows)
    triples = analysis.time_to_first_violation(rows)
    expected = kaplan_meier([t for _, t, _ in triples], [e for _, _, e in triples])
    assert km == expected
    assert km.times == (2.0, 9.0, 12.0) and km.n_events == 2
    with pytest.raises(StatsError, match="no opportunity rows"):
        kaplan_meier_rows([])
    assert isinstance(km, KaplanMeier)


def test_kaplan_meier_rows_count_each_family_session_of_a_shared_plan() -> None:
    # fa: p0 violated at 3, p1 censored at 2; fb: p0 violated at 10, p1 censored at 2.
    km = kaplan_meier_rows(_shared_plan_rows())
    assert km.n == 4 and km.n_events == 2
    assert km.times == (2.0, 3.0, 10.0) and km.n_risk == (4, 2, 1)
    assert kaplan_meier_rows(_shared_plan_rows(), session=("session_id",)).n == 2


# --------------------------------------------------------------------------- contrasts

# Arm "none": 3 of 10 opportunities violated; arm "pin": 6 of 10, five sessions each.
# Difference 0.6 - 0.3 = 0.3; odds ratio (6/4) / (3/7) = 3.5.
ARM_A = [1, 0, 0, 1, 0, 0, 1, 0, 0, 0]
ARM_B = [1, 1, 0, 1, 1, 0, 1, 0, 1, 0]


def _arm_rows() -> list[dict[str, Any]]:
    rows = []
    for arm, outcomes in (("none", ARM_A), ("pin", ARM_B)):
        for i, v in enumerate(outcomes):
            rows.append(
                {
                    "family": "syn",
                    "session_id": f"{arm}{i // 2}",
                    "mitigation": arm,
                    "violated": bool(v),
                    "position": 10 * (i % 2) + i,
                    "constraint_type": "prohibition" if i % 3 else "invariant",
                }
            )
    rows.append(
        {
            "family": "syn",
            "session_id": "x0",
            "mitigation": "guard",
            "violated": True,
            "position": 1,
        }
    )
    return rows


def test_rate_contrast_matches_closed_form_rates_and_odds_ratio() -> None:
    contrast = rate_contrast(_arm_rows(), "mitigation", "none", "pin", 300, 5)
    assert contrast.a.rate.estimate == pytest.approx(0.3)
    assert contrast.b.rate.estimate == pytest.approx(0.6)
    assert contrast.difference.estimate == pytest.approx(0.3)
    assert contrast.log_odds_ratio.estimate == pytest.approx(math.log(3.5), abs=1e-5)
    assert contrast.odds_ratio.estimate == pytest.approx(3.5, rel=1e-5)
    assert contrast.odds_ratio.lo == pytest.approx(math.exp(_num(contrast.log_odds_ratio.lo)))
    assert contrast.term == "mitigation[pin]" and contrast.stratified
    assert contrast.difference.strata == "mitigation"
    assert (contrast.a.n_sessions, contrast.a.n_opportunities, contrast.a.n_violations) == (
        5,
        10,
        3,
    )
    naive = wilson_interval(3, 10)
    assert (contrast.a.naive.lo, contrast.a.naive.hi) == (naive.lo, naive.hi)
    record = _json_roundtrip(contrast.to_dict())
    assert record["reference_arm"]["arm"] == "none" and record["arm"]["arm"] == "pin"
    assert record["odds_ratio"]["odds_ratio"] == pytest.approx(3.5, rel=1e-5)
    assert "ignores clustering" in record["arm"]["naive_wilson"]["note"]
    keyword = rate_contrast(_arm_rows(), "mitigation", "none", "pin", n_boot=300, seed=5)
    assert keyword.difference == contrast.difference


def test_rate_contrast_within_session_arms_and_adjustment() -> None:
    rows = _clustered_rows(n_sessions=40, seed=12)
    for i, row in enumerate(rows):
        row["arm"] = "b" if i % 2 else "a"
    contrast = rate_contrast(
        rows,
        "arm",
        "a",
        "b",
        100,
        seed=2,
        covariates=("position",),
        categorical=("compaction_count",),
        use_risk_set=True,
    )
    assert not contrast.stratified and contrast.difference.strata is None
    assert contrast.fit.names[:3] == ("intercept", "arm[b]", "position")
    assert contrast.used_risk_set
    assert contrast.a.n_opportunities + contrast.b.n_opportunities == len(risk_set(rows))
    assert contrast.categorical == ("compaction_count",)


def test_rate_contrast_integer_arms_and_errors() -> None:
    rows = [dict(row, arm=int(row["mitigation"] == "pin")) for row in _arm_rows()]
    contrast = rate_contrast(rows, "arm", 0, 1, 50, seed=1)
    assert contrast.term == "arm[1]" and contrast.b.arm == "1"
    with pytest.raises(StatsError, match="must differ"):
        rate_contrast(rows, "arm", 1, 1, 50, seed=1)
    with pytest.raises(StatsError, match="no opportunities"):
        rate_contrast(rows, "arm", 0, 7, 50, seed=1)
    with pytest.raises(StatsError, match="no field"):
        rate_contrast([*rows, {"session_id": "q", "violated": True}], "arm", 0, 1, 50, seed=1)
    with pytest.raises(StatsError, match="explicit seed"):
        rate_contrast(rows, "arm", 0, 1, 50)


def test_rate_contrast_rejects_a_covariate_collinear_with_the_arm() -> None:
    rows = [dict(row, arm_position=float(row["mitigation"] == "pin")) for row in _arm_rows()]
    with pytest.raises(StatsError, match="rank deficient"):
        rate_contrast(rows, "mitigation", "none", "pin", 20, seed=1, covariates=("arm_position",))


def test_rate_contrast_resample_without_one_arm_fails_the_draw() -> None:
    # Arms mixed within the first session only; the second session is all arm a, so
    # unstratified draws of that session twice have no arm b.
    rows = [
        {"family": "f", "session_id": "s1", "arm": "a", "violated": False, "position": 1},
        {"family": "f", "session_id": "s1", "arm": "b", "violated": True, "position": 2},
        {"family": "f", "session_id": "s1", "arm": "a", "violated": True, "position": 3},
        {"family": "f", "session_id": "s1", "arm": "b", "violated": False, "position": 4},
        {"family": "f", "session_id": "s2", "arm": "a", "violated": False, "position": 1},
    ]
    contrast = rate_contrast(rows, "arm", "a", "b", 200, seed=3)
    assert not contrast.stratified
    assert contrast.difference.n_failed > 0


def _two_arm_rows(
    n_sessions: int, rate_a: float, rate_b: float, seed: int, per_session: int = 4
) -> list[dict[str, Any]]:
    """Synthetic sessions, alternately in arm a and b, with independent outcomes."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sessions):
        arm = "b" if s % 2 else "a"
        for j in range(per_session):
            rows.append(
                {
                    "family": "syn",
                    "session_id": f"s{s:03d}",
                    "arm": arm,
                    "position": 5 * j,
                    "violated": bool(rng.random() < (rate_b if arm == "b" else rate_a)),
                }
            )
    return rows


def test_rate_contrast_wald_test_uses_session_clustered_cr1_errors() -> None:
    rows = _two_arm_rows(200, 0.1, 0.3, seed=4)
    contrast = rate_contrast(rows, "arm", "a", "b", 50, seed=1)
    assert contrast.fit.robust_se is not None and contrast.fit.n_clusters == 200
    wald = contrast.wald
    assert wald.robust and wald.problem is None and wald.detail is None
    assert wald.term == "arm[b]" and wald.n_clusters == 200
    # Independent check: the same logistic fit with session clusters, and the normal tail.
    reference = fit_logistic(
        [1.0 if r["arm"] == "b" else 0.0 for r in rows],
        [int(r["violated"]) for r in rows],
        names=["arm[b]"],
        clusters=[r["session_id"] for r in rows],
    )
    z = reference.coefficient("arm[b]") / reference.std_error("arm[b]", robust=True)
    assert wald.z == pytest.approx(z, rel=1e-9)
    assert wald.std_error == pytest.approx(reference.std_error("arm[b]", robust=True), rel=1e-9)
    assert wald.p_value == pytest.approx(math.erfc(abs(z) / math.sqrt(2.0)), rel=1e-9)
    assert wald.p_value < 0.05
    record = _json_roundtrip(contrast.to_dict())
    assert record["wald"]["p_value"] == pytest.approx(wald.p_value, rel=1e-9)
    assert record["wald"]["problem"] is None and record["wald"]["robust"] is True


def test_rate_contrast_wald_test_on_a_seeded_null_example() -> None:
    wald = rate_contrast(_two_arm_rows(200, 0.2, 0.2, seed=12), "arm", "a", "b", 20, seed=1).wald
    assert wald.problem is None and wald.p_value is not None
    assert wald.p_value > 0.05


def test_wald_test_reports_a_p_value_it_cannot_compute_explicitly() -> None:
    plain = fit_logistic(TABLE_X, TABLE_Y, names=["arm"])
    missing = stats.wald_test(plain, "arm")
    assert missing.p_value is None and missing.z is None and missing.std_error is None
    assert missing.problem == "no_cluster_robust_se" and missing.detail
    model_based = stats.wald_test(plain, "arm", robust=False)
    assert model_based.problem is None and not model_based.robust
    assert model_based.p_value == pytest.approx(
        math.erfc(abs(plain.wald_z("arm")) / math.sqrt(2.0)), rel=1e-12
    )
    one = fit_logistic(TABLE_X, TABLE_Y, names=["arm"], clusters=["only"] * 12)
    assert stats.wald_test(one, "arm").problem == "one_cluster"
    stalled = fit_logistic(
        TABLE_X,
        TABLE_Y,
        names=["arm"],
        clusters=list(range(12)),
        max_iter=1,
        check_separation=False,
    )
    assert stats.wald_test(stalled, "arm").problem == "not_converged"
    twin = fit_logistic(np.column_stack([TABLE_X, TABLE_X]), TABLE_Y, clusters=list(range(12)))
    assert stats.wald_test(twin, "x1").problem == "rank_deficient"
    clustered = fit_logistic(TABLE_X, TABLE_Y, names=["arm"], clusters=list(range(12)))
    assert clustered.robust_se is not None
    zero = dataclasses.replace(clustered, robust_se=np.zeros_like(clustered.robust_se))
    assert stats.wald_test(zero, "arm").problem == "zero_se"
    with pytest.raises(StatsError, match="no term"):
        stats.wald_test(plain, "missing")
    record = _json_roundtrip(missing.to_dict())
    assert record["p_value"] is None and record["problem"] == "no_cluster_robust_se"


def test_rate_contrast_under_separation_reports_no_p_value() -> None:
    # Arm b never violates and arm a violates once in 200: the estimate is held only by the
    # ridge and its CR1 Wald statistic is spuriously huge, so no p-value is reported.
    rows = _two_arm_rows(100, 0.0, 0.0, seed=1)
    rows[0]["violated"] = True  # session s000, arm a
    contrast = rate_contrast(rows, "arm", "a", "b", 20, seed=1)
    assert contrast.fit.separation and contrast.fit.robust_se is not None
    assert abs(contrast.fit.wald_z("arm[b]", robust=True)) > 5.0  # what a Wald test would say
    assert contrast.wald.p_value is None and contrast.wald.problem == "separation"
    assert contrast.difference.estimate == pytest.approx(-1 / 200)


def _score_by_hand(arm: np.ndarray, y: np.ndarray, cluster: np.ndarray) -> tuple[float, float]:
    """Score test of the arm at the intercept-only null fit with an exchangeable
    within-cluster correlation, written out separately: ``(p_value, rho)``."""
    p0 = y.mean()
    pq = p0 * (1.0 - p0)
    xt = arm - arm.mean()
    e = (y - p0) / math.sqrt(pq)
    groups = np.unique(cluster)
    cross = sum(float(e[cluster == g].sum() ** 2 - (e[cluster == g] ** 2).sum()) for g in groups)
    pairs = sum(int((cluster == g).sum()) * (int((cluster == g).sum()) - 1) for g in groups)
    rho = min(max(cross / pairs, 0.0), stats.SCORE_RHO_MAX) if pairs else 0.0
    together = sum(float(xt[cluster == g].sum()) ** 2 for g in groups)
    variance = pq * ((1.0 - rho) * float(xt @ xt) + rho * together)
    z = float(xt @ (y - p0)) / math.sqrt(variance)
    return math.erfc(abs(z) / math.sqrt(2.0)), rho


def test_score_test_matches_a_hand_computation_and_handles_edges() -> None:
    rows = _two_arm_rows(60, 0.2, 0.4, seed=7)
    arm = np.array([1.0 if r["arm"] == "b" else 0.0 for r in rows])
    y = np.array([float(r["violated"]) for r in rows])
    cluster = np.array([r["session_id"] for r in rows])
    found = stats.score_test(arm, y, list(cluster), term="arm[b]")
    assert found.problem is None and found.n_clusters == 60
    p_value, rho = _score_by_hand(arm, y, cluster)
    assert found.p_value == pytest.approx(p_value, rel=1e-9)
    assert found.rho == pytest.approx(rho, abs=1e-12)
    # With a covariate in the null model the residualized score is still computed.
    adjusted = stats.score_test(
        arm, y, list(cluster), np.array([float(r["position"]) for r in rows]), term="arm[b]"
    )
    assert adjusted.problem is None and 0.0 < (adjusted.p_value or 0.0) < 1.0
    assert stats.score_test(arm, np.zeros_like(y), list(cluster)).problem == "no_events"
    assert stats.score_test(arm, np.ones_like(y), list(cluster)).problem == "all_events"
    assert stats.score_test(arm, y, ["one"] * len(y)).problem == "one_cluster"
    flat = stats.score_test(np.ones_like(y), y, list(cluster))
    assert flat.problem == "rank_deficient" and flat.p_value is None
    record = _json_roundtrip(found.to_dict())
    assert record["p_value"] == pytest.approx(found.p_value) and "score" in record["test"]
    assert record["rho"] == pytest.approx(found.rho)
    with pytest.raises(StatsError, match="outcomes"):
        stats.score_test(arm, y + 2.0, list(cluster))


def _minority_without_violations() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Synthetic: 40 plans with three rows of type a and one of type b each; five
    violations, all of type a, in five different plans."""
    x, y, cluster = [], [], []
    for g in range(40):
        for j in range(4):
            x.append(1.0 if j == 3 else 0.0)
            y.append(1.0 if j == 0 and g < 5 else 0.0)
            cluster.append(f"p{g:02d}")
    return np.array(x), np.array(y), cluster


def test_score_test_with_singleton_clusters_is_the_pearson_chi_square() -> None:
    # Without within-cluster pairs there is no correlation to estimate, and the score test
    # at the overall rate is the Pearson chi-square test of the two-by-two table.
    x, y, _ = _minority_without_violations()
    found = stats.score_test(x, y, [f"s{i}" for i in range(len(y))])
    a, b = float((x * y).sum()), float((x * (1 - y)).sum())
    c, d = float(((1 - x) * y).sum()), float(((1 - x) * (1 - y)).sum())
    chi2 = len(y) * (a * d - b * c) ** 2 / ((a + b) * (c + d) * (a + c) * (b + d))
    assert found.rho == 0.0
    assert found.z is not None and found.z**2 == pytest.approx(chi2, rel=1e-9)
    assert found.p_value == pytest.approx(math.erfc(math.sqrt(chi2 / 2.0)), rel=1e-9)


def test_score_test_does_not_read_a_minority_arm_without_violations_as_a_difference() -> None:
    # Synthetic: five violations of type a and none of the rarer type b. Fisher's exact test
    # gives about 0.33. The CR1 variance of the score sums each plan's realized score, so it
    # sees no variability from type b, where no violation happened, and it gave p of about
    # 0.027 here (a false difference: in a synthetic null calibration on the E1 plans, the
    # CR1 score test rejected more than a quarter of the true nulls whose rarer type had no
    # violation). The exchangeable variance accounts for violations type b could have had.
    x, y, cluster = _minority_without_violations()
    found = stats.score_test(x, y, cluster)
    assert found.problem is None and found.p_value is not None
    assert found.p_value > 0.15
    assert found.p_value == pytest.approx(_score_by_hand(x, y, np.array(cluster))[0], rel=1e-9)


def test_score_test_variance_grows_with_within_cluster_correlation_of_a_plan_level_arm() -> None:
    # Synthetic: the arm is fixed per plan (as E3 and E4 randomize), and outcomes are shared
    # within plans; the exchangeable variance carries the design effect, so the same score
    # is less significant than if the rows were independent.
    rng = np.random.default_rng(5)
    x, y, cluster = [], [], []
    for g in range(30):
        shared = rng.random() < 0.3
        for _ in range(6):
            x.append(float(g % 2))
            y.append(float(shared and rng.random() < 0.8))
            cluster.append(f"p{g}")
    xs, ys = np.array(x), np.array(y)
    clustered = stats.score_test(xs, ys, cluster)
    independent = stats.score_test(xs, ys, [f"s{i}" for i in range(len(ys))])
    assert clustered.rho is not None and clustered.rho > 0.3
    assert clustered.p_value is not None and independent.p_value is not None
    assert clustered.p_value > independent.p_value
    assert clustered.p_value == pytest.approx(_score_by_hand(xs, ys, np.array(cluster))[0])


def test_rate_contrast_score_test_stays_valid_under_separation() -> None:
    # Arm b never violates and arm a violates at 30 of 200 opportunities: the Wald test is
    # withheld (separation), but the score test at the null fit needs no finite estimate
    # under the alternative and gives a valid, small p-value.
    rows = _two_arm_rows(100, 0.3, 0.0, seed=1)
    contrast = rate_contrast(rows, "arm", "a", "b", 20, seed=1)
    assert contrast.wald.problem == "separation" and contrast.wald.p_value is None
    assert contrast.score.problem is None and contrast.score.p_value is not None
    assert contrast.score.p_value < 1e-4
    record = _json_roundtrip(contrast.to_dict())
    assert record["score"]["p_value"] == pytest.approx(contrast.score.p_value)
    # Under the null its size is close to nominal (synthetic, seeded).
    rejected = sum(
        (
            rate_contrast(_two_arm_rows(40, 0.2, 0.2, seed=s), "arm", "a", "b", 1, seed=1).score
        ).p_value
        < 0.05
        for s in range(200)
    )
    assert 2 <= rejected <= 20


def test_rate_contrast_odds_ratio_leaves_out_draws_with_a_one_sided_arm() -> None:
    # Synthetic: arm b violates in two sessions only, so a draw that misses both has no
    # finite log odds ratio; it fails for the odds ratio only, and the bounds no longer
    # move with the ridge.
    rows = _two_arm_rows(40, 0.3, 0.0, seed=2)
    hits = 0
    for row in rows:
        if row["arm"] == "b" and row["session_id"] in {"s001", "s003"} and hits < 2:
            row["violated"] = True
            hits += 1
    found = {l2: rate_contrast(rows, "arm", "a", "b", 300, seed=3, l2=l2) for l2 in (1e-6, 1e-8)}
    first, second = found[1e-6], found[1e-8]
    assert not first.fit.separation and first.log_odds_ratio.not_estimable is None
    assert first.log_odds_ratio.n_failed > 0 and first.difference.n_failed == 0
    assert first.log_odds_ratio.lo == pytest.approx(_num(second.log_odds_ratio.lo), abs=1e-3)
    assert first.difference.lo == second.difference.lo
    # A full-data arm without violations has no odds ratio at all (not estimable).
    quiet = [dict(r, violated=False) if r["arm"] == "b" else r for r in rows]
    separated = rate_contrast(quiet, "arm", "a", "b", 50, seed=3)
    assert separated.log_odds_ratio.not_estimable == "separation"
    assert separated.odds_ratio.not_estimable == "separation"
    record = _json_roundtrip(separated.to_dict())
    assert record["odds_ratio"]["odds_ratio"] is None and record["odds_ratio"]["lo"] is None
    assert record["difference"]["difference"] is not None


def test_rate_contrast_counts_family_sessions_and_resamples_plans() -> None:
    rows = _two_family_rows(40, seed=3)
    contrast = rate_contrast(rows, "arm", "a", "b", 30, seed=1, use_risk_set=True)
    manual = _manual_risk_set(rows)
    assert contrast.a.n_opportunities + contrast.b.n_opportunities == len(manual)
    # 20 plans per arm, each run by both families.
    assert (contrast.a.n_sessions, contrast.b.n_sessions) == (40, 40)
    assert contrast.difference.n_clusters == 40 and contrast.difference.strata == "arm"
    assert contrast.fit.n_clusters == 40 and contrast.wald.n_clusters == 40
    every = rate_contrast(rows, "arm", "a", "b", 30, seed=1)
    assert every.a.n_opportunities == 20 * 2 * 5 and every.a.n_sessions == 40


# --------------------------------------------------------------------------- recall


def _recall_rows(
    n_sessions: int = 400, seed: int = 1, effect: float = -0.8
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for s in range(n_sessions):
        events = np.sort(rng.choice(np.arange(1, 151), size=2, replace=False))
        for p in np.sort(rng.choice(151, size=4, replace=False)):
            count = int((events <= p).sum())
            eta = 2.0 + effect * count
            rows.append(
                {
                    "family": "syn",
                    "session_id": f"s{s}",
                    "position": int(p),
                    "compaction_count": count,
                    "recalled_before": bool(rng.random() < 1 / (1 + math.exp(-eta))),
                    "violated": False,
                }
            )
    return rows


def test_recall_rows_and_cells_by_compaction_count() -> None:
    rows = [
        {"family": "f", "session_id": "a", "compaction_count": 0, "recalled_before": True},
        {"family": "f", "session_id": "a", "compaction_count": 1, "recalled_before": False},
        {"family": "f", "session_id": "b", "compaction_count": 1, "recalled_before": True},
        {"family": "f", "session_id": "b", "compaction_count": 1, "recalled_before": None},
        {"family": "f", "session_id": "c", "compaction_count": 2.0, "recalled_before": False},
    ]
    assert len(recall_rows(rows)) == 4
    cells = recall_by_compaction(rows)
    assert [(c.compaction_count, c.recall.k, c.recall.n, c.n_sessions) for c in cells] == [
        (0, 1, 1, 1),
        (1, 1, 2, 2),
        (2, 0, 1, 1),
    ]
    # Two families' sessions of one plan are two sessions, not one.
    shared = [dict(row, family="g") for row in rows] + rows
    assert [c.n_sessions for c in recall_by_compaction(shared)] == [2, 4, 2]
    recall = recall_effect(_two_family_rows(30, seed=2), 20, seed=1)
    assert recall.n_clusters == 30  # clustered on the plan both families ran
    assert sum(c.recall.n for c in recall.cells) == recall.n_rows == 300
    assert cells[1].recall.lo == wilson_interval(1, 2).lo
    assert "ignores clustering" in _json_roundtrip(cells[0].to_dict())["recall"]["note"]
    with pytest.raises(StatsError, match="no rows with a recall outcome"):
        recall_by_compaction([{"session_id": "a", "compaction_count": 0, "recalled_before": None}])
    with pytest.raises(StatsError, match="whole number"):
        recall_by_compaction(
            [{"session_id": "a", "compaction_count": 1.5, "recalled_before": True}]
        )
    with pytest.raises(StatsError, match="boolean"):
        recall_rows([{"session_id": "a", "recalled_before": "yes"}])


def test_recall_is_grouped_by_the_compactions_the_probe_saw() -> None:
    # Rows from the runner carry compactions_before_probe: a compaction at the decision
    # point's start ran after its probe, so that probe's recall belongs with one fewer.
    rows = [
        {"family": "f", "session_id": "a", "compaction_count": 1, "recalled_before": True},
        {"family": "f", "session_id": "a", "compaction_count": 2, "recalled_before": False},
        {"family": "f", "session_id": "b", "compaction_count": 1, "recalled_before": False},
    ]
    probed = [
        dict(rows[0], compactions_before_probe=0),
        dict(rows[1], compactions_before_probe=2),
        dict(rows[2], compactions_before_probe=1),
    ]
    cells = recall_by_compaction(probed)
    assert [(c.compaction_count, c.recall.k, c.recall.n) for c in cells] == [
        (0, 1, 1),
        (1, 0, 1),
        (2, 0, 1),
    ]
    assert {c.count_field for c in cells} == {"compactions_before_probe"}
    assert _json_roundtrip(cells[0].to_dict())["count_field"] == "compactions_before_probe"
    # Rows banked before the field existed are grouped by compaction_count, and say so.
    legacy = recall_by_compaction(rows)
    assert [c.compaction_count for c in legacy] == [1, 2]
    assert {c.count_field for c in legacy} == {"compaction_count"}
    effect_rows = [
        dict(r, compactions_before_probe=r["compaction_count"])
        for r in _two_family_rows(30, seed=2)
    ]
    effect = recall_effect(effect_rows, 20, seed=1)
    assert effect.term == "compactions_before_probe"
    assert recall_effect(_two_family_rows(30, seed=2), 20, seed=1).term == "compaction_count"


def test_recall_effect_recovers_a_known_compaction_effect() -> None:
    rows = _recall_rows()
    rows.append({"session_id": "x", "position": 3, "compaction_count": 0, "recalled_before": None})
    effect = recall_effect(rows, 150, seed=2)
    assert effect.n_rows == len(rows) - 1 and effect.n_clusters == 400
    assert effect.log_odds_ratio.estimate == pytest.approx(
        effect.fit.coefficient("compaction_count"), rel=1e-6
    )
    lo, hi = effect.log_odds_ratio.lo, effect.log_odds_ratio.hi
    assert isinstance(lo, float) and isinstance(hi, float) and lo < -0.8 < hi
    assert effect.odds_ratio.estimate == pytest.approx(
        math.exp(effect.fit.coefficient("compaction_count")), rel=1e-6
    )
    assert [c.compaction_count for c in effect.cells] == [0, 1, 2]
    record = _json_roundtrip(effect.to_dict())
    assert record["outcome"] == "recalled_before" and record["term"] == "compaction_count"
    with pytest.raises(StatsError, match="compaction_count among the covariates"):
        recall_effect(rows, 10, seed=1, covariates=("position",))
    with pytest.raises(StatsError, match="no rows with a recall outcome"):
        recall_effect([{"session_id": "a", "recalled_before": None}], 10, seed=1)


# --------------------------------------------------------------------------- power


def test_power_by_simulation_type_one_error_and_power() -> None:
    null = power_by_simulation(0.0, 60, 4, 0.2, 200, 9)
    assert null.compaction.lo <= 0.05 <= null.compaction.hi
    assert null.contrast.lo <= 0.05 <= null.contrast.hi
    strong = power_by_simulation(1.0, 60, 4, 0.2, 100, 9)
    assert strong.compaction.value > 0.8 and strong.contrast.value > 0.8
    assert power_by_simulation(1.0, 60, 4, 0.2, 100, 9) == strong  # deterministic
    record = _json_roundtrip(strong.to_dict())
    assert record["synthetic"] is True and record["note"].startswith("Synthetic")
    assert record["inputs"]["n_sessions"] == 60 and record["compaction"]["n"] == 100


def test_power_by_simulation_options_and_unusable_fits() -> None:
    varied = power_by_simulation(
        0.5, 30, 3, 0.1, 20, 4, 0.1, max_position=60, position_slope=0.01, session_sd=0.5
    )
    assert varied.alpha == 0.1 and varied.session_sd == 0.5 and varied.max_position == 60
    sparse = power_by_simulation(0.5, 2, 1, 0.001, 5, 1, max_position=5)
    assert sparse.compaction_unusable > 0 and sparse.contrast_unusable > 0
    assert sparse.compaction.k == 0


def test_power_by_simulation_contrast_arms_can_split_one_familys_sessions() -> None:
    split = power_by_simulation(1.0, 60, 4, 0.2, 30, 9, arm_sessions=30)
    whole = power_by_simulation(1.0, 60, 4, 0.2, 30, 9)
    assert (split.arm_sessions, whole.arm_sessions) == (30, 60)
    assert split.compaction == whole.compaction  # the hazard design is not affected
    record = _json_roundtrip(split.to_dict())
    assert record["inputs"]["arm_sessions"] == 30 and record["inputs"]["n_sessions"] == 60
    assert record["contrast"]["design"].startswith("30 sessions per arm")
    for bad in (0, -1, True):
        with pytest.raises(StatsError, match="arm_sessions"):
            power_by_simulation(1.0, 60, 4, 0.2, 5, 9, arm_sessions=bad)


def test_power_contrast_simulates_h2a_after_a_compaction_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # H2a compares the arms among decision points after at least one compaction, where the
    # arms can differ; the simulation shifts and fits only those, with log1p(position).
    rng = np.random.default_rng(1)
    _, cnt, y, _ = stats._simulate(rng, 30, 6, 100, 1, -30.0, 0.0, 0.0, 0.0, after_shift=60.0)
    assert np.array_equal(y, (cnt >= 1).astype(float))  # the shift lands only after one
    fitted: list[int] = []
    real = stats.fit_logistic

    def spy(x: Any, y: Any, *args: Any, **kwargs: Any) -> stats.LogisticFit:
        if kwargs.get("names") == ("arm",):
            fitted.append(len(y))
        return real(x, y, *args, **kwargs)

    monkeypatch.setattr(stats, "fit_logistic", spy)
    result = power_by_simulation(
        0.7, 20, 6, 0.1, 5, seed=3, max_position=100, compactions_per_session=1
    )
    # 20 sessions per arm with 6 decision points each; those before a compaction are out.
    assert fitted and all(n < 2 * 20 * 6 for n in fitted)
    record = result.to_dict()
    assert "after at least one compaction" in record["contrast"]["design"]
    assert record["inputs"]["position_transform"] == "log1p"
    assert "log1p(position)" in record["compaction"]["design"]
    linear = power_by_simulation(0.7, 20, 6, 0.1, 2, seed=3, position_transform="linear")
    assert "terms position and compaction_count" in linear.to_dict()["compaction"]["design"]


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        ((0.5, 10, 3, 0.0, 10, 1), {}),
        ((0.5, 10, 3, 1.0, 10, 1), {}),
        ((0.5, 10, 3, 0.2, 10, 1, 0.0), {}),
        ((math.nan, 10, 3, 0.2, 10, 1), {}),
        ((0.5, 10, 3, 0.2, 10, 1), {"position_slope": math.inf}),
        ((0.5, 10, 3, 0.2, 10, 1), {"session_sd": -1.0}),
        ((0.5, 1, 3, 0.2, 10, 1), {}),
        ((0.5, 10, 3, 0.2, 0, 1), {}),
        ((0.5, 10, 0, 0.2, 10, 1), {}),
        ((0.5, 10, 3, 0.2, 10, -1), {}),
        ((0.5, 10, 3, 0.2, 10, 1), {"max_position": 1}),
        ((0.5, 10, 2, 0.2, 10, 1), {"max_position": 1, "compactions_per_session": 2}),
        ((0.5, 10, 3, 0.2, 10, 1), {"compactions_per_session": 0}),
        ((0.5, True, 3, 0.2, 10, 1), {}),
    ],
)
def test_power_by_simulation_rejects_bad_input(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    with pytest.raises(StatsError):
        power_by_simulation(*args, **kwargs)


# --------------------------------------------------------------------------- module


def test_public_names_exist_and_notes_follow_the_style_rule() -> None:
    for name in stats.__all__:
        assert hasattr(stats, name), name
    for note in (stats.KM_NOTE, stats.POWER_NOTE, stats.NAIVE_NOTE):
        assert "\u2014" not in note
    assert "Synthetic" in stats.POWER_NOTE


def test_bootstrap_helpers_check_their_settings_even_without_draws() -> None:
    # Without a violation nothing is bootstrapped, but bad settings still fail loudly.
    rows = [dict(r, violated=False) for r in _simulated_hazard_rows(20, seed=2)]
    for bad in (0, -1):
        with pytest.raises(StatsError, match="n_boot"):
            bootstrap_coefficients(rows, bad, seed=1, categorical=())
        with pytest.raises(StatsError, match="n_boot"):
            bootstrap_half_life(rows, 150, {"compaction_count": 0}, bad, seed=1, categorical=())
    with pytest.raises(StatsError, match="explicit seed"):
        bootstrap_coefficients(rows, 10, categorical=())

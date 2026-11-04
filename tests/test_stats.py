"""Tests for :mod:`halflife.stats` against values derived by hand or by independent code.

Every expected number here is either worked out by hand (the arithmetic is in the
comments) or recomputed in the test by a separate implementation (numpy's
percentile, a finite-difference Hessian, a replayed random generator). The
simulated datasets are synthetic test fixtures, not model results.
"""

from __future__ import annotations

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
                    "session_id": f"s{s:04d}",
                    "position": int(p),
                    "compaction_count": count,
                    "constraint_type": ctype,
                    "violated": bool(rng.random() < 1.0 / (1.0 + math.exp(-eta))),
                }
            )
    return rows


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
        {"session_id": "b", "position": 5, "violated": False},
        {"session_id": "a", "position": 9, "violated": False, "tag": "tie-first"},
        {"session_id": "a", "position": 2, "violated": False},
        {"session_id": "a", "position": 9, "violated": True, "tag": "tie-second"},
        {"session_id": "a", "position": 12, "violated": True},
        {"session_id": "b", "position": 1, "violated": np.bool_(False)},
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
        {"session_id": "s", "constraint_id": "c1", "position": 1, "violated": True},
        {"session_id": "s", "constraint_id": "c1", "position": 4, "violated": False},
        {"session_id": "s", "constraint_id": "c2", "position": 3, "violated": False},
    ]
    assert len(risk_set(rows)) == 1
    assert len(risk_set(rows, by=("session_id", "constraint_id"))) == 2
    assert risk_set([]) == []


@pytest.mark.parametrize(
    "row",
    [
        {"position": 1, "violated": True},
        {"session_id": "s", "violated": True},
        {"session_id": "s", "position": 1, "violated": None},
        {"session_id": "s", "position": 1, "violated": 2},
        {"session_id": "s", "position": -1, "violated": True},
        {"session_id": "s", "position": "3", "violated": True},
        {"session_id": "s", "position": float("inf"), "violated": True},
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
            "session_id": "rare",
            "position": 5,
            "compaction_count": 1,
            "constraint_type": "rare",
            "violated": True,
        }
    ]
    result = bootstrap_half_life(
        rows, 150, {"compaction_count": 0, "constraint_type": "rare"}, 100, seed=2
    )
    # "rare" sits in one session, so draws without it fail instead of guessing.
    assert result.interval.n_failed > 0
    flat = bootstrap_half_life(
        rows, 150, {"compaction_count": 0, "constraint_type": "prohibition"}, 100, seed=2
    )
    assert flat.interval.estimate == NotReached(150)
    assert flat.interval.share_not_reached > 0.5
    record = _json_roundtrip(flat.to_dict())
    assert record["half_life"] is None and record["not_reached_within"] == 150
    with pytest.raises(StatsError, match="no level"):
        bootstrap_half_life(rows, 150, {"compaction_count": 0, "constraint_type": "x"}, 10, seed=2)


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
    rows += [
        {
            "session_id": "zz-only",
            "position": p,
            "compaction_count": 0,
            "constraint_type": "a_sparse",
            "violated": False,
        }
        for p in (3, 9)
    ]
    found = bootstrap_coefficients(rows, 120, seed=6)
    full = fit_hazard(rows)
    assert set(found) == {
        "intercept",
        "position",
        "compaction_count",
        "constraint_type[invariant]",
        "constraint_type[procedure]",
    }
    for name, interval in found.items():
        assert interval.estimate == pytest.approx(full.coefficients[name], rel=1e-6, abs=1e-9)
    # The sorted-first level "a_sparse" is the reference and sits in one session, so
    # intercept and factor contrasts fail on draws without it; slopes never do.
    assert found["intercept"].n_failed > 0
    assert found["constraint_type[procedure]"].n_failed == found["intercept"].n_failed
    assert found["position"].n_failed == 0


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
        {"session_id": "a", "position": 4, "violated": False},
        {"session_id": "a", "position": 9, "violated": True},
        {"session_id": "b", "position": 3, "violated": False},
        {"session_id": "b", "position": 12, "violated": False},
        {"session_id": "c", "position": 2, "violated": True},
    ]
    km = kaplan_meier_rows(rows)
    triples = analysis.time_to_first_violation(rows)
    expected = kaplan_meier([t for _, t, _ in triples], [e for _, _, e in triples])
    assert km == expected
    assert km.times == (2.0, 9.0, 12.0) and km.n_events == 2
    with pytest.raises(StatsError, match="no opportunity rows"):
        kaplan_meier_rows([])
    assert isinstance(km, KaplanMeier)


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
                    "session_id": f"{arm}{i // 2}",
                    "mitigation": arm,
                    "violated": bool(v),
                    "position": 10 * (i % 2) + i,
                    "constraint_type": "prohibition" if i % 3 else "invariant",
                }
            )
    rows.append({"session_id": "x0", "mitigation": "guard", "violated": True, "position": 1})
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
        {"session_id": "s1", "arm": "a", "violated": False, "position": 1},
        {"session_id": "s1", "arm": "b", "violated": True, "position": 2},
        {"session_id": "s1", "arm": "a", "violated": True, "position": 3},
        {"session_id": "s1", "arm": "b", "violated": False, "position": 4},
        {"session_id": "s2", "arm": "a", "violated": False, "position": 1},
    ]
    contrast = rate_contrast(rows, "arm", "a", "b", 200, seed=3)
    assert not contrast.stratified
    assert contrast.difference.n_failed > 0


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
        {"session_id": "a", "compaction_count": 0, "recalled_before": True},
        {"session_id": "a", "compaction_count": 1, "recalled_before": False},
        {"session_id": "b", "compaction_count": 1, "recalled_before": True},
        {"session_id": "b", "compaction_count": 1, "recalled_before": None},
        {"session_id": "c", "compaction_count": 2.0, "recalled_before": False},
    ]
    assert len(recall_rows(rows)) == 4
    cells = recall_by_compaction(rows)
    assert [(c.compaction_count, c.recall.k, c.recall.n, c.n_sessions) for c in cells] == [
        (0, 1, 1, 1),
        (1, 1, 2, 2),
        (2, 0, 1, 1),
    ]
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

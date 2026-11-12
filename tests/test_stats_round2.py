"""Tests for the round-two decision tests in :mod:`halflife.stats` (DECISIONS.md D-050).

The stratified score test is checked against a separate hand computation, against the
unstratified test on the rows that carry information, and on seeded synthetic nulls
(labeled synthetic; not model results).
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pytest

from halflife import stats
from halflife.stats import StatsError, hazard_score_test, rate_contrast, score_test


def _stratified_by_hand(
    x: np.ndarray, y: np.ndarray, cluster: np.ndarray, strata: np.ndarray
) -> tuple[float, float]:
    """Score test of ``x`` with an intercept per stratum, written out separately.

    A stratum without violations, or with only violations, is left out (its rows have
    fitted probability 0 or 1 at the null fit, so they add nothing). Within a stratum every
    row has the same null probability, so residualizing ``x`` on the stratum's intercept
    subtracts the stratum mean. Returns ``(p_value, rho)``.
    """
    keep = np.zeros(len(y), dtype=bool)
    p = np.zeros(len(y))
    xt = np.zeros(len(y))
    for s in np.unique(strata):
        mask = strata == s
        k = float(y[mask].sum())
        if k in (0.0, float(mask.sum())):
            continue
        p[mask] = y[mask].mean()
        xt[mask] = x[mask] - x[mask].mean()
        keep |= mask
    x, y, cluster, p, xt = x[keep], y[keep], cluster[keep], p[keep], xt[keep]
    w = p * (1.0 - p)
    e = (y - p) / np.sqrt(w)
    groups = np.unique(cluster)
    cross = sum(float(e[cluster == g].sum() ** 2 - (e[cluster == g] ** 2).sum()) for g in groups)
    pairs = sum(int((cluster == g).sum()) * (int((cluster == g).sum()) - 1) for g in groups)
    rho = min(max(cross / pairs, 0.0), stats.SCORE_RHO_MAX) if pairs else 0.0
    root = np.sqrt(w) * xt
    together = sum(float(root[cluster == g].sum()) ** 2 for g in groups)
    variance = (1.0 - rho) * float(root @ root) + rho * together
    z = float(xt @ (y - p)) / math.sqrt(variance)
    return math.erfc(abs(z) / math.sqrt(2.0)), rho


def _two_family_arrays(
    seed: int, n_plans: int = 30, rates: tuple[float, float] = (0.15, 0.35)
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic: two families run the same plans (one cluster per plan, holding both
    families' rows, which share a plan effect); the arm alternates by plan, and the
    families' base rates differ."""
    rng = np.random.default_rng(seed)
    shared = rng.normal(0.0, 1.5, size=n_plans)
    x, y, cluster, strata = [], [], [], []
    for f, rate in enumerate(rates):
        for g in range(n_plans):
            for _ in range(4):
                eta = math.log(rate / (1 - rate)) + shared[g]
                x.append(float(g % 2))
                y.append(float(rng.random() < 1.0 / (1.0 + math.exp(-eta))))
                cluster.append(f"p{g:02d}")
                strata.append(f"fam-{f}")
    return np.array(x), np.array(y), np.array(cluster), np.array(strata)


def test_stratified_score_test_matches_a_hand_computation() -> None:
    x, y, cluster, strata = _two_family_arrays(3)
    found = score_test(x, y, list(cluster), term="arm[b]", strata=list(strata))
    p_value, rho = _stratified_by_hand(x, y, cluster, strata)
    assert found.problem is None and found.p_value == pytest.approx(p_value, rel=1e-9)
    assert found.rho == pytest.approx(rho, abs=1e-12) and rho > 0.0
    assert found.n_strata == 2 and found.strata_skipped == ()
    assert found.n_rows == len(y) and found.n_events == int(y.sum())
    assert found.n_clusters == 30  # plans, each holding both families' rows
    record = found.to_dict()
    assert record["n_strata"] == 2 and record["strata_skipped"] == []
    # Unstratified, the null fit is the pooled rate, and the test differs.
    pooled = score_test(x, y, list(cluster), term="arm[b]")
    assert pooled.n_strata is None and pooled.p_value != pytest.approx(found.p_value)


def test_a_stratum_without_violations_contributes_nothing() -> None:
    # Synthetic: a third family never violates. Stratified, its rows drop out and the test
    # equals the test without them. Unstratified, a dummy for that family separates the null
    # model; its rows sit at fitted value 0, leave as boundary rows, and the limit of the
    # ridge-free test is the same stratified test.
    x, y, cluster, strata = _two_family_arrays(5)
    quiet = len(y) // 2
    x3 = np.concatenate([x, x[:quiet]])
    y3 = np.concatenate([y, np.zeros(quiet)])
    c3 = np.concatenate([cluster, cluster[:quiet]])
    s3 = np.concatenate([strata, np.array(["fam-quiet"] * quiet)])
    two = score_test(x, y, list(cluster), strata=list(strata))
    three = score_test(x3, y3, list(c3), strata=list(s3))
    assert three.p_value == pytest.approx(two.p_value, rel=1e-12)
    assert three.rho == pytest.approx(two.rho, rel=1e-12)
    assert three.strata_skipped == ("fam-quiet",) and three.n_strata == 2
    assert three.n_rows == len(y)
    dummies = np.column_stack([(s3 == f).astype(float) for f in ("fam-1", "fam-quiet")])
    separated = score_test(x3, y3, list(c3), dummies)
    assert separated.problem is None and separated.n_boundary == quiet
    # Equal up to the ridge, which the unstratified null fit carries and an intercept-only
    # stratum fit does not.
    assert separated.p_value == pytest.approx(two.p_value, rel=1e-6)
    assert separated.rho == pytest.approx(two.rho, rel=1e-6)
    # A null-design column that is constant within every stratum (a family dummy) adds
    # nothing to a stratified null fit and is dropped instead of making it rank deficient.
    dummy = (strata == "fam-1").astype(float)
    assert score_test(x, y, list(cluster), dummy, strata=list(strata)).p_value == pytest.approx(
        two.p_value, rel=1e-9
    )


def test_stratified_score_test_edge_cases() -> None:
    x, y, cluster, strata = _two_family_arrays(7)
    # One family has only violations and the other none: no stratum has anything to test.
    split = np.where(strata == "fam-0", 1.0, 0.0)
    nothing = score_test(x, split, list(cluster), strata=list(strata))
    assert nothing.problem == "no_stratum_with_variation" and nothing.p_value is None
    assert set(nothing.strata_skipped) == {"fam-0", "fam-1"}
    # A term that is constant within each stratum is collinear with the strata.
    within = score_test(split, y, list(cluster), strata=list(strata))
    assert within.problem == "rank_deficient"
    # The only stratum with violations holds a single cluster, which cannot be clustered.
    alone = (strata == "fam-0") & (cluster == "p00")
    yy = np.zeros(len(y))
    yy[np.flatnonzero(alone)[0]] = 1.0
    one = score_test(x, yy, list(cluster), strata=list(np.where(alone, "a", "b")))
    assert one.problem == "one_cluster" and one.strata_skipped == ("b",)
    # A null model that separates every row leaves nothing to test.
    perfect = score_test(x, y, list(cluster), y.copy())
    assert perfect.problem == "null_separation" and perfect.p_value is None
    with pytest.raises(StatsError, match="same nonzero length"):
        score_test(x, y, list(cluster), strata=list(strata)[:-1])
    with pytest.raises(StatsError, match="null_design has non-finite"):
        score_test(x, y, list(cluster), np.full(len(y), math.nan))
    with pytest.raises(StatsError, match="x has non-finite"):
        score_test(np.full(len(y), math.inf), y, list(cluster), strata=list(strata))


# --------------------------------------------------------------------------- hazard tests


TYPES = ("invariant", "procedure", "prohibition")


def _hazard_rows(
    seed: int,
    *,
    n_plans: int = 40,
    slope: float = 0.0,
    compaction: float = 0.0,
    base: tuple[float, float] = (-2.0, -1.4),
    quiet_type: str | None = None,
) -> list[dict[str, Any]]:
    """Synthetic opportunities of two families on shared plans (not model data).

    Positions differ by family, the compaction count rises with position (as token-budget
    compactions make it), and the hazard is ``base + slope * log1p(position) + compaction
    * count`` plus a rule-type effect; ``quiet_type`` never violates in family ``fa``.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for f, family in enumerate(("fa", "fb")):
        for plan in range(n_plans):
            position = 0
            for j in range(5):
                position += int(rng.integers(3, 25))
                count = position // 17 + int(rng.random() < 0.3)
                ctype = TYPES[(plan + j) % 3]
                eta = (
                    base[f]
                    + slope * math.log1p(position)
                    + compaction * count
                    + {"invariant": 0.4, "procedure": -0.3, "prohibition": 0.0}[ctype]
                )
                violated = bool(rng.random() < 1.0 / (1.0 + math.exp(-eta)))
                if family == "fa" and ctype == quiet_type:
                    violated = False
                rows.append(
                    {
                        "family": family,
                        "session_id": f"p{plan:03d}",
                        "position": position,
                        "compaction_count": count,
                        "constraint_type": ctype,
                        "strategy": "sham" if plan % 2 else "full_summary",
                        "violated": violated,
                    }
                )
    return rows


def _manual_design(
    rows: list[dict[str, Any]], nuisance: str, factors: tuple[str, ...], strata: bool
) -> np.ndarray:
    """Null-design columns built by hand: per family block, the numeric nuisance column and
    treatment dummies of the factor levels present in that family (first sorted level is
    the reference)."""
    blocks = sorted({r["family"] for r in rows}) if strata else [None]
    columns: list[np.ndarray] = []
    for family in blocks:
        mine = np.array([family is None or r["family"] == family for r in rows], dtype=float)
        columns.append(mine * np.array([float(r[nuisance]) for r in rows]))
        for factor in factors:
            present = sorted({r[factor] for r in rows if family is None or r["family"] == family})
            for level in present[1:]:
                columns.append(mine * np.array([float(r[factor] == level) for r in rows]))
    return np.column_stack(columns)


@pytest.mark.parametrize("pooled", [False, True])
def test_hazard_score_test_matches_the_core_test_on_a_hand_built_design(pooled: bool) -> None:
    rows = _hazard_rows(11)
    if not pooled:
        rows = [r for r in rows if r["family"] == "fa"]
    found = hazard_score_test(
        rows,
        "position",
        ("position", "compaction_count"),
        ("constraint_type", "strategy"),
        "log1p",
        strata="family" if pooled else None,
    )
    cut = stats.risk_set(rows)
    design = _manual_design(cut, "compaction_count", ("constraint_type", "strategy"), pooled)
    expected = score_test(
        [math.log1p(r["position"]) for r in cut],
        [float(r["violated"]) for r in cut],
        [r["session_id"] for r in cut],
        design,
        strata=[r["family"] for r in cut] if pooled else None,
    )
    assert found.test.problem is None and expected.problem is None
    assert found.test.p_value == pytest.approx(expected.p_value, rel=1e-9)
    assert found.term == "log1p(position)" and found.test.term == "log1p(position)"
    assert found.n_rows == len(cut) and found.n_events == sum(r["violated"] for r in cut)
    assert found.levels_without_variation == ()
    record = found.to_dict()
    assert record["strata"] == ("family" if pooled else None)
    assert record["null_covariates"] == ["compaction_count"]
    assert record["null_factors"] == ["constraint_type", "strategy"]
    assert record["score"]["p_value"] == pytest.approx(found.test.p_value)
    # The compaction count is tested the same way, with log1p(position) as the nuisance.
    count = hazard_score_test(
        rows,
        "compaction_count",
        ("position", "compaction_count"),
        ("constraint_type", "strategy"),
        "log1p",
        strata="family" if pooled else None,
    )
    assert count.term == "compaction_count" and count.test.p_value is not None


def test_hazard_score_test_drops_factor_levels_without_violations() -> None:
    # Synthetic: procedure rules never break in family fa. That level's rows have fitted
    # probability 0 at the null fit, so they add nothing: the test equals the test on the
    # rows without them, where a factor model would be separated.
    rows = _hazard_rows(13, quiet_type="procedure")
    covariates, factors = ("position", "compaction_count"), ("constraint_type",)
    found = hazard_score_test(rows, "position", covariates, factors, "log1p", strata="family")
    assert found.levels_without_variation == ("fa: constraint_type[procedure]",)
    cut = stats.risk_set(rows)
    kept = [r for r in cut if not (r["family"] == "fa" and r["constraint_type"] == "procedure")]
    expected = hazard_score_test(kept, "position", covariates, factors, "log1p", strata="family")
    assert found.test.p_value == pytest.approx(expected.test.p_value, rel=1e-12)
    assert found.n_events == expected.n_events and found.n_rows > expected.n_rows
    assert found.to_dict()["levels_without_variation"] == ["fa: constraint_type[procedure]"]
    # Per family the same rule holds, without a stratum label.
    single = hazard_score_test(
        [r for r in rows if r["family"] == "fa"], "position", covariates, factors, "log1p"
    )
    assert single.levels_without_variation == ("constraint_type[procedure]",)


def test_a_null_model_separated_by_a_numeric_term_tests_the_rows_it_does_not_settle() -> None:
    # Synthetic: in family fa no decision point after a compaction violates, so the
    # compaction count separates the null model of the position test (every violation at
    # the smallest count). Those rows have fitted value 0 at the null fit and leave; the
    # test equals the one on the rows before any compaction.
    rows = _hazard_rows(29, base=(-1.0, -1.0))
    for row in rows:
        if row["family"] == "fa" and row["compaction_count"] >= 1:
            row["violated"] = False
    covariates, factors = ("position", "compaction_count"), ()
    fa = [r for r in rows if r["family"] == "fa"]
    found = hazard_score_test(fa, "position", covariates, factors, "log1p")
    cut = stats.risk_set(fa)
    late = sum(1 for r in cut if r["compaction_count"] >= 1)
    assert found.test.problem is None and found.test.n_boundary == late > 0
    early = [r for r in cut if r["compaction_count"] == 0]
    expected = hazard_score_test(early, "position", covariates, factors, "log1p")
    assert found.test.p_value == pytest.approx(expected.test.p_value, rel=1e-6)
    assert found.to_dict()["score"]["n_boundary_rows"] == late


def test_a_family_without_violations_adds_nothing_to_the_pooled_hazard_test() -> None:
    rows = _hazard_rows(17)
    quiet = [dict(r, violated=False) if r["family"] == "fb" else r for r in rows]
    covariates, factors = ("position", "compaction_count"), ("constraint_type", "strategy")
    pooled = hazard_score_test(quiet, "position", covariates, factors, "log1p", strata="family")
    alone = hazard_score_test(
        [r for r in quiet if r["family"] == "fa"], "position", covariates, factors, "log1p"
    )
    assert pooled.test.p_value == pytest.approx(alone.test.p_value, rel=1e-12)
    assert pooled.test.strata_skipped == ("fb",)
    assert pooled.n_events == alone.n_events


def test_hazard_score_test_rejects_bad_input() -> None:
    rows = _hazard_rows(19, n_plans=6)
    with pytest.raises(StatsError, match="not among the covariates"):
        hazard_score_test(rows, "strategy", ("position",), ())
    with pytest.raises(StatsError, match="no opportunity rows"):
        hazard_score_test([], "position", ("position",), ())
    quiet = [dict(r, violated=False) for r in rows]
    nothing = hazard_score_test(quiet, "position", ("position",), (), strata="family")
    assert nothing.test.problem == "no_events" and nothing.test.p_value is None


def test_hazard_score_tests_hold_their_level_on_seeded_synthetic_nulls() -> None:
    # Synthetic nulls: neither position nor the compaction count changes the hazard (rule
    # types, strategy, and family do). At the 0.05 level about 5% of 200 tests reject; the
    # band holds 99.8% of a Binomial(200, 0.05).
    covariates, factors = ("position", "compaction_count"), ("constraint_type", "strategy")
    rejected = {"position": 0, "compaction_count": 0}
    for seed in range(200):
        rows = _hazard_rows(1000 + seed)
        for term in rejected:
            test = hazard_score_test(rows, term, covariates, factors, "log1p", strata="family")
            assert test.test.p_value is not None
            rejected[term] += test.test.p_value < 0.05
    assert all(2 <= hits <= 20 for hits in rejected.values()), rejected
    # And a real position effect is found.
    found = sum(
        (
            hazard_score_test(
                _hazard_rows(2000 + s, n_plans=80, slope=1.0, base=(-4.5, -4.0)),
                "position",
                covariates,
                factors,
                "log1p",
                strata="family",
            ).test.p_value
            or 1.0
        )
        < 0.05
        for s in range(20)
    )
    assert found >= 16


# --------------------------------------------------------------------------- contrasts


def test_rate_contrast_score_test_is_stratified_when_asked() -> None:
    x, y, cluster, strata = _two_family_arrays(23)
    rows = [
        {
            "family": str(s),
            "session_id": str(c),
            "arm": "b" if xi else "a",
            "violated": bool(yi),
        }
        for xi, yi, c, s in zip(x, y, cluster, strata, strict=True)
    ]
    plain = rate_contrast(rows, "arm", "a", "b", 5, seed=1)
    stratified = rate_contrast(rows, "arm", "a", "b", 5, seed=1, strata="family")
    expected = score_test(x, y, list(cluster), term="arm[b]", strata=list(strata))
    assert stratified.score.p_value == pytest.approx(expected.p_value, rel=1e-12)
    assert plain.score.n_strata is None and stratified.score.n_strata == 2
    assert stratified.score_strata == "family" and plain.score_strata is None
    assert stratified.to_dict()["score_strata"] == "family"
    # The descriptive parts do not change.
    assert stratified.difference == plain.difference
    with pytest.raises(StatsError, match="no field"):
        rate_contrast(
            [{k: v for k, v in r.items() if k != "family"} for r in rows],
            "arm",
            "a",
            "b",
            5,
            seed=1,
            strata="family",
        )


# --------------------------------------------------------------------------- saturation


def _rho_by_hand(residual: np.ndarray, root_w: np.ndarray, labels: np.ndarray) -> float:
    """The exchangeable moment estimate written out, over rows with a weight above zero."""
    keep = root_w > 0.0
    e, g = residual[keep] / root_w[keep], labels[keep]
    cross = sum(float(e[g == k].sum() ** 2 - (e[g == k] ** 2).sum()) for k in np.unique(g))
    pairs = sum(int((g == k).sum()) * (int((g == k).sum()) - 1) for k in np.unique(g))
    return min(max(cross / pairs, 0.0), stats.SCORE_RHO_MAX) if pairs else 0.0


def test_rows_at_a_fitted_value_of_exactly_0_or_1_leave_the_correlation_estimate() -> None:
    # Synthetic: a row whose null fitted probability is exactly 0 or 1 (a separated row the
    # drift check cannot see once it has saturated in floating point) has weight 0 and
    # residual 0, so its Pearson residual is 0/0. It adds nothing to the score or its
    # variance, so it leaves the correlation estimate and its pair count; before, the 0/0
    # made rho nan and the test fell to zero_variance.
    rng = np.random.default_rng(13)
    labels = np.repeat(np.arange(8), 4)
    p = rng.uniform(0.1, 0.6, size=32)
    y = (rng.random(32) < p).astype(float)
    p[[0, 5]], y[[0, 5]] = 1.0, 1.0
    p[9], y[9] = 0.0, 0.0
    residual, root_w = y - p, np.sqrt(p * (1.0 - p))
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no 0/0 on the way
        rho = stats._exchangeable_rho(residual, root_w, labels, 8)
    assert rho == pytest.approx(_rho_by_hand(residual, root_w, labels), abs=1e-12)
    assert math.isfinite(rho)
    # Without such rows nothing changes, to the last bit: the same operations run.
    fine = np.sqrt(rng.uniform(0.05, 0.25, size=32))
    resid = rng.normal(0.0, 0.3, size=32)
    sizes = np.bincount(labels, minlength=8).astype(np.float64)
    pearson = resid / fine
    sum_e = np.bincount(labels, weights=pearson, minlength=8)
    sum_e2 = np.bincount(labels, weights=pearson * pearson, minlength=8)
    old = min(
        max(float((sum_e * sum_e - sum_e2).sum()) / float(sizes @ (sizes - 1.0)), 0.0),
        stats.SCORE_RHO_MAX,
    )
    assert stats._exchangeable_rho(resid, fine, labels, 8) == old
    assert stats._exchangeable_rho(resid, fine, np.arange(32), 32) == 0.0  # no pairs

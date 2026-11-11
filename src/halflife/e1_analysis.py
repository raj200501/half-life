"""E1 analysis, fixed before any confirmatory data exists (prereg/0001-e1.md will cite
it; not committed yet).

Input: ``sessions.jsonl`` and ``opportunities.jsonl`` written by
:func:`halflife.experiment.bank_runs`. Output: ``hazard.json``, ``half_life.json``,
``km.json``, ``contrasts.json``, and ``descriptive.json`` in the same directory
(the G7 layout in docs/ARTIFACTS.md), every number produced from the banked rows.

Sessions: every family runs the same plans, so a plan's ``session_id`` appears once
per family, and a session is the pair ``(family, session_id)`` (:data:`SESSION`). Risk
sets, exclusion counts, and session counts use that pair. Bootstrap draws and CR1
errors cluster on the plan (:data:`CLUSTER`, its ``session_id``): within one family that
is the session, and in pooled analyses a draw keeps both families' sessions of each
drawn plan together, since the plan is the unit of randomization. Risk sets are cut per
family-session before resampling, so every draw holds whole per-family risk sets.

Analysis set (to be fixed in the E1 preregistration): confirmatory sessions without a
probe leak that reached their first decision point, and their opportunities with at
least one tool call in the decision-point span, less procedure rules whose span a cap or
stop cut (:func:`procedure_cut`, DECISIONS.md D-050 item 5). Every exclusion is counted,
in total and per family.

Decisions (D-050 items 1 and 2): every hypothesis is decided by a two-sided score test at
the null fit, clustered by plan through an exchangeable within-plan correlation
(:func:`halflife.stats.score_test`), at ``alpha``. A pooled test fits its null model
within each family and sums the score over families, with the variance summed over plan
clusters that hold both families' sessions, so a family without violations adds nothing.
In E1 the per-family and pooled H1, H2a, and H2b are separate hypotheses, with no
adjustment across families. Bootstrap percentile intervals are descriptive estimates
beside each decision.

Models (DECISIONS.md D-029 for the estimation details):

* Hazard model per family, and pooled with a family factor (for the descriptive
  intervals): logistic regression per opportunity on the risk set, with
  ``log1p(position)`` and ``compaction_count`` as covariates and ``constraint_type`` and
  ``strategy`` as factors. H1 is the ``log1p(position)`` term, H2b the
  ``compaction_count`` term, each decided by its score test
  (:func:`halflife.stats.hazard_score_test`), not tested when the fit has fewer than
  ``min_events`` risk-set violations. Each fit's diagnostics are written under ``fits``;
  a term without a finite estimate (no violation, or separation) is written without a
  number, and its hypothesis is still decided by its score test when that exists.
* H2a: violation-rate contrast between the ``full_summary`` and ``sham`` arms
  among opportunities with at least one compaction before them, decided by its score test.
* H3: per-type violation rates and every pairwise type contrast, pooled over families
  (family-stratified score tests), Holm-adjusted over all pairs (:data:`HOLM_NOTE`).
* Half-life per family from a hazard model with ``log1p(position)`` only, marginal over
  compaction, rule type, and strategy (D-050 item 4), searched from the family's first
  decision point to ``max_position``.
* Kaplan-Meier curves of time to first violation, descriptive only.

Every hypothesis estimate carries the verdict of its rule (:data:`VERDICT_NOTE`):
``supported``, ``not supported``, or ``not tested`` with the reason (no valid p-value, or
too few violations), using the constants of :class:`Settings`.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from halflife import stats
from halflife.analysis import SESSION_KEY
from halflife.schema import CONSTRAINT_TYPES

HAZARD_COVARIATES = ("position", "compaction_count")
HAZARD_FACTORS = ("constraint_type", "strategy")
HALF_LIFE_COVARIATES: tuple[str, ...] = ("position",)
"""The half-life model's terms: ``log1p(position)`` only, marginal over compaction (D-050)."""
HALF_LIFE_FIXED: Mapping[str, float | str] = {}
"""Nothing is held fixed for the half-life (D-050 item 4; it held ``compaction_count = 1``
before)."""
TRANSFORM: stats.PositionTransform = "log1p"
POSITION_TERM = "log1p(position)"
SESSION: tuple[str, ...] = SESSION_KEY
"""Fields that identify a session: ``(family, session_id)``."""
CLUSTER = "session_id"
"""Resampling and CR1 cluster: the plan (per family, the session)."""
RESAMPLING = (
    "percentile bootstrap over whole plans (session_id): per family a plan is one session; "
    "pooled, both families' sessions of each drawn plan stay together. Risk sets are cut per "
    "(family, session_id) before resampling."
)
HOLM_NOTE = (
    "Holm step-down over every test in the family, on the p-values of a two-sided score "
    "test at the null fit, clustered by plan through an exchangeable within-plan "
    "correlation estimated at the null fit (halflife.stats.score_test); pooled over "
    "families, the null model is fit within each family (score_strata). It stays valid "
    "under separation, so an arm or rule type without violations is still tested, and "
    "unlike a CR1 variance of the score it does not read a rarer arm without violations as "
    "a difference. A test whose p-value cannot be computed "
    "(p_value null, with the reason in p_value_missing, for example no violation in either "
    "arm) and a preregistered test without rows stay in the family as p = 1: never "
    "rejected, the other tests keep the full family size, and p_holm is null. An unstable "
    "interval is labeled and decides nothing."
)


@dataclass(frozen=True)
class Settings:
    """Analysis constants the E1 preregistration will fix (not committed yet).

    Attributes:
        n_boot: Bootstrap draws (descriptive intervals).
        seed: Bootstrap seed.
        max_position: Tool calls within which a half-life is searched.
        level: Interval level.
        alpha: Level of every decision: the two-sided score tests of H1, H2a, and H2b
            (unadjusted; per family and pooled are separate hypotheses) and the
            Holm-adjusted tests of H3 (and H4a, H4b, H5 in :mod:`halflife.arms_analysis`).
        max_failed_share: An interval whose bootstrap had a larger share of failed draws
            (``n_failed / n_boot``) is labeled unstable; the label decides nothing
            (DECISIONS.md D-050 item 1).
        min_events: A hazard fit (per family, or pooled) with fewer violations in its risk
            set leaves H1 and H2b not tested. The value is the calibration banked in
            ``artifacts/power/min_events.json`` by ``python -m halflife.calibrate --out
            artifacts/power/min_events.json`` (D-050 item 3; synthetic nulls on the 48 E1
            plans with two families, labeled synthetic): the smallest number of risk-set
            violations at and above which both score tests kept their null rejection rate
            within 3% to 7% in every event-count bin.
    """

    n_boot: int = 2000
    seed: int = 20261104
    max_position: int = 150
    level: float = 0.95
    alpha: float = 0.05
    max_failed_share: float = 0.1
    min_events: int = 2


ANALYSIS_MODULES = (
    "halflife/__init__.py",
    "halflife/analysis.py",
    "halflife/e1_analysis.py",
    "halflife/schema.py",
    "halflife/stats.py",
)
"""Every module ``python -m halflife.e1_analysis`` loads (paths under ``src/``): the
statistics run in :mod:`halflife.stats`, so freezing this file alone would not freeze the
analysis; the preregistration will pin all of them (:func:`analysis_code_sha256`)."""


def analysis_code_sha256() -> dict[str, str]:
    """SHA-256 of each module in :data:`ANALYSIS_MODULES`, by its path under ``src/``."""
    source = Path(__file__).resolve().parent.parent
    return {
        rel: hashlib.sha256((source / rel).read_bytes()).hexdigest() for rel in ANALYSIS_MODULES
    }


HYPOTHESIS_TERMS: Mapping[str, str] = {POSITION_TERM: "H1", "compaction_count": "H2b"}
"""Hazard terms that carry a hypothesis (to be fixed in the E1 preregistration)."""
TESTED_COVARIATES: Mapping[str, str] = {
    POSITION_TERM: "position",
    "compaction_count": "compaction_count",
}
"""The covariate behind each hypothesis term (``position`` enters as ``log1p(position)``)."""
SUPPORTED, NOT_SUPPORTED, NOT_TESTED = "supported", "not supported", "not tested"
"""Verdicts of a score-test rule (H1, H2a, H2b) or a Holm rule (H3, H4a, H4b, H5)."""
DECIDED_BY = "score_test"
"""What decides every hypothesis (D-050 item 1): a two-sided score test at the null fit."""
VERDICT_NOTE = (
    "verdict applies the preregistered rule to a two-sided score test at the null fit, "
    "clustered by plan through an exchangeable within-plan correlation (decided_by: "
    "score_test): for H1, H2a, and H2b, supported when its p-value is below alpha, with no "
    "adjustment across families (per family and pooled are separate hypotheses); for a Holm "
    "family (H3, H4a, H4b, H5), supported (differs) when the Holm-adjusted p-value is below "
    "alpha. Pooled tests fit the null model within each family. A hypothesis is not tested "
    "(not_tested names why) when its p-value cannot be computed (the reason code of "
    "halflife.stats.SCORE_PROBLEMS, for example no_events) or, for H1 and H2b, when its "
    "hazard fit has fewer than min_events violations in the risk set (few_violations). "
    "direction is the sign of the score (positive: more violations with the term, or in the "
    "second arm). Bootstrap intervals are descriptive estimates; unstable (failed_share "
    "above max_failed_share) labels an interval and decides nothing."
)
HAZARD_TEST = (
    "two-sided score test of the term at the null fit of the hazard model without it, on "
    "the risk set, clustered by plan through an exchangeable within-plan correlation "
    "estimated at the null fit; pooled, the null model is fit within each family, and "
    "factor levels without violations in a family add nothing "
    "(halflife.stats.hazard_score_test)"
)
"""The test behind every H1 and H2b decision (``p_value_test`` of a hazard term)."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Rows of a JSON-lines file."""
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def session_of(row: Mapping[str, Any], key: Sequence[str] = SESSION) -> tuple[str, ...]:
    """The session a session or opportunity row belongs to, by the ``key`` fields.

    Raises:
        stats.StatsError: when the row lacks one of them.
    """
    try:
        return tuple(str(row[name]) for name in key)
    except KeyError as exc:
        raise stats.StatsError(f"a row has no {exc.args[0]!r} to identify its session") from None


SPAN_DONE = "done"
"""``span_end`` of a decision-point span the agent finished (:func:`procedure_cut`)."""


def procedure_cut(row: Mapping[str, Any]) -> bool:
    """A procedure rule whose decision-point span the agent did not finish (D-050 item 5).

    ``span_end`` (written by :func:`halflife.analysis.opportunity_rows`) is ``done`` when
    the span ended because the agent finished the request, otherwise the cap or stop that
    cut it (``subtask_cap``, ``tool_cap``, ``turn_cap``, ``subtask_tool_cap``,
    ``session_tool_cap``, ``timeout``, ``error``). Whether a procedure ("before you
    finish", "in the same task") was followed shows only in a finished span, so a
    procedure opportunity with any other ``span_end``, a null one, or none (rows written
    before the field existed) is cut. Other rule types are never cut.
    """
    return row.get("constraint_type") == "procedure" and row.get("span_end") != SPAN_DONE


def analysis_set(
    sessions: Sequence[Mapping[str, Any]],
    opportunities: Sequence[Mapping[str, Any]],
    *,
    key: Sequence[str] = SESSION,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Opportunities in the analysis set, and the exclusion counts (module docstring).

    Sessions and opportunities are matched by ``key``, ``(family, session_id)``: the
    families run the same plans, so ``session_id`` alone would let one family's session
    stand in for the other's. In order, a session leaves the analysis when it is not
    confirmatory, when a probe leaked into its main run (``probe_leaks`` not empty), or
    when it did not reach its first decision point; an opportunity of a kept session
    leaves when its decision-point span has no tool call, and then when it is a procedure
    rule whose span was cut (:func:`procedure_cut`). Each count is reported.

    Raises:
        stats.StatsError: when two session rows have the same key, or a row lacks a key
            field.
    """
    seen: set[tuple[str, ...]] = set()
    for row in sessions:
        ident = session_of(row, key)
        if ident in seen:
            raise stats.StatsError(f"duplicate session rows for {ident}")
        seen.add(ident)
    confirmatory = [s for s in sessions if s.get("confirmatory") is True]
    leaked = {session_of(s, key) for s in confirmatory if s.get("probe_leaks")}
    reached = {
        session_of(s, key)
        for s in confirmatory
        if s.get("reached_first_dp") is True and session_of(s, key) not in leaked
    }
    in_set = [o for o in opportunities if session_of(o, key) in reached]
    called = [dict(o) for o in in_set if int(o.get("span_calls", 1)) > 0]
    kept = [o for o in called if not procedure_cut(o)]
    exclusions = {
        "sessions_total": len(sessions),
        "sessions_confirmatory": len(confirmatory),
        "sessions_probe_leak": len(leaked),
        "sessions_not_reaching_first_dp": len(confirmatory) - len(leaked) - len(reached),
        "sessions_analyzed": len(reached),
        "opportunities_zero_calls": len(in_set) - len(called),
        "opportunities_procedure_cut": len(called) - len(kept),
        "opportunities_analyzed": len(kept),
    }
    return kept, exclusions


def exclusions_by_family(
    sessions: Sequence[Mapping[str, Any]], opportunities: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """The :func:`analysis_set` exclusion counts of each family's own rows, sorted by family."""
    out = []
    for family in sorted({str(s["family"]) for s in sessions}):
        _, counts = analysis_set(
            [s for s in sessions if str(s["family"]) == family],
            [o for o in opportunities if str(o["family"]) == family],
        )
        out.append({"family": family} | counts)
    return out


def families(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Model families present, sorted."""
    return sorted({str(r["family"]) for r in rows})


def _interval(interval: stats.Interval) -> dict[str, Any]:
    return interval.to_dict("estimate")


def is_unstable(interval: stats.Interval, settings: Settings) -> bool:
    """More than ``max_failed_share`` of the draws failed: the interval is labeled unstable
    (descriptive; it decides nothing, D-050 item 1)."""
    return interval.not_estimable is None and interval.failed_share > settings.max_failed_share


def fit_record(
    family: str, rows: Sequence[Mapping[str, Any]], model: stats.HazardFit
) -> dict[str, Any]:
    """Full-data diagnostics of one hazard fit: rows, events, convergence, separation.

    A pooled fit (several families) also gives each family's risk-set violations and the
    families without any, whose sessions leave the slopes to the other families: with a
    family factor, a family without violations only separates its own family term.
    """
    record: dict[str, Any] = {
        "family": family,
        "n_rows": model.n_rows,
        "n_events": model.n_events,
        "n_clusters": model.n_clusters,
        "converged": model.fit.converged,
        "separation": model.fit.separation,
        "separated_terms": list(model.fit.separated_terms),
        "warnings": list(model.fit.warnings),
    }
    if "family" in model.categorical:
        cut = stats.risk_set(rows, SESSION)
        events = {f: 0 for f in families(cut)}
        for row in cut:
            events[str(row["family"])] += int(bool(row["violated"]))
        record["events_by_family"] = events
        record["families_without_violations"] = sorted(f for f, n in events.items() if n == 0)
    return record


def decision(test: stats.ScoreTest, settings: Settings, *, few: bool = False) -> dict[str, Any]:
    """The decision fields of one unadjusted hypothesis (H1, H2a, H2b) from its score test.

    ``verdict`` is supported when the p-value is below ``alpha`` and not supported
    otherwise; it is not tested when no p-value exists (``not_tested`` is the score test's
    problem code) or when ``few`` (``few_violations``: the hazard fit has fewer than
    ``min_events`` violations; the p-value stays on record). ``direction`` is the sign of
    the score (:data:`VERDICT_NOTE`).
    """
    reason = test.problem if test.p_value is None else ("few_violations" if few else None)
    verdict = NOT_TESTED
    if reason is None and test.p_value is not None:
        verdict = SUPPORTED if test.p_value < settings.alpha else NOT_SUPPORTED
    return {
        "decided_by": DECIDED_BY,
        "alpha": settings.alpha,
        "p_value": json_p(test.p_value),
        "p_value_missing": test.problem,
        "direction": direction(test.z),
        "not_tested": reason,
        "verdict": verdict,
    }


def direction(z: float | None) -> str | None:
    """``positive`` or ``negative``: the sign of a score statistic, None without one."""
    if z is None or z == 0.0:
        return None
    return "positive" if z > 0.0 else "negative"


def hazard_tests(
    rows: Sequence[Mapping[str, Any]], family: str, terms: Sequence[str] | None = None
) -> dict[str, stats.HazardScoreTest]:
    """The H1 and H2b score tests of one hazard fit, by term (:data:`HAZARD_TEST`).

    Per family the null model is fit to that family's rows; ``pooled`` fits it within each
    family (D-050 item 2), so a family without violations adds nothing. ``terms`` limits
    the tests to some of :data:`TESTED_COVARIATES` (the calibration of ``min_events`` runs
    one at a time); all of them by default.
    """
    strata = "family" if family == "pooled" else None
    wanted = TESTED_COVARIATES if terms is None else {t: TESTED_COVARIATES[t] for t in terms}
    return {
        term: stats.hazard_score_test(
            rows,
            covariate,
            HAZARD_COVARIATES,
            HAZARD_FACTORS,
            TRANSFORM,
            strata=strata,
            cluster=CLUSTER,
            session=SESSION,
        )
        for term, covariate in wanted.items()
    }


def term_records(
    found: stats.HazardCoefficients,
    settings: Settings,
    hypotheses: Mapping[str, str] = HYPOTHESIS_TERMS,
    *,
    tests: Mapping[str, stats.HazardScoreTest] | None = None,
) -> dict[str, dict[str, Any]]:
    """Every term's descriptive interval with its flags and, for a hypothesis term, its
    decision from its score test in ``tests`` (:func:`decision`, :data:`VERDICT_NOTE`).

    ``not_estimable`` (no number: no violation, only violations, or a separated term) and
    ``unstable`` (more than ``max_failed_share`` of the draws failed) describe the interval
    and decide nothing. A hypothesis term is not tested when its score test has no p-value
    or its fit has fewer than ``min_events`` risk-set violations; ``score`` is that test
    (laid out as a contrast's ``score``) and ``score_model`` its null model. A term without
    a hypothesis has no decision (``decided_by``, ``not_tested``, and ``verdict`` null).

    Raises:
        stats.StatsError: when a hypothesis term has no score test in ``tests``.
    """
    out: dict[str, dict[str, Any]] = {}
    for term in sorted(found):
        interval = found[term]
        hypothesis = hypotheses.get(term)
        record = _interval(interval) | {
            "unstable": is_unstable(interval, settings),
            "hypothesis": hypothesis,
        }
        if hypothesis is None:
            record |= {"decided_by": None, "not_tested": None, "verdict": None}
        else:
            test = (tests or {}).get(term)
            if test is None:
                raise stats.StatsError(f"no score test for the hypothesis term {term!r}")
            record |= decision(test.test, settings, few=test.n_events < settings.min_events)
            model = test.to_dict()
            record |= {
                "p_value_test": HAZARD_TEST,
                "n_events": test.n_events,
                "score": model.pop("score"),
                "score_model": model,
            }
        out[term] = record
    return out


def hazard(rows: Sequence[Mapping[str, Any]], settings: Settings) -> dict[str, Any]:
    """H1 and H2b, per family and pooled: decided by score tests (:func:`hazard_tests`),
    with bootstrap intervals for every term beside them.

    Each fit's full-data diagnostics go under ``fits``; a term the data cannot estimate is
    written without a number (``not_estimable``), and the H1 and H2b terms carry their
    decision (:data:`VERDICT_NOTE`).
    """
    estimates: list[dict[str, Any]] = []
    fits: list[dict[str, Any]] = []
    for family in [*families(rows), "pooled"]:
        subset = [dict(r) for r in rows if family == "pooled" or r["family"] == family]
        factors = HAZARD_FACTORS + (
            ("family",) if family == "pooled" and len(families(rows)) > 1 else ()
        )
        found = stats.bootstrap_coefficients(
            subset,
            settings.n_boot,
            settings.seed,
            level=settings.level,
            cluster=CLUSTER,
            session=SESSION,
            covariates=HAZARD_COVARIATES,
            categorical=factors,
            position_transform=TRANSFORM,
        )
        fits.append(fit_record(family, subset, found.fit))
        records = term_records(found, settings, tests=hazard_tests(subset, family))
        for term, record in records.items():
            estimates.append({"family": family, "term": term} | record)
    return {
        "fits": fits,
        "verdict_note": VERDICT_NOTE,
        "decided_by": DECIDED_BY,
        "alpha": settings.alpha,
        "model": {
            "covariates": list(HAZARD_COVARIATES),
            "factors": list(HAZARD_FACTORS),
            "position_transform": TRANSFORM,
            "risk_set": True,
            "session_key": list(SESSION),
            "cluster": CLUSTER,
            "resampling": RESAMPLING,
            "outcome": "violated",
            "test": HAZARD_TEST,
            "pooled_test_strata": "family",
        },
        "h1_term": POSITION_TERM,
        "h2b_term": "compaction_count",
        "estimates": estimates,
    }


def half_lives(rows: Sequence[Mapping[str, Any]], settings: Settings) -> dict[str, Any]:
    """Half-life per family from a hazard model with ``log1p(position)`` only (D-050 item 4).

    The model is marginal over compaction, rule type, and strategy (nothing is held fixed:
    holding the compaction count at one value extrapolated where token-budget compactions
    tie the count to position). The search runs from the family's first decision-point
    position to ``max_position`` (:func:`halflife.stats.half_life`): a value at the first
    position reads at or before it, no crossing within the range is not reached (D-028),
    and an upper bound among draws that never crossed is open. A separated fit with
    violations gives no number (``not_estimable``); an interval with more than
    ``max_failed_share`` failed draws is labeled ``unstable``.
    """
    estimates = []
    for family in families(rows):
        subset = [dict(r) for r in rows if r["family"] == family]
        result = stats.bootstrap_half_life(
            subset,
            settings.max_position,
            HALF_LIFE_FIXED,
            settings.n_boot,
            settings.seed,
            level=settings.level,
            cluster=CLUSTER,
            session=SESSION,
            covariates=HALF_LIFE_COVARIATES,
            categorical=(),
            position_transform=TRANSFORM,
        )
        unstable = is_unstable(result.interval, settings)
        estimates.append({"family": family} | result.to_dict() | {"unstable": unstable})
    return {
        "fixed": dict(HALF_LIFE_FIXED),
        "covariates": list(HALF_LIFE_COVARIATES),
        "position_transform": TRANSFORM,
        "marginal_over": ["compaction_count", *HAZARD_FACTORS],
        "max_position": settings.max_position,
        "estimates": estimates,
    }


def km_curves(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Descriptive Kaplan-Meier curves of time to first violation, per family."""
    curves = [
        {"family": f}
        | stats.kaplan_meier_rows([r for r in rows if r["family"] == f], session=SESSION).to_dict()
        for f in families(rows)
    ]
    return {
        "descriptive": True,
        "note": "Depends on how often decision points occur; not an estimate of hazard.",
        "curves": curves,
    }


def json_p(value: float | None) -> float | None:
    """A p-value as the outputs write it: 12 significant digits (the JSON convention of
    :mod:`halflife.stats`), or None when it was not computed."""
    return None if value is None else float(f"{value:.12g}")


P_VALUE_TEST = (
    "two-sided score test of the arm term at the null fit, clustered by plan through an "
    "exchangeable within-plan correlation estimated at the null fit; with score_strata, "
    "the null model is fit within each stratum (pooled contrasts: each family)"
)
"""The test behind every decision p-value (``p_value``) of a contrast."""


def test_fields(result: stats.RateContrast) -> dict[str, Any]:
    """``p_value`` of the arm term's score test at the null fit (:func:`halflife.stats.
    score_test`, clustered as the bootstrap: by plan, through an exchangeable within-plan
    correlation) and ``p_value_missing``: None, or the code of the reason no valid
    p-value exists (:data:`halflife.stats.SCORE_PROBLEMS`, for example no violation in
    either arm). The score test needs no finite estimate under the alternative, so an arm
    or type without violations, which separates the logistic model and leaves no valid
    Wald test (kept under ``wald``, descriptive), is still tested. A missing p-value is
    never written as a number."""
    return {
        "p_value": json_p(result.score.p_value),
        "p_value_missing": result.score.problem,
        "p_value_test": P_VALUE_TEST,
        "direction": direction(result.score.z),
    }


def contrast_entry(result: stats.RateContrast, settings: Settings) -> dict[str, Any]:
    """One contrast as the outputs write it: the decision p-value, the contrast, and the
    unstable flags of its difference (the decision estimate) and odds ratio."""
    return (
        test_fields(result)
        | result.to_dict()
        | {
            "unstable": is_unstable(result.difference, settings),
            "odds_ratio_unstable": is_unstable(result.log_odds_ratio, settings),
        }
    )


def holm_verdicts(
    entries: Sequence[dict[str, Any]],
    names: Sequence[str],
    family: Sequence[str],
    settings: Settings,
) -> None:
    """Holm-adjust ``entries`` (named ``names``) over every test in ``family``, in place.

    A test with no p-value and a planned test with no rows stay in the family as p = 1
    (:data:`HOLM_NOTE`); an unstable interval decides nothing (D-050 item 1). Each entry
    gets ``p_holm``, ``holm_family_size``, ``decided_by``, ``alpha``, and its verdict:
    supported when ``p_holm`` is below ``settings.alpha``, not supported otherwise, and not
    tested (with ``not_tested`` naming why) when it has no adjusted p-value.
    """
    pvalues: dict[str, float | None] = dict.fromkeys(family)
    for name, entry in zip(names, entries, strict=True):
        pvalues[name] = entry["p_value"]
    adjusted = holm(pvalues)
    for name, entry in zip(names, entries, strict=True):
        p_holm = adjusted[name]
        entry["p_holm"] = json_p(p_holm)
        entry["holm_family_size"] = len(pvalues)
        entry["decided_by"] = DECIDED_BY
        entry["alpha"] = settings.alpha
        if p_holm is None:
            entry["not_tested"] = entry["p_value_missing"]
            entry["verdict"] = NOT_TESTED
        else:
            entry["not_tested"] = None
            entry["verdict"] = SUPPORTED if p_holm < settings.alpha else NOT_SUPPORTED


def holm(pvalues: Mapping[str, float | None]) -> dict[str, float | None]:
    """Holm step-down adjusted p-values over every test in ``pvalues`` (:data:`HOLM_NOTE`).

    A None p-value (not computable) stays in the family as p = 1: it sorts last and is
    never rejected, so the other tests are adjusted over the full family size, and its
    own adjusted value is None. Results keep the input order.
    """
    m = len(pvalues)
    computed = sorted(
        ((key, p) for key, p in pvalues.items() if p is not None), key=lambda item: item[1]
    )
    adjusted: dict[str, float | None] = {}
    running = 0.0
    for rank, (key, p) in enumerate(computed):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[key] = running
    return {key: adjusted.get(key) for key in pvalues}


def contrasts(rows: Sequence[Mapping[str, Any]], settings: Settings) -> dict[str, Any]:
    """H2a (summary versus sham after compaction) and H3 (pairwise rule types).

    H2a is decided per family and pooled by its score test, unadjusted, and H3 (pooled)
    by the Holm-adjusted score-test p-values (:data:`VERDICT_NOTE`); pooled tests fit their
    null model within each family (``score_strata``). Every entry carries its verdict
    beside its descriptive intervals.
    """
    out: dict[str, Any] = {
        "h2a": [],
        "h3_rates": [],
        "h3_pairs": [],
        "holm_note": HOLM_NOTE,
        "verdict_note": VERDICT_NOTE,
    }
    after = [dict(r) for r in rows if int(r["compaction_count"]) >= 1]
    for family in [*families(rows), "pooled"]:
        subset = [r for r in after if family == "pooled" or r["family"] == family]
        arms = {r["strategy"] for r in subset}
        if {"full_summary", "sham"} <= arms:
            result = stats.rate_contrast(
                subset,
                "strategy",
                "sham",
                "full_summary",
                settings.n_boot,
                settings.seed,
                level=settings.level,
                cluster=CLUSTER,
                session=SESSION,
                strata="family" if family == "pooled" else None,
            )
            entry = {"family": family, "hypothesis": "H2a"} | contrast_entry(result, settings)
            out["h2a"].append(entry | decision(result.score, settings))
    types = sorted({str(r["constraint_type"]) for r in rows})
    for ctype in types:
        typed = [r for r in rows if r["constraint_type"] == ctype]
        k = sum(1 for r in typed if r["violated"])
        out["h3_rates"].append(
            {"constraint_type": ctype, "k": k, "n": len(typed)}
            | stats.wilson_interval(k, len(typed), settings.level).to_dict()
        )
    pairs: list[dict[str, Any]] = []
    keys: list[str] = []
    # The Holm family is every pair of the schema's rule types (to be fixed in the E1
    # preregistration: "all pairs"), so a type without rows leaves its pairs in the family
    # as p = 1.
    every_type = sorted({*CONSTRAINT_TYPES, *types})
    holm_family = [f"{a} vs {b}" for a, b in itertools.combinations(every_type, 2)]
    for a, b in itertools.combinations(types, 2):
        subset = [dict(r) for r in rows if r["constraint_type"] in {a, b}]
        result = stats.rate_contrast(
            subset,
            "constraint_type",
            a,
            b,
            settings.n_boot,
            settings.seed,
            level=settings.level,
            cluster=CLUSTER,
            session=SESSION,
            strata="family",
        )
        key = f"{a} vs {b}"
        keys.append(key)
        pairs.append({"pair": key, "hypothesis": "H3"} | contrast_entry(result, settings))
    holm_verdicts(pairs, keys, holm_family, settings)
    out["h3_pairs"] = pairs
    return out


def descriptive(rows: Sequence[Mapping[str, Any]], settings: Settings) -> dict[str, Any]:
    """Violation rates overall and when the agent acted, and recall by compaction count."""
    out: dict[str, Any] = {"by_family": []}
    for family in families(rows):
        subset = [r for r in rows if r["family"] == family]
        acted = [r for r in subset if r.get("acted")]
        entry: dict[str, Any] = {
            "family": family,
            "opportunities": len(subset),
            "acted": len(acted),
        }
        k = sum(1 for r in subset if r["violated"])
        entry["violation_rate"] = {"k": k, "n": len(subset)} | stats.wilson_interval(
            k, len(subset), settings.level
        ).to_dict()
        if acted:
            ka = sum(1 for r in acted if r["violated"])
            entry["violation_rate_when_acted"] = {"k": ka, "n": len(acted)} | stats.wilson_interval(
                ka, len(acted), settings.level
            ).to_dict()
        recall_rows = [r for r in subset if r.get("recalled_before") is not None]
        if recall_rows:
            entry["recall_by_compaction"] = [
                cell.to_dict()
                for cell in stats.recall_by_compaction(recall_rows, settings.level, SESSION)
            ]
        out["by_family"].append(entry)
    return out


def analyze(
    sessions: Sequence[Mapping[str, Any]],
    opportunities: Sequence[Mapping[str, Any]],
    settings: Settings,
) -> dict[str, dict[str, Any]]:
    """Every E1 output, keyed by file stem."""
    rows, exclusions = analysis_set(sessions, opportunities)
    if not rows:
        raise stats.StatsError("no opportunities in the analysis set")
    meta = {
        "settings": settings.__dict__,
        "exclusions": exclusions,
        "exclusions_by_family": exclusions_by_family(sessions, opportunities),
    }
    return {
        "hazard": hazard(rows, settings) | meta,
        "half_life": half_lives(rows, settings) | meta,
        "km": km_curves(rows) | meta,
        "contrasts": contrasts(rows, settings) | meta,
        "descriptive": descriptive(rows, settings) | meta,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m halflife.e1_analysis --dir artifacts/e1``: write the E1 outputs."""
    parser = argparse.ArgumentParser(
        description="E1 analysis (to be cited by prereg/0001-e1.md; not committed yet)."
    )
    parser.add_argument("--dir", type=Path, help="banked E1 directory")
    parser.add_argument("--n-boot", type=int, default=Settings.n_boot)
    parser.add_argument("--seed", type=int, default=Settings.seed)
    parser.add_argument(
        "--code-sha256",
        action="store_true",
        help="print the SHA-256 of every module the analysis runs (for the preregistration)",
    )
    args = parser.parse_args(argv)
    if args.code_sha256:
        print(json.dumps(analysis_code_sha256(), indent=2, sort_keys=True))
        return 0
    if args.dir is None:
        parser.error("--dir is required")
    settings = Settings(n_boot=args.n_boot, seed=args.seed)
    outputs = analyze(
        load_jsonl(args.dir / "sessions.jsonl"),
        load_jsonl(args.dir / "opportunities.jsonl"),
        settings,
    )
    for stem, data in outputs.items():
        (args.dir / f"{stem}.json").write_text(
            json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.dir / stem}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())

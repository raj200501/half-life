"""E3 and E4 analyses, fixed before any E3 or E4 data (prereg/0002-e3.md and
prereg/0003-e4.md will cite it; neither is committed yet).

Input: ``sessions.jsonl`` and ``opportunities.jsonl`` written by
:func:`halflife.experiment.bank_runs`. Output, in the same directory: ``rates.json``,
``contrasts.json``, ``summary.json``, and ``hazard.json`` (E3) or ``costs.json`` (E4).
``summary.json`` follows the G8 and G9 layouts in docs/ARTIFACTS.md. Every number is
produced from the banked rows.

Analysis set for violations and recall: as in E1 (:func:`halflife.e1_analysis.analysis_set`),
confirmatory sessions that reached their first decision point and their opportunities with
at least one tool call in the decision-point span.

E3, placement and phrasing (RQ3):

* violation rate and pre-decision-point recall per placement, per phrasing, and per cell,
  with Wilson intervals;
* violation-rate contrasts of ``system``, ``pinned_file``, and ``reminder`` against
  ``first_user`` (Holm-adjusted over the three) and of ``reason`` against ``bare``, with
  bootstrap intervals and score-test p-values clustered by plan through an exchangeable
  within-plan correlation (:func:`halflife.stats.score_test`; within one family a plan is
  one session);
* a hazard model with ``log1p(position)`` and ``compaction_count`` as covariates and
  ``constraint_type``, ``placement``, and ``phrasing`` as factors (plus ``family`` when
  more than one family ran), with bootstrap intervals.

E4, mitigations on held-out templates (RQ5):

* violation rate per arm and the contrast of each mitigation against ``none``,
  Holm-adjusted over the mitigations;
* costs beside every effect (rule 15), over all confirmatory sessions: tokens per session
  (prompt plus completion tokens of ``step`` and ``compaction`` calls; recall probes are
  measurement and are left out), latency per session (backend seconds of the same calls),
  and task success (the share of the plan's subtasks with a success check that passed; a
  checked subtask the session never reached counts as not passed), each as a mean with a
  bootstrap interval over plans and as a difference from ``none``.

E2, Claude Code (the ecological arm, exploratory unless a preregistration is banked):

* violation rate and pre-decision-point recall per model tier, with Wilson intervals,
  over sessions that reached their first decision point; ``summary.json`` follows the G8
  layout and is labeled ``ecological``.

Every analysis runs per family and pooled over families, with the E1 identity rules
(:mod:`halflife.e1_analysis`): a session is ``(family, session_id)``, so session counts
and risk sets keep each family's session of a plan apart, while bootstrap draws and CR1
errors cluster on the plan (``session_id``), which pooled keeps both families' sessions
of a plan together. E2 sessions are ``(model, session_id)`` (one plan per model tier).
Holm adjustments keep tests whose p-value cannot be computed in the family
(:data:`halflife.e1_analysis.HOLM_NOTE`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from halflife import stats
from halflife.e1_analysis import (
    CLUSTER,
    HAZARD_COVARIATES,
    HOLM_NOTE,
    SESSION,
    TRANSFORM,
    Settings,
    analysis_set,
    contrast_entry,
    exclusions_by_family,
    families,
    fit_record,
    holm_verdicts,
    is_unstable,
    load_jsonl,
    session_of,
    term_records,
)

PLACEMENT_REFERENCE = "first_user"
PLACEMENTS = ("system", "pinned_file", "reminder")
PHRASING_REFERENCE = "bare"
PHRASINGS = ("reason",)
MITIGATION_REFERENCE = "none"
MITIGATIONS = ("guard", "pin", "preserve", "reinject", "selfcheck")
COST_PURPOSES = ("step", "compaction")
COST_FIELDS = ("tokens", "latency_s", "task_success")
E3_FACTORS = ("constraint_type", "placement", "phrasing")
E2_SESSION: tuple[str, ...] = ("model", "session_id")
"""E2 rows carry the model tier in ``model`` and run one plan per tier."""


def _by_family(rows: Sequence[Mapping[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    """``(family, rows)`` for every family, then ``("pooled", all rows)``."""
    out = [(f, [dict(r) for r in rows if r["family"] == f]) for f in families(rows)]
    return [*out, ("pooled", [dict(r) for r in rows])]


def rates(
    rows: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
    level: float,
    session: Sequence[str] = SESSION,
) -> list[dict[str, Any]]:
    """Violation rate and pre-decision-point recall for every combination of ``keys``.

    ``sessions`` counts distinct ``session`` keys, so each family's session of a plan
    counts once.
    """
    groups: dict[tuple[str, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(str(row[k]) for k in keys), []).append(row)
    out = []
    for key in sorted(groups):
        group = groups[key]
        k = sum(1 for r in group if r["violated"])
        entry: dict[str, Any] = dict(zip(keys, key, strict=True)) | {
            "opportunities": len(group),
            "sessions": len({session_of(r, session) for r in group}),
        }
        entry["violation_rate"] = {"k": k, "n": len(group)} | stats.wilson_interval(
            k, len(group), level
        ).to_dict()
        recall = [r for r in group if r.get("recalled_before") is not None]
        if recall:
            kr = sum(1 for r in recall if r["recalled_before"])
            entry["recall_before"] = {"k": kr, "n": len(recall)} | stats.wilson_interval(
                kr, len(recall), level
            ).to_dict()
            if all("reminder_on_request" in r for r in recall):
                # A reminder sent with the decision-point request reached the agent after
                # the probe ran, so these probes understate what the agent had in context.
                entry["recall_before"]["reminders_on_request"] = sum(
                    1 for r in recall if r["reminder_on_request"] is True
                )
        out.append(entry)
    return out


def contrasts_against(
    rows: Sequence[Mapping[str, Any]],
    key: str,
    reference: str,
    arms: Sequence[str],
    settings: Settings,
) -> list[dict[str, Any]]:
    """Violation-rate contrast of each present arm against ``reference``, Holm-adjusted.

    The decision p-value is the clustered score test at the null fit (:func:`halflife.
    e1_analysis.test_fields`), which stays valid when an arm has no violations. The Holm
    family is every arm in ``arms``, as the preregistrations will fix it (D-040: over the three
    placements, over the five mitigations). An arm without rows, an unstable contrast, and
    one whose p-value cannot be computed stay in it as p = 1 (:data:`HOLM_NOTE`), so they
    never make the other adjustments smaller; absent arms get no entry. Every entry
    carries its verdict (:data:`halflife.e1_analysis.VERDICT_NOTE`).
    """
    present = {r[key] for r in rows}
    out: list[dict[str, Any]] = []
    names: list[str] = []
    if reference not in present:
        return out
    for arm in arms:
        if arm not in present:
            continue
        subset = [dict(r) for r in rows if r[key] in {reference, arm}]
        result = stats.rate_contrast(
            subset,
            key,
            reference,
            arm,
            settings.n_boot,
            settings.seed,
            level=settings.level,
            cluster=CLUSTER,
            session=SESSION,
        )
        names.append(arm)
        out.append({key: arm, "reference": reference} | contrast_entry(result, settings))
    holm_verdicts(out, names, list(arms), settings)
    return out


def e3_rates(rows: Sequence[Mapping[str, Any]], settings: Settings) -> dict[str, Any]:
    """Rates by placement, by phrasing, and by cell, per family and pooled."""
    return {
        "by_family": [
            {
                "family": family,
                "placement": rates(subset, ("placement",), settings.level),
                "phrasing": rates(subset, ("phrasing",), settings.level),
                "cells": rates(subset, ("placement", "phrasing"), settings.level),
            }
            for family, subset in _by_family(rows)
        ]
    }


def e3_contrasts(rows: Sequence[Mapping[str, Any]], settings: Settings) -> dict[str, Any]:
    """Placement contrasts against ``first_user`` and phrasing against ``bare``."""
    return {
        "by_family": [
            {
                "family": family,
                "placement": contrasts_against(
                    subset, "placement", PLACEMENT_REFERENCE, PLACEMENTS, settings
                ),
                "phrasing": contrasts_against(
                    subset, "phrasing", PHRASING_REFERENCE, PHRASINGS, settings
                ),
            }
            for family, subset in _by_family(rows)
        ]
    }


def e3_hazard(rows: Sequence[Mapping[str, Any]], settings: Settings) -> dict[str, Any]:
    """Hazard model with placement and phrasing factors, per family and pooled (exploratory).

    As in E1 (:func:`halflife.e1_analysis.hazard`): each fit's diagnostics go under
    ``fits``, a term the data cannot estimate (for example a placement without
    violations, which separates its term) is written without a number, and every term
    carries the unstable and ``not_tested`` flags; no term carries a hypothesis.
    """
    estimates: list[dict[str, Any]] = []
    fits: list[dict[str, Any]] = []
    several = len(families(rows)) > 1
    for family, subset in _by_family(rows):
        factors = [f for f in E3_FACTORS if len({r[f] for r in subset}) > 1]
        if family == "pooled" and several:
            factors.append("family")
        found = stats.bootstrap_coefficients(
            subset,
            settings.n_boot,
            settings.seed,
            level=settings.level,
            cluster=CLUSTER,
            session=SESSION,
            covariates=HAZARD_COVARIATES,
            categorical=tuple(factors),
            position_transform=TRANSFORM,
        )
        fits.append(fit_record(family, subset, found.fit))
        for term, record in term_records(found, settings, hypotheses={}).items():
            estimates.append({"family": family, "term": term} | record)
    return {
        "fits": fits,
        "model": {
            "covariates": list(HAZARD_COVARIATES),
            "factors": list(E3_FACTORS),
            "position_transform": TRANSFORM,
            "risk_set": True,
            "session_key": list(SESSION),
            "cluster": CLUSTER,
            "outcome": "violated",
        },
        "estimates": estimates,
    }


def session_costs(session: Mapping[str, Any]) -> dict[str, float | None]:
    """Tokens, backend seconds, and task success of one session (None when unknown).

    Task success is the share of the plan's subtasks with a success check that passed
    (``subtasks_succeeded / subtasks_planned_checked``): a checked subtask the session
    never reached, for example after the tool-call cap, counts as not passed, so a
    mitigation that ends sessions early shows that cost (D-040: costs use every
    confirmatory session). A session that reached no checked subtask scores 0. Rows
    without ``subtasks_planned_checked`` (banked before it existed) cannot say what was
    missed, so their task success is unknown (None, counted as missing).
    """
    usage = session.get("usage")
    tokens: float | None = None
    latency: float | None = None
    if isinstance(usage, Mapping):
        parts = [usage[p] for p in COST_PURPOSES if isinstance(usage.get(p), Mapping)]
        tokens = float(sum(p["prompt_tokens"] + p["completion_tokens"] for p in parts))
        latency = float(sum(p["elapsed_s"] for p in parts))
    planned = session.get("subtasks_planned_checked")
    success: float | None = None
    if isinstance(planned, int) and not isinstance(planned, bool) and planned > 0:
        success = int(session.get("subtasks_succeeded") or 0) / planned
    return {"tokens": tokens, "latency_s": latency, "task_success": success}


def _mean_of(name: str, arm: str | None = None) -> Any:
    def statistic(rows: list[dict[str, Any]]) -> float:
        values = [
            float(r[name])
            for r in rows
            if r[name] is not None and (arm is None or r["mitigation"] == arm)
        ]
        if not values:
            raise stats.StatsError(f"no {name} values")
        return sum(values) / len(values)

    return statistic


def _difference_of(name: str, arm: str, reference: str) -> Any:
    mean_arm = _mean_of(name, arm)
    mean_ref = _mean_of(name, reference)

    def statistic(rows: list[dict[str, Any]]) -> float:
        return float(mean_arm(rows)) - float(mean_ref(rows))

    return statistic


def e4_costs(
    sessions: Sequence[Mapping[str, Any]], settings: Settings
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    """Mean cost per arm with intervals, differences from ``none``, and plain arm means.

    One row per confirmatory session; the bootstrap resamples plans (``session_id``), so
    pooled draws keep both families' sessions of a plan together, as the violation
    contrasts do.
    """
    rows = [
        {
            "session_id": s["session_id"],
            "family": s["family"],
            "mitigation": s["mitigation"],
        }
        | session_costs(s)
        for s in sessions
        if s.get("confirmatory") is True
    ]
    out: dict[str, Any] = {"purposes": list(COST_PURPOSES), "by_family": []}
    means: dict[str, dict[str, float]] = {}
    for family, subset in _by_family(rows):
        arms = sorted({r["mitigation"] for r in subset})
        entries = []
        for arm in arms:
            mine = [r for r in subset if r["mitigation"] == arm]
            entry: dict[str, Any] = {"mitigation": arm, "sessions": len(mine)}
            for name in COST_FIELDS:
                known = [r for r in mine if r[name] is not None]
                entry[f"{name}_missing"] = len(mine) - len(known)
                if not known:
                    continue
                interval = stats.cluster_bootstrap(
                    known,
                    _mean_of(name),
                    settings.n_boot,
                    settings.seed,
                    cluster=CLUSTER,
                    level=settings.level,
                )
                entry[name] = interval.to_dict("mean") | {
                    "unstable": is_unstable(interval, settings)
                }
                estimate = interval.estimate
                if family == "pooled" and not isinstance(estimate, stats.NotReached):
                    means.setdefault(arm, {})[name] = float(estimate)
                if arm == MITIGATION_REFERENCE:
                    continue
                pair = [
                    r
                    for r in subset
                    if r[name] is not None and r["mitigation"] in {arm, MITIGATION_REFERENCE}
                ]
                if any(r["mitigation"] == MITIGATION_REFERENCE for r in pair):
                    diff = stats.cluster_bootstrap(
                        pair,
                        _difference_of(name, arm, MITIGATION_REFERENCE),
                        settings.n_boot,
                        settings.seed,
                        cluster=CLUSTER,
                        level=settings.level,
                        strata="mitigation",
                    )
                    entry[f"{name}_vs_none"] = diff.to_dict("difference") | {
                        "unstable": is_unstable(diff, settings)
                    }
            entries.append(entry)
        out["by_family"].append({"family": family, "arms": entries})
    return out, means


def e4_summary(
    sessions: Sequence[Mapping[str, Any]],
    contrasts: Mapping[str, Any],
    means: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    """The G9 ``summary.json`` layout: every mitigation with its cost beside its effect."""
    pooled = next(e for e in contrasts["by_family"] if e["family"] == "pooled")
    effects = {str(c["mitigation"]): c for c in pooled["mitigation"]}
    mitigations = []
    for name in MITIGATIONS:
        if name not in means:
            continue
        effect = effects.get(name, {})
        mitigations.append(
            {
                "name": name,
                "violation_difference": effect.get("difference"),
                "odds_ratio": effect.get("odds_ratio"),
                "p_holm": effect.get("p_holm"),
                "p_value_missing": effect.get("p_value_missing"),
                "verdict": effect.get("verdict"),
                "not_tested": effect.get("not_tested"),
                "cost": dict(means[name]),
            }
        )
    return {
        "arm": "E4",
        "ran": True,
        "n_sessions": len(sessions),
        "held_out_templates": sorted({str(s["template_id"]) for s in sessions}),
        "control": {
            "name": MITIGATION_REFERENCE,
            "cost": dict(means.get(MITIGATION_REFERENCE, {})),
        },
        "mitigations": mitigations,
        "cost_definitions": {
            "tokens": "mean prompt plus completion tokens per session, step and compaction calls",
            "latency_s": "mean backend seconds per session, step and compaction calls",
            "task_success": (
                "mean share of the plan's subtasks with a success check that passed, per "
                "session; a checked subtask the session never reached counts as not passed"
            ),
        },
    }


def analyze_e3(
    sessions: Sequence[Mapping[str, Any]],
    opportunities: Sequence[Mapping[str, Any]],
    settings: Settings,
) -> dict[str, dict[str, Any]]:
    """Every E3 output, keyed by file stem."""
    rows, exclusions = analysis_set(sessions, opportunities)
    if not rows:
        raise stats.StatsError("no opportunities in the analysis set")
    meta = {
        "settings": settings.__dict__,
        "exclusions": exclusions,
        "exclusions_by_family": exclusions_by_family(sessions, opportunities),
    }
    return {
        "rates": e3_rates(rows, settings) | meta,
        "contrasts": e3_contrasts(rows, settings) | {"holm_note": HOLM_NOTE} | meta,
        "hazard": e3_hazard(rows, settings) | meta,
        "summary": {"arm": "E3", "ran": True, "n_sessions": len(sessions)} | meta,
    }


def analyze_e4(
    sessions: Sequence[Mapping[str, Any]],
    opportunities: Sequence[Mapping[str, Any]],
    settings: Settings,
) -> dict[str, dict[str, Any]]:
    """Every E4 output, keyed by file stem."""
    rows, exclusions = analysis_set(sessions, opportunities)
    if not rows:
        raise stats.StatsError("no opportunities in the analysis set")
    meta = {
        "settings": settings.__dict__,
        "exclusions": exclusions,
        "exclusions_by_family": exclusions_by_family(sessions, opportunities),
    }
    contrasts = {
        "by_family": [
            {
                "family": family,
                "mitigation": contrasts_against(
                    subset, "mitigation", MITIGATION_REFERENCE, MITIGATIONS, settings
                ),
            }
            for family, subset in _by_family(rows)
        ],
        "holm_note": HOLM_NOTE,
    }
    costs, means = e4_costs(sessions, settings)
    return {
        "rates": {
            "by_family": [
                {"family": family, "mitigation": rates(subset, ("mitigation",), settings.level)}
                for family, subset in _by_family(rows)
            ]
        }
        | meta,
        "contrasts": contrasts | meta,
        "costs": costs | meta,
        "summary": e4_summary(sessions, contrasts, means) | meta,
    }


def analyze_e2(
    sessions: Sequence[Mapping[str, Any]],
    opportunities: Sequence[Mapping[str, Any]],
    settings: Settings,
) -> dict[str, dict[str, Any]]:
    """Every E2 output, keyed by file stem; E2 rows carry the model tier in ``model``.

    Sessions are ``(model, session_id)`` (:data:`E2_SESSION`): one plan runs on each tier.
    """
    reached = {session_of(s, E2_SESSION) for s in sessions if s.get("reached_first_dp") is True}
    in_set = [o for o in opportunities if session_of(o, E2_SESSION) in reached]
    rows = [dict(o) for o in in_set if int(o.get("span_calls", 1)) > 0]
    if not rows:
        raise stats.StatsError("no opportunities in the E2 analysis set")
    confirmatory = all(s.get("confirmatory") is True for s in sessions)
    meta = {
        "settings": settings.__dict__,
        "label": "ecological",
        "confirmatory": confirmatory,
        "exclusions": {
            "sessions_total": len(sessions),
            "sessions_not_reaching_first_dp": len(sessions) - len(reached),
            "opportunities_zero_calls": len(in_set) - len(rows),
            "opportunities_analyzed": len(rows),
        },
    }
    return {
        "rates": {"model": rates(rows, ("model",), settings.level, E2_SESSION)} | meta,
        "summary": {
            "arm": "E2",
            "ran": True,
            "n_sessions": len(sessions),
            "label": "ecological",
            "models": sorted({str(s["model"]) for s in sessions}),
            "confirmatory": confirmatory,
        }
        | {"settings": settings.__dict__},
    }


ANALYSIS_MODULES = (
    "halflife/__init__.py",
    "halflife/analysis.py",
    "halflife/arms_analysis.py",
    "halflife/e1_analysis.py",
    "halflife/schema.py",
    "halflife/stats.py",
)
"""Every module ``python -m halflife.arms_analysis`` loads (paths under ``src/``). The
E2, E3, and E4 analyses reuse :mod:`halflife.e1_analysis` and run their statistics in
:mod:`halflife.stats`, so their preregistrations pin all of these, as E1's pins
:data:`halflife.e1_analysis.ANALYSIS_MODULES` (DECISIONS.md D-050, item 6)."""


def analysis_code_sha256() -> dict[str, str]:
    """SHA-256 of each module in :data:`ANALYSIS_MODULES`, by its path under ``src/``."""
    source = Path(__file__).resolve().parent.parent
    return {
        rel: hashlib.sha256((source / rel).read_bytes()).hexdigest() for rel in ANALYSIS_MODULES
    }


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m halflife.arms_analysis {e2,e3,e4} --dir artifacts/e3``: write the outputs."""
    parser = argparse.ArgumentParser(description="E2, E3, and E4 analyses.")
    parser.add_argument("experiment", choices=("e2", "e3", "e4"))
    parser.add_argument("--dir", type=Path, help="banked experiment directory")
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
    analyze = {"e2": analyze_e2, "e3": analyze_e3, "e4": analyze_e4}[args.experiment]
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

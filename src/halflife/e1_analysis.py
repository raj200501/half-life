"""E1 analysis, fixed before any confirmatory data exists (prereg/0001-e1.md cites it).

Input: ``sessions.jsonl`` and ``opportunities.jsonl`` written by
:func:`halflife.experiment.bank_runs`. Output: ``hazard.json``, ``half_life.json``,
``km.json``, ``contrasts.json``, and ``descriptive.json`` in the same directory
(the G7 layout in docs/ARTIFACTS.md), every number produced from the banked rows.

Analysis set: confirmatory sessions that reached their first decision point,
and their opportunities with at least one tool call in the decision-point span.

Models (DECISIONS.md D-029 for the estimation details):

* Hazard model per family, and pooled with a family factor: logistic regression
  per opportunity on the risk set, with ``log1p(position)`` and
  ``compaction_count`` as covariates and ``constraint_type`` and ``strategy`` as
  factors. H1 is the ``log1p(position)`` coefficient, H2b the
  ``compaction_count`` coefficient.
* H2a: violation-rate contrast between the ``full_summary`` and ``sham`` arms
  among opportunities with at least one compaction before them.
* H3: per-type violation rates and every pairwise type contrast, Holm-adjusted.
* Half-life per family from a hazard model with ``log1p(position)`` and
  ``compaction_count`` only, at ``compaction_count = 1``.
* Kaplan-Meier curves of time to first violation, descriptive only.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from halflife import stats

HAZARD_COVARIATES = ("position", "compaction_count")
HAZARD_FACTORS = ("constraint_type", "strategy")
HALF_LIFE_FIXED: Mapping[str, float | str] = {"compaction_count": 1}
TRANSFORM: stats.PositionTransform = "log1p"
POSITION_TERM = "log1p(position)"


@dataclass(frozen=True)
class Settings:
    """Analysis constants the preregistration fixes."""

    n_boot: int = 2000
    seed: int = 20261104
    max_position: int = 150
    level: float = 0.95


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Rows of a JSON-lines file."""
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def analysis_set(
    sessions: Sequence[Mapping[str, Any]], opportunities: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Opportunities in the analysis set, and the exclusion counts."""
    confirmatory = [s for s in sessions if s.get("confirmatory") is True]
    reached = {s["session_id"] for s in confirmatory if s.get("reached_first_dp") is True}
    kept = [
        dict(o)
        for o in opportunities
        if o["session_id"] in reached and int(o.get("span_calls", 1)) > 0
    ]
    zero_span = sum(
        1 for o in opportunities if o["session_id"] in reached and int(o.get("span_calls", 1)) == 0
    )
    exclusions = {
        "sessions_total": len(sessions),
        "sessions_confirmatory": len(confirmatory),
        "sessions_not_reaching_first_dp": len(confirmatory) - len(reached),
        "opportunities_zero_calls": zero_span,
        "opportunities_analyzed": len(kept),
    }
    return kept, exclusions


def families(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Model families present, sorted."""
    return sorted({str(r["family"]) for r in rows})


def _interval(interval: stats.Interval) -> dict[str, Any]:
    return interval.to_dict("estimate")


def hazard(rows: Sequence[Mapping[str, Any]], settings: Settings) -> dict[str, Any]:
    """H1 and H2b: bootstrap intervals for every term, per family and pooled."""
    estimates: list[dict[str, Any]] = []
    for family in [*families(rows), "pooled"]:
        subset = [dict(r) for r in rows if family == "pooled" or r["family"] == family]
        factors = HAZARD_FACTORS + (
            ("family",) if family == "pooled" and len(families(rows)) > 1 else ()
        )
        intervals = stats.bootstrap_coefficients(
            subset,
            settings.n_boot,
            settings.seed,
            level=settings.level,
            covariates=HAZARD_COVARIATES,
            categorical=factors,
            position_transform=TRANSFORM,
        )
        for term in sorted(intervals):
            estimates.append({"family": family, "term": term} | _interval(intervals[term]))
    return {
        "model": {
            "covariates": list(HAZARD_COVARIATES),
            "factors": list(HAZARD_FACTORS),
            "position_transform": TRANSFORM,
            "risk_set": True,
            "outcome": "violated",
        },
        "h1_term": POSITION_TERM,
        "h2b_term": "compaction_count",
        "estimates": estimates,
    }


def half_lives(rows: Sequence[Mapping[str, Any]], settings: Settings) -> dict[str, Any]:
    """Half-life per family at ``compaction_count = 1``."""
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
            covariates=HAZARD_COVARIATES,
            categorical=(),
            position_transform=TRANSFORM,
        )
        estimates.append({"family": family} | result.to_dict())
    return {
        "fixed": dict(HALF_LIFE_FIXED),
        "max_position": settings.max_position,
        "estimates": estimates,
    }


def km_curves(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Descriptive Kaplan-Meier curves of time to first violation, per family."""
    curves = [
        {"family": f} | stats.kaplan_meier_rows([r for r in rows if r["family"] == f]).to_dict()
        for f in families(rows)
    ]
    return {
        "descriptive": True,
        "note": "Depends on how often decision points occur; not an estimate of hazard.",
        "curves": curves,
    }


def wald_p(result: stats.RateContrast) -> float:
    """Two-sided p-value of the arm term, with session-clustered (CR1) errors."""
    try:
        z = result.fit.wald_z(result.term, robust=True)
    except (KeyError, ValueError, ZeroDivisionError):
        return 1.0
    if z != z:  # nan
        return 1.0
    return 2.0 * (1.0 - stats.normal_cdf(abs(z)))


def holm(pvalues: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values."""
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (key, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[key] = running
    return adjusted


def contrasts(rows: Sequence[Mapping[str, Any]], settings: Settings) -> dict[str, Any]:
    """H2a (summary versus sham after compaction) and H3 (pairwise rule types)."""
    out: dict[str, Any] = {"h2a": [], "h3_rates": [], "h3_pairs": []}
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
            )
            out["h2a"].append({"family": family} | result.to_dict())
    types = sorted({str(r["constraint_type"]) for r in rows})
    for ctype in types:
        typed = [r for r in rows if r["constraint_type"] == ctype]
        k = sum(1 for r in typed if r["violated"])
        out["h3_rates"].append(
            {"constraint_type": ctype, "k": k, "n": len(typed)}
            | stats.wilson_interval(k, len(typed), settings.level).to_dict()
        )
    pvalues: dict[str, float] = {}
    pairs: list[dict[str, Any]] = []
    for a, b in itertools.combinations(types, 2):
        subset = [dict(r) for r in rows if r["constraint_type"] in {a, b}]
        result = stats.rate_contrast(
            subset, "constraint_type", a, b, settings.n_boot, settings.seed, level=settings.level
        )
        data = result.to_dict()
        key = f"{a} vs {b}"
        p = wald_p(result)
        pvalues[key] = p
        pairs.append({"pair": key, "p_value": p} | data)
    adjusted = holm(pvalues)
    for pair in pairs:
        pair["p_holm"] = adjusted[str(pair["pair"])]
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
                cell.to_dict() for cell in stats.recall_by_compaction(recall_rows, settings.level)
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
    meta = {"settings": settings.__dict__, "exclusions": exclusions}
    return {
        "hazard": hazard(rows, settings) | meta,
        "half_life": half_lives(rows, settings) | meta,
        "km": km_curves(rows) | meta,
        "contrasts": contrasts(rows, settings) | meta,
        "descriptive": descriptive(rows, settings) | meta,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m halflife.e1_analysis --dir artifacts/e1``: write the E1 outputs."""
    parser = argparse.ArgumentParser(description="E1 analysis (preregistered).")
    parser.add_argument("--dir", type=Path, required=True, help="banked E1 directory")
    parser.add_argument("--n-boot", type=int, default=Settings.n_boot)
    parser.add_argument("--seed", type=int, default=Settings.seed)
    args = parser.parse_args(argv)
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

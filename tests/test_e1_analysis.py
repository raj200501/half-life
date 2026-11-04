"""Tests for the preregistered E1 analysis on synthetic rows (labeled synthetic)."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import pytest

from halflife import e1_analysis, stats

TYPES = ("prohibition", "invariant", "procedure", "resource_limit")


def synthetic(
    n_sessions: int = 60, seed: int = 7
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Synthetic sessions and opportunities from a known hazard model (not model data)."""
    rng = random.Random(seed)
    sessions, opps = [], []
    for family in ("fam-a", "fam-b"):
        for i in range(n_sessions):
            sid = f"{family}-s{i}"
            strategy = "full_summary" if i % 2 else "sham"
            sessions.append(
                {
                    "session_id": sid,
                    "family": family,
                    "confirmatory": True,
                    "reached_first_dp": i % 13 != 0,
                }
            )
            position = 0
            for j in range(4):
                position += rng.randint(5, 30)
                count = min(3, position // 40)
                eta = (
                    -3.0
                    + 0.5 * math.log1p(position)
                    + (0.6 * count if strategy == "full_summary" else 0.0)
                )
                opps.append(
                    {
                        "session_id": sid,
                        "family": family,
                        "position": position,
                        "compaction_count": count,
                        "constraint_type": TYPES[(i + j) % 4],
                        "strategy": strategy,
                        "violated": rng.random() < 1 / (1 + math.exp(-eta)),
                        "acted": rng.random() < 0.8,
                        "recalled_before": rng.random() < 0.6,
                        "span_calls": 0 if (i == 1 and j == 3) else 3,
                    }
                )
    return sessions, opps


SETTINGS = e1_analysis.Settings(n_boot=60, seed=3)


def test_analysis_set_exclusions() -> None:
    sessions, opps = synthetic()
    rows, exclusions = e1_analysis.analysis_set(sessions, opps)
    assert exclusions["sessions_not_reaching_first_dp"] == 10
    assert exclusions["opportunities_zero_calls"] == 2
    assert all(r["span_calls"] > 0 for r in rows)
    assert exclusions["opportunities_analyzed"] == len(rows)


def test_full_analysis_outputs_are_complete_and_json_safe(tmp_path: Path) -> None:
    sessions, opps = synthetic()
    out = e1_analysis.analyze(sessions, opps, SETTINGS)
    assert set(out) == {"hazard", "half_life", "km", "contrasts", "descriptive"}
    terms = {(e["family"], e["term"]) for e in out["hazard"]["estimates"]}
    assert ("pooled", "log1p(position)") in terms and ("fam-a", "compaction_count") in terms
    assert any(t == "family[fam-b]" for _, t in terms)
    for est in out["hazard"]["estimates"]:
        assert est["lo"] <= est["estimate"] <= est["hi"]
    assert [e["family"] for e in out["half_life"]["estimates"]] == ["fam-a", "fam-b"]
    assert out["km"]["descriptive"] is True and len(out["km"]["curves"]) == 2
    assert {e["family"] for e in out["contrasts"]["h2a"]} == {"fam-a", "fam-b", "pooled"}
    pairs = out["contrasts"]["h3_pairs"]
    assert len(pairs) == 6 and all(
        0 <= p["p_holm"] <= 1 and p["p_holm"] >= p["p_value"] for p in pairs
    )
    assert out["descriptive"]["by_family"][0]["acted"] > 0
    for data in out.values():
        json.dumps(data, allow_nan=False)
    sdir = tmp_path / "e1"
    sdir.mkdir()
    (sdir / "sessions.jsonl").write_text("\n".join(json.dumps(s) for s in sessions) + "\n")
    (sdir / "opportunities.jsonl").write_text("\n".join(json.dumps(o) for o in opps) + "\n")
    assert e1_analysis.main(["--dir", str(sdir), "--n-boot", "40", "--seed", "1"]) == 0
    assert (sdir / "half_life.json").is_file() and (sdir / "contrasts.json").is_file()


def test_holm_and_empty_input() -> None:
    adjusted = e1_analysis.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
    with pytest.raises(stats.StatsError):
        e1_analysis.analyze([], [], SETTINGS)

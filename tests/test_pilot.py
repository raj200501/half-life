"""Tests for the E1 family choice on synthetic pilot rows (labeled synthetic)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from halflife import pilot

TEMPLATES = ("tpl-a", "tpl-b", "tpl-c")


def _family(
    family: str, reached: int, acted: int, wall: float, templates: tuple[str, ...] = TEMPLATES
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    model = f"{family}-q4"
    sessions = [
        {
            "session_id": f"syn-{t}",
            "family": family,
            "model": model,
            "template_id": t,
            "reached_first_dp": i < reached,
        }
        for i, t in enumerate(templates)
    ]
    opportunities = (
        [
            {"session_id": f"syn-{templates[i % reached]}", "family": family, "acted": j < acted}
            for j, i in enumerate(range(4 * reached))
        ]
        if reached
        else []
    )
    ledger = [
        {"experiment": "pilot", "purpose": "step", "model": model, "wall_s": wall + k}
        for k in range(3)
    ] + [{"experiment": "pilot", "purpose": "probe", "model": model, "wall_s": 999.0}]
    return sessions, opportunities, ledger


def _merge(*parts: tuple[list[Any], list[Any], list[Any]]) -> tuple[list[Any], ...]:
    return tuple([row for part in parts for row in part[i]] for i in range(3))


def test_rule_needs_reach_engagement_and_full_coverage() -> None:
    sessions, opportunities, ledger = _merge(
        _family("fast-idle", 3, 1, 5.0),  # reaches, but acts at 1 of 12
        _family("quick", 3, 12, 10.0),
        _family("steady", 3, 6, 20.0),  # acts at exactly half: qualifies
        _family("slow", 3, 12, 40.0),
        _family("partial", 2, 8, 1.0),  # reach 2 of 3 < 80%
        _family("unpiloted", 1, 4, 2.0, templates=("tpl-a",)),
    )
    table = {r["family"]: r for r in pilot.family_table(sessions, opportunities, ledger, TEMPLATES)}
    assert table["fast-idle"]["qualifies"] is False and table["fast-idle"]["acted"] == 1
    assert table["steady"]["qualifies"] is True
    assert table["partial"]["qualifies"] is False and table["partial"]["reached_first_dp"] == 2
    assert table["unpiloted"]["fully_piloted"] is False and table["unpiloted"]["qualifies"] is False
    assert table["quick"]["median_step_wall_s"] == 11.0 and table["quick"]["step_calls"] == 3
    assert pilot.choose(list(table.values())) == ["quick", "steady"]


def test_no_decision_points_means_no_engagement() -> None:
    sessions, opportunities, ledger = _family("silent", 0, 0, 3.0)
    (row,) = pilot.family_table(sessions, opportunities, ledger, TEMPLATES)
    assert row["acted_share"] is None and row["qualifies"] is False
    assert pilot.choose([row]) == []


def test_main_banks_the_choice(tmp_path: Path) -> None:
    sessions, opportunities, ledger = _merge(_family("quick", 3, 12, 10.0), _family("b", 3, 9, 9.0))
    for name, rows in (("sessions", sessions), ("opportunities", opportunities)):
        (tmp_path / f"{name}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
    ledger_path = tmp_path / "ledger.jsonl"
    ledger_path.write_text("".join(json.dumps(r) + "\n" for r in ledger), encoding="utf-8")
    argv = ["--dir", str(tmp_path), "--ledger", str(ledger_path), "--templates", *TEMPLATES]
    assert pilot.main(argv) == 0
    banked = json.loads((tmp_path / "family_choice.json").read_text(encoding="utf-8"))
    assert banked["chosen"] == ["b", "quick"] and banked["exploratory"] is True
    assert banked["command"].startswith("python -m halflife.pilot --dir")

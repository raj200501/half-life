"""Tests for design files, plan generation, and banking runs."""

from __future__ import annotations

import json
import random
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from halflife import bench, experiment
from halflife.schema import SchemaError

EXEMPLAR = bench.TEMPLATES_DIR / "invoice-ledger.yaml"

DESIGN: dict[str, Any] = {
    "experiment": "e1",
    "template_ids": ["invoice-ledger"],
    "n_sessions": 6,
    "seed_base": 100,
    "confirmatory": True,
    "conditions": [
        {"placement": "first_user", "strategy": "full_summary"},
        {"placement": "first_user", "strategy": "sham"},
    ],
    "settings": {"compactions": [1, 2], "compaction_window": [5, 60], "budget_tokens": 2000},
}


def test_design_parsing_and_validation(tmp_path: Path) -> None:
    path = tmp_path / "design.yaml"
    import yaml

    path.write_text(yaml.safe_dump(DESIGN), encoding="utf-8")
    design = experiment.load_design(path)
    assert design.settings.compactions == (1, 2) and design.settings.budget_tokens == 2000
    assert design.confirmatory and len(design.conditions) == 2
    for bad in ({"conditions": []}, {"split": "main"}, {"n_sessions": 0}):
        with pytest.raises(SchemaError):
            experiment.Design.from_dict(DESIGN | bad)
    path.write_text("[1]", encoding="utf-8")
    with pytest.raises(SchemaError, match="mapping"):
        experiment.load_design(path)


def test_blocked_randomization_is_balanced() -> None:
    picks = experiment.blocked(["a", "b", "c"], 9, random.Random(0))
    assert Counter(picks) == {"a": 3, "b": 3, "c": 3}
    assert len(experiment.blocked(["a", "b"], 3, random.Random(0))) == 3


def test_make_plans_deterministic_and_balanced() -> None:
    design = experiment.Design.from_dict(DESIGN)
    templates = bench.load_all(split=None)
    plans = experiment.make_plans(design, templates)
    assert len(plans) == 6
    assert [p.seed for p in plans] == list(range(100, 106))
    assert Counter(p.compaction.strategy for p in plans) == {"full_summary": 3, "sham": 3}
    assert all(p.confirmatory and p.experiment == "e1" for p in plans)
    assert plans == experiment.make_plans(design, templates)
    with pytest.raises(SchemaError, match="do not exist"):
        experiment.make_plans(
            experiment.Design.from_dict(DESIGN | {"template_ids": ["nope"]}), templates
        )
    with pytest.raises(SchemaError, match="no templates"):
        experiment.make_plans(
            experiment.Design.from_dict(DESIGN | {"template_ids": [], "split": "heldout"}), []
        )


def _fake_run(run_dir: Path, sid: str) -> None:
    sdir = run_dir / sid
    (sdir / "compactions").mkdir(parents=True)
    (sdir / "compactions" / "c1.json").write_text('{"x": 1}', encoding="utf-8")
    (sdir / "probes.json").write_text(
        json.dumps([{"checkpoint": "before_dp:d", "recalled": {}}]), encoding="utf-8"
    )
    record = {
        "run_id": run_dir.name,
        "session": {
            "session_id": sid,
            "confirmatory": True,
            "family": "f",
            "reached_first_dp": True,
        },
        "opportunities": [{"session_id": sid, "position": 4, "violated": False}],
        "compactions": [{"event_id": "c1", "log_path": "compactions/c1.json", "log_sha256": "x"}],
    }
    (sdir / "result.json").write_text(json.dumps(record), encoding="utf-8")


def test_bank_runs(tmp_path: Path) -> None:
    root = tmp_path
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@e.invalid",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "x",
        ],
        check=True,
    )
    runs = root / "runs" / "r1"
    _fake_run(runs, "s2")
    _fake_run(runs, "s1")
    design = root / "design.yaml"
    shutil.copy(EXEMPLAR, design)
    out = root / "artifacts" / "e1"
    prov = experiment.bank_runs(
        [runs],
        out,
        root,
        experiment="e1",
        prereg="prereg/0001-e1.md",
        design_path=design,
        models=[{"family": "f"}],
        command="test",
    )
    sessions = [json.loads(line) for line in (out / "sessions.jsonl").read_text().splitlines()]
    assert [s["session_id"] for s in sessions] == ["s1", "s2"]
    assert all(s["prereg"] == "prereg/0001-e1.md" for s in sessions)
    events = json.loads((out / "compaction_logs.json").read_text())["events"]
    assert events[0]["log_path"] == "artifacts/e1/compactions/r1/s1/c1.json"
    assert (root / events[0]["log_path"]).is_file()
    assert (
        prov["n_sessions"] == 2 and prov["confirmatory"] is True and prov["design"] == "design.yaml"
    )
    assert len((out / "probes.jsonl").read_text().splitlines()) == 2
    exploratory = experiment.bank_runs(
        [runs],
        root / "artifacts" / "x",
        root,
        experiment="x",
        prereg=None,
        design_path=None,
        models=[],
        command="t",
    )
    assert exploratory["confirmatory"] is False and exploratory["design"] is None

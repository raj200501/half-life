"""Tests for design files, plan generation, and banking runs."""

from __future__ import annotations

import dataclasses
import hashlib
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


PREREG = "prereg/0001-e1.md"


def _git_repo(root: Path) -> None:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    _commit(root, "x", allow_empty=True)


def _commit(root: Path, message: str, *, allow_empty: bool = False) -> None:
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    command = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@e.invalid"]
    command += ["commit", "-q", "-m", message] + (["--allow-empty"] if allow_empty else [])
    subprocess.run(command, check=True)


def _plan(sid: str, *, experiment_name: str = "e1", confirmatory: bool = True) -> Any:
    from halflife.schema import CompactionPlan, SessionPlan, load_template

    template = load_template(EXEMPLAR)
    return SessionPlan(
        session_id=sid,
        template_id=template.id,
        template_sha256="0" * 64,
        seed=1,
        placement="first_user",
        phrasing="bare",
        compaction=CompactionPlan(strategy="full_summary", schedule=(3,), budget_tokens=0),
        mitigation="none",
        subtasks=template.subtasks[:2],
        experiment=experiment_name,
        confirmatory=confirmatory,
    )


def _dump(path: Path, data: Any) -> None:
    """JSON the way halflife.runner writes it (indented, sorted keys, final newline)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fake_run(
    root: Path,
    run_id: str,
    plans: list[Any],
    *,
    prereg: str | None,
    plans_file: str | None = "plans.jsonl",
) -> Path:
    """A finished run shaped like halflife.runner output (synthetic sessions, no model)."""
    from halflife import analysis, runner

    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True)
    if plans_file is not None:
        runner.write_plans(root / plans_file, plans)
        record = {
            "run_id": run_id,
            "plans": plans_file,
            "plans_sha256": experiment._sha256(root / plans_file),
            "prereg": prereg,
            "git_sha": "a" * 40,
            "started": "2026-11-04T10:00:00Z",
        }
        _dump(run_dir / "run.json", record)
        (run_dir / "invocations.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    for plan in plans:
        sdir = run_dir / plan.session_id
        log = {"event_id": "c1", "strategy": "full_summary", "tool_call_index": 3}
        _dump(sdir / "compactions" / "c1.json", log)
        canonical = json.dumps(log, sort_keys=True, indent=2).encode("utf-8")
        _dump(sdir / "probes.json", [{"checkpoint": "before_dp:d", "recalled": {}}])
        _dump(sdir / "plan.json", plan.to_dict())
        (sdir / "action_log.json").write_text("{}\n", encoding="utf-8")
        _dump(sdir / "violations.json", [])
        session = {
            "session_id": plan.session_id,
            "experiment": plan.experiment,
            "confirmatory": plan.confirmatory,
            "prereg": prereg,
            "plan_sha256": analysis.plan_sha256(plan),
            "family": "f",
            "reached_first_dp": True,
        }
        result = {
            "run_id": run_id,
            "session": session,
            "opportunities": [{"session_id": plan.session_id, "position": 4, "violated": False}],
            "compactions": [
                {
                    "event_id": "c1",
                    "log_path": "compactions/c1.json",
                    "log_sha256": hashlib.sha256(canonical).hexdigest(),
                }
            ],
            "settings": {"temperature": 0.7, "max_step_tokens": 384},
        }
        _dump(sdir / "result.json", result)
    return run_dir


def _bank(root: Path, runs: list[Path], name: str, prereg: str | None, kind: str = "e1") -> Any:
    return experiment.bank_runs(
        runs,
        root / "artifacts" / name,
        root,
        experiment=kind,
        prereg=prereg,
        design_path=None,
        models=[],
        command="test",
    )


def test_bank_runs(tmp_path: Path) -> None:
    root = tmp_path
    _git_repo(root)
    runs = _fake_run(root, "r1", [_plan("s2"), _plan("s1")], prereg=PREREG)
    usage = {"step": {"calls": 2, "prompt_tokens": 30, "completion_tokens": 8, "elapsed_s": 1.5}}
    (runs / "s1" / "usage.json").write_text(json.dumps(usage), encoding="utf-8")
    meta = {"git_sha": "b" * 40, "git_dirty": False, "started": "x", "finished": "y"}
    _dump(runs / "s1" / "run_meta.json", meta)
    stamp = {"run_id": "r1", "git_sha": "c" * 40, "started": "a", "finished": "b"}
    result = json.loads((runs / "s2" / "result.json").read_text())
    _dump(runs / "s2" / "result.json", result | {"run_provenance": stamp})
    design = root / "design.yaml"
    shutil.copy(EXEMPLAR, design)
    out = root / "artifacts" / "e1"
    prov = experiment.bank_runs(
        [runs],
        out,
        root,
        experiment="e1",
        prereg=PREREG,
        design_path=design,
        models=[{"family": "f"}],
        command="test",
    )
    sessions = [json.loads(line) for line in (out / "sessions.jsonl").read_text().splitlines()]
    assert [s["session_id"] for s in sessions] == ["s1", "s2"]
    assert all(s["prereg"] == PREREG for s in sessions)  # the run-time tag, kept
    assert sessions[0]["usage"] == usage and sessions[1]["usage"] is None
    assert sessions[0]["run_meta"] == meta and sessions[1]["run_meta"] is None
    assert sessions[0]["run_provenance"] is None and sessions[1]["run_provenance"] == stamp
    assert sessions[0]["run_settings"] == {"temperature": 0.7, "max_step_tokens": 384}
    files = sessions[0]["files_sha256"]
    assert files["plan.json"] == experiment._sha256(runs / "s1" / "plan.json")
    assert set(files) == {"plan.json", "action_log.json", "violations.json"}
    lines = (out / "opportunities.jsonl").read_text().splitlines()
    opportunities = [json.loads(line) for line in lines]
    assert all(o["prereg"] == PREREG for o in opportunities)
    events = json.loads((out / "compaction_logs.json").read_text())["events"]
    assert events[0]["log_path"] == "artifacts/e1/compactions/r1/s1/c1.json"
    assert (root / events[0]["log_path"]).is_file()
    assert events[0]["log_sha256"] == experiment._sha256(root / events[0]["log_path"])
    assert events[0]["run_log_sha256"] != events[0]["log_sha256"]  # canonical JSON, no newline
    assert (
        prov["n_sessions"] == 2 and prov["confirmatory"] is True and prov["design"] == "design.yaml"
    )
    assert prov["run_records"]["r1"]["run"]["git_sha"] == "a" * 40
    assert len(prov["run_records"]["r1"]["invocations"]) == 1
    assert len((out / "probes.jsonl").read_text().splitlines()) == 2
    plain = _fake_run(root, "r2", [_plan("p1", confirmatory=False)], prereg=None, plans_file=None)
    exploratory = _bank(root, [plain], "x", None)
    assert exploratory["confirmatory"] is False and exploratory["design"] is None


def _refused(root: Path, runs: list[Path], prereg: str | None, match: str) -> None:
    with pytest.raises(experiment.BankError, match=match):
        _bank(root, runs, "refused", prereg)
    assert not (root / "artifacts" / "refused").exists()  # nothing is written


def test_bank_runs_refuses_sessions_that_did_not_run_as_banked(tmp_path: Path) -> None:
    root = tmp_path
    _git_repo(root)
    # Run without a preregistration tag, banked as preregistered: refused, not retagged.
    untagged = _fake_run(root, "r1", [_plan("s1")], prereg=None, plans_file="p1.jsonl")
    _refused(root, [untagged], PREREG, "ran under preregistration None")
    # A tagged session banked as exploratory, or under another preregistration.
    tagged = _fake_run(root, "r2", [_plan("s2")], prereg=PREREG, plans_file="p2.jsonl")
    _refused(root, [tagged], None, "ran under preregistration 'prereg/0001-e1.md'")
    _refused(root, [tagged], "prereg/0002-e3.md", "ran under preregistration")
    # A session of another experiment (an E3 run listed when banking E1).
    e3 = _fake_run(root, "r3", [_plan("s3", experiment_name="e3")], prereg=PREREG, plans_file="p3")
    _refused(root, [e3], PREREG, "experiment 'e3'")
    # The plans file regenerated after the run: the session ran another plan.
    stale = _fake_run(root, "r4", [_plan("s4")], prereg=PREREG, plans_file="p4.jsonl")
    from halflife import runner

    runner.write_plans(root / "p4.jsonl", [dataclasses.replace(_plan("s4"), seed=2)])
    _refused(root, [stale], PREREG, "plans file changed since the run")
    record = json.loads((stale / "run.json").read_text())
    _dump(stale / "run.json", record | {"plans_sha256": experiment._sha256(root / "p4.jsonl")})
    _refused(root, [stale], PREREG, "ran a different plan than the run's plans file")
    # An orphaned session directory whose plan is not in the run's plans file.
    orphan = _fake_run(root, "r5", [_plan("s5")], prereg=PREREG, plans_file="p5.jsonl")
    runner.write_plans(root / "p5.jsonl", [_plan("other")])
    record = json.loads((orphan / "run.json").read_text())
    _dump(orphan / "run.json", record | {"plans_sha256": experiment._sha256(root / "p5.jsonl")})
    _refused(root, [orphan], PREREG, "is not in the run's plans file")
    # A confirmatory bank needs each run's record to check the plans against.
    bare = _fake_run(root, "r6", [_plan("s6")], prereg=PREREG, plans_file=None)
    _refused(root, [bare], PREREG, "no run.json")
    # A compaction log edited after its session.
    edited = _fake_run(root, "r7", [_plan("s7")], prereg=PREREG, plans_file="p7.jsonl")
    _dump(edited / "s7" / "compactions" / "c1.json", {"event_id": "c1", "summary": "edited"})
    _refused(root, [edited], PREREG, "compaction log changed since its session")
    assert _bank(root, [tagged], "ok", PREREG)["n_sessions"] == 1


def test_cmd_bank_records_the_command_without_the_script_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import argparse

    root = tmp_path
    _git_repo(root)
    run_dir = _fake_run(root, "r1", [_plan("s1")], prereg=PREREG)
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    _dump(run_dir / "run.json", record | {"model": {"key": "m", "family": "f"}})
    # Run as ``python -m``, sys.argv[0] is the module's absolute path (a home directory).
    flags = ["--experiment", "e1", "--runs", "runs/r1", "--out", "artifacts/e1"]
    monkeypatch.setattr("sys.argv", ["/home/someone/src/halflife/experiment.py", "bank", *flags])
    args = argparse.Namespace(
        experiment="e1", runs=["runs/r1"], out="artifacts/e1", prereg=PREREG, design=None
    )
    assert experiment.cmd_bank(args, root) == 0
    prov = json.loads((root / "artifacts" / "e1" / "provenance.json").read_text(encoding="utf-8"))
    assert prov["command"] == "python -m halflife.experiment bank " + " ".join(flags)
    assert "/home/someone" not in (root / "artifacts" / "e1" / "provenance.json").read_text()


def _run_args(plans: Path, **over: Any) -> Any:
    import argparse

    values = {
        "plans": plans,
        "run_id": "r1",
        "model": "no-such-model",
        "n_ctx": 8192,
        "temperature": 0.7,
        "max_step_tokens": 384,
        "prereg": None,
        "limit": None,
        "index": None,
        "out_root": "runs",
    }
    return argparse.Namespace(**(values | over))


def test_cmd_run_refuses_confirmatory_plans_without_a_committed_preregistration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from halflife import models as models_mod
    from halflife import runner

    root = tmp_path
    _git_repo(root)
    template = next(t for t in bench.load_all() if t.id == "invoice-ledger")
    plan = dataclasses.replace(_plan("s1"), template_sha256=template.sha256)
    runner.write_plans(root / "plans.jsonl", [plan])
    assert experiment.cmd_run(_run_args(root / "plans.jsonl"), root) == 2
    assert "confirmatory plans need --prereg" in capsys.readouterr().out
    assert not (root / "runs" / "r1").exists()
    for prereg, message in (
        ("notes/e1.md", "not a prereg/NNNN-slug.md path"),
        (PREREG, "does not exist"),
    ):
        args = _run_args(root / "plans.jsonl", prereg=prereg)
        assert experiment.cmd_run(args, root) == 2
        assert message in capsys.readouterr().out
    (root / "prereg").mkdir()
    (root / PREREG).write_text("# E1\n", encoding="utf-8")
    assert experiment.cmd_run(_run_args(root / "plans.jsonl", prereg=PREREG), root) == 2
    assert "is not committed" in capsys.readouterr().out
    _commit(root, "freeze E1")
    (root / PREREG).write_text("# E1, edited\n", encoding="utf-8")
    assert experiment.cmd_run(_run_args(root / "plans.jsonl", prereg=PREREG), root) == 2
    assert "has uncommitted changes" in capsys.readouterr().out
    subprocess.run(["git", "-C", str(root), "checkout", "--", PREREG], check=True)

    class PastTheChecks(Exception):
        pass

    def stop(key: str) -> Any:
        raise PastTheChecks(key)

    monkeypatch.setattr(models_mod, "get_spec", stop)
    with pytest.raises(PastTheChecks):
        experiment.cmd_run(_run_args(root / "plans.jsonl", prereg=PREREG), root)
    # A preregistration tag on plans that are not confirmatory is refused too.
    loose = dataclasses.replace(plan, confirmatory=False)
    runner.write_plans(root / "loose.jsonl", [loose])
    assert experiment.cmd_run(_run_args(root / "loose.jsonl", prereg=PREREG), root) == 2
    assert "is not confirmatory" in capsys.readouterr().out


def test_cmd_run_refuses_to_resume_a_session_whose_stored_plan_differs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from halflife import runner

    root = tmp_path
    _git_repo(root)
    template = next(t for t in bench.load_all() if t.id == "invoice-ledger")
    plan = dataclasses.replace(_plan("s1", confirmatory=False), template_sha256=template.sha256)
    runner.write_plans(root / "plans.jsonl", [plan])
    stored = dataclasses.replace(plan, seed=99)
    _dump(root / "runs" / "r1" / "s1" / "plan.json", stored.to_dict())
    assert experiment.cmd_run(_run_args(root / "plans.jsonl"), root) == 2
    out = capsys.readouterr().out
    assert "s1: the stored plan differs from the plan being run" in out
    assert not (root / "runs" / "r1" / "pid").exists()


def test_e0_validation_from_real_runs(tmp_path: Path) -> None:
    from halflife import runner
    from halflife.detect import detect_all
    from halflife.schema import CompactionPlan, SessionPlan, load_template

    sys_path_root = Path(__file__).resolve().parents[1]
    import sys

    sys.path.insert(0, str(sys_path_root / "tests"))
    from test_runner import ScriptedPolicy

    template = load_template(EXEMPLAR)
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
    ctx = runner.RunContext(run_id="val", family="fake", model="scripted")
    for seed in (1, 2):
        plan = SessionPlan(
            session_id=f"v{seed}",
            template_id=template.id,
            template_sha256="0" * 64,
            seed=seed,
            placement="first_user",
            phrasing="bare",
            compaction=CompactionPlan(strategy="full_summary", schedule=(3,), budget_tokens=0),
            mitigation="none",
            subtasks=template.subtasks[:3],
            experiment="pilot",
        )
        runner.run_session(plan, template, ScriptedPolicy(), detect_all, ctx, root / "runs")
    banked = root / "artifacts" / "pilot"
    experiment.bank_runs(
        [root / "runs" / "val"],
        banked,
        root,
        experiment="pilot",
        prereg=None,
        design_path=None,
        models=[],
        command="test",
    )
    ids = {template.id: [c.id for c in template.constraints]}
    summary = experiment.e0_validation(banked, root / "artifacts" / "e0", root, ids)
    assert summary == {"validation_runs": 2, "leaks": 0, "events": 2, "incomplete": 0}
    leakage = json.loads((root / "artifacts" / "e0" / "probe_leakage.json").read_text())
    assert leakage["validation_runs"] == len(leakage["runs"]) == 2 and leakage["leaks"] == 0
    logs = json.loads((root / "artifacts" / "e0" / "compaction_logs.json").read_text())
    event = logs["events"][0]
    for name in (
        "run_id",
        "event_id",
        "strategy",
        "tool_call_index",
        "tokens_before",
        "tokens_after",
        "log_path",
        "log_sha256",
    ):
        assert event[name] is not None, name
    tampered = json.loads((root / event["log_path"]).read_text())
    del tampered["constraint_text_retained"]["changelog"]
    assert experiment.log_problems(tampered, ids[template.id]) == [
        "retention not logged for changelog"
    ]


def test_log_problems_flags_missing_parts() -> None:
    assert experiment.log_problems({}, ["c"])[0] == "missing event_id"
    base = {
        "event_id": "c1",
        "strategy": "full_summary",
        "tool_call_index": 1,
        "scheduled": True,
        "tokens_before": 10,
        "tokens_after": 5,
        "before": [{"role": "system", "content": "s"}],
        "after": [{"role": "user", "content": "x"}],
        "summary": "",
        "constraint_text_retained": {"c": False},
    }
    problems = experiment.log_problems(base, ["c"])
    assert "compacted context lost the system prompt" in problems and "no summary text" in problems
    assert "empty before or after context" in experiment.log_problems(base | {"after": []}, ["c"])


def test_kit_hooks_wire_the_e4_arms() -> None:
    from halflife.kit.compaction_prompt import preserve_prompt
    from halflife.kit.reinject import reinject_text
    from halflife.schema import Action
    from halflife.workspace import Workspace

    hooks = experiment.kit_hooks()
    assert hooks.summary_prompt == preserve_prompt()
    assert hooks.reinject_text is reinject_text
    assert all(len(v) == 64 for k, v in hooks.provenance.items() if k.endswith("_sha256"))
    template = next(t for t in bench.load_all() if t.id == "invoice-ledger").template
    with Workspace(template.files) as workspace:
        guard = hooks.guard_factory(template, workspace)
        refusal = guard(Action("write_file", {"path": "tests/test_new.py", "content": "x = 1\n"}))
        assert refusal is not None and "tests/" in refusal
        assert guard(Action("read_file", {"path": "tests/test_invoice.py"})) is None

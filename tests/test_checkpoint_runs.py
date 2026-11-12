"""Tests for tools/checkpoint_runs.py: finished sessions archived once, reproducibly."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

TOOL = Path(__file__).resolve().parents[1] / "tools" / "checkpoint_runs.py"


def load_tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("checkpoint_runs", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["checkpoint_runs"] = module
    spec.loader.exec_module(module)
    return module


def fake_run(root: Path) -> Path:
    """A run directory with one finished session, one unfinished, and run-level files."""
    run = root / "runs" / "e1-smollm"
    done = run / "s-done"
    (done / "compactions").mkdir(parents=True)
    (done / "result.json").write_text(json.dumps({"run_id": "e1-smollm"}), encoding="utf-8")
    (done / "plan.json").write_text("{}\n", encoding="utf-8")
    (done / "compactions" / "c1.json").write_text("{}\n", encoding="utf-8")
    partial = run / "s-partial"
    partial.mkdir()
    (partial / "plan.json").write_text("{}\n", encoding="utf-8")
    (run / "run.json").write_text("{}\n", encoding="utf-8")
    return run


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_only_finished_sessions_are_archived_once(tmp_path: Path) -> None:
    tool = load_tool()
    run = fake_run(tmp_path)
    out = tmp_path / "artifacts" / "e1" / "raw"
    written = tool.checkpoint([run], out)
    archive = out / "e1-smollm" / "s-done.tar.gz"
    assert written == [archive] and archive.is_file()
    assert not (out / "e1-smollm" / "s-partial.tar.gz").exists()
    first = digest(archive)
    assert tool.checkpoint([run], out) == []  # already archived: never rewritten
    assert digest(archive) == first


def test_archives_are_reproducible_and_restore_the_session(tmp_path: Path) -> None:
    tool = load_tool()
    run = fake_run(tmp_path)
    one, two = tmp_path / "one", tmp_path / "two"
    tool.checkpoint([run], one)
    (run / "s-done" / "plan.json").touch()  # a newer mtime must not change the bytes
    tool.checkpoint([run], two)
    name = Path("e1-smollm") / "s-done.tar.gz"
    assert digest(one / name) == digest(two / name)
    restored = tmp_path / "restored"
    assert tool.restore(one, restored) == [restored / "e1-smollm" / "s-done"]
    session = restored / "e1-smollm" / "s-done"
    assert json.loads((session / "result.json").read_text(encoding="utf-8")) == {
        "run_id": "e1-smollm"
    }
    assert (session / "compactions" / "c1.json").is_file()
    assert tool.restore(one, restored) == []  # an existing session is left alone


def test_restore_refuses_paths_outside_the_target(tmp_path: Path) -> None:
    import tarfile

    tool = load_tool()
    evil = tmp_path / "raw" / "e1-x" / "s.tar.gz"
    evil.parent.mkdir(parents=True)
    payload = tmp_path / "payload"
    payload.write_text("x", encoding="utf-8")
    with tarfile.open(evil, "w:gz") as tar:
        tar.add(payload, arcname="../../escape.txt")
    target = tmp_path / "target"
    try:
        tool.restore(tmp_path / "raw", target)
    except ValueError as exc:
        assert "outside" in str(exc)
    else:  # pragma: no cover - the restore must refuse
        raise AssertionError("restore accepted a path outside the target")
    assert not (tmp_path / "escape.txt").exists()


def test_cli_checkpoints_and_restores(tmp_path: Path) -> None:
    tool = load_tool()
    run = fake_run(tmp_path)
    out = tmp_path / "raw"
    assert tool.main(["--out", str(out), str(run)]) == 0
    assert (out / "e1-smollm" / "s-done.tar.gz").is_file()
    assert tool.main(["--restore", "--out", str(out), str(tmp_path / "back")]) == 0
    assert (tmp_path / "back" / "e1-smollm" / "s-done" / "result.json").is_file()


def test_run_records_are_archived_and_refreshed_but_progress_is_not(tmp_path: Path) -> None:
    """G7 checks every invocation (D-066), so each run's run.json and invocations.jsonl
    are kept beside the session archives and refreshed as they grow; progress.jsonl, which
    carries outcomes, is not copied (amendment 2: no interim analysis)."""
    tool = load_tool()
    run = fake_run(tmp_path)
    (run / "invocations.jsonl").write_text('{"run_id": "e1-smollm"}\n', encoding="utf-8")
    (run / "progress.jsonl").write_text('{"violations": 1}\n', encoding="utf-8")
    out = tmp_path / "raw"
    tool.checkpoint([run], out)
    for name in ("run.json", "invocations.jsonl"):
        assert (out / "e1-smollm" / name).read_bytes() == (run / name).read_bytes()
    assert not (out / "e1-smollm" / "progress.jsonl").exists()
    with (run / "invocations.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"run_id": "e1-smollm", "second": true}\n')
    tool.checkpoint([run], out)
    copied = (out / "e1-smollm" / "invocations.jsonl").read_text(encoding="utf-8")
    assert copied.count("\n") == 2


def test_restore_puts_run_records_back_when_missing(tmp_path: Path) -> None:
    tool = load_tool()
    run = fake_run(tmp_path)
    (run / "invocations.jsonl").write_text('{"run_id": "e1-smollm"}\n', encoding="utf-8")
    out = tmp_path / "raw"
    tool.checkpoint([run], out)
    target = tmp_path / "new-machine"
    tool.restore(out, target)
    for name in ("run.json", "invocations.jsonl"):
        assert (target / "e1-smollm" / name).read_bytes() == (run / name).read_bytes()
    (target / "e1-smollm" / "run.json").write_text('{"kept": true}\n', encoding="utf-8")
    tool.restore(out, target)  # an existing run record is left alone
    assert (target / "e1-smollm" / "run.json").read_text(encoding="utf-8") == '{"kept": true}\n'

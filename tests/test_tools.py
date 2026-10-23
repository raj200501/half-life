"""Tests for halflife.tools (real workspace, real sandbox for run)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from halflife import tools
from halflife.sandbox import SandboxError, SandboxOptions
from halflife.schema import AGENT_TOOLS, Action, ToolEvent
from halflife.tools import (
    TOOL_SPECS,
    ToolArgumentError,
    check_args,
    execute,
    execute_call,
    invalid_event,
    json_schema_for_tool_call,
    render_tool_docs,
    tool_spec,
)
from halflife.workspace import Workspace

FILES = {"src/app.py": "def f(x):\n    return x\n", "notes.txt": "one two two\n"}


@pytest.fixture
def ws() -> Iterator[Workspace]:
    with Workspace(FILES) as workspace:
        yield workspace


def _call(ws: Workspace, tool: str, args: Any, index: int = 0) -> ToolEvent:
    return execute_call(tool, args, ws, index=index, subtask_id="s1")


def test_specs_match_agent_tools() -> None:
    assert tuple(spec.name for spec in TOOL_SPECS) == AGENT_TOOLS
    for spec in TOOL_SPECS:
        assert spec.doc and spec.doc.endswith(".")
        assert spec.parameters["type"] == "object"
        assert spec.parameters["additionalProperties"] is False
    assert tool_spec("list_files").signature() == 'list_files(path=".")'
    assert tool_spec("replace_in_file").signature() == "replace_in_file(path, old, new)"
    docs = render_tool_docs()
    assert docs.count("\n") == len(AGENT_TOOLS) - 1
    assert "- run(cmd): " in docs
    with pytest.raises(KeyError):
        tool_spec("bash")


def test_json_schema_for_tool_call() -> None:
    schema = json_schema_for_tool_call()
    branches = schema["anyOf"]
    assert [b["properties"]["tool"]["const"] for b in branches] == list(AGENT_TOOLS)
    for branch in branches:
        assert branch["required"] == ["tool", "args"]
        assert branch["properties"]["args"]["type"] == "object"
    loose = json_schema_for_tool_call(per_tool=False)
    assert loose["properties"]["tool"]["enum"] == list(AGENT_TOOLS)
    json.dumps(schema)


def test_check_args() -> None:
    assert check_args("list_files", {}) == {"path": "."}
    assert check_args("run", {"cmd": "ls"}) == {"cmd": "ls"}
    with pytest.raises(ToolArgumentError, match="unknown tool"):
        check_args("bash", {})
    with pytest.raises(ToolArgumentError, match="JSON object"):
        check_args("run", ["ls"])
    with pytest.raises(ToolArgumentError, match="missing"):
        check_args("write_file", {"path": "a"})
    with pytest.raises(ToolArgumentError, match="unexpected"):
        check_args("read_file", {"path": "a", "mode": "r"})
    with pytest.raises(ToolArgumentError, match="must be a string"):
        check_args("read_file", {"path": 3})


def test_list_and_read(ws: Workspace) -> None:
    event = _call(ws, "list_files", {})
    assert event.output == "notes.txt\nsrc/app.py"
    assert event.tool == "list_files" and event.exit_code is None
    assert event.fs_after == ws.baseline
    assert _call(ws, "list_files", {"path": "src"}).output == "src/app.py"
    (ws.root / "empty").mkdir()
    assert _call(ws, "list_files", {"path": "empty"}).output == "(no files)"
    assert _call(ws, "read_file", {"path": "src/app.py"}).output == FILES["src/app.py"]
    assert _call(ws, "read_file", {"path": "src"}).output.startswith("error: src is a directory")
    assert _call(ws, "read_file", {"path": "nope.py"}).output == "error: nope.py does not exist"
    (ws.root / "bin.dat").write_bytes(b"\xff\xfe")
    assert "not a UTF-8 text file" in _call(ws, "read_file", {"path": "bin.dat"}).output


def test_write_file_updates_fs_after(ws: Workspace) -> None:
    event = _call(ws, "write_file", {"path": "src/new/mod.py", "content": "x = 1\n"})
    assert event.output == "wrote 6 characters to src/new/mod.py"
    assert "src/new/mod.py" in event.fs_after
    assert ws.blobs[event.fs_after["src/new/mod.py"]] == "x = 1\n"
    assert _call(ws, "write_file", {"path": "src", "content": ""}).output.startswith("error:")


def test_replace_in_file(ws: Workspace) -> None:
    ok = _call(ws, "replace_in_file", {"path": "src/app.py", "old": "x\n", "new": "x + 1\n"})
    assert ok.output == "replaced 1 occurrence in src/app.py"
    assert ws.read_text("src/app.py") == "def f(x):\n    return x + 1\n"
    missing = _call(ws, "replace_in_file", {"path": "notes.txt", "old": "three", "new": "3"})
    assert missing.output.startswith("error: old text not found")
    ambiguous = _call(ws, "replace_in_file", {"path": "notes.txt", "old": "two", "new": "2"})
    assert "occurs 2 times" in ambiguous.output
    ws.write_text("aaa.txt", "aaa")
    overlap = _call(ws, "replace_in_file", {"path": "aaa.txt", "old": "aa", "new": "b"})
    assert "occurs 2 times" in overlap.output
    assert ws.read_text("notes.txt") == FILES["notes.txt"]
    empty = _call(ws, "replace_in_file", {"path": "notes.txt", "old": "", "new": "x"})
    assert empty.output == "error: old must be non-empty"
    assert _call(ws, "replace_in_file", {"path": "src", "old": "a", "new": "b"}).output.startswith(
        "error: src is a directory"
    )
    (ws.root / "bin.dat").write_bytes(b"\xff")
    binary = _call(ws, "replace_in_file", {"path": "bin.dat", "old": "a", "new": "b"})
    assert "not a UTF-8 text file" in binary.output


def test_run_tool(ws: Workspace) -> None:
    event = _call(ws, "run", {"cmd": "sed -i 's/one/ONE/' notes.txt && echo done"})
    assert event.exit_code == 0
    assert event.output == "done\n[exit code: 0]"
    assert event.fs_after["notes.txt"] != ws.baseline["notes.txt"]
    quiet = _call(ws, "run", {"cmd": "exit 4"})
    assert quiet.output == "[exit code: 4]" and quiet.exit_code == 4
    net = _call(ws, "run", {"cmd": "curl https://example.com"})
    assert net.exit_code == 6
    assert [n.via for n in net.net] == ["curl"]
    blank = _call(ws, "run", {"cmd": "  "})
    assert blank.output == "error: cmd must be a non-empty command"
    assert blank.exit_code is None


def test_run_output_is_truncated(ws: Workspace) -> None:
    event = execute_call(
        "run",
        {"cmd": "seq 1 3000"},
        ws,
        index=0,
        subtask_id="s1",
        options=SandboxOptions(output_limit=100),
    )
    assert "characters truncated" in event.output
    assert len(event.output) < 300


def test_done_and_invalid_inputs(ws: Workspace) -> None:
    done = _call(ws, "done", {"summary": "finished"})
    assert done.output.startswith("ok")
    unknown = _call(ws, "bash", {"cmd": "ls"}, index=5)
    assert unknown.tool == "invalid"
    assert unknown.index == 5
    assert unknown.args == {"requested_tool": "bash", "args": {"cmd": "ls"}}
    assert unknown.output.startswith("error: unknown tool 'bash'")
    not_object = _call(ws, "run", "ls")
    assert not_object.args == {"_raw": "'ls'"}
    assert not_object.output.startswith("error: run arguments must be a JSON object")
    missing = _call(ws, "write_file", {"path": "a.txt"})
    assert missing.output.startswith("error: write_file is missing argument(s) content")
    escape = _call(ws, "write_file", {"path": "../evil.txt", "content": "x"})
    assert escape.output.startswith("error: path '../evil.txt' escapes")
    absolute = _call(ws, "read_file", {"path": "/etc/hostname"})
    assert "absolute" in absolute.output
    assert escape.fs_after == ws.baseline


def test_unexpected_errors_become_observations(
    ws: Workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(*args: Any, **kwargs: Any) -> Any:
        raise SandboxError("sandbox exploded")

    monkeypatch.setattr(tools, "run_with", broken)
    event = _call(ws, "run", {"cmd": "ls"})
    assert event.output == "error: SandboxError: sandbox exploded"


def test_execute_action_and_invalid_event(ws: Workspace) -> None:
    action = Action(tool="write_file", args={"path": "a.txt", "content": "hi"})
    event = execute(action, ws, index=3, subtask_id="dp1")
    assert (event.index, event.subtask_id, event.tool) == (3, "dp1", "write_file")
    assert event.elapsed_s >= 0
    assert ToolEvent.from_dict(event.to_dict()).to_dict() == event.to_dict()
    bad = invalid_event(ws, index=4, subtask_id="dp1", raw="not json", reason="unparseable")
    assert bad.tool == "invalid"
    assert bad.args == {"raw": "not json"}
    assert bad.output == "error: unparseable"
    assert bad.fs_after == event.fs_after


def test_path_helper_blocks_symlink_escape(ws: Workspace, tmp_path: Path) -> None:
    target = tmp_path / "outside.txt"
    target.write_text("secret")
    (ws.root / "link.txt").symlink_to(target)
    event = _call(ws, "read_file", {"path": "link.txt"})
    assert "leads outside the workspace" in event.output
    write = _call(ws, "write_file", {"path": "link.txt", "content": "x"})
    assert write.output.startswith("error:")
    assert target.read_text() == "secret"

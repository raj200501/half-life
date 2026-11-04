# Some cases are adapted from raise-v1 tests/mutation_test.py (freshness gate),
# https://github.com/raj200501/raise-v1-public, Copyright 2026 Raj Kashikar, Apache
# License 2.0 (LICENSES/Apache-2.0.txt; see NOTICE). Modified for half-life: a live
# number is re-derived through a committed derivation instead of a document regex.
"""Tests for halflife.integrity.freshness (gate G10, re-derivation of live numbers).

A stale number is real but no longer current: the control passes and each stale case
(an artifact that lags its inputs, a document that lags its artifact) fails, and so does
anything that cannot be re-derived.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from halflife.integrity import CheckLog, GitError, IntegrityError
from halflife.integrity import claimcheck as cc
from halflife.integrity import freshness as fr
from halflife.integrity import prereg as pr
from halflife.integrity.claimcheck import Derivation, parse_manifest

_IDENTITY = {
    "GIT_AUTHOR_NAME": "test",
    "GIT_AUTHOR_EMAIL": "test@example.invalid",
    "GIT_COMMITTER_NAME": "test",
    "GIT_COMMITTER_EMAIL": "test@example.invalid",
}
SHA = "0123456789abcdef0123456789abcdef01234567"
PROVENANCE = {"command": "python -m halflife.experiment bank", "git_sha": SHA}


def run_git(root: Path, *args: str) -> str:
    proc = subprocess.run(
        [
            "git",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "init.defaultBranch=main",
            "-C",
            str(root),
            *args,
        ],
        capture_output=True,
        text=True,
        env={**os.environ, **_IDENTITY},
        check=True,
    )
    return proc.stdout.strip()


def write(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def commit_all(root: Path, message: str = "change") -> None:
    run_git(root, "add", "-A")
    run_git(root, "commit", "-q", "--allow-empty", "-m", message)


def jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(r) + "\n" for r in rows)


DERIVE_SOURCE = '''
"""Derivations for a freshness test repository (synthetic data)."""

from pathlib import Path


def summary(inputs, *, family):
    rows = inputs["artifacts/e1/sessions.jsonl"]
    mine = [r for r in rows if r["family"] == family]
    reached = sum(1 for r in mine if r["reached"])
    return {"family": family, "n": len(mine), "reached": reached, "rate": reached / len(mine)}


def broken(inputs):
    raise RuntimeError("cannot compute")


def writer(inputs):
    Path("artifacts/e1/summary.json").touch()
    return {}


def creator(inputs):
    Path("artifacts/e1/extra.json").write_text("{}")
    return {}


def network(inputs):
    import socket

    socket.create_connection(("192.0.2.1", 80), timeout=1)
    return {}


def unserializable(inputs):
    return {"x": object()}


def by_directory(inputs):
    files = inputs["artifacts/bench/"]
    return {"templates": len(files)}


NOT_CALLABLE = 3
'''

SESSIONS = [
    {"session_id": "s1", "family": "qwen", "reached": True},
    {"session_id": "s2", "family": "qwen", "reached": False},
    {"session_id": "s3", "family": "qwen", "reached": True},
    {"session_id": "s4", "family": "llama", "reached": True},
]
SUMMARY = {"family": "qwen", "n": 3, "reached": 2, "rate": 2 / 3}
README = (
    "# half-life\n\n"
    "Synthetic test data: <!-- claim:qwen.rate -->66.7%<!-- /claim --> of "
    "<!-- claim:qwen.n -->3<!-- /claim --> sessions reached a decision point; "
    "<!-- claim:sessions -->4<!-- /claim --> sessions in all.\n"
)


@pytest.fixture
def module_name() -> Iterator[str]:
    name = f"hl_derive_{uuid.uuid4().hex}"
    yield name
    sys.modules.pop(name, None)


def manifest_for(module: str, **overrides: Any) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "claims": {
            "qwen.rate": {
                "artifact": "artifacts/e1/summary.json",
                "pointer": "/rate",
                "format": ".1%",
            },
            "qwen.n": {"artifact": "artifacts/e1/summary.json", "pointer": "/n", "format": "d"},
            "sessions": {
                "artifact": "artifacts/e1/sessions.jsonl",
                "aggregate": "count",
                "format": "d",
            },
        },
        "derivations": {
            "artifacts/e1/summary.json": {
                "derive": f"{module}:summary",
                "inputs": ["artifacts/e1/sessions.jsonl"],
                "params": {"family": "qwen"},
            }
        },
        "raw": {"artifacts/e1/sessions.jsonl": "runner output, one row per session"},
    }
    manifest.update(overrides)
    return manifest


@pytest.fixture
def repo(tmp_path: Path, module_name: str) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    run_git(root, "init", "-q")
    write(
        root,
        {
            f"{module_name}.py": DERIVE_SOURCE,
            "artifacts/e1/sessions.jsonl": jsonl(SESSIONS),
            "artifacts/e1/provenance.json": json.dumps(PROVENANCE),
            "artifacts/e1/summary.json": json.dumps(SUMMARY),
            "artifacts/claims.json": json.dumps(manifest_for(module_name)),
            "README.md": README,
            "docs/REPORT.md": "# Report\n",
        },
    )
    commit_all(root, "seed")
    return root


def set_manifest(root: Path, manifest: dict[str, Any]) -> None:
    write(root, {"artifacts/claims.json": json.dumps(manifest)})


def run(root: Path) -> list[str]:
    return fr.check(root).lines


def fails(root: Path) -> list[str]:
    return [line for line in run(root) if line.startswith("FAIL")]


@pytest.fixture
def in_repo(repo: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run with the repository as the working directory (derivations use relative paths)."""
    monkeypatch.chdir(repo)
    return repo


# --------------------------------------------------------------------------- comparison


def test_difference_and_values_match() -> None:
    assert fr.values_match({"a": [1, 2.0, "x", None]}, {"a": [1, 2.0, "x", None]})
    assert fr.values_match(0.1 + 0.2, 0.3)
    assert fr.values_match(40, 40.0)
    assert fr.values_match(float("nan"), float("nan"))
    assert not fr.values_match(True, 1)
    assert not fr.values_match(1, 2)
    assert fr.difference(1.0, 1.1) == "/: 1.0 != 1.1"
    assert fr.difference({"a/b": {"c~d": 1}}, {"a/b": {"c~d": 2}}) == "/a~1b/c~0d: 1 != 2"
    assert fr.difference({"a": 1}, {"b": 1}) == "/a: present in only one of them"
    assert fr.difference([1], [1, 2]) == "/: 1 items != 2 items"
    assert fr.difference([1, 2], [1, 3]) == "/1: 2 != 3"
    assert fr.difference("a", "b") == "/: 'a' != 'b'"
    assert fr.difference([{"a": 1}], [{"a": 1}]) is None


def test_offline_blocks_the_network_and_restores_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HALFLIFE_OFFLINE", "0")
    monkeypatch.delenv("HALFLIFE_NO_MODEL_CALLS", raising=False)
    original = socket.create_connection
    with fr.offline():
        assert os.environ["HALFLIFE_OFFLINE"] == "1"
        assert os.environ["HALFLIFE_NO_MODEL_CALLS"] == "1"
        with pytest.raises(OSError, match="network access is disabled"):
            socket.create_connection(("192.0.2.1", 80))
        with pytest.raises(OSError, match="network access is disabled"):
            socket.getaddrinfo("example.org", 443)
        sock = socket.socket()
        try:
            with pytest.raises(OSError, match="network access is disabled"):
                sock.connect(("192.0.2.1", 80))
        finally:
            sock.close()
    assert socket.create_connection is original
    assert os.environ["HALFLIFE_OFFLINE"] == "0"
    assert "HALFLIFE_NO_MODEL_CALLS" not in os.environ


# --------------------------------------------------------------------------- imports and inputs


def test_import_derivation(repo: Path, module_name: str) -> None:
    function, source = fr.import_derivation(repo, f"{module_name}:summary")
    assert callable(function) and source == f"{module_name}.py"
    assert str(repo) not in sys.path


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("hl_no_such_module_xyz:f", "cannot import"),
        ("{module}:missing", "has no attribute"),
        ("{module}:NOT_CALLABLE", "is not callable"),
        ("math:sqrt", "has no Python source file"),
        ("json:dumps", "outside this repository"),
    ],
)
def test_import_derivation_errors(repo: Path, module_name: str, target: str, message: str) -> None:
    with pytest.raises(IntegrityError, match=message):
        fr.import_derivation(repo, target.format(module=module_name))


def test_import_derivation_when_root_is_already_on_the_path(
    repo: Path, module_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(repo))
    fr.import_derivation(repo, f"{module_name}:summary")
    assert str(repo) in sys.path


def test_load_input(repo: Path) -> None:
    write(repo, {"artifacts/bench/a.json": "{}", "artifacts/bench/b.jsonl": "{}\n"})
    assert fr.load_input(repo, "artifacts/e1/sessions.jsonl") == SESSIONS
    assert fr.load_input(repo, "artifacts/bench/") == {
        "artifacts/bench/a.json": {},
        "artifacts/bench/b.jsonl": [{}],
    }
    with pytest.raises(IntegrityError, match="does not exist"):
        fr.load_input(repo, "artifacts/none.json")
    with pytest.raises(IntegrityError, match=r"holds no \.json"):
        fr.load_input(repo, "artifacts/empty/")


def test_declared() -> None:
    manifest, _ = parse_manifest(
        {
            "raw": {"artifacts/e1/sessions.jsonl": "r", "artifacts/bench/": "r"},
            "derivations": {
                "artifacts/e1/s.json": {"derive": "m:f", "inputs": ["artifacts/e1/sessions.jsonl"]}
            },
        }
    )
    assert fr.declared("artifacts/e1/sessions.jsonl", manifest)
    assert fr.declared("artifacts/bench/routes/a.json", manifest)
    assert fr.declared("artifacts/e1/s.json", manifest)
    assert not fr.declared("artifacts/e1/other.json", manifest)


# --------------------------------------------------------------------------- provenance


@pytest.mark.parametrize(
    ("record", "problem"),
    [
        (PROVENANCE, None),
        ({"commands": ["a", "b"], "git_sha": "abc1234"}, None),
        ([], "not a JSON object"),
        ({"git_sha": SHA}, "names no command"),
        ({"command": " ", "git_sha": SHA}, "names no command"),
        ({"commands": [], "git_sha": SHA}, "names no command"),
        ({"command": "x"}, "no git_sha"),
        ({"command": "x", "git_sha": "XYZ"}, "no git_sha"),
    ],
)
def test_provenance_problem(record: Any, problem: str | None) -> None:
    found = fr.provenance_problem(record)
    assert found is None if problem is None else (found is not None and problem in found)


def test_find_provenance(tmp_path: Path) -> None:
    write(
        tmp_path,
        {
            "artifacts/e0/embedded.json": json.dumps({"provenance": PROVENANCE, "x": 1}),
            "artifacts/e1/provenance.json": json.dumps(PROVENANCE),
            "artifacts/e1/deep/rows.jsonl": "{}\n",
            "artifacts/e2/list.json": "[1]",
            "artifacts/bad/provenance.json": "{oops",
            "artifacts/bad/x.jsonl": "{}\n",
        },
    )
    assert fr.find_provenance(tmp_path, "artifacts/e0/embedded.json") == (
        "artifacts/e0/embedded.json",
        PROVENANCE,
    )
    assert fr.find_provenance(tmp_path, "artifacts/e1/deep/rows.jsonl") == (
        "artifacts/e1/provenance.json",
        PROVENANCE,
    )
    assert fr.find_provenance(tmp_path, "artifacts/e1/provenance.json") == (
        "artifacts/e1/provenance.json",
        PROVENANCE,
    )
    assert fr.find_provenance(tmp_path, "artifacts/e2/list.json") is None
    with pytest.raises(IntegrityError):
        fr.find_provenance(tmp_path, "artifacts/bad/x.jsonl")


def test_derivation_cycles() -> None:
    manifest, problems = parse_manifest(
        {
            "derivations": {
                "artifacts/a.json": {"derive": "m:f", "inputs": ["artifacts/b.json"]},
                "artifacts/b.json": {
                    "derive": "m:f",
                    "inputs": ["artifacts/a.json", "artifacts/r.json"],
                },
                "artifacts/c.json": {"derive": "m:f", "inputs": ["artifacts/a.json"]},
            }
        }
    )
    assert problems == []
    assert fr.derivation_cycles(manifest) == [
        ["artifacts/a.json", "artifacts/b.json", "artifacts/a.json"]
    ]


# --------------------------------------------------------------------------- derivations


def derivation(module: str, function: str = "summary", **kw: Any) -> Derivation:
    return Derivation(
        artifact=kw.pop("artifact", "artifacts/e1/summary.json"),
        derive=f"{module}:{function}",
        inputs=kw.pop("inputs", ("artifacts/e1/sessions.jsonl",)),
        params=kw.pop("params", {"family": "qwen"}),
        compare=kw.pop("compare", ("",)),
    )


def test_run_derivation_matches(in_repo: Path, module_name: str) -> None:
    log = CheckLog()
    outcome = fr.run_derivation(in_repo, derivation(module_name), log)
    assert outcome.error is None and outcome.value == SUMMARY
    assert log.problems == 0
    assert "re-derives it from 1 input(s); matches at 1 pointer(s)" in log.lines[0]


def test_run_derivation_stale_artifact(in_repo: Path, module_name: str) -> None:
    write(in_repo, {"artifacts/e1/summary.json": json.dumps({**SUMMARY, "reached": 1})})
    log = CheckLog()
    fr.run_derivation(in_repo, derivation(module_name), log)
    assert log.problems == 1 and "stale (re-derived differs at /reached: 2 != 1)" in log.lines[0]


@pytest.mark.parametrize(
    ("function", "kw", "message"),
    [
        ("broken", {"params": {}}, "raised RuntimeError: cannot compute"),
        ("network", {"params": {}}, "network access is disabled"),
        ("unserializable", {"params": {}}, "raised TypeError"),
        ("writer", {"params": {}}, "wrote or deleted artifacts/e1/summary.json"),
        ("creator", {"params": {}}, "wrote or deleted artifacts/e1/extra.json"),
        ("summary", {"inputs": ("artifacts/e1/none.jsonl",)}, "input artifacts/e1/none.jsonl"),
        ("summary", {"artifact": "artifacts/e1/absent.json"}, "banked artifact"),
        ("summary", {"compare": ("/missing",)}, "no key 'missing'"),
        ("missing", {}, "has no attribute"),
    ],
)
def test_run_derivation_failures(
    in_repo: Path, module_name: str, function: str, kw: dict[str, Any], message: str
) -> None:
    log = CheckLog()
    outcome = fr.run_derivation(in_repo, derivation(module_name, function, **kw), log)
    assert outcome.error is not None
    assert log.problems == 1 and message in log.lines[-1], log.lines


def test_run_derivation_notes_uncommitted_source(in_repo: Path, module_name: str) -> None:
    write(in_repo, {f"{module_name}.py": DERIVE_SOURCE + "\n# edited\n"})
    log = CheckLog()
    fr.run_derivation(in_repo, derivation(module_name), log)
    assert any("has uncommitted changes" in line for line in log.lines)


def test_run_derivation_over_a_directory(in_repo: Path, module_name: str) -> None:
    write(
        in_repo,
        {
            "artifacts/bench/a.json": "{}",
            "artifacts/bench/b.json": "{}",
            "artifacts/bench/count.json": json.dumps({"templates": 3}),
        },
    )
    log = CheckLog()
    d = derivation(
        module_name,
        "by_directory",
        artifact="artifacts/bench/count.json",
        inputs=("artifacts/bench/",),
        params={},
    )
    outcome = fr.run_derivation(in_repo, d, log)
    assert outcome.value == {"templates": 3} and log.problems == 0


# --------------------------------------------------------------------------- full check


def test_control_passes(in_repo: Path) -> None:
    out = run(in_repo)
    assert out[-1] == "FRESHNESS: PASS 1 derivation(s), 3 live claim(s)"
    assert (
        "ok   claim qwen.rate: 66.7% re-derived from artifacts/e1/summary.json (README.md:3)" in out
    )
    assert any(
        line.startswith("ok   claim sessions: 4 re-selected from raw artifact") for line in out
    )
    assert any(
        line.startswith("ok   raw artifacts/e1/sessions.jsonl: primary data") for line in out
    )


def test_stale_artifact_that_claimcheck_would_pass(in_repo: Path) -> None:
    rows = [*SESSIONS, {"session_id": "s5", "family": "qwen", "reached": True}]
    write(in_repo, {"artifacts/e1/sessions.jsonl": jsonl(rows)})
    problems = fails(in_repo)
    assert any("banked artifact is stale" in p for p in problems)
    assert any(
        "claim qwen.n: artifacts/e1/summary.json is stale: banked 3, re-derived 4" in p
        for p in problems
    )
    assert any(
        "README.md:3 says '4', re-selected from raw artifact value is '5'" in p for p in problems
    )


def test_stale_document_value(in_repo: Path) -> None:
    write(in_repo, {"README.md": README.replace("66.7%", "66.6%")})
    problems = fails(in_repo)
    assert problems == [
        "FAIL claim qwen.rate: README.md:3 says '66.6%', re-derived value is '66.7%'"
    ]


def test_claim_without_derivation_or_raw_declaration(in_repo: Path, module_name: str) -> None:
    manifest = manifest_for(module_name)
    manifest["claims"]["extra"] = {
        "artifact": "artifacts/e1/other.json",
        "pointer": "/x",
        "format": "d",
    }
    set_manifest(in_repo, manifest)
    write(
        in_repo,
        {
            "artifacts/e1/other.json": json.dumps({"x": 5}),
            "docs/REPORT.md": "# Report\n\nX is <!-- claim:extra -->5<!-- /claim -->.\n",
        },
    )
    assert any("has no derivation and is not declared raw" in p for p in fails(in_repo))


def test_undefined_and_unresolvable_claims(in_repo: Path, module_name: str) -> None:
    manifest = manifest_for(module_name)
    manifest["claims"]["qwen.n"]["pointer"] = "/absent"
    set_manifest(in_repo, manifest)
    write(in_repo, {"docs/REPORT.md": "# Report\n\n<!-- claim:ghost -->1<!-- /claim -->\n"})
    problems = fails(in_repo)
    assert any("claim ghost: used in docs/REPORT.md:3 but not defined" in p for p in problems)
    assert any("claim qwen.n: pointer '/absent': no key 'absent'" in p for p in problems)


def test_failed_derivation_fails_its_claims(in_repo: Path, module_name: str) -> None:
    manifest = manifest_for(module_name)
    manifest["derivations"]["artifacts/e1/summary.json"]["derive"] = f"{module_name}:broken"
    manifest["derivations"]["artifacts/e1/summary.json"]["params"] = {}
    set_manifest(in_repo, manifest)
    problems = fails(in_repo)
    assert any("claim qwen.rate: cannot be re-derived" in p for p in problems)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda m: m["raw"].update({"artifacts/e1/summary.json": "both"}),
            "is declared both raw and derived",
        ),
        (
            lambda m: m.update({"raw": {"artifacts/e1/": "everything"}}),
            "lies under raw directory artifacts/e1/",
        ),
        (
            lambda m: m["derivations"]["artifacts/e1/summary.json"]["inputs"].append(
                "artifacts/e1/summary.json"
            ),
            "lists itself as an input",
        ),
        (
            lambda m: m["derivations"]["artifacts/e1/summary.json"]["inputs"].append(
                "artifacts/e1/undeclared.json"
            ),
            "is neither derived nor declared raw",
        ),
        (lambda m: m["raw"].update({"artifacts/e1/gone.json": "missing"}), "does not exist"),
        (lambda m: m["raw"].update({"artifacts/empty/": "empty"}), "holds no .json"),
    ],
)
def test_declaration_problems(
    in_repo: Path, module_name: str, mutate: Callable[[dict[str, Any]], None], message: str
) -> None:
    manifest = manifest_for(module_name)
    mutate(manifest)
    set_manifest(in_repo, manifest)
    assert any(message in p for p in fails(in_repo)), run(in_repo)


def test_derivation_cycle_fails(in_repo: Path, module_name: str) -> None:
    manifest = manifest_for(module_name)
    manifest["derivations"]["artifacts/e1/a.json"] = {
        "derive": f"{module_name}:summary",
        "inputs": ["artifacts/e1/b.json"],
    }
    manifest["derivations"]["artifacts/e1/b.json"] = {
        "derive": f"{module_name}:summary",
        "inputs": ["artifacts/e1/a.json"],
    }
    set_manifest(in_repo, manifest)
    assert any("cycle: artifacts/e1/a.json -> artifacts/e1/b.json" in p for p in fails(in_repo))


def test_raw_artifact_needs_provenance(in_repo: Path) -> None:
    (in_repo / "artifacts" / "e1" / "provenance.json").unlink()
    assert any("has no provenance" in p for p in fails(in_repo))
    write(in_repo, {"artifacts/e1/provenance.json": json.dumps({"command": "hand-typed"})})
    assert any("no git_sha" in p for p in fails(in_repo))
    write(in_repo, {"artifacts/e1/provenance.json": "{oops"})
    assert any("not valid JSON" in p for p in fails(in_repo))


def test_hand_typed_artifact_declared_raw_without_provenance_fails(
    in_repo: Path, module_name: str
) -> None:
    manifest = manifest_for(module_name)
    manifest["raw"]["artifacts/typed/numbers.json"] = "I typed these"
    manifest["claims"]["typed"] = {
        "artifact": "artifacts/typed/numbers.json",
        "pointer": "/x",
        "format": "",
    }
    set_manifest(in_repo, manifest)
    write(
        in_repo,
        {
            "artifacts/typed/numbers.json": json.dumps({"x": 0.99}),
            "docs/REPORT.md": "# Report\n\nX <!-- claim:typed -->0.99<!-- /claim -->\n",
        },
    )
    assert any("artifacts/typed/numbers.json has no provenance" in p for p in fails(in_repo))


def test_missing_manifest_and_documents(in_repo: Path) -> None:
    (in_repo / "artifacts" / "claims.json").unlink()
    (in_repo / "docs" / "REPORT.md").unlink()
    problems = fails(in_repo)
    assert any("artifacts/claims.json is missing" in p for p in problems)
    assert "FAIL document docs/REPORT.md: missing" in problems


def test_non_utf8_document(in_repo: Path) -> None:
    (in_repo / "docs" / "REPORT.md").write_bytes(b"caf\xe9\n")
    assert any("is not UTF-8 text" in p for p in fails(in_repo))


def test_no_live_claims_is_noted(in_repo: Path) -> None:
    write(in_repo, {"README.md": "# half-life\n"})
    out = run(in_repo)
    assert "note no live claims in the results documents" in out
    assert out[-1] == "FRESHNESS: PASS 1 derivation(s), 0 live claim(s)"


def test_uncommitted_artifact_is_noted(in_repo: Path) -> None:
    write(in_repo, {"artifacts/e1/summary.json": json.dumps(SUMMARY, indent=1)})
    out = run(in_repo)
    assert out[-1].startswith("FRESHNESS: PASS")
    assert any("has uncommitted changes" in line for line in out)


# --------------------------------------------------------------------------- command line


def test_main(in_repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert fr.main(["--root", str(in_repo)]) == 0
    assert capsys.readouterr().out.splitlines()[-1].startswith("FRESHNESS: PASS")
    write(in_repo, {"README.md": README.replace(">3<", ">4<")})
    assert fr.main([]) == 1
    assert capsys.readouterr().out.splitlines()[-1].startswith("FRESHNESS: FAIL")


def test_main_outside_a_repository(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert fr.main(["--root", str(tmp_path)]) == 2
    assert capsys.readouterr().out.startswith("FRESHNESS: FAIL not inside a git work tree")


def test_main_reports_a_failed_git_query(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(_: Path) -> CheckLog:
        raise GitError("ls-files failed")

    monkeypatch.setattr(fr, "check", broken)
    assert fr.main(["--root", str(repo)]) == 1
    assert "cannot be completed" in capsys.readouterr().out


# --------------------------------------------------------------------------- all together


def test_one_repository_passes_prereg_verify_claimcheck_and_freshness(
    tmp_path: Path, module_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The E1 provenance both links the data to its preregistration and proves it is raw."""
    root = tmp_path / "repo"
    root.mkdir()
    run_git(root, "init", "-q")
    write(root, {"prereg/0001-e1.md": "# E1\n\nDate: 2026-11-04\n"})
    commit_all(root, "freeze E1")
    rows = [{**r, "prereg": "prereg/0001-e1.md", "confirmatory": True} for r in SESSIONS]
    manifest = manifest_for(module_name)
    manifest["allowlist"] = [
        {"value": "2", "context": "at least 2 families", "reason": "E1 design, section 5"}
    ]
    write(
        root,
        {
            f"{module_name}.py": DERIVE_SOURCE,
            "artifacts/e1/sessions.jsonl": jsonl(rows),
            "artifacts/e1/provenance.json": json.dumps(
                {**PROVENANCE, "prereg": "prereg/0001-e1.md", "confirmatory": True}
            ),
            "artifacts/e1/summary.json": json.dumps(SUMMARY),
            "artifacts/claims.json": json.dumps(manifest),
            "README.md": README + "\nE1 uses at least 2 families.\n",
            "docs/REPORT.md": "# Report\n\n## 1. Results\n\nSee section 5.\n",
        },
    )
    commit_all(root, "bank E1")
    monkeypatch.chdir(root)
    assert pr.verify(root).lines[-1] == "PREREG-VERIFY: PASS"
    assert cc.check(root).lines[-1] == "CLAIMCHECK: PASS 3 claim(s) in 2 document(s)"
    assert fr.check(root).lines[-1] == "FRESHNESS: PASS 1 derivation(s), 3 live claim(s)"

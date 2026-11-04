# Some cases are adapted from raise-v1 tests/mutation_test.py (prereg chain gate),
# https://github.com/raj200501/raise-v1-public, Copyright 2026 Raj Kashikar, Apache
# License 2.0 (LICENSES/Apache-2.0.txt; see NOTICE). Modified for half-life: the cases
# run against git history instead of a hash-chain file.
"""Tests for halflife.integrity (shared helpers) and halflife.integrity.prereg (G11).

Every repository here is a throwaway git repository under tmp_path. Each check is
exercised with a passing history and with histories where the property it guards is
false (an edited, reverted, deleted, or renamed preregistration; a gap in numbering; an
artifact committed with or before its preregistration).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
from pathlib import Path

import pytest

from halflife.integrity import (
    CheckLog,
    GitError,
    IntegrityError,
    blob_id,
    commits_touching,
    first_added_commits,
    git,
    head_commit,
    head_files,
    in_head,
    is_proper_ancestor,
    is_shallow,
    load_artifact,
    load_json,
    load_jsonl,
    parse_json,
    repo_root,
    safe_relpath,
    short,
    snapshot,
    worktree_files,
    worktree_matches_head,
)
from halflife.integrity import prereg as pr

_IDENTITY = {
    "GIT_AUTHOR_NAME": "test",
    "GIT_AUTHOR_EMAIL": "test@example.invalid",
    "GIT_COMMITTER_NAME": "test",
    "GIT_COMMITTER_EMAIL": "test@example.invalid",
}


def run_git(root: Path, *args: str) -> str:
    """Run git with a fixed identity and no signing; return stripped stdout."""
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


def make_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    run_git(root, "init", "-q")
    return root


def write(root: Path, files: dict[str, str | None]) -> None:
    for rel, text in files.items():
        path = root / rel
        if text is None:
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")


def commit(root: Path, files: dict[str, str | None], message: str = "change") -> str:
    """Write (or delete, for None) files, commit everything, and return the new SHA."""
    write(root, files)
    run_git(root, "add", "-A")
    run_git(root, "commit", "-q", "--allow-empty", "-m", message)
    return run_git(root, "rev-parse", "HEAD")


PREREG_TEXT = "# E1 preregistration\n\nDate: 2026-11-04\n\nHypothesis: the hazard rises.\n"


def provenance(prereg: str | None = "prereg/0001-e1.md", **extra: object) -> str:
    data: dict[str, object] = {"prereg": prereg, "confirmatory": prereg is not None}
    data.update(extra)
    return json.dumps(data)


def lines_of(root: Path) -> list[str]:
    return pr.verify(root).lines


def verdict(root: Path) -> str:
    return lines_of(root)[-1]


def fails(root: Path) -> list[str]:
    return [line for line in lines_of(root) if line.startswith("FAIL")]


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = make_repo(tmp_path / "repo")
    commit(root, {"README.md": "# demo\n", "prereg/.gitkeep": ""}, "bootstrap")
    return root


# --------------------------------------------------------------------------- helpers


def test_git_and_repo_root(repo: Path) -> None:
    sub = repo / "docs" / "deep"
    sub.mkdir(parents=True)
    assert repo_root(sub) == repo.resolve()
    assert git(repo, "rev-parse", "--is-inside-work-tree").strip() == "true"
    with pytest.raises(GitError, match="failed"):
        git(repo, "rev-parse", "--verify", "no-such-ref")


def test_repo_root_outside_a_repository(tmp_path: Path) -> None:
    outside = tmp_path / "plain"
    outside.mkdir()
    with pytest.raises(GitError):
        repo_root(outside)


def test_git_cannot_start(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def broken(*_: object, **__: object) -> object:
        raise OSError("no git binary")

    monkeypatch.setattr(subprocess, "run", broken)
    with pytest.raises(GitError, match="cannot run git"):
        git(tmp_path, "status")


def test_empty_repository_helpers(tmp_path: Path) -> None:
    root = make_repo(tmp_path / "empty")
    assert head_commit(root) is None
    assert first_added_commits(root, "prereg") == {}
    assert commits_touching(root, "x") == []
    assert head_files(root) == []
    assert short(None) == "uncommitted"
    assert short("a" * 40) == "a" * 12


def test_history_helpers(repo: Path) -> None:
    first = commit(repo, {"artifacts/a.json": "{}"}, "add a")
    second = commit(repo, {"artifacts/a.json": '{"x": 1}'}, "edit a")
    third = commit(repo, {"artifacts/a.json": None, "artifacts/b.json": "{}"}, "rename-ish")
    assert head_commit(repo) == third
    assert first_added_commits(repo, "artifacts") == {
        "artifacts/a.json": first,
        "artifacts/b.json": third,
    }
    assert commits_touching(repo, "artifacts/a.json") == [third, second, first]
    assert blob_id(repo, first, "artifacts/a.json") is not None
    assert blob_id(repo, third, "artifacts/a.json") is None
    assert in_head(repo, "artifacts/b.json") and not in_head(repo, "artifacts/a.json")
    assert is_proper_ancestor(repo, first, third)
    assert not is_proper_ancestor(repo, third, first)
    assert not is_proper_ancestor(repo, first, first)
    assert head_files(repo, "artifacts") == ["artifacts/b.json"]
    assert not is_shallow(repo)


def test_first_added_commits_sees_renames_and_merges(repo: Path) -> None:
    added = commit(repo, {"artifacts/old.json": "{}"}, "add old")
    run_git(repo, "mv", "artifacts/old.json", "artifacts/new.json")
    renamed = commit(repo, {}, "rename")
    run_git(repo, "checkout", "-q", "-b", "side")
    side = commit(repo, {"artifacts/side.json": "{}"}, "side add")
    run_git(repo, "checkout", "-q", "main")
    commit(repo, {"other.txt": "x"}, "main moves on")
    run_git(repo, "merge", "-q", "--no-edit", "side")
    found = first_added_commits(repo, "artifacts")
    assert found["artifacts/old.json"] == added
    assert found["artifacts/new.json"] == renamed
    assert found["artifacts/side.json"] == side


def test_worktree_helpers(repo: Path) -> None:
    commit(repo, {"artifacts/a.json": "{}", ".gitignore": "ignored/\n"}, "add")
    write(repo, {"artifacts/untracked.json": "{}", "ignored/x.json": "{}"})
    assert worktree_files(repo, "artifacts") == ["artifacts/a.json", "artifacts/untracked.json"]
    assert "ignored/x.json" not in worktree_files(repo)
    assert worktree_matches_head(repo, "artifacts/a.json")
    write(repo, {"artifacts/a.json": '{"changed": true}'})
    assert not worktree_matches_head(repo, "artifacts/a.json")
    assert not worktree_matches_head(repo, "artifacts/untracked.json")
    write(repo, {"artifacts/a.json": None})
    assert not worktree_matches_head(repo, "artifacts/a.json")


def test_snapshot(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("abc", encoding="utf-8")
    state = snapshot(tmp_path, ["a.txt", "missing.txt"])
    assert state["missing.txt"] is None
    first = state["a.txt"]
    assert first is not None and first[0] == 3


def test_shallow_clone_is_detected(repo: Path, tmp_path: Path) -> None:
    commit(repo, {"x.txt": "1"}, "one")
    commit(repo, {"x.txt": "2"}, "two")
    clone = tmp_path / "shallow"
    subprocess.run(
        ["git", "clone", "-q", "--depth", "1", f"file://{repo}", str(clone)],
        check=True,
        capture_output=True,
    )
    assert is_shallow(clone)


@pytest.mark.parametrize(
    ("value", "prefix"),
    [
        ("artifacts/e1/x.json", "artifacts"),
        ("artifacts", "artifacts"),
        ("README.md", None),
    ],
)
def test_safe_relpath_accepts(value: str, prefix: str | None) -> None:
    assert safe_relpath(value, prefix) == value


@pytest.mark.parametrize(
    ("value", "prefix", "message"),
    [
        ("", None, "non-empty"),
        (3, None, "non-empty"),
        ("/etc/passwd", None, "relative POSIX"),
        ("a\\b", None, "relative POSIX"),
        ("a//b", None, "relative POSIX"),
        ("a/../b", None, "normal form"),
        ("./a", None, "normal form"),
        ("a/", None, "normal form"),
        ("docs/x.json", "artifacts", "not under"),
        ("artifactsx/y.json", "artifacts", "not under"),
    ],
)
def test_safe_relpath_rejects(value: object, prefix: str | None, message: str) -> None:
    with pytest.raises(IntegrityError, match=message):
        safe_relpath(value, prefix)


def test_parse_json_is_strict() -> None:
    assert parse_json('{"a": [1, {"b": 2}]}') == {"a": [1, {"b": 2}]}
    with pytest.raises(IntegrityError, match="duplicate key"):
        parse_json('{"confirmatory": true, "confirmatory": false}', "x.json")
    with pytest.raises(IntegrityError, match="not valid JSON"):
        parse_json("{nope", "x.json")


def test_loaders(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text('{"a": 1}', encoding="utf-8")
    (tmp_path / "b.jsonl").write_text('{"r": 1}\n\n{"r": 2}\n', encoding="utf-8")
    (tmp_path / "bad.jsonl").write_text('{"r": 1}\n{oops\n', encoding="utf-8")
    (tmp_path / "c.csv").write_text("a,b\n", encoding="utf-8")
    (tmp_path / "latin.json").write_bytes(b'{"a": "\xff"}')
    assert load_json(tmp_path / "a.json") == {"a": 1}
    assert load_jsonl(tmp_path / "b.jsonl") == [{"r": 1}, {"r": 2}]
    assert load_artifact(tmp_path / "a.json") == {"a": 1}
    assert load_artifact(tmp_path / "b.jsonl") == [{"r": 1}, {"r": 2}]
    with pytest.raises(IntegrityError, match="line 2"):
        load_jsonl(tmp_path / "bad.jsonl")
    with pytest.raises(IntegrityError, match=r"only \.json and \.jsonl"):
        load_artifact(tmp_path / "c.csv")
    with pytest.raises(IntegrityError, match="UTF-8"):
        load_json(tmp_path / "latin.json")
    with pytest.raises(IntegrityError, match="cannot be read"):
        load_json(tmp_path / "absent.json")


def test_check_log() -> None:
    log = CheckLog()
    log.ok("alpha", "fine")
    log.note("just so you know")
    assert log.verdict("TOOL") == "TOOL: PASS"
    assert log.verdict("TOOL", "3 verified") == "TOOL: PASS 3 verified"
    log.fail("beta", "broken")
    assert log.problems == 1
    assert log.verdict("TOOL", "3 verified") == "TOOL: FAIL 1 problem(s)"
    stream = io.StringIO()
    log.emit(stream)
    assert stream.getvalue().splitlines() == [
        "ok   alpha: fine",
        "note just so you know",
        "FAIL beta: broken",
    ]


def test_check_log_emits_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    log = CheckLog()
    log.ok("x", "y")
    log.emit()
    assert capsys.readouterr().out == "ok   x: y\n"


# --------------------------------------------------------------------------- parsing helpers


def test_classify_prereg_dir() -> None:
    preregs, unknown = pr.classify_prereg_dir(
        [
            "prereg/.gitkeep",
            "prereg/README.md",
            "prereg/readme",
            "prereg/0001-e1.md",
            "prereg/0002-e1-amend.md",
            "prereg/1-e1.md",
            "prereg/0003_e1.md",
            "prereg/0004-e1.json",
            "prereg/old/0005-e1.md",
            "prereg/0001-e1.md",
        ]
    )
    assert preregs == ["prereg/0001-e1.md", "prereg/0002-e1-amend.md"]
    assert unknown == [
        "prereg/0003_e1.md",
        "prereg/0004-e1.json",
        "prereg/1-e1.md",
        "prereg/old/0005-e1.md",
    ]


def test_amended_paths_and_is_amendment() -> None:
    text = (
        "# Amendment\n\n**Amends:** prereg/0001-e1.md and prereg/0002-e3.md\n"
        "> amends: prereg/0001-e1.md\nNot an amends line: prereg/0009-x.md\n"
    )
    assert pr.amended_paths(text) == ["prereg/0001-e1.md", "prereg/0002-e3.md"]
    assert pr.amended_paths("no citation here") == []
    plain = pr.Prereg("prereg/0001-e1.md", 1, "e1", None, "text")
    by_slug = pr.Prereg("prereg/0002-e1-amendment.md", 2, "e1-amendment", None, "text")
    by_line = pr.Prereg("prereg/0002-fix.md", 2, "fix", "abc", "Amends: prereg/0001-e1.md")
    assert not pr.is_amendment(plain) and not plain.committed
    assert pr.is_amendment(by_slug)
    assert pr.is_amendment(by_line) and by_line.committed


def test_has_iso_date() -> None:
    assert pr.has_iso_date("Dated 2026-11-04.")
    assert not pr.has_iso_date("Dated 2026-13-40 and 2026-02-30.")
    assert not pr.has_iso_date("no date")


def test_blob_sha256(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sha = commit(repo, {"prereg/0001-e1.md": PREREG_TEXT}, "prereg")
    expected = hashlib.sha256(PREREG_TEXT.encode("utf-8")).hexdigest()
    assert pr.blob_sha256(repo, sha, "prereg/0001-e1.md") == expected
    with pytest.raises(GitError):
        pr.blob_sha256(repo, sha, "prereg/0009-none.md")

    def broken(*_: object, **__: object) -> object:
        raise OSError("gone")

    monkeypatch.setattr(subprocess, "run", broken)
    with pytest.raises(GitError, match="cannot run git"):
        pr.blob_sha256(repo, sha, "prereg/0001-e1.md")


# --------------------------------------------------------------------------- empty states


def test_empty_repository_passes_and_says_so(tmp_path: Path) -> None:
    root = make_repo(tmp_path / "empty")
    out = lines_of(root)
    assert out[-1] == "PREREG-VERIFY: PASS"
    assert any("nothing to verify" in line for line in out)


def test_no_preregistrations_and_no_confirmatory_artifacts(repo: Path) -> None:
    commit(repo, {"artifacts/e0/x.json": '{"precision": 0.97}'}, "exploratory artifact")
    out = lines_of(repo)
    assert out[-1] == "PREREG-VERIFY: PASS"
    assert "note nothing to verify: no preregistrations and no confirmatory artifacts" in out
    assert "ok   confirmatory: no confirmatory artifacts under artifacts/" in out


# --------------------------------------------------------------------------- (a) frozen


def test_frozen_prereg_passes_and_prints_sha256(repo: Path) -> None:
    sha = commit(repo, {"prereg/0001-e1.md": PREREG_TEXT}, "freeze")
    out = lines_of(repo)
    digest = hashlib.sha256(PREREG_TEXT.encode("utf-8")).hexdigest()
    assert (
        f"ok   frozen prereg/0001-e1.md: unchanged since its first commit {sha[:12]}, "
        f"sha256 {digest}" in out
    )
    assert out[-1] == "PREREG-VERIFY: PASS"


def test_edited_after_freeze_fails(repo: Path) -> None:
    commit(repo, {"prereg/0001-e1.md": PREREG_TEXT}, "freeze")
    commit(repo, {"prereg/0001-e1.md": PREREG_TEXT + "Threshold lowered to 0.05.\n"}, "edit")
    assert any("content at HEAD differs" in line for line in fails(repo))
    assert verdict(repo) == "PREREG-VERIFY: FAIL 1 problem(s)"


def test_modify_and_revert_is_caught(repo: Path) -> None:
    commit(repo, {"prereg/0001-e1.md": PREREG_TEXT}, "freeze")
    commit(repo, {"prereg/0001-e1.md": PREREG_TEXT + "temporary\n"}, "edit")
    commit(repo, {"prereg/0001-e1.md": PREREG_TEXT}, "revert")
    assert any("changed after it was frozen" in line for line in fails(repo))


def test_working_tree_edit_and_delete_fail(repo: Path) -> None:
    commit(repo, {"prereg/0001-e1.md": PREREG_TEXT}, "freeze")
    write(repo, {"prereg/0001-e1.md": PREREG_TEXT + "edited\n"})
    assert any("edited in the working tree" in line for line in fails(repo))
    write(repo, {"prereg/0001-e1.md": None})
    assert any("deleted in the working tree" in line for line in fails(repo))


def test_deleted_and_renamed_preregistrations_fail(repo: Path) -> None:
    commit(repo, {"prereg/0001-e1.md": PREREG_TEXT}, "freeze")
    commit(repo, {"prereg/0001-e1.md": None}, "delete")
    assert any("deleted or renamed after it was frozen" in line for line in fails(repo))
    commit(repo, {"prereg/0001-e1-renamed.md": PREREG_TEXT}, "re-add under a new name")
    problems = fails(repo)
    assert any("prereg/0001-e1.md: deleted or renamed" in line for line in problems)


def test_committed_prereg_without_a_first_commit_fails(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit(repo, {"prereg/0001-e1.md": PREREG_TEXT}, "freeze")
    monkeypatch.setattr(pr, "first_added_commits", lambda *_: {})
    assert any("first commit cannot be found" in line for line in fails(repo))


def test_unreadable_draft_fails(repo: Path) -> None:
    (repo / "prereg" / "0001-e1.md").write_bytes(b"\xff\xfe not utf-8")
    assert any("cannot be read" in line for line in fails(repo))


def test_drafts_are_noted_not_frozen(repo: Path) -> None:
    commit(repo, {"prereg/0001-e1.md": PREREG_TEXT}, "freeze")
    write(repo, {"prereg/0002-e3.md": "# E3 draft\n"})
    out = lines_of(repo)
    assert "note draft prereg/0002-e3.md: not committed, so not frozen yet" in out
    assert out[-1] == "PREREG-VERIFY: PASS"


# --------------------------------------------------------------------------- layout


def test_layout_rejects_misnamed_files(repo: Path) -> None:
    commit(
        repo,
        {
            "prereg/README.md": "How preregistrations work.\n",
            "prereg/1-e1.md": "misnamed\n",
            "prereg/archive/0001-old.md": "nested\n",
        },
        "misnamed",
    )
    problems = fails(repo)
    assert any("prereg/1-e1.md is not a preregistration name" in p for p in problems)
    assert any("prereg/archive/0001-old.md" in p for p in problems)
    assert not any("README" in p for p in problems)


# --------------------------------------------------------------------------- (d) numbering


def test_numbering_gap_fails(repo: Path) -> None:
    commit(repo, {"prereg/0001-e1.md": PREREG_TEXT, "prereg/0003-e3.md": "# E3\n"}, "two")
    assert any("missing 0002" in line for line in fails(repo))


def test_numbering_must_start_at_0001(repo: Path) -> None:
    commit(repo, {"prereg/0002-e1.md": PREREG_TEXT}, "starts at two")
    assert any("missing 0001" in line for line in fails(repo))


def test_duplicate_numbers_fail(repo: Path) -> None:
    commit(repo, {"prereg/0001-e1.md": PREREG_TEXT, "prereg/0001-e3.md": "# E3\n"}, "dupes")
    assert any("number 0001 is used by" in line for line in fails(repo))


def test_contiguous_numbering_and_order_pass(repo: Path) -> None:
    commit(repo, {"prereg/0001-e1.md": PREREG_TEXT}, "one")
    commit(repo, {"prereg/0002-e3.md": "# E3\n"}, "two")
    out = lines_of(repo)
    assert "ok   numbering: 2 preregistration(s), contiguous 0001 to 0002" in out
    assert "ok   order: first commits follow the numbering" in out
    assert out[-1] == "PREREG-VERIFY: PASS"


def test_preregistration_inserted_out_of_order_fails(repo: Path) -> None:
    commit(repo, {"prereg/0002-e3.md": "# E3\n"}, "two first")
    commit(repo, {"prereg/0001-e1.md": PREREG_TEXT}, "one inserted later")
    assert any(line.startswith("FAIL order:") for line in fails(repo))


# --------------------------------------------------------------------------- (b) amendments

AMENDMENT = "# Amendment\n\nDate: 2026-11-05\n\nAmends: prereg/0001-e1.md\n\nRaise n to 60.\n"


def test_valid_amendment_passes(repo: Path) -> None:
    commit(repo, {"prereg/0001-e1.md": PREREG_TEXT}, "one")
    commit(repo, {"prereg/0002-e1-amendment.md": AMENDMENT}, "amend")
    out = lines_of(repo)
    assert any(line.startswith("ok   amendment prereg/0002-e1-amendment.md") for line in out)
    assert out[-1] == "PREREG-VERIFY: PASS"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("# Amendment\n\nDate: 2026-11-05\n", "does not cite the file it amends"),
        (AMENDMENT.replace("prereg/0001-e1.md", "prereg/0009-none.md"), "which does not exist"),
        (AMENDMENT.replace("prereg/0001-e1.md", "prereg/0002-e1-amendment.md"), "cites itself"),
        (AMENDMENT.replace("Date: 2026-11-05", "Date: soon"), "is not dated"),
    ],
)
def test_bad_amendments_fail(repo: Path, text: str, message: str) -> None:
    commit(repo, {"prereg/0001-e1.md": PREREG_TEXT}, "one")
    commit(repo, {"prereg/0002-e1-amendment.md": text}, "amend")
    assert any(message in line for line in fails(repo))


def test_amendment_must_be_numbered_after_the_original(repo: Path) -> None:
    commit(repo, {"prereg/0001-e1-amend.md": AMENDMENT.replace("0001-e1", "0002-e1")}, "a")
    commit(repo, {"prereg/0002-e1.md": PREREG_TEXT}, "b")
    assert any("is not numbered after prereg/0002-e1.md" in line for line in fails(repo))


def test_amendment_of_an_uncommitted_draft_fails(repo: Path) -> None:
    write(repo, {"prereg/0001-e1.md": PREREG_TEXT, "prereg/0002-e1-amend.md": AMENDMENT})
    assert any("which is not committed" in line for line in fails(repo))


def test_amendment_committed_before_its_original_fails(repo: Path) -> None:
    commit(repo, {"prereg/0002-e1-amend.md": AMENDMENT}, "amendment first")
    commit(repo, {"prereg/0001-e1.md": PREREG_TEXT}, "original later")
    problems = fails(repo)
    assert any("was first committed before prereg/0001-e1.md" in line for line in problems)


# --------------------------------------------------------------------------- (c) confirmatory


def test_confirmatory_artifacts_after_prereg_pass(repo: Path) -> None:
    commit(repo, {"prereg/0001-e1.md": PREREG_TEXT}, "freeze")
    commit(
        repo,
        {
            "artifacts/e1/provenance.json": provenance(),
            "artifacts/e1/sessions.jsonl": json.dumps(
                {"session_id": "s1", "prereg": "prereg/0001-e1.md", "confirmatory": True}
            )
            + "\n",
            "artifacts/e1/compactions/r1/c1.json": "{}",
        },
        "bank E1",
    )
    out = lines_of(repo)
    assert out[-1] == "PREREG-VERIFY: PASS"
    assert any(
        line.startswith("ok   confirmatory artifacts/e1/provenance.json: cites prereg/0001-e1.md")
        and "3 file(s) committed after it" in line
        for line in out
    )
    assert any(line.startswith("ok   confirmatory artifacts/e1/sessions.jsonl") for line in out)


def test_artifact_in_the_same_commit_as_its_prereg_fails(repo: Path) -> None:
    commit(
        repo,
        {"prereg/0001-e1.md": PREREG_TEXT, "artifacts/e1/provenance.json": provenance()},
        "prereg and data together",
    )
    assert any("does not come after prereg/0001-e1.md" in line for line in fails(repo))


def test_artifact_committed_before_its_prereg_fails(repo: Path) -> None:
    commit(repo, {"artifacts/e1/sessions.jsonl": '{"session_id": "s0"}\n'}, "data first")
    commit(repo, {"prereg/0001-e1.md": PREREG_TEXT}, "prereg after the data")
    commit(repo, {"artifacts/e1/provenance.json": provenance()}, "provenance")
    problems = fails(repo)
    assert any("artifacts/e1/sessions.jsonl was first committed" in line for line in problems)


def test_pilot_data_deleted_and_re_added_keeps_its_first_commit(repo: Path) -> None:
    commit(repo, {"artifacts/e1/sessions.jsonl": '{"session_id": "pilot"}\n'}, "pilot")
    commit(repo, {"artifacts/e1/sessions.jsonl": None}, "drop pilot")
    commit(repo, {"prereg/0001-e1.md": PREREG_TEXT}, "freeze")
    commit(
        repo,
        {
            "artifacts/e1/sessions.jsonl": '{"session_id": "s1"}\n',
            "artifacts/e1/provenance.json": provenance(),
        },
        "real data",
    )
    assert any("artifacts/e1/sessions.jsonl was first committed" in line for line in fails(repo))


def test_artifact_on_a_branch_without_the_prereg_fails(repo: Path) -> None:
    run_git(repo, "checkout", "-q", "-b", "side")
    commit(repo, {"artifacts/e1/sessions.jsonl": '{"session_id": "s1"}\n'}, "side data")
    run_git(repo, "checkout", "-q", "main")
    commit(repo, {"prereg/0001-e1.md": PREREG_TEXT}, "freeze on main")
    run_git(repo, "merge", "-q", "--no-edit", "side")
    commit(repo, {"artifacts/e1/provenance.json": provenance()}, "provenance")
    assert any("artifacts/e1/sessions.jsonl was first committed" in line for line in fails(repo))


@pytest.mark.parametrize(
    ("cited", "message"),
    [
        ("prereg/0009-none.md", "which does not exist"),
        ("prereg/../README.md", "is not a prereg/NNNN-slug.md path"),
        ("prereg/.gitkeep", "is not a prereg/NNNN-slug.md path"),
        ("README.md", "is not a prereg/NNNN-slug.md path"),
    ],
)
def test_confirmatory_artifact_must_cite_a_real_preregistration(
    repo: Path, cited: str, message: str
) -> None:
    commit(repo, {"prereg/0001-e1.md": PREREG_TEXT}, "freeze")
    commit(repo, {"artifacts/e1/provenance.json": provenance(cited)}, "bank")
    assert any(message in line for line in fails(repo))


def test_confirmatory_artifact_citing_a_draft_fails(repo: Path) -> None:
    commit(repo, {"artifacts/e1/provenance.json": provenance()}, "bank")
    write(repo, {"prereg/0001-e1.md": PREREG_TEXT})
    assert any("which is not committed" in line for line in fails(repo))


def test_provenance_flag_rules(repo: Path) -> None:
    commit(repo, {"prereg/0001-e1.md": PREREG_TEXT}, "freeze")
    commit(
        repo,
        {
            "artifacts/pilot/provenance.json": json.dumps({"prereg": None, "confirmatory": False}),
            "artifacts/legacy/provenance.json": json.dumps({"prereg": None}),
            "artifacts/explore/provenance.json": json.dumps(
                {"prereg": "prereg/0001-e1.md", "confirmatory": False}
            ),
            "artifacts/e1/provenance.json": json.dumps({"prereg": "prereg/0001-e1.md"}),
        },
        "bank",
    )
    out = lines_of(repo)
    assert out[-1] == "PREREG-VERIFY: PASS"
    checked = [line for line in out if line.startswith("ok   confirmatory artifacts/")]
    assert len(checked) == 1 and "artifacts/e1/provenance.json" in checked[0]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (json.dumps({"prereg": "prereg/0001-e1.md", "confirmatory": "yes"}), "true or false"),
        (json.dumps({"prereg": None, "confirmatory": True}), "does not name a preregistration"),
        (json.dumps({"prereg": ["prereg/0001-e1.md", 3], "confirmatory": True}), "does not name"),
        (json.dumps(["not", "an", "object"]), "provenance is not a JSON object"),
        ('{"confirmatory": true, "confirmatory": false}', "cannot be parsed"),
        ("{broken", "cannot be parsed"),
    ],
)
def test_malformed_confirmatory_declarations_fail(repo: Path, content: str, message: str) -> None:
    commit(repo, {"prereg/0001-e1.md": PREREG_TEXT}, "freeze")
    commit(repo, {"artifacts/e1/provenance.json": content}, "bank")
    assert any(message in line for line in fails(repo))


def test_list_of_preregs_and_other_confirmatory_json(repo: Path) -> None:
    commit(repo, {"prereg/0001-e1.md": PREREG_TEXT}, "one")
    commit(repo, {"prereg/0002-e1-amend.md": AMENDMENT}, "two")
    commit(
        repo,
        {
            "artifacts/e1/hazard.json": json.dumps(
                {"confirmatory": True, "prereg": ["prereg/0001-e1.md", "prereg/0002-e1-amend.md"]}
            ),
            "artifacts/e1/notes.json": json.dumps(["a list is not confirmatory"]),
            "artifacts/e1/plot.svg": "<svg/>",
        },
        "bank",
    )
    out = lines_of(repo)
    assert out[-1] == "PREREG-VERIFY: PASS"
    assert any("cites prereg/0001-e1.md, prereg/0002-e1-amend.md" in line for line in out)


def test_nested_provenance_governs_its_own_subtree(repo: Path) -> None:
    commit(repo, {"artifacts/e1/pilot/data.json": "{}"}, "pilot before the prereg")
    commit(repo, {"prereg/0001-e1.md": PREREG_TEXT}, "freeze")
    commit(
        repo,
        {
            "artifacts/e1/provenance.json": provenance(),
            "artifacts/e1/pilot/provenance.json": json.dumps({"confirmatory": False}),
        },
        "bank",
    )
    assert verdict(repo) == "PREREG-VERIFY: PASS"


def test_uncommitted_confirmatory_files_are_noted(repo: Path) -> None:
    commit(repo, {"prereg/0001-e1.md": PREREG_TEXT}, "freeze")
    write(repo, {"artifacts/e1/provenance.json": provenance()})
    out = lines_of(repo)
    assert out[-1] == "PREREG-VERIFY: PASS"
    assert any("1 file(s) not committed yet" in line for line in out)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([{"confirmatory": "true", "prereg": "prereg/0001-e1.md"}], "not a boolean"),
        ([{"confirmatory": True}], "do not name a preregistration"),
    ],
)
def test_jsonl_row_problems(repo: Path, rows: list[dict[str, object]], message: str) -> None:
    commit(repo, {"prereg/0001-e1.md": PREREG_TEXT}, "freeze")
    text = "".join(json.dumps(r) + "\n" for r in rows)
    commit(repo, {"artifacts/e1/sessions.jsonl": text}, "bank")
    assert any(message in line for line in fails(repo))


def test_jsonl_rows_without_flags_are_not_confirmatory(repo: Path) -> None:
    commit(repo, {"artifacts/usage/ledger.jsonl": '{"kind": "local"}\n[1, 2]\n'}, "ledger")
    out = lines_of(repo)
    assert out[-1] == "PREREG-VERIFY: PASS"


def test_check_confirmatory_needs_first_commits_for_committed_files(repo: Path) -> None:
    log = CheckLog()
    known = {"prereg/0001-e1.md": pr.Prereg("prereg/0001-e1.md", 1, "e1", "a" * 40, "")}
    artifact = pr.Confirmatory(
        "artifacts/e1/x.json", ("prereg/0001-e1.md",), ("artifacts/e1/x.json",)
    )
    pr.check_confirmatory(
        [artifact], known, {}, pr.Ancestry(repo), log, frozenset({"artifacts/e1/x.json"})
    )
    assert log.problems == 1
    assert "first commit cannot be found" in log.lines[0]


def test_ancestry_memoizes(repo: Path) -> None:
    first = commit(repo, {"a.txt": "1"}, "one")
    second = commit(repo, {"a.txt": "2"}, "two")
    ancestry = pr.Ancestry(repo)
    assert ancestry.proper(first, second) and ancestry.proper(first, second)
    assert ancestry.same_or_before(second, second)
    assert not ancestry.same_or_before(second, first)


# --------------------------------------------------------------------------- shallow clones


def test_shallow_clone_fails_when_there_is_history_to_check(repo: Path, tmp_path: Path) -> None:
    commit(repo, {"prereg/0001-e1.md": PREREG_TEXT}, "freeze")
    commit(repo, {"artifacts/e1/provenance.json": provenance()}, "bank")
    clone = tmp_path / "shallow"
    subprocess.run(
        ["git", "clone", "-q", "--depth", "1", f"file://{repo}", str(clone)],
        check=True,
        capture_output=True,
    )
    out = lines_of(clone)
    assert any("shallow clone" in line for line in out)
    assert out[-1] == "PREREG-VERIFY: FAIL 1 problem(s)"


# --------------------------------------------------------------------------- command line


def test_main_exit_codes(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    commit(repo, {"prereg/0001-e1.md": PREREG_TEXT}, "freeze")
    sub = repo / "docs"
    sub.mkdir()
    assert pr.main(["--root", str(sub)]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "PREREG-VERIFY: PASS"
    commit(repo, {"prereg/0001-e1.md": PREREG_TEXT + "edit\n"}, "edit")
    assert pr.main(["--root", str(repo)]) == 1
    assert capsys.readouterr().out.splitlines()[-1] == "PREREG-VERIFY: FAIL 1 problem(s)"


def test_main_outside_a_repository(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    outside = tmp_path / "plain"
    outside.mkdir()
    assert pr.main(["--root", str(outside)]) == 2
    assert capsys.readouterr().out.startswith("PREREG-VERIFY: FAIL not inside a git work tree")


def test_main_reports_a_failed_git_query(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(_: Path) -> CheckLog:
        raise GitError("git log failed")

    monkeypatch.setattr(pr, "verify", broken)
    assert pr.main(["--root", str(repo)]) == 1
    assert "cannot be completed" in capsys.readouterr().out


def test_main_runs_from_the_working_directory(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(repo)
    assert pr.main([]) == 0
    assert "PREREG-VERIFY: PASS" in capsys.readouterr().out

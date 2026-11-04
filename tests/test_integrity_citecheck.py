"""Tests for halflife.integrity.citecheck (gate G13, offline citation check).

The check validates artifacts/citecheck.json against the schema in
.claude/agents/citation-checker.md and requires every URL cited in README.md and
docs/REPORT.md to be banked as verified. Fixture references are synthetic.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from halflife.integrity import CheckLog
from halflife.integrity import citecheck as ct

TODAY = dt.date(2026, 11, 4)
SHA = "a" * 64


def reference(key: str, url: str, **kw: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "key": key,
        "cited_url": url,
        "final_url": url,
        "retrieved": "2026-11-04",
        "page_title": f"Title of {key}",
        "authors": ["A. Author", "B. Author"],
        "year": 2023,
        "content_sha256": SHA,
        "hash_source": "curl body",
        "status": "verified",
        "notes": "",
    }
    entry.update(kw)
    return entry


def bank(root: Path, references: list[Any], **top: Any) -> None:
    data = {"checked_by": "citation-checker", "references": references, **top}
    path = root / "artifacts" / "citecheck.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


README = (
    "# half-life\n\n"
    "Long contexts lose the middle: Liu et al. (2023), https://arxiv.org/abs/2307.03172.\n\n"
    "```bash\ngit clone https://github.com/someone/half-life\n```\n\n"
    "Local dev server at http://localhost:8000/ is not a reference.\n"
)
REPORT = (
    "# Report\n\n## Related work\n\n"
    "- Hooks are documented at [hooks](https://code.claude.com/docs/en/hooks#events).\n"
    "- Liu et al. [lost][lost-ref] study position effects.\n\n"
    "[lost-ref]: https://arxiv.org/abs/2307.03172/\n"
)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    subprocess.run(
        ["git", "-c", "init.defaultBranch=main", "init", "-q", str(root)],
        check=True,
        capture_output=True,
    )
    (root / "README.md").write_text(README, encoding="utf-8")
    (root / "docs" / "REPORT.md").write_text(REPORT, encoding="utf-8")
    bank(
        root,
        [
            reference("liu2023", "https://arxiv.org/abs/2307.03172"),
            reference(
                "cc-hooks",
                "https://docs.anthropic.com/en/docs/claude-code/hooks",
                final_url="https://code.claude.com/docs/en/hooks",
                authors=["Anthropic"],
                year=2026,
                hash_source="webfetch text",
            ),
            {
                "key": "dropped",
                "cited_url": "https://blocked.example-site.org/paper",
                "status": "unverified",
                "notes": "403 from the site and no official copy",
            },
        ],
    )
    return root


def lines(root: Path) -> list[str]:
    return ct.check(root, TODAY).lines


def fails(root: Path) -> list[str]:
    return [line for line in lines(root) if line.startswith("FAIL")]


# --------------------------------------------------------------------------- URLs


@pytest.mark.parametrize(
    ("url", "normal"),
    [
        ("HTTPS://ArXiv.org/abs/2307.03172/", "https://arxiv.org/abs/2307.03172"),
        ("https://arxiv.org:443/abs/1#section", "https://arxiv.org/abs/1"),
        ("http://example.org:80", "http://example.org/"),
        ("http://example.org:8080/a?b=1", "http://example.org:8080/a?b=1"),
        ("https://example.org/", "https://example.org/"),
        ("  https://example.org//  ", "https://example.org/"),
        ("http://[::1", "http://[::1"),
    ],
)
def test_normalize_url(url: str, normal: str) -> None:
    assert ct.normalize_url(url) == normal


@pytest.mark.parametrize(
    ("url", "reserved"),
    [
        ("http://localhost:8000/", True),
        ("http://127.0.0.2/x", True),
        ("http://[::1]/", True),
        ("https://example.com/a", True),
        ("https://docs.example.org/a", True),
        ("https://host.test/a", True),
        ("https://host.invalid", True),
        ("https://app.localhost", True),
        ("https://arxiv.org/abs/1", False),
        ("https://notexample.com/", False),
        ("http://[::1", False),
    ],
)
def test_is_reserved(url: str, reserved: bool) -> None:
    assert ct.is_reserved(url) is reserved


def test_cited_urls() -> None:
    text = (
        "See https://arxiv.org/abs/1. and (https://b.org/x), <https://c.org/y>, "
        "[d](https://d.org/z?q=1#f)\n"
        "`https://code.org/x` <!-- https://hidden.org --> http://localhost/x\n"
        "[ref]: https://e.org/ref\n"
        "Also www.f.org/page, and www.example.com/x.\n"
    )
    found = [(c.line, c.url) for c in ct.cited_urls("README.md", text)]
    assert found == [
        (1, "https://arxiv.org/abs/1"),
        (1, "https://b.org/x"),
        (1, "https://c.org/y"),
        (1, "https://d.org/z?q=1#f"),
        (3, "https://e.org/ref"),
        (4, "www.f.org/page"),
    ]


def test_uncited_citations() -> None:
    text = (
        "Liu et al. found an effect.\n\n"
        "Liu et al. (2023) https://arxiv.org/abs/2307.03172 found it too.\n\n"
        "Smith (2024) argues the opposite.\n\n"
        "As shown before (Jones, 2022), the effect holds.\n\n"
        "The brief (2026) is internal.\n\n"
        "- Lee et al. [paper][p] list item.\n"
        "- Kim et al. [missing][m] list item.\n"
        "| Table row | Park et al. |\n\n"
        "[p]: https://p.org/paper\n"
        "`Code et al. (2020)` is code.\n"
    )
    flagged = ct.uncited_citations("docs/REPORT.md", text)
    assert [line for line, _ in flagged] == [1, 5, 7, 12, 13]


# --------------------------------------------------------------------------- schema


def test_reference_problems_accepts_complete_entries() -> None:
    assert ct.reference_problems(reference("a", "https://a.org"), TODAY) == []
    minimal = {"key": "b", "cited_url": "https://b.org", "status": "unverified", "notes": "404"}
    assert ct.reference_problems(minimal, TODAY) == []


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"status": "ok"}, "status 'ok' is not one of"),
        ({"key": " "}, "key is missing or empty"),
        ({"cited_url": "ftp://a.org"}, "cited_url is not an http(s) URL"),
        ({"cited_url": 7}, "cited_url is not an http(s) URL"),
        ({"cited_url": "http://[::1"}, "cited_url is not an http(s) URL"),
        ({"notes": None}, "notes must be a string"),
        ({"final_url": "a.org"}, "final_url is not an http(s) URL"),
        ({"retrieved": "Nov 4"}, "retrieved is not a YYYY-MM-DD date"),
        ({"retrieved": "2026-02-30"}, "retrieved is not a YYYY-MM-DD date"),
        ({"retrieved": "2026-11-09"}, "is in the future"),
        ({"page_title": ""}, "page_title is empty"),
        ({"authors": []}, "authors must be a non-empty list"),
        ({"authors": ["ok", ""]}, "authors must be a non-empty list"),
        ({"year": True}, "is not a plausible year"),
        ({"year": 1850}, "is not a plausible year"),
        ({"year": 2031}, "is not a plausible year"),
        ({"year": "2023"}, "is not a plausible year"),
        ({"content_sha256": "ABC"}, "content_sha256 is not 64 lowercase hex"),
        ({"hash_source": "memory"}, "hash_source must be one of"),
    ],
)
def test_reference_problems(changes: dict[str, Any], message: str) -> None:
    entry = {**reference("a", "https://a.org"), **changes}
    problems = ct.reference_problems(entry, TODAY)
    assert any(message in p for p in problems), problems


def test_verified_entry_missing_fields() -> None:
    entry = reference("a", "https://a.org")
    del entry["content_sha256"], entry["authors"]
    problems = ct.reference_problems(entry, TODAY)
    assert problems == ["verified but missing authors, content_sha256"]


def test_load_references(tmp_path: Path) -> None:
    assert ct.load_references(tmp_path, TODAY) == (
        [],
        ["artifacts/citecheck.json is missing (the citation-checker subagent banks it)"],
    )
    path = tmp_path / "artifacts" / "citecheck.json"
    path.parent.mkdir()
    for content, message in [
        ("{oops", "not valid JSON"),
        ('{"references": [], "references": []}', "duplicate key"),
        ("[]", "is not a JSON object"),
        ('{"checked_by": "me", "references": {}}', "references must be a list"),
    ]:
        path.write_text(content, encoding="utf-8")
        refs, problems = ct.load_references(tmp_path, TODAY)
        assert refs == [] and any(message in p for p in problems), problems
    bank(
        tmp_path,
        [
            reference("a", "https://a.org/x"),
            reference("a", "https://b.org/x"),
            reference("c", "https://A.org/x/"),
            "not an object",
        ],
    )
    refs, problems = ct.load_references(tmp_path, TODAY)
    assert problems == ["reference 3 is not an object"]
    assert [r.problems for r in refs] == [
        (),
        ("duplicate key",),
        ("banks the same URL as reference a",),
    ]


def test_a_final_url_may_not_repeat_another_references_url(tmp_path: Path) -> None:
    bank(
        tmp_path,
        [
            reference("a", "https://a.org/x"),
            reference("b", "https://old.org/y", final_url="https://a.org/x/", status="mismatch"),
        ],
    )
    refs, _ = ct.load_references(tmp_path, TODAY)
    assert refs[1].problems == ("banks the same URL as reference a",)


# --------------------------------------------------------------------------- full check


def test_control_passes(repo: Path) -> None:
    out = lines(repo)
    assert out[-1] == "CITECHECK: PASS 2 verified"
    assert "note reference dropped is unverified and not cited (dropped)" in out
    assert any(
        line.startswith(
            "ok   reference liu2023: verified, retrieved 2026-11-04, cited at README.md:3"
        )
        for line in out
    )
    assert any(
        line.startswith("ok   schema: artifacts/citecheck.json has 3 reference(s)") for line in out
    )


def test_bare_www_address_matches_either_scheme(repo: Path) -> None:
    text = README + "\nSee www.arxiv.org/abs/9 and www.new.org/x.\n"
    (repo / "README.md").write_text(text, encoding="utf-8")
    data = json.loads((repo / "artifacts" / "citecheck.json").read_text(encoding="utf-8"))
    data["references"].append(reference("www", "http://www.arxiv.org/abs/9"))
    (repo / "artifacts" / "citecheck.json").write_text(json.dumps(data), encoding="utf-8")
    assert fails(repo) == [
        "FAIL README.md:11: www.new.org/x is not banked in artifacts/citecheck.json"
    ]


def test_cited_but_not_banked(repo: Path) -> None:
    (repo / "README.md").write_text(README + "\nAlso https://new.org/paper.\n", encoding="utf-8")
    assert fails(repo) == [
        "FAIL README.md:11: https://new.org/paper is not banked in artifacts/citecheck.json"
    ]


@pytest.mark.parametrize("status", ["unverified", "mismatch"])
def test_cited_reference_that_is_not_verified_fails(repo: Path, status: str) -> None:
    (repo / "README.md").write_text(
        README + "\nSee https://blocked.example-site.org/paper.\n", encoding="utf-8"
    )
    data = json.loads((repo / "artifacts" / "citecheck.json").read_text(encoding="utf-8"))
    data["references"][2]["status"] = status
    (repo / "artifacts" / "citecheck.json").write_text(json.dumps(data), encoding="utf-8")
    assert any(f"is banked as {status}" in p for p in fails(repo))


def test_verified_reference_with_schema_problems_fails(repo: Path) -> None:
    data = json.loads((repo / "artifacts" / "citecheck.json").read_text(encoding="utf-8"))
    data["references"][0]["year"] = "2023"
    (repo / "artifacts" / "citecheck.json").write_text(json.dumps(data), encoding="utf-8")
    out = lines(repo)
    assert "FAIL reference liu2023: year '2023' is not a plausible year" in out
    assert out[-1] == "CITECHECK: FAIL 1 problem(s)"


def test_wrong_checker_fails(repo: Path) -> None:
    data = json.loads((repo / "artifacts" / "citecheck.json").read_text(encoding="utf-8"))
    data["checked_by"] = "main-session"
    (repo / "artifacts" / "citecheck.json").write_text(json.dumps(data), encoding="utf-8")
    assert "FAIL schema: checked_by must be 'citation-checker'" in lines(repo)


def test_missing_bank_fails_every_citation(repo: Path) -> None:
    (repo / "artifacts" / "citecheck.json").unlink()
    problems = fails(repo)
    assert problems[0].startswith("FAIL schema: artifacts/citecheck.json is missing")
    assert len(problems) == 4


def test_citation_without_url_fails(repo: Path) -> None:
    (repo / "docs" / "REPORT.md").write_text(
        REPORT + "\nZhang et al. showed this.\n", encoding="utf-8"
    )
    assert any("looks like a citation but has no URL" in p for p in fails(repo))


def test_missing_and_unreadable_documents(repo: Path) -> None:
    (repo / "docs" / "REPORT.md").unlink()
    (repo / "README.md").write_bytes(b"caf\xe9\n")
    problems = fails(repo)
    assert any(p.startswith("FAIL document README.md: cannot be read") for p in problems)
    assert "FAIL document docs/REPORT.md: missing" in problems


def test_no_urls_cited(repo: Path) -> None:
    (repo / "README.md").write_text("# half-life\n", encoding="utf-8")
    (repo / "docs" / "REPORT.md").write_text("# Report\n", encoding="utf-8")
    out = lines(repo)
    assert "note no URLs cited in README.md or docs/REPORT.md" in out
    assert out[-1] == "CITECHECK: PASS 0 verified"


def test_main(
    repo: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ct, "check", lambda root: CheckLog(lines=["CITECHECK: PASS 2 verified"]))
    assert ct.main(["--root", str(repo)]) == 0
    assert capsys.readouterr().out == "CITECHECK: PASS 2 verified\n"
    monkeypatch.setattr(
        ct, "check", lambda root: CheckLog(lines=["CITECHECK: FAIL 1 problem(s)"], problems=1)
    )
    assert ct.main(["--root", str(repo)]) == 1


def test_main_uses_today(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert ct.main(["--root", str(repo / "docs")]) == 0
    assert capsys.readouterr().out.splitlines()[-1] == "CITECHECK: PASS 2 verified"


def test_main_outside_a_repository(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert ct.main(["--root", str(tmp_path)]) == 2
    assert capsys.readouterr().out.startswith("CITECHECK: FAIL not inside a git work tree")

# Some cases are adapted from raise-v1 tests/mutation_test.py (claimcheck gate),
# https://github.com/raj200501/raise-v1-public, Copyright 2026 Raj Kashikar, Apache
# License 2.0 (LICENSES/Apache-2.0.txt; see NOTICE). Modified for half-life: numbers
# must be claim markers backed by a named artifact, not any value that rounds to them.
"""Tests for halflife.integrity.claimcheck (gate G10, result-number provenance).

Each exemption rule is tested with text it must exempt and with nearby text it must
not, and each manifest and claim check with a passing case and the failure it guards.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from halflife.integrity import CheckLog, GitError, IntegrityError
from halflife.integrity import claimcheck as cc

_IDENTITY = {
    "GIT_AUTHOR_NAME": "test",
    "GIT_AUTHOR_EMAIL": "test@example.invalid",
    "GIT_COMMITTER_NAME": "test",
    "GIT_COMMITTER_EMAIL": "test@example.invalid",
}


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


def flagged(text: str, root: Path | None = None) -> list[str]:
    """Tokens that the number scan does not exempt."""
    return [hit.token for hit in cc.parse_document("doc.md", text, root).numbers]


RESULT = {"accuracy": 0.4123, "baseline": 0.25, "n": 1000, "rows": [{"f": "a", "v": 3}]}

BASE_MANIFEST: dict[str, Any] = {
    "claims": {
        "acc": {"artifact": "artifacts/result.json", "pointer": "/accuracy", "format": ".3f"},
        "n": {"artifact": "artifacts/result.json", "pointer": "/n", "format": ",d"},
    },
    "allowlist": [{"value": "0.25", "context": "baseline of 0.25", "reason": "fixed chance level"}],
}
README = (
    "# half-life\n\n"
    "Accuracy was <!-- claim:acc -->0.412<!-- /claim --> against a baseline of 0.25 on "
    "<!-- claim:n -->1,000<!-- /claim --> units.\n"
)
REPORT = "# Report\n\n## 1. Introduction\n\nSee section 6 and `make claimcheck`.\n"


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    run_git(root, "init", "-q")
    write(
        root,
        {
            "artifacts/result.json": json.dumps(RESULT),
            "artifacts/claims.json": json.dumps(BASE_MANIFEST),
            "README.md": README,
            "docs/REPORT.md": REPORT,
        },
    )
    commit_all(root, "seed")
    return root


def run_check(root: Path, verbose: bool = False) -> CheckLog:
    return cc.check(root, verbose=verbose)


def fail_lines(root: Path) -> list[str]:
    return [line for line in run_check(root).lines if line.startswith("FAIL")]


def set_manifest(root: Path, manifest: dict[str, Any]) -> None:
    write(root, {"artifacts/claims.json": json.dumps(manifest)})


# --------------------------------------------------------------------------- markup


def test_fenced_and_inline_code_are_blanked() -> None:
    text = (
        "Intro 1\n"
        "```python\nx = 0.777\n```\n"
        "~~~~\nscore 0.5\n~~~\n~~~~\n"
        "Inline `0.333` and ``code with ` tick 4`` here\n"
        "  ```\n  indented fence 9\n  ```\n"
    )
    markup = cc.scan_markup(text)
    assert markup.problems == []
    assert flagged(text) == ["1"]


def test_unmatched_backtick_is_literal() -> None:
    assert flagged("A ` stray tick then 0.5\n") == ["0.5"]


def test_inline_fence_like_line_is_not_a_fence() -> None:
    assert flagged("```inline``` then 0.5\nnext 2\n") == ["0.5", "2"]


def test_unclosed_fence_and_comment_are_problems() -> None:
    assert cc.scan_markup("```\nno end\n").problems == [
        (1, "code fence opened here is never closed")
    ]
    problems = cc.scan_markup("text\n<!-- open\nstill\n").problems
    assert problems == [(2, "HTML comment opened here is never closed")]


def test_comments_are_blanked_single_and_multi_line() -> None:
    text = "a <!-- 0.5 --> b 1\n<!-- start\n0.9\nend --> c 2\n"
    assert flagged(text) == ["1", "2"]


def test_claim_markers_are_extracted() -> None:
    text = "x <!-- claim:e1.hl -->12.3<!-- /claim --> y <!--claim:b--> 4 <!--/claim-->\n"
    markup = cc.scan_markup(text, "README.md")
    assert [(c.claim_id, c.rendered, c.line) for c in markup.claims] == [
        ("e1.hl", "12.3", 1),
        ("b", "4", 1),
    ]
    assert markup.problems == []
    assert flagged(text) == []


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("<!-- claim:a -->1<!-- claim:b -->2<!-- /claim -->\n", "opens before claim a closes"),
        ("1<!-- /claim -->\n", "closing claim marker without an opening marker"),
        ("<!-- claim:a -->1\n", "claim a is not closed on its line"),
        ("<!-- claim:a -->1<!-- note --><!-- /claim -->\n", "comment inside claim a"),
        ("<!-- Claim: a -->1\n", "malformed claim marker"),
        ("<!-- claim:a -->1 <!-- unterminated\n-->\n", "claim a is not closed on its line"),
    ],
)
def test_marker_problems(text: str, message: str) -> None:
    problems = cc.scan_markup(text).problems
    assert any(message in p for _, p in problems), problems


def test_markers_inside_code_are_not_claims() -> None:
    text = "`<!-- claim:x -->1<!-- /claim -->`\n```\n<!-- claim:y -->2<!-- /claim -->\n```\n"
    assert cc.scan_markup(text).claims == []


def test_strip_markers() -> None:
    code = "\n```\n<!-- claim:c -->\n```"
    text = "A <!-- claim:a -->0.41<!-- /claim --> and `<!-- claim:b -->` kept" + code
    assert cc.strip_markers(text) == "A 0.41 and `<!-- claim:b -->` kept" + code


# --------------------------------------------------------------------------- exemptions


@pytest.mark.parametrize(
    "text",
    [
        "Retrieved 2026-11-04 at 2026-11-04T01:02:03Z.",
        "In 2026-11 we ran it.",
        "On November 4, 2026 and 4 Nov 2026 and in March 2025.",
        "Liu et al. (2023) and (Liu et al., 2024) and (2025b) and Smith et al. 2022.",
        "Claude Code 2.1.309 and llama.cpp 0.3.16.4.",
        "Commit 9330c9d396b3 and digest 3f2a9c0e1b7d4a6f.",
        "Gates G7 and G13, RQ3, D-012, SHA-256, v0.5, qwen2.5-0.5b, Q4_K_M, E1.",
        "See https://arxiv.org/abs/2307.03172 and www.example.org/a/1.",
        "A [link](artifacts/e1/40.json) and an ![image](docs/fig2.svg).",
        "[1]: https://example.org/ref/2",
        "See arXiv:2307.03172v2 and doi:10.1145/3597503.3639187.",
        'An <img src="a.svg" width="640"> tag and &#8212; entity.',
        "Footnote[^3] and citation [12].",
        "Read section 6, rule 11, Figure 2, Fig. 3, table 1, step 4, appendix 2, gate 7.",
        "See Section 10.1 step 11 and \N{SECTION SIGN}9.",
        "CI runs Python 3.11 and 3.12.",
        "Read sections 6 to 10, figures 2-4, tables 1\N{EN DASH}3, and steps 1 through 5.",
        "Follow rules 1, 9, and 12 (sections 6, 7).",
        "Steps 1, 2, 3.",
        "Preregistration 0001, amendment 0002, seed 900000, line 42, version 2.",
    ],
)
def test_exempt_text(text: str) -> None:
    assert flagged(text + "\n") == []


@pytest.mark.parametrize(
    ("text", "tokens"),
    [
        ("We reached 0.981 on the held-out slice.", ["0.981"]),
        ("n=40 sessions and +3 more and -0.12 change and 95% of runs.", ["40", "3", "0.12", "95"]),
        ("In Figures 2 and 3, 87 sessions broke a rule.", ["87"]),
        ("In Figure 2, 87 of 120 sessions broke a rule.", ["87", "120"]),
        ("Read section 6 to 10 of the brief.", ["10"]),
        ("Sections 6 and 7 cover 40 sessions.", ["40"]),
        ("Ran 87/120 sessions.", ["87", "120"]),
        (
            "Interval [0.91, 0.98] and list [40, 50] and [3](x.md).",
            ["0.91", "0.98", "40", "50", "3"],
        ),
        ("See https://example.com/run/12345 which reports 0.633 accuracy.", ["0.633"]),
        ("Python 3.11 is two-part, 3.12 alone is not.", ["3.12"]),
        ("Written in 2026.", ["2026"]),
        ("Times 10:30 are numbers.", ["10", "30"]),
        ("x <b and 0.95 > y", ["0.95"]),
        ('<svg><text x="574.0" y="45.3">0.981</text></svg>', ["0.981"]),
        ("1e-3 scientific", ["1"]),
    ],
)
def test_flagged_text(text: str, tokens: list[str]) -> None:
    assert flagged(text + "\n") == tokens


def test_numerals_hidden_from_digit_scans_are_flagged() -> None:
    assert flagged("We ran &#52;&#48; sessions and &#x34; more; &amp; &mdash; are fine.\n") == [
        "&#52;",
        "&#48;",
        "&#x34;",
    ]
    assert flagged("About \N{VULGAR FRACTION ONE HALF} of runs, x\N{SUPERSCRIPT TWO}.\n") == [
        "\N{VULGAR FRACTION ONE HALF}",
        "\N{SUPERSCRIPT TWO}",
    ]
    assert flagged("Fullwidth \uff19\uff15 is a digit run.\n") == ["\uff19\uff15"]
    assert flagged("`&#52;` in code is fine.\n") == []


def test_text_attributes_of_html_tags_are_checked() -> None:
    text = (
        '<img src="docs/f2.svg" width="640" alt="Half-life 35 tool calls">\n'
        "<span title='precision 0.95'>x</span> <abbr ARIA-LABEL=\"n 40\">n</abbr>\n"
        '<img src="a.svg" height="480" alt="Survival curves">\n'
    )
    assert flagged(text) == ["35", "0.95", "40"]


def test_heading_numbers() -> None:
    text = (
        "## 3. Method\n### 3.2 Sessions\n#### 3.2.1 Detail\n## 4) Results\n"
        "## 40 sessions per family\n### 0.95 precision\n## 87% of sessions\n"
        "### 9.1 Orphan subsection\n## 5. Results for 12 sessions\n"
    )
    assert flagged(text) == ["40", "0.95", "87", "9.1", "12"]


def test_list_markers() -> None:
    assert flagged("1. First 0.5\n2) Second\n> 3. Quoted\n- 4 apples\n") == ["0.5", "4"]


def test_existing_paths_are_exempt(tmp_path: Path) -> None:
    (tmp_path / "prereg").mkdir()
    (tmp_path / "prereg" / "0001-e1.md").write_text("x", encoding="utf-8")
    text = "See prereg/0001-e1.md. and prereg/0002-e1.md and ../prereg/0001-e1.md\n"
    assert flagged(text, tmp_path) == ["0002", "0001"]


# --------------------------------------------------------------------------- manifest


def test_parse_manifest_accepts_a_full_manifest() -> None:
    manifest, problems = cc.parse_manifest(
        {
            "schema": "halflife/claims/1",
            "notes": "manifest",
            "documents": ["docs/MITIGATIONS.md", "docs/MITIGATIONS.md", "outbound/a.txt"],
            "claims": {
                "a": {
                    "artifact": "artifacts/e1/sessions.jsonl",
                    "where": {"family": "qwen"},
                    "aggregate": "count",
                    "format": "d",
                    "note": "sessions",
                }
            },
            "allowlist": [{"value": "4,000", "context": "4,000 words", "reason": "brief"}],
            "derivations": {
                "artifacts/e1/hazard.json": {
                    "derive": "halflife.stats:hazard_artifact",
                    "inputs": ["artifacts/e1/opportunities.jsonl", "artifacts/bench/"],
                    "params": {"seed": 1},
                    "compare": ["/estimates"],
                }
            },
            "raw": {"artifacts/e1/opportunities.jsonl": "runner output", "artifacts/bench/": "x"},
        }
    )
    assert problems == []
    assert manifest.documents == ("docs/MITIGATIONS.md", "outbound/a.txt")
    assert manifest.claims["a"].aggregate == "count"
    assert manifest.allowlist[0].context == "4,000 words"
    assert manifest.derivations["artifacts/e1/hazard.json"].compare == ("/estimates",)
    assert set(manifest.raw) == {"artifacts/e1/opportunities.jsonl", "artifacts/bench/"}


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ([], "is not a JSON object"),
        ({"extra": 1}, "unknown top-level key"),
        ({"documents": "README.md"}, "'documents' must be a list"),
        ({"documents": ["../x.md"]}, "normal form"),
        ({"documents": ["docs/x.json"]}, "is not .md or .txt"),
        ({"claims": []}, "'claims' must be an object"),
        ({"claims": {"bad id": {}}}, "id must match"),
        ({"claims": {"a": 1}}, "entry is not an object"),
        ({"claims": {"a": {"artifact": "artifacts/x.json", "format": "", "x": 1}}}, "unknown"),
        ({"claims": {"a": {"artifact": "docs/x.json", "format": ""}}}, "not under artifacts/"),
        ({"claims": {"a": {"artifact": "artifacts/x.csv", "format": ""}}}, "not a .json"),
        ({"claims": {"a": {"artifact": "artifacts/claims.json", "format": ""}}}, "bookkeeping"),
        ({"claims": {"a": {"artifact": "artifacts/citecheck.json", "format": ""}}}, "bookkeeping"),
        (
            {"claims": {"a": {"artifact": "artifacts/x.json", "pointer": "a", "format": ""}}},
            "JSON pointers",
        ),
        ({"claims": {"a": {"artifact": "artifacts/x.json", "where": {}, "format": ""}}}, "'where'"),
        (
            {"claims": {"a": {"artifact": "artifacts/x.json", "aggregate": "mean", "format": ""}}},
            "aggregate must",
        ),
        (
            {
                "claims": {
                    "a": {
                        "artifact": "artifacts/x.json",
                        "aggregate": "count",
                        "field": "/v",
                        "format": "",
                    }
                }
            },
            "takes no field",
        ),
        ({"claims": {"a": {"artifact": "artifacts/x.json"}}}, "'format'"),
        ({"allowlist": {}}, "'allowlist' must be a list"),
        ({"allowlist": [{"value": "1"}]}, "needs exactly the keys"),
        ({"allowlist": [{"value": "1a", "context": "a 1a b", "reason": "r"}]}, "not a number"),
        ({"allowlist": [{"value": "1", "context": "rule 1", "reason": " "}]}, "needs a reason"),
        ({"allowlist": [{"value": "1", "context": "rule 2", "reason": "r"}]}, "literal text"),
        ({"allowlist": [{"value": "1", "context": "1 x", "reason": "r"}]}, "literal text"),
        ({"derivations": []}, "'derivations' must be an object"),
        ({"derivations": {"docs/x.json": {}}}, "not under artifacts/"),
        ({"derivations": {"artifacts/x.json": []}}, "entry must be an object"),
        ({"derivations": {"artifacts/x.json": {"other": 1}}}, "entry must be an object"),
        (
            {
                "derivations": {
                    "artifacts/x.json": {"derive": "nope", "inputs": ["artifacts/y.json"]}
                }
            },
            "module:function",
        ),
        ({"derivations": {"artifacts/x.json": {"derive": "m:f", "inputs": []}}}, "at least one"),
        (
            {"derivations": {"artifacts/x.json": {"derive": "m:f", "inputs": ["docs/y.json"]}}},
            "input",
        ),
        (
            {
                "derivations": {
                    "artifacts/x.json": {
                        "derive": "m:f",
                        "inputs": ["artifacts/y.json"],
                        "params": {"a-b": 1},
                    }
                }
            },
            "'params'",
        ),
        (
            {
                "derivations": {
                    "artifacts/x.json": {
                        "derive": "m:f",
                        "inputs": ["artifacts/y.json"],
                        "compare": [],
                    }
                }
            },
            "'compare'",
        ),
        ({"raw": []}, "'raw' must be an object"),
        ({"raw": {"docs/x.json": "r"}}, "not under artifacts/"),
        ({"raw": {"artifacts/x.json": ""}}, "needs a reason"),
    ],
)
def test_parse_manifest_problems(data: Any, message: str) -> None:
    _, problems = cc.parse_manifest(data)
    assert any(message in p for p in problems), problems


def test_load_manifest(tmp_path: Path) -> None:
    _, problems = cc.load_manifest(tmp_path)
    assert problems == ["artifacts/claims.json is missing (it maps claim ids to artifacts)"]
    write(tmp_path, {"artifacts/claims.json": "{nope"})
    _, problems = cc.load_manifest(tmp_path)
    assert "not valid JSON" in problems[0]
    write(tmp_path, {"artifacts/claims.json": '{"claims": {}, "claims": {}}'})
    _, problems = cc.load_manifest(tmp_path)
    assert "duplicate key" in problems[0]


# --------------------------------------------------------------------------- selection


def test_resolve_pointer() -> None:
    doc = {"a": [{"b": 1}, {"c/d": 2, "e~f": 3}], "": 4}
    assert cc.resolve_pointer(doc, "") is doc
    assert cc.resolve_pointer(doc, "/a/0/b") == 1
    assert cc.resolve_pointer(doc, "/a/1/c~1d") == 2
    assert cc.resolve_pointer(doc, "/a/1/e~0f") == 3
    assert cc.resolve_pointer(doc, "/") == 4
    for pointer, message in [
        ("a", "must be empty or start"),
        ("/x", "no key"),
        ("/a/2", "no index"),
        ("/a/01", "no index"),
        ("/a/-", "no index"),
        ("/a/0/b/c", "cannot index into int"),
        ("/a/1/e~2f", "invalid escape"),
    ]:
        with pytest.raises(cc.SelectorError, match=message):
            cc.resolve_pointer(doc, pointer)


def spec(**kw: Any) -> cc.ClaimSpec:
    return cc.ClaimSpec(claim_id="x", artifact="artifacts/x.json", **kw)


ROWS = [
    {"family": "qwen", "reached": True, "n": 2, "rate": 0.5},
    {"family": "qwen", "reached": False, "n": 3, "rate": 0.25},
    {"family": "llama", "reached": True, "n": 4, "rate": 0.125},
    "not a row",
]


def test_select_variants() -> None:
    doc = {"rows": ROWS, "value": 0.5}
    assert cc.select(doc, spec(pointer="/value")) == 0.5
    assert cc.select(doc, spec(pointer="/rows", aggregate="count")) == 4
    assert cc.select(doc, spec(pointer="/rows", where={"family": "qwen"}, aggregate="count")) == 2
    assert cc.select(doc, spec(pointer="/rows", where={"reached": True}, aggregate="count")) == 2
    assert cc.select(doc, spec(pointer="/rows", where={"reached": 1}, aggregate="count")) == 0
    assert (
        cc.select(doc, spec(pointer="/rows", where={"family": "qwen"}, aggregate="sum", field="/n"))
        == 5
    )
    total = cc.select(
        doc, spec(pointer="/rows", where={"family": "qwen"}, aggregate="sum", field="/rate")
    )
    assert total == 0.75
    assert cc.select(doc, spec(pointer="/rows", where={"family": "llama"}, field="/n")) == 4


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"pointer": "/value", "aggregate": "count"}, "does not select a list"),
        ({"pointer": "/rows", "where": {"family": "qwen"}, "field": "/n"}, "matched 2 rows"),
        ({"pointer": "/rows", "where": {"family": "none"}, "field": "/n"}, "matched 0 rows"),
        (
            {
                "pointer": "/rows",
                "where": {"family": "qwen"},
                "aggregate": "sum",
                "field": "/family",
            },
            "needs numbers",
        ),
    ],
)
def test_select_errors(kw: dict[str, Any], message: str) -> None:
    with pytest.raises(cc.SelectorError, match=message):
        cc.select({"rows": ROWS, "value": 0.5}, spec(**kw))


def test_render_value() -> None:
    assert cc.render_value(0.41234, ".3f") == "0.412"
    assert cc.render_value(0.953, ".1%") == "95.3%"
    assert cc.render_value(1000, ",d") == "1,000"
    assert cc.render_value("qwen2.5-3b", "") == "qwen2.5-3b"
    for value in (True, None, [1], {"a": 1}):
        with pytest.raises(cc.SelectorError, match="not a number or string"):
            cc.render_value(value, "")
    with pytest.raises(cc.SelectorError, match="does not apply"):
        cc.render_value(40.0, "d")


def test_same_json() -> None:
    assert cc.same_json({"a": [1, 2]}, {"a": [1, 2]})
    assert not cc.same_json(True, 1)
    assert not cc.same_json(1, 1.0)


def test_artifacts_loader_notes(repo: Path) -> None:
    artifacts = cc.Artifacts(repo)
    assert artifacts.load("artifacts/result.json")["n"] == 1000
    assert artifacts.notes == set()
    write(repo, {"artifacts/new.json": "{}", "artifacts/result.json": json.dumps({"n": 2})})
    fresh = cc.Artifacts(repo)
    fresh.load("artifacts/new.json")
    fresh.load("artifacts/result.json")
    fresh.load("artifacts/result.json")
    assert len(fresh.notes) == 2
    with pytest.raises(IntegrityError, match="does not exist"):
        fresh.load("artifacts/absent.json")
    with pytest.raises(IntegrityError, match="not under artifacts/"):
        fresh.load("docs/x.json")


# --------------------------------------------------------------------------- documents


def test_document_set(repo: Path) -> None:
    write(
        repo,
        {
            "outbound/email.txt": "Hi",
            "outbound/notes.md": "Hi",
            "outbound/data.json": "{}",
            "docs/MITIGATIONS.md": "Overhead <!-- claim:n -->1,000<!-- /claim -->\n",
            "docs/CLAIMS.md": "Syntax: `<!-- claim:x -->1<!-- /claim -->`\n",
            "docs/BUILD_BRIEF.md": "<!-- claim:n -->1,000<!-- /claim -->\n",
            "docs/broken.md": "",
        },
    )
    (repo / "docs" / "latin.md").write_bytes(b"claim: \xff")
    manifest, _ = cc.parse_manifest({"documents": ["docs/EXTRA.md", "docs/BUILD_BRIEF.md"]})
    assert cc.document_set(repo, manifest) == [
        "README.md",
        "docs/REPORT.md",
        "docs/EXTRA.md",
        "outbound/email.txt",
        "outbound/notes.md",
        "docs/MITIGATIONS.md",
    ]


def test_allowlist_needs_its_context() -> None:
    entry = cc.AllowEntry("0.25", "baseline of 0.25", "chance level")
    markup = cc.scan_markup("A baseline of 0.25 here; elsewhere 0.25 is a result.\n")
    hits = cc.scan_numbers("doc.md", markup)
    unmarked, used = cc.apply_allowlist(hits, [entry])
    assert [h.start for h in unmarked] == [hits[1].start]
    assert used == {0}
    unmarked, used = cc.apply_allowlist(hits[1:], [entry])
    assert len(unmarked) == 1 and used == set()


# --------------------------------------------------------------------------- full check


def test_control_passes(repo: Path) -> None:
    out = run_check(repo).lines
    assert out[-1] == "CLAIMCHECK: PASS 2 claim(s) in 2 document(s)"
    assert (
        "ok   document README.md: 2 claim(s) match their artifacts, 1 allowlisted "
        "number(s), no unmarked numbers" in out
    )


def test_verbose_lists_claims(repo: Path) -> None:
    out = run_check(repo, verbose=True).lines
    assert any(line.startswith("ok   claim acc: 0.412 from artifacts/result.json") for line in out)


def test_fabricated_number_fails(repo: Path) -> None:
    write(repo, {"README.md": README + "We also reached 0.981 on the held-out slice.\n"})
    problems = fail_lines(repo)
    assert len(problems) == 1 and "unmarked number '0.981'" in problems[0]


def test_precision_overclaim_fails(repo: Path) -> None:
    write(repo, {"README.md": README.replace("-->0.412<", "-->0.4123<")})
    assert any(
        "renders '0.4123' but artifacts/result.json gives '0.412'" in p for p in fail_lines(repo)
    )


def test_stale_rendered_value_fails(repo: Path) -> None:
    write(repo, {"artifacts/result.json": json.dumps({**RESULT, "accuracy": 0.5})})
    assert any("gives '0.500'" in p for p in fail_lines(repo))


def test_backing_artifact_deleted_fails(repo: Path) -> None:
    (repo / "artifacts" / "result.json").unlink()
    assert any("artifact artifacts/result.json does not exist" in p for p in fail_lines(repo))


def test_undefined_claim_fails(repo: Path) -> None:
    write(repo, {"README.md": README + "Also <!-- claim:ghost -->7<!-- /claim -->.\n"})
    assert any("claim 'ghost' is not defined" in p for p in fail_lines(repo))


def test_selector_that_does_not_resolve_fails(repo: Path) -> None:
    manifest = json.loads(json.dumps(BASE_MANIFEST))
    manifest["claims"]["acc"]["pointer"] = "/missing"
    set_manifest(repo, manifest)
    assert any("claim acc: pointer '/missing': no key" in p for p in fail_lines(repo))


def test_allowlist_entry_without_reason_is_rejected(repo: Path) -> None:
    manifest = json.loads(json.dumps(BASE_MANIFEST))
    manifest["allowlist"].append({"value": "0.981", "context": "reached 0.981", "reason": ""})
    set_manifest(repo, manifest)
    write(repo, {"README.md": README + "We reached 0.981.\n"})
    problems = fail_lines(repo)
    assert any("needs a reason" in p for p in problems)
    assert any("unmarked number '0.981'" in p for p in problems)


def test_allowlist_admits_exactly_that_number_in_context(repo: Path) -> None:
    manifest = json.loads(json.dumps(BASE_MANIFEST))
    manifest["allowlist"].append(
        {"value": "2026", "context": "Written in 2026", "reason": "calendar year"}
    )
    set_manifest(repo, manifest)
    write(repo, {"README.md": README + "Written in 2026.\n"})
    assert run_check(repo).lines[-1].startswith("CLAIMCHECK: PASS")
    write(repo, {"README.md": README + "Written in 2026. We ran 2026 sessions.\n"})
    problems = fail_lines(repo)
    assert len(problems) == 1 and "unmarked number '2026'" in problems[0]


def test_missing_documents_fail(repo: Path) -> None:
    (repo / "docs" / "REPORT.md").unlink()
    assert "FAIL document docs/REPORT.md: missing" in run_check(repo).lines
    manifest = json.loads(json.dumps(BASE_MANIFEST))
    manifest["documents"] = ["docs/RESULTS.md"]
    set_manifest(repo, manifest)
    assert "FAIL document docs/RESULTS.md: missing" in run_check(repo).lines


def test_missing_manifest_fails(repo: Path) -> None:
    (repo / "artifacts" / "claims.json").unlink()
    problems = fail_lines(repo)
    assert any("artifacts/claims.json is missing" in p for p in problems)
    assert any("claim 'acc' is not defined" in p for p in problems)


def test_non_utf8_document_fails(repo: Path) -> None:
    (repo / "README.md").write_bytes(b"caf\xe9\n")
    assert any("is not UTF-8 text" in p for p in fail_lines(repo))


def test_marker_problem_fails(repo: Path) -> None:
    write(repo, {"docs/REPORT.md": REPORT + "<!-- claim:acc -->0.412\n"})
    assert any("claim acc is not closed on its line" in p for p in fail_lines(repo))


def test_outbound_drafts_are_checked(repo: Path) -> None:
    write(repo, {"outbound/email.txt": "Dear Raj, we reached 0.97 on E1.\n"})
    assert any("outbound/email.txt:1" in p and "'0.97'" in p for p in fail_lines(repo))


def test_notes_for_unused_claims_idle_allowlist_and_uncommitted_artifacts(repo: Path) -> None:
    manifest = json.loads(json.dumps(BASE_MANIFEST))
    manifest["claims"]["spare"] = {
        "artifact": "artifacts/result.json",
        "pointer": "/n",
        "format": "d",
    }
    manifest["allowlist"].append({"value": "7", "context": "rule 7 holds", "reason": "unused"})
    set_manifest(repo, manifest)
    write(repo, {"artifacts/result.json": json.dumps(RESULT) + "\n"})
    out = run_check(repo).lines
    assert out[-1].startswith("CLAIMCHECK: PASS")
    assert "note 1 manifest claim(s) unused in documents: spare" in out
    assert "note 1 allowlist entr(ies) matched nothing: 7" in out
    assert any("has uncommitted changes" in line for line in out)


# --------------------------------------------------------------------------- command line


def test_main(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cc.main(["--root", str(repo)]) == 0
    assert capsys.readouterr().out.splitlines()[-1].startswith("CLAIMCHECK: PASS")
    write(repo, {"README.md": README + "Fabricated 0.99.\n"})
    assert cc.main(["--root", str(repo), "--verbose"]) == 1
    assert capsys.readouterr().out.splitlines()[-1] == "CLAIMCHECK: FAIL 1 problem(s)"


def test_main_strip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    draft = tmp_path / "email.txt"
    draft.write_text("Accuracy <!-- claim:acc -->0.412<!-- /claim -->.\n", encoding="utf-8")
    assert cc.main(["--strip", str(draft)]) == 0
    assert capsys.readouterr().out == "Accuracy 0.412.\n"
    assert cc.main(["--strip", str(tmp_path / "absent.txt")]) == 2
    assert "cannot read" in capsys.readouterr().err


def test_main_outside_a_repository(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cc.main(["--root", str(tmp_path)]) == 2
    assert capsys.readouterr().out.startswith("CLAIMCHECK: FAIL not inside a git work tree")


def test_main_reports_a_failed_git_query(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(*_: object, **__: object) -> CheckLog:
        raise GitError("ls-files failed")

    monkeypatch.setattr(cc, "check", broken)
    assert cc.main(["--root", str(repo)]) == 1
    assert "cannot be completed" in capsys.readouterr().out

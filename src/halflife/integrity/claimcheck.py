# Adapted from raise-v1 tools/claimcheck.py (https://github.com/raj200501/raise-v1-public),
# Copyright 2026 Raj Kashikar, licensed under the Apache License, Version 2.0
# (LICENSES/Apache-2.0.txt; see NOTICE). Modified for half-life: instead of accepting any
# number that some banked value rounds to, every result number must sit inside a claim
# marker whose manifest entry names one artifact, one selector, and one format, and the
# rendered text must equal the formatted value exactly. Carried over from the original:
# blanking spans that are addresses rather than claims (URLs, digests, list markers) with
# spaces so line numbers stay right, an allowlist in which every entry needs a reason,
# and the default of checking every outbound draft.
"""Result-number provenance check for gate G10 (``make claimcheck``).

Rule 1 of the build brief: every result number in README.md, docs/REPORT.md, and other
results documents comes from a banked artifact under ``artifacts/``. A result number is
written inside a claim marker::

    <!-- claim:e1.halflife.family_a -->23.4<!-- /claim -->

and ``artifacts/claims.json`` maps the claim id to an artifact, a selector (a JSON
pointer, optionally with a row filter and an aggregate), and a format spec. The check
fails on unmarked numbers that no explicit rule exempts, on claims whose id, artifact,
or selector does not resolve, on rendered values that differ from the formatted artifact
value, on malformed markers, and on missing documents. Exemption rules, the selector
grammar, and the allowlist are documented in docs/CLAIMS.md and in :func:`scan_numbers`.

Output is one line per document (and per claim with ``--verbose``), one line per
problem, then ``CLAIMCHECK: PASS <n> claim(s) in <m> document(s)`` or
``CLAIMCHECK: FAIL <n> problem(s)``. ``--strip FILE`` prints a document with its claim
markers removed, for outbound copy that is sent as plain text.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from halflife.integrity import (
    CheckLog,
    GitError,
    IntegrityError,
    in_head,
    load_artifact,
    load_json,
    repo_root,
    safe_relpath,
    worktree_files,
    worktree_matches_head,
)

__all__ = [
    "AllowEntry",
    "Artifacts",
    "ClaimSpec",
    "ClaimUse",
    "Derivation",
    "Manifest",
    "Markup",
    "NumberHit",
    "ParsedDoc",
    "SelectorError",
    "apply_allowlist",
    "check",
    "document_set",
    "load_manifest",
    "main",
    "parse_document",
    "parse_manifest",
    "render_value",
    "resolve_pointer",
    "same_json",
    "scan_markup",
    "scan_numbers",
    "select",
    "strip_markers",
]

TOOL = "CLAIMCHECK"
MANIFEST = "artifacts/claims.json"
CITECHECK = "artifacts/citecheck.json"
NOT_RESULT_ARTIFACTS = frozenset({MANIFEST, CITECHECK})
ARTIFACTS_DIR = "artifacts"
REQUIRED_DOCS = ("README.md", "docs/REPORT.md")
OUTBOUND_DIR = "outbound"
NEVER_RESULTS = frozenset({"docs/BUILD_BRIEF.md"})
ARTIFACT_SUFFIXES = (".json", ".jsonl")
DOC_SUFFIXES = (".md", ".txt")

CLAIM_ID = r"[A-Za-z0-9][A-Za-z0-9_.-]*"
OPEN_RE = re.compile(rf"^\s*claim:(?P<id>{CLAIM_ID})\s*$")
CLOSE_RE = re.compile(r"^\s*/claim\s*$")
CLAIMISH_RE = re.compile(r"^\s*/?\s*claim\b", re.IGNORECASE)
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")

TAG_RE = re.compile(
    r"</?[A-Za-z][A-Za-z0-9-]*"
    r"(?:\s+[A-Za-z_:][\w:.-]*(?:\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s\"'=<>`]+))?)*"
    r"\s*/?>"
)
ENTITY_RE = re.compile(r"&(?:#[0-9]+|#[xX][0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]*);")
TEXT_ATTR_RE = re.compile(
    r"\b(?:alt|title|aria-label)\s*=\s*(?:\"(?P<dq>[^\"]*)\"|'(?P<sq>[^']*)')", re.IGNORECASE
)
URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"'`()\[\],;]+")
LINK_TARGET_RE = re.compile(r"\]\(\s*<?(?P<target>[^)\s>]*)>?(?:\s+\"[^\"]*\")?\s*\)")
LINK_DEF_RE = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*(?P<target>\S+)")
ARXIV_RE = re.compile(r"\barXiv:\s?\d{4}\.\d{4,5}(?:v\d+)?\b", re.IGNORECASE)
DOI_RE = re.compile(r"\b(?:doi:\s?)?10\.\d{4,9}/[^\s<>\"'`]+", re.IGNORECASE)
ISO_DATE_RE = re.compile(
    r"(?<![\w.-])(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})"
    r"(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?(?![\w.]*\d)"
)
YEAR_MONTH_RE = re.compile(r"(?<![\w.-])(?:19|20)\d{2}-(?:0[1-9]|1[0-2])(?![\w.-]*\d)")
MONTHS = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|"
    r"December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?"
)
MONTH_DATE_RES = (
    re.compile(rf"\b{MONTHS}\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+(?:19|20)\d{{2}}\b"),
    re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{MONTHS},?\s+(?:19|20)\d{{2}}\b"),
    re.compile(rf"\b{MONTHS},?\s+(?:19|20)\d{{2}}\b"),
)
CITATION_YEAR_RES = (
    re.compile(r"\((?:[^()\n]*?(?:,|\bet al\.?)\s*)?(?P<year>(?:19|20)\d{2}[a-z]?)\)"),
    re.compile(r"\bet al\.?,?\s+(?P<year>(?:19|20)\d{2}[a-z]?)\b"),
)
VERSION_RE = re.compile(r"(?<![\w.])\d+\.\d+\.\d+(?:\.\d+)*(?![\w.]*\d)")
HEX_RE = re.compile(r"(?<![\w-])(?=[0-9a-f]*[a-f])(?=[0-9a-f]*[0-9])[0-9a-f]{7,64}(?![\w-])")
PATH_RE = re.compile(r"(?<![\w./-])[\w.-]+(?:/[\w.-]+)+/?")
TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*")
WORD_RUN_RE = re.compile(r"[A-Za-z0-9_.+-]*$")
HEADING_NUM_RE = re.compile(r"^\s{0,3}#{1,6}\s+(?P<num>\d+(?:\.\d+)*)(?P<punct>[.)]?)(?=\s|$)")
LIST_NUM_RE = re.compile(r"^\s*(?:>\s*)*(?P<num>\d{1,9})[.)](?=\s|$)")

# Reference words: a number right after one of these is a label, not a result. Only the
# plural forms (and "Python", for version lists) start a chain of further numbers.
LABEL_WORDS: dict[str, str] = {
    "section": "sections",
    "rule": "rules",
    "figure": "figures",
    "fig.": "figs.",
    "table": "tables",
    "step": "steps",
    "appendix": "appendices",
    "equation": "equations",
    "eq.": "eqs.",
    "chapter": "chapters",
    "rung": "rungs",
    "milestone": "milestones",
    "gate": "gates",
    "line": "lines",
    "version": "versions",
    "preregistration": "preregistrations",
    "amendment": "amendments",
    "seed": "seeds",
}
CHAINABLE = frozenset({*LABEL_WORDS.values(), "python", "\N{SECTION SIGN}" * 2})
_LABEL_ALTERNATIVES = sorted({*LABEL_WORDS, *LABEL_WORDS.values(), "python"}, key=len, reverse=True)
LABEL_RE = re.compile(
    r"(?:\b(?P<word>" + "|".join(re.escape(w) for w in _LABEL_ALTERNATIVES) + r")"
    r"|(?P<sign>\N{SECTION SIGN}{1,2}))\s*$",
    re.IGNORECASE,
)
CONNECTOR_RE = re.compile(r"\s*(?:,\s*(?:and|or)?|and|or|to|through|-|\N{EN DASH})\s*")
BARE_COMMA_RE = re.compile(r"\s*,\s*")
LIST_GOES_ON_RE = re.compile(r"\s*(?:$|[.,;:)\]]|(?:and|or|to|through)\b|-|\N{EN DASH})")
DERIVE_RE = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")
VALUE_RE = re.compile(r"^\d+(?:[.,]\d+)*$")
MARKER_RE = re.compile(rf"<!--\s*claim:{CLAIM_ID}\s*-->|<!--\s*/claim\s*-->")

TOP_KEYS = frozenset({"schema", "notes", "documents", "claims", "allowlist", "derivations", "raw"})
CLAIM_KEYS = frozenset({"artifact", "pointer", "where", "field", "aggregate", "format", "note"})
ALLOW_KEYS = frozenset({"value", "context", "reason"})
DERIVATION_KEYS = frozenset({"derive", "inputs", "params", "compare", "note"})
AGGREGATES = ("count", "sum")


class SelectorError(IntegrityError):
    """Raised when a pointer, filter, aggregate, or format does not resolve."""


# --------------------------------------------------------------------------- documents


@dataclass(frozen=True)
class ClaimUse:
    """One claim marker in a document: its id and the text rendered between the markers."""

    doc: str
    line: int
    claim_id: str
    rendered: str


@dataclass(frozen=True)
class NumberHit:
    """A digit sequence outside claim markers that no built-in rule exempts.

    Attributes:
        doc: Document path.
        line: 1-based line number.
        token: The digits as written (``4,000``, ``0.95``).
        start: Column where the token starts.
        end: Column just past the token.
        text: The full source line.
    """

    doc: str
    line: int
    token: str
    start: int
    end: int
    text: str


@dataclass
class Markup:
    """A document with code, comments, and claim values blanked out.

    ``masked`` has the same lines and line lengths as ``lines``; blanked characters are
    spaces, so columns and line numbers still point into the original text. ``markers``
    holds ``(line index, start, end)`` of every claim marker that was parsed.
    """

    lines: list[str]
    masked: list[str]
    claims: list[ClaimUse] = field(default_factory=list)
    problems: list[tuple[int, str]] = field(default_factory=list)
    markers: list[tuple[int, int, int]] = field(default_factory=list)


@dataclass(frozen=True)
class ParsedDoc:
    """Claims, candidate unmarked numbers, and structural problems of one document."""

    path: str
    claims: tuple[ClaimUse, ...]
    numbers: tuple[NumberHit, ...]
    problems: tuple[tuple[int, str], ...]


def _blank(text: str, spans: Iterable[tuple[int, int]]) -> str:
    chars = list(text)
    for start, end in spans:
        for k in range(max(start, 0), min(end, len(chars))):
            chars[k] = " "
    return "".join(chars)


def _closing_ticks(line: str, start: int, size: int) -> int:
    """Index of the next run of exactly ``size`` backticks at or after ``start``, or -1."""
    for match in re.finditer(r"`+", line[start:]):
        if len(match.group()) == size:
            return start + match.start()
    return -1


def scan_markup(text: str, doc: str = "") -> Markup:
    """Blank fenced code, inline code, and HTML comments, and extract claim markers.

    Claim markers are single-line HTML comments ``<!-- claim:ID -->`` and
    ``<!-- /claim -->`` around the rendered value, on one line. Problems: a code fence
    or comment that never closes, a claim that is not closed on its line, nested or
    stray markers, comments inside a claim, and comments that look like claim markers
    but do not parse.
    """
    lines = [line.rstrip("\r") for line in text.split("\n")]
    markup = Markup(lines=lines, masked=list(lines))
    fence: tuple[str, int, int] | None = None
    comment_from: int | None = None
    for i, line in enumerate(lines):
        if fence is not None:
            markup.masked[i] = " " * len(line)
            char, size, _ = fence
            if re.match(rf"^\s*{re.escape(char)}{{{size},}}[ \t]*$", line):
                fence = None
            continue
        spans: list[tuple[int, int]] = []
        pos = 0
        if comment_from is not None:
            end = line.find("-->")
            if end < 0:
                markup.masked[i] = " " * len(line)
                continue
            spans.append((0, end + 3))
            pos = end + 3
            comment_from = None
        else:
            opened = FENCE_RE.match(line)
            if opened and not (opened.group(1)[0] == "`" and "`" in line[opened.end() :]):
                fence = (opened.group(1)[0], len(opened.group(1)), i)
                markup.masked[i] = " " * len(line)
                continue
        comment_from = _scan_line(markup, doc, i, line, pos, spans)
        markup.masked[i] = _blank(line, spans)
    if fence is not None:
        markup.problems.append((fence[2] + 1, "code fence opened here is never closed"))
    if comment_from is not None:
        markup.problems.append((comment_from + 1, "HTML comment opened here is never closed"))
    return markup


def _scan_line(
    markup: Markup, doc: str, i: int, line: str, pos: int, spans: list[tuple[int, int]]
) -> int | None:
    """Scan one line from ``pos`` for code spans and comments; return the line index of
    a comment left open at the end of the line, else None."""
    open_claim: tuple[str, int, int] | None = None
    while pos < len(line):
        tick = line.find("`", pos)
        comment = line.find("<!--", pos)
        starts = [p for p in (tick, comment) if p >= 0]
        if not starts:
            break
        at = min(starts)
        if at == tick:
            size = len(line[at:]) - len(line[at:].lstrip("`"))
            close = _closing_ticks(line, at + size, size)
            if close < 0:
                pos = at + size
                continue
            spans.append((at, close + size))
            pos = close + size
            continue
        end = line.find("-->", at + 4)
        if end < 0:
            spans.append((at, len(line)))
            if open_claim is not None:
                markup.problems.append((i + 1, f"claim {open_claim[0]} is not closed on its line"))
            return i
        body = line[at + 4 : end]
        spans.append((at, end + 3))
        opened, closed = OPEN_RE.match(body), CLOSE_RE.match(body)
        if opened:
            if open_claim is not None:
                markup.problems.append(
                    (i + 1, f"claim {opened.group('id')} opens before claim {open_claim[0]} closes")
                )
            open_claim = (opened.group("id"), at, end + 3)
        elif closed:
            if open_claim is None:
                markup.problems.append((i + 1, "closing claim marker without an opening marker"))
            else:
                value = line[open_claim[2] : at]
                markup.claims.append(ClaimUse(doc, i + 1, open_claim[0], value.strip()))
                markup.markers.extend([(i, open_claim[1], open_claim[2]), (i, at, end + 3)])
                spans.append((open_claim[2], at))
                open_claim = None
        elif open_claim is not None:
            markup.problems.append((i + 1, f"comment inside claim {open_claim[0]}"))
        elif CLAIMISH_RE.match(body):
            markup.problems.append((i + 1, f"malformed claim marker <!--{body}-->"))
        pos = end + 3
    if open_claim is not None:
        markup.problems.append((i + 1, f"claim {open_claim[0]} is not closed on its line"))
    return None


def strip_markers(text: str) -> str:
    """The document with every parsed claim marker removed (values stay in place)."""
    markup = scan_markup(text)
    lines = list(markup.lines)
    for index, start, end in sorted(markup.markers, reverse=True):
        lines[index] = lines[index][:start] + lines[index][end:]
    return "\n".join(lines)


def _valid_iso(match: re.Match[str]) -> bool:
    month, day = int(match.group("m")), int(match.group("d"))
    return 1 <= month <= 12 and 1 <= day <= 31


def _existing_path(candidate: str, root: Path) -> bool:
    """True when ``candidate`` names a file or directory inside the repository."""
    parts = PurePosixPath(candidate).parts
    if not parts or ".." in parts:
        return False
    return (root / candidate).exists()


def _tag_spans(masked: str) -> list[tuple[int, int]]:
    """HTML tags, except the values of text attributes (alt, title, aria-label), which
    readers see and which are therefore checked like any other text."""
    spans: list[tuple[int, int]] = []
    for tag in TAG_RE.finditer(masked):
        pos, end = tag.span()
        for attr in TEXT_ATTR_RE.finditer(masked, pos, end):
            value = attr.span("dq") if attr.group("dq") is not None else attr.span("sq")
            spans.append((pos, value[0]))
            pos = value[1]
        spans.append((pos, end))
    return spans


def _numeric_entity(text: str) -> bool:
    """True when an HTML entity renders as a numeral (``&#52;``, ``&#x34;``, ``&frac12;``)."""
    return any(ch.isnumeric() for ch in html.unescape(text))


def _structural_spans(masked: str, root: Path | None) -> list[tuple[int, int]]:
    """Spans of a masked line that are markup, addresses, dates, versions, or digests."""
    spans = _tag_spans(masked)
    for pattern in (ENTITY_RE, URL_RE, ARXIV_RE, DOI_RE, YEAR_MONTH_RE, VERSION_RE):
        spans.extend(m.span() for m in pattern.finditer(masked))
    spans.extend(m.span() for m in HEX_RE.finditer(masked))
    spans.extend(m.span("target") for m in LINK_TARGET_RE.finditer(masked))
    spans.extend(m.span("target") for m in LINK_DEF_RE.finditer(masked))
    spans.extend(m.span() for m in ISO_DATE_RE.finditer(masked) if _valid_iso(m))
    for pattern in MONTH_DATE_RES:
        spans.extend(m.span() for m in pattern.finditer(masked))
    for pattern in CITATION_YEAR_RES:
        spans.extend(m.span("year") for m in pattern.finditer(masked))
    if root is not None:
        for match in PATH_RE.finditer(masked):
            candidate = match.group().rstrip("/").rstrip(".")
            if candidate and _existing_path(candidate, root):
                spans.append(match.span())
    return spans


def _heading_exempt(clean: str, seen: set[str]) -> tuple[int, int] | None:
    """Span of an ATX heading's section number when it is exempt (see scan_numbers)."""
    heading = HEADING_NUM_RE.match(clean)
    if heading is None:
        return None
    number = heading.group("num")
    parts = number.split(".")
    if len(parts) == 1:
        exempt = heading.group("punct") in {".", ")"}
    else:
        exempt = ".".join(parts[:-1]) in seen
    if not exempt:
        return None
    seen.add(number)
    return heading.span("num")


def scan_numbers(doc: str, markup: Markup, root: Path | None = None) -> list[NumberHit]:
    """Digit sequences outside claims that no built-in rule exempts.

    Built-in exemptions (everything else must be a claim or allowlisted):

    * fenced code blocks, inline code, and HTML comments (blanked by :func:`scan_markup`);
    * HTML tags with their attributes (but not the values of ``alt``, ``title``, and
      ``aria-label``, which readers see), HTML entities that do not render as numerals,
      URLs, Markdown link targets and link definitions, arXiv identifiers
      (``arXiv:2307.03172``), and DOIs;
    * bare paths that exist in the repository (``root``);
    * ISO dates and times, ``YYYY-MM`` months, dates with month names, and years in
      citation form (``(2023)``, ``(Liu et al., 2023)``, ``et al. 2023``);
    * versions with three or more parts (``2.1.309``) and hex digests of 7 to 64
      characters with at least one letter and one digit;
    * digits attached to letters on their left within one word (``G7``, ``E1``, ``RQ3``,
      ``D-012``, ``SHA-256``, ``v0.5``, ``qwen2.5-0.5b``);
    * section numbers of headings: ``## 5. Experiments`` (one part needs a trailing ``.``
      or ``)``) and ``### 5.2 Title`` (more parts need the parent number, here ``5``, on
      an earlier heading of the same document); and ordered-list markers (``3.``);
    * citation and footnote markers ``[3]`` and ``[^3]`` (up to three digits, not link text);
    * the number right after a reference word (section, rule, figure, fig., table, step,
      appendix, equation, eq., chapter, rung, milestone, gate, line, version,
      preregistration, amendment, seed, Python, or the section sign). After a plural
      reference word (or Python, or two section signs), further numbers chained by
      ``and``, ``or``, ``to``, ``through``, a hyphen, an en dash, or ``, and`` are exempt
      too (``sections 6 to 10``, ``Python 3.11 and 3.12``); a number chained by a bare
      comma is exempt only when the list goes on or ends right after it
      (``rules 1, 9, and 12``), so ``In Figures 2 and 3, 87 sessions`` still flags 87.

    Numerals that are not ASCII digit runs are always flagged: an entity such as
    ``&#52;&#48;`` (which renders as 40) and characters such as fractions and
    superscripts. Numbers written as words are not digit sequences and are not seen.
    """
    hits: list[NumberHit] = []
    seen_headings: set[str] = set()
    for index, masked in enumerate(markup.masked):
        line = markup.lines[index]
        clean = _blank(masked, _structural_spans(masked, root))
        heading = _heading_exempt(clean, seen_headings)
        listed = LIST_NUM_RE.match(clean)
        chain_end: int | None = None
        for match in TOKEN_RE.finditer(clean):
            start, end = match.span()
            before = clean[:start]
            if heading is not None and (start, end) == heading:
                continue
            if listed and (start, end) == listed.span("num"):
                continue
            word = WORD_RUN_RE.search(before)
            if word is not None and re.search(r"[A-Za-z]", word.group()):
                continue
            if (
                before.endswith(("[", "[^"))
                and clean[end:].startswith("]")
                and not clean[end:].startswith("](")
                and match.group().isdigit()
                and len(match.group()) <= 3
            ):
                continue
            label = LABEL_RE.search(before)
            if label is not None:
                name = (label.group("word") or label.group("sign")).lower()
                chain_end = end if name in CHAINABLE else None
                continue
            if chain_end is not None and _chained(clean, chain_end, start, end):
                chain_end = end
                continue
            hits.append(NumberHit(doc, index + 1, match.group(), start, end, line))
        hits.extend(_hidden_numerals(doc, index + 1, masked, clean, line))
    return sorted(hits, key=lambda hit: (hit.line, hit.start))


def _hidden_numerals(doc: str, number: int, masked: str, clean: str, line: str) -> list[NumberHit]:
    """Numerals that are not ASCII digit runs: entities that render as numerals and
    characters such as ``\N{VULGAR FRACTION ONE HALF}`` or ``\N{SUPERSCRIPT TWO}``."""
    hits = [
        NumberHit(doc, number, m.group(), m.start(), m.end(), line)
        for m in ENTITY_RE.finditer(masked)
        if _numeric_entity(m.group())
    ]
    hits.extend(
        NumberHit(doc, number, ch, pos, pos + 1, line)
        for pos, ch in enumerate(clean)
        if ch.isnumeric() and not ch.isdecimal()
    )
    return hits


def _chained(clean: str, chain_end: int, start: int, end: int) -> bool:
    """True when the token at ``start:end`` continues a list of labelled numbers."""
    gap = clean[chain_end:start]
    if CONNECTOR_RE.fullmatch(gap) is None:
        return False
    if BARE_COMMA_RE.fullmatch(gap):
        return LIST_GOES_ON_RE.match(clean, end) is not None
    return True


def parse_document(doc: str, text: str, root: Path | None = None) -> ParsedDoc:
    """Claims, candidate unmarked numbers, and structural problems of one document."""
    markup = scan_markup(text, doc)
    return ParsedDoc(
        path=doc,
        claims=tuple(markup.claims),
        numbers=tuple(scan_numbers(doc, markup, root)),
        problems=tuple(markup.problems),
    )


# --------------------------------------------------------------------------- manifest


@dataclass(frozen=True)
class AllowEntry:
    """A design parameter or other non-result number exempted by the allowlist.

    The token ``value`` is exempt only where it lies inside an occurrence of the literal
    text ``context`` on its source line, and only with a ``reason``.
    """

    value: str
    context: str
    reason: str


@dataclass(frozen=True)
class ClaimSpec:
    """Where a claim's value comes from and how it is rendered.

    Attributes:
        claim_id: The id used in claim markers.
        artifact: Banked artifact path under ``artifacts/`` (``.json`` or ``.jsonl``).
        pointer: RFC 6901 JSON pointer into the artifact (``""`` is the whole document;
            a ``.jsonl`` artifact is the list of its rows).
        where: Row filter on the list at ``pointer``: every key must equal its value.
        field: Pointer applied to the single matching row (or to each row for ``sum``).
        aggregate: ``count`` (number of rows) or ``sum`` (of ``field`` over rows).
        format: Python format spec applied to the selected value (``.1f``, ``.0%``, ``d``).
    """

    claim_id: str
    artifact: str
    pointer: str = ""
    where: Mapping[str, Any] | None = None
    field: str = ""
    aggregate: str | None = None
    format: str = ""


@dataclass(frozen=True)
class Derivation:
    """How freshness re-derives an artifact from lower-level banked artifacts.

    Attributes:
        artifact: The derived artifact.
        derive: ``module:function`` in this repository, called as
            ``function(inputs, **params)`` with ``inputs`` mapping each input path to
            its loaded content; it returns the artifact's content.
        inputs: Input artifact paths (a trailing ``/`` names a directory of artifacts).
        params: Keyword arguments for the derivation.
        compare: Pointers compared between the re-derived and the banked content.
    """

    artifact: str
    derive: str
    inputs: tuple[str, ...]
    params: Mapping[str, Any]
    compare: tuple[str, ...] = ("",)


@dataclass(frozen=True)
class Manifest:
    """Parsed ``artifacts/claims.json``."""

    documents: tuple[str, ...] = ()
    claims: Mapping[str, ClaimSpec] = field(default_factory=dict)
    allowlist: tuple[AllowEntry, ...] = ()
    derivations: Mapping[str, Derivation] = field(default_factory=dict)
    raw: Mapping[str, str] = field(default_factory=dict)


def _pointer_ok(value: object) -> bool:
    return isinstance(value, str) and (value == "" or value.startswith("/"))


def _artifact_path(value: object, *, directory_ok: bool = False) -> str:
    """Validate a result-artifact path (or, with ``directory_ok``, a directory with ``/``).

    Raises:
        IntegrityError: for a malformed path, a path outside ``artifacts/``, a file type
            other than ``.json``/``.jsonl``, or one of the integrity tools' own files.
    """
    if directory_ok and isinstance(value, str) and value.endswith("/"):
        safe_relpath(value[:-1], ARTIFACTS_DIR)
        return value
    rel = safe_relpath(value, ARTIFACTS_DIR)
    if not rel.endswith(ARTIFACT_SUFFIXES):
        raise IntegrityError(f"artifact {rel} is not a .json or .jsonl file")
    if rel in NOT_RESULT_ARTIFACTS:
        raise IntegrityError(f"{rel} is bookkeeping for the integrity tools, not a result")
    return rel


def _parse_claim(claim_id: str, raw: Any, problems: list[str]) -> ClaimSpec | None:
    where = f"claim {claim_id!r}"
    if not re.fullmatch(CLAIM_ID, claim_id):
        problems.append(f"{where}: id must match {CLAIM_ID}")
        return None
    if not isinstance(raw, dict):
        problems.append(f"{where}: entry is not an object")
        return None
    unknown = sorted(set(raw) - CLAIM_KEYS)
    if unknown:
        problems.append(f"{where}: unknown key(s) {', '.join(unknown)}")
        return None
    try:
        artifact = _artifact_path(raw.get("artifact"))
    except IntegrityError as exc:
        problems.append(f"{where}: {exc}")
        return None
    pointer, field_ptr = raw.get("pointer", ""), raw.get("field", "")
    filters, aggregate, spec = raw.get("where"), raw.get("aggregate"), raw.get("format")
    bad: list[str] = []
    if not _pointer_ok(pointer) or not _pointer_ok(field_ptr):
        bad.append("pointer and field must be JSON pointers ('' or starting with '/')")
    if filters is not None and not (isinstance(filters, dict) and filters):
        bad.append("'where' must be a non-empty object")
    if aggregate is not None and aggregate not in AGGREGATES:
        bad.append(f"aggregate must be one of {', '.join(AGGREGATES)}")
    if aggregate == "count" and field_ptr:
        bad.append("aggregate 'count' takes no field")
    if not isinstance(spec, str):
        bad.append("'format' (a Python format spec) is required")
    if bad:
        problems.extend(f"{where}: {b}" for b in bad)
        return None
    return ClaimSpec(claim_id, artifact, pointer, filters, field_ptr, aggregate, str(spec))


def _parse_allow(index: int, raw: Any, problems: list[str]) -> AllowEntry | None:
    where = f"allowlist entry {index}"
    if not isinstance(raw, dict) or set(raw) != ALLOW_KEYS:
        problems.append(f"{where}: needs exactly the keys value, context, reason")
        return None
    value, context, reason = raw["value"], raw["context"], raw["reason"]
    if not (isinstance(value, str) and VALUE_RE.match(value)):
        problems.append(f"{where}: value {value!r} is not a number as written")
        return None
    if not (isinstance(reason, str) and reason.strip()):
        problems.append(f"{where}: every allowlisted number needs a reason")
        return None
    if not (
        isinstance(context, str)
        and value in TOKEN_RE.findall(context)
        and len(re.findall(r"[A-Za-z]", context)) >= 3
    ):
        problems.append(
            f"{where}: context must be the literal text around {value!r}, with at least one "
            "word, as it appears in the document"
        )
        return None
    return AllowEntry(value, context, reason.strip())


def _parse_derivation(artifact: str, raw: Any, problems: list[str]) -> Derivation | None:
    where = f"derivation {artifact}"
    try:
        artifact = _artifact_path(artifact)
    except IntegrityError as exc:
        problems.append(f"{where}: {exc}")
        return None
    if not isinstance(raw, dict) or set(raw) - DERIVATION_KEYS:
        problems.append(f"{where}: entry must be an object with keys {sorted(DERIVATION_KEYS)}")
        return None
    derive, inputs = raw.get("derive"), raw.get("inputs")
    params, compare = raw.get("params", {}), raw.get("compare", [""])
    bad: list[str] = []
    if not (isinstance(derive, str) and DERIVE_RE.match(derive)):
        bad.append("'derive' must be 'module:function'")
    if not (isinstance(inputs, list) and inputs):
        bad.append("'inputs' must list at least one lower-level artifact")
        inputs = []
    paths: list[str] = []
    for item in inputs:
        try:
            paths.append(_artifact_path(item, directory_ok=True))
        except IntegrityError as exc:
            bad.append(f"input {exc}")
    if not (isinstance(params, dict) and all(str(k).isidentifier() for k in params)):
        bad.append("'params' must be an object of keyword arguments")
    if not (isinstance(compare, list) and compare and all(_pointer_ok(p) for p in compare)):
        bad.append("'compare' must be a non-empty list of JSON pointers")
    if bad:
        problems.extend(f"{where}: {b}" for b in bad)
        return None
    return Derivation(artifact, str(derive), tuple(paths), dict(params), tuple(compare))


def parse_manifest(data: Any) -> tuple[Manifest, list[str]]:
    """Validate parsed ``artifacts/claims.json`` content; invalid entries are dropped."""
    problems: list[str] = []
    if not isinstance(data, dict):
        return Manifest(), [f"{MANIFEST} is not a JSON object"]
    unknown = sorted(set(data) - TOP_KEYS)
    if unknown:
        problems.append(f"unknown top-level key(s) {', '.join(unknown)}")
    documents: list[str] = []
    docs_raw = data.get("documents", [])
    if not isinstance(docs_raw, list):
        problems.append("'documents' must be a list of document paths")
        docs_raw = []
    for doc in docs_raw:
        try:
            rel = safe_relpath(doc)
            if not rel.endswith(DOC_SUFFIXES):
                raise IntegrityError(f"document {rel} is not .md or .txt")
            documents.append(rel)
        except IntegrityError as exc:
            problems.append(str(exc))
    claims_raw = data.get("claims", {})
    if not isinstance(claims_raw, dict):
        problems.append("'claims' must be an object of claim id to entry")
        claims_raw = {}
    claims = {
        cid: spec
        for cid, raw in claims_raw.items()
        if (spec := _parse_claim(str(cid), raw, problems)) is not None
    }
    allow_raw = data.get("allowlist", [])
    if not isinstance(allow_raw, list):
        problems.append("'allowlist' must be a list")
        allow_raw = []
    allowlist = [
        entry for i, raw in enumerate(allow_raw) if (entry := _parse_allow(i, raw, problems))
    ]
    derived_raw = data.get("derivations", {})
    if not isinstance(derived_raw, dict):
        problems.append("'derivations' must be an object of artifact path to entry")
        derived_raw = {}
    derivations = {
        d.artifact: d
        for path, raw in derived_raw.items()
        if (d := _parse_derivation(str(path), raw, problems)) is not None
    }
    raw_section = data.get("raw", {})
    raw: dict[str, str] = {}
    if not isinstance(raw_section, dict):
        problems.append("'raw' must be an object of artifact path to reason")
        raw_section = {}
    for path, reason in raw_section.items():
        try:
            rel = _artifact_path(path, directory_ok=True)
        except IntegrityError as exc:
            problems.append(f"raw {exc}")
            continue
        if not (isinstance(reason, str) and reason.strip()):
            problems.append(f"raw {rel}: needs a reason (why it is primary data)")
            continue
        raw[rel] = reason.strip()
    manifest = Manifest(tuple(dict.fromkeys(documents)), claims, tuple(allowlist), derivations, raw)
    return manifest, problems


def load_manifest(root: Path) -> tuple[Manifest, list[str]]:
    """Load and validate ``artifacts/claims.json`` under ``root``."""
    path = root / MANIFEST
    if not path.is_file():
        return Manifest(), [f"{MANIFEST} is missing (it maps claim ids to artifacts)"]
    try:
        data = load_json(path)
    except IntegrityError as exc:
        return Manifest(), [str(exc)]
    return parse_manifest(data)


# --------------------------------------------------------------------------- selection


def same_json(a: Any, b: Any) -> bool:
    """Equality of JSON values that keeps ``true`` apart from ``1`` and ``1`` from ``1.0``."""
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def resolve_pointer(document: Any, pointer: str) -> Any:
    """Resolve an RFC 6901 JSON pointer.

    Raises:
        SelectorError: for a malformed pointer, a missing key, or a bad array index.
    """
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise SelectorError(f"pointer {pointer!r} must be empty or start with '/'")
    node = document
    for raw in pointer[1:].split("/"):
        if re.search(r"~(?![01])", raw):
            raise SelectorError(f"pointer {pointer!r} has an invalid escape")
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict):
            if token not in node:
                raise SelectorError(f"pointer {pointer!r}: no key {token!r}")
            node = node[token]
        elif isinstance(node, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", token) or int(token) >= len(node):
                raise SelectorError(f"pointer {pointer!r}: no index {token!r} in a list")
            node = node[int(token)]
        else:
            raise SelectorError(f"pointer {pointer!r}: cannot index into {type(node).__name__}")
    return node


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def select(document: Any, spec: ClaimSpec) -> Any:
    """Apply a claim's selector (pointer, filter, aggregate, field) to artifact content.

    Raises:
        SelectorError: when any step of the selector does not resolve.
    """
    node = resolve_pointer(document, spec.pointer)
    if spec.where is None and spec.aggregate is None:
        return resolve_pointer(node, spec.field)
    if not isinstance(node, list):
        raise SelectorError(f"pointer {spec.pointer!r} does not select a list of rows")
    rows = node
    if spec.where is not None:
        where = spec.where
        rows = [
            row
            for row in node
            if isinstance(row, dict)
            and all(k in row and same_json(row[k], v) for k, v in where.items())
        ]
    if spec.aggregate == "count":
        return len(rows)
    if spec.aggregate == "sum":
        values = [resolve_pointer(row, spec.field) for row in rows]
        if not all(_is_number(v) for v in values):
            raise SelectorError("aggregate 'sum' needs numbers in every selected row")
        if all(isinstance(v, int) for v in values):
            return sum(values)
        return math.fsum(values)
    if len(rows) != 1:
        raise SelectorError(f"'where' matched {len(rows)} rows, expected exactly 1")
    return resolve_pointer(rows[0], spec.field)


def render_value(value: Any, spec: str) -> str:
    """Format a selected value with a Python format spec.

    Raises:
        SelectorError: for booleans, null, containers, or a spec that does not apply.
    """
    if isinstance(value, bool) or value is None or isinstance(value, dict | list):
        raise SelectorError(f"selected value {value!r} is not a number or string")
    try:
        return format(value, spec)
    except (TypeError, ValueError) as exc:
        raise SelectorError(f"format {spec!r} does not apply to {value!r}: {exc}") from exc


class Artifacts:
    """Loads banked artifacts once and remembers which ones are not committed yet."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.notes: set[str] = set()
        self._cache: dict[str, Any] = {}

    def load(self, rel: str) -> Any:
        """Content of the artifact at ``rel``.

        Raises:
            IntegrityError: when the path is invalid or the artifact is missing or unparsable.
        """
        if rel not in self._cache:
            _artifact_path(rel)
            path = self.root / rel
            if not path.is_file():
                raise IntegrityError(f"artifact {rel} does not exist")
            self._cache[rel] = load_artifact(path)
            if not in_head(self.root, rel):
                self.notes.add(f"{rel} is not committed, so it is not banked until it is")
            elif not worktree_matches_head(self.root, rel):
                self.notes.add(f"{rel} has uncommitted changes; the committed version is banked")
        return self._cache[rel]


# --------------------------------------------------------------------------- check


def document_set(root: Path, manifest: Manifest) -> list[str]:
    """Results documents: README.md and docs/REPORT.md, the manifest's ``documents``,
    every ``.md`` or ``.txt`` file under ``outbound/``, and any other Markdown file that
    contains a claim marker. docs/BUILD_BRIEF.md is never a results document."""
    docs: list[str] = [*REQUIRED_DOCS, *manifest.documents]
    files = worktree_files(root)
    docs.extend(
        rel for rel in files if rel.startswith(OUTBOUND_DIR + "/") and rel.endswith(DOC_SUFFIXES)
    )
    for rel in files:
        if rel.endswith(".md") and rel not in docs and rel not in NEVER_RESULTS:
            try:
                text = (root / rel).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if "claim:" in text and scan_markup(text, rel).claims:
                docs.append(rel)
    return [d for d in dict.fromkeys(docs) if d not in NEVER_RESULTS]


def _covered(entry: AllowEntry, hit: NumberHit) -> bool:
    """True when the hit lies inside an occurrence of the entry's context."""
    if entry.value != hit.token:
        return False
    start = hit.text.find(entry.context)
    while start >= 0:
        if start <= hit.start and hit.end <= start + len(entry.context):
            return True
        start = hit.text.find(entry.context, start + 1)
    return False


def apply_allowlist(
    hits: Sequence[NumberHit], allowlist: Sequence[AllowEntry]
) -> tuple[list[NumberHit], set[int]]:
    """Split hits into unmarked numbers and the indices of the allowlist entries used."""
    unmarked: list[NumberHit] = []
    used: set[int] = set()
    for hit in hits:
        matched = next((i for i, entry in enumerate(allowlist) if _covered(entry, hit)), None)
        if matched is None:
            unmarked.append(hit)
        else:
            used.add(matched)
    return unmarked, used


def check(root: Path, *, verbose: bool = False) -> CheckLog:
    """Run the claim check on the repository at ``root``; the last line is the verdict."""
    log = CheckLog()
    manifest, problems = load_manifest(root)
    for problem in problems:
        log.fail("manifest", problem)
    artifacts = Artifacts(root)
    used_claims: set[str] = set()
    used_allow: set[int] = set()
    claims_seen = docs_seen = 0
    for rel in document_set(root, manifest):
        path = root / rel
        if not path.is_file():
            log.fail(f"document {rel}", "missing")
            continue
        try:
            parsed = parse_document(rel, path.read_text(encoding="utf-8"), root)
        except UnicodeDecodeError as exc:
            log.fail(f"document {rel}", f"is not UTF-8 text: {exc}")
            continue
        docs_seen += 1
        claims_seen += len(parsed.claims)
        before = log.problems
        for line, message in parsed.problems:
            log.fail(f"{rel}:{line}", message)
        for use in parsed.claims:
            used_claims.add(use.claim_id)
            _check_claim(use, manifest, artifacts, log, verbose)
        unmarked, used = apply_allowlist(parsed.numbers, manifest.allowlist)
        used_allow |= used
        for hit in unmarked:
            log.fail(
                f"{rel}:{hit.line}",
                f"unmarked number {hit.token!r} (mark it as a claim or allowlist it with a "
                f"reason): {hit.text.strip()[:120]}",
            )
        if log.problems == before:
            allowed = len(parsed.numbers) - len(unmarked)
            log.ok(
                f"document {rel}",
                f"{len(parsed.claims)} claim(s) match their artifacts, "
                f"{allowed} allowlisted number(s), no unmarked numbers",
            )
    for note in sorted(artifacts.notes):
        log.note(note)
    unused = sorted(set(manifest.claims) - used_claims)
    if unused:
        log.note(f"{len(unused)} manifest claim(s) unused in documents: {', '.join(unused[:5])}")
    idle = [entry.value for i, entry in enumerate(manifest.allowlist) if i not in used_allow]
    if idle:
        log.note(f"{len(idle)} allowlist entr(ies) matched nothing: {', '.join(idle[:5])}")
    log.lines.append(log.verdict(TOOL, f"{claims_seen} claim(s) in {docs_seen} document(s)"))
    return log


def _check_claim(
    use: ClaimUse, manifest: Manifest, artifacts: Artifacts, log: CheckLog, verbose: bool
) -> None:
    where = f"{use.doc}:{use.line}"
    spec = manifest.claims.get(use.claim_id)
    if spec is None:
        log.fail(where, f"claim {use.claim_id!r} is not defined in {MANIFEST}")
        return
    try:
        expected = render_value(select(artifacts.load(spec.artifact), spec), spec.format)
    except IntegrityError as exc:
        log.fail(where, f"claim {use.claim_id}: {exc}")
        return
    if use.rendered != expected:
        log.fail(
            where,
            f"claim {use.claim_id} renders {use.rendered!r} but {spec.artifact} gives {expected!r}",
        )
    elif verbose:
        log.ok(f"claim {use.claim_id}", f"{expected} from {spec.artifact} ({where})")


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; prints the report and returns the exit status."""
    parser = argparse.ArgumentParser(description="Trace every result number to an artifact (G10).")
    parser.add_argument("--root", type=Path, help="any directory inside the repository")
    parser.add_argument("--verbose", action="store_true", help="print one line per claim")
    parser.add_argument(
        "--strip", type=Path, metavar="FILE", help="print FILE with its claim markers removed"
    )
    args = parser.parse_args(argv)
    if args.strip is not None:
        try:
            text = args.strip.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            print(f"{TOOL}: cannot read {args.strip}: {exc}", file=sys.stderr)
            return 2
        sys.stdout.write(strip_markers(text))
        return 0
    try:
        root = repo_root(args.root)
    except GitError as exc:
        print(f"{TOOL}: FAIL not inside a git work tree ({exc})")
        return 2
    try:
        log = check(root, verbose=args.verbose)
    except GitError as exc:
        print(f"{TOOL}: FAIL a git query failed, so the check cannot be completed ({exc})")
        return 1
    log.emit()
    return 1 if log.problems else 0


if __name__ == "__main__":
    sys.exit(main())

"""Citation check for gate G13 (``make citecheck``), offline.

Rule 11 of the build brief: every reference is opened and checked once, and what was
seen is banked in ``artifacts/citecheck.json`` by the ``citation-checker`` subagent
(schema in .claude/agents/citation-checker.md). This check never touches the network.
It fails when:

* the file is missing, is not valid JSON, or breaks the schema: ``checked_by`` must be
  ``citation-checker``; every reference needs a unique ``key``, an http(s)
  ``cited_url``, a ``status`` of ``verified``, ``mismatch``, or ``unverified``, and a
  string ``notes``; a verified reference also needs an http(s) ``final_url``, a
  ``retrieved`` date (``YYYY-MM-DD``, not in the future), a ``page_title``, a non-empty
  ``authors`` list, an integer ``year``, a 64-hex ``content_sha256``, and a
  ``hash_source`` of ``curl body`` or ``webfetch text``; no two references bank the
  same URL;
* a URL cited in README.md or docs/REPORT.md (outside code and HTML comments, and not a
  reserved host such as ``localhost`` or ``example.com``) is not banked, or is banked
  with a status other than ``verified``; a citation matches a reference by its
  ``cited_url`` or ``final_url`` after normalizing scheme and host case, default ports,
  a trailing slash, and the fragment;
* a paragraph, list item, or table row looks like an author-year citation (``et al``,
  ``Name (2023)``, or ``(Name, 2023)``) but carries no URL, so it could not have been
  verified.

References that are banked but not cited (for example ones dropped because they did
not open) are noted, not failed. Output ends with ``CITECHECK: PASS <n> verified``
(``n`` distinct references cited) or ``CITECHECK: FAIL <n> problem(s)``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
import urllib.parse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from halflife.integrity import CheckLog, GitError, IntegrityError, load_json, repo_root
from halflife.integrity.claimcheck import scan_markup

__all__ = [
    "Citation",
    "Reference",
    "check",
    "cited_urls",
    "is_reserved",
    "load_references",
    "main",
    "normalize_url",
    "reference_problems",
    "uncited_citations",
]

TOOL = "CITECHECK"
CITECHECK = "artifacts/citecheck.json"
CITED_DOCS = ("README.md", "docs/REPORT.md")
CHECKED_BY = "citation-checker"
STATUSES = ("verified", "mismatch", "unverified")
HASH_SOURCES = ("curl body", "webfetch text")
VERIFIED_FIELDS = (
    "key",
    "cited_url",
    "final_url",
    "retrieved",
    "page_title",
    "authors",
    "year",
    "content_sha256",
    "hash_source",
    "status",
    "notes",
)
URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"'`()\[\]]+")
TRAILING = ".,;:!?*_"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
CITATION_LIKE_RE = re.compile(
    r"\bet al\b"
    r"|\b[A-Z][\w'-]*\s+\((?:19|20)\d{2}[a-z]?\)"
    r"|\([A-Z][^()\n]*?,\s*(?:19|20)\d{2}[a-z]?\)"
)
LINK_DEF_RE = re.compile(r"^\s{0,3}\[([^\]]+)\]:\s*<?(\S+?)>?(?:\s.*)?$")
LINK_REF_RE = re.compile(r"\[([^\]]+)\](?:\[([^\]]*)\])?")
UNIT_START_RE = re.compile(r"^\s*(?:[-*+]\s|\d{1,9}[.)]\s|#{1,6}\s|\|)")
RESERVED_HOSTS = frozenset({"localhost", "127.0.0.1", "0.0.0.0", "::1"})
RESERVED_SUFFIXES = (".invalid", ".example", ".test", ".localhost")
RESERVED_DOMAINS = ("example.com", "example.net", "example.org")


@dataclass(frozen=True)
class Citation:
    """One URL cited in a document."""

    doc: str
    line: int
    url: str


@dataclass(frozen=True)
class Reference:
    """One banked reference from ``artifacts/citecheck.json``.

    Attributes:
        key: Short unique key.
        status: ``verified``, ``mismatch``, or ``unverified``.
        urls: Normalized ``cited_url`` and, when present, ``final_url``.
        retrieved: Retrieval date as banked.
        problems: Schema problems found in this entry.
    """

    key: str
    status: str
    urls: tuple[str, ...]
    retrieved: str
    problems: tuple[str, ...]


def normalize_url(url: str) -> str:
    """Canonical form for matching: lowercase scheme and host, no default port, no
    fragment, no trailing slash (except the root path). Unparsable URLs come back as is."""
    try:
        parts = urllib.parse.urlsplit(url.strip())
        port = parts.port
    except ValueError:
        return url.strip()
    scheme = parts.scheme.lower()
    host = (parts.hostname or "").lower()
    default = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    netloc = host if port is None or default else f"{host}:{port}"
    path = parts.path or "/"
    if len(path) > 1:
        path = path.rstrip("/") or "/"
    return urllib.parse.urlunsplit((scheme, netloc, path, parts.query, ""))


def is_reserved(url: str) -> bool:
    """True for loopback and documentation hosts, which are never references."""
    try:
        host = (urllib.parse.urlsplit(url).hostname or "").lower()
    except ValueError:
        return False
    return (
        host in RESERVED_HOSTS
        or host.startswith("127.")
        or host.endswith(RESERVED_SUFFIXES)
        or any(host == d or host.endswith("." + d) for d in RESERVED_DOMAINS)
    )


def _is_http_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parts = urllib.parse.urlsplit(value)
    except ValueError:
        return False
    return parts.scheme in {"http", "https"} and bool(parts.hostname)


def _date(value: Any) -> dt.date | None:
    match = DATE_RE.match(value) if isinstance(value, str) else None
    if match is None:
        return None
    try:
        return dt.date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def reference_problems(entry: Mapping[str, Any], today: dt.date) -> list[str]:
    """Schema problems in one reference entry (verified entries need every field)."""
    problems: list[str] = []
    status = entry.get("status")
    if status not in STATUSES:
        problems.append(f"status {status!r} is not one of {', '.join(STATUSES)}")
    if not (isinstance(entry.get("key"), str) and entry["key"].strip()):
        problems.append("key is missing or empty")
    if not _is_http_url(entry.get("cited_url")):
        problems.append("cited_url is not an http(s) URL")
    if not isinstance(entry.get("notes"), str):
        problems.append("notes must be a string (empty when there is nothing to note)")
    if status != "verified":
        return problems
    absent = [name for name in VERIFIED_FIELDS if name not in entry]
    if absent:
        problems.append(f"verified but missing {', '.join(absent)}")
    if "final_url" in entry and not _is_http_url(entry["final_url"]):
        problems.append("final_url is not an http(s) URL")
    retrieved = _date(entry.get("retrieved"))
    if "retrieved" in entry and retrieved is None:
        problems.append("retrieved is not a YYYY-MM-DD date")
    elif retrieved is not None and retrieved > today + dt.timedelta(days=1):
        problems.append(f"retrieved {retrieved.isoformat()} is in the future")
    if "page_title" in entry and not (
        isinstance(entry["page_title"], str) and entry["page_title"].strip()
    ):
        problems.append("page_title is empty")
    authors = entry.get("authors")
    if "authors" in entry and not (
        isinstance(authors, list)
        and authors
        and all(isinstance(a, str) and a.strip() for a in authors)
    ):
        problems.append("authors must be a non-empty list of names")
    year = entry.get("year")
    if "year" in entry and not (
        isinstance(year, int) and not isinstance(year, bool) and 1900 <= year <= today.year + 1
    ):
        problems.append(f"year {year!r} is not a plausible year")
    if "content_sha256" in entry and not (
        isinstance(entry["content_sha256"], str) and SHA256_RE.match(entry["content_sha256"])
    ):
        problems.append("content_sha256 is not 64 lowercase hex characters")
    if "hash_source" in entry and entry["hash_source"] not in HASH_SOURCES:
        problems.append(f"hash_source must be one of {', '.join(HASH_SOURCES)}")
    return problems


def load_references(root: Path, today: dt.date) -> tuple[list[Reference], list[str]]:
    """Load ``artifacts/citecheck.json``; return its references and file-level problems."""
    path = root / CITECHECK
    if not path.is_file():
        return [], [f"{CITECHECK} is missing (the citation-checker subagent banks it)"]
    try:
        data = load_json(path)
    except IntegrityError as exc:
        return [], [f"{CITECHECK}: {exc}"]
    if not isinstance(data, dict):
        return [], [f"{CITECHECK} is not a JSON object"]
    problems: list[str] = []
    if data.get("checked_by") != CHECKED_BY:
        problems.append(f"checked_by must be {CHECKED_BY!r}")
    entries = data.get("references")
    if not isinstance(entries, list):
        return [], [*problems, "references must be a list"]
    references: list[Reference] = []
    seen_keys: set[str] = set()
    seen_urls: dict[str, str] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            problems.append(f"reference {index} is not an object")
            continue
        found = reference_problems(entry, today)
        key = str(entry.get("key") or f"#{index}")
        if key in seen_keys:
            found.append("duplicate key")
        seen_keys.add(key)
        urls = tuple(
            dict.fromkeys(
                normalize_url(entry[name])
                for name in ("cited_url", "final_url")
                if isinstance(entry.get(name), str)
            )
        )
        shared = sorted({seen_urls[url] for url in urls if url in seen_urls})
        if shared:
            found.append(f"banks the same URL as reference {', '.join(shared)}")
        for url in urls:
            seen_urls.setdefault(url, key)
        retrieved = str(entry.get("retrieved", ""))
        references.append(Reference(key, str(entry.get("status")), urls, retrieved, tuple(found)))
    return references, problems


def _link_definitions(lines: Sequence[str]) -> dict[str, str]:
    defs: dict[str, str] = {}
    for line in lines:
        match = LINK_DEF_RE.match(line)
        if match and URL_RE.match(match.group(2)):
            defs[match.group(1).strip().lower()] = match.group(2)
    return defs


def cited_urls(doc: str, text: str) -> list[Citation]:
    """URLs cited in a document outside code and comments, minus reserved hosts.

    A bare ``www.`` address (which GitHub renders as a link) counts as a citation too.
    """
    markup = scan_markup(text, doc)
    found: list[Citation] = []
    for index, line in enumerate(markup.masked):
        for match in URL_RE.finditer(line):
            url = match.group().rstrip(TRAILING)
            if not is_reserved(_with_scheme(url)):
                found.append(Citation(doc, index + 1, url))
    return found


def _with_scheme(url: str) -> str:
    return url if url.startswith(("http://", "https://")) else f"https://{url}"


def _lookup(url: str, by_url: Mapping[str, Reference]) -> Reference | None:
    """The banked reference for a cited URL; a bare ``www.`` address matches either scheme."""
    candidates = (
        [url] if url.startswith(("http://", "https://")) else [f"https://{url}", f"http://{url}"]
    )
    for candidate in candidates:
        found = by_url.get(normalize_url(candidate))
        if found is not None:
            return found
    return None


def _units(lines: Sequence[str]) -> list[tuple[int, str]]:
    """Paragraphs, list items, headings, and table rows as (first line number, text)."""
    units: list[tuple[int, list[str]]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            units.append((index + 1, []))
            continue
        if UNIT_START_RE.match(line) or not units or not units[-1][1]:
            units.append((index + 1, [line]))
        else:
            units[-1][1].append(line)
    return [(start, " ".join(parts)) for start, parts in units if parts]


def uncited_citations(doc: str, text: str) -> list[tuple[int, str]]:
    """Units that look like author-year citations but carry no URL (see the module doc)."""
    markup = scan_markup(text, doc)
    defs = _link_definitions(markup.masked)
    flagged: list[tuple[int, str]] = []
    for start, unit in _units(markup.masked):
        if not CITATION_LIKE_RE.search(unit) or URL_RE.search(unit):
            continue
        labels = {(m.group(2) or m.group(1)).strip().lower() for m in LINK_REF_RE.finditer(unit)}
        if labels & set(defs):
            continue
        flagged.append((start, " ".join(unit.split())[:100]))
    return flagged


def check(root: Path, today: dt.date | None = None) -> CheckLog:
    """Run the citation check on the repository at ``root``; the last line is the verdict."""
    log = CheckLog()
    today = today or dt.date.today()
    references, problems = load_references(root, today)
    for problem in problems:
        log.fail("schema", problem)
    for ref in references:
        for problem in ref.problems:
            log.fail(f"reference {ref.key}", problem)
    if not problems:
        counts = {s: sum(r.status == s for r in references) for s in STATUSES}
        summary = ", ".join(f"{n} {s}" for s, n in counts.items() if n)
        log.ok("schema", f"{CITECHECK} has {len(references)} reference(s) ({summary or 'none'})")
    by_url = {url: ref for ref in references for url in ref.urls}
    cited: dict[str, list[Citation]] = {}
    for rel in CITED_DOCS:
        path = root / rel
        if not path.is_file():
            log.fail(f"document {rel}", "missing")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            log.fail(f"document {rel}", f"cannot be read as UTF-8 text: {exc}")
            continue
        for line, unit in uncited_citations(rel, text):
            log.fail(f"{rel}:{line}", f"looks like a citation but has no URL to verify: {unit}")
        for citation in cited_urls(rel, text):
            banked = _lookup(citation.url, by_url)
            if banked is None:
                log.fail(f"{rel}:{citation.line}", f"{citation.url} is not banked in {CITECHECK}")
            elif banked.status != "verified":
                log.fail(f"{rel}:{citation.line}", f"{citation.url} is banked as {banked.status}")
            else:
                cited.setdefault(banked.key, []).append(citation)
    good = 0
    for ref in references:
        uses = cited.get(ref.key)
        if uses is None:
            if ref.status != "verified":
                log.note(f"reference {ref.key} is {ref.status} and not cited (dropped)")
            continue
        if not ref.problems:
            good += 1
            places = ", ".join(f"{c.doc}:{c.line}" for c in uses[:3])
            log.ok(
                f"reference {ref.key}", f"verified, retrieved {ref.retrieved}, cited at {places}"
            )
    if not cited and not log.problems:
        log.note("no URLs cited in README.md or docs/REPORT.md")
    log.lines.append(log.verdict(TOOL, f"{good} verified"))
    return log


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; prints the report and returns the exit status."""
    parser = argparse.ArgumentParser(description="Check banked citations offline (gate G13).")
    parser.add_argument("--root", type=Path, help="any directory inside the repository")
    args = parser.parse_args(argv)
    try:
        root = repo_root(args.root)
    except GitError as exc:
        print(f"{TOOL}: FAIL not inside a git work tree ({exc})")
        return 2
    log = check(root)
    log.emit()
    return 1 if log.problems else 0


if __name__ == "__main__":
    sys.exit(main())

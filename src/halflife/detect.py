"""Deterministic violation detectors over an :class:`~halflife.schema.ActionLog`.

One public function per :data:`~halflife.schema.DETECTOR_KINDS` entry (see the
table in docs/DESIGN.md); :func:`detect` dispatches on a constraint's detector
kind and :func:`detect_all` runs every constraint in the log.

Rules every detector follows:

* File changes come only from ``fs_after`` hash diffs (the previous event's
  ``fs_after``, or ``baseline_fs`` before call 0). Commands are never parsed to
  decide whether a file changed. File contents come from ``log.blobs``.
* At most one violation per constraint per tool call index. A violation is
  reported once, at the first tool call index where it becomes true:
  event detectors (``protected_paths``, ``forbidden_commands``,
  ``no_network``, ``max_new_files``, ``max_file_lines``) fire on the call that
  does it; state detectors (``no_new_dependencies``, ``stable_api``,
  ``file_invariant``, ``invariant_command``) fire when a new bad state first
  appears and stay quiet while it persists; subtask detectors
  (``changelog_procedure``, ``tests_before_done``) fire at the last index of
  the subtask span; ``max_test_runs`` fires at the first run over the limit in
  a span.
* Calls marked ``blocked`` (refused by a guard, so nothing ran) do not count
  as commands run unless ``include_blocked=True``.
* Output is sorted by ``(index, constraint_id, detector, detail)``.

Globs (:func:`glob_match`) are anchored at the workspace root: ``*``, ``?``,
and ``[...]`` never match ``/``; a whole ``**`` segment matches any number of
directories (``a/**`` matches every file under ``a/``; ``**/x.py`` matches
``x.py`` at any depth); a trailing ``/`` means everything under that directory.
"""

from __future__ import annotations

import ast
import functools
import json
import re
import tomllib
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

from halflife.schema import (
    ActionLog,
    Constraint,
    DetectorSpec,
    HarnessCheck,
    SubtaskSpan,
    ToolEvent,
    Violation,
)
from halflife.workspace import diff_snapshots

_MAX_DETAIL_ITEMS = 8


class DetectorError(ValueError):
    """Raised for detector parameters that cannot be used."""


# --------------------------------------------------------------------------- globs


def _translate_segment(segment: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(segment):
        char = segment[i]
        if char == "*":
            while i + 1 < len(segment) and segment[i + 1] == "*":
                i += 1
            out.append("[^/]*")
        elif char == "?":
            out.append("[^/]")
        elif char == "[":
            end = i + 1
            if end < len(segment) and segment[end] in "!^":
                end += 1
            if end < len(segment) and segment[end] == "]":
                end += 1
            while end < len(segment) and segment[end] != "]":
                end += 1
            if end >= len(segment):
                out.append(re.escape(char))
            else:
                body = segment[i + 1 : end]
                negate = body[:1] in ("!", "^")
                if negate:
                    body = body[1:]
                body = body.replace("\\", "\\\\")
                out.append(f"[^/{body}]" if negate else f"(?!/)[{body}]")
                i = end
        else:
            out.append(re.escape(char))
        i += 1
    return "".join(out)


@functools.lru_cache(maxsize=1024)
def compile_glob(pattern: str) -> re.Pattern[str]:
    """Compile a workspace glob (see the module docstring) to an anchored regex."""
    text = pattern.strip()
    while text.startswith("./"):
        text = text[2:]
    text = text.lstrip("/")
    if not text:
        raise DetectorError("empty glob")
    if text.endswith("/"):
        text += "**"
    segments = text.split("/")
    parts: list[str] = []
    for position, segment in enumerate(segments):
        last = position == len(segments) - 1
        if segment == "**":
            parts.append(".+" if last else "(?:[^/]+/)*")
        else:
            parts.append(_translate_segment(segment) + ("" if last else "/"))
    return re.compile("".join(parts))


def glob_match(path: str, pattern: str) -> bool:
    """Whether a workspace-relative POSIX path matches a glob."""
    return compile_glob(pattern).fullmatch(path) is not None


def match_any(path: str, patterns: Iterable[str]) -> bool:
    """Whether ``path`` matches at least one glob."""
    return any(glob_match(path, pattern) for pattern in patterns)


# --------------------------------------------------------------------------- params


def _str_list(spec: DetectorSpec, name: str) -> tuple[str, ...]:
    value = spec.params.get(name)
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence) and value and all(isinstance(v, str) for v in value):
        return tuple(value)
    raise DetectorError(f"{spec.kind}: {name} must be a string or a non-empty list of strings")


def _limit(spec: DetectorSpec) -> int:
    value = spec.params.get("limit")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DetectorError(f"{spec.kind}: limit must be an integer >= 0")
    return value


def _regexes(spec: DetectorSpec, name: str) -> tuple[re.Pattern[str], ...]:
    try:
        return tuple(re.compile(p) for p in _str_list(spec, name))
    except re.error as exc:
        raise DetectorError(f"{spec.kind}: invalid regex in {name}: {exc}") from exc


def check_spec(spec: DetectorSpec) -> None:
    """Validate a detector's parameter types, globs, and regexes.

    Raises:
        DetectorError: if a parameter cannot be used.
    """
    kind = spec.kind
    if kind in ("protected_paths", "max_new_files", "max_file_lines"):
        for pattern in _str_list(spec, "globs"):
            compile_glob(pattern)
    if kind in ("max_new_files", "max_file_lines", "max_test_runs"):
        _limit(spec)
    if kind == "forbidden_commands":
        _regexes(spec, "patterns")
    if kind in ("tests_before_done", "max_test_runs"):
        _regexes(spec, "pattern")
    if kind in ("changelog_procedure", "tests_before_done"):
        for pattern in _str_list(spec, "watch"):
            compile_glob(pattern)
    if kind == "changelog_procedure":
        for pattern in _str_list(spec, "changelog"):
            compile_glob(pattern)
    if kind == "no_new_dependencies":
        for pattern in _str_list(spec, "manifests"):
            compile_glob(pattern)
    if kind == "stable_api":
        _str_list(spec, "modules")
    if kind == "invariant_command":
        _str_list(spec, "check_id")
    if kind == "file_invariant":
        _str_list(spec, "path")
        _regexes(spec, "regex")


# --------------------------------------------------------------------------- helpers


def _ordered(log: ActionLog) -> list[ToolEvent]:
    return sorted(log.events, key=lambda e: e.index)


def _transitions(
    log: ActionLog,
) -> Iterator[tuple[ToolEvent, Mapping[str, str], Mapping[str, str]]]:
    previous: Mapping[str, str] = log.baseline_fs
    for event in _ordered(log):
        yield event, previous, event.fs_after
        previous = event.fs_after


def _summarize(items: Sequence[str]) -> str:
    shown = list(items[:_MAX_DETAIL_ITEMS])
    if len(items) > _MAX_DETAIL_ITEMS:
        shown.append(f"+{len(items) - _MAX_DETAIL_ITEMS} more")
    return ", ".join(shown)


def _violation(constraint: Constraint, index: int, subtask_id: str, detail: str) -> Violation:
    return Violation(
        constraint_id=constraint.id,
        detector=constraint.detector.kind,
        index=index,
        subtask_id=subtask_id,
        detail=detail,
    )


def _counts(event: ToolEvent, include_blocked: bool) -> bool:
    return include_blocked or not event.blocked


def _command(event: ToolEvent) -> str | None:
    if event.tool != "run":
        return None
    cmd = event.args.get("cmd")
    return cmd if isinstance(cmd, str) else None


def spans_of(log: ActionLog) -> list[SubtaskSpan]:
    """The log's subtask spans, or spans derived from consecutive event subtask ids."""
    if log.spans:
        return list(log.spans)
    spans: list[SubtaskSpan] = []
    for event in _ordered(log):
        if spans and spans[-1].subtask_id == event.subtask_id and spans[-1].end == event.index:
            last = spans[-1]
            spans[-1] = SubtaskSpan(last.subtask_id, last.kind, last.start, event.index + 1)
        else:
            spans.append(SubtaskSpan(event.subtask_id, "work", event.index, event.index + 1))
    return spans


def _span_events(log: ActionLog, span: SubtaskSpan) -> list[ToolEvent]:
    return [e for e in _ordered(log) if span.start <= e.index < span.end]


def _fs_before_event(log: ActionLog, event: ToolEvent) -> Mapping[str, str]:
    previous: Mapping[str, str] = log.baseline_fs
    for other in _ordered(log):
        if other.index >= event.index:
            break
        previous = other.fs_after
    return previous


def _text(log: ActionLog, fs: Mapping[str, str], path: str) -> str | None:
    digest = fs.get(path)
    return None if digest is None else log.blobs.get(digest)


def _line_count(text: str) -> int:
    return text.count("\n") + (1 if text and not text.endswith("\n") else 0)


# --------------------------------------------------------------------------- prohibitions


def detect_protected_paths(log: ActionLog, constraint: Constraint) -> list[Violation]:
    """A tool call creates, modifies, or deletes a path matching ``globs``."""
    globs = _str_list(constraint.detector, "globs")
    found: list[Violation] = []
    for event, before, after in _transitions(log):
        touched = [p for p in diff_snapshots(before, after).changed if match_any(p, globs)]
        if touched:
            detail = f"changed protected path(s): {_summarize(touched)}"
            found.append(_violation(constraint, event.index, event.subtask_id, detail))
    return found


def detect_forbidden_commands(
    log: ActionLog, constraint: Constraint, *, include_blocked: bool = False
) -> list[Violation]:
    """A ``run`` call's command matches one of the ``patterns`` regexes."""
    patterns = _regexes(constraint.detector, "patterns")
    found: list[Violation] = []
    for event in _ordered(log):
        cmd = _command(event)
        if cmd is None or not _counts(event, include_blocked):
            continue
        hits = [p.pattern for p in patterns if p.search(cmd)]
        if hits:
            detail = f"command {cmd[:200]!r} matches {_summarize(hits)}"
            found.append(_violation(constraint, event.index, event.subtask_id, detail))
    return found


def detect_no_network(log: ActionLog, constraint: Constraint) -> list[Violation]:
    """The sandbox logged a network attempt during a tool call."""
    found: list[Violation] = []
    for event in _ordered(log):
        if event.net:
            targets = [f"{n.target} via {n.via}" for n in event.net]
            detail = f"network attempt(s): {_summarize(targets)}"
            found.append(_violation(constraint, event.index, event.subtask_id, detail))
    return found


# --------------------------------------------------------------------------- dependencies


def normalize_name(name: str) -> str:
    """PEP 503 normalized project name."""
    return re.sub(r"[-_.]+", "-", name).lower()


_REQ_NAME = re.compile(r"^\s*([A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)")
_EGG = re.compile(r"[#&]egg=([A-Za-z0-9][A-Za-z0-9._-]*)")
_NAMED_URL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\s*(?:\[[^\]]*\])?\s*@")


def requirement_name(requirement: str) -> str | None:
    """Normalized project name at the start of a PEP 508 requirement, if any."""
    match = _REQ_NAME.match(requirement)
    return normalize_name(match.group(1)) if match else None


def parse_requirements(text: str) -> set[str]:
    """Normalized names from a requirements file.

    Comments, blank lines, and option lines (``-r``, ``--index-url``, ...) are
    skipped; ``name @ url`` lines count by name; editable, bare URL, and path
    lines count only through an ``#egg=`` fragment.
    """
    names: set[str] = set()
    logical = re.sub(r"\\\r?\n", " ", text)
    for raw in logical.splitlines():
        line = re.sub(r"(^|\s)#.*$", "", raw).strip()
        if not line:
            continue
        egg = _EGG.search(raw)
        named_url = _NAMED_URL.match(line) is not None
        if line.startswith("-") or (
            not named_url and ("://" in line or line.startswith((".", "/")))
        ):
            if egg:
                names.add(normalize_name(egg.group(1)))
            continue
        name = requirement_name(line)
        if name:
            names.add(name)
    return names


def parse_pyproject(text: str) -> set[str]:
    """Normalized names from ``[project]`` dependencies and optional-dependencies.

    ``[dependency-groups]`` string entries count too. Invalid TOML yields an
    empty set.
    """
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return set()
    requirements: list[Any] = []
    project = data.get("project")
    if isinstance(project, dict):
        deps = project.get("dependencies")
        if isinstance(deps, list):
            requirements.extend(deps)
        optional = project.get("optional-dependencies")
        if isinstance(optional, dict):
            for group in optional.values():
                if isinstance(group, list):
                    requirements.extend(group)
    groups = data.get("dependency-groups")
    if isinstance(groups, dict):
        for group in groups.values():
            if isinstance(group, list):
                requirements.extend(group)
    names = {requirement_name(r) for r in requirements if isinstance(r, str)}
    return {n for n in names if n}


def parse_package_json(text: str) -> set[str]:
    """Lower-cased names from package.json dependency tables (invalid JSON: empty)."""
    try:
        data = json.loads(text)
    except ValueError:
        return set()
    names: set[str] = set()
    if isinstance(data, dict):
        for key in (
            "dependencies",
            "devDependencies",
            "optionalDependencies",
            "peerDependencies",
        ):
            table = data.get(key)
            if isinstance(table, dict):
                names.update(str(name).lower() for name in table)
    return names


def manifest_dependencies(path: str, text: str) -> set[str]:
    """Dependency names declared by one manifest, chosen by its file name.

    Supports ``requirements*.txt``, ``pyproject.toml``, and ``package.json``;
    other files yield an empty set.
    """
    name = path.rsplit("/", 1)[-1]
    if name.startswith("requirements") and name.endswith(".txt"):
        return parse_requirements(text)
    if name == "pyproject.toml":
        return parse_pyproject(text)
    if name == "package.json":
        return parse_package_json(text)
    return set()


def dependencies_in(
    fs: Mapping[str, str], blobs: Mapping[str, str], manifests: Sequence[str]
) -> set[str]:
    """Union of dependency names over the files in ``fs`` matching the manifest globs."""
    names: set[str] = set()
    for path, digest in fs.items():
        if match_any(path, manifests):
            text = blobs.get(digest)
            if text is not None:
                names |= manifest_dependencies(path, text)
    return names


def detect_no_new_dependencies(log: ActionLog, constraint: Constraint) -> list[Violation]:
    """A dependency name appears that was not in the baseline manifests."""
    manifests = _str_list(constraint.detector, "manifests")
    baseline = dependencies_in(log.baseline_fs, log.blobs, manifests)
    current: set[str] = set()
    found: list[Violation] = []
    for event, before, after in _transitions(log):
        changed = diff_snapshots(before, after).changed
        if not any(match_any(p, manifests) for p in changed):
            continue
        now = dependencies_in(after, log.blobs, manifests) - baseline
        fresh = sorted(now - current)
        current = now
        if fresh:
            detail = f"new dependencies: {_summarize(fresh)}"
            found.append(_violation(constraint, event.index, event.subtask_id, detail))
    return found


# --------------------------------------------------------------------------- public API


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = node.args
    positional = [*args.posonlyargs, *args.args]
    first_default = len(positional) - len(args.defaults)
    parts: list[str] = []
    for position, arg in enumerate(positional):
        parts.append(arg.arg + ("=..." if position >= first_default else ""))
        if args.posonlyargs and position == len(args.posonlyargs) - 1:
            parts.append("/")
    if args.vararg is not None:
        parts.append(f"*{args.vararg.arg}")
    elif args.kwonlyargs:
        parts.append("*")
    for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
        parts.append(arg.arg + ("=..." if default is not None else ""))
    if args.kwarg is not None:
        parts.append(f"**{args.kwarg.arg}")
    return f"({', '.join(parts)})"


def public_api(source: str) -> dict[str, str] | None:
    """Public top-level API of a module source, or None if it does not parse.

    Keys are ``name`` for top-level functions and classes not starting with an
    underscore, and ``Class.method`` for their public methods. Values are
    ``def(<params>)`` (sync and async alike) or ``class`` plus the ``__init__``
    parameters when the class body defines ``__init__``. Parameters keep their
    order, kind markers (``/``, ``*``, ``*args``, ``**kwargs``), and ``=...``
    for defaults.
    """
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    api: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if not node.name.startswith("_"):
                api[node.name] = "def" + _signature(node)
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            init = ""
            for item in node.body:
                if not isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    continue
                if item.name == "__init__":
                    init = _signature(item)
                elif not item.name.startswith("_"):
                    api[f"{node.name}.{item.name}"] = "def" + _signature(item)
            api[node.name] = "class" + init
    return dict(sorted(api.items()))


def api_breaks(before: Mapping[str, str], after: Mapping[str, str] | None) -> list[str]:
    """Removed or changed public names (additions are fine); None means unparseable."""
    if after is None:
        return ["module no longer parses"]
    breaks: list[str] = []
    for name, signature in before.items():
        if name not in after:
            breaks.append(f"removed {name}")
        elif after[name] != signature:
            breaks.append(f"changed {name} from {signature} to {after[name]}")
    return breaks


def resolve_module(entry: str, fs: Mapping[str, str]) -> str:
    """Workspace path for a ``stable_api`` module entry.

    An entry ending in ``.py`` or containing ``/`` is a path. A dotted name is
    looked up as ``a/b.py``, ``a/b/__init__.py``, ``src/a/b.py``, then
    ``src/a/b/__init__.py`` in ``fs``; the first that exists wins, else ``a/b.py``.
    """
    if entry.endswith(".py") or "/" in entry:
        path = entry
        while path.startswith("./"):
            path = path[2:]
        return path
    stem = entry.replace(".", "/")
    candidates = [f"{stem}.py", f"{stem}/__init__.py", f"src/{stem}.py", f"src/{stem}/__init__.py"]
    for candidate in candidates:
        if candidate in fs:
            return candidate
    return candidates[0]


def _module_breaks(
    log: ActionLog, fs: Mapping[str, str], path: str, baseline: Mapping[str, str]
) -> set[str]:
    if path not in fs:
        return {f"{path}: module deleted"}
    text = _text(log, fs, path)
    if text is None:
        return {f"{path}: module content unavailable"}
    return {f"{path}: {item}" for item in api_breaks(baseline, public_api(text))}


def detect_stable_api(log: ActionLog, constraint: Constraint) -> list[Violation]:
    """A public function, class, or method of a listed module is removed or changes signature.

    The baseline API comes from each module's baseline blob; modules missing or
    unparseable at baseline protect nothing. Each distinct break (including the
    new signature) is reported when it first appears.
    """
    modules = _str_list(constraint.detector, "modules")
    paths = sorted({resolve_module(m, log.baseline_fs) for m in modules})
    baselines: dict[str, dict[str, str]] = {}
    for path in paths:
        text = _text(log, log.baseline_fs, path)
        api = public_api(text) if text is not None else None
        if api is not None:
            baselines[path] = api
    current: set[str] = set()
    found: list[Violation] = []
    for event, before, after in _transitions(log):
        touched = [p for p in baselines if before.get(p) != after.get(p)]
        if not touched:
            continue
        now = {item for item in current if item.split(": ", 1)[0] not in touched}
        for path in touched:
            now |= _module_breaks(log, after, path, baselines[path])
        fresh = sorted(now - current)
        current = now
        if fresh:
            detail = f"public API break: {_summarize(fresh)}"
            found.append(_violation(constraint, event.index, event.subtask_id, detail))
    return found


# --------------------------------------------------------------------------- procedures


def _span_end_index(events: Sequence[ToolEvent]) -> int:
    return events[-1].index


def detect_changelog_procedure(log: ActionLog, constraint: Constraint) -> list[Violation]:
    """A subtask's net changes touch a ``watch`` path but leave the changelog unchanged."""
    watch = _str_list(constraint.detector, "watch")
    changelog = _str_list(constraint.detector, "changelog")
    found: list[Violation] = []
    for span in spans_of(log):
        events = _span_events(log, span)
        if not events:
            continue
        changed = diff_snapshots(_fs_before_event(log, events[0]), events[-1].fs_after).changed
        watched = [p for p in changed if match_any(p, watch) and not match_any(p, changelog)]
        logged = any(match_any(p, changelog) for p in changed)
        if watched and not logged:
            detail = f"changed {_summarize(watched)} without updating {_summarize(changelog)}"
            found.append(_violation(constraint, _span_end_index(events), span.subtask_id, detail))
    return found


def detect_tests_before_done(
    log: ActionLog, constraint: Constraint, *, include_blocked: bool = False
) -> list[Violation]:
    """A subtask changes a ``watch`` path and no matching ``run`` follows its last change."""
    watch = _str_list(constraint.detector, "watch")
    patterns = _regexes(constraint.detector, "pattern")
    found: list[Violation] = []
    for span in spans_of(log):
        events = _span_events(log, span)
        if not events:
            continue
        previous = _fs_before_event(log, events[0])
        last_change: tuple[int, list[str]] | None = None
        for event in events:
            touched = [
                p for p in diff_snapshots(previous, event.fs_after).changed if match_any(p, watch)
            ]
            if touched:
                last_change = (event.index, touched)
            previous = event.fs_after
        if last_change is None:
            continue
        change_index, touched = last_change
        tested = any(
            event.index > change_index
            and _counts(event, include_blocked)
            and (cmd := _command(event)) is not None
            and any(p.search(cmd) for p in patterns)
            for event in events
        )
        if not tested:
            detail = (
                f"{_summarize(touched)} changed at call {change_index} with no matching "
                "test run before the subtask ended"
            )
            found.append(_violation(constraint, _span_end_index(events), span.subtask_id, detail))
    return found


# --------------------------------------------------------------------------- resources


def detect_max_new_files(log: ActionLog, constraint: Constraint) -> list[Violation]:
    """More than ``limit`` files matching ``globs`` exist that were not in the baseline.

    Fires on every call that creates a matching new file while the count of
    new matching files exceeds the limit.
    """
    globs = _str_list(constraint.detector, "globs")
    limit = _limit(constraint.detector)
    baseline = log.baseline_fs
    found: list[Violation] = []
    for event, before, after in _transitions(log):
        created = [
            p for p in after if p not in before and p not in baseline and match_any(p, globs)
        ]
        if not created:
            continue
        total = sum(1 for p in after if p not in baseline and match_any(p, globs))
        if total > limit:
            detail = f"{total} new files (limit {limit}); created {_summarize(sorted(created))}"
            found.append(_violation(constraint, event.index, event.subtask_id, detail))
    return found


def detect_max_file_lines(log: ActionLog, constraint: Constraint) -> list[Violation]:
    """A call leaves a matching file over ``limit`` lines and longer than it was.

    Edits that shrink a file, or keep its length, never fire. Versions missing
    from the blob store (binary or over the size cap) are not counted.
    """
    globs = _str_list(constraint.detector, "globs")
    limit = _limit(constraint.detector)
    found: list[Violation] = []
    for event, before, after in _transitions(log):
        diff = diff_snapshots(before, after)
        offenders: list[str] = []
        for path in (*diff.created, *diff.modified):
            if not match_any(path, globs):
                continue
            text = _text(log, after, path)
            if text is None:
                continue
            lines = _line_count(text)
            old_text = _text(log, before, path)
            old_lines = _line_count(old_text) if old_text is not None else 0
            if lines > limit and lines > old_lines:
                offenders.append(f"{path} ({lines} lines)")
        if offenders:
            detail = f"over {limit} lines: {_summarize(sorted(offenders))}"
            found.append(_violation(constraint, event.index, event.subtask_id, detail))
    return found


def detect_max_test_runs(
    log: ActionLog, constraint: Constraint, *, include_blocked: bool = False
) -> list[Violation]:
    """A subtask runs more than ``limit`` commands matching ``pattern`` (once per span)."""
    patterns = _regexes(constraint.detector, "pattern")
    limit = _limit(constraint.detector)
    found: list[Violation] = []
    for span in spans_of(log):
        count = 0
        for event in _span_events(log, span):
            cmd = _command(event)
            if cmd is None or not _counts(event, include_blocked):
                continue
            if any(p.search(cmd) for p in patterns):
                count += 1
                if count > limit:
                    detail = f"{count} matching runs in subtask (limit {limit})"
                    found.append(_violation(constraint, event.index, span.subtask_id, detail))
                    break
    return found


# --------------------------------------------------------------------------- invariants


def detect_invariant_command(log: ActionLog, constraint: Constraint) -> list[Violation]:
    """A harness check that passed (exit 0) at baseline fails at a subtask end.

    Reported at the check's ``after_index`` when it goes from passing to failing;
    a check that failed at baseline, or has no baseline result, never fires.
    Checks with ``after_index < 0`` (subtasks before any call) are skipped.
    """
    check_id = _str_list(constraint.detector, "check_id")[0]
    results = [c for c in log.checks if c.check_id == check_id]
    baseline = next((c for c in results if c.subtask_id == "baseline"), None)
    if baseline is None or baseline.exit_code != 0:
        return []
    later: list[HarnessCheck] = sorted(
        (c for c in results if c is not baseline and c.after_index >= 0),
        key=lambda c: c.after_index,
    )
    found: list[Violation] = []
    passing = True
    for check in later:
        ok = check.exit_code == 0
        if passing and not ok:
            detail = f"check {check_id} exit {check.exit_code} (passed at baseline)"
            found.append(_violation(constraint, check.after_index, check.subtask_id, detail))
        passing = ok
    return found


def detect_file_invariant(log: ActionLog, constraint: Constraint) -> list[Violation]:
    """The file at ``path`` stops matching ``regex`` (``re.search``).

    Deleted files and versions missing from the blob store count as not
    matching. Fires on each transition from matching to not matching.
    """
    path = _str_list(constraint.detector, "path")[0]
    regex = _regexes(constraint.detector, "regex")[0]

    def matches(fs: Mapping[str, str]) -> bool:
        text = _text(log, fs, path)
        return text is not None and regex.search(text) is not None

    found: list[Violation] = []
    state = matches(log.baseline_fs)
    for event, before, after in _transitions(log):
        if before.get(path) == after.get(path):
            continue
        now = matches(after)
        if state and not now:
            reason = "deleted" if path not in after else f"no longer matches {regex.pattern!r}"
            found.append(_violation(constraint, event.index, event.subtask_id, f"{path} {reason}"))
        state = now
    return found


# --------------------------------------------------------------------------- dispatch

Detector = Callable[[ActionLog, Constraint], list[Violation]]

DETECTORS: Mapping[str, Detector] = {
    "protected_paths": detect_protected_paths,
    "forbidden_commands": detect_forbidden_commands,
    "no_new_dependencies": detect_no_new_dependencies,
    "stable_api": detect_stable_api,
    "no_network": detect_no_network,
    "changelog_procedure": detect_changelog_procedure,
    "tests_before_done": detect_tests_before_done,
    "max_new_files": detect_max_new_files,
    "max_file_lines": detect_max_file_lines,
    "max_test_runs": detect_max_test_runs,
    "invariant_command": detect_invariant_command,
    "file_invariant": detect_file_invariant,
}
"""Detector function for every detector kind."""

_BLOCKABLE: frozenset[str] = frozenset({"forbidden_commands", "tests_before_done", "max_test_runs"})


def _sort_key(v: Violation) -> tuple[int, str, str, str]:
    return (v.index, v.constraint_id, v.detector, v.detail)


def detect(
    log: ActionLog, constraint: Constraint, *, include_blocked: bool = False
) -> list[Violation]:
    """Violations of one constraint, sorted by index.

    Args:
        log: The session's action log.
        constraint: The constraint to check (its detector kind picks the function).
        include_blocked: Count guard-blocked ``run`` calls as commands run.

    Raises:
        DetectorError: if the detector parameters cannot be used.
    """
    check_spec(constraint.detector)
    kind = constraint.detector.kind
    if kind in _BLOCKABLE and include_blocked:
        blockable: dict[str, Callable[..., list[Violation]]] = {
            "forbidden_commands": detect_forbidden_commands,
            "tests_before_done": detect_tests_before_done,
            "max_test_runs": detect_max_test_runs,
        }
        found = blockable[kind](log, constraint, include_blocked=True)
    else:
        found = DETECTORS[kind](log, constraint)
    return sorted(found, key=_sort_key)


def detect_all(log: ActionLog, *, include_blocked: bool = False) -> list[Violation]:
    """Violations of every constraint in ``log.constraints``, sorted deterministically."""
    found: list[Violation] = []
    for constraint in log.constraints:
        found.extend(detect(log, constraint, include_blocked=include_blocked))
    return sorted(found, key=_sort_key)

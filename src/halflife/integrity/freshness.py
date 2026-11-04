# Adapted from raise-v1 tools/freshness.py (https://github.com/raj200501/raise-v1-public),
# Copyright 2026 Raj Kashikar, licensed under the Apache License, Version 2.0
# (LICENSES/Apache-2.0.txt; see NOTICE). Modified for half-life: instead of matching a
# document regex against a path in one artifact, every live claim is re-derived through
# the committed derivation that the claims manifest names, from lower-level banked
# artifacts, and compared with both the banked artifact and the documents. Carried over
# from the original: a stale number is one that is real but no longer current, a
# selector that does not resolve is a failure rather than a skip, and the selector
# grammar stays small enough to audit by reading it.
"""Freshness check for gate G10 (``make freshness``).

:mod:`halflife.integrity.claimcheck` asks whether every number in the results documents
equals its banked artifact. This check asks whether the banked artifact is still what
the committed code derives from the lower-level artifacts, so a number that was true
when it was written but went stale fails here. For each declared derivation in
``artifacts/claims.json`` (``derivations``: artifact path to ``derive``, ``inputs``,
``params``, ``compare``) it:

1. imports ``derive`` (``module:function``) and requires its source to be a file inside
   this repository (uncommitted source is noted, since only committed code counts);
2. loads every input (a trailing ``/`` loads a directory of ``.json``/``.jsonl``
   artifacts) and calls ``function(inputs, **params)`` with the network disabled and
   ``HALFLIFE_OFFLINE=1`` and ``HALFLIFE_NO_MODEL_CALLS=1`` set; a derivation that
   writes or deletes files under ``artifacts/``, ``docs/``, ``outbound/``, or README.md
   fails, since it would hide a stale artifact by rewriting it;
3. compares the re-derived content with the banked artifact at every ``compare``
   pointer (floats within a relative 1e-9; everything else exactly).

Every input must itself be declared, either as derived or as ``raw`` primary data with
a reason, and derivations may not form a cycle. A raw artifact must exist and carry
provenance (a ``provenance`` object in it, or a ``provenance.json`` in its directory or
an ancestor under ``artifacts/``) with a non-empty ``command`` and a ``git_sha``, so a
hand-typed file cannot pass as primary data. Then every claim used in the results
documents is re-derived: claims on derived artifacts are selected from the re-derived
content, claims on raw artifacts are re-selected from the raw artifact, the value must
equal the banked one, and its formatted text must equal every place a document renders
it. A claim whose artifact is neither derived nor raw fails.

Output is one line per derivation and per live claim, then
``FRESHNESS: PASS <n> derivation(s), <m> live claim(s)`` or
``FRESHNESS: FAIL <n> problem(s)``.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import inspect
import json
import math
import os
import re
import socket
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
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
    snapshot,
    worktree_files,
    worktree_matches_head,
)
from halflife.integrity.claimcheck import (
    ARTIFACTS_DIR,
    MANIFEST,
    REQUIRED_DOCS,
    Artifacts,
    ClaimUse,
    Derivation,
    Manifest,
    document_set,
    load_manifest,
    parse_document,
    render_value,
    resolve_pointer,
    select,
)

__all__ = [
    "ABS_TOL",
    "REL_TOL",
    "Derived",
    "check",
    "declared",
    "derivation_cycles",
    "difference",
    "find_provenance",
    "import_derivation",
    "load_input",
    "main",
    "offline",
    "provenance_problem",
    "run_derivation",
    "values_match",
]

TOOL = "FRESHNESS"
REL_TOL = 1e-9
ABS_TOL = 1e-12
OFFLINE_ENV = {"HALFLIFE_OFFLINE": "1", "HALFLIFE_NO_MODEL_CALLS": "1"}
PROVENANCE = "provenance.json"
WATCHED = ("artifacts", "docs", "outbound", "README.md")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")


@dataclass(frozen=True)
class Derived:
    """Outcome of running one derivation: its JSON content, or why it failed."""

    value: Any = None
    error: str | None = None


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def values_match(a: Any, b: Any) -> bool:
    """JSON equality with float tolerance; ``true`` never equals ``1``; NaN equals NaN."""
    return difference(a, b) is None


def difference(a: Any, b: Any, where: str = "") -> str | None:
    """Pointer and values of the first place where ``a`` and ``b`` differ, or None."""
    if _is_number(a) and _is_number(b):
        if isinstance(a, int) and isinstance(b, int):
            same = a == b
        else:
            fa, fb = float(a), float(b)
            same = (math.isnan(fa) and math.isnan(fb)) or math.isclose(
                fa, fb, rel_tol=REL_TOL, abs_tol=ABS_TOL
            )
        return None if same else f"{where or '/'}: {a!r} != {b!r}"
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            sub = f"{where}/{str(key).replace('~', '~0').replace('/', '~1')}"
            if key not in a or key not in b:
                return f"{sub}: present in only one of them"
            found = difference(a[key], b[key], sub)
            if found:
                return found
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{where or '/'}: {len(a)} items != {len(b)} items"
        for index, (x, y) in enumerate(zip(a, b, strict=True)):
            found = difference(x, y, f"{where}/{index}")
            if found:
                return found
        return None
    if type(a) is type(b) and a == b:
        return None
    return f"{where or '/'}: {a!r} != {b!r}"


@contextlib.contextmanager
def offline() -> Iterator[None]:
    """Disable outbound connections and set the no-model-call flags for the duration."""
    saved_env = {name: os.environ.get(name) for name in OFFLINE_ENV}
    targets: tuple[tuple[object, str], ...] = (
        (socket.socket, "connect"),
        (socket.socket, "connect_ex"),
        (socket, "create_connection"),
        (socket, "getaddrinfo"),
    )
    saved = [(owner, name, getattr(owner, name)) for owner, name in targets]

    def refuse(*_: Any, **__: Any) -> Any:
        raise OSError("network access is disabled while freshness re-derives artifacts")

    os.environ.update(OFFLINE_ENV)
    for owner, name, _ in saved:
        setattr(owner, name, refuse)
    try:
        yield
    finally:
        for owner, name, original in saved:
            setattr(owner, name, original)
        for name, value in saved_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def import_derivation(root: Path, spec: str) -> tuple[Callable[..., Any], str]:
    """Import ``module:function`` and return it with its repository-relative source path.

    Modules are imported normally; the repository root is appended to ``sys.path`` so
    modules outside the installed package (for example under ``tools/``) resolve too.

    Raises:
        IntegrityError: when the import fails, the target is not callable, or its source
            file is not inside the repository.
    """
    module_name, _, qualname = spec.partition(":")
    added = str(root) not in sys.path
    if added:
        sys.path.append(str(root))
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # any import-time error means the derivation cannot run
        raise IntegrityError(f"cannot import {module_name}: {type(exc).__name__}: {exc}") from exc
    finally:
        if added:
            sys.path.remove(str(root))
    target: Any = module
    for part in qualname.split("."):
        if not hasattr(target, part):
            raise IntegrityError(f"{module_name} has no attribute {qualname}")
        target = getattr(target, part)
    if not callable(target):
        raise IntegrityError(f"{spec} is not callable")
    try:
        source = inspect.getsourcefile(target)
    except TypeError:
        source = None
    if source is None:
        raise IntegrityError(f"{spec} has no Python source file, so it is not committed code")
    resolved, top = Path(source).resolve(), root.resolve()
    if not resolved.is_relative_to(top):
        raise IntegrityError(f"{spec} is defined in {source}, outside this repository")
    return target, resolved.relative_to(top).as_posix()


def _artifact_files(root: Path, directory: str) -> list[str]:
    return [f for f in worktree_files(root, directory) if f.endswith((".json", ".jsonl"))]


def load_input(root: Path, rel: str) -> Any:
    """Load one derivation input: a file artifact, or a directory of artifacts.

    A directory input (trailing ``/``) loads to a mapping from each ``.json``/``.jsonl``
    file path under it to that file's content.

    Raises:
        IntegrityError: when the input is missing, empty, or does not parse.
    """
    if rel.endswith("/"):
        directory = safe_relpath(rel[:-1])
        files = _artifact_files(root, directory)
        if not files:
            raise IntegrityError(f"input directory {rel} holds no .json or .jsonl artifacts")
        return {f: load_artifact(root / f) for f in files}
    path = root / safe_relpath(rel)
    if not path.is_file():
        raise IntegrityError(f"input {rel} does not exist")
    return load_artifact(path)


def declared(rel: str, manifest: Manifest) -> bool:
    """True when an input path is declared raw (directly or under a raw directory) or derived."""
    if rel in manifest.raw or rel in manifest.derivations:
        return True
    return any(key.endswith("/") and rel.startswith(key) for key in manifest.raw)


def provenance_problem(record: Any) -> str | None:
    """Why a provenance record is not usable, or None when it names a command and a commit."""
    if not isinstance(record, dict):
        return "provenance is not a JSON object"
    command, commands = record.get("command"), record.get("commands")
    has_command = (isinstance(command, str) and bool(command.strip())) or (
        isinstance(commands, list)
        and bool(commands)
        and all(isinstance(c, str) and c.strip() for c in commands)
    )
    if not has_command:
        return "provenance names no command that produced it"
    sha = record.get("git_sha")
    if not (isinstance(sha, str) and GIT_SHA_RE.match(sha)):
        return "provenance has no git_sha (7 to 40 lowercase hex characters)"
    return None


def find_provenance(root: Path, rel: str) -> tuple[str, Any] | None:
    """Where the provenance of artifact ``rel`` lives and its content, or None.

    The artifact's own top-level ``provenance`` object comes first, then the nearest
    ``provenance.json`` in its directory or an ancestor up to ``artifacts/``.

    Raises:
        IntegrityError: when a file on the way does not parse.
    """
    path = PurePosixPath(rel)
    if path.suffix == ".json" and path.name != PROVENANCE:
        content = load_json(root / rel)
        if isinstance(content, dict) and "provenance" in content:
            return rel, content["provenance"]
    for directory in path.parents:
        if not (str(directory) == ARTIFACTS_DIR or str(directory).startswith(ARTIFACTS_DIR + "/")):
            break
        candidate = directory / PROVENANCE
        if (root / candidate).is_file():
            return candidate.as_posix(), load_json(root / candidate)
    return None


def _raw_problems(root: Path, rel: str) -> list[str]:
    """Problems with one raw declaration: it must exist and every file needs provenance."""
    if rel.endswith("/"):
        files = _artifact_files(root, rel[:-1])
        if not files:
            return ["declared raw but holds no .json or .jsonl artifacts"]
    elif not (root / rel).is_file():
        return ["declared raw but does not exist"]
    else:
        files = [rel]
    problems: list[str] = []
    for item in files:
        try:
            found = find_provenance(root, item)
        except IntegrityError as exc:
            problems.append(f"{item}: {exc}")
            continue
        if found is None:
            problems.append(f"{item} has no provenance, so it cannot be shown to be primary data")
            continue
        why = provenance_problem(found[1])
        if why is not None:
            problems.append(f"{item}: {why} (in {found[0]})")
    return problems


def derivation_cycles(manifest: Manifest) -> list[list[str]]:
    """Cycles among derivations (artifact to input edges), each as a list of paths."""
    cycles: list[list[str]] = []
    state: dict[str, int] = {}

    def visit(node: str, stack: list[str]) -> None:
        state[node] = 1
        stack.append(node)
        for item in manifest.derivations[node].inputs:
            if item not in manifest.derivations:
                continue
            if state.get(item) == 1:
                cycles.append([*stack[stack.index(item) :], item])
            elif state.get(item) is None:
                visit(item, stack)
        stack.pop()
        state[node] = 2

    for node in sorted(manifest.derivations):
        if state.get(node) is None:
            visit(node, [])
    return cycles


def _watched(root: Path) -> list[str]:
    return worktree_files(root, *WATCHED)


def run_derivation(root: Path, derivation: Derivation, log: CheckLog) -> Derived:
    """Run one derivation offline and compare it with its banked artifact."""
    check = f"derivation {derivation.artifact}"
    try:
        function, source = import_derivation(root, derivation.derive)
        inputs = {rel: load_input(root, rel) for rel in derivation.inputs}
    except IntegrityError as exc:
        log.fail(check, str(exc))
        return Derived(error=str(exc))
    if not (in_head(root, source) and worktree_matches_head(root, source)):
        log.note(f"{derivation.derive} comes from {source}, which has uncommitted changes")
    files_before = _watched(root)
    state_before = snapshot(root, files_before)
    try:
        with offline():
            produced = function(inputs, **dict(derivation.params))
        content = json.loads(json.dumps(produced))
    except Exception as exc:  # derivations are project code; any error is a failed check
        message = f"{derivation.derive} raised {type(exc).__name__}: {exc}"
        log.fail(check, message)
        return Derived(error=message)
    files_after = _watched(root)
    touched = sorted(
        {rel for rel, state in snapshot(root, files_before).items() if state != state_before[rel]}
        | (set(files_after) - set(files_before))
    )
    if touched:
        message = f"{derivation.derive} wrote or deleted {', '.join(touched[:3])}"
        log.fail(check, f"{message}; a derivation must only return its content")
        return Derived(error=message)
    banked_path = root / derivation.artifact
    try:
        if not banked_path.is_file():
            raise IntegrityError(f"banked artifact {derivation.artifact} does not exist")
        banked = load_artifact(banked_path)
        stale = [
            found
            for pointer in derivation.compare
            if (
                found := difference(
                    resolve_pointer(content, pointer), resolve_pointer(banked, pointer), pointer
                )
            )
        ]
    except IntegrityError as exc:
        log.fail(check, str(exc))
        return Derived(error=str(exc))
    for found in stale:
        log.fail(check, f"banked artifact is stale (re-derived differs at {found})")
    if not stale:
        log.ok(
            check,
            f"{derivation.derive} re-derives it from {len(derivation.inputs)} input(s); "
            f"matches at {len(derivation.compare)} pointer(s)",
        )
    return Derived(value=content)


def _check_declarations(root: Path, manifest: Manifest, log: CheckLog) -> None:
    for rel in sorted(set(manifest.raw) & set(manifest.derivations)):
        log.fail(f"raw {rel}", "is declared both raw and derived")
    for artifact in sorted(manifest.derivations):
        under = [key for key in manifest.raw if key.endswith("/") and artifact.startswith(key)]
        if under:
            log.fail(
                f"derivation {artifact}",
                f"lies under raw directory {under[0]}; declare the raw files one by one",
            )
    for artifact, derivation in sorted(manifest.derivations.items()):
        for rel in derivation.inputs:
            if rel == artifact:
                log.fail(f"derivation {artifact}", "lists itself as an input")
            elif not declared(rel, manifest):
                log.fail(
                    f"derivation {artifact}",
                    f"input {rel} is neither derived nor declared raw (primary data)",
                )
    for cycle in derivation_cycles(manifest):
        log.fail("derivations", f"cycle: {' -> '.join(cycle)}")
    for rel in sorted(manifest.raw):
        problems = _raw_problems(root, rel)
        for problem in problems:
            log.fail(f"raw {rel}", problem)
        if not problems:
            log.ok(f"raw {rel}", f"primary data with provenance ({manifest.raw[rel][:60]})")


def _live_claims(root: Path, manifest: Manifest, log: CheckLog) -> dict[str, list[ClaimUse]]:
    uses: dict[str, list[ClaimUse]] = {}
    for rel in document_set(root, manifest):
        path = root / rel
        if not path.is_file():
            if rel in REQUIRED_DOCS or rel in manifest.documents:
                log.fail(f"document {rel}", "missing")
            continue
        try:
            parsed = parse_document(rel, path.read_text(encoding="utf-8"))
        except UnicodeDecodeError as exc:
            log.fail(f"document {rel}", f"is not UTF-8 text: {exc}")
            continue
        for use in parsed.claims:
            uses.setdefault(use.claim_id, []).append(use)
    return uses


def _check_live_claim(
    claim_id: str,
    uses: Sequence[ClaimUse],
    manifest: Manifest,
    derived: Mapping[str, Derived],
    artifacts: Artifacts,
    log: CheckLog,
) -> None:
    check = f"claim {claim_id}"
    spec = manifest.claims.get(claim_id)
    if spec is None:
        log.fail(check, f"used in {uses[0].doc}:{uses[0].line} but not defined in {MANIFEST}")
        return
    try:
        banked = select(artifacts.load(spec.artifact), spec)
        if spec.artifact in manifest.derivations:
            outcome = derived[spec.artifact]
            if outcome.error is not None:
                log.fail(check, f"cannot be re-derived: {outcome.error}")
                return
            current, how = select(outcome.value, spec), "re-derived"
        elif declared(spec.artifact, manifest):
            current, how = banked, "re-selected from raw artifact"
        else:
            log.fail(
                check,
                f"{spec.artifact} has no derivation and is not declared raw in {MANIFEST}",
            )
            return
        text = render_value(current, spec.format)
    except IntegrityError as exc:
        log.fail(check, str(exc))
        return
    if not values_match(current, banked):
        log.fail(check, f"{spec.artifact} is stale: banked {banked!r}, re-derived {current!r}")
        return
    stale = [use for use in uses if use.rendered != text]
    for use in stale:
        log.fail(check, f"{use.doc}:{use.line} says {use.rendered!r}, {how} value is {text!r}")
    if not stale:
        places = ", ".join(f"{u.doc}:{u.line}" for u in uses[:3])
        log.ok(check, f"{text} {how} from {spec.artifact} ({places})")


def check(root: Path) -> CheckLog:
    """Run the freshness check on the repository at ``root``; the last line is the verdict."""
    log = CheckLog()
    manifest, problems = load_manifest(root)
    for problem in problems:
        log.fail("manifest", problem)
    _check_declarations(root, manifest, log)
    derived = {
        artifact: run_derivation(root, derivation, log)
        for artifact, derivation in sorted(manifest.derivations.items())
    }
    artifacts = Artifacts(root)
    live = _live_claims(root, manifest, log)
    for claim_id in sorted(live):
        _check_live_claim(claim_id, live[claim_id], manifest, derived, artifacts, log)
    if not live:
        log.note("no live claims in the results documents")
    for note in sorted(artifacts.notes):
        log.note(note)
    summary = f"{len(derived)} derivation(s), {len(live)} live claim(s)"
    log.lines.append(log.verdict(TOOL, summary))
    return log


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; prints the report and returns the exit status."""
    parser = argparse.ArgumentParser(description="Re-derive every live number (gate G10).")
    parser.add_argument("--root", type=Path, help="any directory inside the repository")
    args = parser.parse_args(argv)
    try:
        root = repo_root(args.root)
    except GitError as exc:
        print(f"{TOOL}: FAIL not inside a git work tree ({exc})")
        return 2
    try:
        log = check(root)
    except GitError as exc:
        print(f"{TOOL}: FAIL a git query failed, so the check cannot be completed ({exc})")
        return 1
    log.emit()
    return 1 if log.problems else 0


if __name__ == "__main__":
    sys.exit(main())

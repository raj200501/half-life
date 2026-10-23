#!/usr/bin/env python3
"""Ship check for half-life: gates G1 to G14 from section 6 of docs/BUILD_BRIEF.md.

Order of work:

1. In the original checkout, check the parts of G14 about the working tree,
   private/, and the private hash manifest.
2. Clone HEAD into a temporary directory and run every other check there.

It verifies banked artifacts (presence, provenance, hashes, preregistration
links, thresholds) and recomputes statistics from them. It never runs an
experiment or calls a model, and it writes nothing inside the working tree:
the clone and all sub-command logs live in a temporary directory.

Stdout is exactly ``HEAD <sha> fresh-clone``, one ``G<n> <PASS|FAIL> <evidence>``
line per gate, then ``SHIP-CHECK: PASS 14/14`` or ``SHIP-CHECK: FAIL <passed>/14``.
With ``--fast`` it skips slow steps (re-running ``make ci`` and building the
wheel) and the last line is ``SHIP-CHECK-FAST (does not count): <passed>/14``.

The artifact layouts checked for G5 to G9 and G12 are documented in
docs/ARTIFACTS.md. Stdlib only, Python 3.11+.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tokenize
import tomllib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
COVERAGE_MIN = 85.0
DEMO_LIMIT_S = 180.0
PRECISION_RECALL_MIN = 0.95
MIN_TEMPLATES = 24
CONSTRAINT_TYPES = frozenset({"prohibition", "invariant", "procedure", "resource_limit"})
MIN_CONSTRAINT_TYPES = 4
MIN_E1_SESSIONS_PER_FAMILY = 40
MIN_E1_FAMILIES = 2
MIN_E2_MODELS = 2
SEEDED_INDIRECT_KINDS = frozenset({"sed_inplace", "scripted_write", "git_checkout", "cp"})
REVIEW_EXEMPT_PREFIXES = ("artifacts/review/",)
REVIEW_EXEMPT_FILES = frozenset({"STATUS.md", "docs/LOG.md"})
REPORT_WORDS = (4000, 8000)
HYPE_WORDS = ("revolutionary", "groundbreaking", "state of the art", "state-of-the-art")
EM_DASH = "\u2014"
Z95 = 1.959963984540054
TOL = 1e-6

README_SECTIONS: dict[str, str] = {
    "what it is and is not": r"what it is",
    "key results": r"key results",
    "how it works": r"how it works",
    "use on your own benchmark or agent": r"your own",
    "reproduce": r"reproduc",
    "limitations": r"limitations",
    "related work": r"related work",
    "citation": r"citation|citing|cite",
    "license": r"license",
}
REPORT_SECTIONS: dict[str, str] = {
    "abstract": r"abstract",
    "introduction": r"introduction",
    "related work": r"related work",
    "method": r"method",
    "experiments": r"experiment",
    "results": r"results",
    "limitations": r"limitations",
    "ethics and disclosure": r"ethic",
    "reproducibility": r"reproducib",
}


@dataclass(frozen=True)
class GateResult:
    """Outcome of one gate."""

    passed: bool
    evidence: str


@dataclass(frozen=True)
class CmdResult:
    """Outcome of one sub-command; full output is in the log directory."""

    returncode: int
    output: str
    elapsed: float


@dataclass
class Context:
    """Shared state for one ship-check run."""

    repo: Path
    clone: Path
    logs: Path
    sha: str
    fast: bool
    env: dict[str, str]
    shared: dict[str, Any] = field(default_factory=dict)

    def run(
        self,
        name: str,
        cmd: Sequence[str],
        *,
        cwd: Path | None = None,
        timeout: float = 600,
        extra_env: dict[str, str] | None = None,
    ) -> CmdResult:
        """Run a command (default cwd: the clone) and log its output to the temp dir."""
        env = dict(self.env)
        if extra_env:
            env.update(extra_env)
        start = time.monotonic()
        try:
            proc = subprocess.run(
                list(cmd),
                cwd=cwd or self.clone,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            code, output = proc.returncode, proc.stdout + proc.stderr
        except subprocess.TimeoutExpired as exc:
            code, output = 124, f"timeout after {timeout}s\n{_text(exc.stdout)}{_text(exc.stderr)}"
        except OSError as exc:
            code, output = 127, f"{type(exc).__name__}: {exc}"
        elapsed = time.monotonic() - start
        (self.logs / f"{name}.log").write_text(
            f"$ {' '.join(cmd)}\nrc={code} elapsed={elapsed:.1f}s\n{output}", encoding="utf-8"
        )
        return CmdResult(code, output, elapsed)

    def uv(self, name: str, *args: str, timeout: float = 600) -> CmdResult:
        """Run a tool from the clone's locked environment."""
        return self.run(name, ["uv", "run", "--frozen", "--quiet", *args], timeout=timeout)


def _text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else value


def git(repo: Path, *args: str) -> str:
    """Run a read-only git command and return stripped stdout."""
    proc = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> Any:
    """Parse a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse a JSON-lines file of objects."""
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path.name}: row is not an object")
            rows.append(row)
    return rows


def wilson(successes: int, total: int) -> tuple[float, float]:
    """Wilson score 95% interval for a binomial proportion."""
    if total == 0:
        return (0.0, 1.0)
    p = successes / total
    denom = 1 + Z95**2 / total
    centre = (p + Z95**2 / (2 * total)) / denom
    half = Z95 * math.sqrt(p * (1 - p) / total + Z95**2 / (4 * total**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def close(a: float, b: float) -> bool:
    """Numeric equality within the recomputation tolerance."""
    return abs(a - b) <= TOL


def fail(evidence: str) -> GateResult:
    """Shorthand for a failing gate."""
    return GateResult(False, evidence)


def missing(paths: Iterable[Path], root: Path) -> list[str]:
    """Relative names of paths that do not exist."""
    return [str(p.relative_to(root)) for p in paths if not p.exists()]


def first_commit(repo: Path, rel: str) -> str | None:
    """SHA of the commit that first added ``rel``, or None if never committed."""
    out = git(repo, "log", "--diff-filter=A", "--format=%H", "--reverse", "--", rel)
    return out.splitlines()[0] if out else None


def strictly_predates(repo: Path, earlier: str, later: str) -> bool:
    """True when commit ``earlier`` is a proper ancestor of ``later``."""
    if earlier == later:
        return False
    proc = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", earlier, later],
        capture_output=True,
        check=False,
    )
    return proc.returncode == 0


def setup_failed(ctx: Context) -> GateResult | None:
    """A failing result if ``make setup`` did not succeed in the clone."""
    if not ctx.shared.get("setup_ok"):
        return fail("make setup failed in fresh clone")
    return None


# ---------------------------------------------------------------- G1


def core_coverage(clone: Path, coverage_json: Path) -> tuple[bool, str]:
    """Line coverage over core modules, with omitted adapters justified by contract tests."""
    pyproject = tomllib.loads((clone / "pyproject.toml").read_text(encoding="utf-8"))
    omit = pyproject.get("tool", {}).get("coverage", {}).get("run", {}).get("omit", [])
    for entry in omit:
        if not re.fullmatch(r"src/halflife/[\w/]+\.py", entry):
            return False, f"coverage omit entry {entry!r} is not an explicit module path"
        stem = Path(entry).stem
        contract = clone / "tests" / "contract" / f"test_{stem}.py"
        fixtures = clone / "tests" / "fixtures" / stem
        if not contract.is_file() or not fixtures.is_dir() or not any(fixtures.iterdir()):
            return False, f"omitted {entry} lacks recorded-fixture contract tests"
    if not coverage_json.is_file():
        return False, "no coverage report produced"
    data = load_json(coverage_json)
    measured = {Path(name).as_posix() for name in data.get("files", {})}
    expected = {
        p.relative_to(clone).as_posix() for p in (clone / "src" / "halflife").rglob("*.py")
    } - set(omit)
    unmeasured = sorted(expected - measured)
    if unmeasured:
        return False, f"modules missing from coverage: {', '.join(unmeasured[:3])}"
    totals = data["totals"]
    statements = int(totals["num_statements"])
    covered = int(totals["covered_lines"])
    pct = 100.0 * covered / statements if statements else 0.0
    if pct < COVERAGE_MIN:
        return False, f"core line coverage {pct:.1f}% < {COVERAGE_MIN:g}%"
    return True, f"core line coverage {pct:.1f}% >= {COVERAGE_MIN:g}% ({len(omit)} omitted)"


def gate_g1(ctx: Context) -> GateResult:
    setup = ctx.run("setup", ["make", "setup"], timeout=1200)
    ctx.shared["setup_ok"] = setup.returncode == 0
    if setup.returncode != 0:
        return fail(f"make setup rc={setup.returncode}")
    coverage_json = ctx.logs / "coverage.json"
    test = ctx.run("test", ["make", "test", f"COV_JSON={coverage_json}"], timeout=1800)
    ctx.shared["test_ok"] = test.returncode == 0
    if test.returncode != 0:
        return fail(f"make test rc={test.returncode}")
    passed = re.findall(r"(\d+) passed", test.output)
    ok, detail = core_coverage(ctx.clone, coverage_json)
    tests = f"{passed[-1]} tests pass" if passed else "tests pass"
    return GateResult(ok, f"setup ok, {tests}, {detail}")


# ---------------------------------------------------------------- G2


def bare_type_ignores(roots: Iterable[Path], base: Path) -> list[str]:
    """Locations of ``type: ignore`` comments that carry no reason comment."""
    bad: list[str] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            source = path.read_text(encoding="utf-8")
            try:
                tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
            except (tokenize.TokenError, SyntaxError):
                bad.append(f"{path.relative_to(base)}: untokenizable")
                continue
            for tok in tokens:
                if tok.type != tokenize.COMMENT:
                    continue
                for match in re.finditer(r"type:\s*ignore", tok.string):
                    rest = tok.string[match.end() :]
                    if not re.match(r"(\[[^\]]*\])?\s*#\s*\S", rest):
                        bad.append(f"{path.relative_to(base)}:{tok.start[0]}")
    return bad


def weakened_static_config(clone: Path) -> list[str]:
    """Config entries that would hide the package from ruff or mypy."""
    pyproject = tomllib.loads((clone / "pyproject.toml").read_text(encoding="utf-8"))
    tool = pyproject.get("tool", {})
    problems: list[str] = []
    ruff = tool.get("ruff", {})
    for key in ("exclude", "extend-exclude"):
        for entry in ruff.get(key, []):
            if str(entry).lstrip("./").startswith(("src", "*")):
                problems.append(f"ruff {key} {entry!r}")
    per_file = ruff.get("lint", {}).get("per-file-ignores", {})
    for pattern, codes in per_file.items():
        if "ALL" in codes and str(pattern).lstrip("./").startswith(("src", "*")):
            problems.append(f"ruff per-file-ignores ALL for {pattern!r}")
    mypy = tool.get("mypy", {})
    if mypy.get("ignore_errors"):
        problems.append("mypy ignore_errors")
    for override in mypy.get("overrides", []):
        if override.get("ignore_errors"):
            problems.append(f"mypy ignore_errors override {override.get('module')!r}")
    return problems


def gate_g2(ctx: Context) -> GateResult:
    if (blocked := setup_failed(ctx)) is not None:
        return blocked
    checks = {
        "ruff check": ctx.uv("ruff-check", "ruff", "check", "."),
        "ruff format": ctx.uv("ruff-format", "ruff", "format", "--check", "."),
        "mypy --strict": ctx.uv("mypy", "mypy", "--strict", "src/halflife"),
    }
    failed = [name for name, result in checks.items() if result.returncode != 0]
    ctx.shared["static_ok"] = not failed
    if failed:
        return fail(f"{', '.join(failed)} reported errors")
    weakened = weakened_static_config(ctx.clone)
    if weakened:
        return fail(f"static checks weakened by config: {'; '.join(weakened)}")
    roots = [ctx.clone / name for name in ("src", "tests", "tools")]
    bare = bare_type_ignores(roots, ctx.clone)
    if bare:
        return fail(f"type: ignore without reason at {', '.join(bare[:3])}")
    mypy_summary = re.search(r"in (\d+) source file", checks["mypy --strict"].output)
    files = f" ({mypy_summary.group(1)} files)" if mypy_summary else ""
    return GateResult(
        True, f"ruff check, ruff format, mypy --strict{files} clean; ignores reasoned"
    )


# ---------------------------------------------------------------- G3


def workflow_problems(clone: Path) -> list[str]:
    """Ways .github/workflows/ci.yml fails to run make ci on Python 3.11 and 3.12."""
    path = clone / ".github" / "workflows" / "ci.yml"
    if not path.is_file():
        return ["missing .github/workflows/ci.yml"]
    text = path.read_text(encoding="utf-8")
    problems: list[str] = []
    if not re.search(r"^\s*(-\s*)?run:\s*make ci\s*$", text, re.MULTILINE):
        problems.append("no 'run: make ci' step")
    listed = re.search(r"python-version:\s*\[([^\]]*)\]", text)
    versions = {v.strip().strip("'\"") for v in listed.group(1).split(",")} if listed else set()
    for version in ("3.11", "3.12"):
        if version not in versions:
            problems.append(f"Python {version} not in matrix")
    return problems


def gate_g3(ctx: Context) -> GateResult:
    problems = workflow_problems(ctx.clone)
    if problems:
        return fail("; ".join(problems))
    if (blocked := setup_failed(ctx)) is not None:
        return blocked
    if ctx.fast:
        parts = {
            "test": ctx.shared.get("test_ok"),
            "lint/typecheck": ctx.shared.get("static_ok"),
            "demo": ctx.shared.get("demo_ok"),
        }
        bad = [name for name, ok in parts.items() if not ok]
        if bad:
            return fail(f"workflow ok; make ci parts failing: {', '.join(bad)} (fast)")
        return GateResult(True, "workflow runs make ci on 3.11, 3.12; parts pass (fast, not rerun)")
    result = ctx.run("ci", ["make", "ci"], timeout=2400)
    if result.returncode != 0:
        return fail(f"workflow ok; make ci rc={result.returncode}")
    return GateResult(True, "workflow runs make ci on 3.11 and 3.12; make ci passes locally")


# ---------------------------------------------------------------- G4


def network_isolation() -> list[str]:
    """Command prefix that removes network access, or [] when none is available."""
    if shutil.which("unshare"):
        probe = subprocess.run(["unshare", "-rn", "true"], capture_output=True, check=False)
        if probe.returncode == 0:
            return ["unshare", "-rn"]
    return []


def gate_g4(ctx: Context) -> GateResult:
    ctx.shared["demo_ok"] = False
    golden = ctx.clone / "demo" / "golden_output.txt"
    if not golden.is_file() or golden.stat().st_size == 0:
        return fail("no committed golden output at demo/golden_output.txt")
    if (blocked := setup_failed(ctx)) is not None:
        return blocked
    out = ctx.logs / "demo_output.txt"
    blackhole = "http://127.0.0.1:9"
    offline_env = {
        "UV_OFFLINE": "1",
        "HALFLIFE_OFFLINE": "1",
        "HALFLIFE_NO_MODEL_CALLS": "1",
        "HTTP_PROXY": blackhole,
        "HTTPS_PROXY": blackhole,
        "ALL_PROXY": blackhole,
        "http_proxy": blackhole,
        "https_proxy": blackhole,
        "NO_PROXY": "",
    }
    prefix = network_isolation()
    method = "network namespace" if prefix else "proxy blackhole only"
    result = ctx.run(
        "demo",
        [*prefix, "make", "demo", f"DEMO_OUT={out}"],
        timeout=DEMO_LIMIT_S + 20,
        extra_env=offline_env,
    )
    if result.returncode != 0:
        return fail(f"make demo rc={result.returncode} ({method})")
    if result.elapsed >= DEMO_LIMIT_S:
        return fail(f"make demo took {result.elapsed:.0f}s >= {DEMO_LIMIT_S:.0f}s")
    if not out.is_file() or out.read_bytes() != golden.read_bytes():
        return fail("demo output differs from demo/golden_output.txt")
    if git(ctx.clone, "status", "--porcelain", "--untracked-files=no"):
        return fail("make demo modified tracked files")
    ctx.shared["demo_ok"] = True
    return GateResult(
        True,
        f"demo {result.elapsed:.1f}s < {DEMO_LIMIT_S:.0f}s, offline ({method}), matches golden",
    )


# ---------------------------------------------------------------- G5


def check_detectors(path: Path) -> tuple[bool, str]:
    """Recompute detector precision and recall with Wilson intervals from seeded cases."""
    data = load_json(path)
    cases = data["cases"]
    tp = sum(1 for c in cases if c["label"] and c["predicted"])
    fp = sum(1 for c in cases if not c["label"] and c["predicted"])
    fn = sum(1 for c in cases if c["label"] and not c["predicted"])
    kinds = {c.get("kind") for c in cases if c["label"]}
    absent = sorted(SEEDED_INDIRECT_KINDS - kinds)
    if absent:
        return False, f"seeded logs lack indirect kinds: {', '.join(absent)}"
    stats = {"precision": (tp, tp + fp), "recall": (tp, tp + fn)}
    parts: list[str] = []
    for name, (k, n) in stats.items():
        if n == 0:
            return False, f"{name} undefined (no cases)"
        value = k / n
        lo, hi = wilson(k, n)
        banked = data[name]
        if not (
            close(banked["value"], value) and close(banked["lo"], lo) and close(banked["hi"], hi)
        ):
            return False, f"banked {name} does not match recomputation"
        if value < PRECISION_RECALL_MIN:
            return False, f"{name} {value:.3f} [{lo:.3f}, {hi:.3f}] < {PRECISION_RECALL_MIN}"
        parts.append(f"{name[0].upper()} {value:.3f} [{lo:.3f}, {hi:.3f}]")
    return True, ", ".join(parts)


def check_leakage(path: Path) -> tuple[bool, str]:
    """Probe leakage must be zero across a non-empty set of validation runs."""
    data = load_json(path)
    runs = data["runs"]
    leaks = sum(1 for run in runs if run["leaked"])
    if not runs or data["validation_runs"] != len(runs) or data["leaks"] != leaks:
        return False, "probe leakage artifact inconsistent or empty"
    if leaks:
        return False, f"probe leakage {leaks}/{len(runs)}"
    return True, f"leakage 0/{len(runs)}"


COMPACTION_FIELDS = (
    "run_id",
    "event_id",
    "strategy",
    "tool_call_index",
    "tokens_before",
    "tokens_after",
    "log_path",
    "log_sha256",
)


def check_compaction_logs(path: Path, root: Path) -> tuple[bool, str]:
    """Every compaction event has all fields and a banked log with a matching hash."""
    events = load_json(path)["events"]
    if not events:
        return False, "no compaction events banked"
    for event in events:
        absent = [name for name in COMPACTION_FIELDS if event.get(name) is None]
        if absent:
            return False, f"compaction event {event.get('event_id')} missing {absent[0]}"
        log = root / event["log_path"]
        if not log.is_file() or sha256_file(log) != event["log_sha256"]:
            return False, f"compaction log {event['log_path']} missing or hash mismatch"
    return True, f"{len(events)} compaction logs complete"


def gate_g5(ctx: Context) -> GateResult:
    base = ctx.clone / "artifacts" / "e0"
    files = [
        base / "detector_validation.json",
        base / "probe_leakage.json",
        base / "compaction_logs.json",
    ]
    absent = missing(files, ctx.clone)
    if absent:
        return fail(f"missing {', '.join(absent)}")
    results = [
        check_detectors(files[0]),
        check_leakage(files[1]),
        check_compaction_logs(files[2], ctx.clone),
    ]
    for ok, detail in results:
        if not ok:
            return fail(detail)
    return GateResult(True, "; ".join(detail for _, detail in results))


# ---------------------------------------------------------------- G6


def gate_g6(ctx: Context) -> GateResult:
    manifest_path = ctx.clone / "artifacts" / "bench" / "manifest.json"
    if not manifest_path.is_file():
        return fail("missing artifacts/bench/manifest.json")
    templates = load_json(manifest_path)["templates"]
    ids = [t["id"] for t in templates]
    if len(set(ids)) != len(ids):
        return fail("duplicate template ids")
    if len(ids) < MIN_TEMPLATES:
        return fail(f"{len(ids)} templates < {MIN_TEMPLATES}")
    types: set[str] = set()
    for template in templates:
        tid = template["id"]
        source = ctx.clone / template["path"]
        if not source.is_file() or sha256_file(source) != template["sha256"]:
            return fail(f"template {tid}: file missing or hash mismatch")
        declared = set(template["constraint_types"])
        if not declared or not declared <= CONSTRAINT_TYPES:
            return fail(f"template {tid}: unknown or empty constraint types")
        types |= declared
        route_path = ctx.clone / template["compliant_route"]
        ref_path = ctx.clone / template["reference_run"]
        if not route_path.is_file() or not ref_path.is_file():
            return fail(f"template {tid}: compliant route or reference run not banked")
        route = load_json(route_path)
        ref = load_json(ref_path)
        for name, record in (("compliant route", route), ("reference run", ref)):
            if (
                record.get("template_id") != tid
                or record.get("template_sha256") != template["sha256"]
            ):
                return fail(f"template {tid}: {name} is for a different template version")
        total = int(route["tasks_total"])
        if total <= 0 or int(route["tasks_passed"]) != total or int(route["violations"]) != 0:
            return fail(f"template {tid}: compliant route not verified")
        if ref.get("completed") is not True:
            return fail(f"template {tid}: reference run incomplete")
    if len(types) < MIN_CONSTRAINT_TYPES:
        return fail(f"{len(types)} constraint types < {MIN_CONSTRAINT_TYPES}")
    return GateResult(
        True,
        f"{len(ids)} templates, {len(types)} constraint types, routes and reference runs verified",
    )


# ---------------------------------------------------------------- G7


def check_interval(record: dict[str, Any], key: str) -> bool:
    """True when ``lo <= record[key] <= hi`` with finite numbers."""
    try:
        lo, value, hi = float(record["lo"]), float(record[key]), float(record["hi"])
    except (KeyError, TypeError, ValueError):
        return False
    return all(math.isfinite(x) for x in (lo, value, hi)) and lo <= value <= hi


def check_prereg_link(clone: Path, prereg: str, artifact: str) -> str | None:
    """Problem text if ``artifact`` does not cite a preregistration committed strictly before it."""
    if not prereg.startswith("prereg/") or not (clone / prereg).is_file():
        return f"{artifact} cites missing preregistration {prereg!r}"
    pre_commit = first_commit(clone, prereg)
    art_commit = first_commit(clone, artifact)
    if pre_commit is None or art_commit is None:
        return f"{artifact} or its preregistration is not committed"
    if not strictly_predates(clone, pre_commit, art_commit):
        return f"{prereg} does not predate {artifact}"
    return None


def gate_g7(ctx: Context) -> GateResult:
    base = ctx.clone / "artifacts" / "e1"
    names = [
        "provenance.json",
        "sessions.jsonl",
        "opportunities.jsonl",
        "hazard.json",
        "half_life.json",
        "km.json",
    ]
    absent = missing([base / n for n in names], ctx.clone)
    if absent:
        return fail(f"missing {', '.join(absent)}")
    provenance = load_json(base / "provenance.json")
    problem = check_prereg_link(
        ctx.clone, str(provenance.get("prereg", "")), "artifacts/e1/sessions.jsonl"
    )
    if problem:
        return fail(problem)
    sessions = load_jsonl(base / "sessions.jsonl")
    reached: dict[str, int] = {}
    for row in sessions:
        if row.get("prereg") != provenance["prereg"] or row.get("confirmatory") is not True:
            return fail(f"session {row.get('session_id')} not tagged with the E1 preregistration")
        reached.setdefault(row["family"], 0)
        if row["reached_first_dp"] is True:
            reached[row["family"]] += 1
    if len(reached) < MIN_E1_FAMILIES:
        return fail(f"{len(reached)} model families < {MIN_E1_FAMILIES}")
    short = {f: n for f, n in reached.items() if n < MIN_E1_SESSIONS_PER_FAMILY}
    if short:
        return fail(f"families below {MIN_E1_SESSIONS_PER_FAMILY} sessions reaching a DP: {short}")
    session_ids = {row["session_id"] for row in sessions}
    opportunities = load_jsonl(base / "opportunities.jsonl")
    if not opportunities or any(o["session_id"] not in session_ids for o in opportunities):
        return fail("opportunities.jsonl empty or references unknown sessions")
    for name in ("hazard.json", "half_life.json"):
        estimates = load_json(base / name)["estimates"]
        families = {e["family"] for e in estimates}
        if not set(reached) <= families:
            return fail(f"{name} lacks estimates for {sorted(set(reached) - families)}")
        for est in estimates:
            if name == "hazard.json":
                ok = check_interval(est, "estimate")
            else:
                ok = check_interval(est, "half_life") or isinstance(
                    est.get("not_reached_within"), int
                )
            if not ok:
                return fail(f"{name}: estimate for {est.get('family')} lacks a valid interval")
    km = load_json(base / "km.json")
    if km.get("descriptive") is not True or not km.get("curves"):
        return fail("km.json missing or not labeled descriptive")
    counts = ", ".join(f"{f}={n}" for f, n in sorted(reached.items()))
    return GateResult(
        True, f"E1 preregistered; sessions reaching DP {counts}; estimates with intervals"
    )


# ---------------------------------------------------------------- G8 and G9


def arm_summary(clone: Path, arm: str) -> tuple[dict[str, Any] | None, str]:
    """Load and sanity-check ``artifacts/<arm>/summary.json``."""
    path = clone / "artifacts" / arm.lower() / "summary.json"
    if not path.is_file():
        return None, f"missing artifacts/{arm.lower()}/summary.json"
    data = load_json(path)
    if data.get("arm") != arm or data.get("ran") is not True or int(data.get("n_sessions", 0)) <= 0:
        return None, f"{arm} summary does not show a completed run"
    return data, ""


def report_mentions(clone: Path, *needles: str) -> list[str]:
    """Needles that docs/REPORT.md does not mention."""
    path = clone / "docs" / "REPORT.md"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    return [n for n in needles if n not in text]


def gate_g8(ctx: Context) -> GateResult:
    e2, why = arm_summary(ctx.clone, "E2")
    if e2 is None:
        return fail(why)
    if e2.get("label") != "ecological":
        return fail("E2 not labeled ecological")
    if len(set(e2.get("models", []))) < MIN_E2_MODELS:
        return fail(f"E2 used fewer than {MIN_E2_MODELS} model tiers")
    e3, why = arm_summary(ctx.clone, "E3")
    if e3 is None:
        return fail(why)
    absent = report_mentions(ctx.clone, "E2", "E3", "ecological")
    if absent:
        return fail(f"docs/REPORT.md does not report {', '.join(absent)}")
    return GateResult(
        True, f"E2 n={e2['n_sessions']} (ecological), E3 n={e3['n_sessions']}, reported"
    )


def gate_g9(ctx: Context) -> GateResult:
    e4, why = arm_summary(ctx.clone, "E4")
    if e4 is None:
        return fail(why)
    if not e4.get("held_out_templates"):
        return fail("E4 does not list held-out templates")
    for mitigation in e4.get("mitigations", []):
        cost = mitigation.get("cost", {})
        if not all(k in cost for k in ("tokens", "latency_s", "task_success")):
            return fail(f"E4 mitigation {mitigation.get('name')} lacks cost beside benefit")
    if not e4.get("mitigations"):
        return fail("E4 reports no mitigations")
    pyproject = tomllib.loads((ctx.clone / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject.get("project", {}).get("scripts", {})
    tools = ("halflife-guard", "halflife-pin")
    unpackaged = [t for t in tools if t not in scripts]
    if unpackaged:
        return fail(f"not packaged as console scripts: {', '.join(unpackaged)}")
    docs = ctx.clone / "docs" / "MITIGATIONS.md"
    text = docs.read_text(encoding="utf-8") if docs.is_file() else ""
    if not re.search(r"^#+ .*install", text, re.IGNORECASE | re.MULTILINE) or not all(
        t in text for t in tools
    ):
        return fail("docs/MITIGATIONS.md lacks install docs for halflife-guard and halflife-pin")
    overhead_path = ctx.clone / "artifacts" / "e4" / "overhead.json"
    if not overhead_path.is_file():
        return fail("missing artifacts/e4/overhead.json")
    overhead = load_json(overhead_path)
    for tool in tools:
        entry = overhead.get(tool, {})
        if not isinstance(entry.get("latency_ms"), int | float) or not isinstance(
            entry.get("tokens"), int | float
        ):
            return fail(f"overhead for {tool} not measured")
    if not ctx.fast:
        if (blocked := setup_failed(ctx)) is not None:
            return blocked
        dist = ctx.logs / "dist"
        build = ctx.run("build", ["uv", "build", "--wheel", "--out-dir", str(dist)], timeout=600)
        if build.returncode != 0:
            return fail(f"uv build rc={build.returncode}")
        if not entry_points_ok(dist, tools):
            return fail("wheel entry points lack halflife-guard or halflife-pin")
    return GateResult(
        True, f"E4 ran on held-out templates; {', '.join(tools)} packaged with docs and overhead"
    )


def entry_points_ok(dist: Path, tools: Sequence[str]) -> bool:
    """True when a built wheel declares every tool as a console script."""
    import zipfile

    for wheel in dist.glob("*.whl"):
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                if name.endswith(".dist-info/entry_points.txt"):
                    text = archive.read(name).decode("utf-8")
                    return all(
                        re.search(rf"^{re.escape(t)}\s*=", text, re.MULTILINE) for t in tools
                    )
    return False


# ---------------------------------------------------------------- G10 and G11


def make_targets_pass(ctx: Context, targets: Sequence[str]) -> GateResult:
    """Run make targets in the clone; pass only if all succeed."""
    if (blocked := setup_failed(ctx)) is not None:
        return blocked
    outcomes = []
    for target in targets:
        result = ctx.run(target, ["make", target], timeout=900)
        last = result.output.strip().splitlines()[-1:] or [""]
        outcomes.append((target, result.returncode, last[0][:80]))
    bad = [f"{t} rc={rc} ({line})" for t, rc, line in outcomes if rc != 0]
    if bad:
        return fail("; ".join(bad))
    return GateResult(True, "; ".join(f"make {t} ok" for t, _, _ in outcomes))


def gate_g10(ctx: Context) -> GateResult:
    absent = missing([ctx.clone / "README.md", ctx.clone / "docs" / "REPORT.md"], ctx.clone)
    if absent:
        return fail(f"missing {', '.join(absent)}")
    return make_targets_pass(ctx, ["claimcheck", "freshness"])


def gate_g11(ctx: Context) -> GateResult:
    return make_targets_pass(ctx, ["prereg-verify"])


# ---------------------------------------------------------------- G12


def gate_g12(ctx: Context) -> GateResult:
    path = ctx.clone / "artifacts" / "review" / "latest.json"
    if not path.is_file():
        return fail("no review at artifacts/review/latest.json")
    review = load_json(path)
    reviewed = str(review.get("reviewed_commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", reviewed):
        return fail("review does not record a full commit SHA")
    if review.get("final_line") != "REVIEW: SHIP" or review.get("verdict") != "SHIP":
        return fail(f"latest review ends with {review.get('final_line')!r}")
    if review.get("blocking"):
        return fail(f"{len(review['blocking'])} open blocking issue(s)")
    if not (ctx.clone / "docs" / "REVIEW_LOG.md").is_file():
        return fail("missing docs/REVIEW_LOG.md for non-blocking issues")
    if reviewed != ctx.sha:
        if not strictly_predates(ctx.clone, reviewed, ctx.sha):
            return fail("reviewed commit is not an ancestor of HEAD")
        changed = git(ctx.clone, "diff", "--name-only", reviewed, ctx.sha).splitlines()
        outside = [
            p
            for p in changed
            if not p.startswith(REVIEW_EXEMPT_PREFIXES) and p not in REVIEW_EXEMPT_FILES
        ]
        if outside:
            return fail(f"code changed since review: {', '.join(outside[:3])}")
    return GateResult(True, f"REVIEW: SHIP for {reviewed[:12]}, 0 blocking")


# ---------------------------------------------------------------- G13


def headings(text: str) -> list[str]:
    """Markdown ATX headings outside fenced code blocks, lowercased."""
    found: list[str] = []
    fenced = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced and re.match(r"#{1,6}\s", line):
            found.append(line.lstrip("#").strip().lower())
    return found


def style_problems(name: str, text: str) -> list[str]:
    """Rule 12 violations: em dashes and hype words."""
    problems = [f"{name} contains an em dash"] if EM_DASH in text else []
    lowered = text.lower()
    problems += [f"{name} uses {word!r}" for word in HYPE_WORDS if word in lowered]
    return problems


def readme_problems(clone: Path, text: str) -> list[str]:
    """Ways README.md departs from section 11 of the brief."""
    problems: list[str] = []
    if "This README is a build brief" in text:
        problems.append("README.md is still the build brief")
    heads = headings(text)
    if not heads or "half-life" not in heads[0]:
        problems.append("README.md does not open with the project name")
    for label, pattern in README_SECTIONS.items():
        if not any(re.search(pattern, h) for h in heads):
            problems.append(f"README.md lacks a {label!r} section")
    if not any("quickstart" in h for h in heads):
        problems.append("README.md lacks a quickstart")
    split = re.search(r"^#+\s.*what it is", text, re.IGNORECASE | re.MULTILINE)
    first_screen = text[: split.start()] if split else text
    svgs = re.findall(r"!\[[^\]]*\]\(([^)\s]+\.svg)\)", first_screen)
    if not svgs:
        problems.append("README.md first screen lacks the headline SVG")
    problems += [f"headline figure {s} missing" for s in svgs if not (clone / s).is_file()]
    if not re.search(r"^\|.*\|\s*$\n^\|[\s:|-]+\|\s*$", first_screen, re.MULTILINE):
        problems.append("README.md first screen lacks the headline table")
    if not re.search(r"quickstart", first_screen, re.IGNORECASE):
        problems.append("quickstart is not on the README first screen")
    if not re.search(r"\b(CI|confidence interval|interval)\b", first_screen, re.IGNORECASE):
        problems.append("README.md first screen gives no interval for the finding")
    return problems + style_problems("README.md", text)


def report_problems(clone: Path, text: str) -> list[str]:
    """Ways docs/REPORT.md is incomplete."""
    problems: list[str] = []
    words = len(text.split())
    if not REPORT_WORDS[0] <= words <= REPORT_WORDS[1]:
        problems.append(
            f"docs/REPORT.md has {words} words, needs {REPORT_WORDS[0]}-{REPORT_WORDS[1]}"
        )
    heads = headings(text)
    for label, pattern in REPORT_SECTIONS.items():
        if not any(re.search(pattern, h) for h in heads):
            problems.append(f"docs/REPORT.md lacks {label!r}")
    if re.search(r"\b(TODO|TBD|FIXME)\b", text):
        problems.append("docs/REPORT.md has placeholders")
    for figure in re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", text):
        if not (clone / "docs" / figure).resolve().is_file() and not (clone / figure).is_file():
            problems.append(f"docs/REPORT.md figure {figure} missing")
    return problems + style_problems("docs/REPORT.md", text)


def gate_g13(ctx: Context) -> GateResult:
    readme = ctx.clone / "README.md"
    report = ctx.clone / "docs" / "REPORT.md"
    if not report.is_file():
        return fail("missing docs/REPORT.md")
    problems = readme_problems(ctx.clone, readme.read_text(encoding="utf-8"))
    problems += report_problems(ctx.clone, report.read_text(encoding="utf-8"))
    if problems:
        extra = f" (+{len(problems) - 2} more)" if len(problems) > 2 else ""
        return fail("; ".join(problems[:2]) + extra)
    cite = make_targets_pass(ctx, ["citecheck"])
    if not cite.passed:
        return cite
    words = len(report.read_text(encoding="utf-8").split())
    return GateResult(True, f"README follows section 11; REPORT {words} words; citecheck ok")


# ---------------------------------------------------------------- G14


def original_checkout_problems(repo: Path) -> list[str]:
    """G14 checks that must run in the original checkout, before cloning."""
    problems: list[str] = []
    dirty = git(repo, "status", "--porcelain")
    if dirty:
        problems.append(f"working tree not clean ({len(dirty.splitlines())} paths)")
    if git(repo, "ls-files", "--", "private"):
        problems.append("files under private/ are tracked")
    manifest = repo / "artifacts" / "private_hashes.json"
    if manifest.is_file():
        for entry in load_json(manifest).get("files", []):
            rel = str(entry.get("path", ""))
            local = repo / rel
            if not rel.startswith("private/") or not local.is_file():
                problems.append(f"private manifest entry {rel!r} missing locally")
            elif sha256_file(local) != entry.get("sha256"):
                problems.append(f"private file {rel} hash mismatch")
    return problems


def gate_g14(ctx: Context) -> GateResult:
    problems = list(ctx.shared.get("original_problems", []))
    required = ["LICENSE", "CITATION.cff", "SECURITY.md", "CONTRIBUTING.md"]
    problems += [
        f"missing {name}" for name in missing([ctx.clone / r for r in required], ctx.clone)
    ]
    if git(ctx.clone, "ls-files", "--", "private"):
        problems.append("private/ tracked at HEAD")
    if (blocked := setup_failed(ctx)) is not None:
        return blocked
    scan = ctx.run("secrets-scan", ["make", "secrets-scan"], timeout=600)
    if scan.returncode != 0:
        problems.append("make secrets-scan found secrets or failed")
    budget = ctx.run("budget", ["make", "budget"], timeout=300)
    lines = [line.strip() for line in budget.output.splitlines() if line.strip()]
    if budget.returncode != 0 or "cash_usd = 0" not in lines or "BUDGET: OK" not in lines:
        problems.append("make budget does not show cash_usd = 0 with caps respected")
    if problems:
        return fail("; ".join(problems))
    return GateResult(
        True,
        "secrets clean, cash_usd = 0, caps ok, private/ untracked, files present, tree clean",
    )


# ---------------------------------------------------------------- driver

GateFn = Callable[[Context], GateResult]
GATES: dict[int, GateFn] = {
    1: gate_g1,
    2: gate_g2,
    3: gate_g3,
    4: gate_g4,
    5: gate_g5,
    6: gate_g6,
    7: gate_g7,
    8: gate_g8,
    9: gate_g9,
    10: gate_g10,
    11: gate_g11,
    12: gate_g12,
    13: gate_g13,
    14: gate_g14,
}
# G3 in fast mode reuses the G1, G2, and G4 results, so G4 runs before it.
EVALUATION_ORDER = (1, 2, 4, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)


def child_env() -> dict[str, str]:
    """Environment for sub-commands: the caller's, minus any active virtualenv."""
    env = {
        k: v for k, v in os.environ.items() if k not in {"VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"}
    }
    env["HALFLIFE_SHIP_CHECK"] = "1"
    return env


def one_line(text: str) -> str:
    """Collapse evidence to a single short line."""
    return re.sub(r"\s+", " ", text).strip()[:240]


def run_gates(ctx: Context) -> dict[int, GateResult]:
    """Evaluate every gate, turning unexpected errors into failures."""
    results: dict[int, GateResult] = {}
    for number in EVALUATION_ORDER:
        try:
            results[number] = GATES[number](ctx)
        except Exception as exc:  # a broken artifact must fail its gate, not crash the check
            results[number] = fail(f"error: {type(exc).__name__}: {exc}")
    return results


def main(argv: Sequence[str] | None = None) -> int:
    """Run the ship check and print the gate report."""
    parser = argparse.ArgumentParser(description="Check ship gates G1 to G14 on a fresh clone.")
    parser.add_argument(
        "--fast", action="store_true", help="skip slow steps; result does not count"
    )
    parser.add_argument("--keep", action="store_true", help="keep the temporary clone and logs")
    args = parser.parse_args(argv)

    repo = Path(git(REPO, "rev-parse", "--show-toplevel"))
    sha = git(repo, "rev-parse", "HEAD")
    original = original_checkout_problems(repo)

    tmp = Path(tempfile.mkdtemp(prefix="halflife-ship-check-"))
    logs = tmp / "logs"
    logs.mkdir()
    clone = tmp / "clone"
    try:
        subprocess.run(
            ["git", "clone", "--quiet", "--no-hardlinks", "--no-checkout", str(repo), str(clone)],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(clone), "checkout", "--quiet", "--detach", sha],
            check=True,
            capture_output=True,
        )
        ctx = Context(repo=repo, clone=clone, logs=logs, sha=sha, fast=args.fast, env=child_env())
        ctx.shared["original_problems"] = original
        results = run_gates(ctx)
    finally:
        if args.keep:
            print(f"kept {tmp}", file=sys.stderr)
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    print(f"HEAD {sha} fresh-clone")
    for number in sorted(results):
        result = results[number]
        print(f"G{number} {'PASS' if result.passed else 'FAIL'} {one_line(result.evidence)}")
    passed = sum(1 for r in results.values() if r.passed)
    if args.fast:
        print(f"SHIP-CHECK-FAST (does not count): {passed}/14")
    elif passed == len(GATES):
        print("SHIP-CHECK: PASS 14/14")
    else:
        print(f"SHIP-CHECK: FAIL {passed}/14")
    return 0 if passed == len(GATES) else 1


if __name__ == "__main__":
    sys.exit(main())

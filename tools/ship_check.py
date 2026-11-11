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
With ``--fast`` it skips slow steps (re-running ``make ci``, building the wheel,
and recomputing the E1 estimates, which G7's evidence then says) and the last
line is ``SHIP-CHECK-FAST (does not count): <passed>/14``.

The artifact layouts checked for G5 to G9 and G12 are documented in
docs/ARTIFACTS.md. Stdlib only, Python 3.11+. Benchmark template files are YAML,
so G5, G6, and G9 read them with the clone's own ``halflife.bench`` loader, run
as a sub-command in the clone's locked environment after ``make setup``; G7
reruns the clone's own ``halflife.e1_analysis`` the same way.
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
import zipfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeGuard

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
E1_RECOMPUTE_LIMIT_S = 1800.0
REVIEW_EXEMPT_PREFIXES = ("artifacts/review/",)
# The reviewer appends its non-blocking items to docs/REVIEW_LOG.md after reviewing
# (.claude/agents/reviewer.md), so the log may follow the reviewed commit (D-051).
REVIEW_EXEMPT_FILES = frozenset({"STATUS.md", "docs/LOG.md", "docs/REVIEW_LOG.md"})
REPORT_WORDS = (4000, 8000)
HYPE_WORDS = ("revolutionary", "groundbreaking", "state of the art", "state-of-the-art")
EM_DASH = "\u2014"
# The character itself, or an HTML entity that renders as one (named, decimal, hex).
EM_DASH_RE = re.compile(r"\u2014|&mdash;|&#0*8212(?![0-9]);?|&#[xX]0*2014(?![0-9A-Fa-f]);?")
# Upper-case markers are matched case-sensitively ("todo-cli" is a template name);
# placeholder words and phrases in any case.
PLACEHOLDER_RE = re.compile(
    r"\b(?:TODO|TBD|FIXME|XXX)\b|(?i:\b(?:pending|to\s+be\s+written|coming\s+soon)\b)"
)
# A first-screen interval: "95% CI" and the like, a named interval, or a bracketed
# pair of numbers together with the word CI or interval.
PERCENT_INTERVAL_RE = re.compile(
    r"\b\d{1,2}(?:\.\d+)?\s?%\s?(?:CI\b|(?:confidence\s+|credible\s+)?interval\b)", re.IGNORECASE
)
NAMED_INTERVAL_RE = re.compile(r"\b(?:confidence|credible)\s+interval\b", re.IGNORECASE)
_NUMBER = r"[-+]?\d+(?:\.\d+)?"
BRACKETED_PAIR_RE = re.compile(rf"[\[(]\s*{_NUMBER}\s*(?:,|;|to|-|\u2013)\s*{_NUMBER}\s*[\])]")
INTERVAL_WORD_RE = re.compile(r"\bCI\b|(?i:\binterval\b)")
# Same rule as halflife.integrity.prereg (make prereg-verify).
PREREG_PATH_RE = re.compile(r"prereg/\d{4}-[A-Za-z0-9][A-Za-z0-9_-]*\.md")
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
    """Shared state for one ship-check run.

    ``python`` is the command prefix that runs Python in the clone's locked
    environment (tests point it at their own interpreter).
    """

    repo: Path
    clone: Path
    logs: Path
    sha: str
    fast: bool
    env: dict[str, str]
    shared: dict[str, Any] = field(default_factory=dict)
    python: tuple[str, ...] = ("uv", "run", "--frozen", "--quiet", "python")

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


def is_count(value: object) -> TypeGuard[int]:
    """True for a non-negative int that is not a boolean."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def finite_nonnegative(value: object) -> bool:
    """True for a finite, non-negative int or float that is not a boolean."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(value) and value >= 0
    except OverflowError:  # an int too large for a float
        return False


def inside(path: Path, root: Path) -> bool:
    """True when ``path``, with symlinks followed, lies inside ``root`` (or is ``root``)."""
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def clone_file(clone: Path, rel: object) -> Path | None:
    """The path ``rel`` names in the clone, or None unless it is clean and stays inside.

    Clean means a relative POSIX path with no empty, ``.``, or ``..`` segment and no
    backslash; the resolved path (symlinks followed) must still lie inside ``clone``.
    """
    if not isinstance(rel, str) or not rel or "\\" in rel or "\x00" in rel:
        return None
    parts = rel.split("/")
    if any(part in ("", ".", "..") for part in parts):
        return None
    path = clone.joinpath(*parts)
    return path if inside(path, clone) else None


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


# ---------------------------------------------------------------- benchmark templates

TEMPLATE_FACTS_MARK = "HALFLIFE-TEMPLATE-FACTS "
# Runs in the clone's environment: loads each template file with halflife.bench and
# prints what the file itself declares, so no gate trusts the manifest for it.
TEMPLATE_LOADER = """\
import json, sys
from pathlib import Path
from halflife import bench
root = Path(sys.argv[1])
facts = {}
for rel in json.loads(sys.argv[2]):
    try:
        template = bench.load(root / rel).template
        facts[rel] = {
            "id": template.id,
            "split": template.split,
            "constraint_types": sorted({c.type for c in template.constraints}),
            "constraint_ids": sorted(c.id for c in template.constraints),
        }
    except Exception as exc:
        facts[rel] = {"error": type(exc).__name__ + ": " + str(exc)}
print("HALFLIFE-TEMPLATE-FACTS " + json.dumps(facts))
"""


@dataclass(frozen=True)
class TemplateFacts:
    """What a template file declares, as read by the clone's own loader.

    ``constraint_ids`` is empty when the loader reported none; G5 then fails, since
    every compaction log records retention for at least one constraint.
    """

    id: str
    split: str
    constraint_types: frozenset[str]
    constraint_ids: frozenset[str] = frozenset()


def template_facts(entry: Any) -> TemplateFacts | str:
    """Facts from one loader result, or a short reason the file did not load."""
    if not isinstance(entry, dict):
        return "no result from the template loader"
    if "error" in entry:
        return " ".join(str(entry["error"]).split())[:120]
    tid, split, types = entry.get("id"), entry.get("split"), entry.get("constraint_types")
    cids = entry.get("constraint_ids", [])
    if (
        not isinstance(tid, str)
        or not isinstance(split, str)
        or not isinstance(types, list)
        or not all(isinstance(t, str) for t in types)
        or not isinstance(cids, list)
        or not all(isinstance(c, str) for c in cids)
    ):
        return "template loader returned malformed facts"
    return TemplateFacts(tid, split, frozenset(types), frozenset(cids))


def load_templates(ctx: Context, rels: Sequence[str]) -> dict[str, TemplateFacts | str]:
    """Load template files (clean clone-relative paths) with the clone's ``halflife.bench``.

    Results are cached in ``ctx.shared`` so G6 and G9 share one loader run.
    """
    cache: dict[str, TemplateFacts | str] = ctx.shared.setdefault("template_facts", {})
    todo = sorted(set(rels) - set(cache))
    if todo:
        result = ctx.run(
            f"templates-{len(cache)}",
            [*ctx.python, "-I", "-c", TEMPLATE_LOADER, str(ctx.clone), json.dumps(todo)],
            timeout=300,
        )
        lines = [ln for ln in result.output.splitlines() if ln.startswith(TEMPLATE_FACTS_MARK)]
        parsed: Any = None
        if result.returncode == 0 and len(lines) == 1:
            try:
                parsed = json.loads(lines[0][len(TEMPLATE_FACTS_MARK) :])
            except ValueError:
                parsed = None
        for rel in todo:
            if isinstance(parsed, dict):
                cache[rel] = template_facts(parsed.get(rel))
            else:
                cache[rel] = f"template loader failed in the clone (rc={result.returncode})"
    return {rel: cache[rel] for rel in rels}


def bench_template_facts(
    ctx: Context, purpose: str = "held-out templates"
) -> tuple[dict[str, TemplateFacts] | None, str]:
    """Bench templates by id: the files the bench manifest lists, loaded in the clone.

    ``purpose`` names what the caller needs them for, in the evidence when the
    manifest is missing.
    """
    manifest = ctx.clone / "artifacts" / "bench" / "manifest.json"
    if not manifest.is_file():
        return None, f"missing artifacts/bench/manifest.json (needed for {purpose})"
    entries = load_json(manifest)["templates"]
    ids = [entry.get("id") for entry in entries]
    if len({str(i) for i in ids}) != len(ids):
        return None, "bench manifest has duplicate template ids"
    rels: list[str] = []
    for entry in entries:
        rel = entry.get("path")
        if clone_file(ctx.clone, rel) is None:
            return None, f"bench template {entry.get('id')}: path is not inside the clone"
        rels.append(str(rel))
    if (blocked := setup_failed(ctx)) is not None:
        return None, blocked.evidence
    loaded = load_templates(ctx, rels)
    out: dict[str, TemplateFacts] = {}
    for entry, rel in zip(entries, rels, strict=True):
        facts = loaded[rel]
        if isinstance(facts, str):
            return None, f"bench template {entry.get('id')} does not load ({facts})"
        if facts.id != entry.get("id"):
            return None, f"bench manifest id {entry.get('id')} names a file declaring {facts.id}"
        out[facts.id] = facts
    return out, ""


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
    """Recompute detector precision and recall with Wilson intervals from seeded cases.

    Every case has a unique ``case_id`` (so a repeated row cannot inflate a count)
    and boolean ``label`` and ``predicted`` values.
    """
    data = load_json(path)
    cases = data["cases"]
    if not isinstance(cases, list):
        return False, "detector cases are not a list"
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            return False, "detector case is not an object"
        cid = case.get("case_id")
        if not isinstance(cid, str) or not cid:
            return False, "detector case without a case_id"
        if cid in seen:
            return False, f"duplicate case_id {cid} in seeded cases"
        seen.add(cid)
        for name in ("label", "predicted"):
            if not isinstance(case.get(name), bool):
                return False, f"case {cid}: {name} is not a boolean"
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
    """Probe leakage must be zero across a non-empty set of validation runs.

    Each run has a ``run_id``, a boolean ``leaked`` (null means never checked), and a
    unique ``(run_id, session_id)``; the integer counts must match the runs.
    """
    data = load_json(path)
    runs = data["runs"]
    counts = (data.get("validation_runs"), data.get("leaks"))
    if not isinstance(runs, list) or not runs or not all(is_count(c) for c in counts):
        return False, "probe leakage artifact inconsistent or empty"
    seen: set[tuple[str, str]] = set()
    for run in runs:
        run_id = run.get("run_id") if isinstance(run, dict) else None
        if not isinstance(run_id, str) or not run_id:
            return False, "probe leakage run without a run_id"
        key = (run_id, str(run.get("session_id")))
        if key in seen:
            return False, f"duplicate probe leakage run {run_id}"
        seen.add(key)
        if not isinstance(run.get("leaked"), bool):
            return False, f"probe leakage run {run_id}: leaked is not a boolean"
    leaks = sum(1 for run in runs if run["leaked"])
    if data["validation_runs"] != len(runs) or data["leaks"] != leaks:
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
# Fields every compaction log file holds (halflife.experiment.LOG_FIELDS).
COMPACTION_LOG_FIELDS = (
    "event_id",
    "strategy",
    "tool_call_index",
    "scheduled",
    "tokens_before",
    "tokens_after",
    "before",
    "after",
    "summary",
    "constraint_text_retained",
)
# Fields an event row repeats from its log; when present they must agree.
COMPACTION_EVENT_LOG_FIELDS = (
    "event_id",
    "strategy",
    "tool_call_index",
    "scheduled",
    "tokens_before",
    "tokens_after",
    "constraint_text_retained",
)


def compaction_log_problems(log: Any) -> list[str]:
    """Ways a parsed compaction log falls short of complete.

    The rules of ``halflife.experiment.log_problems``: every field present, non-empty
    before and after contexts, the system prompt first after compaction, a summary
    unless the strategy is ``sliding_window``, and retention logged per constraint
    (here: a non-empty map of constraint id to boolean). Counts must be non-negative
    integers and ``scheduled`` a boolean.
    """
    if not isinstance(log, dict):
        return ["is not a JSON object"]
    problems = [f"lacks {name}" for name in COMPACTION_LOG_FIELDS if name not in log]
    if problems:
        return problems
    before, after = log["before"], log["after"]
    if not isinstance(before, list) or not before or not isinstance(after, list) or not after:
        problems.append("has an empty before or after context")
    elif not isinstance(after[0], dict) or after[0].get("role") != "system":
        problems.append("lost the system prompt")
    summary = log["summary"]
    if log["strategy"] != "sliding_window" and not (isinstance(summary, str) and summary.strip()):
        problems.append("has no summary text")
    retained = log["constraint_text_retained"]
    if (
        not isinstance(retained, dict)
        or not retained
        or not all(isinstance(v, bool) for v in retained.values())
    ):
        problems.append("does not log constraint text retention")
    if not all(isinstance(log[name], str) and log[name] for name in ("event_id", "strategy")):
        problems.append("has no event_id or strategy")
    if not all(
        is_count(log[name]) for name in ("tool_call_index", "tokens_before", "tokens_after")
    ):
        problems.append("has a count that is not a non-negative integer")
    if not isinstance(log["scheduled"], bool):
        problems.append("has a scheduled flag that is not a boolean")
    return problems


def check_compaction_logs(path: Path, root: Path) -> tuple[bool, str]:
    """Every compaction event has all fields and its own complete, banked log.

    A log path must be clean and inside ``root`` (the clone), the file must match
    its SHA-256, its content must be complete (:func:`compaction_log_problems`),
    and the event's copies of the log fields must agree with it.
    """
    data = load_json(path)
    events = data["events"]
    if not isinstance(events, list) or not events:
        return False, "no compaction events banked"
    if data.get("incomplete") not in (None, 0):
        return False, f"compaction log artifact counts {data.get('incomplete')!r} incomplete logs"
    seen_events: set[tuple[str, str, str]] = set()
    seen_logs: set[Path] = set()
    for event in events:
        if not isinstance(event, dict):
            return False, "compaction event is not an object"
        absent = [name for name in COMPACTION_FIELDS if event.get(name) is None]
        if absent:
            return False, f"compaction event {event.get('event_id')} missing {absent[0]}"
        eid = str(event["event_id"])
        key = (str(event["run_id"]), str(event.get("session_id")), eid)
        if key in seen_events:
            return False, f"duplicate compaction event {eid} in run {event['run_id']}"
        seen_events.add(key)
        rel = event["log_path"]
        log_file = clone_file(root, rel)
        if log_file is None:
            return False, f"compaction event {eid}: log_path is not a clean path inside the clone"
        if log_file.resolve() in seen_logs:
            return False, f"compaction log {rel} is shared by two events"
        seen_logs.add(log_file.resolve())
        if not log_file.is_file() or sha256_file(log_file) != event["log_sha256"]:
            return False, f"compaction log {rel} missing or hash mismatch"
        if event.get("complete", True) is not True or event.get("problems"):
            return False, f"compaction event {eid} is marked incomplete"
        try:
            log = json.loads(log_file.read_text(encoding="utf-8"))
        except ValueError:
            return False, f"compaction log {rel} is not valid JSON"
        problems = compaction_log_problems(log)
        if problems:
            return False, f"compaction log {rel} {problems[0]}"
        for name in COMPACTION_EVENT_LOG_FIELDS:
            if name in event and event[name] != log[name]:
                return False, f"compaction log {rel} disagrees with its event on {name}"
    return True, f"{len(events)} compaction logs complete"


def source_sessions(clone: Path, source: object) -> tuple[dict[tuple[str, str], list[Any]], str]:
    """Template ids by ``(run_id, session_id)`` in the banked experiment ``source`` names.

    ``source`` (the ``source`` field of compaction_logs.json, as
    ``halflife.experiment.e0_validation`` writes it) must be a clean path to a
    directory inside the clone whose ``sessions.jsonl`` lies inside the clone too.
    Returns an empty map and the reason when it does not.
    """
    if not isinstance(source, str) or not source:
        return {}, "compaction logs name no source (the banked experiment directory)"
    directory = clone_file(clone, source)
    if directory is None:
        return {}, f"compaction log source {source!r} is not a clean path inside the clone"
    sessions = clone_file(clone, f"{source}/sessions.jsonl")
    if not directory.is_dir() or sessions is None or not sessions.is_file():
        return {}, f"compaction log source {source} has no sessions.jsonl"
    templates: dict[tuple[str, str], list[Any]] = {}
    for row in load_jsonl(sessions):
        run_id, sid = row.get("run_id"), row.get("session_id")
        if run_id is not None and isinstance(sid, str) and sid:
            templates.setdefault((str(run_id), sid), []).append(row.get("template_id"))
    return templates, ""


def check_compaction_retention(ctx: Context, path: Path) -> tuple[bool, str]:
    """Each compaction log records retention for exactly its session's template rules.

    The session of each event (its ``run_id`` and ``session_id``) is looked up in the
    ``sessions.jsonl`` of the experiment directory the file names as ``source``; the
    template id found there is resolved through the bench manifest and loaded from the
    template file with the clone's ``halflife.bench`` (the loader G6 uses). The log's
    ``constraint_text_retained`` must have exactly that template's constraint ids.
    Run after :func:`check_compaction_logs` has passed.
    """
    data = load_json(path)
    sessions, why = source_sessions(ctx.clone, data.get("source"))
    if not sessions:
        return False, why or f"compaction log source {data['source']} lists no sessions"
    bench, why = bench_template_facts(ctx, "compaction retention")
    if bench is None:
        return False, why
    source = data["source"]
    used: set[str] = set()
    for event in data["events"]:
        eid, run_id, sid = event["event_id"], event["run_id"], event.get("session_id")
        if not isinstance(sid, str) or not sid:
            return False, f"compaction event {eid} in run {run_id} names no session_id"
        found = sessions.get((str(run_id), sid))
        if not found:
            return False, (
                f"compaction event {eid}: session {sid} of run {run_id} "
                f"is not in {source}/sessions.jsonl"
            )
        tid = found[0]
        if not isinstance(tid, str) or not tid or any(other != tid for other in found):
            return False, f"session {sid} of run {run_id} names no single template in {source}"
        facts = bench.get(tid)
        if facts is None:
            return False, f"template {tid} of session {sid} is not in the bench manifest"
        if not facts.constraint_ids:
            return False, f"template {tid}: the template loader reported no constraint ids"
        rel = event["log_path"]
        log_file = clone_file(ctx.clone, rel)
        if log_file is None:  # check_compaction_logs rejects this before
            return False, f"compaction event {eid}: log_path is not a clean path inside the clone"
        logged = set(json.loads(log_file.read_text(encoding="utf-8"))["constraint_text_retained"])
        absent = sorted(facts.constraint_ids - logged)
        if absent:
            return False, f"compaction log {rel} does not log retention of {absent[0]} ({tid})"
        extra = sorted(logged - facts.constraint_ids)
        if extra:
            return False, f"compaction log {rel} logs retention of {extra[0]}, not a rule of {tid}"
        used.add(tid)
    return True, f"retention logged for every rule ({len(used)} templates)"


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
    ok, retention = check_compaction_retention(ctx, files[2])
    if not ok:
        return fail(retention)
    return GateResult(True, "; ".join([*(detail for _, detail in results), retention]))


# ---------------------------------------------------------------- G6


def bench_records_problem(clone: Path, template: dict[str, Any]) -> str | None:
    """Why a template's compliant route or reference run is not verified, or None.

    Both paths must be clean and inside the clone, and they must be two different
    records (different files with different content).
    """
    tid = template["id"]
    route_path = clone_file(clone, template.get("compliant_route"))
    ref_path = clone_file(clone, template.get("reference_run"))
    if route_path is None or ref_path is None:
        return "compliant route or reference run path is not a clean path inside the clone"
    if not route_path.is_file() or not ref_path.is_file():
        return "compliant route or reference run not banked"
    if route_path.resolve() == ref_path.resolve() or sha256_file(route_path) == sha256_file(
        ref_path
    ):
        return "compliant route and reference run are the same record"
    route, ref = load_json(route_path), load_json(ref_path)
    for name, record in (("compliant route", route), ("reference run", ref)):
        if (
            not isinstance(record, dict)
            or record.get("template_id") != tid
            or record.get("template_sha256") != template["sha256"]
        ):
            return f"{name} is for a different template version"
    total, passed, violations = (
        route.get(k) for k in ("tasks_total", "tasks_passed", "violations")
    )
    if not (is_count(total) and total > 0 and is_count(passed) and passed == total):
        return "compliant route not verified"
    if not is_count(violations) or violations != 0:
        return "compliant route not verified"
    if ref.get("completed") is not True:
        return "reference run incomplete"
    return action_log_problem(clone, tid, ref)


# The fields halflife.schema.ActionLog.to_dict writes, with their JSON types.
ACTION_LOG_FIELDS: dict[str, type] = {
    "session_id": str,
    "template_id": str,
    "constraints": list,
    "baseline_fs": dict,
    "blobs": dict,
    "events": list,
    "checks": list,
    "spans": list,
}
# Counts a reference-run record repeats from its action log (halflife.bench.verify.bank).
REFERENCE_LOG_COUNTS = (("n_tool_calls", "events"), ("n_subtasks", "spans"))
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def action_log_problem(clone: Path, tid: object, ref: dict[str, Any]) -> str | None:
    """Why a reference run's action log is not the banked log of a run of template ``tid``.

    The record names ``action_log`` and ``action_log_sha256`` (as
    ``halflife.bench.verify.bank`` writes them): the path must be clean and inside the
    clone (symlinks followed), and the file must have that SHA-256. The file must be a
    JSON object in the ``halflife.schema.ActionLog`` layout whose ``template_id`` is the
    template's id, whose ``session_id`` is a non-empty string, and which records at
    least one tool call and one subtask span. Counts the record repeats from the log
    (``n_tool_calls``, ``n_subtasks``), when present, must equal the log's.
    """
    rel = ref.get("action_log")
    if rel is None:
        return "reference run names no action_log"
    log_path = clone_file(clone, rel)
    if log_path is None:
        return "reference run action_log is not a clean path inside the clone"
    digest = ref.get("action_log_sha256")
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        return "reference run names no action_log_sha256"
    if not log_path.is_file() or sha256_file(log_path) != digest:
        return "reference run action log missing or hash mismatch"
    try:
        log = load_json(log_path)
    except ValueError:
        return "reference run action log is not valid JSON"
    if not isinstance(log, dict):
        return "reference run action log is not a JSON object"
    for name, kind in ACTION_LOG_FIELDS.items():
        if not isinstance(log.get(name), kind):
            return f"reference run action log lacks {name} (not an action log)"
    if log["template_id"] != tid:
        return f"reference run action log is a session of template {log['template_id']!r}"
    if not log["session_id"]:
        return "reference run action log names no session"
    if not log["events"] or not log["spans"]:
        return "reference run action log records no tool calls or no subtask spans"
    for count, name in REFERENCE_LOG_COUNTS:
        if count in ref and (not is_count(ref[count]) or ref[count] != len(log[name])):
            return (
                f"reference run record says {count}={ref[count]!r} "
                f"but its action log has {len(log[name])} {name}"
            )
    return None


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
    sources: list[Path] = []
    for template in templates:
        source = clone_file(ctx.clone, template.get("path"))
        if source is None:
            return fail(f"template {template['id']}: path is not a clean path inside the clone")
        sources.append(source)
    if len({source.resolve() for source in sources}) != len(sources):
        return fail("duplicate template paths: one file listed under several ids")
    if len({str(t.get("sha256")) for t in templates}) != len(templates):
        return fail("duplicate template sha256 values: one template counted twice")
    if (blocked := setup_failed(ctx)) is not None:
        return blocked
    facts = load_templates(ctx, [str(t["path"]) for t in templates])
    types: set[str] = set()
    for template, source in zip(templates, sources, strict=True):
        tid = template["id"]
        if not source.is_file() or sha256_file(source) != template["sha256"]:
            return fail(f"template {tid}: file missing or hash mismatch")
        declared = template["constraint_types"]
        if not isinstance(declared, list) or not declared or not set(declared) <= CONSTRAINT_TYPES:
            return fail(f"template {tid}: unknown or empty constraint types")
        loaded = facts[str(template["path"])]
        if isinstance(loaded, str):
            return fail(f"template {tid}: file does not load ({loaded})")
        if loaded.id != tid:
            return fail(f"template {tid}: file declares id {loaded.id!r}")
        if not loaded.constraint_types or not loaded.constraint_types <= CONSTRAINT_TYPES:
            return fail(f"template {tid}: file has unknown or no constraint types")
        if loaded.constraint_types != set(declared):
            return fail(f"template {tid}: manifest constraint types differ from the template file")
        types |= loaded.constraint_types
        problem = bench_records_problem(ctx.clone, template)
        if problem:
            return fail(f"template {tid}: {problem}")
    if len(types) < MIN_CONSTRAINT_TYPES:
        return fail(f"{len(types)} constraint types < {MIN_CONSTRAINT_TYPES}")
    return GateResult(
        True,
        f"{len(ids)} templates, {len(types)} constraint types (read from the template files), "
        "routes and reference runs with their action logs verified",
    )


# ---------------------------------------------------------------- G7

# Every E1 file the gate reads; each must be first committed after the preregistration.
E1_FILES = (
    "sessions.jsonl",
    "opportunities.jsonl",
    "provenance.json",
    "hazard.json",
    "half_life.json",
    "km.json",
)


def check_interval(record: dict[str, Any], key: str) -> bool:
    """True when ``lo <= record[key] <= hi`` with finite numbers."""
    try:
        lo, value, hi = float(record["lo"]), float(record[key]), float(record["hi"])
    except (KeyError, TypeError, ValueError):
        return False
    return all(math.isfinite(x) for x in (lo, value, hi)) and lo <= value <= hi


def check_half_life(record: dict[str, Any]) -> bool:
    """A half-life estimate with a valid interval, in one of three forms.

    * finite: ``lo <= half_life <= hi``, all finite;
    * open upper bound: finite ``lo <= half_life``, ``hi`` null, and an integer
      ``hi_not_reached_within`` at least ``half_life`` (some bootstrap draws never
      reached 0.5 within that many tool calls; DECISIONS.md D-028);
    * not reached: ``half_life`` null and a positive integer ``not_reached_within``.
    """
    within = record.get("not_reached_within")
    if record.get("half_life") is None:
        return isinstance(within, int) and not isinstance(within, bool) and within > 0
    open_hi = record.get("hi_not_reached_within")
    if record.get("hi") is None and isinstance(open_hi, int) and not isinstance(open_hi, bool):
        try:
            lo, value = float(record["lo"]), float(record["half_life"])
        except (KeyError, TypeError, ValueError):
            return False
        return math.isfinite(lo) and math.isfinite(value) and lo <= value <= open_hi
    return check_interval(record, "half_life")


def check_prereg_link(clone: Path, prereg: str, artifact: str) -> str | None:
    """Problem text if ``artifact`` does not cite a preregistration committed strictly before it.

    The preregistration must be a regular file at a ``prereg/NNNN-slug.md`` path
    (the naming rule of ``make prereg-verify``), so ``..`` segments and
    ``prereg/.gitkeep`` do not count.
    """
    if not prereg.startswith("prereg/") or not (clone / prereg).is_file():
        return f"{artifact} cites missing preregistration {prereg!r}"
    if not PREREG_PATH_RE.fullmatch(prereg):
        return f"{artifact} cites {prereg!r}, which is not a prereg/NNNN-slug.md path"
    if (clone / prereg).is_symlink():
        return f"{artifact} cites {prereg!r}, a symlink rather than a preregistration file"
    pre_commit = first_commit(clone, prereg)
    art_commit = first_commit(clone, artifact)
    if pre_commit is None or art_commit is None:
        return f"{artifact} or its preregistration is not committed"
    if not strictly_predates(clone, pre_commit, art_commit):
        return f"{prereg} does not predate {artifact}"
    return None


GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")


def run_provenance_problem(
    clone: Path, pre_commit: str, row: dict[str, Any], contains: dict[str, bool]
) -> str | None:
    """Problem text unless a confirmatory session ran after its preregistration.

    ``halflife.runner.save_session`` stamps every session with ``run_provenance``
    (DECISIONS.md D-050, item 6). The stamp must name a commit that is the
    preregistration's commit or descends from it, and a clean checkout
    (``git_dirty`` false), so the session ran under code that already held the frozen
    preregistration. A commit the clone does not have fails. ``contains`` caches the
    ancestry answer per commit.
    """
    sid = row.get("session_id")
    stamp = row.get("run_provenance")
    if not isinstance(stamp, dict):
        return f"session {sid} has no run_provenance stamp"
    sha = stamp.get("git_sha")
    if not isinstance(sha, str) or not GIT_SHA_RE.fullmatch(sha):
        return f"session {sid}: run_provenance names no git commit"
    if stamp.get("git_dirty") is not False:
        return f"session {sid} did not run from a clean checkout (git_dirty must be false)"
    if sha not in contains:
        contains[sha] = sha == pre_commit or strictly_predates(clone, pre_commit, sha)
    if not contains[sha]:
        return f"session {sid} ran at {sha[:12]}, which does not contain the preregistration commit"
    return None


# Analysis outputs G7 recomputes from the banked rows (halflife.e1_analysis writes them).
E1_RECOMPUTED = ("hazard.json", "half_life.json", "km.json")
E1_ROWS = ("sessions.jsonl", "opportunities.jsonl")


def analysis_settings(hazard: Any) -> tuple[int, int] | None:
    """``(n_boot, seed)`` from the ``settings`` the banked hazard.json records (D-029).

    None unless ``n_boot`` is a positive integer and ``seed`` an integer (booleans are
    not integers here).
    """
    settings = hazard.get("settings") if isinstance(hazard, dict) else None
    if not isinstance(settings, dict):
        return None
    n_boot, seed = settings.get("n_boot"), settings.get("seed")
    if not is_count(n_boot) or n_boot == 0 or isinstance(seed, bool) or not isinstance(seed, int):
        return None
    return n_boot, seed


def json_difference(banked: Any, recomputed: Any, where: str = "") -> str | None:
    """Where two parsed JSON values first differ, or None when they are the same value.

    Object keys are visited in sorted order and array items by index; the result is a
    path such as ``estimates[2].hi``. Types must match too, so ``1``, ``1.0``, and
    ``true`` all differ from each other, and NaN differs from everything.
    """
    here = where or "(top level)"
    if type(banked) is not type(recomputed):
        return here
    if isinstance(banked, dict):
        for key in sorted(set(banked) | set(recomputed), key=str):
            path = f"{where}.{key}" if where else str(key)
            if key not in banked or key not in recomputed:
                return path
            found = json_difference(banked[key], recomputed[key], path)
            if found is not None:
                return found
        return None
    if isinstance(banked, list):
        for index, (a, b) in enumerate(zip(banked, recomputed, strict=False)):
            found = json_difference(a, b, f"{where}[{index}]")
            if found is not None:
                return found
        if len(banked) != len(recomputed):
            return f"{where}[{min(len(banked), len(recomputed))}]"
        return None
    return None if banked == recomputed else here


def scratch_parent(ctx: Context) -> Path | None:
    """Parent for scratch directories: the log directory unless it lies inside the clone.

    None means the system temporary directory (tests keep their logs inside the clone).
    """
    return None if inside(ctx.logs, ctx.clone) else ctx.logs


PREREG_BLOCK = re.compile(r"^```json[ \t]*\n(.*?)^```[ \t]*$", re.MULTILINE | re.DOTALL)


def prereg_settings(text: str) -> dict[str, Any] | None:
    """The ``halflife_prereg`` object of the first fenced JSON block that has one, or None."""
    for match in PREREG_BLOCK.finditer(text):
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("halflife_prereg"), dict):
            block: dict[str, Any] = data["halflife_prereg"]
            return block
    return None


def prereg_settings_problem(ctx: Context, prereg: str, base: Path) -> str | None:
    """Why the banked E1 analysis settings differ from the preregistered ones, or None.

    The preregistration fixes the analysis module, ``n_boot``, and ``seed`` in a fenced
    JSON block ``{"halflife_prereg": {...}}``; hazard.json must record the same values.
    """
    path = ctx.clone / prereg
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    fixed = prereg_settings(text)
    if fixed is None:
        return f"{prereg} has no halflife_prereg settings block"
    if fixed.get("analysis") != "halflife.e1_analysis":
        return f"{prereg} names analysis {fixed.get('analysis')}, not halflife.e1_analysis"
    settings = analysis_settings(load_json(base / "hazard.json"))
    if settings is None:
        return None  # malformed settings: recompute_e1 reports them
    for key, value in zip(("n_boot", "seed"), settings, strict=True):
        want = fixed.get(key)
        if not isinstance(want, int) or isinstance(want, bool) or want != value:
            return f"hazard.json settings {key} {value} != {want!r} fixed in {prereg}"
    return None


def recompute_e1(ctx: Context, base: Path) -> str | None:
    """Problem text unless rerunning the E1 analysis on the banked rows reproduces its outputs.

    Copies ``sessions.jsonl`` and ``opportunities.jsonl`` into a temporary directory
    outside the clone, runs ``python -m halflife.e1_analysis --dir <tmp>`` there in the
    clone's environment with the ``n_boot`` and ``seed`` that hazard.json records, and
    requires hazard.json, half_life.json, and km.json to equal the banked files as
    parsed JSON (:func:`json_difference`). The banked files are read before the rerun,
    and nothing is written inside the clone.
    """
    banked = {name: load_json(base / name) for name in E1_RECOMPUTED}
    settings = analysis_settings(banked["hazard.json"])
    if settings is None:
        return "hazard.json does not record the analysis settings (integer n_boot > 0 and seed)"
    if (blocked := setup_failed(ctx)) is not None:
        return blocked.evidence
    n_boot, seed = settings
    with tempfile.TemporaryDirectory(prefix="e1-recompute-", dir=scratch_parent(ctx)) as tmp:
        work = Path(tmp)
        if inside(work, ctx.clone):
            return "no temporary directory outside the clone for the E1 recomputation"
        for name in E1_ROWS:
            shutil.copyfile(base / name, work / name)
        result = ctx.run(
            "e1-recompute",
            [
                *ctx.python,
                "-I",
                "-m",
                "halflife.e1_analysis",
                "--dir",
                str(work),
                "--n-boot",
                str(n_boot),
                "--seed",
                str(seed),
            ],
            timeout=E1_RECOMPUTE_LIMIT_S,
        )
        if result.returncode != 0:
            lines = [line.strip() for line in result.output.splitlines() if line.strip()]
            last = lines[-1][:100] if lines else "no output"
            return f"halflife.e1_analysis failed on the banked rows: rc={result.returncode} {last}"
        for name in E1_RECOMPUTED:
            produced = work / name
            if not produced.is_file():
                return f"halflife.e1_analysis wrote no {name}"
            where = json_difference(banked[name], load_json(produced))
            if where is not None:
                return f"recomputed {name} differs from the banked file at {where}"
    return None


def gate_g7(ctx: Context) -> GateResult:
    base = ctx.clone / "artifacts" / "e1"
    absent = missing([base / n for n in E1_FILES], ctx.clone)
    if absent:
        return fail(f"missing {', '.join(absent)}")
    for name in E1_FILES:
        if clone_file(ctx.clone, f"artifacts/e1/{name}") is None:
            return fail(f"artifacts/e1/{name} resolves to a file outside the clone")
    provenance = load_json(base / "provenance.json")
    prereg = str(provenance.get("prereg", ""))
    for name in E1_FILES:
        problem = check_prereg_link(ctx.clone, prereg, f"artifacts/e1/{name}")
        if problem:
            return fail(problem)
    pre_commit = first_commit(ctx.clone, prereg)
    if pre_commit is None:  # pragma: no cover - check_prereg_link already required it
        return fail(f"{prereg} is not committed")
    contains: dict[str, bool] = {}
    sessions = load_jsonl(base / "sessions.jsonl")
    reached: dict[str, int] = {}
    keys: set[tuple[str, str]] = set()
    claimed: set[tuple[str, str]] = set()
    for row in sessions:
        sid = row.get("session_id")
        if row.get("prereg") != provenance["prereg"] or row.get("confirmatory") is not True:
            return fail(f"session {sid} not tagged with the E1 preregistration")
        problem = run_provenance_problem(ctx.clone, pre_commit, row, contains)
        if problem:
            return fail(problem)
        family = row.get("family")
        if not isinstance(sid, str) or not sid or not isinstance(family, str) or not family:
            return fail(f"session {sid} lacks a session_id or model family")
        # Every family runs the same plans, so a session is one plan run by one family.
        key = (family, sid)
        if key in keys:
            return fail(f"duplicate session rows for {sid} ({family})")
        keys.add(key)
        if not isinstance(row.get("reached_first_dp"), bool):
            return fail(f"session {sid}: reached_first_dp is not a boolean")
        reached.setdefault(family, 0)
        if row["reached_first_dp"]:
            reached[family] += 1
            claimed.add(key)
    if len(reached) < MIN_E1_FAMILIES:
        return fail(f"{len(reached)} model families < {MIN_E1_FAMILIES}")
    short = {f: n for f, n in reached.items() if n < MIN_E1_SESSIONS_PER_FAMILY}
    if short:
        return fail(f"families below {MIN_E1_SESSIONS_PER_FAMILY} sessions reaching a DP: {short}")
    opportunities = load_jsonl(base / "opportunities.jsonl")
    seen = {(str(o.get("family")), str(o.get("session_id"))) for o in opportunities}
    if not opportunities or not seen <= keys:
        return fail("opportunities.jsonl empty or references unknown sessions")
    if seen - claimed:
        return fail(
            f"{len(seen - claimed)} session(s) have opportunities but reached_first_dp false"
        )
    if claimed - seen:
        return fail(
            f"{len(claimed - seen)} session(s) claim reached_first_dp with no opportunities"
        )
    for name in ("hazard.json", "half_life.json"):
        estimates = load_json(base / name)["estimates"]
        families = {e["family"] for e in estimates}
        if not set(reached) <= families:
            return fail(f"{name} lacks estimates for {sorted(set(reached) - families)}")
        for est in estimates:
            ok = check_interval(est, "estimate") if name == "hazard.json" else check_half_life(est)
            if not ok:
                return fail(f"{name}: estimate for {est.get('family')} lacks a valid interval")
    km = load_json(base / "km.json")
    if km.get("descriptive") is not True or not km.get("curves"):
        return fail("km.json missing or not labeled descriptive")
    counts = ", ".join(f"{f}={n}" for f, n in sorted(reached.items()))
    summary = f"E1 preregistered; sessions reaching DP {counts}; estimates with intervals"
    if ctx.fast:
        return GateResult(True, f"{summary}; not recomputed (fast)")
    problem = prereg_settings_problem(ctx, prereg, base) or recompute_e1(ctx, base)
    if problem:
        return fail(problem)
    return GateResult(True, f"{summary}, recomputed from the banked rows")


# ---------------------------------------------------------------- G8 and G9


def arm_summary(clone: Path, arm: str) -> tuple[dict[str, Any] | None, str]:
    """Load and sanity-check ``artifacts/<arm>/summary.json`` (``n_sessions`` an integer)."""
    path = clone / "artifacts" / arm.lower() / "summary.json"
    if not path.is_file():
        return None, f"missing artifacts/{arm.lower()}/summary.json"
    data = load_json(path)
    n_sessions = data.get("n_sessions") if isinstance(data, dict) else None
    if (
        not isinstance(data, dict)
        or data.get("arm") != arm
        or data.get("ran") is not True
        or not is_count(n_sessions)
        or n_sessions <= 0
    ):
        return None, f"{arm} summary does not show a completed run"
    return data, ""


def arm_sessions(
    clone: Path, arm: str, summary: dict[str, Any]
) -> tuple[list[dict[str, Any]] | None, str]:
    """Banked ``artifacts/<arm>/sessions.jsonl`` rows; their count must equal ``n_sessions``.

    Rows are unique by ``(session_id, model)``: arms may run one plan per model.
    """
    rel = f"artifacts/{arm.lower()}/sessions.jsonl"
    path = clone / rel
    if not path.is_file():
        return None, f"missing {rel}"
    rows = load_jsonl(path)
    keys: set[tuple[str, str]] = set()
    for row in rows:
        sid = row.get("session_id")
        if not isinstance(sid, str) or not sid:
            return None, f"{rel} has a row without a session_id"
        key = (sid, str(row.get("model")))
        if key in keys:
            return None, f"{rel} has duplicate session rows ({sid})"
        keys.add(key)
    if len(rows) != summary["n_sessions"]:
        return None, f"{arm} n_sessions {summary['n_sessions']} != {len(rows)} banked session rows"
    return rows, ""


def report_mentions(clone: Path, *needles: str) -> list[str]:
    """Needles that docs/REPORT.md does not mention as whole tokens.

    A token is not part of a longer word, number, or hyphenated compound, so
    'E20' does not mention E2 and 'non-ecological' does not mention ecological.
    """
    path = clone / "docs" / "REPORT.md"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    return [
        n
        for n in needles
        if not re.search(rf"(?<![A-Za-z0-9_-]){re.escape(n)}(?![A-Za-z0-9_-])", text)
    ]


# The sentence halflife.report writes for an arm with no banked results names the arm
# (for example "E3 has not run, so ..."), so mentioning the arm is not reporting it.
NOT_RUN_SENTENCE = r"\b{arm}\b[^.\n]*\bhas not run\b"


def report_says_not_run(clone: Path, *arms: str) -> list[str]:
    """Arms that docs/REPORT.md still describes with a "has not run" sentence."""
    path = clone / "docs" / "REPORT.md"
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    return [arm for arm in arms if re.search(NOT_RUN_SENTENCE.format(arm=arm), text)]


def gate_g8(ctx: Context) -> GateResult:
    e2, why = arm_summary(ctx.clone, "E2")
    if e2 is None:
        return fail(why)
    if e2.get("label") != "ecological":
        return fail("E2 not labeled ecological")
    models = e2.get("models", [])
    if (
        not isinstance(models, list)
        or not all(isinstance(m, str) and m for m in models)
        or len(set(models)) < MIN_E2_MODELS
    ):
        return fail(f"E2 used fewer than {MIN_E2_MODELS} model tiers")
    rows, why = arm_sessions(ctx.clone, "E2", e2)
    if rows is None:
        return fail(why)
    if any(row.get("label") != "ecological" for row in rows):
        return fail("E2 session rows not all labeled ecological")
    if {row.get("model") for row in rows} != set(models):
        return fail("E2 session rows do not match the summary's model tiers")
    e3, why = arm_summary(ctx.clone, "E3")
    if e3 is None:
        return fail(why)
    e3_rows, why = arm_sessions(ctx.clone, "E3", e3)
    if e3_rows is None:
        return fail(why)
    absent = report_mentions(ctx.clone, "E2", "E3", "ecological")
    if absent:
        return fail(f"docs/REPORT.md does not report {', '.join(absent)}")
    stale = report_says_not_run(ctx.clone, "E2", "E3")
    if stale:
        return fail(f"docs/REPORT.md still says {' and '.join(stale)} has not run (stale report)")
    return GateResult(
        True,
        f"E2 n={e2['n_sessions']} (ecological), E3 n={e3['n_sessions']}, rows banked, reported",
    )


COST_FIELDS = ("tokens", "latency_s", "task_success")


def held_out_problem(ctx: Context, held_out: Sequence[str]) -> str | None:
    """Why E4's held-out templates are not held out, or None.

    Each must be a bench template whose file declares the ``heldout`` split, and
    none may appear as a ``template_id`` in the E1 sessions.
    """
    bench, why = bench_template_facts(ctx)
    if bench is None:
        return why
    sessions = ctx.clone / "artifacts" / "e1" / "sessions.jsonl"
    if not sessions.is_file():
        return "missing artifacts/e1/sessions.jsonl (needed for held-out templates)"
    used: set[str] = set()
    for row in load_jsonl(sessions):
        tid = row.get("template_id")
        if not isinstance(tid, str) or not tid:
            return "E1 session rows lack template_id, so held-out templates cannot be checked"
        used.add(tid)
    for tid in held_out:
        facts = bench.get(tid)
        if facts is None:
            return f"E4 held-out template {tid} is not in the bench"
        if tid in used:
            return f"E4 held-out template {tid} was used in E1"
        if facts.split != "heldout":
            return f"E4 held-out template {tid} is not in the heldout split"
    return None


def has_install_heading(text: str) -> bool:
    """True when a Markdown heading outside code fences has a word starting 'install'."""
    fenced = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced and re.match(r"#{1,6} .*\binstall", line, re.IGNORECASE):
            return True
    return False


def e4_rows_problem(
    ctx: Context, e4: dict[str, Any], held_out: Sequence[str], mitigations: Sequence[Any]
) -> str | None:
    """Why the banked E4 session rows do not back the summary, or None.

    ``artifacts/e4/sessions.jsonl`` must hold ``n_sessions`` unique rows, their
    templates must be exactly the held-out templates, every reported mitigation must
    have rows, and every row's arm must be a reported mitigation or the ``none`` control.
    """
    rows, why = arm_sessions(ctx.clone, "E4", e4)
    if rows is None:
        return why
    templates = {row.get("template_id") for row in rows}
    if templates != set(held_out):
        return "E4 session rows' templates differ from held_out_templates"
    names = {m.get("name") for m in mitigations if isinstance(m, dict)}
    arms = {row.get("mitigation") for row in rows}
    for name in sorted(str(n) for n in names):
        if name not in arms:
            return f"E4 reports mitigation {name} with no banked session rows"
    for arm in sorted(str(a) for a in arms):
        if arm not in names and arm != "none":
            return f"E4 session rows name mitigation {arm}, which the summary does not report"
    return None


def gate_g9(ctx: Context) -> GateResult:
    e4, why = arm_summary(ctx.clone, "E4")
    if e4 is None:
        return fail(why)
    held_out = e4.get("held_out_templates")
    if not held_out:
        return fail("E4 does not list held-out templates")
    if (
        not isinstance(held_out, list)
        or not all(isinstance(t, str) and t for t in held_out)
        or len(set(held_out)) != len(held_out)
    ):
        return fail("E4 held_out_templates is not a list of distinct template ids")
    mitigations = e4.get("mitigations")
    if not isinstance(mitigations, list) or not mitigations:
        return fail("E4 reports no mitigations")
    for mitigation in mitigations:
        record = mitigation if isinstance(mitigation, dict) else {}
        cost = record.get("cost")
        bad = [
            k
            for k in COST_FIELDS
            if not isinstance(cost, dict) or not finite_nonnegative(cost.get(k))
        ]
        if bad:
            return fail(
                f"E4 mitigation {record.get('name')} lacks cost beside benefit: "
                f"{bad[0]} is not a finite number >= 0"
            )
    problem = held_out_problem(ctx, held_out)
    if problem:
        return fail(problem)
    problem = e4_rows_problem(ctx, e4, held_out, mitigations)
    if problem:
        return fail(problem)
    pyproject = tomllib.loads((ctx.clone / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject.get("project", {}).get("scripts", {})
    tools = ("halflife-guard", "halflife-pin")
    unpackaged = [t for t in tools if t not in scripts]
    if unpackaged:
        return fail(f"not packaged as console scripts: {', '.join(unpackaged)}")
    docs = ctx.clone / "docs" / "MITIGATIONS.md"
    text = docs.read_text(encoding="utf-8") if docs.is_file() else ""
    if not has_install_heading(text) or not all(t in text for t in tools):
        return fail("docs/MITIGATIONS.md lacks install docs for halflife-guard and halflife-pin")
    overhead_path = ctx.clone / "artifacts" / "e4" / "overhead.json"
    if not overhead_path.is_file():
        return fail("missing artifacts/e4/overhead.json")
    overhead = load_json(overhead_path)
    for tool in tools:
        entry = overhead.get(tool) if isinstance(overhead, dict) else None
        if (
            not isinstance(entry, dict)
            or not finite_nonnegative(entry.get("latency_ms"))
            or not finite_nonnegative(entry.get("tokens"))
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


def console_scripts(text: str) -> set[str]:
    """Names declared under ``[console_scripts]`` in an ``entry_points.txt``.

    The section header must be exactly ``[console_scripts]`` (importlib.metadata does
    not strip spaces inside the brackets), and keys start in column 0 (indented lines
    are continuations, as in INI files).
    """
    names: set[str] = set()
    section = ""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", ";")):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped[1:-1]
            continue
        match = re.match(r"([^\s=][^=]*?)\s*=", line)
        if section == "console_scripts" and match:
            names.add(match.group(1))
    return names


def entry_points_ok(dist: Path, tools: Sequence[str]) -> bool:
    """True when a built wheel declares every tool under ``[console_scripts]``."""
    for wheel in sorted(dist.glob("*.whl")):
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                if name.endswith(".dist-info/entry_points.txt"):
                    scripts = console_scripts(archive.read(name).decode("utf-8"))
                    return all(t in scripts for t in tools)
    return False


# ---------------------------------------------------------------- G10 and G11


def make_targets_pass(ctx: Context, targets: Sequence[str]) -> GateResult:
    """Run make targets in the clone; pass only if all succeed."""
    if (blocked := setup_failed(ctx)) is not None:
        return blocked
    outcomes = []
    for target in targets:
        result = ctx.run(target, ["make", target], timeout=900)
        lines = [ln for ln in result.output.splitlines() if ln.strip() and "make: ***" not in ln]
        outcomes.append((target, result.returncode, (lines[-1] if lines else "")[:80]))
    bad = [f"{t} rc={rc} ({line})" for t, rc, line in outcomes if rc != 0]
    if bad:
        return fail("; ".join(bad))
    return GateResult(True, "; ".join(f"make {t} ok" for t, _, _ in outcomes))


def gate_g10(ctx: Context) -> GateResult:
    absent = missing([ctx.clone / "README.md", ctx.clone / "docs" / "REPORT.md"], ctx.clone)
    if absent:
        return fail(f"missing {', '.join(absent)}")
    return make_targets_pass(ctx, ["claimcheck", "freshness", "report-check"])


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
    blocking = review.get("blocking")
    if not isinstance(blocking, list):
        return fail("review has no blocking list")
    if blocking:
        return fail(f"{len(blocking)} open blocking issue(s)")
    log_path = ctx.clone / "docs" / "REVIEW_LOG.md"
    if not log_path.is_file():
        return fail("missing docs/REVIEW_LOG.md for non-blocking issues")
    non_blocking = review.get("non_blocking", [])
    if not isinstance(non_blocking, list):
        return fail("review has no non_blocking list")
    log_text = " ".join(log_path.read_text(encoding="utf-8").split())
    unlisted = [str(item) for item in non_blocking if " ".join(str(item).split()) not in log_text]
    if unlisted:
        return fail(f"non-blocking issue not listed in docs/REVIEW_LOG.md: {unlisted[0][:60]}")
    if reviewed != ctx.sha:
        if not strictly_predates(ctx.clone, reviewed, ctx.sha):
            return fail("reviewed commit is not an ancestor of HEAD")
        changed = git(
            ctx.clone, "diff", "--name-only", "--no-renames", reviewed, ctx.sha
        ).splitlines()
        outside = [
            p
            for p in changed
            if not p.startswith(REVIEW_EXEMPT_PREFIXES) and p not in REVIEW_EXEMPT_FILES
        ]
        if outside:
            return fail(f"code changed since review: {', '.join(outside[:3])}")
    tool = ctx.clone / "tools" / "review.py"
    if not tool.is_file():
        return fail("missing tools/review.py (make review)")
    try:
        proc = subprocess.run(
            [sys.executable, "-I", str(tool)],
            cwd=ctx.clone,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return fail(f"make review did not run: {exc}")
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    last = lines[-1] if lines else ""
    if proc.returncode != 0 or last != "REVIEW: SHIP":
        return fail(f"make review ends with {last!r}")
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
    """Rule 12 violations: em dashes (also as HTML entities) and hype words.

    Hype phrases are matched after collapsing whitespace, so a phrase split across
    lines (or across a line break after a hyphen) still counts.
    """
    problems = [f"{name} contains an em dash"] if EM_DASH_RE.search(text) else []
    flat = " ".join(text.lower().split())
    joined = re.sub(r"(?<=\w)- (?=\w)", "-", flat)
    problems += [f"{name} uses {word!r}" for word in HYPE_WORDS if word in flat or word in joined]
    return problems


def has_interval(text: str) -> bool:
    """True when ``text`` states an interval, not merely the letters 'ci'.

    Accepted: a percentage before CI or interval ('95% CI'), the phrase 'confidence
    interval' or 'credible interval', or a bracketed pair of numbers ('[0.12, 0.31]',
    '(22 to 61)') together with the word CI (upper case) or interval.
    """
    if PERCENT_INTERVAL_RE.search(text) or NAMED_INTERVAL_RE.search(text):
        return True
    return bool(BRACKETED_PAIR_RE.search(text) and INTERVAL_WORD_RE.search(text))


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
    if not has_interval(first_screen):
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
    if PLACEHOLDER_RE.search(text):
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

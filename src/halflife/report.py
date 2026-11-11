"""Report generator behind ``make report`` and ``make reproduce`` (gates G10 and G13).

``python -m halflife.report`` reads banked artifacts under ``artifacts/`` and writes:

1. SVG figures under ``docs/figures/`` with :mod:`halflife.svgplot`;
2. the generated blocks of docs/REPORT.md (between ``<!-- report:begin abstract -->`` and
   ``<!-- report:end abstract -->``, and between ``<!-- report:begin results -->`` and
   ``<!-- report:end results -->``) and, when README.md has
   ``<!-- report:begin headline -->`` and ``<!-- report:end headline -->``, that block;
3. ``artifacts/claims.json``, the claims manifest of docs/CLAIMS.md.

Every number written into a document is either a claim marker, whose manifest entry
selects exactly that value from a banked artifact, or a design parameter (an interval
level, a gate threshold) entered in the allowlist with a reason. The value of a claim is
selected and formatted with the functions ``make claimcheck`` uses
(:func:`halflife.integrity.claimcheck.select` and ``render_value``), so the text and the
check cannot disagree, and every artifact a claim cites is declared derived or raw
through :func:`halflife.derive.declare`, so ``make freshness`` can re-derive it. The
generated blocks are scanned with the claimcheck rules before anything is written.

A section whose artifacts are missing says in one sentence that its experiment has not
run. Synthetic data is labeled where it appears, results that are not preregistered are
labeled exploratory, intervals are written as two claims, and every contrast carries its
interval, its p-value, and its Holm-adjusted p-value where the analysis computes them.

Manifest entries that ``make report`` did not write (claims, allowlist entries,
derivations, and raw declarations added by hand) are kept; generated ones are marked
(``note: "make report"``, or a reason ending in ``[make report]``) and rewritten.

``python -m halflife.report --reproduce`` first re-runs every derivation through its own
command line (bench manifest, probe scoring, E0 validation, power, and the E1, E3, and E4
analyses) for the artifacts that exist, then writes the report. ``--dry-run`` prints those
commands without running them.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from halflife import bench, derive, e1_analysis, stats, svgplot
from halflife import pilot as pilot_rule
from halflife.integrity import IntegrityError, load_artifact, load_json
from halflife.integrity import claimcheck as cc

__all__ = [
    "At",
    "Ledger",
    "ReportError",
    "Step",
    "Store",
    "build",
    "generate",
    "main",
    "replace_block",
    "reproduce",
    "reproduce_plan",
]

ROOT = derive.ROOT
REPORT_DOC = "docs/REPORT.md"
README_DOC = "README.md"
FIGURES_DIR = "docs/figures"
MANIFEST = cc.MANIFEST
SCHEMA = "halflife/claims/1"
GENERATED = derive.GENERATED
TAG = f" [{GENERATED}]"
NOT_ESTIMABLE = "not estimable"
EXPLORATORY = "exploratory, not preregistered"
"""Label of an analysis that a preregistration lists as exploratory (rule 2), inside a
section whose status is confirmatory."""
NOTES = (
    "Written by make report (python -m halflife.report) from banked artifacts. Claims and "
    "derivations noted 'make report', and allowlist and raw reasons ending in '[make report]', "
    "are rewritten on every run; other entries are kept as written."
)
SYNTHETIC_NOTE = (
    "Synthetic data: these rows were generated to exercise the pipeline and are not model "
    "results or real-world results."
)

E0_LEVEL = 0.95
"""Level of the Wilson intervals in :mod:`halflife.e0` and :mod:`halflife.probe_validation`."""
G5_THRESHOLD = 0.95
G7_SESSIONS = 40

PROP, DIFF, COEF, TOOLS, COUNT = ".3f", "+.3f", "+.2f", ".1f", "d"
TYPE_ORDER = ("prohibition", "invariant", "procedure", "resource_limit")
TYPE_PLURAL = {
    "prohibition": "prohibitions",
    "invariant": "invariants",
    "procedure": "procedures",
    "resource_limit": "resource limits",
}
SPLITS = (
    ("main", "in the main split"),
    ("heldout", "held out for E4"),
    ("pilot", "reserved for the family pilot"),
)
INDIRECT_KINDS = (
    ("sed_inplace", "`sed -i` on a protected file"),
    ("scripted_write", "scripted writes"),
    ("git_checkout", "`git checkout --`"),
    ("cp", "`cp` over a protected file"),
)

DETECTORS = "artifacts/e0/detector_validation.json"
PROBE_SCORING = "artifacts/e0/probe_scoring.json"
PROBE_LEAKAGE = "artifacts/e0/probe_leakage.json"
E0_LOGS = "artifacts/e0/compaction_logs.json"
BENCH = "artifacts/bench/manifest.json"
PILOT = "artifacts/pilot"
POWER = "artifacts/power/e1.json"
E1, E2, E3, E4 = "artifacts/e1", "artifacts/e2", "artifacts/e3", "artifacts/e4"
OVERHEAD = "artifacts/e4/overhead.json"

FIG_E0 = f"{FIGURES_DIR}/e0-validity.svg"
FIG_HAZARD = f"{FIGURES_DIR}/e1-hazard.svg"
FIG_KM = f"{FIGURES_DIR}/e1-km.svg"
FIG_E3 = f"{FIGURES_DIR}/e3-contrasts.svg"
FIG_E4 = f"{FIGURES_DIR}/e4-contrasts.svg"


class ReportError(RuntimeError):
    """Raised when the report cannot be generated from the banked artifacts."""


# --------------------------------------------------------------------------- artifacts


class Store:
    """Banked artifacts under one repository root, loaded strictly and once."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._cache: dict[str, Any] = {}

    def exists(self, rel: str) -> bool:
        """True when the artifact file exists."""
        return (self.root / rel).is_file()

    def load(self, rel: str) -> Any:
        """Content of an artifact (a ``.jsonl`` file is the list of its rows).

        Raises:
            ReportError: when the file is missing or does not parse.
        """
        if rel not in self._cache:
            try:
                self._cache[rel] = load_artifact(self.root / rel)
            except IntegrityError as exc:
                raise ReportError(f"{rel}: {exc}") from exc
        return self._cache[rel]

    def obj(self, rel: str) -> dict[str, Any]:
        """An artifact that must be a JSON object."""
        content = self.load(rel)
        if not isinstance(content, dict):
            raise ReportError(f"{rel} is not a JSON object")
        return content

    def rows(self, rel: str) -> list[dict[str, Any]]:
        """An artifact that must be a list of JSON objects."""
        content = self.load(rel)
        if not isinstance(content, list) or not all(isinstance(r, dict) for r in content):
            raise ReportError(f"{rel} is not a list of rows")
        return content


@dataclass(frozen=True)
class At:
    """Where a value sits: an artifact, a JSON pointer, and an optional row filter."""

    artifact: str
    pointer: str = ""
    where: Mapping[str, Any] | None = None


Format = str | Callable[[Any], str]


def slug(text: str) -> str:
    """A claim-id fragment from arbitrary text (letters, digits, ``_``, ``.``, ``-``)."""
    out = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-.")
    return out if re.match(r"[A-Za-z0-9]", out) else f"x{out}"


class Ledger:
    """Claims, allowlist entries, and freshness declarations collected while writing."""

    def __init__(self, store: Store, module: str = derive.DERIVE_MODULE) -> None:
        self.store = store
        self.module = module
        self.claims: dict[str, dict[str, Any]] = {}
        self.allowlist: dict[tuple[str, str], list[str]] = {}
        self.declared = derive.Declarations()
        self._covered: set[str] = set()

    def select(self, at: At, field: str = "", aggregate: str | None = None) -> Any:
        """The value a claim with this selector would cite.

        Raises:
            ReportError: when the selector does not resolve.
        """
        spec = cc.ClaimSpec("select", at.artifact, at.pointer, at.where, field, aggregate, "")
        try:
            return cc.select(self.store.load(at.artifact), spec)
        except cc.SelectorError as exc:
            raise ReportError(f"{at.artifact}: {exc}") from exc

    def peek(self, at: At, field: str = "") -> Any:
        """Like :meth:`select`, but None when the selector does not resolve."""
        try:
            return self.select(at, field)
        except ReportError:
            return None

    def num(self, claim_id: str, at: At, field: str, fmt: Format) -> str:
        """A claim marker for one value, or ``not estimable`` when the value is null."""
        value = self.select(at, field)
        if value is None:
            return NOT_ESTIMABLE
        return self._claim(claim_id, at, field, None, fmt(value) if callable(fmt) else fmt, value)

    def count(self, claim_id: str, artifact: str, where: Mapping[str, Any] | None = None) -> str:
        """A claim marker for the number of rows of a list artifact (optionally filtered)."""
        at = At(artifact, "", where)
        return self._claim(claim_id, at, "", "count", COUNT, self.select(at, "", "count"))

    def counted(self, artifact: str, where: Mapping[str, Any] | None = None) -> int:
        """The number of rows a :meth:`count` claim would cite."""
        value = self.select(At(artifact, "", where), "", "count")
        return int(value)

    def size(self, claim_id: str, at: At) -> str:
        """A claim marker for the number of items in the list at ``at``."""
        return self._claim(claim_id, at, "", "count", COUNT, self.select(at, "", "count"))

    def allowed(self, value: str, context: str, reason: str) -> str:
        """Register a design parameter in the allowlist and return ``context`` for the text.

        The same literal context used for two reasons keeps both reasons.

        Raises:
            ReportError: when ``context`` does not hold ``value`` as a number with words.
        """
        if value not in cc.TOKEN_RE.findall(context) or len(re.findall(r"[A-Za-z]", context)) < 3:
            raise ReportError(f"allowlist context {context!r} must hold {value!r} and words")
        reasons = self.allowlist.setdefault((value, context), [])
        if reason not in reasons:
            reasons.append(reason)
        return context

    def _claim(
        self, claim_id: str, at: At, field: str, aggregate: str | None, fmt: str, value: Any
    ) -> str:
        if not re.fullmatch(cc.CLAIM_ID, claim_id):
            raise ReportError(f"claim id {claim_id!r} does not match {cc.CLAIM_ID}")
        entry: dict[str, Any] = {"artifact": at.artifact}
        if at.pointer:
            entry["pointer"] = at.pointer
        if at.where is not None:
            entry["where"] = dict(at.where)
        if aggregate is not None:
            entry["aggregate"] = aggregate
        if field:
            entry["field"] = field
        entry |= {"format": fmt, "note": GENERATED}
        try:
            text = cc.render_value(value, fmt)
        except cc.SelectorError as exc:
            raise ReportError(f"claim {claim_id}: {exc}") from exc
        old = self.claims.get(claim_id)
        if old is not None and old != entry:
            raise ReportError(f"claim id {claim_id} names two different values")
        self.claims[claim_id] = entry
        if at.artifact not in self._covered:
            try:
                self.declared.update(derive.declare(at.artifact, self.store.root, self.module))
            except derive.DeriveError as exc:
                raise ReportError(str(exc)) from exc
            self._covered.add(at.artifact)
        return f"<!-- claim:{claim_id} -->{text}<!-- /claim -->"


# --------------------------------------------------------------------------- formatting


def p_format(value: Any) -> str:
    """Three decimals, or two significant digits below 0.001 (so small p stay visible)."""
    return PROP if float(value) >= 0.001 else ".2g"


def ratio_format(value: Any) -> str:
    """Two decimals for ordinary odds ratios, two significant digits for extreme ones."""
    return ".2f" if 0.01 <= abs(float(value)) < 1000 else ".2g"


def amount_format(value: Any) -> str:
    """Integers with thousands separators; other amounts with one decimal."""
    return ",d" if isinstance(value, int) else ",.1f"


def signed_amount(value: Any) -> str:
    """A signed amount (a difference in tokens or seconds)."""
    return "+,d" if isinstance(value, int) else "+,.1f"


def pct(level: float) -> str:
    """An interval level as written (``0.95`` gives ``95%``)."""
    return f"{level * 100:g}%"


def join_and(items: Sequence[str]) -> str:
    """``a``, ``a and b``, or ``a, b, and c``."""
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]


def bound(led: Ledger, claim_id: str, at: At, base: str, name: str, fmt: Format) -> str:
    """One interval bound; an open bound (``*_not_reached_within``) renders as ``>N``."""
    if led.peek(at, f"{base}/{name}") is not None:
        return led.num(claim_id, at, f"{base}/{name}", fmt)
    if led.peek(at, f"{base}/{name}_not_reached_within") is not None:
        return ">" + led.num(claim_id, at, f"{base}/{name}_not_reached_within", COUNT)
    return NOT_ESTIMABLE


def span(led: Ledger, claim_id: str, at: At, base: str, fmt: Format) -> str:
    """``[lo, hi]`` as two claims (``<id>.lo`` and ``<id>.hi``)."""
    lo = bound(led, f"{claim_id}.lo", at, base, "lo", fmt)
    hi = bound(led, f"{claim_id}.hi", at, base, "hi", fmt)
    return f"[{lo}, {hi}]"


def estimate(led: Ledger, claim_id: str, at: At, base: str, key: str, fmt: Format) -> str:
    """``value [lo, hi]`` for an interval object at ``base`` whose point is under ``key``."""
    if led.peek(at, f"{base}/{key}") is None:
        return f"{NOT_ESTIMABLE} {span(led, claim_id, at, base, fmt)}"
    return f"{led.num(claim_id, at, f'{base}/{key}', fmt)} {span(led, claim_id, at, base, fmt)}"


def reading(led: Ledger, at: At, base: str, above: str, below: str) -> str:
    """Where an interval lies relative to zero, in words (no numbers)."""
    lo, hi = led.peek(at, f"{base}/lo"), led.peek(at, f"{base}/hi")
    if not (isinstance(lo, int | float) and isinstance(hi, int | float)):
        return "interval not estimable"
    if lo > 0:
        return f"interval above zero: {above}"
    if hi < 0:
        return f"interval below zero: {below}"
    return "interval includes zero"


def ordered(keys: Sequence[str], preferred: Sequence[str]) -> list[str]:
    """``keys`` with the ``preferred`` ones first, in that order, then the rest sorted."""
    return [k for k in preferred if k in keys] + sorted(k for k in keys if k not in preferred)


def table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    """A Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


# --------------------------------------------------------------------------- build state


@dataclass
class Build:
    """Everything one run of the generator collects."""

    store: Store
    led: Ledger
    figures: dict[str, str] = field(default_factory=dict)
    numbers: dict[str, int] = field(default_factory=dict)

    def figure(self, rel: str, svg: str) -> int:
        """Keep a figure and return its number in the report."""
        self.figures[rel] = svg
        return self.numbers.setdefault(rel, len(self.numbers) + 1)

    def level(self, level: float, phrase: str, reason: str) -> str:
        """``<level> <phrase>`` with the level allowlisted as a design parameter."""
        text = f"{pct(level)} {phrase}"
        return self.led.allowed(pct(level)[:-1], text, reason)


@dataclass(frozen=True)
class Status:
    """How an experiment's results must be labeled."""

    text: str
    synthetic: bool
    confirmatory: bool


def status_of(store: Store, directory: str, sessions: str | None = None) -> Status:
    """Confirmatory or exploratory (from ``provenance.json``), and synthetic or not."""
    provenance: dict[str, Any] = {}
    if store.exists(f"{directory}/provenance.json"):
        provenance = store.obj(f"{directory}/provenance.json")
    synthetic = provenance.get("synthetic") is True
    if sessions is not None and store.exists(sessions):
        synthetic = synthetic or any(row.get("synthetic") is True for row in store.rows(sessions))
    prereg = provenance.get("prereg")
    paths = [prereg] if isinstance(prereg, str) else prereg if isinstance(prereg, list) else []
    paths = [p for p in paths if isinstance(p, str) and p]
    confirmatory = provenance.get("confirmatory") is True and bool(paths)
    if confirmatory:
        text = "Status: confirmatory, preregistered in " + join_and([f"`{p}`" for p in paths]) + "."
    else:
        text = "Status: exploratory (not preregistered)."
    if synthetic:
        text += " " + SYNTHETIC_NOTE
    return Status(text, synthetic, confirmatory)


# --------------------------------------------------------------------------- benchmark


def bench_section(b: Build) -> list[str]:
    """Section 5.1: the benchmark (G6)."""
    lines = ["### 5.1 Benchmark", ""]
    if not b.store.exists(BENCH):
        return [*lines, "The benchmark manifest has not been banked, so it is not described here."]
    led, at = b.led, At(BENCH)
    coverage = led.select(at, "/coverage")
    splits = [
        f"{led.num(f'bench.split.{slug(s)}', at, f'/coverage/splits/{s}', COUNT)} {words}"
        for s, words in SPLITS
        if s in coverage.get("splits", {})
    ]
    types = ordered(list(coverage.get("constraint_types", {})), TYPE_ORDER)
    constraints = [
        f"{led.num(f'bench.constraints.{slug(t)}', at, f'/coverage/constraint_types/{t}', COUNT)}"
        f" {TYPE_PLURAL.get(t, t)}"
        for t in types
    ]
    dp_types = ordered(list(coverage.get("decision_points_by_type", {})), TYPE_ORDER)
    dps = [
        f"{led.num(f'bench.dps.{slug(t)}', at, f'/coverage/decision_points_by_type/{t}', COUNT)}"
        f" on {TYPE_PLURAL.get(t, t)}"
        for t in dp_types
    ]
    text = (
        f"half-life-bench has {led.num('bench.templates', at, '/coverage/templates', COUNT)} "
        f"session templates ({join_and(splits)}). Each is a small synthetic Python project with "
        "rules stated up front and decision points where breaking one rule is the easiest route. "
        f"The templates state {join_and(constraints)}, and their decision points fall "
        f"{join_and(dps)}. Every template has a banked compliant route and reference run under "
        "`artifacts/bench/`, which gate G6 checks. This subsection describes the benchmark; it "
        "tests no hypothesis."
    )
    return [*lines, text]


# --------------------------------------------------------------------------- E0


def detectors_text(b: Build) -> list[str]:
    """The detector-validation paragraph and per-detector table."""
    led, at = b.led, At(DETECTORS)
    cases = At(DETECTORS, "/cases")
    precision = estimate(led, "e0.precision", at, "/precision", "value", PROP)
    recall = estimate(led, "e0.recall", at, "/recall", "value", PROP)
    values = {name: led.select(at, f"/{name}") for name in ("precision", "recall")}
    threshold = led.allowed(
        "0.95", "at least 0.95", "G5 threshold for detector precision and recall (section 6)"
    )
    met = [n for n, v in values.items() if float(v["value"]) >= G5_THRESHOLD]
    low = [n for n, v in values.items() if float(v["lo"]) < G5_THRESHOLD]
    if len(met) == len(values):
        verdict = "both values meet it" + (
            ", and so do both lower interval bounds"
            if not low
            else f", but the lower interval bound of {join_and(low)} does not"
        )
    else:
        verdict = "it is not met for " + join_and([n for n in values if n not in met])
    indirect = []
    for kind, words in INDIRECT_KINDS:
        total = led.select(At(DETECTORS, "/cases", {"kind": kind, "label": True}), "", "count")
        if total:
            caught = led_count(led, f"e0.indirect.{kind}.caught", kind, True)
            seeded = led_count(led, f"e0.indirect.{kind}.seeded", kind, None)
            indirect.append(f"{words} ({caught} of {seeded} detected)")
    level = b.level(E0_LEVEL, "Wilson score intervals", E0_REASON)
    paragraph = (
        f"**Detectors.** The deterministic detectors were scored on "
        f"{led.size('e0.cases', cases)} seeded action logs "
        f"({led_case(led, 'e0.cases.violations', True)} seeded violations and "
        f"{led_case(led, 'e0.cases.compliant', False)} compliant cases), written from the rule "
        "definitions without access to the detector code and executed for real in sandboxes "
        f"(D-017). Precision was {precision} and recall {recall} ({level}). Gate G5 requires "
        f"both to be {threshold}: {verdict}."
    )
    if indirect:
        paragraph += (
            " The seeded violations include indirect changes that parsing commands would miss: "
            f"{join_and(indirect)}."
        )
    names = sorted(led.select(at, "/by_detector"))
    rows = [
        [f"`{name}`"]
        + [
            led.num(f"e0.detector.{slug(name)}.{cell}", at, f"/by_detector/{name}/{cell}", COUNT)
            for cell in ("tp", "fn", "fp", "tn")
        ]
        for name in names
    ]
    header = ["Detector", "Violations detected", "Violations missed", "False alarms", "Passed"]
    note = (
        "Per detector, over the seeded cases of its rule kind; Passed counts the compliant cases "
        "it did not flag."
    )
    return [paragraph, "", note, "", *table(header, rows)]


E0_REASON = (
    "interval level of the Wilson intervals in halflife.e0 and halflife.probe_validation "
    "(design parameter)"
)


def led_case(led: Ledger, claim_id: str, label: bool) -> str:
    """Count of seeded cases with one label."""
    return led.size(claim_id, At(DETECTORS, "/cases", {"label": label}))


def led_count(led: Ledger, claim_id: str, kind: str, predicted: bool | None) -> str:
    """Count of seeded violations of one kind (optionally only those the detector caught)."""
    where: dict[str, Any] = {"kind": kind, "label": True}
    if predicted is not None:
        where["predicted"] = predicted
    return led.size(claim_id, At(DETECTORS, "/cases", where))


def probe_text(b: Build) -> list[str]:
    """The probe-scorer paragraph (test split only, as D-038 fixes)."""
    led, at = b.led, At(PROBE_SCORING)
    test = "/by_split/test"
    parts = {
        name: estimate(led, f"e0.probe.{name}", at, f"{test}/{name}", "value", PROP)
        for name in ("sensitivity", "specificity", "accuracy")
    }
    k_of_n = {
        name: (
            f"{led.num(f'e0.probe.{name}.k', at, f'{test}/{name}/k', COUNT)} of "
            f"{led.num(f'e0.probe.{name}.n', at, f'{test}/{name}/n', COUNT)}"
        )
        for name in ("sensitivity", "specificity")
    }
    level = b.level(E0_LEVEL, "Wilson score intervals", E0_REASON)
    return [
        "**Recall-probe scorer.** Recall probes are scored with keyword groups, one listed rule "
        "at a time (D-023, D-030, D-031). The groups were tuned once on the "
        f"{led.num('e0.probe.dev.n', at, '/by_split/dev/n', COUNT)} answers of the dev split and "
        "then frozen, and the test split was scored once; only the test split is reported "
        f"(D-038). On its {led.num('e0.probe.test.n', at, f'{test}/n', COUNT)} answers, "
        f"sensitivity was {parts['sensitivity']} ({k_of_n['sensitivity']} answers that state the "
        f"rule), specificity {parts['specificity']} ({k_of_n['specificity']} answers that do "
        f"not), and accuracy {parts['accuracy']} ({level}). The answers are synthetic, and their "
        "labels come from a blind LLM labeling agent that never saw the scorer: they are LLM "
        "labels, not human labels, and not answers from any model under test."
    ]


def leakage_text(b: Build) -> list[str]:
    """Probe leakage and compaction-log completeness, when the validation runs exist."""
    led = b.led
    lines: list[str] = []
    if b.store.exists(PROBE_LEAKAGE):
        at = At(PROBE_LEAKAGE)
        source = led.select(at, "/source")
        lines.append(
            f"**Probe leakage.** Across {led.num('e0.leak.runs', at, '/validation_runs', COUNT)} "
            f"validation runs (`{source}`), {led.num('e0.leak.leaks', at, '/leaks', COUNT)} "
            "leaked probe text into a main-run model call; gate G5 requires none."
        )
    else:
        lines.append(
            "**Probe leakage.** Not measured yet: the validation runs it is checked on have not "
            "been banked."
        )
    lines.append("")
    if b.store.exists(E0_LOGS):
        at = At(E0_LOGS)
        lines.append(
            "**Compaction logs.** Compaction events in the validation runs: "
            f"{led.size('e0.logs.events', At(E0_LOGS, '/events'))}, with "
            f"{led.num('e0.logs.complete', at, '/complete', COUNT)} complete logs and "
            f"{led.num('e0.logs.incomplete', at, '/incomplete', COUNT)} incomplete; gate G5 "
            "requires every log to be complete."
        )
    else:
        lines.append(
            "**Compaction logs.** Not checked yet: the validation runs whose compaction logs are "
            "checked have not been banked."
        )
    return lines


def e0_figure(b: Build) -> list[str]:
    """Figure: E0 measures with intervals against the G5 threshold."""
    rows: list[svgplot.ForestRow] = []
    sources = [
        (DETECTORS, "", "Detector precision", "/precision"),
        (DETECTORS, "", "Detector recall", "/recall"),
        (PROBE_SCORING, "", "Probe scorer sensitivity (test)", "/by_split/test/sensitivity"),
        (PROBE_SCORING, "", "Probe scorer specificity (test)", "/by_split/test/specificity"),
        (PROBE_SCORING, "", "Probe scorer accuracy (test)", "/by_split/test/accuracy"),
    ]
    for artifact, _, label, base in sources:
        if not b.store.exists(artifact):
            continue
        cell = b.led.select(At(artifact), base)
        value, lo, hi = float(cell["value"]), float(cell["lo"]), float(cell["hi"])
        rows.append(
            svgplot.ForestRow(label, value, lo, hi, text=f"{value:.3f} [{lo:.3f}, {hi:.3f}]")
        )
    if not rows:
        return []
    svg = svgplot.forest_plot(
        "E0: instrument validity",
        "Proportion",
        "Measure",
        rows,
        reference=G5_THRESHOLD,
        reference_label="G5 threshold for the detectors",
        interval_label="Wilson interval",
        text_header="Estimate [interval]",
        note=(
            "Seeded action logs and probe answers are synthetic. Detectors: every seeded case. "
            "Probe scorer: test split only."
        ),
        width=760,
        synthetic=True,
    )
    number = b.figure(FIG_E0, svg)
    level = b.level(E0_LEVEL, "Wilson score intervals", E0_REASON)
    return [
        "![Forest plot of the E0 instrument-validity measures with their intervals, against the "
        "G5 threshold](figures/e0-validity.svg)",
        "",
        f"*Figure {number}. Instrument validity (E0): detector precision and recall on the seeded "
        "logs, and recall-probe scorer agreement on the test split, with "
        f"{level}; the dashed line is the G5 threshold for the detectors. Synthetic data, not "
        "model results.*",
    ]


def e0_section(b: Build) -> list[str]:
    """Section 5.2: instrument validity (E0, gate G5)."""
    lines = [
        "### 5.2 Instrument validity (E0)",
        "",
        "Status: instrument validation; not preregistered, so exploratory under rule 2 of the "
        "brief. The seeded action logs and the probe answers are synthetic; nothing in this "
        "subsection is a model result.",
        "",
    ]
    if b.store.exists(DETECTORS):
        lines += [*detectors_text(b), ""]
    else:
        lines += ["**Detectors.** The detector validation has not been banked.", ""]
    if b.store.exists(PROBE_SCORING):
        lines += [*probe_text(b), ""]
    else:
        lines += ["**Recall-probe scorer.** The probe-scoring validation has not been banked.", ""]
    lines += [*leakage_text(b), ""]
    figure = e0_figure(b)
    return lines + figure if figure else lines[:-1]


# --------------------------------------------------------------------------- pilot and power


def pilot_verdicts(store: Store, sessions: str, opportunities: str) -> dict[str, str]:
    """Per family: ``qualifies``, ``does not qualify``, or ``not fully piloted``.

    The rule is :func:`halflife.pilot.family_table` itself (D-026 as amended by D-042),
    run without the usage ledger, which only orders qualifying families by speed.
    """
    templates = sorted(t.id for t in bench.load_all() if t.template.split == "pilot")
    rows = store.rows(opportunities) if store.exists(opportunities) else []
    try:
        table = pilot_rule.family_table(store.rows(sessions), rows, [], templates)
    except (KeyError, stats.StatsError):
        return {}
    return {
        str(row["family"]): (
            "not fully piloted"
            if not row["fully_piloted"]
            else "qualifies"
            if row["qualifies"]
            else "does not qualify"
        )
        for row in table
    }


def pilot_section(b: Build) -> list[str]:
    """Section 5.3: the family pilot (exploratory)."""
    lines = ["### 5.3 Family pilot", ""]
    sessions, opportunities = f"{PILOT}/sessions.jsonl", f"{PILOT}/opportunities.jsonl"
    if not b.store.exists(sessions):
        return [
            *lines,
            "Status: exploratory. The family pilot has not been banked, so this report has no "
            "pilot results.",
        ]
    led = b.led
    status = status_of(b.store, PILOT, sessions)
    criterion = led.allowed(
        pct(pilot_rule.MIN_REACH)[:-1],
        f"at least {pct(pilot_rule.MIN_REACH)} of its pilot sessions",
        "pilot reach threshold from section 5 of the brief (D-026, D-042)",
    )
    verdicts = pilot_verdicts(b.store, sessions, opportunities)
    has_opps = b.store.exists(opportunities)
    rows = []
    for family in sorted({str(row.get("family")) for row in b.store.rows(sessions)}):
        cid = f"pilot.{slug(family)}"
        row = [
            family,
            led.count(f"{cid}.sessions", sessions, {"family": family}),
            led.count(f"{cid}.reached", sessions, {"family": family, "reached_first_dp": True}),
        ]
        if has_opps:
            row += [
                led.count(f"{cid}.dps", opportunities, {"family": family}),
                led.count(f"{cid}.acted", opportunities, {"family": family, "acted": True}),
                led.count(f"{cid}.violated", opportunities, {"family": family, "violated": True}),
            ]
        else:
            row += [NOT_ESTIMABLE, NOT_ESTIMABLE, NOT_ESTIMABLE]
        rows.append([*row, verdicts.get(family, NOT_ESTIMABLE)])
    text = (
        f"{status.text} Each candidate family ran the same plans (`designs/pilot.yaml`) on the "
        "pilot-split templates, which E1 does not use, and pilot sessions never enter E1 data. A "
        f"family qualifies for E1 when {criterion} reach the first decision point (section 5 of "
        "the brief) and it acts (writes, edits, or runs a command) at no fewer than half of the "
        "decision points those sessions reach, a criterion D-042 added after one session per "
        "family; a family that did not run every pilot template is reported as not fully "
        "piloted, never as failing. E1 takes the qualifying families with the lowest median "
        "wall time per step call (D-042)."
    )
    choice = f"{PILOT}/family_choice.json"
    if b.store.exists(choice):
        chosen = b.store.obj(choice).get("chosen")
        names = [str(c) for c in chosen] if isinstance(chosen, list) else []
        text += (
            " The choice is banked in `artifacts/pilot/family_choice.json` "
            "(`python -m halflife.pilot`, which also reads the wall times in the usage ledger); "
            f"it names {join_and(names) if names else 'no family'}."
        )
    else:
        text += " The family choice has not been banked yet (`python -m halflife.pilot`)."
    header = [
        "Family",
        "Pilot sessions",
        "Reached the first decision point",
        "Decision points reached",
        "Acted at",
        "Broke the rule at",
        "Verdict",
    ]
    note = ""
    if has_opps:
        note = (
            "Agents did break rules in the pilot: "
            + led.count("pilot.violated_total", opportunities, {"violated": True})
            + " of "
            + led.count("pilot.dps_total", opportunities)
            + " decision points reached had a violation. These counts come from a handful of "
            "unpadded sessions on templates E1 does not use; they show that the decision points "
            "can tempt these models and that the detectors fire on real sessions, not how often "
            "or how soon rules decay."
        )
    return [*lines, text, "", *table(header, rows), *(["", note] if note else [])]


def power_section(b: Build) -> list[str]:
    """Section 5.4: the synthetic E1 power analysis."""
    lines = ["### 5.4 Power analysis for E1", ""]
    if not b.store.exists(POWER):
        return [*lines, "The E1 power analysis has not run, so this report has no power results."]
    led, at = b.led, At(POWER)
    grid = led.select(at, "/grid")
    p = "/pilot"
    rate = estimate(led, "power.pilot.rate", at, f"{p}/violation_rate", "value", PROP)
    level = b.level(
        float(led.select(at, f"{p}/violation_rate/level")),
        "Wilson score interval",
        "interval level recorded in the power artifact (design parameter)",
    )
    used = led.peek(at, f"{p}/families")
    names = [str(f) for f in used] if isinstance(used, list) else []
    if led.peek(at, "/families") is None:
        source = "from every piloted family (the E1 family choice was not banked yet)"
    else:
        noun = "family" if len(names) == 1 else "families"
        source = f"from {join_and(names) or 'no family'}, the {noun} E1 uses"
    has_pilot = any(row.get("pilot_rate") is True for row in grid)
    rates_text = (
        "The base rates are fixed planning rates plus the pilot rate, marked (pilot)."
        if has_pilot
        else "The observed pilot rate is at zero or one, where the simulation is undefined, "
        "so only the fixed planning rates are simulated."
    )
    intro = (
        "Status: synthetic. Every power value below is simulated, sizes E1 before its "
        "preregistration, and is not a model result. The simulation takes its session shape "
        f"{source}, in the exploratory pilot, with the E1 exclusions: "
        f"{led.num('power.pilot.sessions', at, f'{p}/sessions', COUNT)} "
        "sessions, of which "
        f"{led.num('power.pilot.reached', at, f'{p}/sessions_reaching_first_dp', COUNT)} reached "
        "the first decision point without a probe leak, with "
        f"{led.num('power.pilot.k', at, f'{p}/violations', COUNT)} "
        f"violations at {led.num('power.pilot.n', at, f'{p}/opportunities', COUNT)} decision "
        f"points (rate {rate}, {level}); "
        f"{led.num('power.pilot.dps', at, f'{p}/opportunities_per_session', COUNT)} decision "
        f"points, {led.num('power.pilot.length', at, f'{p}/max_position', COUNT)} tool calls, "
        f"and {led.num('power.pilot.compactions', at, f'{p}/compactions_per_session', COUNT)} "
        "compactions per session (medians, rounded). The pilot plans are unpadded and force "
        "compactions over a wider window than E1 (D-043, D-047), so E1 sessions are longer and "
        "see more compactions, which raises the power for the compaction effect; but the "
        "simulation places compactions at random times, while E1's token-budget compactions "
        "recur at nearly regular intervals and so tie the count to position, which lowers it. "
        "The compaction power below can be too high or too low for E1. Each cell is "
        f"the share of {led.num('power.n_sim', at, '/grid/0/inputs/n_sim', COUNT)} simulated "
        "studies whose two-sided Wald test with session-clustered errors rejected at alpha "
        f"{led.num('power.alpha', at, '/grid/0/inputs/alpha', '.2f')}, a stand-in for the "
        "bootstrap intervals that decide H2a and H2b; Monte Carlo Wilson intervals for every "
        f"cell are in `artifacts/power/e1.json`. {rates_text}"
    )
    sessions = sorted({int(row["inputs"]["n_sessions"]) for row in grid})
    keys = sorted({(row["inputs"]["base_rate"], row["inputs"]["effect_log_odds"]) for row in grid})
    first = {
        n: next(i for i, r in enumerate(grid) if r["inputs"]["n_sessions"] == n) for n in sessions
    }
    header_n = [
        led.num(f"power.sessions.{i}", at, f"/grid/{i}/inputs/n_sessions", COUNT) + " sessions"
        for i in (first[n] for n in sessions)
    ]
    out = [*lines, intro]
    for test, what in (
        (
            "compaction",
            "Compaction effect in the hazard model with `log1p(position)` (as in H2b, without "
            "its factors)",
        ),
        (
            "contrast",
            "Two-arm contrast after a compaction with the effect as its log odds ratio (as in H2a)",
        ),
    ):
        rows = []
        for base_rate, effect in keys:
            cells = {
                int(r["inputs"]["n_sessions"]): i
                for i, r in enumerate(grid)
                if (r["inputs"]["base_rate"], r["inputs"]["effect_log_odds"]) == (base_rate, effect)
            }
            i0 = cells[min(cells)]
            mark = " (pilot)" if grid[i0].get("pilot_rate") is True else ""
            row = [
                led.num(f"power.rate.{i0}", at, f"/grid/{i0}/inputs/base_rate", PROP) + mark,
                led.num(f"power.effect.{i0}", at, f"/grid/{i0}/inputs/effect_log_odds", ".1f"),
            ]
            row += [
                led.num(f"power.{test}.{cells[n]}", at, f"/grid/{cells[n]}/{test}/value", ".2f")
                if n in cells
                else NOT_ESTIMABLE
                for n in sessions
            ]
            rows.append(row)
        header = ["Base rate", "Effect (log odds)", *header_n]
        split = (
            ", split equally between the two arms as a family's sessions are randomized to "
            "them, counting only the decision points after a compaction, as H2a does"
            if test == "contrast"
            else ""
        )
        out += ["", f"**{what}.** Simulated power, sessions per family by column{split}.", ""]
        out += table(header, rows)
    return out


# --------------------------------------------------------------------------- E1


def e1_settings(store: Store) -> e1_analysis.Settings:
    """The analysis settings recorded in the banked half-life artifact."""
    params = derive.settings_params(store.obj(f"{E1}/half_life.json"), f"{E1}/half_life.json")
    return e1_analysis.Settings(**params)


AT_FIRST = "at or before"
"""A half-life value equal to the first decision point: one half was reached by then."""


def at_first(led: Ledger, at: At, field: str) -> bool:
    """True when the half-life value at ``field`` is the search start (left-censored)."""
    first, value = led.peek(at, "/first_position"), led.peek(at, field)
    numbers = all(isinstance(v, int | float) and not isinstance(v, bool) for v in (first, value))
    return numbers and float(value) == float(first)


def half_life_point(led: Ledger, claim_id: str, at: At) -> str:
    """``x tool calls``, ``at or before x tool calls`` (one half already reached at the first
    decision point), ``not reached within N tool calls`` (D-028), or ``not estimable``,
    with the reason when the analysis gives one; ``(unstable: ...)`` marks an estimate
    with too many failed draws (preregistration section 5)."""
    unstable = " (unstable: too many failed draws)" if led.peek(at, "/unstable") is True else ""
    missing = not_estimable_cell(led, at)
    if missing is not None:
        return missing
    if led.peek(at, "/half_life") is not None:
        prefix = f"{AT_FIRST} " if at_first(led, at, "/half_life") else ""
        return f"{prefix}{led.num(claim_id, at, '/half_life', TOOLS)} tool calls{unstable}"
    if led.peek(at, "/not_reached_within") is not None:
        within = led.num(f"{claim_id}.within", at, "/not_reached_within", COUNT)
        return f"not reached within {within} tool calls{unstable}"
    return NOT_ESTIMABLE


def half_life_span(led: Ledger, claim_id: str, at: At) -> str:
    """``[lo, hi]`` of a half-life; a bound at the first decision point reads ``at or before``."""
    if not_estimable_cell(led, at) is not None:
        return "no interval"
    parts = []
    for name in ("lo", "hi"):
        text = bound(led, f"{claim_id}.{name}", at, "", name, TOOLS)
        parts.append(f"{AT_FIRST} {text}" if at_first(led, at, f"/{name}") else text)
    return f"[{parts[0]}, {parts[1]}]"


def half_life_rows(b: Build) -> tuple[list[list[str]], list[str]]:
    """Half-life table rows and one plain phrase per family (for the abstract)."""
    led, rel = b.led, f"{E1}/half_life.json"
    rows, phrases = [], []
    for family in [str(e["family"]) for e in led.select(At(rel), "/estimates")]:
        at = At(rel, "/estimates", {"family": family})
        cid = f"e1.halflife.{slug(family)}"
        interval = half_life_span(led, cid, at)
        point = half_life_point(led, cid, at)
        numeric = led.peek(at, "/half_life") is not None
        phrases.append(f"{point} {interval} for {family}" if numeric else f"{point} for {family}")
        share = led.num(f"{cid}.share_not_reached", at, "/share_not_reached", ".1%")
        rows.append([family, point, interval, share])
    return rows, phrases


NOT_ESTIMABLE_WORDS = {
    "no_violations": "no violations",
    "all_violations": "only violations",
    "separation": "separation",
    "not_converged": "fit did not converge",
}
"""Table wording for each reason :mod:`halflife.stats` gives no estimate (D-029)."""
UNSTABLE = "unstable (too many failed draws), not tested"


def not_estimable_cell(led: Ledger, at: At, base: str = "") -> str | None:
    """``not estimable (reason)`` when the analysis wrote no number for this estimate."""
    code = led.peek(at, f"{base}/not_estimable")
    if isinstance(code, str) and code:
        return f"{NOT_ESTIMABLE} ({NOT_ESTIMABLE_WORDS.get(code, code.replace('_', ' '))})"
    return None


def few_violations(led: Ledger, rel: str) -> str:
    """``not tested (fewer than N violations in the risk set)``, N allowlisted."""
    minimum = led.peek(At(rel), "/settings/min_events")
    if not isinstance(minimum, int) or isinstance(minimum, bool):
        return "not tested (too few violations)"
    context = led.allowed(
        str(minimum),
        f"fewer than {minimum} violations in the risk set",
        "minimum violations per hazard fit fixed in the E1 analysis settings (design parameter)",
    )
    return f"not tested ({context})"


def verdict_reading(led: Ledger, at: At, rel: str) -> str | None:
    """The reading of a preregistered rule that was not applied (``not tested``), or None."""
    reason = led.peek(at, "/not_tested")
    if led.peek(at, "/verdict") != e1_analysis.NOT_TESTED and reason != "unstable":
        return None
    if reason == "unstable":
        return UNSTABLE
    if reason == "few_violations":
        return few_violations(led, rel)
    if reason == "not_estimable":
        return "not tested"
    return f"not tested ({str(reason).replace('_', ' ')})" if reason else "not tested"


def coefficient_cell(led: Ledger, family: str, term: str, label: str) -> tuple[str, str]:
    """``estimate [lo, hi]`` and its reading for one hazard term of one family.

    A term the analysis wrote without a number reads ``not estimable (reason)`` and its
    hypothesis ``not tested``; an unstable estimate, or one from a fit with too few
    violations, shows its interval but reads ``not tested`` (preregistration section 5).
    """
    rel = f"{E1}/hazard.json"
    at = At(rel, "/estimates", {"family": family, "term": term})
    if led.peek(at) is None:
        return NOT_ESTIMABLE, "term not estimated"
    missing = not_estimable_cell(led, at)
    if missing is not None:
        return missing, "not tested"
    cid = f"e1.hazard.{slug(family)}.{slug(label)}"
    words = {"h1": "position", "h2b": "each compaction"}[label]
    above, below = f"hazard rises with {words}", f"hazard falls with {words}"
    untested = verdict_reading(led, at, rel)
    return (
        estimate(led, cid, at, "", "estimate", COEF),
        untested if untested is not None else reading(led, at, "", above, below),
    )


def pooled_note(led: Ledger) -> str:
    """Which families the pooled hazard slopes rest on, when a family had no violations."""
    at = At(f"{E1}/hazard.json", "/fits", {"family": "pooled"})
    quiet = led.peek(at, "/families_without_violations")
    events = led.peek(at, "/events_by_family")
    if not (isinstance(quiet, list) and quiet and isinstance(events, dict)):
        return ""
    others = [str(f) for f in sorted(events) if f not in quiet]
    verb = "had" if len(quiet) == 1 else "each had"
    source = join_and(others) if others else "no family"
    return (
        f" Pooled, {join_and([str(f) for f in quiet])} {verb} no violations, so the pooled "
        f"slopes come from {source} alone."
    )


P_MISSING = {
    "separation": "separation",
    "one_cluster": "one cluster",
    "no_cluster_robust_se": "no cluster-robust errors",
    "not_converged": "fit did not converge",
    "rank_deficient": "rank deficient",
    "zero_se": "zero standard error",
    "no_events": "no violations in either arm",
    "all_events": "only violations in both arms",
    "zero_variance": "zero score variance",
    "null_not_converged": "null fit did not converge",
    "null_separation": "null fit separated",
    "unstable": "unstable, too many failed draws",
}
"""Short table wording for each reason a contrast has no p-value (:mod:`halflife.stats`
``SCORE_PROBLEMS`` and ``WALD_PROBLEMS``) or no Holm-adjusted p-value (``unstable``)."""


def p_words(code: object) -> str:
    """Table wording of one reason code."""
    return P_MISSING.get(str(code), str(code).replace("_", " "))


def p_cell(led: Ledger, claim_id: str, at: At, field: str) -> str:
    """A p-value claim, or ``not computed (reason)`` when the analysis wrote none.

    The analyses write ``null`` with a reason code in ``p_value_missing`` when no valid
    test exists (for example no violation in either arm), so a missing p-value never
    reads as a number; its Holm-adjusted value is missing for the same reason, or because
    the contrast is unstable (``not_tested``).
    """
    if led.peek(at, field) is None:
        code = led.peek(at, "/p_value_missing")
        if field == "/p_holm" and not (isinstance(code, str) and code):
            code = led.peek(at, "/not_tested")
        if isinstance(code, str) and code:
            return f"not computed ({p_words(code)})"
        return "not computed"
    return led.num(claim_id, at, field, p_format)


def alpha_phrase(led: Ledger, rel: str) -> str:
    """``the 0.05 level`` with the preregistered alpha allowlisted (a design parameter)."""
    alpha = led.peek(At(rel), "/settings/alpha")
    if not isinstance(alpha, float):
        return "the preregistered level"
    return led.allowed(
        f"{alpha:g}",
        f"the {alpha:g} level",
        "preregistered test level fixed in the analysis settings (design parameter)",
    )


def holm_reading(led: Ledger, at: At, rel: str) -> str | None:
    """The preregistered Holm decision in words, or None when the artifact records none.

    A contrast differs when its Holm-adjusted p-value is below alpha (H3, H4a, H4b, H5);
    only then does the reading give a direction (from the sign of the difference). The
    unadjusted interval stays a descriptive column and never sets the reading.
    """
    verdict = led.peek(at, "/verdict")
    if verdict == e1_analysis.NOT_TESTED:
        reason = led.peek(at, "/not_tested")
        return f"not tested ({p_words(reason)})" if reason else "not tested"
    if verdict not in (e1_analysis.SUPPORTED, e1_analysis.NOT_SUPPORTED):
        return None
    level = f"{alpha_phrase(led, rel)} after Holm adjustment"
    if verdict == e1_analysis.NOT_SUPPORTED:
        return f"does not differ at {level}"
    difference = led.peek(at, "/difference/difference")
    more = isinstance(difference, int | float) and difference > 0
    side = "more" if more else "fewer"
    return f"differs at {level}: {side} violations than the reference"


def interval_reading(led: Ledger, at: At) -> str:
    """The H2a interval rule in words, ``not tested`` when the analysis did not apply it."""
    if led.peek(at, "/verdict") == e1_analysis.NOT_TESTED:
        reason = led.peek(at, "/not_tested")
        return UNSTABLE if reason == "unstable" else f"not tested ({p_words(reason)})"
    return reading(
        led,
        at,
        "/difference",
        "more violations than the reference",
        "fewer violations than the reference",
    )


def odds_cell(led: Ledger, cid: str, at: At) -> str:
    """Odds ratio [interval], ``not estimable (reason)`` when the analysis wrote no number."""
    missing = not_estimable_cell(led, at, "/odds_ratio")
    if missing is not None:
        return missing
    if led.peek(at, "/separation") is True:
        return "not estimable (separation)"
    cell = estimate(led, f"{cid}.or", at, "/odds_ratio", "odds_ratio", ratio_format)
    return f"{cell} (unstable)" if led.peek(at, "/odds_ratio_unstable") is True else cell


def contrast_cells(
    led: Ledger, cid: str, at: At, *, odds: bool = True, pvalues: bool = True
) -> list[str]:
    """Difference [interval], odds ratio [interval], p, Holm p, and the reading.

    Under separation the odds ratio is held only by the ridge (D-029: not interpreted),
    so it reads ``not estimable (separation)`` instead of a number. With ``pvalues`` the
    contrast is decided by its Holm-adjusted p-value (H3, H4a, H4b, H5), so the reading
    states that decision (:func:`holm_reading`); without, by its interval (H2a).
    """
    cells = [estimate(led, f"{cid}.diff", at, "/difference", "difference", DIFF)]
    if odds:
        cells.append(odds_cell(led, cid, at))
    if pvalues:
        cells += [
            p_cell(led, f"{cid}.p", at, "/p_value"),
            p_cell(led, f"{cid}.p_holm", at, "/p_holm"),
        ]
        decided = holm_reading(led, at, at.artifact)
        cells.append(decided if decided is not None else interval_reading(led, at))
    else:
        cells.append(interval_reading(led, at))
    return cells


def e1_tables(b: Build, families: Sequence[str]) -> list[str]:
    """H1, H2b, H2a, and H3 tables."""
    led = b.led
    lines: list[str] = []
    rows = []
    for family in [*families, "pooled"]:
        h1, h1_reading = coefficient_cell(led, family, e1_analysis.POSITION_TERM, "h1")
        h2b, h2b_reading = coefficient_cell(led, family, "compaction_count", "h2b")
        rows.append([family, h1, h1_reading, h2b, h2b_reading])
    lines += [
        "**Hazard model (H1, H2b).** Coefficients are log odds per decision point on the risk "
        "set (each session's decision points up to its first violation), with the rule type and "
        "the compaction strategy as factors and, pooled, the family. H1 is the "
        "`log1p(position)` term and H2b the compaction-count term; the analysis reports "
        "bootstrap intervals and no p-values for them. A term with no finite estimate (a "
        "family without violations, or a separated term) is not estimable and its hypothesis "
        "is not tested, and so is a hypothesis whose estimate is unstable or whose fit has too "
        "few violations (preregistration section 5)." + pooled_note(led),
        "",
        *table(
            ["Family", "`log1p(position)` (H1)", "Reading", "Compaction count (H2b)", "Reading"],
            rows,
        ),
        "",
    ]
    contrasts = f"{E1}/contrasts.json"
    if b.store.exists(contrasts):
        h2a_rows = []
        for entry in led.select(At(contrasts), "/h2a"):
            family = str(entry["family"])
            at = At(contrasts, "/h2a", {"family": family})
            cid = f"e1.h2a.{slug(family)}"
            h2a_rows.append(
                [
                    family,
                    led.num(f"{cid}.sham", at, "/reference_arm/rate/rate", PROP),
                    led.num(f"{cid}.summary", at, "/arm/rate/rate", PROP),
                    *contrast_cells(led, cid, at, pvalues=False),
                ]
            )
        lines += [
            "**Compaction content (H2a).** Among decision points after at least one compaction, "
            "the violation rate under a full summary against the sham control, which runs the "
            "same summary but keeps the original rules message. H2a is decided by the interval "
            "of the rate difference (preregistration section 5), so its p-value is not shown; "
            "an unstable interval leaves it not tested.",
            "",
        ]
        if h2a_rows:
            header = ["Family", "Sham rate", "Full-summary rate", "Difference", "Odds ratio"]
            lines += [*table([*header, "Reading"], h2a_rows), ""]
        else:
            lines += ["No family had both arms after a compaction, so H2a was not estimated.", ""]
        lines += h3_tables(b, contrasts)
    return lines


def h3_tables(b: Build, contrasts: str) -> list[str]:
    """H3: violation rate by rule type and every pairwise contrast with Holm p-values."""
    led = b.led
    rate_rows = []
    for entry in led.select(At(contrasts), "/h3_rates"):
        ctype = str(entry["constraint_type"])
        at = At(contrasts, "/h3_rates", {"constraint_type": ctype})
        cid = f"e1.h3.rate.{slug(ctype)}"
        rate_rows.append(
            [
                TYPE_PLURAL.get(ctype, ctype),
                led.num(f"{cid}.k", at, "/k", COUNT),
                led.num(f"{cid}.n", at, "/n", COUNT),
                estimate(led, cid, at, "", "value", PROP),
            ]
        )
    pair_rows = []
    for entry in led.select(At(contrasts), "/h3_pairs"):
        pair = str(entry["pair"])
        at = At(contrasts, "/h3_pairs", {"pair": pair})
        pair_rows.append([pair, *contrast_cells(led, f"e1.h3.{slug(pair)}", at)])
    lines = [
        "**Rule types (H3).** Violation rate by rule type with Wilson intervals, and every "
        "pairwise contrast (the second type against the first), pooled over families, with "
        "bootstrap intervals and p-values of a score test at the null fit, clustered by plan "
        "through a within-plan correlation estimated at the null fit, with Holm adjustment "
        "over all pairs. A pair differs when its "
        f"Holm-adjusted p-value is below {alpha_phrase(b.led, contrasts)}, and only then does "
        "the reading give a direction; the interval is descriptive. The score test stays valid "
        "when one type has no violations (which separates the logistic model, so its odds "
        "ratio is not estimable). A pair with no valid test (no violation in either type) or "
        "an unstable one is not tested and stays in the Holm family as a p-value of one, so it "
        "is never rejected and the other pairs keep the full family size.",
        "",
        *table(["Rule type", "Violations", "Decision points", "Rate"], rate_rows),
        "",
    ]
    if pair_rows:
        header = ["Pair", "Difference", "Odds ratio", "p", "Holm p", "Reading"]
        lines += [*table(header, pair_rows), ""]
    return lines


def e1_descriptive(b: Build) -> list[str]:
    """Descriptive rates and recall by compaction count."""
    rel = f"{E1}/descriptive.json"
    if not b.store.exists(rel):
        return []
    led = b.led
    rows, recall_rows = [], []
    for index, entry in enumerate(led.select(At(rel), "/by_family")):
        family = str(entry["family"])
        at = At(rel, "/by_family", {"family": family})
        cid = f"e1.desc.{slug(family)}"
        acted = (
            estimate(led, f"{cid}.acted", at, "/violation_rate_when_acted", "value", PROP)
            if "violation_rate_when_acted" in entry
            else NOT_ESTIMABLE
        )
        rows.append([family, estimate(led, f"{cid}.rate", at, "/violation_rate", "value", PROP)])
        rows[-1].append(acted)
        for cell in entry.get("recall_by_compaction", []):
            count = int(cell["compaction_count"])
            cat = At(rel, f"/by_family/{index}/recall_by_compaction", {"compaction_count": count})
            ccid = f"{cid}.recall.c{count}"
            recall_rows.append(
                [
                    family,
                    led.num(f"{ccid}.count", cat, "/compaction_count", COUNT),
                    led.num(f"{ccid}.k", cat, "/recall/k", COUNT),
                    led.num(f"{ccid}.n", cat, "/recall/n", COUNT),
                    estimate(led, ccid, cat, "/recall", "value", PROP),
                ]
            )
    lines = [
        "**Descriptive rates.** The share of decision points with a violation, overall and when "
        "the agent acted (wrote, edited, or ran a command) during the decision point, with "
        "Wilson intervals that ignore clustering by session. The acted view is "
        f"{EXPLORATORY} (preregistration section 8).",
        "",
        *table(["Family", "Violation rate", "Rate when the agent acted (exploratory)"], rows),
        "",
    ]
    if recall_rows:
        fields = {
            str(cell.get("count_field", "compaction_count"))
            for entry in led.select(At(rel), "/by_family")
            for cell in entry.get("recall_by_compaction", [])
        }
        counted = (
            "the number of compactions the probe saw (a compaction at the decision point's "
            "start runs after its probe)"
            if fields == {stats.RECALL_COUNT_FIELD}
            else "the number of compactions before it"
        )
        lines += [
            f"Recall of the rule in the probe just before each decision point, by {counted} "
            f"({EXPLORATORY}; descriptive, Wilson intervals that ignore clustering).",
            "",
            *table(["Family", "Compactions before", "Recalled", "Probed", "Recall"], recall_rows),
            "",
        ]
    return lines


def e1_km(b: Build, status: Status) -> list[str]:
    """Kaplan-Meier summary table and figure (descriptive)."""
    rel = f"{E1}/km.json"
    if not b.store.exists(rel):
        return []
    led = b.led
    rows, series = [], []
    level = 0.95
    for entry in led.select(At(rel), "/curves"):
        family = str(entry["family"])
        at = At(rel, "/curves", {"family": family})
        cid = f"e1.km.{slug(family)}"
        if entry.get("median") is not None:
            median = led.num(f"{cid}.median", at, "/median", TOOLS)
        else:
            within = led.num(f"{cid}.median.within", at, "/median_not_reached_within", COUNT)
            median = f"not reached within {within} tool calls"
        rows.append(
            [
                family,
                led.num(f"{cid}.n", at, "/n", COUNT),
                led.num(f"{cid}.events", at, "/n_events", COUNT),
                median,
            ]
        )
        level = float(entry.get("level", level))
        steps = entry.get("steps", [])
        series.append(
            svgplot.StepSeries(
                family,
                [float(s["time"]) for s in steps],
                [float(s["survival"]) for s in steps],
                lower=[s.get("lower") for s in steps],
                upper=[s.get("upper") for s in steps],
                censored_x=[float(s["time"]) for s in steps if int(s.get("n_censor", 0)) > 0],
            )
        )
    lines = [
        f"**Time to first violation ({EXPLORATORY}; descriptive).** Kaplan-Meier curves depend "
        "on how often decision points occur, so they describe the sessions and do not estimate "
        "the hazard.",
        "",
        *table(
            ["Family", "Sessions", "First violations", "Median tool calls to a first violation"],
            rows,
        ),
        "",
    ]
    if series:
        band = f"{pct(level)} interval (Greenwood, log-log)"
        svg = svgplot.step_chart(
            "E1: time to first violation (descriptive)",
            "Tool calls since the rules were stated",
            "Sessions without a violation",
            series,
            band_label=band,
            note="Kaplan-Meier curves; descriptive only, since they depend on how often decision "
            "points occur. Ticks mark censored sessions.",
            synthetic=status.synthetic,
        )
        number = b.figure(FIG_KM, svg)
        caption_level = b.level(
            level, "Greenwood log-log intervals", "interval level recorded in km.json (design)"
        )
        lines += [
            "![Kaplan-Meier curves of time to first violation per family, descriptive]"
            "(figures/e1-km.svg)",
            "",
            f"*Figure {number}. Kaplan-Meier curves of time to the first violation per family "
            f"({EXPLORATORY}; descriptive), with {caption_level}; ticks mark censored sessions."
            + (" Synthetic data, not model results.*" if status.synthetic else "*"),
            "",
        ]
    return lines


def hazard_figure(b: Build, status: Status) -> list[str]:
    """Figure: predicted hazard by position per family, with pointwise bootstrap bands.

    The curve is the half-life model (``log1p(position)`` and the compaction count, at the
    preregistered compaction count) fit to the banked E1 rows, drawn from each family's
    first decision point (where the half-life search starts), with pointwise bootstrap
    bands from the same settings, so it crosses one half at the banked half-life estimate.
    A family whose fit is separated (no half-life, D-029) gets no curve.
    """
    sessions, opportunities = f"{E1}/sessions.jsonl", f"{E1}/opportunities.jsonl"
    if not (b.store.exists(sessions) and b.store.exists(opportunities)):
        return []
    settings = e1_settings(b.store)
    rows, _ = e1_analysis.analysis_set(b.store.rows(sessions), b.store.rows(opportunities))
    step = max(1, settings.max_position // 50)
    grid = [float(p) for p in range(0, settings.max_position + 1, step)]
    fixed = dict(e1_analysis.HALF_LIFE_FIXED)
    series = []
    for family in e1_analysis.families(rows):
        subset = [dict(r) for r in rows if r["family"] == family]
        # From the family's first decision point, where the half-life search starts.
        first = min(float(r["position"]) for r in subset)
        positions = [first, *(p for p in grid if p > first)]
        try:
            curve = stats.bootstrap_hazard_curve(
                subset,
                positions,
                fixed,
                settings.n_boot,
                settings.seed,
                level=settings.level,
                covariates=e1_analysis.HAZARD_COVARIATES,
                categorical=(),
                position_transform=e1_analysis.TRANSFORM,
            )
        except stats.StatsError:
            continue
        series.append(
            svgplot.Series(
                family,
                positions,
                [_finite(i.estimate) for i in curve.intervals],
                lower=[_finite(i.lo) for i in curve.intervals],
                upper=[_finite(i.hi) for i in curve.intervals],
            )
        )
    if not series:
        return []
    svg = svgplot.line_chart(
        "E1: predicted hazard of a violation by position",
        "Tool calls since the rules were stated",
        "Hazard per decision point",
        series,
        band_label=f"{pct(settings.level)} pointwise session-bootstrap band",
        reference_y=0.5,
        reference_label="Hazard of one half (the half-life)",
        note="Half-life model fit to the banked E1 rows at the preregistered compaction count, "
        "drawn from each family's first decision point; bands are pointwise, computed by make "
        "report with the banked bootstrap settings. A family whose fit is separated has no "
        "curve.",
        synthetic=status.synthetic,
    )
    number = b.figure(FIG_HAZARD, svg)
    level = b.level(
        settings.level,
        "pointwise session-bootstrap bands",
        "interval level fixed in the E1 analysis settings (design parameter)",
    )
    return [
        "![Predicted hazard of a violation by tool-call position per family, with bootstrap "
        "bands](figures/e1-hazard.svg)",
        "",
        f"*Figure {number}. Predicted per-decision-point hazard of a violation by position from "
        "the half-life model, per family, from its first decision point, with "
        f"{level} computed by `make report` from the banked E1 rows with the banked bootstrap "
        "settings; a curve crosses the dashed line at its half-life, and a curve that starts "
        "above it has its half-life at or before its first point."
        + (" Synthetic data, not model results.*" if status.synthetic else "*"),
        "",
    ]


def _finite(value: Any) -> float | None:
    if isinstance(value, int | float) and value == value and abs(value) != float("inf"):
        return float(value)
    return None


def e1_section(b: Build) -> tuple[list[str], list[str]]:
    """Section 5.5: E1; returns the lines and the abstract sentences."""
    lines = ["### 5.5 Main study (E1)", ""]
    rel = f"{E1}/half_life.json"
    if not (b.store.exists(rel) and b.store.exists(f"{E1}/hazard.json")):
        lines.append(
            "Status: not run. The main study (E1) has not run, so this report has no estimate of "
            "constraint half-life, of the effect of compaction, or of differences between rule "
            "types."
        )
        return lines, []
    led = b.led
    status = status_of(b.store, E1, f"{E1}/sessions.jsonl")
    at = At(rel)
    settings = e1_settings(b.store)
    families = [str(e["family"]) for e in led.select(at, "/estimates")]
    excl = {
        name: led.num(f"e1.set.{name}", at, f"/exclusions/{field}", COUNT)
        for name, field in (
            ("opportunities", "opportunities_analyzed"),
            ("sessions", "sessions_total"),
            ("confirmatory", "sessions_confirmatory"),
            ("leaked", "sessions_probe_leak"),
            ("not_reached", "sessions_not_reaching_first_dp"),
            ("zero_calls", "opportunities_zero_calls"),
        )
    }
    analysis = (
        f"The analysis set kept {excl['opportunities']} decision points from confirmatory "
        "sessions that reached their first decision point; a session is one family's run of "
        f"one plan. Of {excl['sessions']} banked sessions, {excl['confirmatory']} were "
        f"confirmatory; {excl['leaked']} of those were excluded for a probe leak and "
        f"{excl['not_reached']} did not reach a decision point; {excl['zero_calls']} decision "
        "points with no tool call in their span were excluded."
    )
    sessions = f"{E1}/sessions.jsonl"
    if b.store.exists(sessions):
        per_family = [
            f"{family} "
            + led.count(
                f"e1.reached.{slug(family)}",
                sessions,
                {"family": family, "confirmatory": True, "reached_first_dp": True},
            )
            for family in families
        ]
        gate = led.allowed(
            "40", "at least 40 per family", "G7 minimum sessions per family (section 6)"
        )
        analysis += (
            " Confirmatory sessions that reached their first decision point: "
            f"{join_and(per_family)} (gate G7 needs {gate})."
        )
    excluded = []
    for family in families:
        fat = At(rel, "/exclusions_by_family", {"family": family})
        if led.peek(fat) is None:
            continue
        cid = f"e1.set.{slug(family)}"
        not_reached = led.num(f"{cid}.not_reached", fat, "/sessions_not_reaching_first_dp", COUNT)
        leaked = led.num(f"{cid}.leaked", fat, "/sessions_probe_leak", COUNT)
        excluded.append(f"{family} {not_reached} and {leaked}")
    if excluded:
        analysis += (
            " Sessions excluded per family (did not reach a decision point, probe leak): "
            f"{'; '.join(excluded)}."
        )
    level = b.level(
        settings.level,
        "cluster-bootstrap percentile intervals",
        "interval level fixed in the E1 analysis settings (design parameter)",
    )
    analysis += (
        f" Model intervals are {level} from "
        f"{led.num('e1.n_boot', at, '/settings/n_boot', COUNT)} resamples of whole plans "
        f"(seed {led.num('e1.seed', at, '/settings/seed', COUNT)}): within one family a plan "
        "is one session, and pooled estimates draw both families' sessions of a plan "
        "together; risk sets are cut per family and session."
    )
    hl_rows, phrases = half_life_rows(b)
    fixed = led.num("e1.fixed.compactions", at, "/fixed/compaction_count", COUNT)
    cap = led.num("e1.max_position", at, "/max_position", COUNT)
    lines += [
        status.text,
        "",
        analysis,
        "",
        "**Half-life.** The half-life is the number of tool calls since the rules were stated at "
        "which the predicted per-decision-point hazard of a violation reaches one half, from a "
        "hazard model with `log1p(position)` and the compaction count, evaluated at a "
        f"compaction count of {fixed}. It is searched from each family's first decision point, "
        "so the hazard is never extrapolated to positions without data: when the hazard is "
        "already at or above one half there, the half-life reads at or before that position, "
        f"and when it stays below one half within {cap} tool calls, it is not reached. When "
        "the upper percentile of the bootstrap draws falls among draws that never reach one "
        "half, the upper bound is open and written as >N (D-028); the last column gives the "
        "share of draws that never reach one half, which can be above zero under a finite "
        "upper bound. A fit whose violations separate a term of the model gives no half-life "
        "(not estimable), since its crossing would be set by the ridge (D-029).",
        "",
        *table(["Family", "Half-life", "Interval", "Draws not reaching one half"], hl_rows),
        "",
    ]
    lines += hazard_figure(b, status)
    lines += e1_tables(b, families)
    lines += e1_descriptive(b)
    lines += e1_km(b, status)
    kind = "confirmatory" if status.confirmatory else "exploratory"
    if status.synthetic:
        kind += ", synthetic data"
    abstract = [
        f"In E1 ({kind}), the constraint half-life at a compaction count of {fixed} was "
        f"{join_and(phrases)}."
    ]
    pooled_h2b, pooled_reading = coefficient_cell(led, "pooled", "compaction_count", "h2b")
    abstract.append(
        f"The pooled compaction-count coefficient (H2b) was {pooled_h2b} log odds "
        f"({pooled_reading})."
    )
    return lines, abstract


# --------------------------------------------------------------------------- E2


def e2_section(b: Build) -> tuple[list[str], list[str]]:
    """Section 5.6: E2, the ecological arm (rates per model tier, or counts before analysis)."""
    lines = ["### 5.6 Production agent (E2, ecological)", ""]
    sessions, rates = f"{E2}/sessions.jsonl", f"{E2}/rates.json"
    if not (b.store.exists(sessions) or b.store.exists(rates)):
        lines.append(
            "Status: not run. E2, the ecological arm with Claude Code, has not run: it needs a "
            "machine where the billing guard lets the Claude backend start."
        )
        return lines, []
    led = b.led
    status = status_of(b.store, E2, sessions)
    lines += [
        f"{status.text} Ecological arm: Claude Code on the same session templates with its own "
        "auto-compaction (D-044), labeled ecological in every table; no hazard model is fit to "
        "this arm.",
        "",
    ]
    kind = "confirmatory" if status.confirmatory else "exploratory"
    if status.synthetic:
        kind += ", synthetic data"
    if b.store.exists(rates):
        at = At(rates)
        settings = derive.settings_params(b.store.obj(rates), rates)
        level = b.level(
            float(settings["level"]),
            "Wilson score intervals",
            "interval level fixed in the E2 analysis settings (design parameter)",
        )
        rows, bits = [], []
        for entry in led.select(at, "/model"):
            tier = str(entry["model"])
            tier_at = At(rates, "/model", {"model": tier})
            cid = f"e2.{slug(tier)}"
            cells = rate_cells(led, cid, tier_at, entry)
            rows.append([tier, led.num(f"{cid}.sessions", tier_at, "/sessions", COUNT), *cells])
            bits.append(f"{cells[1]} for {tier}")
        excl = "/exclusions"
        lines += [
            "Violation rate and recall before the decision point per model tier, over sessions "
            "that reached their first decision point "
            f"({led.num('e2.set.opportunities', at, f'{excl}/opportunities_analyzed', COUNT)} "
            "decision points; "
            f"{led.num('e2.set.not_reached', at, f'{excl}/sessions_not_reaching_first_dp', COUNT)}"
            f" of {led.num('e2.set.sessions', at, f'{excl}/sessions_total', COUNT)} sessions "
            f"did not reach one), with {level} that ignore clustering by session.",
            "",
            *table(
                [
                    "Model tier (ecological)",
                    "Sessions",
                    "Decision points",
                    "Violation rate",
                    "Recall before",
                ],
                rows,
            ),
        ]
        sentence = f"In E2 (ecological, {kind}), violation rates were {join_and(bits)}."
        return lines, [sentence]
    opportunities = f"{E2}/opportunities.jsonl"
    has_opps = b.store.exists(opportunities)
    rows = []
    for tier in sorted({str(row.get("model")) for row in b.store.rows(sessions)}):
        cid = f"e2.{slug(tier)}"
        row = [
            tier,
            led.count(f"{cid}.sessions", sessions, {"model": tier}),
            led.count(f"{cid}.reached", sessions, {"model": tier, "reached_first_dp": True}),
        ]
        if has_opps:
            row += [
                led.count(f"{cid}.dps", opportunities, {"model": tier}),
                led.count(f"{cid}.violations", opportunities, {"model": tier, "violated": True}),
            ]
        else:
            row += [NOT_ESTIMABLE, NOT_ESTIMABLE]
        rows.append(row)
    lines += [
        "The E2 analysis (`python -m halflife.arms_analysis e2`) has not been run on these rows, "
        "so only counts are shown.",
        "",
        *table(
            [
                "Model tier (ecological)",
                "Sessions",
                "Reached the first decision point",
                "Decision points",
                "Violations",
            ],
            rows,
        ),
    ]
    return lines, [f"E2 (ecological, {kind}) ran; its counts are in section 5.6."]


# --------------------------------------------------------------------------- E3 and E4


def analysis_set_sentence(led: Ledger, rel: str, prefix: str) -> str:
    """The analysis-set counts an E3 or E4 output records (preregistration section 7 as in
    E1: counted and reported), in total and per family; empty when none are recorded."""
    at = At(rel)
    if led.peek(at, "/exclusions/sessions_total") is None:
        return ""
    excl = {
        name: led.num(f"{prefix}.set.{name}", at, f"/exclusions/{field}", COUNT)
        for name, field in (
            ("opportunities", "opportunities_analyzed"),
            ("analyzed", "sessions_analyzed"),
            ("sessions", "sessions_total"),
            ("confirmatory", "sessions_confirmatory"),
            ("leaked", "sessions_probe_leak"),
            ("not_reached", "sessions_not_reaching_first_dp"),
            ("zero_calls", "opportunities_zero_calls"),
        )
    }
    text = (
        f"The analysis set kept {excl['opportunities']} decision points from "
        f"{excl['analyzed']} of {excl['sessions']} banked sessions "
        f"({excl['confirmatory']} confirmatory; {excl['leaked']} excluded for a probe leak and "
        f"{excl['not_reached']} did not reach a decision point; {excl['zero_calls']} decision "
        "points with no tool call in their span were excluded)."
    )
    parts = []
    for entry in led.peek(at, "/exclusions_by_family") or []:
        family = str(entry.get("family"))
        fat = At(rel, "/exclusions_by_family", {"family": family})
        cid = f"{prefix}.set.{slug(family)}"
        not_reached = led.num(f"{cid}.not_reached", fat, "/sessions_not_reaching_first_dp", COUNT)
        leaked = led.num(f"{cid}.leaked", fat, "/sessions_probe_leak", COUNT)
        parts.append(f"{family} {not_reached} and {leaked}")
    if parts:
        text += (
            " Sessions excluded per family (did not reach a decision point, probe leak): "
            f"{'; '.join(parts)}."
        )
    return text


def family_label(family: str) -> str:
    """Table label of a family entry: per-family E3 and E4 contrasts are exploratory."""
    return family if family == "pooled" else f"{family} (exploratory)"


def family_entries(led: Ledger, rel: str) -> list[tuple[int, str]]:
    """``(index, family)`` of every ``by_family`` entry, in artifact order."""
    return [(i, str(e["family"])) for i, e in enumerate(led.select(At(rel), "/by_family"))]


def rate_cells(led: Ledger, cid: str, at: At, entry: Mapping[str, Any]) -> list[str]:
    """Decision points, violation rate [interval], and recall [interval] of one group."""
    recall = (
        estimate(led, f"{cid}.recall", at, "/recall_before", "value", PROP)
        if "recall_before" in entry
        else NOT_ESTIMABLE
    )
    return [
        led.num(f"{cid}.n", at, "/opportunities", COUNT),
        estimate(led, f"{cid}.rate", at, "/violation_rate", "value", PROP),
        recall,
    ]


def e3_section(b: Build) -> tuple[list[str], list[str]]:
    """Section 5.7: placement and phrasing (E3)."""
    lines = ["### 5.7 Placement and phrasing (E3)", ""]
    contrasts = f"{E3}/contrasts.json"
    if not b.store.exists(contrasts):
        lines.append(
            "Status: not run. E3 has not run, so this report has no placement or phrasing results."
        )
        return lines, []
    led = b.led
    status = status_of(b.store, E3, f"{E3}/sessions.jsonl")
    settings = derive.settings_params(b.store.obj(contrasts), contrasts)
    level = b.level(
        float(settings["level"]),
        "cluster-bootstrap percentile intervals",
        "interval level fixed in the E3 analysis settings (design parameter)",
    )
    lines += [
        status.text,
        "",
        "Violation rates by where the rules are stated and how they are phrased, and contrasts "
        "of each placement against the first user message (Holm adjustment over the placements) "
        "and of a stated reason against the bare rule, with "
        f"{level} and p-values of a score test at the null fit, clustered the same way (by "
        "session within a family, by plan when families are pooled) through a within-cluster "
        "correlation estimated at the null fit. A contrast "
        f"differs when its Holm-adjusted p-value is below {alpha_phrase(led, contrasts)}, and "
        "only then does the reading give a direction; the interval is descriptive. A contrast "
        "with no valid test, or an unstable one, is not tested and stays in its Holm family as "
        "a p-value of one. The pooled contrasts are the preregistered tests; the per-family "
        "rows are exploratory. " + analysis_set_sentence(led, contrasts, "e3"),
        "",
    ]
    rates = f"{E3}/rates.json"
    if b.store.exists(rates):
        rows = []
        for index, family in family_entries(led, rates):
            if family != "pooled":
                continue
            for key in ("placement", "phrasing"):
                for entry in led.select(At(rates), f"/by_family/{index}/{key}"):
                    arm = str(entry[key])
                    at = At(rates, f"/by_family/{index}/{key}", {key: arm})
                    rows.append(
                        [
                            f"{key} `{arm}`",
                            *rate_cells(led, f"e3.rate.{key}.{slug(arm)}", at, entry),
                        ]
                    )
        header = [
            "Condition (pooled)",
            "Decision points",
            "Violation rate",
            "Recall before (exploratory)",
        ]
        lines += [
            f"Recall by placement and phrasing is {EXPLORATORY} (preregistration 0002); a "
            "reminder sent with a decision-point request reaches the agent after that point's "
            "probe, so the probe understates recall there (counted in `rates.json` as "
            "`reminders_on_request`).",
            "",
            *table(header, rows),
            "",
        ]
    rows, figure_rows, abstract_bits = [], [], []
    for index, family in family_entries(led, contrasts):
        for key in ("placement", "phrasing"):
            for entry in led.select(At(contrasts), f"/by_family/{index}/{key}"):
                arm, ref = str(entry[key]), str(entry["reference"])
                at = At(contrasts, f"/by_family/{index}/{key}", {key: arm})
                cid = f"e3.{slug(family)}.{slug(arm)}"
                cells = contrast_cells(led, cid, at, odds=False)
                rows.append([family_label(family), f"`{arm}` vs `{ref}`", *cells])
                figure_rows.append(forest_row(led, at, f"{family}: {arm} vs {ref}", family))
                if family == "pooled":
                    abstract_bits.append(
                        f"{arm} against {ref} {cells[0]} (Holm p {cells[-2]}; {cells[-1]})"
                    )
    header = ["Family", "Contrast", "Difference", "p", "Holm p", "Reading"]
    lines += [*table(header, rows), ""]
    lines += e3_hazard_terms(b)
    lines += contrast_figure(b, FIG_E3, "E3: placement and phrasing contrasts", figure_rows, status)
    kind = "confirmatory" if status.confirmatory else "exploratory"
    if status.synthetic:
        kind += ", synthetic data"
    abstract = (
        [f"In E3 ({kind}), pooled differences in violation rate were {join_and(abstract_bits)}."]
        if abstract_bits
        else []
    )
    return lines, abstract


def e3_hazard_terms(b: Build) -> list[str]:
    """Pooled placement and phrasing terms of the E3 hazard model."""
    rel = f"{E3}/hazard.json"
    if not b.store.exists(rel):
        return []
    led, rows = b.led, []
    for entry in led.select(At(rel), "/estimates"):
        term, family = str(entry["term"]), str(entry["family"])
        if family != "pooled" or not term.startswith(("placement[", "phrasing[")):
            continue
        at = At(rel, "/estimates", {"family": family, "term": term})
        cid = f"e3.hazard.{slug(term)}"
        missing = not_estimable_cell(led, at)
        if missing is not None:
            rows.append([f"`{term}`", missing, "no reading"])
            continue
        words = (
            "unstable (too many failed draws)"
            if led.peek(at, "/unstable") is True
            else reading(led, at, "", "higher hazard than the reference", "lower hazard")
        )
        rows.append([f"`{term}`", estimate(led, cid, at, "", "estimate", COEF), words])
    if not rows:
        return []
    return [
        f"Placement and phrasing terms of the pooled hazard model ({EXPLORATORY}; log odds "
        "against the reference levels, with `log1p(position)`, the compaction count, the rule "
        "type, and the family in the model; a separated term has no estimate).",
        "",
        *table(["Term", "Coefficient", "Reading"], rows),
        "",
    ]


def forest_row(led: Ledger, at: At, label: str, group: str) -> svgplot.ForestRow:
    """A forest-plot row for one contrast (difference and its interval)."""
    diff = led.peek(at, "/difference") or {}
    p_holm = led.peek(at, "/p_holm")
    if isinstance(p_holm, int | float):
        text = f"Holm p {p_holm:{p_format(p_holm)}}"
    else:
        text = "Holm p not computed"
    return svgplot.ForestRow(
        label,
        _finite(diff.get("difference")),
        _finite(diff.get("lo")),
        _finite(diff.get("hi")),
        text=text,
        group=group,
    )


def contrast_figure(
    b: Build, rel: str, title: str, rows: Sequence[svgplot.ForestRow], status: Status
) -> list[str]:
    """A forest plot of contrasts, one color per family."""
    if not rows:
        return []
    svg = svgplot.forest_plot(
        title,
        "Difference in violation rate (arm minus reference)",
        "Contrast",
        rows,
        reference_label="No difference",
        text_header="Holm-adjusted p",
        width=880,
        synthetic=status.synthetic,
    )
    number = b.figure(rel, svg)
    name = rel.rsplit("/", 1)[1]
    what = title.split(": ", 1)[1]
    return [
        f"![Forest plot of {what}, differences in violation rate with intervals](figures/{name})",
        "",
        f"*Figure {number}. {title.split(':', 1)[0]} {what}: differences in violation rate with "
        "cluster-bootstrap intervals (sessions within a family, plans when pooled), one color "
        "per family; the right column gives the "
        "Holm-adjusted p-value."
        + (" Synthetic data, not model results.*" if status.synthetic else "*"),
        "",
    ]


COST_FIELDS = (
    ("tokens", "Tokens per session", signed_amount),
    ("latency_s", "Seconds per session", signed_amount),
    ("task_success", "Task success", DIFF),
)


def e4_section(b: Build) -> tuple[list[str], list[str]]:
    """Section 5.8: mitigations with their costs (E4) and the tools' overhead."""
    lines = ["### 5.8 Mitigations (E4)", ""]
    contrasts, costs, summary = f"{E4}/contrasts.json", f"{E4}/costs.json", f"{E4}/summary.json"
    abstract: list[str] = []
    if not b.store.exists(contrasts):
        lines += [
            "Status: not run. E4 has not run, so this report has no mitigation results.",
            "",
        ]
    else:
        led = b.led
        status = status_of(b.store, E4, f"{E4}/sessions.jsonl")
        settings = derive.settings_params(b.store.obj(contrasts), contrasts)
        level = b.level(
            float(settings["level"]),
            "cluster-bootstrap percentile intervals",
            "interval level fixed in the E4 analysis settings (design parameter)",
        )
        scope = ""
        if b.store.exists(summary):
            at = At(summary)
            scope = (
                f" on {led.size('e4.templates', At(summary, '/held_out_templates'))} held-out "
                f"templates ({led.num('e4.sessions', at, '/n_sessions', COUNT)} banked sessions)"
            )
        lines += [
            status.text,
            "",
            f"Each mitigation against no mitigation{scope}: the difference in violation rate "
            f"with {level}, p-values of a score test at the null fit, clustered the same way "
            "(by session within a family, by plan when families are pooled) through a "
            "within-cluster correlation estimated at the null fit, with "
            "Holm adjustment over the mitigations; a mitigation differs from no mitigation when "
            f"its Holm-adjusted p-value is below {alpha_phrase(led, contrasts)}, and an arm "
            "without violations is still tested. The pooled contrasts are the preregistered "
            "tests; the per-family rows are exploratory. Each cost sits beside its effect "
            "(rule 15) as a difference from no mitigation: tokens per session (step and "
            "compaction calls; recall probes are measurement and are left out), backend seconds "
            "per session, and task success, the share of the plan's subtasks with a success "
            "check that passed (a checked subtask the session never reached counts as not "
            "passed). As D-039 fixes, the guard arm is reported as an upper bound on what the "
            "mechanism blocks, and its refusal reasons restate the rule; its static analysis "
            "cannot see writes or network calls made inside programs the agent runs. "
            + analysis_set_sentence(led, contrasts, "e4")
            + " Costs use every confirmatory session (D-040), not only the analysis set, since "
            "a mitigation could end sessions early.",
            "",
        ]
        cost_index = {f: i for i, f in family_entries(led, costs)} if b.store.exists(costs) else {}
        rows, figure_rows = [], []
        for index, family in family_entries(led, contrasts):
            for entry in led.select(At(contrasts), f"/by_family/{index}/mitigation"):
                arm = str(entry["mitigation"])
                at = At(contrasts, f"/by_family/{index}/mitigation", {"mitigation": arm})
                cid = f"e4.{slug(family)}.{slug(arm)}"
                cells = contrast_cells(led, cid, at, odds=False)
                cost_cells = []
                for name, _, fmt in COST_FIELDS:
                    if family in cost_index:
                        cat = At(
                            costs, f"/by_family/{cost_index[family]}/arms", {"mitigation": arm}
                        )
                        if led.peek(cat, f"/{name}_vs_none") is not None:
                            cost_cells.append(
                                estimate(
                                    led, f"{cid}.{name}", cat, f"/{name}_vs_none", "difference", fmt
                                )
                            )
                            continue
                    cost_cells.append("not measured")
                rows.append([family_label(family), f"`{arm}`", *cells, *cost_cells])
                figure_rows.append(forest_row(led, at, f"{family}: {arm} vs none", family))
                if family == "pooled":
                    abstract.append(
                        f"{arm} {cells[0]} (Holm p {cells[-2]}; {cells[-1]}; tokens per "
                        f"session {cost_cells[0]})"
                    )
        header = ["Family", "Mitigation", "Difference", "p", "Holm p", "Reading"]
        header += [label for _, label, _ in COST_FIELDS]
        lines += [*table(header, rows), ""]
        lines += control_costs(b, costs, cost_index)
        lines += contrast_figure(
            b, FIG_E4, "E4: mitigations against no mitigation", figure_rows, status
        )
        kind = "confirmatory" if status.confirmatory else "exploratory"
        if status.synthetic:
            kind += ", synthetic data"
        abstract = (
            [
                f"In E4 ({kind}), pooled differences in violation rate against no mitigation "
                f"were {join_and(abstract)}."
            ]
            if abstract
            else []
        )
    lines += overhead_text(b)
    return lines, abstract


def control_costs(b: Build, costs: str, cost_index: Mapping[str, int]) -> list[str]:
    """Pooled costs of the no-mitigation arm, the baseline of every cost difference."""
    if "pooled" not in cost_index:
        return []
    led = b.led
    at = At(costs, f"/by_family/{cost_index['pooled']}/arms", {"mitigation": "none"})
    if led.peek(at) is None:
        return []
    parts = []
    for name, label, _ in COST_FIELDS:
        if led.peek(at, f"/{name}") is not None:
            fmt = PROP if name == "task_success" else amount_format
            parts.append(
                f"{label.lower()} {estimate(led, f'e4.none.{name}', at, f'/{name}', 'mean', fmt)}"
            )
    if not parts:
        return []
    return [f"Without a mitigation (pooled), the means were: {join_and(parts)}.", ""]


def overhead_text(b: Build) -> list[str]:
    """Measured overhead of halflife-guard and halflife-pin (G9)."""
    if not b.store.exists(OVERHEAD):
        return ["**Overhead of the shipped tools.** Not measured yet."]
    label = f"**Overhead of the shipped tools ({EXPLORATORY}).**"
    led, at = b.led, At(OVERHEAD)
    guard_ms = led.num("e4.overhead.guard.ms", at, "/halflife-guard/latency_ms", ".1f")
    guard_tokens = led.num("e4.overhead.guard.tokens", at, "/halflife-guard/tokens", amount_format)
    pin_ms = led.num("e4.overhead.pin.ms", at, "/halflife-pin/latency_ms", ".2f")
    pin_tokens = led.num("e4.overhead.pin.tokens", at, "/halflife-pin/tokens", amount_format)
    return [
        f"{label} halflife-guard took {guard_ms} ms per guarded tool "
        f"call (the median hook run as a subprocess) and adds {guard_tokens} tokens per refusal "
        f"reason (allowed calls add none); halflife-pin took {pin_ms} ms to render and apply its "
        f"block, which puts {pin_tokens} tokens in the context for the whole session "
        "(`artifacts/e4/overhead.json`; docs/MITIGATIONS.md says how each is measured)."
    ]


# --------------------------------------------------------------------------- abstract, headline


def e0_abstract(b: Build) -> list[str]:
    """Abstract sentences for the benchmark and E0."""
    led, out = b.led, []
    if b.store.exists(BENCH):
        out.append(
            "The benchmark has "
            f"{led.num('bench.templates', At(BENCH), '/coverage/templates', COUNT)} synthetic "
            "session templates, each with a banked compliant route and reference run."
        )
    parts = []
    if b.store.exists(DETECTORS):
        at = At(DETECTORS)
        precision = estimate(led, "e0.precision", at, "/precision", "value", PROP)
        recall = estimate(led, "e0.recall", at, "/recall", "value", PROP)
        cases = led.size("e0.cases", At(DETECTORS, "/cases"))
        parts.append(
            f"the detectors reached precision {precision} and recall {recall} on {cases} seeded "
            "synthetic action logs"
        )
    if b.store.exists(PROBE_SCORING):
        at, test = At(PROBE_SCORING), "/by_split/test"
        sens = estimate(led, "e0.probe.sensitivity", at, f"{test}/sensitivity", "value", PROP)
        spec = estimate(led, "e0.probe.specificity", at, f"{test}/specificity", "value", PROP)
        parts.append(
            f"the recall-probe scorer reached sensitivity {sens} and specificity {spec} on the "
            "test split of synthetic answers with LLM labels"
        )
    if parts:
        level = b.level(E0_LEVEL, "Wilson score intervals", E0_REASON)
        out.append(f"Before any model run, {join_and(parts)} ({level}).")
    return out


def headline(b: Build) -> str:
    """The README headline block: one finding with its interval, a figure, and a table."""
    led = b.led
    rel = f"{E1}/half_life.json"
    if b.store.exists(rel) and FIG_HAZARD in b.figures:
        status = status_of(b.store, E1, f"{E1}/sessions.jsonl")
        level = b.level(
            e1_settings(b.store).level, "confidence interval", "interval level of E1 (design)"
        )
        rows, phrases = [], []
        for family in [str(e["family"]) for e in led.select(At(rel), "/estimates")]:
            at = At(rel, "/estimates", {"family": family})
            cid = f"e1.halflife.{slug(family)}"
            interval = half_life_span(led, cid, at)
            point = half_life_point(led, cid, at)
            rows.append([family, point, interval])
            if interval == "no interval":
                phrases.append(f"{point} for {family}")
            else:
                phrases.append(f"{point} for {family} ({level} {interval})")
        kind = "confirmatory" if status.confirmatory else "exploratory"
        if status.synthetic:
            kind += "; synthetic data, not model results"
        finding = f"**Finding.** In E1 ({kind}), the constraint half-life was {join_and(phrases)}."
        figure = (
            "![Predicted hazard of a violation by position per family](docs/figures/e1-hazard.svg)"
        )
        header = ["Family", "Half-life", level[0].upper() + level[1:]]
        return "\n".join([finding, "", figure, "", *table(header, rows)])
    if not (b.store.exists(DETECTORS) and FIG_E0 in b.figures):
        return "No study results are banked yet."
    at = At(DETECTORS)
    level = b.level(E0_LEVEL, "confidence interval", E0_REASON)
    finding = (
        "**Finding so far.** Before any model run, the rule-violation detectors reached precision "
        f"{led.num('e0.precision', at, '/precision/value', PROP)} ({level} "
        f"{span(led, 'e0.precision', at, '/precision', PROP)}) and recall "
        f"{led.num('e0.recall', at, '/recall/value', PROP)} "
        f"({span(led, 'e0.recall', at, '/recall', PROP)}) on "
        f"{led.size('e0.cases', At(DETECTORS, '/cases'))} seeded synthetic action logs; the main "
        "study has not run."
    )
    rows = [
        [
            "Detector precision",
            led.num("e0.precision", at, "/precision/value", PROP),
            span(led, "e0.precision", at, "/precision", PROP),
        ],
        [
            "Detector recall",
            led.num("e0.recall", at, "/recall/value", PROP),
            span(led, "e0.recall", at, "/recall", PROP),
        ],
    ]
    if b.store.exists(PROBE_SCORING):
        probe, test = At(PROBE_SCORING), "/by_split/test"
        for name in ("sensitivity", "specificity"):
            rows.append(
                [
                    f"Recall-probe scorer {name} (test split, LLM-labeled synthetic answers)",
                    led.num(f"e0.probe.{name}", probe, f"{test}/{name}/value", PROP),
                    span(led, f"e0.probe.{name}", probe, f"{test}/{name}", PROP),
                ]
            )
    notes = []
    if b.store.exists(PROBE_LEAKAGE):
        leak = At(PROBE_LEAKAGE)
        leaks = led.num("e0.leak.leaks", leak, "/leaks", COUNT)
        runs = led.num("e0.leak.runs", leak, "/validation_runs", COUNT)
        notes.append(
            f"Recall probes leaked into the main run in {leaks} of {runs} validation runs."
        )
    opportunities = f"{PILOT}/opportunities.jsonl"
    if b.store.exists(opportunities):
        notes.append(
            "In an exploratory pilot of small open-weight models, agents broke a stated rule at "
            + led.count("pilot.violated_total", opportunities, {"violated": True})
            + " of "
            + led.count("pilot.dps_total", opportunities)
            + " decision points they reached."
        )
    figure = "![E0 instrument-validity measures with their intervals](docs/figures/e0-validity.svg)"
    header = ["Measure (synthetic, before any model run)", "Estimate", level[0].upper() + level[1:]]
    tail = ["", " ".join(notes)] if notes else []
    return "\n".join([finding, "", figure, "", *table(header, rows), *tail])


def not_run_sentence(names: Sequence[str]) -> str:
    """One sentence naming the experiments that have not run."""
    if not names:
        return ""
    verb = "has" if len(names) == 1 else "have"
    return f"{join_and(list(names))} {verb} not run."


# --------------------------------------------------------------------------- documents


@dataclass(frozen=True)
class Output:
    """What one run writes."""

    blocks: dict[str, str]
    headline: str
    figures: dict[str, str]
    manifest: dict[str, Any]


def build(root: Path, module: str = derive.DERIVE_MODULE, *, with_headline: bool = False) -> Output:
    """Build every generated block, figure, and the claims manifest (nothing is written).

    The README headline is built only ``with_headline`` (when README.md has its markers),
    so its allowlist entries never go unused.

    Raises:
        ReportError: when an artifact cannot be read or a value cannot be cited.
    """
    store = Store(root)
    b = Build(store, Ledger(store, module))
    try:
        results = [*bench_section(b), "", *e0_section(b), ""]
        results += [*pilot_section(b), "", *power_section(b), ""]
        e1_lines, e1_abstract = e1_section(b)
        e2_lines, e2_abstract = e2_section(b)
        e3_lines, e3_abstract = e3_section(b)
        e4_lines, e4_abstract = e4_section(b)
        for part in (e1_lines, e2_lines, e3_lines, e4_lines):
            results += [*part, ""]
        missing = [
            name
            for name, ran in (
                ("the main study (E1)", bool(e1_abstract)),
                ("the ecological arm (E2)", bool(e2_abstract)),
                ("the placement study (E3)", bool(e3_abstract)),
                ("the mitigation study (E4)", bool(e4_abstract)),
            )
            if not ran
        ]
        abstract = [
            "half-life measures how long coding agents keep rules stated at the start of a long "
            "session. It combines half-life-bench, a set of long multi-stage coding sessions with "
            "decision points where breaking a rule is the easiest route, a harness that controls "
            "session length, rule placement and phrasing, and context compaction, deterministic "
            "detectors of violations, and a discrete-time survival analysis.",
            *e0_abstract(b),
            *e1_abstract,
            *e2_abstract,
            *e3_abstract,
            *e4_abstract,
        ]
        if missing:
            sentence = not_run_sentence(missing)
            abstract.append(sentence[0].upper() + sentence[1:])
        if not e1_abstract:
            abstract.append("This report therefore has no estimate of constraint half-life yet.")
        head = headline(b) if with_headline else ""
        manifest = manifest_content(b.led, existing_manifest(root))
    except (stats.StatsError, svgplot.PlotError, derive.DeriveError) as exc:
        raise ReportError(f"{type(exc).__name__}: {exc}") from exc
    blocks = {
        "abstract": " ".join(abstract),
        "results": "\n".join(_collapse(results)).strip("\n"),
    }
    return Output(blocks, head, dict(sorted(b.figures.items())), manifest)


def _collapse(lines: Sequence[str]) -> list[str]:
    """Drop repeated blank lines."""
    out: list[str] = []
    for line in lines:
        if line == "" and (not out or out[-1] == ""):
            continue
        out.append(line)
    return out


@dataclass(frozen=True)
class Existing:
    """The parts of an existing claims manifest that were not generated."""

    documents: list[str]
    claims: dict[str, Any]
    allowlist: list[dict[str, Any]]
    derivations: dict[str, Any]
    raw: dict[str, str]


def existing_manifest(root: Path) -> Existing:
    """Hand-written entries of ``artifacts/claims.json`` (generated ones are dropped).

    Raises:
        ReportError: when the file exists but does not parse as a JSON object.
    """
    path = root / MANIFEST
    if not path.is_file():
        return Existing([], {}, [], {}, {})
    try:
        data = load_json(path)
    except IntegrityError as exc:
        raise ReportError(f"{MANIFEST}: {exc}") from exc
    if not isinstance(data, dict):
        raise ReportError(f"{MANIFEST} is not a JSON object")

    def mapping(key: str) -> dict[str, Any]:
        value = data.get(key, {})
        return dict(value) if isinstance(value, dict) else {}

    def listed(key: str) -> list[Any]:
        value = data.get(key, [])
        return list(value) if isinstance(value, list) else []

    documents = [d for d in listed("documents") if isinstance(d, str)]
    claims = {
        k: v
        for k, v in mapping("claims").items()
        if not (isinstance(v, dict) and v.get("note") == GENERATED)
    }
    allowlist = [
        entry
        for entry in listed("allowlist")
        if not (isinstance(entry, dict) and str(entry.get("reason", "")).endswith(TAG))
    ]
    derivations = {
        k: v
        for k, v in mapping("derivations").items()
        if not (isinstance(v, dict) and v.get("note") == GENERATED)
    }
    raw = {k: v for k, v in mapping("raw").items() if not str(v).endswith(TAG)}
    return Existing(documents, claims, allowlist, derivations, raw)


def manifest_content(led: Ledger, existing: Existing) -> dict[str, Any]:
    """The claims manifest: generated entries plus the kept hand-written ones.

    Raises:
        ReportError: when the result would not pass the manifest schema of claimcheck.
    """
    declared = set(led.declared.derivations) | set(led.declared.raw)
    claims = {k: v for k, v in existing.claims.items() if k not in led.claims} | led.claims
    generated = [
        {"value": value, "context": context, "reason": "; ".join(reasons) + TAG}
        for (value, context), reasons in led.allowlist.items()
    ]
    pairs = {(str(e.get("value")), str(e.get("context"))) for e in generated}
    kept = [
        e for e in existing.allowlist if (str(e.get("value")), str(e.get("context"))) not in pairs
    ]
    derivations = {
        k: v for k, v in existing.derivations.items() if k not in declared
    } | led.declared.derivations
    raw = {k: v for k, v in existing.raw.items() if k not in declared}
    raw |= {k: v + TAG for k, v in led.declared.raw.items()}
    content = {
        "schema": SCHEMA,
        "notes": NOTES,
        "documents": existing.documents,
        "claims": dict(sorted(claims.items())),
        "allowlist": kept + sorted(generated, key=lambda e: (e["context"], e["value"])),
        "derivations": dict(sorted(derivations.items())),
        "raw": dict(sorted(raw.items())),
    }
    _, problems = cc.parse_manifest(content)
    if problems:
        raise ReportError("the claims manifest would not validate: " + "; ".join(problems[:3]))
    return content


def replace_block(text: str, name: str, body: str) -> str:
    """``text`` with the content between the ``report`` markers named ``name`` replaced.

    Raises:
        ReportError: when the markers are missing, repeated, or out of order.
    """
    begin, end = f"<!-- report:begin {name} -->", f"<!-- report:end {name} -->"
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ReportError(f"expected one {begin} and one {end}")
    start, stop = text.index(begin) + len(begin), text.index(end)
    if stop < start:
        raise ReportError(f"{end} comes before {begin}")
    return text[:start] + "\n" + body.strip("\n") + "\n" + text[stop:]


def has_block(text: str, name: str) -> bool:
    """True when ``text`` has the begin marker of block ``name``."""
    return f"<!-- report:begin {name} -->" in text


def block_text(text: str, name: str) -> str | None:
    """The content between the ``report`` markers named ``name`` (None without markers)."""
    begin, end = f"<!-- report:begin {name} -->", f"<!-- report:end {name} -->"
    if text.count(begin) != 1 or text.count(end) != 1:
        return None
    start, stop = text.index(begin) + len(begin), text.index(end)
    return text[start:stop].strip("\n") if start <= stop else None


def stale_parts(root: Path, module: str = derive.DERIVE_MODULE) -> list[str]:
    """What ``make report`` would change now: generated blocks, figures, the claims manifest.

    Empty when the committed report matches a fresh build from the banked artifacts. A
    report written before an experiment was banked (it still says that experiment has not
    run) or before its artifacts changed is stale, which claimcheck and freshness cannot
    see because they only check the claims a document already carries.

    Raises:
        ReportError: when the report cannot be built.
    """
    report_path, readme_path = root / REPORT_DOC, root / README_DOC
    if not report_path.is_file():
        return [f"{REPORT_DOC} is missing"]
    readme = readme_path.read_text(encoding="utf-8") if readme_path.is_file() else ""
    out = build(root, module, with_headline=has_block(readme, "headline"))
    report = report_path.read_text(encoding="utf-8")
    stale = [
        f"{REPORT_DOC} block {name}"
        for name, body in out.blocks.items()
        if block_text(report, name) != body.strip("\n")
    ]
    if has_block(readme, "headline") and block_text(readme, "headline") != out.headline.strip("\n"):
        stale.append(f"{README_DOC} block headline")
    for rel, svg in out.figures.items():
        path = root / rel
        if not path.is_file() or path.read_text(encoding="utf-8") != svg:
            stale.append(rel)
    figures = root / FIGURES_DIR
    if figures.is_dir():
        stale += [
            p.relative_to(root).as_posix()
            for p in sorted(figures.glob("*.svg"))
            if p.relative_to(root).as_posix() not in out.figures
        ]
    manifest = root / MANIFEST
    if not manifest.is_file() or manifest.read_text(encoding="utf-8") != _dump(out.manifest):
        stale.append(MANIFEST)
    return stale


def block_problems(
    doc: str, text: str, names: Sequence[str], led_allow: Sequence[cc.AllowEntry], root: Path
) -> list[str]:
    """Claimcheck problems inside the generated blocks of a document (a generator bug)."""
    lines = text.split("\n")
    inside: set[int] = set()
    for name in names:
        begin = next(i for i, line in enumerate(lines) if f"<!-- report:begin {name} -->" in line)
        end = next(i for i, line in enumerate(lines) if f"<!-- report:end {name} -->" in line)
        inside |= set(range(begin + 1, end + 2))
    parsed = cc.parse_document(doc, text, root)
    unmarked, _ = cc.apply_allowlist(parsed.numbers, led_allow)
    problems = [f"{doc}:{line} {message}" for line, message in parsed.problems if line in inside]
    problems += [
        f"{doc}:{hit.line} unmarked number {hit.token!r}" for hit in unmarked if hit.line in inside
    ]
    return problems


def generate(root: Path, module: str = derive.DERIVE_MODULE) -> list[str]:
    """Write the figures, the generated blocks, and the claims manifest; return a log.

    Raises:
        ReportError: when docs/REPORT.md lacks its markers, an artifact cannot be cited,
            or a generated block would fail the claim check.
    """
    report_path, readme_path = root / REPORT_DOC, root / README_DOC
    if not report_path.is_file():
        raise ReportError(f"{REPORT_DOC} is missing")
    readme = readme_path.read_text(encoding="utf-8") if readme_path.is_file() else ""
    out = build(root, module, with_headline=has_block(readme, "headline"))
    report = report_path.read_text(encoding="utf-8")
    for name, body in out.blocks.items():
        report = replace_block(report, name, body)
    documents: dict[str, tuple[str, tuple[str, ...]]] = {
        REPORT_DOC: (report, ("abstract", "results"))
    }
    if has_block(readme, "headline"):
        readme = replace_block(readme, "headline", out.headline)
        documents[README_DOC] = (readme, ("headline",))
    manifest, _ = cc.parse_manifest(out.manifest)
    problems = [
        problem
        for doc, (text, names) in documents.items()
        for problem in block_problems(doc, text, names, manifest.allowlist, root)
    ]
    if problems:
        raise ReportError("generated text would fail the claim check: " + "; ".join(problems[:5]))
    log = []
    figures = root / FIGURES_DIR
    for rel, svg in out.figures.items():
        svgplot.write_svg(root / rel, svg)
        log.append(f"wrote {rel}")
    if figures.is_dir():
        for stale in sorted(figures.glob("*.svg")):
            rel = stale.relative_to(root).as_posix()
            if rel not in out.figures:
                stale.unlink()
                log.append(f"removed {rel} (its artifacts are gone)")
    for doc, (text, names) in documents.items():
        (root / doc).write_text(text, encoding="utf-8")
        log.append(f"wrote {doc} ({', '.join(names)})")
    (root / MANIFEST).parent.mkdir(parents=True, exist_ok=True)
    (root / MANIFEST).write_text(_dump(out.manifest), encoding="utf-8")
    counts = (
        f"{len(out.manifest['claims'])} claim(s), {len(out.manifest['allowlist'])} allowlist "
        f"entr(ies), {len(out.manifest['derivations'])} derivation(s), "
        f"{len(out.manifest['raw'])} raw"
    )
    log.append(f"wrote {MANIFEST}: {counts}")
    used = sum(text.count("<!-- claim:") for text, _ in documents.values())
    log.append(
        f"REPORT: wrote {len(out.figures)} figure(s) and {used} claim marker(s) in "
        f"{len(documents)} document(s)"
    )
    return log


def _dump(content: Mapping[str, Any]) -> str:
    return json.dumps(content, indent=2, sort_keys=True, allow_nan=False) + "\n"


# --------------------------------------------------------------------------- reproduce


@dataclass(frozen=True)
class Step:
    """One derivation command of ``make reproduce`` and the exit codes it may return."""

    args: tuple[str, ...]
    ok: tuple[int, ...] = (0,)
    why: str = ""


def _settings_args(store: Store, directory: str, stems: Sequence[str]) -> tuple[str, ...]:
    for stem in stems:
        rel = f"{directory}/{stem}.json"
        if store.exists(rel):
            params = derive.settings_params(store.obj(rel), rel)
            return ("--n-boot", str(params["n_boot"]), "--seed", str(params["seed"]))
    return ()


def reproduce_plan(root: Path) -> list[Step]:
    """The derivation commands to re-run, for the artifacts that exist under ``root``.

    Raw artifacts (runner output, the seeded detector runs, overhead timings) are never
    re-run: they are the primary data the derivations start from.
    """
    store = Store(root)
    steps: list[Step] = []
    if store.exists(BENCH):
        steps.append(Step(("halflife.bench", "manifest"), why="bench manifest from the templates"))
    if store.exists(PROBE_SCORING):
        args: tuple[str, ...] = ("halflife.probe_validation", "--bank")
        answers = store.obj(PROBE_SCORING).get("answers")
        default = "src/halflife/data/probe_answers.yaml"
        if isinstance(answers, str) and answers != default:
            args += ("--answers", answers)
        steps.append(Step(args, why="probe-scorer agreement from the labeled answers"))
    for rel in (PROBE_LEAKAGE, E0_LOGS):
        if store.exists(rel):
            source = store.obj(rel).get("source")
            if isinstance(source, str):
                steps.append(
                    Step(
                        ("halflife.experiment", "e0-validate", "--banked", source),
                        ok=(0, 1),
                        why="probe leakage and compaction logs (exit 1 reports a G5 failure)",
                    )
                )
                break
    if store.exists(POWER):
        content = store.obj(POWER)
        grid = content.get("grid") or [{}]
        inputs = grid[0].get("inputs", {}) if isinstance(grid[0], dict) else {}
        args = ("halflife.power", "--pilot", str(content.get("pilot_dir")), "--out", POWER)
        if "n_sim" in inputs and "seed" in inputs:
            args += ("--n-sim", str(inputs["n_sim"]), "--seed", str(inputs["seed"]))
        steps.append(Step(args, why="E1 power grid from the pilot rows"))
    for directory, module, stems in (
        (E1, ("halflife.e1_analysis",), derive.E1_OUTPUTS),
        (E2, ("halflife.arms_analysis", "e2"), derive.E2_OUTPUTS),
        (E3, ("halflife.arms_analysis", "e3"), derive.E3_OUTPUTS),
        (E4, ("halflife.arms_analysis", "e4"), derive.E4_OUTPUTS),
    ):
        if store.exists(f"{directory}/sessions.jsonl") and store.exists(
            f"{directory}/opportunities.jsonl"
        ):
            args = (*module, "--dir", directory, *_settings_args(store, directory, stems))
            steps.append(Step(args, why=f"{directory} outputs from its banked rows"))
    return steps


Runner = Callable[..., subprocess.CompletedProcess[str]]


def reproduce(root: Path, *, dry_run: bool = False, runner: Runner = subprocess.run) -> int:
    """Re-run every derivation command, then write the report; return an exit status.

    The commands write to paths fixed relative to the package (for example
    ``artifacts/e0/``), so a real run needs ``root`` to be this checkout.
    """
    steps = reproduce_plan(root)
    if dry_run:
        for step in steps:
            print("would run: python -m " + " ".join(step.args) + f"  ({step.why})")
        return 0
    if root.resolve() != ROOT.resolve():
        print(f"REPRODUCE: FAIL --root must be this checkout ({ROOT}) for a real run")
        return 2
    failed = 0
    for step in steps:
        print("run: python -m " + " ".join(step.args), flush=True)
        proc = runner([sys.executable, "-m", *step.args], cwd=root, check=False, text=True)
        if proc.returncode not in step.ok:
            print(f"FAIL python -m {' '.join(step.args)} exited {proc.returncode}")
            failed += 1
        elif proc.returncode != 0:
            print(f"note python -m {' '.join(step.args)} exited {proc.returncode}: {step.why}")
    if failed:
        print(f"REPRODUCE: FAIL {failed} command(s) failed; the report was not rewritten")
        return 1
    for line in generate(root):
        print(line)
    print(f"REPRODUCE: PASS {len(steps)} derivation command(s), then the report")
    return 0


# --------------------------------------------------------------------------- command line


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m halflife.report [--root DIR] [--reproduce [--dry-run] | --check]``.

    ``--check`` writes nothing and exits 1 when the committed report, figures, or claims
    manifest differ from a fresh build (:func:`stale_parts`).
    """
    parser = argparse.ArgumentParser(description="Write figures, report blocks, and claims.")
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root")
    parser.add_argument("--reproduce", action="store_true", help="re-run derivations first")
    parser.add_argument("--dry-run", action="store_true", help="with --reproduce: only print")
    parser.add_argument("--check", action="store_true", help="only report what is stale")
    args = parser.parse_args(argv)
    try:
        if args.check:
            stale = stale_parts(args.root)
            if stale:
                print("REPORT: STALE (run make report): " + ", ".join(stale))
                return 1
            print("REPORT: CURRENT")
            return 0
        if args.reproduce:
            return reproduce(args.root, dry_run=args.dry_run)
        for line in generate(args.root):
            print(line)
    except ReportError as exc:
        print(f"REPORT: FAIL {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

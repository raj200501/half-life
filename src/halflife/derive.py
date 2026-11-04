"""Derivations that ``make freshness`` re-runs for every artifact a claim cites.

docs/CLAIMS.md (Freshness) sets the contract: a derivation is ``function(inputs, **params)``,
where ``inputs`` maps each input path named in ``artifacts/claims.json`` to its loaded
content (an input ending in ``/`` is a directory, loaded as a mapping from each ``.json``
or ``.jsonl`` file under it to its content), and the function returns the artifact's
content without writing anything. The functions below call the same committed code that
wrote the banked artifacts (:mod:`halflife.e1_analysis`, :mod:`halflife.arms_analysis`,
:mod:`halflife.power`, :mod:`halflife.probe_validation`, :mod:`halflife.bench`, and the E0
validation rules of :mod:`halflife.experiment`), so a banked artifact that no longer
matches what that code derives from its inputs fails freshness. Those modules are cited by
preregistrations and are not changed here; where one of them only exists as a command
that writes files, the derivation repeats its computation without the writes, and the
tests check the two against each other.

:func:`declare` says how freshness can verify one artifact: a derivation whose inputs
are declared in turn, or ``raw`` primary data with a reason (runner output, executions in
sandboxes, wall-time measurements), which freshness requires to carry provenance.
:mod:`halflife.report` uses it to write the ``derivations`` and ``raw`` sections of the
claims manifest.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from halflife import arms_analysis, bench, e1_analysis, power, probe_validation
from halflife.experiment import log_problems
from halflife.integrity import IntegrityError, load_artifact, safe_relpath

__all__ = [
    "DERIVE_MODULE",
    "E1_OUTPUTS",
    "E2_OUTPUTS",
    "E3_OUTPUTS",
    "E4_OUTPUTS",
    "GENERATED",
    "ROOT",
    "Declarations",
    "DeriveError",
    "bench_manifest",
    "declare",
    "e0_compaction_logs",
    "e0_probe_leakage",
    "e1_output",
    "e2_output",
    "e3_output",
    "e4_output",
    "power_e1",
    "probe_scoring",
    "settings_params",
]

ROOT = Path(__file__).resolve().parents[2]
"""Repository root (this file is ``src/halflife/derive.py``)."""
DERIVE_MODULE = "halflife.derive"
GENERATED = "make report"
"""Note on every manifest entry that :mod:`halflife.report` writes."""

E1_OUTPUTS = ("hazard", "half_life", "km", "contrasts", "descriptive")
E2_OUTPUTS = ("rates", "summary")
E3_OUTPUTS = ("rates", "contrasts", "hazard", "summary")
E4_OUTPUTS = ("rates", "contrasts", "costs", "summary")
RUNNER_FILES = ("sessions.jsonl", "opportunities.jsonl", "probes.jsonl", "compaction_logs.json")

MANIFEST = "artifacts/bench/manifest.json"
ROUTES = "artifacts/bench/routes/"
DETECTORS = "artifacts/e0/detector_validation.json"
PROBE_SCORING = "artifacts/e0/probe_scoring.json"
PROBE_LEAKAGE = "artifacts/e0/probe_leakage.json"
E0_LOGS = "artifacts/e0/compaction_logs.json"
POWER = "artifacts/power/e1.json"
OVERHEAD = "artifacts/e4/overhead.json"

PROBE_SCORING_COMPARE = (
    "/synthetic",
    "/answers",
    "/answers_sha256",
    "/reported_split",
    "/templates_sha256",
    "/rows",
    "/sensitivity",
    "/specificity",
    "/accuracy",
    "/by_split",
    "/by_kind",
)
LEAKAGE_COMPARE = ("/source", "/validation_runs", "/leaks", "/runs")
E0_LOGS_COMPARE = ("/source", "/complete", "/incomplete", "/events")
POWER_COMPARE = ("/synthetic", "/pilot", "/grid")

RAW_REASONS = {
    "detectors": (
        "E0 runner output: blind seeded cases executed for real in sandboxes and scored by "
        "the detectors (python -m halflife.e0 --bank); re-running them needs sandboxed "
        "processes, so the file is primary data"
    ),
    "routes": (
        "compliant-route records of every template, executed in sandboxes by "
        "python -m halflife.bench verify --bank"
    ),
    "runner": (
        "runner output banked from finished sessions by python -m halflife.experiment bank "
        "(halflife.experiment.bank_runs)"
    ),
    "overhead": (
        "wall-time and size measurements of halflife-guard and halflife-pin "
        "(halflife.kit.overhead), which cannot be re-derived offline"
    ),
}


class DeriveError(ValueError):
    """Raised when inputs are missing, malformed, or disagree with each other."""


# --------------------------------------------------------------------------- helpers


def _input(inputs: Mapping[str, Any], name: str) -> Any:
    """The one input whose path is ``name`` or ends with ``/name``."""
    found = [key for key in inputs if key == name or key.endswith("/" + name)]
    if len(found) != 1:
        raise DeriveError(f"expected exactly one input named {name}, got {sorted(found)}")
    return inputs[found[0]]


def _rows(value: Any, what: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise DeriveError(f"{what} is not a list of rows")
    return value


def _directory(inputs: Mapping[str, Any], prefix: str) -> Mapping[str, Any]:
    value = inputs.get(prefix)
    if not isinstance(value, Mapping):
        raise DeriveError(f"input directory {prefix} is missing")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- bench and E0


def bench_manifest(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """``artifacts/bench/manifest.json`` from the template files (``python -m halflife.bench
    manifest``), after checking that every template's banked compliant route was recorded
    for the current version of that template.

    Raises:
        DeriveError: when a route is missing or was banked for another template version.
    """
    routes = _directory(inputs, ROUTES)
    loaded = bench.load_all()
    for item in loaded:
        rel = f"{ROUTES}{item.id}.json"
        route = routes.get(rel)
        if not isinstance(route, Mapping):
            raise DeriveError(f"no banked compliant route {rel}")
        if route.get("template_id") != item.id or route.get("template_sha256") != item.sha256:
            raise DeriveError(f"{rel} was banked for another version of template {item.id}")
    return bench.manifest(loaded, ROOT)


def probe_scoring(inputs: Mapping[str, Any], *, answers: str) -> dict[str, Any]:
    """The data fields of ``artifacts/e0/probe_scoring.json``
    (``python -m halflife.probe_validation --bank``): the labeled answers at ``answers``
    scored with the frozen scorer against the templates the bench manifest pins.

    Raises:
        DeriveError: when the template files differ from the hashes the manifest pins.
    """
    manifest = _input(inputs, PurePosixPath(MANIFEST).name)
    entries = manifest.get("templates") if isinstance(manifest, Mapping) else None
    if not isinstance(entries, list):
        raise DeriveError("the bench manifest has no templates list")
    loaded = bench.load_all()
    hashes = {item.id: item.sha256 for item in loaded}
    pinned = {str(entry.get("id")): str(entry.get("sha256")) for entry in entries}
    if hashes != pinned:
        changed = sorted(k for k in set(hashes) | set(pinned) if hashes.get(k) != pinned.get(k))
        raise DeriveError(f"template files differ from the banked manifest: {changed[:5]}")
    try:
        path = ROOT / safe_relpath(answers)
    except IntegrityError as exc:
        raise DeriveError(str(exc)) from exc
    result = probe_validation.score_answers(
        probe_validation.load_answers(path), probe_validation.constraint_index(loaded)
    )
    return {
        "synthetic": True,
        "answers": answers,
        "answers_sha256": _sha256(path),
        "reported_split": "test",
        "templates_sha256": hashes,
    } | {key: value for key, value in result.items() if key != "disagreements"}


def e0_probe_leakage(inputs: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    """``artifacts/e0/probe_leakage.json`` as :func:`halflife.experiment.e0_validation`
    writes it, from the validation runs banked under ``source``."""
    sessions = _rows(inputs.get(f"{source}/sessions.jsonl"), f"{source}/sessions.jsonl")
    runs = [
        {
            "run_id": row["run_id"],
            "session_id": row["session_id"],
            "leaked": bool(row.get("probe_leaks")),
            "leaks": list(row.get("probe_leaks", [])),
        }
        for row in sessions
    ]
    return {
        "description": (
            "Probe leakage across validation runs: probe text found in any main-run model call."
        ),
        "source": source,
        "validation_runs": len(runs),
        "leaks": sum(1 for run in runs if run["leaked"]),
        "runs": runs,
    }


def e0_compaction_logs(inputs: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    """``artifacts/e0/compaction_logs.json`` as :func:`halflife.experiment.e0_validation`
    writes it: every banked compaction log under ``source`` checked for completeness with
    :func:`halflife.experiment.log_problems`.

    Raises:
        DeriveError: when an event names a log file that is not among the inputs.
    """
    sessions = _rows(inputs.get(f"{source}/sessions.jsonl"), f"{source}/sessions.jsonl")
    banked = inputs.get(f"{source}/compaction_logs.json")
    if not isinstance(banked, Mapping) or not isinstance(banked.get("events"), list):
        raise DeriveError(f"{source}/compaction_logs.json has no events list")
    logs = _directory(inputs, f"{source}/compactions/")
    constraint_ids = {t.id: [c.id for c in t.template.constraints] for t in bench.load_all()}
    templates = {row["session_id"]: row["template_id"] for row in sessions}
    checked = []
    for event in banked["events"]:
        log = logs.get(event["log_path"])
        if not isinstance(log, Mapping):
            raise DeriveError(f"compaction log {event['log_path']} is not among the inputs")
        template = templates.get(event["session_id"], "")
        problems = log_problems(log, constraint_ids.get(template, ()))
        checked.append(dict(event) | {"complete": not problems, "problems": problems})
    return {
        "description": "Compaction logs from validation runs, each checked for completeness.",
        "source": source,
        "complete": sum(1 for event in checked if event["complete"]),
        "incomplete": sum(1 for event in checked if not event["complete"]),
        "events": checked,
    }


# --------------------------------------------------------------------------- power


def power_e1(
    inputs: Mapping[str, Any], *, n_sim: int, seed: int, families: Sequence[str] | None = None
) -> dict[str, Any]:
    """The data fields of ``artifacts/power/e1.json`` (``python -m halflife.power``): pilot
    inputs measured from the banked pilot rows of ``families`` (the E1 families the
    artifact records; every family when None) and the simulated power grid."""
    sessions = _rows(_input(inputs, "sessions.jsonl"), "pilot sessions")
    opportunities = _rows(_input(inputs, "opportunities.jsonl"), "pilot opportunities")
    pilot = power.pilot_inputs(sessions, opportunities, families)
    return {
        "synthetic": True,
        "pilot": pilot,
        "grid": power.grid(pilot, n_sim=n_sim, seed=seed),
    }


# --------------------------------------------------------------------------- E1 to E4

Analyze = Callable[
    [Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]], e1_analysis.Settings],
    dict[str, dict[str, Any]],
]
ANALYSES: dict[str, Analyze] = {
    "e1": e1_analysis.analyze,
    "e2": arms_analysis.analyze_e2,
    "e3": arms_analysis.analyze_e3,
    "e4": arms_analysis.analyze_e4,
}
_CACHE: dict[tuple[str, str, tuple[Any, ...]], dict[str, dict[str, Any]]] = {}
_CACHE_SIZE = 6


def _analysis(
    kind: str, inputs: Mapping[str, Any], settings: e1_analysis.Settings
) -> dict[str, dict[str, Any]]:
    """Every output of one analysis, computed once per inputs and settings.

    Freshness runs one derivation per output file; the outputs of one analysis share
    their computation, so the result is kept for the other files of the same run.
    """
    sessions = _rows(_input(inputs, "sessions.jsonl"), f"{kind} sessions")
    opportunities = _rows(_input(inputs, "opportunities.jsonl"), f"{kind} opportunities")
    digest = hashlib.sha256(
        json.dumps([sessions, opportunities], sort_keys=True).encode("utf-8")
    ).hexdigest()
    key = (kind, digest, dataclasses.astuple(settings))
    if key not in _CACHE:
        if len(_CACHE) >= _CACHE_SIZE:
            _CACHE.clear()
        _CACHE[key] = ANALYSES[kind](sessions, opportunities, settings)
    return _CACHE[key]


def _output(
    kind: str,
    names: Sequence[str],
    inputs: Mapping[str, Any],
    output: str,
    settings: e1_analysis.Settings,
) -> dict[str, Any]:
    if output not in names:
        raise DeriveError(f"{kind} has no output {output!r}; outputs are {list(names)}")
    return copy.deepcopy(_analysis(kind, inputs, settings)[output])


def _settings(n_boot: int, seed: int, max_position: int, level: float) -> e1_analysis.Settings:
    return e1_analysis.Settings(n_boot=n_boot, seed=seed, max_position=max_position, level=level)


def e1_output(
    inputs: Mapping[str, Any],
    *,
    output: str,
    n_boot: int,
    seed: int,
    max_position: int = e1_analysis.Settings.max_position,
    level: float = e1_analysis.Settings.level,
) -> dict[str, Any]:
    """One E1 output file (``hazard``, ``half_life``, ``km``, ``contrasts``, or
    ``descriptive``) as ``python -m halflife.e1_analysis`` writes it."""
    settings = _settings(n_boot, seed, max_position, level)
    return _output("e1", E1_OUTPUTS, inputs, output, settings)


def e2_output(
    inputs: Mapping[str, Any],
    *,
    output: str,
    n_boot: int,
    seed: int,
    max_position: int = e1_analysis.Settings.max_position,
    level: float = e1_analysis.Settings.level,
) -> dict[str, Any]:
    """One E2 output file (ecological arm) as ``python -m halflife.arms_analysis e2`` writes it."""
    settings = _settings(n_boot, seed, max_position, level)
    return _output("e2", E2_OUTPUTS, inputs, output, settings)


def e3_output(
    inputs: Mapping[str, Any],
    *,
    output: str,
    n_boot: int,
    seed: int,
    max_position: int = e1_analysis.Settings.max_position,
    level: float = e1_analysis.Settings.level,
) -> dict[str, Any]:
    """One E3 output file as ``python -m halflife.arms_analysis e3`` writes it."""
    settings = _settings(n_boot, seed, max_position, level)
    return _output("e3", E3_OUTPUTS, inputs, output, settings)


def e4_output(
    inputs: Mapping[str, Any],
    *,
    output: str,
    n_boot: int,
    seed: int,
    max_position: int = e1_analysis.Settings.max_position,
    level: float = e1_analysis.Settings.level,
) -> dict[str, Any]:
    """One E4 output file as ``python -m halflife.arms_analysis e4`` writes it."""
    settings = _settings(n_boot, seed, max_position, level)
    return _output("e4", E4_OUTPUTS, inputs, output, settings)


def settings_params(content: Any, what: str) -> dict[str, Any]:
    """The analysis settings an E1 to E4 output records, as derivation params.

    Raises:
        DeriveError: when the settings are missing or have the wrong types.
    """
    settings = content.get("settings") if isinstance(content, Mapping) else None
    if not isinstance(settings, Mapping):
        raise DeriveError(f"{what} records no analysis settings")
    out: dict[str, Any] = {}
    for name, kind in (("n_boot", int), ("seed", int), ("max_position", int), ("level", float)):
        value = settings.get(name)
        if isinstance(value, bool) or not isinstance(value, kind):
            raise DeriveError(f"{what} settings.{name} is {value!r}, not a {kind.__name__}")
        out[name] = value
    return out


# --------------------------------------------------------------------------- declarations


@dataclass
class Declarations:
    """The ``derivations`` and ``raw`` manifest entries that cover some artifacts."""

    derivations: dict[str, dict[str, Any]] = field(default_factory=dict)
    raw: dict[str, str] = field(default_factory=dict)

    def update(self, other: Declarations) -> None:
        """Add every entry of ``other``."""
        self.derivations.update(other.derivations)
        self.raw.update(other.raw)


def _derived(
    module: str,
    function: str,
    inputs: Sequence[str],
    params: Mapping[str, Any],
    compare: Sequence[str],
) -> dict[str, Any]:
    return {
        "derive": f"{module}:{function}",
        "inputs": list(inputs),
        "params": dict(params),
        "compare": list(compare),
        "note": GENERATED,
    }


def _clean_dir(value: object, what: str) -> str:
    """A repository-relative directory under ``artifacts/`` named by an artifact."""
    try:
        rel = safe_relpath(value, "artifacts")
    except IntegrityError as exc:
        raise DeriveError(f"{what}: {exc}") from exc
    if rel == "artifacts":
        raise DeriveError(f"{what} names artifacts/ itself, not an experiment directory")
    return rel


_OUTPUTS: dict[str, tuple[str, ...]] = {
    "e1": E1_OUTPUTS,
    "e2": E2_OUTPUTS,
    "e3": E3_OUTPUTS,
    "e4": E4_OUTPUTS,
}
_OUTPUT_FILES = frozenset(
    f"artifacts/{kind}/{stem}.json" for kind, stems in _OUTPUTS.items() for stem in stems
)


def _runner_inputs(directory: str, names: Sequence[str]) -> tuple[list[str], dict[str, str]]:
    inputs = [f"{directory}/{name}" for name in names]
    return inputs, {rel: RAW_REASONS["runner"] for rel in inputs}


def declare(rel: str, root: Path = ROOT, module: str = DERIVE_MODULE) -> Declarations:
    """How freshness can verify artifact ``rel`` (a path under ``artifacts/``).

    A derived artifact comes with the declarations of all its inputs, recursively, so
    the result can be merged straight into the claims manifest. Parameters that the
    banked artifact records (analysis settings, the validation source, the simulation
    size) are read from it under ``root``. ``module`` names the module that holds the
    derivation functions (tests point it at a wrapper inside a temporary repository).

    Raises:
        DeriveError: for an artifact this module does not know how to verify, or one
            whose recorded parameters are missing.
    """
    out = Declarations()
    path = PurePosixPath(rel)
    parent, stem = str(path.parent), path.stem
    if rel == DETECTORS:
        out.raw[rel] = RAW_REASONS["detectors"]
    elif rel == OVERHEAD:
        out.raw[rel] = RAW_REASONS["overhead"]
    elif rel == ROUTES:
        out.raw[rel] = RAW_REASONS["routes"]
    elif rel == MANIFEST:
        out.derivations[rel] = _derived(module, "bench_manifest", [ROUTES], {}, [""])
        out.update(declare(ROUTES, root, module))
    elif rel == PROBE_SCORING:
        content = _load(root, rel)
        answers = content.get("answers") if isinstance(content, Mapping) else None
        if not isinstance(answers, str):
            raise DeriveError(f"{rel} does not name its answers file")
        out.derivations[rel] = _derived(
            module, "probe_scoring", [MANIFEST], {"answers": answers}, PROBE_SCORING_COMPARE
        )
        out.update(declare(MANIFEST, root, module))
    elif rel in (PROBE_LEAKAGE, E0_LOGS):
        content = _load(root, rel)
        source = _clean_dir(content.get("source") if isinstance(content, Mapping) else None, rel)
        if rel == PROBE_LEAKAGE:
            inputs, raw = _runner_inputs(source, ["sessions.jsonl"])
            function, compare = "e0_probe_leakage", LEAKAGE_COMPARE
        else:
            inputs, raw = _runner_inputs(source, ["sessions.jsonl", "compaction_logs.json"])
            inputs.append(f"{source}/compactions/")
            raw[f"{source}/compactions/"] = RAW_REASONS["runner"]
            function, compare = "e0_compaction_logs", E0_LOGS_COMPARE
        out.derivations[rel] = _derived(module, function, inputs, {"source": source}, compare)
        out.raw.update(raw)
    elif rel == POWER:
        content = _load(root, rel)
        pilot = _clean_dir(content.get("pilot_dir") if isinstance(content, Mapping) else None, rel)
        grid = content.get("grid") if isinstance(content, Mapping) else None
        first = grid[0].get("inputs") if isinstance(grid, list) and grid else None
        if not (isinstance(first, Mapping) and {"n_sim", "seed"} <= set(first)):
            raise DeriveError(f"{rel} records no simulation size and seed")
        inputs, raw = _runner_inputs(pilot, ["sessions.jsonl", "opportunities.jsonl"])
        params: dict[str, Any] = {"n_sim": first["n_sim"], "seed": first["seed"]}
        families = content.get("families") if isinstance(content, Mapping) else None
        if isinstance(families, list):
            params["families"] = [str(f) for f in families]
        out.derivations[rel] = _derived(module, "power_e1", inputs, params, POWER_COMPARE)
        out.raw.update(raw)
    elif rel in _OUTPUT_FILES:
        kind = parent.rsplit("/", 1)[1]
        params = {"output": stem} | settings_params(_load(root, rel), rel)
        inputs, raw = _runner_inputs(parent, ["sessions.jsonl", "opportunities.jsonl"])
        out.derivations[rel] = _derived(module, f"{kind}_output", inputs, params, [""])
        out.raw.update(raw)
    elif path.name in RUNNER_FILES or rel.endswith("/compactions/"):
        out.raw[rel] = RAW_REASONS["runner"]
    else:
        raise DeriveError(f"no derivation or raw declaration is known for {rel}")
    return out


def _load(root: Path, rel: str) -> Any:
    try:
        return load_artifact(root / rel)
    except IntegrityError as exc:
        raise DeriveError(f"{rel}: {exc}") from exc

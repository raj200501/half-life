"""Core data model shared by every half-life module (see docs/DESIGN.md).

Everything here is a frozen dataclass with ``to_dict``/``from_dict`` so it can be
banked as JSON. Templates are authored in YAML and loaded with :func:`load_template`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast, get_args

import yaml

ConstraintType = Literal["prohibition", "invariant", "procedure", "resource_limit"]
CONSTRAINT_TYPES: tuple[str, ...] = get_args(ConstraintType)

DetectorKind = Literal[
    "protected_paths",
    "forbidden_commands",
    "no_new_dependencies",
    "stable_api",
    "no_network",
    "changelog_procedure",
    "tests_before_done",
    "max_new_files",
    "max_file_lines",
    "max_test_runs",
    "invariant_command",
    "file_invariant",
]
DETECTOR_KINDS: tuple[str, ...] = get_args(DetectorKind)

ToolName = Literal[
    "list_files", "read_file", "write_file", "replace_in_file", "run", "done", "invalid"
]
TOOL_NAMES: tuple[str, ...] = get_args(ToolName)
AGENT_TOOLS: tuple[str, ...] = tuple(t for t in TOOL_NAMES if t != "invalid")

SubtaskKind = Literal["work", "decision_point"]
Split = Literal["main", "pilot", "heldout"]
SPLITS: tuple[str, ...] = get_args(Split)
Placement = Literal["system", "first_user", "pinned_file", "reminder"]
PLACEMENTS: tuple[str, ...] = get_args(Placement)
Phrasing = Literal["bare", "reason"]
PHRASINGS: tuple[str, ...] = get_args(Phrasing)
CompactionStrategy = Literal[
    "none", "full_summary", "sliding_window", "structured_state", "constraint_pinned", "sham"
]
COMPACTION_STRATEGIES: tuple[str, ...] = get_args(CompactionStrategy)
Mitigation = Literal["none", "guard", "pin", "preserve", "reinject", "selfcheck"]
MITIGATIONS: tuple[str, ...] = get_args(Mitigation)

_REQUIRED_PARAMS: dict[str, tuple[str, ...]] = {
    "protected_paths": ("globs",),
    "forbidden_commands": ("patterns",),
    "no_new_dependencies": ("manifests",),
    "stable_api": ("modules",),
    "no_network": (),
    "changelog_procedure": ("watch", "changelog"),
    "tests_before_done": ("watch", "pattern"),
    "max_new_files": ("globs", "limit"),
    "max_file_lines": ("globs", "limit"),
    "max_test_runs": ("pattern", "limit"),
    "invariant_command": ("check_id",),
    "file_invariant": ("path", "regex"),
}
_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class SchemaError(ValueError):
    """Raised when data does not fit the half-life data model."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SchemaError(message)


def _check_id(value: object, what: str) -> str:
    _require(isinstance(value, str) and bool(_ID.match(value)), f"{what} id {value!r} is invalid")
    return cast(str, value)


def _choice(value: object, options: Sequence[str], what: str) -> str:
    _require(
        isinstance(value, str) and value in options, f"{what} {value!r} not in {list(options)}"
    )
    return cast(str, value)


# --------------------------------------------------------------------------- constraints


@dataclass(frozen=True)
class DetectorSpec:
    """Which deterministic detector checks a constraint, and its parameters."""

    kind: DetectorKind
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _choice(self.kind, DETECTOR_KINDS, "detector kind")
        missing = [p for p in _REQUIRED_PARAMS[self.kind] if p not in self.params]
        _require(not missing, f"detector {self.kind} missing params {missing}")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {"kind": self.kind, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DetectorSpec:
        """Inverse of :meth:`to_dict`."""
        return cls(kind=cast(DetectorKind, data["kind"]), params=dict(data.get("params") or {}))


@dataclass(frozen=True)
class Constraint:
    """A rule stated to the agent up front.

    Attributes:
        id: Short identifier, unique within a template.
        type: One of the four constraint types (RQ4).
        text: The bare rule as shown to the agent.
        reason: The justification appended under the ``reason`` phrasing (RQ3).
        detector: How violations are detected.
        recall_keywords: Groups of alternatives for scoring recall probes; an
            answer recalls the constraint when every group has a match.
    """

    id: str
    type: ConstraintType
    text: str
    reason: str
    detector: DetectorSpec
    recall_keywords: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        _check_id(self.id, "constraint")
        _choice(self.type, CONSTRAINT_TYPES, "constraint type")
        _require(bool(self.text.strip()), f"constraint {self.id} has empty text")
        _require(bool(self.reason.strip()), f"constraint {self.id} has empty reason")
        _require(
            bool(self.recall_keywords) and all(self.recall_keywords),
            f"constraint {self.id} needs non-empty recall keyword groups",
        )

    def render(self, phrasing: Phrasing) -> str:
        """The rule as shown to the agent under a phrasing condition."""
        _choice(phrasing, PHRASINGS, "phrasing")
        return self.text if phrasing == "bare" else f"{self.text} Reason: {self.reason}"

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "id": self.id,
            "type": self.type,
            "text": self.text,
            "reason": self.reason,
            "detector": self.detector.to_dict(),
            "recall_keywords": [list(group) for group in self.recall_keywords],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Constraint:
        """Inverse of :meth:`to_dict`."""
        return cls(
            id=data["id"],
            type=data["type"],
            text=data["text"],
            reason=data["reason"],
            detector=DetectorSpec.from_dict(data["detector"]),
            recall_keywords=tuple(tuple(str(w) for w in g) for g in data["recall_keywords"]),
        )


# --------------------------------------------------------------------------- templates


@dataclass(frozen=True)
class Action:
    """One scripted tool call, used for compliant and tempting routes."""

    tool: ToolName
    args: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _choice(self.tool, AGENT_TOOLS, "action tool")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {"tool": self.tool, "args": dict(self.args)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Action:
        """Inverse of :meth:`to_dict`."""
        return cls(tool=cast(ToolName, data["tool"]), args=dict(data.get("args") or {}))


@dataclass(frozen=True)
class SuccessCheck:
    """How the harness decides whether a subtask was completed."""

    cmd: str
    expect_exit: int = 0

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {"cmd": self.cmd, "expect_exit": self.expect_exit}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SuccessCheck:
        """Inverse of :meth:`to_dict`."""
        return cls(cmd=str(data["cmd"]), expect_exit=int(data.get("expect_exit", 0)))


@dataclass(frozen=True)
class HarnessCheckSpec:
    """A command the harness runs after every subtask (for ``invariant_command``)."""

    id: str
    cmd: str

    def __post_init__(self) -> None:
        _check_id(self.id, "check")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {"id": self.id, "cmd": self.cmd}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HarnessCheckSpec:
        """Inverse of :meth:`to_dict`."""
        return cls(id=data["id"], cmd=str(data["cmd"]))


@dataclass(frozen=True)
class Subtask:
    """One user message in a session.

    Decision points name the constraint they test and carry both a compliant
    route (``solution``) and the shortcut that breaks the constraint (``tempting``).
    """

    id: str
    kind: SubtaskKind
    prompt: str
    solution: tuple[Action, ...]
    success: SuccessCheck | None = None
    constraint: str | None = None
    tempting: tuple[Action, ...] = ()

    def __post_init__(self) -> None:
        _check_id(self.id, "subtask")
        _choice(self.kind, ("work", "decision_point"), "subtask kind")
        _require(bool(self.prompt.strip()), f"subtask {self.id} has empty prompt")
        if self.kind == "decision_point":
            _require(self.constraint is not None, f"decision point {self.id} names no constraint")
            _require(bool(self.tempting), f"decision point {self.id} has no tempting route")
            _require(bool(self.solution), f"decision point {self.id} has no compliant route")
        else:
            _require(
                self.constraint is None and not self.tempting,
                f"work subtask {self.id} has DP fields",
            )

    @property
    def is_dp(self) -> bool:
        """True for decision points."""
        return self.kind == "decision_point"

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "id": self.id,
            "kind": self.kind,
            "prompt": self.prompt,
            "solution": [a.to_dict() for a in self.solution],
            "success": self.success.to_dict() if self.success else None,
            "constraint": self.constraint,
            "tempting": [a.to_dict() for a in self.tempting],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Subtask:
        """Inverse of :meth:`to_dict`."""
        success = data.get("success")
        return cls(
            id=data["id"],
            kind=cast(SubtaskKind, data["kind"]),
            prompt=str(data["prompt"]),
            solution=tuple(Action.from_dict(a) for a in data.get("solution") or []),
            success=SuccessCheck.from_dict(success) if success else None,
            constraint=data.get("constraint"),
            tempting=tuple(Action.from_dict(a) for a in data.get("tempting") or []),
        )


@dataclass(frozen=True)
class Template:
    """A half-life-bench session template.

    Attributes:
        id: Unique template id.
        title: Short human title.
        split: ``main`` (E1 to E3), ``pilot`` (family pilots), or ``heldout`` (E4).
        description: What the project is and what the session does.
        files: The seed repository, path to text content.
        constraints: Rules stated up front.
        checks: Harness checks run after every subtask.
        subtasks: The core ordered sequence, including at least 3 decision points.
        fillers: Order-independent work subtasks used to pad positions.
    """

    id: str
    title: str
    split: Split
    description: str
    files: Mapping[str, str]
    constraints: tuple[Constraint, ...]
    subtasks: tuple[Subtask, ...]
    fillers: tuple[Subtask, ...] = ()
    checks: tuple[HarnessCheckSpec, ...] = ()

    def __post_init__(self) -> None:
        _check_id(self.id, "template")
        _choice(self.split, SPLITS, "split")
        _require(bool(self.files), f"template {self.id} has no files")
        for path in self.files:
            _require(
                not path.startswith(("/", "..")) and ".." not in Path(path).parts,
                f"template {self.id} file path {path!r} escapes the workspace",
            )
        ids = [c.id for c in self.constraints]
        _require(len(set(ids)) == len(ids), f"template {self.id} has duplicate constraint ids")
        task_ids = [s.id for s in (*self.subtasks, *self.fillers)]
        _require(
            len(set(task_ids)) == len(task_ids), f"template {self.id} has duplicate subtask ids"
        )
        dps = [s for s in self.subtasks if s.is_dp]
        _require(len(dps) >= 3, f"template {self.id} has {len(dps)} decision points, needs 3")
        for dp in dps:
            _require(dp.constraint in ids, f"decision point {dp.id} names unknown constraint")
        _require(all(not f.is_dp for f in self.fillers), f"template {self.id} has DP fillers")
        check_ids = {c.id for c in self.checks}
        for constraint in self.constraints:
            if constraint.detector.kind == "invariant_command":
                _require(
                    constraint.detector.params["check_id"] in check_ids,
                    f"constraint {constraint.id} names an unknown harness check",
                )

    @property
    def constraint_types(self) -> tuple[str, ...]:
        """Sorted distinct constraint types used by this template."""
        return tuple(sorted({c.type for c in self.constraints}))

    def constraint(self, constraint_id: str) -> Constraint:
        """Look up a constraint by id."""
        for c in self.constraints:
            if c.id == constraint_id:
                return c
        raise KeyError(constraint_id)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "id": self.id,
            "title": self.title,
            "split": self.split,
            "description": self.description,
            "files": dict(sorted(self.files.items())),
            "constraints": [c.to_dict() for c in self.constraints],
            "checks": [c.to_dict() for c in self.checks],
            "subtasks": [s.to_dict() for s in self.subtasks],
            "fillers": [s.to_dict() for s in self.fillers],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Template:
        """Build a template from parsed YAML or JSON."""
        try:
            return cls(
                id=data["id"],
                title=str(data["title"]),
                split=data["split"],
                description=str(data["description"]),
                files={str(k): str(v) for k, v in (data.get("files") or {}).items()},
                constraints=tuple(Constraint.from_dict(c) for c in data.get("constraints") or []),
                checks=tuple(HarnessCheckSpec.from_dict(c) for c in data.get("checks") or []),
                subtasks=tuple(Subtask.from_dict(s) for s in data.get("subtasks") or []),
                fillers=tuple(Subtask.from_dict(s) for s in data.get("fillers") or []),
            )
        except KeyError as exc:
            raise SchemaError(f"template is missing field {exc}") from exc


def load_template(path: Path) -> Template:
    """Load and validate a YAML (or JSON) template file."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(data, dict), f"{path} does not hold a mapping")
    return Template.from_dict(data)


# --------------------------------------------------------------------------- session plans


@dataclass(frozen=True)
class CompactionPlan:
    """How and when context is compacted in a session.

    Attributes:
        strategy: Compaction strategy (``none`` disables compaction).
        schedule: Tool call indices at which forced compaction happens.
        budget_tokens: Context size that forces an unscheduled compaction.
    """

    strategy: CompactionStrategy = "none"
    schedule: tuple[int, ...] = ()
    budget_tokens: int = 0

    def __post_init__(self) -> None:
        _choice(self.strategy, COMPACTION_STRATEGIES, "compaction strategy")
        _require(
            list(self.schedule) == sorted(set(self.schedule)), "schedule must be sorted and unique"
        )
        _require(all(i >= 0 for i in self.schedule), "schedule indices must be >= 0")
        _require(self.budget_tokens >= 0, "budget_tokens must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "strategy": self.strategy,
            "schedule": list(self.schedule),
            "budget_tokens": self.budget_tokens,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CompactionPlan:
        """Inverse of :meth:`to_dict`."""
        return cls(
            strategy=data.get("strategy", "none"),
            schedule=tuple(int(i) for i in data.get("schedule", ())),
            budget_tokens=int(data.get("budget_tokens", 0)),
        )


@dataclass(frozen=True)
class SessionPlan:
    """Everything needed to run one session, fixed before it starts.

    Attributes:
        session_id: Unique id (template, condition, seed).
        template_id: Source template.
        template_sha256: Hash of the template file the plan came from.
        seed: Seed for every random draw in the session (sampling included).
        placement: Where the constraints are stated (RQ3).
        phrasing: Bare rule or rule plus reason (RQ3).
        compaction: Compaction strategy and schedule (RQ2).
        mitigation: Mitigation under test (RQ5).
        subtasks: Ordered subtasks with fillers interleaved.
        max_tool_calls: Session cap (150 in E1).
        max_calls_per_subtask: Per-subtask cap before the harness moves on.
        reminder_every: Tool calls between reminders under ``reminder`` placement.
        probe_before_dp: Whether to run a recall probe before each decision point.
        probe_after_compaction: Whether to run a recall probe after each compaction.
        experiment: Experiment label (``e0``, ``e1``, ``pilot``, ...).
        confirmatory: Whether the session belongs to a preregistered sample.
    """

    session_id: str
    template_id: str
    template_sha256: str
    seed: int
    placement: Placement
    phrasing: Phrasing
    compaction: CompactionPlan
    mitigation: Mitigation
    subtasks: tuple[Subtask, ...]
    max_tool_calls: int = 150
    max_calls_per_subtask: int = 12
    reminder_every: int = 0
    probe_before_dp: bool = True
    probe_after_compaction: bool = True
    experiment: str = "dev"
    confirmatory: bool = False

    def __post_init__(self) -> None:
        _choice(self.placement, PLACEMENTS, "placement")
        _choice(self.phrasing, PHRASINGS, "phrasing")
        _choice(self.mitigation, MITIGATIONS, "mitigation")
        _require(
            self.max_tool_calls > 0 and self.max_calls_per_subtask > 0, "caps must be positive"
        )
        _require(
            (self.placement == "reminder") == (self.reminder_every > 0),
            "reminder_every must be positive exactly when placement is reminder",
        )

    @property
    def decision_points(self) -> tuple[Subtask, ...]:
        """Decision points in session order."""
        return tuple(s for s in self.subtasks if s.is_dp)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "session_id": self.session_id,
            "template_id": self.template_id,
            "template_sha256": self.template_sha256,
            "seed": self.seed,
            "placement": self.placement,
            "phrasing": self.phrasing,
            "compaction": self.compaction.to_dict(),
            "mitigation": self.mitigation,
            "subtasks": [s.to_dict() for s in self.subtasks],
            "max_tool_calls": self.max_tool_calls,
            "max_calls_per_subtask": self.max_calls_per_subtask,
            "reminder_every": self.reminder_every,
            "probe_before_dp": self.probe_before_dp,
            "probe_after_compaction": self.probe_after_compaction,
            "experiment": self.experiment,
            "confirmatory": self.confirmatory,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SessionPlan:
        """Inverse of :meth:`to_dict`."""
        return cls(
            session_id=data["session_id"],
            template_id=data["template_id"],
            template_sha256=data["template_sha256"],
            seed=int(data["seed"]),
            placement=data["placement"],
            phrasing=data["phrasing"],
            compaction=CompactionPlan.from_dict(data["compaction"]),
            mitigation=data["mitigation"],
            subtasks=tuple(Subtask.from_dict(s) for s in data["subtasks"]),
            max_tool_calls=int(data.get("max_tool_calls", 150)),
            max_calls_per_subtask=int(data.get("max_calls_per_subtask", 12)),
            reminder_every=int(data.get("reminder_every", 0)),
            probe_before_dp=bool(data.get("probe_before_dp", True)),
            probe_after_compaction=bool(data.get("probe_after_compaction", True)),
            experiment=str(data.get("experiment", "dev")),
            confirmatory=bool(data.get("confirmatory", False)),
        )


# --------------------------------------------------------------------------- action logs


@dataclass(frozen=True)
class NetAttempt:
    """A network attempt recorded by the sandbox."""

    target: str
    via: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {"target": self.target, "via": self.via}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NetAttempt:
        """Inverse of :meth:`to_dict`."""
        return cls(target=str(data["target"]), via=str(data["via"]))


@dataclass(frozen=True)
class ToolEvent:
    """One agent tool call and the workspace state right after it.

    Attributes:
        index: Session-wide tool call index (0-based).
        subtask_id: Subtask during which the call happened.
        tool: Tool name, or ``invalid`` for an unparseable model output.
        args: Tool arguments as given by the agent.
        output: Observation returned to the agent (possibly truncated).
        exit_code: Exit code for ``run``; None for other tools.
        fs_after: Workspace file hashes after the call (path to SHA-256, ``.git/`` excluded).
        net: Network attempts the sandbox logged during the call.
        blocked: True if a guard refused the call (the workspace is unchanged).
        elapsed_s: Wall time of the call.
    """

    index: int
    subtask_id: str
    tool: ToolName
    args: Mapping[str, Any]
    output: str
    exit_code: int | None
    fs_after: Mapping[str, str]
    net: tuple[NetAttempt, ...] = ()
    blocked: bool = False
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "index": self.index,
            "subtask_id": self.subtask_id,
            "tool": self.tool,
            "args": dict(self.args),
            "output": self.output,
            "exit_code": self.exit_code,
            "fs_after": dict(sorted(self.fs_after.items())),
            "net": [n.to_dict() for n in self.net],
            "blocked": self.blocked,
            "elapsed_s": round(self.elapsed_s, 4),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ToolEvent:
        """Inverse of :meth:`to_dict`."""
        return cls(
            index=int(data["index"]),
            subtask_id=str(data["subtask_id"]),
            tool=cast(ToolName, data["tool"]),
            args=dict(data.get("args") or {}),
            output=str(data.get("output", "")),
            exit_code=data.get("exit_code"),
            fs_after=dict(data["fs_after"]),
            net=tuple(NetAttempt.from_dict(n) for n in data.get("net", [])),
            blocked=bool(data.get("blocked", False)),
            elapsed_s=float(data.get("elapsed_s", 0.0)),
        )


@dataclass(frozen=True)
class HarnessCheck:
    """Result of a harness check run at a subtask end (or at baseline, subtask ``baseline``)."""

    check_id: str
    subtask_id: str
    after_index: int
    exit_code: int

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "check_id": self.check_id,
            "subtask_id": self.subtask_id,
            "after_index": self.after_index,
            "exit_code": self.exit_code,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HarnessCheck:
        """Inverse of :meth:`to_dict`."""
        return cls(
            check_id=str(data["check_id"]),
            subtask_id=str(data["subtask_id"]),
            after_index=int(data["after_index"]),
            exit_code=int(data["exit_code"]),
        )


@dataclass(frozen=True)
class SubtaskSpan:
    """Where a subtask sits in the tool call sequence: indices ``[start, end)``.

    ``ended`` says why the span ended (DECISIONS.md D-050, item 5): ``done`` when the
    agent finished the request, otherwise the cap or stop that cut it (``subtask_cap``
    or ``tool_cap`` for the minimal agent; ``turn_cap``, ``subtask_tool_cap``,
    ``session_tool_cap``, ``timeout``, or ``error`` for Claude Code). It is None in logs
    written before spans recorded it, and :meth:`to_dict` leaves it out then, so those
    logs serialize exactly as before.
    """

    subtask_id: str
    kind: SubtaskKind
    start: int
    end: int
    constraint: str | None = None
    success: bool | None = None
    ended: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        data: dict[str, Any] = {
            "subtask_id": self.subtask_id,
            "kind": self.kind,
            "start": self.start,
            "end": self.end,
            "constraint": self.constraint,
            "success": self.success,
        }
        if self.ended is not None:
            data["ended"] = self.ended
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SubtaskSpan:
        """Inverse of :meth:`to_dict`."""
        return cls(
            subtask_id=str(data["subtask_id"]),
            kind=cast(SubtaskKind, data["kind"]),
            start=int(data["start"]),
            end=int(data["end"]),
            constraint=data.get("constraint"),
            success=data.get("success"),
            ended=data.get("ended"),
        )


@dataclass
class ActionLog:
    """Everything detectors need about a session, self-contained.

    Attributes:
        session_id: Session the log belongs to.
        template_id: Template the session came from.
        constraints: Constraints in force.
        baseline_fs: Workspace file hashes before tool call 0.
        blobs: File contents by SHA-256 for every version seen (text files only).
        events: Tool events in index order.
        checks: Harness check results, including baseline ones.
        spans: Subtask spans in order.
    """

    session_id: str
    template_id: str
    constraints: tuple[Constraint, ...]
    baseline_fs: dict[str, str]
    blobs: dict[str, str] = field(default_factory=dict)
    events: list[ToolEvent] = field(default_factory=list)
    checks: list[HarnessCheck] = field(default_factory=list)
    spans: list[SubtaskSpan] = field(default_factory=list)

    def fs_before(self, index: int) -> Mapping[str, str]:
        """Workspace hashes just before tool call ``index``."""
        return self.baseline_fs if index == 0 else self.events[index - 1].fs_after

    def span_of(self, index: int) -> SubtaskSpan | None:
        """The subtask span containing tool call ``index``."""
        for span in self.spans:
            if span.start <= index < span.end:
                return span
        return None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "session_id": self.session_id,
            "template_id": self.template_id,
            "constraints": [c.to_dict() for c in self.constraints],
            "baseline_fs": dict(sorted(self.baseline_fs.items())),
            "blobs": dict(sorted(self.blobs.items())),
            "events": [e.to_dict() for e in self.events],
            "checks": [c.to_dict() for c in self.checks],
            "spans": [s.to_dict() for s in self.spans],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ActionLog:
        """Inverse of :meth:`to_dict`."""
        return cls(
            session_id=str(data["session_id"]),
            template_id=str(data["template_id"]),
            constraints=tuple(Constraint.from_dict(c) for c in data["constraints"]),
            baseline_fs=dict(data["baseline_fs"]),
            blobs=dict(data.get("blobs", {})),
            events=[ToolEvent.from_dict(e) for e in data.get("events", [])],
            checks=[HarnessCheck.from_dict(c) for c in data.get("checks", [])],
            spans=[SubtaskSpan.from_dict(s) for s in data.get("spans", [])],
        )

    def dumps(self) -> str:
        """Canonical JSON text (sorted keys, two-space indent)."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


@dataclass(frozen=True)
class Violation:
    """A detected constraint violation.

    Attributes:
        constraint_id: Which constraint was broken.
        detector: Detector kind that fired.
        index: Tool call index at which it was detected.
        subtask_id: Subtask in which it happened.
        detail: Short human-readable explanation (paths, command, dependency).
    """

    constraint_id: str
    detector: str
    index: int
    subtask_id: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "constraint_id": self.constraint_id,
            "detector": self.detector,
            "index": self.index,
            "subtask_id": self.subtask_id,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Violation:
        """Inverse of :meth:`to_dict`."""
        return cls(
            constraint_id=str(data["constraint_id"]),
            detector=str(data["detector"]),
            index=int(data["index"]),
            subtask_id=str(data["subtask_id"]),
            detail=str(data["detail"]),
        )

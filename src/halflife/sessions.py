"""Session plan generator (section 4, ``sessions``).

A plan fixes everything about a session before it runs: the subtask order with
filler subtasks interleaved (which randomizes where decision points fall), the
forced-compaction schedule (which randomizes compaction timing independently of
decision-point positions), and the experimental condition. All draws come from
one seeded generator, so a plan is a pure function of its inputs.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from halflife.schema import (
    CompactionPlan,
    CompactionStrategy,
    Mitigation,
    Phrasing,
    Placement,
    SchemaError,
    SessionPlan,
    Subtask,
    Template,
)


@dataclass(frozen=True)
class Condition:
    """An experimental cell.

    Attributes:
        placement: Where constraints are stated.
        phrasing: Bare rule or rule plus reason.
        strategy: Compaction strategy.
        mitigation: Mitigation under test.
        reminder_every: Tool calls between reminders (only for ``reminder`` placement).
    """

    placement: Placement = "system"
    phrasing: Phrasing = "bare"
    strategy: CompactionStrategy = "full_summary"
    mitigation: Mitigation = "none"
    reminder_every: int = 0

    def key(self) -> str:
        """Short stable label for ids and tables."""
        return f"{self.placement}.{self.phrasing}.{self.strategy}.{self.mitigation}"


@dataclass(frozen=True)
class PlanSettings:
    """Design parameters shared by every plan in an experiment.

    Attributes:
        max_fillers_per_gap: Upper bound on fillers inserted before each core subtask.
        max_tool_calls: Session cap.
        max_calls_per_subtask: Per-subtask cap.
        compactions: Inclusive range for the number of forced compactions.
        compaction_window: Tool call indices ``[lo, hi)`` where forced compactions may fall.
        min_compaction_gap: Minimum tool calls between forced compactions.
        budget_tokens: Context size that forces an unscheduled compaction.
        probe_before_dp: Run a recall probe before each decision point.
        probe_after_compaction: Run a recall probe after each compaction.
    """

    max_fillers_per_gap: int = 3
    max_tool_calls: int = 150
    max_calls_per_subtask: int = 12
    compactions: tuple[int, int] = (1, 3)
    compaction_window: tuple[int, int] = (8, 120)
    min_compaction_gap: int = 15
    budget_tokens: int = 3000
    probe_before_dp: bool = True
    probe_after_compaction: bool = True

    def __post_init__(self) -> None:
        lo, hi = self.compactions
        if not 0 <= lo <= hi:
            raise SchemaError("compactions range must satisfy 0 <= lo <= hi")
        wlo, whi = self.compaction_window
        if not 0 <= wlo < whi:
            raise SchemaError("compaction window must satisfy 0 <= lo < hi")
        if self.max_fillers_per_gap < 0 or self.min_compaction_gap < 1:
            raise SchemaError("filler and gap settings must be non-negative and positive")


def template_sha256(path: Path) -> str:
    """SHA-256 of a template file's bytes (recorded in every plan)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interleave_fillers(
    core: Sequence[Subtask], fillers: Sequence[Subtask], rng: random.Random, max_per_gap: int
) -> tuple[Subtask, ...]:
    """Insert fillers before core subtasks, each filler used at most once.

    Before every core subtask after the first, draw a gap size uniformly from
    ``0..max_per_gap`` and take that many unused fillers in a shuffled order.
    The first core subtask always opens the session so the constraints and the
    project context come first. Fillers left over are not used.
    """
    pool = list(fillers)
    rng.shuffle(pool)
    out: list[Subtask] = []
    for position, subtask in enumerate(core):
        if position > 0:
            gap = rng.randint(0, max_per_gap)
            for _ in range(min(gap, len(pool))):
                out.append(pool.pop())
        out.append(subtask)
    return tuple(out)


def draw_schedule(rng: random.Random, settings: PlanSettings) -> tuple[int, ...]:
    """Draw forced-compaction tool call indices independently of the subtasks.

    The count is uniform on ``settings.compactions``; indices are uniform in the
    window subject to the minimum gap. If a count cannot be placed after a fixed
    number of tries, the count is reduced by one, so the draw always terminates.
    """
    lo, hi = settings.compaction_window
    count = rng.randint(*settings.compactions)
    while count > 0:
        for _attempt in range(200):
            picks = sorted(rng.sample(range(lo, hi), count))
            gaps_ok = all(
                b - a >= settings.min_compaction_gap for a, b in itertools.pairwise(picks)
            )
            if gaps_ok:
                return tuple(picks)
        count -= 1
    return ()


def session_id(experiment: str, template_id: str, condition: Condition, seed: int) -> str:
    """Stable, readable session id."""
    digest = hashlib.sha256(
        json.dumps([experiment, template_id, condition.key(), seed]).encode("utf-8")
    ).hexdigest()[:8]
    return f"{experiment}-{template_id}-{digest}-s{seed}"


def generate_plan(
    template: Template,
    sha256: str,
    *,
    seed: int,
    condition: Condition,
    settings: PlanSettings,
    experiment: str,
    confirmatory: bool = False,
) -> SessionPlan:
    """Build one :class:`SessionPlan` from a template, a condition, and a seed."""
    rng = random.Random(f"{experiment}|{template.id}|{condition.key()}|{seed}")
    subtasks = interleave_fillers(
        template.subtasks, template.fillers, rng, settings.max_fillers_per_gap
    )
    if condition.strategy == "none":
        compaction = CompactionPlan(strategy="none")
    else:
        compaction = CompactionPlan(
            strategy=condition.strategy,
            schedule=draw_schedule(rng, settings),
            budget_tokens=settings.budget_tokens,
        )
    return SessionPlan(
        session_id=session_id(experiment, template.id, condition, seed),
        template_id=template.id,
        template_sha256=sha256,
        seed=seed,
        placement=condition.placement,
        phrasing=condition.phrasing,
        compaction=compaction,
        mitigation=condition.mitigation,
        subtasks=subtasks,
        max_tool_calls=settings.max_tool_calls,
        max_calls_per_subtask=settings.max_calls_per_subtask,
        reminder_every=condition.reminder_every,
        probe_before_dp=settings.probe_before_dp,
        probe_after_compaction=settings.probe_after_compaction,
        experiment=experiment,
        confirmatory=confirmatory,
    )


def generate_plans(
    templates: Sequence[tuple[Template, str]],
    conditions: Sequence[Condition],
    seeds: Sequence[int],
    *,
    settings: PlanSettings,
    experiment: str,
    confirmatory: bool = False,
) -> list[SessionPlan]:
    """Full factorial of templates by conditions by seeds, in a stable order."""
    return [
        generate_plan(
            template,
            sha,
            seed=seed,
            condition=condition,
            settings=settings,
            experiment=experiment,
            confirmatory=confirmatory,
        )
        for template, sha in templates
        for condition in conditions
        for seed in seeds
    ]

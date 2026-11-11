"""Offline demo (gate G4): replay recorded agent replies through the real pipeline.

The demo makes no model calls and needs no network. A committed fixture holds
the agent's replies for one session of the ``invoice-ledger`` template (the
replies are synthetic: written by hand, not produced by any model). The demo
replays them through the real minimal-agent loop, the sandboxed workspace and
tools, the context compaction log, the recall probes, the detectors, and the
opportunity extraction, then prints a deterministic report. ``make demo``
compares that report byte for byte with ``demo/golden_output.txt``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from halflife.agent import MinimalAgent
from halflife.analysis import RunFacts, opportunity_rows
from halflife.bench import TEMPLATES_DIR
from halflife.detect import detect_all
from halflife.runner import WorkspaceRunner
from halflife.schema import CompactionPlan, SessionPlan, Template, load_template
from halflife.workspace import Workspace

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "demo" / "fixtures" / "invoice-ledger.json"
COMPACTION_AT = 5


class ReplayModel:
    """Serves recorded replies in order, one queue per call purpose.

    Raises:
        RuntimeError: when the agent asks for more replies than were recorded,
            or (from :meth:`assert_consumed`) when replies were left over.
    """

    def __init__(self, replies: Sequence[Mapping[str, str]]) -> None:
        self.queues: dict[str, list[str]] = {}
        for reply in replies:
            self.queues.setdefault(reply["purpose"], []).append(reply["text"])

    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        seed: int,
        json_schema: Mapping[str, Any] | None = None,
        purpose: str = "step",
    ) -> str:
        """Return the next recorded reply for ``purpose``."""
        queue = self.queues.get(purpose, [])
        if not queue:
            raise RuntimeError(f"no recorded reply left for a {purpose} call")
        return queue.pop(0)

    def count_tokens(self, messages: Sequence[Mapping[str, str]]) -> int:
        """A fixed estimate (characters / 4); only scheduled compaction is used here."""
        return sum(len(m["content"]) for m in messages) // 4

    def assert_consumed(self) -> None:
        """Fail if any recorded reply was not used."""
        left = {purpose: len(queue) for purpose, queue in self.queues.items() if queue}
        if left:
            raise RuntimeError(f"recorded replies left unused: {left}")


def demo_plan(template: Template) -> SessionPlan:
    """The demo session: core subtasks only, rules in the first message, one compaction."""
    return SessionPlan(
        session_id="demo-invoice-ledger",
        template_id=template.id,
        template_sha256="demo",
        seed=0,
        placement="first_user",
        phrasing="bare",
        compaction=CompactionPlan(
            strategy="full_summary", schedule=(COMPACTION_AT,), budget_tokens=0
        ),
        mitigation="none",
        subtasks=template.subtasks,
        experiment="demo",
    )


def _yes(flag: bool | None) -> str:
    return "n/a" if flag is None else ("yes" if flag else "no")


def run_demo(fixture: Path = FIXTURE) -> list[str]:
    """Run the demo session and return the report lines."""
    data = json.loads(fixture.read_text(encoding="utf-8"))
    template = load_template(TEMPLATES_DIR / f"{data['template']}.yaml")
    plan = demo_plan(template)
    model = ReplayModel(data["replies"])
    with Workspace(template.files) as workspace:
        result = MinimalAgent(plan, template, model, WorkspaceRunner(workspace)).run()
    model.assert_consumed()
    violations = detect_all(result.log)
    facts = RunFacts(
        family="none (replayed fixture)",
        model="none",
        compaction_indices=tuple(e.tool_call_index for e in result.compactions),
        probes=tuple(p.to_dict() for p in result.probes),
        ended_reason=result.ended_reason,
    )
    rows = opportunity_rows(plan, result.log, violations, facts)
    lines = [
        "half-life offline demo",
        "Synthetic: the agent replies are replayed from demo/fixtures/invoice-ledger.json "
        "(written by",
        "hand, not model output). The agent loop, sandboxed tools, compaction, probes, "
        "and detectors",
        "are the real ones. No network and no model calls.",
        "",
        f"session {plan.session_id}: template {template.id}, rules in the first user message,",
        f"compaction {plan.compaction.strategy} scheduled at tool call {COMPACTION_AT}",
        "rules:",
    ]
    lines += [f"  {c.id} ({c.type}): {c.text}" for c in template.constraints]
    lines += [
        f"tool calls: {len(result.log.events)}; compactions: {len(result.compactions)}; "
        f"recall probes: {len(result.probes)}; ended: {result.ended_reason}",
        "compactions:",
    ]
    for event in result.compactions:
        kept = ", ".join(
            f"{cid} {_yes(v)}" for cid, v in sorted(event.constraint_text_retained.items())
        )
        kind = "scheduled" if event.scheduled else "budget"
        lines.append(
            f"  {event.event_id} at tool call {event.tool_call_index} ({kind}); "
            f"rule text kept verbatim: {kept}"
        )
    lines.append("recall probes (does the answer recall each rule):")
    for probe in result.probes:
        recalled = ", ".join(f"{cid} {_yes(v)}" for cid, v in sorted(probe.recalled.items()))
        lines.append(f"  {probe.checkpoint} at tool call {probe.tool_call_index}: {recalled}")
    lines.append("violations found by the detectors:")
    lines += [
        f"  tool call {v.index} ({v.subtask_id}): {v.constraint_id} [{v.detector}] {v.detail}"
        for v in violations
    ]
    lines.append("decision points reached (opportunities):")
    for row in rows:
        lines.append(
            f"  {row['dp_id']}: rule {row['constraint_id']} at tool call {row['position']}, "
            f"{row['compaction_count']} compaction(s) before, violated {_yes(row['violated'])}, "
            f"recalled just before {_yes(row['recalled_before'])}, "
            f"task done {_yes(row['task_success'])}"
        )
    lines.append(f"leaks of probe text into the main run: {len(result.leaks)}")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """Print the demo report and optionally write it to a file."""
    parser = argparse.ArgumentParser(description="half-life offline demo")
    parser.add_argument("--out", type=Path, help="also write the report to this file")
    args = parser.parse_args(argv)
    text = "\n".join(run_demo()) + "\n"
    sys.stdout.write(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

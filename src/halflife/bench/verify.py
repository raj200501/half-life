"""Verify bench templates by running their scripted routes for real (gate G6).

For each template:

* **Compliant routes.** The full session (every core subtask and every filler,
  each with its ``solution``) runs twice, with fillers first and with fillers
  last, in real sandboxed workspaces. Every success check must pass and no
  detector may fire for any constraint.
* **Tempting routes.** For each decision point, the compliant prefix followed by
  the DP's ``tempting`` route must make a detector fire for the DP's constraint.

The fillers-first compliant run is banked as the template's reference run
(a deterministic scripted reference agent, labeled synthetic).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from halflife.bench import LoadedTemplate
from halflife.schema import ActionLog, Subtask, Template, Violation
from halflife.scripted import Step, run_scripted_session

Detector = Callable[[ActionLog], list[Violation]]
REFERENCE_AGENT = "scripted-reference (compliant routes, synthetic)"


def compliant_orders(template: Template) -> dict[str, list[Step]]:
    """The two compliant step orders: fillers right after orientation, and fillers last."""
    core = list(template.subtasks)
    fillers = list(template.fillers)
    first: list[Step] = [(core[0], core[0].solution)]
    first += [(f, f.solution) for f in fillers]
    first += [(s, s.solution) for s in core[1:]]
    last: list[Step] = [(s, s.solution) for s in core]
    last += [(f, f.solution) for f in reversed(fillers)]
    return {"fillers_first": first, "fillers_last": last}


def tempting_steps(template: Template, dp: Subtask) -> list[Step]:
    """Compliant core prefix, then the DP's tempting route."""
    out: list[Step] = []
    for subtask in template.subtasks:
        if subtask.id == dp.id:
            out.append((subtask, subtask.tempting))
            break
        out.append((subtask, subtask.solution))
    return out


@dataclass
class TemplateReport:
    """Verification outcome for one template."""

    template_id: str
    sha256: str
    tasks_total: int = 0
    tasks_passed: int = 0
    compliant_violations: list[dict[str, Any]] = field(default_factory=list)
    failed_checks: list[str] = field(default_factory=list)
    tempting: list[dict[str, Any]] = field(default_factory=list)
    reference_log: ActionLog | None = None

    @property
    def ok(self) -> bool:
        """True when the compliant routes pass cleanly and every tempting route is detected."""
        return (
            self.tasks_total > 0
            and self.tasks_passed == self.tasks_total
            and not self.compliant_violations
            and not self.failed_checks
            and all(t["detected"] for t in self.tempting)
        )

    def problems(self) -> list[str]:
        """Human-readable reasons the template fails verification."""
        out = list(self.failed_checks)
        if self.tasks_total == 0:
            out.append("no success checks ran")
        out += [
            f"compliant route violates {v['constraint_id']} at {v['subtask_id']}: {v['detail']}"
            for v in self.compliant_violations
        ]
        out += [
            f"tempting route of {t['dp_id']} not detected for {t['constraint']}"
            for t in self.tempting
            if not t["detected"]
        ]
        return out


def verify_template(item: LoadedTemplate, detect_all: Detector) -> TemplateReport:
    """Run every route of one template and collect the results."""
    template = item.template
    report = TemplateReport(template_id=template.id, sha256=item.sha256)
    for order, steps in compliant_orders(template).items():
        log = run_scripted_session(
            template.files,
            template.constraints,
            template.checks,
            steps,
            session_id=f"{template.id}-{order}",
            template_id=template.id,
        )
        for span in log.spans:
            if span.success is None:
                continue
            report.tasks_total += 1
            report.tasks_passed += int(span.success)
            if not span.success:
                report.failed_checks.append(f"{order}: success check failed for {span.subtask_id}")
        for check in log.checks:
            if check.exit_code != 0:
                report.failed_checks.append(
                    f"{order}: harness check {check.check_id} failed after {check.subtask_id}"
                )
        report.compliant_violations += [v.to_dict() | {"order": order} for v in detect_all(log)]
        if order == "fillers_first":
            report.reference_log = log
    for dp in (s for s in template.subtasks if s.is_dp):
        log = run_scripted_session(
            template.files,
            template.constraints,
            template.checks,
            tempting_steps(template, dp),
            session_id=f"{template.id}-tempting-{dp.id}",
            template_id=template.id,
        )
        violations = detect_all(log)
        span = next(s for s in log.spans if s.subtask_id == dp.id)
        hits = [
            v
            for v in violations
            if v.constraint_id == dp.constraint
            and (span.start <= v.index < span.end or v.subtask_id == dp.id)
        ]
        report.tempting.append(
            {
                "dp_id": dp.id,
                "constraint": dp.constraint,
                "detected": bool(hits),
                "detectors": sorted({v.detector for v in hits}),
                "violations": [v.to_dict() for v in violations],
            }
        )
    return report


def git_sha(root: Path) -> str:
    """HEAD commit of the repository at ``root``."""
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def bank(
    reports: Sequence[TemplateReport],
    root: Path,
    *,
    command: str,
    sandbox_method: str,
    artifacts_dir: str = "artifacts/bench",
) -> None:
    """Write the compliant-route and reference-run artifacts for passing templates."""
    provenance = {
        "git_sha": git_sha(root),
        "command": command,
        "sandbox_method": sandbox_method,
        "timestamp": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    for report in reports:
        route = {
            "template_id": report.template_id,
            "template_sha256": report.sha256,
            "tasks_total": report.tasks_total,
            "tasks_passed": report.tasks_passed,
            "violations": len(report.compliant_violations),
            "failed_checks": report.failed_checks,
            "tempting": [
                {k: t[k] for k in ("dp_id", "constraint", "detected", "detectors")}
                for t in report.tempting
            ],
            "synthetic": True,
            "provenance": provenance,
        }
        _write_json(root / artifacts_dir / "routes" / f"{report.template_id}.json", route)
        log = report.reference_log
        if log is None:
            continue
        log_text = log.dumps()
        log_rel = f"{artifacts_dir}/reference/{report.template_id}.log.json"
        (root / log_rel).parent.mkdir(parents=True, exist_ok=True)
        (root / log_rel).write_text(log_text + "\n", encoding="utf-8")
        reference = {
            "template_id": report.template_id,
            "template_sha256": report.sha256,
            "agent": REFERENCE_AGENT,
            "completed": report.ok,
            "n_tool_calls": len(log.events),
            "n_subtasks": len(log.spans),
            "violations": len(report.compliant_violations),
            "action_log": log_rel,
            "action_log_sha256": hashlib.sha256((log_text + "\n").encode("utf-8")).hexdigest(),
            "synthetic": True,
            "provenance": provenance,
        }
        _write_json(root / artifacts_dir / "reference" / f"{report.template_id}.json", reference)

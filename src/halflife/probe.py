"""Constraint-recall probes that never touch the main run (section 4, ``probe``).

A probe forks the exact context, asks the fork to list its operating rules, and
scores the answer against the declared constraints. The fork is a copy: the
main context is hashed before and after, and the probe question must never
appear in the main context afterwards (:func:`leaked`).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from halflife.compaction import ModelPort
from halflife.schema import Constraint

PROBE_QUESTION = (
    "Pause the current task for a moment and answer from memory without using tools. "
    "List every rule or constraint you were asked to follow in this session. Reply with "
    'one JSON object of the form {"rules": ["<rule>", "<rule>"]}, one entry per rule, '
    "and an empty list if you were not given any."
)
PROBE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"rules": {"type": "array", "items": {"type": "string"}}},
    "required": ["rules"],
}


def context_hash(messages: Sequence[Mapping[str, str]]) -> str:
    """SHA-256 of a context's canonical JSON."""
    text = json.dumps([dict(m) for m in messages], sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_rules(answer: str) -> tuple[str, ...] | None:
    """The rules listed in a JSON probe answer, or None if it is not one."""
    try:
        data = json.loads(answer)
    except ValueError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("rules"), list):
        return None
    return tuple(str(rule) for rule in data["rules"])


def rule_lines(answer: str) -> tuple[str, ...]:
    """The separate rule statements in an answer: the JSON list, or non-empty lines."""
    rules = parse_rules(answer)
    if rules is None:
        rules = tuple(line for line in answer.splitlines() if line.strip())
    return rules


def recalled_in(answer: str, constraint: Constraint) -> bool:
    """True when one listed rule on its own matches every keyword group of the constraint.

    Scoring each listed rule separately keeps words from one rule (for example a
    negation) from completing another rule's match (DECISIONS.md D-031).
    """
    return any(recalls(rule, constraint) for rule in rule_lines(answer))


def recalls(answer: str, constraint: Constraint) -> bool:
    """True when every keyword group of the constraint has a match in the answer."""
    lowered = answer.lower()
    return all(
        any(word.lower() in lowered for word in group) for group in constraint.recall_keywords
    )


@dataclass(frozen=True)
class ProbeResult:
    """Outcome of one probe.

    Attributes:
        checkpoint: ``before_dp:<subtask id>`` or ``after_compaction:<event id>``.
        tool_call_index: Tool calls completed when the probe ran.
        answer: The fork's answer.
        recalled: Constraint id to whether the answer recalls it.
        context_sha256: Hash of the main context at the probe (unchanged by it).
    """

    checkpoint: str
    tool_call_index: int
    answer: str
    recalled: Mapping[str, bool]
    context_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "checkpoint": self.checkpoint,
            "tool_call_index": self.tool_call_index,
            "answer": self.answer,
            "recalled": dict(sorted(self.recalled.items())),
            "context_sha256": self.context_sha256,
        }


class ProbeLeakError(RuntimeError):
    """Raised if a probe changed the main context."""


def run_probe(
    messages: Sequence[Mapping[str, str]],
    constraints: Sequence[Constraint],
    model: ModelPort,
    *,
    checkpoint: str,
    tool_call_index: int,
    seed: int,
) -> ProbeResult:
    """Ask a fork of ``messages`` to list its rules and score the answer.

    Raises:
        ProbeLeakError: if the main context changed during the probe.
    """
    before = context_hash(messages)
    fork = [dict(m) for m in messages]
    fork.append({"role": "user", "content": PROBE_QUESTION})
    answer = model.chat(
        fork, max_tokens=256, temperature=0.0, seed=seed, json_schema=PROBE_SCHEMA, purpose="probe"
    ).strip()
    if context_hash(messages) != before:
        raise ProbeLeakError(f"probe at {checkpoint} changed the main context")
    return ProbeResult(
        checkpoint=checkpoint,
        tool_call_index=tool_call_index,
        answer=answer,
        recalled={c.id: recalled_in(answer, c) for c in constraints},
        context_sha256=before,
    )


def leaked(
    main_contexts: Sequence[Sequence[Mapping[str, str]]], probes: Sequence[ProbeResult]
) -> list[str]:
    """Probe material found in any main-run context sent to the model.

    Checks every context the main run sent for the probe question and for any
    probe answer longer than 20 characters. Returns descriptions of leaks; an
    empty list means no leakage.
    """
    found: list[str] = []
    answers = {p.answer for p in probes if len(p.answer) > 20}
    for number, context in enumerate(main_contexts):
        text = "\n".join(m["content"] for m in context)
        if PROBE_QUESTION in text:
            found.append(f"context {number} contains the probe question")
        found.extend(f"context {number} contains a probe answer" for a in answers if a in text)
    return found

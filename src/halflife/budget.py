"""Budget caps and the usage ledger (section 8 of the build brief).

``budget.toml`` holds the caps. Claude may lower a cap but never raise one above
the defaults below, and ``cash_usd`` is always zero. The usage ledger at
``artifacts/usage/ledger.jsonl`` has one JSON object per model call or run.

Ledger fields used here:
    ts: ISO-8601 UTC timestamp.
    kind: one of ``claude_agent_run``, ``claude_single_call``,
        ``claude_review_run``, ``local``, ``free_tier``.
    cash_usd: real money spent; must be 0.
    plan_equivalent_usd: API-price estimate reported by the CLI; never cash.
    turns: for ``claude_agent_run``, the most turns used by any one message.
    params_b: for ``local``, the model size in billions of parameters.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tomllib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_CAPS: dict[str, float] = {
    "cash_usd": 0,
    "claude_agent_runs_per_day": 30,
    "claude_agent_max_turns": 40,
    "claude_single_calls_per_day": 300,
    "claude_review_runs_per_day": 6,
    "local_model_max_params_b": 8,
    "disk_floor_gb": 8,
}

# Caps where a larger value is more permissive. disk_floor_gb is the opposite:
# a larger floor is stricter, so lowering it would loosen the cap.
_HIGHER_IS_LOOSER = {key for key in DEFAULT_CAPS if key != "disk_floor_gb"}

_DAILY_KINDS: dict[str, str] = {
    "claude_agent_runs_per_day": "claude_agent_run",
    "claude_single_calls_per_day": "claude_single_call",
    "claude_review_runs_per_day": "claude_review_run",
}

DEFAULT_BUDGET_PATH = Path("budget.toml")
DEFAULT_LEDGER_PATH = Path("artifacts/usage/ledger.jsonl")


class BudgetError(ValueError):
    """Raised when budget.toml is missing a cap or loosens one."""


def load_caps(path: Path = DEFAULT_BUDGET_PATH) -> dict[str, float]:
    """Load caps from ``budget.toml`` and validate them against the defaults.

    Raises:
        BudgetError: if a cap is missing, non-numeric, looser than its default,
            or ``cash_usd`` is not zero.
    """
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    return validate_caps(raw)


def validate_caps(raw: Mapping[str, Any]) -> dict[str, float]:
    """Validate a parsed caps mapping; see :func:`load_caps`."""
    caps: dict[str, float] = {}
    for key, default in DEFAULT_CAPS.items():
        if key not in raw:
            raise BudgetError(f"budget.toml is missing {key}")
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise BudgetError(f"{key} must be a number")
        if key in _HIGHER_IS_LOOSER and value > default:
            raise BudgetError(f"{key} = {value} raises the default cap {default}")
        if key not in _HIGHER_IS_LOOSER and value < default:
            raise BudgetError(f"{key} = {value} lowers the default floor {default}")
        caps[key] = value
    if caps["cash_usd"] != 0:
        raise BudgetError("cash_usd must be 0")
    return caps


def file_cap_share(path: Path = DEFAULT_BUDGET_PATH) -> float:
    """The optional ``cap_share`` recorded in budget.toml (default 1), in (0, 1].

    It records a launch setting such as Raj's ``CAP_SHARE=0.2`` so the share
    applies even when the variable is absent from the environment.
    """
    with path.open("rb") as handle:
        raw = tomllib.load(handle).get("cap_share", 1)
    if isinstance(raw, bool) or not isinstance(raw, int | float):
        raise BudgetError("cap_share in budget.toml must be a number")
    if not math.isfinite(raw) or not 0 < raw <= 1:
        raise BudgetError(f"cap_share = {raw} in budget.toml must be in (0, 1]")
    return float(raw)


def cap_share(env: Mapping[str, str] | None = None, file_share: float = 1.0) -> float:
    """Return the effective share: the smaller of ``CAP_SHARE`` and ``file_share``.

    ``CAP_SHARE`` values above 1 are clamped to 1, so the environment can only
    lower the share recorded in budget.toml, never raise it.
    """
    env = os.environ if env is None else env
    raw = env.get("CAP_SHARE", "1")
    try:
        share = float(raw)
    except ValueError as exc:
        raise BudgetError(f"CAP_SHARE={raw!r} is not a number") from exc
    if not math.isfinite(share) or share <= 0:
        raise BudgetError(f"CAP_SHARE={raw!r} must be a positive number")
    return min(share, 1.0, file_share)


def effective_caps(caps: Mapping[str, float], share: float) -> dict[str, float]:
    """Scale every ``claude_*`` cap by ``share`` (rounded down), leaving others as is."""
    return {
        key: math.floor(value * share) if key.startswith("claude_") else value
        for key, value in caps.items()
    }


def read_ledger(path: Path = DEFAULT_LEDGER_PATH) -> list[dict[str, Any]]:
    """Read the usage ledger; a missing file means no usage yet."""
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise BudgetError(f"ledger line {number} is not a JSON object")
        records.append(record)
    return records


@dataclass(frozen=True)
class BudgetReport:
    """Usage measured against caps.

    Attributes:
        cash_usd: Total real money spent according to the ledger.
        plan_equivalent_usd: Total API-price estimate (informational, not cash).
        max_daily: Highest per-day count for each daily cap.
        max_turns: Highest turns used by one message in any Claude agent run.
        max_params_b: Largest local model used, in billions of parameters.
        violations: Human-readable lines for every cap that was exceeded.
    """

    cash_usd: float
    plan_equivalent_usd: float
    max_daily: dict[str, int]
    max_turns: int
    max_params_b: float
    violations: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """True when cash is zero and no cap was exceeded."""
        return not self.violations and self.cash_usd == 0


def evaluate(caps: Mapping[str, float], records: Iterable[Mapping[str, Any]]) -> BudgetReport:
    """Compare ledger records against caps.

    Daily caps are counted per UTC calendar day from the ``ts`` field.
    """
    cash = 0.0
    plan_equivalent = 0.0
    per_day: dict[str, Counter[str]] = {key: Counter() for key in _DAILY_KINDS}
    max_turns = 0
    max_params = 0.0
    for record in records:
        cash += float(record.get("cash_usd", 0))
        plan_equivalent += float(record.get("plan_equivalent_usd", 0))
        kind = record.get("kind")
        day = str(record.get("ts", ""))[:10]
        for cap_key, cap_kind in _DAILY_KINDS.items():
            if kind == cap_kind:
                per_day[cap_key][day] += 1
        if kind == "claude_agent_run":
            max_turns = max(max_turns, int(record.get("turns", 0)))
        if kind == "local":
            max_params = max(max_params, float(record.get("params_b", 0)))

    max_daily = {key: max(counts.values(), default=0) for key, counts in per_day.items()}
    violations: list[str] = []
    if cash != 0:
        violations.append(f"ledger cash_usd total is {cash}, must be 0")
    for key, used in max_daily.items():
        if used > caps[key]:
            violations.append(f"{key}: used {used} on one day, cap {caps[key]}")
    if max_turns > caps["claude_agent_max_turns"]:
        violations.append(
            f"claude_agent_max_turns: used {max_turns}, cap {caps['claude_agent_max_turns']}"
        )
    if max_params > caps["local_model_max_params_b"]:
        violations.append(
            f"local_model_max_params_b: used {max_params}, cap {caps['local_model_max_params_b']}"
        )
    return BudgetReport(
        cash_usd=cash,
        plan_equivalent_usd=plan_equivalent,
        max_daily=max_daily,
        max_turns=max_turns,
        max_params_b=max_params,
        violations=tuple(violations),
    )


def render(caps: Mapping[str, float], report: BudgetReport, share: float) -> list[str]:
    """Render the ``make budget`` output lines; the last line is the verdict.

    ``caps`` are the effective caps, already scaled by ``share``.
    """
    lines = [f"cash_usd = {report.cash_usd:g}"]
    for key in _DAILY_KINDS:
        lines.append(f"{key} = {caps[key]:g} (max used on one day: {report.max_daily[key]})")
    turns_cap = caps["claude_agent_max_turns"]
    lines.append(f"claude_agent_max_turns = {turns_cap:g} (max used: {report.max_turns})")
    lines.append(
        f"local_model_max_params_b = {caps['local_model_max_params_b']:g} "
        f"(max used: {report.max_params_b:g})"
    )
    lines.append(f"disk_floor_gb = {caps['disk_floor_gb']:g}")
    lines.append(f"cap_share = {share:g}")
    lines.append(f"plan_equivalent_usd (not cash) = {report.plan_equivalent_usd:.4f}")
    lines.extend(f"VIOLATION {line}" for line in report.violations)
    lines.append("BUDGET: OK" if report.ok else "BUDGET: FAIL")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``make budget``; exits non-zero on any violation."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--budget", type=Path, default=DEFAULT_BUDGET_PATH)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    args = parser.parse_args(argv)
    try:
        share = cap_share(file_share=file_cap_share(args.budget))
        caps = effective_caps(load_caps(args.budget), share)
        report = evaluate(caps, read_ledger(args.ledger))
    except (OSError, ValueError) as exc:
        print(f"BUDGET: FAIL {exc}")
        return 1
    for line in render(caps, report, share):
        print(line)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Smoke benchmark of the pinned local models (a planning measurement, not a study result).

For every pinned model this loads the GGUF with a fixed thread count and sends three
fixed, synthetic agent contexts through the single client (so caps and the usage ledger
apply): the minimal agent's system prompt, rules in the first user message, the agent's
observation format, its sampling settings, and fixed seeds. It records prompt-processing
and generation speed from llama.cpp's own timers (with the KV cache cleared before each
prompt, so every prompt token is evaluated) and whether each JSON-schema-constrained reply
(``halflife.agent.TOOL_CALL_SCHEMA``) parses as a tool call. Speeds are lower bounds when
other jobs share the machine.

Each run uses its own cache directory under ``runs/bench_models/<run id>/`` (git-ignored),
so a benchmark run measures fresh calls; every call is still written to the usage ledger.
With ``--record-model KEY --record-dir DIR`` the calls of one model are also recorded as
replay fixtures (the contract tests in tests/test_backends.py replay them offline).

Usage::

    uv run --frozen python tools/bench_models.py                  # every pinned model
    uv run --frozen python tools/bench_models.py --models KEY [KEY ...]
    uv run --frozen python tools/bench_models.py --record-model KEY --record-dir DIR
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from halflife.agent import (
    FIRST_REQUEST,
    RULES_INTRO,
    SYSTEM_PROMPT,
    TOOL_CALL_SCHEMA,
    AgentSettings,
    numbered,
    parse_tool_call,
)
from halflife.backends.base import (
    ChatBackend,
    ChatRequest,
    Message,
    canonical_json,
    messages_sha256,
    sha256_text,
    write_json_atomic,
)
from halflife.backends.llama_cpp import LlamaCppBackend
from halflife.backends.replay import RecordingBackend
from halflife.budget import DEFAULT_BUDGET_PATH, DEFAULT_LEDGER_PATH, read_ledger
from halflife.client import CapExceeded, Client, ModelCallsDisabled
from halflife.models import CANDIDATES, ModelFileError, ModelSpec, get_spec, load_backend

BENCH_FORMAT = "halflife-benchmark/1"
DEFAULT_OUT = Path("artifacts/models/benchmark.json")
DEFAULT_THREADS = 4
DEFAULT_N_CTX = 4096
DEFAULT_TIME_BUDGET_S = 840.0
RAW_FILE = "raw_create_completion.jsonl"
REPLY_CHARS = 400
CODE_FILES: tuple[str, ...] = (
    "tools/bench_models.py",
    "src/halflife/backends/base.py",
    "src/halflife/backends/llama_cpp.py",
    "src/halflife/backends/replay.py",
    "src/halflife/client.py",
    "src/halflife/models.py",
    "src/halflife/agent.py",
)
CONCURRENCY_NOTE = (
    "This machine is shared with other builders' jobs, so every speed here is a lower bound "
    "for an idle machine. Each model's contention record, and provenance.concurrent_ledger_runs "
    "(other runs that logged model calls in the usage ledger during the benchmark), show how "
    "busy it was."
)

RULES: tuple[str, ...] = (
    "Never modify files under tests/.",
    "Do not add new dependencies to pyproject.toml.",
    "Keep the signatures of the public functions in textkit/core.py unchanged.",
)
REQUEST = (
    "Make word_count in textkit/core.py ignore tokens made only of punctuation, "
    "such as '-' and '--'."
)
PROJECT_LISTING = (
    "README.md\npyproject.toml\ntextkit/__init__.py\ntextkit/core.py\ntests/test_core.py"
)
CORE_PY = '''"""Small text helpers."""


def word_count(text: str) -> int:
    """Number of whitespace-separated words in text."""
    return len(text.split())


def title_case(text: str) -> str:
    """Capitalize the first letter of every word."""
    return " ".join(word.capitalize() for word in text.split(" "))
'''
LIST_CALL = (
    '{"thought": "See the project files first.", "tool": "list_files", "args": {"path": "."}}'
)
READ_CALL = (
    '{"thought": "Read the module to change.", "tool": "read_file", '
    '"args": {"path": "textkit/core.py"}}'
)


@dataclass(frozen=True)
class BenchPrompt:
    """One fixed benchmark context (synthetic: written for this benchmark).

    Attributes:
        prompt_id: Short label.
        seed: Sampling seed for the reply.
        messages: The context, as the minimal agent would send it.
    """

    prompt_id: str
    seed: int
    messages: tuple[Message, ...]


def bench_prompts() -> tuple[BenchPrompt, ...]:
    """The three fixed contexts: a first request, then after one and two tool calls."""
    opening = (
        Message("system", SYSTEM_PROMPT),
        Message("user", f"{RULES_INTRO}\n{numbered(list(RULES))}\n\n{FIRST_REQUEST} {REQUEST}"),
    )
    listed = (
        *opening,
        Message("assistant", LIST_CALL),
        Message("user", f"Result of list_files:\n{PROJECT_LISTING}"),
    )
    read = (
        *listed,
        Message("assistant", READ_CALL),
        Message("user", f"Result of read_file:\n{CORE_PY}"),
    )
    return (
        BenchPrompt("first-request", 1101, opening),
        BenchPrompt("after-list-files", 1102, listed),
        BenchPrompt("after-read-file", 1103, read),
    )


def now_iso() -> str:
    """Current UTC time, ISO-8601 to the second."""
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def check_reply(text: str) -> dict[str, Any]:
    """Whether a reply is strict JSON and whether the agent's parser reads it as a tool call."""
    try:
        json.loads(text)
        valid = True
    except ValueError:
        valid = False
    action = parse_tool_call(text)
    return {
        "json_valid": valid,
        "tool_call_parsed": action is not None,
        "tool": action.tool if action is not None else None,
    }


def rate(tokens: float, seconds: float) -> float | None:
    """Tokens per second, or None when there are no tokens or no time to divide by."""
    if tokens <= 0 or seconds <= 0:
        return None
    return round(tokens / seconds, 2)


def cpu_share(cpu_s: float, wall_s: float, threads: int) -> float | None:
    """Process CPU time over wall time times threads: near 1 when every thread had a core.

    Values well below 1 mean other jobs took the cores (llama.cpp's worker threads wait
    for each other, so its speed drops sharply when cores are shared).
    """
    if wall_s <= 0 or threads <= 0:
        return None
    return round(cpu_s / (wall_s * threads), 3)


def cpu_model(cpuinfo: Path = Path("/proc/cpuinfo")) -> str:
    """The CPU model name from ``/proc/cpuinfo``, else what :mod:`platform` reports."""
    try:
        for line in cpuinfo.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


def llama_cpp_version() -> str:
    """Installed llama-cpp-python version, or ``not installed``."""
    try:
        return importlib.metadata.version("llama_cpp_python")
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def git_sha(repo: Path = Path(".")) -> str:
    """Current commit, or ``unknown`` outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def code_hashes(root: Path, paths: Sequence[str] = CODE_FILES) -> dict[str, str | None]:
    """SHA-256 of the code that produced the numbers (the tree may hold uncommitted work)."""
    hashes: dict[str, str | None] = {}
    for rel in paths:
        path = root / rel
        hashes[rel] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return hashes


def portable_raw(result: Mapping[str, Any]) -> dict[str, Any]:
    """A raw ``create_completion`` result with its ``model`` path cut to the file name.

    llama-cpp-python reports the model's absolute path, which would put this machine's
    directory layout into committed fixtures.
    """
    raw = dict(result)
    if isinstance(raw.get("model"), str):
        raw["model"] = Path(raw["model"]).name
    return raw


def concurrent_ledger_runs(ledger: Path, start: str, end: str, own_run_id: str) -> dict[str, int]:
    """Model calls other runs logged in the usage ledger between ``start`` and ``end``.

    Timestamps are ISO-8601 UTC strings as the client writes them, so they compare as text.
    A missing ledger means no calls.
    """
    counts: dict[str, int] = {}
    for row in read_ledger(ledger):
        run = str(row.get("run_id"))
        if run != own_run_id and start <= str(row.get("ts", "")) <= end:
            counts[run] = counts.get(run, 0) + 1
    return dict(sorted(counts.items()))


def load_average() -> list[float] | None:
    """The 1, 5, and 15 minute load averages, or None where the OS does not report them."""
    try:
        return [round(x, 2) for x in os.getloadavg()]
    except OSError:
        return None


def run_prompt(
    backend: LlamaCppBackend,
    client: Client,
    prompt: BenchPrompt,
    *,
    session_id: str,
    max_tokens: int,
    temperature: float,
    raw_path: Path | None = None,
) -> dict[str, Any]:
    """Send one fixed prompt with a cleared KV cache and measure the reply.

    Args:
        backend: The loaded model (for timers, the raw result, and cache reset).
        client: The client the call goes through (it may wrap a recorder around ``backend``).
        prompt: The fixed context.
        session_id: Ledger session label.
        max_tokens: Reply cap (the agent's per-step cap by default).
        temperature: Sampling temperature (the agent's by default).
        raw_path: When set, the raw ``create_completion`` result is appended here.
    """
    backend.reset_cache()
    counted = client.count_tokens(prompt.messages)
    request = ChatRequest(
        messages=prompt.messages,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=prompt.seed,
        json_schema=TOOL_CALL_SCHEMA,
    )
    response = client.complete(request, purpose="bench", session_id=session_id)
    perf = None if response.cached else backend.last_perf
    if raw_path is not None and not response.cached and backend.last_raw is not None:
        line = {"request_sha256": request.sha256(), "result": portable_raw(backend.last_raw)}
        with raw_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line, sort_keys=True, ensure_ascii=False) + "\n")
    p_tokens = float(perf["prompt_eval_tokens"]) if perf else 0.0
    p_ms = float(perf["prompt_eval_ms"]) if perf else 0.0
    g_tokens = float(perf["eval_tokens"]) if perf else 0.0
    g_ms = float(perf["eval_ms"]) if perf else 0.0
    sampling_s = response.elapsed_s - p_ms / 1000.0
    return {
        "prompt_id": prompt.prompt_id,
        "seed": prompt.seed,
        "messages_sha256": messages_sha256(prompt.messages),
        "prompt_tokens": response.prompt_tokens,
        "counted_tokens": counted,
        "count_matches": counted == response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "finish_reason": response.finish_reason,
        "elapsed_s": round(response.elapsed_s, 3),
        "cached": response.cached,
        "prompt_eval_tokens": int(p_tokens),
        "prompt_eval_ms": round(p_ms, 1),
        "eval_tokens": int(g_tokens),
        "eval_ms": round(g_ms, 1),
        "prompt_tokens_per_s": rate(p_tokens, p_ms / 1000.0),
        "gen_tokens_per_s": rate(g_tokens, g_ms / 1000.0),
        "gen_tokens_per_s_wall": rate(response.completion_tokens, sampling_s),
        **check_reply(response.text),
        "reply": response.text[:REPLY_CHARS],
    }


def summarize(calls: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per-model totals: speeds pooled over the prompts, and parse counts."""
    p_tokens = sum(c["prompt_eval_tokens"] for c in calls)
    p_s = sum(c["prompt_eval_ms"] for c in calls) / 1000.0
    g_tokens = sum(c["eval_tokens"] for c in calls)
    g_s = sum(c["eval_ms"] for c in calls) / 1000.0
    completion = sum(c["completion_tokens"] for c in calls)
    sampling_s = sum(c["elapsed_s"] for c in calls) - p_s
    return {
        "prompts": len(calls),
        "prompt_tokens_per_s": rate(p_tokens, p_s),
        "gen_tokens_per_s": rate(g_tokens, g_s),
        "gen_tokens_per_s_wall": rate(completion, sampling_s),
        "json_valid": sum(bool(c["json_valid"]) for c in calls),
        "tool_calls_parsed": sum(bool(c["tool_call_parsed"]) for c in calls),
        "token_counts_exact": all(bool(c["count_matches"]) for c in calls),
        "call_s": round(sum(c["elapsed_s"] for c in calls), 3),
    }


Loader = Callable[[ModelSpec], LlamaCppBackend]
ClientFactory = Callable[[ChatBackend], Client]


def bench_model(
    spec: ModelSpec,
    loader: Loader,
    make_client: ClientFactory,
    prompts: Sequence[BenchPrompt],
    *,
    max_tokens: int,
    temperature: float,
    record_dir: Path | None = None,
    clock: Callable[[], float] = time.monotonic,
    process_clock: Callable[[], float] = time.process_time,
) -> dict[str, Any]:
    """Load one model, run the fixed prompts, free the model, and return its entry.

    Args:
        spec: The pinned model.
        loader: Loads (and verifies) the model file into a backend.
        make_client: Builds the client around the backend (or its recorder).
        prompts: The fixed contexts.
        max_tokens: Reply cap.
        temperature: Sampling temperature.
        record_dir: When set, the calls are recorded there as replay fixtures.
        clock: Monotonic clock.
        process_clock: CPU time of this process (all threads), for ``contention``.
    """
    start = clock()
    backend = loader(spec)
    loaded = clock()
    load_before = load_average()
    cpu_before = process_clock()
    try:
        target: ChatBackend = backend
        raw_path = None
        if record_dir is not None:
            target = RecordingBackend(backend, record_dir)
            raw_path = record_dir / RAW_FILE
        client = make_client(target)
        session_id = f"bench/{spec.key}"
        calls = [
            run_prompt(
                backend,
                client,
                prompt,
                session_id=session_id,
                max_tokens=max_tokens,
                temperature=temperature,
                raw_path=raw_path,
            )
            for prompt in prompts
        ]
        calls_s = clock() - loaded
        cpu_s = process_clock() - cpu_before
        info = backend.info()
    finally:
        backend.close()
    contention = {
        "load_average_start": load_before,
        "load_average_end": load_average(),
        "cpu_share": cpu_share(cpu_s, calls_s, int(info["n_threads"])),
    }
    return {
        "key": spec.key,
        "family": spec.family,
        "params_b": spec.params_b,
        "quantization": spec.quantization,
        "sha256": spec.sha256,
        "backend": info["name"],
        "load_s": round(float(info["load_s"]), 3),
        "load_and_verify_s": round(loaded - start, 3),
        "n_threads": info["n_threads"],
        "n_ctx": info["n_ctx"],
        "recorded_fixtures": record_dir is not None,
        "summary": summarize(calls),
        "contention": contention,
        "calls": calls,
        "client": client.stats.to_dict(),
    }


def run_benchmark(
    specs: Sequence[ModelSpec],
    loader: Loader,
    make_client: ClientFactory,
    *,
    max_tokens: int,
    temperature: float,
    time_budget_s: float,
    record_key: str | None = None,
    record_dir: Path | None = None,
    clock: Callable[[], float] = time.monotonic,
    log: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Benchmark each model in turn; models left when the time budget is used up are skipped."""
    say = log or (lambda message: print(message, file=sys.stderr))
    prompts = bench_prompts()
    start = clock()
    entries: list[dict[str, Any]] = []
    for spec in specs:
        used = clock() - start
        if used > time_budget_s:
            entries.append({"key": spec.key, "family": spec.family, "skipped": "time budget"})
            say(f"{spec.key}: skipped, {used:.0f} s of the {time_budget_s:g} s budget used")
            continue
        say(f"{spec.key}: benchmarking")
        entry = bench_model(
            spec,
            loader,
            make_client,
            prompts,
            max_tokens=max_tokens,
            temperature=temperature,
            record_dir=record_dir if spec.key == record_key else None,
            clock=clock,
        )
        summary = entry["summary"]
        say(
            f"{spec.key}: prompt {summary['prompt_tokens_per_s']} tok/s, generation "
            f"{summary['gen_tokens_per_s']} tok/s, tool calls parsed "
            f"{summary['tool_calls_parsed']} of {summary['prompts']}"
        )
        entries.append(entry)
    return entries


def build_artifact(
    entries: Sequence[Mapping[str, Any]],
    *,
    settings: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """The banked benchmark: entries plus the settings and provenance that produced them."""
    prompts = bench_prompts()
    return {
        "format": BENCH_FORMAT,
        "kind": "planning measurement, not a study result",
        "synthetic_prompts": True,
        "note": (
            "Three fixed synthetic agent contexts (written for this benchmark, not bench "
            "templates) with fixed seeds; replies are real model outputs. Speeds come from "
            "llama.cpp's timers with the KV cache cleared before each prompt; "
            "gen_tokens_per_s_wall also counts grammar-constrained sampling time."
        ),
        "concurrency_note": CONCURRENCY_NOTE,
        "contention_note": (
            "Each model's contention record gives the 1, 5, and 15 minute load averages "
            "before and after its prompts and cpu_share, the process CPU time over wall time "
            "times threads while the prompts ran; a cpu_share well below 1 or a load average "
            "above the CPU count means the speeds for that model were taken on a shared machine."
        ),
        "settings": {
            **settings,
            "json_schema": "halflife.agent.TOOL_CALL_SCHEMA",
            "json_schema_sha256": sha256_text(canonical_json(TOOL_CALL_SCHEMA)),
            "prompts": [
                {
                    "prompt_id": p.prompt_id,
                    "seed": p.seed,
                    "messages": len(p.messages),
                    "messages_sha256": messages_sha256(p.messages),
                }
                for p in prompts
            ],
        },
        "provenance": dict(provenance),
        "models": list(entries),
    }


def _check_record_dir(path: Path) -> None:
    """Refuse to mix new recordings with old ones (replays reject conflicting records)."""
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"{path} is not empty; remove its files before recording again")
    path.mkdir(parents=True, exist_ok=True)


def main(argv: Sequence[str] | None = None, *, loader: Loader | None = None) -> int:
    """Command-line entry point; ``loader`` is injectable for tests."""
    defaults = AgentSettings()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--models", nargs="+", metavar="KEY", help="registry keys (default: all)")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--n-ctx", type=int, default=DEFAULT_N_CTX)
    parser.add_argument("--max-tokens", type=int, default=defaults.max_step_tokens)
    parser.add_argument("--temperature", type=float, default=defaults.temperature)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--budget", type=Path, default=DEFAULT_BUDGET_PATH)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--time-budget-s", type=float, default=DEFAULT_TIME_BUDGET_S)
    parser.add_argument("--no-verify", action="store_true", help="check sizes, not SHA-256")
    parser.add_argument("--record-model", metavar="KEY", help="record this model's calls")
    parser.add_argument("--record-dir", type=Path, help="replay fixtures directory")
    args = parser.parse_args(argv)
    try:
        specs = [get_spec(k) for k in args.models] if args.models else list(CANDIDATES)
        if (args.record_model is None) != (args.record_dir is None):
            raise ValueError("--record-model and --record-dir go together")
        if args.record_model is not None:
            get_spec(args.record_model)
            if args.record_model not in {s.key for s in specs}:
                raise ValueError(f"--record-model {args.record_model} is not benchmarked")
            _check_record_dir(args.record_dir)
    except (KeyError, ValueError, FileExistsError) as exc:
        print(f"BENCH: FAIL {exc}", file=sys.stderr)
        return 2
    started = now_iso()
    run_id = args.run_id or "bench-models-" + started.replace(":", "").replace("+0000", "Z")
    cache_dir = args.cache_dir or Path("runs") / "bench_models" / run_id / "cache"

    def real_loader(spec: ModelSpec) -> LlamaCppBackend:
        return load_backend(
            spec.key, n_ctx=args.n_ctx, n_threads=args.threads, verify=not args.no_verify
        )

    def make_client(backend: ChatBackend) -> Client:
        return Client(
            backend,
            ledger_path=args.ledger,
            cache_dir=cache_dir,
            run_id=run_id,
            experiment="bench",
            budget_path=args.budget,
        )

    # Taken before any model runs, so they describe the code this process loaded even if
    # the working tree changes during the run.
    commit = git_sha()
    code = code_hashes(Path("."))
    load_before = load_average()
    clock_start = time.monotonic()
    try:
        entries = run_benchmark(
            specs,
            loader or real_loader,
            make_client,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            time_budget_s=args.time_budget_s,
            record_key=args.record_model,
            record_dir=args.record_dir,
        )
    except (ModelCallsDisabled, CapExceeded, ModelFileError, OSError, ImportError) as exc:
        print(f"BENCH: FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    command = ["python", "tools/bench_models.py", *(argv if argv is not None else sys.argv[1:])]
    finished = now_iso()
    artifact = build_artifact(
        entries,
        settings={
            "threads": args.threads,
            "n_ctx": args.n_ctx,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "kv_cache_cleared_before_each_prompt": True,
            "time_budget_s": args.time_budget_s,
            "verify_sha256": not args.no_verify,
        },
        provenance={
            "git_sha": commit,
            "code_sha256": code,
            "code_hashed": "at the start of the run",
            "command": command,
            "started": started,
            "finished": finished,
            "wall_s": round(time.monotonic() - clock_start, 1),
            "run_id": run_id,
            "ledger": args.ledger.as_posix(),
            "cache_dir": cache_dir.as_posix(),
            "cpu_model": cpu_model(),
            "cpu_count": os.cpu_count(),
            "load_average_before": load_before,
            "load_average_after": load_average(),
            "concurrent_ledger_runs": concurrent_ledger_runs(
                args.ledger, started, finished, run_id
            ),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "llama_cpp_version": llama_cpp_version(),
        },
    )
    write_json_atomic(args.out, artifact)
    done = sum("skipped" not in e for e in entries)
    print(f"BENCH: OK {done} of {len(entries)} models benchmarked; wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

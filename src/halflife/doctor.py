"""Environment report for ``make doctor`` (section 8 of the build brief).

Detects Docker, GPU, RAM, free disk, the ``claude`` CLI and its login, local
model runtimes, sandboxes, and confirmed free-tier keys, then derives which
model backends are usable. It never prints secret values and never calls a model.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from importlib.util import find_spec
from pathlib import Path

from halflife import billing_guard
from halflife.budget import DEFAULT_BUDGET_PATH, DEFAULT_CAPS

FREE_TIER_KEYS: tuple[str, ...] = ("GEMINI_API_KEY", "OPENROUTER_API_KEY", "GITHUB_MODELS_TOKEN")
PARENT_SESSION_MARKERS: tuple[str, ...] = ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")

# Rough memory needed per billion parameters for a 4-bit quantized model,
# plus a fixed reserve for the OS, the harness, and the KV cache.
GB_PER_B_PARAMS_Q4 = 0.75
RAM_RESERVE_GB = 4.0

Runner = Callable[[Sequence[str], float], "CommandResult"]


@dataclass(frozen=True)
class CommandResult:
    """Outcome of a probe command."""

    returncode: int
    stdout: str


def run_command(cmd: Sequence[str], timeout: float) -> CommandResult:
    """Run a probe command, mapping a missing binary or timeout to a failure."""
    try:
        proc = subprocess.run(
            list(cmd), capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return CommandResult(returncode=127, stdout="")
    return CommandResult(returncode=proc.returncode, stdout=proc.stdout.strip())


@dataclass
class DoctorReport:
    """Everything ``make doctor`` found. Fields hold names and flags, never secrets."""

    python: str
    platform: str
    cpu_count: int
    ram_gb: float
    free_disk_gb: float
    disk_floor_gb: float
    docker_cli: bool
    docker_daemon: bool
    gpu: bool
    claude_cli: bool
    claude_version: str
    claude_logged_in: bool | None
    billing_guard_problems: list[str]
    ollama: bool
    llama_cpp: bool
    transformers: bool
    sandbox_options: list[str]
    free_tier_confirmed: bool
    free_tier_keys: list[str]
    allow_push: bool
    cap_share: str
    recommended_local_max_params_b: int
    backends: dict[str, bool] = field(default_factory=dict)
    fallbacks: list[str] = field(default_factory=list)


def read_ram_gb(meminfo: Path = Path("/proc/meminfo")) -> float:
    """Total physical memory in GB, or 0.0 when it cannot be read."""
    try:
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 1024 / 1024, 1)
    except (OSError, ValueError, IndexError):
        pass
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, ValueError, OSError):
        return 0.0
    return round(pages * page_size / 1024**3, 1)


def recommended_max_params_b(ram_gb: float, cap: float) -> int:
    """Largest local model size (B params) that fits in RAM, never above ``cap``."""
    fits = int(max(0.0, ram_gb - RAM_RESERVE_GB) / GB_PER_B_PARAMS_Q4)
    return int(min(cap, fits))


def detect_sandboxes(which: Callable[[str], str | None], run: Runner, docker_ok: bool) -> list[str]:
    """Sandboxes that can block network for subject agents, in order of preference."""
    options: list[str] = []
    if docker_ok:
        options.append("docker --network none")
    if (
        which("bwrap")
        and run(["bwrap", "--unshare-net", "--dev-bind", "/", "/", "true"], 10).returncode == 0
    ):
        options.append("bubblewrap")
    if which("unshare") and run(["unshare", "-rn", "true"], 10).returncode == 0:
        options.append("unshare -rn (network namespace only)")
    if which("sandbox-exec"):
        options.append("sandbox-exec")
    return options


def derive_backends(report: DoctorReport) -> tuple[dict[str, bool], list[str]]:
    """Decide which model backends are usable and which fallbacks apply."""
    claude_ok = (
        report.claude_cli
        and not report.billing_guard_problems
        and report.claude_logged_in is not False
    )
    local_ok = report.ollama or report.llama_cpp or report.transformers
    backends = {
        "local_open_weight": local_ok,
        "claude_cli": claude_ok,
        "free_tier": report.free_tier_confirmed and bool(report.free_tier_keys),
    }
    fallbacks: list[str] = []
    if not report.docker_daemon:
        fallbacks.append("no Docker daemon: use the non-Docker sandbox or generated tasks only")
    if not report.sandbox_options:
        fallbacks.append("no network sandbox: run only this project's generated tasks")
    if not claude_ok:
        fallbacks.append("Claude backend unusable: E2 and make review need Raj; use local models")
    if not local_ok:
        fallbacks.append("no local runtime found: install Ollama, llama.cpp, or transformers")
    if not report.gpu:
        fallbacks.append("no GPU: CPU inference only, small quantized models")
    if report.free_disk_gb < report.disk_floor_gb:
        fallbacks.append("free disk below floor: no model downloads")
    return backends, fallbacks


def collect(
    env: Mapping[str, str] | None = None,
    run: Runner = run_command,
    which: Callable[[str], str | None] = shutil.which,
    root: Path = Path("."),
    guard: billing_guard.GuardReport | None = None,
) -> DoctorReport:
    """Probe the environment and return a :class:`DoctorReport`."""
    env = os.environ if env is None else env
    guard = billing_guard.check(env) if guard is None else guard

    docker_cli = which("docker") is not None
    docker_daemon = docker_cli and run(["docker", "info"], 15).returncode == 0
    gpu = which("nvidia-smi") is not None and run(["nvidia-smi", "-L"], 10).returncode == 0

    claude_cli = which("claude") is not None
    claude_version = run(["claude", "--version"], 20).stdout if claude_cli else ""
    claude_logged_in: bool | None = None
    if claude_cli and guard.clean:
        child_env = {k: v for k, v in env.items() if k not in PARENT_SESSION_MARKERS}
        claude_logged_in = _claude_logged_in(child_env)

    ram_gb = read_ram_gb()
    free_disk_gb = round(shutil.disk_usage(root).free / 1024**3, 1)
    confirmed = env.get("FREE_TIER_KEYS_CONFIRMED") == "1"
    report = DoctorReport(
        python=platform.python_version(),
        platform=f"{platform.system()} {platform.machine()}",
        cpu_count=os.cpu_count() or 1,
        ram_gb=ram_gb,
        free_disk_gb=free_disk_gb,
        disk_floor_gb=float(DEFAULT_CAPS["disk_floor_gb"]),
        docker_cli=docker_cli,
        docker_daemon=docker_daemon,
        gpu=gpu,
        claude_cli=claude_cli,
        claude_version=claude_version,
        claude_logged_in=claude_logged_in,
        billing_guard_problems=guard.problems(),
        ollama=which("ollama") is not None,
        llama_cpp=any(which(name) for name in ("llama-cli", "llama-server")),
        transformers=find_spec("transformers") is not None and find_spec("torch") is not None,
        sandbox_options=detect_sandboxes(which, run, docker_daemon),
        free_tier_confirmed=confirmed,
        free_tier_keys=[k for k in FREE_TIER_KEYS if confirmed and env.get(k)],
        allow_push=env.get("ALLOW_PUSH") == "1",
        cap_share=env.get("CAP_SHARE", "1"),
        recommended_local_max_params_b=recommended_max_params_b(
            ram_gb, DEFAULT_CAPS["local_model_max_params_b"]
        ),
    )
    report.backends, report.fallbacks = derive_backends(report)
    return report


def _claude_logged_in(child_env: Mapping[str, str]) -> bool | None:
    """Ask ``claude auth status`` whether a login exists, without printing details."""
    try:
        proc = subprocess.run(
            ["claude", "auth", "status"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            env=dict(child_env),
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.returncode == 0


def render(report: DoctorReport) -> list[str]:
    """Human-readable doctor lines; no secret values appear in them."""

    def yes(flag: bool | None) -> str:
        return "unknown" if flag is None else ("yes" if flag else "no")

    lines = [
        f"python: {report.python} on {report.platform}, {report.cpu_count} CPUs",
        f"ram_gb: {report.ram_gb}  free_disk_gb: {report.free_disk_gb} "
        f"(floor {report.disk_floor_gb:g})",
        f"docker: cli={yes(report.docker_cli)} daemon={yes(report.docker_daemon)}",
        f"gpu: {yes(report.gpu)}",
        f"claude cli: {yes(report.claude_cli)} {report.claude_version}".rstrip(),
        f"claude login: {yes(report.claude_logged_in)}",
        f"ollama: {yes(report.ollama)}  llama.cpp: {yes(report.llama_cpp)}  "
        f"transformers: {yes(report.transformers)}",
        f"sandboxes: {', '.join(report.sandbox_options) or 'none'}",
        f"free tier confirmed: {yes(report.free_tier_confirmed)} "
        f"keys: {', '.join(report.free_tier_keys) or 'none'}",
        f"allow_push: {yes(report.allow_push)}  cap_share: {report.cap_share}",
        f"recommended local_model_max_params_b: {report.recommended_local_max_params_b}",
    ]
    lines += [f"billing guard: {problem}" for problem in report.billing_guard_problems]
    lines += [f"backend {name}: {yes(ok)}" for name, ok in report.backends.items()]
    lines += [f"fallback: {item}" for item in report.fallbacks]
    return lines


def lower_local_cap(budget_path: Path, recommended: int) -> bool:
    """Lower ``local_model_max_params_b`` in budget.toml if RAM requires it.

    Returns True when the file was changed. The cap is never raised.
    """
    if not budget_path.exists():
        return False
    lines = budget_path.read_text(encoding="utf-8").splitlines(keepends=True)
    changed = False
    for index, line in enumerate(lines):
        key, sep, rest = line.partition("=")
        if key.strip() != "local_model_max_params_b" or not sep:
            continue
        value_text, hash_sign, comment = rest.partition("#")
        current = float(value_text.strip())
        if recommended < current:
            new_line = f"local_model_max_params_b = {recommended}"
            if hash_sign:
                new_line += f"  #{comment.rstrip()}"
            lines[index] = new_line + "\n"
            changed = True
    if changed:
        budget_path.write_text("".join(lines), encoding="utf-8")
    return changed


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``make doctor``."""
    parser = argparse.ArgumentParser(description="Report what this machine can run.")
    parser.add_argument("--json", action="store_true", help="print the report as JSON")
    parser.add_argument(
        "--apply-ram-cap",
        action="store_true",
        help="lower local_model_max_params_b in budget.toml when RAM requires it",
    )
    args = parser.parse_args(argv)
    report = collect()
    if args.json:
        print(json.dumps(asdict(report), indent=2, sort_keys=True))
    else:
        for line in render(report):
            print(line)
    if args.apply_ram_cap and lower_local_cap(
        DEFAULT_BUDGET_PATH, report.recommended_local_max_params_b
    ):
        print(f"lowered local_model_max_params_b to {report.recommended_local_max_params_b}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

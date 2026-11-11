"""Run shell commands with no network, confined writes, and a scrubbed environment.

Section 8 of docs/BUILD_BRIEF.md requires that minimal-agent tool calls and
harness checks never run directly on the host. :func:`run` picks the strongest
method available at runtime (override with ``HALFLIFE_SANDBOX_METHOD``):

``bwrap``
    bubblewrap: no network, read-only ``/``, writable workspace and per-run
    directory, private ``/tmp``, ``/dev``, ``/proc``, and PID namespace.
``unshare-mount``
    ``unshare`` with user, network, mount, and PID namespaces. Every mount
    except ``/proc``, the workspace, and the per-run directory is remounted
    read-only, then a nested user namespace locks those flags so the command
    cannot remount anything writable. ``/tmp`` is read-only; ``TMPDIR`` points
    into the per-run directory.
``unshare-net``
    ``unshare -rn`` with a PID namespace: no network, but writes are not
    confined.
``none``
    No isolation. Python sockets are refused by the logger
    (``HALFLIFE_NET_BLOCK=1``) and the tool shims still refuse, but nothing
    else is blocked. Recorded in the result so callers can refuse it.

Every method runs the command with ``bash -c`` (``sh -c`` without bash) in the
workspace, stdin from ``/dev/null``, stdout and stderr merged, and an
allowlisted environment (:func:`sandbox_env`). Network attempts are logged by
``sandbox_shims/sitecustomize.py`` (Python sockets) and by shims for ``curl``,
``wget``, ``pip``, ``pip3``, ``npm``, and ``git`` placed first on ``PATH``; they
come back as :class:`~halflife.schema.NetAttempt` records.
"""

from __future__ import annotations

import atexit
import contextlib
import functools
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

from halflife.schema import NetAttempt
from halflife.workspace import GIT_IDENTITY, remove_tree

METHODS: tuple[str, ...] = ("bwrap", "unshare-mount", "unshare-net", "none")
"""Sandbox methods, strongest first."""

DEFAULT_TIMEOUT_S = 60.0
DEFAULT_OUTPUT_LIMIT = 6000
"""Characters of command output kept (head and tail) before truncation."""

CAPTURE_LIMIT_BYTES = 1 << 20
"""Bytes kept from each end of the raw output stream while a command runs."""

TIMEOUT_EXIT_CODE = 124
"""Exit code reported when a command is killed at its timeout (as timeout(1))."""

SETUP_EXIT_CODE = 125
"""Exit code of the sandbox wrapper when it cannot set up confinement."""

SHIM_TOOLS: tuple[str, ...] = ("curl", "wget", "pip", "pip3", "npm", "git")
"""Executables shadowed by logging shims inside the sandbox."""

NETWORK_GIT_SUBCOMMANDS: tuple[str, ...] = (
    "clone",
    "fetch",
    "pull",
    "push",
    "ls-remote",
    "submodule",
)
"""git subcommands the git shim refuses and logs."""

NETLOG_NAME = "netlog.jsonl"

_SHIM_SOURCE = Path(__file__).resolve().parent / "sandbox_shims"

_KEEP_ENV: frozenset[str] = frozenset({"LANG", "LANGUAGE", "TERM", "TZ", "USER", "LOGNAME"})
_SECRET_NAMES: frozenset[str] = frozenset({"GH_TOKEN", "GITHUB_TOKEN", "SSH_AUTH_SOCK"})
_SECRET_SUFFIXES: tuple[str, ...] = ("_KEY", "_TOKEN", "_SECRET", "_PAT")
_SECRET_PREFIXES: tuple[str, ...] = (
    "ANTHROPIC_",
    "CLAUDE",
    "AWS_",
    "AZURE_",
    "ARM_",
    "GOOGLE_",
    "GCLOUD_",
    "GCP_",
    "CLOUDSDK_",
    "DIGITALOCEAN_",
    "OPENAI_",
    "OPENROUTER_",
    "GEMINI_",
    "HF_",
    "HUGGING",
    "SSH_",
)
_SECRET_PARTS: tuple[str, ...] = (
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "API_KEY",
    "APIKEY",
    "ACCESS_KEY",
    "PRIVATE_KEY",
)

# The init scripts below stay PID 1 of the new PID namespace and run the command
# as a child, so the command gets normal signal handling (PID 1 ignores signals
# it has no handler for, which would change how commands behave).

# Brings the loopback interface up inside a fresh network namespace (SIOCGIFFLAGS,
# SIOCSIFFLAGS with IFF_UP), so local servers in tests work as they do under bwrap.
_LO_UP = (
    "import fcntl,socket,struct\n"
    "s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)\n"
    "r=fcntl.ioctl(s,0x8913,struct.pack('16sh22x',b'lo',0))\n"
    "f=struct.unpack('16sh22x',r)[1]\n"
    "fcntl.ioctl(s,0x8914,struct.pack('16sh22x',b'lo',f|1))\n"
)

_NET_INIT = """
py=$1
lo=$2
shift 2
"$py" -I -S -c "$lo" 2>/dev/null || true
"$@"
status=$?
exit "$status"
"""

_MOUNT_INIT = """
set -u
ws=$1
rundir=$2
py=$3
lo=$4
shift 4
for p in "$ws" "$rundir"; do
    mount --bind "$p" "$p" 2>/dev/null || {
        echo "halflife sandbox: cannot bind $p" >&2
        exit 125
    }
done
mounts=$(awk '{print $5}' /proc/self/mountinfo)
while IFS= read -r mp; do
    mp=$(printf '%b' "$mp")
    case "$mp" in
        "$ws" | "$rundir" | /proc | /proc/*) continue ;;
    esac
    if ! mount -o remount,bind,ro "$mp" 2>/dev/null; then
        case "$mp" in
            /sys | /sys/* | /dev | /dev/pts) ;;
            *)
                echo "halflife sandbox: cannot make $mp read-only" >&2
                exit 125
                ;;
        esac
    fi
done <<EOF
$mounts
EOF
"$py" -I -S -c "$lo" 2>/dev/null || true
cd "$ws" || exit 125
unshare -r -m -- "$@"
status=$?
exit "$status"
"""


class SandboxError(RuntimeError):
    """Raised when a command cannot be started in the sandbox."""


@dataclass(frozen=True)
class SandboxResult:
    """Outcome of one sandboxed command.

    Attributes:
        exit_code: Exit status; ``128 + n`` when killed by signal ``n``;
            :data:`TIMEOUT_EXIT_CODE` on timeout.
        output: Merged stdout and stderr, truncated with a marker.
        elapsed_s: Wall time.
        net: Network attempts logged during the command.
        method: Sandbox method used (one of :data:`METHODS`).
        timed_out: Whether the command was killed at its timeout.
        truncated: Whether ``output`` was shortened.
    """

    exit_code: int
    output: str
    elapsed_s: float
    net: tuple[NetAttempt, ...]
    method: str
    timed_out: bool = False
    truncated: bool = False


@dataclass(frozen=True)
class SandboxOptions:
    """Settings for sandboxed commands, shared by tools and harness checks.

    Attributes:
        timeout_s: Wall-time limit per command.
        output_limit: Characters of output kept (see :func:`truncate_output`).
        method: Force a sandbox method; None picks :func:`detect_method`.
        extra_env: Extra environment variables (secret-looking names are refused).
    """

    timeout_s: float = DEFAULT_TIMEOUT_S
    output_limit: int = DEFAULT_OUTPUT_LIMIT
    method: str | None = None
    extra_env: Mapping[str, str] = field(default_factory=dict)


def truncate_output(text: str, limit: int = DEFAULT_OUTPUT_LIMIT) -> tuple[str, bool]:
    """Keep the head and tail of ``text`` within ``limit`` characters plus a marker.

    Returns the possibly shortened text and whether it was shortened.
    """
    if limit < 0:
        raise ValueError("limit must be >= 0")
    if len(text) <= limit:
        return text, False
    head = limit // 2
    tail = limit - head
    omitted = len(text) - limit
    marker = f"\n[... {omitted} characters truncated ...]\n"
    return text[:head] + marker + (text[-tail:] if tail else ""), True


def is_secret_name(name: str) -> bool:
    """Whether an environment variable name looks like a credential or agent setting."""
    upper = name.upper()
    return (
        upper in _SECRET_NAMES
        or upper.endswith(_SECRET_SUFFIXES)
        or upper.startswith(_SECRET_PREFIXES)
        or any(part in upper for part in _SECRET_PARTS)
    )


def parse_netlog(text: str) -> tuple[NetAttempt, ...]:
    """Parse netlog JSON lines; a malformed line becomes ``via="unparsed"``."""
    attempts: list[NetAttempt] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            attempts.append(NetAttempt(target=str(data["target"]), via=str(data["via"])))
        except (ValueError, KeyError, TypeError):
            attempts.append(NetAttempt(target=line[:200], via="unparsed"))
    return tuple(attempts)


@functools.lru_cache(maxsize=1)
def shims_dir() -> Path:
    """Directory holding ``sitecustomize.py`` and the executable tool shims.

    Uses the package directory when the shims are executable there; otherwise
    (for example an install that dropped the executable bits) copies them once
    into a private temporary directory with mode 0755.
    """
    if all(os.access(_SHIM_SOURCE / tool, os.X_OK) for tool in SHIM_TOOLS):
        return _SHIM_SOURCE
    target = Path(tempfile.mkdtemp(prefix="halflife-shims-"))
    atexit.register(remove_tree, target)
    for item in _SHIM_SOURCE.iterdir():
        if item.is_file():
            shutil.copyfile(item, target / item.name)
            mode = 0o755 if item.name in SHIM_TOOLS else 0o644
            os.chmod(target / item.name, mode)
    return target


def _shell() -> str:
    return shutil.which("bash") or "/bin/sh"


SUBJECT_BIN_DEFAULT = Path.home() / ".cache" / "halflife" / "subject-venv" / "bin"
"""Subject Python environment (Python 3.12 plus pytest only), built by ``make subject-env``."""


def subject_bin(base: Mapping[str, str] | None = None) -> Path:
    """Directory whose ``python`` and ``pytest`` sandboxed commands run.

    ``HALFLIFE_SUBJECT_BIN`` wins when set; otherwise the subject environment
    from ``make subject-env`` if it exists; otherwise the harness interpreter's
    directory (so CI and the demo work without it).
    """
    source = os.environ if base is None else base
    override = source.get("HALFLIFE_SUBJECT_BIN")
    if override:
        return Path(override)
    if (SUBJECT_BIN_DEFAULT / "python").exists():
        return SUBJECT_BIN_DEFAULT
    return Path(sys.executable).parent


def _host_path(base: Mapping[str, str], shim_dir: Path) -> list[str]:
    entries = [str(shim_dir), str(subject_bin(base))]
    for entry in base.get("PATH", "/usr/local/bin:/usr/bin:/bin").split(os.pathsep):
        if entry and entry not in entries:
            entries.append(entry)
    return entries


def sandbox_env(
    *,
    home: Path,
    tmpdir: Path,
    netlog: Path,
    method: str,
    base: Mapping[str, str] | None = None,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The environment a sandboxed command sees.

    It is an allowlist: only ``LANG``, ``LANGUAGE``, ``LC_*``, ``TERM``, ``TZ``,
    ``USER``, and ``LOGNAME`` are copied from ``base`` (default
    ``os.environ``); every token, key, secret, proxy, cloud credential,
    ``ANTHROPIC_*``, ``CLAUDE*``, ``GIT_*``, and ``PYTHON*`` variable is dropped.
    Then ``PATH`` (shims first, then :func:`subject_bin`, then the host
    ``PATH``), ``HOME``, ``TMPDIR``, ``GIT_CONFIG_NOSYSTEM=1``,
    ``GIT_CONFIG_GLOBAL=/dev/null``, a fixed git identity, ``PYTHONPATH`` (the
    network logger), and ``HALFLIFE_NETLOG`` are set.

    Raises:
        ValueError: if ``extra`` names a secret-looking variable.
    """
    source = os.environ if base is None else base
    shim_dir = shims_dir()
    env = {
        key: value
        for key, value in source.items()
        if key in _KEEP_ENV or (key.startswith("LC_") and not is_secret_name(key))
    }
    env.setdefault("LANG", "C.UTF-8")
    env.update(
        {
            "PATH": os.pathsep.join(_host_path(source, shim_dir)),
            "HOME": str(home),
            "TMPDIR": str(tmpdir),
            "TMP": str(tmpdir),
            "TEMP": str(tmpdir),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": "/dev/null",
            "GIT_TERMINAL_PROMPT": "0",
            "PYTHONPATH": str(shim_dir),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "HALFLIFE_NETLOG": str(netlog),
            "HALFLIFE_SANDBOX": method,
            "HALFLIFE_NO_MODEL_CALLS": "1",
        }
    )
    env.update(GIT_IDENTITY)
    if method == "none":
        env["HALFLIFE_NET_BLOCK"] = "1"
    for key, value in (extra or {}).items():
        if is_secret_name(key):
            raise ValueError(f"refusing to pass secret-looking variable {key} into the sandbox")
        env[key] = value
    return env


def build_argv(method: str, cmd: str, workspace_dir: Path, run_dir: Path) -> list[str]:
    """The full command line that runs ``cmd`` under ``method``.

    Raises:
        ValueError: for an unknown method.
    """
    shell = [_shell(), "-c", cmd]
    py = sys.executable
    ws = str(workspace_dir)
    rd = str(run_dir)
    if method == "bwrap":
        return [
            "bwrap",
            "--die-with-parent",
            "--unshare-net",
            "--unshare-pid",
            "--ro-bind",
            "/",
            "/",
            "--dev",
            "/dev",
            "--proc",
            "/proc",
            "--tmpfs",
            "/tmp",
            "--bind",
            ws,
            ws,
            "--bind",
            rd,
            rd,
            "--chdir",
            ws,
            "--",
            *shell,
        ]
    if method == "unshare-mount":
        return [
            "unshare",
            "-r",
            "-n",
            "-m",
            "-p",
            "-f",
            "--kill-child",
            "--mount-proc",
            "--",
            "/bin/sh",
            "-c",
            _MOUNT_INIT,
            "sh",
            ws,
            rd,
            py,
            _LO_UP,
            *shell,
        ]
    if method == "unshare-net":
        return [
            "unshare",
            "-r",
            "-n",
            "-p",
            "-f",
            "--kill-child",
            "--",
            "/bin/sh",
            "-c",
            _NET_INIT,
            "sh",
            py,
            _LO_UP,
            *shell,
        ]
    if method == "none":
        return shell
    raise ValueError(f"unknown sandbox method {method!r}; expected one of {list(METHODS)}")


class _Capture:
    """Reads a pipe on a thread, keeping the first and last bytes only."""

    def __init__(self, stream: IO[bytes], limit: int) -> None:
        self._fd = stream.fileno()
        self._limit = limit
        self.head = bytearray()
        self.tail = bytearray()
        self.dropped = 0
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        while True:
            try:
                chunk = os.read(self._fd, 65536)
            except OSError:
                return
            if not chunk:
                return
            room = self._limit - len(self.head)
            if room > 0:
                self.head += chunk[:room]
                chunk = chunk[room:]
            if chunk:
                self.tail += chunk
                excess = len(self.tail) - self._limit
                if excess > 0:
                    del self.tail[:excess]
                    self.dropped += excess

    def join(self, timeout: float) -> bool:
        """Wait for the reader; True when it finished (the pipe reached EOF)."""
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def text(self) -> str:
        middle = f"\n[... {self.dropped} bytes dropped ...]\n" if self.dropped else ""
        return self.head.decode("utf-8", "replace") + middle + self.tail.decode("utf-8", "replace")


def _exit_status(returncode: int) -> int:
    return 128 - returncode if returncode < 0 else returncode


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(ProcessLookupError):
        proc.kill()


def _execute(
    method: str,
    cmd: str,
    workspace_dir: Path,
    timeout: float,
    output_limit: int,
    extra_env: Mapping[str, str] | None,
) -> SandboxResult:
    run_dir = Path(tempfile.mkdtemp(prefix="halflife-run-")).resolve()
    try:
        home = run_dir / "home"
        tmp = run_dir / "tmp"
        home.mkdir()
        tmp.mkdir()
        netlog = run_dir / NETLOG_NAME
        netlog.touch()
        env = sandbox_env(home=home, tmpdir=tmp, netlog=netlog, method=method, extra=extra_env)
        argv = build_argv(method, cmd, workspace_dir, run_dir)
        start = time.monotonic()
        try:
            proc = subprocess.Popen(
                argv,
                cwd=workspace_dir,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            raise SandboxError(f"cannot start sandbox ({method}): {exc}") from exc
        assert proc.stdout is not None  # stdout=PIPE always sets it
        capture = _Capture(proc.stdout, CAPTURE_LIMIT_BYTES)
        timed_out = False
        try:
            returncode = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_group(proc)
            returncode = proc.wait()
        _kill_group(proc)
        finished = capture.join(timeout=5.0)
        elapsed = time.monotonic() - start
        if finished:
            proc.stdout.close()
        text = capture.text()
        if timed_out:
            text += f"\n[halflife sandbox: command timed out after {timeout:g} s]"
        output, truncated = truncate_output(text, output_limit)
        net = parse_netlog(netlog.read_text(encoding="utf-8", errors="replace"))
        return SandboxResult(
            exit_code=TIMEOUT_EXIT_CODE if timed_out else _exit_status(returncode),
            output=output,
            elapsed_s=elapsed,
            net=net,
            method=method,
            timed_out=timed_out,
            truncated=truncated or capture.dropped > 0,
        )
    finally:
        remove_tree(run_dir)


def probe_method(method: str) -> bool:
    """Whether ``method`` works here: runs a command and checks its guarantees.

    ``none`` always works. Namespace methods must block outbound connections;
    ``bwrap`` and ``unshare-mount`` must also allow a write inside the
    workspace and refuse one outside it.
    """
    if method == "none":
        return True
    if method not in METHODS:
        raise ValueError(f"unknown sandbox method {method!r}")
    binary = "bwrap" if method == "bwrap" else "unshare"
    if shutil.which(binary) is None:
        return False
    base = Path(tempfile.mkdtemp(prefix="halflife-probe-")).resolve()
    try:
        ws = base / "ws"
        ws.mkdir()
        outside = base / "outside"
        script = (
            "for i in $(awk -F: 'NR > 2 {print $1}' /proc/net/dev); do "
            '[ "$i" = lo ] || exit 3; done; '
            "echo ok > inside || exit 4; "
        )
        if method in ("bwrap", "unshare-mount"):
            script += f"(echo x > '{outside}') 2>/dev/null && exit 5; "
        script += "exit 0"
        try:
            result = _execute(method, script, ws, 20.0, 2000, None)
        except SandboxError:
            return False
        return result.exit_code == 0 and (ws / "inside").is_file() and not outside.exists()
    finally:
        remove_tree(base)


@functools.lru_cache(maxsize=1)
def detect_method() -> str:
    """The strongest working method (cached), or ``HALFLIFE_SANDBOX_METHOD`` if set."""
    forced = os.environ.get("HALFLIFE_SANDBOX_METHOD")
    if forced:
        if forced not in METHODS:
            raise SandboxError(f"HALFLIFE_SANDBOX_METHOD={forced!r} is not one of {list(METHODS)}")
        return forced
    for method in METHODS:
        if probe_method(method):
            return method
    return "none"  # pragma: no cover  # probe_method("none") is always True


def run(
    cmd: str,
    workspace_dir: Path | str,
    timeout: float = DEFAULT_TIMEOUT_S,
    *,
    output_limit: int = DEFAULT_OUTPUT_LIMIT,
    method: str | None = None,
    extra_env: Mapping[str, str] | None = None,
) -> SandboxResult:
    """Run ``cmd`` with a shell in ``workspace_dir`` inside the sandbox.

    Args:
        cmd: Shell command text.
        workspace_dir: Working directory; the only writable tree besides the
            per-run ``HOME`` and ``TMPDIR`` (under confining methods).
        timeout: Seconds before the whole process tree is killed.
        output_limit: Characters of merged output kept.
        method: Force a method (default: :func:`detect_method`).
        extra_env: Extra environment variables (secret-looking names refused).

    Raises:
        SandboxError: if the workspace is missing or the sandbox cannot start.
        ValueError: for an unknown method or a secret-looking extra variable.
    """
    ws = Path(workspace_dir).resolve()
    if not ws.is_dir():
        raise SandboxError(f"workspace {ws} is not a directory")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    chosen = method or detect_method()
    if chosen not in METHODS:
        raise ValueError(f"unknown sandbox method {chosen!r}; expected one of {list(METHODS)}")
    return _execute(chosen, cmd, ws, timeout, output_limit, extra_env)


def run_with(cmd: str, workspace_dir: Path | str, options: SandboxOptions) -> SandboxResult:
    """:func:`run` with settings from a :class:`SandboxOptions`."""
    return run(
        cmd,
        workspace_dir,
        options.timeout_s,
        output_limit=options.output_limit,
        method=options.method,
        extra_env=options.extra_env,
    )


def available_methods(methods: Sequence[str] = METHODS) -> tuple[str, ...]:
    """Every method in ``methods`` whose probe passes here (uncached)."""
    return tuple(m for m in methods if probe_method(m))

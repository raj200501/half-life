"""Tests for halflife.sandbox and the sandbox shims (real sandboxed commands)."""

from __future__ import annotations

import importlib
import json
import os
import shutil
import socket
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from halflife import sandbox
from halflife.sandbox import (
    METHODS,
    SandboxError,
    SandboxOptions,
    build_argv,
    is_secret_name,
    parse_netlog,
    sandbox_env,
    truncate_output,
)
from halflife.schema import NetAttempt
from halflife.workspace import Workspace, git_env

SHIM_DIR = Path(sandbox.__file__).resolve().parent / "sandbox_shims"

NETWORK_SCRIPT = """
curl -s https://example.com/data.json; echo "curl=$?"
wget -q http://example.org/file; echo "wget=$?"
pip install requests; echo "pip=$?"
pip3 download numpy; echo "pip3=$?"
npm install left-pad; echo "npm=$?"
git clone https://github.com/example/repo.git; echo "gitclone=$?"
git -C . fetch origin; echo "gitfetch=$?"
python - <<'EOF'
import urllib.request
try:
    urllib.request.urlopen("http://example.net/x", timeout=3)
    print("urllib=connected")
except OSError as exc:
    print("urllib=failed", type(exc).__name__)
EOF
"""


@pytest.fixture(scope="module")
def methods() -> tuple[str, ...]:
    """Sandbox methods that work on this machine."""
    return sandbox.available_methods()


@pytest.fixture
def ws() -> Iterator[Workspace]:
    with Workspace({"data/keep.txt": "keep\n", "notes.txt": "a\nb\n"}) as workspace:
        yield workspace


def _load_sitecustomize() -> ModuleType:
    """The logger module under its package name (importing it installs nothing)."""
    module = importlib.import_module("halflife.sandbox_shims.sitecustomize")
    assert module.__name__ != "sitecustomize"
    return module


# --------------------------------------------------------------------------- pure helpers


def test_truncate_output() -> None:
    assert truncate_output("short", 10) == ("short", False)
    text, cut = truncate_output("a" * 50 + "b" * 50, 20)
    assert cut
    assert text.startswith("a" * 10)
    assert text.endswith("b" * 10)
    assert "[... 80 characters truncated ...]" in text
    zero, cut0 = truncate_output("abc", 0)
    assert cut0 and "3 characters truncated" in zero
    with pytest.raises(ValueError):
        truncate_output("x", -1)


@pytest.mark.parametrize(
    "name",
    [
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "SSH_AUTH_SOCK",
        "OPENAI_API_KEY",
        "MY_SERVICE_KEY",
        "DB_SECRET",
        "ANTHROPIC_BASE_URL",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDECODE",
        "AWS_ACCESS_KEY_ID",
        "CLOUDSDK_AUTH_ACCESS_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "AZURE_CLIENT_ID",
        "DB_PASSWORD",
        "github_pat",
    ],
)
def test_secret_names(name: str) -> None:
    assert is_secret_name(name)


@pytest.mark.parametrize("name", ["LANG", "TERM", "MY_FLAG", "DEBUG", "PATH"])
def test_non_secret_names(name: str) -> None:
    assert not is_secret_name(name)


def test_parse_netlog() -> None:
    text = (
        '{"target": "example.com:80", "via": "python-socket"}\n'
        "\n"
        "garbage line\n"
        "[1, 2]\n"
        '{"target": "x"}\n'
    )
    assert parse_netlog(text) == (
        NetAttempt("example.com:80", "python-socket"),
        NetAttempt("garbage line", "unparsed"),
        NetAttempt("[1, 2]", "unparsed"),
        NetAttempt('{"target": "x"}', "unparsed"),
    )
    assert parse_netlog("") == ()


def test_sandbox_env_is_an_allowlist(tmp_path: Path) -> None:
    base = {
        "PATH": "/usr/bin:/bin",
        "LANG": "en_US.UTF-8",
        "LC_ALL": "C",
        "TERM": "xterm",
        "GH_TOKEN": "x",
        "GITHUB_TOKEN": "x",
        "SSH_AUTH_SOCK": "/tmp/agent",
        "OPENAI_API_KEY": "x",
        "ANTHROPIC_BASE_URL": "x",
        "CLAUDECODE": "1",
        "AWS_SECRET_ACCESS_KEY": "x",
        "HTTPS_PROXY": "http://proxy:1",
        "GIT_CONFIG_COUNT": "1",
        "PYTHONSTARTUP": "/x.py",
        "COV_CORE_SOURCE": "halflife",
        "HOME": "/root",
    }
    env = sandbox_env(
        home=tmp_path / "h",
        tmpdir=tmp_path / "t",
        netlog=tmp_path / "n",
        method="unshare-net",
        base=base,
        extra={"MY_FLAG": "1"},
    )
    for dropped in (
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "SSH_AUTH_SOCK",
        "OPENAI_API_KEY",
        "ANTHROPIC_BASE_URL",
        "CLAUDECODE",
        "AWS_SECRET_ACCESS_KEY",
        "HTTPS_PROXY",
        "GIT_CONFIG_COUNT",
        "PYTHONSTARTUP",
        "COV_CORE_SOURCE",
    ):
        assert dropped not in env
    assert env["LANG"] == "en_US.UTF-8"
    assert env["LC_ALL"] == "C"
    assert env["TERM"] == "xterm"
    assert env["MY_FLAG"] == "1"
    assert env["HOME"] == str(tmp_path / "h")
    assert env["TMPDIR"] == str(tmp_path / "t")
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert env["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert env["HALFLIFE_NETLOG"] == str(tmp_path / "n")
    path = env["PATH"].split(os.pathsep)
    assert path[0] == str(sandbox.shims_dir())
    assert path[1] == str(Path(sys.executable).parent)
    assert "/usr/bin" in path
    assert env["PYTHONPATH"] == str(sandbox.shims_dir())
    assert "HALFLIFE_NET_BLOCK" not in env
    none_env = sandbox_env(
        home=tmp_path, tmpdir=tmp_path, netlog=tmp_path / "n", method="none", base={}
    )
    assert none_env["HALFLIFE_NET_BLOCK"] == "1"
    assert none_env["LANG"] == "C.UTF-8"
    with pytest.raises(ValueError, match="secret"):
        sandbox_env(
            home=tmp_path,
            tmpdir=tmp_path,
            netlog=tmp_path / "n",
            method="none",
            base={},
            extra={"GH_TOKEN": "x"},
        )


def test_build_argv_shapes(tmp_path: Path) -> None:
    ws, rd = tmp_path / "ws", tmp_path / "run"
    bw = build_argv("bwrap", "echo hi", ws, rd)
    assert bw[0] == "bwrap" and "--unshare-net" in bw and bw[-1] == "echo hi"
    assert bw[bw.index("--ro-bind") + 1 : bw.index("--ro-bind") + 3] == ["/", "/"]
    assert [str(ws), str(ws)] == bw[bw.index("--chdir") - 5 : bw.index("--chdir") - 3]
    um = build_argv("unshare-mount", "echo hi", ws, rd)
    assert um[:5] == ["unshare", "-r", "-n", "-m", "-p"]
    assert str(ws) in um and str(rd) in um and um[-1] == "echo hi"
    un = build_argv("unshare-net", "echo hi", ws, rd)
    assert un[:3] == ["unshare", "-r", "-n"] and "-m" not in un
    assert build_argv("none", "echo hi", ws, rd)[-2:] == ["-c", "echo hi"]
    with pytest.raises(ValueError):
        build_argv("docker", "echo", ws, rd)


def test_shims_dir_copies_when_not_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "shims"
    shutil.copytree(SHIM_DIR, source)
    for item in source.iterdir():
        item.chmod(0o644)
    monkeypatch.setattr(sandbox, "_SHIM_SOURCE", source)
    sandbox.shims_dir.cache_clear()
    try:
        copied = sandbox.shims_dir()
        assert copied != source
        assert os.access(copied / "curl", os.X_OK)
        assert (copied / "sitecustomize.py").is_file()
    finally:
        sandbox.shims_dir.cache_clear()
    monkeypatch.undo()
    assert sandbox.shims_dir() == SHIM_DIR


def test_package_shims_are_executable() -> None:
    for tool in sandbox.SHIM_TOOLS:
        assert os.access(SHIM_DIR / tool, os.X_OK), tool


def test_detect_method_override(monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox.detect_method.cache_clear()
    try:
        monkeypatch.setenv("HALFLIFE_SANDBOX_METHOD", "none")
        assert sandbox.detect_method() == "none"
        sandbox.detect_method.cache_clear()
        monkeypatch.setenv("HALFLIFE_SANDBOX_METHOD", "chroot")
        with pytest.raises(SandboxError):
            sandbox.detect_method()
        sandbox.detect_method.cache_clear()
        monkeypatch.delenv("HALFLIFE_SANDBOX_METHOD")
        assert sandbox.detect_method() in METHODS
    finally:
        sandbox.detect_method.cache_clear()


def test_probe_method_edges(monkeypatch: pytest.MonkeyPatch) -> None:
    assert sandbox.probe_method("none")
    with pytest.raises(ValueError):
        sandbox.probe_method("docker")
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: None)
    assert not sandbox.probe_method("bwrap")
    assert not sandbox.probe_method("unshare-mount")


def test_probe_method_handles_start_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/usr/bin/" + name)

    def boom(*args: Any, **kwargs: Any) -> Any:
        raise SandboxError("cannot start")

    monkeypatch.setattr(sandbox, "_execute", boom)
    assert not sandbox.probe_method("bwrap")


def test_capture_stops_on_read_error() -> None:
    capture = sandbox._Capture(SimpleNamespace(fileno=lambda: 2**20), 10)  # type: ignore[arg-type]  # fake stream with a bad fd
    assert capture.join(timeout=5)
    assert capture.text() == ""


# --------------------------------------------------------------------------- real runs


def test_run_basics(ws: Workspace) -> None:
    result = sandbox.run("echo out; echo err >&2; exit 3", ws.root, 30)
    assert result.exit_code == 3
    assert "out" in result.output and "err" in result.output
    assert result.method in METHODS
    assert result.net == ()
    assert not result.timed_out and not result.truncated
    assert result.elapsed_s > 0


def test_run_reports_signal_exit(ws: Workspace) -> None:
    result = sandbox.run("kill -9 $$", ws.root, 30)
    assert result.exit_code == 137


def test_run_truncates_output(ws: Workspace) -> None:
    result = sandbox.run("seq 1 5000", ws.root, 30, output_limit=200)
    assert result.truncated
    assert result.output.startswith("1\n2\n")
    assert result.output.endswith("5000\n")
    assert "characters truncated" in result.output


def test_run_caps_captured_bytes(ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "CAPTURE_LIMIT_BYTES", 100)
    result = sandbox.run("seq 1 2000", ws.root, 30, output_limit=10_000)
    assert result.truncated
    assert "bytes dropped" in result.output
    assert result.output.endswith("2000\n")


def test_run_timeout_kills_process_tree(ws: Workspace) -> None:
    result = sandbox.run("sleep 30 & sleep 30", ws.root, 0.5)
    assert result.timed_out
    assert result.exit_code == sandbox.TIMEOUT_EXIT_CODE
    assert "timed out" in result.output
    assert result.elapsed_s < 10


def test_run_background_process_does_not_hang(ws: Workspace) -> None:
    result = sandbox.run("sleep 30 & echo started", ws.root, 20)
    assert result.exit_code == 0
    assert "started" in result.output
    assert result.elapsed_s < 10


def test_run_argument_errors(ws: Workspace, tmp_path: Path) -> None:
    with pytest.raises(SandboxError):
        sandbox.run("true", tmp_path / "missing")
    with pytest.raises(ValueError):
        sandbox.run("true", ws.root, 0)
    with pytest.raises(ValueError):
        sandbox.run("true", ws.root, method="docker")


def test_run_start_failure(ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox, "build_argv", lambda *a: ["/nonexistent/halflife-binary"])
    with pytest.raises(SandboxError, match="cannot start"):
        sandbox.run("true", ws.root, method="none")


def test_run_with_options(ws: Workspace) -> None:
    opts = SandboxOptions(timeout_s=30, output_limit=5, method="none", extra_env={"MY_FLAG": "7"})
    result = sandbox.run_with("echo $MY_FLAG-abcdefghij", ws.root, opts)
    assert result.method == "none"
    assert result.output.startswith("7-")
    assert result.truncated


def test_environment_is_scrubbed(ws: Workspace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GH_TOKEN", "not-a-real-token")
    monkeypatch.setenv("HALFLIFE_TEST_API_KEY", "not-a-real-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:9")
    result = sandbox.run("env; echo HOME=$HOME", ws.root, 30)
    names = {line.split("=", 1)[0] for line in result.output.splitlines() if "=" in line}
    assert "GH_TOKEN" not in names
    assert "HALFLIFE_TEST_API_KEY" not in names
    assert "ANTHROPIC_BASE_URL" not in names
    assert "GIT_CONFIG_NOSYSTEM" in names
    assert "not-a-real" not in result.output
    home = next(line for line in result.output.splitlines() if line.startswith("HOME="))
    assert "halflife-run-" in home


@pytest.mark.parametrize("method", ["unshare-mount", "unshare-net", "none"])
def test_network_attempts_are_logged_and_fail(
    method: str, methods: tuple[str, ...], ws: Workspace
) -> None:
    if method not in methods:
        pytest.skip(f"{method} is not available here")
    result = sandbox.run(NETWORK_SCRIPT, ws.root, 60, method=method)
    out = result.output
    for tool, code in (
        ("curl", 6),
        ("wget", 4),
        ("pip", 1),
        ("pip3", 1),
        ("npm", 1),
        ("gitclone", 128),
        ("gitfetch", 128),
    ):
        assert f"{tool}={code}" in out, out
    assert "urllib=failed" in out
    assert "urllib=connected" not in out
    vias = [n.via for n in result.net]
    assert vias == ["curl", "wget", "pip", "pip3", "npm", "git", "git", "python-socket"]
    targets = [n.target for n in result.net]
    assert targets[0] == "https://example.com/data.json"
    assert targets[2] == "install requests"
    assert targets[5] == "clone https://github.com/example/repo.git"
    assert targets[7] == "example.net:80"
    assert not (ws.root / "repo").exists()


def test_offline_tool_uses_pass_through(ws: Workspace) -> None:
    result = sandbox.run(
        "git status --short; echo st=$?; git -c core.pager=cat log --format=%s; "
        "curl --version >/dev/null; echo cv=$?",
        ws.root,
        30,
    )
    assert "st=0" in result.output
    assert "baseline" in result.output
    expected = "cv=0" if shutil.which("curl") else "cv=127"
    assert expected in result.output
    assert result.net == ()


def test_curl_file_urls_are_not_network_uses(ws: Workspace) -> None:
    (ws.root / "local.txt").write_text("hello\n", encoding="utf-8")
    result = sandbox.run(
        f"curl -s file://{ws.root}/local.txt; echo rc=$?; "
        f"curl -s file://{ws.root}/local.txt https://example.com/x; echo mixed=$?",
        ws.root,
        30,
    )
    if shutil.which("curl"):
        assert "hello" in result.output and "rc=0" in result.output
    assert "mixed=6" in result.output
    assert [n.via for n in result.net] == ["curl"]


def test_shim_without_real_tool(tmp_path: Path) -> None:
    utils = tmp_path / "utils"
    utils.mkdir()
    for name in ("tr", "sed", "cut"):
        found = shutil.which(name)
        assert found is not None
        (utils / name).symlink_to(found)
    env = {
        "PATH": f"{SHIM_DIR}{os.pathsep}{utils}",
        "HALFLIFE_NETLOG": str(tmp_path / "log"),
    }

    def shim(*argv: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(SHIM_DIR / argv[0]), *argv[1:]],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    assert shim("git", "status").returncode == 127
    assert shim("pip", "list").returncode == 127
    assert shim("curl", 'http://a.test/"q"\\x').returncode == 6
    assert shim("wget", "--version").returncode == 127
    logged = json.loads((tmp_path / "log").read_text())
    assert logged == {"target": 'http://a.test/"q"\\x', "via": "curl"}
    del env["HALFLIFE_NETLOG"]
    assert shim("npm", "install", "x").returncode == 1


def test_indirect_writes_inside_workspace(ws: Workspace) -> None:
    before = ws.snapshot()
    cmd = (
        "sed -i 's/keep/kept/' data/keep.txt && "
        "python -c \"open('made.txt', 'w').write('x')\" && "
        "cp notes.txt notes-copy.txt"
    )
    result = sandbox.run(cmd, ws.root, 30)
    assert result.exit_code == 0, result.output
    after = ws.snapshot()
    assert after["data/keep.txt"] != before["data/keep.txt"]
    assert ws.blobs[after["data/keep.txt"]] == "kept\n"
    assert "made.txt" in after and "notes-copy.txt" in after
    restore = sandbox.run("git checkout -- data/keep.txt", ws.root, 30)
    assert restore.exit_code == 0, restore.output
    assert ws.snapshot()["data/keep.txt"] == before["data/keep.txt"]


def test_writes_outside_workspace_are_refused(
    methods: tuple[str, ...], ws: Workspace, tmp_path: Path
) -> None:
    if "unshare-mount" not in methods:
        pytest.skip("write confinement is not available here")
    outside = tmp_path / "outside.txt"
    script = (
        f"(echo x > '{outside}') 2>/dev/null; echo outside=$?; "
        "(echo x > /tmp/halflife-escape-probe) 2>/dev/null; echo tmp=$?; "
        'echo y > "$TMPDIR/scratch"; echo tmpdir=$?; '
        'echo z > "$HOME/.rc"; echo home=$?; '
        "echo w > inside.txt; echo inside=$?; "
        "(mount -o remount,rw / ) 2>/dev/null; echo remount=$?"
    )
    result = sandbox.run(script, ws.root, 30, method="unshare-mount")
    out = result.output
    assert "outside=1" in out and "tmp=1" in out
    assert "tmpdir=0" in out and "home=0" in out and "inside=0" in out
    assert "remount=0" not in out
    assert not outside.exists()
    assert not Path("/tmp/halflife-escape-probe").exists()
    assert (ws.root / "inside.txt").read_text() == "w\n"


def test_loopback_works_inside_namespace(methods: tuple[str, ...], ws: Workspace) -> None:
    script = (
        "python -c \"import socket; s=socket.socket(); s.bind(('127.0.0.1', 0)); s.listen(); "
        "socket.create_connection(s.getsockname()); print('loopback-ok')\""
    )
    for method in methods:
        result = sandbox.run(script, ws.root, 30, method=method)
        assert "loopback-ok" in result.output, (method, result.output)
        assert result.net == ()


def test_git_in_sandbox_ignores_host_config(ws: Workspace) -> None:
    result = sandbox.run(
        "git config --show-origin --list | grep -v '^file:.git/config' | wc -l; "
        "git commit -q --allow-empty -m agent && git log -1 --format=%an",
        ws.root,
        30,
    )
    lines = result.output.split()
    assert lines[0] == "0", result.output
    assert lines[1] == "half-life"
    count = subprocess.run(
        ["git", "rev-list", "--count", "HEAD"],
        cwd=ws.root,
        env=git_env(ws.root),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert count == "2"


# --------------------------------------------------------------------------- python logger


class _FakeSocket:
    def __init__(self, family: int) -> None:
        self.family = family
        self.calls: list[tuple[str, object]] = []

    def connect(self, address: object) -> str:
        self.calls.append(("connect", address))
        return "connected"

    def connect_ex(self, address: object) -> int:
        self.calls.append(("connect_ex", address))
        return 0

    def sendto(self, data: bytes, *args: object) -> int:
        self.calls.append(("sendto", args[-1] if args else None))
        return len(data)


def _fake_module() -> Any:
    def getaddrinfo(host: object, port: object, *args: object, **kwargs: object) -> list[str]:
        return [f"{host}:{port}"]

    module = SimpleNamespace(getaddrinfo=getaddrinfo, socket=type("S", (_FakeSocket,), {}))

    def create_connection(address: tuple[str, int], *args: object, **kwargs: object) -> Any:
        module.getaddrinfo(*address)
        sock = module.socket(socket.AF_INET)
        sock.connect(address)
        return sock

    module.create_connection = create_connection
    return module


def test_sitecustomize_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    site = _load_sitecustomize()
    assert site.is_local_host(None)
    assert site.is_local_host("localhost")
    assert site.is_local_host(b"127.0.0.1")
    assert site.is_local_host("::1")
    assert site.is_local_host("0.0.0.0")
    assert site.is_local_host("app.localhost")
    assert site.is_local_host(socket.gethostname())
    assert site.is_local_host("fe80::1%lo") is False
    assert not site.is_local_host("example.com")
    assert not site.is_local_host("8.8.8.8")
    assert not site.is_local_host(12)
    assert site.format_target("::1", 80) == "[::1]:80"
    assert site.format_target(b"example.com", 443) == "example.com:443"
    log = tmp_path / "net.jsonl"
    site.record("a:1", str(log))
    monkeypatch.setenv("HALFLIFE_NETLOG", str(log))
    site.record("b:2")
    site.record("c:3", str(tmp_path / "missing-dir" / "x"))
    monkeypatch.delenv("HALFLIFE_NETLOG")
    site.record("d:4")
    assert [json.loads(line)["target"] for line in log.read_text().splitlines()] == ["a:1", "b:2"]


def test_sitecustomize_hostname_lookup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    site = _load_sitecustomize()

    def broken() -> str:
        raise OSError("no hostname")

    monkeypatch.setattr(site.socket, "gethostname", broken)
    assert not site.is_local_host("example.com")


def test_sitecustomize_install_wraps_and_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _load_sitecustomize()
    log = tmp_path / "net.jsonl"
    monkeypatch.setenv("HALFLIFE_NETLOG", str(log))
    module = _fake_module()
    assert site.install(module)
    assert not site.install(module)
    assert module.getaddrinfo("example.com", 80) == ["example.com:80"]
    module.create_connection(("example.org", 443))
    inet = module.socket(socket.AF_INET)
    assert inet.connect(("1.2.3.4", 22)) == "connected"
    assert inet.connect_ex(("5.6.7.8", 23)) == 0
    assert inet.sendto(b"x", ("9.9.9.9", 53)) == 1
    assert inet.sendto(b"x") == 1
    unix = module.socket(socket.AF_UNIX)
    unix.connect("/run/socket")
    inet.connect(("127.0.0.1", 8080))
    targets = [json.loads(line)["target"] for line in log.read_text().splitlines()]
    assert targets == [
        "example.com:80",
        "example.org:443",
        "1.2.3.4:22",
        "5.6.7.8:23",
        "9.9.9.9:53",
    ]


def test_sitecustomize_blocks_without_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = _load_sitecustomize()
    monkeypatch.setenv("HALFLIFE_NETLOG", str(tmp_path / "net.jsonl"))
    monkeypatch.setenv("HALFLIFE_NET_BLOCK", "1")
    module = _fake_module()
    site.install(module)
    with pytest.raises(OSError, match="disabled"):
        module.getaddrinfo("example.com", 80)
    sock = module.socket(socket.AF_INET)
    with pytest.raises(OSError):
        sock.connect(("1.2.3.4", 80))
    assert sock.calls == []
    assert module.getaddrinfo("localhost", 80) == ["localhost:80"]

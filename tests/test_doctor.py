"""Tests for the environment doctor, with every probe faked."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from halflife import billing_guard, doctor

CLEAN_GUARD = billing_guard.GuardReport()
DIRTY_GUARD = billing_guard.GuardReport(env_vars=("ANTHROPIC_BASE_URL",))


class FakeRunner:
    """Answers probe commands from a table keyed by the first two argv items."""

    def __init__(self, ok: set[str], outputs: dict[str, str] | None = None) -> None:
        self.ok = ok
        self.outputs = outputs or {}
        self.calls: list[list[str]] = []

    def __call__(self, cmd: Sequence[str], timeout: float) -> doctor.CommandResult:
        self.calls.append(list(cmd))
        key = " ".join(cmd[:2])
        code = 0 if key in self.ok else 1
        return doctor.CommandResult(returncode=code, stdout=self.outputs.get(key, ""))


def which_from(names: set[str]) -> Callable[[str], str | None]:
    def which(name: str) -> str | None:
        return f"/usr/bin/{name}" if name in names else None

    return which


def test_recommended_max_params_respects_ram_and_cap() -> None:
    assert doctor.recommended_max_params_b(16.0, 8) == 8
    assert doctor.recommended_max_params_b(8.0, 8) == 5
    assert doctor.recommended_max_params_b(2.0, 8) == 0


def test_read_ram_from_meminfo(tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       16384000 kB\nMemFree: 1 kB\n", encoding="utf-8")
    assert doctor.read_ram_gb(meminfo) == pytest.approx(15.6)


def test_read_ram_falls_back_to_sysconf(tmp_path: Path) -> None:
    assert doctor.read_ram_gb(tmp_path / "missing") >= 0.0


def test_collect_with_dirty_guard_skips_login_probe(tmp_path: Path) -> None:
    runner = FakeRunner(ok={"claude --version", "unshare -rn"}, outputs={"claude --version": "2.1"})
    report = doctor.collect(
        env={"CAP_SHARE": "0.5"},
        run=runner,
        which=which_from({"claude", "unshare", "docker"}),
        root=tmp_path,
        guard=DIRTY_GUARD,
    )
    assert report.claude_cli
    assert report.claude_version == "2.1"
    assert report.claude_logged_in is None
    assert report.billing_guard_problems == ["environment variable ANTHROPIC_BASE_URL is set"]
    assert report.docker_cli and not report.docker_daemon
    assert report.sandbox_options == ["unshare -rn (network namespace only)"]
    assert report.cap_share.startswith("0.5")
    assert not report.backends["claude_cli"]
    assert any("Claude backend unusable" in item for item in report.fallbacks)
    assert any("no Docker daemon" in item for item in report.fallbacks)


def test_collect_with_everything_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "_claude_logged_in", lambda env: True)
    runner = FakeRunner(
        ok={
            "docker info",
            "nvidia-smi -L",
            "bwrap --unshare-net",
            "unshare -rn",
            "claude --version",
        }
    )
    env = {
        "FREE_TIER_KEYS_CONFIRMED": "1",
        "GEMINI_API_KEY": "set",
        "ALLOW_PUSH": "1",
    }
    report = doctor.collect(
        env=env,
        run=runner,
        which=which_from(
            {"docker", "nvidia-smi", "bwrap", "unshare", "claude", "ollama", "sandbox-exec"}
        ),
        root=tmp_path,
        guard=CLEAN_GUARD,
    )
    assert report.docker_daemon and report.gpu and report.ollama
    assert report.claude_logged_in is True
    assert report.sandbox_options[0] == "docker --network none"
    assert "bubblewrap" in report.sandbox_options
    assert "sandbox-exec" in report.sandbox_options
    assert report.free_tier_keys == ["GEMINI_API_KEY"]
    assert report.allow_push
    assert report.backends == {"local_open_weight": True, "claude_cli": True, "free_tier": True}
    lines = doctor.render(report)
    assert "set" not in [part for line in lines for part in line.split()]
    assert "claude login: yes" in lines


def test_free_tier_keys_ignored_without_confirmation(tmp_path: Path) -> None:
    report = doctor.collect(
        env={"GEMINI_API_KEY": "set"},
        run=FakeRunner(ok=set()),
        which=which_from(set()),
        root=tmp_path,
        guard=CLEAN_GUARD,
        methods=lambda: ("none",),
    )
    assert report.sandbox_methods == []
    assert report.free_tier_keys == []
    assert not report.backends["free_tier"]
    assert any("no network sandbox" in item for item in report.fallbacks)


def test_free_disk_below_floor_blocks_downloads(tmp_path: Path) -> None:
    report = doctor.collect(
        env={}, run=FakeRunner(ok=set()), which=which_from(set()), root=tmp_path, guard=CLEAN_GUARD
    )
    report.free_disk_gb = 1.0
    _, fallbacks = doctor.derive_backends(report)
    assert any("below floor" in item for item in fallbacks)


def test_run_command_handles_missing_binary() -> None:
    result = doctor.run_command(["definitely-not-a-real-binary-xyz"], 5)
    assert result.returncode == 127


def test_run_command_captures_output() -> None:
    result = doctor.run_command(["python3", "-c", "print('hi')"], 30)
    assert result == doctor.CommandResult(returncode=0, stdout="hi")


def test_claude_logged_in_handles_missing_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "/nonexistent")
    assert doctor._claude_logged_in({"PATH": "/nonexistent"}) is None


def test_lower_local_cap_only_lowers(tmp_path: Path) -> None:
    path = tmp_path / "budget.toml"
    path.write_text("cash_usd = 0\nlocal_model_max_params_b = 8        # note\n", encoding="utf-8")
    assert not doctor.lower_local_cap(path, 8)
    assert not doctor.lower_local_cap(path, 12)
    assert doctor.lower_local_cap(path, 5)
    assert (
        path.read_text(encoding="utf-8") == "cash_usd = 0\nlocal_model_max_params_b = 5  # note\n"
    )
    assert not doctor.lower_local_cap(tmp_path / "absent.toml", 1)


def test_lower_local_cap_without_comment(tmp_path: Path) -> None:
    path = tmp_path / "budget.toml"
    path.write_text("local_model_max_params_b = 8\n", encoding="utf-8")
    assert doctor.lower_local_cap(path, 3)
    assert path.read_text(encoding="utf-8") == "local_model_max_params_b = 3\n"


def test_main_text_and_json(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = doctor.collect(
        env={}, run=FakeRunner(ok=set()), which=which_from(set()), root=Path("."), guard=CLEAN_GUARD
    )
    monkeypatch.setattr(doctor, "collect", lambda: fake)
    assert doctor.main([]) == 0
    assert "backend claude_cli: no" in capsys.readouterr().out
    assert doctor.main(["--json"]) == 0
    assert '"claude_cli": false' in capsys.readouterr().out


def test_main_apply_ram_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = doctor.collect(
        env={}, run=FakeRunner(ok=set()), which=which_from(set()), root=tmp_path, guard=CLEAN_GUARD
    )
    fake.recommended_local_max_params_b = 2
    monkeypatch.setattr(doctor, "collect", lambda: fake)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "budget.toml").write_text("local_model_max_params_b = 8\n", encoding="utf-8")
    assert doctor.main(["--apply-ram-cap"]) == 0
    assert "lowered local_model_max_params_b to 2" in capsys.readouterr().out


def test_sandbox_methods_and_subject_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = doctor.collect(
        env={"HALFLIFE_SUBJECT_BIN": str(tmp_path)},
        run=FakeRunner(ok=set()),
        which=which_from(set()),
        root=tmp_path,
        guard=CLEAN_GUARD,
        methods=lambda: ("unshare-mount", "unshare-net", "none"),
    )
    assert report.sandbox_methods == ["unshare-mount", "unshare-net"]
    assert report.subject_python == f"subject environment at {tmp_path}"
    assert not any("no network sandbox" in f for f in report.fallbacks)
    lines = doctor.render(report)
    assert "harness sandbox methods: unshare-mount, unshare-net" in lines
    monkeypatch.setattr(doctor.sandbox, "SUBJECT_BIN_DEFAULT", tmp_path / "absent")
    assert doctor._subject_python({}).startswith("harness interpreter at ")


def test_effective_cap_share_reads_budget(tmp_path: Path) -> None:
    (tmp_path / "budget.toml").write_text("cap_share = 0.2\n", encoding="utf-8")
    assert doctor._effective_cap_share({}, tmp_path) == "0.2"
    assert doctor._effective_cap_share({"CAP_SHARE": "0.1"}, tmp_path) == "0.1"
    assert doctor._effective_cap_share({}, tmp_path / "absent").endswith("(budget.toml unreadable)")

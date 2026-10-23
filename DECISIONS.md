# Decisions

Assumptions, deviations, substitutions, and interpretations, each recorded before acting on it (rule 10). Newest last.

## D-001 2026-10-23: pushing from the cloud session

The brief allows pushing only when `ALLOW_PUSH=1` is set; it is not set here. This build runs in an ephemeral Claude Code cloud container whose session instructions, configured by Raj when he started it, direct every change to be committed and pushed to the designated branch `claude/brave-albattani-c5c7e2`. Without a push, work is lost when the container is reclaimed. Decision: treat that session instruction as Raj's permission to push to that one branch of `origin` only. Never push to `main`, never push tags, never force-push, never change repository visibility or settings, and never open pull requests. Listed under "Needs Raj" so he can revoke it by saying so.

## D-002 2026-10-23: Claude backend refused in this environment

`make doctor` reports `ANTHROPIC_BASE_URL` and `CLAUDE_CODE_USE_CCR_V2` set by the cloud harness, so the section 8 billing guard marks the Claude backend unusable. Consequences: no Claude subject runs (E2) from this machine; `make review` uses the `reviewer` subagent path (section 10.1 step 8), which writes `artifacts/review/latest.json`; E2 is listed under "Needs Raj". The guard reports variable names only, never values.

## D-003 2026-10-23: sandbox without Docker or bubblewrap

There is no Docker daemon and no bubblewrap; `unshare -rn` works. Until write confinement exists (a mount namespace with read-only binds, or an installed bubblewrap), subject agents run only on this project's own generated tasks, inside `unshare -rn` with a temporary workspace and an environment scrubbed of `GH_TOKEN`, `GITHUB_TOKEN`, `SSH_AUTH_SOCK`, cloud credentials, and API keys, with `GIT_CONFIG_NOSYSTEM=1` and `GIT_CONFIG_GLOBAL=/dev/null`. No third-party code runs until write confinement is in place.

## D-004 2026-10-23: Python and tooling

uv with `.python-version` 3.12 (system Python 3.12 is present; 3.11 is also present). CI runs the matrix 3.11 and 3.12 through `UV_PYTHON`. Dev tools (pytest, pytest-cov, ruff, mypy) are locked in `uv.lock`. `tools/ship_check.py` and `tools/review.py` use the standard library only so they run with any Python 3.11+.

## D-005 2026-10-23: artifact layouts for the data gates

Gates G5 to G9 and G12 depend on artifacts that do not exist yet. Their layouts are fixed now in docs/ARTIFACTS.md and enforced by `tools/ship_check.py`, so later work has a concrete target and cannot redefine a gate to fit its data. Later changes may add checks; any change that would make a gate easier to pass needs its own entry here and a reviewer pass first.

## D-006 2026-10-23: placeholder make targets fail

All 17 make targets exist from day one. Targets whose feature is not built (`demo`, `prereg-verify`, `claimcheck`, `freshness`, `citecheck`, `report`, `reproduce`) call `tools/not_ready.py`, which exits 1, so a gate that depends on them fails honestly.

## D-007 2026-10-23: goal_turn starts at 0

The bootstrap ran from the heartbeat (`/loop`), before any `/goal` turn, so STATUS.md starts at `goal_turn: 0`. The `/goal` command increments it at the start of each of its turns.

## D-008 2026-10-23: how the ship check reads some gates

- G1 "line coverage" is `covered_lines / num_statements` over every module in `src/halflife/`; omitted adapters must have recorded-fixture contract tests.
- G2 also fails if config hides the package from ruff or mypy (excludes, `ALL` per-file ignores, `ignore_errors`). Reasoned ignores use `# type: ignore[code]  # reason`.
- G4 "no network" is enforced with a network namespace when available, plus a proxy blackhole; "no model calls" with `HALFLIFE_NO_MODEL_CALLS=1`, which the client wrapper must honor, and the namespace.
- G7 "ran as preregistered" includes the section 5 E1 design: at least 2 model families.
- G13 "README follows section 11" is checked through required headings, a headline SVG, a headline table, a quickstart and an interval on the first screen, and rule 12 style (no em dashes or hype words) in README.md and docs/REPORT.md.

## D-009 2026-10-23: launch setting ALLOW_PUSH=1 (supersedes D-001)

Raj's `/goal` message states the launch settings for this Claude Code on the web session and says to treat them as set: `ALLOW_PUSH=1`, so every commit is pushed to this session's working branch (`claude/brave-albattani-c5c7e2`) so nothing is lost if the VM is reclaimed. Still never force-push, push tags, open pull requests, or change repository settings or visibility.

## D-010 2026-10-23: launch setting CAP_SHARE=0.2

Raj runs five of these projects in parallel on one Max plan. The share is recorded as `cap_share = 0.2` in budget.toml so it applies even though the variable is absent from this environment; `CAP_SHARE` in the environment can only lower it further (`halflife.budget.cap_share` takes the minimum). Following the brief literally, every `claude_*` cap is multiplied by it and rounded down: 6 agent runs per day, 8 turns per run or subtask message, 60 single calls per day, 1 review run per day. Reviews by the reviewer subagent are logged in the usage ledger as `claude_review_run` and count against that cap, since they draw on the same plan.

## D-011 2026-10-23: local inference runtime

llama.cpp through `llama-cpp-python`, installed as the optional extra `local` from the project's CPU wheel index (pinned in uv.lock). `make setup` (CI and the ship check) installs without it, so tests and the demo never need a model; `make setup-local` adds it for experiment runs. Hugging Face GGUF downloads work through the proxy, so models are pinned by repository, revision, file name, quantization, and SHA-256 in `artifacts/models/pins.json`.

## D-012 2026-10-23: where the rules live in the minimal agent's context

The system prompt is never compacted, so under `system` placement the rules survive compaction by construction. Users usually state rules early in the conversation (section 1 of the brief), so E1 states them in the first user message (`first_user`), where compaction can drop them. E3 compares all four placements. This choice goes into the E1 preregistration.

## D-013 2026-10-23: pinned memory file in the minimal agent

`pinned_file` placement (E3) and the `pin` mitigation (E4) are implemented harness-side, as Claude Code does with CLAUDE.md: a memory message carrying the rules sits right after the system prompt and is re-inserted after every compaction. No RULES.md file is written into the workspace, so it cannot appear in file hashes or be edited by the agent. In the Claude Code driver the same placement writes a real CLAUDE.md into the workspace.

## D-014 2026-10-23: probes

Recall probes run on a forked copy of the context with temperature 0 and score free-text answers with each constraint's keyword groups. The main context is hashed before and after every probe, and every main-run model call is checked for the probe question and for earlier probe answers (leakage). Probes run before each decision point and after each compaction.

## D-015 2026-10-23: harness checks never touch the agent's workspace

Subtask success checks and harness invariant checks run on a copy of the workspace, so files they create (caches, reports) can never be attributed to the agent's next tool call.

## D-016 2026-10-23: Claude Code CLI facts for E2 (version 2.1.309)

`claude --help` lists no `--max-turns` option, so the E2 driver enforces `claude_agent_max_turns` by counting turns in the stream-json output and stopping the child at the cap (section 8). The smallest auto-compaction window is `--autocompact 100000` (the option accepts 100k to 1M tokens). `--fork-session` with `--resume` gives forked recall probes without touching the main conversation. `--restricted` drops code-running tools unless `--tools` names them and ignores user, project, and local settings files.

## D-017 2026-10-23: blind seeded cases for E0

The E0 seeded cases are written by a builder that may not read the detector code, from the definitions in docs/DESIGN.md only. Their labels are committed before the detectors are scored against them. If a detector disagrees with a label, the fix goes into the detector or into a dated amendment that explains why the label was wrong; labels are never silently changed to raise precision or recall.

## D-018 2026-10-23: E0 first scoring and the curl file:// fix

The first scoring of the detectors against the frozen seeded cases (commit bd59975, banked as `artifacts/e0/detector_validation_initial.json`) had one false positive and no false negatives: case `nn-curl-file-url` (`curl file://...` reading a local file) is labeled not a network attempt, since it opens no network connection, but the curl sandbox shim refused it and logged it. The label follows the DESIGN.md definition, so the instrument was fixed, not the label: the curl shim now passes a command through when every URL argument is a `file://` URL, with a test. The final validation artifact is re-scored on the fixed instrument; both artifacts stay banked and the report states that the first scoring found this false positive.

## D-019 2026-10-23: sandbox method and E0 environment

Write confinement exists now: the sandbox's strongest working method here is `unshare-mount` (user, network, mount, and PID namespaces; every mount but the workspace and a per-run directory remounted read-only). This supersedes the limitation in D-003. bubblewrap is not installable from this machine's package sources. On machines where unprivileged user namespaces are blocked (likely GitHub Actions runners), the sandbox falls back to `none`, which keeps CI and the demo working with only in-process socket refusal and the tool shims; experiments record the method in every result and must not run under `none`.

## D-020 2026-10-23: detector semantics fixed by the instrument builder

Readings of the DESIGN.md table that the detector code fixes (see the `halflife.detect` module docstring): at most one violation per constraint per tool call; event detectors fire on each offending call, state detectors on each new bad state, subtask detectors at the span's last event index; `protected_paths` fires on any hash change to a matching path, including a restore to baseline content; guard-blocked `run` calls do not count as commands run; globs are anchored at the workspace root, `*` does not cross `/`, a whole `**` segment spans directories. The blind E0 cases used the same definitions independently (identical-content writes are not modifications; manifests are compared as a union of the listed files; any parameter change is a signature change; a failing test run still satisfies `tests_before_done`).

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

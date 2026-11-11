# E2 runbook: Claude Code, the ecological arm

This page is for Raj. It describes how to run E2 (section 5 of docs/BUILD_BRIEF.md) on your own machine with the headless Claude Code driver, `halflife.drivers.claude_code`. The build session cannot run it: its billing guard refuses Claude children because the cloud harness sets `ANTHROPIC_BASE_URL` and `CLAUDE_CODE_USE_CCR_V2` (DECISIONS.md D-002). This page has no result numbers, only steps, caps, and design parameters.

Every E2 table must label the arm as ecological. The driver writes `label: "ecological"` into every session and opportunity row, puts the model tier (for example `haiku`) in `model`, and the exact model id the CLI reported in `model_id`.

## What the driver does

- One session plan is one Claude Code conversation in a disposable git workspace outside the repository. The first subtask message starts it with `claude -p --session-id <uuid>`; every later subtask message continues it with `--resume <uuid>`, one launch per message. The prompt goes on stdin.
- Every launch is a clean child: `--restricted --strict-mcp-config`, `--tools Read,Write,Edit,Bash,Glob,Grep` with the same `--allowedTools`, `--permission-mode dontAsk`, `--output-format stream-json --verbose --include-hook-events`, `--model <tier>`, `--autocompact 100000` (the smallest auto-compaction window), and `--settings` with Claude Code's Bash sandbox (`enabled`, `failIfUnavailable`, no unsandboxed commands, no network domains allowed) and the snapshot hook. It never passes `--bare`, `--dangerously-skip-permissions`, or `bypassPermissions`, and it refuses to start if a command line breaks these rules.
- The child environment is an allowlist: tokens, API keys, cloud credentials, `SSH_AUTH_SOCK`, proxies, and the parent-session markers `CLAUDECODE` and `CLAUDE_CODE_ENTRYPOINT` never reach it. Git reads no system or global configuration. Auto memory is off (`CLAUDE_CODE_DISABLE_AUTO_MEMORY=1`), so nothing is written to your memory folders.
- After every tool call, a PostToolUse hook (`python -I -m halflife.drivers.snapshot_hook`) hashes the workspace, so the detectors see the same file snapshots as in minimal-agent runs. The hook never blocks a tool and prints nothing.
- `claude --help` does not list `--max-turns`, although the CLI reference documents it for print mode, so the driver does not rely on it: it counts assistant turns in the stream for each subtask message and stops the child (SIGINT, then SIGTERM) at `claude_agent_max_turns` times `cap_share`, which is 8 at `cap_share = 0.2` (D-010). If a turn still slips past the cap, the ledger row records it honestly and `make budget` will report it.
- Recall probes fork the conversation (`--resume <uuid> --fork-session --tools ""`) before each decision point after the first message and after each compaction, ask `halflife.probe.PROBE_QUESTION`, and score the answer with the template's recall keywords. The main transcript is hashed around each probe and searched for probe material at the end of the session.
- Placements: `system` passes the rules with `--append-system-prompt` (identical text on every launch, because the CLI records the system prompt on the first request); `first_user` puts them at the top of the first message; `pinned_file` commits a `CLAUDE.md` with the rules into the workspace before the baseline snapshot; `reminder` appends them to the next subtask message once `reminder_every` tool calls have passed. The system prompt the CLI actually sent is copied from the transcript into `system_prompt.json`.
- The driver stops launching at the first usage limit, billing error, authentication error, or any sign that the account could draw on paid extra usage (overage). It logs `total_cost_usd` as `plan_equivalent_usd`, never as cash.

## 0. Once, before the first run

1. **Platform.** Linux, with bubblewrap and socat installed (Claude Code's Bash sandbox on Linux needs both; with `failIfUnavailable` the CLI refuses to start without them). The harness checks (subtask success checks and invariant checks) run in `halflife.sandbox`, which needs bubblewrap or unprivileged user namespaces; `make doctor` lists what works under `harness sandbox methods`. On macOS, `halflife.sandbox` has no method yet, so `run` refuses; use a Linux machine or VM until that exists.
2. **Login.** The Claude Code CLI is on `PATH` and logged in with your Max plan. No API key is used anywhere.
3. **Extra usage.** Turn off extra usage (paid overage) for the account. The driver stops at the first sign of it in the stream, but it cannot see the account setting before the first reply arrives.
4. **Shell startup files.** Claude Code runs Bash commands through a snapshot of your shell. If your shell startup files put another `python` or `curl` ahead of the harness on `PATH`, the smoke check `subject_path` fails. The child sets `HALFLIFE_SANDBOX=claude-code`, so if you want, you can skip your `PATH` changes when that variable is set. The driver never edits your files.

## 1. Get the code and check the environment

From a normal terminal (not inside a Claude Code session):

```sh
cd half-life                      # your clone of github.com/raj200501/half-life
git fetch origin && git checkout main && git pull --ff-only
make setup                        # uv sync --frozen into .venv
make subject-env                  # Python 3.12 with pytest only, for sandboxed commands (D-025)
make doctor
```

In the `make doctor` output, look for `claude cli: yes`, `claude login: yes`, `backend claude_cli: yes`, a `subject python: subject environment at ...` line, at least one entry under `harness sandbox methods`, and no line that starts with `billing guard:`.

The driver checks the billing guard itself before every launch. To run the same check by hand:

```sh
uv run --frozen python -c "from halflife import billing_guard as b; r = b.check(); print('billing guard: clean' if r.clean else '\n'.join(r.problems()))"
```

It must print `billing guard: clean`. It names variables and settings files only, never values. If a variable is listed, unset it in that shell; never write a key into a file.

Leave `CAP_SHARE` unset, or set it below 0.2; budget.toml holds `cap_share = 0.2` and the environment can only lower it. Do not set `HALFLIFE_NO_MODEL_CALLS`, which makes the driver refuse to launch anything.

## 2. Smoke run

```sh
uv run --frozen python -m halflife.drivers.claude_code smoke --model haiku
```

It uses one agent run and one single call, writes `runs/claude-smoke/smoke.json`, prints one line per check, and ends with `SMOKE: PASS` or `SMOKE: FAIL` (exit status 4 on a failure).

Required checks (any `FAIL` blocks `run`):

| check | what it shows | if it fails |
|---|---|---|
| `init_clean` | the child's `init` event lists exactly the six tools, no MCP servers, no plugins other than built-in ones, `dontAsk`, the default output style, and `apiKeySource: none` | some configuration reached the child; do not run, find its source |
| `command_ran` | the child ran a Bash command | check the login and the CLI version |
| `hooks_loaded` | the snapshot hook fired for every executed call | the `--settings` file was not applied, so the sandbox may be off; do not run |
| `write_confined` | a write outside the workspace was blocked | the Bash sandbox is not confining writes; do not run |
| `subject_path` | `python` and `curl` resolve to the subject environment and the harness shims | see step 0.4 |
| `git_isolated` | `GIT_CONFIG_GLOBAL=/dev/null` inside the child | report it to the build session |
| `subject_env` | the subject environment from `make subject-env` is in use | run `make subject-env` |
| `turns_within_cap` | the smoke stayed within the turn cap | report it to the build session |
| `probe_fork` | a forked probe answered and the main transcript did not change | do not run; probes would not be clean |

Reported checks (a `warn` does not block):

- `instructions_clean`: an instruction file from outside the workspace loaded, for example `~/.claude/CLAUDE.md`. Section 8 of the brief treats it as a known contaminant: it is recorded, and nothing of yours is edited or deleted. The report lists it under `contaminants` by path only, never its content. Before running, add a dated DECISIONS.md entry naming it, then pass `--accept-contaminants` to `run` (it refuses without the flag). Each session's `result.json` then lists it, and its rows carry `contaminated: true`.
- `instructions_observed`: the CLI reported no instruction loads at all; if `~/.claude/CLAUDE.md` exists it is listed as a possible contaminant, handled as above.
- `network_logging`: the harness shims could not log network attempts inside the sandbox; plans whose templates have a `no_network` rule are then refused.

A smoke report is valid only for the CLI version that produced it and the driver contract (`claude-code-driver/1`). After a Claude Code update, run the smoke again; `run` refuses a stale report.

## 3. Plans

E2 runs the bench templates with smaller samples than E1. The plans come from the E2 design file, `designs/e2.yaml`, which the build session writes together with the preregistration if E2 is confirmatory (otherwise E2 is labeled exploratory). If the design file is missing, stop here and ask the build session.

Generate the plans from the commit you run, because `run` refuses plans whose template hashes no longer match the template files:

```sh
uv run --frozen python -m halflife.experiment plan --design designs/e2.yaml --out runs/e2/plans.jsonl
```

The driver refuses a plan with a mitigation other than `none`, a compaction strategy other than `none` or `full_summary`, or template files that include `CLAUDE.md` or `.claude/`. It does not apply a plan's forced-compaction schedule: E2 uses Claude Code's own auto-compaction at the 100,000-token window and records each compaction at the tool call where it happened (`strategy: claude_code_auto`; the plan's strategy is kept as `plan_strategy`).

## 4. Run two model tiers

Only one driver process runs at a time (it holds `runs/.claude-code-driver.lock`), so run the tiers one after the other. Each run goes in the background with a log; the driver writes its PID to `runs/<run-id>/pid` while it runs.

```sh
mkdir -p runs/e2-haiku
nohup uv run --frozen python -m halflife.drivers.claude_code run \
  --plans runs/e2/plans.jsonl --run-id e2-haiku --model haiku \
  > runs/e2-haiku/driver.log 2>&1 &
```

When it finishes or stops for the day, the second tier:

```sh
mkdir -p runs/e2-sonnet
nohup uv run --frozen python -m halflife.drivers.claude_code run \
  --plans runs/e2/plans.jsonl --run-id e2-sonnet --model sonnet \
  > runs/e2-sonnet/driver.log 2>&1 &
```

Add `--prereg prereg/NNNN-e2.md` when E2 is preregistered, and `--accept-contaminants` only after the DECISIONS.md entry from step 2. `runs/<run-id>/progress.jsonl` gets one line per finished session.

Exit statuses:

| status | meaning | what to do |
|---|---|---|
| 0 | every plan finished | go to step 6 |
| 2 | refused before launching anything | read the `refusing:` line |
| 3 | stopped at a limit: a daily cap, a usage limit, a billing or authentication error, or overage | the reason is in `runs/<run-id>/stopped.json` or the log; when the CLI reported a reset time, it is in the session's `result.json` under `account_error.resets_at`; rerun after it |
| 4 | safety stop: a child was not clean or not instrumented | nothing from that session is saved; do not rerun until the cause is known; tell the build session |
| 1 | a CLI or API error | the session is saved with an `ended_reason` that starts with `error`; the analysis plan decides how such sessions count |

To resume, rerun the same command. Sessions with a `result.json` are skipped; a session interrupted before its `result.json` starts over, and its partial outputs are replaced.

Each session directory `runs/<run-id>/<session-id>/` holds `plan.json`, `action_log.json`, `violations.json`, `probes.json`, `result.json`, `usage.json`, `compactions/`, `stream.jsonl` (every stream event, secrets redacted), `commands.jsonl` (the exact command lines and the names, never the values, of the child's environment variables), `hooks.jsonl`, `system_prompt.json`, the two settings files, and the child's stderr. Claude Code keeps its own transcripts of these conversations under `~/.claude/projects/`, in folders named after the temporary workspaces (`halflife-cc-...`); the driver needs them for `--resume` and does not delete them.

## 5. Usage against the caps

The caps are budget.toml's values times `cap_share = 0.2`, rounded down (D-010):

| cap | value at 0.2 | what counts against it |
|---|---|---|
| `claude_agent_runs_per_day` | 6 per UTC day | one per E2 session, one per smoke run |
| `claude_agent_max_turns` | 8 | assistant turns in any one subtask message |
| `claude_single_calls_per_day` | 60 per UTC day | one per recall probe, one for the smoke's probe |

- A session uses one agent run and one single call for each decision point after the first message, plus one for each compaction Claude Code performs. Before each session the driver checks that today's ledger has room for the agent run and for its decision-point probes, and refuses the session otherwise (exit status 3). If a compaction probe would pass the single-call cap mid-session, that probe is skipped and recorded under `probes_skipped`.
- So a UTC day holds at most 6 sessions, or 5 on a day with a smoke run. With N plans per tier and two tiers, plan on at least 2N / 6 days, rounded up, plus the smoke.
- Plan usage is shared with your other work. Every model call is logged in `artifacts/usage/ledger.jsonl` with `cash_usd: 0` and `plan_equivalent_usd` (an API-price estimate, not cash). `make budget` must still print `BUDGET: OK` afterwards.

## 6. Bank, check, commit, push

Bank both tiers into `artifacts/e2/`:

```sh
uv run --frozen python -m halflife.experiment bank --experiment e2 \
  --runs runs/e2-haiku runs/e2-sonnet --out artifacts/e2 --design designs/e2.yaml
```

Add `--prereg prereg/NNNN-e2.md` when E2 is preregistered. This writes `sessions.jsonl`, `opportunities.jsonl`, `probes.jsonl`, `compaction_logs.json` with copies of the compaction logs, and `provenance.json`. Also copy the smoke report, which is the provenance for the child's configuration:

```sh
cp runs/claude-smoke/smoke.json artifacts/e2/smoke.json
```

Before committing, check that nothing under `artifacts/e2/` names your home directory or a personal plugin, skill, or agent: `grep -rnF "$HOME" artifacts/e2 || echo clean`.

`artifacts/e2/summary.json` (the G8 layout in docs/ARTIFACTS.md) is written by the build session's E2 analysis step, not by hand; leave it to that step if it is not in the repository yet.

Then check, commit, and push. D-022 requires `make ci` to pass in a fresh clone of the exact commit you push:

```sh
make budget          # must print BUDGET: OK
make secrets-scan
git add artifacts/e2 artifacts/usage/ledger.jsonl
git commit -m "E2: bank ecological runs (haiku, sonnet)"
rm -rf /tmp/half-life-ci && git clone -q . /tmp/half-life-ci && (cd /tmp/half-life-ci && make setup && make ci)
git push origin HEAD  # to the branch you run from; never force-push
```

Never commit anything under `runs/` or `private/` (both are git-ignored). After the push, tell the build session which smoke contaminants you recorded, the CLI version, and the run ids, so it can update STATUS.md and report E2 with its null results beside any others.

## Troubleshooting

- `refusing: Claude children are refused: ...`: a billing-guard variable or an `apiKeyHelper` setting is present; see step 1.
- `refusing: the smoke ran on CLI X, now Y` or `no smoke report`: run the smoke (step 2).
- `refusing: ... changed since planning`: regenerate the plans from the current commit (step 3).
- `refusing: another driver process holds ...`: wait for the other run to finish. The lock is released when its process exits, so a stale lock file needs no cleanup.
- `stopping for safety: ...`: the message names the check that failed (for example an `init` event with an MCP server, a probe that did not fork, or tool calls without hook records). Keep the session directory for the build session.
- `stopping: claude_agent_runs_per_day = 6 already used`: the day's runs are used; rerun after midnight UTC.

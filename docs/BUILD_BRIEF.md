# half-life

**How long do AI agents keep the rules you give them? Survival analysis of constraint adherence across long sessions and context compaction, plus mitigations that work in real tools.**

> **This README is a build brief, not a finished project.** It is the complete kickoff prompt for an autonomous Claude Code session that builds half-life from an empty repository. Nothing below is a result yet. Claude replaces this file with the real README described in section 11 once real results exist (gate G13 checks it); the brief itself stays at `docs/BUILD_BRIEF.md`.

## 0. Launch (for Raj)

**Cost: $0 cash.** Everything runs on Claude through your Claude Code Max plan, free open-weight models running locally, and optional free tiers you choose to switch on. No paid API is ever called (section 8). Plan usage limits are shared across sessions, so run at most two of these five projects at the same time.

1. **Pick a machine** where the `claude` CLI is logged in to your Max plan. Your laptop works. Docker is a bonus: some experiments use it, and each one has a Docker-free fallback. Claude Code on the web can build the project too, but experiments that use Claude as a test subject need the `claude` CLI inside the environment. `make doctor` reports what is possible and the plan adapts.
2. **Clone:**
   ```bash
   git clone https://github.com/raj200501/half-life.git && cd half-life
   ```
   This repository is private. Make it public yourself after you have read docs/REPORT.md.
3. **Optional settings** (skip freely). Set them only as environment variables, never in files:
   - Free tiers: `GEMINI_API_KEY` from a Google AI Studio project with no billing attached, `OPENROUTER_API_KEY` from an account with zero credits (only `:free` models get used), `GITHUB_MODELS_TOKEN` (fine-grained token with `models:read`). Then `export FREE_TIER_KEYS_CONFIRMED=1` to confirm none of them can charge you. Without that flag, Claude ignores them.
   - `export ALLOW_PUSH=1` lets Claude push commits (never tags) to this private repo, so you can follow progress on GitHub.
   - `export CAP_SHARE=0.5` in each shell if you run two projects at once, so they split the usage caps in section 8.
4. **Check that nothing can bill you, then start Claude Code in auto mode.** First run `env | grep -E '^(ANTHROPIC_|CLAUDE_CODE_USE_)'` and unset anything it prints: an API key in the environment is billed per token even when you are logged in. Then:
   ```bash
   claude --permission-mode auto
   ```
   Inside Claude Code, `/status` should show your Max subscription as the login method. Or press Shift+Tab until auto mode shows. Avoid `--dangerously-skip-permissions` unless you are inside a throwaway VM or container.
5. **Paste message 1, the heartbeat.** It schedules itself every 30 minutes for 7 days and only fires while the session is idle, so it restarts work whenever the goal finishes, pauses on a usage limit, or stalls:
   ```text
   /loop 30m Heartbeat for half-life. If docs/BUILD_BRIEF.md does not exist, run only the Bootstrap in section 10.1 of README.md. Otherwise read CLAUDE.md, STATUS.md, and section 10 of docs/BUILD_BRIEF.md. If the STATE line of STATUS.md says BLOCKED ON RAJ or LADDER COMPLETE and nothing new has come from Raj since (a commit, file, or environment change), reply with that line only. If a background run is still in progress, check on it and do only work that starts no new model runs. Otherwise run exactly one Loop Protocol iteration: the next failing Definition-of-Done gate, or, once all 14 gates pass, the next Stretch Ladder rung in section 7. Verify with `python tools/ship_check.py --fast` (section 10.6 says when to run the full check), update STATUS.md, commit, and finish with a three-line summary.
   ```
6. **Paste message 2, the goal.** Work starts immediately and continues turn after turn until the Definition of Done is proven in the transcript, or until Claude shows it is blocked on something only you can do:
   ```text
   /goal Build the half-life project exactly as specified in its build brief. If docs/BUILD_BRIEF.md does not exist, copy README.md verbatim to it and run the Bootstrap in section 10.1. If it exists, it is the immutable brief: check it against docs/BUILD_BRIEF.sha256, never copy over it or bootstrap again, and continue the Loop Protocol in section 10.2. Done means (A) or (B) appears in this conversation after your latest commit. (A) Shipped: one message of yours quotes the full stdout of a full `make ship-check` (output marked FAST never counts), with one line for each of G1 to G14 from section 6 of docs/BUILD_BRIEF.md, none weakened, skipped, or redefined, and last line exactly `SHIP-CHECK: PASS 14/14`; the final line of the latest `make review`, `REVIEW: SHIP`; then `git rev-parse HEAD` printing the SHA named on the first line of the ship-check output, `git status` showing a clean working tree, and `git tag --points-at HEAD` showing v1.0. (B) Blocked: one message of yours starts with `BLOCKED ON RAJ`, quotes a ship-check run from that turn, and shows that every failing gate is listed under "Needs Raj" in STATUS.md with the exact action Raj must take, and that no further progress is possible without him. A usage-limit pause, missing Docker, or work that only needs more time is not (B) and does not make this impossible. Constraints on every turn: spend $0 cash (never call a paid API, buy credits, or change any plan); never edit docs/BUILD_BRIEF.md; never invent or hand-type a result number (every reported number comes from a banked artifact and passes claimcheck); never send email, open issues or pull requests, post, or publish anything, and push only as section 10.1 step 11 allows; work autonomously and record assumptions in DECISIONS.md instead of asking questions. Keep `goal_turn: N` as the first line of STATUS.md, add 1 at the start of every turn, and end each turn by stating goal_turn and which gates pass. Or stop once goal_turn reaches 300.
   ```
7. **Check in** with `/goal` (status and the evaluator's latest reason), `head -2 STATUS.md`, or `git log --oneline`. To stop: `/goal clear`, then ask Claude to cancel the scheduled loop. When the 7-day loop expires, a bare `/loop 30m` restarts it from `.claude/loop.md`. If a session ends, start a new one in the repo and paste the same two messages; the goal text resumes from the existing state instead of starting over.

**If your Claude Code surface does not support `/goal` or `/loop`**, paste this single message instead:

```text
Read docs/BUILD_BRIEF.md in full (README.md if that file does not exist yet) and execute it as your build brief. If docs/BUILD_BRIEF.md does not exist, start with the Bootstrap in section 10.1. Then repeat the Loop Protocol in section 10.2 without stopping to ask questions (record assumptions in DECISIONS.md), finish with the final sequence in section 10.6, and keep climbing the Stretch Ladder in section 7. Follow every rule in section 9 and the $0 budget in section 8.
```

### 0.1 For Claude: if you were started without the two messages above

If Raj only asked you to "do the README" or "complete the prompt in the README", do not stop to ask. (a) If you have a scheduling tool that delivers its prompt back into this same session (for example CronCreate), schedule a recurring task every 30 minutes whose prompt is the message-1 text above without the leading `/loop 30m`. Never use a scheduler that starts a separate session per firing: two sessions must never work in one repository at once. (b) Then follow the fallback message exactly. (c) End your first reply with one line telling Raj he can paste the message-2 `/goal` line from section 0 to have an independent evaluator keep you going until the Definition of Done is proven.

## 1. Mission

Agents now run for hours. Users set rules early ("never edit tests/", "no new dependencies", "keep the public API stable", "no network calls"), and the agent may compact its context several times before a rule matters. Whether those rules survive is a reliability question and a safety question, and today the evidence is anecdotal or observational.

Build **half-life**:

- **half-life-bench**: long, realistic, multi-stage coding sessions with constraints stated up front and later decision points where breaking a constraint is the easiest path, with deterministic violation detection. Every decision point also has a reasonable compliant route.
- A harness that controls session length, constraint placement and phrasing, and compaction (strategy and timing), for a controlled minimal agent and for real agent CLIs.
- The first survival-analysis estimates of **constraint half-life** (tool calls until violation becomes as likely as not) and of the causal effect of compaction.
- Mitigations that work in real tools: pinned memory files, constraint-preserving compaction, re-injection after compaction, and policy hooks, each with measured effect and cost.

**Quality bar.** Teams building long-running agents (Claude Code, Codex, background agents) should see this as the reference measurement of constraint persistence and adopt at least one of the mitigations. Hit the signals frontier teams say they value: measurement science, product impact on real model behavior, clean engineering, and honest reporting.

## 2. Why now, and what to beat

Read and position against these. Verify each before citing and drop any you cannot open.

- "How Coding Agents Fail Their Users" (2026): constraint violations are among the most common failures in real sessions and context loss is one coded cause, but the study is observational. https://arxiv.org/abs/2605.29442
- PushBench (2026): long multi-artifact tasks where external state tracking helps; it does not study prohibitions or compaction. https://www.alphaxiv.org/abs/2605.23574
- "Evaluating Goal Drift in Language Model Agents" (2025), which predates production compaction: https://arxiv.org/abs/2505.02709
- "Lost in the Middle" (2023), position effects in long contexts: https://arxiv.org/abs/2307.03172
- Claude Code documentation on memory and hooks, which states that the project CLAUDE.md is re-read after compaction. That is a testable claim about a mitigation. https://code.claude.com/docs/en/memory and https://code.claude.com/docs/en/hooks

**The gap.** A controlled, randomized measurement of how constraint adherence decays with session length and with compaction, plus a head-to-head test of mitigations inside real tools.

## 3. Research questions

- **RQ1 Hazard.** The probability of a violation at a decision point as a function of tool calls since the constraint was stated (a discrete-time hazard per opportunity, with decision-point positions randomized), and the half-life per agent, model, and constraint type: the position where that probability reaches 0.5, reported as "not reached within N tool calls" when it does not. Kaplan-Meier curves of time to first violation are descriptive only, because they depend on how often decision points occur.
- **RQ2 Compaction.** The causal effect of a compaction event (randomized timing, fixed strategy) on violation hazard and on constraint recall.
- **RQ3 Placement and phrasing.** System prompt versus first user message versus pinned memory file versus periodic reminder; bare rule versus rule plus reason.
- **RQ4 Constraint type.** Prohibitions, invariants, procedures, and resource limits: which decay fastest?
- **RQ5 Mitigations.** Constraint-preserving compaction summaries, re-injection after compaction, a PreToolUse policy guard (hard enforcement), and periodic self-checks: effects on violations, task success, tokens, and latency.

Write formal hypotheses with frozen thresholds into `prereg/` before confirmatory runs. Do not presume directions: "constraints persist well" is a valid and publishable finding.

## 4. What to build

Package `halflife` (Python 3.11+, src layout, uv, MIT):

- **`sessions`**: a generator for long sessions: a repository plus an ordered list of 20 to 60 subtasks, constraints declared up front, and decision points inserted at controlled positions where the fastest route breaks a constraint (for example a failing test whose quickest fix is editing the test, a task that tempts adding a dependency, or a refactor that tempts changing a public signature). Every decision point has a reasonable compliant route.
- **`detect`**: deterministic violation detectors over the action log: paths touched (from filesystem hashes taken after every tool call, not from parsing commands), commands run, package manifests, public API changes via AST diff, and network attempts from sandbox logs.
- **`compaction`**: pluggable strategies for the minimal agent (full summary, sliding window, structured state summary, constraint-pinned summary) with exact logs of what was kept, forced compaction at randomized points, and a sham control in which compaction provably keeps the constraint text verbatim.
- **`probe`**: constraint-recall probes that never contaminate the main run. Fork the exact context at checkpoints, ask the fork to list its operating rules, and score the answer against the declared constraints.
- **Drivers**: `minimal-agent` for local open-weight models with the scaffold held constant; headless Claude Code (`claude -p`, stream-json), using its real auto-compaction with the smallest supported window (check current docs for the setting); and an optional Gemini CLI free-tier driver.
- **`stats`**: discrete-time hazard models per decision point (with compaction count and task effects), half-life estimates with bootstrap intervals, descriptive Kaplan-Meier curves, and preregistered contrasts.
- **Mitigation kit**: `halflife-guard` (a Claude Code PreToolUse hook that enforces a YAML constraint spec), `halflife-pin` (CLAUDE.md and AGENTS.md templates), a constraint-preserving compaction prompt, and a re-injection hook. Check the current Claude Code hook events related to compaction before building it.

## 5. Experiments

- **E0 Instrument validity.** Detectors on seeded action logs (precision and recall), verified compaction logs, and a check that probes never leak into the main run. Seeded logs include indirect violations such as `sed -i` on a protected file, scripted writes, `git checkout --`, and `cp`.
- **E1 Controlled decay study (confirmatory).** minimal-agent with at least 2 open-weight model families, randomized compaction timing and decision-point positions, and at least 3 constraint types. Sample sizes by power analysis under the section 8 caps. Sessions are capped at 150 tool calls with at least 3 decision points each, and a small context budget forces compaction. Before E1, pilot candidate families on 3 templates not used in E1 and keep only families that reach the first decision point in at least 80% of pilot sessions; record the choice in DECISIONS.md. Size E3 and E4 from the compute budget in docs/PLAN.md.
- **E2 Production agents (ecological arm).** Claude Code at two or more model tiers, plus any optional free-tier CLI, on the same session templates with smaller samples. Label this arm as ecological in every table. Each E2 session is one `claude -p` conversation continued with `--resume` once per subtask; it counts once against `claude_agent_runs_per_day`, and `claude_agent_max_turns` applies to each subtask message. The CLI records the system prompt on the first request and reuses it on every resume until compaction (see `--system-prompt-snapshot` in `claude --help`), so pass identical system-prompt text on every launch, deliver reminders and re-injection as user-message text or hooks, and log the system prompt actually in effect.
- **E3 Placement and phrasing.**
- **E4 Mitigation study** on held-out session templates.

## 6. Definition of Done (ship gates)

`make ship-check` must print PASS for all 14 gates, checked on a fresh clone.

- **G1 Build and tests.** `make setup` succeeds on a fresh clone and `make test` passes. Line coverage is at least 85% on core modules. Adapters that need external services may be excluded from coverage only if recorded-fixture contract tests cover them.
- **G2 Static quality.** `ruff check`, `ruff format --check`, and `mypy --strict` on the package pass with zero errors. Every `type: ignore` carries a reason.
- **G3 CI parity.** `.github/workflows/ci.yml` runs `make ci` (lint, typecheck, test, offline demo) on Python 3.11 and 3.12, and `make ci` passes locally.
- **G4 Offline demo.** `make demo` finishes in under 3 minutes with no network and no model calls (recorded fixtures only) and matches its committed golden output.
- **G5 Instrument validity.** Detector precision and recall are both at least 0.95 on the seeded logs, with intervals printed. Probe leakage is zero across validation runs. Compaction logs are complete.
- **G6 Benchmark complete.** half-life-bench has at least 24 session templates spanning at least 4 constraint types, each with a verified compliant route and a reference run.
- **G7 Main study.** E1 ran as preregistered with at least 40 sessions per model family that reached their first decision point. Per-opportunity hazard and half-life estimates (or "not reached within N tool calls") are banked with intervals, and Kaplan-Meier curves are banked as descriptive.
- **G8 Ecological and placement arms.** E2 and E3 ran and are reported, including null results.
- **G9 Mitigations shipped.** E4 ran. halflife-guard and halflife-pin are packaged with install docs and measured overhead.
- **G10 Claim integrity.** `make claimcheck` and `make freshness` pass on README.md and docs/REPORT.md.
- **G11 Preregistration integrity.** `make prereg-verify` passes: every confirmatory artifact cites a preregistration whose commit predates the artifact's first commit, and no preregistration changed after it was frozen.
- **G12 Independent review.** The latest `make review` ends with `REVIEW: SHIP` for the current code, with zero open blocking issues. Non-blocking issues are listed in docs/REVIEW_LOG.md.
- **G13 Documentation.** README.md follows section 11, docs/REPORT.md is complete, and `make citecheck` shows every reference verified.
- **G14 Hygiene.** `make secrets-scan` is clean. `make budget` shows `cash_usd = 0` and every cap respected. Nothing under private/ is tracked. LICENSE, CITATION.cff, SECURITY.md, and CONTRIBUTING.md exist. The original checkout's working tree is clean.

## 7. Stretch Ladder (after the Definition of Done, one rung per iteration)

1. Multi-day sessions under real goal-driven autonomy, with constraints checked across restarts and resumes.
2. A portable constraint spec (YAML) with converters to CLAUDE.md, AGENTS.md, and other agents' settings.
3. Agent memory features (memory files, auto memory) as carriers of constraints.
4. Larger open-weight models if the hardware allows.
5. An interactive explorer of the survival curves (static HTML).
6. A workshop-paper version of docs/REPORT.md for Raj to submit.
7. Last rung: when every rung is done or listed under "Blocked", set `STATE: LADDER COMPLETE` in STATUS.md and stop.

## 8. Budget and compute

**Cash budget: $0. This is absolute.** Never call a paid API, add a payment method, buy credits, upgrade a plan, or rent paid compute. If any call returns a billing, credit, or payment error, stop that path, log it, and continue with other work.

Allowed model backends, in order of preference:

1. **Local open-weight models** through Ollama, llama.cpp, or transformers. Pin the model, revision, quantization, and file hash. Respect the disk floor below.
2. **Claude through Raj's Claude Code plan.** As an experimental subject or a pipeline worker, call headless Claude Code: `claude -p --model <haiku|sonnet|opus> --output-format stream-json --verbose`, run in a disposable workspace outside this repo. Plan usage is shared with Raj's other work, so be frugal; for a handful of pipeline calls, Claude Code subagents are fine.
   - **Billing guard.** If `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_BASE_URL`, any `CLAUDE_CODE_USE_*` variable, or an `apiKeyHelper` setting is present, the client wrapper refuses to launch Claude children and adds the problem to "Needs Raj". Never use `--bare`: it ignores the Max login and authenticates only with an API key. The `total_cost_usd` field in stream-json output is an API-price estimate, not cash: log it as `plan_equivalent_usd`, never add it to `cash_usd`, and never treat `--max-budget-usd` as the $0 control.
   - **Nesting.** If a child refuses to start or hangs because it detects the parent session, launch it with the parent-session markers removed from its environment only (for example `env -u CLAUDECODE -u CLAUDE_CODE_ENTRYPOINT claude -p ...`), the usual pattern for scripted automation. If it still fails, use subagents for small pipeline jobs and local models for subject runs, and add the limitation to "Needs Raj".
3. **Optional free tiers, only if `FREE_TIER_KEYS_CONFIRMED=1` is set:** the Gemini API free tier or Gemini CLI with a personal login, OpenRouter models whose IDs end in `:free` (check that the models endpoint lists a zero price before every batch), and the GitHub Models free tier. Obey each provider's rate limits and terms. Never send anything under `private/` to a free-tier provider.

Caps live in `budget.toml`, created at bootstrap from the defaults below. Claude may lower a cap and must never raise one. If a cap blocks the Definition of Done, add it to STATUS.md under "Needs Raj" and keep working on everything else.

```toml
cash_usd = 0                        # absolute; never change
claude_agent_runs_per_day = 30      # multi-turn headless runs used as test subjects
claude_agent_max_turns = 40         # per headless run (or per subtask message in resumed sessions)
claude_single_calls_per_day = 300   # short single-turn calls (labeling, judging, extraction)
claude_review_runs_per_day = 6      # make review
local_model_max_params_b = 8        # make doctor may lower this to fit RAM
disk_floor_gb = 8                   # never download if free disk would fall below this
```

If `CAP_SHARE` is set below 1 (Raj sets it when two projects run at once), the wrapper multiplies every `claude_*` cap by it. docs/PLAN.md schedules the Claude subject arms against the caps and starts them as early as the preregistration allows.

Usage hygiene:

- Every model call goes through one client wrapper that enforces the caps, logs tokens and wall time to `artifacts/usage/ledger.jsonl`, and caches responses by content hash. A banked call is never re-run.
- Prefer haiku for bulk work, sonnet for main arms, and opus for small confirmatory slices.
- When a child run reports a usage limit, stop launching model runs, record the reset time in STATUS.md, commit, and end the turn; the heartbeat resumes after the reset. Runs longer than a few minutes go in the background with a PID file and a log under `runs/<id>/` (git-ignored), and at most one model-heavy run is active at a time.
- `make doctor` detects Docker, GPU, RAM, free disk, the `claude` CLI and its login, Ollama, and any confirmed free-tier keys. Choose backends from its output and record them in STATUS.md.
- **Sandboxing.** Subject agents, minimal-agent tool calls, and third-party code (graders, task repositories, benchmark scripts) never run directly on the host. Use Docker with `--network none` when available. Otherwise use an OS sandbox that blocks network and allows writes only inside a temporary workspace (bubblewrap on Linux, sandbox-exec on macOS, or Claude Code's built-in Bash sandbox for Claude subjects; check current docs), with `GH_TOKEN`, `GITHUB_TOKEN`, `SSH_AUTH_SOCK`, cloud credentials, and all API keys removed from the environment, and with `GIT_CONFIG_NOSYSTEM=1` and `GIT_CONFIG_GLOBAL=/dev/null`. If no sandbox works, run only this project's own generated tasks, record that in DECISIONS.md, and add it to "Needs Raj". Never use `bypassPermissions` outside a container.
- **Clean Claude children.** Launch every Claude child with `--restricted --strict-mcp-config`, an explicit `--tools` list, and the matching permission rules (for example `--permission-mode dontAsk` plus `--allowedTools` scoped to the sandbox), so Raj's settings files, the hooks and plugins they enable, and his MCP servers do not load. Pass any hooks an experiment needs through `--settings`. On a smoke run, confirm the child can run the task's commands and that none of Raj's personal configuration reaches it; if his user-level CLAUDE.md or memory still loads, record it as a known contaminant in DECISIONS.md and in run provenance, and never edit or delete Raj's files. Single calls use `--tools ""` and a short `--system-prompt`, batching 20 to 50 items per call. If no max-turns option works, enforce `claude_agent_max_turns` by counting turns in the stream-json output and stopping the child at the cap. Record the exact command line in each run's provenance.
- CPU first. Docker and GPU are optional and never required for the Definition of Done. Every Docker-dependent step needs a Docker-free fallback or a substitution recorded in DECISIONS.md before any data is collected.

## 9. Non-negotiables

These hold on every turn. Copy them verbatim into CLAUDE.md during bootstrap so they survive context compaction.

1. **No fabricated or hand-typed numbers.** Every result number (a measured rate, count of outcomes, estimate, interval, or test statistic) in README.md, docs/REPORT.md, other results documents, figures, and tables is produced by a committed script from a banked artifact under `artifacts/`. Design parameters, thresholds, caps, dates, versions, and the text of docs/BUILD_BRIEF.md are not result numbers. `make claimcheck` enforces this and `make freshness` re-derives every live number.
2. **Preregister before confirmatory runs.** Hypotheses, metrics, analysis plan, thresholds, sample sizes, exclusions, and stopping rules go in `prereg/NNNN-*.md`, committed and hashed before any confirmatory data exists. A needed change is a new, dated amendment file that cites the original, is committed before any data it affects, and is reported as a deviation; the original is never edited. Anything not preregistered is labeled exploratory.
3. **Publish failures beside successes.** Null and negative results get the same prominence as positive ones. Every changed or withdrawn claim gets a dated entry in CORRECTIONS.md.
4. **Label synthetic and mock data everywhere.** Synthetic, procedurally generated, or mock results are never presented as model results or real-world results.
5. **$0 cash, with caps enforced in code** (section 8).
6. **Secrets stay in the environment.** Never write keys or tokens to files, logs, artifacts, or command lines. `make secrets-scan` runs in CI and in ship-check.
7. **Defensive scope only.** Mock tools and disposable sandboxes. No real credentials, real systems, or harmful payloads. Build nothing that helps an agent evade graders, monitors, or safety measures beyond what the measurement strictly requires.
8. **No outbound actions.** Never send email, open issues or pull requests, post, publish packages, datasets, or sites, or contact anyone. Pushing is allowed only under section 10.1 step 11. Prepare drafts in `outbound/` for Raj to review and send.
9. **Respect licenses and terms.** Record the source, license, and retrieval date of all third-party data in docs/DATA.md. Do not redistribute what a license does not allow. Add a canary string to any dataset meant for release.
10. **The brief is immutable.** Never edit docs/BUILD_BRIEF.md. Record deviations, substitutions, and interpretations in DECISIONS.md, with a reason, before acting on them.
11. **Verified citations only.** Never cite from memory. Open and check every reference (title, authors, year, URL) once and bank what you saw in `artifacts/citecheck.json` (retrieval date, final URL, page title, content hash); `make citecheck` validates that file offline, so ship-check never needs the network. If a site blocks scripted access, use the arXiv, DOI, or another official copy, and never work around bot protection. If nothing opens, drop the reference and list it under "Needs Raj" for a manual check.
12. **Plain writing.** Precise, scoped claims. No hype words such as "revolutionary", "groundbreaking", or "state of the art". No em dashes anywhere (Raj's style rule).
13. **Honest progress.** Never pass a gate by weakening its check. If a gate cannot pass, say so in STATUS.md and keep working on what can.
14. **Benign decision points.** Every decision point is a realistic situation with a reasonable compliant route. Never instruct an agent to break its constraints, and never search for prompts that make agents drop them.
15. **Report costs beside benefits.** Every mitigation result includes its cost in tokens, latency, and task success.

## 10. Operating protocol

### 10.1 Bootstrap (first turn only)

1. Copy README.md verbatim to `docs/BUILD_BRIEF.md` and write its SHA-256 to `docs/BUILD_BRIEF.sha256`.
2. Write and run `make doctor`. Record the environment, the usable backends, and the resulting fallbacks in STATUS.md. `make doctor` marks the Claude backend unusable while any variable or setting named in the section 8 billing guard is present.
3. Create `CLAUDE.md` (under 150 lines): a one-paragraph mission, section 9 verbatim, the state files, the command list, and this line: "After any compaction or restart, re-read CLAUDE.md, STATUS.md, and sections 6 to 10 of docs/BUILD_BRIEF.md before acting."
4. Create `.claude/loop.md` containing the message-1 heartbeat text from section 0 without the leading `/loop 30m`, so a bare `/loop 30m` after the 7-day expiry behaves the same.
5. Create project subagents in `.claude/agents/`:
   - `reviewer.md`: an adversarial reviewer with fresh eyes. It tries to falsify every claim and checks statistics, leakage, preregistration adherence, reproducibility, and the gate implementations themselves (could a gate pass while the property it guards is false?). An issue is blocking only if it names a file, line, or artifact and shows one of: a gate that can pass while its property is false; a reported number that is wrong, unreproducible, or not from an artifact; a preregistration violation; a broken section 9 rule; or a defect that changes a reported result. Everything else is non-blocking and goes to docs/REVIEW_LOG.md. It reads the previous review and the fixes since, and re-raises a resolved issue only with new evidence. It ends with `REVIEW: SHIP` or `REVIEW: BLOCK` plus a numbered list of blocking issues, each with its evidence.
   - `citation-checker.md`: opens each reference, confirms title, authors, year, and URL, and banks what it saw in `artifacts/citecheck.json` (rule 11).
6. Scaffold: `pyproject.toml` (uv, Python 3.11+), `src/halflife/`, `tests/`, `Makefile`, `.github/workflows/ci.yml`, `.gitignore` (private/, runs/, .env*, caches, model weights), `LICENSE` (MIT), `CITATION.cff`, `SECURITY.md`, `CONTRIBUTING.md`, `budget.toml`, `STATUS.md`, `DECISIONS.md`, `CORRECTIONS.md`, `prereg/`, `artifacts/`, `outbound/`, `private/` (git-ignored), and `runs/` (git-ignored).
7. Make targets, all present from day one even if some fail at first: `setup doctor test lint typecheck ci demo prereg-verify claimcheck freshness citecheck secrets-scan budget review report reproduce ship-check`.
8. `make review` runs the reviewer through headless `claude -p --model opus` when available. Otherwise the `reviewer` subagent runs and writes `artifacts/review/latest.json` itself; the main session never edits that file, and `make review` validates and prints it. Either way the file records the commit that was reviewed, which may differ from HEAD only in `artifacts/review/`, `STATUS.md`, and `docs/LOG.md`. When the daily review cap is used up, work on other gates or end the turn; never wait in a loop.
9. Write `tools/ship_check.py`. It implements G1 to G14 exactly as section 6 states them. It first checks, in the original checkout, the parts of G14 about the working tree and private/ (and any private hash manifests), then makes a fresh clone of HEAD in a temporary directory and runs every other check there. It verifies banked artifacts (presence, provenance, hashes, preregistration links, thresholds) and recomputes statistics from them; it never runs an experiment or calls a model. It writes nothing inside the working tree; sub-command logs go to the temporary directory. Its stdout is exactly `HEAD <sha> fresh-clone`, then one line per gate, `G<n> <PASS|FAIL> <short evidence>`, then `SHIP-CHECK: PASS 14/14` or `SHIP-CHECK: FAIL <passed>/14`. With `--fast` it may skip slow steps, and its last line is then `SHIP-CHECK-FAST (does not count): <passed>/14`. Only a full run counts.
10. Write `docs/PLAN.md`: milestones mapped to gates, risks, environment-dependent fallbacks, and usage estimates.
11. Commit ("bootstrap"). Push only if `ALLOW_PUSH=1` is set, only to this repository's `origin`, and never force-push, push tags, or change repository visibility or settings.

### 10.2 Loop Protocol (every iteration)

1. **Orient.** Read STATUS.md. Run `python tools/ship_check.py --fast`.
2. **Choose one thing.** The highest-leverage next step: a failing gate on the critical path first; once all gates pass, the next Stretch Ladder rung in section 7. Write it under "Now" in STATUS.md.
3. **Build it test-first** in small commits, with types, docstrings, and tests for every public function.
4. **Verify.** `make lint typecheck test`, plus the specific check for what you built. For experiments: dry run, usage estimate, preregistration check, then the real run. Pilots and dry runs use tasks and seeds disjoint from the confirmatory sample, recorded in the preregistration, and are never pooled with confirmatory data.
5. **Bank** results as artifacts with provenance: git SHA, config hash, model IDs and revisions, seeds, timestamps, environment, exact command lines, and usage.
6. **Review** milestone-level changes with the `reviewer` subagent and fix every blocking issue.
7. **Record.** Update STATUS.md (gate table, what changed, next three actions, usage to date, "Needs Raj" items), plus DECISIONS.md and CORRECTIONS.md as needed. The first line of STATUS.md is `goal_turn: N` and the second is `STATE: ACTIVE`, `STATE: BLOCKED ON RAJ`, or `STATE: LADDER COMPLETE`. Keep STATUS.md under 200 lines and move history to docs/LOG.md.
8. **Commit** with a descriptive message, and push only under the rule in 10.1 step 11. Tag milestones locally: v0.1 (offline demo), v0.5 (first real results), v1.0 (Definition of Done, set during the final sequence in 10.6).
9. **Raise the bar.** At each milestone ask what would make a research engineer on an OpenAI or Anthropic evals, safety, or agents team say "we should use this", and add one concrete suggestion to the backlog in STATUS.md. Backlog items are suggestions for Raj, not Stretch Ladder rungs.

### 10.3 When stuck

After three failed attempts at the same problem, write what you tried under "Blocked" in STATUS.md, switch to the next highest-leverage item, and come back later. Never fake progress or quietly narrow a gate.

### 10.4 Subagents and parallelism

Use subagents for literature verification, wide code review, and data audits so the main context stays focused. A subagent never declares a gate passed. Only `make ship-check` does.

### 10.5 Reuse from Raj's existing repositories

You may clone these public repositories (same author) and port code with attribution instead of rewriting it:

- https://github.com/raj200501/raise-v1-public (Apache-2.0): the preregistration chain (`tools/prereg.py`), the outbound-copy number gate (`tools/claimcheck.py`), the freshness check (`tools/freshness.py`), and mutation tests for gates (`tests/mutation_test.py`).
- https://github.com/raj200501/armour (MIT): the trace schema, rule monitors, and policy packs for tool-using agents.
- https://github.com/raj200501/arc-ttt-public (MIT): Fencecheck (`tools/fencecheck.py`), which finds eval code that scores correct-but-fenced JSON as zero.

### 10.6 Final sequence

When every gate other than G12 and G13 passes in the fast check, do the following in one turn, in this order, with no other commits:

1. Make sure README.md follows section 11, update STATUS.md, and commit.
2. Run `make review`. On `REVIEW: SHIP`, commit only `artifacts/review/`. On `REVIEW: BLOCK`, fix the issues and start again at step 1 in a later turn.
3. Run the full `make ship-check`.
4. If it ends with `SHIP-CHECK: PASS 14/14`, create or move the local tag v1.0 to HEAD, then run `git rev-parse HEAD`, `git status`, and `git tag --points-at HEAD`.
5. Quote the ship-check stdout, the review's final line, and the three git outputs in your final message.

Never commit after step 3 in this turn. Record the outcome in STATUS.md at the start of the next iteration.

## 11. What the shipped README must look like

Once real results exist, replace this file with the real README (the brief already lives at docs/BUILD_BRIEF.md). G13 checks it.

- **First screen:** the project name, a one-sentence finding with its confidence interval, the headline figure (an SVG generated by `make report`), a small headline table, and a 60-second quickstart that runs offline.
- **Then:** what it is and is not; key results, each linked to its artifact and preregistration; how it works (one diagram); how to use it on your own benchmark or agent; how to reproduce every figure with one command; limitations and threats to validity; a responsible-disclosure note where relevant; verified related work; citation; license.
- **Tone:** plain and exact. No hype, no em dashes, and every number from an artifact.
- **Also ship `docs/REPORT.md`:** a technical report (abstract, introduction, related work, method, experiments, results with intervals, limitations, ethics and disclosure, reproducibility) of 4,000 to 8,000 words, with figures generated from artifacts.

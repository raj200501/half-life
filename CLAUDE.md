# CLAUDE.md

After any compaction or restart, re-read CLAUDE.md, STATUS.md, and sections 6 to 10 of docs/BUILD_BRIEF.md before acting.

## Mission

half-life measures how long AI coding agents keep the rules they are given. It builds half-life-bench (long multi-stage coding sessions with constraints stated up front and later decision points where breaking a constraint is the easiest path, each with a reasonable compliant route), a harness that controls session length, constraint placement and phrasing, and context compaction, survival-analysis estimates of constraint half-life and of the causal effect of compaction, and mitigations for real tools (pinned memory files, constraint-preserving compaction, re-injection after compaction, and a PreToolUse policy guard), each with measured effect and cost. The build brief at docs/BUILD_BRIEF.md is immutable (hash in docs/BUILD_BRIEF.sha256); the Definition of Done is its section 6, checked only by `make ship-check`.

## Non-negotiables (section 9 of the brief, verbatim)

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

## State files

- `STATUS.md`: line 1 `goal_turn: N`, line 2 `STATE: ACTIVE | BLOCKED ON RAJ | LADDER COMPLETE`; gate table, Now, next three actions, usage, Needs Raj, Blocked, backlog. Under 200 lines; history goes to `docs/LOG.md`.
- `DECISIONS.md`: every assumption, deviation, substitution, and interpretation, recorded before acting on it.
- `CORRECTIONS.md`: dated entry for every changed or withdrawn claim.
- `docs/PLAN.md`: milestones mapped to gates, risks, fallbacks, usage estimates.
- `docs/ARTIFACTS.md`: the artifact layouts that `tools/ship_check.py` verifies.
- `prereg/NNNN-*.md`: preregistrations, frozen once committed; changes are new amendment files.
- `artifacts/`: banked results with provenance; `artifacts/usage/ledger.jsonl` logs every model call.
- `budget.toml`: caps; lower only, never raise. `private/` and `runs/` are git-ignored.
- `.claude/loop.md`: the heartbeat prompt. `.claude/agents/`: `reviewer`, `citation-checker`.

## Commands

| Command | What it does |
|---|---|
| `make setup` | `uv sync --frozen` into `.venv` |
| `make doctor` | report Docker, GPU, RAM, disk, claude CLI and billing guard, local runtimes, sandboxes |
| `make test` | pytest with coverage (fails under 85%) |
| `make lint` / `make typecheck` | ruff check and format check / mypy --strict on `src/halflife` |
| `make ci` | lint, typecheck, test, offline demo (what CI runs on 3.11 and 3.12) |
| `make demo` | offline demo from recorded fixtures, compared to `demo/golden_output.txt` |
| `make prereg-verify` | preregistration chain check |
| `make claimcheck` / `make freshness` | every result number traced to an artifact / re-derived |
| `make citecheck` | validate `artifacts/citecheck.json` offline |
| `make secrets-scan` | scan tracked files for secrets |
| `make budget` | `cash_usd = 0` and every cap respected |
| `make review` | validate and print `artifacts/review/latest.json` (written by the reviewer) |
| `make report` / `make reproduce` | figures and tables from artifacts / rebuild them all |
| `make ship-check` | gates G1 to G14 on a fresh clone; `python tools/ship_check.py --fast` does not count |

## Working rules

- One Loop Protocol iteration (brief section 10.2) per heartbeat: orient with the fast ship check, pick one thing, build it test-first, verify, bank, record, commit.
- At most one model-heavy run at a time, in the background, with a PID file and log under `runs/<id>/`.
- The Claude backend is refused while any billing-guard variable or `apiKeyHelper` is present (`make doctor`).
- Push every commit to this session's branch (`ALLOW_PUSH=1`, DECISIONS.md D-009); never tags, never force-push, never pull requests.
- `cap_share = 0.2` in budget.toml scales every `claude_*` cap (D-010). Experiments with local models need `make setup-local`.

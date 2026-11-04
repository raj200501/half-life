# Artifact layouts checked by the ship check

`tools/ship_check.py` verifies banked artifacts against the layouts below and recomputes what it can from them. It never runs an experiment or calls a model. These layouts may gain fields and checks over time; a change that would let a gate pass more easily needs a DECISIONS.md entry and a reviewer pass first (rule 13: never pass a gate by weakening its check).

All paths are relative to the repository root. JSON files are UTF-8. Every artifact directory also carries provenance (git SHA, config hash, model IDs and revisions, seeds, timestamps, environment, exact command lines, usage) as section 10.2 step 5 requires.

## G1 coverage

- Line coverage = `covered_lines / num_statements` from the pytest-cov JSON report, over every module under `src/halflife/`.
- A module may appear in `[tool.coverage.run] omit` only as an explicit path (`src/halflife/<...>.py`) and only if `tests/contract/test_<stem>.py` and a non-empty `tests/fixtures/<stem>/` exist.

## G4 demo

- `make demo DEMO_OUT=<path>` writes the demo output to `<path>`; the committed golden copy is `demo/golden_output.txt` and must match byte for byte.
- The ship check runs it inside a network namespace (`unshare -rn`) when available, with `UV_OFFLINE=1`, `HALFLIFE_OFFLINE=1`, `HALFLIFE_NO_MODEL_CALLS=1`, and proxies pointed at a closed port. It must finish in under 180 s and leave tracked files unchanged.

## G5 instrument validity (`artifacts/e0/`)

- `detector_validation.json`
  - `cases`: list of `{case_id, constraint_id, kind, label, predicted}`; `label` and `predicted` are booleans (violation or not). Seeded violations must include the kinds `sed_inplace`, `scripted_write`, `git_checkout`, and `cp`.
  - `precision` and `recall`: `{value, lo, hi}` with Wilson 95% intervals. The ship check recomputes all six numbers from `cases` (tolerance 1e-6) and requires both values to be at least 0.95.
- `probe_leakage.json`: `{validation_runs, leaks, runs: [{run_id, leaked}]}`. Counts must match `runs`, `runs` must be non-empty, and `leaks` must be 0.
- `compaction_logs.json`: `{events: [...]}`, non-empty. Each event has `run_id`, `event_id`, `strategy`, `tool_call_index`, `tokens_before`, `tokens_after`, `log_path`, `log_sha256`, and the file at `log_path` exists with that SHA-256.

## G6 benchmark (`artifacts/bench/`)

- `manifest.json`: `{templates: [{id, path, sha256, constraint_types, compliant_route, reference_run}]}`.
  - At least 24 unique ids. `path` exists with the given SHA-256.
  - `constraint_types` is a non-empty subset of `prohibition`, `invariant`, `procedure`, `resource_limit`; at least 4 distinct types across templates.
  - `compliant_route` points to `{template_id, template_sha256, tasks_total, tasks_passed, violations}` with `tasks_passed == tasks_total > 0` and `violations == 0`.
  - `reference_run` points to `{template_id, template_sha256, completed: true, ...}`.

## G7 main study (`artifacts/e1/`)

- `provenance.json`: includes `prereg`, a path under `prereg/` whose first commit is a proper ancestor of the first commit of `artifacts/e1/sessions.jsonl`.
- `sessions.jsonl`: one row per session, `{session_id, family, model, prereg, confirmatory: true, reached_first_dp}`. At least 2 families (section 5, E1) and at least 40 sessions per family with `reached_first_dp: true`.
- `opportunities.jsonl`: one row per decision point reached, `{session_id, family, constraint_type, position, compaction_count, violated}`; every `session_id` exists in `sessions.jsonl`.
- `hazard.json`: `{estimates: [{family, ..., estimate, lo, hi}]}` covering every family.
- `half_life.json`: `{estimates: [...]}` covering every family; each estimate is one of: finite `{half_life, lo, hi}` with `lo <= half_life <= hi`; an open upper bound `{half_life, lo, hi: null, hi_not_reached_within: N}` with `lo <= half_life <= N` (some bootstrap draws never reached 0.5 within N tool calls; D-028); or not reached `{half_life: null, not_reached_within: N}`.
- `km.json`: `{descriptive: true, curves: [...]}`.

## G8 ecological and placement arms

- `artifacts/e2/summary.json`: `{arm: "E2", ran: true, n_sessions > 0, label: "ecological", models: [at least 2 tiers]}`.
- `artifacts/e3/summary.json`: `{arm: "E3", ran: true, n_sessions > 0}`.
- `docs/REPORT.md` mentions `E2`, `E3`, and `ecological`.

## G9 mitigations

- `artifacts/e4/summary.json`: `{arm: "E4", ran: true, n_sessions > 0, held_out_templates: [...], mitigations: [{name, ..., cost: {tokens, latency_s, task_success}}]}`.
- `artifacts/e4/overhead.json`: `{"halflife-guard": {latency_ms, tokens}, "halflife-pin": {latency_ms, tokens}}`.
- `pyproject.toml` declares console scripts `halflife-guard` and `halflife-pin`; the full check builds the wheel and confirms the entry points.
- `docs/MITIGATIONS.md` has an install heading and covers both tools.

## G12 review (`artifacts/review/latest.json`)

`{reviewed_commit, reviewer, timestamp, verdict, blocking, non_blocking, final_line}` as described in `tools/review.py`. The ship check requires `verdict: SHIP`, `final_line: "REVIEW: SHIP"`, an empty `blocking` list, `docs/REVIEW_LOG.md`, and that `reviewed_commit` is HEAD or an ancestor that differs from HEAD only in `artifacts/review/`, `STATUS.md`, and `docs/LOG.md`.

## G14 private manifest (optional)

`artifacts/private_hashes.json`: `{files: [{path, sha256}]}` for files under `private/` that results depend on. The ship check verifies each in the original checkout.

## Usage ledger

`artifacts/usage/ledger.jsonl`: one row per model call or run, with `ts`, `kind` (`claude_agent_run`, `claude_single_call`, `claude_review_run`, `local`, `free_tier`), `cash_usd` (always 0), `plan_equivalent_usd`, tokens, wall time, and for agent runs `turns` (most turns used by one message). `make budget` checks it against `budget.toml`.

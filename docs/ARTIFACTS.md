# Artifact layouts checked by the ship check

`tools/ship_check.py` verifies banked artifacts against the layouts below and recomputes what it can from them. It never runs an experiment or calls a model. These layouts may gain fields and checks over time; a change that would let a gate pass more easily needs a DECISIONS.md entry and a reviewer pass first (rule 13: never pass a gate by weakening its check).

All paths are relative to the repository root. JSON files are UTF-8. Every artifact directory also carries provenance (git SHA, config hash, model IDs and revisions, seeds, timestamps, environment, exact command lines, usage) as section 10.2 step 5 requires.

Rules that hold for every layout below:

- **Paths.** A path that an artifact names (a compaction log, the experiment directory compaction logs came from, a template, a compliant route, a reference run, a reference run's action log) is a clean repository-relative POSIX path: no leading `/`, no empty, `.`, or `..` segment, no backslash, and the file it names, with symlinks followed, lies inside the clone. Anything else fails, so an artifact cannot point at a file that is not banked.
- **Counts and flags.** A count is an integer that is not a boolean; a flag is a JSON boolean (`null` and `"false"` are not booleans); a measured number is finite and not a boolean.
- **No padding.** Rows are unique by their identity (named per layout), so a repeated row can never raise a count.
- **Templates are read from their files.** G5, G6, and G9 load template files with the clone's own `halflife.bench` (a `uv run --frozen` sub-command in the clone, after `make setup`), so a template's id, split, constraint types, and constraint ids always come from the template file, never from a manifest.
- **Estimates are recomputed, not trusted.** A full check reruns the clone's own `halflife.e1_analysis` on the banked E1 rows and requires its output to equal the banked estimates (G7). The rerun works on copies in a temporary directory outside the clone, so it can never compare a banked file with itself.

## G1 coverage

- Line coverage = `covered_lines / num_statements` from the pytest-cov JSON report, over every module under `src/halflife/`.
- A module may appear in `[tool.coverage.run] omit` only as an explicit path (`src/halflife/<...>.py`) and only if `tests/contract/test_<stem>.py` and a non-empty `tests/fixtures/<stem>/` exist.

## G4 demo

- `make demo DEMO_OUT=<path>` writes the demo output to `<path>`; the committed golden copy is `demo/golden_output.txt` and must match byte for byte.
- The ship check runs it inside a network namespace (`unshare -rn`) when available, with `UV_OFFLINE=1`, `HALFLIFE_OFFLINE=1`, `HALFLIFE_NO_MODEL_CALLS=1`, and proxies pointed at a closed port. It must finish in under 180 s and leave tracked files unchanged.

## G5 instrument validity (`artifacts/e0/`)

- `detector_validation.json`
  - `cases`: list of `{case_id, constraint_id, kind, label, predicted}`; `case_id` is a non-empty string, unique across cases (a repeated case fails); `label` and `predicted` are booleans (violation or not). Seeded violations must include the kinds `sed_inplace`, `scripted_write`, `git_checkout`, and `cp`.
  - `precision` and `recall`: `{value, lo, hi}` with Wilson 95% intervals. The ship check recomputes all six numbers from `cases` (tolerance 1e-6) and requires both values to be at least 0.95.
- `probe_leakage.json`: `{validation_runs, leaks, runs: [{run_id, session_id, leaked}]}`. `validation_runs` and `leaks` are integers that match `runs`; `runs` is non-empty; every run has a non-empty `run_id`, a boolean `leaked` (null, meaning leakage was never checked, fails), and a unique `(run_id, session_id)` (`session_id` may be absent); `leaks` must be 0. `python -m halflife.experiment e0-validate` writes this layout and `compaction_logs.json`.
- `compaction_logs.json`: `{source, events: [...]}`, `events` non-empty; if it has an `incomplete` count, that count is 0. `source` is the repository-relative path of the banked experiment directory the validation runs came from, as `halflife.experiment.e0_validation` writes it.
  - Each event has `run_id`, `event_id`, `strategy`, `tool_call_index`, `tokens_before`, `tokens_after`, `log_path`, `log_sha256`; events are unique by `(run_id, session_id, event_id)`, and no two events share a log file.
  - `log_path` is a clean path inside the clone, and the file there exists with that SHA-256.
  - The log file is complete, checked from its content with the rules of `halflife.experiment.log_problems`: it is a JSON object with `event_id`, `strategy`, `tool_call_index`, `scheduled`, `tokens_before`, `tokens_after`, `before`, `after`, `summary`, and `constraint_text_retained`; `before` and `after` are non-empty message lists and `after` starts with the system message; `summary` is non-empty text unless `strategy` is `sliding_window`; `constraint_text_retained` is a non-empty map from constraint id to boolean. `event_id` and `strategy` are non-empty strings, the three counts are non-negative integers, and `scheduled` is a boolean.
  - Fields the event repeats from its log (`event_id`, `strategy`, `tool_call_index`, `tokens_before`, `tokens_after`, and, when present, `scheduled` and `constraint_text_retained`) equal the log's values.
  - An event that carries `complete` must have `complete: true` and no `problems`.
  - **Retention for every rule.** `source` is a clean path inside the clone to a directory whose `sessions.jsonl` (rows `{run_id, session_id, template_id, ...}`, as `halflife.experiment.bank_runs` writes them) also lies inside the clone. Every event names a non-empty `session_id`, and its `(run_id, session_id)` appears in that file with exactly one `template_id`. That template is listed in the bench manifest, its file loads with the clone's `halflife.bench`, and the log's `constraint_text_retained` has exactly the constraint ids the template file declares: a missing id fails, and so does an id the template does not declare. A missing or unclean `source`, a source without `sessions.jsonl`, and an unknown session all fail.

## G6 benchmark (`artifacts/bench/`)

- `manifest.json`: `{templates: [{id, path, sha256, constraint_types, compliant_route, reference_run}]}` (as `python -m halflife.bench manifest` writes it).
  - At least 24 unique ids. `path` is a clean path inside the clone and exists with the given SHA-256. Paths and SHA-256 values are each unique, so one template file cannot count twice.
  - The template file loads with `halflife.bench` and declares the entry's `id`.
  - `constraint_types` is a non-empty subset of `prohibition`, `invariant`, `procedure`, `resource_limit` and equals the set of constraint types in the template file. The at-least-4-distinct-types rule counts the types read from the template files.
  - `compliant_route` and `reference_run` are clean paths inside the clone that name two different records (different files with different content).
  - `compliant_route` points to `{template_id, template_sha256, tasks_total, tasks_passed, violations}` with integer counts, `tasks_passed == tasks_total > 0`, and `violations == 0`.
  - `reference_run` points to `{template_id, template_sha256, completed: true, action_log, action_log_sha256, ...}` (as `halflife.bench.verify.bank` writes it).
  - `action_log` is a clean path inside the clone (symlinks followed), and the file there has the SHA-256 `action_log_sha256` (64 lower-case hex digits). The file is a JSON object in the `halflife.schema.ActionLog` layout: `session_id` (a non-empty string), `template_id`, `constraints`, `baseline_fs`, `blobs`, `events`, `checks`, and `spans`, with the JSON types `ActionLog.to_dict` writes. Its `template_id` equals the manifest entry's `id`, so the run is a session of that template (two templates cannot share one log), and it records at least one tool call (`events`) and one subtask span (`spans`).
  - Counts the reference record repeats from its log, `n_tool_calls` and `n_subtasks`, when present, are integers equal to the number of `events` and `spans` in the log.

## G7 main study (`artifacts/e1/`)

- Every E1 file the gate reads (`sessions.jsonl`, `opportunities.jsonl`, `provenance.json`, `hazard.json`, `half_life.json`, and `km.json`), with symlinks followed, lies inside the clone.
- `provenance.json`: includes `prereg`, a regular file (not a symlink) at a `prereg/NNNN-slug.md` path (the naming rule of `make prereg-verify`, so `..` segments and `prereg/.gitkeep` do not count). Its first commit is a proper ancestor of the first commit of every E1 file the gate reads.
- `sessions.jsonl`: one row per session, `{session_id, family, model, prereg, confirmatory: true, reached_first_dp}`. Every family runs the same plans, so a row is identified by `(family, session_id)`, and that pair is unique. `session_id` and `family` are non-empty strings and `reached_first_dp` is a boolean. At least 2 families (section 5, E1) and at least 40 sessions per family with `reached_first_dp: true`.
- `opportunities.jsonl`: one row per decision point reached, `{session_id, family, constraint_type, position, compaction_count, violated}`; every `(family, session_id)` exists in `sessions.jsonl`. A session has `reached_first_dp: true` exactly when it has at least one opportunity row.
- `hazard.json`: `{estimates: [{family, ..., estimate, lo, hi}], settings: {n_boot, seed, ...}}` covering every family. `settings` is what `halflife.e1_analysis` records (D-029): `n_boot` a positive integer and `seed` an integer (booleans do not count).
- `half_life.json`: `{estimates: [...]}` covering every family; each estimate is one of: finite `{half_life, lo, hi}` with `lo <= half_life <= hi`; an open upper bound `{half_life, lo, hi: null, hi_not_reached_within: N}` with `lo <= half_life <= N` (some bootstrap draws never reached 0.5 within N tool calls; D-028); or not reached `{half_life: null, not_reached_within: N}`.
- `km.json`: `{descriptive: true, curves: [...]}`.
- **Recomputed in a full check.** The ship check copies `sessions.jsonl` and `opportunities.jsonl` into a temporary directory outside the clone and runs `python -m halflife.e1_analysis --dir <tmp> --n-boot <n_boot> --seed <seed>` in the clone's environment, with the `n_boot` and `seed` from `hazard.json`'s `settings`. The recomputed `hazard.json`, `half_life.json`, and `km.json` must equal the banked files as parsed JSON: the same keys, the same values, and the same JSON types (`20` and `20.0` differ), so a change of one unit in the last place fails. The evidence names the file and the first differing key path, for example `recomputed hazard.json differs from the banked file at estimates[0].hi` (object keys are visited in sorted order, array items by index). The analysis failing on the banked rows also fails the gate. A `--fast` check skips the rerun and its evidence ends with `not recomputed (fast)`; only a full check counts.

## G8 ecological and placement arms

- `artifacts/e2/summary.json`: `{arm: "E2", ran: true, n_sessions, label: "ecological", models}`, where `n_sessions` is a positive integer and `models` is a list of at least 2 distinct model tier names.
- `artifacts/e2/sessions.jsonl`: one row per E2 session, `{session_id, model, label: "ecological", ...}`. Rows are unique by `(session_id, model)` (one plan may run on each tier), the row count equals `n_sessions`, every row is labeled `ecological`, and the set of row `model` values equals the summary's `models`.
- `artifacts/e3/summary.json`: `{arm: "E3", ran: true, n_sessions}` with `n_sessions` a positive integer.
- `artifacts/e3/sessions.jsonl`: one row per E3 session with a non-empty `session_id`; rows unique by `(session_id, model)`, and the row count equals `n_sessions`.
- `docs/REPORT.md` mentions `E2`, `E3`, and `ecological` as whole tokens: not inside a longer word, number, or hyphenated compound (`E20`, `E3-like`, and `non-ecological` do not count; `E2's` and `**E3**` do).

## G9 mitigations

- `artifacts/e4/summary.json`: `{arm: "E4", ran: true, n_sessions, held_out_templates: [...], mitigations: [{name, ..., cost: {tokens, latency_s, task_success}}]}`.
  - `n_sessions` is a positive integer.
  - `mitigations` is a non-empty list, and every cost field (`tokens`, `latency_s`, `task_success`) is a finite, non-negative number (not a boolean or null), so each benefit has its cost beside it (rule 15).
  - `held_out_templates` is a non-empty list of distinct template ids. Each is a template in the bench manifest (`artifacts/bench/manifest.json`) whose file declares `split: heldout`, and none appears as a `template_id` in `artifacts/e1/sessions.jsonl` (every E1 row must carry a `template_id`, so the check can be made).
- `artifacts/e4/overhead.json`: `{"halflife-guard": {latency_ms, tokens}, "halflife-pin": {latency_ms, tokens}}`, every value a finite, non-negative number that is not a boolean.
- `pyproject.toml` declares console scripts `halflife-guard` and `halflife-pin`; the full check builds the wheel and confirms both under `[console_scripts]` in its `entry_points.txt` (another section, such as `[gui_scripts]`, does not count).
- `docs/MITIGATIONS.md` covers both tools and has an install heading: a Markdown heading (`#` to `######` and a space) outside code fences with a word that starts with `install` (`Install`, `Installation`); `Uninstall` and `Reinstall` do not count.

## G12 review (`artifacts/review/latest.json`)

`{reviewed_commit, reviewer, timestamp, verdict, blocking, non_blocking, final_line}` as described in `tools/review.py`. The ship check requires `verdict: SHIP`, `final_line: "REVIEW: SHIP"`, a `blocking` field that is a list and is empty (a missing field, null, or empty string fails), `docs/REVIEW_LOG.md`, and that `reviewed_commit` is HEAD or an ancestor that differs from HEAD only in `artifacts/review/`, `STATUS.md`, and `docs/LOG.md`.

## G13 documents (`README.md`, `docs/REPORT.md`)

Checked from the text; no artifact layout, but the rules are fixed here so writers know them.

- **Style (both files).** No em dash: the character U+2014 or an HTML entity that renders as one (`&mdash;`, `&#8212;`, `&#x2014;`, with any leading zeros). No hype phrases (`revolutionary`, `groundbreaking`, `state of the art`, `state-of-the-art`, any case), matched after collapsing whitespace, so a phrase split across lines (or after a hyphen at a line break) still counts.
- **README first screen** (everything before the "What it is" heading): the headline SVG, the headline table, the quickstart, and an interval for the finding. An interval is a percentage before `CI` or `interval` (`95% CI`, `90% credible interval`), the phrase `confidence interval` or `credible interval`, or a bracketed pair of numbers (`[0.12, 0.31]`, `(22 to 61)`) together with the word `CI` (upper case) or `interval`. The letters `ci` alone (as in `make ci`) do not count.
- **REPORT completeness.** 4,000 to 8,000 words, every section heading, every figure present, and no placeholder: `TODO`, `TBD`, `FIXME`, and `XXX` (upper case, as whole words, so the `todo-cli` template name is fine) or `pending`, `to be written`, and `coming soon` in any case.

## G14 private manifest (optional)

`artifacts/private_hashes.json`: `{files: [{path, sha256}]}` for files under `private/` that results depend on. The ship check verifies each in the original checkout.

## Usage ledger

`artifacts/usage/ledger.jsonl`: one row per model call or run, with `ts`, `kind` (`claude_agent_run`, `claude_single_call`, `claude_review_run`, `local`, `free_tier`), `cash_usd` (always 0), `plan_equivalent_usd`, tokens, wall time, and for agent runs `turns` (most turns used by one message). `make budget` checks it against `budget.toml`.

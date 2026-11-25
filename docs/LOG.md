# Log

History moved out of STATUS.md, newest first.

## 2026-11-11 to 2026-11-12: crunch-time notes (moved from STATUS.md on 2026-11-25)

- `tools/checkpoint_runs.py` now keeps each run's `run.json` and `invocations.jsonl` beside the session archives (refreshed on every checkpoint, never `progress.jsonl`) and restores them, so a lost machine does not lose what G7 checks for every invocation (review 5, non-blocking).
- E1 running (2026-11-12T02:31Z, from 05d4b2c, `src/` and `designs/` identical to the preregistration commit, sandbox pinned): `runs/e1/driver.sh` in the background, PID in `runs/e1/driver.pid`. Only `runs/e1/exits.log` (plan index, model, exit code) and the reach-only command are read while it runs; `runs/e1/full.log` holds the children's output with violation counts and is not read before the last planned session (amendment 2). Main updated to 05d4b2c under D-056 (RELEASE-READY).
- E1 preregistration frozen (2026-11-12, prereg/0001-e1.md at 084bed1) before any E1 session, with two same-day amendments (0002: a withdrawn masking sentence and a sensitivity refit; 0003: corrected counts and references, the fixed order of decision points with a refit, operating rules for no interim analysis and the sandbox). Statistics round two landed (a722586) and the calibration re-run from that clean commit reproduced `min_events` 7 (the margin is thin: H2b's highest per-family null rate is just under 7%, D-063). G7 now ties sessions to the pinned plans, reach rule, harness, models, settings, every invocation, and the sandbox (D-062, D-064, D-066); G10 fails on numbers in Mermaid diagrams (D-065). A pre-freeze audit workflow narrowed the claims; its remaining findings that need code or simulations are backlog (below).
- Review 4 (BLOCK at 526ae0d, B1 only; B2 and B3 held): `report --check` now renders README.md and docs/REPORT.md with cmark-gfm (new dev dependency `cmarkgfm`) and fails closed on any image file named in the raw text (D-057, CORRECTIONS.md).
- Crunch time (Raj, 2026-11-11): main holds the release commit (merge of pull request #1) and the repository is public. Order: finish D-050 and freeze the E1 preregistration; run E1 (G7); run E3 (G8; E2 stays for Raj's machine); run E4 (G9); a fresh review (G12). Main is updated at RELEASE-READY points under D-056. The heartbeat runs every 30 minutes in this session.
- Review 3 (BLOCK at af5ddc6) fixed: `report --check` now finds embedded images in every Markdown and HTML form (B1); every bench template carries the half-life-bench canary and the bench routes, reference runs, manifest, and probe-scoring hashes are re-banked with their scripts (B2, D-055); README.md no longer calls the analysis preregistered, plus four smaller wording fixes (B3, CORRECTIONS.md).
- E1 power analysis banked (`artifacts/power/e1.json`, synthetic, score tests): power is low for small effects at every size the compute allows, so the preregistration keeps 48 plans per family and states that a null result is inconclusive (report section 5.4).
- Statistics round two is in (206234d, D-058; `min_events` = 2 from the banked synthetic calibration) and under an adversarial audit before the preregistration freezes it. G7 gate pass, second part: a not-estimable estimate passes only with its reason and the fit diagnostics behind it, every family needs a descriptive violation-rate interval, the full check reruns only the preregistered code (`code_sha256` pins), and it now also recomputes `contrasts.json` (H2a, H3) and `descriptive.json`.
- D-050 gate pass, first part: G7 now requires every confirmatory session's `run_provenance` to show a clean checkout at or after the preregistration's commit (six mutation tests). `make ci` passes on Python 3.11 as well as 3.12 at 526ae0d.
- D-050 item 5, harness side: every span records how it ended (`done` or the cap that cut it) and opportunity rows carry `span_end`; the statistics items run in a subagent.
- Release sprint (2026-11-11, Raj): experiments on hold. Publish-safety pass: `make secrets-scan` clean, nothing under private/ tracked, no personal data beyond Raj's name and GitHub noreply email (the only other addresses are reserved example domains in tests and templates), the bench templates are original synthetic projects whose only outside names are reserved example domains and documentation links. README.md gains a Status section with the gate state. The third independent review was paused for the sprint and resumes after it.
- Pre-release pass (2026-11-11): the build brief in README.md is replaced by the real project README (section 11), generated in part from artifacts by `make report`; a pilot figure and violation counts join the report; a pre-release correction is logged (CORRECTIONS.md); G12 accepts the reviewer's own log after the reviewed commit (D-051).
- Banked so far: E0 instrument validity (detectors, probe scorer, probe leakage, compaction logs), the half-life-bench templates with routes and reference runs, the family pilot with its D-042 choice (SmolLM2 and Granite), and the overhead of halflife-guard and halflife-pin.
- Statistics: audit round 1 is committed (D-049). Round 2 (D-050: score tests for every hypothesis, family-stratified pooled tests, the marginal half-life, the cut-span exclusion, harness hashes and provenance, power that mirrors the tests) is decided; its first attempt was stopped and discarded unverified. Item 6 is partly in: compaction log hashes equal the written file's hash, every saved session carries a `run_provenance` stamp, and `arms_analysis` pins its modules (`--code-sha256`), and the E2 driver refuses to skip a finished session whose plan changed; the power simulation and the report decisions remain.

## 2026-11-12: statistics round two lands; the E1 preregistration is frozen

- Statistics round two and its audit landed (D-058 to D-061, D-063): H1 and H2b each in its own model, per-family hazard tests confirmatory and pooled ones exploratory, and `min_events` 7 from a synthetic calibration of the final models on the E1 plans (60,000 datasets per scenario; the margin is thin for H2b). Rows whose fitted null probability saturates now leave the score test's correlation.
- G7 now ties every E1 session to the preregistered plans and the reach rule (D-062) and to the preregistered harness, models, and run settings (D-064); G10 fails on numbers inside Mermaid diagrams (D-065).
- A pre-freeze adversarial audit (a workflow of independent lenses) narrowed what the preregistration claims (D-064): H1 and H2b are adjusted associations, H2a is the randomized test, H3 is exploratory, and the limitations are stated in advance.
- Froze prereg/0001-e1.md (084bed1) before any E1 session, and the same day amendment 1 withdrew one wrong limitation sentence and added an exploratory refit (prereg/0002-e1-amendment-1.md, CORRECTIONS.md). docs/E1_RUNBOOK.md gives the commands to run E1 on any machine that stays up for days.

## 2026-11-11: pre-release pass

- Replaced the build brief in README.md with the project README (section 11), its first screen generated from artifacts by `make report`; added the pilot figure and violation counts to the report; logged the first correction (CORRECTIONS.md).
- A read-only audit before going public found 37 issues in documentation and provenance; the high and medium ones are fixed (D-052). G12 now accepts the reviewer's own log after the reviewed commit (D-051).
- Two independent reviews returned BLOCK; their findings are fixed (D-053, D-054).
- Raj opened and merged pull request #1, which put the release commit on main (this session opens no pull requests, rule 8); from then on main is updated only at RELEASE-READY points under D-056.
- Release sprint: publish-safety pass (secrets, private/, personal data, third-party content in the templates) and a README Status section with the gate state; experiments on hold for the sprint.

## 2026-11-04: pilot, audits, and instruments

- Ran the family pilot (D-026, amended by D-042 before any further session): SmolLM2, Granite, and Qwen ran every pilot template; the rule picks SmolLM2 and Granite. Banked it with the E0 validation runs (no probe leaks; complete compaction logs).
- Banked the probe-scorer validation (D-038), the tool overhead, and the thirty-template bench.
- Committed the mitigation kit (D-039), the E2 Claude Code driver and runbook (D-044), the E3 and E4 analysis plans (D-040), and the report pipeline with claims and freshness (D-048).
- A statistics audit before any preregistration fixed two known and twenty-two further confirmed defects (D-049); round two of the decisions is recorded (D-050).
- Hardened the ship check three times (D-037, D-041, D-045, D-046).

## 2026-10-23: bootstrap

- Copied the build brief to docs/BUILD_BRIEF.md with its SHA-256 in docs/BUILD_BRIEF.sha256.
- Ran `make doctor`; recorded the environment in STATUS.md.
- Scaffolded the package, Makefile, CI, state files, subagents, and tools/ship_check.py.

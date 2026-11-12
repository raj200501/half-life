goal_turn: 4
STATE: ACTIVE

# Status

Updated 2026-11-12 (goal_turn 4). Crunch time: E1 preregistration frozen; final review and the merge to main next. History lives in docs/LOG.md.

## Gates (fast check at 05d4b2c, 2026-11-12: 10/14; fast results do not count)

| Gate | State | Evidence |
|---|---|---|
| G1 Build and tests | PASS | fresh-clone `make setup` and `make test`, coverage gate 85% |
| G2 Static quality | PASS | ruff check, ruff format, mypy --strict clean; every ignore has a reason |
| G3 CI parity | PASS (fast) | workflow runs `make ci` on Python 3.11 and 3.12 |
| G4 Offline demo | PASS | demo in a network namespace with no model calls, matches the golden output |
| G5 Instrument validity | PASS | detector precision and recall with intervals, probe leakage, complete compaction logs with per-rule retention |
| G6 Benchmark complete | PASS | thirty templates, four rule types, routes and reference runs with action logs verified |
| G7 Main study | FAIL | E1 is preregistered and frozen (prereg/0001-e1.md at 084bed1, two amendments the same day) and running since 2026-11-12T02:31Z from 05d4b2c (no outcome read before the last session); G7 ties every session to the pinned plans and reach rule, harness, models, settings, every invocation, and the sandbox (D-062, D-064, D-066) |
| G8 Ecological and placement arms | FAIL | E2 needs Raj (billing guard); E3 not run |
| G9 Mitigations shipped | FAIL | kit packaged with install docs and banked overhead; E4 not run |
| G10 Claim integrity | PASS | claimcheck, freshness, and report-check pass on README.md and docs/REPORT.md |
| G11 Preregistration integrity | PASS | `make prereg-verify`: prereg/0001-e1.md unchanged since its first commit, amendments 0002 and 0003 dated and in order; no confirmatory artifact yet |
| G12 Independent review | FAIL | review 5 (of 1f9a4c5) returned BLOCK on two findings, both fixed: G10's Mermaid check matched fences by hand (B1; it now reads the cmark-gfm rendering, D-067) and changed claims lacked CORRECTIONS.md entries (B2); the next review waits for the daily allowance (2026-11-13 UTC) |
| G13 Documentation | PASS | README follows section 11, REPORT complete, citecheck verified |
| G14 Hygiene | PASS | secrets clean, cash_usd = 0, caps respected, tree clean |

## Environment (`make doctor`, 2026-11-04)

- Cloud container, Linux x86_64, 4 CPUs, 15.7 GB RAM, no GPU, about 28 GB free disk (floor 8 GB).
- Docker CLI yes, daemon no. Harness sandbox methods: `unshare-mount` (no network, writes confined to the workspace) and `unshare-net`; no bubblewrap (D-019).
- `claude` CLI 2.1.318 present; Claude backend refused by the billing guard (`ANTHROPIC_BASE_URL`, `CLAUDE_CODE_USE_CCR_V2` set by the cloud harness).
- Local runtime: llama-cpp-python CPU wheel (`make setup-local`); Hugging Face GGUF downloads work through the proxy.
- Subject Python for sandboxed commands: `make subject-env` (Python 3.12, pytest only, no pip; D-025).
- Launch settings: commits authored as Raj (D-021), CI only on main and manual dispatch with `make ci` run on a fresh clone before every push (D-022), push every commit (D-009), `cap_share = 0.2` (D-010).

## Backends and fallbacks

- Local open-weight models through llama.cpp: usable; five families pinned by file hash (`artifacts/models/pins.json`), behind the capped client and usage ledger (`halflife.client`).
- Claude: unusable here (billing guard); `make review` uses the reviewer subagent; E2 needs Raj and docs/E2_RUNBOOK.md.

## Now

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

## Next three actions

1. Keep E1 running; every few hours checkpoint finished sessions (`tools/checkpoint_runs.py --out artifacts/e1/raw runs/e1-*`, plus each run's `run.json` and `invocations.jsonl`) with the usage ledger, commit, `tools/prepush_ci.sh`, push.
2. Review 6 (G12) when the daily allowance permits (2026-11-13 UTC), on the then-current commit.
3. Start E1 on a machine that stays up for several days (docs/E1_RUNBOOK.md; never `make setup` in that checkout while it runs) and checkpoint finished sessions every few hours; then the E3 and E4 preregistrations with plan pins in G8 and G9 (as D-062 and D-064 do for E1).

## Usage to date

- cash_usd = 0. Claude subject runs 0, single calls 0, review runs 0.
- Ledger rows: 584, all local llama.cpp calls (dry runs, the planning benchmark, fixture recording, and pilot sessions); no confirmatory data yet.
- Build-session subagents (builders, the D-038 labeling agent, the citation checker, the reviewer) run on the Max plan outside the harness and are not in the ledger.
- Claude plan usage: subagent builders hit the weekly limit at about 04:00 UTC on 2026-11-04 (reset 07:00 UTC); relaunched after the reset.

## Needs Raj

0. **E1 compute (blocks G7).** E1 needs several days of CPU on a machine that stays up. This cloud container can run it while the session lives, and finished sessions are checkpointed into the repository so a run resumes anywhere; if the container is reclaimed, run docs/E1_RUNBOOK.md on your machine.
1. **Claude backend (blocks E2 in G8, and headless review).** This cloud container sets `ANTHROPIC_BASE_URL` and `CLAUDE_CODE_USE_CCR_V2`, so the billing guard refuses Claude children. To run E2: on a Linux machine with bubblewrap and socat where `env | grep -E '^(ANTHROPIC_|CLAUDE_CODE_USE_)'` prints nothing and `claude` is logged in to the Max plan, turn off extra usage (paid overage) on the account, then follow docs/E2_RUNBOOK.md (smoke run, plans from `designs/e2.yaml`, two tiers, bank, push). Record any contaminant the smoke run reports (for example a user-level CLAUDE.md) so it can go in DECISIONS.md. E2 needs about 3 days at the capped 6 sessions per day.
2. **Private vulnerability reporting (not blocking).** To accept private security reports as SECURITY.md describes, turn on private vulnerability reporting in the repository's security settings.
3. **Model licenses (confirm, not blocking).** Llama-3.2-3B-Instruct is under the Llama 3.2 Community License (its acceptable-use policy file is gated and could not be read here) and Qwen2.5-3B-Instruct under the Qwen Research License (non-commercial research and evaluation). This project uses both only for research evaluation and redistributes no weights (D-035). If you do not accept either license, say so and that family's data is dropped and recorded in CORRECTIONS.md.

## Blocked

Nothing yet.

## Backlog (suggestions for Raj, not ladder rungs)

0. **Audit backlog (2026-11-12).** From the pre-freeze audit, needing code or simulations (each becomes an amendment or an exploratory analysis before E1 is analyzed): H2b counting only scheduled compactions; G7 re-deriving the banked rows from the raw archives; the half-life's observed-range reading in the report; H3 labeled exploratory in the report's tables; power on the final models (D-063); and G8 and G9 tied to their plans as D-062 ties E1.
1. Publish half-life-bench templates with a canary string and a stable YAML constraint spec so other teams can run their own agents against it. The templates carry no canary yet (rule 9 asks for one in any dataset meant for release). Adding one changes each template's file hash, which the E0, pilot, and reference-run artifacts pin, so it needs a recorded decision and re-banked references, and it should land before E1 so E1 pins the final template bytes.

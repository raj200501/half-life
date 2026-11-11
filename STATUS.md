goal_turn: 4
STATE: ACTIVE

# Status

Updated 2026-11-11 (goal_turn 4). Release sprint before the repository goes public. History lives in docs/LOG.md.

## Gates (fast check at 8c2bde7, 2026-11-11: 10/14; fast results do not count)

| Gate | State | Evidence |
|---|---|---|
| G1 Build and tests | PASS | fresh-clone `make setup` and `make test`, coverage gate 85% |
| G2 Static quality | PASS | ruff check, ruff format, mypy --strict clean; every ignore has a reason |
| G3 CI parity | PASS (fast) | workflow runs `make ci` on Python 3.11 and 3.12 |
| G4 Offline demo | PASS | demo in a network namespace with no model calls, matches the golden output |
| G5 Instrument validity | PASS | detector precision and recall with intervals, probe leakage, complete compaction logs with per-rule retention |
| G6 Benchmark complete | PASS | thirty templates, four rule types, routes and reference runs with action logs verified |
| G7 Main study | FAIL | E1 has not run: its preregistration waits on statistics round 2 (D-050) |
| G8 Ecological and placement arms | FAIL | E2 needs Raj (billing guard); E3 not run |
| G9 Mitigations shipped | FAIL | kit packaged with install docs and banked overhead; E4 not run |
| G10 Claim integrity | PASS | claimcheck, freshness, and report-check pass on README.md and docs/REPORT.md |
| G11 Preregistration integrity | PASS (trivially) | no preregistration committed yet |
| G12 Independent review | FAIL | review 4 (of 526ae0d) returned BLOCK on one finding, embedded-image forms the hand parser missed; fixed by rendering with cmark-gfm plus a fail-closed raw scan (D-057); a new review waits for the daily allowance |
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

- Review 4 (BLOCK at 526ae0d, B1 only; B2 and B3 held): `report --check` now renders README.md and docs/REPORT.md with cmark-gfm (new dev dependency `cmarkgfm`) and fails closed on any image file named in the raw text (D-057, CORRECTIONS.md).
- Crunch time (Raj, 2026-11-11): main holds the release commit (merge of pull request #1) and the repository is public. Order: finish D-050 and freeze the E1 preregistration; run E1 (G7); run E3 (G8; E2 stays for Raj's machine); run E4 (G9); a fresh review (G12). Main is updated at RELEASE-READY points under D-056. The heartbeat runs every 30 minutes in this session.
- Review 3 (BLOCK at af5ddc6) fixed: `report --check` now finds embedded images in every Markdown and HTML form (B1); every bench template carries the half-life-bench canary and the bench routes, reference runs, manifest, and probe-scoring hashes are re-banked with their scripts (B2, D-055); README.md no longer calls the analysis preregistered, plus four smaller wording fixes (B3, CORRECTIONS.md).
- D-050 gate pass, first part: G7 now requires every confirmatory session's `run_provenance` to show a clean checkout at or after the preregistration's commit (six mutation tests). `make ci` passes on Python 3.11 as well as 3.12 at 526ae0d.
- D-050 item 5, harness side: every span records how it ended (`done` or the cap that cut it) and opportunity rows carry `span_end`; the statistics items run in a subagent.
- Release sprint (2026-11-11, Raj): experiments on hold. Publish-safety pass: `make secrets-scan` clean, nothing under private/ tracked, no personal data beyond Raj's name and GitHub noreply email (the only other addresses are reserved example domains in tests and templates), the bench templates are original synthetic projects whose only outside names are reserved example domains and documentation links. README.md gains a Status section with the gate state. The third independent review was paused for the sprint and resumes after it.
- Pre-release pass (2026-11-11): the build brief in README.md is replaced by the real project README (section 11), generated in part from artifacts by `make report`; a pilot figure and violation counts join the report; a pre-release correction is logged (CORRECTIONS.md); G12 accepts the reviewer's own log after the reviewed commit (D-051).
- Banked so far: E0 instrument validity (detectors, probe scorer, probe leakage, compaction logs), the half-life-bench templates with routes and reference runs, the family pilot with its D-042 choice (SmolLM2 and Granite), and the overhead of halflife-guard and halflife-pin.
- Statistics: audit round 1 is committed (D-049). Round 2 (D-050: score tests for every hypothesis, family-stratified pooled tests, the marginal half-life, the cut-span exclusion, harness hashes and provenance, power that mirrors the tests) is decided; its first attempt was stopped and discarded unverified. Item 6 is partly in: compaction log hashes equal the written file's hash, every saved session carries a `run_provenance` stamp, and `arms_analysis` pins its modules (`--code-sha256`), and the E2 driver refuses to skip a finished session whose plan changed; the power simulation and the report decisions remain.

## Next three actions

1. Finish statistics round 2 (D-050 items 1 to 4 in a subagent; then the report decisions, the power simulation, the `min_events` calibration, and the rest of the gate pass), with its adversarial review.
2. Fill in and commit prereg/0001-e1.md (settings and code-pin block per D-046, sample size from the power run), generate the E1 plans, and start E1 for SmolLM2 and Granite with `tools/round_robin.py ... --prereg prereg/0001-e1.md` in the background (about two days on this CPU).
3. After E1: E3, then E4 on the held-out templates, one model-heavy run at a time; E2 on Raj's machine; a fresh review when the daily allowance permits.

## Usage to date

- cash_usd = 0. Claude subject runs 0, single calls 0, review runs 0.
- Ledger rows: 584, all local llama.cpp calls (dry runs, the planning benchmark, fixture recording, and pilot sessions); no confirmatory data yet.
- Build-session subagents (builders, the D-038 labeling agent, the citation checker, the reviewer) run on the Max plan outside the harness and are not in the ledger.
- Claude plan usage: subagent builders hit the weekly limit at about 04:00 UTC on 2026-11-04 (reset 07:00 UTC); relaunched after the reset.

## Needs Raj

1. **Claude backend (blocks E2 in G8, and headless review).** This cloud container sets `ANTHROPIC_BASE_URL` and `CLAUDE_CODE_USE_CCR_V2`, so the billing guard refuses Claude children. To run E2: on a Linux machine with bubblewrap and socat where `env | grep -E '^(ANTHROPIC_|CLAUDE_CODE_USE_)'` prints nothing and `claude` is logged in to the Max plan, turn off extra usage (paid overage) on the account, then follow docs/E2_RUNBOOK.md (smoke run, plans from `designs/e2.yaml`, two tiers, bank, push). Record any contaminant the smoke run reports (for example a user-level CLAUDE.md) so it can go in DECISIONS.md. E2 needs about 3 days at the capped 6 sessions per day.
2. **Private vulnerability reporting (not blocking).** To accept private security reports as SECURITY.md describes, turn on private vulnerability reporting in the repository's security settings.
3. **Model licenses (confirm, not blocking).** Llama-3.2-3B-Instruct is under the Llama 3.2 Community License (its acceptable-use policy file is gated and could not be read here) and Qwen2.5-3B-Instruct under the Qwen Research License (non-commercial research and evaluation). This project uses both only for research evaluation and redistributes no weights (D-035). If you do not accept either license, say so and that family's data is dropped and recorded in CORRECTIONS.md.

## Blocked

Nothing yet.

## Backlog (suggestions for Raj, not ladder rungs)

1. Publish half-life-bench templates with a canary string and a stable YAML constraint spec so other teams can run their own agents against it. The templates carry no canary yet (rule 9 asks for one in any dataset meant for release). Adding one changes each template's file hash, which the E0, pilot, and reference-run artifacts pin, so it needs a recorded decision and re-banked references, and it should land before E1 so E1 pins the final template bytes.

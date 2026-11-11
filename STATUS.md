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
| G12 Independent review | FAIL | the latest review (of 15c4319) returned BLOCK; its findings are fixed in 8c2bde7 (D-054); a new review of the release commit is pending |
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

- Release sprint (2026-11-11, Raj): experiments on hold. Publish-safety pass: `make secrets-scan` clean, nothing under private/ tracked, no personal data beyond Raj's name and GitHub noreply email (the only other addresses are reserved example domains in tests and templates), the bench templates are original synthetic projects whose only outside names are reserved example domains and documentation links. README.md gains a Status section with the gate state. The third independent review was paused for the sprint and resumes after it.
- Pre-release pass (2026-11-11): the build brief in README.md is replaced by the real project README (section 11), generated in part from artifacts by `make report`; a pilot figure and violation counts join the report; a pre-release correction is logged (CORRECTIONS.md); G12 accepts the reviewer's own log after the reviewed commit (D-051).
- Banked so far: E0 instrument validity (detectors, probe scorer, probe leakage, compaction logs), the half-life-bench templates with routes and reference runs, the family pilot with its D-042 choice (SmolLM2 and Granite), and the overhead of halflife-guard and halflife-pin.
- Statistics: audit round 1 is committed (D-049). Round 2 (D-050: score tests for every hypothesis, family-stratified pooled tests, the marginal half-life, the cut-span exclusion, harness hashes and provenance, power that mirrors the tests) is decided; its first attempt was stopped and discarded unverified. Item 6 is partly in: compaction log hashes equal the written file's hash, every saved session carries a `run_provenance` stamp, and `arms_analysis` pins its modules (`--code-sha256`); the E2 changed-plan refusal, the power simulation, and the report decisions remain.

## Next three actions

1. Resume the independent review on the release commit and fix what it finds until it records SHIP (G12).
2. Implement statistics round 2 (D-050) with its adversarial review, then run `python -m halflife.power --pilot artifacts/pilot --out artifacts/power/e1.json`; decide how the bench canary goes in before E1 (backlog item 1).
3. Fill in and commit prereg/0001-e1.md (settings and code-pin block per D-046), generate the E1 plans, and run E1 for SmolLM2 and Granite with `tools/round_robin.py ... --prereg prereg/0001-e1.md` (about two days on this CPU); while it runs, the gate pass D-050 lists, then E3 and E4 preregistrations and runs; E2 on Raj's machine.

## Usage to date

- cash_usd = 0. Claude subject runs 0, single calls 0, review runs 0.
- Ledger rows: 584, all local llama.cpp calls (dry runs, the planning benchmark, fixture recording, and pilot sessions); no confirmatory data yet.
- Build-session subagents (builders, the D-038 labeling agent, the citation checker, the reviewer) run on the Max plan outside the harness and are not in the ledger.
- Claude plan usage: subagent builders hit the weekly limit at about 04:00 UTC on 2026-11-04 (reset 07:00 UTC); relaunched after the reset.

## Needs Raj

1. **Claude backend (blocks E2 in G8, and headless review).** This cloud container sets `ANTHROPIC_BASE_URL` and `CLAUDE_CODE_USE_CCR_V2`, so the billing guard refuses Claude children. To run E2: on a Linux machine with bubblewrap and socat where `env | grep -E '^(ANTHROPIC_|CLAUDE_CODE_USE_)'` prints nothing and `claude` is logged in to the Max plan, turn off extra usage (paid overage) on the account, then follow docs/E2_RUNBOOK.md (smoke run, plans from `designs/e2.yaml`, two tiers, bank, push). Record any contaminant the smoke run reports (for example a user-level CLAUDE.md) so it can go in DECISIONS.md. E2 needs about 3 days at the capped 6 sessions per day.
2. **Before making the repository public: move `main` to this branch while the repository is still private.** `main` still holds only the first commit (the build brief), and GitHub shows the default branch to visitors. This branch is a fast-forward of `main`: from a machine with push rights run `git fetch origin && git push origin origin/claude/brave-albattani-c5c7e2:main`. Do not use Squash and merge or Rebase and merge: banked provenance cites these commit SHAs and `make prereg-verify` reads commit order, so `main` must contain the same commits; keep the branch until it does. The push starts the first GitHub Actions run of `make ci` on Python 3.11 and 3.12 (D-022); make the repository public after both jobs pass, and check that Settings shows `main` as the default branch. To accept private security reports as SECURITY.md describes, turn on private vulnerability reporting in the repository's security settings.
3. **Model licenses (confirm, not blocking).** Llama-3.2-3B-Instruct is under the Llama 3.2 Community License (its acceptable-use policy file is gated and could not be read here) and Qwen2.5-3B-Instruct under the Qwen Research License (non-commercial research and evaluation). This project uses both only for research evaluation and redistributes no weights (D-035). If you do not accept either license, say so and that family's data is dropped and recorded in CORRECTIONS.md.

## Blocked

Nothing yet.

## Backlog (suggestions for Raj, not ladder rungs)

1. Publish half-life-bench templates with a canary string and a stable YAML constraint spec so other teams can run their own agents against it. The templates carry no canary yet (rule 9 asks for one in any dataset meant for release). Adding one changes each template's file hash, which the E0, pilot, and reference-run artifacts pin, so it needs a recorded decision and re-banked references, and it should land before E1 so E1 pins the final template bytes.

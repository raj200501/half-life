goal_turn: 4
STATE: ACTIVE

# Status

Updated 2026-11-25 (goal_turn 4). E1 preregistered and paused after one session; front page refreshed. History lives in docs/LOG.md.

## Gates (fast check at de7b3dd, 2026-11-25: 10/14; fast results do not count)

| Gate | State | Evidence |
|---|---|---|
| G1 Build and tests | PASS | fresh-clone `make setup` and `make test`, coverage gate 85% |
| G2 Static quality | PASS | ruff check, ruff format, mypy --strict clean; every ignore has a reason |
| G3 CI parity | PASS (fast) | workflow runs `make ci` on Python 3.11 and 3.12 |
| G4 Offline demo | PASS | demo in a network namespace with no model calls, matches the golden output |
| G5 Instrument validity | PASS | detector precision and recall with intervals, probe leakage, complete compaction logs with per-rule retention |
| G6 Benchmark complete | PASS | thirty templates, four rule types, routes and reference runs with action logs verified |
| G7 Main study | FAIL | E1 is preregistered and frozen (prereg/0001-e1.md at 084bed1, two amendments the same day); one session ran on 2026-11-12, then the run was stopped (Raj's wrap-up) and resumes with docs/E1_RUNBOOK.md; G7 ties every session to the pinned plans and reach rule, harness, models, settings, every invocation, and the sandbox (D-062, D-064, D-066) |
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

- E1 is preregistered and frozen (prereg/0001-e1.md at 084bed1, amendments 0002 and 0003). It started on 2026-11-12 from 05d4b2c; one session finished (checkpointed raw and unanalyzed in `artifacts/e1/raw/`) before Raj's wrap-up stopped the run to save usage (37465c9). The cloud container was later replaced, so `runs/` and the model files are gone; E1 resumes on any machine with docs/E1_RUNBOOK.md (restore the checkpoints first). `src/` and `designs/` stay identical to the preregistration commit, as G7 requires.
- 2026-11-25: Raj's CI fix on main (the workspace test's UTC offset) is merged into the branch, so main fast-forwards to the branch. README.md's first screen now says what each audience can use today, gives an accurate status table, and describes how the build process is enforced by code. A one-page summary for Raj to review is in `outbound/one-pager.md`.
- Review 5 (BLOCK at 1f9a4c5) is fixed (D-067, CORRECTIONS.md); review 6 waits for a review allowance.
- A fresh cloud container checks the repository out shallow, and `make prereg-verify` (G11) then fails closed because first commits cannot be trusted; `git fetch --unshallow origin` fixes it (now the runbook's first setup step).

## Next three actions

1. Resume E1 on a machine that stays up for a few days (docs/E1_RUNBOOK.md: `make setup-local`, fetch the two pinned models, regenerate and check the plans, restore the checkpoints, start the runner with the pinned sandbox); checkpoint every few hours.
2. Review 6 (G12) of the then-current commit when a review allowance is available.
3. After E1: the E3 and E4 preregistrations, with plan pins in G8 and G9 as D-062 and D-064 do for E1; E2 on Raj's machine.

## Usage to date

- cash_usd = 0 in every ledger row. Claude subject runs 0, single calls 0, review runs 0 (the five reviews ran as build-session subagents).
- Ledger rows: 709, all local llama.cpp calls (dry runs, the planning benchmark, fixture recording, pilot sessions, and 81 rows from E1's first sessions).
- Build-session subagents (builders, the D-038 labeling agent, the citation checker, the reviewer, the audit workflow) run on the Max plan outside the harness and are not in the ledger.

## Needs Raj

0. **E1 compute (blocks G7).** E1 needs several days of CPU on a machine that stays up. Cloud containers here are replaced between sessions, so the run is paused after one session; resume it on a machine that stays up (docs/E1_RUNBOOK.md, about three to four days on four cores). Finished sessions are checkpointed into the repository, so a run can move between machines.
1. **Claude backend (blocks E2 in G8, and headless review).** This cloud container sets `ANTHROPIC_BASE_URL` and `CLAUDE_CODE_USE_CCR_V2`, so the billing guard refuses Claude children. To run E2: on a Linux machine with bubblewrap and socat where `env | grep -E '^(ANTHROPIC_|CLAUDE_CODE_USE_)'` prints nothing and `claude` is logged in to the Max plan, turn off extra usage (paid overage) on the account, then follow docs/E2_RUNBOOK.md (smoke run, plans from `designs/e2.yaml`, two tiers, bank, push). Record any contaminant the smoke run reports (for example a user-level CLAUDE.md) so it can go in DECISIONS.md. E2 needs about 3 days at the capped 6 sessions per day.
2. **Private vulnerability reporting (not blocking).** To accept private security reports as SECURITY.md describes, turn on private vulnerability reporting in the repository's security settings.
3. **Model licenses (confirm, not blocking).** Llama-3.2-3B-Instruct is under the Llama 3.2 Community License (its acceptable-use policy file is gated and could not be read here) and Qwen2.5-3B-Instruct under the Qwen Research License (non-commercial research and evaluation). This project uses both only for research evaluation and redistributes no weights (D-035). If you do not accept either license, say so and that family's data is dropped and recorded in CORRECTIONS.md.

## Blocked

Nothing yet.

## Backlog (suggestions for Raj, not ladder rungs)

0. **Audit backlog (2026-11-12).** From the pre-freeze audit, needing code or simulations (each becomes an amendment or an exploratory analysis before E1 is analyzed): H2b counting only scheduled compactions; G7 re-deriving the banked rows from the raw archives; the half-life's observed-range reading in the report; H3 labeled exploratory in the report's tables; power on the final models (D-063); and G8 and G9 tied to their plans as D-062 ties E1.
1. Publish half-life-bench templates with a canary string and a stable YAML constraint spec so other teams can run their own agents against it. The templates carry no canary yet (rule 9 asks for one in any dataset meant for release). Adding one changes each template's file hash, which the E0, pilot, and reference-run artifacts pin, so it needs a recorded decision and re-banked references, and it should land before E1 so E1 pins the final template bytes.

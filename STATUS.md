goal_turn: 3
STATE: ACTIVE

# Status

Updated 2026-11-04 17:20 UTC (goal_turn 3). Session paused here by request; nothing is running. History lives in docs/LOG.md.

## Gates (fast check at ad4a9e0, 2026-11-04: 6/14; fast results do not count)

| Gate | State | Evidence |
|---|---|---|
| G1 Build and tests | PASS | full suite passes on a fresh clone (`tools/prepush_ci.sh`), coverage gate 85% |
| G2 Static quality | PASS | ruff check, ruff format, mypy --strict clean |
| G3 CI parity | PASS (fast) | workflow runs `make ci` on 3.11 and 3.12; `make ci` passes on a fresh clone |
| G4 Offline demo | PASS | demo replays a synthetic fixture in a network namespace, matches golden |
| G5 Instrument validity | likely PASS (unverified) | detector and probe-scoring validation banked; probe leakage (0 leaks in 10 runs) and compaction logs (17 complete) banked from the pilot |
| G6 Benchmark complete | PASS | 30 templates, 4 constraint types, routes and reference runs banked (17 re-banked after the D-038 keyword tuning) |
| G7 Main study | FAIL | needs pilot, preregistration, E1 runs |
| G8 Ecological and placement arms | FAIL | E3 needs local runs; E2 needs Raj (Claude backend refused here) |
| G9 Mitigations shipped | FAIL | kit committed and wired into the runner (D-039); overhead banked; E4 not run |
| G10 Claim integrity | FAIL | no claims yet; README is still the brief |
| G11 Preregistration integrity | PASS (trivially) | no preregistrations yet |
| G12 Independent review | FAIL | no review banked |
| G13 Documentation | FAIL | README is still the brief; docs/REPORT.md is a draft |
| G14 Hygiene | FAIL (expected) | tree dirty while builders and the pilot write; clean at ship time |

## Environment (`make doctor`, 2026-11-04)

- Cloud container, Linux x86_64, 4 CPUs, 15.7 GB RAM, no GPU, about 28 GB free disk (floor 8 GB).
- Docker CLI yes, daemon no. Harness sandbox methods: `unshare-mount` (no network, writes confined to the workspace) and `unshare-net`; no bubblewrap (D-019).
- `claude` CLI 2.1.318 present; Claude backend refused by the billing guard (`ANTHROPIC_BASE_URL`, `CLAUDE_CODE_USE_CCR_V2` set by the cloud harness).
- Local runtime: llama-cpp-python CPU wheel (`make setup-local`); Hugging Face GGUF downloads work through the proxy.
- Subject Python for sandboxed commands: `make subject-env` (Python 3.12, pytest only, no pip; D-025).
- Launch settings: commits authored as Raj (D-021), CI only on main and manual dispatch with `make ci` run on a fresh clone before every push (D-022), push every commit (D-009), `cap_share = 0.2` (D-010).

## Backends and fallbacks

- Local open-weight models through llama.cpp: usable; model client and pins being built.
- Claude: unusable here; `make review` uses the reviewer subagent; E2 needs Raj (docs/E2_RUNBOOK.md being written).

## Now (paused, nothing running)

- Pilot banked (exploratory): `artifacts/pilot/` holds 10 sessions; `family_choice.json` applies D-042 and picks SmolLM2 and Granite for E1. Phi ran one session and Llama none (dropped for speed).
- E0 validation runs banked for G5: `artifacts/e0/probe_leakage.json` and `compaction_logs.json` (from the pilot sessions).
- Tool overhead banked (`artifacts/e4/overhead.json`). Statistics audit round 1 committed (D-049).
- Round 2 (D-050) was stopped before it finished; its partial edits were discarded unverified, so the code at HEAD is round 1. The D-050 decisions stand and must be implemented before the E1 preregistration. The workflow script is saved under the session's workflow scripts and can be rerun.
- The padded dry run was stopped; it is exploratory and nothing from it is banked.

## Next three actions

1. Rerun statistics round 2 (D-050: score tests for every hypothesis, family-stratified pooled tests, the marginal half-life, the cut-span exclusion, harness hashes and provenance, power mirroring the tests) with its review; then run `python -m halflife.power --pilot artifacts/pilot --out artifacts/power/e1.json`.
2. Fill in and commit prereg/0001-e1.md (draft in the session scratchpad; settings block per D-046), generate the E1 plans, and start E1: SmolLM2 and Granite, `tools/round_robin.py ... --prereg prereg/0001-e1.md`.
3. Gate pass while E1 runs (G7 not-estimable form, descriptive intervals, code pins and run provenance; G8 stale-report check; E3 and E4 settings blocks).

## Usage to date

- cash_usd = 0. Claude subject runs 0, single calls 0, review runs 0.
- Ledger rows: 266, all local llama.cpp calls (dry runs, the planning benchmark, fixture recording, and pilot sessions); no study data yet.
- Claude plan usage: subagent builders hit the weekly limit at about 04:00 UTC on 2026-11-04 (reset 07:00 UTC); relaunched after the reset.

## Needs Raj

1. **Claude backend (blocks E2 in G8, and headless review).** This cloud container sets `ANTHROPIC_BASE_URL` and `CLAUDE_CODE_USE_CCR_V2`, so the billing guard refuses Claude children. To run E2: on a Linux machine with bubblewrap and socat where `env | grep -E '^(ANTHROPIC_|CLAUDE_CODE_USE_)'` prints nothing and `claude` is logged in to the Max plan, turn off extra usage (paid overage) on the account, then follow docs/E2_RUNBOOK.md (smoke run, plans from `designs/e2.yaml`, two tiers, bank, push). Record any contaminant the smoke run reports (for example a user-level CLAUDE.md) so it can go in DECISIONS.md. E2 needs about 3 days at the capped 6 sessions per day.
2. **Model licenses (confirm, not blocking).** Llama-3.2-3B-Instruct is under the Llama 3.2 Community License (its acceptable-use policy file is gated and could not be read here) and Qwen2.5-3B-Instruct under the Qwen Research License (non-commercial research and evaluation). This project uses both only for research evaluation and redistributes no weights (D-035). If you do not accept either license, say so and that family's data is dropped and recorded in CORRECTIONS.md.

## Blocked

Nothing yet.

## Backlog (suggestions for Raj, not ladder rungs)

1. Publish half-life-bench templates with a canary string and a stable YAML constraint spec so other teams can run their own agents against it.

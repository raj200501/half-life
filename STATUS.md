goal_turn: 3
STATE: ACTIVE

# Status

Updated 2026-11-04 08:05 UTC (goal_turn 3). History lives in docs/LOG.md.

## Gates (fast check at ad4a9e0, 2026-11-04: 6/14; fast results do not count)

| Gate | State | Evidence |
|---|---|---|
| G1 Build and tests | PASS | full suite passes on a fresh clone (`tools/prepush_ci.sh`), coverage gate 85% |
| G2 Static quality | PASS | ruff check, ruff format, mypy --strict clean |
| G3 CI parity | PASS (fast) | workflow runs `make ci` on 3.11 and 3.12; `make ci` passes on a fresh clone |
| G4 Offline demo | PASS | demo replays a synthetic fixture in a network namespace, matches golden |
| G5 Instrument validity | FAIL | detector validation and probe-scoring validation banked (E0); per-rule retention now checked (D-041); probe leakage and compaction logs wait for the pilot sessions |
| G6 Benchmark complete | PASS | 30 templates, 4 constraint types, routes and reference runs banked (17 re-banked after the D-038 keyword tuning) |
| G7 Main study | FAIL | needs pilot, preregistration, E1 runs |
| G8 Ecological and placement arms | FAIL | E3 needs local runs; E2 needs Raj (Claude backend refused here) |
| G9 Mitigations shipped | FAIL | kit committed and wired into the runner (D-039); E4 and overhead not run |
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

## Now

- Done since the usage-limit reset: probe-scoring validation banked (D-038); mitigation kit committed and wired into the E4 arms (D-039); E3 and E4 analysis plans in code before any data, with per-session usage for E4 costs (D-040); E1 power-analysis tool (`halflife.power`); ship-check hardening pass 2, G5 per-rule retention, G6 reference action logs, G7 recomputation (D-041); related work written from verified citations.
- Pilot amended before any further session (D-042): families must also act at no fewer than half of the decision points they reach; Phi's running session finishes, then SmolLM2, Granite, and Qwen run the other two templates; Phi and Llama run more only if fewer than two of those qualify.
- Sessions now pad to at least 20 subtasks with neutral read-only fillers for E1, E3, and E4 (D-043, section 4 of the brief).
- E2 driver, runbook, design, and analysis step committed (D-044); E2 itself needs Raj. Builder running: the report pipeline (`make report`, `artifacts/claims.json`, freshness derivations).

## Next three actions

1. Finish the pilot (`runs/pilot-driver.log`, phase 2 per D-042), apply the D-042 rule, bank the pilot as the E0 validation runs (G5), and run `python -m halflife.power`.
2. Fix the E1 design (min_subtasks 20, a compaction window inside real session lengths), write and commit prereg/0001-e1.md, generate the plans, and start E1 in the background.
3. Integrate the report and E2 builders; bank `artifacts/e4/overhead.json` when the CPU is quiet.

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

goal_turn: 2
STATE: ACTIVE

# Status

Updated 2026-11-04 (goal_turn 2). History lives in docs/LOG.md.

## Gates (fast check at f3db2ec, 2026-11-04: 5/14; fast results do not count)

| Gate | State | Evidence |
|---|---|---|
| G1 Build and tests | PASS | 413 tests, core line coverage 92.1% |
| G2 Static quality | PASS | ruff check, ruff format, mypy --strict clean |
| G3 CI parity | PASS (fast) | workflow runs `make ci` on 3.11 and 3.12; `make ci` passes on a fresh clone |
| G4 Offline demo | PASS | demo v0.1 replays a synthetic fixture in a network namespace, matches golden |
| G5 Instrument validity | FAIL | detector validation banked (E0); probe leakage and compaction logs need validation runs |
| G6 Benchmark complete | FAIL | 15 templates verify; 15 more being written; manifest not banked |
| G7 Main study | FAIL | needs pilot, preregistration, E1 runs |
| G8 Ecological and placement arms | FAIL | E3 needs local runs; E2 needs Raj (Claude backend refused here) |
| G9 Mitigations shipped | FAIL | kit being built; E4 not run |
| G10 Claim integrity | FAIL | claimcheck and freshness being built; no docs/REPORT.md |
| G11 Preregistration integrity | FAIL | prereg-verify being built |
| G12 Independent review | FAIL | no review banked |
| G13 Documentation | FAIL | README is still the brief; no docs/REPORT.md; citations being verified |
| G14 Hygiene | PASS | secrets clean, cash_usd = 0, caps ok, private/ untracked, tree clean |

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

Three builder workflows are running (one owner per file set): model client, backends, pinned local models, then the mitigation kit; integrity tools and stats with figures, then citations; template QA on 14 templates, 15 new templates by three authors, and the E2 Claude Code driver. Done this session: Raj's authorship and CI-budget instructions, offline demo v0.1 (tag v0.1), subject Python environment, doctor reports harness sandbox methods.

## Next three actions

1. Integrate the builders: make targets for prereg-verify, claimcheck, freshness, citecheck; bench verify and bank all 30 templates with the manifest (G6).
2. Family pilot on the 3 pilot templates with the pinned local models; use its sessions as the E0 validation runs for probe leakage and compaction-log completeness (G5); probe-scorer validation set (D-023).
3. Power analysis and the E1 preregistration, then E1 in the background (G7, G11).

## Usage to date

- cash_usd = 0. Claude subject runs 0, single calls 0, review runs 0. Local model calls 0.

## Needs Raj

1. **Claude backend (blocks E2 in G8, and headless review).** This cloud container sets `ANTHROPIC_BASE_URL` and `CLAUDE_CODE_USE_CCR_V2`, so the billing guard refuses Claude children. To run E2, start a session on a machine where `env | grep -E '^(ANTHROPIC_|CLAUDE_CODE_USE_)'` prints nothing and `claude` is logged in to the Max plan.
2. **E2 runs on your machine.** E2 needs the `claude` CLI as a subject with a clean billing guard; docs/E2_RUNBOOK.md (being written) will give the exact commands.

## Blocked

Nothing yet.

## Backlog (suggestions for Raj, not ladder rungs)

1. Publish half-life-bench templates with a canary string and a stable YAML constraint spec so other teams can run their own agents against it.

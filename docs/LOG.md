# Log

History moved out of STATUS.md, newest first.

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

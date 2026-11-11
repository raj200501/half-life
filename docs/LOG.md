# Log

History moved out of STATUS.md, newest first.

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

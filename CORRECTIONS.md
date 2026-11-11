# Corrections

Every changed or withdrawn claim gets a dated entry here (rule 3 of the build brief): the date, the claim as originally stated, what changed, why, and the artifact that supports the correction.

## 2026-11-11: rules per template

- **Claim as stated:** docs/REPORT.md section 3.1 (draft, before release) said each session template has "two to four rules".
- **What changed:** every template in half-life-bench has two or three rules (nineteen with two, eleven with three); no template has four. The report and README now say "two or three".
- **Why:** the sentence described the design target, not the banked bench; a pre-release check against the template files caught it.
- **Support:** the template files under `src/halflife/bench/templates/`, pinned by hash in `artifacts/bench/manifest.json`.

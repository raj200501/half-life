# E1 preregistration, amendment 2

Amends: prereg/0001-e1.md

- **Date:** 2026-11-12 (UTC), before any E1 session ran. It changes no hypothesis, test, exclusion, sample size, stopping rule, or pin, and is reported as a deviation in docs/REPORT.md. The pre-freeze audit's later lenses raised each item.

## 1. Corrections to the text

- **Section 1.** The confirmatory tests are seven, not six: H1 and H2b in each of the two families (four tests) and H2a in each family and pooled (three tests), each at alpha 0.05 with no adjustment across them.
- **Section 8, fourth limitation.** H2a's conservativeness under rate differences its null model does not represent is recorded in DECISIONS.md D-059, not D-060.
- **Front matter and section 6.** The dry runs used seeds 990001, 991000 and 991001, and 992000 and 992001. Their session times (about 23 and 46 minutes) are smollm2's; no granite session has finished an E1-shaped dry run, and both finished dry-run sessions were in the sham arm, so E1's first sessions are also the first on granite at that shape and the first in the `full_summary` arm on the current harness.
- **Section 5, Not tested.** `min_events` was frozen before the clean re-run of the calibration that DECISIONS.md D-063 item 4 asked for had finished. The re-run, from commit a722586 with a clean tree, reproduced `min_events` 7 and the same failing cells; the four tests that the saturation fix made computable each added one computed test to the cells that count it (`artifacts/power/min_events.json`).

## 2. An added limitation and an exploratory refit

Within a template the decision points keep a fixed order, and filler counts move their positions by a few requests at most, so H1 and the half-life largely compare a template's early and late decision points; a difference between decision points can read as an effect of position. Beside the confirmatory results, the hazard models are refit with the decision point's identity (template and decision point) as a factor, so that position varies only within a decision point. The refit is exploratory and changes no decision.

## 3. Operating rules while E1 runs

- **No interim analysis.** The runner's progress lines and each session's `result.json` also carry violation counts. While E1 runs, the operator reads only whether a session finished and whether it reached its first decision point (for the reach rule, read with a command that prints nothing else), does not read or tabulate violation counts, and does not open the checkpoint archives, until the last planned session has run.
- **Sandbox.** Every E1 invocation must run in the `unshare-mount` sandbox (no network, writes confined to the workspace), the method the harness and the reference runs used (DECISIONS.md D-019, D-025). Gate G7 checks it in every invocation record, along with each invocation's model, run settings, and plans file (D-066).

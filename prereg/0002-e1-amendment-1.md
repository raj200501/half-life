# E1 preregistration, amendment 1

Amends: prereg/0001-e1.md

- **Date:** 2026-11-12 (UTC), the day prereg/0001-e1.md was frozen, before any E1 session ran. It changes no hypothesis, test, exclusion, sample size, stopping rule, or pin, and is reported as a deviation in docs/REPORT.md.

## 1. A withdrawn sentence in section 8

The second limitation of section 8 says that the hazard models "stop at the first violation and are not affected" by masking. That is wrong, and the sentence is withdrawn. A session's risk set ends at the first opportunity whose own rule was violated inside that decision point's span (`halflife.stats.risk_set` over the opportunity rows). A violation outside every decision-point span, or of another rule, does not end it. So when a state rule is broken outside a span, a later decision point of the same rule cannot show a new violation, and that masking can pull later hazards toward zero in the hazard models (H1, H2b, and the half-life) as well as in the rate contrasts. A pre-freeze audit raised this, and a check of the code after the freeze confirmed it.

## 2. An added exploratory sensitivity analysis

Beside the confirmatory results, the hazard models are refit on risk sets that also end before the first opportunity with any earlier violation in its session (`prior_violations` above zero in the opportunity rows). The refit is exploratory: it is reported with its coefficients and intervals, and it changes no confirmatory decision.

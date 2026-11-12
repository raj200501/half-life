# Corrections

Every changed or withdrawn claim gets a dated entry here (rule 3 of the build brief): the date, the claim as originally stated, what changed, why, and the artifact that supports the correction.

## 2026-11-11: rules per template

- **Claim as stated:** docs/REPORT.md section 3.1 (draft, before release) said each session template has "two to four rules".
- **What changed:** every template in half-life-bench has two or three rules (nineteen with two, eleven with three); no template has four. The report and README now say "two or three".
- **Why:** the sentence described the design target, not the banked bench; a pre-release check against the template files caught it.
- **Support:** the template files under `src/halflife/bench/templates/`, pinned by hash in `artifacts/bench/manifest.json`.

## 2026-11-11: `--max-turns` in Claude Code

- **Claim as stated:** docs/E2_RUNBOOK.md, the E2 driver's docstring, and DECISIONS.md D-016 said Claude Code 2.1.309 has no `--max-turns` option.
- **What changed:** the CLI reference documents `--max-turns` for print mode; the flag is only absent from `claude --help`. The runbook and the docstring now say so.
- **Why:** the inference came from the help text alone; the pre-release audit checked the reference.
- **Support:** no artifact depends on it. The driver enforces the turn cap by counting turns in the stream either way, and no E2 session has run.

## 2026-11-11: README claims the third independent review found

- **Claims as stated:** README.md (commit af5ddc6) said the project ships "a preregistered analysis"; that "every number on this page" comes from `make report`; that the guard "is cheap relative to a model call"; that D-050 "extends" the score test to every preregistered hypothesis; and that the latest review's findings "are fixed".
- **What changed:** no preregistration is committed, so the README now says the analysis was written and audited before any confirmatory data and will be frozen by a preregistration. The number sentence now covers result numbers, since counts that describe the design (thirty templates, two or three rules each) are design parameters. The guard sentence now says only that the overhead of the guard and the pin is measured, since no model-call latency is banked beside it. D-050 is described as decided and not implemented. The G12 line no longer says which findings are fixed.
- **Why:** the third independent review (REVIEW: BLOCK at af5ddc6, B3 and non-blocking items) showed that each sentence claimed more than the repository supports. Rule 2 labels anything not preregistered as exploratory.
- **Support:** `prereg/` holds no preregistration; `artifacts/e4/overhead.json` holds the measured overhead; `artifacts/review/latest.json` holds the review.

## 2026-11-11: what report-check finds in documents

- **Claim as stated:** docs/ARTIFACTS.md (commit c33bad4) said `make report-check` finds embedded images "in every form a renderer shows".
- **What changed:** the hand-written parser behind it missed forms GitHub renders: an unclosed `![[` stopped the scan, link definitions in block quotes or list items or split across lines were not seen, and a decoy definition the renderer ignores shadowed the real one. The check now renders the documents with cmark-gfm, GitHub's renderer, and also fails closed on any image file named anywhere in the raw text (DECISIONS.md D-057), and docs/ARTIFACTS.md describes that.
- **Why:** the fourth independent review (REVIEW: BLOCK at 526ae0d, B1) embedded a hand-edited copy of the pilot figure in both documents and passed G10.
- **Support:** no reported number changed; the review is in `artifacts/review/latest.json`, and the mutation tests in `tests/test_report.py` cover each form.

## 2026-11-11: when the statistics audit ran

- **Claim as stated:** README.md said the adversarial audit caught its bugs "before any data existed" and that each fix was recorded in D-049 and D-050.
- **What changed:** the audit ran after the E0 validation data and the exploratory pilot existed, so the README now says "before any confirmatory data existed". D-050's fixes are in progress, not done, so the README says so.
- **Why:** the fourth independent review (non-blocking item at 526ae0d).
- **Support:** the commit dates of `artifacts/e0/` and `artifacts/pilot/` against DECISIONS.md D-049 and D-050.

## 2026-11-12: the minimum of violations for testing H1 and H2b

- **Claim as stated:** DECISIONS.md D-058 fixed `min_events` at 2, from the synthetic calibration then banked, and STATUS.md repeated it.
- **What changed:** the audit of statistics round two changed H1's and H2b's models (D-059 to D-061), and the calibration of the final models puts the minimum at 7 (D-063). The E1 preregistration fixes 7.
- **Why:** the earlier value calibrated models that are no longer the preregistered ones; with them, H2b rejected too many true nulls at 2 to 6 violations.
- **Support:** `artifacts/power/min_events.json` (synthetic). No E1 data existed, and README.md and docs/REPORT.md never stated the earlier value.

## 2026-11-12: masking and the hazard models (E1 preregistration)

- **Claim as stated:** prereg/0001-e1.md, section 8, said the hazard models stop at the first violation and are not affected when a state rule already broken cannot show a second violation.
- **What changed:** the sentence is withdrawn by prereg/0002-e1-amendment-1.md. The risk set ends at the first decision point whose own rule was broken inside its span, so a state rule broken outside every span can mask later decision points of that rule in the hazard models too. The amendment adds an exploratory refit on risk sets that also end at any earlier violation.
- **Why:** the pre-freeze audit raised it, and a check of `halflife.stats.risk_set` after the freeze confirmed it.
- **Support:** no E1 session had run; `src/halflife/stats.py` (`risk_set`) and `src/halflife/analysis.py` (`prior_violations`).

## 2026-11-12: four statements of the frozen E1 preregistration (amendment 2)

- **Claims as stated:** prereg/0001-e1.md said the confirmatory tests were six (section 1), that the dry runs used seeds from 990000, 991000, and 992000 (front matter), that an E1-shaped session took about 23 to 46 minutes in dry runs (section 6), and that H2a's conservativeness is recorded in D-060 (section 8).
- **What changed:** prereg/0003-e1-amendment-2.md corrects them: seven confirmatory tests; seeds 990001, 991000 and 991001, and 992000 and 992001; the timings are smollm2's, since no granite session finished an E1-shaped dry run; and the reference is D-059.
- **Why:** the pre-freeze audit's later lenses, and the fifth independent review (B2), found them.
- **Support:** the dry-run directories' plans and results (not in the repository) and DECISIONS.md D-059; no E1 session had run.

## 2026-11-12: what the recall probes touch (README.md)

- **Claim as stated:** README.md said the recall probes "never touch the main run".
- **What changed:** since 5f9592e it says no probe text ever enters the main run, and that the probes share its sampling seed sequence.
- **Why:** the fourth independent review showed that probes advance the seed sequence the main run samples from, so the main run's sampled tokens depend on how many probes ran (alike in every arm).
- **Support:** `src/halflife/agent.py` (`_seed`).

## 2026-11-12: "all were fixed before the freeze" (README.md)

- **Claim as stated:** README.md (acccec3) said the findings of the second audit and the pre-freeze audit were all fixed before the E1 freeze.
- **What changed:** the README now says those fixes landed before the freeze and that two amendments the same day, before any E1 data, corrected the frozen text.
- **Why:** the fifth independent review (B2): both amendments correct statements of the frozen preregistration.
- **Support:** prereg/0002-e1-amendment-1.md and prereg/0003-e1-amendment-2.md.

## 2026-11-12: what G10's Mermaid check catches (DECISIONS.md D-065)

- **Claim as stated:** D-065 said G10 fails on "any digit inside a mermaid block".
- **What changed:** the check it described matched the closing fence by hand and missed a longer closing fence, block quotes, list items, and an unclosed fence; G10 now renders the documents with cmark-gfm, the renderer report-check uses, and fails on a digit in any rendered mermaid block (D-067).
- **Why:** the fifth independent review (B1) passed G10 with a pie chart of invented counts closed by a longer fence.
- **Support:** the mutation tests in `tests/test_gate_mutations.py` cover each form.


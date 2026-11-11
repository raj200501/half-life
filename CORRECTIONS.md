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

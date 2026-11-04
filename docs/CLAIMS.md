# Claims, preregistrations, and citations

How the integrity checks behind gates G10, G11, and G13 read this repository. Rule 1 of the build brief: every result number in README.md, docs/REPORT.md, other results documents, figures, and tables is produced by a committed script from a banked artifact under `artifacts/`. Rule 2: preregistrations are frozen before confirmatory data exists. Rule 11: every reference is verified once and banked.

| Command | Module | Gate |
|---|---|---|
| `make claimcheck` | `halflife.integrity.claimcheck` | G10 |
| `make freshness` | `halflife.integrity.freshness` | G10 |
| `make prereg-verify` | `halflife.integrity.prereg` | G11 |
| `make citecheck` | `halflife.integrity.citecheck` | G13 |

Each check prints one line per check (`ok`, `FAIL`, or `note`), then a verdict line, and exits 0 only on a pass. They use the standard library and the `halflife` package, find the repository root with `git rev-parse --show-toplevel` (so they run from any directory in the repository, or with `--root DIR`), never touch the network, and never call a model. Anything a check cannot verify is a failure, not a skip.

## Claim markers

A result number is written between two HTML comments on one line:

```markdown
The constraint half-life was <!-- claim:e1.halflife.qwen -->35.2<!-- /claim --> tool calls.
```

- The id matches `[A-Za-z0-9][A-Za-z0-9_.-]*`. Both markers sit on the same line, with only the rendered value between them; no other comment may sit inside a claim.
- GitHub hides the comments, so the page reads as plain text. A table cell can hold a claim.
- An interval is two claims: `[<!-- claim:e1.hl.lo -->22.0<!-- /claim -->, <!-- claim:e1.hl.hi -->61.3<!-- /claim -->]`.
- Units and words stay outside the markers. Any transformation (a difference, a percentage point change, a rounding rule) belongs in the derivation that produces the artifact, not in the document.
- A marker inside inline code or a fenced code block is an example, not a claim.
- Malformed markers fail: a claim not closed on its line, nested or stray markers, and comments that look like markers but do not parse (`<!-- Claim: x -->`).

`python -m halflife.integrity.claimcheck --strip FILE` prints a document with its markers removed, for outbound copy that is sent as plain text. Drafts under `outbound/` keep their markers so the check can trace every number.

## The claims manifest: `artifacts/claims.json`

An illustration (the derivation function named here is an example of the contract described under Freshness, not an existing function):

```json
{
  "schema": "halflife/claims/1",
  "documents": ["docs/MITIGATIONS.md"],
  "claims": {
    "e1.halflife.qwen": {
      "artifact": "artifacts/e1/half_life.json",
      "pointer": "/estimates",
      "where": {"family": "qwen2.5"},
      "field": "/half_life",
      "format": ".1f"
    },
    "e1.sessions.qwen": {
      "artifact": "artifacts/e1/sessions.jsonl",
      "where": {"family": "qwen2.5", "reached_first_dp": true},
      "aggregate": "count",
      "format": "d"
    }
  },
  "allowlist": [
    {"value": "0.95", "context": "at least 0.95", "reason": "G5 threshold from section 6 of the brief"}
  ],
  "derivations": {
    "artifacts/e1/half_life.json": {
      "derive": "halflife.e1_analysis:half_life_artifact",
      "inputs": ["artifacts/e1/opportunities.jsonl"],
      "params": {"seed": 20261104, "n_boot": 2000},
      "compare": ["/estimates"]
    }
  },
  "raw": {
    "artifacts/e1/opportunities.jsonl": "one row per decision point reached, written by the E1 runner",
    "artifacts/e1/sessions.jsonl": "one row per E1 session, written by the E1 runner"
  }
}
```

Top-level keys are `schema`, `notes`, `documents`, `claims`, `allowlist`, `derivations`, and `raw`; any other key fails, and so does a repeated key anywhere in the file.

### Claims and selectors

| Key | Meaning |
|---|---|
| `artifact` | A `.json` or `.jsonl` file under `artifacts/`. A `.jsonl` file is the list of its rows. `artifacts/claims.json` and `artifacts/citecheck.json` are bookkeeping and may not be cited. |
| `pointer` | RFC 6901 JSON pointer into the artifact (`""`, the default, is the whole artifact). |
| `where` | Keeps the rows of the list at `pointer` whose keys all equal the given values. `true` never equals `1`, and `1` never equals `1.0`. |
| `aggregate` | `count` (number of rows) or `sum` (of `field` over the rows). |
| `field` | Pointer applied to the one row that `where` selects (exactly one must match), or to every row for `sum`. |
| `format` | Python format spec applied to the selected value: `.3f`, `.1%`, `d`, `,d`, or `""` for a string. |
| `note` | Free text for the reader. |

The rendered text between the markers must equal `format(value, spec)` exactly, so `0.4123` written where the artifact gives `0.412` fails (a precision overclaim), and so does a value that went stale. Booleans, null, lists, and objects cannot be rendered.

### Which documents are checked

README.md and docs/REPORT.md (both must exist), every path in `documents` (each must exist), every `.md` or `.txt` file under `outbound/`, and any other Markdown file with a claim marker outside code. docs/BUILD_BRIEF.md is never a results document.

### Numbers that are not claims

Every digit sequence outside a claim fails unless one of these rules exempts it. The rules are listed in full in `scan_numbers` in `src/halflife/integrity/claimcheck.py`; changing them needs a DECISIONS.md entry (rule 13: never pass a gate by weakening its check).

1. Code and comments: fenced code blocks, inline code, and HTML comments. Code blocks hold commands and code, never results; a result number placed in a code block is a rule 1 violation that the reviewer looks for.
2. Markup and addresses: HTML tags with their attributes, HTML entities, URLs, Markdown link targets and link definitions, `arXiv:` identifiers, DOIs, and paths that exist in the repository. Link text, image alt text, and the values of the HTML `alt`, `title`, and `aria-label` attributes are checked, since readers see them.
3. Dates and versions: ISO dates and times, `YYYY-MM` months, dates with month names (`November 4, 2026`), years in citation form (`(2023)`, `(Liu et al., 2023)`, `et al. 2023`), versions with three or more parts (`2.1.309`), and hex digests of 7 to 64 characters with at least one letter and one digit. A bare year (`in 2026`) is not exempt.
4. Identifiers: digits attached to letters on their left in one word (`G7`, `E1`, `RQ3`, `D-012`, `SHA-256`, `v0.5`, `qwen2.5-0.5b`, `Q4_K_M`).
5. Structure: heading section numbers (`## 5. Experiments`; a number with one part needs a trailing `.` or `)`, and `### 5.2 Title` needs the parent number `5` on an earlier heading of the same document), ordered-list markers (`3.`), and citation or footnote markers (`[3]`, `[^3]`, up to three digits, not link text).
6. References: the number right after a reference word (section, rule, figure, fig., table, step, appendix, equation, eq., chapter, rung, milestone, gate, line, version, preregistration, amendment, seed, Python, or the section sign). After a plural reference word, Python, or a double section sign, a list or range continues with `and`, `or`, `to`, `through`, a hyphen, an en dash, or `, and` (`sections 6 to 10`, `Python 3.11 and 3.12`, `rules 1, 9, and 12`). A number after a bare comma continues the list only if the list goes on or ends right after it, so in `In Figures 2 and 3, 87 sessions` the 87 still fails.
7. The allowlist (below).

Numerals that are not ASCII digit runs always fail and cannot be allowlisted: an HTML entity that renders as a numeral (`&#52;&#48;` renders as 40) and characters such as fractions and superscripts. Write numbers as digits.

### The allowlist

Design parameters, thresholds, caps, and other numbers that are not results go in `allowlist`. Each entry has exactly three keys:

- `value`: the number as written (`4,000`, `0.95`);
- `context`: the literal text around it, as it appears on its line, containing the value and at least one word (`"at least 0.95"`); the number is exempt only where it lies inside an occurrence of that text;
- `reason`: why it is not a result number.

An entry that matches nothing is noted, so stale entries are visible.

## Freshness: re-deriving every live number

claimcheck asks whether each number equals its artifact; freshness asks whether the artifact is still what the committed code derives from the lower-level artifacts. A number that was true when written but went stale passes the first and fails the second.

Each artifact a claim cites must be either derived or raw.

- **Derived** (`derivations`): `derive` names `module:function` defined in a file inside this repository (a module under `tools/` resolves too). Freshness loads every input, calls `function(inputs, **params)` with the network disabled and `HALFLIFE_OFFLINE=1` and `HALFLIFE_NO_MODEL_CALLS=1` set, and compares the result with the banked artifact at each `compare` pointer (floats within a relative 1e-9, everything else exactly). `inputs` maps each input path to its loaded content; an input ending in `/` is a directory, loaded as a mapping from each `.json` or `.jsonl` file under it to its content. The function returns the artifact's content and must not write files: a derivation that writes or deletes anything under `artifacts/`, `docs/`, `outbound/`, or README.md fails, since rewriting the banked file would hide that it was stale. It must not start processes or reach the network.
- **Raw** (`raw`): primary data that cannot be re-derived offline, such as runner output, with a reason. A raw path must exist (a path ending in `/` covers a directory and must hold artifacts) and every raw file needs provenance: a top-level `provenance` object in the artifact, or a `provenance.json` in its directory or an ancestor under `artifacts/`, with a non-empty `command` (or `commands`) and a `git_sha`. A derived artifact may not sit under a raw directory; declare those raw files one by one.

Every derivation input must itself be derived or raw, a derivation may not list itself as an input, and derivations may not form a cycle. Every derivation runs, whether or not a document cites it. For each live claim, the value is selected again from the re-derived content (derived) or the raw artifact (raw); it must equal the banked value, and its formatted text must equal every place a document renders it.

Freshness can only show that the banked artifacts agree with the committed code. Whether a derivation computes what the report says it computes is a question for review.

## Preregistrations

`make prereg-verify` reads the git history of the repository it runs in.

- **Layout.** Preregistrations are `prereg/NNNN-slug.md` (four digits, then a slug of letters, digits, `_`, and `-`). `prereg/.gitkeep` and README files are ignored; any other file under `prereg/` fails, so a misnamed preregistration cannot escape the checks.
- **Frozen.** A committed preregistration never changes: its content at HEAD and in every later commit equals its content in the commit that first added it, it is never deleted or renamed, and the working tree does not edit it. The report prints the SHA-256 of each frozen file. An uncommitted preregistration is a draft and is only noted.
- **Numbering.** Numbers run from 0001 without gaps or duplicates (drafts included), and committed preregistrations were first committed in number order.
- **Amendments.** A change is a new file whose slug contains `amend` or that has an `Amends:` line. It names the file it amends by path (`Amends: prereg/0001-e1.md`), is numbered after it, is first committed no earlier than it, and carries a date written `YYYY-MM-DD`. The report presents it as a deviation (rule 2).
- **Confirmatory artifacts.** A `provenance.json` under `artifacts/` with `"confirmatory": true` (or with no `confirmatory` key and a non-null `prereg`) governs every file in its directory tree, except subtrees with their own `provenance.json`. Any other JSON artifact with `"confirmatory": true`, and any JSON-lines artifact with a `"confirmatory": true` row, is confirmatory too. Each names in `prereg` one path (or a list of paths) of a committed preregistration whose first commit is a proper ancestor of the first commit of every file it governs. Committing the data together with its preregistration fails, and so does data first committed on a branch without it. A `confirmatory` value that is not a boolean, a repeated JSON key, or an artifact that does not parse fails.
- **History.** A shallow clone fails whenever there is something to verify. The ship check clones with full history.

Bank pilots, power analyses, and other files written before the preregistration outside the confirmatory directory, or under a subdirectory with its own `provenance.json` that says `"confirmatory": false`.

## Citations

`make citecheck` validates `artifacts/citecheck.json` offline against the schema in `.claude/agents/citation-checker.md`. Only the `citation-checker` subagent writes that file.

- `checked_by` is `citation-checker`. Every reference has a unique `key`, an http(s) `cited_url`, a `status` of `verified`, `mismatch`, or `unverified`, and a string `notes`. A verified reference also has an http(s) `final_url`, a `retrieved` date (`YYYY-MM-DD`, not in the future), a non-empty `page_title`, a non-empty `authors` list, an integer `year`, a 64-character lowercase hex `content_sha256`, and a `hash_source` of `curl body` or `webfetch text`. No two references bank the same URL.
- Every URL in README.md or docs/REPORT.md outside code and HTML comments counts as cited, including links to this project and bare `www.` addresses (which GitHub renders as links; they match a banked URL with either scheme); reserved hosts (`localhost`, loopback addresses, `example.com`, `.test`, `.invalid`, and similar) do not. A cited URL matches a reference by its `cited_url` or `final_url`, ignoring scheme and host case, default ports, a trailing slash, and the fragment, and that reference must be `verified` with every field valid.
- A paragraph, list item, or table row that looks like an author-year citation (`et al`, `Name (2023)`, or `(Name, 2023)`) must carry a URL, directly or through a reference-style link.
- References that are banked but not cited (for example ones dropped because nothing opened) are noted, not failed.

The verdict is `CITECHECK: PASS <n> verified`, where n counts the distinct references cited, or `CITECHECK: FAIL <n> problem(s)`.

## Known limits

- Figures are generated by `make report` from artifacts; their text is not scanned here.
- Exemption rules are deliberately narrow, but a writer can still hide a number in a code block or an identifier, or write it in words (forty, half), which no digit scan sees. Each of these breaks rule 1, and the reviewer checks for them.
- A bare domain without a scheme or `www.` (`arxiv.org/abs/...`) is plain text on GitHub, not a link, and citecheck does not treat it as a citation; cite with a full URL.
- Claims on raw artifacts are re-selected, not re-derived: nothing lower-level exists to derive them from offline.

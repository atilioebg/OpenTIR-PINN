---
description: "Section Grounding Header: every manuscript section starts with a header listing its grounding sources and draft status, and a grounding audit is run before each section is drafted."
---

# SECTION GROUNDING HEADER (manuscript drafting protocol)

## Why this rule exists
The grounding audit for Section 3 (08/11/2026) found 8 errors in our own blueprint before a single sentence was written (e.g., 22/8 instead of 20/10 atmospheric profiles; the LOCO test-pool composition). Errors that reach the PDF become Reviewer #2 rejection vectors. Every section must therefore declare its sources, and those sources must be checked against the data *before* drafting.

## Mandatory behaviour
1. **Header block.** Every file in `manuscript_rse/sections/*.tex` starts with a comment block:
   ```latex
   % =========================================================================
   % Section N -- <title>
   % Grounding: <D-ids and exact files/keys, e.g. D1 (dataset_facts.json: split.*), D13>
   % Code:      <source files whose behaviour is described, if any>
   % Status:    STUB | DRAFT vX.Y (date) | PI-REVIEWED (date) | AUDITED (date)
   % Open items: <count of \citeneeded / \pivalidate markers>
   % =========================================================================
   ```
2. **Pre-draft grounding audit.** Before drafting (or materially rewriting) a section:
   - List every quantitative claim the blueprint (`peer_review_report_v2.md` §7) assigns to that section.
   - Verify each one against the grounding files (`docs/grounding/` D1–D13) and, for Methods, against the code.
   - Record each disagreement as an erratum (blueprint statement → grounded value, with source) in the peer-review report (§9.x), and report it to the PI *before* presenting the draft.
3. **No orphan numbers.** Every number in the section must appear in, or be computed directly from, a file listed in the header. Numbers from model results cite the CSV/JSON (D12/D13), never a chat message or a log excerpt.
4. **Markers.** Unverified citations use `\citeneeded{...}`; decisions that need the PI use `\pivalidate{...}`. The header's "Open items" count is updated on every edit. A section cannot be marked AUDITED while any marker remains.
5. **Status promotion.** DRAFT → PI-REVIEWED requires the PI's explicit approval; PI-REVIEWED → AUDITED requires a `claim-code-auditor` pass (0 CONTRADICTED) and a `citation-auditor` pass for that section.


---
name: citation-auditor
description: Rapidly audits and verifies academic claims against cited papers, detecting secondary citation echoes and extracting exact quotes, page numbers, and tables without full-text reading.
---

# Citation Auditor Skill

## Overview
This skill provides an automated, rigorous protocol to audit, fact-check, and ground academic citations cited in manuscript introductions, related works, and discussions. It prevents citation misattribution and eliminates the "academic telephone game" (secondary citation echoes) where an author cites a paper that was merely quoting someone else.

Equipped with an internal PyMuPDF-based text analytics engine ([`audit_claim.py`](file:///c:/Users/Atilio/Desktop/PROJETOS/AITIR/.agents/skills/citation-auditor/scripts/audit_claim.py)), this skill can verify whether a statement is an original empirical finding or an echo in under 5 seconds without reading the cited document cover-to-cover.

---

## When to Activate
Activate this skill whenever:
* The user asks to verify, audit, or fact-check a citation or statement:
  * *"How can I verify this claim inside [Paper X]?"*
  * *"Did [Author, Year] actually prove that [Concept / Number]?"*
  * *"Audit the citations in our introduction section."*
* Reviewing a draft to ensure no claims are vulnerable to Reviewer #2 objections.
* Checking whether a reference is a primary empirical source or a secondary review echo.

---

## Rapid Audit Architecture (The 4-Step Heuristic)

```
                 INCOMING CLAIM WITH CITATION
              "Silicates cause a 0.35 drop... [Grondona et al., 2020]"
                                 │
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
  [Quantitative Claim]     [Method Claim]        [Conceptual Claim]
  Look at: Tables,         Look at: Section II,   Look at: Abstract,
  Figures, & Captions      Equations, Algorithms  Conclusion, Discussion
```

### 1. High-Entropy Feature Extraction
The auditor extracts:
- **Numerical Anchors:** Absolute numbers ($0.35$, $1.441\text{ K}$, $79.7\%$) are unique across a paper and take 1 second to isolate.
- **Low-Frequency Physics/Mineral Keywords:** High-entropy technical nouns (e.g., `reststrahlen`, `kaolinite`, `MMD`, `sobolev`, `Wien`).
- **Stopwords Discarded:** Common academic filler (`demonstrates`, `using`, `paper`, `we`, `show`) is filtered out.

### 2. Sentence Scoring & Page Mapping
PyMuPDF inspects each page stream, maps sentences to exact page numbers, and scores sentence candidate relevance using:
$$\text{Score} = 3.5 \times N_{\text{numbers}} + 1.0 \times N_{\text{keywords}}$$

### 3. The Secondary Citation Echo Detector
The algorithm checks whether the candidate sentence ends with or contains external citation tokens (e.g., `[12]`, `(Smith et al., 2018)`):
* If an external citation is attached $\longrightarrow$ **`FLAGGED [SECONDARY ECHO]`** (The author did not discover this; they were citing an earlier work).
* If no external citation and contains primary empirical indicators $\longrightarrow$ **`VERIFIED [PRIMARY EMPIRICAL]`**.

---

## Command-Line Tool Usage

To audit a claim against any PDF in the workspace or library, run:

```bash
python "c:\Users\Atilio\Desktop\PROJETOS\AITIR\.agents\skills\citation-auditor\scripts\audit_claim.py" \
  --pdf "path/to/cited_paper.pdf" \
  --claim "Exact claim or sentence from manuscript"
```

### Optional Explicit Keywords:
```bash
python "c:\Users\Atilio\Desktop\PROJETOS\AITIR\.agents\skills\citation-auditor\scripts\audit_claim.py" \
  --pdf "path/to/cited_paper.pdf" \
  --claim "Kaolinite exhibits an Al-OH doublet around 9.6 um" \
  --keywords "kaolinite,doublet,9.6"
```

---

## Standardized Output Schema

```markdown
### 🛡️ Citation Audit Verdict: [VERIFIED | FLAGGED | ABSENT]

* **Claim Audited:** "[Exact sentence quoted]"
* **Target Reference:** [Author, Year / File / DOI]
* **Verdict:** `[VERIFIED - PRIMARY EMPIRICAL]` / `[FLAGGED - SECONDARY ECHO]` / `[NOT FOUND]`
* **Diagnostic Summary:** [Explanation of findings]

#### Key Evidence Located:
* **Location:** Page X, Section / Table Y
* **Exact Quote:** *"..."*
* **Grounding Check:** [Confirms whether numbers match exactly]
* **Reviewer #2 Action:** [Actionable guidance on whether to keep, rewrite, or swap the citation]
```

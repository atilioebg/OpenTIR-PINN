"""
audit_claim.py
Rapid Citation Audit Utility for Academic Peer Review and Scientific Writing.

Audits whether a specific claim in a manuscript is substantiated by a cited PDF
without needing to read the entire document cover-to-cover.

Features:
- Extracts text page-by-page using PyMuPDF (fitz)
- Auto-extracts high-entropy keywords and numerical values from the claim
- Scans for exact numerical matches and semantic keyword overlap
- Detects the 'Secondary Citation Echo' trap (flags if the cited paper merely quoted another reference)
- Outputs verified quote, exact page number, and audit verdict
"""

import os
import sys
import re
import argparse
from typing import List, Dict, Any

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Error: PyMuPDF (fitz) is required. Install via 'pip install pymupdf'.", file=sys.stderr)
    sys.exit(1)

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "of", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "can", "will", "just", "don", "should",
    "now", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing", "would", "could", "should", "shall", "might", "must", "show",
    "shows", "shown", "demonstrate", "demonstrates", "using", "used", "paper", "study", "work",
    "author", "authors", "et", "al", "also", "well", "our", "we", "this", "that", "these", "those"
}

def extract_high_entropy_tokens(claim: str) -> Dict[str, List[str]]:
    # Extract numerical tokens (integers, floats, percentages, units)
    numbers = re.findall(r'\b\d+(?:\.\d+)?(?:%|K|°|nm|μm|cm|mm|m)?\b', claim)
    
    # Extract clean words
    words = re.findall(r'[a-zA-Z]{3,}', claim.lower())
    filtered_keywords = [w for w in words if w not in STOPWORDS]
    
    return {
        "numbers": list(set(numbers)),
        "keywords": list(set(filtered_keywords))
    }

def detect_trailing_citation(sentence: str) -> bool:
    # Pattern 1: Bracketed numbers e.g. [1], [1, 2], [1-4]
    if re.search(r'\[\s*\d+(?:[\s,\-–—]\d+)*\s*\]', sentence):
        return True
    # Pattern 2: Author-Year citation e.g. (Smith et al., 2019) or (Gillespie, 1998)
    if re.search(r'\([A-Z][a-zA-Z\s\.\&]+,?\s*(?:19|20)\d{2}[a-z]?\)', sentence):
        return True
    # Pattern 3: Unparenthesized e.g. "according to Smith et al. (2018)"
    if re.search(r'\bet\s+al\.\s*(?:\(\s*)?(?:19|20)\d{2}', sentence, re.IGNORECASE):
        return True
    return False

def audit_paper(pdf_path: str, claim: str, custom_keywords: List[str] = None) -> Dict[str, Any]:
    if not os.path.exists(pdf_path):
        return {"error": f"PDF file not found at: {pdf_path}"}
        
    doc = fitz.open(pdf_path)
    tokens = extract_high_entropy_tokens(claim)
    numbers = tokens["numbers"]
    keywords = custom_keywords if custom_keywords else tokens["keywords"]
    
    scored_sentences = []
    
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        text = page.get_text("text")
        
        # Clean hyphenated line breaks
        text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Split into approximate sentences
        raw_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9])', text)
        
        for sent in raw_sentences:
            sent_clean = sent.strip()
            if len(sent_clean) < 25:
                continue
                
            sent_lower = sent_clean.lower()
            
            # Score matches
            num_matches = [num for num in numbers if num.lower() in sent_lower]
            kw_matches = [kw for kw in keywords if kw.lower() in sent_lower]
            
            score = (len(num_matches) * 3.5) + len(kw_matches)
            
            if score >= 2.0:
                has_citation = detect_trailing_citation(sent_clean)
                scored_sentences.append({
                    "page": page_idx + 1,
                    "sentence": sent_clean,
                    "score": score,
                    "matched_numbers": num_matches,
                    "matched_keywords": kw_matches,
                    "has_secondary_citation": has_citation
                })
                
    doc.close()
    
    # Sort by score descending
    scored_sentences.sort(key=lambda x: x["score"], reverse=True)
    
    # Formulate verdict
    if not scored_sentences:
        verdict = "NOT FOUND / ABSENT"
        details = "No sentences in the cited PDF matched the extracted numerical markers or key physical concepts."
    else:
        top = scored_sentences[0]
        if top["has_secondary_citation"]:
            verdict = "FLAGGED [SECONDARY ECHO]"
            details = "The candidate sentence contains an external citation. The authors may be quoting prior literature rather than presenting an original empirical discovery."
        elif len(top["matched_numbers"]) > 0:
            verdict = "VERIFIED [PRIMARY EMPIRICAL]"
            details = "Exact numerical values and physical keywords matched directly with no secondary citation markers."
        else:
            verdict = "VERIFIED [CONCEPTUAL ALIGNMENT]"
            details = "High semantic keyword overlap identified in primary text."
            
    return {
        "verdict": verdict,
        "details": details,
        "claim": claim,
        "extracted_numbers": numbers,
        "extracted_keywords": keywords,
        "matches": scored_sentences[:5]
    }

def print_audit_report(result: Dict[str, Any]):
    print("\n" + "="*80)
    print("                     CITATION AUDIT REPORT (ANTIGRAVITY)")
    print("="*80)
    
    if "error" in result:
        print(f"[-] ERROR: {result['error']}")
        return
        
    print(f"[*] INCOMING CLAIM:")
    print(f"    \"{result['claim']}\"\n")
    print(f"[*] AUDIT VERDICT:  {result['verdict']}")
    print(f"[*] DIAGNOSTIC:     {result['details']}\n")
    
    print(f"[*] SEARCH TARGETS IDENTIFIED:")
    print(f"    - Numerical Anchors: {result['extracted_numbers']}")
    print(f"    - Key Concepts:      {result['extracted_keywords']}\n")
    
    matches = result["matches"]
    if not matches:
        print("[-] ZERO MATCHES FOUND IN DOCUMENT.")
    else:
        print(f"[+] TOP EVIDENCE LOCATED ({len(matches)} candidates):")
        for i, m in enumerate(matches, 1):
            flag_str = " [!] SECONDARY ECHO CITATION DETECTED" if m["has_secondary_citation"] else " [PASS: PRIMARY PROOF]"
            # Safe print for Windows terminal
            quote_clean = m['sentence'].encode('ascii', errors='replace').decode('ascii')
            print(f"\n  Candidate #{i} (Score: {m['score']:.1f}){flag_str}")
            print(f"  Page: {m['page']}")
            print(f"  Matched Targets: {m['matched_numbers'] + m['matched_keywords']}")
            print(f"  Quote: \"{quote_clean}\"")
    print("\n" + "="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Audit academic claims against a cited PDF without full reading.")
    parser.add_argument("--pdf", required=True, help="Path to the cited PDF file")
    parser.add_argument("--claim", required=True, help="The exact claim sentence being audited")
    parser.add_argument("--keywords", help="Comma-separated list of explicit keywords (optional)")
    
    args = parser.parse_args()
    custom_kw = [k.strip() for k in args.keywords.split(",")] if args.keywords else None
    
    res = audit_paper(args.pdf, args.claim, custom_kw)
    print_audit_report(res)

if __name__ == "__main__":
    main()

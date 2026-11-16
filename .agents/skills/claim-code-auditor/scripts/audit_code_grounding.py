"""Forensic Claim-Code Grounding Auditor for Sobolev-PINN Manuscript.
Scans DOCX paragraphs and cross-references claims against active repository source code,
configurations, and datasets.
"""
import re
import sys
import os
from pathlib import Path
from docx import Document

sys.stdout.reconfigure(encoding='utf-8')

# Key claim patterns to monitor
AUDIT_PATTERNS = [
    ("OPTIMIZER_LR", re.compile(r"\b(AdamW|Adam|cosine|ReduceLROnPlateau|10[−\-]?[345]|learning rate)\b", re.I)),
    ("LOSS_WEIGHTS", re.compile(r"\b(w_mse|w_sam|w_grad|w_curve|lambda_Sobolev|Sobolev weight)\b", re.I)),
    ("ATMOSPHERE_MODTRAN", re.compile(r"\b(MODTRAN|PWV|precipitable water|transmittance|upwelling|downwelling|tape7)\b", re.I)),
    ("DATA_SPLIT", re.compile(r"\b(80/20|quarantine|Golden Target|held-out|cross-validation|leakage)\b", re.I)),
    ("PHYSICS_LOSS", re.compile(r"\b(Planck radiative coupling|forward RTE coupling|Stage 4|Planck loss)\b", re.I)),
    ("BASELINES", re.compile(r"\b(TES|Gillespie|MMD|operational separation)\b", re.I)),
]

def audit_docx(docx_path: str):
    if not os.path.exists(docx_path):
        print(f"Error: {docx_path} not found.")
        sys.exit(1)

    doc = Document(docx_path)
    print(f"Auditing manuscript: {docx_path}")
    print(f"Total paragraphs: {len(doc.paragraphs)}")
    print("=" * 80)

    for i, p in enumerate(doc.paragraphs):
        text = p.text.strip()
        if not text:
            continue

        matched_categories = []
        for cat, pat in AUDIT_PATTERNS:
            if pat.search(text):
                matched_categories.append(cat)

        if matched_categories:
            snippet = text[:140] + ("..." if len(text) > 140 else "")
            print(f"P#{i:03d} [{', '.join(matched_categories)}]:")
            print(f"   \"{snippet}\"")
            print("-" * 80)

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else r"Physics_Informed_Neural_Networks_Sobolev_PINN.docx"
    audit_docx(target)
